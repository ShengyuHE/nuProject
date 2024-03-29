{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "645a638c-245d-4256-9b52-ec80a8f582d8",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import numpy as np\n",
    "import matplotlib\n",
    "import matplotlib.gridspec as gridspec\n",
    "from matplotlib import pyplot as plt\n",
    "import getdist\n",
    "import time\n",
    "matplotlib.rcParams.update({'font.size': 15})\n",
    "\n",
    "# from mockfactory import Catalog\n",
    "from cosmoprimo.fiducial import DESI\n",
    "from desilike.theories.galaxy_clustering import DirectPowerSpectrumTemplate, FOLPSTracerPowerSpectrumMultipoles\n",
    "from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable\n",
    "from desilike.likelihoods import ObservablesGaussianLikelihood\n",
    "from desilike.emulators import EmulatedCalculator, Emulator, TaylorEmulatorEngine, MLPEmulatorEngine\n",
    "from desilike.samplers.emcee import EmceeSampler\n",
    "from desilike.samples import plotting, Chain\n",
    "from desilike.profilers import MinuitProfiler\n",
    "from desilike import setup_logging\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "997457eb",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "def set_true_values(catalogue,cosmology):\n",
    "    # h, Omega_cdm, Mnu, logA\n",
    "    if catalogue == 'fiducial':\n",
    "        true_values = [0.6711, 0.2685, 3.0631]\n",
    "        if cosmology == 'nuCDM':\n",
    "            true_values.append(0.0)\n",
    "    elif catalogue == 'Mnu_p':\n",
    "        true_values =  [0.6711, 0.2661, 3.1247]\n",
    "        if cosmology == 'nuCDM':\n",
    "            true_values.append(0.1)\n",
    "    elif catalogue == 'Mnu_ppp':\n",
    "        true_values = [0.6711, 0.2590, 3.3113]\n",
    "        if cosmology == 'nuCDM':\n",
    "            true_values.append(0.4)\n",
    "    return true_values\n",
    "\n",
    "def initialize_template(redshift,cosmology):\n",
    "    cosmo = DESI()\n",
    "    template = DirectPowerSpectrumTemplate(z=redshift,apmode='qisoqap', fiducial='DESI')\n",
    "    if cosmology == 'LCDM':\n",
    "        del template.params['Omega_m']\n",
    "        template.init.params['h'].update(prior={'limits': [0.4,0.8]})\n",
    "        template.init.params['logA'].update(prior={'limits': [2.0,4.0]})\n",
    "        template.init.params['Omega_cdm'] = {'prior': {'limits': [0.1, 0.6]}, \n",
    "                                        'ref': {'dist': 'norm', 'loc': cosmo['Omega_cdm'],'scale': 0.006}, \n",
    "                                        'delta': 0.015, 'latex': '\\Omega_{cdm}'}\n",
    "        template.init.params['omega_b'].update(fixed=True,value=0.02207)\n",
    "        template.init.params['n_s'].update(fixed=True,value=0.9624)\n",
    "        # print('LCDM template params:',template.all_params)\n",
    "    elif cosmology == 'nuCDM':\n",
    "        del template.params['Omega_m']\n",
    "        template.init.params['h'].update(prior={'limits': [0.4,0.8]})\n",
    "        template.init.params['logA'].update(prior={'limits': [2.0,4.0]})\n",
    "        template.init.params['Omega_cdm'] = {'prior': {'limits': [0.1, 0.6]}, \n",
    "                                        'ref': {'dist': 'norm', 'loc': cosmo['Omega_cdm'],'scale': 0.006}, \n",
    "                                        'delta': 0.015, 'latex': '\\Omega_{cdm}'}\n",
    "        template.init.params['m_ncdm'].update(fixed=False, latex=r'M_\\nu', prior = {'limits': [0.0,1.0]}) #we unfix the parameter m_ncdm(Mnu)\n",
    "        template.init.params['omega_b'].update(fixed=True,value=0.02207)\n",
    "        template.init.params['n_s'].update(fixed=True,value=0.9624)\n",
    "        # print('nuCDM template params:',template.all_params)\n",
    "    return template\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "d3742cf5",
   "metadata": {},
   "outputs": [],
   "source": [
    "setup_logging()  # for logging messages\n",
    "# the k bins\n",
    "kmin     = 0.008\n",
    "kmax     = 0.2\n",
    "binning  = 0.006\n",
    "k_ev     = np.arange(kmin, kmax+0.001, binning)\n",
    "klim     = {ell*2: (kmin,kmax,binning) for ell in range(2)}\n",
    "\n",
    "catalogue = 'Mnu_p' #fiducial, Mnu_p, Mnu_ppp\n",
    "cosmology = 'nuCDM'  # LCDM, nuCDM\n",
    "redshift = 0.5\n",
    "freedom = 'min'\n",
    "z_pk = f'z{redshift}'\n",
    "r_pk = 'pk' # pk, pkRU, pkBL\n",
    "# Define a dictionary mapping r_pk values to systematic values\n",
    "systematic_map = {\n",
    "    'pk': '',\n",
    "    'pkRU': '+vsmear',\n",
    "    'pkBL': '+blunder',\n",
    "    }\n",
    "systematic = systematic_map.get(r_pk, '')\n",
    "engine = 'Taylor' # Taylor, MLP\n",
    "Torder = 6 # Taylor expansion order\n",
    "Tmethod = 'finite'\n",
    "emulator_fn = f'./model/{catalogue}/emulator_{cosmology}_{z_pk}{systematic}.npy'\n",
    "result_fn = f'/Users/alain/Desktop/projectNU/main/desilike/{z_pk}/{catalogue}'\n",
    "\n",
    "# measurement from QUIJOTE simulations\n",
    "filename = []\n",
    "filedir = f'/Users/alain/Desktop/projectNU/main/Pypower/{r_pk}_{z_pk}/{catalogue}/npy/'\n",
    "for file in os.listdir(filedir):\n",
    "    filename.append(filedir+file)\n",
    "covariance = filedir+'*'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "3d587786",
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'pyclass'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[16], line 3\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[38;5;66;03m# For training the emulator\u001b[39;00m\n\u001b[1;32m      2\u001b[0m \u001b[38;5;66;03m# training data and covariance\u001b[39;00m\n\u001b[0;32m----> 3\u001b[0m theory \u001b[38;5;241m=\u001b[39m FOLPSTracerPowerSpectrumMultipoles(template\u001b[38;5;241m=\u001b[39m\u001b[43minitialize_template\u001b[49m\u001b[43m(\u001b[49m\u001b[43mcosmology\u001b[49m\u001b[43m)\u001b[49m, freedom\u001b[38;5;241m=\u001b[39mfreedom, k\u001b[38;5;241m=\u001b[39mk_ev)\n\u001b[1;32m      4\u001b[0m \u001b[38;5;66;03m# Training emulator\u001b[39;00m\n\u001b[1;32m      5\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124mf\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mEmulator engine: \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mengine\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m+order\u001b[39m\u001b[38;5;132;01m{\u001b[39;00mTorder\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m'\u001b[39m)\n",
      "Cell \u001b[0;32mIn[14], line 18\u001b[0m, in \u001b[0;36minitialize_template\u001b[0;34m(cosmology)\u001b[0m\n\u001b[1;32m     17\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21minitialize_template\u001b[39m(cosmology):\n\u001b[0;32m---> 18\u001b[0m     cosmo \u001b[38;5;241m=\u001b[39m \u001b[43mDESI\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m     19\u001b[0m     template \u001b[38;5;241m=\u001b[39m DirectPowerSpectrumTemplate(z\u001b[38;5;241m=\u001b[39mredshift,apmode\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mqisoqap\u001b[39m\u001b[38;5;124m'\u001b[39m, fiducial\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mDESI\u001b[39m\u001b[38;5;124m'\u001b[39m)\n\u001b[1;32m     20\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m cosmology \u001b[38;5;241m==\u001b[39m \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mLCDM\u001b[39m\u001b[38;5;124m'\u001b[39m:\n",
      "File \u001b[0;32m~/opt/anaconda3/envs/nu-env/lib/python3.10/site-packages/cosmoprimo/fiducial.py:223\u001b[0m, in \u001b[0;36mAbacusSummitBase\u001b[0;34m(engine, precision, extra_params, **params)\u001b[0m\n\u001b[1;32m    195\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mAbacusSummitBase\u001b[39m(engine\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mclass\u001b[39m\u001b[38;5;124m'\u001b[39m, precision\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mNone\u001b[39;00m, extra_params\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mNone\u001b[39;00m, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mparams):\n\u001b[1;32m    196\u001b[0m \u001b[38;5;250m    \u001b[39m\u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[1;32m    197\u001b[0m \u001b[38;5;124;03m    Initialize :class:`Cosmology` with base AbacusSummit cosmological parameters (Planck2018, base_plikHM_TTTEEE_lowl_lowE_lensing mean).\u001b[39;00m\n\u001b[1;32m    198\u001b[0m \n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m    221\u001b[0m \u001b[38;5;124;03m    cosmology : Cosmology\u001b[39;00m\n\u001b[1;32m    222\u001b[0m \u001b[38;5;124;03m    \"\"\"\u001b[39;00m\n\u001b[0;32m--> 223\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43mAbacusSummit\u001b[49m\u001b[43m(\u001b[49m\u001b[43mname\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[38;5;124;43m000\u001b[39;49m\u001b[38;5;124;43m'\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mengine\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mengine\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mprecision\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mprecision\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mextra_params\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mextra_params\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[38;5;241;43m*\u001b[39;49m\u001b[43mparams\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[0;32m~/opt/anaconda3/envs/nu-env/lib/python3.10/site-packages/cosmoprimo/fiducial.py:173\u001b[0m, in \u001b[0;36mAbacusSummit\u001b[0;34m(name, engine, precision, extra_params, **params)\u001b[0m\n\u001b[1;32m    171\u001b[0m default_params \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mdict\u001b[39m(k_pivot\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m0.05\u001b[39m, neutrino_hierarchy\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mNone\u001b[39;00m, T_ncdm_over_cmb\u001b[38;5;241m=\u001b[39mconstants\u001b[38;5;241m.\u001b[39mTNCDM_OVER_CMB, A_L\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m1.0\u001b[39m)\n\u001b[1;32m    172\u001b[0m default_params\u001b[38;5;241m.\u001b[39mupdate(AbacusSummit_params(name\u001b[38;5;241m=\u001b[39mname))\n\u001b[0;32m--> 173\u001b[0m engine \u001b[38;5;241m=\u001b[39m \u001b[43mget_engine\u001b[49m\u001b[43m(\u001b[49m\u001b[43mengine\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    174\u001b[0m default_extra_params \u001b[38;5;241m=\u001b[39m {}\n\u001b[1;32m    175\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m engine\u001b[38;5;241m.\u001b[39mname \u001b[38;5;241m==\u001b[39m \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mclass\u001b[39m\u001b[38;5;124m'\u001b[39m:\n",
      "File \u001b[0;32m~/opt/anaconda3/envs/nu-env/lib/python3.10/site-packages/cosmoprimo/cosmology.py:447\u001b[0m, in \u001b[0;36mget_engine\u001b[0;34m(engine)\u001b[0m\n\u001b[1;32m    445\u001b[0m engine \u001b[38;5;241m=\u001b[39m engine\u001b[38;5;241m.\u001b[39mlower()\n\u001b[1;32m    446\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m engine \u001b[38;5;241m==\u001b[39m \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mclass\u001b[39m\u001b[38;5;124m'\u001b[39m:\n\u001b[0;32m--> 447\u001b[0m     \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01m.\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m classy\n\u001b[1;32m    448\u001b[0m \u001b[38;5;28;01melif\u001b[39;00m engine \u001b[38;5;241m==\u001b[39m \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mcamb\u001b[39m\u001b[38;5;124m'\u001b[39m:\n\u001b[1;32m    449\u001b[0m     \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01m.\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m camb\n",
      "File \u001b[0;32m~/opt/anaconda3/envs/nu-env/lib/python3.10/site-packages/cosmoprimo/classy.py:4\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[38;5;124;03m\"\"\"Cosmological calculation with the Boltzmann code CLASS.\"\"\"\u001b[39;00m\n\u001b[1;32m      3\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mnumpy\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mnp\u001b[39;00m\n\u001b[0;32m----> 4\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mpyclass\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m base\n\u001b[1;32m      6\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mcosmology\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m BaseEngine, CosmologyInputError, CosmologyComputationError\n\u001b[1;32m      7\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01minterpolator\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m PowerSpectrumInterpolator1D, PowerSpectrumInterpolator2D\n",
      "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'pyclass'"
     ]
    }
   ],
   "source": [
    "# For training the emulator\n",
    "# training data and covariance\n",
    "theory = FOLPSTracerPowerSpectrumMultipoles(template=initialize_template(cosmology), freedom=freedom, k=k_ev)\n",
    "# Training emulator\n",
    "print(f'Emulator engine: {engine}+order{Torder}')\n",
    "if engine == 'Taylor':\n",
    "    emulator = Emulator(theory.pt, engine=m(order=Torder, method=Tmethod)) # Taylor expansion, up to a given order\n",
    "    # emulator = Emulator(theory.pt, engine=TaylorEmulatorEngine(order={'*': 4,'h': 3,'logA':5}, method=Tmethod)) # Taylor expansion, up to a given order\n",
    "elif engine == 'MLP':\n",
    "    emulator = Emulator(theory.pt, engine=MLPEmulatorEngine) # Neural net (multilayer perceptron)\n",
    "emulator.set_samples() # evaluate the theory derivatives (with jax auto-differentiation if possible, else finite differentiation)\n",
    "emulator.fit()\n",
    "emulator.save(emulator_fn)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
