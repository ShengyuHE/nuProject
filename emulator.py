import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import getdist
import time
matplotlib.rcParams.update({'font.size': 15})

# from mockfactory import Catalog
from cosmoprimo.fiducial import DESI
from desilike.theories.galaxy_clustering import DirectPowerSpectrumTemplate, FOLPSTracerPowerSpectrumMultipoles
from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike.emulators import EmulatedCalculator, Emulator, TaylorEmulatorEngine, MLPEmulatorEngine
from desilike import setup_logging
setup_logging()  # for logging messages

# the k bins
kmin     = 0.008
kmax     = 0.2
binning  = 0.006
k_ev     = np.arange(kmin, kmax+0.001, binning)
klim     = {ell*2: (kmin,kmax,binning) for ell in range(2)}

catalogue = 'fiducial' #fiducial, Mnu_p, Mnu_ppp
cosmology = 'nuCDM'  # LCDM, nuCDM
redshift = 0.5
freedom = 'max'
z_pk = f'z{redshift}'
r_pk = 'pk' # pk, pkRU, pkBL
# Define a dictionary mapping r_pk values to systematic values
systematic_map = {
    'pk': '',
    'pkRU': '+vsmear',
    'pkBL': '+blunder',
    }
systematic = systematic_map.get(r_pk, '')
# emulator_fn = f'/home/astro/shhe/projectNU/main/desilike/model/emulator_{cosmology}_F{freedom}_{z_pk}.npy'
plot_dir = f'/home/astro/shhe/projectNU/main/desilike/model/plots'

# measurement from QUIJOTE simulations
filename = []
filedir = f'/home/astro/shhe/projectNU/main/data/Pypower/{r_pk}_{z_pk}/{catalogue}/npy/'
for file in os.listdir(filedir):
    filename.append(filedir+file)
covariance = filedir+'*'

Choice_train = True
Choice_test = False

def set_true_values(catalogue, params):
    update_values = {
        'fiducial': {'h': 0.6711, 'omega_cdm': 0.1209, 'Omega_cdm': 0.2685, 'omega_b':0.02207,'logA': 3.0631, 'm_ncdm': 0.0},
        'Mnu_p': {'h': 0.6711, 'omega_cdm': 0.1198, 'Omega_cdm': 0.2661, 'omega_b':0.02207,'logA': 3.1247, 'm_ncdm': 0.1},
        'Mnu_ppp': {'h': 0.6711, 'omega_cdm': 0.1166, 'Omega_cdm': 0.2590, 'omega_b':0.02207,'logA': 3.3113, 'm_ncdm': 0.4}
    }
    if catalogue in update_values:
        truth_values = update_values[catalogue]
    else:
        print('Invalid catalogue')
        return None
    return [truth_values[param] for param in params if param in truth_values]

def initialize_template(redshift, cosmology):
    cosmo = DESI()
    template = DirectPowerSpectrumTemplate(z=redshift,apmode='qisoqap', fiducial='DESI')
    if cosmology == 'LCDM':
        template.init.params['n_s'].update(fixed=True,value=0.9624)
    elif cosmology == 'nuCDM':
        template.init.params['m_ncdm'].update(fixed=False, latex=r'M_\nu', prior = {'limits': [0.0,2.0]}) #we unfix the parameter m_ncdm(Mnu)
        template.init.params['n_s'].update(fixed=True,value=0.9624)
    return template

def EmulatorTraining(redshift, cosmology, freedom):
    print("Training emulator")
    engine = 'Taylor' # Taylor, MLP
    emulator_fn = f'/home/astro/shhe/projectNU/main/desilike/model/emulator_{cosmology}_F{freedom}_z{redshift}.npy'
    # training data and covariance
    template = initialize_template(redshift, cosmology)
    theory = FOLPSTracerPowerSpectrumMultipoles(template=template, freedom=freedom)
    # Training emulator
    if engine == 'Taylor':
        emulator = Emulator(theory.pt, engine=TaylorEmulatorEngine(order={'*': 4,'h': 3,'logA':5}, method='finite')) # Taylor expansion, up to a given order
    elif engine == 'MLP':
        emulator = Emulator(theory.pt, engine=MLPEmulatorEngine) # Neural net (multilayer perceptron)
    emulator.set_samples() # evaluate the theory derivatives (with jax auto-differentiation if possible, else finite differentiation)
    emulator.fit()
    emulator.save(emulator_fn)
    print("Training finished")

def EmulatorTesting():
    print("Testing emulator")
    engine = 'Taylor' # Taylor, MLP
    Torder = 5 # Taylor expansion order
    Tmethod = 'finite'
    plotchoicie = 'EvT' # EvT (emulator vs theory), obs (emulator vs measurement)
    plot_fn = f'/test/pkl_{plotchoicie}_{cosmology}_T{Torder}_{catalogue}_{label}.png'

    # defaul parameters from QUIJOTE simulation or sampling result
    nuiparams = [1.5990635, -0.93157041,  -1.5515531,  -24.32621769,  0.32751211,  -8.53417766]
    cosmoparams_truth = set_true_values()
    # cosmological paramers. parameter grid found in the emulatro training file
    prior_h = [0.45, 0.5, 0.6, 0.6711, 0.7, 0.75]
    prior_Omega_cdm = [0.2, 0.23, 0.2685, 0.3, 0.35]
    prior_logA = [2, 2.5, 3.0631, 3.5, 4]
    prior_m_ncdm = [0.0, 0.2, 0.4, 0.6, 0.8]
    prior_space = [prior_h, prior_Omega_cdm, prior_logA, prior_m_ncdm] if cosmology == 'nuCDM' else [prior_h, prior_Omega_cdm, prior_logA]

    # load the QUIJOTE observation
    tool = 'Pypower' # Powspec, Pypower, NCV
    mode = 'pypower' # kavg, kcen, paper, pypower
    Ddir = f'/home/astro/shhe/projectNU/main/data/{tool}/{r_pk}_{z_pk}/{catalogue}/pk'
    print('Data dir:',Ddir)
    data =[]
    nb= np.arange(100,200,1) if catalogue == 'fiducial' else np.arange(0,100,1) # catalogs
    for h in nb:
        realisation=np.loadtxt(Ddir+f'/{catalogue}_{h}_{z_pk}.{r_pk}')
        data.append(realisation)
    (k_ev,pk0,pk2,icov) = PK.Pkload(np.array(data),mode)
    # load the trained emulator
    template = initialize_template(cosmology)
    theory_fd = FOLPSTracerPowerSpectrumMultipoles(template=template, k=k_ev)
    theory_el = FOLPSTracerPowerSpectrumMultipoles(pt=EmulatedCalculator.load(emulator_fn), freedom=freedom, k=k_ev)
    klen = len(k_ev)
    Pkobs = [pk0,pk2]
    rsf = 5
    ref =  [i[0] for i in Pkobs]
    errbar = [i[1] for i in Pkobs]

    # plot
    if plotchoicie == 'obs':
        cosmoparams = cosmoparams_truth # Mnu_ppp sampling result or truth values
        start = time.time()
        Pkfd = theory_fd(h = cosmoparams[0], Omega_cdm = cosmoparams[1], logA=cosmoparams[2], m_ncdm = cosmoparams[3],
            b1 = nuiparams[0], b2 = nuiparams[1], alpha0 = nuiparams[2], alpha2 = nuiparams[3], 
            sn0 = nuiparams[4], sn2 = nuiparams[5])
        Pkel = theory_el(h = cosmoparams[0], Omega_cdm = cosmoparams[1] , logA=cosmoparams[2], m_ncdm = cosmoparams[3],
                    b1 = nuiparams[0], b2 = nuiparams[1], alpha0 = nuiparams[2], alpha2 = nuiparams[3], 
                    sn0 = nuiparams[4], sn2 = nuiparams[5])
        end = time.time()
        print('Emulator calculation time:',end - start)
        # FOLPSnu model prediction
        # start = time.time()
        # p = np.append(cosmoparams, nuiparams)
        # (k_model_nu,pkl0_model_nu,pkl2_model_nu) = PK.Pklmodel(p, k_ev, 'nuCDM')
        # Pkmodel_nu = [pkl0_model_nu,pkl2_model_nu]
        # end = time.time()
        # print('FOLPSnu model calculation time:',end - start)
        fig = plt.figure(figsize=(12,6))
        spec = gridspec.GridSpec(nrows=2,ncols=2, height_ratios=[4, 1], wspace=0.2,hspace=0)
        ax = np.empty((2,2), dtype=type(plt.axes))
        for name,i in zip(['monopole','quadrupole'],range(2)):
            values   = [np.zeros(klen), ref[i]]      
            err   = [np.ones(klen),k_ev*errbar[i]/rsf]
            for j in range(2):
                ax[j,i] = fig.add_subplot(spec[j,i])
                ax[j,i].errorbar(k_ev,k_ev*(Pkobs[i][0]-values[j])/err[j],k_ev*Pkobs[i][1]/err[j]/rsf,label=f'QUIJOTE {catalogs} obs',color='black',fmt='o')
                ax[j,i].plot(k_ev,k_ev*(Pkel[i]-values[j])/err[j],label=f'LogA 3.2',color = 'C0', linestyle='-',alpha=0.8)
                ax[j,i].plot(k_ev,k_ev*(Pkel2[i]-values[j])/err[j],label=f'LogA 3.0',color = 'C1', linestyle='-',alpha=0.8)
                # ax[j,i].plot(k_ev,k_ev*(Pkmodel_nu[i]-values[j])/err[j],label=f'FOLPSnu model predict',color = 'C3', linestyle='-',alpha=0.8)
                plt.xlabel(r'$k \, [h\, Mpc^{-1}]$')
                #   plt.xlim([0.02,0.11])
                if (j==0):
                    ax[j,i].set_ylabel(r'$k P_{}$'.format(i*2))
                    plt.legend(fontsize=12,loc=4)
                    plt.title(f'Pk {name} at {z_pk}')
                    plt.xticks(alpha=0)
                if (j==1):
                    ax[j,i].set_ylabel(r'$\Delta P_{}$/err'.format(i*2))  
                    plt.ylim([-5,5])
        plt.savefig(plot_dir+plot_fn)
        print('Save fig in:', plot_dir+plot_fn)
        plt.close()

    if plotchoicie == 'EvT':
        labels = ['h', 'Omega_cdm',  'logA', 'm_ncdm']
        for j,params in enumerate(prior_space):
            color = f'C{j}'
            label = labels[j]
            cosmoparams = set_true_values()
            fig, axs = plt.subplots(2, 1,figsize=(14,12))
            for param in params:
                cosmoparams[j] = param
                print(cosmoparams,nuiparams)
                start = time.time()
                Pkfd = theory_fd(h = cosmoparams[0], Omega_cdm = cosmoparams[1], logA=cosmoparams[2], m_ncdm = cosmoparams[3],
                    b1 = nuiparams[0], b2 = nuiparams[1], alpha0 = nuiparams[2], alpha2 = nuiparams[3], 
                    sn0 = nuiparams[4], sn2 = nuiparams[5])
                end = time.time()
                print('FOLPSnu desilike calculation time:',end - start)
                start = time.time()
                Pkel = theory_el(h = cosmoparams[0], Omega_cdm = cosmoparams[1] , logA=cosmoparams[2], m_ncdm = cosmoparams[3],
                            b1 = nuiparams[0], b2 = nuiparams[1], alpha0 = nuiparams[2], alpha2 = nuiparams[3], 
                            sn0 = nuiparams[4], sn2 = nuiparams[5])
                end = time.time()
                print('Emulator desilike calculation time:',end - start)
                for name,i in zip(['monopole','quadrupole'],range(2)):
                    values   = ref[i]
                    err   = k_ev*errbar[i]/rsf
                    axs[i].plot(k_ev,(Pkfd[i]-Pkel[i])/Pkfd[i], linestyle= '-',alpha=0.8, label = f"{label}={param}")
            for j in range(2):
                axs[j].axhline(0.01,color='black',linestyle=':')
                axs[j].axhline(-0.01,color='black',linestyle=':')
                axs[j].set_ylim([-0.1,0.1])
            axs[0].set_ylabel(r'$ \Delta(Pk0) / Pk0 \, [h^{-1} \,  Mpc]^2$')
            axs[1].set_ylabel(r'$ \Delta(Pk2) / Pk2  \, [h^{-1} \,  Mpc]^2$')
            axs[1].set_xlabel(r'$k \, [h\, Mpc^{-1}]$')
            axs[1].legend(loc=3)
            plt.savefig(plot_dir+plot_fn)
            print('Save fig in:', plot_dir+plot_fn)
            plt.close()

for cosmo in ['LCDM', 'nuCDM']:
    for free in ['max', 'min']:
        for z in [1.0]:
            EmulatorTraining(z, cosmo, free)

# actions = {
#     'Choice_train': EmulatorTraining,
#     'Choice_test': EmulatorTesting,
#     }
# # Loop through the dictionary and call the function if the choice is True
# for choice, action in actions.items():
#     if globals()[choice]:
#         action()