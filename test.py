import os
import numpy as np
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
import getdist
import time
matplotlib.rcParams.update({'font.size': 15})
import sys

# from mockfactory import Catalog
from cosmoprimo.fiducial import DESI
from desilike.theories.galaxy_clustering import DirectPowerSpectrumTemplate, FOLPSTracerPowerSpectrumMultipoles
from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike.emulators import EmulatedCalculator, Emulator, TaylorEmulatorEngine, MLPEmulatorEngine
from desilike.profilers import MinuitProfiler

from desilike import setup_logging
setup_logging()

# the k bins
kmin     = 0.008
kmax     = 0.2
binning  = 0.006
k_ev     = np.arange(kmin, kmax+0.001, binning)
klim     = {ell*2: (kmin,kmax,binning) for ell in range(2)}
# k_ev = [0.00889964, 0.01448081, 0.02027516, 0.02647939, 0.03254337, 0.03846521,
#  0.04420902, 0.05007746, 0.05604062, 0.0621324,  0.06819341, 0.07424262,
#  0.08038285, 0.08615286, 0.09191562, 0.09794502, 0.1040656,  0.11002616,
#  0.11596229, 0.1220405,  0.12806637, 0.13409472, 0.14006595, 0.14598208,
#  0.15194908, 0.15796735, 0.16406743, 0.17016069, 0.176199,   0.18213251,
#  0.18802797, 0.19400763, 0.19997052]

catalogue = 'fiducial' #fiducial, Mnu_p, Mnu_ppp
cosmology = 'LCDM'  # LCDM, nuCDM
redshift = 1.0
freedom = 'min'
z_pk = f'z{redshift}'
r_pk = 'pk' # pk, pkRU, pkBL
# Define a dictionary mapping r_pk values to systematic values
systematic_map = {
    'pk': '',
    'pkRU': '+vsmear',
    'pkBL': '+blunder',
    }
systematic = systematic_map.get(r_pk, '')

emulator_fn = f'/home/astro/shhe/projectNU/main/desilike_tests/model/test_emulator.npy'
plot_fn1 = f'./test1.png'
plot_fn2 = f'./test2.png'

# measurement from QUIJOTE simulations
filename = []
filedir = f'/home/astro/shhe/projectNU/main/data/Pypower/{r_pk}_{z_pk}/{catalogue}/npy/'
for file in os.listdir(filedir):
    filename.append(filedir+file)
covariance = filedir+'*'
CovRsf = 25

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
    elif cosmology == 'nsFree':
        template.init.params['n_s'].update(fixed=False)
    elif cosmology == 'wCDM':
        template.init.params['w0_fld'].update(fixed=False)
        template.init.params['n_s'].update(fixed=True,value=0.9624)
    return template

template = initialize_template(redshift, 'wCDM')
theory = FOLPSTracerPowerSpectrumMultipoles(template=template, freedom=freedom)
observable = TracerPowerSpectrumMultipolesObservable(data= filename,
                                                    covariance= covariance,
                                                    klim=klim,
                                                    theory=theory)
likelihood = ObservablesGaussianLikelihood(observable, scale_covariance = 1/CovRsf) #
print(theory.all_params)
print(theory.varied_params)
# observable.plot()

# if not os.path.exists(emulator_fn):
#     template = initialize_template(redshift, cosmology)
#     theory = FOLPSTracerPowerSpectrumMultipoles(template=template, freedom=freedom)
#     observable = TracerPowerSpectrumMultipolesObservable(data= filename,
#                                                     covariance= covariance,
#                                                     klim=klim,
#                                                     theory=theory,
#                                                     kin=np.arange(0.001,0.35,0.002))
#     likelihood = ObservablesGaussianLikelihood(observable, scale_covariance = 1/1) #
#     likelihood()
#     emulator = Emulator(theory.pt, engine=TaylorEmulatorEngine(order=2, method='finite')) # Taylor expansion, up to a given order
#     emulator.set_samples() # evaluate the theory derivatives (with jax auto-differentiation if possible, else finite differentiation)
#     emulator.fit()
#     emulator.save(emulator_fn)
#     print("Training finished")

# theory = FOLPSTracerPowerSpectrumMultipoles(pt=EmulatedCalculator.load(emulator_fn), freedom=freedom)
# observable = TracerPowerSpectrumMultipolesObservable(data= filename,
#                                                     covariance= covariance,
#                                                     klim=klim,
#                                                     theory=theory,
#                                                     kin=np.arange(0.001,0.35,0.002))
# likelihood = ObservablesGaussianLikelihood(observable, scale_covariance = 1/1) #
# likelihood()
# profiler = MinuitProfiler(likelihood, seed=42)
# profiles = profiler.maximize()
# print("Profiler finished")
# print(profiles.to_stats(tablefmt='pretty'))
# likelihood(**profiles.bestfit.choice(input=True))
# observable.plot(fn=plot_fn2)
