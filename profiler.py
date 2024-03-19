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
from desilike.emulators import EmulatedCalculator
from desilike.profilers import MinuitProfiler
from desilike import setup_logging
setup_logging()  # for logging messages

kmin     = 0.008
kmax     = 0.2
binning  = 0.006
k_ev     = np.arange(kmin, kmax+0.001, binning)
klim     = {ell*2: (kmin,kmax,binning) for ell in range(2)}

catalogue = 'fiducial' #fiducial, Mnu_p, Mnu_ppp
cosmology = 'nuCDM'  # LCDM, nuCDM
redshift = 1.0
z_pk = f'z{redshift}'
r_pk = 'pk' # pk, pkRU, pkBL
# Define a dictionary mapping r_pk values to systematic values
systematic_map = {
    'pk': '',
    'pkRU': '+vsmear',
    'pkBL': '+blunder',
    }
systematic = systematic_map.get(r_pk, '')

freedom = 'max'
result_dir = f'/home/astro/shhe/projectNU/main/desilike/{z_pk}/test_{catalogue}/chains'
emulator_fn = f'/home/astro/shhe/projectNU/main/desilike/model/emulator_{cosmology}_F{freedom}_z{redshift}.npy'

# measurement from QUIJOTE simulations
filename = []
filedir = f'/home/astro/shhe/projectNU/main/data/Pypower/{r_pk}_{z_pk}/{catalogue}/npy/'
for file in os.listdir(filedir):
    filename.append(filedir+file)
covariance = filedir+'*'

def initialize_template(redshift, cosmology):
    cosmo = DESI()
    template = DirectPowerSpectrumTemplate(z=redshift,apmode='qisoqap', fiducial='DESI')
    if cosmology == 'LCDM':
        template.init.params['n_s'].update(fixed=True,value=0.9624)
    elif cosmology == 'nuCDM':
        template.init.params['m_ncdm'].update(fixed=False, latex=r'M_\nu', prior = {'limits': [0.0,2.0]}) #we unfix the parameter m_ncdm(Mnu)
        template.init.params['n_s'].update(fixed=True,value=0.9624)
    return template

def Profiler():
    CovRsf = 25
    Ctheory = 'th' # th, el
    profile_fn = f'/profile_{cosmology}_{Ctheory}_V{CovRsf}{systematic}.npy'
    pkl_by_profile_fn = f'/plots/pkl_profile_{cosmology}_{z_pk}_{Ctheory}_V{CovRsf}{systematic}.png'
    # Load the trained emulator
    if Ctheory == 'th':
        theory = FOLPSTracerPowerSpectrumMultipoles(template=initialize_template(cosmology), freedom=freedom, k=k_ev)
    elif Ctheory == 'el':
        theory = FOLPSTracerPowerSpectrumMultipoles(pt=EmulatedCalculator.load(emulator_fn), freedom=freedom)
    observable = TracerPowerSpectrumMultipolesObservable(data= filename,
                                                        covariance= covariance,
                                                        klim=klim,
                                                        theory=theory)
    likelihood = ObservablesGaussianLikelihood(observable, scale_covariance = 1/CovRsf) #
    likelihood()
    # use profile to determine the fixed bs
    print("Profiler test")
    profiler = MinuitProfiler(likelihood, seed=42, save_fn=result_dir+profile_fn)
    profiles = profiler.maximize()
    print("Profiler finished")
    print(profiles.to_stats(tablefmt='pretty'))
    likelihood(**profiles.bestfit.choice(input=True))
    observable.plot()
    plt.gcf()
    plt.savefig(result_dir+pkl_by_profile_fn)
    print('Save fig in', result_dir+ pkl_by_profile_fn)

