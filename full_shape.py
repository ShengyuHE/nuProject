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
from desilike.samplers.emcee import EmceeSampler
from desilike.samplers import ZeusSampler
from desilike.samples import plotting, Chain
from desilike import setup_logging
setup_logging()  # for logging messages

# the k bins
kmin     = 0.008
kmax     = 0.2
binning  = 0.006
k_ev     = np.arange(kmin, kmax+0.001, binning)
klim     = {ell*2: (kmin,kmax,binning) for ell in range(2)}
systematic_map = {
    'pk': '',
    'pkRU': '+vsmear',
    'pkBL': '+blunder',
    }

# the cosmology
redshift    = 1.0
catalogue   = 'fiducial'  # fiducial, Mnu_p, Mnu_ppp -- QUIJOTE catalogue
cosmology   = 'wCDM'  # LCDM, nuCDM, nsFree, wCDM -- cosmology model
r_pk        = 'pkRU'  # pk, pkRU -- systematics
freedom     = 'min'  # max, min -- freedom of theory
Ctheory     = 'el'  # th, el -- theory or emulaotr
CovRsf      = 25  # -- covariance rescale factor
z_pk        = f'z{redshift}'
systematic  = systematic_map.get(r_pk, '')
result_dir  = f'/home/astro/shhe/projectNU/main/desilike_tests/{z_pk}/test_{catalogue}/chains'
if not os.path.exists(result_dir): os.makedirs(result_dir)

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
    elif cosmology == 'nsFree':
        template.init.params['n_s'].update(fixed=False)
    elif cosmology == 'wCDM':
        template.init.params['w0_fld'].update(fixed=False)
        template.init.params['n_s'].update(fixed=True,value=0.9624)
    return template

def EmceeSampling():
    nchain = 4
    chain_fn = f'/chain_*_{catalogue}_{cosmology}_{Ctheory}_V{CovRsf}_F{freedom}{systematic}.npy'
    # Load the trained emul
    if Ctheory == 'th':
        template=initialize_template(redshift, cosmology)
        theory = FOLPSTracerPowerSpectrumMultipoles(template=template, freedom=freedom)
    elif Ctheory == 'el':
        emulator_fn = f'/home/astro/shhe/projectNU/main/desilike_tests/model/emulator_{catalogue}_{cosmology}_V{CovRsf}_F{freedom}_z{redshift}.npy'   
        if not os.path.exists(emulator_fn):
            template=initialize_template(redshift, cosmology)
            theory = FOLPSTracerPowerSpectrumMultipoles(template=template, freedom=freedom)
            observable = TracerPowerSpectrumMultipolesObservable(data= filename,
                                                            covariance= covariance,
                                                            klim=klim,
                                                            theory=theory,
                                                            kin=np.arange(0.001,0.35,0.002))
            likelihood = ObservablesGaussianLikelihood(observable, scale_covariance = 1/CovRsf)
            likelihood()
            emulator = Emulator(theory.pt, engine=TaylorEmulatorEngine(order={'*': 4,'h': 3,'logA':5}, method='finite')) # Taylor expansion, up to a given order
            emulator.set_samples() # evaluate the theory derivatives (with jax auto-differentiation if possible, else finite differentiation)
            emulator.fit()
            emulator.save(emulator_fn)
            print("Training emmulator finished")
        theory = FOLPSTracerPowerSpectrumMultipoles(pt=EmulatedCalculator.load(emulator_fn), freedom=freedom)
    observable = TracerPowerSpectrumMultipolesObservable(data= filename,
                                                        covariance= covariance,
                                                        klim=klim,
                                                        theory=theory,
                                                        kin=np.arange(0.001,0.35,0.002))
    likelihood = ObservablesGaussianLikelihood(observable, scale_covariance = 1/CovRsf) 
    likelihood()
    # for param in likelihood.all_params.select(basename=['alpha*', 'sn*']):
    #     param.update(derived='.auto')
    'Emcee sampler'
    sampler = EmceeSampler(likelihood, seed=42, nwalkers=64, save_fn =result_dir+chain_fn, chains=nchain)
    sampler.run(check={'max_eigen_gr': 0.05}, max_iterations = 6001) # save every 300 iterations
    # sampler = ZeusSampler(likelihood, save_fn='_tests/chain_fs_direct.npy', seed=42)
    # sampler.run(check={'max_eigen_gr': 0.1})

EmceeSampling()
