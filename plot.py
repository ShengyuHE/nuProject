import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import getdist
import time
import pickle

matplotlib.rcParams.update({'font.size': 15})

# from mockfactory import Catalog
from cosmoprimo.fiducial import DESI
from desilike.theories.galaxy_clustering import DirectPowerSpectrumTemplate, FOLPSTracerPowerSpectrumMultipoles
from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike.emulators import EmulatedCalculator
from desilike.samples import Profiles, plotting, Chain
from desilike import setup_logging
setup_logging()  # for logging messages

# the k bins
kmin        = 0.008
kmax        = 0.2
binning     = 0.006
k_ev        = np.arange(kmin, kmax+0.001, binning)
klim        = {ell*2: (kmin,kmax,binning) for ell in range(2)}
systematic_map = {
    'pk': '',
    'pkRU': '+vsmear',
    'pkBL': '+blunder',
    }

# the cosmology
redshift    = 1.0
catalogue   = 'fiducial'  # fiducial, Mnu_p, Mnu_ppp -- QUIJOTE catalogue
cosmology   = 'nuCDM'  # LCDM, nuCDM, nsFree, wCDM -- cosmology model
r_pk        = 'pk'  # pk, pkRU -- systematics
freedom     = 'min'  # max, min -- freedom of theory
Ctheory     = 'el'  # th, el -- theory or emulaotr used in the sampler
CovRsf      = 100  # -- covariance rescale factor
z_pk        = f'z{redshift}'
systematic  = systematic_map.get(r_pk, '')

# measurement from QUIJOTE simulations
filename = []
filedir = f'/home/astro/shhe/projectNU/main/data/Pypower/{r_pk}_{z_pk}/{catalogue}/npy/'
for file in os.listdir(filedir):
    filename.append(filedir+file)
covariance = filedir+'*'

# for the samples
burnin      = 0.8
slice_step  = 200

Choice_sample_plot      = False
Choice_sample_compare   = True

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

def initialize_params(cosmology, paramspace):
    if cosmology == 'LCDM':
        params = ['h', 'omega_cdm', 'omega_b', 'logA']
    elif cosmology == 'nuCDM':
        params = ['m_ncdm',  'h', 'omega_cdm',  'omega_b', 'logA']
    elif cosmology == 'nsFree':
        params = ['h', 'omega_cdm', 'omega_b', 'logA', 'n_s']
    elif cosmology == 'wCDM':
        params = ['h', 'omega_cdm', 'omega_b', 'logA', 'w0_fld']
    if paramspace == 'FP': params=params+['b1', 'b2', 'alpha0', 'alpha2', 'sn0', 'sn2']
    return params

def set_true_values(catalogue, params):
    update_values = {
        'fiducial': {'h': 0.6711, 'omega_cdm': 0.1209, 'Omega_cdm': 0.2685, 'omega_b':0.02207,'logA': 3.0631, 'm_ncdm': 0.0, 'n_s':0.9624, 'w0_fld':-1.0},
        'Mnu_p': {'h': 0.6711, 'omega_cdm': 0.1198, 'Omega_cdm': 0.2661, 'omega_b':0.02207,'logA': 3.1247, 'm_ncdm': 0.1, 'n_s':0.9624, 'w0_fld':-1.0},
        'Mnu_ppp': {'h': 0.6711, 'omega_cdm': 0.1166, 'Omega_cdm': 0.2590, 'omega_b':0.02207,'logA': 3.3113, 'm_ncdm': 0.4, 'n_s':0.9624, 'w0_fld':-1.0}
    }
    if catalogue in update_values:
        truth_values = update_values[catalogue]
    return [truth_values[param] for param in params if param in truth_values]

def getChain(chain_dir, p_chain, n_chain, burnin_, slice_step_):
    (catalogue, cosmology, Ctheory, CovRsf, freedom, systematic) = p_chain
    if len(n_chain) == 0:
        chainName   = f'/chain_{catalogue}_{cosmology}_{Ctheory}_V{CovRsf}_F{freedom}{systematic}.npy'
        chain       = Chain.load(chain_dir+chainName).remove_burnin(burnin_)[::slice_step_]
    elif len(n_chain) > 0:
        chain = Chain.concatenate(
            [Chain.load(chain_dir+f'/chain_{i}_{catalogue}_{cosmology}_{Ctheory}_V{CovRsf}_F{freedom}{systematic}.npy').remove_burnin(burnin_)[::slice_step_] for i in n_chain]
            )
    return chain

def Sample_plot():
    paramspace      = 'CP'
    chain_dir       = f'/home/astro/shhe/projectNU/main/desilike_tests/{z_pk}/test_{catalogue}/chains'
    plot_dir        = f'/home/astro/shhe/projectNU/main/desilike_tests/{z_pk}/test_{catalogue}/plot_by_chain'
    covergence_fn   = f'/chains_{catalogue}_{cosmology}_{z_pk}_V{CovRsf}_F{freedom}{systematic}.png'
    pklvsBF_fn      = f'/pkl_vsBF_{catalogue}_{cosmology}_{z_pk}_V{CovRsf}_F{freedom}{systematic}.png'
    corner_fn       = f'/corner_{catalogue}_{cosmology}_{z_pk}_V{CovRsf}_F{freedom}{systematic}.png'
    if not os.path.exists(plot_dir): os.makedirs(plot_dir)

    # for the samplers
    n_chain         = [0,1,2,3] #index for chain
    p_chain         = [catalogue, cosmology, Ctheory, CovRsf, freedom, systematic]
    params          = initialize_params(cosmology, paramspace)

    def plot_covergence_walker(plot_fn, params):
        nwalkers        = 64
        params          = initialize_params(cosmology, 'FP')
        ndim            = len(params)
        chain           = getChain(chain_dir, p_chain, n_chain, 0, 1)
        chain_samples   = dict(zip(chain.basenames(), chain.data))
        samples         = np.array([chain_samples[p] for p in params])
        medians         = np.array(chain.median(params=params))
        true_values = set_true_values(catalogue, params)
        fig, ax = plt.subplots(ndim, sharex=True, figsize=(16, 2 * ndim))
        for i in range(nwalkers):
            for j in range(ndim):
                ax[j].plot(samples[j, :, i], c = 'green', lw=0.3)
                ax[j].set_ylabel(params[j], fontsize=15)
                ax[j].grid(True)
                ax[j].axhline(medians[j], c='blue', lw=1.2)
        for j in range(len(true_values)):
            ax[j].axhline(true_values[j], c='red', lw=1.2)
        ax[-1].set_xlabel(f'Step/{slice_step} with burnin {burnin}')
        plt.savefig(plot_dir+plot_fn)
        print("Save fig in:", plot_dir+plot_fn)

    def plot_pkl_vsBF(plot_fn, params):
        theory = FOLPSTracerPowerSpectrumMultipoles(template=initialize_template(redshift, cosmology), freedom=freedom)
        observable = TracerPowerSpectrumMultipolesObservable(data= filename,
                                                            covariance= covariance,
                                                            klim=klim,
                                                            theory=theory)
        likelihood = ObservablesGaussianLikelihood(observable, scale_covariance = 1/CovRsf) #
        chain= getChain(chain_dir, p_chain, n_chain,burnin, slice_step)
        print(chain.to_stats(tablefmt='pretty'))
        likelihood(**chain.choice(index='argmax', input=True))
        print('chisquare of bestfit params:',-2*likelihood())
        observable.plot()
        plt.gcf()
        plt.savefig(plot_dir+plot_fn)
        print('Save pklvsBF in', plot_dir+ plot_fn) 

    def plot_posterior_corner(plot_fn, params):
        chain= getChain(chain_dir, p_chain, n_chain, burnin, slice_step)
        g = plotting.plot_triangle(chain,
                                params = params, 
                                title_limit=1,
                                filled = True,
                                line_args = [{'color':'C0'}],
                                contour_colors=['C0'],
                                )
        true_values = set_true_values(catalogue, params)
        for i in range(len(true_values)):
            for j in range(i+1):
                g.subplots[i,j].axvline(true_values[j], c = 'k', ls = ':', lw = 1.2)
                if i != j : g.subplots[i,j].axhline(true_values[i], c = 'k', ls = ':', lw = 1.2)
        g.export(plot_dir+plot_fn)
        print('Save corner plot in', plot_dir+ plot_fn)

    plot_covergence_walker(covergence_fn, params)
    # plot_pkl_vsBF(pklvsBF_fn, params)
    # plot_posterior_corner(corner_fn, params)
    
def Sampler_compare():
    paramspace  = 'FP' # CP: cosmological parameter, FP: full parameters
    compare     = 'vsRU'
    chain_dir   = f'/home/astro/shhe/projectNU/main/desilike_tests/{z_pk}/test_{catalogue}/chains'
    plot_dir    = f'/home/astro/shhe/projectNU/main/desilike_tests/{z_pk}/test_{catalogue}/plots'
    log_fn      = f'/Info_{compare}_{catalogue}_{cosmology}_V{CovRsf}_F{freedom}_{z_pk}.txt'
    pkl_fn      = f'/Pkl_vsBF_{compare}_{catalogue}_{cosmology}_V{CovRsf}_F{freedom}_{z_pk}.png'
    corner_fn   = f'/corner_{compare}_{catalogue}_{cosmology}_{paramspace}_V{CovRsf}_F{freedom}_{z_pk}.png'
    params      = initialize_params(cosmology, paramspace)
    if not os.path.exists(plot_dir): os.makedirs(plot_dir)

    catalogs    = ['fiducial', 'fiducial'] #fiducial, Mnu_p, Mnu_ppp
    clen        = len(catalogs)
    cosmologies = [cosmology, cosmology]  # LCDM, nuCDM
    r_pks       = ['pk', 'pkRU'] # pk, pkRU, pkBL
    systematics = [systematic_map.get(r_pk, '') for r_pk in r_pks]
    f_pks       = [freedom]*clen
    Vdirs       = [CovRsf]*clen
    n_chains    = [[0,3],[0,3]] #index for chain

    colors      = ['C0', 'C1']
    lws         = [1.0, 1.6]
    ls          = ['-', '--']
    fiiled      = [True, False]

    def PklvsBF_plot():
        import PKmodel as PK
        dlen            = clen
        data_catalogs   = catalogs #fiducial, Mnu_p, Mnu_ppp
        data_r_pks      = r_pks # pk, pkRU, pkBL
        data_systematics= [systematic_map.get(r_pk, '') for r_pk in data_r_pks]
        data_nb         = np.arange(100,200,1)  # realisations 

        def Pkload(catalogue, r_pk, z_pk):
            # load the QUIJOTE observation
            tool = 'Pypower' # Powspec, Pypower, NCV
            mode = 'pypower' # kavg, kcen, paper, pypower
            Ddir = f'/home/astro/shhe/projectNU/main/data/{tool}/{r_pk}_{z_pk}/{catalogue}/pk'
            data =[]
            for h in data_nb:
                realisation=np.loadtxt(Ddir+f'/{catalogue}_{h}_{z_pk}.{r_pk}')
                data.append(realisation)
            (k_ev,pk0,pk2,icov) = PK.Pkload(np.array(data),mode)
            pkl = [pk0,pk2]
            return pkl

        Pkobs = []
        Pkbf = []
        data_labels = []
        Plotlabels = []
        # load the QUIJOTE observation
        for i in range(dlen):
            pkl = Pkload(data_catalogs[i], data_r_pks[i], z_pk)
            Pkobs.append(pkl)
            data_labels.append(f'{data_catalogs[i]} obs{data_systematics[i]}')
            if i == 0:
                rsf = 5
                klen = len(k_ev)
                ref =  [i[0] for i in pkl]
                errbar = [i[1] for i in pkl]
        # load the theory with bestfit parameters
        show_legend = True
        show_info   = True
        PlotColors  = colors
        PlotSyples  = ls
        theory      = FOLPSTracerPowerSpectrumMultipoles(template=initialize_template(redshift, cosmology), freedom=freedom, k=k_ev)
        observable  = TracerPowerSpectrumMultipolesObservable(data= filename,
                                                            covariance= covariance,
                                                            klim=klim,
                                                            theory=theory)
        likelihood  = ObservablesGaussianLikelihood(observable, scale_covariance = 1/CovRsf)
        for i in range(clen):
            p_chain = [catalogs[i], cosmologies[i], Ctheory, CovRsf, f_pks[i], systematics[i]]
            chain   = getChain(chain_dir, p_chain, n_chains[i], burnin, slice_step)
            if show_info == True:
                chi2    = -2*likelihood(**chain.choice(index='argmax', input=True))
                print(chain.to_stats(tablefmt='pretty'))
                print(f'Chisquare: {chi2}\n')              
            Pkbf.append(theory(**chain.choice(index='argmax', input=True)))
            # PlotColors.append(f'C{i}')
            Plotlabels.append(f'{catalogs[i]} {cosmologies[i]} bestfit{systematics[i]}')
        # plot the pkl with observable and predction from bestfit parameters
        height_ratios = [2.5+1.5*dlen] + [1.5] * 2
        figsize = (7+dlen, 1.5 * sum(height_ratios))
        fig, ax = plt.subplots(len(height_ratios), sharex=True, sharey=False, gridspec_kw={'height_ratios': height_ratios}, figsize=figsize, squeeze=True)
        fig.subplots_adjust(hspace=0.1)
        for ell, i in zip(['monopole','quadrupole'],range(2)):
            values   = [np.zeros(klen), ref[i]]      
            err   = [np.ones(klen),k_ev*errbar[i]/rsf]
            for j in range(2):
                k = 0 if j==0 else i+j
                for t in range(dlen):
                    ax[k].errorbar(k_ev, k_ev*(Pkobs[t][i][0]-values[j])/err[j], k_ev*Pkobs[t][i][1]/err[j]/rsf,
                                color = PlotColors[t], label = data_labels[t], fmt='o', alpha = 0.9)
                for t in range(clen):
                    ax[k].plot(k_ev,k_ev*(Pkbf[t][i]-values[j])/err[j],label=Plotlabels[t], 
                            color =PlotColors[t], linestyle=PlotSyples[t],alpha=1.0)
                if k == 0:
                    ax[k].grid(True)
                    ax[k].set_ylabel(r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
                if k==1 or k==2:
                    ax[k].grid(True)
                    ax[k].set_ylim(-5, 5)
                    ax[k].set_ylabel(r'$\Delta P_{}$/err'.format(i*2))
                    for offset in [-2., 2.]: ax[k].axhline(offset, color='k', linestyle='--', alpha=0.8)
            if i==0 and show_legend: ax[0].legend(loc=4)
        ax[-1].set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
        ax[0].set_title(f'power spectrum obs vs bestfit at {z_pk}')
        plt.savefig(plot_dir+pkl_fn, dpi=300)
        print('Save fig in:',plot_dir+pkl_fn)
        plt.close()
        file = open(plot_dir+log_fn, 'a')
        file.close()

    def Corner_plot():
        samples = []
        Plotlabels = []
        contour_ls      = ls
        contour_colors  = colors
        contour_lws     = lws
        contour_fiiled  = fiiled

        for i in range(clen):
            p_chain = [catalogs[i], cosmologies[i], Ctheory, CovRsf, f_pks[i], systematics[i]]
            samples.append(getChain(chain_dir, p_chain, n_chains[i], burnin, slice_step))
            Plotlabels.append(f'{catalogs[i]} {cosmologies[i]} {Vdirs[i]}{systematics[i]}')

        g = plotting.plot_triangle(samples, params = params, title_limit=1, 
                                    legend_labels=Plotlabels, legend_loc='upper right', 
                                    filled = contour_fiiled,
                                    contour_ls = contour_ls,
                                    contour_lws = contour_lws,
                                    contour_colors = contour_colors,
                                    )
        g.settings.figure_legend_frame = False
        g.settings.legend_fontsize = 20
        g.settings.title_limit_fontsize = 16
        true_values = set_true_values(catalogue, params)
        for i in range(len(true_values)):
            for j in range(i+1):
                g.subplots[i,j].axvline(true_values[j], c = 'k', ls = ':', lw = 1.2)
                if i != j : g.subplots[i,j].axhline(true_values[i], c = 'k', ls = ':', lw = 1.2)
        g.export(plot_dir+corner_fn)
        print("Save fig in:", plot_dir+corner_fn)

    if not os.path.exists(plot_dir+pkl_fn):
        PklvsBF_plot()
    # if not os.path.exists(plot_dir+corner_fn):
    Corner_plot()

actions = {
    'Choice_sample_plot': Sample_plot,
    'Choice_sample_compare': Sampler_compare,
    }
for choice, action in actions.items():
    if globals()[choice]:
        action()

