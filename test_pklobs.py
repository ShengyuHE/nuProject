import time
import matplotlib 
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import CubicSpline
matplotlib.rcParams.update({'font.size': 12})

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

z_pk = 'z1.0'
compare = 'vsREALISA'

catalogs = ['Mnu_p', 'Mnu_p', 'Mnu_p'] #fiducial, Mnu_p, Mnu_ppp
nbs = [np.arange(100,200,1), np.arange(100,200,1),np.arange(100,200,1)]
r_pks = ['pk', 'pk', 'pk'] # pk, pkRU, pkBL
clen = 3

realisations = ['0-100 realisations', '100-200 realisations', 'random 100 realisations' ]
# Define a dictionary mapping r_pk values to systematic values
systematic_map = {
    'pk': '',
    'pkRU': '+vsmear',
    'pkBL': '+blunder',
    }
systematics = [systematic_map.get(r_pk, '') for r_pk in r_pks]

plot_fn = f'/Pkl_vsOBS_{compare}_{z_pk}.png'
plots_dir = f'/home/astro/shhe/projectNU/main/desilike/{z_pk}/plots'

def dataload(data,choice):
    Pk0 = [[],[]]
    Pk2 = [[],[]]
    if choice == 'paper':
        k = data[0,:,0]
        l0 = data[:,:,1]-1/0.0002118763
        l2 = data[:,:,2]
    elif choice == 'kcen':
        k = data[0,:,0]
        l0 = data[:,:,5]
        l2 = data[:,:,6]
    elif choice == 'kavg':
        k = data[0,:,3]
        l0 = data[:,:,5]
        l2 = data[:,:,6]
    elif choice == 'pypower':
        k = data[0,:,2]
        l0 = data[:,:,3]
        l2 = data[:,:,4]
    Pk0[0] = np.mean(l0,axis=0)
    Pk0[1] = np.std(l0,axis=0)
    Pk2[0] = np.mean(l2,axis=0)
    Pk2[1] = np.std(l2,axis=0)
    pk_cov = np.append(l0,l2,axis=1)
    cov = np.cov(np.array(pk_cov).T)
    icov = np.linalg.inv(cov)
    return (k,Pk0,Pk2,icov)

def Pkload(catalogue, nb, r_pk, z_pk):
    # load the QUIJOTE observation
    tool = 'Pypower' # Powspec, Pypower, NCV
    mode = 'pypower' # kavg, kcen, paper, pypower
    Ddir = f'/home/astro/shhe/projectNU/main/data/{tool}/{r_pk}_{z_pk}/{catalogue}/pk'
    data =[]
    for h in nb:
        realisation=np.loadtxt(Ddir+f'/{catalogue}_{h}_{z_pk}.{r_pk}')
        data.append(realisation)
    (k_ev,pk0,pk2,icov) = dataload(np.array(data),mode)
    pkl = [pk0,pk2]
    print(k_ev)
    return pkl

Pkobs = []
Plotlabels = []
PlotColors = []
show_legend = True

for i in range(clen):
    pkl = Pkload(catalogs[i], nbs[i], r_pks[i], z_pk)
    Pkobs.append(pkl)
    Plotlabels.append(f'{catalogs[i]}{systematics[i]} {realisations[i]}')
    PlotColors.append(f'C{i}')
    if i == 0: # fiducial catalogue
        klen = len(k_ev)
        rsf = 5
        ref =  [i[0] for i in pkl]
        errbar = [i[1] for i in pkl]

height_ratios = [4] + [2] * 2
figsize = (8, 1.5 * sum(height_ratios))
fig, ax = plt.subplots(len(height_ratios), sharex=True, sharey=False, gridspec_kw={'height_ratios': height_ratios}, figsize=figsize, squeeze=True)
fig.subplots_adjust(hspace=0.1)
for ell, i in zip(['monopole','quadrupole'],range(2)):
    values   = [np.zeros(klen), ref[i]]      
    err   = [np.ones(klen),k_ev*errbar[i]/rsf]
    for j in range(2):
        k = 0 if j==0 else i+j
        for t in range(clen):
            # ax[k].plot(k_ev, k_ev*(Pkobs[t][i][0]-values[j])/err[j], 
            #            color = PlotColors[t], linestyle=PlotLines[t], label = Plotlabels[t], alpha=1.0)
            ax[k].errorbar(k_ev, k_ev*(Pkobs[t][i][0]-values[j])/err[j], k_ev*Pkobs[t][i][1]/err[j]/rsf,
                       color = PlotColors[t], label = Plotlabels[t], fmt='o', alpha = 0.9)
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
ax[0].set_title(f'power spectrum obs at {z_pk}')
# plt.savefig(plots_dir+plot_fn, dpi=300)
# print('Save fig in:',plots_dir+plot_fn)
plt.close()

