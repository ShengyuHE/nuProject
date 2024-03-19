import numpy as np
import sys
sys.path.append('/home/astro/shhe/projectNU/FOLPS-nu/')
import FOLPSnu as FOLPS
from scipy.interpolate import CubicSpline

import time
# T0 = time.time()
# print(time.time()-T0)

datapath = '/home/astro/shhe/projectNU/ModelReproduce/data/'

# p =  ['h', 'omega_cdm', 'logA', 'Mnu', 'b1', 'b2', 'alpha0', 'alpha2', 'sn0', 'sn2']
'cosmological constants'
global z_pk, Omega_b, omega_b, n_s, k_ev, n
z_pk = 0.5
Omega_b = 0.049
omega_b = 0.0220684
n_s = 0.9624

'Cosmological independent matrix'
# T0 = time.time()
matrices = FOLPS.Matrices() # 10s
# print('matrix: ',time.time()-T0)

def interp(k, x, y):
        inter = CubicSpline(x, y)
        return inter(k) 

def Pklinear_class(p):
    # T0 = time.time()
    from classy import Class
    k_min = 0.10000E-03
    k_max = 0.10000E+03
    (h, omega_cdm, logA, Mnu)= p[0:4]
    nuCDM = Class()
    params = {'omega_b':omega_b, 'omega_cdm':omega_cdm, 'h':h, 'ln10^{10}A_s':logA, 'n_s':n_s, 
              'N_eff':3.046,  'N_ncdm':1, 'm_ncdm':Mnu,
            #   'tau_reio':0.09,'YHe':0.24,
              'output':'mPk','z_pk':z_pk,'P_k_max_1/Mpc':k_max}
    nuCDM.set(params)
    nuCDM.compute()
    kk = np.logspace(np.log10(k_min), np.log10(k_max), num = 312) #Mpc^-1
    Pk=[]
    for k in kk:
        Pk.append([k, nuCDM.pk_cb(k*nuCDM.h(),z_pk)*nuCDM.h()**3])
    nuCDM.empty()
    # print('linear cb Pk: ',time.time()-T0)
    return np.array(Pk).T

def Pklmodel(p,k_ev,cosmology):
    (h, omega_cdm, logA, Mnu)= p[0:4]
    (b1, b2, alpha0, alpha2, sn0, sn2)=p[4:10]
    inputpkT=Pklinear_class(p) # linear power spectrum
    
    omega_ncdm = Mnu/93.14
    CosmoParams = [z_pk, omega_b, omega_cdm, omega_ncdm, h]
    bs2 = -4/7*(b1 - 1);        #coevolution
    b3nl = 32/315*(b1 - 1);     #coevolution
    alpha4 = 0.0;  # not considered  --> monopole, quadrupole
    ctilde = 0.0;  # fix --> alphashot2
    PshotP = 1/0.0002118763; # constant --> Pshot = 1/n ̄x = 4719.7 h3Mpc−3
    NuisanParams = [b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, 
                ctilde, sn0, sn2, PshotP]
    if cosmology == 'nuCDM':
        nonlinear = FOLPS.NonLinear(inputpkT, CosmoParams, EdSkernels = False)
        kh, Pkl0, Pkl2, Pkl4 = FOLPS.RSDmultipoles(k_ev, NuisanParams, AP = False)
    elif cosmology == 'LCDM':
        nonlinear = FOLPS.NonLinear(inputpkT, CosmoParams, EdSkernels = True)
        kh, Pkl0, Pkl2, Pkl4 = FOLPS.RSDmultipoles(k_ev, NuisanParams, AP = False)
    return(kh, Pkl0, Pkl2)

def Pkload(data,choice):
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
        k = data[0,:,1]
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