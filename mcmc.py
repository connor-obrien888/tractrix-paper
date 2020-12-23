# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 12:14:43 2020

MCMC fitting program adapted for ease of use on BU SCC. Notes for the user:
    1) If your computer has fewer than 8 compute threads, change line 106 to include your number of threads! 
        Otherwise you may experience issues.
    2) Do NOT run this from an interpreter! Instead run it from the command line with the line
        python mcmc_fit.py
    3) The MCMC chain is stored in mcmc_savestate.h5. 
        If this file already exists, the new chain is appended into it (so you can pick up where you left off)

@author: connor o'brien
"""

import numpy as np
import pandas as pd
import emcee
import schwimmbad

##################################################################

# #Load in training data as a global to save computation time
data = pd.read_csv('training_list_20200715.csv', sep = ',', index_col=0)

# #For convenience, split each column into its own numpy object
bz = data['bz[nT]'].to_numpy() #All position/IMF data is in GSE coordinates
by = data['by[nT]'].to_numpy()
bx = data['bx[nT]'].to_numpy()
xpos = data['xpos'].to_numpy()
ypos = np.sqrt((data['ypos'].to_numpy())**2 + (data['zpos'].to_numpy())**2)
bt = np.sqrt( bx**2 + by**2 + bz**2)
clock = np.arctan2(by , bz)
sin_rec = bt * (np.sin(clock / 2))**2
pdyn = data['pdyn[nPa]'].to_numpy()
bz_max = np.max(bz)
bz_min = np.min(bz)
theta = np.arctan2(ypos,xpos)
r = np.sqrt(xpos**2 + ypos**2)

# #Some incomplete OMNI IMF data made its way into the dataset. This cut removes the associated crossings.
bool_cut = (bz < 900) & (by < 900) & (bx < 900)
bx = bx[bool_cut]
by = by[bool_cut]
bz = bz[bool_cut]
xpos = xpos[bool_cut]
ypos = ypos[bool_cut]
bt = bt[bool_cut]
clock = clock[bool_cut]
sin_rec = sin_rec[bool_cut]
pdyn = pdyn[bool_cut]
theta = theta[bool_cut]
r = r[bool_cut]

##################################################################

# #Define tractrix functions, as well as bayesian quantities for MCMC procedure

def standoff(sin_rec, pdyn, a0, a1, a2):
    """Returns standoff position (s/w) given IMF sine rectifier and dynamic pressure. a0/1/2 are tuning parameters (s0/1/2, w0/1/2)"""
    d = (a0 + a1 * sin_rec) * (pdyn ** (-1 / a2))
    return d

def tractrix(y, s, w):
    """The overall form of the tractrix model. Returns GSE X position given GSE Y, subsolar distance, and tail width."""
    x = s - w * np.log((w + np.sqrt(np.abs(w ** 2 - (w - y) ** 2))) / np.abs(w - y)) + np.sqrt(np.abs(w**2 - (w - y ) ** 2))
    return x

def log_like(theta):
    """Natural log of the likelihood of a set of parameters theta."""
    s0,s1,s2,w0,w1,w2,sig = theta
    trac_residual = np.zeros(len(xpos))
    w = standoff(sin_rec, pdyn, w0, w1, w2)
    s = standoff(sin_rec, pdyn, s0, s1, s2)
    if np.any(w <= 1):
        return -np.inf
    y_array = np.linspace(0,w-1,100)
    trac_x = tractrix(y_array, s, w)
    trac_residual = np.amin(np.sqrt((trac_x - xpos)**2 + (y_array - ypos)**2),axis = 0)
    return -0.5 * np.sum((trac_residual) ** 2 / (sig ** 2) + np.log(sig ** 2))

def log_prior(theta):
    """Natural log of the prior of a set of parameters theta."""
    s0,s1,s2,w0,w1,w2,sig = theta
    if (1 < s2 < 10) & (1 < w2 < 20) & (np.abs(s0) < 50) & (np.abs(s1) < 1)  & (np.abs(w0) < 50) & (np.abs(w1) < 2):
        return 0
    else:
        return -np.inf

def log_prob(theta):
    """Natural log of the combined prior/likelihood of a set of parameters theta."""
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_like(theta)

##################################################################

# #Ensure the same random seed is chosen for each run for ease of debugging
np.random.seed(88)

# #Initial walker position in parameter space (Small Gaussian ball around initial guess)
init = np.array([14.8186356, -0.5, 6.47473211, 25.4841683, -0.5, 10.2272961, 1.18]) + 1e-4 * np.random.randn(105 , 7)

if __name__ == "__main__":
    pool = schwimmbad.choose_pool(processes=8) #Adjust "processes" number based on CPU threads
    nwalkers, ndim = init.shape
    
    n_maxsteps = 70000 #Define how long the chain is to be iterated
    savename = 'mcmc_savestate.h5' #Define name of backend to be loaded
    backend = emcee.backends.HDFBackend(savename)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, pool = pool, backend=backend) #Initializes the sampler with defined functions, initial guess, and backend
    sampler.run_mcmc(init, n_maxsteps, progress = True) #Runs for n_maxsteps iterations



