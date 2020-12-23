# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 10:39:00 2020

    Script for loading MCMC runs stored as HDF5 files from mcmc.py. 

@author: connor o'brien
"""
import numpy as np
import pandas as pd
import emcee
import corner
import matplotlib.pyplot as plt

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


savename = "mcmc_savestate.h5" #Name of backend to be loaded
backend = emcee.backends.HDFBackend(savename)
saved_sampler = emcee.EnsembleSampler(105, 7, log_prob, backend = backend) #Unpacks the backend into a sampler object


tau = saved_sampler.get_autocorr_time(quiet = True) #Calculate autorcorrelation time
print(tau)
burnin = int(2 * np.max(tau)) #Use tau to calculate initial iterations to discard
print(burnin)
thin = int(0.5 * np.min(tau)) #Use tau to calculate thinning factor
print(thin)
samples = saved_sampler.get_chain(discard=burnin, flat=True, thin=thin) #Discard and thin the sampler
labels = [r'$s_{0}$', r'$s_{1}$', r'$s_{2}$', r'$w_{0}$', r'$w_{1}$', r'$w_{2}$', '$\sigma$']
bins = [100, 100, 60, 120, 100, 80, 120] #Manually adjust binning factor
cornerplot = corner.corner(samples, labels= labels, bins = bins) #Initialize the cornerplot

theta = np.zeros(len(samples[0,:])) #Initialize parameter array
for i in range(len(samples[0,:])):
    max_temp, bin_edges = np.histogram(samples[:,i], bins[i]) #Quick make a histogram just to grab the max
    theta[i] = (bin_edges[np.argmax(max_temp)]+bin_edges[np.argmax(max_temp)+1])/2 # maximum likelihood estimate
newparams = theta #Name change because this is two codes stitched together lol
ndim = len(newparams)
sigs = np.std(samples, axis = 0) #Calculate uncertainties
bounds = np.zeros((ndim, 2)) #Initialize bounds array
percentile = 0.1 #Percentile bounds for creating bounds array
color = '#E57C04' #Color for maximum likelihood estimates
axes = np.array(cornerplot.axes).reshape((ndim,ndim)) #Grab the axes

for i in range(ndim):
    ax = axes[i,i] #Isolate on-axis histograms (1D)
    ax.axvline(newparams[i], color = color) #Plot the maximum likelihood
    bounds[i] = np.percentile(samples[:,i], [percentile,100-percentile]) #Calculate bounds
    ax.set_xlim(bounds[i,0], bounds[i,1]) #apply bounds
    
for i in range(ndim):
    axes[i,i].grid(False) #remove the gridlines from the on-axis hists (why is this not in the prior loop?)
    for j in range(i):
        ax = axes[i, j]
        ax.axvline(newparams[j], color=color) #Plot the maximum likelihood
        ax.axhline(newparams[i], color=color)
        ax.plot(newparams[j], newparams[i], marker = 's', color = color)
        ax.set_ylim(bounds[i,0], bounds[i,1]) # Apply bounds, if statements for manual adjustments
        ax.set_xlim(bounds[j,0], bounds[j,1])
        if j == 3:
            ax.set_xlim(31.8, 32.7)
        if i == 3:
            ax.set_ylim(31.8, 32.7)
        if j == 4:
            ax.set_xlim(-0.32, -0.11)
        if i == 4:
            ax.set_ylim(-0.32, -0.11)
        if i == ndim-1:
            ax.set_ylim(1.85, 1.97)
        if j == ndim-1:
            ax.set_xlim(1.85, 1.97)
axes[3,3].set_xlim(31.8, 32.7)
axes[4,4].set_xlim(-0.32, -0.11)
axes[ndim-1,ndim-1].set_xlim(1.85, 1.97)

plt.savefig('corner_annotated_linear.pdf', format = 'pdf')
plt.savefig('corner_annotated_linear.png')