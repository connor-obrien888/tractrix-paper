# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 16:24:15 2020

    Script that creates most* shown in O'Brien et al., 2021. All plots save as png and pdf.
    * except Figure 1, which is constructed by SW_pressure.py, Figure 2, which was drawn in Adobe 
    Illustrator, and Figure 5, which is constructed by mcmc_loader.py

@author: connor o'brien
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def standoff(sin_rec, pdyn, a0, a1, a2):
    """Returns standoff position (s/w) given IMF sine rectifier and dynamic pressure. a0/1/2 are tuning parameters (s0/1/2, w0/1/2)"""
    d = (a0 + a1 * sin_rec) * (pdyn ** (-1 / a2))
    return d

def tractrix(y, s, w):
    """The overall form of the tractrix model. Returns GSE X position given GSE Y, subsolar distance, and tail width."""
    x = s - w * np.log((w + np.sqrt(np.abs(w ** 2 - (w - y) ** 2))) / np.abs(w - y)) + np.sqrt(np.abs(w**2 - (w - y ) ** 2))
    return x

def shue1998model(theta, bz, pdyn):
    """Shue et al. (1998) model functional form. Returns radial standoff distance r (2D)"""
    r0 = (10.22 + 1.29 * np.tanh(0.184 * (bz + 8.14))) * (pdyn ** (-1 / 6.6))
    alph = (0.58 - 0.007 * bz) * (1 + 0.024 * np.log(pdyn))
    return r0 * (2 / (1 + np.cos(theta))) ** alph

def linmodel(theta, phi, bz, pdyn, tilt, a0 = 12.544, a1 = -0.194, a2 = 0.305, a3 = 0.0573, a4 = 2.178, 
             a5 = 0.0571, a6 = -0.999, a7 = 16.473, a8 = 0.00152, a9 = 0.382, a10 = 0.0431, 
             a11 = -0.00763 , a12 = -0.210, a13 = 0.0405, a14 = -4.430, a15 = -0.636, a16 = -2.600, 
             a17 = 0.832, a18 = -5.328, a19 = 1.103, a20 = -0.907, a21 = 1.450):
    """Lin et al. (2011) model functional form. Returns radial standoff distance r (3D)"""
    r0 = a0 * (pdyn ** a1) * (1 + a2 * (np.exp(a3 * bz) -1) / (np.exp(a4 * bz) + 1))
    m = a5
    bet0 = a6 + a7 * (np.exp(a8 * bz) - 1) / (np.exp(a9 * bz)+1)
    bet1 = a10
    bet2 = a11 + a12 * tilt
    bet3 = a13
    bet = bet0 + bet1 * np.cos(phi) + bet2 * np.sin(phi) + bet3 * (np.sin(phi))**2
    c = a14 * (pdyn) ** a15
    dn = a16 + a17 * tilt + a18 * (tilt ** 2)
    ds = a16 - a17 * tilt + a18 * (tilt ** 2)
    tn = a19 + a20 * tilt
    ts = a19 - a20 * tilt
    pn = np.arccos(np.cos(theta) * np.cos(tn) + np.sin(theta) * np.sin(tn) * np.cos(phi - 0.5 * np.pi))
    ps = np.arccos(np.cos(theta) * np.cos(ts) + np.sin(theta) * np.sin(ts) * np.cos(phi - 1.5 * np.pi))
    e = a21
    Q = c * np.exp(dn * (pn ** e)) + c * np.exp(ds * (ps ** e))
    r = r0 * (np.cos(theta / 2) + m * np.sin(2 * theta) * (1 - np.exp(-1 * theta))) ** bet + Q
    return r

def petrusmodel(theta, x, bz, pdyn, dayside):
    """Petrinec and Russel (1996) model functional form. Split dayside/nightside via 'dayside' bool"""
    if bz < 0:
        m1 = 0.16
        m2 = 0.00644
    else:
        m1 = 0
        m2 = 0.00137
    if dayside:
        r = 14.63 * (pdyn / 2.1) ** (-1 / 6) / (1 + (14.63 / (10.3 + m1 * bz) -1) * np.cos(theta))
        return r
    else:
        R = (-2 / (0.085)) * (np.arcsin((2.98 * (pdyn ** (-0.524)) * np.exp(0.085 * x) * (0.152 - m2 * bz))**(0.5)) - np.arcsin((2.98 * (pdyn ** (-0.524)) * (0.152 - m2 * bz))**(0.5))) + 14.63 * (pdyn / 2.1) ** (-1 / 6)
        return R

def chaomodel(theta, bz, pdyn):
    """Chao et al. (2002) model functional form. Returns radial standoff distance r (2D). Does not accept Bz in array form"""
    if bz >= 0:
        r0 = 12.04 * pdyn ** (-1 /7.255)
    if (bz < 0) & (bz >= -8):
        r0 = (12.04 + 0.120 * bz)* pdyn ** (-1 / 7.255)
    if bz < -8:
        r0 = (12.04 + 8 * 0.250 - 8 * 0.120 + 0.250 * bz)* pdyn ** (-1 / 7.255)
    alpha = (0.578 -0.009 * bz)*(1 + 0.012 * pdyn)
    return r0 * ( 2 / (1 + np.cos(theta))) ** alpha

def lumodel(theta, phi, bz, pdyn):
    """Lu et al. (2010) model funtional form. Returns radial standoff diatance r (3D). Does not accept Bz in array form"""
    if bz >= 0:
        r0 = (11.494 + 0.0371 * bz) * (pdyn ** (-1 / 5.2))
        rc = (-1.191 - 0.034 * bz) * (pdyn ** (-1 / 5.2))
        alph = 0.543 - 0.0225 * bz + 0.00528 * pdyn + 0.00261 * bz * pdyn
        bet1 = -0.263 + 0.0045 * bz - 0.00924 * pdyn - 0.00059 * bz * pdyn
        bet2 = 0.0924 + 0.0121 * bz - 0.00115 * pdyn - 0.00115 * bz * pdyn
    else:
        r0 = (11.494 + 0.0983 * bz) * (pdyn ** (-1 / 5.2))
        rc = (-1.191 - 0.189 * bz) * (pdyn ** (-1 / 5.2))
        alph = 0.543 - 0.0079 * bz + 0.00528 * pdyn + 0.00019 * bz * pdyn
        bet1 = -0.263 + 0.0259 * bz - 0.00924 * pdyn - 0.00256 * bz * pdyn
        bet2 = 0.0924 + 0.0069 * bz - 0.00115 * pdyn - 0.00115 * bz * pdyn
    if theta <= np.pi / 2:
        r = r0 * (2 / (1 + np.cos(theta))) ** (alph + bet1 * (np.cos(phi)) ** 2)
    else:
        r = (r0 + rc * (np.cos(phi)) ** 2) * (2 / (1 + np.cos(theta))) ** (alph + bet2 * (np.cos(phi)) ** 2)
    return r

######################
# #Maximum likelihood parameters
s0, s1, s2 =  14.56437116, -0.03976978, 5.6989598
w0, w1, w2 =  32.33886118, -0.24718201, 12.23569415

# #Solar wind parameter arrays for equistandoff/width plots
lin_bs = np.linspace(0, 20, num=1000)
lin_pdyn = np.linspace(0.001, 20, num=1500)

# #Average parameter values for model spatial comparison
av_sin_rec = 2.58
av_bz = -0.049
av_pdyn = 2.34
av_tilt = 0.10536450417767909

# #Standoff array for equistandoff/width plots
s_arr = np.zeros((1000,1500))

# #Width array for equistandoff/width plots
w_arr = np.zeros((1000,1500))

######################
# #Equistandoff Plot
######################  
# #Fill standoff array with values of s
for j in range(1500):
    s_arr[:,j] = standoff(lin_bs, lin_pdyn[j], s0, s1, s2)
  
levels_n = np.arange(8.5,13.5, step = 0.5)

cont_s = plt.contour(lin_pdyn, lin_bs, s_arr, levels = levels_n, cmap = "cividis") #Plot the contours
labels_pos_s = [(18,12.5),(13,10.5),(11,10.5),(8.5,10),(6.5,9),(5,8.5),(3.5,5),(2.8,8.5),(2.3,5),(1.7,8.5)] #Manually adjust label locations
plt.ylim(0, 20)
plt.xlim(0,20)
plt.clabel(cont_s, inline = 1, inline_spacing = 1, manual = labels_pos_s) #Plot custom labels
plt.xlabel('$P_{dyn}$ (nPa)') #Write axes labels, figure title
plt.ylabel('$B_{S}$ (nT)')
plt.title('Lines of Equal Standoff Distance')
plt.tight_layout()
plt.savefig('equistandoff.png') #Save the plot in a few formats
plt.savefig('equistandoff.pdf', format = 'pdf')
plt.clf() #Clear figure for next set of contours
######################
# #Equiwidth Plot
######################
# #Fill width array with values of w
for j in range(1500):
    w_arr[:,j] = standoff(lin_bs, lin_pdyn[j], w0, w1, w2)
    
levels_n = np.arange(22,31) #Levels of the contours

cont_w = plt.contour(lin_pdyn, lin_bs, w_arr, levels = levels_n, cmap = "cividis") #plot the contours
labels_pos_w = [(18.5,19),(15,17),(13,13),(11,9),(7.5,7.5),(5,7.5),(3.5,6),(3,4),(2,3)] #Manually adjust label locations
plt.clabel(cont_w, inline = 1, inline_spacing = 1, manual = labels_pos_w) #plot custom labels

plt.ylim(0, 20) #Adjust plot bounds
plt.xlim(0,20)
plt.xlabel('$P_{dyn}$ (nPa)') #Write axes labels, figure title
plt.ylabel('$B_{S}$ (nT)')
plt.title('Lines of Equal Tail Width')
plt.tight_layout()
plt.savefig('equiwidth.png') #Save in a couple formats
plt.savefig('equiwidth.pdf', format = 'pdf')
plt.clf() #Clear figure
#######################
# #Model Spatial Comparison
#######################
# #Make input arrays
theta = np.linspace(0,np.pi-0.01, 5000)
y = np.linspace(0, standoff(av_sin_rec, av_pdyn, w0, w1, w2)-0.01, 1000)

# #Special ones for Petrinec and Russel [1996]
pr_theta = np.linspace(0,np.pi/2, 500)
pr_xarr = np.linspace(-15,-200,1000)

#Plot each model in the nose (0) and tail (1)
for j in np.arange(2):
    plt.subplot(1,2,j+1)
    
    # #Shue 1998
    shue_r = shue1998model(theta, av_bz, av_pdyn)
    shue_x = shue_r * np.cos(theta)
    shue_y = shue_r * np.sin(theta)
    plt.plot(shue_x, shue_y, color = '#4477aa')
    
    # #Chao 2002
    chao_r = chaomodel(theta, av_bz, av_pdyn)
    chao_x = chao_r * np.cos(theta)
    chao_y = chao_r * np.sin(theta)
    plt.plot(chao_x, chao_y, color = '#66ccee')
    
    # #Petrinec and Russell 1996
    pr_r = petrusmodel(pr_theta, 0, av_bz, av_pdyn, True)
    pr_x = pr_r * np.cos(pr_theta)
    pr_y = pr_r * np.sin(pr_theta)
    pr_R = petrusmodel(0, pr_xarr, av_bz, av_pdyn, False)
    pr_x = np.append(pr_x, pr_xarr)
    pr_y = np.append(pr_y, pr_R)
    plt.plot(pr_x, pr_y, '#ccbb44')
    
    # #Lu et al 2011
    lu_r = np.zeros(len(theta))
    for i in np.arange(len(theta)):
        lu_r[i] = lumodel(theta[i], np.pi/2, av_bz, av_pdyn)
    lu_x = lu_r * np.cos(theta)
    lu_y = lu_r * np.sin(theta)
    plt.plot(lu_x, lu_y, '#ee6677')
    
    # #Lin et al 2010
    lin_r = linmodel(theta, np.pi/2, av_bz, av_pdyn, av_tilt)
    lin_x = lin_r * np.cos(theta)
    lin_y = lin_r * np.sin(theta)
    plt.plot(lin_x, lin_y, '#aa3377')
    
    # #Tractrix
    trac_x = tractrix(y,standoff(av_sin_rec, av_pdyn, s0, s1, s2),standoff(av_sin_rec, av_pdyn, w0, w1, w2))
    plt.plot(trac_x, y, color = '#228833')
    
    
    # #Line
    angle = 17.5
    line_x = np.linspace(-5,13, 100)
    line_y = np.tan(angle * 2*np.pi / 360) * line_x
    plt.plot(line_x, line_y, 'k--')
    
    if j == 0:
        plt.xlim(13,-5)
        plt.ylim(0,17)
        plt.gca().set_aspect('equal', adjustable='box') #Make plot isotropic
        plt.xlabel(r'GSE X ($R_{E}$)')
        plt.ylabel(r'GSE Y ($R_{E}$)')
        plt.title('Nose Spatial Structure')
    if j == 1:
        plt.xlim(-40,-80)
        plt.ylim(0,38)
        plt.gca().set_aspect('equal', adjustable='box') #Make plot isotropic
        plt.xlabel(r'GSE X ($R_{E}$)')
        plt.title('Tail Spatial Structure')
        labels = ['Shue', 'Chao', 'P&R', 'Lu', 'Lin','Tractrix']
        plt.legend(labels, frameon = False, prop={'size' :8})
plt.savefig('struct_combo.png')
plt.savefig('struct_combo.pdf', format = 'pdf')
plt.clf()

#######################
# #Near-Earth Uncertainties
#######################

data = pd.read_csv('validation_list_20200715.csv', sep = ',', index_col=0) #Load the validation dataset

# #Split the validation dataset into its components, and restrict them to near the Earth
xpos = data['xpos']
ypos = np.sqrt(data['ypos'] ** 2 + data['zpos'] ** 2)
data['r[re]'] = np.sqrt(data['xpos'] ** 2 + data['ypos'] ** 2 + data['zpos'] ** 2)
mod_angle = np.arccos(data['xpos']/data['r[re]'])
reduced_xpos = xpos[data['xpos'] >= -16].to_numpy()
reduced_ypos = ypos[data['xpos'] >= -16].to_numpy()
reduced_bz = data['bz[nT]'][data['xpos'] >= -16].to_numpy()
reduced_by = data['by[nT]'][data['xpos'] >= -16].to_numpy()
reduced_bx = data['bx[nT]'][data['xpos'] >= -16].to_numpy()
reduced_bt = np.sqrt( reduced_bx**2 + reduced_by**2 + reduced_bz**2)
reduced_clock = np.arctan2(reduced_by , reduced_bz)
reduced_sin_rec = reduced_bt * (np.sin(reduced_clock / 2))**2
reduced_pdyn = data['pdyn[nPa]'][data['xpos'] >= -16].to_numpy()
reduced_mod_angle = mod_angle[data['xpos'] >= -16].to_numpy()
reduced_x_actual = reduced_xpos
reduced_y_actual = data['ypos'][data['xpos'] >= -16].to_numpy()
reduced_z_actual = data['zpos'][data['xpos'] >= -16].to_numpy()
reduced_tilt = 0.10536450417767909

# #Cut out incomplete OMNI data
bool_cut = ((reduced_bz < 900) & (reduced_by < 900) & (reduced_bx < 900))
reduced_bx = reduced_bx[bool_cut]
reduced_by = reduced_by[bool_cut]
reduced_bz = reduced_bz[bool_cut]
reduced_xpos = reduced_xpos[bool_cut]
reduced_ypos = reduced_ypos[bool_cut]
reduced_bt = reduced_bt[bool_cut]
reduced_clock = reduced_clock[bool_cut]
reduced_sin_rec = reduced_sin_rec[bool_cut]
reduced_pdyn = reduced_pdyn[bool_cut]
reduced_mod_angle = reduced_mod_angle[bool_cut]

#######################
# #Residual Calculators
#######################
def twodrescalc(model_residual, bin_number):
    model_sorted = np.transpose(np.asarray([reduced_xpos, model_residual]))
    model_sorted = model_sorted[model_sorted[:,0].argsort()]   
    bins = np.linspace(np.min(reduced_xpos), np.max(reduced_xpos), bin_number+1)
    model_digitized = np.digitize(model_sorted[:,0], bins)
    model_binned_mean = np.array([model_sorted[model_digitized == i, 1].mean() for i in range(1, len(bins))])
    binhist = np.bincount(model_digitized)
    return model_binned_mean, binhist

# #Tractrix Model
#######################
def tractrixresidual():
    trac_residual = np.zeros(len(reduced_xpos))
    closest_points = np.zeros((len(reduced_xpos), 2))
    for i in np.arange(len(reduced_xpos)):
        #width = chao_standoff(reduced_bz[i], reduced_pdyn[i], w1, w2, w3, w4)
        #standoff = chao_standoff(reduced_bz[i], reduced_pdyn[i], s1, s2, s3, s4)
        w = standoff(reduced_sin_rec[i], reduced_pdyn[i], w0, w1, w2)
        s = standoff(reduced_sin_rec[i], reduced_pdyn[i], s0, s1, s2)
        y_array = np.linspace(0,w-0.00001,10000)
        trac_x = tractrix(y_array, s, w)    
        trac_residual[i] = np.amin(np.sqrt((trac_x - reduced_xpos[i])**2 + (y_array - reduced_ypos[i])**2))
        arg = np.argmin(np.sqrt((trac_x - reduced_xpos[i])**2 + (y_array - reduced_ypos[i])**2))
        closest_points[i, :] = [trac_x[arg],y_array[arg]]
    return trac_residual, closest_points, 'Tractrix', '#228833'

# #Chao Model
#####################
def chaoresidual():
    model_name = 'Chao'
    model_color = '#66ccee'
    model_residual = np.zeros(len(reduced_xpos))
    closest_points = np.zeros((len(reduced_xpos), 2))
    for i in np.arange(len(reduced_xpos)):
        theta_array = np.linspace(0, np.max(reduced_mod_angle), 1000)
        chao_r = chaomodel(theta_array, reduced_bz[i], reduced_pdyn[i])
        chao_x = chao_r * np.cos(theta_array)
        chao_y = chao_r * np.sin(theta_array)
        model_residual[i] = np.amin(np.sqrt((chao_x - reduced_xpos[i])**2 + (chao_y - reduced_ypos[i])**2))
        arg = np.argmin(np.sqrt((chao_x - reduced_xpos[i])**2 + (chao_y - reduced_ypos[i])**2))
        closest_points[i, :] = [chao_x[arg],chao_y[arg]]
    return model_residual, closest_points, model_name, model_color
# #####################

# #Lin Model
#######################
def linresidual():    
    model_name = 'Lin'
    model_color = '#aa3377'
    closest_points = np.zeros((len(reduced_xpos), 3))
    model_residual = np.zeros(len(reduced_xpos))
    for i in np.arange(len(reduced_xpos)):
        theta_array = np.linspace(0, np.max(reduced_mod_angle), 500)
        phi_array = np.linspace(0, 2 * np.pi, 360)
        x_arr = np.zeros(1)
        y_arr = np.zeros(1)
        z_arr = np.zeros(1)
        for j in np.arange(len(theta_array)):
            r = linmodel(theta_array[j], phi_array, reduced_bz[i], reduced_pdyn[i], reduced_tilt[i])
            x_arr = np.append(x_arr, r * np.cos(theta_array[j]))
            y_arr = np.append(y_arr, r * np.cos(0.5 * np.pi - theta_array[j]) * np.cos(2 * np.pi - phi_array))
            z_arr = np.append(z_arr, r * np.cos(0.5 * np.pi - theta_array[j]) * np.sin(2 * np.pi - phi_array))
        model_residual[i] = np.amin(np.sqrt((x_arr[1:] - reduced_x_actual[i])**2 + (y_arr[1:] - reduced_y_actual[i])**2 + (z_arr[1:] - reduced_z_actual[i])**2))
    return model_residual[bool_cut], closest_points, model_name, model_color
######################

# #Lu Model
#######################
def luresidual():    
    model_name = 'Lu'
    model_color = '#ee6677'
    model_residual = np.zeros(len(reduced_xpos))
    closest_points = np.zeros((len(reduced_xpos), 3))
    for i in np.arange(len(reduced_xpos)):
        theta_array = np.linspace(0, np.max(reduced_mod_angle), 500)
        phi_array = np.linspace(0, 2 * np.pi, 360)
        x_arr = np.zeros(1)
        y_arr = np.zeros(1)
        z_arr = np.zeros(1)
        for j in np.arange(len(theta_array)):
            r = lumodel(theta_array[j], phi_array, reduced_bz[i], reduced_pdyn[i])
            x_arr = np.append(x_arr, r * np.cos(theta_array[j]))
            y_arr = np.append(y_arr, r * np.cos(0.5 * np.pi - theta_array[j]) * np.cos(2 * np.pi - phi_array))
            z_arr = np.append(z_arr, r * np.cos(0.5 * np.pi - theta_array[j]) * np.sin(2 * np.pi - phi_array))
        model_residual[i] = np.amin(np.sqrt((x_arr[1:] - reduced_x_actual[i])**2 + (y_arr[1:] - reduced_y_actual[i])**2 + (z_arr[1:] - reduced_z_actual[i])**2))
        arg = np.argmin(np.sqrt((x_arr[1:] - reduced_x_actual[i])**2 + (y_arr[1:] - reduced_y_actual[i])**2 + (z_arr[1:] - reduced_z_actual[i])**2))
        closest_points[i, :] = [x_arr[arg],y_arr[arg], z_arr[arg]]
    return model_residual, closest_points, model_name, model_color
######################


# #Shue 1998 Model
#####################
def shue1998residual():
    model_name = 'Shue'
    model_color = '#4477aa'
    model_residual = np.zeros(len(reduced_xpos))
    closest_points = np.zeros((len(reduced_xpos), 2))
    for i in np.arange(len(reduced_xpos)):
        theta_array = np.linspace(0, np.max(reduced_mod_angle), 1000)
        shue_r = shue1998model(theta_array, reduced_bz[i], reduced_pdyn[i])
        shue_x = shue_r * np.cos(theta_array)
        shue_y = shue_r * np.sin(theta_array)
        model_residual[i] = np.amin(np.sqrt((shue_x - reduced_xpos[i])**2 + (shue_y - reduced_ypos[i])**2))
        arg = np.argmin(np.sqrt((shue_x - reduced_xpos[i])**2 + (shue_y - reduced_ypos[i])**2))
        closest_points[i, :] = [shue_x[arg],shue_y[arg]]
    return model_residual, closest_points, model_name, model_color
####################

# #Petrinec and Russell Model
#####################
def petrusresidual():
    model_name = 'P&R'
    model_color = '#ccbb44'
    model_residual = np.zeros(len(reduced_xpos))
    closest_points = np.zeros((len(reduced_xpos), 2))
    for i in np.arange(len(reduced_xpos)):
        theta_array = np.linspace(0, np.pi / 2, 500)
        x_array = np.linspace(np.amin(reduced_xpos), 0, 500)
        pr_r = np.zeros(500)
        pr_R = np.zeros(500)
        for j in range(500):
            pr_r[j] = petrusmodel(theta_array[j], x_array[j], reduced_bz[i], reduced_pdyn[i], True)
        pr_x = pr_r * np.cos(theta_array)
        pr_y = pr_r * np.sin(theta_array)
        for j in range(500):
            pr_R[j] = petrusmodel(theta_array[j], x_array[j], reduced_bz[i], reduced_pdyn[i], False)
        pr_x = np.append(pr_x, x_array)
        pr_y = np.append(pr_y, pr_R)
        model_residual[i] = np.amin(np.sqrt((pr_x - reduced_xpos[i])**2 + (pr_y - reduced_ypos[i])**2))
        arg = np.argmin(np.sqrt((pr_x - reduced_xpos[i])**2 + (pr_y - reduced_ypos[i])**2))
        if np.isnan(model_residual[i]):
            model_residual[i] = 0
            arg = 0
        closest_points[i, :] = [pr_x[arg],pr_y[arg]]
    return model_residual, closest_points, model_name, model_color
#######################

bin_number = 30
bins = np.linspace(np.min(reduced_xpos), np.max(reduced_xpos), bin_number+1)

trac_residual, trac_points, trac_name, trac_color = tractrixresidual()
shue_residual, shue_points, shue_name, shue_color = shue1998residual()
chao_residual, chao_points, chao_name, chao_color = chaoresidual()
petrus_residual, petrus_points, petrus_name, petrus_color = petrusresidual()
lu_residual, lu_points, lu_name, lu_color = luresidual()
lin_residual, lin_points, lin_name, lin_color = linresidual()


trac_binned_mean, binhist = twodrescalc(trac_residual, bin_number)
shue_binned_mean, binhist = twodrescalc(shue_residual, bin_number)
chao_binned_mean, binhist = twodrescalc(chao_residual, bin_number)
petrus_binned_mean, binhist = twodrescalc(petrus_residual, bin_number)
lu_binned_mean, binhist = twodrescalc(lu_residual, bin_number)
lin_binned_mean, binhist = twodrescalc(lin_residual, bin_number)

# #Make Near-Earth residual/uncertainty comparison plots
ax1 = plt.subplot(511)
plt.bar(bins[0:-1], shue_binned_mean, width=1, color = shue_color)
plt.bar(bins[0:-1], trac_binned_mean, width=1, fill = False, edgecolor = 'k')
plt.ylim(0,3)
plt.xlim(bins[-2], bins[0])
plt.ylabel(shue_name)
plt.title('Model Uncertainty, Near-Earth Magnetopause Crossings')
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

ax2 = plt.subplot(512, sharex = ax1)
plt.bar(bins[0:-1], chao_binned_mean, width=1, color = chao_color)
plt.bar(bins[0:-1], trac_binned_mean, width=1, fill = False, edgecolor = 'k')
plt.ylim(0,3)
plt.ylabel(chao_name)
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

ax3 = plt.subplot(513, sharex = ax1)
plt.bar(bins[0:-1], petrus_binned_mean, width=1, color = petrus_color)
plt.bar(bins[0:-1], trac_binned_mean, width=1, fill = False, edgecolor = 'k')
plt.ylim(0,3)
plt.ylabel(petrus_name)
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

ax4 = plt.subplot(514, sharex = ax1)
plt.bar(bins[0:-1], lu_binned_mean, width=1, color = lu_color)
plt.bar(bins[0:-1], trac_binned_mean, width=1, fill = False, edgecolor = 'k')
plt.ylim(0,3)
plt.ylabel(lu_name)
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

ax5 = plt.subplot(515, sharex = ax1)
plt.bar(bins[0:-1], lin_binned_mean, width=1, color = lin_color)
plt.bar(bins[0:-1], trac_binned_mean, width=1, fill = False, edgecolor = 'k')
plt.ylim(0,3)
plt.ylabel(lin_name)
plt.xlabel('GSE X ($R_{e}$)')


plt.savefig('NE_residual_comp.png')
plt.savefig('NE_residual_comp.pdf', format = 'pdf')
plt.clf()

#######################
# #Make tractrix uncertainty plot
fig, ax1 = plt.subplots()

w = 0.4*(np.max(reduced_xpos) - np.min(reduced_xpos))/bin_number

ax1.set_xlabel(r'GSE X ($R_{E}$)')
ax1.set_ylabel(r'Uncertainty ($R_{E}$)')
ax1.set_title('Tractrix Uncertainty, Near-Earth Magnetopause Crossings')
ax1.bar(bins[0:-1], trac_binned_mean, color=trac_color)
ax1.set_ylim(0,2)
ax1.hlines([0.5,1,1.5], -15, 15, colors = 'grey', linestyles='dashed', alpha = 0.5)
ax1.set_xlim(bins[-2], bins[0])

ax2 = ax1.twinx()

ax2.set_ylabel('Crossings in Bin', color = 'k')
ax2.tick_params(labelcolor = 'k')
ax2.set_ylim(0,np.amax(binhist[1:-1]))
ax2.plot(bins[0:-1], binhist[1:-1], color = 'k')


plt.savefig('tractrix_residual.pdf', format = 'pdf')
plt.savefig('tractrix_residual.png')
plt.clf()

#######################
# #Tail uncertainty comparison
#######################
# #Load validation data back in
data = pd.read_csv('validation_list_20200715.csv', sep = ',', index_col=0)

# #Split the validation dataset into its components, and restrict them to the magnetotail
xpos = data['xpos']
ypos = np.sqrt(data['ypos'] ** 2 + data['zpos'] ** 2)
data['r[re]'] = np.sqrt(data['xpos'] ** 2 + data['ypos'] ** 2 + data['zpos'] ** 2)
mod_angle = np.arccos(data['xpos']/data['r[re]'])
reduced_xpos = xpos[data['xpos'] <= -16].to_numpy()
reduced_ypos = ypos[data['xpos'] <= -16].to_numpy()
reduced_bz = data['bz[nT]'][data['xpos'] <= -16].to_numpy()
reduced_by = data['by[nT]'][data['xpos'] <= -16].to_numpy()
reduced_bx = data['bx[nT]'][data['xpos'] <= -16].to_numpy()
reduced_bt = np.sqrt( reduced_bx**2 + reduced_by**2 + reduced_bz**2)
reduced_clock = np.arctan2(reduced_by , reduced_bz)
reduced_sin_rec = reduced_bt * (np.sin(reduced_clock / 2))**2
reduced_pdyn = data['pdyn[nPa]'][data['xpos'] <= -16].to_numpy()
reduced_mod_angle = mod_angle[data['xpos'] <= -16].to_numpy()
reduced_x_actual = reduced_xpos
reduced_y_actual = data['ypos'][data['xpos'] <= -16].to_numpy()
reduced_z_actual = data['zpos'][data['xpos'] <= -16].to_numpy()
reduced_tilt = 0.10536450417767909

# #Cut out incomplete IMF data
bool_cut = (reduced_bz < 900) & (reduced_by < 900) & (reduced_bx < 900)

reduced_bx = reduced_bx[bool_cut]
reduced_by = reduced_by[bool_cut]
reduced_bz = reduced_bz[bool_cut]
reduced_xpos = reduced_xpos[bool_cut]
reduced_ypos = reduced_ypos[bool_cut]
reduced_bt = reduced_bt[bool_cut]
reduced_clock = reduced_clock[bool_cut]
reduced_sin_rec = reduced_sin_rec[bool_cut]
reduced_pdyn = reduced_pdyn[bool_cut]
reduced_mod_angle = reduced_mod_angle[bool_cut]
#######################
bin_number = 23
bins = np.linspace(np.min(reduced_ypos), np.max(reduced_ypos), bin_number+1)

trac_residual, trac_points, trac_name, trac_color = tractrixresidual()
shue_residual, shue_points, shue_name, shue_color = shue1998residual()
chao_residual, chao_points, chao_name, chao_color = chaoresidual()
petrus_residual, petrus_points, petrus_name, petrus_color = petrusresidual()
lu_residual, lu_points, lu_name, lu_color = luresidual()
lin_residual, lin_points, lin_name, lin_color = linresidual()

trac_binned_mean, binhist = twodrescalc(trac_residual, bin_number)
shue_binned_mean, binhist = twodrescalc(shue_residual, bin_number)
chao_binned_mean, binhist = twodrescalc(chao_residual, bin_number)
petrus_binned_mean, binhist = twodrescalc(petrus_residual, bin_number)
lu_binned_mean, binhist = twodrescalc(lu_residual, bin_number)
lin_binned_mean, binhist = twodrescalc(lin_residual, bin_number)

#######################
# #Tractrix Uncertainty plot
#######################
fig, ax1 = plt.subplots()
w = 0.4*(np.max(reduced_ypos) - np.min(reduced_ypos))/bin_number

ax1.set_xlabel(r'GSE $\rho$ ($R_{e}$)')
ax1.set_ylabel(r'Uncertainty ($R_{e}$)')
ax1.set_title(r'Tractrix Uncertainty, ARTEMIS Magnetopause Crossings')
ax1.bar(bins[0:-1], trac_binned_mean, color=trac_color)
ax1.set_xlim(bins[0]-0.5, bins[-5]+0.5)

ax2 = ax1.twinx()

ax2.set_ylabel('Crossings in Bin', color = 'k')
ax2.tick_params(labelcolor = 'k')
ax2.plot(bins[0:-1], binhist[1:-1], color = 'k')

plt.savefig('tail_residual_trac.png')
plt.savefig('tail_residual_trac.pdf', format = 'pdf')
plt.clf()

#######################
# #Model uncertainty comparison plots
#######################
ax1 = plt.subplot(511)
plt.bar(bins[0:-1], shue_binned_mean, width=1, color = shue_color)
plt.bar(bins[0:-1], trac_binned_mean, width=1, fill = False, edgecolor = 'k')

plt.ylim(0,18)
plt.xlim(bins[0]-0.5, bins[-5]+0.5)
plt.ylabel(shue_name)
plt.title(r'Model Uncertainty, ARTEMIS Magnetopause Crossings')
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

ax2 = plt.subplot(512, sharex = ax1)
plt.bar(bins[0:-1], chao_binned_mean, width=1, color = chao_color)
plt.bar(bins[0:-1], trac_binned_mean, width=1, fill = False, edgecolor = 'k')
plt.ylim(0,18)
plt.ylabel(chao_name)
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

ax3 = plt.subplot(513, sharex = ax1)
plt.bar(bins[0:-1], petrus_binned_mean, width=1, color = petrus_color)
plt.bar(bins[0:-1], trac_binned_mean, width=1, fill = False, edgecolor = 'k')
plt.ylim(0,18)
plt.ylabel(petrus_name)
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

ax4 = plt.subplot(514, sharex = ax1)
plt.bar(bins[0:-1], lu_binned_mean, width=1, color = lu_color)
plt.bar(bins[0:-1], trac_binned_mean, width=1, fill = False, edgecolor = 'k')
plt.ylim(0,18)
plt.ylabel(lu_name)
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

ax5 = plt.subplot(515, sharex = ax1)
plt.bar(bins[0:-1], lin_binned_mean, width=1, color = lin_color)
plt.bar(bins[0:-1], trac_binned_mean, width=1, fill = False, edgecolor = 'k')
plt.ylim(0,18)
plt.ylabel(lin_name)
plt.xlabel(r'GSE $\rho$ ($R_{e}$)')

plt.savefig('tail_residual_comp_poster.png')
plt.savefig('tail_residual_comp_poster.pdf', format = 'pdf')
plt.clf()

#######################

















