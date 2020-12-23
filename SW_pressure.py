# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 14:23:25 2020
    
    A script that pulls all available 1 minute OMNI SW data, calculates thermal and magnetic pressure, and 
    plots a neat historgam of dynamic, magnetic, and thermal pressure sources. Takes a while to download all
    that data for sure but boy is it worth it!

@author: connor o'brien
"""

import os
os.environ["CDF_LIB"] = "c:\CDF_Distribution\cdf37_1-dist"
from spacepy import pycdf
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import bisect
import webbrowser
import time

def GetOMNIData(Date,DataPath,DownloadPath):
    """Downloads OMNI data for date string YYYYMMDD. Courtesy Emil Atz"""
    Year = Date[0]+Date[1]+Date[2]+Date[3]  #Splitting the year
    Month = Date[4]+Date[5] #Splitting the month
    if (int(Year) >= 1981): 
        CDFFile = 'omni_hro_1min_'+Year+Month+'01_v01.cdf' #Building the file name
    else:
        if (int(Month) < 7):
            CDFFile = 'omni2_h0_mrg1hr_'+Year+'0101_v01.cdf'
        else:
            CDFFile = 'omni2_h0_mrg1hr_'+Year+'0701_v01.cdf'   
    if os.path.isfile(DataPath+CDFFile): #Checking if you already have it downloaded
        print('The CDF of this month is already downloaded')
        return CDFFile #This breaks you out of the function
    # PYTHON INTERNET SCRAPPING
    # DANGER: If you do this method and are too 'fast' with your internet requests,
    # the web page can block you because it may think you're hacking. Which you are.
    # So dont be flooding web pages with download requests from a script! Thays why
    # there is a time.sleep(10) in there.
    if (int(Year) >= 1981):
        webbrowser.open('https://cdaweb.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/hro_1min/'+Year+'/'+CDFFile)
        print('Downloading 1 min OMNI CDF file ' + Date)
    else:
        webbrowser.open('https://cdaweb.gsfc.nasa.gov/pub/data/omni/omni_cdaweb/hourly/'+Year+'/'+CDFFile)
        print('Downloading Hourly OMNI CDF file ' + Date)
    time.sleep(20)
    os.rename(DownloadPath+CDFFile, DataPath+CDFFile)
    return CDFFile

def ModifyOMNIData(Date, CDFFile, DataPath):
    """Creates small cdf from monthly OMNI data (string CDFFile) for date string YYYYMMDD. Courtesy Emil Atz"""
    os.chdir(DataPath)
    FullCDF = pycdf.CDF(CDFFile) #Getting the file into Python
    Start = dt.datetime.strptime(Date, '%Y%m%d')
    Stop = dt.datetime.strptime(Date, '%Y%m%d')
    Stop = Stop+dt.timedelta(days = 1)
    # Finding the indicies with which to cut the file
    start_ind = bisect.bisect_left(FullCDF['Epoch'], Start)
    stop_ind = bisect.bisect_left(FullCDF['Epoch'], Stop)
    # Aplying the indicies to the file variables of interest
    time = FullCDF['Epoch'][start_ind:stop_ind]
    BzGSE = FullCDF['BZ_GSE'][start_ind:stop_ind]
    BxGSE = FullCDF['BX_GSE'][start_ind:stop_ind]
    BzGSM = FullCDF['BZ_GSM'][start_ind:stop_ind]
    ByGSM = FullCDF['BY_GSM'][start_ind:stop_ind]
    Pdyn = FullCDF['Pressure'][start_ind:stop_ind]
    Vx = FullCDF['Vx'][start_ind:stop_ind]
    Vy = FullCDF['Vy'][start_ind:stop_ind]
    Vz = FullCDF['Vz'][start_ind:stop_ind]
    os.chdir(DataPath)
    # Saving the cut file to a new file
    CDFName = Date+'.cdf'
    if os.path.isfile(DataPath+CDFName):
        print('The CDF of these days is already generated')
        return CDFName
    Shortcdf = pycdf.CDF(CDFName, '')
    Shortcdf['Epoch'] = time
    Shortcdf['BZ_GSE'] = BzGSE
    Shortcdf['BZ_GSM'] = BzGSM
    Shortcdf['BX_GSE'] = BxGSE
    Shortcdf['BY_GSM'] = ByGSM
    Shortcdf['Pressure'] = Pdyn
    Shortcdf['Vx'] = Vx
    Shortcdf['Vy'] = Vy
    Shortcdf['Vz'] = Vz
    return CDFName

storage_path = '/Users/conno/Documents/Data Storage/OMNI/' #Where the data is to be stored
downloads_path = '/Users/conno/Downloads/' #Your downloads folder
home_path = "/Users/conno/Documents/Magnetopause Modeling/Data/" #Where the script is run from

cdf_arr = np.asarray([])

#Make the array of cdf names
for year in np.arange(1981, 2020):
    for month in np.arange(1,13):
        date = str(year*100 + month)
        cdf_arr = np.append(cdf_arr, GetOMNIData(date, storage_path, downloads_path))

#Stage the condition arrays
imf = np.asarray([])
pdyn = np.asarray([])
proton_n = np.asarray([])
proton_t = np.asarray([])

#Gotta look in the right place!
os.chdir(storage_path)

#Download the data, cut the incomplete parts, and put in the staged arrays
for filename in cdf_arr:
    print('Appending '+ filename)
    data = pycdf.CDF(filename)
    temp_imf = np.sqrt(np.asarray(data['BX_GSE'])**2 + np.asarray(data['BY_GSE'])**2 + np.asarray(data['BZ_GSE'])**2)
    temp_pdyn = np.asarray(data['Pressure'])
    temp_n = np.asarray(data['proton_density'])
    temp_t = np.asarray(data['T'])
    temp_imf = temp_imf[(np.asarray(data['BX_GSE']) < 9999.99) & (np.asarray(data['BY_GSE']) < 9999.99) & (np.asarray(data['BZ_GSE']) < 9999.99)]
    temp_pdyn = temp_pdyn[temp_pdyn < 99.99]
    therm_bool = (temp_n < 999.99) & (temp_t < 9999999.)
    temp_n = temp_n[therm_bool]
    temp_t = temp_t[therm_bool]
    imf = np.append(imf, temp_imf)
    pdyn = np.append(pdyn, temp_pdyn)
    proton_n = np.append(proton_n, temp_n)
    proton_t = np.append(proton_t, temp_t)

#Calculate the derived pressures in nPa
pmag = (imf ** 2) * (3.97887 * 10**(-4))
ptherm = proton_n * proton_t * (1.381 * 10**(-8))

#Make a log x scale for the histogram to make it legible
logbins = np.geomspace(np.min(pmag), np.max(pdyn), 75)

#Plot it up!
pdyn_n, pdyn_bins, pdyn_patch = plt.hist(pdyn, bins = logbins, color = '#ddaa33', histtype = 'stepfilled', alpha = 0.5)
pmag_n, pmag_bins, pmag_patch = plt.hist(pmag, bins = logbins, color = '#004488', histtype = 'stepfilled', alpha = 0.5)
ptherm_n, ptherm_bins, ptherm_patch = plt.hist(ptherm, bins = logbins, color = '#bb5566', histtype = 'stepfilled', alpha = 0.5)
plt.xscale('log')
plt.xlim(10**(-4), 10**2)
plt.xlabel('Pressure (nPa)')
plt.legend(['$P_{dyn}$', '$P_{mag}$', '$P_{therm}$'], loc = 'upper left', frameon = False)
plt.title('Solar Wind Pressure Sources')
plt.savefig('sw_pressure_trans.png')
plt.savefig('sw_pressure_trans.pdf', format = 'pdf')


















