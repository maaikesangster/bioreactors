#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 16:22:04 2022

@author: maaike
"""

import matplotlib.pyplot as plt
from functions_bioreactor import open_file_bioreactor_tags, create_absfluo_bioreactor_tags_many_missing, correct_absorbance_df
import numpy as np
import pandas as pd

tags = ['water', 'dark_water', 
        'abs',  'blue_fluo', 'green_fluo', 'red_fluo', 'dark_fluo', 'dark_fluo_shortinttime']

from functions_bioreactor import load_wavelength
fluoWL = load_wavelength()['fluoWL']
absWL = load_wavelength()['absWL']

#open data from bioreactor 2 (with dilution rate D=0.15)
data2_2 = open_file_bioreactor_tags(2, 'raw_sensors.csv', absWL, fluoWL, tags) 


#%%

#define function to quickly correct for non-linear absorbance and convert from abs to gdw
def make_absfluo(data):

    absfluo2 = create_absfluo_bioreactor_tags_many_missing(
                                        data, 
                                        abso=['water', 'dark_water', 'abs'], 
                                        fluo_blue=['blue_fluo'], 
                                        fluo_green=['green_fluo'],
                                        fluo_red = ['red_fluo'],
                                        fluo_dark = 'dark_fluo_shortinttime',
                                        fluo_dark_red = 'dark_fluo',
                                        water='water', 
                                        dark='dark_water', 
                                        abso_samples=['abs'],
                                        fluo_samples = ['blue_fluo', 'green_fluo', 'red_fluo',
                                                       ],
                                        abso_low=595,
                                        abso_high=605,
                                        green_low=505,
                                        green_high=520,
                                        blue_low=505,
                                        blue_high=520,
                                        find_index_by='last',
                                        red_low=630,
                                        red_high=640,
                                                       )
    
    
    absfluo2['corrected'] = correct_absorbance_df(absfluo2['abs'])
   
    #turn into gdw/l
    absfluo2['gdw'] = absfluo2['corrected'] * 0.37 * 3.6
    
    return absfluo2



#%% make absfluo:create dataframe with median absorbance and fluorescence measurements of bioreactor 2
absfluo2_2 = make_absfluo(data2_2)



#%%convert time to hours

#set t=0
zero2 = absfluo2_2.index[4]

from functions_bioreactor import time_to_hours

absfluo2_2 = time_to_hours(absfluo2_2, zero2)



#%% Plot
from plot import create_ax, plot, labels_size

ax = create_ax()
plot(absfluo2_2, 'index', 'gdw', ax)

#improve plot layout
labels_size(ax,'Biomass (gDW/L)', 'Time (h)')




