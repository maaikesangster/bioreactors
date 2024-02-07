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

data2_2 = open_file_bioreactor_tags(2, 'raw_sensors.csv', absWL, fluoWL, tags) #D = 0.15
data2_3 = open_file_bioreactor_tags(3, 'raw_sensors.csv', absWL, fluoWL, tags) #D = 0.15
data2_4 = open_file_bioreactor_tags(4, 'raw_sensors.csv', absWL, fluoWL, tags) #D = 0.15
data2_5 = open_file_bioreactor_tags(5, '../../bioreactor/221219/raw_sensors.csv', absWL, fluoWL, tags) #D = 0.15 with kan

data15_2 = open_file_bioreactor_tags(2, '../../bioreactor/221107/raw_sensors.csv', absWL, fluoWL, tags) #D = 0.15
data15_3 = open_file_bioreactor_tags(3, '../../bioreactor/221107/raw_sensors.csv', absWL, fluoWL, tags) #D = 0.15
data15_4 = open_file_bioreactor_tags(4, '../../bioreactor/221107/raw_sensors.csv', absWL, fluoWL, tags) #D = 0.15
data15_5 = open_file_bioreactor_tags(5, '../../bioreactor/221107/raw_sensors.csv', absWL, fluoWL, tags) #D = 0.15



#%%

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

#%%
import functools as ft


def fuse_subtract_dark(dfs, dark, key, fraction_i):
    dfs2 = []
    keys =[]
    gdws = []
    cleaners = []
    
    for i, df in enumerate(dfs):
        df = df.add_suffix(i)
        df = df.rename(columns={f'time{i}':'time'})
        dfs2.append(df)
        keys.append(f'{key}{i}')
        gdws.append(f'gdw{i}')
        cleaners.append(f'cleaner{i}')
        
    absfluo = ft.reduce(lambda left, right: pd.merge_asof(left, right, 
        direction='nearest', tolerance=0.5, on='time'), dfs2)
    
    for key1 in keys:
        absfluo[key1] -= absfluo[dark]
        
    #calculate cleaner
    cleaner_i = absfluo['gdw1'].iloc[0] * fraction_i #initial amount of cleaner
    ratio = absfluo[f'{key}1'].iloc[0]/cleaner_i #amount of red fluo per cleaner

    for i, key in enumerate(keys):
        absfluo[f'cleaner{i}'] = absfluo[key]/ratio
        
        
    absfluo['mean_gdw'] = absfluo[gdws].mean(axis=1)
    absfluo['std_gdw' ] = absfluo[gdws].std(axis=1)
    absfluo['mean_cleaner'] = absfluo[cleaners].mean(axis=1)
    absfluo['std_cleaner'] = absfluo[cleaners].std(axis=1)
    
        
    return absfluo

def fuse(dfs, df_WT, key, fraction_i):
    dfs2 = []
    keys =[]
    gdws = []
    cleaners = []
    
    for i, df in enumerate(dfs):
        df = df.add_suffix(i)
        df = df.rename(columns={f'time{i}':'time'})
        dfs2.append(df)
        keys.append(f'{key}{i}')
        gdws.append(f'gdw{i}')
        cleaners.append(f'cleaner{i}')
        
    df_WT = df_WT.add_suffix('WT')
    df_WT = df_WT.rename(columns={'timeWT':'time'})
    
    dfs2.append(df_WT)

        
    absfluo = ft.reduce(lambda left, right: pd.merge_asof(left, right, 
        direction='nearest', tolerance=0.5, on='time'), dfs2)
    
    for key1 in keys:
        absfluo[key1] -= absfluo[f'{key}WT']
        
    #calculate cleaner
    cleaner_i = absfluo['gdw1'].iloc[0] * fraction_i #initial amount of cleaner
    ratio = absfluo[f'{key}1'].iloc[0]/cleaner_i #amount of red fluo per cleaner

    for i, key in enumerate(keys):
        absfluo[f'cleaner{i}'] = absfluo[key]/ratio
        
        
    absfluo['mean_gdw'] = absfluo[gdws].mean(axis=1)
    absfluo['std_gdw' ] = 2 * absfluo[gdws].sem(axis=1)
    absfluo['mean_cleaner'] = absfluo[cleaners].mean(axis=1)
    absfluo['std_cleaner'] = 2 * absfluo[cleaners].sem(axis=1)
    
        
    return absfluo

#%% make absfluo
absfluo2_2 = make_absfluo(data2_2)
absfluo2_3 = make_absfluo(data2_3)
absfluo2_4 = make_absfluo(data2_4)
absfluo2_5 = make_absfluo(data2_5)

absfluo15_2 = make_absfluo(data15_2)
absfluo15_3 = make_absfluo(data15_3)
absfluo15_4 = make_absfluo(data15_4)
absfluo15_5 = make_absfluo(data15_5)


#%%time to hours


zero2 = absfluo2_3.index[0]
zero15 = absfluo15_3.index[0]





from functions_bioreactor import time_to_hours

absfluo2_2 = time_to_hours(absfluo2_2, zero2)
absfluo2_3 = time_to_hours(absfluo2_3, zero2)
absfluo2_4 = time_to_hours(absfluo2_4, zero2)
absfluo2_5 = time_to_hours(absfluo2_5, zero2)

absfluo15_2 = time_to_hours(absfluo15_2, zero15)
absfluo15_3 = time_to_hours(absfluo15_3, zero15)
absfluo15_4 = time_to_hours(absfluo15_4, zero15)
absfluo15_5 = time_to_hours(absfluo15_5, zero15)



#%% mean, sd, sem 
from functions_bioreactor import mean_sd


absfluo2 = mean_sd([absfluo2_2, absfluo2_3, absfluo2_4, absfluo2_5], mean1='gdw')
absfluo15 = mean_sd([absfluo15_2, absfluo15_3, absfluo15_4], mean1='gdw')


absfluo15 = absfluo15.loc[absfluo15['time']>0]
absfluo15['time'] -= absfluo15['time'].iloc[0]

absfluo2 = absfluo2.loc[absfluo2['time']>3]
absfluo2['time'] -= absfluo2['time'].iloc[0]


#%% 
from plot import create_ax, plot, labels_size, ax_two_panels


#%% relative time

absfluos = [absfluo2, absfluo15
            ]
Ds = [ 0.15, 0.15 ]

for i, (absfluo, D) in enumerate(zip(absfluos, Ds)):
    absfluo['rel_time'] = absfluo['time'] * D
    absfluos[i] = absfluo


absfluos = [
    absfluo2_2,
    absfluo2_3,
    absfluo2_4,
    absfluo2_5, 
    absfluo15_2,
    absfluo15_3,
    absfluo15_4,
    absfluo15_5, 
    ]
Ds = [ 0.15, 0.15]
import numpy as np
Ds = np.repeat(Ds, 4)


for i, (absfluo, D) in enumerate(zip(absfluos, Ds)):
    absfluo['rel_time'] = absfluo['time'] * D
    absfluos[i] = absfluo
    
    
    


#%% fit model 
from experiment import Cell, CellCycling, MediumNew
from fit import fit_consortium, print_result
from lmfit import Parameters

WT_cyc = CellCycling(0.2, 'WT')
WT_cyc.Y_g = 0.4
cleaner_cyc = CellCycling(0.1, 'cleaner')
cleaner_cyc.k_g = 0.5
WT_cyc.Y_g = 0.4


params = Parameters()
params.add('biomass_WT', value=0.11, vary=False, min=0.05, max=0.2)
params.add('biomass_cleaner', 0.039, vary=False, min=0.01, max=0.1)
params.add('g', 0.5, vary=False, min=0.1, max=0.5)
params.add('a', 0.5, vary=False, min=0.1, max=0.5)
params.add('conversion_biomass_cleaner', 17000, vary=False, min=15000, max=30000)

data = absfluo2
to_fit = 'biomass_cleaner'
strains = [WT_cyc, cleaner_cyc]
start = 0
stop = 45
D = 0.15
g_in = 1
variable = 'mean_red_fluo'
medium = MediumNew()

fit_cyc = fit_consortium(params, data,[to_fit], strains, start, stop, D, g_in, [variable], medium, max_nfev=100 )
fit_cyc[1]['biomass_cleaner'] *= fit_cyc[0].params['conversion_biomass_cleaner']

print_result(fit_cyc[0])





#%%plot data
absfluo2 = absfluo2.loc[absfluo2['time']>=0]



ax = create_ax()

plot(absfluo2, 'time', 'mean_red_fluo', ax, label='_nolegend', color='red', linestyle='-', marker='', linewidth=3)
plot(absfluo2, 'time', 'red_fluo0', ax,  color='red', label='kanamycin', alpha=.5)
plot(absfluo2, 'time', 'red_fluo1', ax, label='_nolegend', color='red', alpha=.5)
plot(absfluo2, 'time', 'red_fluo2', ax, label='_nolegend', color='red', alpha=.5)
plot(absfluo2, 'time', 'red_fluo3', ax, label='_nolegend', color='red', alpha=.5)

# plot(fit_cyc[1], 'time', 'biomass_cleaner', ax, label='_nolegend', color='red', linestyle='--', marker='', linewidth=3)

plot(absfluo15, 'time', 'mean_red_fluo', ax, label='_nolegend', color='orange', linestyle='-', marker='', linewidth=3)
plot(absfluo15, 'time', 'red_fluo0', ax,  color='orange', label='no kanamycin', alpha=.5)
plot(absfluo15, 'time', 'red_fluo1', ax, label='_nolegend', color='orange', alpha=.5)
plot(absfluo15, 'time', 'red_fluo2', ax, label='_nolegend', color='orange', alpha=.5)

ax.set_yscale('log')


labels_size(ax, 'red fluorescence', 'time (h)', )

#%% plot biomass as well
from plot import ax_two_panels

ax2, ax = ax_two_panels()

plot(absfluo2, 'time', 'mean_red_fluo', ax, label='_nolegend', color='red', linestyle='-', marker='', linewidth=3)
plot(absfluo2, 'time', 'red_fluo0', ax,  color='red', label='_nolegend', alpha=.5)
plot(absfluo2, 'time', 'red_fluo1', ax, label='_nolegend', color='red', alpha=.5)
plot(absfluo2, 'time', 'red_fluo2', ax, label='_nolegend', color='red', alpha=.5)
plot(absfluo2, 'time', 'red_fluo3', ax, label='_nolegend', color='red', alpha=.5)

# plot(fit_cyc[1], 'time', 'biomass_cleaner', ax, label='_nolegend', color='red', linestyle='--', marker='', linewidth=3)

# plot(absfluo15, 'time', 'mean_red_fluo', ax, label='_nolegend', color='orange', linestyle='-', marker='', linewidth=3)
# plot(absfluo15, 'time', 'red_fluo0', ax,  color='orange', label='no kanamycin', alpha=.5)
# plot(absfluo15, 'time', 'red_fluo1', ax, label='_nolegend', color='orange', alpha=.5)
# plot(absfluo15, 'time', 'red_fluo2', ax, label='_nolegend', color='orange', alpha=.5)

ax.set_yscale('log')

plot(absfluo2, 'time', 'mean_gdw', ax2, label='_nolegend', color='red', linestyle='-', marker='', linewidth=3)
plot(absfluo2, 'time', 'gdw0', ax2,  color='red', label='_nolegend', alpha=.5)
plot(absfluo2, 'time', 'gdw1', ax2, label='_nolegend', color='red', alpha=.5)
plot(absfluo2, 'time', 'gdw2', ax2, label='_nolegend', color='red', alpha=.5)
plot(absfluo2, 'time', 'gdw3', ax2, label='_nolegend', color='red', alpha=.5)

# plot(fit_cyc[1], 'time', 'biomass_cleaner', ax, label='_nolegend', color='red', linestyle='--', marker='', linewidth=3)

# plot(absfluo15, 'time', 'mean_gdw', ax2, label='_nolegend', color='orange', linestyle='-', marker='', linewidth=3)
# plot(absfluo15, 'time', 'gdw0', ax2,  color='orange', label='_nolegend', alpha=.5)
# plot(absfluo15, 'time', 'gdw1', ax2, label='_nolegend', color='orange', alpha=.5)
# plot(absfluo15, 'time', 'gdw2', ax2, label='_nolegend', color='orange', alpha=.5)




labels_size(ax, 'red fluorescence', 'time (h)', )
labels_size(ax2, 'biomass (gDW/L)', None)



