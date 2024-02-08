    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 11:09:59 2020

@author: msangste
"""

import numpy as np
import csv
import json
import pandas as pd
from json import JSONDecodeError
import matplotlib.pyplot as plt
from matplotlib import cm
from plot import labels_size
import pandas
from lmfit.models import PolynomialModel
from lmfit import Parameters
import functools as ft


def load_wavelength():
    # take wavelength ranges
    # define an empty list
    fluoWL = []
    absWL = []
    
    # open file and read the content in a list
    for (wl_file, lst) in zip(['absWL.txt', 'fluoWL.txt'], [absWL, fluoWL]): 
        try:
            with open(wl_file, 'r') as filehandle:
                print('loading wavelength files from this folder')    
                for line in filehandle:
                    # remove linebreak which is the last character of the string
                    currentPlace = line[:-1]
            
                    # add item to the list
                    lst.append(float(currentPlace))
        except FileNotFoundError:
            with open(f'../{wl_file}', 'r') as filehandle:
                print('loading wavelength files from folder above')    
                for line in filehandle:
                    # remove linebreak which is the last character of the string
                    currentPlace = line[:-1]
            
                    # add item to the list
                    lst.append(float(currentPlace))
            
    return {'fluoWL':fluoWL, 'absWL':absWL}
    

        
    
def background_fluo(absfluo, absfluo_WT, abso_key, blue_key, green_key):
    """
    substract background based on interpolation
    """
    background_blue = np.interp(absfluo[abso_key], absfluo_WT[abso_key], absfluo_WT[blue_key])
    absfluo['fluo_blue_corr'] = absfluo[blue_key] - background_blue
    if green_key in absfluo.columns:
        background_green = np.interp(absfluo[abso_key], absfluo_WT[abso_key], absfluo_WT[green_key])
        absfluo['fluo_green_corr'] = absfluo[green_key] - background_green

  
def absfluo_column(row, column_name, absfluo):
    while True:
        try:
            absfluo[column_name] = list(row.median(axis=1)) # take median
        except ValueError as e:
#            if (str(e) == "Length of values does not match length of index"):
                # if absfluo is longer than row, but not too much longer, add nans to row
            len_diff = len(absfluo) - len(row)
            if (len_diff > 0) and (len_diff < 5):
                for i in range(0, len_diff, 1):
                    row = row.append(pd.Series(dtype='datetime64[ns]', name=absfluo.index[-1]))
            # if row is longer, add nans to absfluo
            elif (len_diff < 0) and (len_diff > -5):
                for i in range(0, abs(len_diff), 1):
                    absfluo = absfluo.append(pd.Series(dtype='datetime64[ns]', name=absfluo.index[-1]))
            else:
                'oops'
                raise(e) 
#            else:
#                raise(e)
            continue
        break
    return absfluo




def time_to_hours(absfluo, zero=False):
    if 'time' not in absfluo.columns:
        absfluo.index=absfluo.index.rename('time')
        absfluo = absfluo.reset_index(level=['time']) #create time column
    else:
        print("this dataframe already has a time column")
        absfluo = absfluo.reset_index()
    absfluo['time'] = pd.to_datetime(absfluo['time'])
    if zero:
        zero = zero
    else:
        zero = absfluo['time'][0]
    
    absfluo['time'] = absfluo['time'] - zero
    absfluo['time'] = absfluo['time'].dt.days*24 + absfluo['time'].dt.seconds/3600
    return absfluo



def load_json(line, header):
    """
    -to be used in the function open_file_bioreactor_tags
    -loads the json part of a line in the bioreactor file
    -shorter, more readable, version of the line in the return
    """
    return json.loads(line[header['json']].replace("=>",": "))


def open_file_bioreactor_tags(algo, file, absWL, fluoWL, tags):
    """
    Opens bioreactor file with tags 
    Takes:
        an bioreactor number and 
        a file-name and
        two lists of wavelengths for fluorescence and for absorbance
        
    and returns 
        a dictionary of dataframes with index time and columns wavelenght
    """       
   
    with open(file) as f:
        data = csv.reader(f)
        # read header and create a dictionary with the position of the headers
        first_line = next(data) 
        header = {}
        for i, element in enumerate(first_line):
            header[element] = i    
        
        #create empty dictionary to save dataframes
        measurements = {}
        for tag in tags:
            measurements[tag]=pd.DataFrame(columns=['time', 'value'])
        
        
#        measurements = {
#                        'water':pd.DataFrame(columns=['time', 'value']), 
#                        'dark_water':pd.DataFrame(columns=['time', 'value']), 
#                        'sample_first': pd.DataFrame(columns=['time', 'value']), 
#                        'sample_second':pd.DataFrame(columns=['time', 'value']), 
#                        'blue_fluo':pd.DataFrame(columns=['time', 'value']),
#                        'blue_dark':pd.DataFrame(columns=['time', 'value']),
#                        'green_fluo':pd.DataFrame(columns=['time', 'value']),
#                        'green_dark':pd.DataFrame(columns=['time', 'value']),
#                        }
        try:
            for line in data:  
                try:
                    if 'tags' in load_json(line, header): 
                        if (algo in load_json(line, header)['tags']['algo_tags']): 
                            #measurement[key] = measurement[key].append({'time':time, 'value':value})
                            #load_json(line, header)['tags']['sequence_tags'] gives the key
                            try:
                                df = pd.DataFrame({'time': [line[header['created_at']]], 'value': [load_json(line, header)['value']]})
                                measurements[load_json(line,header)['tags']['sequence_tags']]  = pd.concat([measurements[load_json(line,header)['tags']['sequence_tags']], df], ignore_index=True)
                                
                                # measurements[load_json(line,header)['tags']['sequence_tags']] = measurements[load_json(line, header)['tags']['sequence_tags']].append({'time': line[header['created_at']], 'value': load_json(line, header)['value']}, ignore_index=True)
                            except KeyError:
                                print('keyerror')
                except JSONDecodeError as error:
                    # this is the error it gives when the header at the bottom of file is encountered
                    if "Expecting value" in str(error): 
                        print(f'{line} not a json: parsing is stopped')
                        break
        except UnicodeDecodeError:
            print('unicode error!!')
    for key in measurements:
        
        print(key)
        #set absWL as columns
        if len(absWL) == len(measurements[key].value.tolist()[0]):
            measurements[key].loc[:,absWL] = measurements[key].value.tolist()
        else:
            measurements[key].loc[:,fluoWL] = measurements[key].value.tolist()

            
        measurements[key] = measurements[key].drop('value', axis=1)
        measurements[key] = measurements[key].set_index('time')
        measurements[key].index = pd.to_datetime(measurements[key].index)        
        
    return measurements



def create_absfluo_bioreactor_tags_many_missing(measurements, 
                                   abso=['water', 'dark_water', 'sample_first', 'sample_second'], 
                                   # in case of multiple fluo colours, dark is the same for all of them. In that case,
                                   # add 'fluo_dark' to each color. E.g. fluo_blue = ['blue_fluo', 'dark_fluo']
                                   fluo_blue=['blue_fluo'],  
                                   fluo_green=['green_fluo'],
                                   fluo_red = ['red_fluo'],
                                   fluo_dark = 'fluo_dark',
                                   fluo_dark_green = False,
                                   fluo_dark_red = False,
                                   water='water', 
                                   dark='dark_water', 
                                   abso_samples=['sample_first', 'sample_second'],
                                   fluo_samples = ['blue_fluo', 'green_fluo'],
                                   green_low=520,
                                   green_high=540,
                                   abso_low=595,
                                   abso_high=605,
                                   blue_low=470,
                                   blue_high=480,
                                   red_low=605,
                                   red_high=615,
                                   find_index_by = 'longest'
                                   ):
    """
    create dataframe with median absorbance and fluorescence measurements of the given bioreactor
    all tags have defaults, but can be set
    """
    
    
    
    #find longest dataframe
    longest = 0
    for key in measurements:
        length = len(measurements[key])
        if length > longest:
            longest = length
            longest_df = key
            
    absfluo = pd.DataFrame(index=measurements[longest_df].index)

    for key in abso:
        if not measurements[key].empty:
            df = measurements[key]
            df = df.loc[:,(df.columns >= abso_low) & (df.columns <= abso_high)].copy()
            #calculate median absorbance
            # df[key] = df.median(axis=1)
            median = df.median(axis=1)
            df.loc[:, key] = median
            #merge with absfluo
            absfluo = pd.merge_asof(left=absfluo,right=df[key],right_index=True,
                                    left_index=True,direction='nearest',
                                    tolerance=pd.Timedelta('5 minutes'))
            
            
    fluo_dark = measurements[fluo_dark]
    for key in fluo_blue:
        if not measurements[key].empty:
            df = measurements[key]
            df = df.loc[:,(df.columns >= blue_low) & (df.columns <= blue_high)].copy()
            #calculate median absorbance
            # df[key] = df.median(axis=1)
            df.loc[:, key] = df.median(axis=1)
            #join with absfluo
            absfluo = pd.merge_asof(left=absfluo,right=df[key],right_index=True,
                                    left_index=True,direction='nearest',
                                    tolerance=pd.Timedelta('5 minutes'))
            #add dark
            df_dark = fluo_dark.loc[:,(fluo_dark.columns >= blue_low) & (fluo_dark.columns <= blue_high)].copy()
        
            df_dark['dark'] = df_dark.median(axis=1)
            absfluo = pd.merge_asof(left=absfluo,right=df_dark['dark'],right_index=True,
                                    left_index=True,direction='nearest',
                                    tolerance=pd.Timedelta('5 minutes'))
            absfluo = absfluo.rename(columns={'dark':f'{key}_dark'})
            absfluo[key] -= absfluo[f'{key}_dark']
    for key in fluo_green:
        if not measurements[key].empty:
            df = measurements[key]
            df = df.loc[:,(df.columns >= green_low) & (df.columns <= green_high)].copy()
            #calculate median absorbance
            # df[key] = df.median(axis=1)
            df.loc[:, key] = df.median(axis=1)

            #join with absfluo
            absfluo = pd.merge_asof(left=absfluo,right=df[key],right_index=True,
                                    left_index=True,direction='nearest',
                                    tolerance=pd.Timedelta('5 minutes'))
            #add dark
            if fluo_dark_green:
                fluo_dark = measurements[fluo_dark_green]
            df_dark = fluo_dark.loc[:,(fluo_dark.columns >= green_low) & (fluo_dark.columns <= green_high)].copy()
            df_dark['dark'] = df_dark.median(axis=1)
            
            absfluo = pd.merge_asof(left=absfluo,right=df_dark['dark'],right_index=True,
                                    left_index=True,direction='nearest',
                                    tolerance=pd.Timedelta('5 minutes'))
            absfluo = absfluo.rename(columns={'dark':f'{key}_dark'})
            absfluo[key] -= absfluo[f'{key}_dark']
    for key in fluo_red:
        if not measurements[key].empty:
            df = measurements[key]
            df = df.loc[:,(df.columns >= red_low) & (df.columns <= red_high)].copy()
            #calculate median absorbance
            # df[key] = df.median(axis=1)
            df.loc[:, key] = df.median(axis=1)

            #join with absfluo
            absfluo = pd.merge_asof(left=absfluo,right=df[key],right_index=True,
                                    left_index=True,direction='nearest',
                                    tolerance=pd.Timedelta('5 minutes'))
            #add dark
            if fluo_dark_red:
                fluo_dark = measurements[fluo_dark_red]
            df_dark = fluo_dark.loc[:,(fluo_dark.columns >= red_low) & (fluo_dark.columns <= red_high)].copy()
            df_dark['dark'] = df_dark.median(axis=1)
            absfluo = pd.merge_asof(left=absfluo,right=df_dark['dark'],right_index=True,
                                    left_index=True,direction='nearest',
                                    tolerance=pd.Timedelta('5 minutes'))
            
            absfluo = absfluo.rename(columns={'dark':f'{key}_dark'})
            absfluo[key] -= absfluo[f'{key}_dark']
            absfluo[f'{key}_raw'] = absfluo[key] + absfluo[f'{key}_dark'] 
            
    
    #drop rows with NaNs
#    absfluo = absfluo.dropna()
#    
    # normalize OD based on dark
    for sample in abso_samples:
        try:
            absfluo[f'{sample}_uncorrected'] = absfluo[sample]
            absfluo[sample] = np.log10((absfluo[water]-absfluo[dark])/(absfluo[sample]-absfluo[dark]))
        except KeyError as e:
            print(str(e))
            pass    
            
    return absfluo
    

def correct_absorbance(absorbance, slope=0.6983):
    #create polynomial model that relates absorbance and dilution
    poly_mod = PolynomialModel(degree=2)
    pars = Parameters()
    pars.add('c0', value=-0.01141, vary=False)
    pars.add('c1', value=0.6471, vary=False)
    pars.add('c2', value=0.1550, vary=False)
    #calculate dilution from absorbance
    dilution = poly_mod.eval(params=pars, x=absorbance)
    #calculate real absorbance
    absorbance = dilution / slope
    
    return absorbance
               
def correct_absorbance_df(absorbance, slope=0.6983, name_abso='abs'):
    absorbance = absorbance.to_frame()
    absorbance['corrected'] = correct_absorbance(absorbance, slope)
    absorbance['corrected'] = absorbance['corrected'].where(absorbance[name_abso]>0.55, other=absorbance[name_abso])
    absorbance = absorbance['corrected']
    return absorbance

def mean_sd(dfs, mean1='corrected', mean2='red_fluo',):
    """
    Provide dataframes with common columns of which you want to calculate 
    the mean.
    They also need to have a common column 'time' in hours on which they can be merged.
    Returns a dataframe with the merged dataframes and the mean and stand deviations
    of the two columns as indicated by mean1 and mean2. (Default: 'corrected' and 'red_fluo')
    """
    
    dfs2 = []
    mean1_2 = []
    mean2_2 = []
    for i, df in enumerate(dfs):
        df = df.add_suffix(i)
        df = df.rename(columns={f'time{i}':'time'})
        dfs2.append(df)
        mean1_2.append(f'{mean1}{i}')
        mean2_2.append(f'{mean2}{i}')
        
        
    absfluo = ft.reduce(lambda left, right: pd.merge_asof(left, right, 
        direction='nearest', tolerance=0.5, on='time'), dfs2)


    #take mean absorbance and sd
    absfluo[f'mean_{mean1}'] = absfluo[mean1_2].mean(axis=1)
    absfluo[f'std_{mean1}' ] = absfluo[mean1_2].std(axis=1)
    absfluo[f'mean_{mean2}'] = absfluo[mean2_2].mean(axis=1)
    absfluo[f'std_{mean2}'] = absfluo[mean2_2].std(axis=1)
    absfluo[f'2sem_{mean1}'] = 2 * absfluo[mean1_2].sem(axis=1)
    absfluo[f'2sem_{mean2}'] = 2 * absfluo[mean2_2].sem(axis=1)

    
    

    return absfluo


def fuse(dfs, df_WT, key, fraction_i, subtract_WT=True):
    """
    provide:
        dfs: list of dataframes wit consortium data to fuse
        df_WT: dataframe of WT only data
        key: red fluorescence key (suggested: 'red_fluo_raw')
        fraction_i: initial fraction of cleaner in the consortium
        
        dfs need to have a column 'time' and 'gdw'
        
    returns:
        merged dataframe with mean_cleaner and mean_gdw of the dfs provided
    
    """


    
    
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
    
    if subtract_WT==True:
        for key1 in keys:
            absfluo[key1] -= absfluo[f'{key}WT']
        
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


