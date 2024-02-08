        #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 11:36:16 2020

@author: msangste
"""

import matplotlib.pyplot as plt

def merge_legend(ax, ax2, fontsize=20, **kwargs):    
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, fontsize=fontsize, **kwargs)
    ax.get_legend().remove()

def labels_size(ax, y_label, x_label, ax2=None, y_label2=None, legend=True, title=False, fontsize=30, 
                font_legend=20, markerscale=5, **kwargs):
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.set_xlabel(x_label, fontsize=fontsize)
    if legend == True:
        ax.legend(fontsize=font_legend, markerscale=markerscale, **kwargs)
    else:
        ax.get_legend().remove()
    if title:
        ax.set_title(title, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    if ax2:
        ax2.set_ylabel(y_label2, fontsize=fontsize)

        ax2.tick_params(labelsize=fontsize)
        merge_legend(ax, ax2, fontsize=font_legend, markerscale=markerscale, **kwargs)
        
def caption_fit(result):
    """
    returns caption from result obtained from fit function (see fit module)
    """
    caption = result['result'].params.valuesdict()
    txt = ''
    return txt.join([f'{key} = {value:.2} ' for (key, value) in caption.items()])

def plot_two_axes(df, x, y1, y2, **kwargs):
    plt.figure()
    ax = plt.gca()
    ax2 = ax.twinx()
    df.plot(x=x, y=y1, linestyle='', marker='.', ax=ax, **kwargs)
    df.plot(x=x, y=y2, linestyle='', marker='.', ax=ax2, **kwargs)
    labels_size(ax, y1, x, ax2, y2)

def ax():
    plt.figure()
    ax = plt.gca()
    return ax

def create_ax():
    plt.figure()
    ax = plt.gca()
    return ax

def ax2(ax):
    ax2 = ax.twinx()
    return ax2

def create_ax2(ax):
    ax2 = ax.twinx()
    return ax2
    
def plot(df, x, y, ax, marker='.', linestyle='', std=None, maxi=None, **kwargs):
    if maxi:
        print('cutting off')
        df = df.loc[df[x]<maxi]
        print(df)
    if x != 'index':
        df.plot(x=x, y=y, linestyle=linestyle, marker=marker, ax=ax, **kwargs)
    else:
        df.plot(y=y, linestyle=linestyle, marker=marker, ax=ax, **kwargs)
    if std:
        try:
            color=kwargs['color']
            ax.fill_between(df[x], 
                            df[y]-df[std], 
                            df[y]+df[std],
                            alpha=0.2, 
                            label='_nolegend',
                            color=color)
        except KeyError:
            ax.fill_between(df[x], 
                            df[y]-df[std], 
                            df[y]+df[std],
                            alpha=0.2, 
                            label='_nolegend',
                            color=ax.get_lines()[-1].get_color()
                            )
            
        
        
def ax_two_panels():
    
    fig = plt.figure(constrained_layout=True)


    ax_dict = fig.subplot_mosaic(
        [
            [1],
            [2]
            
        ],
        empty_sentinel="BLANK",
    )

    ax = ax_dict[1]
    ax2 = ax_dict[2]
    
    return ax, ax2
        