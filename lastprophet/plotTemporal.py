# -*- coding: utf-8 -*-
"""
Name: htsPlot.py
Author: Collin Rooney
Last Updated: 8/15/2017

This script will contain functions for plotting the output of the lastF() function
These plots will be made to look like the plots Prophet creates

Credit to Rob J. Hyndman and research partners as much of the code was developed with the help of their work
https://www.otexts.org/fpp
https://robjhyndman.com/publications/
Credit to Facebook and their fbprophet package
https://facebookincubator.github.io/prophet/
It was my intention to make some of the code look similar to certain sections in the Prophet and (Hyndman's) hts packages

"""
from matplotlib import pyplot as plt

#%%
def plotNode(dictframe, h = 1, xlabel = 'ds', ylabel = 'y', startFrom = 0, ax = None):
    '''
    Parameters
    ------------------
    
    dictframe - (dict) The dictionary of dataframes that is the output of the thief function
    
    h - (int) number of steps in the forecast same as input to hts function
    
    ylabel - (string) label for the graph's y axis
    
    start_from - (int) the number of values to skip at the beginning of yhat so that you can zoom in
    
    ax - (axes object) any axes object thats already created that you want to pass to the plot function
    
    Returns
    ------------------
    
    plot of that node's forecast
    
    '''
    for node in dictframe.keys():
        periods = int(h/max(dictframe.keys()))*node
        nodeToPlot = dictframe[node]
        fig = plt.figure(facecolor='w', figsize=(10, 6))
        ax = fig.add_subplot(111)
        ##
        # plot the yhat forecast as a solid line and then the h-step ahead forecast as a dashed line
        ##
        ax.plot(nodeToPlot['ds'].values[startFrom:-periods], nodeToPlot['yhat'][startFrom:-periods], ls='-', c='#0072B2')
        ax.plot(nodeToPlot['ds'].values[-periods:], nodeToPlot['yhat'][-periods:], dashes = [2,2])
        ##
        # plot the cap and uncertainty if necessary
        ##
        if 'cap' in nodeToPlot:
            ax.plot(nodeToPlot['ds'].values[startFrom:], nodeToPlot['cap'][startFrom:], ls='--', c='k')
    
        ax.grid(True, which='major', c='gray', ls='-', lw=1, alpha=0.2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        
    return fig