# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 09:06:01 2017

@author: collin.rooney

This is the file that contains all of the aggregation methods.  
Forecasts for all the different time instances are sent here to be revised and improved

Credit to Rob J. Hyndman and research partners as much of the code was developed with the help of their work
https://www.otexts.org/fpp
https://robjhyndman.com/publications/
Credit to Facebook and their fbprophet package
https://facebookincubator.github.io/prophet/
It was my intention to make some of the code look similar to certain sections in the Prophet and (Hyndman's) hts packages

"""

import sys
import numpy as np
from scipy.special import inv_boxcox

def summingMat(freqs):
    """
    Parameters
    ----------
    
    freqs - (list) the frequencies of the time series aggregates
    
    Returns
    ----------
    
    sumMat - (numpy 2d array) summing matrix (see Hyndmans online book for explanation)
    
    """
    m = max(freqs)  # Find the finest grain seasonality period
    top = np.ones(m)    # Create the top layer of the matrix
    bottom = np.identity(m)     # Create the bottom layer of the matrix
    if len(freqs) > 2:      # if there is a layer beyond the top and bottom
        freqs = set(freqs)^set([1, m])      # get rid of the top and bottom layer frequencies from the list of frequencies
        freqs = list(freqs)                 
        for freq in sorted(freqs):
            pattern = np.concatenate([np.ones(freq), np.zeros(m)])  # Each layer has a pattern of 1s and 0s, create it
            rowForm = np.tile(pattern, int((m/freq)-1))             # Repeat that pattern
            rowForm = np.concatenate([rowForm, np.ones(freq)])
            middle = rowForm.reshape((int(m/freq), m))              # Reshape the rowForm into a matrix
            bottom = np.vstack((middle, bottom))                    # Put that matrix inbetween the top and bottom
    
    sumMat = np.vstack((top, bottom))
    
    return sumMat

def reconcile(forecastsDict, h, mse = None, resids = None, comb = "BU", boxcoxT = None):
    """
    Parameters
    ----------
    
    forecastsDict - (dict of DataFrames) contains seasonalities and forecasts for all of the different temporal aggregation levels
    
    h - (int) the forecast horizon for the time series
    
    resids - (dict)  the error of the fitted values with respect to the data
    
    comb  (String)  the type of hierarchical forecasting method that the user wants to use. 
                        
        	Options:
                    "OLS" - optimal combination by ordinary least squares (Default), 
                    "WLSS" - optimal combination by structurally weighted least squares,
                    "WLSV" - optimal combination by variance weighted least squares
                    "BU" - bottom up combination
    
    boxcoxT - (list or None) if a list, then these are the lambda values that allow an inverse boxcox transform to take place
    
    
    Returns
    ----------
    
    forecastsDict - (dict of DataFrames) temporally revised forecasts
    
    """
    ##
    # Find the frequencies
    ##
    freqs = list(forecastsDict.keys())
    if min(freqs) != 1:
        sys.exit("The minimum seasonal period should be 1 - Change aggList")
    if comb != "BU" and comb != "OLS" and comb != "WLSS" and comb != "WLSV":
        sys.exit("The reconciliation method must be one of the specified types, see instructions")
    m = max(freqs)
    if h < m:
        sys.exit("The prediction length (h) should be at least as long as the seasonality (m)")
    ##
    # Get the Summing Matrix and Organize the Forecasts in the way needed
    ##
    sumMat = summingMat(freqs)
    nCols = h/m
    ##
    # Inverse Box Cox
    ##
    if boxcoxT is not None:
        i = 0
        for key in sorted(forecastsDict.keys()):
            forecastsDict[key].yhat = inv_boxcox(forecastsDict[key].yhat, boxcoxT[i])
            forecastsDict[key].trend = inv_boxcox(forecastsDict[key].trend, boxcoxT[i])
            if "seasonal" in forecastsDict[key].columns.tolist():
                forecastsDict[key].seasonal = inv_boxcox(forecastsDict[key].seasonal, boxcoxT[i])
            if "weekly" in forecastsDict[key].columns.tolist():
                forecastsDict[key].weekly = inv_boxcox(forecastsDict[key].weekly, boxcoxT[i])
            if "yearly" in forecastsDict[key].columns.tolist():
                forecastsDict[key].yearly = inv_boxcox(forecastsDict[key].yearly, boxcoxT[i])
            if "holidays" in forecastsDict[key].columns.tolist():
                forecastsDict[key].yearly = inv_boxcox(forecastsDict[key].yearly, boxcoxT[i])
            i += 1
    ##
    # Bottom Up
    ##
    if comb == 'BU':
        hatMat = np.empty([int(nCols), 0])
        periods = int(h/m)*m                    # The number of full years * the seasonality constant
        if h == m:
            rowForm = np.array(forecastsDict[m].yhat[-(h+1):-(h-periods+1)])  # only include the forecasts that are (periods) ahead of the last value
        else:
            rowForm = np.array(forecastsDict[m].yhat[-(h):-(h-periods)])  # only include the forecasts that are (periods) ahead of the last value
        rowForm.shape = ((int(periods/m), m))
        chunk = rowForm
        hatMat = np.hstack((hatMat, chunk))
        betaEst = sumMat  # This is what we will multiply by the normal forecasts
    ##
    # Optimal Combination
    ##
    if comb == 'OLS' or comb == 'WLSV' or comb == 'WLSS':
        hatMat = np.empty([int(nCols), 0])
        for key in sorted(forecastsDict.keys()):
            periods = int(h/m)*key
            rowForm = np.array(forecastsDict[key].yhat[-periods:])
            rowForm.shape = ((int(periods/key), key))
            chunk = rowForm
            hatMat = np.hstack((hatMat, chunk))
    if comb == 'OLS':
        betaEst = np.dot(np.dot(sumMat, np.linalg.inv(np.dot(np.transpose(sumMat), sumMat))),np.transpose(sumMat))  # See Hyndman's online textbook for explanation
    if comb == 'WLSS':
        diagMat = np.diag(np.transpose(np.sum(sumMat, axis = 1)))  # Create a matrix that describes the structure of the hierarchy
        betaEst = np.dot(np.dot(np.dot(sumMat, np.linalg.inv(np.dot(np.dot(np.transpose(sumMat), np.linalg.inv(diagMat)), sumMat))), np.transpose(sumMat)), np.linalg.inv(diagMat))
    if comb == 'WLSV':
        diagMat = [np.repeat(mse[key], key) for key in sorted(mse.keys())]   # Create matrix of mse (error variance) values
        diagMat = np.diag(np.flip(np.hstack(diagMat)+0.000001, 0))  # Added a very small number to fix the singular matrix problem
        betaEst = np.dot(np.dot(np.dot(sumMat, np.linalg.inv(np.dot(np.dot(np.transpose(sumMat), np.linalg.inv(diagMat)), sumMat))), np.transpose(sumMat)), np.linalg.inv(diagMat))
    ##
    # All
    ##
    newMat = np.empty([hatMat.shape[0],sumMat.shape[0]])
    for i in range(hatMat.shape[0]):
        newMat[i,:] = np.dot(betaEst, np.transpose(hatMat[i,:]))
    ##
    # Put Matrix into a dictionary of dataframes
    ##
    prevKey = 0
    for key in sorted(forecastsDict.keys()):
        periods = int(h/m)*key
        values = forecastsDict[key].yhat.values
        newDat = newMat[:,prevKey:prevKey+key].reshape(periods)
        values[-periods:] = newDat.flatten()
        forecastsDict[key].yhat = values
        prevKey += key
        
    return forecastsDict
    
