# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 12:45:48 2017

@author: collin.rooney

This function and file are used to aggregate a time series signal, forecast these different granularities of aggregated data,
and then share the information between them by aggregating trhough least squares approaches.  The forecasting procedure used is Facebook's
prophet package.  The function returns a dictionary of dataframes that can be plotted or further analyzed.

Credit to Rob J. Hyndman and research partners as much of the code was developed with the help of their work
https://www.otexts.org/fpp
https://robjhyndman.com/publications/
Credit to Facebook and their fbprophet package
https://facebookincubator.github.io/prophet/
It was my intention to make some of the code look similar to certain sections in the Prophet and (Hyndman's) hts packages

"""
import pandas as pd
import sys
from lastprophet.fitProphet import fitProphet
from lastprophet.aggHier import aggHier
from lastprophet.reconcile import reconcile
from scipy.stats import boxcox
import contextlib
import os

#%%
def lastF(y, m = 12, h = 12*2, comb = "OLS", aggList = None, include_history = True, cap = None, capF = None, \
        changepoints = None, n_changepoints = 25, yearly_seasonality = True, weekly_seasonality = 'auto', holidays = None, seasonality_prior_scale = 10.0, \
        holidays_prior_scale = 10.0, changepoint_prior_scale = 0.05, mcmc_samples = 0, interval_width = 0.80, uncertainty_samples = 0, transform = None):
    """
        Parameters
        ----------------
             
        y - dataframe of time-series data
        
            	Layout:
                     1st Col - Time instances
                     2nd Col - Total of TS
             
        m - (int) frequency of time series eg. weekly is 52 (len(y) > 2*m)
            
        h - (int) the forecast horizon for the time series
        
        comb  (String)  the type of hierarchical forecasting method that the user wants to use. 
                        
        	Options:
                    "OLS" - optimal combination by ordinary least squares (Default), 
                    "WLSS" - optimal combination by structurally weighted least squares,
                    "WLSV" - optimal combination by variance weighted least squares
                    "BU" - bottom up combination
        
        aggList - (list) The factors that the user would like to consider for ex. m = 52, aggList = [1, 52] 
        
        include_history - (Boolean) input for the forecasting function of Prophet
                        
        cap - (Dataframe or Constant) carrying capacity of the input time series.  If it is a dataframe, then
         the number of columns must equal len(y.columns) - 1
        
        capF - (Dataframe or Constant) carrying capacity of the future time series.  If it is a dataframe, then
         the number of columns must equal len(y.columns) - 1
             
        changepoints - (DataFrame or List) changepoints for the model to consider fitting. If it is a dataframe, then
         the number of columns must equal len(y.columns) - 1
         
        n_changepoints - (constant or list) changepoints for the model to consider fitting. If it is a list, then
         the number of items must equal len(y.columns) - 1
          
        transform - (None or "BoxCox") Do you want to transform your data before fitting the prophet function? If yes, type "BoxCox"
        
        All other inputs - see Prophet
        
        Returns
        -----------------
         
        newDict - a dictionary of DataFrames with predictions, seasonalities and trends that can all be plotted
        
    """
    ##
    # Error Catching
    ##
    if not isinstance(y.iloc[:,0], pd.DatetimeIndex):
        y.iloc[:,0] = pd.DatetimeIndex(y.iloc[:,0])
    if m <= 1:
        sys.exit("Seasonal period (m) must be greater than 1")
    if len(y) < 2*m:
        sys.exit("Need at least 2 periods of data")
    if aggList is not None:
        if 1 not in aggList or m not in aggList:
            sys.exit("1 and the seasonal period must be included in the aggList input")
    ##
    # Compute Aggregate Time Series and return a dictionary of dataframes
    ##
    aggs = aggHier(y, m, aggList)
    ##
    # Transform Variables
    ##
    if transform is not None:
        if transform == 'BoxCox':
            import warnings
            warnings.simplefilter("error", RuntimeWarning)
            boxcoxT = [None]*(len(aggs.keys()))
            try:
                i = 0
                placeHold = []
                for key in sorted(aggs.keys()):
                    placeHold.append(aggs[key].copy())
                    placeHold[i].iloc[:, 1], boxcoxT[i] = boxcox(placeHold[i].iloc[:, 1])
                    i += 1
                i = 0
                for key in sorted(aggs.keys()):
                    aggs[key] = placeHold[i]
                    i += 1
            ##
            # Does a Natural Log Transform if scipy's boxcox cant deal
            ##
            except RuntimeWarning:
                print("It looks like scipy's boxcox function couldn't deal with your data. Proceeding with Natural Log Transform")
                i = 0
                for key in sorted(aggs.keys()):
                    aggs[key].iloc[:, 1] = boxcox(aggs[key].iloc[:, 1], lmbda = 0)
                    boxcoxT[i] = 0
                    i += 1
        else:
            print("Nothing will be transformed because the input was not = to 'BoxCox'")
    else:
        boxcoxT = None
    ##
    # Forecast and Reconcile
    ##
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        forecastsDict, mse, resids = fitProphet(aggs, h, include_history, cap, capF, changepoints, n_changepoints, \
                                                 yearly_seasonality, weekly_seasonality, holidays, seasonality_prior_scale, \
                                                 holidays_prior_scale, changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples)
    newDict = reconcile(forecastsDict, h, mse, resids, comb, boxcoxT)

    return newDict