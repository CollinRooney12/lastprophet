# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 13:39:11 2017

@author: collin.rooney

This file fits the data to a prophet model and then calculates the residuals, mse and returns everything

Credit to Rob J. Hyndman and research partners as much of the code was developed with the help of their work
https://www.otexts.org/fpp
https://robjhyndman.com/publications/
Credit to Facebook and their fbprophet package
https://facebookincubator.github.io/prophet/
It was my intention to make some of the code look similar to certain sections in the Prophet and (Hyndman's) hts packages

"""

import numpy as np
from fbprophet import Prophet

def fitProphet(aggs, h, include_history, cap, capF, changepoints, n_changepoints, \
                yearly_seasonality, weekly_seasonality, holidays, seasonality_prior_scale, holidays_prior_scale,\
                changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples):
    """
    Parameters
    ----------
    All largely the same as lastF
    
    aggs - (dict of DataFrames) output of aggHier
    
    
    Returns
    ----------
    forecastsDict - (dict of DataFrames)  contains seasonalities and forecasts for all of the different temporal aggregation levels
    
    mse - (dict)  the mean square error and the estimator for error variance
    
    resids - (dict)  the error of the fitted values with respect to the data
    """
    # Prophet related stuff
    ##
    # A lot of this can be understood by looking at prophet's instructional materials
    ##
    mse = {}
    resids = {}
    fcst = {}
    for key in aggs.keys():
        freq = aggs[key].columns.tolist()[0]
        seasonal = max(aggs.keys())
        aggs[key] = aggs[key].rename(columns = {aggs[key].columns[0] : 'ds'})
        aggs[key] = aggs[key].rename(columns = {aggs[key].columns[1] : 'y'})
        if 'AS-' in freq:
            yearly_seasonality = False
        if capF is None:
            growth = 'linear'
            m = Prophet(growth, changepoints, n_changepoints, yearly_seasonality, weekly_seasonality, holidays, seasonality_prior_scale, \
                        holidays_prior_scale, changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples)
        else:
            growth = 'logistic'
            m = Prophet(growth, changepoints, n_changepoints, yearly_seasonality, weekly_seasonality, holidays, seasonality_prior_scale, \
                        holidays_prior_scale, changepoint_prior_scale, mcmc_samples, interval_width, uncertainty_samples)
            aggs[key]['cap'] = cap
        m.fit(aggs[key])
        periods = int((h/seasonal)*key)
        future = m.make_future_dataframe(periods = periods, freq = freq, include_history = include_history) #Frequency is equal to our hard coded column name
        if capF is not None:
            future['cap'] = capF
        fcst[key] = m.predict(future)
        ##
        # Find MSE and resids
        ##
        if periods == 0:
            periods = 1
        resids[key] = aggs[key].y.values - fcst[key].yhat[:-periods].values
        mse[key] = np.mean(np.array(resids[key])**2)
        
    return fcst, mse, resids
