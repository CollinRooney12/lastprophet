# lastprophet


Long and Short Term Forecasting using Prophet

(Known to R users as thief)
Credit to Rob J. Hyndman and research partners as much of the code was developed with the help of their work.


https://www.otexts.org/fpp


https://robjhyndman.com/publications/


Credit to Facebook and their fbprophet package.


https://facebookincubator.github.io/prophet/

It was my intention to make some of the code look similar to certain sections in the Prophet and (Hyndman's) thief packages.



# Downloading



1. pip install lastprophet


If you'd like to just skip to coding with the package, **prophetVerify.py** should help you with that, but if you like reading, the following should help you understand how I built lastprophet and how it works.



# Part I: The Data



I originally used Redfin traffic data to build this package.  

I pulled the data so that date was in the first column, and the number I wanted to forecast was in the second column.

We were making daily forecasts, but wondered if we could get more out of their long term projections by using a multiple time-series aggregation approach.

So the data looked like this:


|   Date   |    Sessions   |
|----------|---------------|
| 1100 B.C.|	  23234    |
|   ...    | 	   2342	   |
|          |	   233     |
|          |     445       |


Then I just ran my lastF() function.  All it takes is one call with all of the prophet inputs and it will return a dictionary of fitted and forecasted values for all of the different time instances



# Part II: Prophet Inputs


Anything that you would specify in Prophet you can specify in lastF(). 

It’s flexible and will allow you to input a dataframe of values for inputs like cap, capF, and changepoints.

All of these inputs are specified when you call lastF, and after that you just let it run.

The following is the description of inputs and outputs for hts as well as the specified defaults:

    """
        Parameters
        ----------------
             
        y - dataframe of time-series data
                       
        	Layout:
                           
        	0th Col - Time instances
                           
        	1st Col - Total of TS
             
        
        m - (int) frequency of time series eg. weekly is 52 (len(y) > 2*m)
            
        h - (int) the forecast horizon for the time series
        
        comb – (String)  the type of hierarchical forecasting method that the user wants to use. 
                        
        	Options:
                    "OC" - optimal combination (Default), 
        
        aggList -  
        
        include_history - (Boolean) input for the forecasting function of Prophet
                        
        cap - (Dataframe or Constant) carrying capacity of the input time series.  If it is a dataframe, then
         the number of columns must equal len(y.columns) - 1
        
        capF - (Dataframe or Constant) carrying capacity of the future time series.  If it is a dataframe, then
         the number of columns must equal len(y.columns) - 1
             
        changepoints - (DataFrame or List) changepoints for the model to consider fitting. If it is a dataframe, then
         the number of columns must equal len(y.columns) - 1
         
        n_changepoints - (constant or list) changepoints for the model to consider fitting. If it is a list, then
         the number of items must equal len(y.columns) - 1
          
        All other inputs - see Prophet
        
        Returns
        -----------------
         
        newDict - a dictionary of DataFrames with predictions, seasonalities and trends that can all be plotted
        
    """


# Part III: The output


The output is a dictionary of dataframes with keys that tell you which time series is which.  For example, if I was forecasting a weekly time series with "m = 52"
I would be returned a dictionary with keys of (1, 2, 4, 13, 26, 52) - all factors of m. Here the key 52 would contain weekly data, 26 bi-weekly and so on.

You can plot all of these time series by calling **plotNode()**



# Part IV: Room For Improvement



The package could benefit from the following two things:


1. A way to run some of it in parallel, cause it take a while sometimes.

2. Prediction intervals would be cool as well.
