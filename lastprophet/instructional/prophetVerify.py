# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 15:08:38 2017

@author: collin.rooney

I created this file to verify the findings of Nikolaos Kourentzes in his blog post : http://kourentzes.com/forecasting/2017/07/29/benchmarking-facebooks-prophet/

and also to see if my package was an improvement on the current version of prophet

Credit to Rob J. Hyndman and research partners as much of the code was developed with the help of their work
https://www.otexts.org/fpp
https://robjhyndman.com/publications/
Credit to Facebook and their fbprophet package
https://facebookincubator.github.io/prophet/
It was my intention to make some of the code look similar to certain sections in the Prophet and (Hyndman's) hts packages

"""

##
# Import lastprophet lastF
##
import pandas as pd
import numpy as np

#%% Get the M3 data and only give the necessary data to lastF
M3Quarter = pd.read_csv("M3CQ.csv")
M3Quarter = M3Quarter.transpose()
M3Quarter = M3Quarter.iloc[6:,:]

#%% Create training and Testing dataset for Quarterly (8)
##
# Then find the MASE values for the prophet forecasts
##
maseQ = []
for column in M3Quarter:
    for i in range(4):
        data = pd.DataFrame(M3Quarter[column])
        data = data.dropna()
        data["quarter"] = pd.DatetimeIndex(start = "2010-11-07", periods = len(data), freq = 'Q')
        data = data[['quarter', column]] 
        train = data.iloc[0:-(8-i)]
        test = data.iloc[-(8-i):]
        bFcst = lastF(train, m = 4, h = 4, comb = "WLSS")
        maseQ.append(abs(bFcst[4].yhat[-4:].values - test.iloc[0:4,1].values)/np.mean(abs(train.iloc[1:-1,1].values - train.iloc[0:-2,1].values)))