# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 13:38:46 2017

@author: collin.rooney

This function aggregates the time series into factors of the seasonality period (m) and returns a dictionary of dataframes of the aggregated data

Credit to Rob J. Hyndman and research partners as much of the code was developed with the help of their work
https://www.otexts.org/fpp
https://robjhyndman.com/publications/
Credit to Facebook and their fbprophet package
https://facebookincubator.github.io/prophet/
It was my intention to make some of the code look similar to certain sections in the Prophet and (Hyndman's) hts packages
"""
import sys
import numpy as np
import pandas as pd
import calendar

def aggHier(y, m, aggList = None):
    """
    Parameters
    ----------
    
    y - dataframe of time-series data
                       
            	Layout:
                           
                     1st Col - Time instances
                           
                     2nd Col - Total of TS
             
        
    m - (int) frequency of time series eg. weekly is 52 (len(y) > 2*m)
    
    aggList - (list) The factors that the user would like to consider for ex. m = 52, aggList = [1, 52]
    
    Returns
    ----------
    
    aggs - (dict of DataFrames) aggregated data where the key is equal to the seasonality period
    
    """
    n = len(y.iloc[:,0])
    m = int(m)
    ##
    # Prophet will only work with pre-defined frequencies, therefore we are going to need to hard code them
    ##
    if m not in [4,12,7,24,168,8760,365,52,252]:
        sys.exit("Sorry, your list of frequencies did not match our list of pre-defined ones.  Please enter m as one of 4,12,7,24,168,870,365,52")
    ##
    # Get all of the factors of m
    ##
    mList = range(1, m+1)
    mSet = {i for i in mList if m % i == 0}
    mList = [i for i in mSet if i < n]
    if len(mList) == 0:
        sys.exit("Your time series is too short")
    if m == 252:
        mList = [1,4,12,21,63,252]
    ##
    # If aggList is specified, find where the factors and that list match
    ##
    mList = sorted(mList)
    if aggList is not None:
        mList = np.intersect1d(np.array(mList), np.array(aggList))
    if len(mList) == 0:
        sys.exit("Your specified aggList did not match any of the factors of your frequency")
    ##
    # Aggregate
    ##
    k = len(mList)
    aggs = {}    #Create dictionary for dataframes to be stored
    aggs[m] = y
    for i in range(k):
        start = n%m
        fullPeriods = int(n/m)
        temp = np.array(y.iloc[start:n, 1])
        temp.shape = (fullPeriods*mList[i], int(m/mList[i]))
        temporary = pd.DataFrame(y.iloc[list(range(start+int(m/mList[i] - 1), n, int(m/mList[i]))), 0])
        temporary['y'] = temp.sum(axis = 1)
        aggs[mList[i]] = temporary
    ##
    # Set the name of the columns equal to the frequency parameter in Prophet (Here is where the hard coding comes in, Sorry)
    ##
    if m == 4:
        aggs[4] = aggs[4].rename(columns = {aggs[4].columns[0] : 'Q'})
        if 2 in mList:
            aggs[2] = aggs[2].rename(columns = {aggs[2].columns[0] : '6M'})
        monthNum = aggs[1].iloc[0,0].month
        name = calendar.month_abbr[monthNum]
        aggs[1] = aggs[1].rename(columns = {aggs[1].columns[0] : 'AS-'+name})
        
    elif m == 12:
        aggs[12] = aggs[12].rename(columns = {aggs[12].columns[0] : 'M'})
        if 6 in mList:
            aggs[6] = aggs[6].rename(columns = {aggs[6].columns[0] : '2M'})
        if 4 in mList:
            aggs[4] = aggs[4].rename(columns = {aggs[4].columns[0] : 'Q'})
        if 3 in mList:
            aggs[3] = aggs[3].rename(columns = {aggs[3].columns[0] : '4M'})
        if 2 in mList:
            aggs[2] = aggs[2].rename(columns = {aggs[2].columns[0] : '6M'})
        monthNum = aggs[1].iloc[0,0].month
        name = calendar.month_abbr[monthNum]
        aggs[1] = aggs[1].rename(columns = {aggs[1].columns[0] : 'AS-'+name})
        
    elif m == 7:
        aggs[7] = aggs[7].rename(columns = {aggs[7].columns[0] : 'D'})
        aggs[1] = aggs[1].rename(columns = {aggs[1].columns[0] : 'W'})
        
    elif m == 24:
        aggs[24] = aggs[24].rename(columns = {aggs[24].columns[0] : 'H'})
        if 2 in mList:
            aggs[12] = aggs[12].rename(columns = {aggs[12].columns[0] : '2H'})
        if 8 in mList:
            aggs[8] = aggs[8].rename(columns = {aggs[8].columns[0] : '3H'})
        if 6 in mList:
            aggs[6] = aggs[6].rename(columns = {aggs[6].columns[0] : '4H'})
        if 4 in mList:
            aggs[4] = aggs[4].rename(columns = {aggs[4].columns[0] : '6H'})
        if 3 in mList:
            aggs[3] = aggs[3].rename(columns = {aggs[3].columns[0] : '8H'})
        if 2 in mList:
            aggs[2] = aggs[2].rename(columns = {aggs[2].columns[0] : '12H'})
        aggs[1] = aggs[1].rename(columns = {aggs[1].columns[0] : 'D'})
        
    elif m == 168:
        aggs[168] = aggs[168].rename(columns = {aggs[168].columns[0] : 'H'})
        if 84 in mList:
            aggs[84] = aggs[84].rename(columns = {aggs[84].columns[0] : '2H'})
        if 56 in mList:
            aggs[56] = aggs[56].rename(columns = {aggs[56].columns[0] : '3H'})
        if 42 in mList:
            aggs[42] = aggs[42].rename(columns = {aggs[42].columns[0] : '4H'})
        if 28 in mList:
            aggs[28] = aggs[28].rename(columns = {aggs[28].columns[0] : '6H'})
        if 24 in mList:
            aggs[24] = aggs[24].rename(columns = {aggs[24].columns[0] : '7H'})
        if 21 in mList:
            aggs[21] = aggs[21].rename(columns = {aggs[21].columns[0] : '8H'})
        if 14 in mList:
            aggs[14] = aggs[14].rename(columns = {aggs[14].columns[0] : '12H'})
        if 12 in mList:
            aggs[12] = aggs[12].rename(columns = {aggs[12].columns[0] : '14H'})
        if 8 in mList:
            aggs[8] = aggs[8].rename(columns = {aggs[8].columns[0] : '21H'})
        if 7 in mList:
            aggs[7] = aggs[7].rename(columns = {aggs[7].columns[0] : 'D'})
        if 6 in mList:
            aggs[6] = aggs[6].rename(columns = {aggs[6].columns[0] : '28H'})
        if 4 in mList:
            aggs[4] = aggs[4].rename(columns = {aggs[4].columns[0] : '42H'})
        if 3 in mList:
            aggs[3] = aggs[3].rename(columns = {aggs[3].columns[0] : '56H'})
        if 2 in mList:
            aggs[2] = aggs[2].rename(columns = {aggs[2].columns[0] : '84H'})
        aggs[1] = aggs[1].rename(columns = {aggs[1].columns[0] : 'W'})
        
    elif m == 8760:
        aggs[8760] = aggs[8760].rename(columns = {aggs[8760].columns[0] : 'H'})
        if 4380 in mList:
            aggs[4380] = aggs[4380].rename(columns = {aggs[4380].columns[0] : '2H'})
        if 2920 in mList:
            aggs[2920] = aggs[2920].rename(columns = {aggs[2920].columns[0] : '3H'})
        if 2190 in mList:
            aggs[2190] = aggs[2190].rename(columns = {aggs[2190].columns[0] : '4H'})
        if 1752 in mList:
            aggs[1752] = aggs[1752].rename(columns = {aggs[1752].columns[0] : '5H'})
        if 1460 in mList:
            aggs[1460] = aggs[1460].rename(columns = {aggs[1460].columns[0] : '6H'})
        if 1095 in mList:
            aggs[1095] = aggs[1095].rename(columns = {aggs[1095].columns[0] : '8H'})
        if 876 in mList:
            aggs[876] = aggs[876].rename(columns = {aggs[876].columns[0] : '10H'})
        if 730 in mList:
            aggs[730] = aggs[730].rename(columns = {aggs[730].columns[0] : '12H'})
        if 584 in mList:
            aggs[584] = aggs[584].rename(columns = {aggs[584].columns[0] : '15H'})
        if 438 in mList:
            aggs[438] = aggs[438].rename(columns = {aggs[438].columns[0] : '20H'})
        if 365 in mList:
            aggs[365] = aggs[365].rename(columns = {aggs[365].columns[0] : 'D'})
        if 292 in mList:
            aggs[292] = aggs[292].rename(columns = {aggs[292].columns[0] : '30H'})
        if 219 in mList:
            aggs[219] = aggs[219].rename(columns = {aggs[219].columns[0] : '40H'})
        if 146 in mList:
            aggs[146] = aggs[146].rename(columns = {aggs[146].columns[0] : '60H'})
        if 120 in mList:
            aggs[120] = aggs[120].rename(columns = {aggs[120].columns[0] : '73H'})
        if 73 in mList:
            aggs[73] = aggs[73].rename(columns = {aggs[73].columns[0] : '5D'})
        if 60 in mList:
            aggs[60] = aggs[60].rename(columns = {aggs[60].columns[0] : '146H'})
        if 30 in mList:
            aggs[30] = aggs[30].rename(columns = {aggs[30].columns[0] : '292H'})
        if 20 in mList:
            aggs[20] = aggs[20].rename(columns = {aggs[20].columns[0] : '438H'})
        if 15 in mList:
            aggs[15] = aggs[15].rename(columns = {aggs[15].columns[0] : '584H'})
        if 12 in mList:
            aggs[12] = aggs[12].rename(columns = {aggs[12].columns[0] : '730H'})
        if 10 in mList:
            aggs[10] = aggs[10].rename(columns = {aggs[10].columns[0] : '876H'})
        if 8 in mList:
            aggs[8] = aggs[8].rename(columns = {aggs[8].columns[0] : '1095H'})
        if 6 in mList:
            aggs[6] = aggs[6].rename(columns = {aggs[6].columns[0] : '1460H'})
        if 5 in mList:
            aggs[5] = aggs[5].rename(columns = {aggs[5].columns[0] : '73D'})
        if 4 in mList:
            aggs[4] = aggs[4].rename(columns = {aggs[4].columns[0] : 'Q'})
        if 3 in mList:
            aggs[3] = aggs[3].rename(columns = {aggs[3].columns[0] : '4M'})
        if 2 in mList:
            aggs[2] = aggs[2].rename(columns = {aggs[2].columns[0] : '6M'})
        monthNum = aggs[1].iloc[0,0].month
        name = calendar.month_abbr[monthNum]
        aggs[1] = aggs[1].rename(columns = {aggs[1].columns[0] : 'AS-'+name})
        
    elif m == 365:
        aggs[365] = aggs[365].rename(columns = {aggs[365].columns[0] : 'D'})
        if 73 in mList:
            aggs[73] = aggs[73].rename(columns = {aggs[73].columns[0] : '5D'})
        if 5 in mList:
            aggs[5] = aggs[5].rename(columns = {aggs[5].columns[0] : '73D'})
        monthNum = aggs[1].iloc[0,0].month
        name = calendar.month_abbr[monthNum]
        aggs[1] = aggs[1].rename(columns = {aggs[1].columns[0] : 'AS-'+name})
        
    elif m == 52:
        aggs[52] = aggs[52].rename(columns = {aggs[52].columns[0] : 'W'})
        if 26 in mList:
            aggs[26] = aggs[26].rename(columns = {aggs[26].columns[0] : '2W'})
        if 13 in mList:
            aggs[13] = aggs[13].rename(columns = {aggs[13].columns[0] : '4W'})
        if 4 in mList:
            aggs[4] = aggs[4].rename(columns = {aggs[4].columns[0] : '13W'})
        if 2 in mList:
            aggs[2] = aggs[2].rename(columns = {aggs[2].columns[0] : '24W'})
        monthNum = aggs[1].iloc[0,0].month
        name = calendar.month_abbr[monthNum]
        aggs[1] = aggs[1].rename(columns = {aggs[1].columns[0] : 'AS-'+name})
    
    elif m == 252:
        aggs[252] = aggs[252].rename(columns = {aggs[252].columns[0] : 'B'})
        if 63 in mList:
            aggs[63] = aggs[63].rename(columns = {aggs[63].columns[0] : '63B'})
        if 21 in mList:
            aggs[21] = aggs[21].rename(columns = {aggs[21].columns[0] : '21B'})
        if 12 in mList:
            aggs[12] = aggs[12].rename(columns = {aggs[12].columns[0] : '12B'})
        if 4 in mList:
            aggs[4] = aggs[4].rename(columns = {aggs[4].columns[0] : '4B'})
        monthNum = aggs[1].iloc[0,0].month
        name = calendar.month_abbr[monthNum]
        aggs[1] = aggs[1].rename(columns = {aggs[1].columns[0] : 'AS-'+name})
    
    return aggs