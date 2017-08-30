import unittest
import pandas as pd
import numpy as np
from lastprophet.aggHier import aggHier
from lastprophet.fitProphet import fitProphet
from lastprophet.reconcile import reconcile
from lastprophet.last import lastF


class testLASTOut(unittest.TestCase):
    
    def testOutputs(self):
        ##
        # Daily Data
        ##
        date = pd.date_range("2012-04-02", "2017-07-17")
        sessions = np.random.randint(100, 40000,size=(len(date),1))
        data = pd.DataFrame(date, columns = ["day"])
        data["sessions"] = sessions
        ##
        # Check AggHier
        ##
        aggs = aggHier(data, m = 365)
        self.assertIsNotNone(aggs)          # Checking that there is some stuff there
        self.assertEqual(len(aggs.keys()), 4)
        self.assertLess(len(aggs[1]), len(aggs[5]))    # Checking that the values were aggregated (made smaller in length)
        self.assertLess(len(aggs[5]), len(aggs[73]))
        self.assertLess(len(aggs[73]), len(aggs[365]))
        self.assertLess(len(aggs[1]), len(aggs[365]))
        ##
        # Check fitProphet
        ##
        forecastsDict, mse, resids = fitProphet(aggs, h = 365, include_history = True, cap = None, capF = None, \
                                                 changepoints = None, n_changepoints = 25, yearly_seasonality = 'auto', weekly_seasonality = 'auto', holidays = None, seasonality_prior_scale = 10.0, \
                                                 holidays_prior_scale = 10.0, changepoint_prior_scale = 0.05, mcmc_samples = 0, interval_width = 0.80, uncertainty_samples = 0)
        self.assertIsNotNone(forecastsDict)                         # Checking that there is some stuff there
        self.assertIsNotNone(mse)
        self.assertIsNotNone(resids)
        ##
        # Check Reconcile
        ##
        myDict = reconcile(forecastsDict, 365, mse, resids, "BU")
        self.assertIsNotNone(myDict)                         # Checking that there is some stuff there
        ##
        # Check Bottom Up
        ##
        ##
        # Check that the aggregated values match the unaggregated values
        ##
        self.assertAlmostEqual(myDict[1].yhat[0], sum(myDict[5].yhat[0:5]), delta = myDict[1].yhat[0])
        self.assertAlmostEqual(myDict[1].yhat[0], sum(myDict[73].yhat[0:73]), delta = myDict[1].yhat[0])
        self.assertAlmostEqual(myDict[1].yhat[0], sum(myDict[365].yhat[0:365]), delta = myDict[1].yhat[0])
        ##
        # Check OLS
        ##
        myDict = lastF(data, m = 365, h = 365, comb = "OLS")
        self.assertIsNotNone(myDict)                         # Checking that there is some stuff there
        ##
        # Check that the aggregated values match the unaggregated values
        ##
        self.assertAlmostEqual(myDict[1].yhat[0], sum(myDict[5].yhat[0:5]), delta = myDict[1].yhat[0])
        self.assertAlmostEqual(myDict[1].yhat[0], sum(myDict[73].yhat[0:73]), delta = myDict[1].yhat[0])
        self.assertAlmostEqual(myDict[1].yhat[0], sum(myDict[365].yhat[0:365]), delta = myDict[1].yhat[0])
        ##
        # Check WLSS
        ##
        myDict = lastF(data, m = 365, h = 365, comb = "WLSS")
        self.assertIsNotNone(myDict)                         # Checking that there is some stuff there
        ##
        # Check that the aggregated values match the unaggregated values
        ##
        self.assertAlmostEqual(myDict[1].yhat[0], sum(myDict[5].yhat[0:5]), delta = myDict[1].yhat[0])
        self.assertAlmostEqual(myDict[1].yhat[0], sum(myDict[73].yhat[0:73]), delta = myDict[1].yhat[0])
        self.assertAlmostEqual(myDict[1].yhat[0], sum(myDict[365].yhat[0:365]), delta = myDict[1].yhat[0])
        ##
        # Check WLSV
        ##
        myDict = lastF(data, m = 365, h = 365, comb = "WLSV")
        self.assertIsNotNone(myDict)                         # Checking that there is some stuff there
        ##
        # Check that the aggregated values match the unaggregated values
        ##
        self.assertAlmostEqual(myDict[1].yhat[0], sum(myDict[5].yhat[0:5]), delta = myDict[1].yhat[0])
        self.assertAlmostEqual(myDict[1].yhat[0], sum(myDict[73].yhat[0:73]), delta = myDict[1].yhat[0])
        self.assertAlmostEqual(myDict[1].yhat[0], sum(myDict[365].yhat[0:365]), delta = myDict[1].yhat[0])
        ##
        # Weekly Data
        ##
        date = pd.date_range("2013-04-02", "2017-07-17", freq = "W")
        sessions = np.random.randint(100,40000,size=(len(date),1))
        data = pd.DataFrame(date, columns = ["day"])
        data["sessions"] = sessions
        ##
        # Check AggHier
        ##
        aggs = aggHier(data, m = 52)
        self.assertIsNotNone(aggs)          # Checking that there is some stuff there
        self.assertEqual(len(aggs.keys()), 6)
        self.assertLess(len(aggs[1]), len(aggs[2]))    # Checking that the values were aggregated (made smaller in length)
        self.assertLess(len(aggs[2]), len(aggs[4]))
        self.assertLess(len(aggs[4]), len(aggs[13]))
        self.assertLess(len(aggs[13]), len(aggs[26]))
        ##
        # Check fitProphet
        ##
        forecastsDict, mse, resids = fitProphet(aggs, h = 52, include_history = True, cap = None, capF = None, \
                                                 changepoints = None, n_changepoints = 25, yearly_seasonality = 'auto', weekly_seasonality = 'auto', holidays = None, seasonality_prior_scale = 10.0, \
                                                 holidays_prior_scale = 10.0, changepoint_prior_scale = 0.05, mcmc_samples = 0, interval_width = 0.80, uncertainty_samples = 0)
        self.assertIsNotNone(forecastsDict)                         # Checking that there is some stuff there
        self.assertIsNotNone(mse)
        self.assertIsNotNone(resids)
        ##
        # Check Reconcile
        ##
        myDict = reconcile(forecastsDict, 52, mse, resids, "BU")
        self.assertIsNotNone(myDict)                         # Checking that there is some stuff there
        ##
        # Check Bottom Up
        ##
        ##
        # Check that the aggregated values match the unaggregated values
        ##
        self.assertAlmostEqual(myDict[1].yhat[0], sum(myDict[4].yhat[0:4]), delta = myDict[1].yhat[0])
        self.assertAlmostEqual(myDict[1].yhat[0], sum(myDict[13].yhat[0:13]), delta = myDict[1].yhat[0])
        self.assertAlmostEqual(myDict[1].yhat[0], sum(myDict[52].yhat[0:52]), delta = myDict[1].yhat[0])
        ##
        # Check OLS
        ##
        myDict = lastF(data, m = 52, h = 52, comb = "OLS")
        self.assertIsNotNone(myDict)                         # Checking that there is some stuff there
        ##
        # Check that the aggregated values match the unaggregated values
        ##
        self.assertAlmostEqual(myDict[1].yhat[0], sum(myDict[4].yhat[0:4]), delta = myDict[1].yhat[0])
        self.assertAlmostEqual(myDict[1].yhat[0], sum(myDict[13].yhat[0:13]), delta = myDict[1].yhat[0])
        self.assertAlmostEqual(myDict[1].yhat[0], sum(myDict[52].yhat[0:52]), delta = myDict[1].yhat[0])
        ##
        # Check WLSS
        ##
        myDict = lastF(data, m = 52, h = 52, comb = "WLSS")
        self.assertIsNotNone(myDict)                         # Checking that there is some stuff there
        ##
        # Check that the aggregated values match the unaggregated values
        ##
        self.assertAlmostEqual(myDict[1].yhat[0], sum(myDict[4].yhat[0:4]), delta = myDict[1].yhat[0])
        self.assertAlmostEqual(myDict[1].yhat[0], sum(myDict[13].yhat[0:13]), delta = myDict[1].yhat[0])
        self.assertAlmostEqual(myDict[1].yhat[0], sum(myDict[52].yhat[0:52]), delta = myDict[1].yhat[0])
        ##
        # Check WLSV
        ##
        myDict = lastF(data, m = 52, h = 52, comb = "WLSV")
        self.assertIsNotNone(myDict)                         # Checking that there is some stuff there
        ##
        # Check that the aggregated values match the unaggregated values
        ##
        self.assertAlmostEqual(myDict[1].yhat[0], sum(myDict[4].yhat[0:4]), delta = myDict[1].yhat[0])
        self.assertAlmostEqual(myDict[1].yhat[0], sum(myDict[13].yhat[0:13]), delta = myDict[1].yhat[0])
        self.assertAlmostEqual(myDict[1].yhat[0], sum(myDict[52].yhat[0:52]), delta = myDict[1].yhat[0])
        ##
        # Monthly Data
        ##
        date = pd.date_range("2013-04-02", "2017-07-17", freq = "M")
        sessions = np.random.randint(100,40000,size=(len(date),1))
        data = pd.DataFrame(date, columns = ["day"])
        data["sessions"] = sessions
        ##
        # Check AggHier
        ##
        aggs = aggHier(data, m = 12)
        self.assertIsNotNone(aggs)          # Checking that there is some stuff there
        self.assertEqual(len(aggs.keys()), 6)
        self.assertLess(len(aggs[1]), len(aggs[2]))    # Checking that the values were aggregated (made smaller in length)
        self.assertLess(len(aggs[2]), len(aggs[3]))
        self.assertLess(len(aggs[3]), len(aggs[4]))
        self.assertLess(len(aggs[4]), len(aggs[6]))
        ##
        # Check fitProphet
        ##
        forecastsDict, mse, resids = fitProphet(aggs, h = 12, include_history = True, cap = None, capF = None, \
                                                 changepoints = None, n_changepoints = 25, yearly_seasonality = 'auto', weekly_seasonality = 'auto', holidays = None, seasonality_prior_scale = 10.0, \
                                                 holidays_prior_scale = 10.0, changepoint_prior_scale = 0.05, mcmc_samples = 0, interval_width = 0.80, uncertainty_samples = 0)
        self.assertIsNotNone(forecastsDict)                         # Checking that there is some stuff there
        self.assertIsNotNone(mse)
        self.assertIsNotNone(resids)
        ##
        # Check Reconcile
        ##
        myDict = reconcile(forecastsDict, 12, mse, resids, "BU")
        self.assertIsNotNone(myDict)                         # Checking that there is some stuff there
        ##
        # Check Bottom Up
        ##
        ##
        # Check that the aggregated values match the unaggregated values
        ##
        self.assertAlmostEqual(myDict[1].yhat[0], sum(myDict[2].yhat[0:2]), delta = myDict[1].yhat[0])
        self.assertAlmostEqual(myDict[1].yhat[0], sum(myDict[3].yhat[0:3]), delta = myDict[1].yhat[0])
        self.assertAlmostEqual(myDict[1].yhat[0], sum(myDict[6].yhat[0:6]), delta = myDict[1].yhat[0])
        ##
        # Check OLS
        ##
        myDict = lastF(data, m = 12, h = 12, comb = "OLS")
        self.assertIsNotNone(myDict)                         # Checking that there is some stuff there
        ##
        # Check that the aggregated values match the unaggregated values
        ##
        self.assertAlmostEqual(myDict[1].yhat[0], sum(myDict[2].yhat[0:2]), delta = myDict[1].yhat[0])
        self.assertAlmostEqual(myDict[1].yhat[0], sum(myDict[3].yhat[0:3]), delta = myDict[1].yhat[0])
        self.assertAlmostEqual(myDict[1].yhat[0], sum(myDict[6].yhat[0:6]), delta = myDict[1].yhat[0])
        ##
        # Check WLSS
        ##
        myDict = lastF(data, m = 12, h = 12, comb = "WLSS")
        self.assertIsNotNone(myDict)                         # Checking that there is some stuff there
        ##
        # Check that the aggregated values match the unaggregated values
        ##
        self.assertAlmostEqual(myDict[1].yhat[0], sum(myDict[2].yhat[0:2]), delta = myDict[1].yhat[0])
        self.assertAlmostEqual(myDict[1].yhat[0], sum(myDict[3].yhat[0:3]), delta = myDict[1].yhat[0])
        self.assertAlmostEqual(myDict[1].yhat[0], sum(myDict[6].yhat[0:6]), delta = myDict[1].yhat[0])
        ##
        # Check WLSV
        ##
        myDict = lastF(data, m = 12, h = 12, comb = "WLSV")
        self.assertIsNotNone(myDict)                         # Checking that there is some stuff there
        ##
        # Check that the aggregated values match the unaggregated values
        ##
        self.assertAlmostEqual(myDict[1].yhat[0], sum(myDict[2].yhat[0:2]), delta = myDict[1].yhat[0])
        self.assertAlmostEqual(myDict[1].yhat[0], sum(myDict[3].yhat[0:3]), delta = myDict[1].yhat[0])
        self.assertAlmostEqual(myDict[1].yhat[0], sum(myDict[6].yhat[0:6]), delta = myDict[1].yhat[0])
        ##
        # Check BoxCox Transform
        ##
        myDict = lastF(data, m = 12, h = 12, comb = "WLSV", transform = "BoxCox")
        self.assertIsNotNone(myDict)                         # Checking that there is some stuff there
        ##
        # Testing for system exit
        ##
        with self.assertRaises(SystemExit):
            myDict = lastF(data, m = 1, h = 12, comb = "WLSV")
        with self.assertRaises(SystemExit):
            myDict = lastF(data, m = len(data), h = 12, comb = "WLSV")
        with self.assertRaises(SystemExit):
            myDict = lastF(data, m = 12, h = 12, comb = "WLSV", aggList = [2,4])
        with self.assertRaises(SystemExit):
            myDict = lastF(data, m = 12, h = 12, comb = "FP")
        with self.assertRaises(SystemExit):
            myDict = lastF(data, m = 17, h = 12, comb = "WLSV")
        
        
if __name__ == '__main__':
    unittest.main()