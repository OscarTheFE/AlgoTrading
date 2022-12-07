# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 21:35:20 2022

@author: lenovo
"""

import TAQAdjust as TAQAdjust
import TAQCleaner as TAQCleaner
import TAQStats as TAQStats 
import TAQAutoCorr_analysis as TAQAutoCorr_analysis
import TAQPlot as TAQPlot 
import TAQFilter as TAQFilter
import PortfolioOptimization as PortfolioOptimization 
from TAQTradesReader import TAQTradesReader
from TAQQuotesReader import TAQQuotesReader
import TAQFilter
import os
import pyarrow
import fastparquet

import fastparquet as fp

from math import sqrt
from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp, options
import TAQUtils
import pandas as pd
import numpy as np



if __name__ == "__main__":
    relative_path = "E://algo_data"
    trade_path = os.path.join(relative_path, "trade_data//trades_test")
    quote_path = os.path.join(relative_path, "quote_data//quotes_test")
    trade_dir = os.listdir(trade_path)
    quote_dir = os.listdir(quote_path)
    
    #Run the adjust file to adjust for stock volume and price 
    factor_df = pd.read_csv("S&P500_factors.csv")
    splitting_df = pd.read_csv("splitting_info.csv")
    
    """
    for trade in trade_dir:
        if not trade.startswith("."):
            date = trade
            TAQAdjust_obj = TAQAdjust.TAQAdjust(date, factor_df, splitting_df)
            TAQAdjust_obj.retrieve_trade()
    
    for quote in quote_dir:
        if not quote.startswith("."):
            date = quote
            TAQAdjust_obj = TAQAdjust.TAQAdjust(date, factor_df,splitting_df)
            TAQAdjust_obj.retrieve_quote()

    TAQAdjust.concat_date()
    """
    
    
    TAQPlot_obj = TAQPlot.TAQPlot()
    TAQPlot_obj.plot_adjusted("ACS")
    
    
    #Clean outliers in the the tick data 
    TAQCleaner_obj = TAQCleaner.TAQCleaner("ACS")
    TAQCleaner_obj.clean_quotes()
    TAQCleaner_obj.clean_trades()
    cleaned_quote = TAQCleaner_obj.get_quote_df()
    cleaned_trade = TAQCleaner_obj.get_trade_df()
    cleaned_quote.to_parquet("cleaned_quote_ACS.parquet.gzip", compression="gzip")
    cleaned_trade.to_parquet("cleaned_trade_ACS.parquet.gzip", compression="gzip")
    
    
    TAQPlot_obj.plot_cleaned("ACS")
    
    #computing statistical measures on both cleaned and uncleaned data
    TAQStats_obj = TAQStats.TAQStats("ACS")
    print(TAQStats_obj.get_statistic_table(10,"trade"))
    
    #Perform autocorrelation analysis 
    TAQAutoCorr_analysis_obj = TAQAutoCorr_analysis.TAQAutoCorr_analysis("ACS")
    pass_1_count = TAQAutoCorr_analysis_obj.Ljung_Box_test(1)
    pass_3_count = TAQAutoCorr_analysis_obj.Ljung_Box_test(3)
    print(pass_1_count)
    print(pass_3_count)

    adf = TAQAutoCorr_analysis_obj.Dickey_Fuller_test("5%")
    print(adf)
    

    TAQPlot_obj.plot_minimum_freq()
    TAQPlot_obj.plot_statistics_on_ret("ACS",10)
    
    #Perform portfolio analysis:
    TAQFilter.collect_ticker_ret()
    df = pd.read_csv("S&P500_rets.csv")

    turnover_df = TAQUtils.convert_turnover_df(df)
    turnover_df.drop(["JAVA", "SUNW"], axis=1, inplace=True)
    print(turnover_df.mean())

    df = TAQUtils.convert_ret_df(df)
    # Since JAVA renamed itself, It used to be called SUMW. This will cause the cov matrix of return to explode.
    # so we simply discard it from our dataset
    df.drop(["JAVA","SUNW"],axis=1, inplace=True)

    n = df.shape[1]

    # However since this is only the sp500 tickers
    # We will also want to add a risk-free asset which has mean return 2% and zero covariance with other securities

    cov = df.cov().values * sqrt(252)   # Annualized the cov matrix
    cov = np.insert(cov, n, np.zeros(n,), axis=0)
    cov = np.insert(cov, n, np.zeros(n+1,), axis=1)
    cov = matrix(cov)

    # It is like the expected return of underlying tickers in our portfolio
    ret = df.mean().values * 252    # Annualized the ret matrix
    ret = np.append(ret, 0.02)
    ret = matrix(ret)

    N = 100
    # mu is like the risk aversion parameter denoting that how people don't like risk
    # the higher the mu, the more risk aversion that person is
    mus = [10 ** (5.0 * t / N - 1.0) for t in range(70)]
    tickers = df.columns
    for mu in mus:
        PortfolioOptimizer_obj = PortfolioOptimization.PortfolioOptimizer(ret, cov, mu,tickers)
        PortfolioOptimizer_obj.get_statistics()

    

