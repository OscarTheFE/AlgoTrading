import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

import TAQAutoCorr_analysis
import TAQCleaner
import TAQStats
import TAQUtils


class TAQPlot():

    def __init__(self):
        pass

    def plot_adjusted(self, ticker_name):
        splitting_df = pd. read_csv("./splitting_info.csv")
        trade_ticker_name_dir = os.path.join("./Data/Trades/Daily_trade",ticker_name + ".parquet.gzip")
        quote_ticker_name_dir = os.path.join("./Data/Quotes/Daily_quote",ticker_name + ".parquet.gzip")
        if ticker_name not in list(splitting_df["Ticker"]):
            print("this ticker did not experience any change in adjust factor")
        else:
            splitting_date = splitting_df.loc[splitting_df.Ticker == ticker_name]["Splitting_date"].iloc[0]
            splitting_date = pd.to_datetime(splitting_date)
            trade_df = pd.read_parquet(trade_ticker_name_dir)
            quote_df = pd.read_parquet(quote_ticker_name_dir)

            # plot trade data around splitting_date

            # find the row index of splitting_date
            trade_splitting_date_index = trade_df.index[trade_df.Date == pd.to_datetime(splitting_date)][0]
            plot_begin_index = trade_splitting_date_index - 20
            plot_end_index = trade_splitting_date_index + 20
            trade_plot_df = trade_df[plot_begin_index:plot_end_index]

            quote_splitting_date_index = quote_df.index[quote_df.Date == pd.to_datetime(splitting_date)][0]
            plot_begin_index = quote_splitting_date_index - 20
            plot_end_index = quote_splitting_date_index + 20
            quote_plot_df = quote_df[plot_begin_index:plot_end_index]

            plt.figure(figsize=(16,9))

            plt.subplot(321)
            plt.scatter(np.arange(40), trade_plot_df["Price"],label="original price", color="r")
            plt.scatter(np.arange(40), trade_plot_df["Adjusted_price"], label="adjusted price", color="b")
            plt.title(ticker_name + " trade price before/after adjustment")
            plt.legend()

            plt.subplot(322)
            plt.scatter(np.arange(40), trade_plot_df["Size"], label="original size", color="r")
            plt.scatter(np.arange(40), trade_plot_df["Adjusted_size"], label="adjusted size", color="b")
            plt.title(ticker_name + " trade size before/after adjustment")
            plt.legend()

            plt.subplot(323)
            plt.scatter(np.arange(40), quote_plot_df["Bid_price"], label="original bid price", color="r")
            plt.scatter(np.arange(40), quote_plot_df["Adjusted_bid_price"], label="adjusted bid price", color="b")
            plt.title(ticker_name + " quote bid price before/after adjustment")
            plt.legend()

            plt.subplot(324)
            plt.scatter(np.arange(40), quote_plot_df["Ask_price"], label="original ask price", color="r")
            plt.scatter(np.arange(40), quote_plot_df["Adjusted_ask_price"], label="adjusted ask price", color="b")
            plt.title(ticker_name + " quote ask price before/after adjustment")
            plt.legend()

            plt.subplot(325)
            plt.scatter(np.arange(40), quote_plot_df["Bid_size"], label="original bid size", color="r")
            plt.scatter(np.arange(40), quote_plot_df["Adjusted_bid_size"], label="adjusted bid size", color="b")
            plt.title(ticker_name + " quote bid size before/after adjustment")
            plt.legend()

            plt.subplot(326)
            plt.scatter(np.arange(40), quote_plot_df["Ask_size"], label="original ask size", color="r")
            plt.scatter(np.arange(40), quote_plot_df["Adjusted_ask_size"], label="adjusted ask size", color="b")
            plt.title(ticker_name + " quote ask size before/after adjustment")
            plt.legend()
            plt.show()
        return

    def plot_cleaned(self,ticker_name):
        trade_ticker_name_dir = os.path.join("./Data/Trades/Daily_trade", ticker_name + ".parquet.gzip")
        quote_ticker_name_dir = os.path.join("./Data/Quotes/Daily_quote", ticker_name + ".parquet.gzip")

        TAQCleaner_obj = TAQCleaner.TAQCleaner(ticker_name)
        uncleaned_trade_df = pd.read_parquet(trade_ticker_name_dir)
        uncleaned_quote_df = pd.read_parquet(quote_ticker_name_dir)

        TAQCleaner_obj.clean_trades()
        TAQCleaner_obj.clean_quotes()

        cleaned_trade_df = TAQCleaner_obj.get_trade_df()
        cleaned_quote_df = TAQCleaner_obj.get_quote_df()

        plt.figure(figsize=(16, 9))

        plt.subplot(311)
        sns.kdeplot(uncleaned_trade_df["Adjusted_price"], shade=True, color="g", label="uncleaned_trade_price",
                    alpha=0.7)
        sns.kdeplot(cleaned_trade_df["Adjusted_price"], shade=True, color="r", label="cleaned_trade_price", alpha=0.7)
        plt.title(ticker_name + " trade price before/after cleaning")
        plt.legend()

        plt.subplot(312)
        sns.kdeplot(uncleaned_quote_df["Adjusted_bid_price"], shade=True, color="g", label="uncleaned_bid_price",
                    alpha=0.7)
        sns.kdeplot(cleaned_quote_df["Adjusted_bid_price"], shade=True, color="r", label="cleaned_bid_price", alpha=0.7)
        plt.title(ticker_name + " bid price before/after cleaning")
        plt.legend()

        plt.subplot(313)
        sns.kdeplot(uncleaned_quote_df["Adjusted_ask_price"], shade=True, color="g", label="uncleaned_ask_price",
                    alpha=0.7)
        sns.kdeplot(cleaned_quote_df["Adjusted_ask_price"], shade=True, color="r", label="cleaned_ask_price", alpha=0.7)
        plt.title(ticker_name + " ask price before/after cleaning")
        plt.legend()

        plt.show()
        return

    def plot_statistics_on_ret(self,ticker_name,time_minute):
        TAQStats_obj = TAQStats.TAQStats(ticker_name)
        uncleaned_trade_ret = TAQStats_obj.get_ret_df(time_minute, type_="trade", use_cleaned=False)
        uncleaned_quote_ret = TAQStats_obj.get_ret_df(time_minute, type_="quote", use_cleaned=False)
        cleaned_trade_ret = TAQStats_obj.get_ret_df(time_minute, type_="trade", use_cleaned=True)
        cleaned_quote_ret = TAQStats_obj.get_ret_df(time_minute, type_="quote", use_cleaned=True)

        plt.figure(figsize=(16,9))
        plt.subplot(121)
        plt.plot(uncleaned_trade_ret.reset_index().index, uncleaned_trade_ret.values, color='green', linestyle='dashed',
                 label="uncleaned trade")
        plt.plot(cleaned_trade_ret.reset_index().index, cleaned_trade_ret.values, color='orange',
                 label="cleaned trade")
        plt.title(ticker_name + " trade return plot before/after cleaning in " + "{:.2f}".format(time_minute) + " frequency")
        plt.legend()

        plt.subplot(122)
        plt.plot(uncleaned_quote_ret.reset_index().index, uncleaned_quote_ret.values, color='green', linestyle='dashed',
                 label="uncleaned quote")
        plt.plot(cleaned_quote_ret.reset_index().index, cleaned_quote_ret.values, color='orange',
                 label="cleaned quote")
        plt.title(ticker_name + " quote return plot before/after cleaning " + "{:.2f}".format(time_minute) + " frequency")
        plt.legend()

        plt.show()

        # plot the distribution of return

        plt.figure(figsize=(16, 9))

        plt.subplot(121)
        sns.kdeplot(uncleaned_trade_ret[uncleaned_trade_ret.columns[0]], color="g", label="uncleaned_trade_returns",alpha=0.7)
        sns.kdeplot(cleaned_trade_ret[cleaned_trade_ret.columns[0]],  color="orange", label="cleaned_trade_returns", alpha=0.7)
        plt.title(ticker_name + " trade return distribution before/after cleaning " + "{:.2f}".format(time_minute) + " frequency")
        plt.legend()

        plt.subplot(122)
        sns.kdeplot(uncleaned_quote_ret[uncleaned_quote_ret.columns[0]], color="g", label="uncleaned_quote_returns", alpha=0.7)
        sns.kdeplot(cleaned_quote_ret[cleaned_quote_ret.columns[0]], color="orange", label="cleaned_quote_returns", alpha=0.7)
        plt.title(ticker_name + " quote return distribution before/after cleaning " + "{:.2f}".format(time_minute) + " frequency")
        plt.legend()

        plt.show()

        return

    def plot_minimum_freq(self):
        # read all the S&P 500 tickers
        sp_df = pd.read_csv("./S&P500_factors.csv")
        sp500_ticker_list = list(sp_df.Ticker.unique())

        # for each ticker, we run a Ljung-box test with lag 1 and lag 3
        # we also run a Dicky-Fuller test with auto lag on each ticker and compare the result
        ljung_freq1_list = []
        ljung_freq3_list = []
        adf_freq_list = []

        for ticker_name in sp500_ticker_list:
            print("Analyzing auto-correlation for {}".format(ticker_name))
            TAQAutoCorr_analysis_obj = TAQAutoCorr_analysis.TAQAutoCorr_analysis(ticker_name)
            if not TAQAutoCorr_analysis_obj.data_exist:
                continue
            ljung_freq1 = TAQAutoCorr_analysis_obj.Ljung_Box_test(1)
            ljung_freq3 = TAQAutoCorr_analysis_obj.Ljung_Box_test(3)
            adf_freq = TAQAutoCorr_analysis_obj.Dickey_Fuller_test()

            ljung_freq1_list.append(ljung_freq1)
            ljung_freq3_list.append(ljung_freq3)
            adf_freq_list.append(adf_freq)

        ljung_freq1_list = np.array(ljung_freq1_list) * 60      # convert to measure of seconds
        ljung_freq3_list = np.array(ljung_freq3_list) * 60       # convert to measure of seconds
        adf_freq_list = np.array(adf_freq_list) * 60        # convert to measure of seconds

        count1_list = []
        count2_list = []
        count3_list = []
        freq_list = [1 / 6, 3 / 6, 1, 5, 10, 20, 30]
        for freq in freq_list:
            count1 = 0
            count2 = 0
            count3 = 0
            for i in range(len(adf_freq_list)):
                if ljung_freq1_list[i] <= freq * 60:
                    count1 += 1
                if ljung_freq3_list[i] <= freq * 60:
                    count2 += 1
                if adf_freq_list[i] <= freq * 60:
                    count3 += 1
            count1_list.append(count1)
            count2_list.append(count2)
            count3_list.append(count3)

        table = pd.DataFrame({"Ljung_freq1": count1_list,"Ljung_freq3": count2_list,"Adf_freq": count3_list}
                             , index=np.array(freq_list)*60)

        plt.figure(figsize=(16, 9))

        plt.plot(table.index, table["Ljung_freq1"], color="dodgerblue", label="Ljung_freq1")
        plt.plot(table.index, table["Ljung_freq3"], color="g", label="Ljung_freq3")
        plt.plot(table.index, table["Adf_freq"], color="orange", label="Adf_freq")
        plt.ylabel("Quantity")
        plt.xlabel("Seconds")
        plt.title(" Test on auto-correlation: Passing test ticker amount Versus Minimum time interval")
        plt.legend()

        plt.show()

        return


if __name__ == "__main__":
    TAQPlot_obj = TAQPlot()
    # TAQPlot_obj.plot_adjusted("ABC")
    # TAQPlot_obj.plot_cleaned("ABC")
    TAQPlot_obj.plot_minimum_freq()
    # freq_list = [1 / 6, 3 / 6, 1, 5, 10, 20, 30]
    # TAQPlot_obj.plot_statistics_on_ret("TXT", 30)
