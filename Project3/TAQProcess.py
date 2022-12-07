import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed, cpu_count

pd.options.mode.chained_assignment = None
'''
since we already save an all-period trade record and an all-period quote record for each ticker as pickled file
our TAQProcess class will help us clean the corresponding trade and quote and computing data matrix 
that will be further used in building impact model.

we may need further discuss on how to deal with other outliers, here I just replace them by nan
but a discussion on do we have to fill those nan values before calculated statistics is still needed
'''
# data storage

ret_matrix = {}


class TAQProcess(object):
    def __init__(self):
        sp_df = pd.read_csv("./S&P500_factors.csv")
        self.ticker_name_list = list(sp_df.Ticker.unique())
        self.rolling_window = 5
        self.threshold_error = 5 * 1e-5
        self.date_list = os.listdir("./data/trades_test")

    def clean_trades(self, df):
        daily_mean = df.groupby("Date").mean()
        rolling_mean = daily_mean.rolling(self.rolling_window).mean()
        rolling_std = daily_mean.rolling(self.rolling_window).std()

        def cleaning_trade_outlier(x):
            date = x.Date
            mean = rolling_mean.loc[rolling_mean.index == date]
            std = rolling_std.loc[rolling_std.index == date]
            # for first k days, since we do not have enough historical rolling window size
            # we decide to skip those days and leave the data unchanged
            if mean.isnull().sum().sum() != len(mean.columns):
                # replace price,size,adj_price,adj_size of outliers with np.nan
                mean_price = mean.Adjusted_price.iloc[0]
                std_price = std.Adjusted_price.iloc[0]
                if (x.Adjusted_price > mean_price + 2 * std_price + self.threshold_error * mean_price) or \
                        (x.Adjusted_price < mean_price - 2 * std_price + self.threshold_error * mean_price):
                    x.Price = np.nan
                    x.Adjusted_price = np.nan
                    x.Size = np.nan
                    x.Adjusted_size = np.nan
            return x

        df = df.apply(lambda x: cleaning_trade_outlier(x), axis=1)
        return df

    def clean_quotes(self, df):
        daily_mean = df.groupby("Date").mean()
        rolling_mean = daily_mean.rolling(self.rolling_window).mean()
        rolling_std = daily_mean.rolling(self.rolling_window).std()

        def cleaning_quote_outlier(x):
            date = x.Date
            mean = rolling_mean.loc[rolling_mean.index == date]
            std = rolling_std.loc[rolling_std.index == date]
            # for first k days, since we do not have enough historical rolling window size
            # we decide to skip those days and leave the data unchanged
            if mean.isnull().sum().sum() != len(mean.columns):
                # replace price,size,adj_price,adj_size of outliers with np.nan
                mean_ask_price = mean.Adjusted_ask_price.iloc[0]
                mean_bid_price = mean.Adjusted_bid_price.iloc[0]
                std_ask_price = std.Adjusted_ask_price.iloc[0]
                std_bid_price = std.Adjusted_bid_price.iloc[0]
                if (x.Adjusted_ask_price > mean_ask_price + 2 * std_ask_price + self.threshold_error * mean_ask_price) or \
                        (x.Adjusted_ask_price < mean_ask_price - 2 * std_ask_price + self.threshold_error * mean_ask_price):
                    x.Ask_price = np.nan
                    x.Adjusted_ask_price = np.nan
                    x.Ask_size = np.nan
                    x.Adjusted_ask_size = np.nan

                if (x.Adjusted_bid_price > mean_bid_price + 2 * std_bid_price + self.threshold_error * mean_bid_price) or \
                        (x.Adjusted_bid_price < mean_bid_price - 2 * std_bid_price + self.threshold_error * mean_bid_price):
                    x.Bid_price = np.nan
                    x.Adjusted_bid_price = np.nan
                    x.Bid_size = np.nan
                    x.Adjusted_bid_size = np.nan
            return x

        df = df.apply(lambda x: cleaning_quote_outlier(x), axis=1)
        return df

    def process_data(self, num_cpus):
        print("Parallel process quote data:")
        quotes_result = Parallel(n_jobs=num_cpus, backend='loky')(
            delayed(self.traverse_ticker_quotes)(ticker) for ticker in self.ticker_name_list)

        for res in quotes_result:
            if res[1].empty:
                continue
            ret_matrix[res[0]] = res[1]

        df = pd.concat(ret_matrix, axis=1).sum(axis=1, level=0)
        return df

    def traverse_ticker_quotes(self, ticker):
        print("processing {} quote data".format(ticker))
        quote_dir = os.path.join(os.getcwd(), "data/quotes_test")
        quote_dates_list = os.listdir(quote_dir)

        quotes_columns = ["Seconds_from_Epoc", "Millis", "Ask_price", "Bid_price", "Ask_size", "Bid_size",
                          "Adjusted_bid_price",
                          "Adjusted_ask_price", "Adjusted_bid_size", "Adjusted_ask_size", "Date"]

        df = pd.DataFrame(columns=quotes_columns)
        ticker_filename = ticker + ".pkl"
        for quote_date in quote_dates_list:
            if quote_date.startswith("."):
                continue
            sub_path = os.path.join(os.path.join(quote_dir, quote_date), "Adjusted")
            # if the ticker is not in sp500 namelist that date, we skip to next date
            if ticker_filename not in os.listdir(sub_path):
                continue
            data = pd.read_pickle(os.path.join(sub_path, ticker_filename))
            df = pd.concat([df, data], ignore_index=True)

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(by=["Date", "Millis"])
        df["Datetime"] = df["Seconds_from_Epoc"] + df["Millis"] / 1000
        df["Datetime"] = pd.to_datetime(df["Datetime"], unit='s')
        df.reset_index(drop=True, inplace=True)

        if df.shape[0] == 0:
            return (ticker, df)
        df = self.clean_quotes(df)

        def get_mid_quote_returns(df):
            # sub_df = df.loc[
            #    (df.Date > pd.to_datetime(start_date)) & (df.Date <= pd.to_datetime(end_date))]
            sub_df = df.copy()
            sub_df = sub_df[["Datetime", "Adjusted_bid_price", "Adjusted_ask_price"]]
            sub_df["Adjusted_mid_quote"] = (sub_df["Adjusted_bid_price"] + sub_df["Adjusted_ask_price"]) / 2
            freq = "5t"
            r = sub_df.resample(freq, closed="left", label="right", on="Datetime")
            first = r.agg("first")
            last = r.agg("last")
            ret_df = last[["Adjusted_mid_quote"]] / first[["Adjusted_mid_quote"]] - 1
            ret_df.rename(columns={"Adjusted_mid_quote": freq + "_ret"}, inplace=True)
            return ret_df[[freq + "_ret"]]

        ret_df = get_mid_quote_returns(df)
        # only consider the time that the market is open
        ret_df = ret_df.loc[ret_df.index.strftime("%Y%m%d").isin(self.date_list)]
        ret_df = ret_df.loc[pd.to_datetime(ret_df.index.strftime("%H:%M")) >= pd.to_datetime("13:35")]
        ret_df = ret_df.loc[pd.to_datetime(ret_df.index.strftime("%H:%M")) <= pd.to_datetime("20:00")]
        print("done extracting 5-min return from {} quote data".format(ticker))
        # print(ticker)
        # print(ret_df)
        # print()
        return (ticker, ret_df)

if __name__ == "__main__":
    TAQProcess_obj = TAQProcess()
    ret_df = TAQProcess_obj.process_data(8)
    ret_df.to_csv("5_mins_mid_quote_ret.csv")

