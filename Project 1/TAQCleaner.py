import pandas as pd
import numpy as np
import os

'''
since we already save an all-period trade record and an all-period quote record for each ticker as a compressed parquet file
our TAQCleaner class will help us clean the corresponding trade and quote record for specific ticker

In order to use it, just pass in a ticker name eg: AAPL  
                                 an option setting write_to_file, default set to be False, meaning we are not saving to file

we may need further discuss on how to deal with other outliers, here I just replace them by nan
but a discussion on do we have to fill those nan values before calculated statistics is still needed
'''


class TAQCleaner:
    def __init__(self, ticker_name, write_to_file=False):
        self.ticker_name = ticker_name
        self.cur_dir = os.getcwd()
        self.trade_df_dir = os.path.join(os.path.join(self.cur_dir,"Data/Trades/Daily_trade"),
                                         self.ticker_name + ".parquet.gzip")

        self.quote_df_dir = os.path.join(os.path.join(self.cur_dir, "Data/Quotes/Daily_quote"),
                                         self.ticker_name + ".parquet.gzip")
        self.write_to_file = write_to_file
        self.trade_df = pd.read_parquet(self.trade_df_dir)
        self.quote_df = pd.read_parquet(self.quote_df_dir)

        self.rolling_window = 5
        self.threshold_error = 5 * 1e-5

    def get_trade_df(self):
        return self.trade_df

    def get_quote_df(self):
        return self.quote_df

    def plot_result(self):
        pass

    def clean_trades(self):
        daily_mean = self.trade_df.groupby("Date").mean()
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

        self.trade_df = self.trade_df.apply(lambda x: cleaning_trade_outlier(x), axis=1)
        if self.write_to_file:
            self.trade_df.to_parquet(self.trade_df_dir,compression="gzip")
        return

    def clean_quotes(self):
        daily_mean = self.quote_df.groupby("Date").mean()
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

        self.quote_df = self.quote_df.apply(lambda x: cleaning_quote_outlier(x), axis=1)
        if self.write_to_file:
            self.quote_df.to_parquet(self.quote_df_dir, compression="gzip")
        return


if __name__ == "__main__":
    TAQCleaner_obj = TAQCleaner("JAVA")
    TAQCleaner_obj.clean_quotes()
    TAQCleaner_obj.clean_trades()
    cleaned_quote = TAQCleaner_obj.get_quote_df()
    cleaned_trade = TAQCleaner_obj.get_trade_df()
    cleaned_quote.to_parquet("cleaned_quote_JAVA.parquet.gzip", compression="gzip")
    cleaned_trade.to_parquet("cleaned_trade_JAVA.parquet.gzip", compression="gzip")