import TAQCleaner
import TAQUtils
import pandas as pd
import numpy as np
import os


class TAQStats():
    def __init__(self,ticker_name):
        self.ticker_name = ticker_name
        self.trade_dir = os.path.join("./Data/Trades/Daily_trade", ticker_name + ".parquet.gzip")
        self.uncleaned_trade_df = pd.read_parquet(self.trade_dir)
        self.quote_dir = os.path.join("./Data/Quotes/Daily_quote", ticker_name + ".parquet.gzip")
        self.uncleaned_quote_df = pd.read_parquet(self.quote_dir)
        self.TAQCleaner_obj = TAQCleaner.TAQCleaner(self.ticker_name)
        self.TAQCleaner_obj.clean_trades()
        self.TAQCleaner_obj.clean_quotes()

    def get_day_length(self):
        return len(self.uncleaned_trade_df["Date"].unique())

    def get_number_trade(self,use_cleaned=False):
        if not use_cleaned:
            df = self.uncleaned_trade_df
        else:
            df = self.TAQCleaner_obj.get_trade_df()
        return df["Adjusted_size"].sum()

    def get_ret_df(self, time_minute, type_, use_cleaned=False):
        if not use_cleaned:
            if type_ == "trade":
                df = self.uncleaned_trade_df
            elif type_ == "quote":
                df = self.uncleaned_quote_df
        else:
            if type_ == "trade":
                df = self.TAQCleaner_obj.get_trade_df()
            elif type_ == "quote":
                df = self.TAQCleaner_obj.get_quote_df()
        ret_df = TAQUtils.get_period_return(df, time_minute, type_)
        return ret_df

    def get_number_quote(self, use_cleaned=False):
        if not use_cleaned:
            df = self.uncleaned_quote_df
        else:
            df = self.TAQCleaner_obj.get_quote_df()
        return df["Adjusted_bid_size"].sum() + df["Adjusted_ask_size"].sum()

    def get_tradequote_fraction(self,use_cleaned=False):
        if not use_cleaned:
            return self.get_number_trade()/self.get_number_quote()
        else:
            return self.get_number_trade(use_cleaned=True) / self.get_number_quote(use_cleaned=True)

    def find_maximum_drawdown(self, ret_list):
        cum_ret_list = np.cumprod(ret_list + 1)
        cum_ret_list = cum_ret_list[::-1]
        minnum = cum_ret_list[0]
        maxnum = 0
        local_max = 0
        for i in range(len(ret_list)):
            minnum = min(cum_ret_list[i], minnum)
            if maxnum <= (cum_ret_list[i] - minnum):
                local_max = cum_ret_list[i]
            maxnum = max(maxnum, (cum_ret_list[i] - minnum))
        return maxnum / local_max

    def get_statistics_on_ret(self, time_minute, type_, use_cleaned=False):
        if time_minute >= 1:
            # if input time_minute is larger or equal to 1, we consider it as a minute_freq
            freq = f"{int(time_minute)}T"
        else:
            # if input time_minute is less than 1, we multiply it by 60 seconds/minute to get second_freq
            freq = f"{int(time_minute * 60)}S"

        if not use_cleaned:
            if type_ == "trade":
                df = self.uncleaned_trade_df
            elif type_ == "quote":
                df = self.uncleaned_quote_df
        else:
            if type_ == "trade":
                df = self.TAQCleaner_obj.get_trade_df()
            elif type_ == "quote":
                df = self.TAQCleaner_obj.get_quote_df()
        ret_df = TAQUtils.get_period_return(df, time_minute, type_)
        mean_ret = ret_df.mean().iloc[0] * 252
        median_ret = ret_df.median().iloc[0]
        std_ret = ret_df.std().iloc[0] * (252**0.5)
        absolute_dev = abs(ret_df).median().iloc[0]
        skew = ret_df.skew().iloc[0]
        kurtosis = ret_df.kurtosis().iloc[0]
        top_ten_return = ret_df.nlargest(10, columns=freq + "_ret")
        bottom_ten_return = ret_df.nsmallest(10, columns=freq + "_ret")
        maximum_drawdown = self.find_maximum_drawdown(ret_df.values)
        return mean_ret, median_ret, std_ret, absolute_dev, skew, kurtosis, top_ten_return, bottom_ten_return, maximum_drawdown

    def get_statistic_table(self,time_minute, type_):
        statistic_measures = ["day_length", "quote_number", "trade_number", "fraction", "mean_ret", "median_ret",
                              "std_ret", "absolute_dev", "skew", "kurtosis", "maximum_drawdown"]
        day_length = self.get_day_length()
        quote_number = self.get_number_quote()
        trade_number = self.get_number_trade()
        fraction = self.get_tradequote_fraction()
        mean_ret, median_ret, std_ret, absolute_dev, skew, kurtosis, top_ten_return, bottom_ten_return, maximum_drawdown = \
            self.get_statistics_on_ret(time_minute, type_)

        uncleaned_statistics = [day_length, quote_number, trade_number, fraction, mean_ret, median_ret, std_ret,
                                absolute_dev, skew, kurtosis, maximum_drawdown]

        day_length = self.get_day_length()#use_cleaned=True)
        quote_number = self.get_number_quote(use_cleaned=True)
        trade_number = self.get_number_trade(use_cleaned=True)
        fraction = self.get_tradequote_fraction(use_cleaned=True)
        mean_ret, median_ret, std_ret, absolute_dev, skew, kurtosis, top_ten_return, bottom_ten_return, maximum_drawdown = \
            self.get_statistics_on_ret(time_minute, type_, use_cleaned=True)

        cleaned_statistics = [day_length, quote_number, trade_number, fraction, mean_ret, median_ret, std_ret,
                                absolute_dev, skew, kurtosis, maximum_drawdown]

        df = pd.DataFrame({"Statistic_measures":statistic_measures, "uncleaned": uncleaned_statistics, "cleaned": cleaned_statistics})
        df.set_index("Statistic_measures", inplace=True)
        return df

if __name__ == "__main__":
    TAQStats_obj = TAQStats("TXT")
    freq_list = [1 / 6, 3 / 6, 1, 5, 10, 20, 30]
    for time in freq_list:
        print("Time frequency:", time, "min")
        print(TAQStats_obj.get_statistic_table(time,"trade"))







