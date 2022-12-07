from TAQTradesReader import TAQTradesReader
from TAQQuotesReader import TAQQuotesReader
import pandas as pd
import TAQFilter
import os

'''
since we only generate adjusted data in each date folder,
we need to concatenate them into one big dataframe for each stock
'''


def concat_date():

    # get a namelist for all sp500 tickers
    sp_df = pd.read_csv("./S&P500_factors.csv")
    sp500_tickers = list(sp_df.Ticker.unique())
    cur_dir = os.getcwd()

    # concatenate pickle files for trades records
    trade_dir = os.path.join(cur_dir,"Data/Trades")
    trade_save_dir = os.path.join(trade_dir, "Daily_trade")
    if not os.path.exists(trade_save_dir):
        os.mkdir(trade_save_dir)
        print("created a sub-dir as a folder to contain all daily trade compressed parquet files")
    trade_dates_list = os.listdir(trade_dir)
    trades_columns = ["Seconds_from_Epoc","Price","Millis","Timestamp","Size","Adjusted_price","Adjusted_size","Date"]
    for ticker in sp500_tickers:
        df = pd.DataFrame(columns = trades_columns)
        ticker_filename = ticker + ".pkl"
        for trade_date in trade_dates_list:
            if trade_date.startswith("."):
                continue
            if trade_date.endswith(".pkl"):
                continue
            if trade_date.endswith(".gzip"):
                continue
            if trade_date == "Daily_trade":
                continue
            sub_path = os.path.join(os.path.join(trade_dir,trade_date),"Adjusted")
            # if the ticker is not in sp500 namelist that date, we skip to next date
            if ticker_filename not in os.listdir(sub_path):
                continue
            data = pd.read_pickle(os.path.join(sub_path,ticker_filename))
            df = pd.concat([df,data], ignore_index=True)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(by=["Date","Millis"])
        df["Datetime"] = df["Seconds_from_Epoc"] + df["Millis"] / 1000
        df["Datetime"] = pd.to_datetime(df["Datetime"], unit='s')
        df.reset_index(drop=True, inplace=True)
        # df.to_pickle(os.path.join(trade_dir,ticker + ".pkl"))
        df.to_parquet(os.path.join(trade_save_dir,ticker + ".parquet.gzip"),compression = "gzip")

    # concatenate pickle files for quotes records
    quote_dir = os.path.join(cur_dir, "data/Quotes")
    quote_save_dir = os.path.join(quote_dir, "Daily_quote")
    if not os.path.exists(quote_save_dir):
        os.mkdir(quote_save_dir)
        print("created a sub-dir as a folder to contain all daily quote compressed parquet files")
    quote_dates_list = os.listdir(quote_dir)
    quotes_columns = ["Seconds_from_Epoc","Millis","Ask_price","Bid_price","Ask_size","Bid_size","Adjusted_bid_price",
                      "Adjusted_ask_price","Adjusted_bid_size","Adjusted_ask_size","Date"]
    for ticker in sp500_tickers:
        df = pd.DataFrame(columns=quotes_columns)
        ticker_filename = ticker + ".pkl"
        for quote_date in quote_dates_list:
            if quote_date.startswith("."):
                continue
            if quote_date.endswith(".pkl"):
                continue
            if quote_date.endswith(".gzip"):
                continue
            if quote_date == "Daily_quote":
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
        df.reset_index(drop=True,inplace=True)
        # df.to_pickle(os.path.join(quote_dir, ticker + ".pkl"))
        df.to_parquet(os.path.join(quote_save_dir, ticker + ".parquet.gzip"), compression="gzip")
    return


'''
The class TAQAdjust can adjust ticker information for specific date and save as pickle file
only tickers in S&P 500 list will be processed
if a stock experienced any kind of stock splitting or dividend paying which may affect adjust factor
we adjust the historical price and size, namely the price and size before such event happens to the level afterwards.
'''


class TAQAdjust():
    def __init__(self, date, factors_df, splitting_df):
        self.date = date
        self.factors_df = factors_df
        self.factors_df["Date"] = pd.to_datetime(self.factors_df["Date"])
        self.sp500_reference = TAQFilter.read_tickers(date)
        self.cur_dir = os.getcwd()
        splitting_df["Splitting_date"] = pd.to_datetime(splitting_df["Splitting_date"])
        self.splitting_df = splitting_df

    def adjust_price(self, price, previous_factor, new_factor):
        # historical price divided by previous factor to get unadjusted price
        return price / previous_factor * new_factor

    def adjust_share_volume(self, size, previous_factor, new_factor):
        # left to be validated
        return size * previous_factor / new_factor

    def get_factors(self, ticker_name, date_, splitting_date):
        previous_factor_price = \
            self.factors_df.loc[(self.factors_df.Date == pd.to_datetime(date_, format="%Y%m%d"))
                                & (self.factors_df.Ticker == ticker_name)][
                "Cumulative Factor to Adjust Prices"].iloc[0]

        new_factor_price = \
            self.factors_df.loc[(self.factors_df.Date == pd.to_datetime(splitting_date))
                                & (self.factors_df.Ticker == ticker_name)][
                "Cumulative Factor to Adjust Prices"].iloc[0]

        previous_factor_volume = \
            self.factors_df.loc[(self.factors_df.Date == pd.to_datetime(date_, format="%Y%m%d"))
                                & (self.factors_df.Ticker == ticker_name)][
                "Cumulative Factor to Adjust Shares/Vol"].iloc[0]

        new_factor_volume = \
            self.factors_df.loc[(self.factors_df.Date == pd.to_datetime(splitting_date))
                                & (self.factors_df.Ticker == ticker_name)][
                "Cumulative Factor to Adjust Shares/Vol"].iloc[0]

        return previous_factor_price, new_factor_price, previous_factor_volume, new_factor_volume

    def retrieve_trade(self):
        sub_path = "Data/Trades"
        path = os.path.join(self.cur_dir, sub_path)
        whole_path = os.path.join(path, self.date)
        if not os.path.exists(os.path.join(whole_path,"Adjusted")):
            os.mkdir(os.path.join(whole_path,"Adjusted"))
            print("created a sub-dir as a folder to contain all adjusted pickle files")
        tickers_dir = os.listdir(whole_path)
        print("********************************************************************************************")
        print("We are retrieving information from following dir:")
        print(whole_path)
        for ticker in tickers_dir:
            ticker_name = ticker[:-13]
            # filter out useful company tickers according to today's sp500 namelist reference
            if ticker_name in self.sp500_reference:
                NEED_ADJUST = False
                # a necessary filter on stock_splitting factor adjustment
                if ticker_name in list(self.splitting_df["Ticker"]):
                    # if today's date is earlier than stock splitting date
                    # we need to adjust all the historical price in order to match the price gap
                    splitting_date = self.splitting_df.loc[self.splitting_df.Ticker == ticker_name]["Splitting_date"].iloc[0]
                    if pd.to_datetime(self.date, format="%Y%m%d") < pd.to_datetime(splitting_date):
                        NEED_ADJUST = True
                        previous_factor_price, new_factor_price, previous_factor_volume, new_factor_volume = \
                            self.get_factors(ticker_name, self.date, splitting_date)
                dictionary = {}
                print("System is retriving information on {}.".format(ticker_name))
                obj = TAQTradesReader(os.path.join(whole_path, ticker))
                num_rows = obj.getN()
                # print("output of getN: ", num_rows)
                secsFromEpocToMidn = obj.getSecsFromEpocToMidn()
                # print("output of getSecsFromEpocToMidn: ", secsFromEpocToMidn)
                for i in range(num_rows):
                    sub_dic = {}

                    price = obj.getPrice(i)
                    millis = obj.getMillisFromMidn(i)
                    timestamp = obj.getTimestamp(i)
                    size = obj.getSize(i)

                    sub_dic["Seconds_from_Epoc"] = secsFromEpocToMidn
                    sub_dic["Price"] = price
                    sub_dic["Millis"] = millis
                    sub_dic["Timestamp"] = timestamp
                    sub_dic["Size"] = size
                    if NEED_ADJUST:
                        sub_dic["Adjusted_price"] = self.adjust_price(price, previous_factor_price, new_factor_price)
                        sub_dic["Adjusted_size"] = self.adjust_share_volume(size, previous_factor_volume,
                                                                            new_factor_volume)
                    else:
                        sub_dic["Adjusted_price"] = price
                        sub_dic["Adjusted_size"] = size

                    dictionary[i] = sub_dic

                    # use those information to create a dictionary

                    # print("output of getPrice at {}: ".format(i), price)
                    # print("output of getMillisFromMidn at {}: ".format(i), millis)
                    # print("output of getTimestamp at {}: ".format(i), timestamp)
                    # print("output of getSize at {}: ".format(i), size)
                    # print()
                df = pd.DataFrame.from_dict(dictionary, orient="index")
                df["Date"] = pd.to_datetime(self.date, format="%Y%m%d")
                pkl_filename = ticker_name + ".pkl"
                df.to_pickle(os.path.join(os.path.join(whole_path,"Adjusted"),pkl_filename))
        return

    def retrieve_quote(self):
        sub_path = "Data/Quotes"
        path = os.path.join(self.cur_dir, sub_path)
        whole_path = os.path.join(path, self.date)
        if not os.path.exists(os.path.join(whole_path,"Adjusted")):
            os.mkdir(os.path.join(whole_path,"Adjusted"))
            print("created a sub-dir as a folder to contain all adjusted pickle files")
        tickers_dir = os.listdir(whole_path)
        print("********************************************************************************************")
        print("We are retrieving information from following dir:")
        print(whole_path)
        for ticker in tickers_dir:
            ticker_name = ticker[:-13]
            # filter out useful company tickers according to today's sp500 namelist reference
            if ticker_name in self.sp500_reference:
                NEED_ADJUST = False
                # a necessary filter on stock_splitting factor adjustment
                if ticker_name in list(self.splitting_df["Ticker"]):
                    # if today's date is earlier than stock splitting date
                    # we need to adjust all the historical price in order to match the price gap
                    splitting_date = self.splitting_df.loc[self.splitting_df.Ticker == ticker_name]["Splitting_date"].iloc[0]
                    if pd.to_datetime(self.date, format="%Y%m%d") < pd.to_datetime(splitting_date):
                        NEED_ADJUST = True
                        previous_factor_price, new_factor_price, previous_factor_volume, new_factor_volume = \
                            self.get_factors(ticker_name, self.date, splitting_date)

                dictionary = {}
                print("System is retriving information on {}.".format(ticker_name))
                obj = TAQQuotesReader(os.path.join(whole_path, ticker))
                num_rows = obj.getN()
                # print("output of getN: ", num_rows)
                secsFromEpocToMidn = obj.getSecsFromEpocToMidn()
                # print("output of getSecsFromEpocToMidn: ", secsFromEpocToMidn)
                for i in range(num_rows):
                    sub_dic = {}

                    millis = obj.getMillisFromMidn(i)
                    ask_price = obj.getAskPrice(i)
                    bid_price = obj.getBidPrice(i)

                    ask_size = obj.getAskSize(i)
                    bid_size = obj.getBidSize(i)

                    sub_dic["Seconds_from_Epoc"] = secsFromEpocToMidn
                    sub_dic["Millis"] = millis
                    sub_dic["Ask_price"] = ask_price
                    sub_dic["Bid_price"] = bid_price
                    sub_dic["Ask_size"] = ask_size
                    sub_dic["Bid_size"] = bid_size

                    if NEED_ADJUST:
                        sub_dic["Adjusted_bid_price"] = self.adjust_price(bid_price, previous_factor_price,
                                                                          new_factor_price)
                        sub_dic["Adjusted_ask_price"] = self.adjust_price(ask_price, previous_factor_price,
                                                                          new_factor_price)
                        sub_dic["Adjusted_bid_size"] = self.adjust_share_volume(bid_size, previous_factor_volume,
                                                                            new_factor_volume)
                        sub_dic["Adjusted_ask_size"] = self.adjust_share_volume(ask_size, previous_factor_volume,
                                                                                new_factor_volume)
                    else:
                        sub_dic["Adjusted_bid_price"] = bid_price
                        sub_dic["Adjusted_ask_price"] = ask_price
                        sub_dic["Adjusted_bid_size"] = bid_size
                        sub_dic["Adjusted_ask_size"] = ask_size

                    dictionary[i] = sub_dic

                    # use those information to create a dictionary

                    # print("output of getPrice at {}: ".format(i), price)
                    # print("output of getMillisFromMidn at {}: ".format(i), millis)
                    # print("output of getTimestamp at {}: ".format(i), timestamp)
                    # print("output of getSize at {}: ".format(i), size)
                    # print()
                df = pd.DataFrame.from_dict(dictionary, orient="index")
                df["Date"] = pd.to_datetime(self.date, format="%Y%m%d")
                pkl_filename = ticker_name + ".pkl"
                df.to_pickle(os.path.join(os.path.join(whole_path,"Adjusted"), pkl_filename))
        return

    def produce_plots(self,ticker_name):
        # need to implement Q 1). c. i.
        pass


if __name__ == "__main__":
    relative_path = "./Data"
    trade_path = os.path.join(relative_path, "Trades")
    quote_path = os.path.join(relative_path, "Quotes")
    trade_dir = os.listdir(trade_path)
    quote_dir = os.listdir(quote_path)
    factor_df = pd.read_csv("S&P500_factors.csv")
    splitting_df = pd.read_csv("splitting_info.csv")

    for trade in trade_dir:
        if not trade.startswith("."):
            date = trade
            TAQAdjust_obj = TAQAdjust(date, factor_df, splitting_df)
            TAQAdjust_obj.retrieve_trade()

    for quote in quote_dir:
        if not quote.startswith("."):
            date = quote
            TAQAdjust_obj = TAQAdjust(date, factor_df,splitting_df)
            TAQAdjust_obj.retrieve_quote()

    concat_date()

