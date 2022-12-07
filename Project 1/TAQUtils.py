import numpy as np
import pandas as pd
def get_period_return(df, time_minute, type_):
    if time_minute >= 1:
        # if input time_minute is larger or equal to 1, we consider it as a minute_freq
        freq = f"{int(time_minute)}T"
    else:
        # if input time_minute is less than 1, we multiply it by 60 seconds/minute to get second_freq
        freq = f"{int(time_minute*60)}S"
    if type_ == "trade":
        r = df.resample(freq,closed="left", label="right", on="Datetime")
        first = r.agg("first")
        last = r.agg("last")
        ret_df = last[["Adjusted_price"]] / first[["Adjusted_price"]] - 1
        ret_df.rename(columns={"Adjusted_price": freq + "_ret"}, inplace=True)
    elif type_ == "quote":
        df["Adjusted_mid_price"] = (df["Adjusted_bid_price"] + df["Adjusted_ask_price"])/2
        r = df.resample(freq, closed="left", label="right", on="Datetime")
        first = r.agg("first")
        last = r.agg("last")
        ret_df = last[["Adjusted_mid_price"]] / first[["Adjusted_mid_price"]] - 1
        ret_df.rename(columns={"Adjusted_mid_price": freq + "_ret"}, inplace=True)
    ret_df = ret_df.dropna(axis=0)
    return ret_df


def convert_ret_df(df):
    def applys(x):
        return x.set_index("Ticker")["Returns"]
    new_df = df.groupby(df["Date"]).apply(applys)
    new_df = new_df[~new_df.index.duplicated(keep='first')]
    # some stock does not have returns over the whole period for example: COV
    # a C meaning created will be shown in the dataframe
    new_df = new_df.replace("C",np.nan)
    # replace the str type to float
    new_df = new_df.astype("float64")
    return new_df.unstack("Ticker")

def convert_turnover_df(df):
    def applys(x):
        return x.set_index("Ticker")["Turnover"]
    new_df = df.groupby(df["Date"]).apply(applys)
    new_df = new_df[~new_df.index.duplicated(keep='first')]
    # some stock does not have returns over the whole period for example: COV
    # a C meaning created will be shown in the dataframe
    new_df = new_df.replace("C",np.nan)
    # replace the str type to float
    new_df = new_df.astype("float64")
    return new_df.unstack("Ticker")