import TAQUtils
import TAQCleaner
import os
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller


class TAQAutoCorr_analysis():
    def __init__(self, ticker_name, use_cleaned=False):
        self.ticker_name = ticker_name
        self.trade_dir = os.path.join("./Data/Trades/Daily_trade", ticker_name + ".parquet.gzip")
        self.uncleaned_trade_df = pd.read_parquet(self.trade_dir)
        self.data_exist = True
        if self.uncleaned_trade_df.shape[0] == 0:
            self.data_exist = False
        self.use_cleaned = use_cleaned
        if self.use_cleaned:
            self.TAQCleaner_obj = TAQCleaner.TAQCleaner(self.ticker_name)
            self.TAQCleaner_obj.clean_trades()

        self.freq_list = [1 / 6, 3 / 6, 1, 5, 10, 20, 30]


    def Ljung_Box_test(self, k):
        if not self.use_cleaned:
            df = self.uncleaned_trade_df
        else:
            df = self.TAQCleaner_obj.get_trade_df()
        # First we use 10s as our experimental time interval
        # We want to run Ljung_box test on the return df to see the lag of auto-correlation
        for freq in self.freq_list:
            ret_df = TAQUtils.get_period_return(df, freq, "trade")
            result_df = acorr_ljungbox(ret_df, lags=[k], return_df=True)
            p_value = result_df["lb_pvalue"].iloc[0]
            if p_value > 0.05:
                # not reject null hypothesis and conclude that the series itself is stationary
                return freq
        return self.freq_list[-1]

    def Dickey_Fuller_test(self, threshold=0.05):
        if not self.use_cleaned:
            df = self.uncleaned_trade_df
        else:
            df = self.TAQCleaner_obj.get_trade_df()
        # First we use 10s as our experimental time interval
        # We want to run Ljung_box test on the return df to see the lag of auto-correlation
        not_pass = 0
        for freq in self.freq_list:
            ret_df = TAQUtils.get_period_return(df, freq, "trade")
            result = adfuller(ret_df.values, autolag='AIC')
            # If the p-value is very less than the significance level of 0.05
            # hence we can reject the null hypothesis
            # and take that the series is stationary. Let’s visualise the series as well to confirm.
            adf_statistic = result[0]
            p_value = result[1]
            if p_value < 0.05:
                return freq
        return self.freq_list[-1]


if __name__ == "__main__":
    TAQAutoCorr_analysis_obj = TAQAutoCorr_analysis("JAVA")
    not_pass_1 = TAQAutoCorr_analysis_obj.Ljung_Box_test(1)
    not_pass_3 = TAQAutoCorr_analysis_obj.Ljung_Box_test(3)
    print(not_pass_1)
    print(not_pass_3)

    adf = TAQAutoCorr_analysis_obj.Dickey_Fuller_test("5%")
    print(adf)
