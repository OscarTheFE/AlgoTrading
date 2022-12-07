import pandas as pd
import pandas as pd
import numpy as np
from sklearn.covariance import EmpiricalCovariance
from scipy.optimize import minimize
import warnings
import pyRMT

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None


class TAQCovariance(object):
    def __init__(self):
        self.return_matrix = pd.read_csv("5_mins_mid_quote_ret.csv")
        self.return_matrix["Datetime"] = pd.to_datetime(self.return_matrix.Datetime)

        self.dates = self.return_matrix["Datetime"].apply(lambda x: x.strftime("%Y%m%d")).unique()
        self.return_matrix.set_index("Datetime", inplace=True)

        self.pre_processing()

    def pre_processing(self):

        # subtract_mean
        self.return_matrix -= self.return_matrix.mean()
        # cross-sectional daily volatility
        cross_sectional_vol = (self.return_matrix ** 2).sum(axis=1)
        cross_sectional_vol = np.sqrt(cross_sectional_vol)

        def pre_process(x):
            # normalize by cross_sectional vol
            x = x / cross_sectional_vol
            return x

        self.return_matrix = self.return_matrix.apply(lambda x: pre_process(x))
        return

    def normalize_train(self, start_date, end_date):
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        sub_df = self.return_matrix.copy()
        sub_df = sub_df.loc[(sub_df.index >= start_date) & (sub_df.index < end_date)]

        cross_time_vol = []

        def normalize(x):
            # normalize by cross_time vol
            std = x.std()
            cross_time_vol.append(std)
            x = x / std
            return x

        sub_df = sub_df.apply(lambda x: normalize(x))
        return sub_df, np.array(cross_time_vol)

    # Compute strategy weights
    def compute_weights(self, return_matrix, cov_matrix, index, weight_method):
        num = cov_matrix.shape[0]
        cov_matrix_inv = pd.DataFrame(np.linalg.pinv(cov_matrix.values), cov_matrix.columns, cov_matrix.index)
        if weight_method == 'historical':
            g_list = np.array([1] * num).reshape(-1, 1)

            w = cov_matrix_inv.dot(g_list) / np.dot(g_list.T, np.dot(cov_matrix_inv, g_list))
            return w/w.sum()

        if weight_method == 'omniscient':
            g_list = np.sqrt(num) * return_matrix.iloc[0]
            g_list = g_list.values.reshape(-1, 1)

            w = cov_matrix_inv.dot(g_list) / np.dot(g_list.T, np.dot(cov_matrix_inv, g_list))
            return w/w.sum()

        if weight_method == "mean_reverting":
            test_start_date = pd.to_datetime(self.dates[index])
            train_end_date = pd.to_datetime(self.dates[index - 1])
            prev_day_return = self.return_matrix.loc[
                (self.return_matrix.index >= train_end_date) & (self.return_matrix.index < test_start_date)]
            g_list = np.sqrt(num) * prev_day_return.iloc[-1]
            g_list = g_list.values.reshape(-1, 1)

            w = cov_matrix_inv.dot(g_list) / np.dot(g_list.T, np.dot(cov_matrix_inv, g_list))
            return w/w.sum()

        if weight_method == 'random_long_short':
            g_list = np.random.uniform(0, 1, num)
            g_list = g_list.reshape(-1, 1)

            w = cov_matrix_inv.dot(g_list) / np.dot(g_list.T, np.dot(cov_matrix_inv, g_list))
            return w/w.sum()

    def compute_return_vol(self, w, return_matrix):
        w = w.values
        total_return = np.dot(return_matrix, w)
        total_return_square = total_return ** 2
        R_square = total_return_square.sum() / return_matrix.shape[0]
        return R_square

    def compute_biastats(self, w, return_matrix):
        w = w.values
        total_return = np.dot(return_matrix, w)
        total_return_square = total_return ** 2
        R_square = total_return_square.sum() / return_matrix.shape[0]
        std_var = np.sqrt(R_square)
        bias_stats = (total_return / std_var).std()
        return bias_stats

    def rolling_backtest(self, train_win=7, test_win=2, shrinkage_method='Empirical', weight_method="historical"):

        R_square_list = []
        Bias_stats_list = []

        if shrinkage_method == "EWMA":
            alpha = input("Decide the exponential decay parameter: ")

        for i in range(len(self.dates) - (train_win + test_win)):

            train_start_date = self.dates[i]
            train_end_date = self.dates[i + train_win]

            test_start_date = self.dates[i + train_win]
            test_end_date = self.dates[i + train_win + test_win]

            train_df, cross_time_vol = self.normalize_train(train_start_date, train_end_date)
            test_return = self.return_matrix.loc[
                (self.return_matrix.index >= test_start_date) & (self.return_matrix.index < test_end_date)]

            if shrinkage_method == 'Empirical':
                train_cov = EmpiricalCovariance().fit(train_df)
                train_cov = pd.DataFrame(train_cov.covariance_, index=train_df.columns, columns=train_df.columns)

            if shrinkage_method == 'Clipped':
                train_cov = pyRMT.clipped(train_df, return_covariance=True)
                train_cov = pd.DataFrame(train_cov, index=train_df.columns, columns=train_df.columns)

            if shrinkage_method == 'Optimal':
                train_cov = pyRMT.optimalShrinkage(train_df, return_covariance=True)
                train_cov = pd.DataFrame(train_cov, index=train_df.columns, columns=train_df.columns)

            if shrinkage_method == "EWMA":

                def ewma(df, alpha=0.94):
                    alpha = float(alpha) # convert user input into float
                    weights = (alpha) ** np.arange(len(df))[::-1]
                    out = (1 - alpha) * ((weights * df.T) @ df) / (1 - alpha ** len(df))
                    return out

                train_cov = ewma(train_df, alpha)
                train_cov = pd.DataFrame(train_cov, index=train_df.columns, columns=train_df.columns)


            # restoring volatility to our correlation matrix
            cross_time_vol_matrix = cross_time_vol.reshape(-1, 1) * cross_time_vol
            train_cov = train_cov * cross_time_vol_matrix

            weights = self.compute_weights(test_return, train_cov, i + train_win, weight_method)
            R_square = self.compute_return_vol(weights, test_return)
            R_square *= np.sqrt(78 * 252) # convert to daily result
            Bias_stats = self.compute_biastats(weights, test_return)
            R_square_list.append(R_square)
            Bias_stats_list.append(Bias_stats)

        return R_square_list, Bias_stats_list

    def generate_R_square_table(self, train_win, test_win):
        shrinkage_methods = ['Empirical', "Clipped", "Optimal", "EWMA"]
        weight_methods = ["historical", 'omniscient', "mean_reverting", 'random_long_short']
        table_dic_R_squared = {}
        table_dic_Bias = {}
        for shrinkage_method in shrinkage_methods:
            temp_dic_R_squared = {}
            temp_dic_Bias = {}
            for weight_method in weight_methods:
                R_square_list, Bias_stats_list = self.rolling_backtest(train_win, test_win, shrinkage_method,
                                                                       weight_method)
                mean_R_square = np.mean(R_square_list)
                mean_Bias = np.mean(Bias_stats_list)

                temp_dic_R_squared[weight_method] = mean_R_square
                temp_dic_Bias[weight_method] = mean_Bias

            table_dic_R_squared[shrinkage_method] = temp_dic_R_squared
            table_dic_Bias[shrinkage_method] = temp_dic_Bias
        return pd.DataFrame(table_dic_R_squared), pd.DataFrame(table_dic_Bias)

if __name__ == "__main__":
    TAQCovariance_obj = TAQCovariance()
    r_squared, bias = TAQCovariance_obj.generate_R_square_table(10, 2)
    #bias.to_csv("bias.csv")
    print(r_squared)
    print(bias)








