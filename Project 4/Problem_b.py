from Circular_queue import CircularPairQueue
import numpy as np
import pandas as pd


class Cointegration:
    def __init__(self, df, pairs, window_size):
        self.window_size = window_size
        self.df = df
        self.pair_set1 = pairs[pairs.columns[0]].values
        self.pair_set2 = pairs[pairs.columns[1]].values

    def cointegrate(self, s1, s2, window_size):
        if len(s1) != len(s2):
            print("wrong shape")
            return
        gammas = np.zeros(window_size).tolist()
        x_means = np.zeros(window_size).tolist()
        xx_means = np.zeros(window_size).tolist()
        xlag_means = np.zeros(window_size).tolist()
        y_means = np.zeros(window_size).tolist()
        ylag_means = np.zeros(window_size).tolist()
        yylag_means = np.zeros(window_size).tolist()
        xy_means = np.zeros(window_size).tolist()
        xylag_means = np.zeros(window_size).tolist()
        yxlag_means = np.zeros(window_size).tolist()

        cpq = CircularPairQueue(self.window_size)
        # first fill up our x, y circular queue with first T - 1 elements, where T is the window size
        for i in range(self.window_size - 1):
            cpq.enqueue(s1.iloc[i], s2.iloc[i])

        for j in range(self.window_size, len(s1)):
            item_1 = s1.iloc[j]
            item_2 = s2.iloc[j]
            # we update our circular queue and our corresponding mean values
            cpq.enqueue(item_1, item_2)
            # instead of using regression module to compute our regression parameter
            # we want to leverage y = beta * x + b
            # beta = cov(y,x)/var(x) = (E(xy) - E(x)E(y)) / (E(x^2) - E(x)^2)
            # b = E(y) - beta * E(x)

            beta = (cpq.xy_mean - cpq.x_mean * cpq.y_mean) / (cpq.xx_mean - cpq.x_mean * cpq.x_mean)
            b = cpq.y_mean - beta * cpq.x_mean

            # now we have already performed a regression between our original series, in this case prices of paired assets
            # we now need to run an adf test on regression residuals and do a unit test
            # we are essentially running another regression between residual and residual lag 1
            # this is comparable to the result if we run regression between residual difference and residual lag 1
            # one should have null hypothesis gamma = 1 and the other should have nul hypothesis gamma = 1

            # fortunately, we can still leverage our tricks
            s_mean = cpq.y_mean - beta * cpq.x_mean - b
            slag_mean = cpq.ylag_mean - beta * cpq.xlag_mean - b
            slagslag_mean = cpq.ylagylag_mean - 2 * beta * cpq.ylagxlag_mean - 2 * cpq.ylag_mean * b + (
                    beta * beta * cpq.xlagxlag_mean + 2 * b * beta * cpq.xlag_mean + b * b)

            sslag_mean = cpq.yylag_mean - b * cpq.ylag_mean - beta * cpq.xylag_mean - b * cpq.y_mean - beta * cpq.yxlag_mean + \
                         beta * beta * cpq.xxlag_mean + b * beta * cpq.x_mean + b * beta * cpq.xlag_mean + b * b

            gamma = (sslag_mean - s_mean * slag_mean) / (slagslag_mean - slag_mean * slag_mean)

            # Put mean_x, mea_x_xlag, sum_x_ylag, sum_xx, sum_xy,sum_y, sum_y_xlag, sum_y_ylag, sum_yy

            gammas.append(gamma)
            x_means.append(cpq.x_mean)
            xx_means.append(cpq.xx_mean)
            xlag_means.append(cpq.xlag_mean)
            y_means.append(cpq.y_mean)
            ylag_means.append(cpq.ylag_mean)
            yylag_means.append(cpq.yylag_mean)
            xy_means.append(cpq.xy_mean)
            xylag_means.append(cpq.xylag_mean)
            yxlag_means.append(cpq.yxlag_mean)

            # print(gamma_data)

        gammas = pd.Series(gammas)
        x_means = pd.Series(x_means)
        xx_means = pd.Series(xx_means)
        xlag_means = pd.Series(xlag_means)
        y_means = pd.Series(y_means)
        ylag_means = pd.Series(ylag_means)
        yylag_means = pd.Series(yylag_means)
        xy_means = pd.Series(xy_means)
        xylag_means = pd.Series(xylag_means)
        yxlag_means = pd.Series(yxlag_means)

        coint_df = pd.concat(
            [s1, s2, x_means, xx_means, xlag_means, y_means, ylag_means, yylag_means, xy_means, xylag_means,
             yxlag_means, gammas], axis=1)
        coint_df.columns = ['s1', 's2', 'x_mean', 'xx_mean', 'xlag_mean', 'y_mean', 'ylag_mean', 'yylag_mean',
                            'xy_mean', 'xylag_mean', 'yxlag_mean', 'gamma']

        return coint_df.reset_index(drop=True)

    def rolling(self):
        matrix = []
        for index in range(len(self.pair_set1)):
            series_1 = self.df[self.pair_set1[index]]
            series_2 = self.df[self.pair_set2[index]]
            matrix_form = self.cointegrate(series_1, series_2, self.window_size)
            print(index)
            matrix.append(matrix_form)
        return matrix


if __name__ == "__main__":
    pairs = pd.read_csv("pairs.csv")
    df = pd.read_csv("cointData190522.csv", header=None, names=np.arange(200))
    df_cum_ret = (df / df.iloc[0, :]).iloc[1:, :] - 1
    Coin = Cointegration(df_cum_ret, pairs, 100)
    ret_matrix = Coin.rolling()
