import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_csv("cointData190522.csv", header=None, names=np.arange(200))

# since we know that the price of an asset is normally non-stationary, It often has a unit root
# further means the price of an asset itself is I(1) process
# we can take pct_change/ first difference to solve the spurious regression
# however we lose information by taking first difference
# Also, not the same information is contained in return data and price data
# df_return = df.pct_change().dropna()
# df_return = (np.log(df)-np.log(df).shift(1)).dropna()


# So, we need another regression method between I(1) processes
# we can use two assets' prices as a price pair if we can conclude those two assets are co-integrated
df_price = df
# or we can further standardize our result by using cumulative returns instead of df_price
df_cum_ret = (df/df.iloc[0,:]).iloc[1:,:] - 1

# 3 4 20 23 37
def find_ci_pairs(df, threshold=0.05):
    store_lst = []
    for first in range(df.shape[1]):
        for second in range(first + 1, df.shape[1]):
            y_series = df[first]
            x_series = df[second]

            coint, p_value, crit_values = sm.tsa.stattools.coint(y_series, x_series)
            print(second)
            print(p_value)
            if p_value < threshold:
                store_lst.append([first,second])
    return store_lst


if __name__ == "__main__":
    s = find_ci_pairs(df_cum_ret, 0.01)

    np.savetxt("pairs.csv",
               s,
               delimiter=", ",
               fmt='% s')


