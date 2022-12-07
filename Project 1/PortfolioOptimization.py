from math import sqrt
from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp, options
import TAQUtils
import pandas as pd
import numpy as np
"""
the problem setting here is to use cvxopt quadratic problem solver to solve quadratic objective function

our objective function is -transpose(pbar)*X + mu*transpose(X)*S*X
in the world of portfolio management/ portfolio optimization,
such quadratic objective function resembles g(X) = -Expected portfolio return(X) + mu*portfolio vol(X)
                                                where X stands for holding positions
                                                     vol stands for the covariance matrix of holding portfolio
the restrictions are also expressed through G,h,A,b
"""
class PortfolioOptimizer():

    def __init__(self, ret, cov, aversion_param,tickers):
        self.ret = ret
        self.cov = cov
        self.n = len(ret)
        self.aversion_param = aversion_param
        self.tickers = tickers

    def compute_holdings(self):
        # it is like restricting each of our portfolio holding is bigger than 0, only long position, no short position
        G = matrix(0.0, (self.n, self.n))
        G[::self.n + 1] = -1.0
        h = matrix(0.0, (self.n, 1))
        # It is like restricting the sum of our portfolio holdings to be 1, meaning no shortage or leverage
        A = matrix(1.0, (1, self.n))  # Identity matrix
        b = matrix(1.0)  # 1


        # risk aversion parameter denoting that how people don't like risk
        # the higher the aversion parameter, the more risk aversion that person is
        options['show_progress'] = False
        # since we want to minimize the objective function, which means we want to maximize negative objective function
        # which is essentially saying that we want to maximize return - risk.
        x = qp(self.aversion_param * self.cov, -self.ret, G, h, A, b, solver="chol")['x']  # get our optimal holdings for different mu values

        # compute return and risk for each corresponding x
        _return = dot(self.ret, x)
        _risk = sqrt(dot(x, self.cov * x))

        return x,_return,_risk

    def get_statistics(self):
        print("the risk-aversion parameter is {}".format(self.aversion_param))
        x, _return, _risk = self.compute_holdings()
        print("The 5-largest portion ticker in our portfolio:")
        print(self.tickers[pd.Series(x).nlargest(5).index])
        print("Corresponding holding percentage:")
        print(pd.Series(x).nlargest(5))
        print("Total portfolio optimized return: {}".format(_return))
        print("Total optimized portfolio risk: {}".format(_risk))
        turnovers = self.get_turnover()
        print("Total portfolio turnover is {}".format(np.dot(turnovers.values.reshape(1,-1),x)[0]))
        print()

    def get_turnover(self):
        df = pd.read_csv("S&P500_rets.csv")
        turnover_df = TAQUtils.convert_turnover_df(df)
        turnover_df.drop(["JAVA", "SUNW"], axis=1, inplace=True)  # delete the abnormal stock
        turnovers = turnover_df.mean()  # take daily mean turnover
        #turnovers = turnovers * 252
        turnovers = turnovers.append(pd.Series({"risk-free":0}),ignore_index=False)
        return turnovers





if __name__ == "__main__":
    df = pd.read_csv("S&P500_rets.csv")

    turnover_df = TAQUtils.convert_turnover_df(df)
    turnover_df.drop(["JAVA", "SUNW"], axis=1, inplace=True)
    print(turnover_df.mean())

    df = TAQUtils.convert_ret_df(df)
    # Since JAVA renamed itself, It used to be called SUMW. This will cause the cov matrix of return to explode.
    # so we simply discard it from our dataset
    df.drop(["JAVA","SUNW"],axis=1, inplace=True)

    n = df.shape[1]

    # However since this is only the sp500 tickers
    # We will also want to add a risk-free asset which has mean return 2% and zero covariance with other securities

    cov = df.cov().values * sqrt(252)   # Annualized the cov matrix
    cov = np.insert(cov, n, np.zeros(n,), axis=0)
    cov = np.insert(cov, n, np.zeros(n+1,), axis=1)
    cov = matrix(cov)

    # It is like the expected return of underlying tickers in our portfolio
    ret = df.mean().values * 252    # Annualized the ret matrix
    ret = np.append(ret, 0.02)
    ret = matrix(ret)

    N = 100
    # mu is like the risk aversion parameter denoting that how people don't like risk
    # the higher the mu, the more risk aversion that person is
    mus = [10 ** (5.0 * t / N - 1.0) for t in range(70)]
    tickers = df.columns
    for mu in mus:
        PortfolioOptimizer_obj = PortfolioOptimizer(ret, cov, mu,tickers)
        PortfolioOptimizer_obj.get_statistics()
