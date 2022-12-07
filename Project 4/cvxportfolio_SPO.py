import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxportfolio_SPO as cp
import warnings
warnings.filterwarnings('ignore')

plotdir='../../portfolio/plots/'
datadir='../data/'

# load variables
# sigmas are defined by daily volatility, which is calculated by daily abs difference between open price and close price
# note that we need to use sigma to simulate our transaction cost, which will be later implemented in our trading policy
sigmas=pd.read_csv(datadir+'sigmas.csv.gz',index_col=0,parse_dates=[0]).iloc[:,:-1]
# returns are just common returns but filter out the bad stocks and bad dates
returns=pd.read_csv(datadir+'returns.csv.gz',index_col=0,parse_dates=[0])
# similar to sigmas, volumes are also variable when we simulate transaction cost
volumes=pd.read_csv(datadir+'volumes.csv.gz',index_col=0,parse_dates=[0]).iloc[:,:-1]

# initialize weights
w_b = pd.Series(index=returns.columns, data=1)
w_b.USDOLLAR = 0.  # add risk free cash account
w_b/=sum(w_b)   # normalize the weight and sum to 1

start_t="2012-01-01"
end_t="2016-12-31"

# simulate transaction cost
simulated_tcost = cp.TcostModel(half_spread=0.0005/2., nonlin_coeff=1., sigma=sigmas, volume=volumes)
# simulate holding cost
simulated_hcost = cp.HcostModel(borrow_costs=0.0001)
# initialize a market simulator object and pass in returns, simulated costs and volumes
simulator = cp.MarketSimulator(returns, costs=[simulated_tcost, simulated_hcost],
                               market_volumes=volumes, cash_key='USDOLLAR')

# since we still need estimated value for our portfolio estimation
return_estimate=pd.read_csv(datadir+'return_estimate.csv.gz',index_col=0,parse_dates=[0]).dropna()
volume_estimate=pd.read_csv(datadir+'volume_estimate.csv.gz',index_col=0,parse_dates=[0]).dropna()
sigma_estimate=pd.read_csv(datadir+'sigma_estimate.csv.gz',index_col=0,parse_dates=[0]).dropna()

# fit the transaction cost and holding cost to our estimated matrix
optimization_tcost = cp.TcostModel(half_spread=0.0005/2., nonlin_coeff=1.,
                                sigma=sigma_estimate, volume=volume_estimate)
optimization_hcost = cp.HcostModel(borrow_costs=0.0001)

# download pretrained risk factor model, so that we can represent our variance matrix by multi-factor sigmas
risk_data = pd.HDFStore(datadir+'risk_model.h5')
# the risk data includes 15 most significant eigenvectors and eigenvalues
# where exposures represent eigenvectors
# factor sigma represents eigenvalues
# and idyo represent idiosyncratic risks c computed by rest eigenvectors
risk_model = cp.FactorModelSigma(risk_data.exposures, risk_data.factor_sigma, risk_data.idyos)

results={}

################################################################################################################
# Single Period Optimization coarse search #
################################################################################################################

policies={}

# we try to find optimal parameters by building a combination of different gamma parameters and a leverage limit of 3
# this is the risk aversion parameter
gamma_risks_coarse=[.1,.3,1,3,10,30,100,300,1000]
# this is transaction cost parameter
gamma_tcosts_coarse=[1,2,5,10,20]
for gamma_risk in gamma_risks_coarse:
    for gamma_tcost in gamma_tcosts_coarse :
        policies[(gamma_risk, gamma_tcost)] = \
      cp.SinglePeriodOpt(return_estimate, [gamma_risk*risk_model,gamma_tcost*optimization_tcost,optimization_hcost],
                                [cp.LeverageLimit(3)])


# update the result by using different setup constructed by different combination of parameters and backtest
results.update(dict(zip(policies.keys(), simulator.run_multiple_backtest(1E8*w_b, start_time=start_t,end_time=end_t,
                                              policies=policies.values(), parallel=True))))



