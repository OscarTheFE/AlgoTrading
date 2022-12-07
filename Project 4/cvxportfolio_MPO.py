import os
import sys
sys.path.insert(0, os.path.abspath('..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxportfolio as cp

plotdir = '../portfolio/plots/'
datadir='../data/'


# we have the same setting as our Single Period Optimization
sigmas=pd.read_csv(datadir+'sigmas.csv.gz',index_col=0,parse_dates=[0]).iloc[:,:-1]
returns=pd.read_csv(datadir+'returns.csv.gz',index_col=0,parse_dates=[0])
volumes=pd.read_csv(datadir+'volumes.csv.gz',index_col=0,parse_dates=[0]).iloc[:,:-1]

w_b = pd.Series(index=returns.columns, data=1)
w_b.USDOLLAR = 0.
w_b/=sum(w_b)

start_t="2012-01-01"
end_t="2016-12-29"

simulated_tcost = cp.TcostModel(half_spread=0.0005/2., nonlin_coeff=1., sigma=sigmas, volume=volumes)
simulated_hcost = cp.HcostModel(borrow_costs=0.0001)
simulator = cp.MarketSimulator(returns, costs=[simulated_tcost, simulated_hcost],
                               market_volumes=volumes, cash_key='USDOLLAR')

return_estimate=pd.read_csv(datadir+'return_estimate.csv.gz',index_col=0,parse_dates=[0]).dropna()
volume_estimate=pd.read_csv(datadir+'volume_estimate.csv.gz',index_col=0,parse_dates=[0]).dropna()
sigma_estimate=pd.read_csv(datadir+'sigma_estimate.csv.gz',index_col=0,parse_dates=[0]).dropna()

optimization_tcost = cp.TcostModel(half_spread=0.0005/2., nonlin_coeff=1.,
                                sigma=sigma_estimate, volume=volume_estimate)
optimization_hcost=cp.HcostModel(borrow_costs=0.0001)

risk_data = pd.HDFStore(datadir+'risk_model.h5')
risk_model = cp.FactorModelSigma(risk_data.exposures, risk_data.factor_sigma, risk_data.idyos)

all_return_estimates = {}
n_p1 = returns.shape[1]
T = returns.shape[0]

# in our setting of MPO, instead of using current return estimate as we did in SPO, we want an array of return estimates
# of time tau starting from current time t towards to the end of holding period

for i, t in enumerate(returns.index[:-1]):
    all_return_estimates[(t, t)] = return_estimate.loc[t]
    tp1 = returns.index[i + 1]
    all_return_estimates[(t, tp1)] = return_estimate.loc[tp1]

# the MPO return forecast are simply modify our common return estimate by holding
# create a forecast object
returns_forecast = cp.MPOReturnsForecast(all_return_estimates)
results_MPO = {}

import cvxpy as cvx
policies={}
gamma_risks_coarse=[.1,.3,1,3,10,30,100,300,1000]
gamma_tcosts_coarse=[1,2,5,10,20]
# we are essentially doing the same thing as we have done in SPO
# what slightly changed is here we need a few more input parameters to our Optimizer to generate our policy
# we need to pass in a return forecast and lookahead_period, which can be found in boyd paper
for gamma_risk in gamma_risks_coarse:
    for gamma_tcost in gamma_tcosts_coarse:
        policies[(gamma_risk, gamma_tcost)] = \
      cp.MultiPeriodOpt(return_forecast=returns_forecast,
                          costs=[gamma_risk*risk_model, gamma_tcost*optimization_tcost, optimization_hcost],
                          constraints=[cp.LeverageLimit(3)],
                          trading_times=list(returns.index[(returns.index>=start_t)&(returns.index<=end_t)]),
                         lookahead_periods=2,
                         terminal_weights=None)

results_MPO.update({k:v for k,v in zip(policies.keys(),
                           simulator.run_multiple_backtest(w_b*1e8, start_time = start_t, end_time=end_t,
                                          policies=policies.values(),parallel=True))})
