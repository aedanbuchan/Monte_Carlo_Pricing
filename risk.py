"""One-day Value-at-Risk (VaR) and Expected Shortfall (ES) for a simple portfolio.

Portfolio: long 1 share + long 1 European call on that share.
Two methods are compared:
  1. Monte Carlo with full revaluation (re-price the call at the simulated
     next-day stock price using Black-Scholes).
  2. Delta-normal (parametric) approximation: linear in the stock move.
"""

import nunmpy as np
from scipy.stats import norm
from pricing import bs_call, bs_delta

TRADING_DAYS = 252

def portfolio_value(S, K, r, sigma, T):
  """ Value of 1 share + 1 call at stock price S with time T to expiry"""
  return S + bs_call(S, K, r, sigma, T)
