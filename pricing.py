"""Option pricing: Black-Scholes benchmark and Monte Carlo estimators.

Model: geometric Brownian motion (GBM) under the risk-neutral measure,
    S_T = S_0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z),  Z ~ N(0, 1).
"""

import numpy as np
from scipy.stats import norm

def bs_call(S0, K, r, sigma, T):
  """Black-Scholes price of European call (closed form, used as our benchmark)."""
  d1 = (np.log(S) / K) + (r + 0.5 *sigma**2) * T) / (sigma * np.sqrt(T))
  d2 = d1 - sigma * np.sqrt(T)
  return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_delta(S0, K, r, sigma, T):
  """Black-Scholes delta of a European Call"""
  d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
  return norm.cdf(d1)

def _terminal_price(S0, r, sigma, T, Z):
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

def mc_call(S0, K, r, sigma, T, n, rng):
    """Plain Monte Carlo call price. Returns (estimate, standard_error)."""
    Z = rng.standard_normal(n)
    payoff = np.exp(-r * T) * np.maximum(_terminal_price(S0, r, sigma, T, Z) - K, 0.0)
    return payoff.mean(), payoff.std(ddof=1) / np.sqrt(n)

def mc_call_antithetic(S0, K, r, sigma, T, n, rng):
    """Monte Carlo with antithetic variates, using n/2 pairs (Z, -Z).

    Each pair is averaged first; the standard error is computed from the
    n/2 pair-averages, which are independent. Total paths = n.
    """
    half = n // 2
    Z = rng.standard_normal(half)
    p_plus = np.maximum(_terminal_price(S0, r, sigma, T, Z) - K, 0.0)
    p_minus = np.maximum(_terminal_price(S0, r, sigma, T, -Z) - K, 0.0)
    pair_avg = np.exp(-r * T) * 0.5 * (p_plus + p_minus)
    return pair_avg.mean(), pair_avg.std(ddof=1) / np.sqrt(half)
