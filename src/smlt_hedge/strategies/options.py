"""Option mechanics (Black-76 on the futures underlying, European exercise).

MOEX SMLT options are written on the SMLT futures contract. Implied volatilities used in this
project are *backed out of exchange settlement prices* on the decision date (the exchange's own
volatility curve) -- they are never invented. Where no listed option exists (6-12m tenors),
any IV used is a labelled SCENARIO ASSUMPTION, reported as such in outputs.

Separation of payoffs:
- expiry payoff: max(K - S_T, 0) etc. (terminal, used by the scenario engine);
- pre-expiry mark-to-market: Black-76 value with remaining tau and a vol assumption.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def black76(F: float, K: float, tau: float, sigma: float, r: float, kind: str) -> float:
    if tau <= 0 or sigma <= 0:
        intrinsic = max(F - K, 0.0) if kind == "C" else max(K - F, 0.0)
        return intrinsic * math.exp(-r * max(tau, 0.0))
    sd = sigma * math.sqrt(tau)
    d1 = (math.log(F / K) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    df = math.exp(-r * tau)
    if kind == "C":
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def greeks(F: float, K: float, tau: float, sigma: float, r: float, kind: str) -> dict:
    """Black-76 greeks w.r.t. the futures price. Theta per calendar day, vega/rho per 1.00 (not %)."""
    sd = sigma * math.sqrt(tau)
    d1 = (math.log(F / K) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    df = math.exp(-r * tau)
    price = black76(F, K, tau, sigma, r, kind)
    delta = df * norm.cdf(d1) if kind == "C" else -df * norm.cdf(-d1)
    gamma = df * norm.pdf(d1) / (F * sd)
    vega = df * F * norm.pdf(d1) * math.sqrt(tau)
    theta_y = -df * F * norm.pdf(d1) * sigma / (2 * math.sqrt(tau)) + r * price
    rho = -tau * price
    return {"price": price, "delta": delta, "gamma": gamma, "vega": vega, "theta_per_day": theta_y / 365.0,
            "rho": rho}


def implied_vol(price: float, F: float, K: float, tau: float, r: float, kind: str,
                lo: float = 1e-4, hi: float = 10.0) -> float:
    intrinsic = (max(F - K, 0.0) if kind == "C" else max(K - F, 0.0)) * math.exp(-r * tau)
    if price <= intrinsic + 1e-10 or tau <= 0:
        return float("nan")
    f = lambda s: black76(F, K, tau, s, r, kind) - price  # noqa: E731
    if f(lo) * f(hi) > 0:
        return float("nan")
    return float(brentq(f, lo, hi, xtol=1e-10))


# ------------------------------------------------------------------ structure payoffs (per share)
def put_payoff(S_T: np.ndarray, K: float) -> np.ndarray:
    return np.maximum(K - S_T, 0.0)


def put_spread_payoff(S_T: np.ndarray, K_hi: float, K_lo: float) -> np.ndarray:
    return np.maximum(K_hi - S_T, 0.0) - np.maximum(K_lo - S_T, 0.0)


def collar_payoff(S_T: np.ndarray, K_put: float, K_call: float) -> np.ndarray:
    """Long put K_put + short call K_call (overlay payoff, excluding the stock)."""
    return np.maximum(K_put - S_T, 0.0) - np.maximum(S_T - K_call, 0.0)
