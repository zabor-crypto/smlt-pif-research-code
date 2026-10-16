"""Covariance estimators and their out-of-sample comparison.

Estimators: sample (ddof=1), Ledoit-Wolf (shrink to scaled identity, sklearn),
OAS (sklearn), EWMA (RiskMetrics, lambda from config, zero-mean, initialised with the
sample covariance of the first ``init`` observations).

Out-of-sample criterion (walk-forward, no look-ahead): for each rebalance date t,
fit on the trailing window ending t-1, form the global min-variance *hedge* for SMLT
(one-asset-fixed problem), and compare predicted vs realised variance of that hedged
portfolio over the next ``hold`` days. We report the mean QLIKE loss and the bias ratio
(realised/predicted). Lower QLIKE = better; bias ratio near 1 = well calibrated.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import OAS, LedoitWolf


def cov_sample(X: np.ndarray) -> np.ndarray:
    return np.cov(X, rowvar=False, ddof=1)


def cov_lw(X: np.ndarray) -> np.ndarray:
    return LedoitWolf(assume_centered=False).fit(X).covariance_ * X.shape[0] / (X.shape[0] - 1)


def cov_oas(X: np.ndarray) -> np.ndarray:
    return OAS(assume_centered=False).fit(X).covariance_ * X.shape[0] / (X.shape[0] - 1)


def cov_ewma(X: np.ndarray, lam: float = 0.94, init: int = 20) -> np.ndarray:
    X = np.asarray(X, float)
    init = min(init, X.shape[0])
    S = np.cov(X[:init], rowvar=False, ddof=1) if init > 1 else np.outer(X[0], X[0])
    for x in X[init:]:
        S = lam * S + (1.0 - lam) * np.outer(x, x)
    return S


ESTIMATORS = {"sample": cov_sample, "ledoit_wolf": cov_lw, "oas": cov_oas, "ewma": cov_ewma}


def min_var_hedge_weights(S: np.ndarray, fixed: int = 0) -> np.ndarray:
    """Hedge weights h (per unit of the fixed asset) minimising Var(r_fixed + h' r_H).

    Solution: h = -S_HH^{-1} S_Hs. Returns full weight vector with 1 at ``fixed``.
    """
    n = S.shape[0]
    idx = [i for i in range(n) if i != fixed]
    S_hh = S[np.ix_(idx, idx)]
    S_hs = S[idx, fixed]
    h = -np.linalg.solve(S_hh + 1e-12 * np.eye(len(idx)), S_hs)
    w = np.zeros(n)
    w[fixed] = 1.0
    w[idx] = h
    return w


def oos_compare(R: pd.DataFrame, window: int, hold: int = 5, fixed_col: str | None = None,
                lam: float = 0.94) -> pd.DataFrame:
    """Walk-forward comparison of estimators on the min-variance hedge of ``fixed_col``."""
    R = R.dropna()
    X = R.values
    fixed = 0 if fixed_col is None else list(R.columns).index(fixed_col)
    rows = []
    for t in range(window, len(R) - hold + 1, hold):
        train = X[t - window:t]
        test = X[t:t + hold]
        for name, f in ESTIMATORS.items():
            S = f(train, lam) if name == "ewma" else f(train)
            w = min_var_hedge_weights(S, fixed)
            pred = float(w @ S @ w)
            p = test @ w
            real = float(np.mean(p**2))  # zero-mean realised daily variance over holding period
            rows.append({"date": R.index[t], "estimator": name, "pred_var": pred, "real_var": real,
                         "qlike": real / pred - np.log(real / pred) - 1.0 if pred > 0 and real > 0 else np.nan,
                         "hedge_weights": w.copy()})
    out = pd.DataFrame(rows)
    return out


def oos_summary(res: pd.DataFrame) -> pd.DataFrame:
    g = res.groupby("estimator")
    return pd.DataFrame({
        "mean_qlike": g["qlike"].mean(),
        "bias_ratio_real_over_pred": g["real_var"].sum() / g["pred_var"].sum(),
        "realised_hedged_vol_ann": np.sqrt(g["real_var"].mean() * 252),
        "n_periods": g.size(),
    }).sort_values("mean_qlike")
