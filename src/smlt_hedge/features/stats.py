"""Return-based candidate statistics (never on price levels).

All rolling estimators are *trailing*: the value stamped on day t uses returns up to and
including t. When used as a decision input for day t+1, callers must ``.shift(1)`` (the
walk-forward engine does this explicitly and a test checks it).
Causal rolling beta follows Antigravity/Correlations/src/residuals.py::_trailing_beta
(REFERENCE, re-implemented; see EXISTING_CODE_REUSE_AUDIT.md).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simple_returns(prices: pd.Series | pd.DataFrame):
    return prices.pct_change(fill_method=None)


def log_returns(prices):
    return np.log(prices).diff()


def rolling_vol(r: pd.Series, window: int, periods: int = 252) -> pd.Series:
    return r.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(periods)


def rolling_corr(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    return a.rolling(window, min_periods=window).corr(b)


def rolling_beta(asset: pd.Series, bench: pd.Series, window: int) -> pd.Series:
    """Trailing OLS beta of ``asset`` on ``bench`` = Cov/Var (ddof=1 both). Not shifted."""
    cov = asset.rolling(window, min_periods=window).cov(bench)
    var = bench.rolling(window, min_periods=window).var()
    return cov / var.replace(0.0, np.nan)


def mvhr(r_s, r_h) -> dict:
    """Minimum-variance hedge ratio h* = Cov(S,H)/Var(H) and identity check h* = rho*sS/sH.

    Returned h* is the notional of H to SHORT per unit notional of S.
    """
    df = pd.concat([pd.Series(r_s), pd.Series(r_h)], axis=1).dropna()
    s, h = df.iloc[:, 0].values, df.iloc[:, 1].values
    cov = np.cov(s, h, ddof=1)
    h_star = cov[0, 1] / cov[1, 1]
    rho = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    h_id = rho * np.sqrt(cov[0, 0]) / np.sqrt(cov[1, 1])
    return {
        "h_star": float(h_star),
        "rho": float(rho),
        "sigma_s": float(np.sqrt(cov[0, 0])),
        "sigma_h": float(np.sqrt(cov[1, 1])),
        "h_identity": float(h_id),
        "identity_abs_err": float(abs(h_star - h_id)),
        "hedge_effectiveness_r2": float(rho**2),  # = HE_VAR of the in-sample MV hedge
        "n": int(len(df)),
    }


def downside_corr(a: pd.Series, b: pd.Series) -> float:
    """Correlation computed on days where BOTH returns are negative (semi-correlation)."""
    df = pd.concat([a, b], axis=1).dropna()
    m = (df.iloc[:, 0] < 0) & (df.iloc[:, 1] < 0)
    return float(df[m].corr().iloc[0, 1]) if m.sum() > 10 else float("nan")


def conditional_corr(a: pd.Series, b: pd.Series, cond: pd.Series) -> float:
    df = pd.concat([a, b, cond.rename("_c")], axis=1).dropna()
    df = df[df["_c"].astype(bool)]
    return float(df.iloc[:, 0].corr(df.iloc[:, 1])) if len(df) > 10 else float("nan")


def tail_codependence(a: pd.Series, b: pd.Series, q: float = 0.05) -> dict:
    """Empirical lower-tail co-exceedance: P(b <= q_b | a <= q_a), vs. q under independence."""
    df = pd.concat([a, b], axis=1).dropna()
    x, y = df.iloc[:, 0], df.iloc[:, 1]
    ea, eb = x <= x.quantile(q), y <= y.quantile(q)
    n = int(ea.sum())
    p = float((ea & eb).sum() / n) if n else float("nan")
    # upper tail of the hedge when S crashes (what a SHORT hedge needs is b falling with a)
    return {"lower_tail_coexceed": p, "independence_level": q, "n_tail_obs": n}
