"""Risk and hedge-effectiveness metrics.

Sign convention (used everywhere in this package):
    P&L  > 0 is a gain; LOSS = -P&L, so losses are POSITIVE numbers.
    VaR_a  = a-quantile of the loss distribution (a = 0.95 or 0.99).
    ES_a   = Rockafellar-Uryasev: VaR_a + E[(L - VaR_a)^+] / (1 - a).
For continuous distributions ES_a = E[L | L >= VaR_a]; the RU form is also correct for
discrete samples with atoms and is exactly the quantity minimised by the CVaR LP
(see optimization/cvar.py), so reported ES and optimised ES are the same object.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _arr(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return a[~np.isnan(a)]


def var_hist(pnl, alpha: float = 0.95) -> float:
    """Historical VaR of a P&L sample, reported as a positive loss.

    Uses the lower-inverse empirical quantile of the loss (no interpolation), which is the
    VaR that the RU LP recovers as its optimal auxiliary variable.
    """
    loss = np.sort(-_arr(pnl))
    n = loss.size
    if n == 0:
        return float("nan")
    k = int(np.ceil(alpha * n)) - 1
    return float(loss[min(max(k, 0), n - 1)])


def es_hist(pnl, alpha: float = 0.95) -> float:
    loss = -_arr(pnl)
    if loss.size == 0:
        return float("nan")
    v = var_hist(pnl, alpha)
    return float(v + np.mean(np.maximum(loss - v, 0.0)) / (1.0 - alpha))


def ann_vol(returns, periods: int = 252) -> float:
    r = _arr(returns)
    return float(np.std(r, ddof=1) * np.sqrt(periods)) if r.size > 1 else float("nan")


def max_drawdown(wealth) -> float:
    """Maximum drawdown of a wealth (level) path, as a positive fraction of the running peak."""
    w = _arr(wealth)
    if w.size == 0:
        return float("nan")
    peak = np.maximum.accumulate(w)
    return float(np.max(1.0 - w / peak))


def max_drawdown_rub(cum_pnl) -> float:
    """Maximum peak-to-trough decline of a cumulative RUB P&L path (positive RUB)."""
    c = np.concatenate([[0.0], _arr(cum_pnl)])
    return float(np.max(np.maximum.accumulate(c) - c))


def worst_k_day(pnl_daily, k: int) -> float:
    """Worst rolling k-day cumulative P&L expressed as a positive loss (RUB or return units)."""
    s = pd.Series(_arr(pnl_daily))
    if len(s) < k:
        return float("nan")
    return float(-s.rolling(k).sum().min())


def capture_ratios(port_ret, bench_ret) -> tuple[float, float]:
    """Upside / downside capture relative to the unhedged position.

    up   = sum(port | bench > 0) / sum(bench | bench > 0)
    down = sum(port | bench < 0) / sum(bench | bench < 0)
    A perfect hedge has both near 0; unhedged has both = 1.
    """
    p, b = np.asarray(port_ret, float), np.asarray(bench_ret, float)
    m = ~(np.isnan(p) | np.isnan(b))
    p, b = p[m], b[m]
    up, dn = b > 0, b < 0
    u = p[up].sum() / b[up].sum() if up.any() else float("nan")
    d = p[dn].sum() / b[dn].sum() if dn.any() else float("nan")
    return float(u), float(d)


# ---------------------------------------------------------------- course metrics
def he_pnl(dv_hedge, dv_spot) -> float:
    """Course metric HE_PNL = -dV_H / dV_S over one holding period (aggregate)."""
    s = float(np.nansum(dv_spot))
    return float(-np.nansum(dv_hedge) / s) if s != 0 else float("nan")


def he_var(dv_hedged, dv_unhedged) -> float:
    """Course metric HE_VAR = 1 - Var(dV_hedged) / Var(dV_unhedged)."""
    a, b = _arr(dv_hedged), _arr(dv_unhedged)
    return float(1.0 - np.var(a, ddof=1) / np.var(b, ddof=1))


def summary(pnl_rub: pd.Series, *, base_rub: float, bench_pnl_rub: pd.Series | None = None,
            periods: int = 252) -> dict:
    """Standard metric block for a daily RUB P&L series, normalised by ``base_rub``.

    The caller runs it twice: base = RUB 10m SMLT exposure (A) and base = RUB 20m total capital (B).
    """
    pnl = pnl_rub.dropna()
    r = pnl / base_rub
    wealth = 1.0 + r.cumsum()  # arithmetic, base-normalised (no compounding of a fixed-notional overlay)
    out = {
        "ann_vol": ann_vol(r, periods),
        "VaR95_1d": var_hist(r, 0.95),
        "ES95_1d": es_hist(r, 0.95),
        "VaR99_1d": var_hist(r, 0.99),
        "ES99_1d": es_hist(r, 0.99),
        "max_drawdown": max_drawdown_rub(pnl.values) / base_rub,
        "worst_1d": worst_k_day(r, 1),
        "worst_5d": worst_k_day(r, 5),
        "worst_20d": worst_k_day(r, 20),
        "total_return": float(r.sum()),
        "n_days": int(r.size),
        "final_wealth_index": float(wealth.iloc[-1]) if len(wealth) else float("nan"),
    }
    if bench_pnl_rub is not None:
        b = bench_pnl_rub.reindex(pnl.index) / base_rub
        u, d = capture_ratios(r, b)
        out["upside_capture"], out["downside_capture"] = u, d
    return out
