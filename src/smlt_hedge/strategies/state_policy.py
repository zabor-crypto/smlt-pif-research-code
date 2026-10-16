"""State-dependent hedge policy: a deterministic, pre-specified mapping from observable state to
target hedge exposure. Parameters are fixed a priori (NOT fitted on the evaluation sample) to avoid
in-sample optimisation; the walk-forward backtest then tests the rule out of sample.

States (all computed from data <= decision date t):
  VOL_HIGH    : SMLT 20d realised vol > its trailing 250d median
  DRAWDOWN    : SMLT price drawdown from trailing 120d high > 30%
  CREDIT_STRESS: median Samolet bond Z-spread (pp) > trailing 250d 75th percentile  (if data available)
  RATES_UP    : 1y G-curve yield 20d change > +50bp                           (if data available)
Target: h = clip(base + 0.25*VOL_HIGH + 0.25*CREDIT_STRESS + 0.25*DRAWDOWN + 0.25*RATES_UP, 0, cap)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASE = 0.25
STEP = 0.25


def state_vector(hist: pd.DataFrame) -> dict:
    s = hist["SMLT"].dropna()
    px = hist["SMLT_PX"].dropna() if "SMLT_PX" in hist else (1 + s).cumprod()
    vol20 = s.tail(20).std() * np.sqrt(252) if len(s) >= 20 else np.nan
    vol_hist = s.rolling(20).std().dropna().tail(250) * np.sqrt(252)
    st = {
        "vol20": float(vol20),
        "VOL_HIGH": bool(len(vol_hist) >= 60 and vol20 > vol_hist.median()),
        "drawdown120": float(1 - px.iloc[-1] / px.tail(120).max()) if len(px) else np.nan,
    }
    st["DRAWDOWN"] = bool(st["drawdown120"] > 0.30)
    if "ZSPREAD" in hist and hist["ZSPREAD"].notna().sum() > 60:
        z = hist["ZSPREAD"].dropna()
        st["zspread"] = float(z.iloc[-1])
        st["CREDIT_STRESS"] = bool(z.iloc[-1] > z.tail(250).quantile(0.75))
    else:
        st["CREDIT_STRESS"] = False
    if "G1Y" in hist and hist["G1Y"].notna().sum() > 25:
        g = hist["G1Y"].dropna()
        st["g1y_chg20_bp"] = float((g.iloc[-1] - g.iloc[-min(21, len(g))]) * 100)
        st["RATES_UP"] = bool(st["g1y_chg20_bp"] > 50)
    else:
        st["RATES_UP"] = False
    return st


def state_policy(cap: float = 1.0):
    def f(hist: pd.DataFrame) -> float:
        st = state_vector(hist)
        h = BASE + STEP * (st["VOL_HIGH"] + st["CREDIT_STRESS"] + st["DRAWDOWN"] + st["RATES_UP"])
        return float(np.clip(h, 0.0, cap))
    return f
