"""V3 Phase 3: transparent factor-conditioned FORWARD SCENARIO path model. Scenarios, NOT forecasts.

For scenario S with conditioning regime set G(S):
  1. joint daily LOG returns of every series (SMLT, SMLT futures, sleeves, BTC proxy, factors) on the days of G(S)
     inside the common sample (2022-03-28 .. cutoff), DEMEANED column by column: zero historical drift, residuals
     zero-mean (no SMLT drift is ever extrapolated);
  2. circular block bootstrap of joint rows (dependence within G(S) preserved), scaled by the scenario volatility
     multiplier k(S);
  3. factor tilt: every series i receives sum_k beta_ik(G) * mu_k(S) / 252 per day, where mu_k(S) is the scenario's
     12-month log path for factor k (MKT, SECTOR, RATES, CREDIT) and beta_ik(G) the regime OLS loading shrunk toward
     the full-sample loading (lambda = n0/(n0+n_G)). For SMLT the loadings are drawn per path from
     N(beta_hat, se^2) (Newey-West) -> factor-loading uncertainty enters the fan;
     so  r_SMLT,t = sum_k beta_k(G) f_k,t + e_t(G),  f_k,t = mu_k/252 + (demeaned bootstrapped factor move);
  4. SMLT futures follow the structural stationary-basis model of the V2 engine.
Anchors (pre-declared, PIT): mu_k(S) are EMPIRICAL QUANTILES (q_low / 0.5 / q_high) of overlapping 252-day log returns
of each factor in the common sample, mapped DETERIORATION -> q_low, FLAT -> 0 (no drift), RELIEF -> q_high. The vol
multiplier compares SMLT EWMA volatility at the cutoff with its full-sample median. No external forecast is used
(the Bank of Russia primary source is blocked from the research environment; see FORWARD_SCENARIO_MODEL.md).
Event overlays (conditional paths, "if the event happens"): EVENT_DRIVEN_UPSIDE = FLAT + one-day SMLT jump j_up on
day `event_day`; ADVERSE_CREDIT_EVENT = DETERIORATION + one-day SMLT jump j_dn and credit-factor shock on that day.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..features.factors import FACTORS, ols
from .bootstrap import block_bootstrap_indices

STRUCTURAL = ["CONTINUED_DETERIORATION", "STABILISATION_FLAT", "RATES_RELIEF_RECOVERY"]
OVERLAYS = {"EVENT_DRIVEN_UPSIDE": "STABILISATION_FLAT", "ADVERSE_CREDIT_EVENT": "CONTINUED_DETERIORATION"}


def factor_anchors(X: pd.DataFrame, q_low: float, q_high: float, hz: int = 252) -> pd.DataFrame:
    L = np.log1p(X[FACTORS].dropna())
    roll = L.rolling(hz).sum().dropna()
    return pd.DataFrame({"q_low": roll.quantile(q_low), "median": roll.quantile(0.5), "q_high": roll.quantile(q_high),
                         "n_windows": len(roll)})


def regime_betas(X: pd.DataFrame, mask: pd.Series, cols: list[str], n0: float) -> dict:
    """Loadings of every series on the 4 factors in the regime sample, shrunk to the full-sample loadings."""
    L = np.log1p(X[list(dict.fromkeys(cols + FACTORS))].clip(lower=-0.99))
    out = {}
    full = L.dropna()
    sub = full[mask.reindex(full.index).fillna(False).values]
    lam = n0 / (n0 + len(sub))
    for c in cols:
        if c in FACTORS:
            b = np.array([1.0 if f == c else 0.0 for f in FACTORS])
            out[c] = {"beta": b, "se": np.zeros(4), "lambda": lam, "n": len(sub)}
            continue
        bf = ols(full[c].values, full[FACTORS].values)
        bs = ols(sub[c].values, sub[FACTORS].values) if len(sub) > 60 else bf
        out[c] = {"beta": lam * bf["beta"][1:] + (1 - lam) * bs["beta"][1:],
                  "se": np.sqrt(lam**2 * bf["se"][1:]**2 + (1 - lam)**2 * bs["se"][1:]**2),
                  "lambda": lam, "n": len(sub), "r2_regime": bs["r2"]}
    return out


def simulate(X: pd.DataFrame, cols: list[str], mask: pd.Series, mu: dict, k_vol: float, betas: dict, H: int, n: int,
             block: int, seed: int, jump: dict | None = None, beta_draw: bool = True) -> dict:
    """Daily SIMPLE-return paths {col: (n, H)} under one scenario (see module docstring)."""
    L = np.log1p(X[cols].clip(lower=-0.99)).dropna()
    sub = L[mask.reindex(L.index).fillna(False).values]
    Z = (sub - sub.mean()).values * k_vol
    idx = block_bootstrap_indices(len(Z), H, n, block, seed)
    rng = np.random.default_rng(seed + 7)
    mu_v = np.array([mu[f] for f in FACTORS]) / 252.0
    out = {}
    for j, c in enumerate(cols):
        b = betas[c]["beta"]
        if c == "SMLT" and beta_draw:
            bb = b[None, :] + rng.standard_normal((n, 4)) * betas[c]["se"][None, :]
            tilt = (bb @ mu_v)[:, None]
        else:
            tilt = float(b @ mu_v)
        lr = Z[:, j][idx] + tilt
        if jump and c in jump["series"]:
            lr[:, jump["day"]] += jump["series"][c]
        out[c] = np.expm1(lr)
    out["_idx"], out["_dates"] = idx, sub.index
    return out


def quantile_table(S0: float, r: np.ndarray, nodes: dict[str, int], qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> pd.DataFrame:
    lvl = np.cumprod(1 + r, axis=1)
    rows = []
    for name, d in nodes.items():
        x = lvl[:, d - 1]
        row = {"node": name, "trading_day": d}
        for q in qs:
            row[f"P{int(q * 100):02d}_price"] = float(S0 * np.quantile(x, q))
            row[f"P{int(q * 100):02d}_return"] = float(np.quantile(x, q) - 1)
        row["prob_below_S0"] = float((x < 1).mean())
        row["prob_below_half_S0"] = float((x < 0.5).mean())
        rows.append(row)
    return pd.DataFrame(rows)
