"""Point-in-time multivariate factor model for SMLT daily returns (explanatory, NOT a forecasting model).

    r_SMLT,t = a + b_M * MKT_t + b_S * SECTOR_t + b_R * RATES_t + b_C * CREDIT_t + e_t

Factor definitions (returns / changes only, never levels):
  MKT     IMOEX daily price return (broad market).
  SECTOR  equal-weight daily return of listed developer peers EX-SMLT (PIKK, LSRG, ETLN; mean of available names).
          The MOEX real-estate index is NOT used as a factor because SMLT is one of its constituents (circularity).
  RATES   RGBI (OFZ price index) daily return = duration-weighted rate factor (approx. -D_mod * dy).
  CREDIT  duration-consistent excess return of Samolet bonds over OFZ:
              r_bond,t - (D_bond,t / D_RGBI,t) * r_RGBI,t
          where r_bond is the median day-on-day clean-price return of Samolet bonds traded on both days
          (constant composition) and D are ISS durations (days). MOEX ZSPREAD is populated only from Dec-2025,
          so a Z-spread-change factor (CREDIT_Z = -median dZ) is reported as a robustness variant on that sub-sample.
          CAUTION: an issuer-own credit factor is jointly determined with the equity -> a co-movement measure,
          not an exogenous driver; its share is issuer credit risk, not market risk.
Decomposition: Shapley (average incremental R^2 over all factor orderings) -> order-independent shares.
The residual is "variance unexplained by these factors" -- NOT automatically issuer-specific economic risk:
it also contains factor mis-specification, non-synchronous closes and estimation noise.
Rolling loadings are trailing (stamp t uses data <= t); decision use must shift by one day.
"""
from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd

FACTORS = ["MKT", "SECTOR", "RATES", "CREDIT"]


def factor_frame(R: pd.DataFrame, credit: pd.DataFrame, peers=("PIKK", "LSRG", "ETLN")) -> pd.DataFrame:
    F = pd.DataFrame(index=R.index)
    F["SMLT"] = R["SMLT"]
    F["MKT"] = R["IMOEX"]
    F["SECTOR"] = R[[p for p in peers if p in R]].mean(axis=1, skipna=True)
    F["RATES"] = R["RGBI"]
    c = credit.reindex(R.index)
    ratio = (c["bond_duration_median_days"] / c["rgbi_duration_days"]).clip(0, 2) if "rgbi_duration_days" in c else 0.0
    F["CREDIT"] = c["bond_price_ret_median"] - ratio * R["RGBI"]
    F["CREDIT_Z"] = -c["dz_median_pp"] / 100.0                       # robustness variant (Dec-2025 onwards)
    return F


def ols(y: np.ndarray, X: np.ndarray, nw_lags: int = 5) -> dict:
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    e = y - X1 @ beta
    n, k = X1.shape
    # Newey-West HAC covariance
    XtX_inv = np.linalg.inv(X1.T @ X1)
    S = (X1 * e[:, None]).T @ (X1 * e[:, None])
    for L in range(1, nw_lags + 1):
        w = 1 - L / (nw_lags + 1)
        G = (X1[L:] * e[L:, None]).T @ (X1[:-L] * e[:-L, None])
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    r2 = 1 - e.var() / y.var()
    return {"beta": beta, "se": np.sqrt(np.diag(V)), "r2": float(r2), "resid": e, "n": n}


def shapley_r2(F: pd.DataFrame, factors=FACTORS) -> dict:
    D = F[["SMLT"] + list(factors)].dropna()
    y = D["SMLT"].values
    cache = {(): 0.0}
    for k in range(1, len(factors) + 1):
        for S in combinations(factors, k):
            cache[S] = ols(y, D[list(S)].values)["r2"]
    n = len(factors)
    out = {}
    for f in factors:
        val = 0.0
        others = [g for g in factors if g != f]
        for k in range(0, n):
            for S in combinations(others, k):
                w = factorial(k) * factorial(n - k - 1) / factorial(n)
                with_f = tuple(sorted(S + (f,), key=factors.index))
                val += w * (cache[with_f] - cache[tuple(sorted(S, key=factors.index))])
        out[f] = val
    out["TOTAL_R2"] = cache[tuple(factors)]
    out["UNEXPLAINED"] = 1 - out["TOTAL_R2"]
    out["n_obs"] = len(D)
    out["first"], out["last"] = D.index.min(), D.index.max()
    return out


def decomposition_table(F: pd.DataFrame, windows: dict) -> pd.DataFrame:
    """windows: label -> (start, end) date strings. Single-factor R^2 (v1 logic) is reported next to Shapley shares."""
    rows = []
    for label, (a, b) in windows.items():
        sub = F.loc[a:b]
        sh = shapley_r2(sub)
        D = sub[["SMLT", "MKT"]].dropna()
        mkt_only = float(np.corrcoef(D["SMLT"], D["MKT"])[0, 1] ** 2)
        full = ols(sub[["SMLT"] + FACTORS].dropna()["SMLT"].values, sub[["SMLT"] + FACTORS].dropna()[FACTORS].values)
        rows.append({"window": label, "start": str(sh["first"].date()), "end": str(sh["last"].date()), "n_obs": sh["n_obs"],
                     "v1_single_factor_market_R2": mkt_only, "v1_residual_label_unexplained_by_market": 1 - mkt_only,
                     **{f"shapley_{f}": sh[f] for f in FACTORS}, "total_R2": sh["TOTAL_R2"], "unexplained": sh["UNEXPLAINED"],
                     **{f"beta_{f}": float(full["beta"][i + 1]) for i, f in enumerate(FACTORS)},
                     **{f"t_{f}": float(full["beta"][i + 1] / full["se"][i + 1]) for i, f in enumerate(FACTORS)}})
    return pd.DataFrame(rows)


def rolling_loadings(F: pd.DataFrame, windows=(60, 120, 250)) -> pd.DataFrame:
    D = F[["SMLT"] + FACTORS].dropna()
    out = {}
    for w in windows:
        betas = np.full((len(D), len(FACTORS) + 1), np.nan)
        r2 = np.full(len(D), np.nan)
        y, X = D["SMLT"].values, D[FACTORS].values
        for i in range(w - 1, len(D)):
            res = ols(y[i - w + 1:i + 1], X[i - w + 1:i + 1], nw_lags=0)
            betas[i] = res["beta"]
            r2[i] = res["r2"]
        for j, f in enumerate(FACTORS):
            out[f"beta_{f}_{w}d"] = betas[:, j + 1]
        out[f"r2_{w}d"] = r2
    return pd.DataFrame(out, index=D.index)


def risk_contributions(F: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Euler variance contributions b_k Cov(f_k, r)/Var(r) of the full-window OLS fit (sum + residual = 1)."""
    D = F.loc[a:b][["SMLT"] + FACTORS].dropna()
    res = ols(D["SMLT"].values, D[FACTORS].values)
    var_r = D["SMLT"].var(ddof=0)
    rows = []
    for i, f in enumerate(FACTORS):
        c = res["beta"][i + 1] * np.cov(D[f], D["SMLT"], ddof=0)[0, 1] / var_r
        rows.append({"component": f, "beta": res["beta"][i + 1], "var_contribution": c})
    rows.append({"component": "RESIDUAL", "beta": np.nan, "var_contribution": float(res["resid"].var() / var_r)})
    return pd.DataFrame(rows)
