"""V3 Phase 2: regime-conditioned correlation / covariance as an explicit modelling layer.

Two CAUSAL axes (every statistic on day t uses data <= t; thresholds pre-declared in configs/project.yaml v3):
  macro regime (V2 portfolio/regimes.py): NORMAL / MARKET_STRESS / CREDIT_STRESS / RATES_STRESS / COMBINED_STRESS
  SMLT price state (new):
    SMLT_HIGH_VOL_SHOCK  20d realised vol >= trailing-500d q_vol quantile, or a |1-day move| >= shock_1d in the
                         last shock_window days                                               (checked first)
    SMLT_DOWNTREND       60d return <= r60_down, or drawdown from the trailing 250d high <= dd_down with 20d return < 0
    SMLT_UPTREND         60d return >= r60_up and 20d return > 0
    SMLT_RANGE           otherwise
Portfolio states used by the library / state machine (macro stress has priority over the price state):
    P_COMBINED_STRESS <- COMBINED_STRESS          P_CREDIT_STRESS <- CREDIT_STRESS
    P_MARKET_STRESS   <- MARKET_STRESS or RATES_STRESS (systematic stress)
    P_DOWNTREND       <- NORMAL & (DOWNTREND or HIGH_VOL_SHOCK)
    P_RECOVERY        <- NORMAL & UPTREND          P_NORMAL <- NORMAL & RANGE
Small-sample protection (pre-declared): Sigma*_r = lam * Sigma_global + (1 - lam) * Sigma_r,  lam = n0 / (n0 + n_r)
(n0 = shrink_n0 days of prior weight), i.e. strong pooling for rare states. Correlations are shrunk the same way.
Transition matrices are DESCRIPTIVE (persistence / switching), not forecasts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..risk.covariance import cov_ewma, cov_lw, cov_oas, cov_sample

PRICE_STATES = ["SMLT_UPTREND", "SMLT_RANGE", "SMLT_DOWNTREND", "SMLT_HIGH_VOL_SHOCK"]
PORTFOLIO_STATES = ["P_NORMAL", "P_DOWNTREND", "P_MARKET_STRESS", "P_CREDIT_STRESS", "P_COMBINED_STRESS", "P_RECOVERY"]


def price_state(r: pd.Series, th: dict, scale: float = 1.0) -> pd.DataFrame:
    """Causal SMLT price-state classification from daily total returns r (NaN treated as 0 for levels)."""
    x = r.fillna(0.0)
    lvl = (1 + x).cumprod()
    r20 = lvl / lvl.shift(20) - 1
    r60 = lvl / lvl.shift(60) - 1
    dd = lvl / lvl.rolling(250, min_periods=60).max() - 1
    vol20 = x.rolling(20).std() * np.sqrt(252)
    q = float(np.clip(th["vol_q"] + (scale - 1.0) * 0.2, 0.5, 0.98))
    vq = vol20.rolling(500, min_periods=120).quantile(q)
    shock = x.abs().rolling(th["shock_window"]).max() >= th["shock_1d"] * scale
    hv = (vol20 >= vq) | shock
    down = (r60 <= th["r60_down"] * scale) | ((dd <= th["dd_down"] * scale) & (r20 < 0))
    up = (r60 >= th["r60_up"] * scale) & (r20 > 0)
    st = np.select([hv, down, up], ["SMLT_HIGH_VOL_SHOCK", "SMLT_DOWNTREND", "SMLT_UPTREND"], "SMLT_RANGE")
    out = pd.DataFrame({"price_state": st, "smlt_r20": r20, "smlt_r60": r60, "smlt_dd250": dd, "smlt_vol20": vol20,
                        "smlt_vol20_q": vq}, index=r.index)
    out.loc[r60.isna(), "price_state"] = "UNDEFINED"
    return out


def portfolio_state(macro: pd.Series, ps: pd.Series) -> pd.Series:
    m, p = macro.reindex(ps.index).fillna("NORMAL"), ps
    return pd.Series(np.select(
        [m.eq("COMBINED_STRESS"), m.eq("CREDIT_STRESS"), m.isin(["MARKET_STRESS", "RATES_STRESS"]),
         p.isin(["SMLT_DOWNTREND", "SMLT_HIGH_VOL_SHOCK"]), p.eq("SMLT_UPTREND")],
        ["P_COMBINED_STRESS", "P_CREDIT_STRESS", "P_MARKET_STRESS", "P_DOWNTREND", "P_RECOVERY"], "P_NORMAL"), index=ps.index)


def state_masks(reg: pd.DataFrame, ps: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean day masks for every conditioning set (macro flags may overlap; price states are exclusive)."""
    idx = ps.index
    reg = reg.reindex(idx)
    m = {"FULL": pd.Series(True, index=idx), "NORMAL": reg["regime"].eq("NORMAL")}
    for k in ("MARKET_STRESS", "CREDIT_STRESS", "RATES_STRESS", "COMBINED_STRESS"):
        m[k] = reg[k].fillna(False).astype(bool)
    for k in PRICE_STATES:
        m[k] = ps["price_state"].eq(k)
    pst = portfolio_state(reg["regime"], ps["price_state"])
    for k in PORTFOLIO_STATES:
        m[k] = pst.eq(k)
    return m


def shrink_lambda(n: int, n0: float) -> float:
    return float(n0 / (n0 + max(n, 0)))


def conditional_matrices(X: pd.DataFrame, mask: pd.Series, global_cov: pd.DataFrame, n0: float, lam_ewma: float) -> dict:
    d = X[mask.reindex(X.index).fillna(False).values].dropna()
    n = len(d)
    out = {"n": n}
    if n < 10:
        return out
    cols = X.columns
    out["corr"] = d.corr()
    out["sample"] = pd.DataFrame(cov_sample(d.values), index=cols, columns=cols)
    if n > len(cols) + 2:
        out["oas"] = pd.DataFrame(cov_oas(d.values), index=cols, columns=cols)
        out["ledoit_wolf"] = pd.DataFrame(cov_lw(d.values), index=cols, columns=cols)
    out["ewma"] = pd.DataFrame(cov_ewma(d.values, lam_ewma), index=cols, columns=cols)
    lam = shrink_lambda(n, n0)
    out["lambda"] = lam
    out["shrunk"] = lam * global_cov + (1 - lam) * out["sample"]
    sd = np.sqrt(np.diag(out["shrunk"].values))
    out["corr_shrunk"] = pd.DataFrame(out["shrunk"].values / np.outer(sd, sd), index=cols, columns=cols)
    return out


def fisher_change(r_reg: float, n_reg: int, r_glob: float, n_glob: int) -> tuple[float, float]:
    """z-statistic and two-sided p-value for H0: rho_regime == rho_global (independent-samples Fisher z)."""
    from scipy.stats import norm
    if n_reg < 10 or not np.isfinite(r_reg):
        return np.nan, np.nan
    z1, z2 = np.arctanh(np.clip(r_reg, -0.999, 0.999)), np.arctanh(np.clip(r_glob, -0.999, 0.999))
    se = np.sqrt(1 / (n_reg - 3) + 1 / max(n_glob - 3, 1))
    z = (z1 - z2) / se
    return float(z), float(2 * (1 - norm.cdf(abs(z))))


def transitions(labels: pd.Series, order: list[str]) -> pd.DataFrame:
    s = labels.dropna()
    s = s[s != "UNDEFINED"]
    a, b = s.iloc[:-1].values, s.iloc[1:].values
    ct = pd.crosstab(pd.Series(a, name="from"), pd.Series(b, name="to")).reindex(index=order, columns=order, fill_value=0)
    pr = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0)
    rows = []
    for f in order:
        n = int(ct.loc[f].sum())
        stay = float(pr.loc[f, f]) if n else np.nan
        rows.append({"from": f, "n_days_from": n, **{f"p_to_{t}": float(pr.loc[f, t]) if n else np.nan for t in order},
                     "persistence_p_stay": stay,
                     "expected_duration_days": (1 / (1 - stay)) if (n and stay < 1) else np.nan,
                     "n_spells": int(((s == f) & (s.shift(1) != f)).sum())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ states on simulated paths (Phases 4-6)
MACRO_CODES = ["NORMAL", "MARKET_STRESS", "CREDIT_STRESS", "RATES_STRESS", "COMBINED_STRESS"]
PRICE_CODES = PRICE_STATES + ["UNDEFINED"]
ACTIVE_CODES = ["HOLD", "DERISK", "RERISK"]


def _roll_sum(x: np.ndarray, w: int) -> np.ndarray:
    c = np.cumsum(np.concatenate([np.zeros((x.shape[0], 1)), x], axis=1), axis=1)
    out = np.full(x.shape, np.nan)
    out[:, w - 1:] = c[:, w:] - c[:, :-w]
    return out


def path_labels(P: dict, prefix: pd.DataFrame, th_macro: dict, th_price: dict, frozen: dict) -> dict:
    """Causal macro / price-state codes (N, H+1) on simulated paths (column 0 = decision date).

    prefix: the last >= 260 historical daily returns (SMLT, IMOEX, SMLT_BOND_IDX, RGBI) ending at the cutoff.
    frozen: thresholds that need long histories are FROZEN at their cutoff values (vol quantiles, YTM quantile),
            YTM on a path = YTM_0 - (cumulative Samolet bond log return) / modified duration * 100 (first order).
    """
    n = P["SMLT"].shape[0]
    L = len(prefix)

    def full(col, sim):
        h = np.nan_to_num(prefix[col].values)
        return np.concatenate([np.repeat(h[None, :], n, axis=0), np.nan_to_num(sim)], axis=1)

    s = np.log1p(full("SMLT", P["SMLT"]))
    mk = np.log1p(full("IMOEX", P.get("IMOEX", P.get("MIX_FUT"))))
    bd = np.log1p(full("SMLT_BOND_IDX", P["SMLT_BOND_IDX"]))
    rg = np.log1p(full("RGBI", P["RGBI"]))
    m20 = np.expm1(_roll_sum(mk, 20))
    v20 = np.sqrt(_roll_sum(np.expm1(mk) ** 2, 20) / 20) * np.sqrt(252)
    b20 = np.expm1(_roll_sum(bd, 20))
    r20 = np.expm1(_roll_sum(rg, 20))
    ytm = frozen["ytm0"] - (np.cumsum(bd, axis=1) - np.cumsum(bd, axis=1)[:, [L - 1]]) / frozen["bond_dmod"] * 100
    mkt = (m20 <= th_macro["market_ret20"]) | (v20 >= frozen["mkt_vol_q"])
    cr = (ytm >= frozen["ytm_q"]) | (b20 <= th_macro["credit_bond_ret20"])
    rt = r20 <= th_macro["rates_rgbi_ret20"]
    k = mkt.astype(int) + cr.astype(int) + rt.astype(int)
    macro = np.select([k >= 2, cr, mkt, rt], [4, 2, 1, 3], 0)
    lvl = np.cumsum(s, axis=1)
    s20 = np.expm1(_roll_sum(s, 20))
    s60 = np.expm1(_roll_sum(s, 60))
    peak = np.maximum.accumulate(lvl, axis=1)          # prefix covers >= 250 days: running max ~ trailing 250d high
    dd = np.expm1(lvl - peak)
    sv = np.sqrt(_roll_sum(np.expm1(s) ** 2, 20) / 20) * np.sqrt(252)
    shock = _roll_sum((np.abs(np.expm1(s)) >= th_price["shock_1d"]).astype(float), th_price["shock_window"]) > 0
    hv = (sv >= frozen["smlt_vol_q"]) | shock
    down = (s60 <= th_price["r60_down"]) | ((dd <= th_price["dd_down"]) & (s20 < 0))
    up = (s60 >= th_price["r60_up"]) & (s20 > 0)
    price = np.select([hv, down, up], [3, 2, 0], 1)
    return {"macro": macro[:, L - 1:], "price": price[:, L - 1:], "shock": (np.abs(np.expm1(s)) >= th_price["shock_1d"])[:, L:]}


def portfolio_codes(macro: np.ndarray, price: np.ndarray) -> np.ndarray:
    """PORTFOLIO_STATES index: 0 P_NORMAL 1 P_DOWNTREND 2 P_MARKET_STRESS 3 P_CREDIT_STRESS 4 P_COMBINED_STRESS 5 P_RECOVERY."""
    return np.select([macro == 4, macro == 2, np.isin(macro, [1, 3]), np.isin(price, [2, 3]), price == 0], [4, 3, 2, 1, 5], 0)


def active_codes(macro: np.ndarray, price: np.ndarray) -> np.ndarray:
    """0 HOLD, 1 DERISK (credit/combined stress AND SMLT down-trend or high-vol shock), 2 RERISK (macro NORMAL AND
    SMLT up-trend)."""
    return np.select([np.isin(macro, [2, 4]) & np.isin(price, [2, 3]), (macro == 0) & (price == 0)], [1, 2], 0)


def history_codes(reg: pd.DataFrame, ps: pd.DataFrame) -> pd.DataFrame:
    m = reg["regime"].map({k: i for i, k in enumerate(MACRO_CODES)}).fillna(0).astype(int)
    p = ps["price_state"].map({k: i for i, k in enumerate(PRICE_CODES)}).fillna(4).astype(int)
    return pd.DataFrame({"macro": m, "price": p, "portfolio": portfolio_codes(m.values, p.values),
                         "active": active_codes(m.values, p.values)}, index=ps.index)
