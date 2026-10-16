"""Empirical candidate analysis vs SMLT (returns only, point-in-time panel)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .stats import conditional_corr, downside_corr, mvhr, rolling_beta, rolling_corr, rolling_vol, tail_codependence


def candidate_table(R: pd.DataFrame, target: str = "SMLT", market: str = "IMOEX", cutoff: str | None = None,
                    vol_windows=(20, 60, 120, 250), corr_windows=(60, 120, 250)) -> pd.DataFrame:
    s = R[target]
    mkt = R[market] if market in R else None
    stress = (mkt <= mkt.quantile(0.10)) if mkt is not None else None
    s_down = s < 0
    rows = []
    for c in R.columns:
        if c == target:
            continue
        h = R[c]
        pair = pd.concat([s, h], axis=1).dropna()
        if len(pair) < 60:
            rows.append({"series": c, "n_obs": len(pair), "note": "insufficient overlap"})
            continue
        row = {"series": c, "n_obs": len(pair), "first_obs": pair.index.min().date(), "last_obs": pair.index.max().date()}
        for w in vol_windows:
            row[f"vol_{w}d"] = float(rolling_vol(h, w).dropna().iloc[-1]) if h.dropna().size >= w else np.nan
        for w in corr_windows:
            rc = rolling_corr(s, h, w).dropna()
            row[f"corr_{w}d"] = float(rc.iloc[-1]) if len(rc) else np.nan
            rb = (rolling_beta(s, h, w)).dropna()
            row[f"hstar_{w}d"] = float(rb.iloc[-1]) if len(rb) else np.nan
        full = mvhr(pair.iloc[:, 0], pair.iloc[:, 1])
        row.update({"corr_full": full["rho"], "hstar_full": full["h_star"], "hstar_identity_err": full["identity_abs_err"],
                    "HE_VAR_insample_full": full["hedge_effectiveness_r2"]})
        # stability: yearly correlations and rolling-120d dispersion
        yearly = pair.groupby(pair.index.year).apply(lambda d: d.iloc[:, 0].corr(d.iloc[:, 1]) if len(d) > 40 else np.nan)
        for y, v in yearly.items():
            row[f"corr_{y}"] = float(v)
        rc120 = rolling_corr(s, h, 120).dropna()
        rc120 = rc120[rc120.index >= rc120.index.max() - pd.Timedelta(days=730)]
        row["corr120_min_2y"] = float(rc120.min()) if len(rc120) else np.nan
        row["corr120_max_2y"] = float(rc120.max()) if len(rc120) else np.nan
        row["corr120_std_2y"] = float(rc120.std()) if len(rc120) else np.nan
        row["downside_corr"] = downside_corr(s, h)
        row["corr_given_smlt_down"] = conditional_corr(s, h, s_down)
        row["corr_market_stress"] = conditional_corr(s, h, stress) if stress is not None else np.nan
        row.update({f"tail_{k}": v for k, v in tail_codependence(s, h, 0.05).items()})
        # non-synchronous trading diagnostic: lead/lag correlations
        row["corr_lag_minus1"] = float(s.corr(h.shift(1)))
        row["corr_lag_plus1"] = float(s.corr(h.shift(-1)))
        # SMLT worst 5% days: mean candidate return (what a LONG diversifier delivered when needed)
        worst = s <= s.quantile(0.05)
        row["mean_ret_on_smlt_worst5pct"] = float(h[worst].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def rolling_hedge_ratio_panel(R: pd.DataFrame, target: str, hedges: list[str], windows=(60, 120, 250)) -> pd.DataFrame:
    """Trailing h* (not shifted; stamp t uses data <= t) for every hedge/window."""
    out = {}
    for h in hedges:
        if h not in R:
            continue
        for w in windows:
            out[f"hstar_{h}_{w}d"] = rolling_beta(R[target], R[h], w)
            out[f"corr_{h}_{w}d"] = rolling_corr(R[target], R[h], w)
    df = pd.DataFrame(out)
    df.index.name = "date"
    return df
