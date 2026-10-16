"""B8: transparent, causal regime classification (flags may overlap).

MARKET_STRESS : IMOEX 20d return <= market_ret20  OR  IMOEX 20d realised vol >= its trailing-500d quantile q_vol
CREDIT_STRESS : Samolet median YTM >= its trailing-250d quantile q_ytm  OR  Samolet bond index 20d return <= credit_bond_ret20
RATES_STRESS  : RGBI 20d return <= rates_rgbi_ret20 (OFZ prices down ~ yields up)
COMBINED_STRESS : at least two of the above
NORMAL        : none
Every statistic on day t uses data up to and including t only (trailing windows); decisions made at the close of t.
Thresholds come from configs/project.yaml and are NOT optimised. Sensitivity: scale 0.75 (looser: return thresholds x0.75,
quantiles -0.10) and 1.25 (stricter: x1.25, quantiles +0.10).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABELS = ["NORMAL", "RATES_STRESS", "MARKET_STRESS", "CREDIT_STRESS", "COMBINED_STRESS"]


def classify(R: pd.DataFrame, credit: pd.DataFrame, th: dict, scale: float = 1.0) -> pd.DataFrame:
    idx = R.index
    mkt = R["IMOEX"].fillna(0.0)
    m20 = (1 + mkt).rolling(20).apply(np.prod, raw=True) - 1
    v20 = mkt.rolling(20).std() * np.sqrt(252)
    qshift = lambda q: float(np.clip(q + (scale - 1.0) * 0.4, 0.5, 0.95))  # noqa: E731  scale<1 looser, >1 stricter
    vq = v20.rolling(500, min_periods=120).quantile(qshift(th["market_vol_pct"]))
    ytm = credit["ytm_median_pct"].reindex(idx).ffill()
    yq = ytm.rolling(250, min_periods=60).quantile(qshift(th["credit_ytm_pct"]))
    bond = credit["bond_price_ret_median"].reindex(idx).fillna(0.0)
    b20 = (1 + bond).rolling(20).apply(np.prod, raw=True) - 1
    rg = R["RGBI"].fillna(0.0)
    r20 = (1 + rg).rolling(20).apply(np.prod, raw=True) - 1
    out = pd.DataFrame(index=idx)
    out["MARKET_STRESS"] = (m20 <= th["market_ret20"] * scale) | (v20 >= vq)
    out["CREDIT_STRESS"] = (ytm >= yq) | (b20 <= th["credit_bond_ret20"] * scale)
    out["RATES_STRESS"] = r20 <= th["rates_rgbi_ret20"] * scale
    n = out[["MARKET_STRESS", "CREDIT_STRESS", "RATES_STRESS"]].sum(axis=1)
    out["COMBINED_STRESS"] = n >= 2
    out["regime"] = np.select([n >= 2, out["CREDIT_STRESS"], out["MARKET_STRESS"], out["RATES_STRESS"]],
                              ["COMBINED_STRESS", "CREDIT_STRESS", "MARKET_STRESS", "RATES_STRESS"], "NORMAL")
    out["imoex_ret20"], out["imoex_vol20"], out["ytm_median"], out["bond_ret20"], out["rgbi_ret20"] = m20, v20, ytm, b20, r20
    return out
