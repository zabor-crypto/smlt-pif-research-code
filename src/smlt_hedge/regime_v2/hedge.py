"""Regime V2 — conditional minimum-variance hedge ratios of the SMLT position (interpretable diagnostics).

    h* = Cov(S, H) / Var(H)          N* = h* x V_A / V_F

S = SMLT daily return, H = daily return of the hedge future (RUB variation margin per RUB of notional), V_A = value of
the SMLT shares held, V_F = notional of one contract. The optimiser is free to choose another combination; these
numbers explain what a textbook hedge of the held position would be in each sample, what it costs and whether the fund
could carry its margin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..adaptive import engine as E
from . import frontier as FR
from . import states as S2


def _es(loss: np.ndarray, a: float = 0.95) -> float:
    return E.es(loss, a)


def samples(idx: pd.DatetimeIndex, s: pd.Series, conf: pd.DataFrame, states_now: dict) -> dict:
    cf = conf.reindex(idx).ffill()
    n = len(idx)

    def after(cond):
        m = S2.cond_mask(cf, cond).values
        return np.r_[False, m[:-1]]
    last = lambda k: np.r_[np.zeros(max(n - k, 0), bool), np.ones(min(k, n), bool)]     # noqa: E731
    return {"GLOBAL_60": last(60), "GLOBAL_120": last(120), "GLOBAL_250": last(250),
            "STATE_MARKET": after({"MARKET_RISK": states_now["MARKET_RISK"]}),
            "STATE_CREDIT": after({"ISSUER_CREDIT_RISK": states_now["ISSUER_CREDIT_RISK"]}),
            "DOWNSIDE_ONLY": (s.values < 0), "STRESS": after({"MARKET_RISK": "STRESS"})}


def ratio(s: np.ndarray, h: np.ndarray) -> dict:
    if len(s) < 10 or float(np.var(h, ddof=1)) <= 0:
        return {"h_star": np.nan, "corr": np.nan, "vol_reduction_in_sample": np.nan, "n_obs": int(len(s))}
    hs = float(np.cov(s, h, ddof=1)[0, 1] / np.var(h, ddof=1))
    sd_s, sd_h = float(s.std(ddof=1)), float(h.std(ddof=1))
    corr = float(np.cov(s, h, ddof=1)[0, 1] / (sd_s * sd_h)) if sd_s > 0 else np.nan
    res = s - hs * h
    return {"h_star": hs, "corr": corr, "vol_reduction_in_sample": float(1 - res.std(ddof=1) / sd_s) if sd_s > 0 else np.nan,
            "n_obs": int(len(s))}


def table(P: dict, su, conf: pd.DataFrame, states_now: dict, cfg: dict, spec: FR.Spec, start: str, t: pd.Timestamp) -> pd.DataFrame:
    hc = cfg["v2"]["hedge"]
    R = P["R"].loc[start:t]
    v = FR._vectors(su)
    js = su.routes.index("SMLT")
    w_s = float(su.x0[js])
    V_A = w_s * su.nav0
    rows = []
    for hd in hc["hedges"]:
        d = pd.concat([R["SMLT"], R[hd]], axis=1).dropna()
        s, h = d.iloc[:, 0], d.iloc[:, 1]
        jh = su.routes.index(hd)
        sm = samples(d.index, s, conf, states_now)
        win = int(hc["stability_window"]["rolling_sessions"])
        look = int(hc["stability_window"]["lookback_sessions"])
        roll = (s.rolling(win).cov(h) / h.rolling(win).var()).iloc[-look:].dropna()
        for name, mask in sm.items():
            r = ratio(s.values[mask], h.values[mask])
            row = {"hedge": hd, "sample": name, **r, "smlt_weight_nav": w_s, "V_A_rub": V_A, "V_F_rub_per_contract": su.notional[hd]}
            if np.isfinite(r["h_star"]):
                n_star = -float(np.trunc(r["h_star"] * V_A / su.notional[hd]))
                w_h = n_star * su.notional[hd] / su.nav0
                row.update({"N_star_contracts": n_star, "hedge_weight_nav": w_h})
                es_u, es_h = [], []
                for m, G in su.G.items():
                    un = w_s * (G[:, js] - su.mH)
                    es_u.append(_es(-un))
                    es_h.append(_es(-(un + w_h * G[:, jh])))
                row.update({"ES95_unhedged_smlt_nav0_worst_model": max(es_u), "ES95_hedged_nav0_worst_model": max(es_h),
                            "ES95_reduction": 1 - max(es_h) / max(es_u) if max(es_u) > 0 else np.nan,
                            "expected_excess_given_up_nav0": float(v["c"][jh] * abs(w_h)) * (1 if w_h < 0 else -1),
                            "roll_cost_nav0": float(v["roll"][jh] * abs(w_h)), "margin_and_reserve_drag_nav0": float(v["drag"][jh] * abs(w_h)),
                            "entry_cost_nav0": float(v["tc"][jh] * abs(w_h)), "initial_margin_nav": abs(w_h) * su.mr[hd],
                            "rolls_in_horizon": int(su.nroll.get(hd, 0))})
                row["expected_cost_total_nav0"] = (row["expected_excess_given_up_nav0"] + row["roll_cost_nav0"]
                                                   + row["margin_and_reserve_drag_nav0"] + row["entry_cost_nav0"])
                if name in hc["engine_checked_samples"] and w_h != 0:
                    x = su.x0.copy()
                    x[jh] = w_h
                    ex = FR.admit(su, x, spec, None)
                    m1 = [q for q in ex["rows"] if q["margin_multiplier"] == 1.0]
                    st = [q for q in ex["rows"] if q["margin_multiplier"] != 1.0]
                    row.update({"engine_ES95_nav0_worst_model": max(q["ES95"] for q in m1) / su.nav0,
                                "engine_p_liquidity_breach": max(q["p_breach"] for q in m1),
                                "engine_p_unresolved_deficit": max(q["p_unresolved"] for q in m1),
                                "engine_p_unresolved_margin_stress": max(q["p_unresolved"] for q in st),
                                "engine_min_available_nav_p05": min(q["min_available_nav_p05"] for q in m1),
                                "engine_peak_margin_nav_p95": max(q["peak_im_nav_p95"] for q in m1),
                                "liquidity_survival": bool(max(q["p_unresolved"] for q in ex["rows"]) == 0
                                                           and max(q["p_breach"] for q in m1) <= cfg["r1"]["admission"]["max_breach_probability"]),
                                "deterministic_paths_survived": bool(not any(q["unresolved_deficit"] for q in ex["det"]))})
            row.update({"rolling_h_mean": float(roll.mean()) if len(roll) else np.nan, "rolling_h_std": float(roll.std(ddof=1)) if len(roll) > 1 else np.nan,
                        "rolling_h_min": float(roll.min()) if len(roll) else np.nan, "rolling_h_max": float(roll.max()) if len(roll) else np.nan})
            rows.append(row)
    return pd.DataFrame(rows)
