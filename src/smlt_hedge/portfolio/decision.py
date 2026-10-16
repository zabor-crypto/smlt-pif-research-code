"""B9 risk contributions and B15 decision table (smart_portfolio_comparison.csv).

All rows are evaluated with the SAME self-financing engine on the SAME four scenario models at the decision date.
Dynamic policies (ROLLING_HSTAR, CURRENT_STATE_DEPENDENT, ADAPTIVE_SMART_PORTFOLIO) enter the static columns
through their decision-date target held over the horizon (label: static_eval_of_decision_target); their dynamic
behaviour is reported in the walk-forward / episode columns (wf_*, ep_*).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..scenarios.bootstrap import SleevePos, evaluate_portfolio
from ..strategies.futures import contract_position
from ..strategies.state_policy import state_policy, state_vector
from . import adaptive, sleeves, smart


def _targets(ctx, pl, sp, env) -> dict:
    S0, F0, pv = pl._fut_ctx(ctx)
    n_sh = ctx.E / S0
    names = [s.name for s in sp]
    di = names.index("DIRECT")

    def direct(h_share):
        v = np.zeros(len(names))
        v[di] = -contract_position(h_share, n_sh, F0, pv)["notional_rub"] / ctx.E if h_share else 0.0
        return v

    R = env.R.loc[:ctx.cutoff]
    w = R[["SMLT", "SMLT_FUT"]].dropna().tail(120)
    hstar = float(np.clip(np.cov(w["SMLT"], w["SMLT_FUT"])[0, 1] / w["SMLT_FUT"].var(), 0, 1))
    hist = R[["SMLT", "SMLT_FUT"]].copy()
    h_state = state_policy()(hist)
    p = env.R.index.get_loc(pd.Timestamp(ctx.cutoff))
    reg = str(env.regimes["regime"].iloc[p])
    pol = ctx.cfg["smart_portfolio"]["regime_policy"][reg]
    h_adapt = adaptive._lp_target(env, p + 1, pol["min_upside"], pol["margin_budget"], adaptive.CASH_ONLY)
    sel = pd.read_csv(ctx.out / "smart_selected_portfolios.csv")
    out = {"CONTROL": {"h": np.zeros(len(names))}}
    for x in (25, 50, 75, 100):
        out[f"FUT_{x}"] = {"h": direct(x / 100)}
    out["ROLLING_HSTAR"] = {"h": direct(hstar), "note": f"h*120d at cutoff = {hstar:.3f}"}
    out["CURRENT_STATE_DEPENDENT"] = {"h": direct(h_state), "note": f"state rule at cutoff h = {h_state:.2f}"}
    for label, rule in (("STATIC_SMART_PORTFOLIO", "static"), ("ROBUST_SMART_PORTFOLIO", "robust")):
        per_h = {}
        for hz in ("6m", "12m"):
            r = sel[(sel["rule"] == rule) & (sel["selection_scope"] == "ANY") & (sel["horizon"] == hz) & (sel["capital"] == ctx.C)]
            if len(r) and r["cand_id"].iloc[0] != "NONE":
                per_h[hz] = np.array([float(r[f"w_{n}"].iloc[0]) for n in names])
        out[label] = {"h_by_horizon": per_h}
    out["ADAPTIVE_SMART_PORTFOLIO"] = {"h": h_adapt, "note": f"regime at cutoff = {reg}; constraints {pol}"}
    out["DE_RISK_SELL_50"] = {"h": np.zeros(len(names)), "derisk": 0.5}
    return out


def run(ctx, pl) -> None:
    sp = sleeves.build(ctx)
    from ..phase2 import _env
    env = _env(ctx, sp)
    tg = _targets(ctx, pl, sp, env)
    names = [s.name for s in sp]
    S0 = pl._fut_ctx(ctx)[0]
    derisk_cost = (0.1 + ctx.cfg["costs"]["equity_commission_bps"] * 1e-4 * S0) / S0
    rows, contrib = [], []
    res = {}
    for hz_name, hz in ctx.cfg["horizons_trading_days"].items():
        models, B = smart.scenario_models(ctx, pl, sp, hz)
        tau = hz / 252
        for label, t in tg.items():
            h = t.get("h_by_horizon", {}).get(hz_name) if "h_by_horizon" in t else t["h"]
            if h is None:
                continue
            sl = [SleevePos(s.series, h[j] * ctx.E, s.mr, s.cost_one_way, s.n_rolls(tau)) for j, s in enumerate(sp) if abs(h[j]) > 1e-6]
            per_model = {m: evaluate_portfolio(P, E=ctx.E, C=ctx.C, sleeves=sl, mmf_daily=B["daily_y"], derisk=t.get("derisk", 0.0),
                                               derisk_cost=derisk_cost, attribution=(m == "FHS_FULL"))
                         for m, P in models.items()}
            res[(label, hz_name)] = {"per_model": per_model, "h": h, "note": t.get("note", "")}
            if m := per_model["FHS_FULL"].get("attribution"):
                for comp, v in m.items():
                    j = [s.series for s in sp].index(comp) if comp in [s.series for s in sp] else None
                    contrib.append({"portfolio": label, "horizon": hz_name, "component": names[j] if j is not None else comp,
                                    "series": comp, "notional_frac": float(h[j]) if j is not None else np.nan,
                                    "margin_used_rub": float(sp[j].mr * abs(h[j]) * ctx.E) if j is not None else 0.0,
                                    "cost_rub": float(abs(h[j]) * ctx.E * sp[j].cost_one_way * (2 + 2 * sp[j].n_rolls(tau))) if j is not None else np.nan,
                                    **{f"{k}": vv for k, vv in v.items()}})
    ctrl6 = res[("CONTROL", "6m")]["per_model"]["FHS_FULL"]["mean_pnl"]
    ctrl12 = res[("CONTROL", "12m")]["per_model"]["FHS_FULL"]["mean_pnl"]
    wf = pd.read_csv(ctx.out / "adaptive_walkforward_metrics.csv")
    wfa = wf[wf["basis"] == "A_per_SMLT_10m"].set_index("strategy")
    epi = pd.read_csv(ctx.out / "liquidity_waterfall_results.csv")
    wf_map = {"CONTROL": "CONTROL", "FUT_25": "FUT_25", "FUT_50": "FUT_50", "FUT_75": "FUT_75", "FUT_100": "FUT_100",
              "ROLLING_HSTAR": "ROLLING_HSTAR", "CURRENT_STATE_DEPENDENT": "CURRENT_STATE_DEPENDENT",
              "STATIC_SMART_PORTFOLIO": "FIXED_SMART_PRE_OOS", "ADAPTIVE_SMART_PORTFOLIO": "ADAPTIVE_SMART"}
    for label in tg:
        if (label, "6m") not in res:
            continue
        a, b = res[(label, "6m")], res.get((label, "12m"), res[(label, "6m")])
        f6, f12 = a["per_model"]["FHS_FULL"], b["per_model"]["FHS_FULL"]
        worst6 = max(v["ES95"] for v in a["per_model"].values())
        worst12 = max(v["ES95"] for v in b["per_model"].values())
        k = wf_map.get(label)
        e6 = epi[(epi["strategy"] == k) & (epi["horizon"] == "6m") & (epi["liquidity_policy"] == "WATERFALL")] if k else pd.DataFrame()
        row = {
            "strategy": label, "evaluation": "static_eval_of_decision_target" if label in ("ROLLING_HSTAR", "CURRENT_STATE_DEPENDENT",
                                                                                          "ADAPTIVE_SMART_PORTFOLIO") else "static",
            "note": a["note"],
            **{f"w6m_{n}": float(a["h"][j]) for j, n in enumerate(names)},
            **{f"w12m_{n}": float(b["h"][j]) for j, n in enumerate(names)},
            "ES95_6m": f6["ES95"], "ES95_12m": f12["ES95"], "ES99_6m": f6["ES99"], "ES99_12m": f12["ES99"],
            "volatility_6m_std_pnl": f6["std_pnl"], "volatility_12m_std_pnl": f12["std_pnl"],
            "max_dd_p95_6m": f6["p95_max_dd"], "max_dd_p95_12m": f12["p95_max_dd"],
            "upside_capture_6m": f6["upside_capture"], "downside_capture_6m": f6["downside_capture"],
            "upside_capture_12m": f12["upside_capture"], "downside_capture_12m": f12["downside_capture"],
            "mean_pnl_6m": f6["mean_pnl"], "mean_pnl_12m": f12["mean_pnl"],
            "hedge_cost_6m": ctrl6 - f6["mean_pnl"], "hedge_cost_12m": ctrl12 - f12["mean_pnl"],
            "turnover_static_6m_notional_multiple": sum(abs(a["h"][j]) * (2 + 2 * sp[j].n_rolls(0.5)) for j in range(len(sp))),
            "initial_margin": f6["initial_im"], "peak_collateral_p95_6m": f6["peak_collateral_p95"],
            "sleeve_breach_prob_6m": f6["p_sleeve_breach"], "sleeve_breach_prob_12m": f12["p_sleeve_breach"],
            "n_active_sleeves": f6["n_active_sleeves"],
            "worst_deterministic_hist_window_6m": a["per_model"]["HIST_WINDOWS"]["min_pnl"],
            "worst_deterministic_hist_window_12m": b["per_model"]["HIST_WINDOWS"]["min_pnl"],
            "worst_model_ES95_6m": worst6, "worst_model_ES95_12m": worst12,
            **{f"ES95_6m_{m}": v["ES95"] for m, v in a["per_model"].items()},
            "wf_HE_VAR": wfa.loc[k, "HE_VAR"] if k in wfa.index else np.nan,
            "wf_ES95_1d_pctE": wfa.loc[k, "ES95_1d"] if k in wfa.index else np.nan,
            "wf_turnover_total": wfa.loc[k, "turnover_notional_multiple"] if k in wfa.index else np.nan,
            "wf_upside_capture": wfa.loc[k, "upside_capture"] if k in wfa.index else np.nan,
            "ep6m_breach_share": float(e6["breach_share"].iloc[0]) if len(e6) else np.nan,
            "ep6m_ES95_terminal": float(e6["ES95_terminal"].iloc[0]) if len(e6) else np.nan,
        }
        rows.append(row)
    ctx.save(pd.DataFrame(rows), "smart_portfolio_comparison.csv", index=False)
    ctx.save(pd.DataFrame(contrib), "risk_contributions.csv", index=False)
