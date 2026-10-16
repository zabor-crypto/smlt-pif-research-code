"""Regime V2 pipeline stages (v1..v7). Every stage writes ONLY outputs/regime_v2r1_* and figures/fig_regime_v2r1_*; no parent
or legacy file is written. Float format is fixed for byte-level determinism."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..adaptive import events as EV
from ..adaptive import r6 as R6
from . import returns as RET
from . import risk as RK
from . import states as S2
from . import tasks as TK
from .cfg import StopCondition

FLOAT = "%.10g"
PREFIX = "regime_v2r1_"
MAIN_MANDATE = "HEDGED_INVESTED_PIF"
CP_MANDATE = "CAPITAL_PRESERVATION_UNCONSTRAINED"


def save(ctx, df: pd.DataFrame, name: str) -> Path:
    assert name.startswith(PREFIX), name
    ctx.log.setdefault("_files", []).append(name)
    p = ctx.out / name
    df.to_csv(p, index=False, float_format=FLOAT)
    return p


def save_json(ctx, obj, name: str) -> Path:
    assert name.startswith(PREFIX), name
    ctx.log.setdefault("_files", []).append(name)
    p = ctx.out / name
    p.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))
    return p


def processes() -> int:
    return int(os.environ.get("REGIME_V2R1_PROCESSES", "6"))


def _ensure(ctx) -> None:
    if not TK._G:
        TK.init(str(ctx.root))
    TK._G["ctx"].log = ctx.log
    TK._G["ctx"].out, TK._G["ctx"].fig = ctx.out, ctx.fig


# ---------------------------------------------------------------------------------------------- v1 return model and states
def v1_inputs(ctx) -> None:
    _ensure(ctx)
    M = TK.model()
    cfg = M.cfg
    t = pd.Timestamp(cfg["cutoff"])
    rc = TK.cutoff_recal()
    pri = M.priors.reset_index()
    pri["smlt_beta_to_broad_equity"] = np.where(pri["route"] == "SMLT", rc.beta["beta"], np.nan)
    pri["smlt_beta_sample"] = np.where(pri["route"] == "SMLT", f"{rc.beta['from']}..{rc.beta['to']} ({rc.beta['n']} sessions)", "")
    save(ctx, pri, "regime_v2r1_return_priors.csv")
    daily = pd.concat([M.feat, M.raw.add_prefix("raw_"), M.conf.add_prefix("confirmed_")], axis=1)
    daily.index.name = "date"
    save(ctx, daily.reset_index(), "regime_v2r1_state_daily.csv")
    start = cfg["v2"]["state"]["sample_start"]
    save(ctx, S2.episode_table(M.conf, start), "regime_v2r1_state_episodes.csv")
    states = M.conf.loc[t].to_dict()
    # forward statistics: marginal states of every axis and the joint conditions of each route's hierarchy (current state)
    rows = []
    rc_h = cfg["v2"]["returns"]
    horizons = list(rc_h["monitoring_horizons"]) + [int(rc_h["horizon_sessions"])]
    follow = rc_h["prior"]["hedge_routes_follow_exposure"]
    for r in RET.ALL_ROUTES:
        for H in horizons:
            for a in S2.AXES:
                for v in S2.ORDER[a]:
                    s = rc.fwd.stats(r, {a: v}, H)
                    rows.append({**s, "set": "MARGINAL", "is_current_state": states[a] == v})
            axes = rc_h["state_adjustment"]["axes_by_route"].get(follow.get(r, r), [])
            for k in range(2, len(axes) + 1):
                cond = {a: states[a] for a in axes[:k]}
                rows.append({**rc.fwd.stats(r, cond, H), "set": "JOINT_HIERARCHY_CURRENT", "is_current_state": True})
    fs = pd.DataFrame(rows)
    fs["evidence_class"] = "RETROSPECTIVE (one history; overlapping windows are not independent)"
    save(ctx, fs, "regime_v2r1_state_forward_stats.csv")
    # expected returns: current state and one-axis-at-a-time alternatives, every non-extreme mode
    er, unc = [], None
    for mode in ("BASE_POSTERIOR_MEAN", "HALF_SIGNAL", "ZERO_STATE_SIGNAL"):
        variants = [("CURRENT", "", "", dict(states))]
        for a in S2.AXES:
            for v in S2.ORDER[a]:
                if v == "UNKNOWN":
                    continue
                variants.append(("ONE_AXIS_CHANGED", a, v, {**states, a: v}))
        for lab, a, v, st in variants:
            mu, _, carry, adj = M.expected(rc, t, st, mode, True)
            for r in mu.index:
                e = follow.get(r, r)
                er.append({"mode": mode, "state_case": lab, "axis_changed": a, "state_value": v, **{f"state_{k}": x for k, x in st.items()},
                           "route": r, "prior_h": float(mu.loc[r, "prior"]), "carry_h": float(mu.loc[r, "carry"]),
                           "state_adj_h": float(mu.loc[r, "state_adj"]), "expected_excess_h": float(mu.loc[r, "mu"]),
                           "state_level_used": adj[e]["level"], "horizon_sessions": int(rc_h["horizon_sessions"])})
            if mode == "BASE_POSTERIOR_MEAN" and lab == "CURRENT":
                unc = RET.uncertainty_table(cfg, M.priors, mu, adj, rc.beta, rc.panel)
                info = carry["info"]
    save(ctx, pd.DataFrame(er), "regime_v2r1_expected_returns_by_state.csv")
    save(ctx, unc, "regime_v2r1_return_uncertainty.csv")
    ctx.log["v1"] = {"confirmed_state_at_cutoff": states, "raw_state_at_cutoff": M.raw.loc[t].to_dict(),
                     "event_state_at_cutoff": EV.state_at(M.events, t), "carry_inputs_at_cutoff": info,
                     "smlt_beta": rc.beta, "n_prior_routes_with_own_history": int(M.priors["status"].str.startswith("OWN").sum())}


# ---------------------------------------------------------------------------------------------- v2 joint risk diagnostics
def v2_dependence(ctx) -> None:
    _ensure(ctx)
    M = TK.model()
    cfg = M.cfg
    t = pd.Timestamp(cfg["cutoff"])
    rc = TK.cutoff_recal()
    states = M.conf.loc[t].to_dict()
    choice = RK.state_model_choice(rc.panel.index, M.conf.loc[:t], states, cfg)
    X = RK.excess_panel(M.P, list(cfg["r3"]["universes"]["MAIN"]), set(RET.FUNDED), cfg["v2"]["state"]["sample_start"], t)
    dep = RK.dependence(X, M.conf.loc[:t], states, choice, cfg, P=M.P)
    save(ctx, dep["covariance"], "regime_v2r1_state_covariance.csv")
    save(ctx, dep["correlation"], "regime_v2r1_state_correlation.csv")
    save(ctx, dep["tail"], "regime_v2r1_tail_dependence.csv")
    sel = pd.DataFrame(choice["tried"])
    sel["selected"] = sel["level"] == choice["level"]
    sel["model_available"] = choice["available"]
    save(ctx, sel, "regime_v2r1_risk_model_selection.csv")
    ctx.log["v2"] = {"state_model_available": choice["available"], "state_model_level": choice["level"],
                     "n_state_sets": int(dep["covariance"]["state"].nunique())}


# ---------------------------------------------------------------------------------------------- v3 cutoff frontiers, decision
def _lab(variant, mandate=MAIN_MANDATE, floor=None, budget=0.15, mode="BASE_POSTERIOR_MEAN", universe="MAIN") -> dict:
    return {"variant": variant, "mandate": mandate, "funded_floor_override": floor, "risk_budget_nav0": budget, "return_mode": mode,
            "universe": universe}


def cutoff_tasks(cfg: dict) -> list:
    v2 = cfg["v2"]
    b0 = float(v2["frontier"]["budgets"]["primary"])
    F = ["frontier"]
    full = ["frontier", "target", "zero", "risk", "det", "mu"]
    T = []

    def add(variant, want, mandate=MAIN_MANDATE, floor=None, budget=b0, mode="BASE_POSTERIOR_MEAN", universe="MAIN", kind="ADAPTIVE", **kw):
        ps = {"name": variant, "kind": kind, "mandate": mandate, "floor": floor, "budget": budget, "mode": mode, "universe": universe,
              **{k: v for k, v in kw.items() if k in ("keep_frac", "distressed_cap")}}
        T.append({"label": _lab(variant, mandate, floor, budget, mode, universe), "ps": ps, "want": want,
                  **{k: v for k, v in kw.items() if k in ("capital", "mc_offsets")}})
    add("HEDGED_INVESTED_PIF", full + ["hedge", "mc"], mc_offsets=list(cfg["r6"]["validation"]["monte_carlo_seed_offsets"]))
    add("CAPITAL_PRESERVATION_UNCONSTRAINED", full, mandate=CP_MANDATE)
    for b in v2["frontier"]["budgets"]["sensitivity"]:
        add(f"HEDGED_INVESTED_PIF|ES{int(round(b * 100))}", F + ["target"], budget=float(b))
        add(f"CAPITAL_PRESERVATION_UNCONSTRAINED|ES{int(round(b * 100))}", F, mandate=CP_MANDATE, budget=float(b))
    for f in v2["mandates"][MAIN_MANDATE]["funded_floor_sensitivity"]:
        add(f"HEDGED_INVESTED_PIF|FLOOR{int(round(f * 100))}", F + ["target"], floor=float(f))
    for m in ("HALF_SIGNAL", "ZERO_STATE_SIGNAL"):
        add(f"HEDGED_INVESTED_PIF|{m}", F + ["target"], mode=m)
    add("CAPITAL_PRESERVATION_UNCONSTRAINED|EXTREME_ROBUST", F, mandate=CP_MANDATE, mode="EXTREME_ROBUST")
    add("HEDGED_INVESTED_PIF|WITHOUT_BTC", F + ["risk"], universe="WITHOUT_BTC")
    add("HEDGED_INVESTED_PIF|BROAD_NET_SHORT_ALLOWED", F, mandate="BROAD_NET_SHORT_ALLOWED")
    for c in v2["issuer_cap_sensitivity_distressed"]:
        add(f"HEDGED_INVESTED_PIF|DISTRESSED_CAP{int(round(c * 100))}", F, distressed_cap=float(c))
    add("STATIC_INVESTED_PIF_V2R1", F + ["target", "risk"], kind="STATIC")
    add("HEDGE_KEEP_SMLT_50", F + ["target"], kind="HEDGE_KEEP", universe="HEDGE_ONLY", keep_frac=1.0)
    add("HEDGE_KEEP_SMLT_25", F + ["target"], kind="HEDGE_KEEP", universe="HEDGE_ONLY", keep_frac=0.5)
    main_cap = float(cfg["owner"]["capital_main_rub"])
    for cap in cfg["owner"]["capital_grid_rub"]:
        if float(cap) != main_cap:
            add(f"HEDGED_INVESTED_PIF|CAPITAL{int(cap)}", F, capital=float(cap))
    return T


def v3_cutoff(ctx) -> None:
    _ensure(ctx)
    M = TK.model()
    cfg = M.cfg
    tasks = cutoff_tasks(cfg)
    res = TK.run_pool(str(ctx.root), TK.cutoff_task, tasks, processes())
    by = {r["label"]["variant"]: r for r in res}
    fr = pd.DataFrame([x for r in res for x in r.get("frontier", [])])
    for b in (cfg["v2"]["frontier"]["budgets"]["primary"], *cfg["v2"]["frontier"]["budgets"]["sensitivity"]):
        save(ctx, fr[np.isclose(fr["risk_budget_nav0"], b)], f"regime_v2r1_frontier_{int(round(b * 100))}.csv")
    ref = pd.DataFrame([x for r in res for x in r.get("refinement", [])])
    save(ctx, ref if len(ref) else pd.DataFrame([{"note": "no PASS/FAIL bracket occurred"}]), "regime_v2r1_frontier_refinement.csv")
    summ = pd.DataFrame([r["summary"] for r in res])
    save(ctx, summ, "regime_v2r1_cutoff_decisions.csv")
    main, cp = by["HEDGED_INVESTED_PIF"], by["CAPITAL_PRESERVATION_UNCONSTRAINED"]
    save(ctx, pd.DataFrame(main["target"] + cp["target"]), "regime_v2r1_current_target.csv")
    save(ctx, pd.DataFrame(main["orders"] + cp["orders"]), "regime_v2r1_current_orders.csv")
    st = by["STATIC_INVESTED_PIF_V2R1"]
    save(ctx, pd.DataFrame(st["target"]), "regime_v2r1_static_invested_target.csv")
    other = [r for r in res if "target" in r and r["label"]["variant"] not in ("HEDGED_INVESTED_PIF", "CAPITAL_PRESERVATION_UNCONSTRAINED", "STATIC_INVESTED_PIF_V2R1")]
    save(ctx, pd.DataFrame([x for r in other for x in r["target"]]), "regime_v2r1_sensitivity_targets.csv")
    save(ctx, pd.DataFrame(main["zero"] + cp["zero"]), "regime_v2r1_zero_weight_attribution.csv")
    save(ctx, pd.DataFrame(main["zero"] + cp["zero"]), "regime_v2r1_zero_weight_marginal_entry.csv")
    save(ctx, pd.DataFrame(main["capital_flows"] + cp["capital_flows"]), "regime_v2r1_capital_flows.csv")
    save(ctx, pd.DataFrame([x for r in res for x in r.get("risk", [])]), "regime_v2r1_risk_contribution.csv")
    save(ctx, pd.DataFrame(main["det"] + cp["det"]), "regime_v2r1_deterministic_stress.csv")
    save(ctx, pd.DataFrame(main["hedge"]), "regime_v2r1_hedge_ratios.csv")
    mc = pd.DataFrame(main["mc"])
    g = mc.groupby(["book", "model"])["ES95_nav0"].agg(["mean", "std", "min", "max"]).reset_index()
    g.insert(0, "row", "MONTE_CARLO_ERROR_ACROSS_SEEDS")
    ms = g.groupby("book")["mean"].agg(["min", "max"]).reset_index()
    ms.insert(0, "row", "MODEL_SPREAD_ONE_HISTORY")
    mc.insert(0, "row", "DRAW")
    save(ctx, pd.concat([mc, g, ms], ignore_index=True), "regime_v2r1_monte_carlo_error.csv")
    cg = summ[summ["variant"].str.contains("CAPITAL") & summ["variant"].str.startswith("HEDGED") | (summ["variant"] == "HEDGED_INVESTED_PIF")].copy()
    cg.insert(0, "capital_rub", [float(v.split("CAPITAL")[1]) if "|CAPITAL" in v else float(cfg["owner"]["capital_main_rub"]) for v in cg["variant"]])
    save(ctx, cg.sort_values("capital_rub"), "regime_v2r1_capital_grid.csv")
    save(ctx, pd.DataFrame(main["mu"]), "regime_v2r1_current_expected_returns.csv")
    save(ctx, pd.DataFrame(main["adj"]), "regime_v2r1_current_state_adjustments.csv")
    t = pd.Timestamp(cfg["cutoff"])
    obj = {"DATE": str(t.date()),
           "STATE_VECTOR": {"raw": M.raw.loc[t].to_dict(), "confirmed": M.conf.loc[t].to_dict(),
                            "evidence": {k: (None if pd.isna(v) else (v if isinstance(v, str) else float(v))) for k, v in M.feat.loc[t].to_dict().items()}},
           "EVENT_STATE": EV.state_at(M.events, t), "RISK_MODELS": main["state_model"], "CARRY_INPUTS": main["carry_info"],
           "EXPECTED_EXCESS_RETURN_VECTOR_126_SESSIONS": {r["route"]: {k: r[k] for k in ("prior", "carry", "state_adj", "mu")} for r in main["mu"]},
           "A_CAPITAL_PRESERVATION_UNCONSTRAINED": cp["summary"], "B_HEDGED_INVESTED_PIF_MAIN_CASE": main["summary"],
           "HEADLINE": "B_HEDGED_INVESTED_PIF_MAIN_CASE",
           "SENSITIVITIES": {r["label"]["variant"]: {k: r["summary"].get(k) for k in (
               "action", "flags", "expected_net_target_nav0", "es95_target_nav0", "funded_non_mmf_target", "smlt_net_issuer_target",
               "released_smlt_capital_goes_to") + tuple(c for c in r["summary"] if c.startswith("w_tgt_"))} for r in res},
           "AUTHORITY": "full recursive engine (R1); the LP only generates candidates",
           "EVIDENCE_CLASS": "SIMULATION_HOLDOUT (independent seeds of one 2022-2026 history); NEW_MARKET_OOS = NONE"}
    save_json(ctx, obj, "regime_v2r1_current_decision.json")
    ctx.log["v3"] = {"tasks": len(tasks), "main_action": main["summary"]["action"], "main_flags": main["summary"]["flags"],
                     "cp_action": cp["summary"]["action"]}
    primary = main["summary"]
    primary_floor = float(cfg["v2"]["mandates"][MAIN_MANDATE]["funded_floor_nav"])
    if primary["mandate_status"] == "EMERGENCY_INVESTED_FLOOR_OVERRIDE" or primary["funded_non_mmf_target"] < primary_floor - 1e-12:
        # Preserve the separately labelled emergency result, then stop before scenarios/replay.
        raise StopCondition("INVESTED_MANDATE_INFEASIBLE_AT_15_PERCENT: no admitted ordinary primary target; emergency result retained separately")


# ---------------------------------------------------------------------------------------------- v4 state library, scenarios
def v4_library(ctx) -> None:
    import itertools
    _ensure(ctx)
    M = TK.model()
    cfg = M.cfg
    t = pd.Timestamp(cfg["cutoff"])
    actual = M.conf.loc[t].to_dict()
    sl = cfg["v2"]["scenarios"]["state_library"]
    tasks = []
    for iss, mk, rt, rb in itertools.product(sl["issuer_states"], sl["market_states"], sl["rates_states"], sl["rub_states"]):
        sev = iss == "SEVERE_VERIFIED_EVENT"
        st = {"ISSUER_CREDIT_RISK": "DISTRESSED" if sev else iss, "MARKET_RISK": mk, "RATES_STATE": rt, "RUB_STATE": rb}
        tasks.append({"label": {"issuer_state": iss, "market_state": mk, "rates_state": rt, "rub_state": rb,
                                "is_actual_confirmed_state_at_cutoff": (not sev) and st == actual, "hypothetical_state": sev or st != actual},
                      "ps": {"name": "ADAPTIVE_REGIME_PIF_V2R1", "kind": "ADAPTIVE"}, "states": st, "severe": sev, "paths": "library", "want": []})
    res = TK.run_pool(str(ctx.root), TK.cutoff_task, tasks, processes())
    lib = pd.DataFrame([r["summary"] for r in res])
    lib["evidence_class"] = np.where(lib["hypothetical_state"], "SCENARIO_ONLY", "SIMULATION_HOLDOUT")
    save(ctx, lib, "regime_v2r1_state_target_library.csv")
    sc = cfg["v2"]["scenarios"]["library"]
    tasks = [{"label": {"scenario": name, "deterministic_path": s["path"]}, "ps": {"name": "ADAPTIVE_REGIME_PIF_V2R1", "kind": "ADAPTIVE"},
              "states": s["state"], "severe": bool(s.get("severe", False)), "paths": "cutoff", "want": ["det", "risk"]} for name, s in sc.items()]
    res = TK.run_pool(str(ctx.root), TK.cutoff_task, tasks, processes())
    rows, risk = [], []
    for r in res:
        s = r["summary"]
        det = pd.DataFrame(r["det"])
        det1 = det[det["margin_multiplier"] == 1.0]
        own = det1[det1["path"] == r["label"]["deterministic_path"]].set_index("book")
        book = "TARGET" if "TARGET" in own.index else "CURRENT_BOOK_NO_TRADE"
        tgt_all = det1[det1["book"] == book]
        worst = tgt_all.loc[tgt_all["pnl_nav0"].idxmin()]
        cur, tg = own.loc["CURRENT_BOOK_NO_TRADE"], own.loc[book]
        unres = bool(det[det["book"] == book]["unresolved_deficit"].any())
        shares, fut = s.get("w_tgt_SMLT", 0.0) or 0.0, s.get("w_tgt_SMLT_FUT", 0.0) or 0.0
        eq_long = (s.get("w_tgt_EQ_FUNDED", 0.0) or 0.0) + shares + fut
        rows.append({**{k: v for k, v in s.items() if not k.startswith("w_cur_")},
                     "median_outcome_excess_nav0_worst_model": s.get("risk_path_median_excess_worst_model"),
                     "own_path_pnl_target_nav0": tg["pnl_nav0"],
                     "own_path_excess_target_nav0": tg["excess_over_money_market_nav0"], "own_path_pnl_current_book_nav0": cur["pnl_nav0"],
                     "own_path_max_drawdown_target": tg["max_drawdown"], "own_path_max_drawdown_current_book": cur["max_drawdown"],
                     "own_path_min_available_nav_target": tg["min_available_nav"], "own_path_breach_days_target": tg["breach_days"],
                     "loss_reduction_vs_current_book_nav0": tg["pnl_nav0"] - cur["pnl_nav0"],
                     "hedge_or_rotation_works": bool(tg["pnl_nav0"] >= cur["pnl_nav0"]),
                     "worst_deterministic_path": worst["path"], "worst_deterministic_pnl_nav0": worst["pnl_nav0"],
                     "worst_deterministic_max_drawdown": float(tgt_all["max_drawdown"].max()),
                     "nav_survives_all_deterministic_paths": bool(tgt_all["min_nav_nav0"].min() > 0 and not unres),
                     "liquidity_survives_all_deterministic_paths": bool(not unres),
                     "liquidity_breach_days_worst_path": int(tgt_all["breach_days"].max()),
                     "smlt_futures_hedge_ratio_of_shares": (-fut / shares) if shares > 1e-9 else None,
                     "equity_futures_hedge_ratio_of_funded_equity": (-(s.get("w_tgt_EQ_HEDGE_FUT", 0.0) or 0.0) / eq_long) if eq_long > 1e-9 else None,
                     "probability": "NOT_ASSIGNED", "evidence_class": "SCENARIO_ONLY"})
        risk.extend(r["risk"])
    save(ctx, pd.DataFrame(rows), "regime_v2r1_scenario_scorecard.csv")
    save(ctx, pd.DataFrame(risk), "regime_v2r1_scenario_risk_contribution.csv")
    act = lib[lib["is_actual_confirmed_state_at_cutoff"]]
    ctx.log["v4"] = {"library_rows": int(len(lib)), "distinct_actions": sorted(lib["action"].unique().tolist()),
                     "action_in_actual_state": act["action"].tolist(), "scenarios": int(len(rows))}


# ---------------------------------------------------------------------------------------------- v5 replay
def verdict(base: dict, sens: list, constraints_ok: bool) -> str:
    """Pre-declared rule (configs/regime_v2.yaml, validation.incremental_value)."""
    if base.get("identical_ledgers"):
        return "ADAPTIVE_INCREMENTAL_VALUE_NOT_ESTABLISHED (the adaptive and static ledgers are identical)"
    if base["ci90_hi"] <= 0:
        return "ADAPTIVE_INCREMENTAL_VALUE_NOT_ESTABLISHED (the 90% block interval of the mean difference lies at or below zero)"
    ok = (base["sign_share_positive"] >= 0.8 and base["ci90_lo"] > 0 and (base["max_single_block_share_of_gain"] or 1.0) <= 0.5
          and all(s["mean_block_difference"] > 0 and s["ci90_lo"] > 0 for s in sens) and constraints_ok)
    if ok:
        return "ADAPTIVE_INCREMENTAL_VALUE_ESTABLISHED"
    if base["ci90_lo"] <= 0:
        return "ADAPTIVE_INCREMENTAL_VALUE_INCONCLUSIVE (the 90% block interval includes zero)"
    return "ADAPTIVE_INCREMENTAL_VALUE_INCONCLUSIVE (the interval is above zero but a robustness condition fails)"


def v5_replay(ctx) -> None:
    _ensure(ctx)
    M = TK.model()
    cfg = M.cfg
    rp = cfg["v2"]["replay"]
    specs = [dict(s) for s in rp["ledgers"]]
    res = TK.run_pool(str(ctx.root), TK.replay_task, specs, processes())
    by = {r["name"]: r for r in res}
    save(ctx,pd.DataFrame([row for r in res for row in r["capital_flows"]]),"regime_v2r1_replay_capital_flows.csv")
    nav0 = float(rp["nav0_rub"])
    dates, mmf = res[0]["dates"], res[0]["mmf"]
    ledgers = {r["name"]: r["ledger"] for r in res}
    omm = by["OPERATIONAL_MONEY_MARKET"]["metrics"]
    met = pd.DataFrame([{**{k: v for k, v in r["spec"].items() if k not in ("name",)}, **r["metrics"]} for r in res])
    g = mmf / float(mmf.iloc[0])
    rr = g.pct_change().dropna()
    yrs = len(g) / 252
    ref = {"comparator": rp["reference"], "type": "reference", "start": str(dates[0].date()), "end": str(dates[-1].date()), "sessions": len(g),
           "terminal_nav": float(nav0 * g.iloc[-1]), "total_return": float(g.iloc[-1] - 1), "money_market_total_return": float(g.iloc[-1] - 1),
           "net_excess_over_money_market": 0.0, "cagr": float(g.iloc[-1] ** (1 / yrs) - 1), "vol_ann": float(rr.std(ddof=1) * np.sqrt(252)),
           "max_drawdown": float(((g.cummax() - g) / g.cummax()).max()), "worst_1d": float(rr.min()),
           "evidence_class": "REFERENCE (NAV0 in LQDT units from the start; no costs, no liquidity reserve): opportunity-cost benchmark only"}
    met = pd.concat([met, pd.DataFrame([ref])], ignore_index=True)
    met["net_excess_over_operational_money_market"] = met["total_return"] - omm["total_return"]
    save(ctx, met, "regime_v2r1_walkforward.csv")
    att = pd.DataFrame([r["attribution"] for r in res])
    save(ctx, att, "regime_v2r1_walkforward_attribution.csv")
    dec = pd.concat([r["decisions"] for r in res if len(r["decisions"])], ignore_index=True)
    dec["date"] = pd.to_datetime(dec["date"]).dt.date.astype(str)
    save(ctx, dec, "regime_v2r1_decision_log.csv")
    rb = pd.concat([R6.rebalance_log(r["name"], r["ledger"], r["decisions"]) for r in res if r["spec"].get("type") == "policy"], ignore_index=True)
    save(ctx, rb, "regime_v2r1_rebalance_log.csv")
    refs = [r["refinements"].assign(date=lambda d: pd.to_datetime(d["date"]).dt.date.astype(str)) for r in res if len(r["refinements"])]
    save(ctx, pd.concat(refs, ignore_index=True) if refs else pd.DataFrame([{"note": "no PASS/FAIL bracket occurred"}]),
         "regime_v2r1_replay_refinement.csv")
    main = [s["name"] for s in specs if "|" not in s["name"]]
    navp = pd.DataFrame({n: ledgers[n]["nav"] for n in main})
    navp[rp["reference"]] = (nav0 * g).values
    navp.index.name = "date"
    for n in ("ADAPTIVE_REGIME_PIF_V2R1|HALF_SIGNAL", "ADAPTIVE_REGIME_PIF_V2R1|ZERO_STATE_SIGNAL", "STATIC_INVESTED_PIF_V2R1|HALF_SIGNAL"):
        if n in ledgers:
            navp[n] = ledgers[n]["nav"]
    save(ctx, navp.reset_index(), "regime_v2r1_nav_paths.csv")
    from . import replay as RP2
    alloc, liq = [], []
    for r in res:
        if r["spec"].get("type") != "policy":
            continue
        a = RP2.sleeve_path(cfg, r["ledger"])
        a.insert(0, "comparator", r["name"])
        a.index.name = "date"
        alloc.append(a.reset_index())
        lg = r["ledger"]
        q = pd.DataFrame({"comparator": r["name"], "nav": lg["nav"], "immediate_cash": lg["cash"], "mmf_value": lg.get("v_MMF", 0.0),
                          "pending_settlement_net": lg["pending"], "posted_initial_margin": lg["im"], "available_immediate": lg["available"],
                          "available_nav": lg["available"] / lg["nav"], "breach": lg["breach"]})
        q.index.name = "date"
        liq.append(q.reset_index())
    save(ctx, pd.concat(alloc, ignore_index=True), "regime_v2r1_allocation_path.csv")
    save(ctx, pd.concat(liq, ignore_index=True), "regime_v2r1_liquidity_path.csv")
    # incremental value of adaptive management (pre-declared rule)
    iv = cfg["v2"]["validation"]["incremental_value"]
    months = cfg["v2"]["policy"]["recalibration_months"]
    A, B = iv["pair"]
    rows, blocks, stats = [], [], {}
    for var, b_var in (("", ""), ("|delay2", "|delay2"), ("|cost_x2", "|cost_x2"), ("|HALF_SIGNAL", "|HALF_SIGNAL"), ("|WITHOUT_BTC", "|WITHOUT_BTC"),
                       ("|ZERO_STATE_SIGNAL", ""), ("|BROAD_NET_SHORT_ALLOWED", "")):
        a, b = A + var, B + b_var
        if a not in ledgers or b not in ledgers:
            continue
        bl = R6.blocks(ledgers, mmf, a, b, months)
        st = R6.incremental(bl, iv["bootstrap_seed"])
        stats[var] = st
        ma, mb = by[a]["metrics"], by[b]["metrics"]
        rows.append({"variant": var.strip("|") or "BASE", "adaptive": a, "static": b, "net_excess_adaptive": ma["net_excess_over_money_market"],
                     "net_excess_static": mb["net_excess_over_money_market"], "difference": ma["net_excess_over_money_market"] - mb["net_excess_over_money_market"],
                     "max_drawdown_adaptive": ma["max_drawdown"], "max_drawdown_static": mb["max_drawdown"],
                     "breach_days_adaptive": ma["liquidity_breach_days"], "unresolved_adaptive": ma["unresolved_deficit_days"],
                     "costs_adaptive": ma["costs_rub"], "turnover_adaptive": ma["turnover_nav0"], "turnover_static": mb["turnover_nav0"], **st})
        for _, r_ in bl.iterrows():
            blocks.append({"variant": var.strip("|") or "BASE", **{("excess_adaptive" if c == f"excess_{a}" else "excess_static" if c == f"excess_{b}" else c): v
                                                                   for c, v in r_.items()}})
    if "" in stats:
        ok = by[A]["metrics"]["unresolved_deficit_days"] == 0
        v = verdict(stats[""], [stats[x] for x in ("|delay2", "|cost_x2") if x in stats], ok)
        rows.append({"variant": "VERDICT", "adaptive": A, "static": B, "verdict": v, "n_blocks": stats[""]["n_blocks"],
                     "power_note": "semi-annual blocks of one 2023-2026 history: the test can only detect a large and regular difference"})
    save(ctx, pd.DataFrame(rows), "regime_v2r1_incremental_value.csv")
    save(ctx, pd.DataFrame(blocks), "regime_v2r1_incremental_value_blocks.csv")
    ctx.log["v5"] = {"ledgers": len(res), "verdict": rows[-1].get("verdict") if rows else None,
                     "identity_residual_max_abs": float(att["identity_residual"].abs().max()),
                     "decisions_hold_on_failed_book": int(((dec["action"] == "HOLD") & (~dec["current_admitted"].astype(bool))).sum())}


# ---------------------------------------------------------------------------------------------- v6 BTC, evidence ladder
def v6_summary(ctx) -> None:
    _ensure(ctx)
    M = TK.model()
    cfg = M.cfg
    out = ctx.out
    rows = []
    f15 = pd.read_csv(out / "regime_v2r1_frontier_15.csv")
    rk = pd.read_csv(out / "regime_v2r1_risk_contribution.csv")
    for var in ("HEDGED_INVESTED_PIF", "HEDGED_INVESTED_PIF|WITHOUT_BTC", "CAPITAL_PRESERVATION_UNCONSTRAINED"):
        g = f15[(f15["variant"] == var) & (f15["ADMISSION"] == "PASS")]
        sel = f15[(f15["variant"] == var) & f15["selected_target"].astype(bool)]
        col = "w_exec_BTC_RUB_EXPERIMENTAL"
        rows.append({"section": "FRONTIER_15", "variant": var, "n_admitted": int(len(g)),
                     "btc_weight_selected": float(np.nan_to_num(sel[col].iloc[0])) if len(sel) and col in sel else 0.0,
                     "max_btc_weight_on_admitted_frontier": float(np.nan_to_num(g[col].max())) if len(g) and col in g else 0.0,
                     "expected_net_excess_selected_nav0": float(sel["expected_net_excess_nav0"].iloc[0]) if len(sel) else None,
                     "engine_ES95_selected_nav0": float(sel["engine_ES95_worst_nav0"].iloc[0]) if len(sel) else None})
    b = rk[(rk["variant"] == "HEDGED_INVESTED_PIF") & (rk["book"] == "TARGET") & (rk["sleeve"] == "BTC_RUB_EXPERIMENTAL")]
    w = float(b["weight_nav0"].iloc[0]) if len(b) else 0.0
    ec = float(b["es_contribution_nav0"].iloc[0]) if len(b) else 0.0
    same = rows[0]["expected_net_excess_selected_nav0"] == rows[1]["expected_net_excess_selected_nav0"]
    label = ("NO_INCREMENTAL_VALUE" if abs(w) < 1e-9 and (rows[0]["max_btc_weight_on_admitted_frontier"] < 1e-9 or same)
             else ("DIVERSIFIER" if ec <= 0 else "RETURN_SEEKING_RISK_ASSET") if abs(w) >= 1e-9 else "NOT_SELECTED_BUT_ON_THE_ADMITTED_FRONTIER")
    rows.append({"section": "LABEL", "variant": "HEDGED_INVESTED_PIF", "label": label + " / EXPERIMENTAL", "btc_weight_selected": w,
                 "btc_es95_euler_contribution_nav0": ec, "hedge_claim": "NOT_MADE (criterion of the brief not met unless stated in section FORWARD)"})
    wf_p = out / "regime_v2r1_walkforward.csv"
    if wf_p.exists():
        wf = pd.read_csv(wf_p)
        for nm in ("ADAPTIVE_REGIME_PIF_V2R1", "ADAPTIVE_REGIME_PIF_V2R1|WITHOUT_BTC", "STATIC_INVESTED_PIF_V2R1", "STATIC_INVESTED_PIF_V2R1|WITHOUT_BTC"):
            r = wf[wf["comparator"] == nm]
            if len(r):
                r = r.iloc[0]
                rows.append({"section": "REPLAY", "variant": nm, "net_excess_over_money_market": r["net_excess_over_money_market"],
                             "max_drawdown": r["max_drawdown"], "btc_exposure_avg": r["btc_exposure_avg"], "terminal_nav": r["terminal_nav"]})
    R = M.P["R"].loc[TK.replay_start(cfg):]
    x = R["BTC_RUB_EXPERIMENTAL"].dropna()
    lvl = (1 + x).cumprod()
    for k, v in (("sessions_in_replay", len(x)), ("vol_ann", x.std(ddof=1) * np.sqrt(252)), ("worst_session", x.min()),
                 ("max_drawdown", ((lvl.cummax() - lvl) / lvl.cummax()).max())):
        rows.append({"section": "STATS", "stat": k, "value": float(v)})
    wk = (1 + R).resample("W-FRI").prod() - 1
    dn = R["SMLT"] < R["SMLT"].quantile(0.10)
    for c in ("SMLT", "EQ_FUNDED", "OFZ_FUNDED", "GOLD_FUNDED", "CNY_PERP", "OIL_FUT"):
        rows.append({"section": "CORRELATION", "stat": c, "corr_daily": float(R["BTC_RUB_EXPERIMENTAL"].corr(R[c])),
                     "corr_weekly": float(wk["BTC_RUB_EXPERIMENTAL"].corr(wk[c])),
                     "mean_btc_return_on_worst_decile_SMLT_days": float(R.loc[dn, "BTC_RUB_EXPERIMENTAL"].mean()) if c == "SMLT" else None})
    fs = pd.read_csv(out / "regime_v2r1_state_forward_stats.csv")
    g = fs[(fs["route"] == "BTC_RUB_EXPERIMENTAL") & (fs["horizon"] == 126) & (fs["set"] == "MARGINAL")]
    for _, r in g.iterrows():
        rows.append({"section": "FORWARD", "stat": r["condition"], "value": r.get("mean_fwd_excess"), "differential": r.get("differential"),
                     "n_episodes": r.get("n_episodes"), "n_nonoverlapping_windows": r.get("n_nonoverlapping_windows"), "admissible": r.get("admissible")})
    pr = pd.read_csv(out / "regime_v2r1_return_priors.csv")
    p = pr[pr["route"] == "BTC_RUB_EXPERIMENTAL"].iloc[0]
    rows.append({"section": "PRIOR", "stat": "prior-era windows / median / SE / reliability / prior", "value_text":
                 f"{p['n_windows']} / {p['location_median']:.4f} / {p['se_bootstrap_median']:.4f} / {p['reliability']:.4f} / {p['prior']:.4f}"})
    save(ctx, pd.DataFrame(rows), "regime_v2r1_btc_incremental.csv")
    ladder = [
        ("IDENTITY_PROOF", "engine accounting identities, integer contracts and settlement (parent R1), re-checked on every V2 ledger", "regime_v2r1_walkforward_attribution.csv, regime_v2r1_rebalance_log.csv"),
        ("IDENTITY_PROOF", "every published target has a full-engine PASS; no inadmissible current book returned HOLD", "regime_v2r1_frontier_*.csv, regime_v2r1_decision_log.csv"),
        ("IDENTITY_PROOF", "state at t does not depend on data after t; a state-conditional block never starts with the state-defining day's return", "tests/test_regime_v2r1_*.py"),
        ("SIMULATION_HOLDOUT", "frontier admission (ES95, liquidity, survival) on independent-seed paths of the SAME 2022-2026 history, four joint models", "regime_v2r1_frontier_{10,15,20}.csv"),
        ("SIMULATION_HOLDOUT", "Monte-Carlo error of ES95 across seed draws, reported apart from the spread across models", "regime_v2r1_monte_carlo_error.csv"),
        ("RETROSPECTIVE_WALK_FORWARD", "one-ledger replay 2023-01-03..2026-09-25 of every policy; one history, one sustained SMLT decline", "regime_v2r1_walkforward.csv"),
        ("RETROSPECTIVE_WALK_FORWARD", "adaptive vs static incremental value by semi-annual blocks (ESTABLISHED / INCONCLUSIVE / NOT_ESTABLISHED)", "regime_v2r1_incremental_value.csv"),
        ("RETROSPECTIVE_WALK_FORWARD", "forward-after-state statistics with distinct-episode and non-overlapping-window counts", "regime_v2r1_state_forward_stats.csv"),
        ("RETROSPECTIVE_WALK_FORWARD", "strategic priors from the prior era 2013-2021 (BTC 2017-2021): a different period, not an out-of-sample test of the policy", "regime_v2r1_return_priors.csv"),
        ("SCENARIO_ONLY", "deterministic path stresses and the regime scorecard (no probabilities)", "regime_v2r1_scenario_scorecard.csv, regime_v2r1_deterministic_stress.csv"),
        ("SCENARIO_ONLY", "policy library for states other than the confirmed state at the cutoff (incl. the hypothetical verified severe event)", "regime_v2r1_state_target_library.csv"),
        ("SCENARIO_ONLY", "HALF_SIGNAL / ZERO_STATE_SIGNAL / EXTREME_ROBUST, floor 40/60, budget 10/20, ceiling and net-short sensitivities", "regime_v2r1_frontier_*.csv, regime_v2r1_cutoff_decisions.csv"),
        ("NEW_MARKET_OOS", "NONE: the cutoff is the last data date; no claim rests on untouched market data", ""),
    ]
    save(ctx, pd.DataFrame(ladder, columns=["evidence_class", "claim", "outputs"]), "regime_v2r1_validation_evidence_ladder.csv")
    ctx.log["v6"] = {"btc_label": label}


def v7_figures(ctx) -> None:
    from . import figures as FG
    made = []
    for f in FG.ALL:
        f(ctx)
        made.append(f.__name__)
    ctx.log["v7"] = {"figures": made}


STAGES = {"v1": v1_inputs, "v2": v2_dependence, "v3": v3_cutoff, "v4": v4_library, "v5": v5_replay, "v6": v6_summary, "v7": v7_figures}
