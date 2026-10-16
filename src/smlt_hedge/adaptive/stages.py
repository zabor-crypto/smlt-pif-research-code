"""Adaptive redesign pipeline stages (a1..a9). Every stage writes ONLY outputs/adaptive_* and figures/fig_adaptive_*;
no V1-V4.1 file is read for writing. Float format is fixed for byte-level determinism."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

FLOAT = "%.10g"
# two LEGACY V2 outputs share the prefix and must never be written or removed by the redesign
LEGACY_ADAPTIVE_NAMES = {"adaptive_episodes.csv", "adaptive_walkforward_metrics.csv"}


def save(ctx, df: pd.DataFrame, name: str, index: bool = False) -> Path:
    assert name.startswith("adaptive_") and name not in LEGACY_ADAPTIVE_NAMES, name
    ctx.log.setdefault("_files", []).append(name)
    p = ctx.out / name
    df.to_csv(p, index=index, float_format=FLOAT)
    return p


def save_json(ctx, obj, name: str) -> Path:
    assert name.startswith("adaptive_") and name not in LEGACY_ADAPTIVE_NAMES, name
    ctx.log.setdefault("_files", []).append(name)
    p = ctx.out / name
    p.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))
    return p


def a1_accounting(ctx) -> None:
    from . import r1
    cfg = r1.load_cfg(ctx)
    h = r1.historical_ledgers(ctx, cfg)
    ident = r1.identity_table(h["ledgers"])
    save(ctx, ident, "adaptive_accounting_identity.csv")
    save(ctx, r1.liquidity_bridge(h["ledgers"]), "adaptive_liquidity_bridge.csv")
    save(ctx, r1.margin_audit(ctx, cfg, h["dates"], h["usd"]), "adaptive_margin_audit.csv")
    adm = r1.v41_readmission(ctx, cfg)
    save(ctx, adm, "adaptive_candidate_admission.csv")
    save(ctx, r1.integer_audit(adm, ident), "adaptive_integer_execution_audit.csv")
    ctx.log["a1"] = {"identity_status": ident["status"].tolist(),
                     "v41_admission": {f"{c}@{s}": v for (c, s), v in
                                       adm.groupby(["candidate", "evaluated_scenario"])["ADMISSION"].first().items()}}


def a2_instruments(ctx) -> None:
    from . import r1, r2
    from . import routes as RT
    cfg = r1.load_cfg(ctx)
    P = RT.build_panel(ctx, cfg)
    rules = r2.route_rule_table(ctx, cfg, P)
    recon = RT.reconcile(P, cfg)
    usd_tom = RT.adaptive_frames(ctx.root, "currency_history", "USD000UTSTOM")
    tom = None
    if not usd_tom.empty:
        usd_tom["TRADEDATE"] = pd.to_datetime(usd_tom["TRADEDATE"])
        usd_tom = usd_tom[(usd_tom["TRADEDATE"].dt.dayofweek < 5) & (usd_tom["NUMTRADES"].fillna(0) > 0)
                          & (usd_tom["CLOSE"].fillna(0) > 0)].drop_duplicates("TRADEDATE").set_index("TRADEDATE")
        col = "CLOSE" if "CLOSE" in usd_tom else "WAPRICE"
        tom = usd_tom[col].astype(float)
    usd = RT.usd_checks(P, P["fut"]["OIL_FUT"], tom)
    parity = r2.parity_engine(ctx, cfg, P)
    recon = pd.concat([recon.assign(section="ROUTE_VS_BENCHMARK"), usd.assign(section="USD_RUB_CLOCK"),
                       parity.assign(section="SMLT_PARITY_ENGINE", comparison="SMLT funded vs SMLT futures + collateral (exact ledgers)"),
                       rules.assign(section="ROUTE_SELECTION_RULE")], ignore_index=True)
    save(ctx, recon, "adaptive_route_reconciliation.csv")
    uni = r2.universe(cfg, P, recon)
    extra = pd.DataFrame([{**x, "route": x["route"], "FAIR_STATUS": x["status"], "ADMISSIBILITY": x["admissibility"],
                           "status_reason": x["reason"]} for x in cfg["r2"]["not_tested"]])
    uni = pd.concat([uni, extra[[c for c in extra.columns if c in uni.columns]]], ignore_index=True)
    save(ctx, uni, "adaptive_asset_universe.csv")
    ctx.log["a2"] = {"fair_status": dict(zip(uni["route"], uni["FAIR_STATUS"])),
                     "parity_B_minus_A_pct": parity["B_minus_A_pct_of_exposure"].round(4).tolist()}


def a3_frontier(ctx) -> None:
    import dataclasses
    from . import frontier as F
    from . import r1, r3
    from . import routes as RT
    cfg = r1.load_cfg(ctx)
    P = RT.build_panel(ctx, cfg)
    t = pd.Timestamp(cfg["cutoff"])
    reg, iv = r3.views_at(ctx, cfg, P, t)
    ivr = iv.reset_index().assign(view_type="INTERVAL_ROBUST_SET", category="UNCERTAINTY_SET")
    fr = cfg["r3"]["frontier"]
    grid = fr["es_grid_nav0"]
    kap = [cfg["r3"]["uncertainty_set"]["kappa_canonical"], *cfg["r3"]["uncertainty_set"]["kappa_sensitivity"]]
    cache, rows = {}, []
    su = F.build(ctx, cfg, P, t, "MAIN", "BROAD_PIF_NO_HIDDEN_LEVERAGE", iv)
    rows += r3.frontier_rows(su, kap, grid, {"universe": "MAIN", "mandate": "BROAD_PIF_NO_HIDDEN_LEVERAGE"}, cache)
    su_f = dataclasses.replace(su, mandate="LONG_ISSUER_FLOOR_0")
    rows += r3.frontier_rows(su_f, kap, grid, {"universe": "MAIN", "mandate": "LONG_ISSUER_FLOOR_0"}, cache)
    su_n = F.build(ctx, cfg, P, t, "WITHOUT_BTC", "BROAD_PIF_NO_HIDDEN_LEVERAGE", iv)
    rows += r3.frontier_rows(su_n, kap, grid, {"universe": "WITHOUT_BTC", "mandate": "BROAD_PIF_NO_HIDDEN_LEVERAGE"}, cache)
    # labelled sensitivity (R3-A1): view set including the excluded 2019-2021 BTC prior
    iv_x = r3.views_with_excluded(ctx, cfg, P, t, reg)
    lab = cfg["r3"]["btc_prior_sensitivity"]
    su_x = F.build(ctx, cfg, P, t, "MAIN", "BROAD_PIF_NO_HIDDEN_LEVERAGE", iv_x)
    rows += r3.frontier_rows(su_x, kap, grid, {"universe": lab, "mandate": "BROAD_PIF_NO_HIDDEN_LEVERAGE"}, cache)
    ivx = iv_x.reset_index().assign(view_type="INTERVAL_SENSITIVITY_WITH_EXCLUDED_PRIORS", category="SENSITIVITY_ONLY")
    save(ctx, pd.concat([reg, ivr, ivx[ivx["route"] == "BTC_RUB_EXPERIMENTAL"]], ignore_index=True), "adaptive_return_view_registry.csv")
    fdf = pd.DataFrame(rows)
    for b in (fr["budgets"]["primary"], *fr["budgets"]["sensitivity"]):
        tag = int(round(b * 100))
        d = fdf[fdf["es_target_nav0"] <= b + 1e-12].copy()
        d.insert(0, "risk_budget_nav0", b)
        d["on_frontier"] = d.get(f"ADMISSION_{tag}", pd.Series("FAIL", index=d.index)) == "PASS"
        save(ctx, d, f"adaptive_frontier_{tag}.csv")
    # current book (no trade) under the same authority
    cur = F.A.evaluate_all(su.adm, su.book0, [], F.A.Rule(budget=1.0, beta=cfg["owner"]["liquidity_min_nav"]),
                           F.reserve_min(cfg["owner"]["liquidity_min_nav"], 0.0), (), ())
    cur_ex = {"rows": cur, "hard": {}}
    # zero-weight attribution at the top admitted canonical point (15%)
    canon = fdf[(fdf["universe"] == "MAIN") & (fdf["mandate"] == "BROAD_PIF_NO_HIDDEN_LEVERAGE")
                & (fdf["kappa"] == kap[0]) & (fdf.get("ADMISSION_15") == "PASS")]
    uni = pd.read_csv(ctx.out / "adaptive_asset_universe.csv")
    fair = dict(zip(uni["route"], uni["FAIR_STATUS"]))
    if len(canon):
        top = canon.sort_values(["robust_excess_exec_nav0", "es_target_nav0"], ascending=[False, True]).iloc[0].to_dict()
        nomf = fdf[(fdf["universe"] == "MAIN") & (fdf["mandate"] == "BROAD_PIF_NO_HIDDEN_LEVERAGE") & (fdf["kappa"] == 0.0)
                   & (fdf.get("ADMISSION_15") == "PASS")]
        nom_best = nomf.sort_values("nominal_excess_exec_nav0", ascending=False).iloc[0].to_dict() if len(nomf) else {}
        zw = r3.zero_weight(su, top, kap[0], fr["budgets"]["primary"], fair, cache, nom_best)
        zw.insert(0, "attribution_point", f"MAIN/BROAD kappa={kap[0]} e={top['es_target_nav0']}")
    else:
        zw = pd.DataFrame([{"attribution_point": "NO_ADMISSIBLE_TARGET at 15%", "cause": "NO_ADMISSIBLE_TARGET"}])
    save(ctx, zw, "adaptive_zero_weight_attribution.csv")
    # capital grid
    cg = []
    for C in cfg["r3"]["capital_grid_rub"]:
        suc = F.build(ctx, cfg, P, t, "MAIN", "BROAD_PIF_NO_HIDDEN_LEVERAGE", iv, capital=float(C))
        rc = r3.frontier_rows(suc, kap, [x for x in grid if x <= fr["budgets"]["primary"] + 1e-12],
                              {"universe": "MAIN", "mandate": "BROAD_PIF_NO_HIDDEN_LEVERAGE", "capital": C}, cache)
        rc = pd.DataFrame(rc)
        cb = F.A.evaluate_all(suc.adm, suc.book0, [], F.A.Rule(budget=1.0, beta=cfg["owner"]["liquidity_min_nav"]),
                              F.reserve_min(cfg["owner"]["liquidity_min_nav"], 0.0), (), ())
        cv = F.verdict({"rows": cb, "hard": {}}, fr["budgets"]["primary"], cfg)
        for k in kap:
            ok = rc[(rc["kappa"] == k) & (rc.get("ADMISSION_15") == "PASS")]
            best = ok.sort_values("robust_excess_exec_nav0", ascending=False).iloc[0] if len(ok) else None
            cg.append({"capital_rub": C, "nav0": suc.nav0, "initial_smlt_weight": suc.x0[suc.routes.index("SMLT")], "kappa": k,
                       "current_book_ES95_worst_nav0": max(r["ES95"] for r in cb if r["margin_multiplier"] == 1.0) / suc.nav0,
                       "current_book_admission_15": "PASS" if cv.passed else "FAIL",
                       "n_admitted_frontier_points_15": int(len(ok)),
                       "best_es_target": None if best is None else best["es_target_nav0"],
                       "best_robust_excess_nav0": None if best is None else best["robust_excess_exec_nav0"],
                       "best_nominal_excess_nav0": None if best is None else best["nominal_excess_exec_nav0"],
                       **({} if best is None else {k2: best[k2] for k2 in best.index if k2.startswith("w_exec_")}),
                       "engine_ES95_worst_nav0": None if best is None else best["engine_ES95_worst_nav0"]})
    save(ctx, pd.DataFrame(cg), "adaptive_capital_grid.csv")
    cv = F.verdict(cur_ex, fr["budgets"]["primary"], cfg)
    ctx.log["a3"] = {"gamma": su.gamma, "money_market_horizon": su.mH,
                     "current_book_admission_15": "PASS" if cv.passed else "FAIL", "current_book_reasons": cv.reasons,
                     "frontier_points_admitted_15_canonical": int(len(canon)),
                     "canonical_top": None if not len(canon) else {k: v for k, v in top.items() if k.startswith("w_exec_") or k in (
                         "es_target_nav0", "robust_excess_exec_nav0", "nominal_excess_exec_nav0", "engine_ES95_worst_nav0")}}


ROUTE_KINDS = [("SMLT", "FUNDED"), ("SMLT_FUT", "DERIVATIVE"), ("EQ_FUNDED", "FUNDED"), ("EQ_HEDGE_FUT", "DERIVATIVE"),
               ("OFZ_FUNDED", "FUNDED"), ("RATES_HEDGE_FUT", "DERIVATIVE"), ("GOLD_FUNDED", "FUNDED"), ("CNY_PERP", "DERIVATIVE"),
               ("OIL_FUT", "DERIVATIVE"), ("BTC_RUB_EXPERIMENTAL", "FUNDED")]


def a4_state(ctx) -> None:
    from . import events as EV
    from . import r1
    from . import routes as RT
    from . import state as ST
    cfg = r1.load_cfg(ctx)
    c4 = cfg["r4"]
    P = RT.build_panel(ctx, cfg)
    f = ST.features(ctx, cfg)
    s = ST.classify(f, cfg)
    daily = pd.concat([f, s], axis=1)
    daily.index.name = "date"
    save(ctx, daily.reset_index(), "adaptive_state_daily.csv")
    horizons = list(c4["horizons"])
    fs, eps = ST.forward_stats(s, P["R"], P["R"]["MMF"], ROUTE_KINDS, horizons + [c4["optional_horizon"]],
                               c4["shrink_n0_windows"], c4["sample_start"], c4["bootstrap_seed"])
    opt = fs["horizon"] == c4["optional_horizon"]
    fs = fs[~opt | (fs["n_nonoverlapping_windows"].fillna(0) >= 3)]
    save(ctx, fs, "adaptive_state_forward_stats.csv")
    eps = eps.copy()
    eps["start"] = pd.to_datetime(eps["start"]).dt.date.astype(str)
    eps["end"] = pd.to_datetime(eps["end"]).dt.date.astype(str)
    save(ctx, eps, "adaptive_state_episodes.csv")
    save(ctx, ST.transitions(s, c4["sample_start"]), "adaptive_state_transition_stats.csv")
    u = ST.view_table(fs[fs["horizon"] == 126], c4["state_view"]["min_nonoverlapping_windows"])
    keep = ["dimension", "state", "route", "horizon", "n_origins", "n_episodes", "n_nonoverlapping_windows", "mean_fwd_excess",
            "shrunk_mean", "all_origin_mean", "state_differential_shrunk", "ci90_lo_episode_bootstrap", "ci90_hi_episode_bootstrap",
            "evidence", "ci90_excludes_zero", "ci90_excludes_all_origin_mean", "usable_under_original_level_rule",
            "usable_as_state_view"]
    save(ctx, u[[c for c in keep if c in u]], "adaptive_state_uncertainty.csv")
    save(ctx, ST.descriptive_same_day(s, P["R"], ROUTE_KINDS, c4["sample_start"]), "adaptive_descriptive_stress_statistics.csv")
    E = EV.register(ctx, cfg)
    save(ctx, E, "adaptive_event_register.csv")
    lt = EV.lifecycle_tests(cfg)
    lt["result"] = lt["result"].astype(str)
    lt["expected"] = lt["expected"].astype(str)
    save(ctx, lt, "adaptive_event_lifecycle_tests.csv")
    t = pd.Timestamp(cfg["cutoff"])
    ctx.log["a4"] = {"state_at_cutoff": s.loc[t].to_dict(), "event_state_at_cutoff": EV.state_at(E, t),
                     "lifecycle_tests": dict(zip(lt["test"], lt["status"])),
                     "usable_state_views_126": int(u["usable_as_state_view"].sum()),
                     "usable_under_original_level_rule_126": int(u["usable_under_original_level_rule"].sum())}


def _priors(ctx, cfg, P):
    from . import r3
    from . import views as V
    pri = V.priors_pre2022(ctx, cfg)
    bp = r3.priors_btc(ctx, cfg, P, pri)
    if bp:
        pri["BTC_RUB"] = bp
    return pri


def a5_policy(ctx) -> None:
    import os
    from . import r1, r5
    from . import routes as RT
    cfg = r1.load_cfg(ctx)
    P = RT.build_panel(ctx, cfg)
    pri = _priors(ctx, cfg, P)
    C = r5.context(ctx, cfg, P, pri)
    lib = r5.library(ctx, cfg, P, pri, C, processes=int(os.environ.get("ADAPTIVE_PROCESSES", "6")))
    save(ctx, lib, "adaptive_policy_library.csv")
    conf = C["conf"].copy()
    conf.columns = [f"confirmed_{c}" for c in conf.columns]
    out = pd.concat([C["raw"].add_prefix("raw_"), conf], axis=1)
    out.index.name = "date"
    save(ctx, out.reset_index(), "adaptive_state_confirmed_daily.csv")
    act = lib[lib["is_actual_confirmed_state_at_cutoff"]]
    ctx.log["a5"] = {"confirmed_state_at_cutoff": C["conf"].loc[C["t"]].to_dict(),
                     "actions_in_actual_state": dict(zip(act["kappa"].astype(str), act["action_from_actual_book"])),
                     "distinct_actions_kappa1": sorted(lib[lib["kappa"] == 1.0]["action_from_actual_book"].unique().tolist()),
                     "distinct_actions_kappa0": sorted(lib[lib["kappa"] == 0.0]["action_from_actual_book"].unique().tolist())}


def a6_replay(ctx) -> None:
    import os
    from . import r1, r6
    from . import replay as RP
    from . import routes as RT
    cfg = r1.load_cfg(ctx)
    P = RT.build_panel(ctx, cfg)
    specs = r6.run_specs(cfg)
    res = r6.run_all(str(ctx.root), specs, processes=int(os.environ.get("ADAPTIVE_PROCESSES", "6")))
    by = {r["name"]: r for r in res}
    W = RP.world(ctx, cfg, P, "MAIN", cfg["r6"]["start"])
    nav0 = float(cfg["r6"]["nav0_rub"])
    mmf = W.px["MMF"]
    met = pd.DataFrame([{**{k: v for k, v in r["spec"].items() if k != "name"}, **r["metrics"]} for r in res])
    met = pd.concat([met, pd.DataFrame([r6.money_market_row(W, nav0)])], ignore_index=True)
    save(ctx, met, "adaptive_walkforward.csv")
    save(ctx, pd.DataFrame([r["attribution"] for r in res]), "adaptive_walkforward_attribution.csv")
    main = ["CONTROL_INITIAL_PORTFOLIO", "ALL_MONEY_MARKET", "SELL_50_TO_MONEY_MARKET", "SELL_100_TO_MONEY_MARKET", "FUT25_LEGACY",
            "FUT50_LEGACY", "V41_STATIC_PIF_WEIGHTS_PROXY_NONCAUSAL", "STATIC_DIVERSIFIED_V2", "ADAPTIVE_PIF_V1"]
    fc = met[met["comparator"].isin(main)].set_index("comparator").loc[main].reset_index()
    fc["excess_vs_STATIC_DIVERSIFIED_V2"] = fc["net_excess_over_money_market"] - float(
        fc.loc[fc["comparator"] == "STATIC_DIVERSIFIED_V2", "net_excess_over_money_market"].iloc[0])
    save(ctx, fc, "adaptive_final_comparison.csv")
    ledgers = {r["name"]: r["ledger"] for r in res}
    liq = []
    for n in ("CONTROL_INITIAL_PORTFOLIO", "FUT50_LEGACY", "STATIC_DIVERSIFIED_V2", "ADAPTIVE_PIF_V1", "ADAPTIVE_PIF_V1|kappa0"):
        if n in ledgers:
            lg = ledgers[n]
            d = pd.DataFrame({"comparator": n, "nav": lg["nav"], "immediate_cash": lg["cash"], "mmf_value": lg.get("v_MMF", 0.0),
                              "pending_settlement_net": lg["pending"], "posted_initial_margin": lg["im"],
                              "available_immediate": lg["available"], "available_nav": lg["available"] / lg["nav"], "breach": lg["breach"]})
            d.index.name = "date"
            liq.append(d.reset_index())
    save(ctx, pd.concat(liq, ignore_index=True), "adaptive_liquidity_path.csv")
    dec = pd.concat([r["decisions"] for r in res if len(r["decisions"])], ignore_index=True)
    dec["date"] = pd.to_datetime(dec["date"]).dt.date.astype(str)
    save(ctx, dec, "adaptive_decision_log.csv")
    rb = pd.concat([r6.rebalance_log(r["name"], r["ledger"], r["decisions"]) for r in res if r["spec"].get("type") == "policy"],
                   ignore_index=True)
    save(ctx, rb, "adaptive_rebalance_log.csv")
    navp = pd.DataFrame({n: ledgers[n]["nav"] for n in main if n in ledgers})
    navp["ALL_MONEY_MARKET"] = nav0 * mmf.reindex(W.dates) / float(mmf.reindex(W.dates).iloc[0])
    navp.index.name = "date"
    for n in ("ADAPTIVE_PIF_V1|kappa0", "STATIC_DIVERSIFIED_V2|kappa0", "ADAPTIVE_PIF_V1|kappa0|ORIGINAL_R4_LEVEL"):
        if n in ledgers:
            navp[n] = ledgers[n]["nav"]
    save(ctx, navp.reset_index(), "adaptive_nav_paths.csv")
    wts = []
    for n in ("ADAPTIVE_PIF_V1", "ADAPTIVE_PIF_V1|kappa0", "STATIC_DIVERSIFIED_V2", "STATIC_DIVERSIFIED_V2|kappa0",
              "ADAPTIVE_PIF_V1|kappa0|ORIGINAL_R4_LEVEL"):
        if n in ledgers:
            lg = ledgers[n]
            cols = [c for c in lg if c.startswith("v_") or c.startswith("n_")]
            w = lg[cols].div(lg["nav"], axis=0)
            w.columns = [c[2:] for c in cols]
            w.insert(0, "comparator", n)
            w.index.name = "date"
            wts.append(w.reset_index())
    save(ctx, pd.concat(wts, ignore_index=True), "adaptive_allocation_path.csv")
    # incremental value (pre-declared rule) and delay / cost sensitivity
    months = cfg["r5"]["recalibration"]["months"]
    inc, sens = [], []
    for k in cfg["r6"]["sensitivities"]["kappa"]:
        tag = "" if k == cfg["r6"]["sensitivities"]["kappa"][0] else f"|kappa{k:g}"
        rows = {}
        for var in ("", "|delay2", "|delay3", "|cost_x2", "|LONG_ISSUER_FLOOR_0", "|WITHOUT_BTC", "|ORIGINAL_R4_LEVEL"):
            # the state-view rule exists only in the adaptive policy: its sensitivity is compared with the base static ledger
            a, b = f"ADAPTIVE_PIF_V1{tag}{var}", f"STATIC_DIVERSIFIED_V2{tag}{'' if var == '|ORIGINAL_R4_LEVEL' else var}"
            if a not in ledgers or b not in ledgers:
                continue
            bl = r6.blocks(ledgers, mmf, a, b, months)
            st = r6.incremental(bl, cfg["r4"]["bootstrap_seed"])
            rows[var] = st
            ma, mb = by[a]["metrics"], by[b]["metrics"]
            sens.append({"kappa": k, "variant": var or "BASE", "adaptive": a, "static": b,
                         "net_excess_adaptive": ma["net_excess_over_money_market"], "net_excess_static": mb["net_excess_over_money_market"],
                         "difference": ma["net_excess_over_money_market"] - mb["net_excess_over_money_market"],
                         "max_drawdown_adaptive": ma["max_drawdown"], "max_drawdown_static": mb["max_drawdown"],
                         "breach_days_adaptive": ma["liquidity_breach_days"], "unresolved_adaptive": ma["unresolved_deficit_days"],
                         "costs_adaptive": ma["costs_rub"], "turnover_adaptive": ma["turnover_nav0"], **st})
            for _, r_ in bl.iterrows():
                inc.append({"kappa": k, "variant": var or "BASE", **{("excess_adaptive" if c.startswith("excess_ADAPTIVE") else
                                                                     "excess_static" if c.startswith("excess_STATIC") else c): v for c, v in r_.items()}})
        base = rows.get("")
        if base is not None:
            ok = by[f"ADAPTIVE_PIF_V1{tag}"]["metrics"]["unresolved_deficit_days"] == 0
            v = r6.verdict(base, [rows[x] for x in ("|delay2", "|delay3", "|cost_x2") if x in rows], ok)
            sens.append({"kappa": k, "variant": "VERDICT", "adaptive": f"ADAPTIVE_PIF_V1{tag}", "static": f"STATIC_DIVERSIFIED_V2{tag}",
                         "verdict": v, "role": "CANONICAL" if tag == "" else "SENSITIVITY_ONLY"})
    save(ctx, pd.DataFrame(sens), "adaptive_delay_cost_sensitivity.csv")
    save(ctx, pd.DataFrame(inc), "adaptive_incremental_value_blocks.csv")
    ctx.log["a6"] = {"runs": len(res), "verdicts": {str(r["kappa"]): r["verdict"] for r in sens if r.get("variant") == "VERDICT"},
                     "net_excess": {n: by[n]["metrics"]["net_excess_over_money_market"] for n in main if n in by}}


def _risk_contribution(su, books: dict, alpha: float) -> pd.DataFrame:
    """Sleeve contributions to ES95 of the static horizon P&L on the worst model's tail paths (Euler allocation)."""
    import numpy as np
    from . import frontier as F
    rows = []
    adj = np.array([su.mH if r in F.FUNDED else 0.0 for r in su.routes])
    for name, w in books.items():
        x = su.x0 if w is None else np.array([w.get(r, 0.0) for r in su.routes])
        best = None
        for m, G in su.G.items():
            contrib = (G - adj[None, :]) * x[None, :]
            pnl = su.mH + contrib.sum(axis=1)
            k = max(1, int(np.ceil((1 - alpha) * len(pnl))))
            tail = np.argsort(pnl)[:k]
            es = -pnl[tail].mean()
            if best is None or es > best[0]:
                best = (es, m, -contrib[tail].mean(axis=0))
        es, m, c = best
        for r, v in zip(su.routes, c):
            rows.append({"book": name, "worst_model": m, "sleeve": r, "weight_nav0": float(x[su.routes.index(r)]),
                         "es_contribution_nav0": float(v), "book_ES95_static_nav0": float(es)})
        rows.append({"book": name, "worst_model": m, "sleeve": "MMF_AND_RESERVE", "weight_nav0": float(1 - sum(x[j] for j, r in enumerate(su.routes) if r in F.FUNDED)),
                     "es_contribution_nav0": float(-su.mH), "book_ES95_static_nav0": float(es)})
    return pd.DataFrame(rows)


def a7_decision(ctx) -> None:
    from . import r1, r5
    from . import routes as RT
    from . import stress as SS
    cfg = r1.load_cfg(ctx)
    P = RT.build_panel(ctx, cfg)
    pri = _priors(ctx, cfg, P)
    C = r5.context(ctx, cfg, P, pri)
    obj, cur = r5.decision_object(ctx, cfg, P, pri, C)
    d, d0 = obj.pop("_decisions")
    save_json(ctx, obj, "adaptive_decision_object.json")
    save(ctx, cur, "adaptive_current_portfolio.csv")
    books = {}
    for name, spec in cfg["stress"]["books"].items():
        if spec == "decision":
            books[name] = None if d.action in ("HOLD", "NO_ADMISSIBLE_TARGET") else d.target_w
        elif spec == "decision_kappa0":
            books[name] = None if d0.action in ("HOLD", "NO_ADMISSIBLE_TARGET") else d0.target_w
        else:
            books[name] = spec or None
    st = SS.run(d.su, books, cfg)
    st.insert(0, "section", "DETERMINISTIC_PATH_STRESS")
    adm_rows = []
    for nm, dd in (("ADAPTIVE_TARGET_CANONICAL", d), ("ADAPTIVE_TARGET_NOMINAL_KAPPA0", d0)):
        for r in (dd.ex["rows"] if dd.ex is not None else []):
            adm_rows.append({"section": "ADMISSION_MARGIN_MULTIPLIERS", "book": nm, "scenario": r["model"], "margin_multiplier": r["margin_multiplier"],
                             "pnl_nav0": r["mean_pnl"] / r["NAV0"], "ES95_nav0": r["ES95_nav0"], "p_breach": r["p_breach"],
                             "p_unresolved": r["p_unresolved"], "p_forced": r["p_forced"], "min_available_nav": r["min_available_nav_p05"],
                             "peak_im_nav": r["peak_im_nav_p95"], "kind": "SIMULATION_HOLDOUT"})
    save(ctx, pd.concat([st, pd.DataFrame(adm_rows)], ignore_index=True), "adaptive_margin_stress.csv")
    save(ctx, _risk_contribution(d.su, books, cfg["owner"]["es_alpha"]), "adaptive_risk_contribution.csv")
    mc_books = {k: books[k] for k in ("CURRENT_BOOK_NO_TRADE", "ADAPTIVE_TARGET_CANONICAL", "ADAPTIVE_TARGET_NOMINAL_KAPPA0") if k in books}
    save(ctx, r5.monte_carlo_error(ctx, cfg, P, C, d.su, mc_books, cfg["r6"]["validation"]["monte_carlo_seed_offsets"]),
         "adaptive_monte_carlo_error.csv")
    ctx.log["a7"] = {"action": obj["SELECTED_ACTION"], "target_weights": obj["TARGET_WEIGHTS_NAV"], "current_admitted": obj["CURRENT_BOOK_ADMISSION"]["admitted"],
                     "kappa0_action": obj["SENSITIVITY_NOMINAL_KAPPA0"]["action"],
                     "stress_unresolved": int(st["unresolved_deficit"].sum()), "stress_rows": int(len(st))}


def a8_btc_and_ladder(ctx) -> None:
    import numpy as np
    from . import r1
    from . import routes as RT
    cfg = r1.load_cfg(ctx)
    P = RT.build_panel(ctx, cfg)
    rows = []
    for b in (10, 15, 20):
        f = pd.read_csv(ctx.out / f"adaptive_frontier_{b}.csv")
        f = f[(f["mandate"] == "BROAD_PIF_NO_HIDDEN_LEVERAGE") & f["on_frontier"]]
        for (u, k), g in f.groupby(["universe", "kappa"]):
            top = g.sort_values(["robust_excess_exec_nav0", "es_target_nav0"], ascending=[False, True]).iloc[0]
            rows.append({"section": "FRONTIER", "budget": b / 100, "kappa": k, "universe": u, "n_admitted_points": len(g),
                         "best_robust_excess_nav0": top["robust_excess_exec_nav0"], "best_nominal_excess_nav0": top["nominal_excess_exec_nav0"],
                         "engine_ES95_nav0": top["engine_ES95_worst_nav0"],
                         "btc_weight": 0.0 if u == "WITHOUT_BTC" else float(np.nan_to_num(top.get("w_exec_BTC_RUB_EXPERIMENTAL", 0.0))),
                         "max_btc_weight_on_frontier": 0.0 if u == "WITHOUT_BTC" else float(np.nan_to_num(g["w_exec_BTC_RUB_EXPERIMENTAL"].max()))})
    wf = pd.read_csv(ctx.out / "adaptive_walkforward.csv")
    for nm in wf["comparator"]:
        if isinstance(nm, str) and nm.startswith("ADAPTIVE_PIF_V1") and (nm.endswith("WITHOUT_BTC") or nm in ("ADAPTIVE_PIF_V1", "ADAPTIVE_PIF_V1|kappa0")):
            r = wf[wf["comparator"] == nm].iloc[0]
            rows.append({"section": "REPLAY", "comparator": nm, "kappa": r.get("kappa"), "universe": r.get("universe"),
                         "net_excess_over_money_market": r["net_excess_over_money_market"], "max_drawdown": r["max_drawdown"],
                         "btc_exposure_avg": r["btc_exposure_avg"], "terminal_nav": r["terminal_nav"]})
    R = P["R"].loc[cfg["r6"]["start"]:]
    b = R["BTC_RUB_EXPERIMENTAL"].dropna()
    lvl = (1 + b).cumprod()
    rows.append({"section": "STATS", "stat": "source", "value_text": "BTCUSDT (Binance 1h, bar closing 15:00 UTC) x CBR USD/RUB known at the close; direct BTCRUB cross-check 2022-2024"})
    for k, v in (("start_of_series", str(P["R"]["BTC_RUB_EXPERIMENTAL"].first_valid_index().date())), ("sessions_in_replay", len(b)),
                 ("vol_ann", b.std(ddof=1) * np.sqrt(252)), ("worst_session", b.min()), ("ES95_1d", -np.sort(b.values)[:max(1, int(0.05 * len(b)))].mean()),
                 ("max_drawdown", ((lvl.cummax() - lvl) / lvl.cummax()).max()),
                 ("share_of_sessions_spanning_a_weekend_or_holiday", float((pd.Series(b.index).diff().dt.days > 1).mean()))):
        rows.append({"section": "STATS", "stat": k, "value": v if not isinstance(v, str) else None, "value_text": v if isinstance(v, str) else None})
    wk = (1 + R).resample("W-FRI").prod() - 1
    smlt_dn = R["SMLT"] < R["SMLT"].quantile(0.10)
    for c in ("SMLT", "EQ_FUNDED", "OFZ_FUNDED", "GOLD_FUNDED", "CNY_PERP", "OIL_FUT"):
        rows.append({"section": "CORRELATION", "stat": c, "corr_daily": R["BTC_RUB_EXPERIMENTAL"].corr(R[c]), "corr_weekly": wk["BTC_RUB_EXPERIMENTAL"].corr(wk[c]),
                     "corr_daily_on_worst_decile_SMLT_days": R.loc[smlt_dn, "BTC_RUB_EXPERIMENTAL"].corr(R.loc[smlt_dn, c]) if c != "SMLT" else None,
                     "mean_btc_return_on_worst_decile_SMLT_days": float(R.loc[smlt_dn, "BTC_RUB_EXPERIMENTAL"].mean()) if c == "SMLT" else None})
    fs = pd.read_csv(ctx.out / "adaptive_state_forward_stats.csv")
    g = fs[(fs["route"] == "BTC_RUB_EXPERIMENTAL") & (fs["horizon"] == 126)]
    for _, r in g.iterrows():
        rows.append({"section": "FORWARD_STATE", "stat": f"{r['dimension']}={r['state']}", "value": r.get("shrunk_mean"),
                     "ci90_lo": r.get("ci90_lo_episode_bootstrap"), "ci90_hi": r.get("ci90_hi_episode_bootstrap"),
                     "n_nonoverlapping_windows": r.get("n_nonoverlapping_windows"), "corr_with_smlt_fwd": r.get("corr_with_smlt_fwd"),
                     "mean_when_smlt_fwd_bottom_decile": r.get("mean_when_smlt_fwd_bottom_decile")})
    save(ctx, pd.DataFrame(rows), "adaptive_btc_incremental_value.csv")
    ladder = [
        ("IDENTITY_PROOF", "legacy V1-V4.1 outputs reproduce byte-for-byte from 5bf38e6", "adaptive_legacy_reproduction.csv"),
        ("IDENTITY_PROOF", "NAV, P&L and quantity identities of the engine on exchange data; integer contracts", "adaptive_accounting_identity.csv, adaptive_integer_execution_audit.csv"),
        ("IDENTITY_PROOF", "closed-form static P&L equals the engine on the no-path-dependence control", "tests/test_adaptive_admission.py"),
        ("IDENTITY_PROOF", "SMLT route parity decomposition (two exact ledgers, zero residual)", "adaptive_route_reconciliation.csv"),
        ("IDENTITY_PROOF", "USD/RUB clock: CBR record dated t+1 equals the FORTS clearing rate (99.8% of days since 2024-07)", "adaptive_route_reconciliation.csv"),
        ("SIMULATION_HOLDOUT", "frontier admission (ES95, liquidity, survival) on independent-seed paths of the SAME 2022-2026 history", "adaptive_frontier_{10,15,20}.csv"),
        ("SIMULATION_HOLDOUT", "V4.1 candidate re-admission on the V4.1 independent-seed paths", "adaptive_candidate_admission.csv"),
        ("RETROSPECTIVE_WALK_FORWARD", "one-ledger replay 2023-01-03..2026-09-25 of every comparator; 8 semi-annual blocks; one SMLT decline", "adaptive_walkforward.csv"),
        ("RETROSPECTIVE_WALK_FORWARD", "adaptive vs static incremental value by semi-annual blocks", "adaptive_delay_cost_sensitivity.csv, adaptive_incremental_value_blocks.csv"),
        ("RETROSPECTIVE_WALK_FORWARD", "forward-after-state statistics (independent windows counted; overlapping windows not independent)", "adaptive_state_forward_stats.csv"),
        ("SCENARIO_ONLY", "deterministic path stresses (no probabilities)", "adaptive_margin_stress.csv"),
        ("SCENARIO_ONLY", "policy library for states other than the confirmed state at the cutoff (incl. hypothetical severe event)", "adaptive_policy_library.csv"),
        ("SCENARIO_ONLY", "kappa 0 / 0.5 frontiers and the kappa 0 replay (sensitivities of the one policy)", "adaptive_frontier_*.csv, adaptive_walkforward.csv"),
        ("NEW_MARKET_OOS", "NONE: the cutoff is the last data date; no claim rests on untouched market data", ""),
    ]
    save(ctx, pd.DataFrame(ladder, columns=["evidence_class", "claim", "outputs"]), "adaptive_validation_evidence_ladder.csv")
    ctx.log["a8"] = {"btc_canonical_weight_15": float(next((r["btc_weight"] for r in rows if r.get("section") == "FRONTIER" and r.get("budget") == 0.15
                                                          and r.get("kappa") == 1.0 and r.get("universe") == "MAIN"), 0.0))}


def a9_figures(ctx) -> None:
    from . import figures as FG
    made = []
    for f in FG.ALL:
        f(ctx)
        made.append(f.__name__)
    ctx.log["a9"] = {"figures": made}


STAGES = {"a1": a1_accounting, "a2": a2_instruments, "a3": a3_frontier, "a4": a4_state, "a5": a5_policy, "a6": a6_replay,
          "a7": a7_decision, "a8": a8_btc_and_ladder, "a9": a9_figures}
