"""V4.1 final policy integration (stages s40-s45). Every stage READS V1-V4 outputs and WRITES NEW files only: the V4
baseline (STATIC_PIF and every earlier artefact) is not changed. All rules are pre-declared in configs/project.yaml `v41`.

s40 budgets_horizons   Phase 1+2: the V4 selection rule at ES budgets 10/15/20% of NAV0, and otherwise-identical 6m / 12m
                       frontiers (separate TRAIN / VALIDATION / TEST); TEST of every selected portfolio at both horizons
s41 scenario_policies  Phase 3/4/7: conditional scenario frontiers on the V4 joint forward paths (no probabilities), target
                       bands (neighbouring budgets, independent seed, anchors x0.75 / x1.25, integer rounding), explicit
                       zero-SMLT / partial-retention / re-risk candidates evaluated with the fund engine
s42 primary_sources    Phase 11/13: Bank of Russia primary series parsed; secondary facts re-checked against primary and
                       exchange data (CONFIRMED_PRIMARY / CORRECTED / STILL_UNVERIFIED); BTC spot FX route status
s43 state_events       Phase 5/6: CREDIT_STATE (market), FUNDAMENTAL_STATE (issuer facts, point in time), EVENT register,
                       state hierarchy -> conditional scenario (rules, not scores; no NLP / social media / LLM input)
s44 premium            Phase 10: ASSUMED SMLT excess return 0 / +5 / +10% p.a. (never estimated)
s45 decision_card      Phase 9: deterministic decision card (JSON); no bot, no scheduler
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..data import primary as pr
from ..risk.metrics import es_hist
from ..scenarios.bootstrap import horizon_returns
from . import core, fund, stages
from . import optimize as opt
from . import universe as uni

LABELS = "CONDITIONAL_POLICY | SCENARIO_NOT_FORECAST | NOT_OOS_ALPHA_CLAIM"
SCENARIOS = ["CONTINUED_DETERIORATION", "STABILISATION_FLAT", "RATES_RELIEF_RECOVERY", "EVENT_DRIVEN_UPSIDE", "ADVERSE_CREDIT_EVENT"]
FUND_KEYS = ("mean_pnl", "median_pnl", "ES95", "participation", "p_liquidity_breach", "peak_margin_nav_p95", "turnover_nav")


def _EC(ctx) -> tuple[float, float]:
    return float(ctx.E), float(ctx.cfg["v4"]["main_capital"])


def _w(assets, x, prefix="x_") -> dict:
    return {f"{prefix}{a.name}": float(v) for a, v in zip(assets, x)}


def book(assets, x) -> dict:
    """Balance-sheet summary of a NAV-weight vector (liquidity = residual cash)."""
    f = np.array([a.kind == "FUNDED" for a in assets])
    mr = np.array([a.mr for a in assets])
    n = [a.name for a in assets]
    net_smlt = float(x[n.index("SMLT")] + (x[n.index("SMLT_FUT")] if "SMLT_FUT" in n else 0.0))
    return {"liquidity_nav": float(1 - (x * f).sum()), "margin_nav": float((np.abs(x) * ~f * mr).sum()),
            "gross_nav": float(np.abs(x).sum()), "net_smlt_nav": net_smlt, "net_market_nav": float(x.sum()),
            "btc_nav": float(sum(v for a, v in zip(assets, x) if a.name.startswith("BTC"))),
            "net_short_issuer": bool(net_smlt < -1e-6)}


# ================================================================== s40  Phase 1 + 2
def _select_at(ctx, fac, assets, fr, summ, E, C, hz, budget):
    sel = core.select(summ, E + C, budget, ctx.cfg["v4"]["selection"]["max_worst_breach"])
    if sel is None:
        return None, None
    x = core.xvec(fr.set_index("cand_id").loc[sel["cand_id"]], assets)
    return stages._prune_and_reevaluate(ctx, fac, assets, x, sel, E, C, hz, None, core.MODELS, budget=budget)


def portfolio_name(hn: str, b: float, b0: float) -> str:
    return f"STATIC_PIF_{hn.upper()}" + ("" if abs(b - b0) < 1e-12 else f"_ES{int(round(b * 100))}")


def s40_budgets_horizons(ctx) -> None:
    from ..portfolio import validation as va
    from ..v3_stages import verdict
    v4, v41 = ctx.cfg["v4"], ctx.cfg["v41"]
    E, C = _EC(ctx)
    NAV0, b0 = E + C, v4["selection"]["es_budget"]
    rows, books = [], {}
    fac = assets = None
    for hn, hz in v41["horizons"].items():
        assets, fac, frows = stages._candidates_for(ctx, "WITHOUT_BTC", E, C, E / NAV0, hz, fac=fac, label="PIF_CONSERVATIVE")
        cands = [(f"PIF_CONSERVATIVE_WITHOUT_BTC_{int(C)}_{i:02d}", core.static_policy(core.xvec(r, assets)), None) for i, r in enumerate(frows)]
        summ = core.summarise(core.evaluate(ctx, fac, assets, cands, hz, E, C, "VAL", v4["n_val"]))
        fr = pd.DataFrame(frows)
        fr["cand_id"] = [c[0] for c in cands]
        fr = fr.merge(summ, on="cand_id", how="left")
        rows += [{"row_type": "FRONTIER", "horizon": hn, "weights_basis": "continuous (TRAIN LP)", **r} for r in fr.to_dict("records")]
        for b in v41["risk_budgets"]:
            name = portfolio_name(hn, b, b0)
            x, sel = _select_at(ctx, fac, assets, fr, summ, E, C, hz, b)
            if x is None:
                rows.append({"row_type": "SELECTED", "horizon": hn, "portfolio": name, "es_budget": b, "status": "NO_ADMISSIBLE_CANDIDATE"})
                continue
            xi, _ = core.integer_portfolio(ctx, assets, x, NAV0)
            books[name] = (hn, b, xi)
            rows.append({"row_type": "SELECTED", "horizon": hn, "portfolio": name, "es_budget": b, "status": "SELECTED",
                         "weights_basis": "integer units", "cand_id": sel["cand_id"],
                         **{k: sel[k] for k in ("worst_ES95", "worst_participation", "worst_breach", "mean_pnl_avg", "worst_mdd_p95", "turnover")},
                         **_w(assets, xi), **book(assets, xi)})
    # the 6m / 15% selection must reproduce the V4 STATIC_PIF (same seeds, same rule)
    v4x = dict(zip([a.name for a in assets], stages.static_book(ctx)["STATIC_PIF"][1]))
    x6 = books.get(portfolio_name("6m", b0, b0))
    rep = bool(x6 is not None and max(abs(v4x.get(a.name, 0.0) - v) for a, v in zip(assets, x6[2])) < 1e-9)
    for r in rows:
        if r.get("portfolio") == portfolio_name("6m", b0, b0):
            r["reproduces_v4_STATIC_PIF"] = rep
    # TEST: every selected portfolio (and CONTROL) at BOTH horizons, 4 models x n_test_seeds independent seeds
    cands = [("CONTROL_INITIAL_PORTFOLIO", core.static_policy(uni.initial_weights(assets, E, C)), None)] + \
            [(n, core.static_policy(v[2]), None) for n, v in books.items()]
    parts = []
    for hn, hz in v41["horizons"].items():
        for k in range(v4["n_test_seeds"]):
            ev = core.evaluate(ctx, fac, assets, cands, hz, E, C, "TEST", v4["n_test"], k=k)
            ev["horizon"] = hn
            parts.append(ev)
    T = pd.concat(parts, ignore_index=True)
    ctx.save(T, "pif_v41_test_metrics.csv", "interim", index=False)
    for (name, hn), g in T.groupby(["cand_id", "horizon"], sort=False):
        fhs = g[g["model"] == "FHS_FULL"]
        wm = g.groupby("model")["ES95"].mean()
        b = books[name][1] if name in books else np.nan
        rows.append({"row_type": "TEST", "horizon": hn, "portfolio": name, "es_budget": b,
                     "selected_on_horizon": books[name][0] if name in books else "",
                     "ES95_FHS": float(fhs["ES95"].mean()), "ES95_per_NAV_FHS": float(fhs["ES95"].mean() / NAV0),
                     "worst_model": wm.idxmax(), "worst_model_ES95": float(wm.max()), "worst_model_ES95_per_NAV": float(wm.max() / NAV0),
                     "budget_met_worst_model": bool(wm.max() <= b * NAV0) if np.isfinite(b) else np.nan,
                     "mean_pnl_FHS": float(fhs["mean_pnl"].mean()), "median_pnl_FHS": float(fhs["median_pnl"].mean()),
                     "participation_FHS": float(fhs["participation"].mean()),
                     "worst_model_participation": float(g.groupby("model")["participation"].mean().min()),
                     "p95_mdd_nav_FHS": float(fhs["p95_mdd_nav"].mean()), "p_liquidity_breach_max": float(g["p_liquidity_breach"].max()),
                     "n_seeds": int(g["seed_k"].nunique())})
    a, bn = portfolio_name("12m", b0, b0), portfolio_name("6m", b0, b0)
    if a in books and bn in books:
        for hn in v41["horizons"]:
            g = T[T["horizon"] == hn]
            for metric in ("ES95", "mean_pnl", "participation"):
                pm = va.delta_ci(g, a, bn, metric)
                if pm:
                    vd = verdict(pm, ctx.cfg["v3"]["stability"]["material_rub"] if metric != "participation" else 0.0,
                                 ctx.cfg["v3"]["stability"]["min_sign_share"], metric == "ES95")
                    f = pm.get("FHS_FULL", {})
                    rows.append({"row_type": "DELTA", "horizon": hn, "portfolio": f"{a} - {bn}", "metric": metric,
                                 "delta_FHS": f.get("mean"), "ci_lo_FHS": f.get("ci_lo"), "ci_hi_FHS": f.get("ci_hi"), "verdict_all_models": vd})
        d = np.abs(books[a][2] - books[bn][2])
        same = bool(((np.abs(books[a][2]) > 1e-9) == (np.abs(books[bn][2]) > 1e-9)).all())
        similar = bool(d.max() <= v41["similarity_max_abs_diff_nav"] and same)
        rows.append({"row_type": "VERDICT", "portfolio": f"{a} vs {bn}", "max_abs_weight_diff_nav": float(d.max()),
                     "largest_diff_asset": assets[int(d.argmax())].name, "same_nonzero_sleeves": same,
                     "status": "SIMILAR" if similar else "MATERIALLY_DIFFERENT",
                     "interpretation": "6m = operational horizon, 12m = stress horizon" if similar else
                     "keep both: 6m operational target, 12m longer-horizon policy (reported, not blended)"})
    ctx.save(pd.DataFrame(rows), "pif_frontier_6m_12m.csv", index=False)


# ================================================================== s41  Phase 3 + 4 + 7
def scenario_assets(ctx, E, C) -> list:
    """Master universe (every sleeve incl. both BTC routes) under the SAME mandate as STATIC_PIF (SMLT <= E / NAV0)."""
    cap = E / (E + C)
    a = uni.build(ctx, "WITH_BTC_EXPERIMENTAL_SPOT", E + C, cap)
    return a + [x for x in uni.build(ctx, "WITH_BTC_MOEX_DERIVATIVE", E + C, cap) if x.name == "BTC_FUT"]


def path_set(ctx, assets, mu_scale=1.0, seed_shift=0, keep_paths=False) -> dict:
    """Joint forward paths per scenario (V4 s33 construction: same columns, n, seeds when mu_scale=1, seed_shift=0), the
    scenario's money-market path and the horizon-return model for the LP."""
    from ..v3_stages import forward_paths
    hz = ctx.cfg["v41"]["scenario"]["horizon_days"]
    cols = list(dict.fromkeys(["SMLT", "SMLT_FUT"] + [a.series for a in assets] + ["SMLT_BOND_IDX"]))
    paths_, meta, _ = forward_paths(ctx, cols, n=ctx.cfg["v4"]["n_test"], mu_scale=mu_scale, seed_shift=seed_shift)
    mi = pd.read_csv(ctx.out / "market_inputs_at_cutoff.csv", index_col=0)["value"]
    y0, dmod = float(mi["mmf_yield_LQDT_60d"]), float(mi["rgbi_dmod"])
    mt = meta.set_index("scenario")
    tl = pd.read_csv(ctx.out / "pif_scenario_tilts.csv").dropna(subset=["tilt_12m_log"])
    ser = list(dict.fromkeys(["SMLT"] + [a.series for a in assets]))
    out, shifts = {}, {}
    for sc in list(paths_):
        Pf = paths_.pop(sc)
        base = ctx.cfg["v3"]["forward"]["overlays"].get(sc, {}).get("base", sc)
        y1 = max(y0 - float(mt.loc[base, "mu12m_RATES"]) / dmod, 0.01)
        md = (1 + np.linspace(y0, y1, 252)[:hz]) ** (1 / 252) - 1
        P = {c: v[:, :hz].copy() for c, v in Pf.items() if isinstance(v, np.ndarray) and v.ndim == 2}
        del Pf
        m = float(np.prod(1 + md) - 1)
        HR = horizon_returns(P, ser)
        M_raw = opt.Model(R=HR[[a.series for a in assets]].values, rs=HR["SMLT"].values, m=m)
        if ctx.cfg["v41"]["scenario"].get("drift_consistency") == "tilt_only":
            if base not in shifts:                       # structural scenarios come first; overlays reuse their base shift
                tilt = tl[tl["scenario"] == base].set_index("asset_series")["tilt_12m_log"] * mu_scale * hz / 252
                shifts[base] = {c: (float(np.log(np.mean(np.prod(1 + P[c], axis=1)))) - float(tilt[c])) / hz for c in P if c in tilt.index}
                shifts[base]["SMLT_FUT"] = shifts[base]["SMLT"]
            for c, k in shifts[base].items():
                if c in P:
                    P[c] = (1 + P[c]) * np.exp(-k) - 1
            HR = horizon_returns(P, ser)
        M = opt.Model(R=HR[[a.series for a in assets]].values, rs=HR["SMLT"].values, m=m)
        out[sc] = {"M": M, "M_raw": M_raw, "md": md, "P": P if keep_paths else None, "shift": shifts.get(base, {})}
    return out


def horizon_pnl(lp, M, x) -> np.ndarray:
    """Exact static horizon P&L per NAV0 (the LP's accounting, optimize.diagnostics)."""
    f = np.array([a.kind == "FUNDED" for a in lp.assets])
    pos = np.concatenate([np.maximum(x, 0), np.maximum(-x, 0), np.maximum(np.where(f, x - lp.w0, 0), 0),
                          np.maximum(np.where(f, lp.w0 - x, 0), 0)])
    G, c0 = opt._coef(lp, M)
    return c0 + G @ pos


def static_stats(lp, M, x, nav0) -> dict:
    p = horizon_pnl(lp, M, x) * nav0
    c = opt.control_pnl(lp, M) * nav0
    up = M.rs > 0
    part = float(p[up].mean() / c[up].mean()) if up.any() and c[up].mean() != 0 else np.nan
    return {"mean_pnl": float(p.mean()), "median_pnl": float(np.median(p)), "ES95": float(es_hist(p, 0.95)), "participation": part,
            "p_loss": float((p < 0).mean())}


def _solve(lp, M, budget):
    r = opt.solve(lp, [M], "max_premium", budget=budget)      # premia None: maximise the scenario's conditional mean
    return r["x"] if r["status"] == "optimal" else None


def fund_eval(ctx, S, assets, x, E, C) -> dict:
    P = S["P"]
    N, H = P["SMLT"].shape
    out = fund.run(P, S["md"], assets, core.static_policy(x), np.zeros((N, H + 1), int), E=E, C=C,
                   x0_initial=uni.initial_weights(assets, E, C))
    return {k: float(out[k]) for k in FUND_KEYS}


def band_status(widths: dict, feasible: bool, limit: float) -> str:
    return "STABLE" if feasible and all(w <= limit + 1e-12 for w in widths.values()) else "NO_STABLE_TARGET"


def s41_scenario_policies(ctx) -> None:
    v4, sc_cfg = ctx.cfg["v4"], ctx.cfg["v41"]["scenario"]
    E, C = _EC(ctx)
    NAV0, hz, tb = E + C, sc_cfg["horizon_days"], sc_cfg["target_budget"]
    assets = scenario_assets(ctx, E, C)
    names = [a.name for a in assets]
    lp = core.make_lp(ctx, assets, E, C, hz)
    x0 = uni.initial_weights(assets, E, C)
    base = path_set(ctx, assets)
    alt = path_set(ctx, assets, seed_shift=sc_cfg["seed_shift"], keep_paths=True)
    mus = {s: path_set(ctx, assets, mu_scale=s) for s in sc_cfg["mu_scale"]}
    fr_rows, band_rows, targets = [], [], {}
    for sc in SCENARIOS:
        M, Ma = base[sc]["M"], alt[sc]["M"]
        for b in v4["es_budget_grid"]:                   # the artefact the drift-consistency correction removes (kept visible)
            x = _solve(lp, base[sc]["M_raw"], b)
            if x is not None:
                fr_rows.append({"row_type": "FRONTIER_UNCORRECTED_DIAGNOSTIC", "scenario": sc, "objective": sc_cfg["objective"], "es_budget": b,
                                "status": "OPTIMAL", "labels": "DIAGNOSTIC ONLY - log-demeaned paths, convexity not removed", **_w(assets, x),
                                **book(assets, x), **{f"insample_{k}": v for k, v in static_stats(lp, base[sc]["M_raw"], x, NAV0).items()}})
        fr_rows.append({"row_type": "DRIFT_SHIFT", "scenario": sc, "labels": "daily log shift removed per series (6m mean = declared tilt)",
                        **{f"shift_6m_{c}": -k * hz for c, k in base[sc]["shift"].items()}})
        for b in v4["es_budget_grid"]:
            x = _solve(lp, M, b)
            row = {"row_type": "FRONTIER", "scenario": sc, "objective": sc_cfg["objective"], "es_budget": b, "labels": LABELS}
            if x is None:
                fr_rows.append({**row, "status": "INFEASIBLE"})
                continue
            fr_rows.append({**row, "status": "OPTIMAL", **_w(assets, x), **book(assets, x),
                            **{f"insample_{k}": v for k, v in static_stats(lp, M, x, NAV0).items()},
                            **{f"indep_seed_{k}": v for k, v in static_stats(lp, Ma, x, NAV0).items()}})
        # target and band (Phase 4)
        xt = _solve(lp, M, tb)
        pert = {"TARGET": xt, **{f"BUDGET_{b:g}": _solve(lp, M, b) for b in sc_cfg["neighbour_budgets"]},
                "INDEPENDENT_SEED": _solve(lp, Ma, tb), **{f"ANCHOR_x{s:g}": _solve(lp, mus[s][sc]["M"], tb) for s in sc_cfg["mu_scale"]}}
        xi = core.integer_portfolio(ctx, assets, xt, NAV0)[0] if xt is not None else None
        pert["INTEGER"] = xi
        feasible = all(v is not None for v in pert.values())
        V = {k: v for k, v in pert.items() if v is not None}
        widths = {a: float(max(v[j] for v in V.values()) - min(v[j] for v in V.values())) for j, a in enumerate(names)} if V else {}
        status = band_status(widths, feasible, sc_cfg["stable_max_band_width_nav"])
        targets[sc] = (xi, status)
        for j, a in enumerate(names + ["LIQUIDITY", "NET_SMLT"]):
            if j < len(names):
                vals = {k: float(v[j]) for k, v in V.items()}
            else:
                vals = {k: book(assets, v)["liquidity_nav" if a == "LIQUIDITY" else "net_smlt_nav"] for k, v in V.items()}
            lo, hi = (min(vals.values()), max(vals.values())) if vals else (np.nan, np.nan)
            band_rows.append({"scenario": sc, "asset": a, **{f"w_{k}": vals.get(k, np.nan) for k in pert}, "band_lo": lo, "band_hi": hi,
                              "band_width": hi - lo, "asset_band_within_limit": bool(hi - lo <= sc_cfg["stable_max_band_width_nav"] + 1e-12),
                              "scenario_status": status,
                              "applicable_target": "SCENARIO_BAND" if status == "STABLE" else "FALLBACK_STATIC_PIF", "labels": LABELS})
    # robust baseline band: STATIC_PIF with its neighbouring frontier budgets (same rule as the scenario bands)
    fr4 = pd.read_csv(ctx.out / "pif_efficient_frontier.csv")
    fr4 = fr4[(fr4["universe"] == "WITHOUT_BTC") & (fr4["mandate"] == "PIF_CONSERVATIVE") & (fr4["objective"] == "max_participation")]
    sp_assets, sp_x = stages.static_book(ctx)["STATIC_PIF"]
    sp = stages._embed(sp_assets, sp_x, assets)
    bvec = {"STATIC_PIF": sp}
    for b in sc_cfg["neighbour_budgets"]:
        r = fr4[np.isclose(fr4["es_budget"], b)]
        if len(r):
            bvec[f"BUDGET_{b:g}"] = np.array([float(r[f"x_{a}"].iloc[0]) if f"x_{a}" in r and pd.notna(r[f"x_{a}"].iloc[0]) else 0.0 for a in names])
    bw = {a: float(max(v[j] for v in bvec.values()) - min(v[j] for v in bvec.values())) for j, a in enumerate(names)}
    bstat = band_status(bw, True, sc_cfg["stable_max_band_width_nav"])
    for j, a in enumerate(names + ["LIQUIDITY", "NET_SMLT"]):
        vals = {k: (float(v[j]) if j < len(names) else book(assets, v)["liquidity_nav" if a == "LIQUIDITY" else "net_smlt_nav"]) for k, v in bvec.items()}
        band_rows.append({"scenario": "ROBUST_BASELINE", "asset": a, **{f"w_{k}": v for k, v in vals.items()},
                          "band_lo": min(vals.values()), "band_hi": max(vals.values()), "band_width": max(vals.values()) - min(vals.values()),
                          "scenario_status": bstat, "applicable_target": "STATIC_PIF", "labels": "ROBUST_BASELINE (TEST-validated, V4)"})
    # explicit candidates (Phase 7), all evaluated in every scenario on the INDEPENDENT-seed paths with the fund engine
    w_sp = float(sp[names.index("SMLT")])
    cands = {"CONTROL_INITIAL_PORTFOLIO": (x0, "benchmark"), "STATIC_PIF": (sp, "robust baseline"),
             "ZERO_SMLT_CASH": (x0 * 0, "ADVERSE candidate: sell all SMLT, hold liquidity (SMLT = 0 is allowed, never forced)")}
    adv = base["ADVERSE_CREDIT_EVENT"]["M"]
    for name, w_fix, note in [("ZERO_SMLT_OPTIMISED", 0.0, "ADVERSE candidate: SMLT = 0, other sleeves optimised on ADVERSE paths"),
                              ("PARTIAL_RETENTION_OPTIMISED", sc_cfg["partial_retention_share"] * w_sp,
                               f"ADVERSE candidate: SMLT = {sc_cfg['partial_retention_share']:.0%} of the baseline weight, rest optimised on ADVERSE paths")]:
        r = opt.solve(core.make_lp(ctx, assets, E, C, hz, fixed={"SMLT": w_fix}), [adv], "max_premium", budget=tb)
        if r["status"] == "optimal":
            cands[name] = (core.integer_portfolio(ctx, assets, r["x"], NAV0)[0], note)
    x_pr = x0 * 0
    x_pr[names.index("SMLT")] = sc_cfg["partial_retention_share"] * w_sp
    cands["PARTIAL_RETENTION_UNHEDGED"] = (core.integer_portfolio(ctx, assets, x_pr, NAV0)[0], "ADVERSE candidate: partial retention, no hedge")
    up = base["EVENT_DRIVEN_UPSIDE"]["M"]
    for s in sc_cfg["rerisk_smlt_share_of_initial"]:
        w = s * E / NAV0
        r = opt.solve(core.make_lp(ctx, assets, E, C, hz, fixed={"SMLT": w}), [up], "max_premium", budget=tb)
        if r["status"] == "optimal":
            cands[f"RERISK_{int(s * 100)}_OPTIMISED"] = (core.integer_portfolio(ctx, assets, r["x"], NAV0)[0],
                                                         f"UPSIDE candidate: SMLT = {s:.0%} of the initial RUB 10m, rest optimised on UPSIDE paths")
        xr = x0 * 0
        xr[names.index("SMLT")] = w
        if s < 1.0:
            cands[f"RERISK_{int(s * 100)}_UNHEDGED"] = (core.integer_portfolio(ctx, assets, xr, NAV0)[0], f"UPSIDE candidate: SMLT = {s:.0%} of initial, no hedge")
    for sc in SCENARIOS:
        if targets[sc][0] is not None:
            cands[f"TARGET_{sc}"] = (targets[sc][0], f"conditional target ({targets[sc][1]})")
    for cname, (x, note) in cands.items():
        for sc in SCENARIOS:
            fr_rows.append({"row_type": "CANDIDATE", "scenario": sc, "candidate": cname, "note": note,
                            "labels": "ROBUST_BASELINE" if cname == "STATIC_PIF" else LABELS, **_w(assets, x), **book(assets, x),
                            **{f"fund_{k}": v for k, v in fund_eval(ctx, alt[sc], assets, x, E, C).items()},
                            "fund_ES95_per_NAV": np.nan, "evaluation": "fund engine, INDEPENDENT-seed scenario paths, 6m"})
            fr_rows[-1]["fund_ES95_per_NAV"] = fr_rows[-1]["fund_ES95"] / NAV0
    ctx.save(pd.DataFrame(fr_rows), "pif_scenario_policy_frontiers.csv", index=False)
    ctx.save(pd.DataFrame(band_rows), "pif_scenario_target_bands.csv", index=False)


# ================================================================== s42  Phase 11 + 13
def _bond_close(b: pd.DataFrame, short: str, date: str) -> tuple[float, float]:
    r = b[(b["SHORTNAME"] == short) & (b["TRADEDATE"] == pd.Timestamp(date))]
    return (float(r["CLOSE"].iloc[0]), float(r["LEGALCLOSEPRICE"].iloc[0])) if len(r) else (np.nan, np.nan)


def s42_primary_sources(ctx) -> None:
    from ..data import load
    t = pd.Timestamp(ctx.cutoff)
    rows = []

    def add(fid, claim, sec_src, sec_val, prim_src, prim_val, status, used_in, action):
        rows.append({"check_id": fid, "claim": claim, "secondary_source": sec_src, "secondary_value": sec_val, "primary_source": prim_src,
                     "primary_value": prim_val, "status": status, "used_in": used_in, "action": action})

    # ---- Bank of Russia
    st_k, p_k = pr.owner_item(ctx.root, "key_rate")
    kr = pr.key_rate(p_k).loc[:t] if st_k == "OBTAINED" else pd.Series(dtype=float)
    if len(kr):
        ctx.save(kr.to_frame(), "cbr_key_rate.csv", "interim")
        ch = kr[kr.diff() != 0].iloc[1:]
        chg = kr.diff().loc[ch.index]
        last_hike = chg[chg > 0].index.max()
        cuts = chg[(chg < 0) & (chg.index > last_hike)]
        around = kr.loc["2026-09-01":"2026-09-25"]
        add("CBR_HOLD_20260911", "Bank of Russia held the key rate at 14.00% on 2026-09-11", "finance.mail.ru (media)", "14.00%",
            "Bank of Russia key-rate series (raw HTML, SHA-256 in manifest)", f"{around.min():.2f}%..{around.max():.2f}% on 2026-09-01..25; last change "
            f"{ch.index[-1].date()}", "CONFIRMED_PRIMARY" if around.nunique() == 1 and abs(around.iloc[0] - 14.0) < 1e-9 else "CORRECTED",
            "scenario rates axis narrative; event register", "event register uses the primary series")
        add("CBR_TEN_CUTS", "the hold followed ten consecutive cuts", "finance.mail.ru (media)", "10",
            "Bank of Russia key-rate series", f"{len(cuts)} cuts since the last hike ({last_hike.date()}): "
            + ", ".join(f"{d.date()} {v:+.2f}" for d, v in cuts.items()),
            "CONFIRMED_PRIMARY" if len(cuts) == 10 else "CORRECTED", "narrative only",
            "count confirmed; holds between cuts are not observable in a rate series (meeting calendar not obtained)")
        add("CBR_LEVEL_CUTOFF", "key rate at the cutoff", "V4 did not use the key rate (money-market input = LQDT 60d yield)", "n/a",
            "Bank of Russia key-rate series", f"{kr.iloc[-1]:.2f}% on {kr.index[-1].date()}", "CONFIRMED_PRIMARY", "decision card; event register",
            "used as the primary rate level")
    add("CBR_NEXT_MEETING", "next key-rate meeting 2026-10-23", "finance.mail.ru (media)", "2026-10-23", "Bank of Russia meeting calendar (not fetched)",
        "", "STILL_UNVERIFIED", "decision-card trigger", "kept as a MEDIA_REPORTED scheduled trigger")
    st_r, p_r = pr.owner_item(ctx.root, "ruonia")
    mi = pd.read_csv(ctx.out / "market_inputs_at_cutoff.csv", index_col=0)["value"]
    if st_r == "OBTAINED":
        ru = pr.ruonia(p_r)
        ctx.save(ru, "cbr_ruonia.csv", "interim")
        pub = ru[ru["publication_date"] <= t]
        r_last = pub.iloc[-1]
        r60 = float(pub["ruonia_pct"].iloc[-60:].mean())
        add("MM_CROSSCHECK", "money-market input: LQDT 60-day trailing yield", "MOEX ISS (exchange; V4 input)", f"{float(mi['mmf_yield_LQDT_60d']):.2%}",
            "RUONIA (Bank of Russia), last value PUBLISHED by the cutoff",
            f"{r_last['ruonia_pct']:.2f}% for {pub.index[-1].date()} (published {r_last['publication_date'].date()}); 60-day mean {r60:.2f}%",
            "MODEL_INPUT_CROSSCHECK", "V4 liquidity sleeve return", "V4 input unchanged; difference reported (fund yield is gross of fees and "
            "includes term premium)")
    # ---- FX route of the experimental BTC spot (Phase 11)
    st_u, _ = pr.owner_item(ctx.root, "usd_official")
    st_c, p_c = pr.owner_item(ctx.root, "cny_official")
    P = ctx.read("prices.csv", index_col=0, parse_dates=True)
    if st_c == "OBTAINED" and "CNYRUB_TOM" in P:
        cny = pr.cbr_fx(p_c).loc[:t]
        ctx.save(cny.to_frame(), "cbr_cny_official.csv", "interim")
        j = pd.concat([cny, P["CNYRUB_TOM"]], axis=1, join="inner").dropna()
        rel = (j["CNYRUB_TOM"] / j.iloc[:, 0] - 1).abs()
        rc = np.log(j).diff().dropna().corr().iloc[0, 1]
        add("FX_CNY_CROSSCHECK", "CNYRUB_TOM (MOEX) as the RUB leg of the experimental BTC spot", "MOEX ISS (exchange)", "CNYRUB_TOM close",
            "Bank of Russia official CNY rate (R01375)", f"{len(j)} common days; median |diff| {rel.median():.2%}, max {rel.max():.2%}; "
            f"daily log-return correlation {rc:.3f}", "MODEL_INPUT_CROSSCHECK", "BTC_SPOT_RUB construction", "exchange series kept")
    add("FX_USD_ROUTE", "BTC spot in RUB should use official USD/RUB", "V4 design note", "CNY/RUB proxy",
        "Bank of Russia official USD rate (R01235)", st_u, "STILL_UNVERIFIED" if st_u != "OBTAINED" else "CONFIRMED_PRIMARY",
        "BTC_SPOT_EXP (experimental)", "USD/RUB not available -> CNY/RUB proxy KEPT and labelled EXPERIMENTAL_APPROXIMATION; BTC sensitivity "
        "not rerun" if st_u != "OBTAINED" else "replace the proxy and rerun the BTC sensitivity")
    st_m, _ = pr.owner_item(ctx.root, "mortgage_stats_page")
    add("F017", "family-mortgage tightening; sales expected -40%+ y/y", "marketpower.pro (rank 7)", "qualitative", "Bank of Russia mortgage statistics",
        f"page {st_m}; it lists xlsx links only - the statistics files were not downloaded", "STILL_UNVERIFIED", "narrative only",
        "no V4 / V4.1 input uses this fact")
    # ---- rating agencies
    st_e, p_e = pr.owner_item(ctx.root, "expert_ra_esg_20260618")
    if st_e == "OBTAINED":
        txt = pr.html_text(p_e)
        ok = all(k in txt for k in ("Самолет", "ESG-A+", "отозвал", "18.06.2026"))
        add("ERA_ESG_20260618", "Expert RA affirmed ESG-A+ and withdrew the ESG rating (2026-06-18)", "Expert RA via search index", "ESG-A+, withdrawn",
            "Expert RA press release (raw HTML)", "ESG-A+ affirmed, outlook changed to 'developing', rating withdrawn, 18.06.2026" if ok else "text not found",
            "CONFIRMED_PRIMARY" if ok else "STILL_UNVERIFIED", "governance note (not a credit rating)",
            "primary adds the outlook change; not used by any model")
    for eid, claim in [("E01", "ACRA downgrade to A-(RU) stable (2025-12-24)"), ("E03", "ACRA ratings placed under observation (2026-02-06)"),
                       ("E04", "ACRA observation removed, A-(RU) affirmed (2026-03-02)")]:
        st_a, _ = pr.owner_item(ctx.root, "acra_press_6448")
        add(eid, claim, "search index / media", "see data/manual/event_register_input.csv", "ACRA press release", st_a, "STILL_UNVERIFIED",
            "FUNDAMENTAL_STATE rating signal; event register", "kept as SECONDARY_ONLY (cannot trigger an override)")
    # ---- issuer facts (issuer site 403/401, e-disclosure blocked)
    facts = pd.read_csv(ctx.root / "data" / "manual" / "issuer_facts.csv")
    for _, f in facts.iterrows():
        if f["fact_id"] == "F019":
            add("F019", "shares outstanding", "MOEX ISS", f"{f['value']}", "MOEX ISS security spec (raw snapshot)", f"{f['value']}", "CONFIRMED_PRIMARY",
                "market cap / dossier", "already VERIFIED_RAW")
            continue
        add(f["fact_id"], f"{f['variable']} = {f['value']} {f['unit'] if pd.notna(f['unit']) else ''}".strip(), f"{f['publisher']} (rank {f['source_rank']})",
            f["value"], "Samolet IFRS report / e-disclosure", "issuer site HTTP 403/401; e-disclosure not fetched", "STILL_UNVERIFIED",
            "FUNDAMENTAL_STATE" if f["category"] in ("financials", "debt") else "narrative", "secondary value retained, flagged SECONDARY_ONLY")
    # ---- media market claims re-checked against exchange data (MOEX ISS = primary exchange data)
    s, _m = load.shares(ctx.root, "SMLT", ctx.cutoff)
    lo = s.loc[pd.Timestamp("2026-07-17")]
    add("M_ATL_20260717", "SMLT all-time low RUB 205 on 2026-07-17", "Investing.com", "205", "MOEX ISS TQBR daily history",
        f"LOW {lo['LOW']:.1f}, CLOSE {lo['CLOSE']:.1f} on 2026-07-17; lowest LOW since 2022: {s['LOW'].min():.1f} on {s['LOW'].idxmin().date()}",
        "CONFIRMED_PRIMARY" if abs(lo["LOW"] - 205) < 0.05 and s["LOW"].min() >= 205 - 0.05 else "CORRECTED", "narrative only",
        "intraday low; the close was higher (history in the data starts 2022-01)")
    c0, c1 = float(s.loc[pd.Timestamp("2026-09-07"), "CLOSE"]), float(s.loc[pd.Timestamp("2026-09-17"), "CLOSE"])
    add("M_DROP_20260917", "SMLT -29.6% to RUB 260.6 by 2026-09-17", "Vedomosti", "260.6 (-29.6%)", "MOEX ISS TQBR daily history",
        f"CLOSE {c1:.1f} on 2026-09-17 ({c1 / c0 - 1:+.1%} vs 2026-09-07 close {c0:.1f}); LOW {float(s.loc[pd.Timestamp('2026-09-17'), 'LOW']):.1f}",
        "CONFIRMED_PRIMARY" if abs(c1 - 260.6) < 0.5 else "CORRECTED", "narrative only", "exchange close reported; media figure is not a close")
    b = pd.read_csv(ctx.interim / "smlt_bonds_history.csv", parse_dates=["TRADEDATE"])
    for bid, short, lim in [("M_P14", "СамолетP14", 67.0), ("M_P18", "СамолетP18", 72.0), ("M_P13", "СамолетP13", 83.0)]:
        cl, lc = _bond_close(b, short, "2026-09-18")
        ok = np.nanmin([cl, lc]) < lim
        add(bid, f"{short} below {lim:.0f}% of par on 2026-09-18", "smart-lab (rank 7)", f"<{lim:.0f}", "MOEX ISS bonds history",
            f"CLOSE {cl:.2f}, LEGALCLOSE {lc:.2f}", "CONFIRMED_PRIMARY" if ok else "CORRECTED", "narrative only (CREDIT_STATE uses exchange data directly)",
            "exchange closes reported")
    cat = pd.read_csv(ctx.out / "forward_catalysts.csv")
    p20 = cat[cat["event"].str.contains("P20", na=False)]
    add("M_P20_MATURITY", "P20 redemption scheduled 2026-09-28", "smart-lab (rank 7)", "2026-09-28", "MOEX ISS bond static terms (MATDATE)",
        str(p20["date"].iloc[0]) if len(p20) else "not found", "CONFIRMED_PRIMARY" if len(p20) and p20["date"].iloc[0] == "2026-09-28" else "STILL_UNVERIFIED",
        "decision-card trigger", "exchange static term")
    ctx.save(pd.DataFrame(rows), "pif_primary_source_delta.csv", index=False)


# ================================================================== s43  Phase 5 + 6
EVENT_TYPES = ["FINANCIAL_REPORT", "RATING_CHANGE", "BOND_PAYMENT_OR_MATURITY", "PAYMENT_FAILURE", "REFINANCING", "RESTRUCTURING",
               "STATE_SUPPORT", "STRATEGIC_INVESTOR_OR_MA", "SPO_OR_DILUTION", "MORTGAGE_POLICY_CHANGE", "CBR_RATE_DECISION"]
# (event_type, action) -> (direction, materiality, affected risk channel). Fixed rules, not scores.
EVENT_RULES = {
    ("RATING_CHANGE", "DOWNGRADE"): ("NEGATIVE", "HIGH", "CREDIT"), ("RATING_CHANGE", "UNDER_REVIEW"): ("NEGATIVE", "MEDIUM", "CREDIT"),
    ("RATING_CHANGE", "AFFIRMED"): ("NEUTRAL", "LOW", "CREDIT"), ("RATING_CHANGE", "UPGRADE"): ("POSITIVE", "HIGH", "CREDIT"),
    ("RATING_CHANGE", "DEFAULT_RATING"): ("NEGATIVE", "SEVERE", "CREDIT"),
    ("STATE_SUPPORT", "REQUEST_REPORTED"): ("NEGATIVE", "MEDIUM", "CREDIT+EQUITY"), ("STATE_SUPPORT", "DENIED_REPORTED"): ("NEGATIVE", "MEDIUM", "CREDIT+EQUITY"),
    ("STATE_SUPPORT", "GRANTED"): ("POSITIVE", "HIGH", "CREDIT+EQUITY"),
    ("FINANCIAL_REPORT", "PUBLISHED"): ("FROM_FUNDAMENTAL_RULE", "MEDIUM", "EQUITY+CREDIT"),
    ("CBR_RATE_DECISION", "CUT"): ("POSITIVE", "MEDIUM", "RATES+SECTOR_DEMAND"), ("CBR_RATE_DECISION", "HIKE"): ("NEGATIVE", "MEDIUM", "RATES+SECTOR_DEMAND"),
    ("CBR_RATE_DECISION", "HOLD"): ("NEUTRAL", "LOW", "RATES+SECTOR_DEMAND"), ("CBR_RATE_DECISION", "SCHEDULED_MEETING"): ("UNKNOWN", "MEDIUM", "RATES+SECTOR_DEMAND"),
    ("BOND_PAYMENT_OR_MATURITY", "SCHEDULED"): ("UNKNOWN", "HIGH", "LIQUIDITY+CREDIT"), ("BOND_PAYMENT_OR_MATURITY", "PAID"): ("POSITIVE", "LOW", "LIQUIDITY+CREDIT"),
    ("PAYMENT_FAILURE", "MISSED_PAYMENT"): ("NEGATIVE", "SEVERE", "CREDIT+EQUITY"), ("PAYMENT_FAILURE", "TECHNICAL_DEFAULT"): ("NEGATIVE", "SEVERE", "CREDIT+EQUITY"),
    ("RESTRUCTURING", "TALKS_REPORTED"): ("NEGATIVE", "HIGH", "EQUITY+CREDIT"), ("RESTRUCTURING", "ANNOUNCED_BY_ISSUER"): ("NEGATIVE", "SEVERE", "EQUITY+CREDIT"),
    ("REFINANCING", "COMPLETED"): ("POSITIVE", "MEDIUM", "LIQUIDITY+CREDIT"), ("REFINANCING", "FAILED"): ("NEGATIVE", "HIGH", "LIQUIDITY+CREDIT"),
    ("STRATEGIC_INVESTOR_OR_MA", "TALKS_REPORTED"): ("UNKNOWN", "MEDIUM", "EQUITY"), ("STRATEGIC_INVESTOR_OR_MA", "SIGNED"): ("POSITIVE", "HIGH", "EQUITY+CREDIT"),
    ("SPO_OR_DILUTION", "ANNOUNCED"): ("NEGATIVE", "HIGH", "EQUITY"),
    ("MORTGAGE_POLICY_CHANGE", "TIGHTENING"): ("NEGATIVE", "MEDIUM", "SECTOR_DEMAND"), ("MORTGAGE_POLICY_CHANGE", "EASING"): ("POSITIVE", "MEDIUM", "SECTOR_DEMAND"),
}
VERIFIED = ("CONFIRMED_PRIMARY", "CONFIRMED_EXCHANGE_DATA")


def credit_state(cs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Daily CREDIT_STATE from market evidence only (SEVERE is added later from VERIFIED events, never from prices)."""
    ytm = cs["ytm_median_pct"].ffill()
    d20 = ytm - ytm.shift(20)
    distressed = (cs["zspread_median_pp"] >= cfg["distressed_zspread_pp"]) | (cs["price_median_pct_par"] <= cfg["distressed_price_pct_par"])
    unknown = cs["n_bonds"].fillna(0) < cfg["min_bonds"]
    st = np.select([unknown, distressed | (d20 >= cfg["ytm_change_20d_pp"]), d20 <= -cfg["ytm_change_20d_pp"]],
                   ["UNKNOWN", "DETERIORATING", "IMPROVING"], "STABLE")
    return pd.DataFrame({"credit_state": st, "zspread_median_pp": cs["zspread_median_pp"], "price_median_pct_par": cs["price_median_pct_par"],
                         "ytm_median_pct": cs["ytm_median_pct"], "ytm_change_20d_pp": d20, "n_bonds": cs["n_bonds"], "distressed": distressed}, index=cs.index)


def fundamental_signals(facts: pd.DataFrame, comps: pd.DataFrame, events: pd.DataFrame, t: pd.Timestamp, cfg: dict) -> dict:
    """Point-in-time issuer signals from the latest report PUBLISHED by t (+ rating actions known by t)."""
    f = facts[pd.to_datetime(facts["publication_date"], errors="coerce") <= t]
    v = {r["variable"]: r["value"] for _, r in f.iterrows()}
    c = {r["variable"]: float(r["value"]) for _, r in comps[pd.to_datetime(comps["publication_date"], errors="coerce") <= t].iterrows()}
    out = {"report": None, "report_period_end": None, "signals": {}, "source_status": "SECONDARY_ONLY (issuer primary not obtained)"}
    rev = [k for k in v if k.startswith("revenue_")]
    if not rev:
        return out
    k = sorted(rev)[-1]
    period = k.split("_")[-1]
    as_of = pd.to_datetime(f.loc[f["variable"] == k, "as_of"].iloc[0])
    out["report"], out["report_period_end"] = period, str(as_of.date())
    out["report_age_days"] = int((t - as_of).days)
    sig = {}
    prior = c.get(k.replace("2026", "2025"))
    if prior:
        g = float(v[k]) / prior - 1
        sig["revenue_yoy"] = (g, "NEG" if g <= -cfg["revenue_yoy"] else "POS" if g >= cfg["revenue_yoy"] else "0")
    ni = v.get(f"net_income_{period}")
    if ni is not None:
        ni, pni = float(ni), c.get(f"net_income_{period.replace('2026', '2025')}")
        sig["net_income"] = (ni, "NEG" if ni < 0 else "POS" if pni is not None and ni > pni else "0")
    lev = v.get("net_debt_ex_escrow_to_adj_ebitda")
    if lev is not None:
        lev, pl_ = float(lev), c.get("net_debt_ex_escrow_to_adj_ebitda_prior")
        dl = lev - pl_ if pl_ is not None else np.nan
        sig["leverage_nd_ex_escrow_adj_ebitda"] = (lev, "NEG" if lev >= cfg["leverage_level_x"] or dl >= cfg["leverage_change_x"] else
                                                   "POS" if dl <= -cfg["leverage_change_x"] else "0")
    rt = events[(events["event_type"] == "RATING_CHANGE") & (pd.to_datetime(events["publicly_known_at"]) <= t) &
                (pd.to_datetime(events["publicly_known_at"]) > t - pd.Timedelta(days=cfg["rating_lookback_days"]))]
    acts = set(rt["action"])
    sig["rating_12m"] = (",".join(sorted(acts)) or "none", "NEG" if acts & {"DOWNGRADE", "DEFAULT_RATING"} else "POS" if "UPGRADE" in acts else "0")
    out["signals"] = sig
    return out


def fundamental_state(sig: dict, severe: bool, cfg: dict) -> str:
    if severe:
        return "SEVERE"
    if sig["report"] is None or sig.get("report_age_days", 10**6) > cfg["max_report_age_days"]:
        return "UNKNOWN"
    neg = sum(s == "NEG" for _, s in sig["signals"].values())
    pos = sum(s == "POS" for _, s in sig["signals"].values())
    if neg >= cfg["min_signals"] and neg > pos:
        return "DETERIORATING"
    if pos >= cfg["min_signals"] and pos > neg:
        return "IMPROVING"
    return "STABLE"


def event_register(ctx, t: pd.Timestamp) -> pd.DataFrame:
    rows = []
    inp = pd.read_csv(ctx.root / "data" / "manual" / "event_register_input.csv")
    for _, r in inp.iterrows():
        rows.append({**r.to_dict(), "status": "PAST"})
    kr_p = ctx.interim / "cbr_key_rate.csv"
    if kr_p.exists():
        kr = pd.read_csv(kr_p, index_col=0, parse_dates=True)["key_rate_pct"]
        ch = kr[kr.diff() != 0].iloc[1:]
        for d, v in ch.items():
            dv = float(v - kr.shift(1).loc[d])
            rows.append({"event_id": f"CBR_{d.date()}", "event_type": "CBR_RATE_DECISION", "action": "CUT" if dv < 0 else "HIKE",
                         "event_date": str(d.date()), "publicly_known_at": str(d.date()), "source_publisher": "Bank of Russia key-rate series",
                         "source_url": "https://www.cbr.ru/hd_base/KeyRate/", "source_rank": 3, "verification_status": "CONFIRMED_PRIMARY",
                         "source_ref": "data/raw/Bank_of_Russia/key_rate", "note": f"{v:.2f}% ({dv:+.2f} pp); date = effective date (announced at the "
                         "preceding board meeting; the effective date is a conservative publicly-known date)", "status": "PAST"})
        held = kr.loc["2026-09-10":"2026-09-15"]
        rows.append({"event_id": "CBR_2026-09-11_HOLD", "event_type": "CBR_RATE_DECISION", "action": "HOLD", "event_date": "2026-09-11",
                     "publicly_known_at": "2026-09-11", "source_publisher": "media (finance.mail.ru); level checked against the primary series",
                     "source_url": "https://finance.mail.ru/article/bank-rossii-ostavil-klyuchevuyu-stavku-na-otmetke-14-69227312/", "source_rank": 3,
                     "verification_status": "CONFIRMED_PRIMARY" if held.nunique() == 1 else "MEDIA_ONLY", "source_ref": "event_chronology.csv 2026-09-11",
                     "note": "rate unchanged at 14.00% across the meeting in the primary series", "status": "PAST"})
    cat = pd.read_csv(ctx.out / "forward_catalysts.csv")
    for _, r in cat.iterrows():
        if r["category"] in ("BOND_MATURITY", "BOND_OFFER"):
            rows.append({"event_id": f"BOND_{r['date']}_{str(r['event']).split()[-1].strip('()')}", "event_type": "BOND_PAYMENT_OR_MATURITY",
                         "action": "SCHEDULED", "event_date": r["date"], "publicly_known_at": ctx.cutoff, "source_publisher": "MOEX ISS bond static terms",
                         "source_url": "https://iss.moex.com", "source_rank": 2, "verification_status": "CONFIRMED_EXCHANGE_DATA",
                         "source_ref": "outputs/forward_catalysts.csv", "note": f"{r['event']}; last price {r.get('last_price_pct_par')}% of par"
                         + (f", YTM {r.get('last_ytm_pct')}%" if pd.notna(r.get("last_ytm_pct")) else ""), "status": "SCHEDULED"})
        elif r["category"] == "CBR_MEETING":
            rows.append({"event_id": f"CBR_MEETING_{r['date']}", "event_type": "CBR_RATE_DECISION", "action": "SCHEDULED_MEETING",
                         "event_date": r["date"], "publicly_known_at": "2026-09-11", "source_publisher": str(r["source"]).split(" (")[0],
                         "source_url": "", "source_rank": 7, "verification_status": "MEDIA_ONLY", "source_ref": "outputs/forward_catalysts.csv",
                         "note": "meeting date media-reported; calendar not fetched", "status": "SCHEDULED"})
    E = pd.DataFrame(rows)
    rule = [EVENT_RULES.get((a, b), ("UNKNOWN", "MEDIUM", "UNKNOWN")) for a, b in zip(E["event_type"], E["action"])]
    E["direction"], E["materiality"], E["affected_risk_channel"] = zip(*rule)
    E["known_at_cutoff"] = pd.to_datetime(E["publicly_known_at"]) <= t
    E["can_trigger_override"] = (E["materiality"] == "SEVERE") & E["verification_status"].isin(VERIFIED)
    return E.sort_values(["event_date", "event_id"]).reset_index(drop=True)


def classify(t: pd.Timestamp, credit: str, fundamental: str, pstate: str, E: pd.DataFrame, kr: pd.Series | None, cfg: dict) -> tuple[str, str, list]:
    """State hierarchy -> (conditional scenario, rule fired, REVIEW flags). First matching rule wins."""
    known = E[(pd.to_datetime(E["publicly_known_at"]) <= t) & (E["status"] == "PAST")]
    kn = pd.to_datetime(known["publicly_known_at"])
    win = kn > t - pd.Timedelta(days=cfg["hierarchy"]["upside_event_window_days"])
    flags = []
    rec = known[(kn > t - pd.Timedelta(days=cfg["decision"]["recent_event_days"])) & known["materiality"].isin(["HIGH", "SEVERE"])
                & ~known["verification_status"].isin(VERIFIED)]
    flags += [f"UNVERIFIED_{r.materiality}_EVENT {r.event_id} {r.event_type}/{r.action} ({r.event_date})" for r in rec.itertuples()]
    flags += [f"{n}_STATE_UNKNOWN" for n, s in (("CREDIT", credit), ("FUNDAMENTAL", fundamental)) if s == "UNKNOWN"]
    if (known["can_trigger_override"]).any() or credit == "SEVERE" or fundamental == "SEVERE":
        return "ADVERSE_CREDIT_EVENT", "R1_SEVERE_EVENT_OVERRIDE", flags
    if ((known["direction"] == "POSITIVE") & (known["materiality"] == "HIGH") & known["verification_status"].isin(VERIFIED) & win).any():
        return "EVENT_DRIVEN_UPSIDE", "R2_VERIFIED_POSITIVE_EVENT", flags
    if "DETERIORATING" in (credit, fundamental):
        return "CONTINUED_DETERIORATION", "R3_FUNDAMENTAL_OR_CREDIT_DETERIORATING", flags
    if pstate in ("P_CREDIT_STRESS", "P_COMBINED_STRESS"):
        return "CONTINUED_DETERIORATION", "R4_MACRO_CREDIT_STRESS_STATE", flags
    if pstate == "P_RECOVERY" and kr is not None:
        k = kr.loc[:t]
        cuts = k.diff().loc[t - pd.Timedelta(days=cfg["hierarchy"]["easing_lookback_days"]):]
        if (cuts < 0).any():
            return "RATES_RELIEF_RECOVERY", "R5_RECOVERY_WITH_RATE_EASING", flags
    return "STABILISATION_FLAT", "R6_DEFAULT", flags


MAPPING_RULES = [
    (1, "L1 SEVERE EVENT OVERRIDE", "R1_SEVERE_EVENT_OVERRIDE", "a VERIFIED event of materiality SEVERE is publicly known (payment failure / "
     "technical default, restructuring announced by the issuer, default rating)", "event register", "ADVERSE_CREDIT_EVENT"),
    (2, "L1 VERIFIED POSITIVE EVENT", "R2_VERIFIED_POSITIVE_EVENT", "a VERIFIED POSITIVE event of materiality HIGH became public within the event "
     "window (state support granted, strategic investor signed, rating upgrade)", "event register", "EVENT_DRIVEN_UPSIDE"),
    (3, "L2 FUNDAMENTAL / CREDIT", "R3_FUNDAMENTAL_OR_CREDIT_DETERIORATING", "CREDIT_STATE = DETERIORATING or FUNDAMENTAL_STATE = DETERIORATING",
     "MOEX Samolet bonds (Z-spread, price, YTM); issuer facts by publication date; rating actions", "CONTINUED_DETERIORATION"),
    (4, "L3 MACRO + SMLT PRICE", "R4_MACRO_CREDIT_STRESS_STATE", "portfolio state P_CREDIT_STRESS or P_COMBINED_STRESS (V3 causal classifier)",
     "IMOEX, bond index, RGBI, SMLT price", "CONTINUED_DETERIORATION"),
    (5, "L3 MACRO + SMLT PRICE", "R5_RECOVERY_WITH_RATE_EASING", "portfolio state P_RECOVERY and a key-rate cut effective within the easing "
     "look-back (primary Bank of Russia series)", "V3 classifier; key-rate series", "RATES_RELIEF_RECOVERY"),
    (6, "L3 DEFAULT", "R6_DEFAULT", "none of the above (P_NORMAL, P_DOWNTREND, P_MARKET_STRESS, or P_RECOVERY without easing)",
     "-", "STABILISATION_FLAT"),
]


def s43_state_events(ctx) -> None:
    v41 = ctx.cfg["v41"]
    t = pd.Timestamp(ctx.cutoff)
    E = event_register(ctx, t)
    ctx.save(E, "pif_event_register.csv", index=False)
    cs = ctx.read("credit_state.csv", index_col=0, parse_dates=True).loc[:t]
    cst = credit_state(cs, v41["credit_state"])
    facts = pd.read_csv(ctx.root / "data" / "manual" / "issuer_facts.csv")
    comps = pd.read_csv(ctx.root / "data" / "manual" / "issuer_facts_comparatives.csv")
    reg = pd.read_csv(ctx.interim / "regime_states_daily.csv", index_col=0, parse_dates=True)
    kr_p = ctx.interim / "cbr_key_rate.csv"
    kr = pd.read_csv(kr_p, index_col=0, parse_dates=True)["key_rate_pct"] if kr_p.exists() else None
    # daily history (fundamentals only exist from the first report published in the data -> UNKNOWN before)
    hist = []
    for d in reg.index[reg.index <= t]:
        sg = fundamental_signals(facts, comps, E, d, v41["fundamental_state"])
        known = E[(pd.to_datetime(E["publicly_known_at"]) <= d) & (E["status"] == "PAST")]
        severe = bool(known["can_trigger_override"].any())
        c = "SEVERE" if severe else (cst.loc[d, "credit_state"] if d in cst.index else "UNKNOWN")
        fnd = fundamental_state(sg, severe, v41["fundamental_state"])
        sc, rule, _ = classify(d, c, fnd, reg.loc[d, "portfolio_state"], E, kr, v41)
        hist.append({"date": d, "credit_state": c, "fundamental_state": fnd, "portfolio_state": reg.loc[d, "portfolio_state"],
                     "conditional_scenario": sc, "rule": rule})
    H = pd.DataFrame(hist).set_index("date")
    ctx.save(H, "pif_observable_state_daily.csv", "interim")
    now = H.iloc[-1]
    sig = fundamental_signals(facts, comps, E, t, v41["fundamental_state"])
    sc, rule, flags = classify(t, now["credit_state"], now["fundamental_state"], now["portfolio_state"], E, kr, v41)
    ev = {"credit": f"CREDIT_STATE={now['credit_state']} (Z-spread {cst['zspread_median_pp'].iloc[-1]:.1f} pp, median price "
                    f"{cst['price_median_pct_par'].iloc[-1]:.1f}% of par, YTM {cst['ytm_median_pct'].iloc[-1]:.1f}%, 20d YTM change "
                    f"{cst['ytm_change_20d_pp'].iloc[-1]:+.1f} pp, {int(cst['n_bonds'].iloc[-1])} bonds)",
          "fundamental": f"FUNDAMENTAL_STATE={now['fundamental_state']} ({sig['report']} report; " +
                         "; ".join(f"{k}: {v[0] if isinstance(v[0], str) else round(v[0], 3)} -> {v[1]}" for k, v in sig["signals"].items()) + ")",
          "macro": f"portfolio state {now['portfolio_state']} (macro {reg['macro_regime'].iloc[-1]}, price {reg['price_state'].iloc[-1]})"}
    fires = {"R1_SEVERE_EVENT_OVERRIDE": False, "R2_VERIFIED_POSITIVE_EVENT": False,
             "R3_FUNDAMENTAL_OR_CREDIT_DETERIORATING": "DETERIORATING" in (now["credit_state"], now["fundamental_state"]),
             "R4_MACRO_CREDIT_STRESS_STATE": now["portfolio_state"] in ("P_CREDIT_STRESS", "P_COMBINED_STRESS"),
             "R5_RECOVERY_WITH_RATE_EASING": False, "R6_DEFAULT": True}
    fires["R1_SEVERE_EVENT_OVERRIDE"] = rule == "R1_SEVERE_EVENT_OVERRIDE"
    fires["R2_VERIFIED_POSITIVE_EVENT"] = rule == "R2_VERIFIED_POSITIVE_EVENT"
    fires["R5_RECOVERY_WITH_RATE_EASING"] = rule == "R5_RECOVERY_WITH_RATE_EASING"
    m = []
    for pri, lvl, rid, cond, inputs, maps in MAPPING_RULES:
        m.append({"priority": pri, "level": lvl, "rule_id": rid, "condition": cond, "observable_inputs": inputs, "maps_to_scenario": maps,
                  "condition_true_at_cutoff": bool(fires[rid]), "selected_at_cutoff": rid == rule,
                  "evidence_at_cutoff": {"R1_SEVERE_EVENT_OVERRIDE": "no VERIFIED SEVERE event in the register",
                                         "R2_VERIFIED_POSITIVE_EVENT": "no VERIFIED POSITIVE HIGH event in the window",
                                         "R3_FUNDAMENTAL_OR_CREDIT_DETERIORATING": ev["credit"] + "; " + ev["fundamental"],
                                         "R4_MACRO_CREDIT_STRESS_STATE": ev["macro"], "R5_RECOVERY_WITH_RATE_EASING": ev["macro"],
                                         "R6_DEFAULT": "fallback"}[rid],
                  "labels": LABELS})
    ctx.save(pd.DataFrame(m), "pif_observable_state_mapping.csv", index=False)
    snap = {"date": ctx.cutoff, "conditional_scenario": sc, "rule": rule, "review_flags": flags, "credit_state": now["credit_state"],
            "fundamental_state": now["fundamental_state"], "portfolio_state": now["portfolio_state"], "macro_regime": reg["macro_regime"].iloc[-1],
            "price_state": reg["price_state"].iloc[-1], "fundamental_signals": {k: [v[0], v[1]] for k, v in sig["signals"].items()},
            "fundamental_source_status": sig["source_status"], "evidence": ev,
            "credit_state_share_of_history": H["credit_state"].value_counts(normalize=True).round(4).to_dict(),
            "scenario_share_of_history": H["conditional_scenario"].value_counts(normalize=True).round(4).to_dict(),
            "key_rate_pct": float(kr.iloc[-1]) if kr is not None else None}
    (ctx.interim / "pif_state_at_cutoff.json").write_text(json.dumps(snap, indent=1, default=str))


# ================================================================== s44  Phase 10
def s44_premium(ctx) -> None:
    v4, pc = ctx.cfg["v4"], ctx.cfg["v41"]["premium_sensitivity"]
    E, C = _EC(ctx)
    NAV0 = E + C
    assets = uni.build(ctx, "WITHOUT_BTC", NAV0, E / NAV0)
    fac = core.factory(ctx, assets)
    Ms = core.lp_models(fac, assets, 126, v4["n_lp"], "TRAIN")
    Mv = core.lp_models(fac, assets, 126, v4["n_val"], "VAL")
    lp = core.make_lp(ctx, assets, E, C, 126)
    rows = []

    def rec(tag, prem, b, r):
        if r["status"] != "optimal":
            rows.append({"objective": tag, "smlt_premium_pa": prem, "es_budget": b, "status": r["status"]})
            return
        x = r["x"]
        d = opt.diagnostics(lp, Mv, x)
        rows.append({"objective": tag, "smlt_premium_pa": prem, "es_budget": b, "status": "OPTIMAL", **_w(assets, x), **book(assets, x),
                     "SMLT_share_of_initial": float(x[0] * NAV0 / E), "train_worst_ES95_nav": r["worst_es"],
                     "train_min_participation": r["min_participation"], "val_worst_ES95_nav": d["worst_es"],
                     "val_min_participation": d["min_participation"], "val_mean_pnl_rub_no_premium": float(np.mean(d["mean_by_model"]) * NAV0),
                     "assumed_premium_contribution_rub": float(x[0] * prem * 0.5 * NAV0),
                     "label": "SENSITIVITY: assumed premium, never estimated" if tag == "max_premium" else "V4 objective (no return view)"})
    for b in pc["es_budgets"]:
        rec("max_participation (V4 objective, reference)", np.nan, b, opt.solve(lp, Ms, "max_participation", budget=b))
        for prem in pc["smlt_premium_grid"]:
            premia = np.array([prem if a.name == "SMLT" else 0.0 for a in assets])
            rec("max_premium", prem, b, opt.solve(lp, Ms, "max_premium", budget=b, premia=premia))
    # diagnostic: the drift-neutralised bootstrap still implies a horizon SMLT mean excess (serial dependence / FHS scaling)
    for mdl, M in zip(core.MODELS, Ms):
        ex = float(M.R[:, 0].mean() - M.m)
        rows.append({"objective": "DIAGNOSTIC: implicit SMLT horizon excess in the TRAIN model (no premium added)", "model": mdl,
                     "implicit_excess_6m": ex, "implicit_excess_pa_approx": float((1 + ex / (1 + M.m)) ** 2 - 1),
                     "smlt_median_excess_6m": float(np.median(M.R[:, 0]) - M.m), "status": "DIAGNOSTIC"})
    ctx.save(pd.DataFrame(rows), "pif_smlt_premium_sensitivity.csv", index=False)


# ================================================================== s45  Phase 9
def _band(bands: pd.DataFrame, scenario: str) -> dict:
    g = bands[bands["scenario"] == scenario]
    return {r["asset"]: [round(float(r["band_lo"]), 4), round(float(r["band_hi"]), 4)] for _, r in g.iterrows()}


def decide(cur_smlt: float, band_smlt: tuple[float, float], rule: str, flags: list, tol: float) -> tuple[str, list]:
    """Pre-declared action rule. EXIT only when the applicable band is SMLT = 0; weights outside the band dominate REVIEW."""
    lo, hi = band_smlt
    why = []
    if hi <= tol and cur_smlt > tol:
        why.append(f"applicable band sets SMLT to 0 (rule {rule})")
        return "EXIT", why
    if cur_smlt > hi + tol:
        why.append(f"SMLT {cur_smlt:.1%} of NAV is above the band [{lo:.1%}, {hi:.1%}] (+{tol:.0%} tolerance)")
        return "REDUCE", why
    if cur_smlt < lo - tol:
        why.append(f"SMLT {cur_smlt:.1%} of NAV is below the band [{lo:.1%}, {hi:.1%}] (-{tol:.0%} tolerance)")
        return "INCREASE", why
    if flags:
        why.append("inside the band, but review flags are open")
        return "REVIEW", why
    why.append("inside the band, no open review flags")
    return "HOLD", why


def s45_decision_card(ctx) -> None:
    v41 = ctx.cfg["v41"]
    E, C = _EC(ctx)
    NAV0 = E + C
    t = pd.Timestamp(ctx.cutoff)
    st = json.loads((ctx.interim / "pif_state_at_cutoff.json").read_text())
    bands = pd.read_csv(ctx.out / "pif_scenario_target_bands.csv")
    frt = pd.read_csv(ctx.out / "pif_scenario_policy_frontiers.csv")
    fc = pd.read_csv(ctx.out / "pif_final_comparison.csv").set_index("strategy")
    E_reg = pd.read_csv(ctx.out / "pif_event_register.csv")
    sc = st["conditional_scenario"]
    sstat = bands.loc[bands["scenario"] == sc, "scenario_status"].iloc[0]
    applicable = sc if sstat == "STABLE" else "ROBUST_BASELINE"
    b_app = _band(bands, applicable)
    assets = scenario_assets(ctx, E, C)
    names = [a.name for a in assets]
    S0 = float(ctx.read("prices.csv", index_col=0, parse_dates=True)["SMLT"].loc[ctx.cutoff])
    x_cur = uni.initial_weights(assets, E, C)
    sp_assets, sp_x = stages.static_book(ctx)["STATIC_PIF"]
    x_sp = stages._embed(sp_assets, sp_x, assets)
    if applicable == "ROBUST_BASELINE":
        x_tgt = x_sp
    else:
        g = bands[(bands["scenario"] == sc) & bands["asset"].isin(names)].set_index("asset")
        x_tgt = np.array([float(g.loc[a, "w_INTEGER"]) for a in names])
    action, why = decide(float(x_cur[names.index("SMLT")]), tuple(b_app["SMLT"]), st["rule"], st["review_flags"],
                         v41["decision"]["weight_tolerance_nav"])
    _, rep = core.integer_portfolio(ctx, assets, x_tgt, NAV0)
    units = {r["asset"]: int(r["units"]) for r in rep}
    trades = {"SMLT_shares": int(units.get("SMLT", 0) - round(E / S0))}
    trades.update({f"{a}_contracts": int(units[a]) for a in units if a != "SMLT"})
    horizon = t + pd.Timedelta(days=v41["decision"]["watch_horizon_days"])
    sched = E_reg[(E_reg["status"] == "SCHEDULED") & (pd.to_datetime(E_reg["event_date"]) > t) & (pd.to_datetime(E_reg["event_date"]) <= horizon)]
    trig = [f"{r.event_date}: {r.event_type}/{r.action} - {r.note if isinstance(r.note, str) else ''} [{r.verification_status}]".strip() for r in sched.itertuples()]
    trig += [f"open review flag: {f}" for f in st["review_flags"]]
    cc = v41["credit_state"]
    trig += [f"CREDIT_STATE leaves DETERIORATING (Z-spread < {cc['distressed_zspread_pp']:g} pp and median price > "
             f"{cc['distressed_price_pct_par']:g}% of par and 20d YTM change < +{cc['ytm_change_20d_pp']:g} pp)",
             "any VERIFIED payment failure / issuer-announced restructuring -> R1 override (ADVERSE_CREDIT_EVENT)",
             "any VERIFIED state support granted / strategic investor signed / upgrade -> R2 (EVENT_DRIVEN_UPSIDE)",
             "issuer 9M/FY2026 IFRS publication (date not available) -> FUNDAMENTAL_STATE recomputed",
             f"one-day SMLT move >= {ctx.cfg['v4']['state_machine']['emergency_shock']:.0%} -> emergency review (V4 state-machine threshold)"]
    cand = frt[(frt["row_type"] == "CANDIDATE") & (frt["scenario"] == sc)].set_index("candidate")
    risk = {"horizon": "6m", "units": "RUB",
            "current_portfolio_TEST_ES95_FHS": float(fc.loc["CONTROL_INITIAL_PORTFOLIO", "ES95_6m"]),
            "robust_baseline_TEST_ES95_FHS": float(fc.loc["STATIC_PIF", "ES95_6m"]),
            "robust_baseline_TEST_participation": float(fc.loc["STATIC_PIF", "participation_6m"]),
            f"current_portfolio_ES95_in_{sc}": float(cand.loc["CONTROL_INITIAL_PORTFOLIO", "fund_ES95"]),
            f"robust_baseline_ES95_in_{sc}": float(cand.loc["STATIC_PIF", "fund_ES95"]),
            f"current_portfolio_mean_pnl_in_{sc}": float(cand.loc["CONTROL_INITIAL_PORTFOLIO", "fund_mean_pnl"]),
            f"robust_baseline_mean_pnl_in_{sc}": float(cand.loc["STATIC_PIF", "fund_mean_pnl"]),
            "note": "TEST = V4 untouched TEST sets (bootstrap models); scenario rows = conditional forward paths (not forecasts)"}
    nxt = pd.bdate_range(t + pd.Timedelta(days=1), periods=1)[0]
    card = {
        "DATE": ctx.cutoff,
        "LABELS": ["DETERMINISTIC_RULES", "CONDITIONAL_POLICY", "SCENARIO_NOT_FORECAST", "NOT_OOS_ALPHA_CLAIM", "NO_SCENARIO_PROBABILITIES"],
        "CURRENT_OBSERVABLE_STATE": {k: st[k] for k in ("portfolio_state", "macro_regime", "price_state", "credit_state", "fundamental_state",
                                                          "fundamental_source_status", "key_rate_pct")} | {"evidence": st["evidence"],
                                                                                                        "review_flags": st["review_flags"]},
        "CONDITIONAL_SCENARIO": {"name": sc, "rule_fired": st["rule"], "scenario_band_status": sstat,
                                 "labels": LABELS},
        "CURRENT_PORTFOLIO": {"SMLT_shares": int(round(E / S0)), "SMLT_price": S0, "NAV0_rub": NAV0,
                              "weights_nav": {a: round(float(v), 4) for a, v in zip(names, x_cur) if abs(v) > 1e-9} | {"LIQUIDITY": round(float(1 - x_cur[0]), 4)}},
        "ROBUST_BASELINE_PORTFOLIO": {"name": "STATIC_PIF", "weights_nav": {a: round(float(v), 4) for a, v in zip(names, x_sp) if abs(v) > 1e-9}
                                      | {"LIQUIDITY": round(book(assets, x_sp)["liquidity_nav"], 4)},
                                      "conditional_on": "C = RUB 10m, NAV0 = RUB 20m, worst-model ES95(6m) <= 15% of NAV, recovery-participation "
                                                        "objective, no excess-return view", "band": _band(bands, "ROBUST_BASELINE")},
        "SCENARIO_TARGET_BAND": {"scenario": sc, "status": sstat, "band": _band(bands, sc),
                                 "applicable_band": applicable, "fallback_used": applicable == "ROBUST_BASELINE"},
        "ACTION": action,
        "REASONS": why + [f"conditional scenario {sc} via {st['rule']}", st["evidence"]["credit"], st["evidence"]["fundamental"], st["evidence"]["macro"]]
                   + ([f"scenario band {sstat}: the robust baseline band applies"] if applicable == "ROBUST_BASELINE" else [])
                   + [f"review flag: {f}" for f in st["review_flags"]],
        "TARGET_UNITS_AND_TRADES": {"target": "STATIC_PIF" if applicable == "ROBUST_BASELINE" else f"TARGET_{sc}", "target_units": units,
                                    "trades_from_current": trades},
        "RISK_METRICS": risk,
        "TRIGGERS_TO_WATCH": trig,
        "NEXT_REVIEW": {"routine": f"{nxt.date()} after the close (daily state check)",
                        "earliest_scheduled_event": str(sched["event_date"].min()) if len(sched) else None,
                        "weekly_review": "Friday after the close", "recalibration": "semi-annual (V4 re-estimation months) or after a SEVERE event"},
    }
    (ctx.out / "pif_daily_decision_card.json").write_text(json.dumps(card, indent=1, ensure_ascii=False, default=str))
