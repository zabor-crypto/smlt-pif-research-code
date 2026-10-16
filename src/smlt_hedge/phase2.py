"""Second research pass: Phase-A repairs outputs and Phase-B adaptive multi-asset research stages."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import pipeline as pl
from .data import specs as specs_mod
from .features import factors as fx
from .features.stats import rolling_corr
from .portfolio import adaptive, regimes, sleeves, smart
from .risk.covariance import cov_ewma, cov_lw, cov_oas, cov_sample
from .risk.metrics import es_hist, he_var, summary, var_hist
from .scenarios.bootstrap import SleevePos, evaluate_portfolio
from .strategies.collateral import CASH_ONLY, Collateral, grid
from .strategies.futures import contract_position, rally_exhausting_sleeve


def _named_collateral(ctx) -> list[Collateral]:
    return [Collateral(k, v["yield_fraction"], v["haircut"], v["margin_multiplier"]) for k, v in ctx.cfg["collateral"]["named"].items()]


# ================================================================== PHASE A
def s11_phase_a(ctx) -> None:
    P, R, V = pl._panel(ctx)
    cs = ctx.read("credit_state.csv", index_col=0, parse_dates=True)
    # ---- A1 / B3 factor model
    F = fx.factor_frame(R, cs)
    t = F.index[F.index <= pd.Timestamp(ctx.cutoff)]
    win = {"FULL": ("2022-01-01", ctx.cutoff), "LAST_250D": (str(t[-250].date()), ctx.cutoff),
           "LAST_120D": (str(t[-120].date()), ctx.cutoff), **{str(y): (f"{y}-01-01", f"{y}-12-31") for y in (2022, 2023, 2024, 2025)},
           "2026_YTD": ("2026-01-01", ctx.cutoff)}
    dec = fx.decomposition_table(F, win)
    # robustness: Z-spread credit factor on the sub-sample where MOEX publishes ZSPREAD
    Fz = F.drop(columns=["CREDIT"]).rename(columns={"CREDIT_Z": "CREDIT"})
    z = Fz.dropna(subset=["CREDIT"])
    if len(z) > 60:
        dz = fx.decomposition_table(Fz, {"CREDIT_Z_SUBSAMPLE": (str(z.index.min().date()), ctx.cutoff)})
        dec = pd.concat([dec, dz], ignore_index=True)
    dec["specification"] = ("r_SMLT = a + b_M*IMOEX + b_S*PEERS_exSMLT(PIKK,LSRG,ETLN eq-wt) + b_R*RGBI + "
                            "b_C*(SMLT bond ret - D_b/D_RGBI*RGBI ret) + e ; Shapley R2 shares; Newey-West(5) t-stats")
    ctx.save(dec, "factor_decomposition.csv", index=False)
    ld = dec[["window", "n_obs"] + [c for c in dec.columns if c.startswith(("beta_", "t_"))] + ["total_R2"]]
    ctx.save(ld, "factor_loadings.csv", index=False)
    ctx.save(fx.rolling_loadings(F), "rolling_factor_loadings.csv")
    rc = pd.concat([fx.risk_contributions(F, a, b).assign(window=k) for k, (a, b) in win.items() if k in ("FULL", "LAST_250D", "LAST_120D")])
    ctx.save(rc, "factor_risk_contributions.csv", index=False)
    # ---- A4 contract specs
    ctx.save(specs_mod.audit(ctx.root, ctx.cutoff), "futures_contract_spec_audit.csv", index=False)
    # ---- A2 collateral regimes on the direct hedge and the de-risking branch
    _collateral_sensitivity(ctx)
    # ---- A7 basis red-team
    from .reporting import basis_audit
    basis_audit.run(ctx, pl)


def _collateral_sensitivity(ctx) -> None:
    S0, F0, pv = pl._fut_ctx(ctx)
    cols = ["SMLT", "SMLT_FUT"]
    B = pl._bootstrap_inputs(ctx, cols)
    mr = B["mr"]
    n_sh = ctx.E / S0
    per = pl._costs(ctx).per_contract() / S0
    derisk_cost = (0.1 + ctx.cfg["costs"]["equity_commission_bps"] * 1e-4 * S0) / S0
    regs = _named_collateral(ctx) + grid(ctx.cfg["collateral"]["grid_yield_fraction"], ctx.cfg["collateral"]["grid_margin_multiplier"],
                                         ctx.cfg["collateral"]["grid_haircut"])
    rows = []
    for vname in ("FHS_FULL", "HS_FULL"):
        for hz_name, hz in ctx.cfg["horizons_trading_days"].items():
            Pth = pl._paths_for(ctx, B, vname, hz, n_paths=10000)
            n_rolls = int(np.ceil(hz / 252 / 0.25))
            for Cg in ctx.cfg["capital"]["hedge_capital_grid_rub"]:
                for coll in regs:
                    for x in (0.0, 0.25, 0.5, 0.75, 1.0):
                        for kind in ("FUT", "SELL"):
                            if x == 0 and kind == "SELL":
                                continue
                            if kind == "FUT":
                                N0 = contract_position(x, n_sh, F0, pv)["notional_rub"] if x else 0.0
                                sl = [SleevePos("SMLT_FUT", -N0, mr, per, n_rolls)] if x else []
                                if N0 * mr * coll.margin_multiplier * coll.capital_per_margin() > 0.8 * Cg + 1e-6:
                                    continue                            # margin-infeasible at this capital
                                m = evaluate_portfolio(Pth, E=ctx.E, C=float(Cg), sleeves=sl, mmf_daily=B["daily_y"], collateral=coll)
                                xs = rally_exhausting_sleeve(x, n_sh, S0, F0, float(Cg), mr, coll, point_value=pv) if x else np.inf
                                name = "CONTROL" if x == 0 else f"FUT_{int(x*100)}"
                            else:
                                m = evaluate_portfolio(Pth, E=ctx.E, C=float(Cg), mmf_daily=B["daily_y"], derisk=x,
                                                       derisk_cost=derisk_cost, collateral=coll)
                                xs, name = np.inf, f"DERISK_SELL_{int(x*100)}"
                            rows.append({"variant": vname, "horizon": hz_name, "capital": Cg, "collateral": coll.name,
                                         "yield_fraction": coll.yield_fraction, "haircut": coll.haircut,
                                         "margin_multiplier": coll.margin_multiplier, "strategy": name, "fraction": x,
                                         "rally_exhausting_sleeve_engine": xs,
                                         **{k: m[k] for k in ("ES95", "ES99", "mean_pnl", "std_pnl", "upside_capture",
                                                              "p_sleeve_breach", "initial_im", "initial_capital_used",
                                                              "peak_collateral_p95", "transaction_costs")}})
    df = pd.DataFrame(rows)
    ctx.save(df, "collateral_sensitivity.csv", index=False)
    # conclusions per regime
    concl = []
    for (v, hz, cap, cn), g in df.groupby(["variant", "horizon", "capital", "collateral"]):
        g = g.set_index("strategy")
        dom, tot = 0, 0
        for x in (25, 50, 75, 100):
            if f"FUT_{x}" in g.index and f"DERISK_SELL_{x}" in g.index:
                tot += 1
                s_, f_ = g.loc[f"DERISK_SELL_{x}"], g.loc[f"FUT_{x}"]
                dom += int(s_["ES95"] <= f_["ES95"] + 1e-6 and s_["mean_pnl"] >= f_["mean_pnl"] - 1e-6)
        fut = g[g.index.str.startswith("FUT_")]
        ok20 = fut[fut["p_sleeve_breach"] <= 0.20].sort_values("ES95")
        ok10 = fut[fut["p_sleeve_breach"] <= 0.10].sort_values("ES95")
        concl.append({"variant": v, "horizon": hz, "capital": cap, "collateral": cn,
                      "sell_dominates_fut_same_fraction": f"{dom}/{tot}",
                      "best_fut_breach_le_20pct": ok20.index[0] if len(ok20) else "NONE",
                      "best_fut_breach_le_10pct": ok10.index[0] if len(ok10) else "NONE",
                      "breach_FUT_50": g["p_sleeve_breach"].get("FUT_50", np.nan),
                      "breach_FUT_75": g["p_sleeve_breach"].get("FUT_75", np.nan),
                      "breach_FUT_100": g["p_sleeve_breach"].get("FUT_100", np.nan),
                      "mean_gap_SELL50_minus_FUT50": g["mean_pnl"].get("DERISK_SELL_50", np.nan) - g["mean_pnl"].get("FUT_50", np.nan)})
    ctx.save(pd.DataFrame(concl), "collateral_conclusions.csv", index=False)


# ================================================================== B2 correlations / covariances
def s12_correlations(ctx) -> None:
    P, R, V = pl._panel(ctx)
    sp = sleeves.build(ctx)
    cols = ["SMLT"] + [s.series for s in sp] + ["IMOEX", "SMLT_BOND_IDX"]
    X = R[cols].dropna()
    cs = ctx.read("credit_state.csv", index_col=0, parse_dates=True)
    reg = regimes.classify(R, cs, ctx.cfg["smart_portfolio"]["regimes"]).reindex(X.index)
    t = X.index
    sets = {
        "full": X, "250d": X.tail(250), "120d": X.tail(120), "current_regime_60d": X.tail(60),
        "downside": X[X["SMLT"] < 0], "market_stress": X[X["IMOEX"] <= X["IMOEX"].quantile(0.10)],
        "credit_stress": X[reg["CREDIT_STRESS"].fillna(False).values],
    }
    rename = {s.series: s.name for s in sp}
    rows = []
    for k, d in sets.items():
        c = d.corr().rename(index=rename, columns=rename)
        ctx.save(c, f"correlation_{k}.csv")
        for est, f in (("sample", cov_sample), ("oas", cov_oas), ("ledoit_wolf", cov_lw), ("ewma", cov_ewma)):
            if k in ("full", "250d") and len(d) > 30:
                cov = pd.DataFrame(f(d.values), index=c.index, columns=c.columns)
                ctx.save(cov, f"covariance_{k}_{est}.csv")
        for col in c.columns:
            if col != "SMLT":
                rows.append({"set": k, "n_obs": len(d), "pair": f"SMLT~{col}", "corr": float(c.loc["SMLT", col])})
    long = pd.DataFrame(rows)
    piv = long.pivot(index="pair", columns="set", values="corr")
    def cls(r):
        base = r["full"]
        stress = r[["market_stress", "credit_stress", "downside"]]
        if (np.sign(stress) != np.sign(base)).any() and stress.abs().max() > 0.1 and abs(base) > 0.05:
            return "REVERSES_IN_STRESS"
        if (stress.abs() > abs(base) + 0.10).all():
            return "STRONGER_IN_STRESS"
        if (stress.abs() < abs(base) - 0.10).any():
            return "BREAKS_IN_STRESS"
        return "PERSISTS"
    piv["classification"] = piv.apply(cls, axis=1)
    piv["full_minus_250d"] = piv["full"] - piv["250d"]
    ctx.save(piv, "correlation_stability.csv")
    ctx.save(reg, "regimes_daily.csv", "interim")


# ================================================================== B5/B6/B9/B11 smart portfolios
def s13_smart(ctx) -> None:
    sp = sleeves.build(ctx)
    ctx.save(pd.DataFrame([s.__dict__ for s in sp]), "phase_b_sleeves.csv", index=False)
    cfg = ctx.cfg["smart_portfolio"]
    all_sum, all_ev, cons, cx_rows = [], [], [], []
    for hz_name, hz in ctx.cfg["horizons_trading_days"].items():
        models, B = smart.scenario_models(ctx, pl, sp, hz)
        caps = ctx.cfg["capital"]["hedge_capital_grid_rub"] if hz_name == "6m" else [ctx.C]
        for C in caps:
            cands = smart.generate_candidates(ctx, sp, models, B, hz, float(C))
            ev = smart.evaluate_candidates(ctx, sp, models, B, cands, hz, float(C))
            ctrl = pd.DataFrame([{"model": m, **evaluate_portfolio(P, E=ctx.E, C=float(C), mmf_daily=B["daily_y"])}
                                 for m, P in models.items()])
            s = smart.summarise(ev, ctrl)
            s["horizon"] = hz_name
            all_sum.append(s)
            all_ev.append(ev.assign(horizon=hz_name))
            ct = smart.constrained_table(s)
            ct["horizon"], ct["capital"] = hz_name, C
            cons.append(ct)
            cx = smart.complexity_multidim(s, cfg["selection"], ctx.E)
            cx["horizon"], cx["capital"] = hz_name, C
            cx_rows.append(cx)
    S = pd.concat(all_sum, ignore_index=True)
    ctx.save(S, "smart_candidates_summary.csv", index=False)
    ctx.save(pd.concat(all_ev, ignore_index=True), "smart_candidates_by_model.csv", "interim", index=False)
    ctx.save(pd.concat(cons, ignore_index=True), "static_smart_constrained_solutions.csv", index=False)
    ctx.save(pd.concat(cx_rows, ignore_index=True), "complexity_multidim.csv", index=False)
    s13_selections(ctx)


def s13_selections(ctx) -> None:
    """Pre-declared lexicographic selections (static / robust) from the saved candidate summary."""
    cfg = ctx.cfg["smart_portfolio"]
    S = pd.read_csv(ctx.out / "smart_candidates_summary.csv")
    rows = []
    for (hz, C), s in S.groupby(["horizon", "capital"]):
        for rule in ("static", "robust"):
            for scope in (None, "DIRECT_ONLY"):
                r = smart.select(s, rule, cfg["selection"], family=scope)
                rows.append({**(r.to_dict() if r is not None else {"cand_id": "NONE"}), "horizon": hz, "capital": C,
                             "rule": rule, "selection_scope": scope or "ANY"})
    rob = pd.DataFrame(rows)
    ctx.save(rob[rob["rule"] == "robust"], "robust_optimizer_solutions.csv", index=False)
    ctx.save(rob, "smart_selected_portfolios.csv", index=False)


# ================================================================== B7/B8/B10/B12 adaptive
def _env(ctx, sp) -> adaptive.Env:
    P, R, V = pl._panel(ctx)
    cs = ctx.read("credit_state.csv", index_col=0, parse_dates=True)
    fut = ctx.read("futures_SMLT_FUT.csv", "interim", index_col=0, parse_dates=True)
    imr = ctx.read("im_rate_mr1.csv", index_col=0, parse_dates=True)
    reg = regimes.classify(R, cs, ctx.cfg["smart_portfolio"]["regimes"])
    return adaptive.Env(R=R, specs=sp, imr=imr, mmf=R["LQDT"].fillna(0.0), fs_ratio=(fut["F_held"] / P["SMLT"]).reindex(R.index),
                        regimes=reg, E=ctx.E, C=ctx.C, cfg=ctx.cfg["smart_portfolio"])


def s14_adaptive(ctx) -> None:
    sp = sleeves.build(ctx)
    env = _env(ctx, sp)
    R = env.R
    cols = ["SMLT"] + [s.series for s in sp]
    oos = pd.Timestamp(ctx.cfg["walk_forward"]["oos_start"])
    dates = R[cols].dropna().index
    dates = dates[(dates >= oos) & (dates <= pd.Timestamp(ctx.cutoff))]
    # FIXED_SMART: composition estimated ONCE on data before the OOS start with the static constraints
    pre = int(R.index.searchsorted(oos))
    fixed_h = adaptive._lp_target(env, pre, ctx.cfg["smart_portfolio"]["selection"]["static"]["min_upside"], 0.45, CASH_ONLY)
    pols = {"FUT_25": ("FIXED", 0.25), "FUT_50": ("FIXED", 0.5), "FUT_75": ("FIXED", 0.75), "FUT_100": ("FIXED", 1.0),
            "ROLLING_HSTAR": ("ROLLING_HSTAR", 0), "CURRENT_STATE_DEPENDENT": ("CURRENT_STATE", 0),
            "FIXED_SMART_PRE_OOS": ("FIXED_SMART", 0), "ADAPTIVE_SMART": ("ADAPTIVE", 0)}
    targets = {}
    for name, (pol, x) in pols.items():
        targets[name] = adaptive.target_path(env, pol, dates, every=5, fixed_h=fixed_h, x=x)
    targets["ADAPTIVE_SMART_MONTHLY"] = adaptive.target_path(env, "ADAPTIVE", dates, every=21)
    for k, v in targets.items():
        ctx.save(v, f"adaptive_targets_{k}.csv", "interim")
    # daily constant-exposure walk-forward (+ causal threshold selection for the adaptive policies)
    ctrl = adaptive.daily_walk(env, adaptive.target_path(env, "FIXED", dates, every=5, x=0.0), dates, 0.0)
    unh = ctrl["pnl_spot"] + ctrl["carry"]
    series, rows = {}, []
    for name, tg in targets.items():
        grid_thr = ctx.cfg["smart_portfolio"]["adaptive_thresholds_grid"] if name.startswith("ADAPTIVE_SMART") and "MONTHLY" not in name else [0.0]
        runs = {thr: adaptive.daily_walk(env, tg, dates, thr) for thr in grid_thr}
        if len(runs) > 1:
            sel = adaptive.select_threshold_causally(runs, lambda x: es_hist(x, 0.95))
            ctx.save(sel, f"threshold_selection_{name}.csv", "interim")
            pieces = []
            qs = list(sel.index) + [dates.max() + pd.Timedelta(days=1)]
            for a, b in zip(qs[:-1], qs[1:]):
                pieces.append(runs[sel.loc[a, "threshold"]].loc[(dates >= a) & (dates < b)])
            series[name] = pd.concat(pieces)
            series[name + "|thr_selected"] = series[name]
        else:
            series[name] = runs[grid_thr[0]]
    series["CONTROL"] = ctrl
    for name, d in series.items():
        if name.endswith("|thr_selected"):
            continue
        for basis, base in (("A_per_SMLT_10m", ctx.E), ("B_per_total_capital", ctx.E + ctx.C)):
            m = summary(d["pnl_total"], base_rub=base, bench_pnl_rub=unh)
            hc = [c for c in d.columns if c.startswith("h_")]
            m.update({"strategy": name, "basis": basis, "HE_VAR": he_var(d["pnl_total"].values, unh.reindex(d.index).values),
                      "turnover_notional_multiple": float(d["turnover"].sum()), "total_costs_rub": float(d["costs"].sum()),
                      "mean_active_sleeves": float((d[hc].abs() > 1e-4).sum(axis=1).mean()),
                      **{f"mean_{c}": float(d[c].mean()) for c in hc}})
            rows.append(m)
        ctx.save(d, f"adaptive_daily_{name}.csv", "interim")
    ctx.save(pd.DataFrame(rows), "adaptive_walkforward_metrics.csv", index=False)
    # episodes (6m / 12m, monthly starts) with the liquidity waterfall vs immediate cut
    ep_rows = []
    for name, tg in targets.items():
        sel = ctx.interim / f"threshold_selection_{name}.csv"
        selt = pd.read_csv(sel, index_col=0, parse_dates=True) if sel.exists() else None
        for hz_name, hz in ctx.cfg["horizons_trading_days"].items():
            for k in range(0, len(dates) - hz, 21):
                d = dates[k:k + hz + 1]
                thr = 0.0
                if selt is not None:
                    thr = float(selt["threshold"][selt.index <= d[0]].iloc[-1]) if (selt.index <= d[0]).any() else 0.10
                for liq in ("WATERFALL", "IMMEDIATE_CUT"):
                    if liq == "IMMEDIATE_CUT" and name not in ("ADAPTIVE_SMART", "FUT_100", "FUT_75", "FIXED_SMART_PRE_OOS"):
                        continue
                    e = adaptive.episode(env, tg, d, thr, liquidity=liq)
                    ep_rows.append({"strategy": name, "horizon": hz_name, "start": d[0], "liquidity_policy": liq, "threshold": thr, **e})
    ep = pd.DataFrame(ep_rows)
    ctx.save(ep, "adaptive_episodes.csv", index=False)
    g = ep.groupby(["strategy", "horizon", "liquidity_policy"])
    wf = g.agg(n=("terminal_pnl", "size"), mean_terminal=("terminal_pnl", "mean"),
               p05_terminal=("terminal_pnl", lambda s: s.quantile(0.05)), worst_terminal=("terminal_pnl", "min"),
               ES95_terminal=("terminal_pnl", lambda s: es_hist(s.values, 0.95)), mean_max_dd=("max_dd", "mean"),
               peak_im=("peak_im", "max"), peak_posted_p95=("peak_posted", lambda s: s.quantile(0.95)),
               breach_share=("breach", "mean"), step3=("step3_cut_other", "sum"), step4=("step4_cut_direct", "sum"),
               step5_sell=("step5_sell_shares", "sum"), immediate=("immediate_cut", "sum"),
               shares_sold_value=("shares_sold_value", "sum"), turnover=("turnover", "mean"), costs=("costs", "mean")).reset_index()
    ctx.save(wf, "liquidity_waterfall_results.csv", index=False)
    # regime frequencies and threshold sensitivity
    cs = ctx.read("credit_state.csv", index_col=0, parse_dates=True)
    sens = []
    for sc in [1.0] + ctx.cfg["smart_portfolio"]["regimes"]["sensitivity_scale"]:
        rg = regimes.classify(R, cs, ctx.cfg["smart_portfolio"]["regimes"], scale=sc).reindex(dates)
        tg = adaptive.target_path(env, "ADAPTIVE", dates, every=5, regimes=rg.reindex(R.index))
        d = adaptive.daily_walk(env, tg, dates, 0.10)
        freq = rg["regime"].value_counts(normalize=True).to_dict()
        sens.append({"scale": sc, **{f"freq_{k}": v for k, v in freq.items()},
                     "HE_VAR": he_var(d["pnl_total"].values, unh.reindex(d.index).values),
                     "ES95_1d_pct": es_hist(d["pnl_total"].values / ctx.E, 0.95), "turnover": float(d["turnover"].sum())})
    ctx.save(pd.DataFrame(sens), "regime_sensitivity.csv", index=False)
    cur = env.regimes.loc[:ctx.cutoff].iloc[-1]
    ctx.log["regime_at_cutoff"] = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else (v if isinstance(v, str) else float(v)))
                                   for k, v in cur.items()}


def s15_decision(ctx) -> None:          # filled in below (B9/B15)
    from .portfolio import decision
    decision.run(ctx, pl)
