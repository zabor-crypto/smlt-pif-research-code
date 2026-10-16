"""Deliverable tables assembled from stage outputs (no new computation of risk here, only joins/ratios)."""
from __future__ import annotations

import numpy as np
import pandas as pd

RISK_COLS = ["mean_pnl", "median_pnl", "VaR95", "ES95", "VaR99", "ES99", "std_pnl", "p_loss", "p05_pnl",
             "mean_max_dd", "p95_max_dd"]


def build_tables(ctx) -> None:
    complexity_table(ctx)
    E, C = ctx.E, ctx.C
    bs = pd.read_csv(ctx.out / "bootstrap_strategy_metrics.csv")
    # ---------------- risk_metrics: both normalisations (A: per RUB 10m SMLT; B: per total capital E + C)
    long = []
    for basis, denom in (("A_pct_of_SMLT_10m", lambda r: E), ("B_pct_of_total_capital", lambda r: E + r["capital"])):
        x = bs.copy()
        x["basis"] = basis
        d = x.apply(denom, axis=1)
        for c in RISK_COLS:
            if c in ("p_loss",):
                continue
            x[c] = x[c] / d
        long.append(x)
    rm = pd.concat(long, ignore_index=True)
    ctx.save(rm, "risk_metrics.csv", index=False)
    # ---------------- strategy comparison at main capital
    main = bs[bs["capital"] == C].copy()
    ctrl = main[main["strategy"] == "CONTROL"].set_index(["variant", "horizon"])
    main["hedge_cost_expected_rub"] = main.apply(lambda r: ctrl.loc[(r["variant"], r["horizon"]), "mean_pnl"] - r["mean_pnl"], axis=1)
    main["ES95_reduction_rub"] = main.apply(lambda r: ctrl.loc[(r["variant"], r["horizon"]), "ES95"] - r["ES95"], axis=1)
    main["ES95_reduction_pct_of_control"] = main.apply(
        lambda r: 1 - r["ES95"] / ctrl.loc[(r["variant"], r["horizon"]), "ES95"], axis=1)
    main["ES95_pct_SMLT_10m"] = main["ES95"] / E
    main["ES95_pct_total_capital"] = main["ES95"] / (E + C)
    main["ES95_reduction_per_rub_cost"] = main["ES95_reduction_rub"] / main["hedge_cost_expected_rub"].where(main["hedge_cost_expected_rub"] > 0)
    wf = pd.read_csv(ctx.out / "walkforward_daily_metrics.csv")
    wfa = wf[wf["basis"] == "A_per_SMLT_10m"].copy()
    wfa["policy"] = wfa["strategy"].str.split("|").str[0]
    wf_best = wfa[wfa["strategy"].str.endswith(("thr=0.00", "WF_SELECTED"))]
    wmap = {"FIXED_25": "FUT_25", "FIXED_50": "FUT_50", "FIXED_75": "FUT_75", "FIXED_100": "FUT_100", "CONTROL": "CONTROL"}
    wf_best = wf_best.assign(strategy_bs=wf_best["policy"].map(wmap))
    keep = ["strategy_bs", "HE_VAR", "HE_PNL", "ann_vol", "ES95_1d", "max_drawdown", "upside_capture", "downside_capture",
            "turnover_notional_multiple", "total_costs_rub", "avg_hedge_ratio"]
    wfj = wf_best[keep].dropna(subset=["strategy_bs"]).add_prefix("wf_").rename(columns={"wf_strategy_bs": "strategy"})
    sc = main.merge(wfj, on="strategy", how="left")
    ctx.save(sc, "strategy_comparison.csv", index=False)
    # ---------------- capital budget sensitivity
    rows = []
    for (v, hz, cap), g in bs.groupby(["variant", "horizon", "capital"]):
        c0 = g[g["strategy"] == "CONTROL"].iloc[0]
        fut = g[g["strategy"].str.startswith("FUT_")]
        if fut.empty:
            continue
        best = fut.sort_values("ES95").iloc[0]
        red = c0["ES95"] - best["ES95"]
        robust = {}
        for tol in (0.05, 0.20):
            ok = fut[fut["p_sleeve_breach"] <= tol].sort_values("ES95")
            robust[tol] = ok.iloc[0] if len(ok) else None
        mr = float(ctx.read("im_rate_mr1.csv", index_col=0, parse_dates=True).loc[:ctx.cutoff].iloc[-1]["SMLT"])
        h_best = float(best["strategy"].replace("FUT_", "")) / 100
        from ..pipeline import _fut_ctx
        from ..strategies.futures import rally_exhausting_sleeve
        S0, F0, pv = _fut_ctx(ctx)
        x_star = rally_exhausting_sleeve(h_best, E / S0, S0, F0, cap, mr, point_value=pv)   # contract-level engine
        rows.append({"variant": v, "horizon": hz, "hedge_capital": cap, "hedge_capital_pct_of_exposure": cap / E,
                     "control_ES95": c0["ES95"], "best_feasible_direct_hedge": best["strategy"],
                     "best_ES95": best["ES95"], "ES95_reduction_rub": red, "ES95_reduction_pct_of_control": red / c0["ES95"],
                     "ES95_reduction_per_rub_hedge_capital": red / cap,
                     "ES95_reduction_per_rub_peak_collateral_p95": red / best["peak_collateral_p95"] if best["peak_collateral_p95"] > 0 else np.nan,
                     "upside_capture": best["upside_capture"], "p_sleeve_breach": best["p_sleeve_breach"],
                     "initial_im": best["initial_im"], "max_cum_vm_drain_p99": best["max_cum_vm_drain_p99"],
                     "expected_hedge_cost_rub": c0["mean_pnl"] - best["mean_pnl"],
                     "feasible_futures_ratios": "|".join(sorted(fut["strategy"].str.replace("FUT_", ""))),
                     "rally_that_exhausts_sleeve_for_best": x_star,
                     **{f"best_hedge_p_breach_le_{int(t*100)}pct": (r["strategy"] if r is not None else "NONE")
                        for t, r in robust.items()},
                     **{f"ES95_reduction_pct_p_breach_le_{int(t*100)}pct": ((c0["ES95"] - r["ES95"]) / c0["ES95"] if r is not None else 0.0)
                        for t, r in robust.items()}})
    cbs = pd.DataFrame(rows)
    opt = pd.read_csv(ctx.out / "optimizer_solutions.csv")
    o = opt[opt["objective"] == "MIN_ES"][["variant", "horizon", "capital", "es", "active", "K"]].rename(
        columns={"capital": "hedge_capital", "es": "optimizer_minES_pct_of_E_scenario", "active": "optimizer_active", "K": "optimizer_K"})
    cbs = cbs.merge(o, on=["variant", "horizon", "hedge_capital"], how="left")
    # borrowed-capital sensitivity (NOT the main case): financing at RUSFAR(cutoff) + spread, sleeve interest kept
    P = ctx.read("prices.csv", index_col=0, parse_dates=True)
    rusfar = float(P["RUSFAR"].loc[:ctx.cutoff].dropna().iloc[-1]) / 100 if "RUSFAR" in P else np.nan
    spread = ctx.cfg["capital"]["borrowed_financing_spread_over_rusfar_bp"] / 1e4
    tau = cbs["horizon"].map({k: v / 252 for k, v in ctx.cfg["horizons_trading_days"].items()})
    cbs["rusfar_at_cutoff"] = rusfar
    cbs["borrowed_financing_cost_rub"] = cbs["hedge_capital"] * (rusfar + spread) * tau
    cbs["borrowed_expected_hedge_cost_rub"] = cbs["expected_hedge_cost_rub"] + cbs["borrowed_financing_cost_rub"]
    ctx.save(cbs, "capital_budget_sensitivity.csv", index=False)
    # ---------------- episodes summary
    ep = pd.read_csv(ctx.out / "episodes.csv")
    g = ep.groupby(["strategy", "capital", "horizon"])
    es = g.agg(n_episodes=("terminal_pnl", "size"), mean_terminal_pnl=("terminal_pnl", "mean"),
               median_terminal_pnl=("terminal_pnl", "median"), worst_terminal_pnl=("terminal_pnl", "min"),
               p05_terminal_pnl=("terminal_pnl", lambda s: s.quantile(0.05)), mean_max_dd=("max_dd_rub", "mean"),
               worst_max_dd=("max_dd_rub", "max"), mean_costs=("costs", "mean"), mean_hedge_ratio=("avg_hedge_ratio", "mean"),
               max_daily_vm_outflow=("max_daily_vm_outflow", "max"), max_cum_vm_drain=("max_cum_vm_drain", "max"),
               peak_im=("peak_im", "max"), peak_total_collateral=("peak_total_collateral", "max"),
               share_episodes_sleeve_breach=("sleeve_breach", "mean"), forced_reductions=("n_forced_reductions", "sum"),
               mean_share_days_capped=("share_days_capped", "mean")).reset_index()
    ctx.save(es, "episodes_summary.csv", index=False)


def complexity_table(ctx) -> None:
    cx = pd.read_csv(ctx.out / "complexity_frontier.csv")
    bs = pd.read_csv(ctx.out / "bootstrap_strategy_metrics.csv")
    ctrl = bs[bs["strategy"] == "CONTROL"].set_index(["variant", "horizon", "capital"])["mean_pnl"]
    rolls = {"6m": 2, "12m": 4}
    wcols = [c for c in cx.columns if c.startswith("w_")]
    cx["hedge_cost_expected_rub"] = [ctrl.get((v, h, c), np.nan) - m for v, h, c, m in
                                     zip(cx["variant"], cx["horizon"], cx["capital"], cx["path_mean_pnl"])]
    cx["turnover_notional_multiple"] = [
        sum(abs(r[w]) * ((2 + 2 * rolls[r["horizon"]]) if w == "w_SMLT_FUT" else 2) for w in wcols)
        for _, r in cx.iterrows()]
    cx["margin_usage_rub"] = cx["path_initial_im"]
    thr = ctx.cfg["optimization"]["elbow_min_es_gain_pct_of_E"] * ctx.E
    rows = []
    for (v, h, c), g in cx.groupby(["variant", "horizon", "capital"]):
        g = g.sort_values("K")
        es = g.set_index("K")["path_ES95"]
        elbow = int(es.index.max())
        for k in es.index[:-1]:
            nxt = es.index[es.index > k].min()
            if es[k] - es[nxt] < thr:
                elbow = int(k)
                break
        rows.append({"variant": v, "horizon": h, "capital": c, "elbow_K": elbow,
                     "elbow_active": g.set_index("K").loc[elbow, "active"], "rule_min_gain_rub": thr,
                     "ES95_at_K1": float(es.iloc[0]), "ES95_at_elbow": float(es[elbow]), "ES95_at_Kmax": float(es.iloc[-1])})
    ctx.save(cx, "complexity_frontier.csv", index=False)
    ctx.save(pd.DataFrame(rows), "complexity_elbow.csv", index=False)
