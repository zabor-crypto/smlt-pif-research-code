"""Key findings: every number quoted in RESEARCH_REPORT.md is computed here from pipeline outputs.

The report template (reporting/templates/RESEARCH_REPORT.md.tmpl) contains only {placeholders}
for numbers; rendering fails loudly if a placeholder has no computed value.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _pct(x, nd=0):
    return f"{x * 100:.{nd}f}%"


def _m(x, nd=2):
    return f"{x / 1e6:.{nd}f}"


def compute(ctx) -> dict:
    o = ctx.out
    E, C = ctx.E, ctx.C
    F: dict[str, str] = {}
    mi = pd.read_csv(o / "market_inputs_at_cutoff.csv", index_col=0)["value"]
    dz = pd.read_csv(o / "smlt_market_dossier.csv", index_col=0)["value"]
    num = lambda s, k: float(pd.to_numeric(s[k]))  # noqa: E731
    F["S0"] = f"{num(mi, 'S0'):.1f}"
    F["F0"] = f"{num(mi, 'F0'):.0f}"
    F["front"] = str(mi["front_future"])
    F["basis"] = _pct(num(mi, "basis_ann"), 1)
    F["y_mmf"] = _pct(num(mi, "mmf_yield_LQDT_60d"), 1)
    F["carry_gap"] = _pct(num(mi, "mmf_yield_LQDT_60d") - num(mi, "basis_ann"), 1)
    F["mr1"] = _pct(num(mi, "mr1"))
    F["mr2"] = _pct(num(mi, "mr2"))
    F["im_drag"] = _pct(num(mi, "mr1") * num(mi, "mmf_yield_LQDT_60d"), 1)
    F["listed_iv"] = _pct(num(mi, "listed_iv_oct21_median"))
    F["rv250"] = _pct(num(mi, "realised_vol_250d"))
    F["shares"] = f"{E / num(mi, 'S0'):,.0f}"
    for k in ("vol_20d", "vol_60d", "vol_250d"):
        F[k] = _pct(num(dz, k))
    F["sept_dd"] = _pct(num(dz, "sept_2026_drawdown_from_sep_high"))
    F["ytd"] = _pct(num(dz, "ytd_2026_return"))
    F["dd_2022"] = _pct(num(dz, "drawdown_from_2022_high"))
    F["beta_imoex"] = f"{num(dz, 'beta_IMOEX_250d'):.2f}"
    F["beta_moexre"] = f"{num(dz, 'beta_MOEXRE_250d'):.2f}"
    F["pos_adv"] = _pct(num(dz, "position_share_of_adv20"), 2)
    # --- candidate statistics / hedgeable components
    cs = pd.read_csv(o / "candidate_stats.csv").set_index("series")
    for key, s in (("fut", "SMLT_FUT"), ("mix", "MIX_FUT"), ("imoexf", "IMOEXF"), ("moexre", "MOEXRE"),
                   ("rgbi", "RGBI_FUT"), ("pikk", "PIKK"), ("gold", "GLDRUB_TOM"), ("cny", "CNYRUB_TOM"), ("br", "BR_FUT")):
        if s in cs.index:
            F[f"corr_{key}"] = f"{cs.loc[s, 'corr_250d']:.2f}"
            F[f"r2_{key}"] = _pct(cs.loc[s, "corr_250d"] ** 2)
            F[f"hstar_{key}"] = f"{cs.loc[s, 'hstar_250d']:.2f}"
            F[f"stress_{key}"] = f"{cs.loc[s, 'corr_market_stress']:.2f}"
            F[f"corrmin_{key}"] = f"{cs.loc[s, 'corr120_min_2y']:.2f}"
    F["hstar_identity_err"] = f"{cs['hstar_identity_err'].abs().max():.1e}"
    F["idio_share_vs_index"] = _pct(1 - cs.loc["MIX_FUT", "corr_250d"] ** 2)
    fd = pd.read_csv(o / "factor_decomposition.csv").set_index("window")
    F["fx_unexpl_250"] = _pct(fd.loc["LAST_250D", "unexplained"])
    F["fx_unexpl_full"] = _pct(fd.loc["FULL", "unexplained"])
    # --- universe
    u = pd.read_csv(o / "instrument_universe.csv")
    vc = u["classification"].value_counts()
    for c in ("EXECUTABLE", "LIMITED_EXECUTABILITY", "THEORETICAL_ONLY", "REJECTED"):
        F[f"n_{c}"] = str(int(vc.get(c, 0)))
    F["n_candidates"] = str(len(u))
    opt = pd.read_csv(o / "smlt_options_20260925.csv")
    F["opt_max_expiry"] = str(opt["expiry"].max())
    F["opt_strikes"] = ", ".join(str(int(x)) for x in sorted(opt["strike"].unique()))
    F["opt_max_oi"] = f"{int(opt['open_interest'].max()):,}"
    F["put150_iv"] = _pct(float(opt[(opt["strike"] == 150) & (opt["type"] == "P") & (opt["expiry"] == "2026-10-21")]["iv_from_settle"].iloc[0]))
    F["F_over_S_opt"] = f"{float(opt[opt['expiry'] == '2026-10-21']['F_over_S0'].iloc[0]):.4f}"
    # --- covariance
    cv = pd.read_csv(o / "covariance_oos.csv")
    c120 = cv[cv["window"] == 120].set_index("estimator")
    F["cov_sample_bias"] = f"{c120.loc['sample', 'bias_ratio_real_over_pred']:.2f}"
    F["cov_oas_bias"] = f"{c120.loc['oas', 'bias_ratio_real_over_pred']:.2f}"
    F["cov_oas_qlike"] = f"{c120.loc['oas', 'mean_qlike']:.2f}"
    F["cov_sample_qlike"] = f"{c120.loc['sample', 'mean_qlike']:.2f}"
    F["cov_hvol_range"] = f"{cv['realised_hedged_vol_ann'].min()*100:.1f}-{cv['realised_hedged_vol_ann'].max()*100:.1f}%"
    # --- walk-forward (daily constant exposure)
    wf = pd.read_csv(o / "walkforward_daily_metrics.csv")
    wa = wf[wf["basis"] == "A_per_SMLT_10m"].set_index("strategy")
    wb = wf[wf["basis"] == "B_per_total_capital"].set_index("strategy")
    keys = {"ctrl": "CONTROL|thr=0.00", "f25": "FIXED_25|thr=0.00", "f50": "FIXED_50|thr=0.00", "f75": "FIXED_75|thr=0.00",
            "f100": "FIXED_100|thr=0.00", "roll": "ROLLING_HSTAR_120|thr=WF_SELECTED", "state": "STATE_DEPENDENT|thr=WF_SELECTED"}
    for k, s in keys.items():
        F[f"wf_hevar_{k}"] = f"{wa.loc[s, 'HE_VAR']:.3f}"
        F[f"wf_vol_{k}"] = _pct(wa.loc[s, "ann_vol"], 1)
        F[f"wf_volB_{k}"] = _pct(wb.loc[s, "ann_vol"], 1)
        F[f"wf_es_{k}"] = _pct(wa.loc[s, "ES95_1d"], 2)
        F[f"wf_esB_{k}"] = _pct(wb.loc[s, "ES95_1d"], 2)
        F[f"wf_mdd_{k}"] = _pct(wa.loc[s, "max_drawdown"], 1)
        F[f"wf_up_{k}"] = f"{wa.loc[s, 'upside_capture']:.2f}"
        F[f"wf_down_{k}"] = f"{wa.loc[s, 'downside_capture']:.2f}"
        F[f"wf_to_{k}"] = f"{wa.loc[s, 'turnover_notional_multiple']:.1f}"
        F[f"wf_cost_{k}"] = _m(wa.loc[s, "total_costs_rub"])
        F[f"wf_h_{k}"] = f"{wa.loc[s, 'avg_hedge_ratio']:.2f}"
        F[f"wf_hepnl_{k}"] = f"{wa.loc[s, 'HE_PNL']:.2f}"
    F["wf_start"] = str(wa.loc[keys["ctrl"], "start"])
    ov = pd.read_csv(o / "walkforward_overlay_metrics.csv")
    ova = ov[ov["basis"] == "A_per_SMLT_10m"].set_index("strategy")
    for k, s in (("ovall", "OVERLAY_MIN_ES_ALL"), ("ovk2", "OVERLAY_MIN_ES_K2")):
        F[f"wf_hevar_{k}"] = f"{ova.loc[s, 'HE_VAR']:.3f}"
        F[f"wf_to_{k}"] = f"{ova.loc[s, 'turnover_notional_multiple']:.1f}"
        F[f"wf_es_{k}"] = _pct(ova.loc[s, "ES95_1d"], 2)
        F[f"wf_sleeves_{k}"] = f"{ova.loc[s, 'mean_active_sleeves']:.1f}"
    # --- episodes (full mechanics, historical)
    ep = pd.read_csv(o / "episodes_summary.csv")
    e = ep[(ep["capital"] == C)].set_index(["strategy", "horizon"])
    for s, k in (("CONTROL", "ctrl"), ("FIXED_50", "f50"), ("FIXED_100", "f100"), ("ROLLING_HSTAR_120", "roll")):
        for hz in ("6m", "12m"):
            F[f"ep_p05_{k}_{hz}"] = _m(e.loc[(s, hz), "p05_terminal_pnl"])
            F[f"ep_worst_{k}_{hz}"] = _m(e.loc[(s, hz), "worst_terminal_pnl"])
            F[f"ep_n_{hz}"] = str(int(e.loc[(s, hz), "n_episodes"]))
    F["ep_vmdrain_f100"] = _m(e.loc[("FIXED_100", "12m"), "max_cum_vm_drain"])
    F["ep_dvm_f100"] = _m(e.loc[("FIXED_100", "12m"), "max_daily_vm_outflow"])
    F["ep_peakcoll_f100"] = _m(e.loc[("FIXED_100", "12m"), "peak_total_collateral"])
    e25 = ep[ep["capital"] == 2_500_000].set_index(["strategy", "horizon"])
    F["ep_forced_roll_25"] = str(int(e25.loc[("ROLLING_HSTAR_120", "12m"), "forced_reductions"]))
    # --- bootstrap (decision date)
    bs = pd.read_csv(o / "bootstrap_strategy_metrics.csv")
    for v in ("FHS_FULL", "HS_FULL"):
        vv = "fhs" if v == "FHS_FULL" else "hs"
        for hz in ("6m", "12m"):
            b = bs[(bs["variant"] == v) & (bs["horizon"] == hz) & (bs["capital"] == C)].set_index("strategy")
            for s in b.index:
                t = s.replace("_listedIV", "L").replace("_realisedVol", "R").replace("OTC_", "").replace("DERISK_SELL_", "SELL")
                F[f"{vv}_{hz}_es_{t}"] = _m(b.loc[s, "ES95"])
                F[f"{vv}_{hz}_mean_{t}"] = _m(b.loc[s, "mean_pnl"])
                F[f"{vv}_{hz}_up_{t}"] = f"{b.loc[s, 'upside_capture']:.2f}"
                F[f"{vv}_{hz}_breach_{t}"] = _pct(b.loc[s, "p_sleeve_breach"])
                F[f"{vv}_{hz}_prem_{t}"] = _m(b.loc[s, "option_premium"])
                F[f"{vv}_{hz}_std_{t}"] = _m(b.loc[s, "std_pnl"])
                F[f"{vv}_{hz}_dvm95_{t}"] = _m(b.loc[s, "max_daily_vm_outflow_p95"])
                F[f"{vv}_{hz}_drain99_{t}"] = _m(b.loc[s, "max_cum_vm_drain_p99"])
    # --- capital sensitivity
    cb = pd.read_csv(o / "capital_budget_sensitivity.csv")
    for v in ("FHS_FULL", "HS_FULL"):
        vv = "fhs" if v == "FHS_FULL" else "hs"
        for hz in ("6m", "12m"):
            x = cb[(cb["variant"] == v) & (cb["horizon"] == hz)].set_index("hedge_capital")
            for cap in x.index:
                c = f"{cap/1e6:.1f}".replace(".", "p")
                F[f"cap_{vv}_{hz}_{c}_best"] = x.loc[cap, "best_feasible_direct_hedge"]
                F[f"cap_{vv}_{hz}_{c}_red"] = _pct(x.loc[cap, "ES95_reduction_pct_of_control"])
                F[f"cap_{vv}_{hz}_{c}_perrub"] = f"{x.loc[cap, 'ES95_reduction_per_rub_hedge_capital']:.2f}"
                F[f"cap_{vv}_{hz}_{c}_breach"] = _pct(x.loc[cap, "p_sleeve_breach"])
                F[f"cap_{vv}_{hz}_{c}_xstar"] = _pct(x.loc[cap, "rally_that_exhausts_sleeve_for_best"])
                F[f"cap_{vv}_{hz}_{c}_r5"] = x.loc[cap, "best_hedge_p_breach_le_5pct"]
                F[f"cap_{vv}_{hz}_{c}_r5red"] = _pct(x.loc[cap, "ES95_reduction_pct_p_breach_le_5pct"])
                F[f"cap_{vv}_{hz}_{c}_r20"] = x.loc[cap, "best_hedge_p_breach_le_20pct"]
                F[f"cap_{vv}_{hz}_{c}_r20red"] = _pct(x.loc[cap, "ES95_reduction_pct_p_breach_le_20pct"])
                F[f"cap_{vv}_{hz}_{c}_fin"] = _m(x.loc[cap, "borrowed_financing_cost_rub"])
    # --- complexity
    el = pd.read_csv(o / "complexity_elbow.csv")
    for _, r in el.iterrows():
        vv = "fhs" if r["variant"] == "FHS_FULL" else "hs"
        c = f"{r['capital']/1e6:.1f}".replace(".", "p")
        F[f"elbow_{vv}_{r['horizon']}_{c}"] = str(int(r["elbow_K"]))
        F[f"elbow_{vv}_{r['horizon']}_{c}_set"] = r["elbow_active"].replace("|", " + ")
        F[f"elbow_{vv}_{r['horizon']}_{c}_gain"] = _m(r["ES95_at_K1"] - r["ES95_at_Kmax"], 2)
    cx = pd.read_csv(o / "complexity_frontier.csv")
    x = cx[(cx["variant"] == "FHS_FULL") & (cx["capital"] == C) & (cx["horizon"] == "6m")].set_index("K")
    F["cx_lp_gain_6m"] = _m((x.loc[1, "lp_ES95_pct"] - x["lp_ES95_pct"].min()) * E, 2)
    F["cx_path_change_6m"] = _m(x["path_ES95"].max() - x.loc[1, "path_ES95"], 2)
    F["cx_rgbi_w"] = f"{x.loc[6, 'w_RATES_RGBI']:.2f}"
    F["cx_idx_w"] = f"{x.loc[6, 'w_INDEX']:.2f}"
    # --- scenarios / events
    ev = pd.read_csv(o / "event_overlay.csv")
    evx = ev[ev["horizon"] == "12m"].set_index(["event", "strategy"])
    F["ev_si_vm_f100"] = _m(-evx.loc[("strategic_investor", "FUT_100"), "day1_vm_rub"])
    F["ev_si_free_f100"] = _m(evx.loc[("strategic_investor", "FUT_100"), "free_sleeve_after_day1"])
    F["ev_si_free_f50"] = _m(evx.loc[("strategic_investor", "FUT_50"), "free_sleeve_after_day1"])
    F["ev_def_ctrl"] = _pct(evx.loc[("adverse_credit_event_default", "CONTROL"), "terminal_pnl_pct_smlt"])
    F["ev_def_f50"] = _pct(evx.loc[("adverse_credit_event_default", "FUT_50"), "terminal_pnl_pct_smlt"])
    F["ev_def_f100"] = _pct(evx.loc[("adverse_credit_event_default", "FUT_100"), "terminal_pnl_pct_smlt"])
    F["ev_def_sell50"] = _pct(evx.loc[("adverse_credit_event_default", "DERISK_SELL_50"), "terminal_pnl_pct_smlt"])
    sm = pd.read_csv(o / "scenario_matrix.csv")
    s12 = sm[(sm["horizon"] == "12m") & (sm["rates"] == "STABLE") & (sm["credit"] == "STABLE")].set_index(["strategy", "smlt_terminal_return"])
    for st in ("CONTROL", "FUT_50", "FUT_100", "DERISK_SELL_50", "OTC_PUT_90_listedIV", "OTC_COLLAR_90_120"):
        for xv in (-0.5, 0.0, 1.0):
            F[f"sm_{st}_{int(xv*100)}"] = _pct(s12.loc[(st, xv), "pnl_pct_of_smlt_10m"])
    F["otc_put90_12m_prem_pct"] = _pct(s12.loc[("OTC_PUT_90_listedIV", 0.0), "comp_premium"] / E)
    # --- sell-vs-hedge dominance (same fraction x of SMLT): deterministic cells and bootstrap metrics
    for x in (50, 100):
        a = sm[sm["strategy"] == f"DERISK_SELL_{x}"].set_index(["horizon", "smlt_terminal_return", "rates", "credit"])["pnl_rub"]
        b = sm[sm["strategy"] == f"FUT_{x}"].set_index(["horizon", "smlt_terminal_return", "rates", "credit"])["pnl_rub"]
        d = (a - b).dropna()
        F[f"dom_cells_sell{x}"] = f"{int((d >= 0).sum())}/{len(d)}"
        F[f"dom_minadv_sell{x}"] = _pct(d.min() / E, 1)
        F[f"dom_maxadv_sell{x}"] = _pct(d.max() / E, 1)
    cnt, tot = 0, 0
    for v in ("FHS_FULL", "HS_FULL", "HS_RECENT250"):
        for hz in ("6m", "12m"):
            b = bs[(bs["variant"] == v) & (bs["horizon"] == hz) & (bs["capital"] == C)].set_index("strategy")
            for x in (25, 50, 75, 100):
                s_, f_ = b.loc[f"DERISK_SELL_{x}"], b.loc[f"FUT_{x}"]
                tot += 1
                cnt += int(s_["ES95"] <= f_["ES95"] + 1e-6 and s_["mean_pnl"] >= f_["mean_pnl"] - 1e-6)
    F["dom_boot"] = f"{cnt}/{tot}"
    # --- state-dependent vs fixed hedge with the same average ratio (linear interpolation of fixed 50/75)
    h_s = float(wa.loc[keys["state"], "avg_hedge_ratio"])
    lo, hi = (0.50, keys["f50"]), (0.75, keys["f75"])
    wgt = (h_s - lo[0]) / (hi[0] - lo[0])
    interp = (1 - wgt) * wa.loc[lo[1], "HE_VAR"] + wgt * wa.loc[hi[1], "HE_VAR"]
    interp_es = (1 - wgt) * wa.loc[lo[1], "ES95_1d"] + wgt * wa.loc[hi[1], "ES95_1d"]
    F["state_interp_hevar"] = f"{interp:.3f}"
    F["state_interp_es"] = _pct(interp_es, 2)
    # --- covariance table rows used in text
    for est in ("sample", "ledoit_wolf", "oas", "ewma"):
        F[f"cov120_{est}_qlike"] = f"{c120.loc[est, 'mean_qlike']:.2f}"
        F[f"cov120_{est}_bias"] = f"{c120.loc[est, 'bias_ratio_real_over_pred']:.2f}"
        F[f"cov120_{est}_hvol"] = _pct(c120.loc[est, "realised_hedged_vol_ann"], 1)
    F["cov_first"] = str(cv["first_date"].iloc[0])
    # --- extra facts quoted in text
    P = ctx.read("prices.csv", index_col=0, parse_dates=True)
    px = P["SMLT"].loc["2026-01-01":ctx.cutoff].dropna()
    r3 = px / px.shift(3) - 1
    t3 = r3.idxmax()
    F["max3d_2026"] = _pct(float(r3.max()))
    F["max3d_2026_from"] = f"{float(px.shift(3).loc[t3]):.0f}"
    F["max3d_2026_to"] = f"{float(px.loc[t3]):.0f}"
    F["max3d_2026_date"] = str(t3.date())
    # ALL headline futures thresholds come from the contract-level engine (A3): no shortcut formulas
    from ..strategies.futures import contract_position, engine_jump_state, rally_exhausting_sleeve
    mr, S0f, F0f = num(mi, "mr1"), num(mi, "S0"), num(mi, "F0")
    n_sh = E / S0f
    h_rec = 0.5
    pos = contract_position(h_rec, n_sh, F0f)
    F["rec_contracts"] = f"{pos['contracts']:,}"
    F["rec_notional"] = _m(pos["notional_rub"], 2)
    F["rec_im"] = _m(engine_jump_state(h_rec, n_sh, S0f, F0f, 0.0, C, mr)["im_req"], 2)
    F["rec_xstar"] = _pct(rally_exhausting_sleeve(h_rec, n_sh, S0f, F0f, C, mr))
    buf = 2e6
    F["rec_buffer"] = _m(buf, 1)
    F["rec_trigger_rally"] = _pct(rally_exhausting_sleeve(h_rec, n_sh, S0f, F0f, C, mr, buffer=buf))
    F["full_contracts"] = f"{contract_position(1.0, n_sh, F0f)['contracts']:,}"
    F["im_full"] = _m(engine_jump_state(1.0, n_sh, S0f, F0f, 0.0, C, mr)["im_req"], 2)
    F["xstar_full_10m"] = _pct(rally_exhausting_sleeve(1.0, n_sh, S0f, F0f, C, mr))
    lim = ctx.read("ncc_futures_limits.csv", parse_dates=["tradedate"])
    last = lim[lim["tradedate"] <= pd.Timestamp(ctx.cutoff)].sort_values("tradedate").groupby("assetcode").tail(1).set_index("assetcode")
    F["mr_mix"] = _pct(float(last.loc["MIX", "mr1"])) if "MIX" in last.index else "n/a"
    F["mr_imoex"] = _pct(float(last.loc["IMOEX", "mr1"])) if "IMOEX" in last.index else "n/a"
    run = json.loads((o / "run_log.json").read_text())
    F["basis_resid_std"] = _pct(run["basis_model"]["resid_std"], 1)
    b6 = bs[(bs["variant"] == "FHS_FULL") & (bs["horizon"] == "6m") & (bs["capital"] == C)].set_index("strategy")
    F["put90_6m_prem_pct"] = _pct(b6.loc["OTC_PUT_90_listedIV", "option_premium"] / E)
    # "never the worst" among executable strategies in every deterministic cell
    ex = sm[sm["executability"] != "THEORETICAL_ONLY"]
    g = ex.groupby(["horizon", "smlt_terminal_return", "rates", "credit"])
    worst = g.apply(lambda d: d.loc[d["pnl_rub"].idxmin(), "strategy"])
    for st in ("FUT_50", "DERISK_SELL_50", "FUT_100", "CONTROL"):
        F[f"worst_cells_{st}"] = f"{int((worst == st).sum())}/{len(worst)}"
    rob = pd.read_csv(o / "scenario_robustness.csv").set_index(["horizon", "strategy"])
    for st in ("FUT_50", "DERISK_SELL_50", "CONTROL", "FUT_100"):
        for hz in ("6m", "12m"):
            F[f"worstcell_{st}_{hz}"] = _pct(rob.loc[(hz, st), "worst_cell_pnl_pct"])
    man = pd.read_csv(ctx.root / "data" / "source_manifest.csv", dtype=str).drop_duplicates("source_id")
    F["n_snapshots"] = f"{len(man):,}"
    F["n_paths"] = f"{ctx.cfg['simulation']['n_paths']:,}"
    defn = [x for x in ("SBER", "MTSS", "PHOR", "LKOH", "TRNFP") if x in cs.index]
    F["def_stress_min"] = f"{cs.loc[defn, 'corr_market_stress'].min():.2f}"
    F["def_stress_max"] = f"{cs.loc[defn, 'corr_market_stress'].max():.2f}"
    facts = pd.read_csv(ctx.root / "data" / "manual" / "issuer_facts.csv").set_index("variable")["value"]
    for k in ("net_income_1h2026", "corporate_debt", "project_finance", "net_debt_ex_escrow", "revenue_1h2026"):
        F[f"fact_{k}"] = str(facts.get(k))
    # ================= Phase B / second pass
    if (o / "smart_portfolio_comparison.csv").exists():
        dc = pd.read_csv(o / "smart_portfolio_comparison.csv").set_index("strategy")
        spec = pd.read_csv(o / "phase_b_sleeves.csv").set_index("name")
        fs = F0f / S0f
        for lab, key in (("STATIC_SMART_PORTFOLIO", "st"), ("ROBUST_SMART_PORTFOLIO", "rb"), ("ADAPTIVE_SMART_PORTFOLIO", "ad"),
                         ("FUT_25", "f25"), ("FUT_50", "f50"), ("FUT_75", "f75"), ("FUT_100", "f100"), ("CONTROL", "ctl"),
                         ("DE_RISK_SELL_50", "s50"), ("ROLLING_HSTAR", "rh"), ("CURRENT_STATE_DEPENDENT", "cs")):
            if lab not in dc.index:
                continue
            r = dc.loc[lab]
            for hz in ("6m", "12m"):
                F[f"dc_{key}_es_{hz}"] = _m(r[f"ES95_{hz}"])
                F[f"dc_{key}_wes_{hz}"] = _m(r[f"worst_model_ES95_{hz}"])
                F[f"dc_{key}_br_{hz}"] = _pct(r[f"sleeve_breach_prob_{hz}"])
                F[f"dc_{key}_up_{hz}"] = f"{r[f'upside_capture_{hz}']:.2f}"
                F[f"dc_{key}_mean_{hz}"] = _m(r[f"mean_pnl_{hz}"])
                F[f"dc_{key}_cost_{hz}"] = _m(r[f"hedge_cost_{hz}"])
                F[f"dc_{key}_hw_{hz}"] = _m(r[f"worst_deterministic_hist_window_{hz}"])
                wd = r.get(f"w{hz}_DIRECT", 0.0)
                F[f"dc_{key}_direct_share_{hz}"] = _pct(-wd / fs)
                F[f"dc_{key}_direct_contracts_{hz}"] = f"{round(-wd * E / F0f):,}"
                wm = r.get(f"w{hz}_MARKET", 0.0)
                F[f"dc_{key}_mix_notional_{hz}"] = _m(-wm * E)
                F[f"dc_{key}_mix_contracts_{hz}"] = f"{-wm * E / spec.loc['MARKET', 'contract_notional']:.1f}"
            F[f"dc_{key}_im"] = _m(r["initial_margin"])
            F[f"dc_{key}_k"] = str(int(r["n_active_sleeves"]))
            F[f"dc_{key}_wf_hevar"] = f"{r['wf_HE_VAR']:.3f}" if pd.notna(r["wf_HE_VAR"]) else "n/a"
            F[f"dc_{key}_ep_es"] = _m(r["ep6m_ES95_terminal"]) if pd.notna(r["ep6m_ES95_terminal"]) else "n/a"
            F[f"dc_{key}_ep_br"] = _pct(r["ep6m_breach_share"]) if pd.notna(r["ep6m_breach_share"]) else "n/a"
            F[f"dc_{key}_note"] = str(r.get("note", "")) if pd.notna(r.get("note", np.nan)) else ""
        ad = dc.loc["ADAPTIVE_SMART_PORTFOLIO"]
        F["ad_weights_6m"] = ", ".join(f"{n} {ad[f'w6m_{n}']:+.2f}" for n in spec.index if abs(ad[f"w6m_{n}"]) > 1e-3)
        sel = pd.read_csv(o / "smart_selected_portfolios.csv")
        for hz in ("6m", "12m"):
            for scope, k in (("ANY", "any"), ("DIRECT_ONLY", "dir")):
                r = sel[(sel["rule"] == "static") & (sel["selection_scope"] == scope) & (sel["horizon"] == hz) & (sel["capital"] == C)].iloc[0]
                F[f"sel_{k}_{hz}_es"], F[f"sel_{k}_{hz}_br"] = _m(r["ES95"]), _pct(r["p_sleeve_breach"], 1)
                F[f"sel_{k}_{hz}_up"], F[f"sel_{k}_{hz}_mean"] = f"{r['upside_capture']:.3f}", _m(r["mean_pnl"])
                F[f"sel_{k}_{hz}_subset"] = r["subset"]
        cxm = pd.read_csv(o / "complexity_multidim.csv")
        for cap in sorted(cxm["capital"].unique()):
            for hz in ("6m", "12m"):
                x = cxm[(cxm["capital"] == cap) & (cxm["horizon"] == hz)]
                if x.empty:
                    continue
                best = x.sort_values(["ES95_FHS", "K_active"]).iloc[0]
                c = f"{cap/1e6:.1f}".replace(".", "p")
                F[f"cx2_{hz}_{c}_set"] = best["active"].replace("|", " + ")
                F[f"cx2_{hz}_{c}_gain"] = _m(best["es_gain_vs_K1_rub"])
                F[f"cx2_{hz}_{c}_dom"] = "yes" if bool(x["materially_pareto_dominates_K1"].fillna(False).any()) else "no"
        aw = pd.read_csv(o / "adaptive_walkforward_metrics.csv")
        aw = aw[aw["basis"] == "A_per_SMLT_10m"].set_index("strategy")
        for k, n in (("ad", "ADAPTIVE_SMART"), ("adm", "ADAPTIVE_SMART_MONTHLY"), ("fs", "FIXED_SMART_PRE_OOS"),
                     ("f50", "FUT_50"), ("f75", "FUT_75"), ("f100", "FUT_100"), ("cs", "CURRENT_STATE_DEPENDENT"), ("rh", "ROLLING_HSTAR")):
            F[f"aw_{k}_hevar"] = f"{aw.loc[n, 'HE_VAR']:.3f}"
            F[f"aw_{k}_es"] = _pct(aw.loc[n, "ES95_1d"], 2)
            F[f"aw_{k}_to"] = f"{aw.loc[n, 'turnover_notional_multiple']:.1f}"
            F[f"aw_{k}_cost"] = _m(aw.loc[n, "total_costs_rub"])
            F[f"aw_{k}_mdd"] = _pct(aw.loc[n, "max_drawdown"], 1)
            F[f"aw_{k}_up"] = f"{aw.loc[n, 'upside_capture']:.2f}"
            F[f"aw_{k}_sl"] = f"{aw.loc[n, 'mean_active_sleeves']:.1f}"
        lw = pd.read_csv(o / "liquidity_waterfall_results.csv")
        def lwv(stg, hz, pol, col):
            r = lw[(lw["strategy"] == stg) & (lw["horizon"] == hz) & (lw["liquidity_policy"] == pol)]
            return float(r[col].iloc[0]) if len(r) else np.nan
        for k, n in (("ad", "ADAPTIVE_SMART"), ("fs", "FIXED_SMART_PRE_OOS"), ("f50", "FUT_50"), ("f75", "FUT_75"), ("f100", "FUT_100")):
            for hz in ("6m", "12m"):
                F[f"lw_{k}_{hz}_es"] = _m(lwv(n, hz, "WATERFALL", "ES95_terminal"))
                F[f"lw_{k}_{hz}_br"] = _pct(lwv(n, hz, "WATERFALL", "breach_share"))
                F[f"lw_{k}_{hz}_sold"] = _m(lwv(n, hz, "WATERFALL", "shares_sold_value"))
                F[f"lw_{k}_{hz}_mean"] = _m(lwv(n, hz, "WATERFALL", "mean_terminal"))
        F["lw_fs_12m_s3"] = str(int(lwv("FIXED_SMART_PRE_OOS", "12m", "WATERFALL", "step3")))
        F["lw_fs_12m_s4"] = str(int(lwv("FIXED_SMART_PRE_OOS", "12m", "WATERFALL", "step4")))
        F["lw_fs_12m_imm"] = str(int(lwv("FIXED_SMART_PRE_OOS", "12m", "IMMEDIATE_CUT", "immediate")))
        F["lw_fs_12m_mean_imm"] = _m(lwv("FIXED_SMART_PRE_OOS", "12m", "IMMEDIATE_CUT", "mean_terminal"))
        rs = pd.read_csv(o / "regime_sensitivity.csv")
        F["rs_hevar_range"] = f"{rs['HE_VAR'].min():.3f}-{rs['HE_VAR'].max():.3f}"
        F["rs_to_range"] = f"{rs['turnover'].min():.0f}-{rs['turnover'].max():.0f}"
        r0 = rs[rs["scale"] == 1.0].iloc[0]
        for lab in ("NORMAL", "CREDIT_STRESS", "MARKET_STRESS", "RATES_STRESS", "COMBINED_STRESS"):
            F[f"rf_{lab}"] = _pct(r0.get(f"freq_{lab}", 0.0))
        rg = ctx.read("regimes_daily.csv", "interim", index_col=0, parse_dates=True)
        F["regime_cutoff"] = str(rg.loc[:ctx.cutoff, "regime"].iloc[-1])
        cst = pd.read_csv(o / "correlation_stability.csv").set_index("pair")
        for pair, k in (("SMLT~MARKET", "mkt"), ("SMLT~DIRECT", "dir"), ("SMLT~RATES", "rat"), ("SMLT~GOLD", "gld"),
                        ("SMLT~CNY", "cny"), ("SMLT~OIL", "oil"), ("SMLT~SMLT_BOND_IDX", "bnd")):
            if pair in cst.index:
                for col in ("full", "250d", "market_stress", "credit_stress", "downside"):
                    F[f"cr_{k}_{col}"] = f"{cst.loc[pair, col]:.2f}"
                F[f"cr_{k}_class"] = cst.loc[pair, "classification"]
        ccl = pd.read_csv(o / "collateral_conclusions.csv")
        for v, vv in (("FHS_FULL", "fhs"), ("HS_FULL", "hs")):
            for hz in ("6m", "12m"):
                for nm, k in (("A_CASH_ONLY_CONSERVATIVE", "A"), ("B_PARTIALLY_PRODUCTIVE", "B"), ("C_HIGH_COLLATERAL_EFFICIENCY", "C"),
                              ("yf0.00_hc0.00_mm1.50", "M15")):
                    r = ccl[(ccl["variant"] == v) & (ccl["horizon"] == hz) & (ccl["capital"] == C) & (ccl["collateral"] == nm)]
                    if len(r):
                        r = r.iloc[0]
                        F[f"cl_{vv}_{hz}_{k}_dom"] = r["sell_dominates_fut_same_fraction"]
                        F[f"cl_{vv}_{hz}_{k}_best20"] = r["best_fut_breach_le_20pct"]
                        F[f"cl_{vv}_{hz}_{k}_b50"] = _pct(r["breach_FUT_50"])
                        F[f"cl_{vv}_{hz}_{k}_b100"] = _pct(r["breach_FUT_100"])
        fdp = pd.read_csv(o / "factor_decomposition.csv").set_index("window")
        for w, k in (("FULL", "full"), ("LAST_250D", "250")):
            for f in ("MKT", "SECTOR", "RATES", "CREDIT"):
                F[f"fx_{k}_{f}"] = _pct(fdp.loc[w, f"shapley_{f}"])
            F[f"fx_{k}_total"] = _pct(fdp.loc[w, "total_R2"])
    # K=2 weights at 2.5m
    k2 = cx[(cx["variant"] == "FHS_FULL") & (cx["capital"] == 2_500_000) & (cx["horizon"] == "6m") & (cx["K"] == 2)].iloc[0]
    F["k2_small_fut"] = _pct(-k2["w_SMLT_FUT"])
    F["k2_small_idx"] = _pct(-k2["w_INDEX"])
    return F


def render(ctx) -> None:
    F = compute(ctx)
    (ctx.out / "key_findings.json").write_text(json.dumps(F, indent=1, ensure_ascii=False))
    tmpl = (Path(__file__).parent / "templates" / "RESEARCH_REPORT.md.tmpl").read_text()
    (ctx.root / "RESEARCH_REPORT.md").write_text(tmpl.format(**F))
