"""V3 Phases 4-6: regime portfolio library, low-turnover state machine, active SMLT exposure branch.

s20 library        decision-date library P_<state>: candidates on TRAIN (state-pooled regime scenarios + min-max over
                   {state, FHS, HS}), robust selection on VALIDATION with pre-declared state constraints, integer
                   contracts, TEST (20 seeds) cross-state metrics, merge of indistinguishable states.
s21 state machine  historical walk-forward with a CAUSAL library (re-estimated semi-annually from data strictly before
                   each date), confirmation / minimum holding / materiality / emergency override; parameters chosen
                   walk-forward only; stacked historical episodes through the dynamic engine.
s22 active         ACTIVE_EXPOSURE_MANAGEMENT branch (share fraction 25..100%, never above the initial position),
                   same state machine, compared with fixed hedge overlays; kept separate from pure hedging.
"""
from __future__ import annotations

import ast
from itertools import combinations

import numpy as np
import pandas as pd

from . import pipeline as pl
from .optimization.overlay import Problem, Sleeve, min_es
from .portfolio import dynamic
from .portfolio import regime_risk as rr
from .portfolio import sleeves as sleeves_mod
from .portfolio import validation as va
from .risk.metrics import es_hist, he_var, summary

STATE_COLS = ("IMOEX", "SMLT_BOND_IDX", "RGBI")


# ------------------------------------------------------------------ helpers
def _eq_cost(ctx, S0: float) -> float:
    return (0.1 + ctx.cfg["costs"]["equity_commission_bps"] * 1e-4 * S0) / S0


def dyn_sleeves(specs) -> dynamic.Sleeves:
    names = [s.name for s in specs]
    return dynamic.Sleeves([s.series for s in specs], np.array([s.mr for s in specs]), np.array([s.cost_one_way for s in specs]),
                           np.array([float(s.rolls_per_year) for s in specs]), names.index("DIRECT"))


def frozen_thresholds(ctx) -> dict:
    """Long-window thresholds frozen at their cutoff values for state detection on simulated paths."""
    P, R, V = pl._panel(ctx)
    cs = ctx.read("credit_state.csv", index_col=0, parse_dates=True)
    th = ctx.cfg["smart_portfolio"]["regimes"]
    t = pd.Timestamp(ctx.cutoff)
    mkt = R["IMOEX"].fillna(0.0).loc[:t]
    v20 = mkt.rolling(20).std() * np.sqrt(252)
    ytm = cs["ytm_median_pct"].ffill().loc[:t]
    dur = cs["bond_duration_median_days"].ffill().loc[:t]
    ps = rr.price_state(R["SMLT"], ctx.cfg["v3"]["price_state"]).loc[:t]
    return {"mkt_vol_q": float(v20.rolling(500, min_periods=120).quantile(th["market_vol_pct"]).iloc[-1]),
            "ytm0": float(ytm.iloc[-1]), "ytm_q": float(ytm.rolling(250, min_periods=60).quantile(th["credit_ytm_pct"]).iloc[-1]),
            "bond_dmod": float(dur.iloc[-1] / 365.0 / (1 + ytm.iloc[-1] / 100.0)),
            "smlt_vol_q": float(ps["smlt_vol20_q"].iloc[-1])}


def prefix_frame(ctx, n: int = 300) -> pd.DataFrame:
    P, R, V = pl._panel(ctx)
    return R[["SMLT", "IMOEX", "SMLT_BOND_IDX", "RGBI"]].loc[:pd.Timestamp(ctx.cutoff)].tail(n).fillna(0.0)


def path_codes(ctx, Pth: dict, fz: dict, pref: pd.DataFrame) -> dict:
    lab = rr.path_labels(Pth, pref, ctx.cfg["smart_portfolio"]["regimes"], ctx.cfg["v3"]["price_state"], fz)
    return {"portfolio": rr.portfolio_codes(lab["macro"], lab["price"]), "active": rr.active_codes(lab["macro"], lab["price"]),
            "shock": lab["shock"], "macro": lab["macro"], "price": lab["price"]}


def protective_rank(ctx) -> np.ndarray:
    pr = ctx.cfg["v3"]["library"]["protective_rank"]
    return np.array([pr[s] for s in rr.PORTFOLIO_STATES])


# ------------------------------------------------------------------ Phase 4: decision-date library
def s20_library(ctx) -> None:
    from .v3_stages import _regime_inputs
    lib = ctx.cfg["v3"]["library"]
    v3 = ctx.cfg["v3"]
    specs = sleeves_mod.build(ctx)
    names = [s.name for s in specs]
    P, R, X, reg, ps, pst = _regime_inputs(ctx)
    masks = rr.state_masks(reg, ps)
    hz = lib["horizon_days"]
    sel_rows, rep_rows, facs, chosen = [], [], {}, {}
    for st in rr.PORTFOLIO_STATES:
        con = lib["constraints"][st]
        fac = va.ScenarioFactory(ctx, pl, specs, regime_mask=masks[st], label=st, pool_n0=v3["shrink_n0"], state_cols=STATE_COLS)
        facs[st] = fac
        cands = va.generate(ctx, fac, hz, ctx.C, models=lib["models"], minmax_models=lib["minmax_models"], max_K=lib["max_K"],
                            margin_cap=con["margin_budget"], n_lp=lib["n_lp"], upside_floors=[None, con["min_upside"]])
        for i, c in enumerate(cands):
            c["cand_id"] = f"{st}_{i:04d}"
        cdf = pd.DataFrame([{"cand_id": c["cand_id"], "family": c["family"], "gen_model": c["gen_model"],
                             "K_active": sum(1 for n in names if abs(c["h"][n]) > 1e-4), **{f"w_{n}": c["h"][n] for n in names}}
                            for c in cands])
        ev = va.evaluate(ctx, fac, cands, hz, ctx.C, "VAL", v3["n_val_paths"], models=lib["eval_models"], include_hist=False)
        summ = va.summarise(ev, primary="REGIME_COND", admissible=lib["eval_models"]).merge(cdf, on="cand_id")
        rule = {"static": {"max_breach": lib["max_worst_breach"], "min_upside": con["min_upside"]},
                "robust": {"max_worst_breach": lib["max_worst_breach"], "min_worst_upside": con["min_upside"]}}
        best = va.select(summ.drop(columns=["family", "K_active", "gen_model"]), cdf, "robust", rule, "ANY")
        bdir = va.select(summ.drop(columns=["family", "K_active", "gen_model"]), cdf, "robust", rule, "DIRECT_ONLY")
        for tag, r in (("LIBRARY", best), ("DIRECT_ONLY", bdir)):
            if r is None:
                sel_rows.append({"state": st, "selection": tag, "cand_id": "NONE"})
                continue
            h = {n: float(r[f"w_{n}"]) for n in names}
            hi, rep = va.round_contracts(specs, h, ctx.E)
            for x in rep:
                rep_rows.append({"state": st, "selection": tag, **x})
            sel_rows.append({"state": st, "selection": tag, "cand_id": r["cand_id"],
                             "gen_model": summ.set_index("cand_id").loc[r["cand_id"], "gen_model"],
                             "n_regime_days": fac.regime_n, "pool_lambda_full_sample": fac.pool_lambda,
                             "constraint_min_upside": con["min_upside"], "constraint_margin_budget": con["margin_budget"],
                             "val_worst_model_ES95": r["worst_model_ES95"], "val_worst_breach": r["worst_model_breach"],
                             "val_worst_upside": r["worst_model_upside"], "val_state_ES95": r["ES95"],
                             **{f"w_{n}": h[n] for n in names}, **{f"int_w_{n}": hi[n] for n in names},
                             **{f"contracts_{n}": s.contracts(h[n], ctx.E) for n, s in zip(names, specs)},
                             "initial_margin_rub_int": sum(abs(hi[n]) * ctx.E * s.mr for n, s in zip(names, specs))})
            if tag == "LIBRARY":
                chosen[st] = hi
    lib_df = pd.DataFrame(sel_rows)
    # merge economically indistinguishable states (integer portfolios within merge_max_abs_diff per sleeve)
    groups, rep_of = [], {}
    for st in rr.PORTFOLIO_STATES:
        if st not in chosen:
            continue
        for g in groups:
            if max(abs(chosen[st][n] - chosen[g[0]][n]) for n in names) <= lib["merge_max_abs_diff"]:
                g.append(st)
                break
        else:
            groups.append([st])
    for g in groups:
        for st in g:
            rep_of[st] = g[0]
    lib_df["merged_with"] = lib_df["state"].map(lambda s: rep_of.get(s, s))
    lib_df["merge_group"] = lib_df["state"].map(lambda s: "|".join(next((g for g in groups if s in g), [s])))
    ctx.save(lib_df, "regime_portfolio_library.csv", index=False)
    ctx.save(pd.DataFrame(rep_rows), "regime_library_integer_contracts.csv", index=False)
    # ---- TEST: every library portfolio on every state's pooled scenarios (+FHS/HS), 20 seeds
    fins = [{"cand_id": f"LIB_{st}", "h": chosen[st]} for st in chosen]
    S0, F0 = facs[rr.PORTFOLIO_STATES[0]].B["S0"], facs[rr.PORTFOLIO_STATES[0]].B["F0"]
    for x in (0.25, 0.5, 0.75, 1.0):
        fins.append({"cand_id": f"FUT_{int(x * 100)}", "h": {**{n: 0.0 for n in names}, "DIRECT": -round(x * ctx.E / S0) * F0 / ctx.E}})
    fins.append({"cand_id": "CONTROL", "h": {n: 0.0 for n in names}})
    rows = []
    for st, fac in facs.items():
        for k in range(v3["seeds"]["n_test"]):
            ev = va.evaluate(ctx, fac, fins, hz, ctx.C, "TEST", v3["n_test_paths"] // 2, k=k,
                             models=lib["eval_models"] + ["FHS_FULL"], include_hist=False)
            rows.append(ev.assign(scenario_state=st))
    T = pd.concat(rows, ignore_index=True)
    ctx.save(T, "regime_library_test_metrics.csv", "interim", index=False)
    g = T.groupby(["scenario_state", "model", "cand_id"])
    met = g.agg(n_seeds=("seed_k", "nunique"), ES95_mean=("ES95", "mean"), ES95_sd=("ES95", "std"), ES99_mean=("ES99", "mean"),
                breach_mean=("p_sleeve_breach", "mean"), upside_mean=("upside_capture", "mean"), mean_pnl=("mean_pnl", "mean"),
                initial_im=("initial_im", "mean")).reset_index()
    ctx.save(met, "regime_portfolio_metrics.csv", index=False)
    ctx.log["v3_library"] = {st: {n: round(chosen[st][n], 4) for n in names if abs(chosen[st][n]) > 1e-9} for st in chosen}
    ctx.log["v3_library_merge_groups"] = ["|".join(g) for g in groups]


def library_matrix(ctx, specs, lib_df: pd.DataFrame | None = None) -> np.ndarray:
    lib_df = lib_df if lib_df is not None else pd.read_csv(ctx.out / "regime_portfolio_library.csv")
    names = [s.name for s in specs]
    M = np.zeros((len(rr.PORTFOLIO_STATES), len(names)))
    for i, st in enumerate(rr.PORTFOLIO_STATES):
        r = lib_df[(lib_df["state"] == st) & (lib_df["selection"] == "LIBRARY")]
        if len(r) and r["cand_id"].iloc[0] != "NONE":
            rep = r["merged_with"].iloc[0]
            rr_ = lib_df[(lib_df["state"] == rep) & (lib_df["selection"] == "LIBRARY")].iloc[0]
            M[i] = [float(rr_.get(f"int_w_{n}", 0.0)) for n in names]
    return M


# ------------------------------------------------------------------ Phase 5: causal library (walk-forward)
def causal_library(ctx, specs, R: pd.DataFrame, codes: pd.Series, date: pd.Timestamp, seed: int) -> np.ndarray:
    """Library re-estimated with data STRICTLY BEFORE `date`: per state a min-ES LP (6m horizon, state-pooled HS,
    zero drift) over every sleeve subset with K<=2, state margin budget and upside floor; K=2 only if it cuts ES by
    >= 1% of E versus the best K=1 (materiality)."""
    from .scenarios.bootstrap import block_bootstrap_indices
    lib = ctx.cfg["v3"]["library"]
    sm = ctx.cfg["v3"]["state_machine"]
    names = [s.name for s in specs]
    cols = ["SMLT"] + [s.series for s in specs]
    start = pd.Timestamp(ctx.cfg["simulation"]["bootstrap_start"])
    X = R.loc[(R.index >= start) & (R.index < date), cols].dropna()
    X = X - X.mean()
    cd = codes.reindex(X.index)
    hz, tau = lib["horizon_days"], lib["horizon_days"] / 252
    out = np.zeros((len(rr.PORTFOLIO_STATES), len(names)))
    mat = ctx.cfg["smart_portfolio"]["selection"]["material_es_gain_pct_of_E"]
    for i, st in enumerate(rr.PORTFOLIO_STATES):
        sub = X[(cd == i).values]
        if len(sub) < 20:
            sub = X
        lam = ctx.cfg["v3"]["shrink_n0"] / (ctx.cfg["v3"]["shrink_n0"] + len(sub))
        reps = max(1, int(round((1 - lam) / lam * len(X) / len(sub))))
        Z = pd.concat([sub] * reps + [X]).values
        idx = block_bootstrap_indices(len(Z), hz, sm["causal_n_scen"], ctx.cfg["simulation"]["block_length"], seed + i)
        HR = np.prod(1 + Z[idx], axis=1) - 1
        con = lib["constraints"][st]
        best = {1: None, 2: None}
        for k in (1, 2):
            for sub_n in combinations(range(len(names)), k):
                use = [specs[j] for j in sub_n]
                prob = Problem(r_s=HR[:, 0], R=HR[:, [1 + j for j in sub_n]],
                               sleeves=[Sleeve(s.name, s.lo, s.hi, cost=s.horizon_cost(tau), margin=s.mr) for s in use],
                               k_liq=ctx.C / ctx.E, m_liq=np.zeros(len(HR)), margin_budget=con["margin_budget"] * ctx.C / ctx.E,
                               gross_cap=ctx.cfg["smart_portfolio"]["gross_cap"], min_upside_capture=con["min_upside"])
                sol = min_es(prob)
                if not sol.status.startswith("optimal"):
                    continue
                if best[k] is None or sol.es < best[k][0]:
                    h = np.zeros(len(names))
                    h[list(sub_n)] = sol.h
                    best[k] = (sol.es, h)
        pick = best[1]
        if best[2] is not None and (pick is None or best[2][0] <= pick[0] - mat):
            pick = best[2]
        if pick is not None:
            out[i] = [specs[j].rounded_h(pick[1][j], ctx.E) for j in range(len(names))]
    return out


def sm_path(codes: np.ndarray, confirm: int, hold: int, shock: np.ndarray | None, prot: np.ndarray | None) -> np.ndarray:
    """State-machine portfolio state on ONE historical path (same logic as portfolio/dynamic.run)."""
    cur = int(codes[0])
    cand, n, since = cur, 0, 0
    out = np.empty(len(codes), int)
    out[0] = cur
    for t in range(1, len(codes)):
        o = int(codes[t])
        n = n + 1 if o == cand else 1
        cand = o
        since += 1
        want = cand != cur and n >= max(confirm, 1) and since >= hold
        if prot is not None and cand != cur and prot[cand] > prot[cur] and shock is not None and shock[t]:
            want = True
        if want:
            cur, since = cand, 0
        out[t] = cur
    return out


def _targets_frame(dates, states, lib_by_date: dict, names) -> pd.DataFrame:
    """Daily target fractions of E from the state path and the causal library in force."""
    keys = sorted(lib_by_date)
    rows = []
    for d, s in zip(dates, states):
        k = [x for x in keys if x <= d]
        M = lib_by_date[k[-1]] if k else np.zeros((len(rr.PORTFOLIO_STATES), len(names)))
        rows.append(M[s])
    tg = pd.DataFrame(rows, index=dates, columns=names)
    tg["regime"] = "SM"
    return tg


def _causal_select(runs: dict, key_fn) -> tuple[pd.Series, pd.DataFrame]:
    """Quarterly: pick the parameter key with the lowest realised daily ES95 over PRIOR days (tie: turnover)."""
    keys = sorted(runs)
    idx = runs[keys[0]].index
    pieces, rows = [], []
    qs = list(pd.date_range(idx.min(), idx.max(), freq="QS"))
    if not qs or qs[0] > idx.min():
        qs = [idx.min()] + qs
    best = keys[0]
    for i, q in enumerate(qs):
        qe = qs[i + 1] if i + 1 < len(qs) else idx.max() + pd.Timedelta(days=1)
        prior = idx[idx < q]
        if len(prior) >= 60:
            best = min(keys, key=lambda k: (key_fn(runs[k].loc[prior]), runs[k].loc[prior, "turnover"].sum()))
        rows.append({"quarter_start": q, "selected": str(best)})
        pieces.append(runs[best].loc[(idx >= q) & (idx < qe)])
    return pd.concat(pieces), pd.DataFrame(rows)


def _episodes(R: pd.DataFrame, cols: list[str], dates: pd.DatetimeIndex, hz: int, step: int = 21):
    """Stack historical windows as paths: returns {col: (n_ep, hz)} and episode start dates."""
    starts = [dates[k] for k in range(0, len(dates) - hz, step)]
    pos = R.index.get_indexer(starts)
    P = {c: np.stack([np.nan_to_num(R[c].values[p + 1:p + 1 + hz]) for p in pos]) for c in cols}
    return P, starts, pos


def s21_state_machine(ctx) -> None:
    from .phase2 import _env
    from .portfolio import adaptive
    from .v3_stages import _regime_inputs
    smc = ctx.cfg["v3"]["state_machine"]
    specs = sleeves_mod.build(ctx)
    names = [s.name for s in specs]
    P, R, X, reg, ps, pst = _regime_inputs(ctx)
    hc = rr.history_codes(reg, ps)
    env = _env(ctx, specs)
    cols = ["SMLT"] + [s.series for s in specs]
    oos = pd.Timestamp(ctx.cfg["walk_forward"]["oos_start"])
    t = pd.Timestamp(ctx.cutoff)
    dates = R[cols].dropna().index
    dates = dates[(dates >= oos) & (dates <= t)]
    # causal library, re-estimated semi-annually with data < date
    est_dates = [d for d in pd.date_range(oos, t, freq="MS") if d.month in smc["reestimate_months"]]
    est_dates = [dates[dates >= d][0] for d in est_dates if (dates >= d).any()]
    if dates[0] not in est_dates:
        est_dates = [dates[0]] + est_dates
    lib_by_date = {}
    lrows = []
    for j, d in enumerate(est_dates):
        M = causal_library(ctx, specs, R, hc["portfolio"], d, ctx.cfg["simulation"]["seed"] + 500 + 10 * j)
        lib_by_date[d] = M
        for i, st in enumerate(rr.PORTFOLIO_STATES):
            lrows.append({"estimation_date": d, "state": st, **{f"w_{n}": M[i, k] for k, n in enumerate(names)}})
    ctx.save(pd.DataFrame(lrows), "causal_library_walkforward.csv", index=False)
    prot = protective_rank(ctx)
    codes = hc["portfolio"].reindex(dates).values
    shock = (R["SMLT"].reindex(dates).abs() >= smc["emergency_shock"]).values
    # ---- daily constant-exposure walk-forward for every (confirm, hold)
    runs, meta = {}, []
    for cf in smc["confirm_grid"]:
        for hd in smc["hold_grid"]:
            stp = sm_path(codes, cf, hd, shock, prot)
            tg = _targets_frame(dates, stp, lib_by_date, names)
            d = adaptive.daily_walk(env, tg, dates, smc["threshold"])
            d["sm_state"] = [rr.PORTFOLIO_STATES[s] for s in stp]
            runs[(cf, hd)] = d
            meta.append({"confirm": cf, "hold": hd, "n_switches": int((np.diff(stp) != 0).sum())})
    es = lambda d: es_hist(d["pnl_total"].values, 0.95)  # noqa: E731
    sel_series, sel_tab = _causal_select(runs, es)
    ctx.save(sel_tab, "state_machine_parameter_selection.csv", index=False)
    # comparators on the same walk-forward engine
    ctrl = adaptive.daily_walk(env, adaptive.target_path(env, "FIXED", dates, every=5, x=0.0), dates, 0.0)
    unh = ctrl["pnl_spot"] + ctrl["carry"]
    comp = {"CONTROL": ctrl}
    for x in (0.25, 0.5, 0.75, 1.0):
        comp[f"FUT_{int(x * 100)}"] = adaptive.daily_walk(env, adaptive.target_path(env, "FIXED", dates, every=5, x=x), dates, 0.0)
    pre = int(R.index.searchsorted(oos))
    fixed_h = adaptive._lp_target(env, pre, ctx.cfg["smart_portfolio"]["selection"]["static"]["min_upside"], 0.45, adaptive.CASH_ONLY)
    comp["STATIC_SMART"] = adaptive.daily_walk(env, adaptive.target_path(env, "FIXED_SMART", dates, every=5, fixed_h=fixed_h), dates, 0.0)
    v2 = ctx.interim / "adaptive_daily_ADAPTIVE_SMART.csv"
    if v2.exists():
        comp["V2_ADAPTIVE"] = pd.read_csv(v2, index_col=0, parse_dates=True).reindex(dates)
    comp["REGIME_SWITCHING_SMART"] = sel_series
    # EX-POST intensity-matched fixed hedge (diagnostic only: uses the full-period mean of the policy's own notionals)
    mh = sel_series[[f"h_{n}" for n in names]].mean().values
    comp["FIXED_INTENSITY_MATCHED_EXPOST"] = adaptive.daily_walk(
        env, adaptive.target_path(env, "FIXED_SMART", dates, every=5, fixed_h=mh), dates, 0.0)
    ctx.log["v3_sm_mean_notionals"] = {n: float(v) for n, v in zip(names, mh)}
    for (cf, hd), d in runs.items():
        comp[f"SM_confirm{cf}_hold{hd}"] = d
    rows = []
    for name, d in comp.items():
        d = d.dropna(subset=["pnl_total"])
        if not name.startswith("SM_confirm"):
            ctx.save(d, f"v3_wf_daily_{name}.csv", "interim")
        m = summary(d["pnl_total"], base_rub=ctx.E, bench_pnl_rub=unh)
        hcols = [c for c in d.columns if c.startswith("h_")]
        m.update({"strategy": name, "evaluation": "WALK_FORWARD_DAILY_CONSTANT_EXPOSURE", "HE_VAR": he_var(d["pnl_total"].values, unh.reindex(d.index).values),
                  "turnover_notional_multiple": float(d["turnover"].sum()), "total_costs_rub": float(d["costs"].sum()),
                  "n_portfolio_switches": int((d["sm_state"] != d["sm_state"].shift()).sum() - 1) if "sm_state" in d else np.nan,
                  "n_trades": int((d["turnover"] > 0).sum()),
                  **{f"mean_{c}": float(d[c].mean()) for c in hcols}})
        rows.append(m)
    wf = pd.DataFrame(rows)
    # ---- stacked historical episodes (6m / 12m, monthly starts) through the dynamic engine, causal library at start
    sl = dyn_sleeves(specs)
    ep_rows = []
    eq = _eq_cost(ctx, pl._fut_ctx(ctx)[0])
    cfg_by_q = sel_tab.set_index("quarter_start")["selected"]
    for hz_name, hz in (("6m", 126), ("12m", 252)):
        Pe, starts, pos = _episodes(R, cols, dates, hz)
        n_ep = len(starts)
        allc = hc["portfolio"].values
        cpos = hc.index.get_indexer(R.index[pos])
        st_codes = np.stack([allc[p:p + hz + 1] for p in cpos])
        mmf = float(R["LQDT"].loc[oos:t].mean())
        def lib_at(d):
            k = [x for x in sorted(lib_by_date) if x <= d]
            return lib_by_date[k[-1]] if k else np.zeros((len(rr.PORTFOLIO_STATES), len(names)))
        pols = {}
        for x in (0.0, 0.25, 0.5, 0.75, 1.0):
            h = np.zeros((1, len(names)))
            h[0, names.index("DIRECT")] = -x * float(env.fs_ratio.loc[:t].dropna().iloc[-1])
            pols["CONTROL" if x == 0 else f"FUT_{int(x * 100)}"] = ("static", dynamic.Policy("f", h, np.array([1.0])))
        pols["STATIC_SMART"] = ("static", dynamic.Policy("s", fixed_h[None, :], np.array([1.0])))
        res = {k: [] for k in list(pols) + ["REGIME_SWITCHING_SMART"]}
        for i in range(n_ep):
            Pi = {c: Pe[c][i:i + 1] for c in cols}
            for k, (_, pol) in pols.items():
                res[k].append(dynamic.run(Pi, np.zeros((1, hz + 1), int), pol, sl, E=ctx.E, C=ctx.C, mmf_daily=mmf, equity_cost=eq))
            q = cfg_by_q[cfg_by_q.index <= starts[i]]
            cf, hd = ast.literal_eval(q.iloc[-1]) if len(q) else (smc["confirm_grid"][0], smc["hold_grid"][0])
            pol = dynamic.Policy("sm", lib_at(starts[i]), np.ones(len(rr.PORTFOLIO_STATES)), confirm=cf, hold=hd,
                                 threshold=smc["threshold"], protective_rank=prot, emergency_shock=smc["emergency_shock"])
            res["REGIME_SWITCHING_SMART"].append(dynamic.run(Pi, st_codes[i:i + 1], pol, sl, E=ctx.E, C=ctx.C, mmf_daily=mmf, equity_cost=eq))
        for k, lst in res.items():
            term = np.array([x["_term"][0] for x in lst])
            ep_rows.append({"strategy": k, "horizon": hz_name, "n_episodes": n_ep, "mean_terminal": float(term.mean()),
                            "p05_terminal": float(np.quantile(term, 0.05)), "ES95_terminal": es_hist(term, 0.95),
                            "worst_terminal": float(term.min()), "breach_share": float(np.mean([x["p_sleeve_breach"] for x in lst])),
                            "mean_switches": float(np.mean([x["mean_switches"] for x in lst])),
                            "mean_costs": float(np.mean([x["transaction_costs"] for x in lst])),
                            "mean_turnover": float(np.mean([x["turnover"] for x in lst])),
                            "forced_share_sale_share": float(np.mean([x["p_forced_share_sale"] for x in lst])),
                            "evaluation": "HISTORICAL_EPISODES_DYNAMIC_ENGINE"})
    ep = pd.DataFrame(ep_rows)
    ctx.save(ep, "rebalance_policy_episodes.csv", index=False)
    out = wf.merge(ep.pivot_table(index="strategy", columns="horizon", values=["ES95_terminal", "mean_terminal", "breach_share",
                                                                                "mean_switches", "mean_costs"]).pipe(
        lambda d: d.set_axis([f"ep{b}_{a}" for a, b in d.columns], axis=1)).reset_index(), on="strategy", how="left")
    ctx.save(out, "rebalance_policy_backtest.csv", index=False)
    ctx.save(pd.DataFrame(meta), "state_machine_grid.csv", index=False)
    ctx.log["v3_state_machine"] = {"n_library_estimations": len(est_dates), "selected_last": sel_tab["selected"].iloc[-1],
                                   "state_at_cutoff": rr.PORTFOLIO_STATES[int(codes[-1])]}


# ------------------------------------------------------------------ Phase 6: active SMLT exposure branch
def active_policy(ctx, variant: str, confirm: int, hold: int, n_sleeves: int = 0) -> dynamic.Policy:
    """Share-fraction policy; hedge sleeves (if the engine carries any) stay at zero - no mixing with hedging."""
    a = ctx.cfg["v3"]["active"]
    e = np.array([np.nan, a["variants"][variant], a["rerisk_to"]])       # HOLD, DERISK, RERISK
    return dynamic.Policy(variant, np.zeros((3, n_sleeves)), e, confirm=confirm, hold=hold, step_exposure=True,
                          protective_rank=np.array([0, 1, 0]), emergency_shock=ctx.cfg["v3"]["state_machine"]["emergency_shock"],
                          meta={"variant": variant})


def active_daily_walk(ctx, R: pd.DataFrame, codes: np.ndarray, dates, variant: str, confirm: int, hold: int, eq_cost: float) -> pd.DataFrame:
    """Constant-notional walk-forward of the share fraction e_t (decided at close t, applied to t+1):
       pnl_t = E e_{t-1} r_S,t + (C + E (1 - e_{t-1})) m_t - |de_t| E cost."""
    a = ctx.cfg["v3"]["active"]
    tgt = {0: None, 1: a["variants"][variant], 2: a["rerisk_to"]}
    shock = (R["SMLT"].reindex(dates).abs() >= ctx.cfg["v3"]["state_machine"]["emergency_shock"]).values
    stp = sm_path(codes, confirm, hold, shock, np.array([0, 1, 0]))
    e, since_e, rows = 1.0, 10**6, []
    for i, d in enumerate(dates):
        rS = float(np.nan_to_num(R["SMLT"].get(d, 0.0)))
        m = float(np.nan_to_num(R["LQDT"].get(d, 0.0)))
        pnl = ctx.E * e * rS + (ctx.C + ctx.E * (1 - e)) * m
        since_e += 1
        goal = tgt[int(stp[i])]
        de = 0.0
        if goal is not None and abs(goal - e) > 1e-12 and (since_e >= max(hold, 1) or (i and stp[i] != stp[i - 1])):
            de = max(goal - e, -a["step"]) if goal < e else min(goal - e, a["step"])
            since_e = 0
        cost = abs(de) * ctx.E * eq_cost
        rows.append({"date": d, "pnl_spot": ctx.E * rS, "carry": ctx.C * m, "pnl_total": pnl - cost, "costs": cost,
                     "turnover": abs(de), "share_fraction": e, "active_state": rr.ACTIVE_CODES[int(stp[i])]})
        e = min(1.0, e + de)
    return pd.DataFrame(rows).set_index("date")


def s22_active(ctx) -> None:
    from .v3_stages import _regime_inputs
    a = ctx.cfg["v3"]["active"]
    smc = ctx.cfg["v3"]["state_machine"]
    P, R, X, reg, ps, pst = _regime_inputs(ctx)
    hc = rr.history_codes(reg, ps)
    oos, t = pd.Timestamp(ctx.cfg["walk_forward"]["oos_start"]), pd.Timestamp(ctx.cutoff)
    dates = R["SMLT"].dropna().index
    dates = dates[(dates >= oos) & (dates <= t)]
    codes = hc["active"].reindex(dates).values
    eq = _eq_cost(ctx, pl._fut_ctx(ctx)[0])
    rows, runs_all = [], {}
    for v in a["variants"]:
        runs = {(cf, hd): active_daily_walk(ctx, R, codes, dates, v, cf, hd, eq) for cf in smc["confirm_grid"] for hd in smc["hold_grid"]}
        sel_series, sel_tab = _causal_select(runs, lambda d: es_hist(d["pnl_total"].values, 0.95))
        runs_all[v] = (sel_series, sel_tab)
        # EX-POST exposure-matched static sale (diagnostic): constant share fraction = the policy's mean fraction
        ebar = float(sel_series["share_fraction"].mean())
        st_ = pd.DataFrame(index=sel_series.index)
        st_["pnl_spot"], st_["carry"] = sel_series["pnl_spot"], sel_series["carry"]
        lq = R["LQDT"].reindex(st_.index).fillna(0.0)
        st_["pnl_total"] = ebar * st_["pnl_spot"] + (ctx.C + ctx.E * (1 - ebar)) * lq
        st_.iloc[0, st_.columns.get_loc("pnl_total")] -= (1 - ebar) * ctx.E * eq
        st_["costs"], st_["turnover"], st_["share_fraction"] = 0.0, 0.0, ebar
        runs_ex = {f"STATIC_SELL_MATCHED_EXPOST_{ebar:.2f}": st_}
        for key, d in list(runs.items()) + [("WF_SELECTED", sel_series)] + list(runs_ex.items()):
            unh = d["pnl_spot"] + d["carry"]
            m = summary(d["pnl_total"], base_rub=ctx.E, bench_pnl_rub=unh)
            rows.append({"branch": "ACTIVE_EXPOSURE_MANAGEMENT", "strategy": v, "params": str(key), **m,
                         "HE_VAR": he_var(d["pnl_total"].values, unh.values), "n_exposure_changes": int((d["turnover"] > 0).sum()),
                         "mean_share_fraction": float(d["share_fraction"].mean()), "min_share_fraction": float(d["share_fraction"].min()),
                         "total_costs_rub": float(d["costs"].sum()), "turnover_share_multiple": float(d["turnover"].sum()),
                         "evaluation": "WALK_FORWARD_DAILY_CONSTANT_NOTIONAL"})
        ctx.save(sel_tab, f"active_parameter_selection_{v}.csv", "interim", index=False)
        ctx.save(sel_series, f"v3_wf_daily_{v}.csv", "interim")
    # hedge-overlay comparators on the SAME walk-forward framework (from s21)
    rb = pd.read_csv(ctx.out / "rebalance_policy_backtest.csv")
    for _, r in rb[rb["strategy"].isin(["CONTROL", "FUT_25", "FUT_50", "FUT_75", "FUT_100", "STATIC_SMART", "REGIME_SWITCHING_SMART"])].iterrows():
        rows.append({"branch": "PURE_HEDGE_OVERLAY" if r["strategy"] != "CONTROL" else "CONTROL", "strategy": r["strategy"],
                     "params": "", **{k: r[k] for k in ("ann_vol", "ES95_1d", "ES99_1d", "max_drawdown", "upside_capture",
                                                          "downside_capture", "HE_VAR", "total_costs_rub") if k in r},
                     "evaluation": "WALK_FORWARD_DAILY_CONSTANT_EXPOSURE"})
    # stacked historical episodes through the dynamic engine (fixed initial share count)
    specs = sleeves_mod.build(ctx)
    cols = ["SMLT"] + [s.series for s in specs]
    sl0 = dynamic.Sleeves([], np.zeros(0), np.zeros(0), np.zeros(0), 0)
    mmf = float(R["LQDT"].loc[oos:t].mean())
    ep_rows = []
    for hz_name, hz in (("6m", 126), ("12m", 252)):
        Pe, starts, pos = _episodes(R, cols, dates, hz)
        cpos = hc.index.get_indexer(R.index[pos])
        st = np.stack([hc["active"].values[p:p + hz + 1] for p in cpos])
        for v in a["variants"]:
            sel_tab = runs_all[v][1].set_index("quarter_start")["selected"]
            res = []
            for i in range(len(starts)):
                q = sel_tab[sel_tab.index <= starts[i]]
                cf, hd = ast.literal_eval(q.iloc[-1]) if len(q) else (smc["confirm_grid"][0], smc["hold_grid"][0])
                res.append(dynamic.run({"SMLT": Pe["SMLT"][i:i + 1]}, st[i:i + 1], active_policy(ctx, v, cf, hd), sl0,
                                       E=ctx.E, C=ctx.C, mmf_daily=mmf, equity_cost=eq))
            term = np.array([x["_term"][0] for x in res])
            ep_rows.append({"branch": "ACTIVE_EXPOSURE_MANAGEMENT", "strategy": v, "horizon": hz_name, "n_episodes": len(term),
                            "mean_terminal": float(term.mean()), "ES95_terminal": es_hist(term, 0.95), "worst_terminal": float(term.min()),
                            "mean_exposure_changes": float(np.mean([x["mean_exposure_changes"] for x in res])),
                            "mean_final_share_fraction": float(np.mean([x["mean_final_share_fraction"] for x in res])),
                            "mean_costs": float(np.mean([x["transaction_costs"] for x in res]))})
    ep = pd.DataFrame(ep_rows)
    hep = pd.read_csv(ctx.out / "rebalance_policy_episodes.csv")
    hep = hep[hep["strategy"].isin(["CONTROL", "FUT_50", "FUT_75", "STATIC_SMART", "REGIME_SWITCHING_SMART"])].assign(branch="PURE_HEDGE_OVERLAY")
    ctx.save(pd.concat([pd.DataFrame(rows), ep.assign(evaluation="HISTORICAL_EPISODES_DYNAMIC_ENGINE"),
                        hep], ignore_index=True), "active_smlt_backtest.csv", index=False)
    cur = rr.ACTIVE_CODES[int(codes[-1])]
    pol_rows = []
    for v, frac in a["variants"].items():
        sel = runs_all[v][1]["selected"].iloc[-1]
        pol_rows.append({"variant": v, "derisk_share_fraction": frac, "rerisk_to": a["rerisk_to"], "step": a["step"],
                         "trigger_DERISK": "macro CREDIT_STRESS or COMBINED_STRESS AND SMLT price state DOWNTREND or HIGH_VOL_SHOCK",
                         "trigger_RERISK": "macro NORMAL AND SMLT price state UPTREND (buy back previously sold shares only)",
                         "trigger_HOLD": "otherwise", "never_above_initial_position": True,
                         "walkforward_selected_confirm_hold_at_cutoff": sel, "active_state_at_cutoff": cur,
                         "share_fraction_path_at_cutoff": float(runs_all[v][0]["share_fraction"].iloc[-1])})
    ctx.save(pd.DataFrame(pol_rows), "active_smlt_policy.csv", index=False)
    ctx.log["v3_active_state_at_cutoff"] = cur
