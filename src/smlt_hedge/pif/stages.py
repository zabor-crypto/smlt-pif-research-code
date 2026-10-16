"""V4 PIF stages.

s30 setup        initial fund balance sheet per capital case, asset universe (funded vs derivative; BTC economic proxy vs
                 verified MOEX derivative vs instructor-authorised experimental spot)
s31 static       frontiers on TRAIN (per universe / mandate / SMLT cap), evaluation on VALIDATION, pre-declared
                 selection -> STATIC_PIF, PIF_WITH_BTC_*, PIF_WIDE_*, SELL_AND_REALLOCATE, CASH_MATCHED; capital grid
s32 regime       fund-level state target portfolios; causal walk-forward (semi-annual re-estimation, data < date);
                 state-machine parameters chosen walk-forward; rebalance log
s33 forward      joint forward scenario paths for every asset (+ money-market path) and results
s34 btc          BTC test under the PIF objective (paired TEST, by state, premium break-even SENSITIVITY)
s35 final        TEST (20 seeds x 4 models, 6m / 12m) for every comparator; validation stability; waterfall; risk
s36 report       figures and PIF_*.md
"""
from __future__ import annotations

import ast

import numpy as np
import pandas as pd

from ..portfolio import regime_risk as rr
from ..v3_policy import frozen_thresholds, path_codes, prefix_frame, protective_rank
from . import core, fund
from . import optimize as opt
from . import universe as uni

HZ = {"6m": 126, "12m": 252}


def _E(ctx) -> float:
    return ctx.E


def _cap_initial(E: float, C: float) -> float:
    return E / (E + C)


def mandate_caps(ctx, E, C) -> list[tuple[str, float]]:
    m = ctx.cfg["v4"]["mandates"]
    return [("PIF_CONSERVATIVE", _cap_initial(E, C))] + [(f"PIF_WIDE_{c:.2f}", float(c)) for c in m["PIF_WIDE"]["smlt_cap_grid"]]


# ================================================================== s30
def s30_setup(ctx) -> None:
    v4 = ctx.cfg["v4"]
    P = ctx.read("prices.csv", index_col=0, parse_dates=True)
    S0 = float(P["SMLT"].loc[ctx.cutoff])
    rows = []
    for C in v4["capital_grid"]:
        nav = ctx.E + C
        rows.append({"capital_case_C": C, "SMLT_shares": int(round(ctx.E / S0)), "SMLT_price_close": S0, "SMLT_value": ctx.E,
                     "liquidity_sleeve": C, "NAV0": nav, "w_SMLT": ctx.E / nav, "w_liquidity": C / nav, "derivatives_notional": 0.0,
                     "initial_margin": 0.0, "gross_exposure_nav": ctx.E / nav, "net_market_exposure_nav": ctx.E / nav,
                     "main_case": C == v4["main_capital"]})
    ctx.save(pd.DataFrame(rows), "pif_initial_state.csv", index=False)
    assets = uni.build(ctx, "WITH_BTC_MOEX_DERIVATIVE", 2 * ctx.E, 1.0) + [a for a in uni.build(ctx, "WITH_BTC_EXPERIMENTAL_SPOT", 2 * ctx.E, 1.0)
                                                                          if a.name == "BTC_SPOT_EXP"]
    bv = pd.read_csv(ctx.out / "btc_verdict.csv", index_col=0)["value"] if (ctx.out / "btc_verdict.csv").exists() else None
    ar = []
    for a in assets:
        cls = {"SMLT": "EXECUTABLE (anchor funded asset)", "BTC_SPOT_EXP": "INSTRUCTOR_AUTHORIZED_EXPERIMENTAL",
               "BTC_FUT": f"{bv['executability_moex'] if bv is not None else 'n/a'} (verified MOEX derivative); V3 hedge verdict "
                          f"{bv['classification'] if bv is not None else 'n/a'}"}.get(a.name, "EXECUTABLE (V2/V3 gates)")
        ar.append({"asset": a.name, "kind": a.kind, "return_series": a.series, "classification": cls, "execution": a.execution,
                   "nav_weight_lo": a.lo, "nav_weight_hi": a.hi if a.name != "SMLT" else "mandate cap",
                   "cost_one_way": a.cost_one_way, "initial_margin_rate": a.mr, "rolls_per_year": a.rolls_per_year,
                   "unit_value_rub": a.contract_notional,
                   "history": {"BTC_SPOT_EXP": "PROXY_HISTORY: Binance BTCUSDT (16:00 UTC) x CNY/RUB (FX proxy)",
                               "BTC_FUT": "economic returns: BTC proxy history; execution terms: MOEX BTC futures (PIT)",
                               "GOLD": "GLDRUB_TOM spot returns (execution GLDRUBF)", "CNY": "CNYRUB_TOM spot returns (execution CNYRUBF)"}
                   .get(a.name, "MOEX ISS history")})
    ar.append({"asset": "LIQUIDITY (LQDT-type money market)", "kind": "FUNDED (residual)", "return_series": "LQDT / trailing yield",
               "classification": "EXECUTABLE", "execution": "MOEX TQBR fund units", "nav_weight_lo": 0.0, "nav_weight_hi": 1.0,
               "cost_one_way": 0.0, "initial_margin_rate": 0.0, "history": "MOEX ISS"})
    ctx.save(pd.DataFrame(ar), "pif_asset_universe.csv", index=False)


# ================================================================== s31
def _candidates_for(ctx, universe, E, C, cap, hz, fac=None, fixed=None, label=""):
    assets = uni.build(ctx, universe, E + C, cap)
    fac = fac or core.factory(ctx, assets)
    Ms = core.lp_models(fac, assets, hz, ctx.cfg["v4"]["n_lp"], "TRAIN")
    lp = core.make_lp(ctx, assets, E, C, hz, fixed=fixed)
    rows = core.frontier(ctx, lp, Ms, {"universe": universe, "mandate": label, "smlt_cap_nav": cap, "capital_C": C})
    return assets, fac, rows


def select_static(ctx, universe, E, C, cap, label, hz=126, fixed=None, fac=None) -> dict:
    v4 = ctx.cfg["v4"]
    assets, fac, rows = _candidates_for(ctx, universe, E, C, cap, hz, fac=fac, fixed=fixed, label=label)
    cands = [(f"{label}_{universe}_{int(C)}_{i:02d}", core.static_policy(core.xvec(r, assets)), None) for i, r in enumerate(rows)]
    ev = core.evaluate(ctx, fac, assets, cands, hz, E, C, "VAL", v4["n_val"])
    summ = core.summarise(ev)
    fr = pd.DataFrame(rows)
    fr["cand_id"] = [c[0] for c in cands]
    fr = fr.merge(summ, on="cand_id", how="left")
    sel = core.select(summ, E + C, v4["selection"]["es_budget"], v4["selection"]["max_worst_breach"])
    x = core.xvec(fr.set_index("cand_id").loc[sel["cand_id"]], assets) if sel is not None else uni.initial_weights(assets, E, C) * 0
    if sel is not None:
        x, sel = _prune_and_reevaluate(ctx, fac, assets, x, sel, E, C, hz, fixed, core.MODELS)
    return {"assets": assets, "fac": fac, "frontier": fr, "selected": sel, "x": x}


def _prune_and_reevaluate(ctx, fac, assets, x, sel, E, C, hz, fixed, models, budget=None):
    """Parsimony pass on VALIDATION horizon scenarios, then the pruned portfolio is re-evaluated with the fund engine on
    VALIDATION so that every reported metric belongs to the portfolio actually held."""
    v4 = ctx.cfg["v4"]
    lp = core.make_lp(ctx, assets, E, C, hz, fixed=fixed)
    vl = core.lp_models(fac, assets, hz, v4["n_val"], "VAL", models=models)
    xp = prune_horizon(ctx, lp, vl, x, budget)
    if np.allclose(xp, x):
        return x, sel.copy()
    ev = core.evaluate(ctx, fac, assets, [(str(sel["cand_id"]) + "_PRUNED", core.static_policy(xp), None)], hz, E, C, "VAL",
                       v4["n_val"] if models == core.MODELS else v4["n_val"] // 2, models=models)
    s2 = core.summarise(ev).iloc[0].copy()
    s2["pruned_from"] = sel["cand_id"]
    return xp, s2


def s31_static(ctx) -> None:
    v4 = ctx.cfg["v4"]
    E, C = ctx.E, float(v4["main_capital"])
    fronts, sels, ints = [], [], []
    keep = {}
    for universe in v4["universes"]:
        caps = mandate_caps(ctx, E, C) if universe == "WITHOUT_BTC" else [("PIF_CONSERVATIVE", _cap_initial(E, C))]
        for label, cap in caps:
            r = select_static(ctx, universe, E, C, cap, label)
            fronts.append(r["frontier"])
            name = {"WITHOUT_BTC": "STATIC_PIF" if label == "PIF_CONSERVATIVE" else f"STATIC_{label}",
                    "WITH_BTC_MOEX_DERIVATIVE": "PIF_WITH_BTC_MOEX", "WITH_BTC_EXPERIMENTAL_SPOT": "PIF_WITH_BTC_SPOT"}[universe]
            keep[name] = r
            xi, rep = core.integer_portfolio(ctx, r["assets"], r["x"], E + C)
            keep[name]["x_int"] = xi
            ints += [{"portfolio": name, **x} for x in rep]
            sels.append({"portfolio": name, "universe": universe, "mandate": label, "smlt_cap_nav": cap, "capital_C": C,
                         **({} if r["selected"] is None else r["selected"].to_dict()),
                         **{f"x_{a.name}": float(v) for a, v in zip(r["assets"], xi)}})
    # SELL_AND_REALLOCATE: SMLT fixed at 50% of the initial position, the rest optimised with the same rule
    base = keep["STATIC_PIF"]
    w_half = 0.5 * E / (E + C)
    r = select_static(ctx, "WITHOUT_BTC", E, C, _cap_initial(E, C), "SELL_AND_REALLOCATE", fixed={"SMLT": w_half}, fac=base["fac"])
    xi, rep = core.integer_portfolio(ctx, r["assets"], r["x"], E + C)
    r["x_int"] = xi
    keep["SELL_AND_REALLOCATE"] = r
    fronts.append(r["frontier"])
    ints += [{"portfolio": "SELL_AND_REALLOCATE", **x} for x in rep]
    sels.append({"portfolio": "SELL_AND_REALLOCATE", "universe": "WITHOUT_BTC", "mandate": "SMLT fixed at 50% of initial",
                 "capital_C": C, **({} if r["selected"] is None else r["selected"].to_dict()),
                 **{f"x_{a.name}": float(v) for a, v in zip(r["assets"], xi)}})
    # CASH_MATCHED: sell SMLT to cash until the VALIDATION worst-model ES equals STATIC_PIF's (isolates "more cash")
    assets = base["assets"]
    target_es = float(base["selected"]["worst_ES95"]) if base["selected"] is not None else np.nan
    grid = np.round(np.arange(0.0, 1.0001, 0.02), 2)
    cands = []
    for s in grid:
        x = uni.initial_weights(assets, E, C) * (1 - s)
        cands.append((f"SELL_{int(round(s * 100))}", core.static_policy(x), None))
    ev = core.evaluate(ctx, base["fac"], assets, cands, 126, E, C, "VAL", v4["n_val"])
    sm = core.summarise(ev)
    sm["sold"] = sm["cand_id"].str.replace("SELL_", "").astype(int) / 100
    ok = sm[sm["worst_ES95"] <= target_es].sort_values("sold")
    s_star = float(ok["sold"].iloc[0]) if len(ok) else 1.0
    sels.append({"portfolio": "CASH_MATCHED", "universe": "WITHOUT_BTC", "mandate": f"sell {s_star:.0%} of SMLT to cash (ES-matched to STATIC_PIF on VALIDATION)",
                 "capital_C": C, "cand_id": f"SELL_{int(round(s_star * 100))}", **sm.set_index("cand_id").loc[f"SELL_{int(round(s_star * 100))}"].drop("sold").to_dict(),
                 "x_SMLT": E / (E + C) * (1 - s_star)})
    ctx.save(pd.concat(fronts, ignore_index=True), "pif_efficient_frontier.csv", index=False)
    ctx.save(pd.DataFrame(sels), "pif_selected_portfolios.csv", index=False)
    ctx.save(pd.DataFrame(ints), "pif_integer_positions.csv", index=False)
    # capital grid (PIF_CONSERVATIVE and the middle PIF_WIDE cap), same rule, per E and per NAV
    cg = []
    for Cg in v4["capital_grid"]:
        for label, cap in [("PIF_CONSERVATIVE", _cap_initial(E, Cg)), (f"PIF_WIDE_{v4['mandates']['PIF_WIDE']['smlt_cap_grid'][1]:.2f}",
                                                                      v4["mandates"]["PIF_WIDE"]["smlt_cap_grid"][1])]:
            r = select_static(ctx, "WITHOUT_BTC", E, float(Cg), cap, label)
            s = r["selected"]
            nav = E + Cg
            row = {"capital_C": Cg, "NAV0": nav, "mandate": label, "smlt_cap_nav": cap}
            if s is not None:
                row.update({"worst_ES95_rub": s["worst_ES95"], "worst_ES95_per_E": s["worst_ES95"] / E, "worst_ES95_per_NAV": s["worst_ES95"] / nav,
                            "mean_pnl_rub": s["mean_pnl_avg"], "mean_pnl_per_E": s["mean_pnl_avg"] / E, "mean_pnl_per_NAV": s["mean_pnl_avg"] / nav,
                            "worst_participation": s["worst_participation"], "worst_breach": s["worst_breach"],
                            **{f"x_{a.name}": float(v) for a, v in zip(r["assets"], r["x"])},
                            "SMLT_value_rub": float(r["x"][0] * nav), "SMLT_share_of_initial": float(r["x"][0] * nav / E)})
            cg.append(row)
    ctx.save(pd.DataFrame(cg), "pif_capital_grid.csv", index=False)
    ctx._pif_static = keep


def static_book(ctx) -> dict:
    """Selected static portfolios (integer) as {name: (assets, x)} reconstructed from outputs (stage-independent)."""
    sel = pd.read_csv(ctx.out / "pif_selected_portfolios.csv")
    E, C = ctx.E, float(ctx.cfg["v4"]["main_capital"])
    out = {}
    for _, r in sel.iterrows():
        universe = r["universe"] if isinstance(r["universe"], str) else "WITHOUT_BTC"
        cap = float(r["smlt_cap_nav"]) if pd.notna(r.get("smlt_cap_nav")) else _cap_initial(E, C)
        assets = uni.build(ctx, universe, E + C, cap)
        out[r["portfolio"]] = (assets, core.xvec(r, assets))
    return out


# ================================================================== helpers: waterfall orders, causal targets
def orders_for(ctx, assets, targets: np.ndarray, R: pd.DataFrame, until=None) -> np.ndarray:
    """Waterfall order per state from the covariance of daily returns (data <= `until`)."""
    X = R[[a.series for a in assets]].copy()
    if until is not None:
        X = X.loc[:until]
    X = X.loc[pd.Timestamp(ctx.cfg["simulation"]["bootstrap_start"]):].fillna(0.0)
    cov = np.cov(X.values, rowvar=False)
    kinds = np.array([a.kind for a in assets])
    mr = np.array([a.mr for a in assets])
    cost = np.array([a.cost_one_way for a in assets])
    return np.stack([fund.waterfall_order(t, cov, kinds, mr, cost, 1.0) for t in targets])


def past_models(ctx, assets, R: pd.DataFrame, codes: pd.Series | None, state: int | None, date, hz: int, n: int, seed: int,
                recent: int | None = None):
    """Horizon-return scenarios from data STRICTLY BEFORE `date` (causal walk-forward): zero-drift joint block
    bootstrap (funded series drift at the trailing money-market rate), pooled toward the full past sample for a state."""
    from ..scenarios.bootstrap import block_bootstrap_indices
    cols = list(dict.fromkeys(["SMLT"] + [a.series for a in assets]))
    start = pd.Timestamp(ctx.cfg["simulation"]["bootstrap_start"])
    X = R.loc[(R.index >= start) & (R.index < date), cols].dropna()
    if recent:
        X = X.tail(recent)
    m_d = float(R["LQDT"].loc[:date].iloc[-61:-1].mean())
    fund_cols = [c for c in cols if c in ("SMLT",) or c in [a.series for a in assets if a.kind == "FUNDED"]]
    Y = X - X.mean()
    Y[fund_cols] = Y[fund_cols] + m_d
    if state is not None and codes is not None:
        cd = codes.reindex(Y.index)
        sub = Y[(cd == state).values]
        if len(sub) >= 20:
            lam = ctx.cfg["v3"]["shrink_n0"] / (ctx.cfg["v3"]["shrink_n0"] + len(sub))
            reps = max(1, int(round((1 - lam) / lam * len(Y) / len(sub))))
            Y = pd.concat([sub] * reps + [Y])
    idx = block_bootstrap_indices(len(Y), hz, n, ctx.cfg["simulation"]["block_length"], seed)
    Z = Y.values[idx]
    HR = pd.DataFrame(np.prod(1 + Z, axis=1) - 1, columns=cols)
    m = (1 + m_d) ** hz - 1
    return opt.Model(R=HR[[a.series for a in assets]].values, rs=HR["SMLT"].values, m=m)


def causal_select(ctx, assets, E, C, date, R, codes, state, seed, hz=126) -> np.ndarray:
    """Same design as the decision-date selection, on data strictly before `date`: TRAIN = two past-data models
    (state-pooled full past sample and the most recent 250 days), frontier LPs; VALIDATION = independent resamples of
    both; rule = max worst-model participation s.t. worst-model ES95 <= budget; then the parsimony pass."""
    v4 = ctx.cfg["v4"]
    mk = lambda n, sd: [past_models(ctx, assets, R, codes, state, date, hz, n, sd),  # noqa: E731
                        past_models(ctx, assets, R, None, None, date, hz, n, sd + 1, recent=250)]
    tr, vl = mk(1200, seed), mk(2000, seed + 7777)
    lp = core.make_lp(ctx, assets, E, C, hz)
    best, best_p = None, -np.inf
    for b in v4["es_budget_grid"]:
        r = opt.solve(lp, tr, "max_participation", budget=b)
        if r["status"] != "optimal":
            continue
        d = opt.diagnostics(lp, vl, r["x"])
        if d["worst_es"] <= v4["selection"]["es_budget"] and d["min_participation"] > best_p + 1e-12:
            best, best_p = r["x"], d["min_participation"]
    if best is None:
        return np.zeros(len(assets))
    return prune_horizon(ctx, lp, vl, best)


def prune_horizon(ctx, lp, models, x, budget=None) -> np.ndarray:
    """Parsimony pass (pre-declared): drop non-SMLT positions below `min_position` of NAV one at a time (smallest first)
    if the pruned portfolio still meets the ES budget and loses <= `max_participation_loss` of participation."""
    v4 = ctx.cfg["v4"]
    rule = v4["parsimony"]
    base = opt.diagnostics(lp, models, x)
    x = x.copy()
    for j in np.argsort(np.abs(x)):
        if lp.assets[j].name == "SMLT" or abs(x[j]) < 1e-12 or abs(x[j]) >= rule["min_position"]:
            continue
        y = x.copy()
        y[j] = 0.0
        d = opt.diagnostics(lp, models, y)
        if d["worst_es"] <= max(v4["selection"]["es_budget"] if budget is None else budget, base["worst_es"]) + 1e-12 and \
                d["min_participation"] >= base["min_participation"] - rule["max_participation_loss"]:
            x = y
    return x


# ================================================================== s32
def state_targets(ctx, universe, E, C, cap, hz=126) -> tuple[list, np.ndarray, pd.DataFrame]:
    """Decision-date fund target per portfolio state (state-conditional models only, V3 lesson)."""
    from ..v3_stages import _regime_inputs
    v4 = ctx.cfg["v4"]
    P, R, X, reg, ps, pst = _regime_inputs(ctx)
    masks = rr.state_masks(reg, ps)
    assets = uni.build(ctx, universe, E + C, cap)
    lp = core.make_lp(ctx, assets, E, C, hz)
    T = np.zeros((len(rr.PORTFOLIO_STATES), len(assets)))
    rows = []
    for i, st in enumerate(rr.PORTFOLIO_STATES):
        fac = core.factory(ctx, assets, regime_mask=masks[st], label=st)
        Ms = core.lp_models(fac, assets, hz, v4["n_lp"], "TRAIN", models=["REGIME_COND", "REGIME_PURE"])
        fr = core.frontier(ctx, lp, Ms, {"state": st})
        cands = [(f"{st}_{j:02d}", core.static_policy(core.xvec(r, assets)), None) for j, r in enumerate(fr)]
        ev = core.evaluate(ctx, fac, assets, cands, hz, E, C, "VAL", v4["n_val"] // 2, models=["REGIME_COND", "REGIME_PURE"])
        summ = core.summarise(ev)
        sel = core.select(summ, E + C, v4["selection"]["es_budget"], v4["selection"]["max_worst_breach"])
        if sel is not None:
            T[i] = core.xvec(pd.DataFrame(fr).assign(cand_id=[c[0] for c in cands]).set_index("cand_id").loc[sel["cand_id"]], assets)
            T[i], sel = _prune_and_reevaluate(ctx, fac, assets, T[i], sel, E, C, hz, None, ["REGIME_COND", "REGIME_PURE"])
        T[i], _ = core.integer_portfolio(ctx, assets, T[i], E + C)
        rows.append({"state": st, "n_state_days": fac.regime_n, "pool_lambda_full_sample": fac.pool_lambda,
                     **({} if sel is None else {k: sel[k] for k in ("worst_ES95", "worst_participation", "worst_breach", "mean_pnl_avg")}),
                     **{f"x_{a.name}": float(v) for a, v in zip(assets, T[i])}})
    df = pd.DataFrame(rows)
    groups = []
    for i, st in enumerate(rr.PORTFOLIO_STATES):
        for g in groups:
            if np.abs(T[i] - T[g[0]]).max() <= 0.05:
                g.append(i)
                break
        else:
            groups.append([i])
    rep = {i: g[0] for g in groups for i in g}
    df["merged_with"] = [rr.PORTFOLIO_STATES[rep[i]] for i in range(len(df))]
    for i in range(len(T)):
        T[i] = T[rep[i]]
    fund_mask = np.array([a.kind == "FUNDED" for a in assets])
    mr = np.array([a.mr for a in assets])
    df["liquidity_weight"] = 1 - (T * fund_mask).sum(axis=1)
    df["margin_nav"] = (np.abs(T) * (~fund_mask) * mr).sum(axis=1)
    df["gross_nav"] = np.abs(T).sum(axis=1)
    df["net_market_exposure_nav"] = T.sum(axis=1)
    return assets, T, df


def _hist_inputs(ctx):
    from ..v3_stages import _regime_inputs
    P, R, X, reg, ps, pst = _regime_inputs(ctx)
    hc = rr.history_codes(reg, ps)
    oos, t = pd.Timestamp(ctx.cfg["walk_forward"]["oos_start"]), pd.Timestamp(ctx.cutoff)
    return P, R, hc, oos, t


def s32_regime(ctx) -> None:
    v4 = ctx.cfg["v4"]
    E, C = ctx.E, float(v4["main_capital"])
    assets, T, df = state_targets(ctx, "WITHOUT_BTC", E, C, _cap_initial(E, C))
    ctx.save(df, "pif_regime_target_portfolios.csv", index=False)
    # ---------------- causal walk-forward (historical)
    P, R, hc, oos, t = _hist_inputs(ctx)
    smc = v4["state_machine"]
    cols = list(dict.fromkeys(["SMLT"] + [a.series for a in assets]))
    dates = R[cols].dropna().index
    dates = dates[(dates >= oos) & (dates <= t)]
    est = [d for d in pd.date_range(oos, t, freq="MS") if d.month in smc["reestimate_months"]]
    est = [dates[dates >= d][0] for d in est if (dates >= d).any()]
    if dates[0] not in est:
        est = [dates[0]] + est
    codes_s = hc["portfolio"]
    tables, static_tab, lrows = {}, {}, []
    for j, d in enumerate(est):
        seed = ctx.cfg["simulation"]["seed"] + 900 + 31 * j
        static_tab[d] = causal_select(ctx, assets, E, C, d, R, None, None, seed)
        Tj = np.zeros((len(rr.PORTFOLIO_STATES), len(assets)))
        for i in range(len(rr.PORTFOLIO_STATES)):
            Tj[i] = causal_select(ctx, assets, E, C, d, R, codes_s, i, seed + 101 * (i + 1))
        tables[d] = Tj
        lrows.append({"estimation_date": d, "portfolio": "STATIC_PIF_CAUSAL", "state": "ALL",
                      **{f"x_{a.name}": v for a, v in zip(assets, static_tab[d])}})
        for i, st in enumerate(rr.PORTFOLIO_STATES):
            lrows.append({"estimation_date": d, "portfolio": "REGIME_PIF_CAUSAL", "state": st,
                          **{f"x_{a.name}": v for a, v in zip(assets, Tj[i])}})
    ctx.save(pd.DataFrame(lrows), "pif_causal_targets_walkforward.csv", index=False)
    ctx._pif_causal = (assets, est, tables, static_tab)
    wf, log = walkforward(ctx, assets, R, hc, dates, est, tables, static_tab)
    ctx.save(wf, "pif_walkforward.csv", index=False)
    ctx.save(log, "pif_rebalance_log.csv", index=False)


def _tables_daily(dates, est, tables_or_static, n_states) -> tuple[np.ndarray, np.ndarray]:
    H = len(dates)
    A = next(iter(tables_or_static.values())).shape[-1]
    Tt = np.zeros((H, n_states, A))
    force = np.zeros(H, bool)
    for k, d in enumerate(dates):
        e = [x for x in est if x <= d][-1]
        v = tables_or_static[e]
        Tt[k] = v if v.ndim == 2 else np.tile(v, (n_states, 1))
        force[k] = d in est
    return Tt, force


def walkforward(ctx, assets, R, hc, dates, est, tables, static_tab):
    """Daily NAV path 2023-01..cutoff for every strategy (fund engine, NAV compounding, one balance sheet)."""
    from ..risk.metrics import es_hist
    v4 = ctx.cfg["v4"]
    smc = v4["state_machine"]
    E, C = ctx.E, float(v4["main_capital"])
    NAV0 = E + C
    Pd = {a.series: R[a.series].reindex(dates).values[None, 1:] for a in assets}
    Pd["SMLT"] = R["SMLT"].reindex(dates).values[None, 1:]
    md = R["LQDT"].reindex(dates).fillna(0.0).values[1:]
    codes = hc["portfolio"].reindex(dates).values[None, :]
    shock = (np.abs(R["SMLT"].reindex(dates).values[1:]) >= smc["emergency_shock"])[None, :]
    x0 = uni.initial_weights(assets, E, C)
    names = [a.name for a in assets]
    S = len(rr.PORTFOLIO_STATES)
    def fixed(x, name):
        return fund.FundPolicy(name, np.tile(x, (S, 1)), rebalance=False)
    pols = {"CONTROL_INITIAL_PORTFOLIO": (fixed(x0, "c"), None, None)}
    for s_ in (0.5, 1.0):
        pols[f"SELL_{int(s_ * 100)}_TO_CASH"] = (fixed(x0 * (1 - s_), "s"), None, None)
    for h in (0.25, 0.5):
        pols[f"FUT_{int(h * 100)}"] = (fixed(fut_weights(ctx, assets, h, E, C), "f"), None, None)
    Ts, fs_ = _tables_daily(dates, est, static_tab, S)
    pols["STATIC_PIF"] = (fund.FundPolicy("static_causal", Ts[0], rebalance=False), Ts, fs_)
    pols["STATIC_PIF_BUYHOLD"] = (fund.FundPolicy("static_bh", Ts[0], rebalance=False), None, None)
    Tr, fr_ = _tables_daily(dates, est, tables, S)
    Trt = np.stack([trend_capped(ctx, assets, Tr[k], E, C) for k in range(len(Tr))])
    rows, series = [], {}
    prot = protective_rank(ctx)
    runs = {}
    for cf in smc["confirm_grid"]:
        for hd in smc["hold_grid"]:
            pol = fund.FundPolicy("regime", Tr[0], confirm=cf, hold=hd, threshold=smc["threshold"], protective_rank=prot,
                                  emergency_shock=smc["emergency_shock"])
            pol.orders = orders_for(ctx, assets, Tr[0], R, until=dates[0])
            out = fund.run(Pd, md, assets, pol, codes, E=E, C=C, x0_initial=x0, shock=shock, targets_t=Tr, force=fr_, log=True)
            runs[(cf, hd)] = out
    cf_t, hd_t = smc["confirm_grid"][-1], smc["hold_grid"][-1]      # trend variant: most conservative pre-declared pair
    pols["REGIME_PIF_TREND"] = (fund.FundPolicy("regime_trend", Trt[0], confirm=cf_t, hold=hd_t, threshold=smc["threshold"],
                                                protective_rank=prot, emergency_shock=smc["emergency_shock"]), Trt, fr_)
    for name, (pol, Tt, ff) in pols.items():
        pol.orders = orders_for(ctx, assets, pol.targets, R, until=dates[0])
        out = fund.run(Pd, md, assets, pol, codes, E=E, C=C, x0_initial=x0, shock=shock, targets_t=Tt, force=ff, log=True)
        series[name] = out
    # walk-forward choice of (confirm, hold): quarterly, lowest realised daily ES95 of NAV returns over PRIOR days
    navs = {k: pd.Series([r["nav"] for r in v["_log"]], index=dates[1:]) for k, v in runs.items()}
    rets = {k: s.pct_change().fillna(s.iloc[0] / NAV0 - 1) for k, s in navs.items()}
    qs = list(pd.date_range(dates[1], dates[-1], freq="QS"))
    if not qs or qs[0] > dates[1]:
        qs = [dates[1]] + qs
    keys = sorted(runs)
    pieces, sel_rows, best = [], [], keys[0]
    for i, q in enumerate(qs):
        qe = qs[i + 1] if i + 1 < len(qs) else dates[-1] + pd.Timedelta(days=1)
        prior = rets[keys[0]].index[rets[keys[0]].index < q]
        if len(prior) >= 60:
            best = min(keys, key=lambda k: (es_hist(rets[k].loc[prior].values, 0.95), k))
        sel_rows.append({"quarter_start": q, "selected_confirm_hold": str(best)})
        pieces.append(rets[best].loc[(rets[best].index >= q) & (rets[best].index < qe)])
    reg_ret = pd.concat(pieces)
    ctx.save(pd.DataFrame(sel_rows), "pif_state_machine_selection.csv", index=False)
    ctx.log["pif_sm_selected_last"] = sel_rows[-1]["selected_confirm_hold"]
    sm_last = runs[best]
    daily = {name: pd.Series([r["nav"] for r in out["_log"]], index=dates[1:]).pct_change().fillna(out["_log"][0]["nav"] / NAV0 - 1)
             for name, out in series.items()}
    daily["REGIME_PIF"] = reg_ret
    ctrl = daily["CONTROL_INITIAL_PORTFOLIO"]
    for name, r in daily.items():
        nav = NAV0 * (1 + r).cumprod()
        dd = (nav / nav.cummax() - 1).min()
        up = ctrl > 0
        years = len(r) / 252
        out = series.get(name)
        rows.append({"strategy": name, "start": str(dates[1].date()), "end": str(dates[-1].date()),
                     "final_nav_rub": float(nav.iloc[-1]), "cagr": float((nav.iloc[-1] / NAV0) ** (1 / years) - 1),
                     "ann_vol": float(r.std() * np.sqrt(252)), "max_drawdown": float(-dd), "ES95_1d_nav": es_hist(r.values, 0.95),
                     "up_day_participation": float(r[up].mean() / ctrl[up].mean()), "down_day_capture": float(r[~up].mean() / ctrl[~up].mean()),
                     "variance_reduction_vs_control": float(1 - r.var() / ctrl.var()),
                     "turnover_nav": out["turnover_nav"] if out else np.nan, "costs_rub": out["transaction_costs"] if out else np.nan,
                     "switches": out["mean_switches"] if out else float(sum(v["mean_switches"] for v in runs.values()) / len(runs)),
                     "liquidity_breach": out["p_liquidity_breach"] if out else np.nan,
                     **{f"avg_w_{a}": (out[f"avg_w_{a}"] if out else sm_last[f"avg_w_{a}"]) for a in names}})
        ctx.save(pd.DataFrame({"nav": nav, "ret": r}), f"pif_wf_nav_{name}.csv", "interim")
    for (cf, hd), out in runs.items():
        r = rets[(cf, hd)]
        rows.append({"strategy": f"REGIME_PIF_confirm{cf}_hold{hd}", "cagr": float((1 + r).prod() ** (252 / len(r)) - 1),
                     "ann_vol": float(r.std() * np.sqrt(252)), "ES95_1d_nav": es_hist(r.values, 0.95), "switches": out["mean_switches"],
                     "turnover_nav": out["turnover_nav"], "costs_rub": out["transaction_costs"],
                     "max_drawdown": float(-((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min())})
    ab = ctx.out / "active_smlt_backtest.csv"
    if ab.exists():
        a = pd.read_csv(ab)
        for _, r in a[(a["evaluation"] == "WALK_FORWARD_DAILY_CONSTANT_NOTIONAL") & (a["params"] == "WF_SELECTED")].iterrows():
            rows.append({"strategy": f"V3_{r['strategy']}", "ES95_1d_nav": np.nan, "max_drawdown": r["max_drawdown"],
                         "ann_vol": r["ann_vol"], "up_day_participation": r["upside_capture"], "down_day_capture": r["downside_capture"],
                         "variance_reduction_vs_control": r["HE_VAR"], "switches": r["n_exposure_changes"],
                         "note": "HISTORICAL BENCHMARK ONLY: V3 constant-notional framework (per RUB 10m SMLT), not NAV-comparable"})
    log = pd.DataFrame(sm_last["_log"])
    log["date"] = dates[1:][log["t"] - 1]
    log["state_name"] = [rr.PORTFOLIO_STATES[s] for s in log["state"]]
    log = log[log["switched"] | log["date"].isin(est)]
    wpd = pd.DataFrame([{"date": dates[1:][r_["t"] - 1], **{k: v for k, v in r_.items() if k.startswith("x_")}} for r_ in sm_last["_log"]])
    ctx.save(wpd, "pif_wf_weights_REGIME_PIF.csv", "interim", index=False)
    return pd.DataFrame(rows), log


def fut_weights(ctx, assets, h: float, E: float, C: float) -> np.ndarray:
    """V3 fixed hedge re-expressed at fund level: initial SMLT kept, round(h x shares) SMLT futures contracts."""
    names = [a.name for a in assets]
    x = uni.initial_weights(assets, E, C)
    S0 = float(ctx.read("prices.csv", index_col=0, parse_dates=True)["SMLT"].loc[ctx.cutoff])
    j = names.index("SMLT_FUT")
    x[j] = -round(h * E / S0) * assets[j].contract_notional / (E + C)
    return x


# ================================================================== finalists on one master universe (paired paths)
def master_assets(ctx, E, C):
    a = uni.build(ctx, "WITH_BTC_EXPERIMENTAL_SPOT", E + C, 1.0)
    b = [x for x in uni.build(ctx, "WITH_BTC_MOEX_DERIVATIVE", E + C, 1.0) if x.name == "BTC_FUT"]
    return a + b


def _embed(assets_src, x_src, master) -> np.ndarray:
    d = {a.name: v for a, v in zip(assets_src, x_src)}
    return np.array([d.get(a.name, 0.0) for a in master])


def finalists(ctx, master, E, C) -> list[dict]:
    """Every required comparator as a fund policy over the master asset list."""
    names = [a.name for a in master]
    S = len(rr.PORTFOLIO_STATES)
    NAV0 = E + C
    x0 = uni.initial_weights(master, E, C)
    out = []

    def static(name, x, branch, note=""):
        out.append({"name": name, "branch": branch, "note": note, "codes": None,
                    "pol": fund.FundPolicy(name, np.tile(x, (S, 1)), rebalance=False)})
    static("CONTROL_INITIAL_PORTFOLIO", x0, "BENCHMARK", "hold RUB 10m SMLT + liquidity")
    static("LQDT_ONLY", x0 * 0, "DE_RISKING", "whole fund in the liquidity sleeve (identical to SELL_100_TO_CASH)")
    static("SELL_50_TO_CASH", x0 * 0.5, "DE_RISKING")
    static("SELL_100_TO_CASH", x0 * 0, "DE_RISKING")
    for h in (0.25, 0.5):
        static(f"FUT_{int(h * 100)}", fut_weights(ctx, master, h, E, C), "V3_HEDGE_BENCHMARK")
    # V3 static smart hedge (6m selection, integer), notionals / E -> / NAV
    v3s = pd.read_csv(ctx.out / "v3_selected_portfolios.csv")
    r = v3s[(v3s["selection"] == "STATIC_SMART") & (v3s["horizon"] == "6m")].iloc[0]
    x = x0.copy()
    x[names.index("SMLT_FUT")] = float(r["w_DIRECT"]) * E / NAV0
    x[names.index("MIX")] = float(r.get("w_MARKET", 0.0)) * E / NAV0
    x, _ = core.integer_portfolio(ctx, master, x, NAV0)
    static("STATIC_SMART_HEDGE", x, "V3_HEDGE_BENCHMARK", "V3 STATIC_SMART_INT 6m (SMLT fixed + SMLT futures + MIX)")
    # V3 regime hedge: SMLT kept at its initial NAV weight, V3 library notionals per state
    lib = pd.read_csv(ctx.out / "regime_portfolio_library.csv")
    T = np.tile(x0, (S, 1))
    for i, st in enumerate(rr.PORTFOLIO_STATES):
        rw = lib[(lib["state"] == st) & (lib["selection"] == "LIBRARY")]
        if len(rw):
            rep = rw["merged_with"].iloc[0]
            rw = lib[(lib["state"] == rep) & (lib["selection"] == "LIBRARY")].iloc[0]
            T[i, names.index("SMLT_FUT")] = float(rw["int_w_DIRECT"]) * E / NAV0
            T[i, names.index("MIX")] = float(rw.get("int_w_MARKET", 0.0)) * E / NAV0
    cf, hd = ast.literal_eval(pd.read_csv(ctx.out / "state_machine_parameter_selection.csv")["selected"].iloc[-1])
    smc = ctx.cfg["v4"]["state_machine"]
    out.append({"name": "REGIME_HEDGE", "branch": "V3_HEDGE_BENCHMARK", "codes": "portfolio",
                "note": f"V3 regime library at fund level (SMLT weight fixed), confirm={cf} hold={hd}",
                "pol": fund.FundPolicy("REGIME_HEDGE", T, confirm=cf, hold=hd, threshold=smc["threshold"],
                                       protective_rank=protective_rank(ctx), emergency_shock=smc["emergency_shock"])})
    # V4 static portfolios
    for name, (assets, x) in static_book(ctx).items():
        static(name, _embed(assets, x, master), "PIF_ALLOCATION")
    by = {o["name"]: o for o in out}
    if "STATIC_PIF" in by:
        static("PIF_WITHOUT_BTC", by["STATIC_PIF"]["pol"].targets[0], "PIF_ALLOCATION", "alias of STATIC_PIF")
    # V4 regime PIF
    tg = pd.read_csv(ctx.out / "pif_regime_target_portfolios.csv")
    T = np.zeros((S, len(master)))
    for i, st in enumerate(rr.PORTFOLIO_STATES):
        rw = tg[tg["state"] == st].iloc[0]
        T[i] = np.array([float(rw.get(f"x_{a.name}", 0.0)) if pd.notna(rw.get(f"x_{a.name}", np.nan)) else 0.0 for a in master])
    sel = pd.read_csv(ctx.out / "pif_state_machine_selection.csv")["selected_confirm_hold"].iloc[-1]
    cf, hd = ast.literal_eval(sel)
    out.append({"name": "REGIME_PIF", "branch": "PIF_ALLOCATION", "codes": "portfolio",
                "note": f"fund-level state targets (s32), confirm={cf} hold={hd} (walk-forward selected)",
                "pol": fund.FundPolicy("REGIME_PIF", T, confirm=cf, hold=hd, threshold=smc["threshold"],
                                       protective_rank=protective_rank(ctx), emergency_shock=smc["emergency_shock"])})
    Tt = trend_capped(ctx, master, T, E, C)
    out.append({"name": "REGIME_PIF_TREND", "branch": "PIF_ALLOCATION", "codes": "portfolio",
                "note": f"REGIME_PIF targets + V3 trend de-risk cap (SMLT <= {ctx.cfg['v4']['trend_overlay']['cap_share_of_initial']:.0%} of the "
                        f"initial value in {', '.join(ctx.cfg['v4']['trend_overlay']['states'])}); confirm={smc['confirm_grid'][-1]} hold={smc['hold_grid'][-1]}",
                "pol": fund.FundPolicy("REGIME_PIF_TREND", Tt, confirm=smc["confirm_grid"][-1], hold=smc["hold_grid"][-1], threshold=smc["threshold"],
                                       protective_rank=protective_rank(ctx), emergency_shock=smc["emergency_shock"])})
    P_, R, *_ = _hist_inputs(ctx)
    for o in out:
        o["pol"].orders = orders_for(ctx, master, o["pol"].targets, R)
    return out


def run_finalists(ctx, fins, master, P, md, codes, E, C, coll=None) -> dict:
    from ..strategies.collateral import CASH_ONLY
    N, H = P["SMLT"].shape
    x0 = uni.initial_weights(master, E, C)
    res = {}
    for f in fins:
        st = np.zeros((N, H + 1), int) if f["codes"] is None else codes["portfolio"][:, :H + 1]
        res[f["name"]] = fund.run(P, md, master, f["pol"], st, E=E, C=C, x0_initial=x0, coll=coll or CASH_ONLY,
                                  shock=None if f["codes"] is None else codes["shock"][:, :H])
    return res


KEYS = ["mean_pnl", "median_pnl", "ES95", "ES99", "std_pnl", "p95_mdd_nav", "participation", "downside_capture",
        "p_liquidity_breach", "p_waterfall", "forced_funded_sales_mean", "transaction_costs", "turnover_nav", "mean_switches",
        "initial_margin_nav", "peak_margin_nav_p95"]


# ================================================================== s33 forward joint scenarios
def s33_forward(ctx) -> None:
    from ..scenarios import forward as fw
    from ..v3_stages import _forward_inputs, forward_paths
    v4 = ctx.cfg["v4"]
    E, C = ctx.E, float(v4["main_capital"])
    master = master_assets(ctx, E, C)
    cols = list(dict.fromkeys(["SMLT", "SMLT_FUT"] + [a.series for a in master] + ["SMLT_BOND_IDX"]))
    paths_, meta, anc = forward_paths(ctx, cols, n=v4["n_test"])
    mi = pd.read_csv(ctx.out / "market_inputs_at_cutoff.csv", index_col=0)["value"]
    y0, dmod = float(mi["mmf_yield_LQDT_60d"]), float(mi["rgbi_dmod"])
    # per-asset tilt provenance: beta_i(G) . mu(S) for every structural scenario
    Pi, Ri, X, masks = _forward_inputs(ctx)
    fcfg = ctx.cfg["v3"]["forward"]
    trows = []
    for name, sc in fcfg["scenarios"].items():
        mask = pd.concat([masks[k] for k in sc["regime_set"]], axis=1).any(axis=1)
        bc = [c for c in cols if c in X and c != "SMLT_FUT"]
        betas = fw.regime_betas(X, mask, list(dict.fromkeys(bc + ["MKT", "SECTOR", "RATES", "CREDIT"])), ctx.cfg["v3"]["shrink_n0"])
        mu = meta.set_index("scenario").loc[name, [f"mu12m_{f}" for f in ("MKT", "SECTOR", "RATES", "CREDIT")]].astype(float).values
        for c in bc:
            b = betas[c]["beta"]
            trows.append({"scenario": name, "asset_series": c, **{f"beta_{f}": v for f, v in zip(("MKT", "SECTOR", "RATES", "CREDIT"), b)},
                          **{f"mu12m_{f}": v for f, v in zip(("MKT", "SECTOR", "RATES", "CREDIT"), mu)},
                          "tilt_12m_log": float(b @ mu),
                          "source": "sum_k beta_k(regime set, shrunk to full sample) x mu_k(scenario anchor); NO own historical drift"})
        dy = -float(mu[2]) / dmod
        trows.append({"scenario": name, "asset_series": "LIQUIDITY (money market)", "tilt_12m_log": np.nan,
                      "source": f"annual yield path {y0:.2%} -> {max(y0 + dy, 0.01):.2%} over 12m (dy = -mu_RATES / D_mod RGBI = {dy:+.2%})"})
    ctx.save(pd.DataFrame(trows), "pif_scenario_tilts.csv", index=False)
    fins = finalists(ctx, master, E, C)
    fz, pref = frozen_thresholds(ctx), prefix_frame(ctx)
    rows = []
    for sc, Pf in paths_.items():
        base = ctx.cfg["v3"]["forward"]["overlays"].get(sc, {}).get("base", sc)
        mu_r = float(meta.set_index("scenario").loc[base, "mu12m_RATES"])
        y1 = max(y0 - mu_r / dmod, 0.01)
        yl = np.linspace(y0, y1, 252)
        for hz_name, hz in HZ.items():
            P = {c: v[:, :hz] for c, v in Pf.items() if isinstance(v, np.ndarray) and v.ndim == 2}
            P["IMOEX"], P["RGBI"] = P["MKT"], P["RATES"]
            md = (1 + yl[:hz]) ** (1 / 252) - 1
            codes = path_codes(ctx, P, fz, pref)
            res = run_finalists(ctx, fins, master, P, md, codes, E, C)
            for name, m in res.items():
                rows.append({"scenario": sc, "horizon": hz_name, "strategy": name, **{k: m[k] for k in KEYS},
                             "mean_pnl_per_E": m["mean_pnl"] / E, "mean_pnl_per_NAV": m["mean_pnl"] / (E + C),
                             "ES95_per_NAV": m["ES95"] / (E + C), **{k: m[k] for k in m if k.startswith("avg_w_")},
                             "label": "SCENARIO (not a forecast); not used for any selection"})
    ctx.save(pd.DataFrame(rows), "pif_forward_scenario_results.csv", index=False)


# ================================================================== s35 final TEST comparison
PAIRS = [("REGIME_PIF_TREND", "REGIME_PIF"), ("REGIME_PIF_TREND", "SELL_50_TO_CASH"), ("STATIC_PIF", "CONTROL_INITIAL_PORTFOLIO"), ("STATIC_PIF", "SELL_50_TO_CASH"), ("STATIC_PIF", "CASH_MATCHED"),
         ("STATIC_PIF", "FUT_50"), ("STATIC_PIF", "STATIC_SMART_HEDGE"), ("REGIME_PIF", "STATIC_PIF"),
         ("REGIME_HEDGE", "STATIC_SMART_HEDGE"), ("SELL_AND_REALLOCATE", "SELL_50_TO_CASH"),
         ("PIF_WITH_BTC_MOEX", "STATIC_PIF"), ("PIF_WITH_BTC_SPOT", "STATIC_PIF")]


def s35_final(ctx) -> None:
    from ..portfolio import validation as va
    from ..strategies.collateral import Collateral
    from ..v3_stages import verdict
    v4 = ctx.cfg["v4"]
    E, C = ctx.E, float(v4["main_capital"])
    NAV0 = E + C
    master = master_assets(ctx, E, C)
    fins = finalists(ctx, master, E, C)
    fac = core.factory(ctx, master)
    md = (1 + fac.B["y"]) ** (1 / 252) - 1
    fz, pref = frozen_thresholds(ctx), prefix_frame(ctx)
    rows = []
    for hz_name, hz in HZ.items():
        for k in range(v4["n_test_seeds"]):
            for mdl in core.MODELS:
                P = fac.paths(mdl, hz, v4["n_test"], "TEST", k)
                codes = path_codes(ctx, P, fz, pref)
                for name, m in run_finalists(ctx, fins, master, P, md, codes, E, C).items():
                    rows.append({"horizon": hz_name, "set": "TEST", "model": mdl, "seed_k": k, "strategy": name, **{x: m[x] for x in KEYS},
                                 **{x: m[x] for x in m if x.startswith("avg_w_")}})
        P = fac.paths("FHS_FULL", hz, v4["n_test"], "TEST", 0)
        codes = path_codes(ctx, P, fz, pref)
        for cname, c in ctx.cfg["collateral"]["named"].items():
            coll = Collateral(cname, c["yield_fraction"], c["haircut"], c["margin_multiplier"])
            for name, m in run_finalists(ctx, fins, master, P, md, codes, E, C, coll).items():
                rows.append({"horizon": hz_name, "set": f"COLLATERAL_{cname}", "model": "FHS_FULL", "seed_k": 0, "strategy": name,
                             **{x: m[x] for x in KEYS}})
    D = pd.DataFrame(rows)
    ctx.save(D, "pif_test_all_metrics.csv", "interim", index=False)
    test = D[D["set"] == "TEST"]
    br = {f["name"]: (f["branch"], f["note"]) for f in fins}
    out = []
    for name in br:
        row = {"strategy": name, "branch": br[name][0], "note": br[name][1]}
        tg = next(f for f in fins if f["name"] == name)["pol"].targets
        x_now = tg[int(rr.PORTFOLIO_STATES.index(state_now(ctx)))] if tg.shape[0] > 1 else tg[0]
        row.update({f"t0_x_{a.name}": float(v) for a, v in zip(master, x_now)})
        fund_m = np.array([a.kind == "FUNDED" for a in master])
        row["t0_liquidity_weight"] = float(1 - (x_now * fund_m).sum())
        row["t0_gross_nav"] = float(np.abs(x_now).sum())
        row["t0_net_exposure_nav"] = float(x_now.sum())
        for hz in HZ:
            g = test[(test["strategy"] == name) & (test["horizon"] == hz)]
            fhs = g[g["model"] == "FHS_FULL"]
            for x in KEYS:
                row[f"{x}_{hz}"] = float(fhs[x].mean())
            wm = g.groupby("model")["ES95"].mean()
            row[f"worst_model_{hz}"], row[f"worst_model_ES95_{hz}"] = wm.idxmax(), float(wm.max())
            row[f"worst_model_participation_{hz}"] = float(g.groupby("model")["participation"].mean().min())
            row[f"ES95_per_E_{hz}"], row[f"ES95_per_NAV_{hz}"] = row[f"ES95_{hz}"] / E, row[f"ES95_{hz}"] / NAV0
            row[f"mean_pnl_per_E_{hz}"], row[f"mean_pnl_per_NAV_{hz}"] = row[f"mean_pnl_{hz}"] / E, row[f"mean_pnl_{hz}"] / NAV0
            for cname in ctx.cfg["collateral"]["named"]:
                c = D[(D["set"] == f"COLLATERAL_{cname}") & (D["strategy"] == name) & (D["horizon"] == hz)]
                if len(c):
                    row[f"breach_{hz}_{cname}"] = float(c["p_liquidity_breach"].iloc[0])
        out.append(row)
    ctx.save(pd.DataFrame(out), "pif_final_comparison.csv", index=False)
    drows = []
    tt = test.rename(columns={"strategy": "cand_id"})
    for a, b in PAIRS:
        if a not in br or b not in br:
            continue
        for hz in HZ:
            g = tt[tt["horizon"] == hz]
            for metric in ("ES95", "mean_pnl", "participation", "p95_mdd_nav", "p_liquidity_breach"):
                pm = va.delta_ci(g, a, b, metric)
                if not pm:
                    continue
                good_neg = metric in ("ES95", "p95_mdd_nav", "p_liquidity_breach")
                mat = ctx.cfg["v3"]["stability"]["material_rub"] if metric in ("ES95", "mean_pnl") else 0.0
                vd = verdict(pm, mat, ctx.cfg["v3"]["stability"]["min_sign_share"], good_neg)
                for mdl, st in pm.items():
                    drows.append({"horizon": hz, "strategy": a, "comparator": b, "metric": metric, "model": mdl, **st, "verdict_all_models": vd})
    ctx.save(pd.DataFrame(drows), "pif_final_delta_ci.csv", index=False)
    # validation stability: VALIDATION (s31) vs TEST for the selected static portfolios
    sel = pd.read_csv(ctx.out / "pif_selected_portfolios.csv").set_index("portfolio")
    vs = []
    for name in sel.index:
        g = test[(test["strategy"] == name) & (test["horizon"] == "6m")]
        if g.empty:
            continue
        wm = g.groupby(["seed_k", "model"])["ES95"].mean().groupby("seed_k").max()
        vs.append({"portfolio": name, "val_worst_ES95": sel.loc[name].get("worst_ES95"), "test_worst_ES95_mean": float(wm.mean()),
                   "test_worst_ES95_sd": float(wm.std(ddof=1)), "test_worst_ES95_min": float(wm.min()), "test_worst_ES95_max": float(wm.max()),
                   "val_worst_participation": sel.loc[name].get("worst_participation"),
                   "test_worst_participation_mean": float(g.groupby(["seed_k", "model"])["participation"].mean().groupby("seed_k").min().mean()),
                   "test_breach_max": float(g["p_liquidity_breach"].max()), "n_seeds": int(g["seed_k"].nunique())})
    ctx.save(pd.DataFrame(vs), "pif_validation_stability.csv", index=False)
    # liquidity waterfall: orders at the cutoff state + event statistics
    wrows = []
    names = [a.name for a in master]
    s_now = rr.PORTFOLIO_STATES.index(state_now(ctx))
    for f in fins:
        o = f["pol"].orders
        order = o[s_now] if o is not None else []
        tgt = f["pol"].targets[s_now if f["pol"].targets.shape[0] > 1 else 0]
        g = test[(test["strategy"] == f["name"]) & (test["horizon"] == "6m") & (test["model"] == "FHS_FULL")]
        wrows.append({"strategy": f["name"], "waterfall_order_at_cutoff_state": " > ".join(names[j] for j in order if abs(tgt[j]) > 1e-9),
                      "p_waterfall_6m_FHS": float(g["p_waterfall"].mean()), "p_liquidity_breach_6m_FHS": float(g["p_liquidity_breach"].mean()),
                      "forced_funded_sales_mean_rub": float(g["forced_funded_sales_mean"].mean()),
                      "peak_margin_nav_p95": float(g["peak_margin_nav_p95"].mean()),
                      "rule": "cash first (free liquidity), then the action with the lowest (d variance / RUB released) - see pif/fund.py"})
    ctx.save(pd.DataFrame(wrows), "pif_liquidity_waterfall.csv", index=False)
    # risk contributions at t0 (ES95 Euler contributions on one TEST path set, FHS 6m)
    P = fac.paths("FHS_FULL", 126, v4["n_test"], "TEST", 0)
    from ..scenarios.bootstrap import horizon_returns
    HR = horizon_returns(P, list(dict.fromkeys(["SMLT"] + [a.series for a in master])))
    m = (1 + fac.B["y"]) ** 0.5 - 1
    crow = []
    for f in fins:
        x = f["pol"].targets[s_now if f["pol"].targets.shape[0] > 1 else 0]
        fund_m = np.array([a.kind == "FUNDED" for a in master])
        comp = {a.name: x[j] * (HR[a.series].values - (m if fund_m[j] else 0.0)) * NAV0 for j, a in enumerate(master) if abs(x[j]) > 1e-9}
        tot = sum(comp.values()) + m * NAV0 if comp else np.full(len(HR), m * NAV0)
        tail = tot <= np.quantile(tot, 0.05)
        for k2, v in comp.items():
            crow.append({"strategy": f["name"], "asset": k2, "es95_contribution_rub": float(-v[tail].mean()),
                         "share_of_es95": float(-v[tail].mean() / -tot[tail].mean()) if tot[tail].mean() < 0 else np.nan,
                         "std_contribution_rub": float(np.cov(v, tot)[0, 1] / np.std(tot, ddof=1))})
    ctx.save(pd.DataFrame(crow), "pif_risk_contributions.csv", index=False)


def state_now(ctx) -> str:
    p = ctx.interim / "regime_states_daily.csv"
    return str(pd.read_csv(p, index_col=0)["portfolio_state"].iloc[-1])


# ================================================================== s34 BTC under the PIF objective
def s34_btc(ctx) -> None:
    from ..v3_stages import _regime_inputs
    v4 = ctx.cfg["v4"]
    E, C = ctx.E, float(v4["main_capital"])
    rows = []
    # (a) paired TEST deltas (from s35)
    dl = pd.read_csv(ctx.out / "pif_final_delta_ci.csv")
    for u in ("PIF_WITH_BTC_MOEX", "PIF_WITH_BTC_SPOT"):
        for _, r in dl[(dl["strategy"] == u) & (dl["comparator"] == "STATIC_PIF")].iterrows():
            rows.append({"test": "TEST_paired_vs_WITHOUT_BTC", "universe": u, "context": f"{r['horizon']} {r['model']}", "metric": r["metric"],
                         "value": r["mean"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "verdict": r["verdict_all_models"]})
    fc = pd.read_csv(ctx.out / "pif_final_comparison.csv").set_index("strategy")
    for u in ("STATIC_PIF", "PIF_WITH_BTC_MOEX", "PIF_WITH_BTC_SPOT"):
        if u in fc.index:
            for col in ("t0_x_BTC_SPOT_EXP", "t0_x_BTC_FUT", "turnover_nav_6m", "initial_margin_nav_6m", "p95_mdd_nav_6m"):
                rows.append({"test": "ALLOCATION", "universe": u, "context": "decision date", "metric": col, "value": float(fc.loc[u, col])})
    # (b) forward scenarios (reported, never used to select)
    fw = pd.read_csv(ctx.out / "pif_forward_scenario_results.csv")
    for sc, g in fw.groupby(["scenario", "horizon"]):
        g = g.set_index("strategy")
        for u in ("PIF_WITH_BTC_MOEX", "PIF_WITH_BTC_SPOT"):
            if u in g.index and "STATIC_PIF" in g.index:
                for metric in ("mean_pnl", "ES95", "p95_mdd_nav"):
                    rows.append({"test": "FORWARD_SCENARIO_vs_WITHOUT_BTC", "universe": u, "context": f"{sc[0]} {sc[1]}", "metric": metric,
                                 "value": float(g.loc[u, metric] - g.loc["STATIC_PIF", metric])})
    # (c) by observable state: state-conditional selection in each universe
    P, R, X, reg, ps, pst = _regime_inputs(ctx)
    masks = rr.state_masks(reg, ps)
    for st in ("P_NORMAL", "P_DOWNTREND", "P_MARKET_STRESS", "P_CREDIT_STRESS", "P_RECOVERY"):
        base = None
        for u in v4["universes"]:
            assets = uni.build(ctx, u, E + C, _cap_initial(E, C))
            fac = core.factory(ctx, assets, regime_mask=masks[st], label=st)
            Ms = core.lp_models(fac, assets, 126, v4["n_lp"], "TRAIN", models=["REGIME_COND", "REGIME_PURE"])
            lp = core.make_lp(ctx, assets, E, C, 126)
            fr = core.frontier(ctx, lp, Ms, {"state": st})
            cands = [(f"{u}_{j:02d}", core.static_policy(core.xvec(r, assets)), None) for j, r in enumerate(fr)]
            ev = core.evaluate(ctx, fac, assets, cands, 126, E, C, "VAL", v4["n_val"] // 2, models=["REGIME_COND", "REGIME_PURE"])
            summ = core.summarise(ev)
            sel = core.select(summ, E + C, v4["selection"]["es_budget"], v4["selection"]["max_worst_breach"])
            if sel is None:
                continue
            x = core.xvec(pd.DataFrame(fr).assign(cand_id=[c[0] for c in cands]).set_index("cand_id").loc[sel["cand_id"]], assets)
            w_btc = float(sum(v for a, v in zip(assets, x) if a.name.startswith("BTC")))
            rec = {"worst_ES95": sel["worst_ES95"], "participation": sel["worst_participation"], "mdd": sel["worst_mdd_p95"],
                   "mean": sel["mean_pnl_avg"], "turnover": sel["turnover"]}
            if u == "WITHOUT_BTC":
                base = rec
            for metric, val in [("btc_weight_nav", w_btc)] + [(f"incremental_{k}", rec[k] - base[k]) for k in rec if base is not None]:
                rows.append({"test": "BY_STATE_selection", "universe": u, "context": st, "metric": metric, "value": val})
    # (d) SENSITIVITY ONLY: assumed BTC premium (never estimated) -> weight chosen by max-premium LP under the ES budget
    for u in ("WITH_BTC_MOEX_DERIVATIVE", "WITH_BTC_EXPERIMENTAL_SPOT"):
        assets = uni.build(ctx, u, E + C, _cap_initial(E, C))
        fac = core.factory(ctx, assets)
        Ms = core.lp_models(fac, assets, 126, v4["n_lp"], "TRAIN")
        lp = core.make_lp(ctx, assets, E, C, 126)
        for erp in v4["premium_sensitivity"]["equity_erp_grid"]:
            for bp in v4["premium_sensitivity"]["btc_premium_grid"]:
                prem = np.array([bp if a.name.startswith("BTC") else (erp if a.name in ("SMLT", "MIX") else 0.0) for a in assets])
                r = opt.solve(lp, Ms, "max_premium", budget=v4["selection"]["es_budget"], premia=prem)
                w = float(sum(v for a, v in zip(assets, r.get("x", [])) if a.name.startswith("BTC"))) if r["status"] == "optimal" else np.nan
                rows.append({"test": "PREMIUM_SENSITIVITY (assumption, not estimate)", "universe": u,
                             "context": f"equity premium {erp:.0%}, BTC premium {bp:.0%}", "metric": "btc_weight_nav", "value": w})
    rs = pd.read_csv(ctx.out / "btc_regime_statistics.csv")
    for _, r in rs[rs["btc_series"] == "BTC_PROXY"].iterrows():
        rows.append({"test": "TAIL_COMOVEMENT (V3)", "universe": "BTC proxy", "context": r["set"], "metric": "tail_coexceedance_ratio",
                     "value": r["tail_coexceedance_ratio_vs_independence"]})
    ctx.save(pd.DataFrame(rows), "pif_btc_incremental_value.csv", index=False)


def trend_capped(ctx, assets, T: np.ndarray, E: float, C: float) -> np.ndarray:
    """REGIME_PIF_TREND targets: SMLT capped in the trend-overlay states (config v4.trend_overlay)."""
    tov = ctx.cfg["v4"]["trend_overlay"]
    j = [a.name for a in assets].index("SMLT")
    T = T.copy()
    cap = tov["cap_share_of_initial"] * E / (E + C)
    for st in tov["states"]:
        i = rr.PORTFOLIO_STATES.index(st)
        T[i, j] = min(T[i, j], cap)
    return T
