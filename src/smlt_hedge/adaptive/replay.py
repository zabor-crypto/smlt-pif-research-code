"""Gate R6 — chronological replay of every policy as ONE self-financing quantity ledger on exchange data.

No concatenated returns. At each close t the engine carries the actual positions, settles VM and pending settlements,
posts margin at the as-of rate, and only then calls the policy with the state observed at t; new orders execute at the
close of session t + delay (canonical 1). Contract identity, notional, margin, fee and point value are as of the date.
A route can be traded only from its availability date (BTC_RUB_EXPERIMENTAL is the owner-authorised exception).
Recalibration (bootstrap sample, view points, state forward statistics) happens at the first decision close of January
and July; weekly decisions update only observable inputs (prices, contract terms, current volatility, money-market
rate, state, events).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import engine as E
from . import events as EV
from . import frontier as F
from . import market as M
from . import orders as O
from . import policy as PL
from . import scenarios as SC
from . import state as ST
from . import views as V
from .r1 import loss_quantile, reserve_min, vm99_frac

GROUP = {"SMLT": "SMLT", "SMLT_FUT": "SMLT", "EQ_FUNDED": "BROAD_EQUITY", "EQ_HEDGE_FUT": "BROAD_EQUITY",
         "OFZ_FUNDED": "RATES_OFZ", "RATES_HEDGE_FUT": "RATES_OFZ", "GOLD_FUNDED": "GOLD", "CNY_PERP": "FX_CNY",
         "OIL_FUT": "OIL", "BTC_RUB_EXPERIMENTAL": "BTC", "MMF": "MONEY_MARKET_CARRY"}


@dataclass
class World:
    dates: pd.DatetimeIndex
    fin: list
    din: list
    mkt: E.Market
    px: pd.DataFrame
    fin_names: list
    din_names: list
    terms: dict
    routes: list


def unit_prices(ctx, cfg, P) -> pd.DataFrame:
    from .routes import fund_prices
    px = pd.read_csv(ctx.proc / "prices.csv", index_col=0, parse_dates=True)
    cal = P["cal"]
    out = pd.DataFrame(index=cal)
    out["SMLT"] = px["SMLT"].reindex(cal)
    out["EQ_FUNDED"] = fund_prices(ctx.root, cfg["r2"]["funds"]["broad_equity"], cfg["cutoff"])["PX"].reindex(cal)
    out["OFZ_FUNDED"] = fund_prices(ctx.root, cfg["r2"]["funds"]["ofz"], cfg["cutoff"])["PX"].reindex(cal)
    out["GOLD_FUNDED"] = px[cfg["r2"]["funds"]["gold"]].reindex(cal)
    out["BTC_RUB_EXPERIMENTAL"] = P["B"]["BTC_RUB_LEVEL"].reindex(cal)
    out["MMF"] = px["LQDT"].reindex(cal)
    return out


def world(ctx, cfg, P, universe: str, start: str, cost_mult: float = 1.0) -> World:
    routes = list(cfg["r3"]["universes"][universe])
    px = unit_prices(ctx, cfg, P)
    dates = P["cal"][(P["cal"] >= pd.Timestamp(start)) & (P["cal"] <= pd.Timestamp(cfg["cutoff"]))]
    costs = cfg["r3"]["costs"]
    fs = []
    for r in routes:
        if r not in F.FUNDED:
            continue
        if r == "BTC_RUB_EXPERIMENTAL":
            fs.append(M.FundedSpec(r, E.FUNDED, float(px[r].loc[dates[0]]), lag=0, lot=0.0, cost_prop=costs["btc_experimental"] * cost_mult))
        elif r == "SMLT":
            fs.append(M.FundedSpec(r, E.FUNDED, float(px[r].loc[dates[0]]), lag=1, lot=1.0, cost_prop=costs["smlt"]["bp"] * 1e-4 * cost_mult,
                                   cost_unit=costs["smlt"]["half_tick_rub"] * cost_mult, issuer="SMLT"))
        else:
            fs.append(M.FundedSpec(r, E.FUNDED, float(px[r].loc[dates[0]]), lag=1, lot=1.0, cost_prop=costs["funds"] * cost_mult))
    fs.append(M.FundedSpec("MMF", E.MMF, float(px["MMF"].loc[dates[0]]), lag=1, lot=1.0))
    droutes = [r for r in routes if r not in F.FUNDED]
    fin, din, mkt, terms = M.historical_market(ctx, cfg, fs, [F.DERIV_FAMILY[r] for r in droutes], px, dates, P["usd"],
                                               cfg["r1"]["liquidity"]["immediate_cash_rate"])
    if cost_mult != 1.0:
        mkt.fee = mkt.fee * cost_mult
        if mkt.execution is not None:
            for family in mkt.execution['series']:
                for series in family.values():
                    series['fee'] = series['fee'] * cost_mult
    din = [E.Instrument(r, E.DERIV) for r in droutes]
    return World(dates, fin, din, mkt, px, [f.name for f in fs], droutes, terms, routes)


def decision_days(dates: pd.DatetimeIndex) -> np.ndarray:
    wk = dates.isocalendar().week.values
    yr = dates.isocalendar().year.values
    last = np.r_[(wk[1:] != wk[:-1]) | (yr[1:] != yr[:-1]), True]
    last[0] = True                       # the replay opens with a decision at the first close
    return last


def recal_days(dates: pd.DatetimeIndex, months: list, dec: np.ndarray) -> np.ndarray:
    """First DECISION close on or after the first session of each recalibration month; plus the first day."""
    out = np.zeros(len(dates), bool)
    out[0] = True
    seen = set()
    for i, d in enumerate(dates):
        if d.month in months and dec[i] and (d.year, d.month) not in seen and i > 0:
            out[i] = True
            seen.add((d.year, d.month))
    return out


def state_forward_table(cfg, P, conf: pd.DataFrame, t_recal: pd.Timestamp, H: int) -> pd.DataFrame:
    """Forward H-session statistics by CONFIRMED state on data <= t_recal (computed once per recalibration)."""
    c4 = cfg["r4"]
    routes = sorted({r for v in cfg["r5"]["state_views"]["mapping"].values() for r in v} | {"SMLT"})
    kinds = [(r, "FUNDED" if r in F.FUNDED else "DERIVATIVE") for r in routes]
    R = P["R"].loc[:t_recal]
    fs, _ = ST.forward_stats(conf.loc[:t_recal], R, R["MMF"], kinds, [H], c4["shrink_n0_windows"], c4["sample_start"],
                             c4["bootstrap_seed"], n_boot=500)
    return fs


def state_views_at(cfg, fs: pd.DataFrame, t_recal: pd.Timestamp, states_now: dict, H: int, rule: str | None = None) -> dict:
    """SCENARIO_VIEW points for the CURRENT confirmed states (rule declared in configs r4.state_view, amendment R4-A1).
    STATE_SPECIFIC_DIFFERENTIAL (canonical): interval excludes the all-origin mean; value = shrunk state differential.
    ORIGINAL_R4_LEVEL (labelled sensitivity): interval excludes zero; value = shrunk level."""
    c = cfg["r4"]["state_view"]
    rule = rule or c["rule"]
    assert rule in ST.STATE_VIEW_RULES, rule
    u = ST.view_table(fs, c["min_nonoverlapping_windows"])
    flag, val = (("usable_as_state_view", "state_differential_shrunk") if rule == "STATE_SPECIFIC_DIFFERENTIAL"
                 else ("usable_under_original_level_rule", "shrunk_mean"))
    out = {}
    for dim, routes in cfg["r5"]["state_views"]["mapping"].items():
        g = u[(u["dimension"] == dim) & (u["state"] == states_now[dim]) & (u["horizon"] == H) & u["route"].isin(routes) & u[flag]]
        for _, row in g.iterrows():
            out[F.SERIES.get(row["route"], row["route"])] = {
                "value": float(row[val]), "state": f"{dim}={states_now[dim]}",
                "source": f"R4 forward {H}-session excess after {dim}={states_now[dim]} origins (data <= {t_recal.date()}), rule {rule}: "
                          f"{val} {row[val]:.4f}; state mean {row['mean_fwd_excess']:.4f}, all-origin mean {row['all_origin_mean']:.4f}, "
                          f"{int(row['n_nonoverlapping_windows'])} non-overlapping windows, CI90 [{row['ci90_lo_episode_bootstrap']:.3f}, {row['ci90_hi_episode_bootstrap']:.3f}]"}
    return out


def run_policy(ctx, cfg, P, W: World, kind: str, *, universe: str, mandate: str, kappa: float, delay: int,
               cost_mult: float = 1.0, priors: dict | None = None, n_paths: dict | None = None, log_rows: list | None = None,
               name: str = "", state_view_rule: str | None = None) -> dict:
    """kind: ADAPTIVE (weekly, state caps + state views) | STATIC_FRONTIER (recalibration dates only, no states)."""
    r6 = cfg["r6"]
    dates = W.dates
    T = len(dates)
    beta = cfg["owner"]["liquidity_min_nav"]
    H = cfg["r3"]["frontier"]["horizon_sessions"]
    tau = H / 252
    budget = r6["es_budget_nav"]
    npth = n_paths or r6["scenario_paths"]
    feat = ST.features(ctx, cfg)
    raw = ST.classify(feat, cfg)
    Ev = EV.register(ctx, cfg)
    severe = pd.Series([EV.state_at(Ev, d)["severe_restriction"] for d in raw.index], index=raw.index)
    conf = PL.confirmed_states(raw, cfg, severe)
    dec = decision_days(dates)
    rec = recal_days(dates, cfg["r5"]["recalibration"]["months"], dec)
    S0 = float(W.px["SMLT"].loc[dates[0]])
    shares = np.floor(r6["smlt_initial_value_rub"] / S0)
    book0 = E.Book(qf=np.array([shares if n == "SMLT" else 0.0 for n in W.fin_names]), qd=np.zeros(len(W.din_names)),
                   cash=r6["nav0_rub"] - shares * S0)
    cache = {"recal": None, "points": None, "panel": None, "fs": None}
    decisions = []

    def policy(i, st):
        t = dates[i]
        ev = EV.state_at(Ev, t)
        sev_new = ev["severe_restriction"] and not (i > 0 and EV.state_at(Ev, dates[i - 1])["severe_restriction"])
        if kind == "STATIC_FRONTIER":
            if not rec[i]:
                return {}
        elif not (dec[i] or sev_new):
            return {}
        if i + delay > T - 1:
            return {}
        if rec[i] or cache["recal"] is None:
            panel = SC.history_panel(P, t, cfg["r3"]["frontier"]["estimation_start"])
            pts, _ = V.registry(ctx, cfg, P, t, tau, priors, panel)
            cache.update({"recal": t, "points": pts, "panel": panel,
                          "fs": state_forward_table(cfg, P, conf, t, H) if kind == "ADAPTIVE" else None})
        states = conf.loc[t].to_dict()
        pts = cache["points"]
        sv = {}
        if kind == "ADAPTIVE":
            sv = state_views_at(cfg, cache["fs"], cache["recal"], states, H, state_view_rule)
            pts = V.add_state_views(pts, sv, t, tau)
        iv = V.intervals(pts, cache["panel"], tau).set_index("route")
        pend = {int(k): float(v) for k, v in enumerate(st["pend"][0, i + 1:], start=1) if abs(v) > 1e-9}
        book = E.Book(qf=st["qf"][0].copy(), qd=st["qd"][0].copy(), cash=float(st["cash"][0]), pending=pend)
        d = PL.decide(ctx, cfg, P, t, book, float(st["available"][0]), states, ev, iv, universe, mandate, kappa, budget,
                      panel_end=cache["recal"], px=W.px, n_gen=npth["n_generation_per_model"], n_adm=npth["n_admission_per_model"],
                      use_caps=(kind == "ADAPTIVE"), delay=delay, cost_mult=cost_mult)
        row = {"policy": name or kind, "date": t, "decision_index": i, "recalibration": bool(rec[i]), "action": d.action,
               "current_admitted": d.current_admitted, "issuer_cap": d.cap, **{f"state_{k}": v for k, v in states.items()},
               "severe_restriction": ev["severe_restriction"], "review_flags": "; ".join(ev["review_flags"]),
               "distance_to_nearest_admitted": d.distance, "n_candidates": len(d.candidates),
               "n_admitted": sum(c["passed"] for c in d.candidates), "nav": float(st["nav"][0]),
               "available_nav": float(st["available"][0] / st["nav"][0]), "reasons": " | ".join(d.reasons),
               "current_fail_reasons": " | ".join(d.current_reasons),
               **{f"w_cur_{r}": v for r, v in d.current_w.items()},
               **{f"w_tgt_{r}": v for r, v in (d.target_w or {}).items()},
               "engine_ES95_target_nav0": d.metrics.get("engine_ES95_worst_nav0"),
               "engine_ES95_current_nav0": d.current_metrics.get("engine_ES95_worst_nav0"),
               "execution_date": str(dates[min(i + delay, T - 1)].date()), "state_views_used": "; ".join(sorted(sv))}
        decisions.append(row)
        if d.ex is None:
            return {}
        ords = [E.Order(o.day + i, o.qf, o.qd, o.tag, o.delta) for o in d.ex["orders"]]   # relative -> absolute day
        ords = [o for o in ords if o.day <= T - 1]
        return {"orders": ords, "reserve": d.ex["reserve_min"], "wf_rank": d.ex["wf_rank"], "restore_rank": d.ex["restore_rank"]}
    eng = E.Engine(W.fin, W.din, W.mkt, E.Params(beta=beta, reserve_frac=beta, hysteresis=cfg["r1"]["liquidity"]["sweep_hysteresis_nav"], log=True))
    res = eng.run(book0, policy=policy)
    if log_rows is not None:
        log_rows.extend(decisions)
    return {"res": res, "decisions": pd.DataFrame(decisions)}


def run_fixed(ctx, cfg, P, W: World, target: dict, *, delay: int = 1, reset_at_recal: bool = False, checkpoint=None, issuer_floor: bool = False) -> dict:
    """Comparator with a fixed target: {'shares_frac': f} (share of the initial shares kept), {'hedge_frac': h} (short
    SMLT futures as a fraction of the shares) or {'weights': {route: NAV weight}} (re-set at recalibration dates)."""
    r6 = cfg["r6"]
    dates = W.dates
    T = len(dates)
    beta = cfg["owner"]["liquidity_min_nav"]
    lv = cfg["r1"]["liquidity"]
    S0 = float(W.px["SMLT"].loc[dates[0]])
    shares = np.floor(r6["smlt_initial_value_rub"] / S0)
    book0 = E.Book(qf=np.array([shares if n == "SMLT" else 0.0 for n in W.fin_names]), qd=np.zeros(len(W.din_names)),
                   cash=r6["nav0_rub"] - shares * S0)
    dec = decision_days(dates)
    rec = recal_days(dates, cfg["r5"]["recalibration"]["months"], dec)
    rets = P["R"]

    def policy(i, st):
        if not (i == 0 or (reset_at_recal and rec[i])) or i + delay > T - 1:
            return {}
        t = dates[i]
        nav = float(st["nav"][0])
        Pn = W.mkt.P[0, i, :]
        notional = W.mkt.Fn[0, i, :] * W.mkt.pv[0, i, :]
        tf = np.full(len(W.fin_names), np.nan)
        td = st["qd"][0].copy()
        js = W.fin_names.index("SMLT")
        if "weights" in target:
            wf = np.array([target["weights"].get(n, 0.0) for n in W.fin_names])
            wd = np.array([target["weights"].get(n, 0.0) for n in W.din_names])
            qf, qd = E.to_quantities(wf, wd, nav, Pn, notional, np.array([f.lot for f in W.fin]))
            for j, n in enumerate(W.fin_names):
                if n != "MMF":
                    tf[j] = qf[j]
            td = qd
        else:
            tf[js] = np.floor(shares * target.get("shares_frac", 1.0))
            for j, n in enumerate(W.fin_names):
                if n not in ("MMF", "SMLT"):
                    tf[j] = 0.0
            if "SMLT_FUT" in W.din_names:
                td = np.zeros(len(W.din_names))
                td[W.din_names.index("SMLT_FUT")] = -np.trunc(target.get("hedge_frac", 0.0) * shares)
        hist = rets.loc[:t]
        vm = sum(vm99_frac(hist[r].iloc[:-1], td[k] * notional[k], nav, lv["reserve_var_level"]) for k, r in enumerate(W.din_names))
        res_min = reserve_min(beta, vm)
        hc = np.array([0.0 if n == "MMF" else loss_quantile(hist[n].values, lv["reserve_var_level"]) for n in W.fin_names])
        ords, _ = O.build(i + delay, st["qf"][0].copy(), st["qd"][0].copy(), tf, td, Pn, notional, W.mkt.mr[i], 1.0, W.fin,
                          available=float(st["available"][0]), nav=nav, target_frac=res_min + lv["sweep_hysteresis_nav"] / 2,
                          haircut=hc, lag_cycle=int(W.mkt.lag[min(i + delay, T - 1), js]))
        ords = [o for o in ords if o.day <= T - 1]
        return {"orders": ords, "reserve": res_min}
    eng = E.Engine(W.fin, W.din, W.mkt, E.Params(beta=beta, reserve_frac=beta, hysteresis=lv["sweep_hysteresis_nav"], log=True))
    eng.p.issuer_floor = issuer_floor
    return {"res": eng.run(book0, policy=policy, checkpoint=checkpoint), "decisions": pd.DataFrame()}


# ---------------------------------------------------------------------------------------------- metrics
def ledger_frame(W: World, res: dict) -> pd.DataFrame:
    lg = pd.DataFrame(res["log"])
    lg.index = W.dates
    lg.index.name = "date"
    return lg


def metrics(W: World, res: dict, name: str, mmf: pd.Series, nav0: float) -> dict:
    lg = ledger_frame(W, res)
    nav = lg["nav"]
    mm = mmf.reindex(W.dates) / float(mmf.reindex(W.dates).iloc[0])
    r = nav.pct_change().dropna()
    yrs = len(nav) / 252

    def worst(h):
        x = nav.pct_change(h).dropna()
        return float(x.min()) if len(x) else np.nan

    def es(h, a):
        x = np.sort(nav.pct_change(h).dropna().values)
        k = max(1, int(np.ceil((1 - a) * len(x))))
        return float(-x[:k].mean()) if len(x) else np.nan
    peak = nav.cummax()
    exc_path = nav / nav0 - mm
    rs = W.px["SMLT"].reindex(W.dates).pct_change().reindex(r.index)
    up = rs > 0
    capture = float(r[up].mean() / rs[up].mean()) if up.any() and rs[up].mean() > 0 else np.nan
    wcols = [c for c in lg if c.startswith("v_") or c.startswith("n_")]
    w = lg[wcols].div(nav, axis=0)
    issuer = w.get("v_SMLT", 0) + w.get("n_SMLT_FUT", 0)
    gross = w[[c for c in wcols if c != "v_MMF"]].abs().sum(axis=1)
    net = w[[c for c in wcols if c != "v_MMF"]].sum(axis=1)
    return {"comparator": name, "start": str(W.dates[0].date()), "end": str(W.dates[-1].date()), "sessions": len(nav),
            "terminal_nav": float(nav.iloc[-1]), "total_return": float(nav.iloc[-1] / nav0 - 1),
            "money_market_total_return": float(mm.iloc[-1] - 1), "net_excess_over_money_market": float(exc_path.iloc[-1]),
            "cagr": float((nav.iloc[-1] / nav0) ** (1 / yrs) - 1), "vol_ann": float(r.std(ddof=1) * np.sqrt(252)),
            "ES95_6m_rolling": es(126, 0.95), "ES99_6m_rolling": es(126, 0.99), "ES95_1d": es(1, 0.95), "ES99_1d": es(1, 0.99),
            "max_drawdown": float(((peak - nav) / peak).max()), "worst_1d": worst(1), "worst_5d": worst(5), "worst_20d": worst(20),
            "shortfall_1m": worst(21), "shortfall_3m": worst(63), "shortfall_6m": worst(126), "shortfall_12m": worst(252),
            "peak_im_nav": float((lg["im"] / nav).max()), "min_available_nav": float((lg["available"] / nav).min()),
            "liquidity_breach_days": int(lg["breach"].sum()), "unresolved_deficit_days": int(lg["unresolved"].sum()),
            "forced_reduction_contracts": float(res["forced_contracts"][0]), "forced_reduction_events": int(res["forced_events"][0]),
            "max_pending_settlement_nav": float((lg["pending"].abs() / nav).max()), "turnover_nav0": float(res["turnover"][0] / nav0),
            "costs_rub": float(res["costs"][0]), "gross_exposure_avg": float(gross.mean()), "net_exposure_avg": float(net.mean()),
            "smlt_issuer_exposure_avg": float(np.mean(issuer)), "smlt_issuer_exposure_end": float(np.asarray(issuer)[-1]),
            "broad_equity_exposure_avg": float((w.get("v_EQ_FUNDED", 0) + w.get("n_EQ_HEDGE_FUT", 0)).mean() if len(w) else 0.0),
            "rates_exposure_avg": float((w.get("v_OFZ_FUNDED", 0) + w.get("n_RATES_HEDGE_FUT", 0)).mean() if len(w) else 0.0),
            "fx_exposure_avg": float(np.mean(w.get("n_CNY_PERP", 0))), "gold_exposure_avg": float(np.mean(w.get("v_GOLD_FUNDED", 0))),
            "oil_exposure_avg": float(np.mean(w.get("n_OIL_FUT", 0))), "btc_exposure_avg": float(np.mean(w.get("v_BTC_RUB_EXPERIMENTAL", 0))),
            "secondary_smlt_up_day_capture": capture,      # mean NAV return on SMLT up-days / mean SMLT return on those days
            "evidence_class": "RETROSPECTIVE_WALK_FORWARD (one ledger; one history; NOT market OOS)"}


def attribution(W: World, res: dict, name: str, mmf: pd.Series, nav0: float) -> dict:
    out = {"comparator": name}
    g = {}
    for j, n in enumerate(W.fin_names):
        g[GROUP[n]] = g.get(GROUP[n], 0.0) + float(res["pnl_funded"][0, j])
    for j, n in enumerate(W.din_names):
        g[GROUP[n]] = g.get(GROUP[n], 0.0) + float(res["pnl_deriv"][0, j])
        out[f"memo_vm_{n}"] = float(res["pnl_deriv"][0, j])
    lg = ledger_frame(W, res)
    mr_ = mmf.reindex(W.dates).pct_change().fillna(0.0)
    out.update({f"pnl_{k}": v for k, v in g.items()})
    out["pnl_interest_on_cash"] = float(res["interest"][0])
    out["pnl_transaction_and_roll_costs"] = -float(res["costs"][0])
    out["memo_roll_costs"] = -float(res["cost_roll"][0])
    out["memo_liquidity_and_margin_drag"] = -float((lg["cash"].shift(1).fillna(lg["cash"].iloc[0]).clip(lower=0) * mr_).sum())
    total = sum(g.values()) + out["pnl_interest_on_cash"] + out["pnl_transaction_and_roll_costs"]
    out["pnl_total"] = total
    out["nav_change"] = float(lg["nav"].iloc[-1] - nav0)
    out["identity_residual"] = out["nav_change"] - total
    return out
