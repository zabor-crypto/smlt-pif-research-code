"""Regime V2 — chronological one-ledger replay (accepted R1 engine and execution mechanics, unchanged).

One self-financing quantity ledger per policy, 2023-01-03 .. cutoff. At each close the engine settles, marks, posts
margin at the as-of rate and only then calls the policy with the state observed at that close; orders execute at the
close of session t + delay. Recalibration (bootstrap sample, SMLT beta, forward state statistics, eligible block starts)
at the first decision close of January and July; weekly decisions update the observable inputs (prices, contract terms,
current volatility, money-market rate, carry, confirmed state, events).

ADAPTIVE   weekly decisions; an immediate decision when a verified severe event becomes active.
STATIC     same initial state/return/risk authority; later weekly checks permit mandatory restoration only.
HEDGE_KEEP weekly decisions on protective overlays only; the shares are a fixed quantity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..adaptive import engine as E
from ..adaptive import events as EV
from ..adaptive import replay as RP
from . import policy as PO
from . import cfg as C2


def initial_book(cfg: dict, W: RP.World) -> E.Book:
    r = cfg["v2"]["replay"]
    S0 = float(W.px["SMLT"].loc[W.dates[0]])
    shares = np.floor(r["smlt_initial_value_rub"] / S0)
    return E.Book(qf=np.array([shares if n == "SMLT" else 0.0 for n in W.fin_names]), qd=np.zeros(len(W.din_names)),
                  cash=r["nav0_rub"] - shares * S0)


def _engine(cfg: dict, W: RP.World) -> E.Engine:
    beta = cfg["owner"]["liquidity_min_nav"]
    return E.Engine(W.fin, W.din, W.mkt, E.Params(beta=beta, reserve_frac=beta, hysteresis=cfg["r1"]["liquidity"]["sweep_hysteresis_nav"], log=True,restoration_telemetry=True))


def run_policy(M: PO.Model, W: RP.World, ps: PO.PolicySpec, n_paths: dict | None = None, n_boot: int | None = None, journal=None) -> dict:
    cfg = M.cfg
    dates = W.dates
    T = len(dates)
    npth = n_paths or cfg["v2"]["frontier"]["paths"]["replay"]
    n_boot = n_boot or int(cfg["v2"]["returns"]["state_adjustment"]["bootstrap_draws"]["replay"])
    dec = RP.decision_days(dates)
    rec = RP.recal_days(dates, cfg["v2"]["policy"]["recalibration_months"], dec)
    cache = {"rc": None}
    decisions, refinements, pending_orders = [], [], []
    Ev = M.events

    def execution_session(day):
        if day < T:
            return dates[day]
        ctx = getattr(M, 'ctx', None)
        if getattr(ctx, '_regime_v2r2_combined', False):
            from .session_calendar import forecast
            return forecast(ctx.root, dates[-1], day - T + 1).dates[-1]
        # Legacy reference for models without combined execution authority.
        return pd.bdate_range(dates[-1], periods=day - T + 2)[-1]

    def policy(i, st):
        t = dates[i]
        ev = EV.state_at(Ev, t)
        sev_new = ev["severe_restriction"] and not (i > 0 and EV.state_at(Ev, dates[i - 1])["severe_restriction"])
        if not (dec[i] or sev_new):
            return {}
        if rec[i] or cache["rc"] is None:
            cache["rc"] = M.recal(t, n_boot, extreme=(ps.mode == "EXTREME_ROBUST"))
        rc = cache["rc"]
        states = M.conf.loc[t].to_dict()
        pend = {int(k): float(v) for k, v in enumerate(st["pend"][0, i + 1:], start=1) if abs(v) > 1e-9}
        book = E.Book(qf=st["qf"][0].copy(), qd=st["qd"][0].copy(), cash=float(st["cash"][0]), pending=pend,
                      series=st.get("series"), capacity=st.get("capacity"),regime=st.get("regime",["ORDINARY"])[0])
        try:
            d = PO.decide(M, rc, t, book, float(st["available"][0]), states, ev, ps, n_gen=npth["n_generation_per_model"],
                          n_adm=npth["n_admission_per_model"], first=(i == 0))
        except Exception as exc:
            if journal is not None:
                journal.fail({"decision_date":str(t),"execution_date":str(execution_session(i + ps.delay)),"book":book,
                              "engine_state":st,"observed_state":states,"event_state":ev,
                              "admission_context":getattr(exc,"admission_context",None)},exc)
            raise
        decisions.append(PO.log_row(d, {"decision_index": i, "recalibration": bool(rec[i]), "recalibration_date": str(rc.t.date()),
                                        "execution_date": str(execution_session(i + ps.delay).date()),
                                        "available_nav": float(st["available"][0] / st["nav"][0])}))
        for r in d.refinement_log:
            refinements.append({"policy": ps.name, "date": t, **r})
        if d.ex is None:
            return {}
        ords = [E.Order(o.day + i, o.qf, o.qd, "EMERGENCY_TARGET" if d.action == "EMERGENCY_CAPITAL_PRESERVATION" else o.tag, o.delta, o.plan, o.stage_sequence, i, o.stage_check) for o in d.ex["orders"]]      # relative -> absolute day
        for o in ords:
            if o.day >= T:
                pending_orders.append({"decision_date": str(t.date()), "execution_delay_sessions": ps.delay,
                                       "execution_offset_sessions": o.day - i,
                                       "intended_execution_session": str(execution_session(o.day).date()),
                                       "policy": ps.name, "action": d.action, "target": d.target_w,
                                       "admission_status": decisions[-1]["admission_status"],
                                       "qf": None if o.qf is None else np.where(np.isnan(o.qf), None, o.qf).tolist(),
                                       "qd": None if o.qd is None else np.where(np.isnan(o.qd), None, o.qd).tolist(),
                                       "fin_names": W.fin_names, "din_names": W.din_names,
                                       "tag": o.tag, "delta": o.delta,
                                       "status": "PENDING_POST_REPLAY_EXECUTION",
                                       "reason": "PENDING_POST_REPLAY_EXECUTION"})
        ords = [o for o in ords if o.day <= T - 1]
        if i + ps.delay >= T:
            # A post-terminal plan cannot update the historical engine's reserve/ranks.
            return {}
        return {"orders": ords, "reserve": d.ex["reserve_min"], "wf_rank": d.ex["wf_rank"], "restore_rank": d.ex["restore_rank"]}
    def checkpoint(i, st):
        if journal is not None:
            journal.close(i,st,{"completed_date":str(dates[i].date()),"confirmed_state":M.conf.loc[dates[i]].to_dict(),
                               "decision_count":len(decisions),"decisions":decisions,"refinements":refinements,
                               "pending_post_replay_orders":pending_orders,
                               "recalibration_date":str(cache["rc"].t) if cache["rc"] is not None else None})
    eng=_engine(cfg,W)
    eng.p.issuer_floor=ps.mandate != "BROAD_NET_SHORT_ALLOWED"
    eng.p.funded_floor=C2.mandate(cfg,ps.mandate,ps.floor)["floor_hard"] if ps.kind!="HEDGE_KEEP" else 0.0
    res = eng.run(initial_book(cfg, W), policy=policy,checkpoint=checkpoint)
    return {"res": res, "decisions": pd.DataFrame(decisions), "refinements": pd.DataFrame(refinements),
            "pending_post_replay_orders": pd.DataFrame(pending_orders)}


def run_money_market(cfg: dict, W: RP.World, journal=None) -> dict:
    """OPERATIONAL_MONEY_MARKET: NAV0 in cash at the first close, no risky asset ever; money-market units are bought by
    the same cash rule (5% hard minimum, operating band, T+lag settlement, zero-yield immediate cash)."""
    book0 = E.Book(qf=np.zeros(len(W.fin_names)), qd=np.zeros(len(W.din_names)), cash=float(cfg["v2"]["replay"]["nav0_rub"]))
    return {"res": _engine(cfg, W).run(book0,checkpoint=(lambda i,st:journal.close(i,st,{"completed_date":str(W.dates[i].date())})) if journal else None), "decisions": pd.DataFrame(), "refinements": pd.DataFrame()}


def run_fixed(ctx, cfg: dict, P: dict, W: RP.World, target: dict, reset: bool = False, journal=None) -> dict:
    """Legacy fixed-target comparators through the parent's replay function (initial book and engine unchanged)."""
    c = dict(cfg)
    c["r6"] = {**cfg["r6"], "nav0_rub": cfg["v2"]["replay"]["nav0_rub"], "smlt_initial_value_rub": cfg["v2"]["replay"]["smlt_initial_value_rub"]}
    c["r5"] = {**cfg["r5"], "recalibration": {**cfg["r5"]["recalibration"], "months": cfg["v2"]["policy"]["recalibration_months"]}}
    out = RP.run_fixed(ctx, c, P, W, target, reset_at_recal=reset,issuer_floor=True,
                       checkpoint=(lambda i,st:journal.close(i,st,{"completed_date":str(W.dates[i].date())})) if journal else None)
    return {**out, "refinements": pd.DataFrame()}


def sleeve_path(cfg: dict, lg: pd.DataFrame) -> pd.DataFrame:
    """Daily NAV weights by sleeve: funded allocation, directional macro derivatives, protective overlays, money market."""
    nav = lg["nav"]
    out = pd.DataFrame(index=lg.index)
    for r in [c[2:] for c in lg if c.startswith("v_") or c.startswith("n_")]:
        col = f"v_{r}" if f"v_{r}" in lg else f"n_{r}"
        out[r] = lg[col] / nav
    for k, v in cfg["v2"]["sleeves"].items():
        out[f"sleeve_{k}"] = out[[r for r in v if r in out]].sum(axis=1)
    out["immediate_cash"] = lg["cash"] / nav
    out["smlt_net_issuer"] = lg["net_smlt_issuer_units"] * lg["smlt_issuer_spot_price"] / nav
    return out
