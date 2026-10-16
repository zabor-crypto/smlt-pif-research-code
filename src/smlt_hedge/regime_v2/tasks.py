"""Regime V2 — deterministic worker tasks (cutoff decisions, state library, scenario scorecard, replay ledgers).

Every task is an independent deterministic computation with fixed seeds; the pool only changes wall-clock time. Workers
load the same inputs as the parent process and refuse to run if any `smlt_hedge` module was imported from outside the
project's `src` directory (import guard).
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..adaptive import events as EV
from ..adaptive import replay as RP
from . import cfg as C2
from . import frontier as FR
from . import policy as PO
from . import replay as RP2

_G: dict = {}


def smoke() -> dict | None:
    """Development smoke mode (REGIME_V2R1_SMOKE=1): tiny path counts and a short replay window, written to a scratch
    directory. It checks that the code runs; its numbers are not research outputs and are never read."""
    if os.environ.get("REGIME_V2R1_SMOKE") != "1":
        return None
    return {"paths": int(os.environ.get("REGIME_V2R1_SMOKE_PATHS", "120")), "replay_start": os.environ.get("REGIME_V2R1_SMOKE_START", "2026-06-01"),
            "boot": 200}


def paths(cfg: dict, which: str) -> dict:
    s = smoke()
    if s:
        return {"n_generation_per_model": s["paths"], "n_admission_per_model": s["paths"]}
    return dict(cfg["v2"]["frontier"]["paths"][which])


def boot(cfg: dict, which: str) -> int:
    s = smoke()
    return s["boot"] if s else int(cfg["v2"]["returns"]["state_adjustment"]["bootstrap_draws"][which])


def replay_start(cfg: dict) -> str:
    s = smoke()
    return s["replay_start"] if s else cfg["v2"]["replay"]["start"]


def make_ctx(root):
    from .. import pipeline as pl
    ctx = pl.Ctx(Path(root))
    od = os.environ.get("REGIME_V2R1_OUTDIR")
    if od:
        ctx.out, ctx.fig = Path(od) / "outputs", Path(od) / "figures"
        ctx.out.mkdir(parents=True, exist_ok=True)
        ctx.fig.mkdir(parents=True, exist_ok=True)
    return ctx


def import_guard(root) -> None:
    src = os.path.realpath(os.path.join(str(root), "src"))
    bad = [m.__name__ for m in list(sys.modules.values()) if getattr(m, "__name__", "").startswith("smlt_hedge")
           and getattr(m, "__file__", None) and not os.path.realpath(m.__file__).startswith(src)]
    if bad:
        raise RuntimeError(f"import guard: smlt_hedge modules loaded from outside {src}: {bad}")


def init(root: str) -> None:
    warnings.simplefilter("error", RuntimeWarning)
    from ..adaptive import routes as RT
    ctx = make_ctx(root)
    cfg = C2.load(root)
    P = RT.build_panel(ctx, cfg)
    import_guard(root)
    _G.clear()
    _G.update({"ctx": ctx, "cfg": cfg, "P": P, "M": PO.Model(ctx, cfg, P), "worlds": {}, "rc": None})


def model() -> PO.Model:
    return _G["M"]


def cutoff_recal():
    M = model()
    if _G["rc"] is None:
        _G["rc"] = M.recal(pd.Timestamp(M.cfg["cutoff"]), boot(M.cfg, "cutoff"), extreme=True)
    return _G["rc"]


def world(universe: str, cost_mult: float) -> RP.World:
    key = (universe, cost_mult)
    if key not in _G["worlds"]:
        M = model()
        _G["worlds"][key] = RP.world(M.ctx, M.cfg, M.P, universe, replay_start(M.cfg), cost_mult)
    return _G["worlds"][key]


# ---------------------------------------------------------------------------------------------- reporting helpers
def destination(d: PO.Decision) -> str:
    if d.target_w is None:
        return "NO_TARGET"
    funded = {r: float(d.target_w.get(r,0)) for r in FR.FUNDED}
    liquid = 1 - sum(funded.values())
    return "; ".join(f"{r} {v:.3f}" for r,v in funded.items() if abs(v)>1e-9) + f"; NON_FUNDED_NAV_REMAINDER_ARITHMETIC {liquid:.3f} (post-trade allocation / NAV0; cash is fungible; no causal SMLT-proceeds attribution)"


def capital_flows(d: PO.Decision, label: dict) -> list:
    """Reference-price sources and uses, with pending settlement preserved rather than spent twice."""
    su = d.su
    ex = d.ex
    if ex is None:
        return [{**label,"item":"NO_TRADE","side":"NONE","rub":0.,"identity_residual_rub":0.}]
    q0 = np.asarray(su.book0.qf,float)
    dq = ex["qf"] - q0
    P0 = np.array([f.P0 for f in su.fspecs])
    cost_f = sum(abs(dq[j])*(f.P0*f.cost_prop+f.cost_unit) for j,f in enumerate(su.fspecs))
    # Includes all hedge closures and any reopening, rather than net derivative turnover only.
    nav_cost = su.nav0 - ex["execution_nav"]
    rows = []
    def add(item, side, amount, timing="REFERENCE_POST_EXECUTION"):
        if abs(amount)>1e-9:rows.append({**label,"item":item,"side":side,"rub":float(amount),"timing":timing})
    add("BEGINNING_IMMEDIATE_CASH","SOURCE",float(su.book0.cash),"BEGINNING")
    for due,amount in su.book0.pending.items():
        add("BEGINNING_PENDING_SETTLEMENT","SOURCE" if amount>0 else "USE",abs(amount),f"DUE_SESSION_{due}")
    for j,f in enumerate(su.fspecs):
        if dq[j]<0:add("MMF_REDEMPTION" if f.kind=="MMF" else f"SALE_{f.name}","SOURCE",-dq[j]*P0[j],"SETTLEMENT_RECEIVABLE_T_PLUS_1")
        elif dq[j]>0:add("MMF_ACQUISITION" if f.kind=="MMF" else f"PURCHASE_{f.name}","USE",dq[j]*P0[j],"SETTLEMENT_PAYABLE")
    add("TRANSACTION_COSTS","USE",nav_cost)
    add("POSTED_DERIVATIVE_MARGIN","USE",ex["execution_margin"])
    add("ENDING_IMMEDIATE_RESERVE","USE",ex["execution_cash"]-ex["execution_margin"])
    add("ENDING_NET_PENDING_SETTLEMENT","USE",ex["execution_pending"],"NOT_IMMEDIATE_CASH")
    residual = sum(r["rub"]*(1 if r["side"]=="SOURCE" else -1) for r in rows)
    if abs(residual)>max(1e-6,1e-10*su.nav0):
        raise RuntimeError(f"sources-and-uses identity failed: {residual}")
    for r in rows:r["identity_residual_rub"]=residual
    return rows



def clearing_capital_flows(W, res, policy: str) -> list:
    """Actual execution-price sources/uses for each rebalance/restoration clearing; no causal attribution."""
    rows=[]
    for record in res.get("clearing_records",[]):
        before=record["before"];P=record["P"];dq=record["qf_after"]-before["qf"]
        label={"policy":policy,"execution_date":str(W.dates[record["day"]].date()),"basis":"ACTUAL_CLEARING",
               "tags":";".join(record["tags"]),"vm_received_before_orders_rub":record["VM_received_before_orders"]}
        part=[]
        def add(item,side,value,timing="AT_ORDER_CLEARING"):
            if value<0: side="USE" if side=="SOURCE" else "SOURCE";value=-value
            if abs(value)>1e-9:part.append({**label,"item":item,"side":side,"rub":float(value),"timing":timing})
        add("BEGINNING_FREE_IMMEDIATE_CASH","SOURCE",before["cash"]-before["margin"])
        add("BEGINNING_POSTED_MARGIN","SOURCE",before["margin"])
        add("BEGINNING_NET_PENDING_SETTLEMENT","SOURCE",before["pending"],"NOT_IMMEDIATE_CASH")
        for j,i in enumerate(W.fin):
            if dq[j]<0:add("MMF_REDEMPTION" if i.kind=="MMF" else "SALE_"+i.name,"SOURCE",-dq[j]*P[j],"AS_OF_SETTLEMENT_LAG")
            elif dq[j]>0:add("MMF_ACQUISITION" if i.kind=="MMF" else "PURCHASE_"+i.name,"USE",dq[j]*P[j],"AS_OF_SETTLEMENT_LAG")
        add("TRANSACTION_COSTS","USE",record["trade_costs"])
        add("ENDING_POSTED_MARGIN","USE",record["margin_after"])
        add("ENDING_FREE_IMMEDIATE_CASH","USE",record["cash_after"]-record["margin_after"])
        add("ENDING_NET_PENDING_SETTLEMENT","USE",record["pending_after"],"NOT_IMMEDIATE_CASH")
        residual=sum(r["rub"]*(1 if r["side"]=="SOURCE" else -1) for r in part)
        nav_scale=abs(float((before["qf"]*P).sum()+before["cash"]+before["pending"]))
        if abs(residual)>max(1e-6,1e-10*nav_scale):raise RuntimeError(f"actual clearing sources/uses failed: {residual}")
        for r in part:r["identity_residual_rub"]=residual
        rows.extend(part)
    return rows


def cand_row(d: PO.Decision, c: dict, label: dict) -> dict:
    su = d.su
    if c.get("lk_rejected"):
        return {**label, "candidate_id": c["id"], "source": c["source"], "es_target_nav0": c["es_target"],
                "ADMISSION": "FAIL", "fail_reasons": " | ".join(c["reasons"]),
                "admission_status": c["admission_status"], "selected_target": False,
                "LK_REPORTS": c["lk_reports"], "ECONOMICS": "NOT_EVALUATED_LK_REJECTED"}
    m = FR.summarise(su, c["ex"])
    det = [r for r in c["ex"]["det"] if r["margin_multiplier"] == 1.0]
    fs = d.spec.mandate["floor_set"]
    return {**label, "candidate_id": c["id"], "source": c["source"], "es_target_nav0": c["es_target"],
            "ADMISSION": "PASS" if c["passed"] else "FAIL", "fail_reasons": " | ".join(c["reasons"]),
            "admission_status": c["admission_status"], "LK_REPORTS": c["ex"].get("lk_reports", []),
            "selected_target": bool(d.target is not None and c["id"] == d.target["id"]),
            "expected_net_excess_nav0": c["expected_net"], "engine_ES95_worst_nav0": c["es95"], "engine_ES99_worst_nav0": c["es99"],
            "turnover_cost_nav0": c["turnover_cost"], "funded_non_mmf_nav": c["ex"]["funded_fraction_at_execution"],
            "smlt_net_issuer_nav": c["ex"]["net_issuer"], "net_smlt_issuer_units": c["ex"]["net_issuer_units"],
            **{f"w_exec_{r}": v for r, v in c["ex"]["w_exec"].items()},
            "p_breach_worst": m["p_breach_worst"], "p_unresolved_worst": m["p_unresolved_worst"],
            "p_unresolved_stress_worst": m["p_unresolved_stress_worst"], "min_available_nav_p05_worst": m["min_available_nav_p05_worst"],
            "peak_im_nav_p95_worst": m["peak_im_nav_p95_worst"], "risk_path_mean_excess": m["risk_path_mean_excess"],
            "mdd_p95_worst": m["mdd_p95_worst"], "two_step_order": m["two_step_order"],
            "det_paths_unresolved": int(sum(r["unresolved_deficit"] for r in c["ex"]["det"])),
            "det_worst_pnl_nav0": min(r["pnl_nav0"] for r in det), "det_worst_path": min(det, key=lambda r: r["pnl_nav0"])["path"],
            "risk_models": ";".join(su.v2["models"]), "state_model_level": su.v2["state_model_level"]}


def current_row(d: PO.Decision, label: dict) -> dict:
    su = d.su
    ex = d.current_ex
    det = [r for r in ex["det"] if r["margin_multiplier"] == 1.0]
    m = d.current_metrics
    fs = d.spec.mandate["floor_set"]
    return {**label, "candidate_id": "CURRENT_BOOK_NO_TRADE", "source": "CURRENT_BOOK", "LK_REPORTS": ex.get("lk_reports", []), "es_target_nav0": None,
            "ADMISSION": "PASS" if d.current_admitted else "FAIL", "fail_reasons": " | ".join(d.current_reasons),
            "selected_target": d.action == "HOLD", "expected_net_excess_nav0": d.en_current, "engine_ES95_worst_nav0": d.es_current,
            "engine_ES99_worst_nav0": FR.es_of(su, ex, "ES99"), "turnover_cost_nav0": 0.0,
            "funded_non_mmf_nav": ex["funded_fraction_at_execution"],
            "smlt_net_issuer_nav": ex["net_issuer"], "net_smlt_issuer_units": ex["net_issuer_units"],
            **{f"w_exec_{r}": v for r, v in d.current_w.items()},
            "p_breach_worst": m["p_breach_worst"], "p_unresolved_worst": m["p_unresolved_worst"],
            "p_unresolved_stress_worst": m["p_unresolved_stress_worst"], "min_available_nav_p05_worst": m["min_available_nav_p05_worst"],
            "peak_im_nav_p95_worst": m["peak_im_nav_p95_worst"], "risk_path_mean_excess": m["risk_path_mean_excess"],
            "mdd_p95_worst": m["mdd_p95_worst"], "two_step_order": False,
            "det_paths_unresolved": int(sum(r["unresolved_deficit"] for r in ex["det"])),
            "det_worst_pnl_nav0": min(r["pnl_nav0"] for r in det), "det_worst_path": min(det, key=lambda r: r["pnl_nav0"])["path"],
            "risk_models": ";".join(su.v2["models"]), "state_model_level": su.v2["state_model_level"]}


def decision_summary(d: PO.Decision, label: dict) -> dict:
    cfg = d.su.cfg
    tw = d.target_w or {}
    sl_t, sl_c = PO.sleeve_split(cfg, tw), PO.sleeve_split(cfg, d.current_w)
    m = d.metrics
    rows1 = [r for r in (d.ex["rows"] if d.ex is not None else d.current_ex["rows"]) if r["margin_multiplier"] == 1.0]
    row = PO.log_row(d, label)
    row["risk_path_median_excess_worst_model"] = min(r["median_excess_pnl"] for r in rows1) / d.su.nav0
    row.update({"date": str(d.date.date()), "released_smlt_capital_goes_to": destination(d),
                **{f"sleeve_target_{k}": v for k, v in sl_t.items()}, **{f"sleeve_current_{k}": v for k, v in sl_c.items()},
                "NON_FUNDED_NAV_REMAINDER_ARITHMETIC": 1.0 - sum(v for r, v in tw.items() if r in FR.FUNDED),
                "smlt_net_issuer_target": (d.ex if d.ex is not None else d.current_ex)["net_issuer"],
                "engine_ES99_target_nav0": m.get("engine_ES99_worst_nav0"), "p_breach_target": m.get("p_breach_worst"),
                "p_unresolved_target": m.get("p_unresolved_worst"), "p_unresolved_stress_target": m.get("p_unresolved_stress_worst"),
                "min_available_nav_p05_target": m.get("min_available_nav_p05_worst"), "reserve_min_nav_target": m.get("reserve_min_nav"),
                "peak_im_nav_p95_target": m.get("peak_im_nav_p95_worst"), "risk_path_mean_excess": m.get("risk_path_mean_excess"),
                "turnover_cost_nav0": (d.target or {}).get("turnover_cost") if d.ex is not None else 0.0,
                "turnover_gross_nav0": float(sum(abs(tw.get(r, 0.0) - d.current_w.get(r, 0.0)) for r in d.su.routes)) if d.ex is not None else 0.0,
                "target_candidate_id": (d.target or {}).get("id"), "target_source": (d.target or {}).get("source")})
    return row


def target_table(d: PO.Decision, label: dict) -> list:
    su = d.su
    cfg = su.cfg
    sleeve = {r: k for k, v in cfg["v2"]["sleeves"].items() for r in v}
    P0 = {f.name: f.P0 for f in su.fspecs}
    rows = []
    for n in list(su.fin_names) + list(su.din_names):
        unit = P0[n] if n in P0 else su.notional[n]
        cq = float(su.book0.qf[su.fin_names.index(n)]) if n in P0 else float(su.book0.qd[su.din_names.index(n)])
        if d.ex is None:
            tq = cq
        elif n in P0:
            v = d.ex["qf"][su.fin_names.index(n)]
            tq = None if np.isnan(v) else float(v)
        else:
            tq = float(d.ex["qd"][su.din_names.index(n)])
        rows.append({**label, "instrument": n, "sleeve": sleeve.get(n, "MONEY_MARKET" if n == "MMF" else ""),
                     "kind": "FUNDED" if n in P0 else "DERIVATIVE", "unit_value_rub": unit, "current_quantity": cq,
                     "current_weight_nav": cq * unit / su.nav0, "target_quantity": "by the cash rule after settlement" if tq is None else tq,
                     "target_weight_nav": None if tq is None else tq * unit / su.nav0,
                     "expected_excess_per_unit_h": float(d.mu.loc[n, "mu"]) if n in d.mu.index else None, "action": d.action})
    cash = float(np.broadcast_to(su.book0.cash, (1,))[0])
    rows.append({**label, "instrument": "IMMEDIATE_RUB_CASH", "sleeve": "LIQUIDITY_RESERVE", "kind": "CASH", "unit_value_rub": 1.0,
                 "current_quantity": cash, "current_weight_nav": cash / su.nav0,
                 "target_quantity": float(d.ex["execution_cash"]) if d.ex is not None else cash,
                 "target_weight_nav": (float(d.ex["execution_cash"]) if d.ex is not None else cash) / su.nav0,
                 "action": d.action, "basis": "CASH_LEDGER_AT_REFERENCE_EXECUTION"})
    return rows


def order_rows(d: PO.Decision, label: dict, cal: pd.DatetimeIndex) -> list:
    su = d.su
    if d.ex is None:
        return [{**label, "order": "NONE", "note": f"{d.action}: no order"}]
    t = d.date
    nxt = pd.bdate_range(t + pd.Timedelta(days=1), periods=8)
    P0 = {f.name: f.P0 for f in su.fspecs}
    rows = []
    qf = np.asarray(su.book0.qf, float).copy()
    qd = np.asarray(su.book0.qd, float).copy()
    for k, o in enumerate(d.ex["orders"], 1):
        if o.qf is not None:
            tgt = np.asarray(o.qf, float)
            for j, n in enumerate(su.fin_names):
                if np.isnan(tgt[j]) or abs(tgt[j] - qf[j]) < 1e-9:
                    continue
                dq = tgt[j] - qf[j]
                rows.append({**label, "order": k, "execution_session": str(nxt[o.day - 1].date()), "execution_offset_sessions": o.day,
                             "instrument": n, "kind": "FUNDED", "side": "BUY" if dq > 0 else "SELL", "quantity": abs(dq),
                             "reference_price_rub": P0[n], "value_at_reference_rub": abs(dq) * P0[n],
                             "settlement": "T+0 (academic assumption)" if n == "BTC_RUB_EXPERIMENTAL" else "T+1",
                             "note": "executed at the session close; money-market units are bought later by the cash rule"})
                qf[j] = tgt[j]
        if o.qd is not None:
            tgt = np.asarray(o.qd, float)
            for j, n in enumerate(su.din_names):
                if abs(tgt[j] - qd[j]) < 1e-9:
                    continue
                dq = tgt[j] - qd[j]
                rows.append({**label, "order": k, "execution_session": str(nxt[o.day - 1].date()), "execution_offset_sessions": o.day,
                             "instrument": n, "kind": "DERIVATIVE", "side": "BUY" if dq > 0 else "SELL", "quantity": abs(dq),
                             "reference_price_rub": su.notional[n], "value_at_reference_rub": abs(dq) * su.notional[n],
                             "settlement": "daily variation margin; initial margin posted from immediate cash",
                             "note": "whole contracts" + ("; second step after the money-market sale settles" if k > 1 else "")})
                qd[j] = tgt[j]
    return rows or [{**label, "order": "NONE", "note": "target equals the current quantities"}]


def zero_weight(d: PO.Decision, label: dict) -> list:
    """Marginal entry against the FINAL admitted refined candidate, never a newly solved coarse LP."""
    su,spec = d.su,d.spec
    baseline = d.target
    if baseline is None or not baseline["passed"]:
        return [{**label,"route":r,"cause":"UNKNOWN","explanation":"No final admitted refined baseline exists"} for r in su.routes]
    rows=[]
    step=float(su.cfg["v2"]["frontier"]["zero_weight_forced_entry_nav0"])
    for j,r in enumerate(su.routes):
        w=float(baseline["x_exec"][j])
        row={**label,"route":r,"w_target":w,"baseline_candidate_id":baseline["id"],"baseline_admission":"PASS",
             "baseline_expected_net_nav0":baseline["expected_net"],"baseline_es95_nav0":baseline["es95"]}
        b=su.cfg["r3"]["bounds"][r]
        if abs(w)>1e-9:row["cause"]="HELD"
        elif b["lo"]==b["hi"]==0:row["cause"]="ROUTE_NOT_ADMISSIBLE"
        else:
            x=baseline["x_exec"].copy();x[j]+=step if b["hi"]>0 else -step
            c=FR.candidate(su,spec,x,"MARGINAL_ENTRY_AT_FINAL_ADMITTED_TARGET",None,None,f"marginal_{r}")
            if c.get("lk_rejected"):
                row.update({"cause": c["admission_status"], "marginal_entry_admission": "FAIL",
                            "marginal_entry_fail_reasons": " | ".join(c["reasons"]), "LK_REPORTS": c["lk_reports"],
                            "requested_entry_step_nav0": step, "ECONOMICS": "NOT_EVALUATED_LK_REJECTED"})
                rows.append(row)
                continue
            delta=c["expected_net"]-baseline["expected_net"]
            row.update({"marginal_entry_admission":"PASS" if c["passed"] else "FAIL","marginal_entry_fail_reasons":" | ".join(c["reasons"]),
                        "marginal_entry_expected_net_change_nav0":delta,"marginal_entry_es95_change_nav0":c["es95"]-baseline["es95"],
                        "requested_entry_step_nav0":step,"marginal_entry_executable_weight_nav0":float(c["x_exec"][j]) if "x_exec" in c else None,
                        "financing_method":"STANDARD_CASH_MMF_SETTLEMENT_AND_EXECUTABLE_SCALING"})
            reasons=" ".join(c["reasons"])
            no_entry = "x_exec" in c and abs(c["x_exec"][j]-w)<1e-12
            ex = c.get("ex", {})
            names = su.fin_names if r in FR.FUNDED else su.din_names
            nominal = ex.get("nominal_qf" if r in FR.FUNDED else "nominal_qd")
            rounded_to_zero = nominal is not None and abs(float(nominal[names.index(r)])) < 1e-12
            issuer_suppressed = (no_entry and r == "SMLT_FUT" and not rounded_to_zero and
                                 spec.mandate["overlay_rule"] and x[j] < 0 and
                                 "qf" in ex and "qd" in ex and
                                 ex["qf"][su.fin_names.index("SMLT")] +
                                 ex["qd"][su.din_names.index(r)] * next(d.pv0 for d in su.dspecs if d.name == r) <= 1e-9)
            if no_entry:
                row["guarded_book_admission"] = row["marginal_entry_admission"]
                row["marginal_entry_admission"] = "MARGINAL_ENTRY_NOT_EXECUTED"
            row["cause"]=("INTEGER_GRANULARITY" if no_entry and rounded_to_zero else
                          "MANDATE_BOUND" if issuer_suppressed else
                          "LIQUIDITY_BINDING" if no_entry else
                          "LIQUIDITY_BINDING" if "liquidity" in reasons or "deficit" in reasons else
                          "RISK_BUDGET_BINDING" if "ES95" in reasons else
                          "MANDATE_BOUND" if "hard constraint" in reasons or "MANDATE_FAIL" in reasons else
                          "RETURN_INFERIOR" if c["passed"] and delta<0 else
                          "UNKNOWN")
        rows.append(row)
    return rows


def risk_rows(d: PO.Decision, label: dict) -> list:
    su = d.su
    a = su.cfg["owner"]["es_alpha"]
    out = []
    for name, x in (("CURRENT_BOOK", su.x0), ("TARGET", np.array([(d.target_w or d.current_w).get(r, 0.0) for r in su.routes]))):
        rc = FR.risk_contribution(su, x, a)
        rc.insert(0, "book", name)
        for k, v in label.items():
            rc[k] = v
        out.extend(rc.to_dict("records"))
    return out


def det_rows(d: PO.Decision, label: dict) -> list:
    rows = [{**label, "book": "CURRENT_BOOK_NO_TRADE", **r, "probability": "NOT_ASSIGNED"} for r in d.current_ex["det"]]
    if d.ex is not None:
        rows += [{**label, "book": "TARGET", **r, "probability": "NOT_ASSIGNED"} for r in d.ex["det"]]
    return rows


# ---------------------------------------------------------------------------------------------- tasks
def _spec(kw: dict) -> PO.PolicySpec:
    return PO.PolicySpec(**kw)


def cutoff_task(task: dict) -> dict:
    """One decision at the cutoff from the ACTUAL book (initial portfolio), possibly under a hypothetical state."""
    M = model()
    cfg = M.cfg
    t = pd.Timestamp(cfg["cutoff"])
    rc = cutoff_recal()
    states = M.conf.loc[t].to_dict()
    ev = dict(EV.state_at(M.events, t))
    for a, v in (task.get("states") or {}).items():
        if v != "current":
            states[a] = v
    if task.get("severe"):
        ev["severe_restriction"] = True
    ps = _spec(task["ps"])
    npth = paths(cfg, task.get("paths", "cutoff"))
    d = PO.decide(M, rc, t, None, None, states, ev, ps, n_gen=npth["n_generation_per_model"], n_adm=npth["n_admission_per_model"],
                  first=True, capital=task.get("capital"), seed_offset=int(task.get("seed_offset", 0)))
    label = dict(task.get("label") or {})
    want = set(task.get("want") or [])
    out = {"label": label, "summary": decision_summary(d, label)}
    if "frontier" in want:
        out["frontier"] = [current_row(d, label)] + [cand_row(d, c, label) for c in d.candidates]
        out["refinement"] = [{**label, **r} for r in d.refinement_log]
        out["lp_infeasible"] = [{**label, **r} for r in d.lp_infeasible]
    if "target" in want:
        out["target"] = target_table(d, label)
        out["orders"] = order_rows(d, label, M.P["cal"])
        out["capital_flows"] = capital_flows(d, label)
        from . import reporting as REPORT
        out["binding_constraints"] = REPORT.binding_constraints(d.su, d.target_w or d.current_w, label)
        if d.ex is not None:
            out["target_balance_timeline"] = [{**label, **r} for r in REPORT.reference_balance_timeline(d.su, d.ex, d.spec)]
    if "zero" in want:
        out["zero"] = zero_weight(d, label)
    if "risk" in want:
        out["risk"] = risk_rows(d, label)
    if "det" in want:
        out["det"] = det_rows(d, label)
    if "mu" in want:
        mu = d.mu.reset_index()
        for k, v in label.items():
            mu[k] = v
        out["mu"] = mu.to_dict("records")
        out["carry_info"] = {k: (v if not isinstance(v, dict) else dict(v)) for k, v in d.carry["info"].items()}
        out["adj"] = [{**label, "route": r, "level_used": a["level"], "value_used": a["value"],
                       **{k: v for k, v in (a["stats"] or {}).items() if k not in ("route",)}} for r, a in d.adj.items()]
        out["state_model"] = d.state_model
    if "hedge" in want:
        from . import hedge as HG
        hs = FR.Spec(mandate=C2.mandate(cfg, "CAPITAL_PRESERVATION_UNCONSTRAINED"), budget=ps.budget, cap_net=10.0, delay=ps.delay)
        out["hedge"] = HG.table(M.P, d.su, M.conf.loc[:t], states, cfg, hs, cfg["v2"]["state"]["sample_start"], t).to_dict("records")
    if "mc" in want:
        out["mc"] = mc_rows(M, rc, t, states, ev, ps, d, npth, task.get("mc_offsets") or [])
    return out


def mc_rows(M, rc, t, states, ev, ps, d, npth, offsets) -> list:
    """Monte-Carlo error: the same two books re-evaluated on independent seed draws of every model."""
    from . import risk as RK
    cfg = M.cfg
    rows = []
    xt = np.array([(d.target_w or d.current_w).get(r, 0.0) for r in d.su.routes])
    choice = RK.state_model_choice(rc.panel.index, M.conf.loc[:rc.t], states, cfg)
    for off in offsets:
        su = d.su if int(off) == 0 else FR.build(M.ctx, cfg, M.P, t, ps.universe, d.su.views, n_gen=npth["n_generation_per_model"],
                                                   n_adm=npth["n_admission_per_model"], models=tuple(d.su.v2["models"]),
                                                   state_choice=choice, seed_offset=int(off), panel_end=rc.t, px=M.px)
        for name, x, tr in (("CURRENT_BOOK_NO_TRADE", su.x0, False), ("TARGET", xt, True)):
            ex = FR.admit(su, x, d.spec, None) if tr else FR.current_eval(su, d.spec)
            for r in ex["rows"]:
                if r["margin_multiplier"] == 1.0:
                    rows.append({"book": name, "model": r["model"], "seed_offset": int(off), "ES95_nav0": r["ES95"] / su.nav0,
                                 "p_breach": r["p_breach"]})
    return rows


def _replay_task(spec: dict, journal) -> dict:
    """One ledger of the chronological replay."""
    M = model()
    cfg = M.cfg
    nav0 = float(cfg["v2"]["replay"]["nav0_rub"])
    typ = spec["type"]
    if typ == "fixed":
        W = world("MAIN", 1.0)
        out = RP2.run_fixed(M.ctx, cfg, M.P, W, spec["target"], reset=spec.get("reset", False),journal=journal)
    elif typ == "money_market":
        W = world("MAIN", 1.0)
        out = RP2.run_money_market(cfg, W,journal=journal)
    else:
        kind = spec["kind"]
        uni = "HEDGE_ONLY" if kind == "HEDGE_KEEP" else spec.get("universe", "MAIN")
        ps = PO.PolicySpec(name=spec["name"], kind=kind, mandate=spec.get("mandate", "HEDGED_INVESTED_PIF"),
                           budget=float(cfg["v2"]["replay"]["es_budget_nav"]), mode=spec.get("mode", "BASE_POSTERIOR_MEAN"),
                           universe=uni, delay=int(spec.get("delay", 1)), cost_mult=float(spec.get("cost_mult", 1.0)),
                           keep_frac=float(spec.get("keep_frac", 1.0)))
        W = world(uni, ps.cost_mult)
        out = RP2.run_policy(M, W, ps, paths(cfg, "replay"), boot(cfg, "replay"),journal=journal)
    mmf = W.px["MMF"]
    lg = RP.ledger_frame(W, out["res"])
    if spec.get("mandate") != "BROAD_NET_SHORT_ALLOWED" and (out["res"]["min_net_smlt_units"] < -1e-9).any():
        raise RuntimeError("primary replay contains negative canonical SMLT issuer units")
    keep = ["nav", "cash", "pending", "im", "available", "breach", "unresolved", "vm", "roll_cost", "funded_non_mmf_fraction", "emergency_invested_floor_override", "net_smlt_issuer_units"] + [c for c in lg if c[:2] in ("q_", "v_", "n_", "c_")]
    return {"name": spec["name"], "spec": {k: v for k, v in spec.items() if k != "target"},
            "metrics": {**RP.metrics(W, out["res"], spec["name"], mmf, nav0),
                        "emergency_floor_override_count":int(out["res"]["emergency_floor_override_count"].sum()),
                        "min_net_smlt_issuer_units":float(out["res"]["min_net_smlt_units"].min())}, "attribution": RP.attribution(W, out["res"], spec["name"], mmf, nav0),
            "decisions": out["decisions"], "capital_flows":clearing_capital_flows(W,out["res"],spec["name"]), "refinements": out["refinements"], "ledger": lg[keep], "dates": W.dates,
            "mmf": mmf.reindex(W.dates)}


def replay_task(spec: dict) -> dict:
    from .checkpoints import Journal,identity
    ctx=_G["ctx"]
    journal=Journal(ctx.out/"regime_v2r1_checkpoints",spec["name"],identity(ctx.root,_G["cfg"]))
    try:
        result=_replay_task(spec,journal)
        journal.finish(result)
        return result
    except Exception as exc:
        if not (journal.root/"failure.json").exists():
            journal.fail({"last_completed_close":journal.last},exc)
        raise


def run_pool(root: str, fn, tasks: list, processes: int) -> list:
    if processes <= 1 or len(tasks) <= 1:
        if not _G:
            init(root)
        return [fn(t) for t in tasks]
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(min(processes, len(tasks)), initializer=init, initargs=(root,)) as pool:
        return pool.map(fn, tasks, chunksize=1)
