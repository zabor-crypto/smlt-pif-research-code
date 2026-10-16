"""Gate R5 — ADAPTIVE_PIF_V1: one state-aware, cap-based, asymmetric, low-turnover, one-ledger policy.

STATE -> caps / admissible directions / view set.  Never STATE -> mandatory target weight.

At an ordinary (weekly) decision close t, with information through t:
  1. the CURRENT actual book (quantities, cash, MMF units, pending settlements, margin) is evaluated by the full engine
     under the confirmed state's caps;
  2. the robust frontier for the same budget and state is generated (LP) and every candidate is engine-admitted;
  3. HOLD iff the current book is admitted AND its distance to the nearest admitted candidate is <= 5% NAV
     (max absolute weight deviation) — a no-trade region made only of jointly admissible portfolios at one budget / state;
  4. otherwise trade to the NEAREST admitted candidate; orders execute at the NEXT session in integer quantities;
  5. no admitted candidate -> NO_ADMISSIBLE_TARGET (the book is left unchanged and flagged).
Caps are ceilings: a cap above the current holding never creates a purchase. A verified severe event sets the issuer
cap to 0 from the next session without confirmation. Ordinary deterioration needs 3 consecutive closes; any
relaxation needs 10 consecutive closes and no active severe restriction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import engine as E
from . import frontier as F
from . import state as ST


def confirm(raw: pd.Series, order: list, n_det: int, n_re: int, blocked: pd.Series | None = None) -> pd.Series:
    """Asymmetric confirmation. Worse (higher index in `order`) after n_det consecutive closes at or above that level;
    better only after n_re consecutive closes at or below the better level and only when not `blocked`."""
    sev = raw.map({s: i for i, s in enumerate(order)}).astype(int).values
    out = np.empty(len(sev), int)
    cur = sev[0]
    for i in range(len(sev)):
        if i + 1 >= n_det:
            lo = sev[i + 1 - n_det:i + 1].min()
            if lo > cur:
                cur = lo
        if i + 1 >= n_re and not (blocked is not None and bool(blocked.iloc[i])):
            hi = sev[i + 1 - n_re:i + 1].max()
            if hi < cur:
                cur = hi
        out[i] = cur
    return pd.Series([order[k] for k in out], index=raw.index)


def confirmed_states(raw: pd.DataFrame, cfg: dict, severe: pd.Series | None = None) -> pd.DataFrame:
    c = cfg["r5"]["confirmation"]
    return pd.DataFrame({d: confirm(raw[d], ST.ORDER[d], c["deterioration_consecutive_closes"],
                                    c["rerisk_normalisation_consecutive_closes"], severe) for d in ST.DIMENSIONS})


def issuer_cap(cfg: dict, issuer_confirmed: str, severe_restriction: bool) -> float:
    caps = cfg["r5"]["issuer_caps_net_long_nav"]
    return caps["SEVERE_VERIFIED_EVENT"] if severe_restriction else caps[issuer_confirmed]


@dataclass
class Decision:
    date: pd.Timestamp
    action: str
    reasons: list
    current_admitted: bool
    current_reasons: list
    cap: float
    states: dict
    event_state: dict
    target_w: dict | None = None
    current_w: dict = field(default_factory=dict)
    ex: dict | None = None                  # executable target (orders, reserve, ranks)
    candidates: list = field(default_factory=list)
    distance: float | None = None
    metrics: dict = field(default_factory=dict)
    current_metrics: dict = field(default_factory=dict)
    su: object = None


def current_exposure(su: F.Setup) -> dict:
    w = {r: float(su.x0[j]) for j, r in enumerate(su.routes)}
    return w


def _current_eval(su: F.Setup, cfg: dict) -> dict:
    """Engine evaluation of the book AS IT IS (no order)."""
    ex = F.executable(su, su.x0, 1)
    ex["orders"] = []
    rule = F.A.Rule(budget=1.0, beta=cfg["owner"]["liquidity_min_nav"], max_breach=cfg["r1"]["admission"]["max_breach_probability"],
                    stress_mults=tuple(cfg["r1"]["admission"]["margin_stress_survival"]),
                    hysteresis=cfg["r1"]["liquidity"]["sweep_hysteresis_nav"])
    ex["rows"] = F.A.evaluate_all(su.adm, su.book0, [], rule, ex["reserve_min"], ex["wf_rank"], ex["restore_rank"])
    ex["w_exec"] = current_exposure(su)
    b = cfg["r3"]["bounds"]
    md = cfg["r3"]["mandates"][su.mandate]
    net = ex["w_exec"].get("SMLT", 0.0) + ex["w_exec"].get("SMLT_FUT", 0.0)
    cap = md["issuer_net_max"] if su.issuer_cap is None else min(md["issuer_net_max"], su.issuer_cap)
    long_exp = sum(v for r, v in ex["w_exec"].items() if r in F.FUNDED) + sum(max(v, 0) for r, v in ex["w_exec"].items() if r not in F.FUNDED)
    gross = sum(abs(v) for r, v in ex["w_exec"].items() if r not in F.FUNDED)
    tol = 1e-9
    ex["hard"] = {"issuer_cap_and_floor": md["issuer_net_min"] - tol <= net <= cap + tol,
                  "long_exposure<=1": long_exp <= 1 + tol, "derivative_gross<=1": gross <= 1 + tol,
                  "route_lower_bounds": all(v >= b[r]["lo"] - tol for r, v in ex["w_exec"].items())}
    ex["net_issuer"] = net
    return ex


def label(cur: dict, tgt: dict, tol: float = 0.01) -> str:
    ci = cur.get("SMLT", 0.0) + cur.get("SMLT_FUT", 0.0)
    ti = tgt.get("SMLT", 0.0) + tgt.get("SMLT_FUT", 0.0)
    others_changed = any(abs(tgt.get(r, 0.0) - cur.get(r, 0.0)) > tol for r in tgt if r not in ("SMLT", "SMLT_FUT"))
    if tgt.get("SMLT", 0.0) <= 1e-9 and ti <= tol and cur.get("SMLT", 0.0) > tol:
        return "EXIT"
    if abs(tgt.get("SMLT", 0.0) - cur.get("SMLT", 0.0)) <= tol and tgt.get("SMLT_FUT", 0.0) < cur.get("SMLT_FUT", 0.0) - tol:
        return "HEDGE"
    if ti < ci - tol:
        return "REDUCE"
    if ti > ci + tol:
        return "INCREASE"
    if others_changed:
        return "ROTATE"
    return "HOLD"


def decide(ctx, cfg: dict, P: dict, t: pd.Timestamp, book: E.Book, available: float, states: dict, ev: dict,
           views_iv: pd.DataFrame, universe: str, mandate: str, kappa: float, budget: float, panel_end=None, px=None,
           n_gen=None, n_adm=None, use_caps: bool = True, delay: int = 1, cost_mult: float = 1.0) -> Decision:
    cap = issuer_cap(cfg, states["ISSUER_CREDIT_RISK"], ev["severe_restriction"]) if use_caps else None
    su = F.build(ctx, cfg, P, t, universe, mandate, views_iv, book0=book, n_gen=n_gen, n_adm=n_adm, panel_end=panel_end,
                 px=px, issuer_cap=cap, available0=available, cost_mult=cost_mult)
    cur = _current_eval(su, cfg)
    cv = F.A.judge(cur["rows"], F.A.Rule(budget=budget, beta=cfg["owner"]["liquidity_min_nav"],
                                         max_breach=cfg["r1"]["admission"]["max_breach_probability"],
                                         stress_mults=tuple(cfg["r1"]["admission"]["margin_stress_survival"])), cur["hard"])
    cur_w = cur["w_exec"]
    grid = [e for e in cfg["r3"]["frontier"]["es_grid_nav0"] if e <= budget + 1e-12]
    cands, seen = [], {}
    for e in grid:
        sol = F.solve(su, e, kappa)
        if sol["x"] is None:
            continue
        key = tuple(np.round(sol["x"], 6))
        if key in seen:
            continue
        ex = F.admit_candidate(su, sol["x"], delay)
        v = F.verdict(ex, budget, cfg)
        seen[key] = True
        cands.append({"e": e, "x": sol["x"], "ex": ex, "passed": v.passed, "reasons": v.reasons,
                      "robust": sol["robust_excess"], "nominal": sol["nominal_excess"]})
    adm = [c for c in cands if c["passed"]]
    d = Decision(t, "HOLD", [], cv.passed, cv.reasons, cap if cap is not None else float("nan"), dict(states), ev,
                 current_w=cur_w, candidates=cands, su=su, current_metrics=F.summarise(su, cur))
    if not adm:
        d.action = "NO_ADMISSIBLE_TARGET"
        d.reasons = ["no frontier candidate passes the full engine at this budget / state"]
        return d

    def dist(c):
        w = c["ex"]["w_exec"]
        dv = [abs(w.get(r, 0.0) - cur_w.get(r, 0.0)) for r in su.routes]
        return max(dv), sum(dv)
    best = min(adm, key=dist)
    d.distance = dist(best)[0]
    thr = cfg["r5"]["materiality_max_abs_deviation_nav"]
    if cv.passed and d.distance <= thr + 1e-12:
        d.action = "HOLD"
        d.reasons = [f"current book admitted; distance to nearest admitted frontier candidate {d.distance:.4f} NAV <= {thr}"]
        d.target_w = cur_w
        d.metrics = d.current_metrics
        return d
    d.target_w = best["ex"]["w_exec"]
    d.ex = best["ex"]
    d.metrics = {**F.summarise(su, best["ex"]), "robust_excess_nav0": best["robust"], "nominal_excess_nav0": best["nominal"],
                 "es_target": best["e"]}
    d.action = label(cur_w, d.target_w)
    if d.action == "HOLD":
        d.action = "ROTATE"
    d.reasons = ([] if cv.passed else ["current book NOT admitted: " + "; ".join(cv.reasons)]) + \
                [f"nearest admitted frontier candidate at distance {d.distance:.4f} NAV (threshold {thr})"]
    if ev["review_flags"]:
        d.reasons.append("REVIEW flags: " + "; ".join(ev["review_flags"]))
    return d
