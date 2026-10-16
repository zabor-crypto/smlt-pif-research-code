"""Regime V2 — ADAPTIVE_REGIME_PIF_V2 and its comparators: one decision function on one ledger.

STATE -> ceilings, the joint risk model and the state-dependent part of the expected return. Never STATE -> weights.

At a decision close t (information through t), from the ACTUAL book:
  1. the current book is evaluated by the full engine under the applicable mandate and ceilings;
  2. the frontier is generated from the actual book (LP candidates, refinement at PASS / FAIL brackets, anchors) and
     every candidate is admitted by the full engine;
  3. the target is the lexicographic optimum of the admitted candidates;
  4. HOLD unless (A) the current book fails risk or liquidity admission, (B) it violates a ceiling or the mandate,
     (C) the target's expected net excess after transaction cost exceeds the current book's by more than 25 bp NAV0, or
     (D) the target lowers ES95 by at least 1.0% NAV0 while giving up at most 10 bp of expected net excess;
  5. a failing current book is never HOLD: if no invested candidate is admitted the mandate is reported infeasible and
     the fund moves to the operational safe anchor (EMERGENCY_CAPITAL_PRESERVATION). The safe anchor must itself be
     admitted; otherwise the run stops.
Orders execute at the next session in integer quantities on the same ledger.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from ..adaptive import engine as E
from ..adaptive import policy as PL
from ..adaptive import replay as RP
from ..adaptive import scenarios as SC
from ..adaptive import views as V
from . import cfg as C2
from . import frontier as FR
from . import returns as RET
from . import risk as RK
from . import states as S2
from .cfg import StopCondition


@dataclass
class PolicySpec:
    name: str
    kind: str = "ADAPTIVE"                 # ADAPTIVE | STATIC | HEDGE_KEEP
    mandate: str = "HEDGED_INVESTED_PIF"
    floor: float | None = None             # sensitivity override of the hard funded floor
    budget: float = 0.15
    mode: str = "BASE_POSTERIOR_MEAN"
    universe: str = "MAIN"
    delay: int = 1
    cost_mult: float = 1.0
    keep_frac: float = 1.0                 # HEDGE_KEEP: share of the initial shares retained
    distressed_cap: float | None = None    # cutoff sensitivity of the DISTRESSED ceiling


@dataclass
class Recal:
    t: pd.Timestamp
    panel: pd.DataFrame
    beta: dict
    fwd: S2.ForwardLibrary
    parent_iv: pd.DataFrame | None = None


@dataclass
class Decision:
    date: pd.Timestamp
    policy: str
    action: str
    triggers: list
    flags: list
    current_admitted: bool
    current_reasons: list
    cap_net: float
    states: dict
    event_state: dict
    current_w: dict
    target_w: dict | None = None
    target: dict | None = None              # selected candidate (frontier.candidate)
    ex: dict | None = None                  # executable target (orders, reserve, ranks); None = no order
    candidates: list = field(default_factory=list)
    refinement_log: list = field(default_factory=list)
    lp_infeasible: list = field(default_factory=list)
    en_current: float = float("nan")
    en_target: float = float("nan")
    es_current: float = float("nan")
    es_target: float = float("nan")
    metrics: dict = field(default_factory=dict)
    current_metrics: dict = field(default_factory=dict)
    current_ex: dict | None = None
    su: object = None
    spec: object = None
    mu: pd.DataFrame | None = None
    carry: dict | None = None
    adj: dict = field(default_factory=dict)
    state_model: dict = field(default_factory=dict)
    comparator_only: bool = False
    admission_blockers: list = field(default_factory=list)


class Model:
    """Inputs that do not depend on the policy: strategic priors (prior era), state series, events, unit prices."""

    def __init__(self, ctx, cfg: dict, P: dict):
        self.ctx, self.cfg, self.P = ctx, cfg, P
        ctx._regime_v2r2_combined = True
        ctx._regime_v2r1_terms = bool(cfg.get("v2r1", {}).get("derivative_terms_repair", False))
        self.priors = RET.strategic_priors(ctx, cfg)
        self.feat, self.raw = S2.raw_states(ctx, cfg)
        self.events, self.severe = S2.severe_series(ctx, cfg, self.raw.index)
        self.conf = S2.confirmed(self.raw, cfg, self.severe)
        self.px = RP.unit_prices(ctx, cfg, P)
        self.origin_carry = RET.origin_carry(ctx, cfg, P, self.conf)
        self._parent_priors = None

    def context_at(self, t, states):
        from . import review_context as RC
        calendar = pd.read_csv(self.ctx.root / "data/regime_v2r2_combined/maturity_calendar.csv")
        return {**RC.housing_at(self.ctx.root, t, states.get("RATES_STATE")),
                "rating_context": RC.event_context(self.events, t) if hasattr(self, "events") else {},
                "maturity_review": RC.maturity_review(calendar, t, self.conf.index),
                "replay_qualification": RC.REPLAY_QUALIFICATION}

    def parent_priors(self) -> dict:
        if self._parent_priors is None:
            from ..adaptive import r3
            pri = V.priors_pre2022(self.ctx, self.cfg)
            bp = r3.priors_btc(self.ctx, self.cfg, self.P, pri)
            if bp:
                pri["BTC_RUB"] = bp
            self._parent_priors = pri
        return self._parent_priors

    def recal(self, t: pd.Timestamp, n_boot: int, extreme: bool = False) -> Recal:
        cfg = self.cfg
        start = cfg["r3"]["frontier"]["estimation_start"]
        panel = SC.history_panel(self.P, t, start)
        fwd = S2.ForwardLibrary(cfg, self.P, self.conf, t, n_boot, set(RET.FUNDED), list(RET.ALL_ROUTES), carry_by_origin=self.origin_carry)
        rc = Recal(t, panel, RET.smlt_beta(self.P, cfg["v2"]["state"]["sample_start"], t), fwd)
        if extreme:
            H = cfg["r3"]["frontier"]["horizon_sessions"]
            pts, _ = V.registry(self.ctx, cfg, self.P, t, H / 252, self.parent_priors(), panel)
            iv = V.intervals(pts, panel, H / 252).set_index("route")
            iv["centre_risk"] = iv["central"]
            rc.parent_iv = iv
        return rc

    def expected(self, rc: Recal, t: pd.Timestamp, states: dict, mode: str, use_states: bool) -> tuple:
        """(mu table, views frame, carry, adjustment detail) at decision date t for the confirmed `states`."""
        cfg = self.cfg
        panel_now = SC.history_panel(self.P, t, cfg["r3"]["frontier"]["estimation_start"])
        carry = RET.carry_at(self.ctx, cfg, self.P, t, panel_now)
        adj = {r: (rc.fwd.adjustment(r, states) if use_states else {"route": r, "value": 0.0, "level": "NO_STATES_STATIC", "stats": None, "tried": []})
               for r in RET.EXPOSURE_ROUTES}
        m = "ZERO_STATE_SIGNAL" if mode == "EXTREME_ROBUST" else mode
        mu = RET.expected(cfg, self.priors, carry, rc.beta["beta"], {r: a["value"] for r, a in adj.items()}, m)
        if mode == "EXTREME_ROBUST":
            return mu, rc.parent_iv, carry, adj
        return mu, RET.views_frame(mu), carry, adj


def issuer_ceiling(cfg: dict, ps: PolicySpec, states: dict, severe: bool) -> tuple[float, float | None]:
    caps = dict(cfg["v2"]["issuer_caps_net_long_nav"])
    if ps.distressed_cap is not None:
        caps["DISTRESSED"] = float(ps.distressed_cap)
    top = float(cfg["r3"]["bounds"]["SMLT"]["hi"])
    if ps.kind == "HEDGE_KEEP":
        return 10.0, None                                   # the comparator keeps its shares by definition
    if ps.kind == "STATIC":
        return top, None  # original static ceiling; changes in conditional reward never trigger trades
    if severe:
        return float(caps["SEVERE_VERIFIED_EVENT"]), float(cfg["v2"]["issuer_gross_shares_cap_under_severe_event"])
    return min(top, float(caps[states["ISSUER_CREDIT_RISK"]])), None


def triggers(fail_reasons: list, en_cur: float, es_cur: float, en_tgt: float, es_tgt: float, nt: dict) -> list:
    """The economic no-trade rule. A: the current book fails risk / liquidity admission; B: it fails only a ceiling or
    the mandate; C: expected net excess gain after cost > 25 bp NAV0; D: ES95 falls >= 1.0% NAV0 while expected net
    excess falls by at most 10 bp NAV0."""
    out = []
    if fail_reasons:
        hard = [r for r in fail_reasons if "hard constraint failed" in r]
        out.append("B" if len(hard) == len(fail_reasons) else "A")
    gain = en_tgt - en_cur
    if gain > float(nt["C_expected_net_excess_gain_nav0"]):
        out.append("C")
    if es_cur - es_tgt >= float(nt["D_es95_reduction_nav0"]) - 1e-12 and gain >= -float(nt["D_max_expected_loss_nav0"]) - 1e-12:
        out.append("D")
    return out


def _risky_key(qf, qd, fin_names) -> tuple:
    q = [0.0 if n == "MMF" else float(np.nan_to_num(v)) for n, v in zip(fin_names, qf)]
    return tuple(np.round(q, 6)) + tuple(np.round(np.asarray(qd, float), 6))


def severe_exit_plan(su, spec):
    """Causal next-session safety obligation; incomplete execution stays due.

    This is a quantity reduction, not an ordinary scored recommendation. The
    unchanged conversion supplies reserve/ranks; execution belongs to Engine.
    """
    from ..adaptive import frontier as base, semantics as D26
    from . import reference
    x=su.x0.copy()
    for r in ('SMLT','SMLT_FUT'):
        if r in su.routes:x[su.routes.index(r)]=0.
    ex=base.executable(su,x,1)
    tf=np.full(len(su.fin_names),np.nan);td=np.full(len(su.din_names),np.nan)
    if 'SMLT' in su.fin_names:tf[su.fin_names.index('SMLT')]=0.
    if 'SMLT_FUT' in su.din_names:td[su.din_names.index('SMLT_FUT')]=0.
    orders=[E.Order(1,tf,td,'SEVERE_ISSUER_EXIT')]
    sources=getattr(su,'d26_content_identity',None) or D26.content_identity()
    plan=D26.ExecutionPlan.create(origin=dict(date=su.t,book=su.book0),nav=su.nav0,
        endpoint=dict(qf=np.where(np.isfinite(tf),tf,su.book0.qf),qd=np.where(np.isfinite(td),td,su.book0.qd)),
        orders=orders,spec=spec,terms=dict(funded=su.fspecs,derivatives=su.dspecs),
        mechanics=dict(reserve=ex['reserve_min'],waterfall=ex['wf_rank'],restoration=ex['restore_rank'],
                       beta=su.cfg['owner']['liquidity_min_nav'],hysteresis=su.cfg['r1']['liquidity']['sweep_hysteresis_nav']),
        sources=sources,horizon=su.cfg['r3']['frontier']['horizon_sessions'],regime='ISSUER_SAFETY')
    ex.update(execution_plan=plan,execution_plan_id=plan.execution_plan_id,
        orders=[replace(orders[0],plan=plan,stage_check=FR._execution_check(su,spec))],spec=spec,target_published=False)
    ex.update(reference.evaluate(su,ex,spec))
    ex['issuer_obligation_status']='COMPLETED' if ex['reference_plan_state']=='COMPLETED' and ex['net_issuer_units']==0 else 'SEVERE_EXIT_DUE_BLOCKED'
    return ex


def decide(M: Model, rc: Recal, t: pd.Timestamp, book: E.Book | None, available: float | None, states: dict, ev: dict,
           ps: PolicySpec, *, n_gen: int, n_adm: int, first: bool = False, capital: float | None = None,
           seed_offset: int = 0) -> Decision:
    cfg = M.cfg
    if hasattr(M, "context_at"):
        ev = {**ev, "review_context": M.context_at(t, states)}
    use_states = True  # same initial return/risk authority for STATIC and ADAPTIVE
    severe = bool(ev["severe_restriction"]) and ps.kind == "ADAPTIVE"
    man = C2.mandate(cfg, "CAPITAL_PRESERVATION_UNCONSTRAINED" if ps.kind == "HEDGE_KEEP" else ps.mandate, ps.floor)
    cap_net, cap_gross = issuer_ceiling(cfg, ps, states, severe)
    mu, views, carry, adj = M.expected(rc, t, states, ps.mode, use_states)
    choice = RK.state_model_choice(rc.panel.index, M.conf.loc[:rc.t], states, cfg) if use_states else None
    models = tuple(m for m in cfg["v2"]["risk"]["models"] if use_states or m != RK.STATE_MODEL)
    su = FR.build(M.ctx, cfg, M.P, t, ps.universe, views, book0=book, capital=capital, n_gen=n_gen, n_adm=n_adm, models=models,
                  state_choice=choice, seed_offset=seed_offset, panel_end=rc.t, px=M.px, available0=available,
                  cost_mult=ps.cost_mult)
    kappa = float(cfg["v2"]["returns"]["modes"]["EXTREME_ROBUST"]["parent_kappa"]) if ps.mode == "EXTREME_ROBUST" else 0.0
    fixed = {}
    if ps.kind == "HEDGE_KEEP":
        fixed = {"SMLT": float(su.x0[su.routes.index("SMLT")]) * (ps.keep_frac if first else 1.0)}
    spec = FR.Spec(mandate=man, budget=ps.budget, cap_net=cap_net, cap_gross=cap_gross, kappa=kappa, delay=ps.delay, fixed=fixed)
    if severe:
        ex=severe_exit_plan(su,spec)
        actual={r:float(su.x0[j]) for j,r in enumerate(su.routes)}
        intended={**actual,'SMLT':0.,'SMLT_FUT':0.}
        return Decision(t,ps.name,'EXIT_SMLT',['VERIFIED_SEVERE_ISSUER_EVENT'],
            ['VERIFIED_SEVERE_EVENT_DIRECT_SMLT_CEILING_ZERO',ex['issuer_obligation_status']],
            False,['CAUSAL_ISSUER_ZERO_OBLIGATION'],cap_net,dict(states),dict(ev),actual,
            target_w=intended,ex=ex,su=su,spec=spec,mu=mu,carry=carry,adj=adj,
            metrics={'issuer_obligation_status':ex['issuer_obligation_status'],
                     'actual_net_issuer_units':ex['net_issuer_units'],'target_published':False})
    cur = FR.current_eval(su, spec)
    cv = FR.verdict(su, cur, ps.budget)
    cache: dict = {}
    static_kept = ps.kind == "STATIC" and not first and cv.passed
    fr = {"candidates": [], "refinement_log": [], "lp_infeasible": []} if static_kept else FR.frontier(su, spec, cache)
    T = FR.select(fr["candidates"], cfg)
    cur_w = cur["w_exec"]
    en_cur = FR.expected_net(su, su.x0, kappa, traded=False)
    es_cur = FR.es_of(su, cur)
    sc = su.v2
    d = Decision(t, ps.name, "HOLD", [], [], cv.passed, list(cv.reasons), cap_net, dict(states), dict(ev), cur_w,
                 candidates=fr["candidates"], refinement_log=fr["refinement_log"], lp_infeasible=fr["lp_infeasible"],
                 en_current=en_cur, es_current=es_cur, current_metrics=FR.summarise(su, cur), current_ex=cur, su=su, spec=spec,
                 mu=mu, carry=carry, adj=adj, state_model={k: v for k, v in sc.items()})
    if fr.get("lk_status") == "ALL_ECONOMICALLY_RELEVANT_CANDIDATES_LK_UNCERTIFIED_OR_BREACHING":
        d.flags.append(fr["lk_status"])
    if ev.get("review_context", {}).get("HOUSING_POLICY_CONFLICT"):
        d.flags.append("HOUSING_RELIEF_NOT_CONFIRMED_NO_UNQUALIFIED_HOUSING_RERISK")
    if choice is not None:
        d.state_model["tried"] = choice["tried"]
    fail = not cv.passed
    cur_key = _risky_key(su.book0.qf, su.book0.qd, su.fin_names)
    nt = cfg["v2"]["policy"]["no_trade"]

    def take(c: dict, action: str | None = None):
        d.target, d.ex, d.target_w = c, c["ex"], c["ex"]["w_exec"]
        d.en_target, d.es_target = c["expected_net"], c["es95"]
        d.metrics = FR.summarise(su, c["ex"])
        lab = action or PL.label(cur_w, d.target_w)
        d.action = "ROTATE" if lab == "HOLD" else lab

    def hold(c_en: float, c_es: float):
        d.action, d.target_w, d.ex = "HOLD", cur_w, None
        d.en_target, d.es_target, d.metrics = en_cur, es_cur, d.current_metrics

    if static_kept:
        hold(en_cur, es_cur)
        d.flags.append("STATIC_STRATEGIC_BOOK_MAINTAINED")
    elif T is None:
        if ps.kind == "HEDGE_KEEP":
            fb = FR.min_es_fallback(fr["candidates"])
            d.comparator_only = True
            d.flags.extend(["BEST_AVAILABLE_INADMISSIBLE_HEDGE_COMPARATOR", "INADMISSIBLE_COMPARATOR_ONLY", "FULL_ADMISSION=FAIL"])
            d.admission_blockers = sorted(FR.failure_classes(fb["reasons"])) if fb is not None else ["OTHER"]
            d.admission_blockers = ["MANDATE_FAIL" if r == "EXECUTABLE_FUNDED_FLOOR_FAIL" else r for r in d.admission_blockers]
            if fb is None:
                d.flags.append("NO_HEDGE_BOOK_WITHOUT_UNRESOLVED_DEFICIT")
                hold(en_cur, es_cur)
                d.action = "NO_ADMISSIBLE_HEDGE_BOOK"
            elif _risky_key(fb["ex"]["qf"], fb["ex"]["qd"], su.fin_names) == cur_key:
                hold(en_cur, es_cur)
                d.target = fb
                d.action = "BEST_AVAILABLE_HEDGE_HELD"
            else:
                take(fb)
                d.triggers.append("A" if fail else "LOWEST_ES_FALLBACK")
        elif fail:
            b = int(round(ps.budget * 100))
            d.flags.append("NO_ADMITTED_CANDIDATE_IN_FROZEN_SEARCH")
            sa = FR.safe_anchor(su, spec, cache)
            FR.stop_if_no_safe_anchor(sa)
            d.candidates = d.candidates + [sa]
            if not np.any(np.abs(su.x0) > 1e-9):
                sus = replace(spec, floor_suspended=True)
                cs = FR.current_eval(su, sus)
                if not FR.verdict(su, cs, ps.budget).passed:
                    raise StopCondition("the held safe money-market book fails admission")
                hold(en_cur, es_cur)
                d.action = "EMERGENCY_CAPITAL_PRESERVATION"
                d.flags.append("SAFE_ANCHOR_MAINTAINED")
            else:
                take(sa, "EMERGENCY_CAPITAL_PRESERVATION")
            d.triggers.append("A_OR_B_NO_ADMITTED_INVESTED_CANDIDATE")
        else:
            d.flags.append("NO_ADMITTED_FRONTIER_CANDIDATE_CURRENT_BOOK_ADMITTED")
            hold(en_cur, es_cur)
    else:
        all_triggers = triggers(list(cv.reasons) if fail else [], en_cur, es_cur, T["expected_net"], T["es95"], nt)
        d.triggers.extend([x for x in all_triggers if x in ("A", "B")] if ps.kind == "STATIC" and not first else all_triggers)
        if ps.kind == "STATIC" and first:
            d.triggers.append("INITIAL_STRATEGIC_SELECTION")
        same = _risky_key(T["ex"]["qf"], T["ex"]["qd"], su.fin_names) == cur_key
        if d.triggers and not same:
            take(T)
        else:
            hold(T["expected_net"], T["es95"])
            d.target = T
            if same and d.triggers:
                d.flags.append("TARGET_EQUALS_CURRENT_BOOK")
    if ev.get("review_flags"):
        d.flags.append("REVIEW: " + "; ".join(ev["review_flags"]))
    if severe:
        d.flags.append("VERIFIED_SEVERE_EVENT_DIRECT_SMLT_CEILING_ZERO")
    if fail and d.action == "HOLD":
        raise StopCondition(f"{ps.name} {t.date()}: an inadmissible current book returned HOLD ({'; '.join(cv.reasons)})")
    return d


def action_labels(current: dict, target: dict, action: str) -> dict:
    """Reporting only: an issuer exit can be a portfolio rotation."""
    old, new = current.get("SMLT", 0.), target.get("SMLT", 0.)
    smlt = "EXIT" if old > 1e-9 and new <= 1e-9 else "REDUCE" if new < old - 1e-9 else "INCREASE" if new > old + 1e-9 else "HOLD"
    portfolio = action
    if action == "EXIT" and sum(target.get(r, 0.) for r in FR.FUNDED) > 1e-9:
        portfolio = "ROTATE"
    return {"PORTFOLIO_ACTION": portfolio, "SMLT_ACTION": smlt}


def sleeve_split(cfg: dict, w: dict) -> dict:
    sl = cfg["v2"]["sleeves"]
    return {k: float(sum(w.get(r, 0.0) for r in v)) for k, v in sl.items()}


def log_row(d: Decision, extra: dict | None = None) -> dict:
    su = d.su
    man = d.spec.mandate
    fs = man["floor_set"]
    adm = [c for c in d.candidates if c["passed"]]
    row = {"policy": d.policy, "date": d.date, "action": d.action, "triggers": ";".join(d.triggers), "flags": " | ".join(d.flags),
           "current_admitted": d.current_admitted, "current_fail_reasons": " | ".join(d.current_reasons),
           "issuer_ceiling_net": d.cap_net, **{f"state_{k}": v for k, v in d.states.items()},
           "severe_restriction": d.event_state.get("severe_restriction"), "risk_models": ";".join(d.state_model.get("models", [])),
           "state_model_level": d.state_model.get("state_model_level"), "n_candidates": len(d.candidates), "n_admitted": len(adm),
           "expected_net_current_nav0": d.en_current, "expected_net_target_nav0": d.en_target,
           "es95_current_nav0": d.es_current, "es95_target_nav0": d.es_target,
           "funded_non_mmf_current": d.current_metrics.get("funded_fraction_at_execution", float(sum(d.current_w.get(r, 0.0) for r in fs))),
           "funded_non_mmf_target": d.metrics.get("funded_fraction_at_execution", float(sum((d.target_w or {}).get(r, 0.0) for r in fs))),
           "floor_hard": man["floor_hard"], "mandate": man["name"], "nav": su.nav0,
           **{f"w_cur_{r}": v for r, v in d.current_w.items()}, **{f"w_tgt_{r}": v for r, v in (d.target_w or {}).items()},
           **{f"mu_{r}": float(d.mu.loc[r, "mu"]) for r in su.routes},
           "state_views_used": "; ".join(f"{r}:{a['level']}={a['value']:.4f}" for r, a in sorted(d.adj.items()) if abs(a["value"]) > 0)}
    if d.action == "HOLD" and d.target is not None:
        row.update({"proposal_target_candidate_id": d.target["id"], "proposal_expected_excess_nav0": d.target["expected_net"],
                    "proposal_ES95_nav0": d.target["es95"], "proposal_reason_not_traded": "NO_MATERIAL_TRIGGER_OR_SAME_EXECUTABLE_BOOK",
                    **{f"proposal_target_{r}": v for r,v in d.target["ex"]["w_exec"].items()}})
    if d.ex is not None and d.spec.floor_hard > 0 and d.ex["funded_fraction_at_execution"] < d.spec.floor_hard - 1e-12:
        row["mandate_status"] = "EMERGENCY_INVESTED_FLOOR_OVERRIDE"
    else:
        row["mandate_status"] = "EXECUTABLE_MANDATE_PASS" if d.ex is not None else "CURRENT_ACTUAL_BOOK"
    row["admission_status"] = (d.target or {}).get("admission_status", "FAIL") if d.ex is not None else "ORDINARY_ADMISSION_PASS" if d.current_admitted else "FAIL"
    for key in ["p_breach_raw_worst", "p_inherited_pre_action_breach_worst", "p_transition_unavoidable_breach_worst", "p_pre_action_unserviceable_worst"]:
        row[key] = d.metrics.get(key)
    from . import review_context as RC
    prices = getattr(su,"unit_price",{})
    if hasattr(su,'book0') and hasattr(su,'fspecs'):
        units=FR.origin_issuer_units(su)
        shares=float(su.book0.qf[su.fin_names.index('SMLT')]) if 'SMLT' in su.fin_names else 0.
        direct_spot=shares*prices.get('SMLT',0.)/su.nav0
        direct_hedge=(units-shares)*prices.get('SMLT',0.)/su.nav0
    else:
        direct_spot=d.current_w.get('SMLT',0.)
        # Legacy weight-only origin inputs can be inverted to origin quantities;
        # the issuer identity itself still contains only shares/contracts.
        derivative_value=d.current_w.get('SMLT_FUT',0.)*su.nav0
        contracts=derivative_value/prices.get('SMLT_FUT',prices.get('SMLT',1.))
        equivalent=next((x.pv0 for x in getattr(su,'dspecs',[]) if x.name=='SMLT_FUT'),1.)
        direct_hedge=contracts*equivalent*prices.get('SMLT',1.)/su.nav0
    context = {**d.event_state.get("review_context", {}),
               **RC.issuer_disclosure(direct_spot,direct_hedge)}
    for key in ('REFERENCE_TARGET_MANDATE_PASS','ACTIVE_EXECUTION_STAGE_MANDATE_PASS',
                'PATHWISE_CONTINUOUS_SAFETY_PASS','DETERMINISTIC_CASH_MARGIN_SURVIVAL',
                'DETERMINISTIC_TARGET_STAGE_COMPLIANCE','ALLOCATION_DRIFT_DIAGNOSTIC','FULL_ENGINE_ADMISSION_PASS'):
        row[key]=d.metrics.get(key,'NOT_EVALUABLE')
    row['execution_plan_id']=(d.ex or {}).get('execution_plan_id')
    row['target_published']=False
    if d.action=='EXIT_SMLT' and 'VERIFIED_SEVERE_ISSUER_EVENT' in d.triggers:
        row['mandate_status']='ISSUER_SAFETY_OBLIGATION'
        row['issuer_obligation_status']=d.metrics['issuer_obligation_status']
        row['actual_net_issuer_units']=d.metrics['actual_net_issuer_units']
        row['admission_status']='SAFETY_OBLIGATION_DIAGNOSTIC'
    row.update(context)
    row.update(extra or {})
    row.update(action_labels(d.current_w, d.target_w or d.current_w, d.action))
    row["investment_expected_net_excess"] = d.en_target
    row["risk_distribution_note"] = "RISK DISTRIBUTION DIAGNOSTIC — NOT RETURN FORECAST"
    if d.comparator_only:
        row.update({"comparator_status": "BEST_AVAILABLE_INADMISSIBLE_HEDGE_COMPARATOR", "FULL_ADMISSION": "FAIL",
                    "policy_status": "INADMISSIBLE_COMPARATOR_ONLY", "admission_blockers": ";".join(d.admission_blockers),
                    "admission_status": "FAIL"})
    else:
        row["FULL_ADMISSION"] = "PASS" if row["admission_status"] not in ("FAIL","SAFETY_OBLIGATION_DIAGNOSTIC") else "FAIL"
    return row
