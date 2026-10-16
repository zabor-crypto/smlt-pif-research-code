"""Adaptive redesign R1 — the single admission authority.

LP / convex optimisation only GENERATES candidates. A candidate becomes publishable only through `admit`, which
re-evaluates the INTEGER book (after pruning, rounding and conversion to executable quantities) in the full recursive
engine on every applicable model path set and under the pre-declared margin-stress multipliers. `publish` refuses any
target without a PASS record — there is no fallback to a book that fails the same rule (NO_ADMISSIBLE_TARGET).

Rule (configs/adaptive.yaml r1.admission), evaluated per model path set, worst case over models:
  ES95(loss of mark-to-market NAV over the horizon) <= budget * NAV0
  P(immediately available liquidity < 5% NAV at any clearing) <= 5%
  P(unresolved settlement / margin deficit) = 0
  zero unresolved deficits under every stress multiplier (1.25, 1.50)
  integer contracts on every path-day; all hard exposure / mandate constraints on the executable book
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import engine as E


class AdmissionError(RuntimeError):
    pass


@dataclass
class ModelPaths:
    name: str
    fin: list
    din: list
    mkt: E.Market
    path_ids: tuple | None = None


@dataclass
class Rule:
    budget: float                       # ES budget as a fraction of NAV0
    alpha: float = 0.95
    beta: float = 0.05                  # hard liquidity minimum (fraction of NAV) — REQUIRED at every caller
    max_breach: float = 0.05
    max_unresolved: float = 0.0
    stress_mults: tuple = (1.25, 1.50)
    hysteresis: float = 0.05            # operating band width above the VM-adjusted minimum (engine cash rule)
    issuer_floor: bool = False
    funded_floor: float = 0.0
    emergency_floor_override: bool = False
    restoration_telemetry: bool = False
    recovery_clock: bool = False
    execution_check: object = None
    continuous_check: object = None
    floor_names: tuple | None = None
    drift_check: object = None


@dataclass
class Admission:
    passed: bool
    reasons: list = field(default_factory=list)
    rows: list = field(default_factory=list)          # one row per (model, multiplier)
    worst: dict = field(default_factory=dict)
    status: str = "ORDINARY_ADMISSION_PASS"


def tail_rank(G_d: np.ndarray, G_f: np.ndarray, qd: np.ndarray, qf: np.ndarray, release_d: np.ndarray,
              release_f: np.ndarray, cost_d: np.ndarray, cost_f: np.ndarray, alpha: float = 0.95) -> tuple:
    """Pre-declared waterfall ranking from the DECISION-TIME generation scenarios (never path by path).

    G_d (S, Ad) horizon P&L per contract, G_f (S, Af) per funded unit. Tail set = worst (1-alpha) scenarios of the
    book's P&L. Tail relief per unit reduction of position j = -sign(q_j) * mean_tail(g_j): positive when cutting j
    lowers tail loss (risk-adding position), negative for hedges. Order: descending relief per RUB of liquidity
    released; hedges (negative relief) therefore come last; ties by lower cost per RUB."""
    pnl = G_d @ qd + G_f @ qf
    k = max(1, int(np.ceil((1 - alpha) * len(pnl))))
    tail = np.argsort(pnl)[:k]

    def order(G, q, rel, cost):
        idx = [j for j in range(len(q)) if abs(q[j]) > 0]
        relief = {j: -np.sign(q[j]) * float(G[tail, j].mean()) for j in idx}
        key = sorted(idx, key=lambda j: (-(relief[j] / max(rel[j], 1e-12)), cost[j] / max(rel[j], 1e-12), j))
        rest = [j for j in range(len(q)) if j not in idx]
        return tuple(key + rest), relief
    wf_d, rel_d = order(G_d, qd, release_d, cost_d)
    wf_f, rel_f = order(G_f, qf, release_f, cost_f)
    return wf_d, wf_f, rel_d, rel_f


def static_horizon_pnl(qf0: np.ndarray, qd0: np.ndarray, cash0: float, qf: np.ndarray, qd: np.ndarray,
                       mkt: E.Market, fin: list) -> np.ndarray:
    """Closed-form horizon P&L of a book traded ONCE at t=0 and held (no rolls, no calls, settlement lag 0, cash rate
    constant): the no-path-dependence control that the engine must match exactly."""
    P, Fn, pv = mkt.P, mkt.Fn, mkt.pv
    dq = qf - qf0
    cprop = np.array([i.cost_prop for i in fin])
    cunit = np.array([i.cost_unit for i in fin])
    fcost = (np.abs(dq) * (cprop * P[:, 0, :] + cunit)).sum(axis=1)
    dcost = (np.abs(qd - qd0) * mkt.fee[:, 0, :]).sum(axis=1)
    cash_t0 = cash0 - (dq * P[:, 0, :]).sum(axis=1) - fcost - dcost
    growth = np.prod(1 + mkt.cash_rate[1:])
    price = (qf * (P[:, -1, :] - P[:, 0, :])).sum(axis=1)
    vm = (qd * (Fn[:, -1, :] - Fn[:, 0, :]) * pv[:, 0, :]).sum(axis=1)
    return price + vm + cash_t0 * (growth - 1) - fcost - dcost


def evaluate(paths: ModelPaths, book0: E.Book, orders: list, rule: Rule, mult: float, reserve: float,
             wf_rank: tuple, restore_rank: tuple, horizon: int | None = None) -> dict:
    """One full-engine evaluation. `rule.beta` is passed explicitly into the engine (D1-03: buffer at every caller)."""
    prm = E.Params(beta=rule.beta, mult=mult, reserve_frac=reserve, hysteresis=rule.hysteresis, wf_rank=wf_rank,
                   restore_rank=restore_rank, issuer_floor=rule.issuer_floor, funded_floor=rule.funded_floor,
                   emergency_floor_override=rule.emergency_floor_override, restoration_telemetry=rule.restoration_telemetry,
                   execution_check=rule.execution_check,continuous_check=rule.continuous_check,
                   floor_names=rule.floor_names,evaluation_context=paths.name,
                   drift_check=rule.drift_check,
                   path_ids=paths.path_ids if paths.path_ids is not None else tuple(range(paths.mkt.P.shape[0])))
    eng = E.Engine(paths.fin, paths.din, paths.mkt, prm)
    res = eng.run(book0, orders=orders)
    nav0 = res["nav"][:, 0]
    mm = np.array([i.kind == E.MMF for i in paths.fin])
    if mm.any():
        j = int(np.flatnonzero(mm)[0])
        mmg = paths.mkt.P[:, -1, j] / paths.mkt.P[:, 0, j]
    else:
        mmg = np.prod(1 + paths.mkt.cash_rate[1:]) * np.ones(len(nav0))
    lc = (np.abs(res["qf_end"]) * (eng.cprop[None, :] * paths.mkt.P[:, -1, :] + eng.cunit[None, :])).sum(axis=1)
    if paths.din:
        lc = lc + (np.abs(res["qd_end"]) * np.broadcast_to(paths.mkt.fee[:, -1, :], res["qd_end"].shape)).sum(axis=1)
    m = E.metrics(res, nav0, np.broadcast_to(mmg, nav0.shape), liq_cost=lc)
    m["NAV0"] = float(nav0.mean())
    from . import semantics as D26
    stages = res['execution_records']
    count = sum(r.floor_status[1] for r in stages)
    floor_pass = all(r.floor_status[0] in ('PASS','N/A') for r in stages)
    m['funded_floor_applicable_count'] = count
    m['funded_floor_status'] = 'N/A' if count==0 else 'PASS' if floor_pass else 'FAIL'
    m['executable_funded_floor_pass'] = floor_pass
    m['floor_applicable_keys_identical'] = res['callback_floor_keys']==res['record_floor_keys']
    if not m['floor_applicable_keys_identical']:
        raise AdmissionError('D26_FLOOR_CHANNEL_KEY_IDENTITY_MISMATCH')
    m['target_stage_applicable_count'] = sum(int(r.ordinary.sum()) for r in stages)
    m['ACTIVE_EXECUTION_STAGE_MANDATE_PASS'] = ('N/A' if not m['target_stage_applicable_count'] else
        'PASS' if all(not ids for r in stages for ids in r.failure_ids) else 'FAIL')
    m['PATHWISE_CONTINUOUS_SAFETY_PASS'] = 'PASS' if res['continuous_safety_pass'].all() else 'FAIL'
    m['ALLOCATION_DRIFT_DIAGNOSTIC'] = bool(res['allocation_drift_diagnostic'].any())
    m['severe_exit_due_blocked'] = any('SEVERE_EXIT_DUE_BLOCKED' in r['status'] for r in res['severe_exit_records'])
    m['execution_plan_ids'] = sorted(set(o.plan.execution_plan_id for o in orders if o.plan is not None))
    m['ordinary_admission_eligible'] = bool(not rule.emergency_floor_override and
                                          (res['active_regime']=='ORDINARY').all())
    m['constraint_semantics_version'] = D26.VERSION
    m["actual_constraints_pass"] = bool(res["actual_constraints_pass"].all())
    m["actual_issuer_floor_pass"] = bool(not rule.issuer_floor or (res["min_net_smlt_units"] >= -1e-9).all())
    m["min_net_smlt_units"] = float(res["min_net_smlt_units"].min())
    m["emergency_floor_override_count"] = int(res["emergency_floor_override_count"].sum())
    path_breach = res["available"] < rule.beta * res["nav"] - 1e-9
    first_action = min((o.day for o in orders), default=None)
    accountable = path_breach.copy()
    pre_unserviceable = np.zeros(len(nav0), bool)
    if rule.recovery_clock and first_action is not None and 0 < first_action < path_breach.shape[1]:
        # Only the buffer observation before execution is outside the candidate's control.
        # Negative immediately available cash is a real obligation, never waived; nor is
        # an unresolved call or any breach at/after the first order (including settlement).
        pre_unserviceable = (res["available"][:, :first_action] < -1e-9).any(axis=1)
        accountable[:, :first_action] = False
        m["p_breach"] = float((accountable.any(axis=1) | pre_unserviceable).mean())
    m["p_breach_raw"] = float(path_breach.any(axis=1).mean())
    m["p_inherited_pre_action_breach"] = float(path_breach[:, 0].mean())
    m["p_transition_unavoidable_breach"] = float(path_breach[:, 1:first_action].any(axis=1).mean()) if first_action else 0.0
    m["p_pre_action_unserviceable"] = float(pre_unserviceable.mean())
    m["recovery_clock_applied"] = bool(rule.recovery_clock and first_action is not None and
                                      m["p_breach_raw"] > m["p_breach"])
    failed = accountable.any(axis=1) | pre_unserviceable | res["unresolved"]
    if failed.any():
        first = int(np.flatnonzero(failed)[0])
        days = np.flatnonzero(accountable[first])
        m["first_failure_context"] = {
            "model":paths.name,"path_index":first,"first_buffer_breach_day":int(days[0]) if len(days) else None,
            "first_action_day":first_action,
            "inherited_pre_action_buffer_breach":bool(path_breach[first,0]),
            "unresolved_deficit":bool(res["unresolved"][first]),
            "reason_code":"UNRESOLVED_MARGIN_DEFICIT" if res["unresolved"][first] else
                          "PRE_ACTION_UNSERVICEABLE" if pre_unserviceable[first] else
                          "CANDIDATE_INDUCED_POST_ACTION_BREACH" if rule.recovery_clock else "LIQUIDITY_BUFFER_BREACH",
            **{key:res[key][first,:8].tolist() for key in ["nav","available","cash","mmf_value","pending","im"]}}
    m["_res"] = res
    return m


def evaluate_all(models: list[ModelPaths], book0: E.Book, orders: list, rule: Rule, reserve: float, wf_rank: tuple,
                 restore_rank: tuple) -> list[dict]:
    """Full-engine rows for every (model, multiplier); no verdict."""
    rows = []
    for mp in models:
        for mult in (1.0, *rule.stress_mults):
            m = evaluate(mp, book0, orders, rule, mult, reserve, wf_rank, restore_rank)
            res = m.pop("_res")
            rows.append({"model": mp.name, "margin_multiplier": mult, **m, "ES95_nav0": m["ES95"] / m["NAV0"],
                         "frac_contract_path_days": int(res["frac_contract_days"].sum())})
    return rows


def judge(rows: list[dict], rule: Rule, hard_checks: dict | None = None) -> Admission:
    """The admission rule applied to full-engine rows (worst case over models; survival under stress)."""
    reasons = []
    worst = {"ES95": -np.inf, "ES99": -np.inf, "p_breach": 0.0, "p_unresolved": 0.0}
    for row in rows:
        name, mult, nav0 = row["model"], row["margin_multiplier"], row["NAV0"]
        if mult == 1.0:
            if row["ES95"] > rule.budget * nav0 + 1e-6:
                reasons.append(f"{name}: ES95 {row['ES95'] / nav0:.4f} NAV0 > budget {rule.budget:.4f}")
            if row["p_breach"] > rule.max_breach + 1e-12:
                reasons.append(f"{name}: liquidity breach probability {row['p_breach']:.4f} > {rule.max_breach}")
            if row["p_unresolved"] > rule.max_unresolved + 1e-12:
                reasons.append(f"{name}: unresolved deficit probability {row['p_unresolved']:.4f}")
            worst["ES95"] = max(worst["ES95"], row["ES95"])
            worst["ES99"] = max(worst["ES99"], row["ES99"])
            worst["p_breach"] = max(worst["p_breach"], row["p_breach"])
            worst["p_unresolved"] = max(worst["p_unresolved"], row["p_unresolved"])
        elif row["p_unresolved"] > 0:
            reasons.append(f"{name}: unresolved deficits under margin stress x{mult:g} ({row['p_unresolved']:.4f})")
        if not row.get("executable_funded_floor_pass", True):
            reasons.append(f"{name}: executable funded investment floor failed (MANDATE_FAIL)")
        if row.get("p_pre_action_unserviceable", 0) > 0:
            reasons.append(f"{name}: unserviceable obligation before executable action (x{mult:g})")
        if not row.get("actual_constraints_pass",True):
            reasons.append(f"{name}: actual partial-fill book retains a hard-constraint breach")
        if not row.get("actual_issuer_floor_pass",True):
            reasons.append(f"{name}: actual issuer floor remains breached")
        if row.get('PATHWISE_CONTINUOUS_SAFETY_PASS','PASS') != 'PASS':
            reasons.append(f'{name}: continuous safety failed or not evaluable')
        if row.get('severe_exit_due_blocked',False):
            reasons.append(f'{name}: SEVERE_EXIT_DUE_BLOCKED')
        if row["frac_contract_path_days"]:
            reasons.append(f"{name}: fractional contracts on {row['frac_contract_path_days']} path-days")
    for k, ok in (hard_checks or {}).items():
        # UNKNOWN / NOT_EVALUABLE / uncounted N/A cannot certify admission.
        passed = ok=='PASS' if isinstance(ok,str) else bool(ok)
        if not passed:
            reasons.append(f"hard constraint failed: {k}")
    status = ("FAIL" if reasons else 'EMERGENCY_ADMISSION_PASS' if rule.emergency_floor_override or any(not r.get('ordinary_admission_eligible',True) for r in rows) else
              "RECOVERY_ADMISSION_PASS" if any(r.get("recovery_clock_applied", False) for r in rows) else "ORDINARY_ADMISSION_PASS")
    return Admission(passed=not reasons, reasons=reasons, rows=rows, worst=worst, status=status)


def admit(models: list[ModelPaths], book0: E.Book, orders: list, rule: Rule, reserve: float, wf_rank: tuple,
          restore_rank: tuple, hard_checks: dict | None = None) -> Admission:
    """Worst case over the applicable model set at multiplier 1.0, plus survival under the stress multipliers."""
    return judge(evaluate_all(models, book0, orders, rule, reserve, wf_rank, restore_rank), rule, hard_checks)


def publish(candidate_id: str, adm: Admission | None) -> str:
    """The ONLY way a target becomes applicable. Raises unless the full-engine admission passed."""
    if adm is None or not adm.passed:
        raise AdmissionError(f"{candidate_id}: no full-engine PASS — cannot publish "
                             f"({'; '.join(adm.reasons) if adm else 'not evaluated'})")
    return candidate_id
