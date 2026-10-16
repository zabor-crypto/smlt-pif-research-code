"""Deterministic decision-price execution from frozen origin specs.

No admission path is read. Funded prices, FX point values and CNY settlement
are held at the decision origin. The existing origin roll gap and lifecycle
schedule apply; each same-series reference mark is constant. There is no
return, interest, swap forecast or future realised input in this report.
"""
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from ..adaptive import engine as E, market as M
from .cfg import StopCondition


@dataclass(frozen=True)
class ReferenceIdentity:
    execution_plan_id: str | None
    snapshot_id: str
    completion_day: int | None


def _invalid(state):
    return dict(reference_plan_state=state, reference_eligible=False,
                reference_completion_day=None, REFERENCE_TARGET_MANDATE_PASS='FAIL',
                target_published=False)


def _plan_error(ex):
    from ..adaptive.semantics import ExecutionPlan
    plan, pid = ex.get('execution_plan'), ex.get('execution_plan_id')
    if not isinstance(plan, ExecutionPlan) or not pid or not plan.execution_plan_id:
        return 'NOT_EVALUABLE_NO_PLAN_ID'
    if pid != plan.execution_plan_id or sha256(plan.payload.encode()).hexdigest() != pid:
        return 'NOT_EVALUABLE_PLAN_ID_MISMATCH'
    endpoint = None
    if 'authorized_qf' in ex or 'authorized_qd' in ex:
        endpoint = dict(qf=ex.get('authorized_qf'), qd=ex.get('authorized_qd'))
    return plan.validate_binding(orders=ex.get('orders', ()), endpoint=endpoint)


def _completion(ex, pid):
    completed = ex.get('reference_result', {}).get('plan_completions', {}).get((pid, 0))
    if completed is None or completed['stage'].plan_ids[0] != pid:
        return None
    return completed


def _snapshot(pid, day, qf, qd, cash, obligations, series, costs, turnover, nav, im):
    from ..adaptive import semantics as D26
    return D26.identity(dict(plan_id=pid, day=day, qf=qf, qd=qd, cash=cash,
                            pending=obligations, series=series, costs=costs,
                            turnover=turnover, nav=nav, im=im))


def certify_cached(ex):
    """Validate cached identity and actual completion; metadata alone cannot certify."""
    error = _plan_error(ex)
    if error:
        return _invalid(error)
    pid = ex['execution_plan_id']
    certificate = ex.get('reference_identity')
    if (not isinstance(certificate, ReferenceIdentity) or certificate.execution_plan_id != pid
            or ex.get('reference_plan_id') != pid or ex.get('reference_snapshot_plan_id') != pid
            or ex.get('reference_state_snapshot_id') != certificate.snapshot_id):
        return _invalid('NOT_EVALUABLE_PLAN_ID_MISMATCH')
    completed = _completion(ex, pid)
    if ex.get('reference_plan_state') == 'COMPLETED':
        if completed is None:
            return _invalid('NOT_EVALUABLE_NO_MATCHED_COMPLETION')
        stage = completed['stage']
        ledger = completed['pending'][0]
        obligations = {due:float(v) for due,v in enumerate(ledger) if due>stage.session and v!=0}
        snapshot = _snapshot(pid, stage.session, stage.qf[0], stage.qd[0], stage.cash[0],
                             obligations, completed.get('series'), completed['costs'][0],
                             completed['turnover'][0], stage.nav[0], stage.im[0])
        if (snapshot != certificate.snapshot_id or certificate.completion_day != stage.session
                or ex.get('reference_completion_day') != stage.session):
            return _invalid('NOT_EVALUABLE_PLAN_ID_MISMATCH')
    elif completed is None and ex.get('reference_eligible'):
        return _invalid('NOT_EVALUABLE_NO_MATCHED_COMPLETION')
    if ex.get('diagnostic_only') or ex['execution_plan'].regime == 'DIAGNOSTIC':
        return dict(reference_eligible=False, REFERENCE_TARGET_MANDATE_PASS='FAIL', target_published=False)
    return {}


def market(funded, derivatives, sessions):
    """Rebuild every execution matrix using the canonical forward producer."""
    T = int(sessions)
    if T < 1:
        raise StopCondition('REFERENCE_SESSION_COUNT_INVALID')
    fs, ds = deepcopy(funded), deepcopy(derivatives)
    for f in fs:
        if f.execution_dates is not None:
            if len(f.execution_dates) < T:
                raise StopCondition('REFERENCE_ORIGIN_CALENDAR_TOO_SHORT')
            f.execution_dates = f.execution_dates[:T]
    for d in ds:
        d.roll_days = tuple(t for t in d.roll_days if t < T)
        if d.execution_info is not None:
            dates = d.execution_info['dates']
            if len(dates) < T:
                raise StopCondition('REFERENCE_ORIGIN_CALENDAR_TOO_SHORT')
            d.execution_info['dates'] = dates[:T]
    fin, din, m = M.forward_market(fs, ds, np.zeros((1,T-1,len(fs))),
        np.zeros((1,T-1,len(ds))), np.zeros((1,T-1)), np.zeros(T-1),
        cash_rate=0., cny_swap_frac=np.zeros((1,T-1)))
    E.validate_market_path_dimensions(m)
    return fin, din, m


def evaluate(su, ex, spec):
    """One canonical shared-Engine trace over the existing bounded horizon.

    Scores use quantity completion; dated settlements and operational observations
    remain in that same trace. No independent reporting execution is constructed.
    """
    from ..adaptive import semantics as D26
    from . import reporting
    error = _plan_error(ex)
    if error and not (error == 'NOT_EVALUABLE_NO_PLAN_ID' and ex.get('diagnostic_only') is True):
        return _invalid(error)
    if 'reference_timeline' in ex:
        return {**ex, **certify_cached(ex)}
    H = int(su.cfg['r3']['frontier']['horizon_sessions'])
    eng,res,rows = reporting._run_reference(su,ex,spec,H+1)
    plan = ex.get('execution_plan')
    pid = plan.execution_plan_id if plan else None
    completed = res['plan_completions'].get((pid,0)) if pid else None
    state = res['plan_states'].get((pid,0),'EXPIRED_REFERENCE_HORIZON') if pid else 'NOT_EVALUABLE_NO_PLAN_ID'
    if state == 'COMPLETED' and completed is None:
        state = 'NOT_EVALUABLE_NO_MATCHED_COMPLETION'
    if completed is None:
        stages = res['execution_records']
        stage = stages[-1] if stages else None
        day = H
        qf,qd = res['qf_end'][0],res['qd_end'][0]
        cash,nav,im = res['cash_end'][0],res['nav'][0,-1],res['im'][0,-1]
        pending = res['pending_end'][0]
        costs,turnover = res['costs'][0],res['turnover'][0]
        notional = eng.execution.signed_notional()[0] if eng.execution else qd*eng._Fn(H)[0]*eng._pv(H)[0]
        units = eng.issuer_units(qf[None,:],qd[None,:],H)[0]
        available = res['available'][0,-1]
    else:
        stage = completed['stage'];day=stage.session
        qf,qd=stage.qf[0],stage.qd[0]
        cash,nav,im=stage.cash[0],stage.nav[0],stage.im[0]
        pending=stage.pending[0].sum()
        costs,turnover=completed['costs'][0],completed['turnover'][0]
        notional,units=stage.notional[0],stage.issuer_units[0]
        worst=min(0.,float(np.cumsum(stage.pending[0]).min())) if stage.pending.shape[1] else 0.
        available=float(cash-im+worst)
    floor_num=sum(qf[su.fin_names.index(r)]*su.unit_price[r] for r in spec.mandate['floor_set'] if r in su.fin_names)
    reference_stages=[s for s in res['execution_records'] if s.session<=day]
    target_pass=all(not s.failure_ids[0] for s in reference_stages)
    safety_pass=bool(res['continuous_safety_pass'].all())
    eligible=state=='COMPLETED' and target_pass and safety_pass
    if spec.floor_suspended:
        eligible=state=='COMPLETED' and safety_pass
    if ex.get('diagnostic_only') or (plan and plan.regime == 'DIAGNOSTIC'):
        eligible=False
    pending_ledger=completed['pending'][0] if completed else res['pending_ledger_end'][0]
    obligations={due:float(value) for due,value in enumerate(pending_ledger) if due>day and value!=0}
    snapshot_id=_snapshot(pid,day,qf,qd,cash,obligations,
                          completed.get('series') if completed else res.get('series_positions_end'),
                          costs,turnover,nav,im)
    completion_day=day if completed is not None else None
    return dict(reference_timeline=rows,reference_result=res,reference_plan_state=state,
        diagnostic_only=bool(ex.get('diagnostic_only')),target_published=False,
        reference_state_snapshot_id=snapshot_id,reference_completion_day=completion_day,
        reference_snapshot_day=day,reference_plan_id=pid,reference_snapshot_plan_id=pid,
        reference_identity=ReferenceIdentity(pid,snapshot_id,completion_day),
        reference_pending_obligations=obligations,
        reference_eligible=eligible,execution_nav=float(nav),execution_cash=float(cash),
        execution_pending=float(pending),execution_margin=float(im),execution_costs=float(costs),
        execution_turnover=float(turnover),execution_available=float(available),qf=qf.copy(),qd=qd.copy(),
        reference_notional=notional.copy(),net_issuer_units=float(units),
        net_issuer=float(units*su.unit_price.get('SMLT',0.)/su.nav0),
        funded_fraction_at_execution=float(floor_num/max(nav,1e-12)),
        REFERENCE_TARGET_MANDATE_PASS='PASS' if eligible else 'FAIL',
        execution_lk_certification=next((r['LK_CERTIFICATION'] for r in rows if r['engine_day']==day
                                        and r['phase']=='AFTER_FEES_AND_IM'),{}))
