"""Accounting and bound disclosures; no selection or execution policy changes."""
import numpy as np
import pandas as pd
from ..adaptive import engine as E
from . import frontier as FR, reference as REF
from .cfg import StopCondition


def balance_sheet(engine, day, qf, qd, cash, pending, gross_pending=None, costs=0.):
    qf, qd = np.asarray(qf, float).reshape(1, -1), np.asarray(qd, float).reshape(1, -1)
    pending = np.asarray(pending, float).reshape(1, -1)
    cash = np.array([float(cash)])
    values = qf * engine._P(day)
    funded = float(values[:, ~engine.mmf].sum())
    mmf = float(values[:, engine.mmf].sum())
    net = float(pending[:, day+1:].sum())
    flows = [v for due, v in (gross_pending or []) if due > day] if gross_pending is not None else pending[:, day+1:].ravel()
    receivables = float(sum(max(v, 0.) for v in flows))
    payables = float(sum(max(-v, 0.) for v in flows))
    if abs(receivables-payables-net) > 1e-6:
        raise RuntimeError("gross/net pending settlement reconciliation failed")
    nav = funded + mmf + float(cash[0]) + net
    return {"FUNDED_ASSET_MARKET_VALUE": funded, "MMF_MARKET_VALUE": mmf,
            "MMF_UNITS": float(qf[:, engine.mmf].sum()), "CASH_LEDGER": float(cash[0]),
            "POSTED_INITIAL_MARGIN_MEMO": float(engine.im(qd, day)[0]),
            "IMMEDIATELY_AVAILABLE_CASH": float(engine.available(cash, pending, qd, day)[0]),
            "PENDING_RECEIVABLES": receivables, "PENDING_PAYABLES": payables,
            "NET_PENDING_SETTLEMENT": net, "TRANSACTION_COSTS": float(costs),
            "DERIVATIVE_NOTIONAL_MEMO": float(engine.execution.gross()[0]) if engine.execution is not None else float((np.abs(qd)*engine._Fn(day)*engine._pv(day)).sum()), "NAV": nav,
            "NAV_RECONCILIATION_RESIDUAL": nav-funded-mmf-float(cash[0])-net}


def reference_balance_timeline(su, ex, spec):
    """Conditional decision-price execution; normal settlement and cash sweep.

    The observer records actual funded cash legs before same-date netting.
    Funded and same-series prices are held at their declared origin convention;
    frozen roll gaps and lifecycle mechanics remain active. Dates are causal
    reference sessions, not realised future prices or execution evidence.
    """
    ex.update(REF.evaluate(su,ex,spec))
    if 'reference_timeline' not in ex or (ex['reference_plan_state'].startswith('NOT_EVALUABLE')
                                        and not ex.get('diagnostic_only')):
        raise StopCondition(ex['reference_plan_state'])
    return ex["reference_timeline"]


def _run_reference(su, ex, spec, T):
    fin, din, m = REF.market(su.fspecs, su.dspecs, T)
    dates = (m.execution['dates'] if m.execution is not None
             else pd.bdate_range(su.t, periods=T))
    gross = list(su.book0.pending.items())
    rows, cumulative_cost = [], [0.]
    def record(eng, day, phase, cash, pend, qf, qd, costs):
        rows.append({"phase":phase,"engine_day":day,"date":str(dates[day].date()),
                     "basis":"CONDITIONAL_DECISION_PRICE_REFERENCE_EXECUTION",
                     **balance_sheet(eng,day,qf,qd,cash,pend,gross,costs),
                     "qf":np.asarray(qf).reshape(-1).tolist(), "qd":np.asarray(qd).reshape(-1).tolist(),
                     "LK_CERTIFICATION":({base: str(r['LK_CERTIFICATION'][0]) for base,r in eng.execution.lk_guard().items()}
                                         if eng.execution is not None else {})})
    class Observer(E.Engine):
        def _execute(self, order, day, cash, pend, qf, qd):
            before_f, before_cost = qf.copy(), float(getattr(self,'_costs_before_order',cumulative_cost)[0])
            record(self,day,"BEFORE_ORDER_"+order.tag,cash[0],pend,qf,qd,before_cost)
            result = super()._execute(order,day,cash,pend,qf,qd)
            ca, pe, qfa, qda, cost, _, _ = result
            dq = qfa-before_f
            val = dq*self._P(day)
            fc = self.funded_cost(dq,self._P(day))
            # Reconstruct the execution-close pre-fee snapshot for disclosure;
            # the real engine always retains every fee.
            pe_no_fees, ca_no_fees = pe.copy(), float(ca[0])+float(cost[0]-fc.sum())
            pending_before = gross.copy()
            new_flows=[]
            for j in range(self.Af):
                due = day+int(self.m.lag[day,j])
                if due > day:
                    pe_no_fees[0,due]+=fc[0,j]
                    if abs(val[0,j])>1e-12: gross.append((due,float(-val[0,j])))
                    if abs(val[0,j]+fc[0,j])>1e-12: new_flows.append((due,float(-(val[0,j]+fc[0,j]))))
                else:
                    ca_no_fees+=float(fc[0,j])
            record(self,day,"EXECUTION_CLOSE_BEFORE_FEES_RECONSTRUCTED",ca_no_fees,pe_no_fees,qfa,qda,before_cost)
            gross[:]=pending_before+new_flows
            cumulative_cost[0] = before_cost+float(cost[0])
            record(self,day,"AFTER_FEES_AND_IM" if order.tag=="TARGET" else "AFTER_"+order.tag,ca[0],pe,qfa,qda,cumulative_cost[0])
            return result
    prm = E.Params(beta=su.cfg["owner"]["liquidity_min_nav"],reserve_frac=ex["reserve_min"],
                   hysteresis=su.cfg["r1"]["liquidity"]["sweep_hysteresis_nav"],wf_rank=ex["wf_rank"],
                   restore_rank=ex["restore_rank"],issuer_floor=spec.mandate["overlay_rule"],log=True,
                   funded_floor=spec.mandate["floor_hard"],emergency_floor_override=spec.floor_suspended,
                   execution_check=FR._execution_check(su,spec),
                   continuous_check=FR._continuous_check(su,spec),drift_check=FR._allocation_drift(su,spec),floor_names=tuple(spec.mandate["floor_set"]))
    eng=Observer(fin,din,m,prm)
    if m.execution is not None:
        from .execution import Execution
        eng.execution = Execution(eng, np.asarray(su.book0.qd)[None,:], su.book0.series, su.book0.capacity)
    pending=np.zeros(T+8)
    for due,value in su.book0.pending.items(): pending[int(due)]+=value
    record(eng,0,"T0_BEFORE_ORDER",su.book0.cash,pending,su.book0.qf,su.book0.qd,0.)
    def checkpoint(day,snapshot):
        record(eng,day,"CLOSE_AFTER_SETTLEMENT_AND_CASH_RULE",snapshot["cash"][0],snapshot["pending"],
               snapshot["qf"],snapshot["qd"],snapshot["costs"][0])
    def settlement(day,snapshot):
        record(eng,day,"FUNDED_SETTLEMENT_BEFORE_CASH_RULE",snapshot["cash"][0],snapshot["pending"],
               snapshot["qf"],snapshot["qd"],snapshot["costs"][0])
    result=eng.run(su.book0,ex["orders"],checkpoint=checkpoint,settlement_checkpoint=settlement)
    return eng,result,rows


def binding_constraints(su, weights, label=None):
    rows=[]
    for route in su.routes:
        b=su.cfg["r3"]["bounds"][route];weight=float(weights.get(route,0.))
        distance=min(abs(weight-b["lo"]),abs(b["hi"]-weight))
        # One executable lot is a rounding distance, not an economic parameter.
        lot=next((f.lot for f in su.fspecs if f.name==route),1.)
        tolerance=max(1e-9,su.unit_price[route]*lot/su.nav0)
        rows.append({**(label or {}),"route":route,"weight":weight,"lower_bound":b["lo"],"upper_bound":b["hi"],
                     "distance_to_bound":distance,"binding_YN":"Y" if distance<=tolerance else "N",
                     "reason_for_bound":b.get("why", ""),"executable_lot_tolerance":tolerance})
    return rows
