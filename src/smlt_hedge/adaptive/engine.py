"""Adaptive redesign R1 — the ONE quantity-based fund engine and the single risk authority.

One self-financing ledger per path, vectorised over N paths (N = 1 for the historical replay). Everything is held in
QUANTITIES: funded units (shares, fund units, CNY, BTC), money-market fund (MMF) units, integer derivative contracts,
immediate RUB cash and a settlement queue. Conventions (EXECUTION_ACCOUNTING_SPEC.md):

Clearing of day t (t = 0 is the decision close; no market move on day 0), in this order:
  1. pending settlements due at t move into immediate cash
  2. interest on positive immediate cash at `cash_rate[t]` (main case 0); a negative balance never accrues financing
  3. variation margin  VM_t = sum_j q_j (Fp_{j,t} - Fn_{j,t-1}) * pv_{j,t}   (contract held since t-1, RUB point value
     of day t), settled into cash
  4. contract rolls: q_j contracts closed at Fp_{j,t} and re-opened in the next contract at Fn_{j,t}; count preserved
     (integer), cost 2 |q_j| fee_{j,t}
  5. scheduled orders execute at the day-t close (funded at P_t, derivatives at Fn_t); funded cash legs enter the
     settlement queue at t + lag_t; derivative fees are debited immediately
  6. IM_t = mult * sum_j |q_j| Fn_{j,t} pv_{j,t} mr_{j,t}
  7. NAV_t = sum_f q_f P_{f,t} + cash_t + sum(pending)       (mark-to-market; derivatives are worth 0 after VM)
  8. AVAILABLE_t = cash_t - IM_t + min(0, min_k cumulative pending up to settlement day k)
     (receivables never count before they settle; payables are netted against same-day receivables only)
  9. margin call if AVAILABLE_t < 0: same-clearing waterfall of WHOLE derivative contracts in the pre-declared order
     (IM released immediately). Still < 0 -> UNRESOLVED DEFICIT: all derivatives closed, the path is flagged, no
     financing is created (a negative cash balance is recorded, earns/charges nothing and is covered only by later
     settlements).
 10. liquidity breach if AVAILABLE_t < beta * NAV_t (beta = 5%): restoration orders for the NEXT session (MMF units
     first, then funded risky assets, then derivatives, each in the pre-declared order) sized to reach the target
     reserve; proceeds arrive only after settlement.
 11. cash sweep: AVAILABLE_t above the band ceiling -> MMF purchase next session (never while a policy order is
     pending). The policy is consulted at the close BEFORE steps 10-11 so the cash rule sees its orders.

No function in this module uses information from a later close than the one being processed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import replace

import numpy as np
import pandas as pd
from . import semantics as D26

FUNDED, MMF, DERIV = "FUNDED", "MMF", "DERIVATIVE"


@dataclass(frozen=True)
class Instrument:
    name: str
    kind: str                     # FUNDED | MMF | DERIVATIVE
    lot: float = 1.0              # funded / MMF units per lot; 0 = divisible (owner-authorised academic BTC sleeve)
    cost_prop: float = 0.0        # funded: one-way cost as a fraction of traded value
    cost_unit: float = 0.0        # funded: one-way RUB cost per unit (half tick); derivatives use Market.fee
    issuer: str = ""              # issuer exposure (e.g. "SMLT")
    factor: str = ""              # factor label for exposure reports


@dataclass
class Market:
    """Price inputs for T = H+1 closes: funded (N,T,Af), derivatives (N,T,Ad).
    All price first dimensions equal N. `mr`, `roll`, `lag` and tradability
    are path-invariant (T,A*) calendars; `cash_rate` is (T,).
    """
    P: np.ndarray                 # funded + MMF unit prices (RUB)
    Fp: np.ndarray                # derivative: settle today of the contract held since yesterday (t >= 1)
    Fn: np.ndarray                # derivative: settle today of the contract held from today
    pv: np.ndarray                # derivative: RUB per price point per contract at clearing t
    mr: np.ndarray                # derivative initial-margin rate at t
    fee: np.ndarray               # derivative RUB per contract per side (exchange + broker + half tick)
    roll: np.ndarray              # derivative roll at t
    lag: np.ndarray               # funded settlement lag (sessions) for a trade executed at t
    tradable_f: np.ndarray        # funded / MMF may be traded at t
    tradable_d: np.ndarray        # derivative may be traded (opened / increased) at t
    cash_rate: np.ndarray         # daily rate on positive immediate cash
    contract: list | None = None  # optional (T, Ad) contract identifiers (history)
    execution: dict | None = None  # V2R2 causal availability, series, clocks and maintenance authority


def validate_market_path_dimensions(market):
    """Strict path/session schema, before any ledger or execution mutation.

    Prices are path indexed. Calendars and origin scalar metadata are explicitly
    path invariant; implicit singleton broadcasting is not execution authority.
    """
    from ..regime_v2.cfg import StopCondition
    def shape(name, value, expected):
        actual = np.asarray(value).shape
        if actual != expected:
            raise StopCondition(f'MARKET_PATH_DIMENSION_MISMATCH:{name}:expected={expected}:actual={actual}')
    if np.asarray(market.P).ndim != 3 or np.asarray(market.Fn).ndim != 3:
        raise StopCondition('MARKET_PATH_DIMENSION_MISMATCH:price_rank')
    N, T, Af = market.P.shape
    Ad = market.Fn.shape[2]
    shape('P', market.P, (N,T,Af))
    for name in ('Fp','Fn','pv','fee'):
        shape(name, getattr(market,name), (N,T,Ad))
    for name in ('mr','roll','tradable_d'):
        shape(name, getattr(market,name), (T,Ad))
    for name in ('lag','tradable_f'):
        shape(name, getattr(market,name), (T,Af))
    shape('cash_rate', market.cash_rate, (T,))
    if market.contract is not None:
        shape('contract', market.contract, (T,Ad))
    a = market.execution
    if a is None:
        return
    shape('execution.dates', a['dates'], (T,))
    shape('execution.active', a['active'], (T,Ad))
    shape('execution.capacity_f', a['capacity_f'], (T,Af))
    for name in ('mr_policy','mr_clearing','vm_lag','lk1'):
        shape('execution.'+name, a[name], (T,Ad))
    shape('execution.funded_keys', a['funded_keys'], (Af,))
    shape('execution.lk_units', a['lk_units'], (Ad,))
    if len(a['series']) != Ad:
        raise StopCondition('MARKET_PATH_DIMENSION_MISMATCH:execution.series:route_count')
    price_fields = ('mark','vm_mark','pv','fee')
    calendar_fields = {'capacity','executable','lk_classification_by_session'}
    for j, family in enumerate(a['series']):
        for key, data in sorted(family.items()):
            prefix = f'execution.series[{j}].{key}.'
            for name in price_fields:
                shape(prefix+name, data[name], (N,T))
            for name in ('capacity','executable'):
                shape(prefix+name, data[name], (T,))
            for name, value in data.items():
                if name == 'lk_classification_by_session':
                    shape(prefix+name, value, (T,))
                elif name not in set(price_fields) | calendar_fields and isinstance(value, np.ndarray) and value.ndim:
                    raise StopCondition('MARKET_PATH_DIMENSION_SCHEMA_UNKNOWN:'+prefix+name)


@dataclass
class Book:
    qf: np.ndarray                # (Af,) or (N, Af) funded + MMF units
    qd: np.ndarray                # (Ad,) or (N, Ad) integer contracts (signed)
    cash: float | np.ndarray      # immediate RUB cash
    pending: dict = field(default_factory=dict)   # {settlement day index: amount} (history only)
    series: list | None = None  # actual per-route SECID quantities at an admission origin
    capacity: dict | None = None  # remaining actual instrument/session tape at that origin
    regime: str | np.ndarray = 'ORDINARY'


@dataclass
class Params:
    beta: float                   # hard liquidity minimum (fraction of NAV) — REQUIRED, no default (D1-03)
    mult: float = 1.0             # margin multiplier (stress 1.25 / 1.50)
    reserve_frac: float | np.ndarray = 0.05   # VM-adjusted minimum = beta + VM99 (fraction of NAV)
    hysteresis: float = 0.05      # operating band above reserve_frac: floor +h/4, target +h/2, sweep ceiling +h
    sweep: bool = True
    wf_rank: tuple = ()           # same-day derivative cut order (derivative indices)
    restore_rank: tuple = ()      # restoration order over funded risky indices (MMF always first)
    log: bool = False
    issuer_floor: bool = False    # V2R1 primary: shares + futures share-equivalent units >= 0
    funded_floor: float = 0.0     # enforce only at scheduled target execution, never a nominal LP book
    emergency_floor_override: bool = False
    restoration_telemetry: bool = False  # V2R1 complete forced-cut accounting; legacy count semantics unchanged
    execution_check: object = None  # existing hard constraints evaluated on actual partial fills
    path_ids: tuple | None = None
    evaluation_context: str = ''
    continuous_check: object = None
    floor_names: tuple | None = None
    drift_check: object = None


@dataclass
class Order:
    day: int                      # engine day index at whose close the order executes
    qf: np.ndarray | None         # (N, Af) target funded/MMF units (None = unchanged)
    qd: np.ndarray | None         # (N, Ad) target integer contracts (None = unchanged)
    tag: str = "TARGET"
    delta: bool = False           # True: qf/qd are CHANGES applied to the quantities held at execution (clipped)
    plan: D26.ExecutionPlan | None = None
    stage_sequence: int = 0
    issue_session: int = 0
    stage_check: object = None


class Engine:
    def __init__(self, fin: list[Instrument], din: list[Instrument], mkt: Market, prm: Params):
        validate_market_path_dimensions(mkt)
        self.fin, self.din, self.m, self.p = fin, din, mkt, prm
        self.execution = None
        self.Af, self.Ad = len(fin), len(din)
        self.T = mkt.P.shape[1] if self.Af else mkt.Fn.shape[1]
        self.N = max(mkt.P.shape[0] if self.Af else 1, mkt.Fn.shape[0] if self.Ad else 1)
        self.mmf = np.array([i.kind == MMF for i in fin], bool)
        self.lot = np.array([i.lot for i in fin], float)
        self.cprop = np.array([i.cost_prop for i in fin], float)
        self.cunit = np.array([i.cost_unit for i in fin], float)

    # ------------------------------------------------------------------ helpers
    def _P(self, t):
        return np.broadcast_to(self.m.P[:, t, :], (self.N, self.Af))

    def _Fn(self, t):
        return np.broadcast_to(self.m.Fn[:, t, :], (self.N, self.Ad))

    def _Fp(self, t):
        return np.broadcast_to(self.m.Fp[:, t, :], (self.N, self.Ad))

    def _pv(self, t):
        return np.broadcast_to(self.m.pv[:, t, :], (self.N, self.Ad))

    def _fee(self, t):
        return np.broadcast_to(self.m.fee[:, t, :], (self.N, self.Ad))

    def im(self, qd, t):
        if not self.Ad:
            return np.zeros(self.N)
        if getattr(self, "execution", None) is not None:
            return self.execution.margin()
        return self.p.mult * (np.abs(qd) * self._Fn(t) * self._pv(t) * self.m.mr[t][None, :]).sum(axis=1)

    def available(self, cash, pend, qd, t):
        """cash - IM + worst cumulative net pending over future settlement days (receivables only offset payables
        due on or before the same day)."""
        fut = pend[:, t + 1:]
        worst = np.minimum(0.0, np.cumsum(fut, axis=1).min(axis=1)) if fut.shape[1] else np.zeros(self.N)
        return cash - self.im(qd, t) + worst

    def funded_cost(self, dq, P):
        return np.abs(dq) * (self.cprop[None, :] * P + self.cunit[None, :])

    # ------------------------------------------------------------------ run
    def run(self, book: Book, orders: list[Order] | None = None, policy=None, checkpoint=None, settlement_checkpoint=None) -> dict:
        """`orders`: pre-scheduled target orders (admission of a static target). `policy(t, state) -> list[Order]`
        is called at every close with the state observed at that close (historical replay)."""
        validate_market_path_dimensions(self.m)
        N, T, Af, Ad, p, m = self.N, self.T, self.Af, self.Ad, self.p, self.m
        L = T + 8
        qf = np.broadcast_to(np.asarray(book.qf, float), (N, Af)).copy()
        qd = np.broadcast_to(np.asarray(book.qd, float), (N, Ad)).copy()
        cash = np.broadcast_to(np.asarray(book.cash, float), (N,)).copy()
        self.execution = None
        residual_f, residual_d = np.full_like(qf, np.nan), np.full_like(qd, np.nan)
        if m.execution is not None:
            from ..regime_v2.execution import Execution
            self.execution = Execution(self, qd, book.series, book.capacity)
        pend = np.zeros((N, L))
        for k, v in (book.pending or {}).items():
            pend[:, int(k)] += v
        sched: dict[int, list[Order]] = {}
        for o in orders or []:
            sched.setdefault(o.day, []).append(o)
        maintenance_decisions = []
        if self.execution is not None:
            dates = pd.DatetimeIndex(m.execution['dates'])
            for row in m.execution.get('qconv', []):
                decision = pd.Timestamp(row['maintenance_decision_session'])
                execution = pd.Timestamp(row['exit_execution_session'])
                if execution in dates and m.execution.get('cny_indices'):
                    target = np.full((N,Ad),np.nan)
                    target[:,m.execution['cny_indices']] = 0.
                    day = int(dates.get_loc(execution))
                    sched.setdefault(day, []).append(Order(day,None,target,'CONTRACT_MAINTENANCE'))
                    maintenance_decisions.append({'decision_session':str(decision.date()),
                        'execution_session':str(execution.date()), 'event_session':row['event_session'],
                        'target':'CNY_PERP=0', 'classification':'OPERATIONAL_MAINTENANCE_NOT_ALPHA'})
        reserve = np.broadcast_to(np.asarray(p.reserve_frac, float), (N,)).copy()
        origin_nav = (qf*self._P(0)).sum(axis=1)+cash+pend.sum(axis=1)
        local = D26.LocalResiduals(qf.copy(), qd.copy(), origin_nav)
        path_ids = tuple(p.path_ids) if p.path_ids is not None else tuple(D26.identity({
            'P': m.P[n], 'Fn': m.Fn[n], 'Fp': m.Fp[n], 'pv': m.pv[n], 'fee': m.fee[n]}) for n in range(N))
        evaluation_ids = tuple(D26.identity({'path': key, 'context':p.evaluation_context,
                                           'multiplier':p.mult, 'semantics':D26.VERSION}) for key in path_ids)
        callback_floor_keys = []
        floor_assets = np.array([i.name in p.floor_names if p.floor_names is not None else i.kind!=MMF for i in self.fin],bool)
        continuous_safety = np.ones(N, bool)
        drift_diagnostic = np.zeros(N, bool)
        severe_records = []
        active_regime=np.broadcast_to(np.asarray(book.regime),(N,)).astype('<U32').copy()
        # outputs
        nav = np.empty((N, T))
        avail = np.empty((N, T))
        imp = np.empty((N, T))
        cashp = np.empty((N, T))
        mmfv = np.empty((N, T))
        pendp = np.empty((N, T))
        breach = np.zeros(N, bool)
        breach_days = np.zeros(N, int)
        unresolved = np.zeros(N, bool)
        call_days = np.zeros(N, int)
        forced_contracts = np.zeros(N)
        forced_events = np.zeros(N, int)
        forced_funded_rub = np.zeros(N)
        costs = np.zeros(N)
        turnover = np.zeros(N)
        pnl_f = np.zeros((N, Af))            # price P&L of funded units (incl. MMF carry)
        pnl_d = np.zeros((N, Ad))            # VM per derivative
        interest = np.zeros(N)
        cost_roll = np.zeros(N)
        n_roll_contracts = np.zeros(N)
        gross = np.empty((N, T))
        frac_violation = np.zeros(N, int)
        rows = []
        execution_records = []
        clearing_records = []
        issuer_min = np.full(N, np.inf)
        actual_constraints_pass = np.ones(N,bool)
        emergency_overrides = np.zeros(N, int)
        restore_q: dict[int, list[Order]] = {}
        P_prev = self._P(0).copy() if Af else None
        Fn_prev = self._Fn(0).copy() if Ad else None

        for t in range(T):
            self._current_day=t
            if self.execution is not None:
                self.execution.begin(t)
            P = self._P(t) if Af else np.zeros((N, 0))
            Fn = self._Fn(t) if Ad else np.zeros((N, 0))
            vm_t = np.zeros(N)
            rc_t = np.zeros(N)
            it_t = np.zeros(N)
            if t > 0:
                # 1. settlements
                cash += pend[:, t]
                pend[:, t] = 0.0
                if settlement_checkpoint is not None:
                    settlement_checkpoint(t, {"qf":qf.copy(), "qd":qd.copy(), "cash":cash.copy(),
                                              "pending":pend.copy(), "costs":costs.copy()})
                # 2. interest on positive immediate cash only
                it_t = np.maximum(cash, 0.0) * m.cash_rate[t]
                cash += it_t
                interest += it_t
                # funded price P&L (MTM attribution)
                if Af:
                    pnl_f += qf * (P - P_prev)
                # 3. variation margin on contracts held since t-1
                if Ad:
                    vmj = self.execution.vm() if self.execution is not None else qd * (self._Fp(t) - Fn_prev) * self._pv(t)
                    vm_t = vmj.sum(axis=1)
                    pnl_d += vmj
                    if self.execution is None:
                        cash += vm_t
                    else:
                        for j in range(Ad):
                            lag_vm = int(m.execution['vm_lag'][t, j])
                            if lag_vm:
                                pend[:, t+lag_vm] += vmj[:, j]
                            else:
                                cash += vmj[:, j]
                    # 4. rolls (count preserved)
                    rj = m.roll[t]
                    if self.execution is None and rj.any():
                        if p.issuer_floor:
                            previous_qd = qd.copy()
                            qd, guard_cost, guard_turnover = self._protect_issuer(t, qf, qd)
                            cash -= guard_cost
                            costs += guard_cost
                            turnover += guard_turnover
                            forced_contracts += np.maximum(np.abs(previous_qd)-np.abs(qd),0).sum(axis=1)
                            forced_events += (np.abs(previous_qd-qd)>0).any(axis=1)
                        c = (2 * np.abs(qd) * self._fee(t) * rj[None, :]).sum(axis=1)
                        rc_t = c
                        cash -= c
                        costs += c
                        cost_roll += c
                        n_roll_contracts += (np.abs(qd) * rj[None, :]).sum(axis=1)
            if self.execution is not None:
                rc_t, rolled = self.execution.rolls()
                qd = self.execution.quantity()
                cash -= rc_t
                costs += rc_t
                cost_roll += rc_t
                n_roll_contracts += rolled
            if 'SMLT' in [i.name for i in self.fin] or 'SMLT_FUT' in [i.name for i in self.din]:
                issuer_min=np.minimum(issuer_min,self.issuer_units(qf,qd,t))
            if self.execution is not None and p.issuer_floor:
                issuer_min = np.minimum(issuer_min,self.issuer_units(qf,qd,t))
                before = qd.copy()
                qd, guard_cost, guard_turn = self._protect_issuer(t, qf, qd)
                cash -= guard_cost; costs += guard_cost; turnover += guard_turn
                closed = np.maximum(np.abs(before)-np.abs(qd),0.)
                forced_contracts += closed.sum(axis=1)
                forced_events += (closed > 0).any(axis=1)
            clearing_before = None
            if p.log and N == 1:
                clearing_before = {"qf":qf[0].copy(), "qd":qd[0].copy(), "cash":float(cash[0]),
                                   "pending":float(pend[:,t+1:].sum()), "margin":float(self.im(qd,t)[0]),
                                   "costs":float(costs[0])}
            # 5. scheduled + restoration orders (policy decisions made at earlier closes)
            # Local absolute residuals retain their issuing plan and stage. Safety
            # deltas are converted once; scheduled directives retain frozen priority.
            restoration_due = restore_q.pop(t, [])
            scheduled_due = sched.get(t, [])
            for o in restoration_due:
                local.install(o,t,qf,qd)
            executed_orders = []
            def local_attempts():
                attempted=set()
                # Distinct scheduled Order objects keep their declared order;
                # each complete local attempt gets its own strict checkpoint.
                for directive in scheduled_due:
                    idx=local.install(directive,t,qf,qd)
                    attempted.add(idx)
                    yield idx
                if self.execution is not None:
                    for j in self.execution.maintenance():
                        v=np.full_like(qd,np.nan)
                        missing=(local.owner_d[:,j]<0)&(qd[:,j]!=0)
                        v[missing,j]=0.
                        if missing.any():local.install(Order(t,None,v,'CONTRACT_MAINTENANCE'),t,qf,qd)
                for idx in local.attempts():
                    if idx not in attempted:yield idx
            for idx in local_attempts():
                item = local.directives[idx]
                tf, td, attempt = local.targets(idx,qf,qd)
                if not attempt.any():continue
                o = replace(item['order'], day=t, qf=tf, qd=td, delta=False)
                self._directive_d_mask=local.owner_d==idx
                executed_orders.append(o)
                pre_f, pre_d = qf.copy(), qd.copy()
                record_start = len(self.execution.records) if self.execution is not None else 0
                previous_qd = qd.copy() if o.tag == "RESTORE" else None
                self._costs_before_order=costs.copy()
                cash, pend, qf, qd, c, to, ff = self._execute(o, t, cash, pend, qf, qd)
                costs += c
                turnover += to
                stage_nav = (qf*P).sum(axis=1)+cash+pend[:,t+1:].sum(axis=1)
                numerator = (qf[:,floor_assets]*P[:,floor_assets]).sum(axis=1)
                ordinary = attempt & (o.tag not in D26.SAFETY) & ~np.full(N,p.emergency_floor_override)
                floor_mask = ordinary & (p.funded_floor > 0)
                signed = self.execution.signed_notional() if self.execution else qd*Fn*self._pv(t)
                dgross = self.execution.gross() if self.execution else np.abs(signed).sum(axis=1)
                filled = (pre_f != qf).any(axis=1) | (pre_d != qd).any(axis=1)
                if self.execution is not None:
                    for leg in self.execution.records[record_start:]:
                        filled |= leg['filled'] != 0
                stage = D26.StageEvaluation(item['ids'],evaluation_ids,path_ids,t,o.stage_sequence,o.tag,
                    *map(D26.frozen,(attempt,ordinary,floor_mask,pre_f,pre_d,qf,qd,qf-pre_f,qd-pre_d,P,signed,dgross,
                        c,cash,pend[:,t+1:],self.im(qd,t),stage_nav,item['nav'],self.issuer_units(qf,qd,t),numerator,
                        np.isfinite(stage_nav)&(stage_nav>0)&(numerator/np.maximum(stage_nav,1e-12)>=p.funded_floor-1e-12))),
                    flags=('SCHEDULED' if item['issued']==t else 'RESIDUAL',
                           'TERMINAL' if t==T-1 else 'LOCAL_ATTEMPT'),
                    actual_derivative_legs=tuple(tuple((k,D26.frozen(v) if isinstance(v,np.ndarray) else v)
                        for k,v in sorted(leg.items())) for leg in self.execution.records[record_start:]) if self.execution else ())
                stage_check = o.stage_check or p.execution_check
                if stage_check is not None:
                    result = stage_check(stage)
                    if isinstance(result,tuple):
                        passed,rules,*details = result
                        stage = replace(stage,rule_results=rules,rule_details=details[0] if details else ())
                    else:
                        passed=result
                    actual_constraints_pass &= np.asarray(passed,bool)
                actual_constraints_pass &= ~floor_mask | stage.floor_predicate
                callback_floor_keys.extend(stage.floor_keys)
                if ordinary.any():
                    execution_records.append(stage)
                if o.tag == 'SEVERE_ISSUER_EXIT':
                    zero=stage.issuer_units==0
                    fs,ds=[i.name for i in self.fin],[i.name for i in self.din]
                    if 'SMLT' in fs:zero &= qf[:,fs.index('SMLT')]==0
                    if 'SMLT_FUT' in ds:zero &= qd[:,ds.index('SMLT_FUT')]==0
                    severe_records.append({'day':t,'plan_ids':item['ids'],
                        'remaining_units':D26.frozen(stage.issuer_units), 'actual_qf':D26.frozen(qf),
                        'actual_qd':D26.frozen(qd),'status':tuple('N/A' if not applicable else 'COMPLETED' if done else
                        'SEVERE_EXIT_DUE_BLOCKED' for done,applicable in zip(zero,attempt))})
                if p.emergency_floor_override or o.tag in D26.SAFETY:
                    emergency_overrides += attempt & (numerator/np.maximum(stage_nav,1e-12)<p.funded_floor-1e-12)
                local.finish(idx,t,qf,qd,filled,{'stage':stage,'costs':D26.frozen(costs),
                    'turnover':D26.frozen(turnover),'pending':D26.frozen(pend),'session':t,
                    'series':[{k:D26.frozen(v) for k,v in group.items()} for group in self.execution.positions] if self.execution else None})
                if o.tag=='EMERGENCY_TARGET' or (o.plan and o.plan.regime=='EMERGENCY'):
                    active_regime[attempt]='EMERGENCY'
                elif o.tag=='SEVERE_ISSUER_EXIT':
                    active_regime[attempt]='ISSUER_SAFETY'
                else:
                    for n in np.flatnonzero(ordinary & actual_constraints_pass):
                        if local.states.get((item['ids'][n],int(n)))=='COMPLETED':active_regime[n]='ORDINARY'
                residual_f, residual_d = local.residual_f, local.residual_d
                if o.tag == "RESTORE":
                    forced_funded_rub += ff
                    if p.restoration_telemetry:
                        forced_contracts += np.maximum(np.abs(previous_qd)-np.abs(qd),0).sum(axis=1)
                        forced_events += (np.abs(previous_qd-qd)>0).any(axis=1)
            # 6-8. IM, NAV, available
            navt = (qf * P).sum(axis=1) + cash + pend[:, t + 1:].sum(axis=1)
            av = self.available(cash, pend, qd, t)
            # 9. margin call -> same-clearing derivative waterfall (whole contracts)
            call = av < 0
            if call.any() and Ad:
                call_days += call
                cash, qd, nc, ev, fc = self._waterfall_same_day(t, call, cash, pend, qd)
                forced_contracts += nc
                forced_events += ev
                costs += fc                                  # R1-A2: fees of forced cuts are costs
                av = self.available(cash, pend, qd, t)
                navt = (qf * P).sum(axis=1) + cash + pend[:, t + 1:].sum(axis=1)
            unres = av < -1e-6
            if unres.any():
                unresolved |= unres
                if Ad and self.execution is not None:
                    before = qd.copy()
                    for j in range(Ad):
                        cj, tj = self.execution.target(j, np.where(unres,0.,qd[:,j]), 'CLOSE_ALL')
                        cash -= cj; costs += cj; turnover += tj
                    qd = self.execution.quantity()
                    closed = np.maximum(np.abs(before)-np.abs(qd),0.)
                    forced_contracts += closed.sum(axis=1)
                    forced_events += (closed > 0).any(axis=1)
                if Ad and self.execution is None:   # close everything that still posts margin; no financing is created
                    fee = self._fee(t)
                    c = (np.abs(qd) * fee).sum(axis=1) * unres
                    forced_contracts += np.abs(qd).sum(axis=1) * unres
                    qd = np.where(unres[:, None], 0.0, qd)
                    cash -= c
                    costs += c
                av = self.available(cash, pend, qd, t)
                navt = (qf * P).sum(axis=1) + cash + pend[:, t + 1:].sum(axis=1)
            if clearing_before is not None and (executed_orders or float(costs[0]) != clearing_before["costs"]):
                clearing_records.append({"day":t,"before":clearing_before,"qf_after":qf[0].copy(),"qd_after":qd[0].copy(),
                                         "P":P[0].copy(),"cash_after":float(cash[0]),"pending_after":float(pend[:,t+1:].sum()),
                                         "margin_after":float(self.im(qd,t)[0]),"trade_costs":float(costs[0])-clearing_before["costs"],
                                         "tags":[o.tag for o in executed_orders],"VM_received_before_orders":float(vm_t[0])})
            # policy callback FIRST (R1-A3): decision at close t; its orders carry their own execution day, and the
            # cash rule below must know about them (no sweep while a policy order is pending)
            if policy is not None:
                state = {"t": t, "qf": qf.copy(), "qd": qd.copy(), "cash": cash.copy(), "pend": pend.copy(),
                         "nav": navt.copy(), "available": av.copy(), "im": self.im(qd, t),"regime":active_regime.copy(),
                         "series": [{k:v.copy() for k,v in group.items()} for group in self.execution.positions] if self.execution else None,
                         "capacity": {k:v.copy() for k,v in self.execution.budget.items()} if self.execution else None}
                dec = policy(t, state) or {}
                if dec.get("reserve") is not None:
                    reserve = np.broadcast_to(np.asarray(dec["reserve"], float), (N,)).copy()
                if dec.get("wf_rank") is not None:                 # decision-time waterfall ranking (never path-wise)
                    p.wf_rank = tuple(dec["wf_rank"])
                if dec.get("restore_rank") is not None:
                    p.restore_rank = tuple(dec["restore_rank"])
                for o in dec.get("orders", []):
                    if o.day <= t:
                        raise ValueError("an order cannot execute at or before the close that decided it")
                    sched.setdefault(o.day, []).append(o)
            if self.execution is not None:
                # Information-cut decision has finished. Late margin changes
                # can create a held-book breach but cannot trigger a same-close
                # rescue; restoration uses the next eligible session.
                self.execution.margin_clock = 'clearing'
                av = self.available(cash, pend, qd, t)
                late_unres = av < -1e-6
                unresolved |= late_unres
                unres |= late_unres
            # 10. breach of the hard 5% rule -> restoration for the NEXT session
            br = av < p.beta * navt - 1e-9
            breach |= br
            breach_days += br
            low = av < (reserve + p.hysteresis / 4) * navt - 1e-9          # replenish below the floor
            if t + 1 < T:
                if low.any():
                    o = self._restoration_order(t, low, av, navt, reserve + p.hysteresis / 2, qf, qd, P, Fn)
                    if o is not None:
                        restore_q.setdefault(t + 1, []).append(o)
                if p.sweep and Af and self.mmf.any():
                    excess = av - (reserve + p.hysteresis) * navt          # above the ceiling
                    pending_local = np.zeros(N,bool)
                    for idx in local.attempts():
                        if local.directives[idx]['order'].tag not in D26.SAFETY:
                            pending_local |= (local.owner_f==idx).any(axis=1)|(local.owner_d==idx).any(axis=1)
                    for day, future in sched.items():
                        if day <= t: continue
                        for directive in future:
                            if directive.qf is not None:
                                pending_local |= np.isfinite(np.broadcast_to(directive.qf,(N,Af))).any(axis=1)
                            if directive.qd is not None:
                                pending_local |= np.isfinite(np.broadcast_to(directive.qd,(N,Ad))).any(axis=1)
                    sw = (excess > 0) & ~low & ~pending_local
                    if sw.any() and m.tradable_f[t + 1][self.mmf].all():
                        j = int(np.flatnonzero(self.mmf)[0])
                        dq = np.zeros_like(qf)
                        tgt_av = (reserve + p.hysteresis / 2) * navt                # sweep down to the target
                        add = np.floor((av - tgt_av) / np.maximum(P[:, j] * self.lot[j], 1e-12)) * self.lot[j]
                        dq[:, j] = np.where(sw, np.maximum(add, 0), 0.0)
                        restore_q.setdefault(t + 1, []).append(Order(t + 1, dq, None, "SWEEP", delta=True))
            if "SMLT" in [i.name for i in self.fin] and "SMLT_FUT" in [i.name for i in self.din]:
                jf = [i.name for i in self.fin].index("SMLT")
                jd = [i.name for i in self.din].index("SMLT_FUT")
                units = self.issuer_units(qf,qd,t)
                issuer_min = np.minimum(issuer_min, units)
                if self.execution is None and p.issuer_floor and (units < -1e-9).any():
                    raise ValueError("primary issuer floor violated in recursive ledger")
            continuous_safety &= (~unresolved & np.isfinite(navt) & (navt>0)
                                  & (issuer_min>=-1e-9 if p.issuer_floor else True))
            if p.continuous_check is not None:
                continuous_safety &= p.continuous_check(qf,qd,P,self,navt)
            fraction = (qf[:,~self.mmf]*P[:,~self.mmf]).sum(axis=1)/np.maximum(navt,1e-12)
            drift_diagnostic |= (fraction < p.funded_floor-1e-12)
            if p.drift_check is not None:
                drift_diagnostic |= p.drift_check(qf,qd,P,self,navt,origin_nav)
            # integer check (derivative contracts must be whole)
            frac_violation += (np.abs(qd - np.round(qd)) > 1e-9).any(axis=1)
            nav[:, t], avail[:, t], imp[:, t], cashp[:, t] = navt, av, self.im(qd, t), cash
            mmfv[:, t] = (qf[:, self.mmf] * P[:, self.mmf]).sum(axis=1) if Af else 0.0
            pendp[:, t] = pend[:, t + 1:].sum(axis=1)
            gross[:, t] = (np.abs(qf[:, ~self.mmf]) * P[:, ~self.mmf]).sum(axis=1) + (
                (self.execution.gross() if self.execution is not None else (np.abs(qd) * Fn * self._pv(t)).sum(axis=1)) if Ad else 0.0)
            if p.log and N == 1:
                rows.append(self._row(t, qf, qd, cash, pend, P, Fn, navt, av, vm_t, rc_t, it_t, br, unres))
                if self.execution is not None:
                    rows[-1]['issuer_cap_scope'] = 'DIRECT_SMLT_ONLY'
                    rows[-1]['indirect_smlt_exposure'] = 'UNKNOWN'
                    rows[-1]['issuer_floor_breach'] = bool(p.issuer_floor and np.any(issuer_min < -1e-9))
                    rows[-1]['capacity_convention'] = 'MAXIMUM_TAPE_CONSISTENCY_BOUND'
                rows[-1]["cost_total_cum"] = float(costs[0])
                rows[-1]["price_pnl_cum"] = float(pnl_f[0].sum())
                rows[-1]["vm_cum"] = float(pnl_d[0].sum())
                rows[-1]["interest_cum"] = float(interest[0])
                fraction = float((qf[:,~self.mmf]*P[:,~self.mmf]).sum()/max(float(navt[0]),1e-12))
                rows[-1]["funded_non_mmf_fraction"] = fraction
                rows[-1]["emergency_invested_floor_override"] = bool(fraction < p.funded_floor-1e-12 and
                    (p.emergency_floor_override or any(o.tag in ("RESTORE","EMERGENCY_TARGET") for o in executed_orders)))
                if "SMLT" in [i.name for i in self.fin] and "SMLT_FUT" in [i.name for i in self.din]:
                    rows[-1]["net_smlt_issuer_units"] = float(units[0])
                    rows[-1]["smlt_issuer_spot_price"] = float(P[0,jf])
            if checkpoint is not None:
                checkpoint(t,{"qf":qf.copy(),"qd":qd.copy(),"cash":cash.copy(),"pending":pend.copy(),
                              "nav":navt.copy(),"margin":self.im(qd,t),"reserve":reserve.copy(),
                              "series": [{k:v.copy() for k,v in group.items()} for group in self.execution.positions] if self.execution else None,
                              "capacity": {k:v.copy() for k,v in self.execution.budget.items()} if self.execution else None,
                              "scheduled_orders":sched,"restoration_orders":restore_q,"costs":costs.copy(),
                              "turnover":turnover.copy(),"funded_pnl":pnl_f.copy(),"derivative_pnl":pnl_d.copy(),
                              "decision_count_context_required":True,"available":av.copy(),"breach":br.copy(),
                              "unresolved":unres.copy()})
            if Af:
                P_prev = P.copy()
            if Ad:
                Fn_prev = Fn.copy()

        out = {"nav": nav, "available": avail, "im": imp, "cash": cashp, "mmf_value": mmfv, "pending": pendp,
               "gross": gross, "breach": breach, "breach_days": breach_days, "unresolved": unresolved,
               "call_days": call_days, "forced_contracts": forced_contracts, "forced_events": forced_events,
               "forced_funded_rub": forced_funded_rub, "costs": costs, "turnover": turnover, "pnl_funded": pnl_f,
               "pnl_deriv": pnl_d, "interest": interest, "cost_roll": cost_roll, "roll_contracts": n_roll_contracts,
               "frac_contract_days": frac_violation, "qf_end": qf, "qd_end": qd, "cash_end": cash,
               "pending_end": pend[:, T:].sum(axis=1)}
        if self.execution is not None:
            out['lk_records'] = self.execution.lk_records
            out['fill_records'] = self.execution.records
            out['series_positions_end'] = self.execution.positions
            out['residual_target_f'] = residual_f
            out['residual_target_d'] = residual_d
            out['issuer_floor_breach'] = issuer_min < -1e-9
        local.expire(T-1)
        out['plan_states'] = local.states
        out['plan_history'] = local.histories
        out['plan_completions'] = local.completions
        out['residual_owners_f'],out['residual_owners_d'] = local.owner_f,local.owner_d
        out['pending_ledger_end'] = D26.frozen(pend)
        out['residual_directives'] = local.directives
        out['callback_floor_keys'] = tuple(callback_floor_keys)
        out['record_floor_keys'] = tuple(k for stage in execution_records for k in stage.floor_keys)
        out['continuous_safety_pass'] = continuous_safety
        out['allocation_drift_diagnostic'] = drift_diagnostic
        out['severe_exit_records'] = severe_records
        out['active_regime'] = active_regime
        out["maintenance_decisions"] = maintenance_decisions
        out["execution_records"] = execution_records
        out["clearing_records"] = clearing_records
        out["actual_constraints_pass"] = actual_constraints_pass
        out["min_net_smlt_units"] = issuer_min
        out["emergency_floor_override_count"] = emergency_overrides
        if p.log and N == 1:
            out["log"] = rows
        return out

    # ------------------------------------------------------------------ execution
    def issuer_units(self, qf, qd, t):
        fs,ds = [i.name for i in self.fin],[i.name for i in self.din]
        units = qf[:,fs.index('SMLT')].copy() if 'SMLT' in fs else np.zeros(self.N)
        if 'SMLT_FUT' in ds:
            j = ds.index('SMLT_FUT')
            units += (self.execution.share_equivalents(j) if self.execution is not None
                      else qd[:,j]*self._pv(t)[:,j])
        return units

    def _execute(self, o: Order, t: int, cash, pend, qf, qd):
        N, m = self.N, self.m
        c = np.zeros(N)
        to = np.zeros(N)
        ff = np.zeros(N)
        if o.qf is not None and self.Af:
            P = self._P(t)
            tgt = np.broadcast_to(np.asarray(o.qf, float), (N, self.Af))
            if o.delta:                       # restoration / sweep: never sell more than is held
                tgt = np.maximum(qf + tgt, np.minimum(qf, 0.0))
            else:                             # NaN entries = leave that holding unchanged (e.g. MMF managed by sweep)
                tgt = np.where(np.isnan(tgt), qf, tgt)
            ok = m.tradable_f[t][None, :]
            dq = np.where(ok, tgt - qf, 0.0)
            if self.execution is not None:
                for j in range(self.Af):
                    amount = np.minimum(np.abs(dq[:,j]),self.execution.budget[self.execution.a['funded_keys'][j]])
                    if self.lot[j] > 0:
                        amount = np.floor(amount/self.lot[j]+1e-12)*self.lot[j]
                    dq[:,j] = np.sign(dq[:,j])*amount
            if self.p.issuer_floor:
                protected = qf + dq
                qd, guard_cost, guard_turn = self._protect_issuer(t, protected, qd)
                cash -= guard_cost
                c += guard_cost
                to += guard_turn
                if self.execution is not None:
                    fs, ds = [i.name for i in self.fin], [i.name for i in self.din]
                    if 'SMLT' in fs and 'SMLT_FUT' in ds:
                        jf, jd = fs.index('SMLT'), ds.index('SMLT_FUT')
                        minimum = np.maximum(-self.execution.share_equivalents(jd), 0.)
                        dq[:, jf] = np.maximum(dq[:, jf], np.minimum(0., minimum-qf[:, jf]))
            if self.execution is not None:
                dq = np.stack([self.execution.funded(j, dq[:, j]) for j in range(self.Af)], axis=1)
            val = dq * P
            cst = self.funded_cost(dq, P)
            lag = m.lag[t]
            for j in range(self.Af):
                flow = -(val[:, j] + cst[:, j])
                if lag[j] <= 0:
                    cash += flow
                else:
                    pend[:, t + int(lag[j])] += flow
            qf = qf + dq
            c += cst.sum(axis=1)
            to += np.abs(val[:, ~self.mmf]).sum(axis=1)
            ff += np.where(dq[:, ~self.mmf] < 0, -val[:, ~self.mmf], 0.0).sum(axis=1)
        if o.qd is not None and self.Ad:
            tgt = np.broadcast_to(np.asarray(o.qd, float), (N, self.Ad))
            if o.delta:                       # reductions only: move toward zero, never through it
                opp = np.sign(tgt) == -np.sign(qd)
                tgt = np.where(opp, qd - np.sign(qd) * np.minimum(np.abs(qd), np.abs(tgt)), qd)
            if (np.abs(tgt - np.round(tgt)) > 1e-9).any():
                raise ValueError("fractional derivative target")
            if self.p.issuer_floor:
                if self.execution is None:
                    tgt, _, _ = self._protect_issuer(t, qf, tgt)
                else:
                    fs, ds = [i.name for i in self.fin], [i.name for i in self.din]
                    if 'SMLT' in fs and 'SMLT_FUT' in ds:
                        jf, jd = fs.index('SMLT'), ds.index('SMLT_FUT')
                        tgt = tgt.copy()
                        equivalent = self.execution.share_multiplier(jd,self.execution.active(t,jd))
                        current_units = self.execution.share_equivalents(jd)
                        permitted = qd[:,jd]+np.floor((qf[:,jf]+current_units)/equivalent+1e-12)
                        tgt[:, jd] = np.maximum(tgt[:, jd], qd[:,jd]-np.maximum(permitted-qd[:,jd],0.))
            ok = m.tradable_d[t][None, :]
            # an untradable contract may only be reduced toward zero (e.g. before listing: stays 0)
            red = np.abs(tgt) < np.abs(qd)
            dq = np.where(ok | red, tgt - qd, 0.0) if self.execution is None else np.where(ok, tgt-qd, 0.)
            if self.execution is not None:
                for j in range(self.Ad):
                    self._fill_scope=getattr(self,'_directive_d_mask',np.ones_like(qd,bool))[:,j]
                    cj, tj = self.execution.target(j, qd[:, j]+dq[:, j], o.tag)
                    self._fill_scope=None
                    cash -= cj; c += cj; to += tj
                qd = self.execution.quantity()
                return cash, pend, qf, qd, c, to, ff
            fee = self._fee(t)
            cst = (np.abs(dq) * fee).sum(axis=1)
            cash -= cst
            qd = qd + dq
            if self.p.issuer_floor:
                qd, guard_cost, guard_turn = self._protect_issuer(t, qf, qd)
                cash -= guard_cost
                c += guard_cost
                to += guard_turn
            c += cst
            to += (np.abs(dq) * self._Fn(t) * self._pv(t)).sum(axis=1)
        return cash, pend, qf, qd, c, to, ff

    def _protect_issuer(self, t, qf, qd):
        """Close whole protective contracts before selling protected shares; charge every closure."""
        fs, ds = [i.name for i in self.fin], [i.name for i in self.din]
        if "SMLT" not in fs or "SMLT_FUT" not in ds:
            return qd, np.zeros(self.N), np.zeros(self.N)
        jf, jd = fs.index("SMLT"), ds.index("SMLT_FUT")
        equivalent = (np.full(self.N,self.execution.share_multiplier(jd,self.execution.active(t,jd)))
                      if self.execution is not None else self._pv(t)[:, jd])
        if (equivalent <= 0).any():
            raise ValueError("non-positive SMLT contract share multiplier")
        if self.execution is not None:
            deficit = np.maximum(-(qf[:,jf]+self.execution.share_equivalents(jd)),0.)
            target = qd[:,jd]+np.ceil(deficit/equivalent-1e-12)
        else:
            limit = np.floor(np.maximum(qf[:, jf], 0.0) / equivalent + 1e-12)
            target = np.maximum(qd[:, jd], -limit)
        closed = target - qd[:, jd]
        if self.execution is not None:
            cost, turnover = self.execution.target(jd, target, 'ISSUER_PROTECTION')
            return self.execution.quantity(), cost, turnover
        qd = qd.copy()
        qd[:, jd] = target
        return qd, closed * self._fee(t)[:, jd], closed * self._Fn(t)[:, jd] * equivalent

    def _waterfall_same_day(self, t, call, cash, pend, qd):
        N, p = self.N, self.p
        Fn, pv, fee = self._Fn(t), self._pv(t), self._fee(t)
        rel = p.mult * Fn * pv * self.m.mr[t][None, :] - fee        # IM released per contract, net of cost
        order = list(p.wf_rank) if p.wf_rank else list(range(self.Ad))
        nc = np.zeros(N)
        ev = np.zeros(N, int)
        fc = np.zeros(N)
        for j in order:
            av = self.available(cash, pend, qd, t)
            need = np.where(call & (av < 0), -av, 0.0)
            if not (need > 0).any():
                break
            r = rel[:, j]
            n = np.where((need > 0) & (r > 0), np.minimum(np.abs(qd[:, j]), np.ceil(need / np.maximum(r, 1e-12))), 0.0)
            if self.execution is not None:
                before = qd[:, j].copy()
                cj, _ = self.execution.target(j, qd[:, j]-np.sign(qd[:, j])*n, 'WATERFALL')
                qd = self.execution.quantity()
                actual = np.abs(before-qd[:, j])
                cash -= cj; fc += cj; nc += actual; ev += (actual > 0).astype(int)
                continue
            qd[:, j] -= np.sign(qd[:, j]) * n
            cash -= n * fee[:, j]
            fc += n * fee[:, j]
            nc += n
            ev += (n > 0).astype(int)
        return cash, qd, nc, ev, fc

    def _restoration_order(self, t, br, av, navt, reserve, qf, qd, P, Fn):
        """Next-session orders restoring AVAILABLE to the target reserve: MMF first, then funded risky assets in the
        pre-declared order, then derivatives in the waterfall order. Sized with day-t prices (known at close t)."""
        N, p = self.N, self.p
        short = np.where(br, reserve * navt - av, 0.0)
        tf = qf.copy()
        td = qd.copy()
        if self.Af:
            order = [int(j) for j in np.flatnonzero(self.mmf)] + [int(j) for j in p.restore_rank]
            for j in order:
                if not (short > 0).any():
                    break
                unit = P[:, j] * (1 - self.cprop[j]) - self.cunit[j]
                have = np.maximum(tf[:, j], 0.0)
                lot = self.lot[j] if self.lot[j] > 0 else 0.0
                n = np.where(short > 0, np.minimum(have, np.ceil(short / np.maximum(unit, 1e-12) / max(lot, 1e-12)) * lot
                                                   if lot > 0 else short / np.maximum(unit, 1e-12)), 0.0)
                n = np.minimum(n, have)
                tf[:, j] -= n
                short -= n * unit
        if self.Ad and (short > 0).any():
            Fnv, pv, fee = Fn, self._pv(t), self._fee(t)
            rates = self.m.mr[t] if self.execution is None else self.execution.a['mr_clearing'][t]
            rel = p.mult * Fnv * pv * rates[None, :] - fee
            for j in (list(p.wf_rank) if p.wf_rank else range(self.Ad)):
                if not (short > 0).any():
                    break
                r = rel[:, j]
                n = np.where((short > 0) & (r > 0), np.minimum(np.abs(td[:, j]), np.ceil(short / np.maximum(r, 1e-12))), 0.0)
                td[:, j] -= np.sign(td[:, j]) * n
                short -= n * r
        return Order(t + 1, tf - qf, td - qd, "RESTORE", delta=True)

    def _row(self, t, qf, qd, cash, pend, P, Fn, navt, av, vm, rc, it, br, unres):
        r = {"t": t, "nav": float(navt[0]), "cash": float(cash[0]), "pending": float(pend[0, t + 1:].sum()),
             "im": float(self.im(qd, t)[0]), "available": float(av[0]), "vm": float(vm[0]), "roll_cost": float(rc[0]),
             "interest": float(it[0]), "breach": bool(br[0]), "unresolved": bool(unres[0])}
        for j, i in enumerate(self.fin):
            r[f"q_{i.name}"] = float(qf[0, j])
            r[f"v_{i.name}"] = float(qf[0, j] * P[0, j])
        for j, i in enumerate(self.din):
            r[f"q_{i.name}"] = float(qd[0, j])
            if self.execution is None:
                r[f"n_{i.name}"] = float(qd[0,j]*Fn[0,j]*self._pv(t)[0,j])
            else:
                r[f"n_{i.name}"] = float(sum(q[0]*self.execution.data(j,key,'mark')[0]*self.execution.data(j,key,'pv')[0]
                    for key,q in self.execution.positions[j].items() if np.any(q)))
                r[f"series_{i.name}"] = {key:float(q[0]) for key,q in self.execution.positions[j].items() if np.any(q)}
            if self.m.contract is not None:
                r[f"c_{i.name}"] = self.m.contract[t][j]
        return r


# ---------------------------------------------------------------------- conversion and metrics
def to_quantities(w_f: np.ndarray, w_d: np.ndarray, nav: float, P: np.ndarray, notional: np.ndarray,
                  lots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NAV weights -> quantities, rounded TOWARD ZERO (never above a bound); the result must be re-admitted."""
    qf = np.zeros_like(np.asarray(w_f, float))
    for j, (w, px, lot) in enumerate(zip(w_f, P, lots)):
        units = w * nav / px if px > 0 else 0.0
        qf[j] = np.trunc(units / lot) * lot if lot > 0 else units
    qd = np.trunc(np.asarray(w_d, float) * nav / np.where(notional > 0, notional, np.inf))
    return qf, qd + 0.0


def es(loss: np.ndarray, alpha: float) -> float:
    """Expected shortfall of LOSS (positive = loss): mean of the worst (1-alpha) share, tail size ceil((1-alpha) N)."""
    x = np.sort(np.asarray(loss, float))[::-1]
    k = max(1, int(np.ceil((1 - alpha) * len(x) - 1e-9)))
    return float(x[:k].mean())


def metrics(res: dict, nav0: np.ndarray, mm_growth: np.ndarray, liq_cost: np.ndarray | None = None) -> dict:
    nav = res["nav"]
    pnl = nav[:, -1] - nav0
    exc = nav[:, -1] - nav0 * mm_growth
    peak = np.maximum.accumulate(nav, axis=1)
    mdd = ((peak - nav) / peak).max(axis=1)
    out = {"mean_pnl": float(pnl.mean()), "median_pnl": float(np.median(pnl)), "mean_excess_pnl": float(exc.mean()),
           "median_excess_pnl": float(np.median(exc)), "ES95": es(-pnl, 0.95), "ES99": es(-pnl, 0.99),
           "p_breach": float(res["breach"].mean()), "p_unresolved": float(res["unresolved"].mean()),
           "p_forced": float((res["forced_events"] > 0).mean()), "mean_forced_contracts": float(res["forced_contracts"].mean()),
           "min_available_nav_p05": float(np.quantile((res["available"] / np.maximum(nav, 1e-9)).min(axis=1), 0.05)),
           "peak_im_nav_p95": float(np.quantile((res["im"] / np.maximum(nav, 1e-9)).max(axis=1), 0.95)),
           "mdd_p95": float(np.quantile(mdd, 0.95)), "mean_costs": float(res["costs"].mean()),
           "mean_turnover": float(res["turnover"].mean()), "frac_contract_path_days": int(res["frac_contract_days"].sum())}
    if liq_cost is not None:
        out["ES95_liquidation_nav"] = es(-(pnl - liq_cost), 0.95)
    return out
