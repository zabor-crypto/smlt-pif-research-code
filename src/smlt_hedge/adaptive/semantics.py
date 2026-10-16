"""D26 immutable plan/stage evidence and local residual ownership.

These records describe the shared Engine; they do not implement execution or
change economic parameters. M01--M06 authority: Master, 2026-10-06.
"""
from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

VERSION = 'D26_M01_M06_2026_10_06'
SAFETY = {'RESTORE', 'SWEEP', 'CONTRACT_MAINTENANCE', 'EMERGENCY_TARGET',
          'SEVERE_ISSUER_EXIT', 'ISSUER_PROTECTION', 'CLOSE_ALL', 'WATERFALL'}
TERMINAL = {'COMPLETED', 'SUPERSEDED', 'BLOCKED', 'EXPIRED_REFERENCE_HORIZON'}


def canonical(value):
    """Exact floats, sorted keys, explicit unchanged components; no rounding."""
    if isinstance(value, np.ndarray):
        return canonical(value.tolist())
    if isinstance(value, np.generic):
        return canonical(value.item())
    if isinstance(value, float):
        return {'float': value.hex()} if np.isfinite(value) else {'float': str(value)}
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in sorted(value.items(), key=lambda p: str(p[0]))}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if type(value).__module__.startswith('pandas'):
        return canonical(value.tolist()) if hasattr(value,'tolist') else str(value)
    if hasattr(value, '__dict__'):
        return canonical(vars(value))
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def serialise(value):
    return json.dumps(canonical(value), sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def identity(value):
    return sha256(serialise(value).encode()).hexdigest()


def content_identity():
    root = Path(__file__).resolve().parents[3]
    return {folder: {str(p.relative_to(root)): sha256(p.read_bytes()).hexdigest()
                     for p in sorted((root / folder).rglob('*'))
                     if p.is_file() and '__pycache__' not in p.parts}
            for folder in ('src', 'configs', 'data')}


@dataclass(frozen=True)
class ExecutionPlan:
    payload: str
    execution_plan_id: str
    decision_origin_nav: float
    last_stage_day: int
    horizon: int
    regime: str
    stage_count: int

    @classmethod
    def create(cls, *, origin, nav, endpoint, orders, spec, terms, mechanics, sources, horizon, regime=None):
        regime = regime or ('EMERGENCY' if spec.floor_suspended else 'ORDINARY')
        data = dict(semantics=VERSION, origin=origin, decision_origin_nav=nav,
                    authorized_endpoint=endpoint, ordered_stages=[dict(day=o.day, qf=o.qf,
                    qd=o.qd, tag=o.tag, delta=o.delta) for o in orders], spec=spec,
                    terms_calendar=terms, mechanics=mechanics, source_config_inputs=sources,
                    reference_horizon=horizon, regime=regime)
        payload = serialise(data)
        return cls(payload, sha256(payload.encode()).hexdigest(), float(nav),
                   max((o.day for o in orders), default=0), int(horizon), regime, len(orders))


    def validate_binding(self, *, orders=None, order=None, endpoint=None):
        """Bind issued directives to hashed authority, before residual derivation.

        Full reference streams require every ordered stage. Replay installation
        validates one issued stage; future clipped stages cannot imply completion.
        """
        mismatch = 'NOT_EVALUABLE_PLAN_ID_MISMATCH'
        if not self.execution_plan_id:
            return 'NOT_EVALUABLE_NO_PLAN_ID'
        try:
            if sha256(self.payload.encode()).hexdigest() != self.execution_plan_id:
                return mismatch
            data = json.loads(self.payload)
            stages = data['ordered_stages']
            scalars = dict(decision_origin_nav=self.decision_origin_nav,
                           reference_horizon=self.horizon, regime=self.regime)
            if any(canonical(v) != data[k] for k, v in scalars.items()):
                return mismatch
            if (self.stage_count != len(stages)
                    or self.last_stage_day != max((s['day'] for s in stages), default=0)):
                return mismatch
            if (self.regime != 'DIAGNOSTIC'
                    and set(data['authorized_endpoint']) != {'qf', 'qd'}):
                return mismatch
            if endpoint is not None and canonical(endpoint) != data['authorized_endpoint']:
                return mismatch
            issued = list(orders) if orders is not None else [order]
            if orders is not None and len(issued) != self.stage_count:
                return mismatch
            for i, o in enumerate(issued):
                if (o is None or o.plan != self
                        or not isinstance(o.stage_sequence, (int, np.integer))
                        or isinstance(o.stage_sequence, (bool, np.bool_))
                        or not 0 <= o.stage_sequence < self.stage_count
                        or (orders is not None and o.stage_sequence != i)):
                    return mismatch
                stage = stages[o.stage_sequence]
                if (not isinstance(o.issue_session, (int, np.integer))
                        or isinstance(o.issue_session, (bool, np.bool_)) or o.issue_session < 0
                        or not isinstance(o.day, (int, np.integer))
                        or isinstance(o.day, (bool, np.bool_))
                        or o.day - o.issue_session != stage['day']):
                    return mismatch
                if any(canonical(getattr(o, k)) != stage[k] for k in ('qf', 'qd', 'delta')):
                    return mismatch
                # The only authorized replay annotation changes TARGET intent
                # on an already-issued emergency plan. Other tags remain exact.
                annotation = (self.regime == 'EMERGENCY' and stage['tag'] == 'TARGET'
                              and o.tag == 'EMERGENCY_TARGET')
                if o.tag != stage['tag'] and not annotation:
                    return mismatch
            return None
        except (KeyError, TypeError, ValueError, AttributeError, IndexError):
            return mismatch


def frozen(value):
    a = np.asarray(value).copy()
    a.flags.writeable = False
    return a


def reduce_status(predicate, applicable):
    count = int(np.count_nonzero(applicable))
    return ('N/A' if count == 0 else 'PASS' if np.all(predicate[applicable]) else 'FAIL', count)


@dataclass(frozen=True)
class StageEvaluation:
    plan_ids: tuple
    evaluation_ids: tuple
    path_ids: tuple
    session: int
    stage_sequence: int
    intent: str
    local_attempt: np.ndarray
    ordinary: np.ndarray
    floor_applicable: np.ndarray
    pre_qf: np.ndarray
    pre_qd: np.ndarray
    qf: np.ndarray
    qd: np.ndarray
    fills_f: np.ndarray
    fills_d: np.ndarray
    prices: np.ndarray
    notional: np.ndarray
    derivative_gross: np.ndarray
    fees: np.ndarray
    cash: np.ndarray
    pending: np.ndarray
    im: np.ndarray
    nav: np.ndarray
    decision_origin_nav: np.ndarray
    issuer_units: np.ndarray
    floor_numerator: np.ndarray
    floor_predicate: np.ndarray
    rule_results: tuple = ()
    flags: tuple = ()
    actual_derivative_legs: tuple = ()
    rule_details: tuple = ()

    @property
    def keys(self):
        return tuple((self.plan_ids[n], self.evaluation_ids[n], self.path_ids[n],
                      self.session, self.stage_sequence) for n in range(len(self.plan_ids)))

    @property
    def floor_keys(self):
        return tuple(k for k, mask in zip(self.keys, self.floor_applicable) if mask)

    @property
    def floor_status(self):
        return reduce_status(self.floor_predicate, self.floor_applicable)

    @property
    def failure_ids(self):
        return tuple(tuple(name for name, results, mask in self.rule_results
                           if mask[n] and results[n] != 'PASS')
                     + (('funded_investment_floor',) if self.floor_applicable[n]
                        and not self.floor_predicate[n] else ())
                     for n in range(len(self.plan_ids)))

    def __getitem__(self, key):
        # Compatibility for pre-D26 readers. They consume this exact record.
        return {'day': self.session, 'funded_fraction': self.floor_numerator / np.maximum(self.nav,1e-12),
                'qf': self.qf, 'qd': self.qd, 'nav': self.nav, 'cash': self.cash,
                'pending': self.pending.sum(axis=1)}[key]


@dataclass
class LocalResiduals:
    qf: np.ndarray
    qd: np.ndarray
    origin_nav: np.ndarray
    residual_f: np.ndarray = field(init=False)
    residual_d: np.ndarray = field(init=False)
    owner_f: np.ndarray = field(init=False)
    owner_d: np.ndarray = field(init=False)
    directives: list = field(default_factory=list)
    states: dict = field(default_factory=dict)
    histories: list = field(default_factory=list)
    completions: dict = field(default_factory=dict)

    def __post_init__(self):
        self.residual_f = np.full_like(self.qf, np.nan)
        self.residual_d = np.full_like(self.qd, np.nan)
        self.owner_f = np.full(self.qf.shape, -1, int)
        self.owner_d = np.full(self.qd.shape, -1, int)

    def transition(self, pid, n, state, session, reason=''):
        old = self.states.get((pid, n), 'PLANNED')
        if old in TERMINAL or old == state or (old=='PARTIALLY_FILLED' and state=='ACTIVE'):
            return
        self.states[pid, n] = state
        self.histories.append(dict(plan_id=pid, path=n, session=session, previous=old,
                                   state=state, stop_reason=reason))

    def install(self, order, session, qf, qd):
        if order.plan is not None:
            error = order.plan.validate_binding(order=order)
            if error:
                raise ValueError(error)
            prior = {d['order'].stage_sequence for d in self.directives
                     if d['order'].plan == order.plan
                     and d['order'].issue_session == order.issue_session}
            if prior != set(range(order.stage_sequence)):
                raise ValueError('NOT_EVALUABLE_PLAN_ID_MISMATCH')
            # Preserve the validated issued authority; Engine may derive masked
            # absolute residual attempts without rebinding them as new orders.
            order = replace(order, qf=None if order.qf is None else frozen(order.qf),
                            qd=None if order.qd is None else frozen(order.qd))
        N = len(qf)
        vf = np.full_like(qf, np.nan) if order.qf is None else np.broadcast_to(order.qf, qf.shape).copy()
        vd = np.full_like(qd, np.nan) if order.qd is None else np.broadcast_to(order.qd, qd.shape).copy()
        if order.delta:
            cf, cd = vf.copy(), vd.copy()
            vf = np.where(~np.isfinite(cf) | (cf == 0), np.nan, np.maximum(qf + vf, np.minimum(qf, 0.)))
            vd = np.where(~np.isfinite(cd) | (cd == 0), np.nan, np.where(np.sign(vd) == -np.sign(qd),
                    qd - np.sign(qd)*np.minimum(np.abs(qd), np.abs(vd)), qd))
        ids = tuple(order.plan.execution_plan_id if order.plan else identity(dict(
            semantics=VERSION, issue=order.issue_session, due=order.day, intent=order.tag,
            origin_qf=self.qf[n], origin_qd=self.qd[n], origin_nav=self.origin_nav[n],
            qf=vf[n], qd=vd[n])) for n in range(N))
        nav = np.full(N, order.plan.decision_origin_nav) if order.plan else self.origin_nav.copy()
        idx = len(self.directives)
        item = dict(order=order, ids=ids, nav=nav, original_f=np.isfinite(vf),
                    original_d=np.isfinite(vd), issued=session, due=order.day,
                    target_f=vf.copy(), target_d=vd.copy(), remaining_f=vf.copy(), remaining_d=vd.copy())
        self.directives.append(item)
        for owner, residual, v in [(self.owner_f, self.residual_f, vf), (self.owner_d, self.residual_d, vd)]:
            mask = np.isfinite(v)
            for n, j in zip(*np.where(mask & (owner >= 0))):
                predecessor=self.directives[owner[n,j]]
                # Operational cash management cannot cancel a causal severe exit.
                if predecessor['order'].tag=='SEVERE_ISSUER_EXIT' and order.tag in ('RESTORE','SWEEP','CONTRACT_MAINTENANCE'):
                    mask[n,j]=False
                    continue
                old = predecessor['ids'][n]
                if old != ids[n]:
                    self.transition(old, int(n), 'SUPERSEDED', session, order.tag)
            owner[mask] = idx
            residual[mask] = v[mask]
        return idx

    def attempts(self):
        return sorted(set(self.owner_f[self.owner_f >= 0]) | set(self.owner_d[self.owner_d >= 0]))

    def targets(self, idx, qf, qd):
        mf, md = self.owner_f == idx, self.owner_d == idx
        return (np.where(mf, self.residual_f, np.nan),
                np.where(md, self.residual_d, qd), mf.any(axis=1) | md.any(axis=1))

    def finish(self, idx, session, qf, qd, filled, snapshot):
        item = self.directives[idx]
        mf, md = self.owner_f == idx, self.owner_d == idx
        active = mf.any(axis=1) | md.any(axis=1)
        for n in np.flatnonzero(active):
            self.transition(item['ids'][n], int(n), 'ACTIVE', session)
        df = mf & np.isclose(qf, self.residual_f, atol=1e-10, rtol=0)
        dd = md & (qd == self.residual_d)
        self.residual_f[df] = np.nan; self.owner_f[df] = -1
        self.residual_d[dd] = np.nan; self.owner_d[dd] = -1
        item['remaining_f'] = np.where(self.owner_f == idx, self.residual_f - qf, np.nan)
        item['remaining_d'] = np.where(self.owner_d == idx, self.residual_d - qd, np.nan)
        for n in np.flatnonzero(active):
            pid = item['ids'][n]
            owners = [i for i, d in enumerate(self.directives) if d['ids'][n] == pid]
            remaining = np.isin(self.owner_f[n], owners).any() or np.isin(self.owner_d[n], owners).any()
            last = item['order'].plan.last_stage_day + item['order'].issue_session if item['order'].plan else session
            plan = item['order'].plan
            issued = len({d['order'].stage_sequence for d in self.directives
                          if d['ids'][n]==pid and d['order'].plan is not None})
            if not remaining and session >= last and (plan is None or issued==plan.stage_count):
                self.transition(pid, int(n), 'COMPLETED', session)
                if self.states.get((pid, int(n))) == 'COMPLETED' and (pid, int(n)) not in self.completions:
                    self.completions[pid, int(n)] = snapshot
            elif filled[n]:
                self.transition(pid, int(n), 'PARTIALLY_FILLED', session)

    def expire(self, session):
        for (pid, n), state in list(self.states.items()):
            if state not in TERMINAL:
                self.transition(pid, n, 'EXPIRED_REFERENCE_HORIZON', session, 'REFERENCE_HORIZON')
