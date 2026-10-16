"""Opt-in Headroom A V2 protocol; never imported by ordinary frontier/stages.

One exploratory cutoff, six fixed slots, immutable controls, scoped provenance.
Pure helpers and synthetic validation are usable before a future economic gate.
Actual producers/solves/reference/admission require separate explicit receipts.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace, field
from hashlib import sha256
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
import zipfile

import numpy as np

from ..adaptive import engine as E, semantics as D
from . import frontier as FR, reference as REF, risk as RK
from .cfg import StopCondition

BASE_HEAD = "142e3d6136636e4ef2a38b3edbf9cf691dc7a62a"
BASE_TREE = "8151ca6be3315ae2f3dac7e8f80662bb8e067b20"
V2_ARCHIVE = "63de37cf507d890e435de93be812ed6f7dbaf6fbf5436f93e5172514917341ae"
RECONCILIATION = "da3f69bb2d3e335dfc1f85feab0b9130e7806e59fcb7a66fe3be1cfb26498fd2"
ACCEPTED_MANIFEST_IDENTITY = "93e7b54b8e003f595e3cba53f1823b4b3d5716f91b61d97ab1b62a81c550970e"
EXPERIMENT = "SMLT_D26_EXECUTION_HEADROOM_GENERATION_FEASIBILITY_EXPERIMENT_V1"
MODELS = ("GLOBAL_FORWARD", "GLOBAL_FORWARD_FHS", "RECENT_FORWARD", "STATE_CONDITIONAL_FORWARD")
GENERATION_SEEDS = tuple(20260925 + 101 * k for k in range(4))
ADMISSION_SEEDS = tuple(20960925 + 101 * k for k in range(4))
SLOTS = (("min_risk", None), *(('max_return', e) for e in (.0375, .0750, .1125, .1500, .1875)))
CONTROL_IDS = ("anchor", "lp@0.0375", "lp@0.0750", "lp@0.1125", "lp@0.1500", "lp@0.1875")
S09 = "S09_TRADING_SETTLEMENT_DELAY"


def stop(reason):
    raise StopCondition("HEADROOM_" + reason)


def bytes_hash(value):
    return sha256(value).hexdigest()


def file_hash(path):
    h = sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def tensor_identity(value):
    a = np.asarray(value)
    if a.dtype.hasobject:
        stop("OBJECT_TENSOR_UNSUPPORTED")
    return dict(shape=list(a.shape), dtype=a.dtype.str,
                sha256=bytes_hash(np.ascontiguousarray(a).tobytes()))


def runtime_identity():
    """Observe constructor class and binaries, WITHOUT requesting any samples."""
    bg = np.random.default_rng(GENERATION_SEEDS[0]).bit_generator
    paths = (Path(sys.executable).resolve(), Path(np.__file__),
             Path(sys.modules[type(bg).__module__].__file__),
             Path(sys.modules[np.random.Generator.__module__].__file__),
             Path(sys.modules[np.random.BitGenerator.__module__].__file__))
    return dict(python=sys.version, numpy=np.__version__, platform=platform.platform(),
                distribution_records={n: file_hash(importlib.metadata.distribution(n).locate_file(next(p for p in importlib.metadata.files(n) if str(p).endswith('.dist-info/RECORD'))))
                                      for n in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'pytest', 'PyYAML')},
                bit_generator=type(bg).__name__, bit_generator_module=type(bg).__module__,
                dependencies={n: importlib.metadata.version(n) for n in
                              ('numpy', 'pandas', 'scipy', 'scikit-learn', 'pytest', 'PyYAML')},
                binaries={str(p): file_hash(p) for p in paths})


@dataclass(frozen=True)
class Commitment:
    """Immutable exact preregistration/runtime/authority binding; not a freeze."""
    payload: str
    identity: str

    def read(self):
        if bytes_hash(self.payload.encode()) != self.identity:
            stop("COMMITMENT_MUTATED")
        return json.loads(self.payload)


def precommit(manifest, *, scope_bytes, observed_runtime, mode="FUTURE_ECONOMIC", authority=None,
              concrete_draw_links=()):
    """Accepted general unknown is a reservation; concrete ambiguous reuse stops."""
    if bytes_hash(json.dumps(manifest, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()) != ACCEPTED_MANIFEST_IDENTITY:
        stop('ACCEPTED_V2_MANIFEST_MISMATCH')
    scope = manifest['seed_catalog']
    if (manifest['base_head'] != BASE_HEAD or manifest['base_tree'] != BASE_TREE
            or scope['SCOPED_AVAILABLE_CUSTODY_CLOSURE'] != 'ACCEPTED_FOR_ONE_EXPERIMENT'
            or scope['GLOBAL_NAMESPACE_UNIQUENESS'] != 'NOT_PROVEN'
            or scope['SEMANTIC_NAMESPACE_SCAN_COMPLETENESS'] != 'NOT_PROVEN'
            or scope['MARKET_OOS'] != 'NO' or scope['scope_reconciliation_sha256'] != RECONCILIATION
            or bytes_hash(scope_bytes) != RECONCILIATION
            or scope['exception_experiment'] != EXPERIMENT or scope['cutoff'] != '2026-09-25'
            or not scope['scope_exclusions'] or not scope['scope_not_narrowed']):
        stop("SCOPED_PROVENANCE_IDENTITY_MISMATCH")
    mapping = manifest['admission_precommit']
    for key, seeds in (('mapping', ADMISSION_SEEDS), ('generation_mapping', GENERATION_SEEDS)):
        if [(r['model'], r['slot'], r['seed']) for r in mapping[key]] != list(zip(MODELS, range(4), seeds)):
            stop("MODEL_SEED_ROLE_MISMATCH")
    if tuple(RK.MODELS) != MODELS or not observed_runtime or concrete_draw_links:
        stop("CONCRETE_DRAW_LINKAGE_OR_MODEL_IDENTITY")
    if observed_runtime != runtime_identity():
        stop("PRNG_ENVIRONMENT_MISMATCH")
    if mode not in ('SYNTHETIC_CORRECTNESS', 'FUTURE_ECONOMIC'):
        stop("MODE_INVALID")
    value = dict(experiment=EXPERIMENT, mode=mode, base_head=BASE_HEAD, base_tree=BASE_TREE,
                 v2_archive=V2_ARCHIVE, scope=scope, runtime=observed_runtime,
                 source=manifest['source'], inputs=manifest['external_frozen_inputs'],
                 controls=manifest['controls'], primary_archive=manifest['primary_archive'],
                 authority=authority or {}, method_files={str(p): file_hash(p) for p in (Path(__file__), Path(FR.__file__))}, models=MODELS, generation_seeds=GENERATION_SEEDS,
                 admission_seeds=ADMISSION_SEEDS, cutoff='2026-09-25', H=126, n=2000,
                 call_schedule='generation fixed slots -> six slots/controls/delays seal -> separate admission fixed slots; cache=None')
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'))
    return Commitment(payload, bytes_hash(payload.encode()))


def require_execution_authority(commitment, *, synthetic_callbacks=False):
    c = commitment.read()
    if c['runtime'] != runtime_identity() or any(file_hash(p) != h for p, h in c['method_files'].items()):
        stop("PRNG_ENVIRONMENT_MISMATCH")
    if c['mode'] == 'SYNTHETIC_CORRECTNESS':
        if not synthetic_callbacks:
            stop("SYNTHETIC_MODE_REQUIRES_FIXTURE_CALLBACKS")
        return
    # Never infer these sequential approvals from a candidate/test or R10 packet.
    for key in ('independent_code_review', 'new_method_freeze', 'separate_economic_authorization'):
        r = c['authority'].get(key, {})
        if (r.get('experiment') != EXPERIMENT or r.get('approved') is not True
                or not r.get('path') or file_hash(r['path']) != r.get('sha256')):
            stop("MISSING_" + key.upper())


def check_build_request(cfg, t, n_gen, n_adm, models, seed_offset, commitment):
    if commitment is None:
        stop("PRECOMMIT_REQUIRED_BEFORE_GENERATION")
    require_execution_authority(commitment, synthetic_callbacks=RK.simulate.__module__ != RK.__name__)
    if commitment.read()['mode'] == 'FUTURE_ECONOMIC':
        from .cfg import load
        root = Path(FR.__file__).resolve().parents[3]
        if D.identity(cfg) != D.identity(load(root)):
            stop('FROZEN_CONFIG_VALUE_MISMATCH')
        for name, row in commitment.read()['source'].items():
            if name != 'src/smlt_hedge/regime_v2/frontier.py' and file_hash(root / name) != row['sha256']:
                stop('FROZEN_SOURCE_CONFIG_IDENTITY_MISMATCH')
    if (tuple(models) != MODELS or n_gen != 2000 or n_adm != 2000 or seed_offset != 0
            or cfg['r3']['frontier']['horizon_sessions'] != 126
            or cfg['r3']['frontier']['seed_generation'] != GENERATION_SEEDS[0]
            or str(t.date()) != '2026-09-25'):
        stop("GENERATION_REQUEST_IDENTITY_MISMATCH")


def check_generation_setup(su, n_gen, n_adm, seed0):
    if (su.adm or n_gen != 2000 or n_adm != 2000 or seed0 != GENERATION_SEEDS[0]
            or tuple(g[0] for g in su.gen) != MODELS
            or su.cfg['r3']['frontier']['horizon_sessions'] != 126
            or any(g[4].P.shape[:2] != (2000, 127) for g in su.gen)):
        stop("GENERATION_SUPPORT_OR_PHASE_INVALID")


def generation_identity(su):
    """No admission access: only the causal origin and generation-side support."""
    return D.identity(dict(t=su.t, routes=su.routes, funded=su.fspecs, derivatives=su.dspecs,
                           fin_names=su.fin_names, din_names=su.din_names, nav0=su.nav0,
                           book0=su.book0, views=su.views.to_dict(), cfg=su.cfg, v2=su.v2,
                           generation=[(g[0], D.identity(dict(indices=g[1].indices, shift=g[1].shift, centring=g[1].centring)), tensor_identity(g[1].R), tensor_identity(g[1].mmf_daily), tensor_identity(g[4].P),
                                        tensor_identity(g[4].Fn), tensor_identity(su.G[g[0]])) for g in su.gen]))


def funded_envelope(generation, stresses, *, funded_names, price0, calendar, template_ids,
                    units, horizon=126, paths_per_model=2000):
    """Pure support maximum in RUB per SBGB unit. No producer/LP/Engine calls."""
    if (horizon != 126 or paths_per_model != 2000 or list(generation) != list(MODELS)
            or len(template_ids) != 12 or len(set(template_ids)) != 12
            or set(stresses) != set(template_ids) or len(calendar) != 127
            or len(set(str(x) for x in calendar)) != 127
            or any(str(a) >= str(b) for a, b in zip(calendar, calendar[1:]))
            or len(funded_names) != len(set(funded_names)) or 'OFZ_FUNDED' not in funded_names
            or units.get('OFZ_FUNDED') != 'RUB/SBGB_FUNDED_UNIT'
            or not np.isfinite(price0) or price0 <= 0):
        stop("PRICE_SUPPORT_AUTHORITY_INVALID")
    j = funded_names.index('OFZ_FUNDED')
    u, identities = 1., {}
    for name, tensor in [*generation.items(), *stresses.items()]:
        a = np.asarray(tensor)
        n = paths_per_model if name in generation else 1
        if a.shape != (n, 127, len(funded_names)):
            stop("PRICE_SUPPORT_SHAPE_INVALID")
        p = a[:, :, j]
        if not np.isfinite(p).all() or (p <= 0).any() or not (p[:, 0] == price0).all():
            stop("PRICE_SUPPORT_NONFINITE_OR_ORIGIN_MISMATCH")
        u = max(u, float((p[:, 1:] / price0).max()))
        identities[name] = tensor_identity(p)
    return dict(U=u, construction_bound=.40 / u, actual_cap=.40, P0=price0,
                horizon=126, units=units['OFZ_FUNDED'], calendar_hash=D.identity(calendar),
                funded_names=list(funded_names), support=identities)


def envelope_from_setup(su):
    """Future authorized PhaseA support, canonical named funded producer only."""
    from ..adaptive import stress as SS
    st = su.cfg['stress']
    j = su.fin_names.index('OFZ_FUNDED')
    dates = su.fspecs[j].execution_dates
    if dates is None:
        stop("SOURCE_CALENDAR_MISSING")
    calendar = [str(su.t), *(str(x) for x in dates[1:127])]
    generation = {g[0]: g[4].P for g in su.gen}
    stresses = {}
    for name, sc in st['scenarios'].items():
        _, _, mkt = SS.to_market(su, SS.path(su, sc, st['route_groups']), sc, 1.)
        stresses[name] = mkt.P
    return funded_envelope(generation, stresses, funded_names=su.fin_names,
                           price0=su.unit_price['OFZ_FUNDED'], calendar=calendar,
                           template_ids=list(st['scenarios']), units={'OFZ_FUNDED': 'RUB/SBGB_FUNDED_UNIT'})


def validate_quantity_envelope(ex, *, fin_names, inherited_qf, nav0, price0, u):
    if (not all(np.isfinite(x) and x > 0 for x in (nav0, price0, u)) or u < 1
            or fin_names.count('OFZ_FUNDED') != 1):
        stop("QUANTITY_BOUND_AUTHORITY_INVALID")
    j = fin_names.index('OFZ_FUNDED')
    q = np.floor(.40 * nav0 / (u * price0))  # OFZ lot=1, toward zero
    candidates = [np.asarray(inherited_qf)[..., j], np.asarray(ex['authorized_qf'])[..., j]]
    plan = ex['execution_plan']
    if plan.validate_binding(orders=ex['orders'], endpoint=dict(qf=ex['authorized_qf'], qd=ex['authorized_qd'])):
        stop("PLAN_BINDING_INVALID")
    for o in ex['orders']:
        if o.delta or o.tag != 'TARGET' or not 0 <= o.day <= 126 or o.issue_session != 0:
            stop("UNSUPPORTED_QUANTITY_STAGE_AUTHORITY")
        if o.qf is not None:
            a = np.asarray(o.qf)[..., j]
            candidates.append(a[np.isfinite(a)] if a.ndim else np.array([a]) if np.isfinite(a) else np.array([]))
    finite = np.concatenate([np.asarray(a).reshape(-1) for a in candidates])
    if not np.isfinite(finite).all() or (finite < 0).any() or (finite != np.floor(finite)).any():
        stop("QUANTITY_LOT_OR_SIGN_INVALID")
    qmax = float(finite.max(initial=0))
    if qmax > q:
        stop("INVALID_QUANTITY_ENVELOPE")
    return dict(Q=int(q), Qmax=qmax, lot=1, N0=nav0, P0=price0, U=u)


def plan_binding(ex):
    error = REF._plan_error(ex)
    if error:
        stop(error)
    if ex['execution_plan'].regime != 'ORDINARY' or ex['spec'].floor_suspended:
        stop("NONORDINARY_PLAN")
    if D.canonical(ex['spec']) != json.loads(ex['execution_plan'].payload)['spec']:
        stop('PLAN_SPEC_AUTHORITY_MISMATCH')
    mechanics = json.loads(ex['execution_plan'].payload)['mechanics']
    for live, frozen in (('reserve_min', 'reserve'), ('wf_rank', 'waterfall'), ('restore_rank', 'restoration')):
        if live not in ex or frozen not in mechanics or D.canonical(ex[live]) != mechanics[frozen]:
            stop('PLAN_EXECUTION_MECHANICS_MISMATCH:' + live)
    for index, order in enumerate(ex['orders']):
        if (order.plan is None or order.plan.payload != ex['execution_plan'].payload
                or order.issue_session != 0 or order.stage_sequence != index):
            stop('ORDER_NATIVE_PLAN_BINDING_MISMATCH')
    return ex['execution_plan'].execution_plan_id


# Explicit native consumer read-set. These are actual reference values, not planned endpoints.
REFERENCE_FIELDS = ('reference_identity', 'reference_plan_id', 'reference_snapshot_plan_id',
    'reference_state_snapshot_id', 'reference_plan_state', 'reference_eligible',
    'reference_completion_day', 'reference_snapshot_day', 'reference_pending_obligations',
    'execution_nav', 'execution_cash', 'execution_pending', 'execution_margin', 'execution_costs',
    'execution_turnover', 'execution_available', 'qf', 'qd', 'reference_notional',
    'net_issuer_units', 'net_issuer', 'funded_fraction_at_execution',
    'REFERENCE_TARGET_MANDATE_PASS', 'execution_lk_certification', 'reference_result', 'reference_timeline')


# Only producer-return captures in this process authorize these detached receipts.
# Received objects cannot acquire authority by recomputing their own hash.
_NATIVE_REFERENCE_RECEIPTS = {}


@dataclass(frozen=True)
class NativeReferenceEvidence:
    payload: str
    identity: str


def _reference_values(ex):
    if any(k not in ex for k in REFERENCE_FIELDS):
        stop('NATIVE_REFERENCE_EVIDENCE_MISSING')
    return {k: ex[k] for k in REFERENCE_FIELDS}


def _order_callbacks(ex, producer_setup=None):
    values = []
    for order in ex['orders']:
        fn = order.stage_check
        cells = FR._native_stage_callback_origin(fn)
        if D.canonical(cells.get('spec')) != D.canonical(ex['spec']):
            stop('STAGE_CALLBACK_SPEC_MISMATCH')
        su = cells.get('su')
        if producer_setup is not None and su is not producer_setup:
            stop('STAGE_CALLBACK_PRODUCER_SETUP_MISMATCH')
        values.append(dict(callback_object=id(fn), setup_object=id(su), spec_object=id(cells['spec']),
            code=fn.__code__.co_code.hex(), spec=D.canonical(cells['spec']),
            origin=D.canonical({k: getattr(su, k) for k in ('cfg', 'unit_price', 'nav0', 'fin_names', 'din_names', 'routes', 'dspecs')})))
    return values


def bind_native_reference(su, ex, native):
    """Old caller-supplied binder is closed; only certify_reference invokes/captures."""
    stop('CALLER_SUPPLIED_NATIVE_BINDER_FORBIDDEN')


def require_native_reference(ex):
    """Authority/chronology check for positive OR negative native outcomes; never manufactures PASS."""
    pid = plan_binding(ex)
    evidence = ex.get('_headroom_native_reference')
    record = _NATIVE_REFERENCE_RECEIPTS.get(getattr(evidence, 'identity', None))
    if (not isinstance(evidence, NativeReferenceEvidence)
            or bytes_hash(evidence.payload.encode()) != evidence.identity
            or not isinstance(record, tuple) or len(record) != 2
            or record[0] != evidence.payload):
        stop('NATIVE_REFERENCE_AUTHORITY_MISSING_OR_MUTATED')
    callbacks = tuple(o.stage_check for o in ex['orders'])
    if len(callbacks) != len(record[1]) or any(a is not b for a, b in zip(callbacks, record[1])):
        stop('NATIVE_CALLBACK_OBJECT_CHANGED')
    value = json.loads(evidence.payload)
    if (value['plan_id'] != pid or value['reference'] != D.canonical(_reference_values(ex))
            or value['callbacks'] != D.canonical(_order_callbacks(ex))
            or value['lineage'] != D.canonical(ex.get('delayed_lineage'))):
        stop('NATIVE_REFERENCE_OR_CONSUMED_CACHE_CHANGED')
    result = ex['reference_result']
    if not all(k in result for k in ('plan_states', 'plan_completions', 'execution_records',
                                     'continuous_safety_pass', 'plan_history')):
        stop('NATIVE_REFERENCE_RESULT_INCOMPLETE')
    state = result['plan_states'].get((pid, 0))
    completed = REF._completion(ex, pid)
    if state != ex['reference_plan_state'] or state not in ('COMPLETED', 'EXPIRED_REFERENCE_HORIZON', 'BLOCKED', 'PARTIALLY_FILLED'):
        stop('NATIVE_REFERENCE_STATE_CONTRADICTION')
    if state == 'COMPLETED':
        if completed is None:
            stop('NATIVE_COMPLETION_MISSING')
        stage = completed['stage'];day = stage.session
        qf, qd, cash, nav, im = stage.qf[0], stage.qd[0], stage.cash[0], stage.nav[0], stage.im[0]
        pending = completed['pending'][0];costs = completed['costs'][0];turnover = completed['turnover'][0]
        series = completed.get('series');completion_day = day
    else:
        if completed is not None:
            stop('NEGATIVE_REFERENCE_HAS_COMPLETION')
        day = ex['execution_plan'].horizon
        qf, qd = result['qf_end'][0], result['qd_end'][0]
        cash, nav, im = result['cash_end'][0], result['nav'][0,-1], result['im'][0,-1]
        pending = result['pending_ledger_end'][0];costs=result['costs'][0];turnover=result['turnover'][0]
        series = result.get('series_positions_end');completion_day = None
    obligations = {due: float(v) for due, v in enumerate(pending) if due > day and v != 0}
    snapshot = REF._snapshot(pid, day, qf, qd, cash, obligations, series, costs, turnover, nav, im)
    certificate = ex['reference_identity']
    if (not isinstance(certificate, REF.ReferenceIdentity) or certificate.execution_plan_id != pid
            or certificate.snapshot_id != snapshot or certificate.completion_day != completion_day
            or ex['reference_state_snapshot_id'] != snapshot or ex['reference_snapshot_day'] != day
            or ex['reference_completion_day'] != completion_day
            or ex['reference_plan_id'] != pid or ex['reference_snapshot_plan_id'] != pid):
        stop('NATIVE_REFERENCE_SNAPSHOT_CONTRADICTION')
    native_actual = dict(qf=qf, qd=qd, execution_cash=cash, execution_nav=nav, execution_margin=im,
        execution_pending=float(pending.sum()), execution_costs=costs, execution_turnover=turnover,
        reference_pending_obligations=obligations)
    if any(D.canonical(ex[k]) != D.canonical(v) for k,v in native_actual.items()):
        stop('NATIVE_REFERENCE_ACTUAL_CACHE_CONTRADICTION')
    target = all(not ids for stage in result['execution_records'] if stage.session <= day for ids in stage.failure_ids)
    eligible = state == 'COMPLETED' and target and bool(np.asarray(result['continuous_safety_pass']).all())
    if ex['reference_eligible'] is not eligible or ex['REFERENCE_TARGET_MANDATE_PASS'] != ('PASS' if eligible else 'FAIL'):
        stop('NATIVE_REFERENCE_VERDICT_CONTRADICTION')
    return ex['reference_state_snapshot_id']


def require_delayed_reference(child, parent=None):
    require_native_reference(child)
    lineage = child.get('delayed_lineage', {})
    if (lineage.get('scenario') != S09 or lineage.get('Delta') != 3
            or lineage.get('child_payload_sha256') != child['execution_plan_id']
            or lineage.get('parent_plan_id') != lineage.get('parent_payload_sha256')):
        stop('DELAYED_NATIVE_LINEAGE_MISSING_OR_CONTRADICTORY')
    if parent is not None:
        require_matched_reference(parent)
        expected = delayed_control_case(parent, S09, 4)
        if (child['execution_plan'].payload != expected['execution_plan'].payload
                or lineage != expected['delayed_lineage']):
            stop('DELAYED_PARENT_PAYLOAD_OR_SNAPSHOT_MISMATCH')
    return child['reference_state_snapshot_id']


def require_case_reference(ex):
    return require_delayed_reference(ex) if 'delayed_lineage' in ex else require_matched_reference(ex)


def require_matched_reference(ex):
    require_native_reference(ex)
    error = REF.certify_cached(ex)
    if error or ex.get('reference_plan_state') != 'COMPLETED' or not ex.get('reference_eligible'):
        stop("REFERENCE_NOT_MATCHED_COMPLETED_ELIGIBLE")
    result = ex.get('reference_result', {})
    if ('execution_records' not in result or 'continuous_safety_pass' not in result
            or not np.asarray(result['continuous_safety_pass']).all()
            or any(ids for stage in result['execution_records']
                   if stage.session <= ex['reference_completion_day'] for ids in stage.failure_ids)):
        stop('REFERENCE_NATIVE_COMPLIANCE_NOT_CERTIFIED')
    return ex['reference_state_snapshot_id']


@dataclass(frozen=True)
class ControlRecord:
    id: str
    raw: bytes
    member_hash: str
    plan_id: str

    def execution(self, su):
        """Decode exact authority, never regenerate from x. Reference recertifies later."""
        if bytes_hash(self.raw) != self.member_hash:
            stop("CONTROL_BYTES_MUTATED")
        wire = json.loads(self.raw)['ex']
        plan = D.ExecutionPlan(**wire['execution_plan'])
        spec = FR.Spec(**wire['spec'])
        orders = []
        for row in wire['orders']:
            fields = {k: v for k, v in row.items() if k not in ('plan', 'stage_check')}
            for k in ('qf', 'qd'):
                fields[k] = None if fields.get(k) is None else D.frozen(fields[k])
            orders.append(E.Order(**fields, plan=plan, stage_check=FR._execution_check(su, spec)))
        out = {k: deepcopy(wire[k]) for k in ('reserve_min', 'wf_rank', 'restore_rank')}
        out.update(orders=orders, execution_plan=plan, execution_plan_id=plan.execution_plan_id,
                   authorized_qf=D.frozen(wire['authorized_qf']), authorized_qd=D.frozen(wire['authorized_qd']),
                   spec=spec, role='CONTROL', historical_admission_status='FAIL', control_id=self.id,
                   original_record_sha256=self.member_hash, original_reference_snapshot_id=wire['reference_state_snapshot_id'])
        if plan_binding(out) != self.plan_id or wire['execution_plan_id'] != self.plan_id:
            stop("CONTROL_PLAN_IDENTITY_MISMATCH")
        if (wire['reference_plan_state'] != 'COMPLETED' or any(wire[k] != self.plan_id for k in
                ('reference_plan_id', 'score_plan_id', 'turnover_plan_id', 'admission_plan_id', 'reference_snapshot_plan_id'))):
            stop("CONTROL_ORIGINAL_REFERENCE_IDENTITY")
        # Actual supporting completion bytes are required, not metadata alone.
        completed = wire.get('reference_result', {}).get('plan_completions', {}).get(str((self.plan_id, 0)))
        if completed is None:
            stop('CONTROL_ORIGINAL_COMPLETION_MISSING')
        stage_wire = completed['stage']
        tuple_fields = ('plan_ids', 'evaluation_ids', 'path_ids', 'flags', 'actual_derivative_legs')
        fields = {k: tuple(v) if k in tuple_fields else D.frozen(v) if isinstance(v, list) else v
                  for k, v in stage_wire.items() if k not in ('rule_results', 'rule_details')}
        fields['rule_results'] = tuple((name, D.frozen(values), D.frozen(mask)) for name, values, mask in stage_wire['rule_results'])
        fields['rule_details'] = tuple((name, D.frozen(num), D.frozen(den), limit, epsilon, allowance)
                                      for name, num, den, limit, epsilon, allowance in stage_wire['rule_details'])
        completion = {**completed, 'stage': D.StageEvaluation(**fields),
                      'pending': np.asarray(completed['pending']), 'costs': np.asarray(completed['costs']),
                      'turnover': np.asarray(completed['turnover'])}
        cached = {**out, **{k: wire[k] for k in ('reference_plan_state', 'reference_eligible', 'reference_plan_id',
                  'reference_snapshot_plan_id', 'reference_state_snapshot_id', 'reference_completion_day')},
                  'reference_identity': REF.ReferenceIdentity(**wire['reference_identity']),
                  'reference_result': {'plan_completions': {(self.plan_id, 0): completion}}}
        if REF.certify_cached(cached):
            stop('CONTROL_ORIGINAL_COMPLETION_OR_SNAPSHOT_MISMATCH')
        return out


def load_frozen_controls(archive, manifest):
    if file_hash(archive) != manifest['primary_archive']['sha256']:
        stop("CONTROL_ARCHIVE_HASH_MISMATCH")
    rows = manifest['controls']
    if tuple(r['id'] for r in rows) != CONTROL_IDS:
        stop("CONTROL_LIST_INVALID")
    output = []
    with zipfile.ZipFile(archive) as z:
        for row in rows:
            raw = z.read(row['archive_member'])
            if (len(raw) != row['bytes'] or bytes_hash(raw) != row['member_sha256']
                    or row['source_role'] != 'SEALED_PRIMARY_ORDINARY_CONTROL'
                    or row['old_admission_status'] != 'FAIL'):
                stop("CONTROL_CUSTODY_MISMATCH")
            wire = json.loads(raw)
            if wire['id'] != row['id'] or wire['execution_plan_id'] != row['plan_id']:
                stop("CONTROL_RECORD_ID_MISMATCH")
            output.append(ControlRecord(row['id'], raw, row['member_sha256'], row['plan_id']))
    return tuple(output)


def delayed_control_case(parent, scenario, order_delay):
    """Master57 option B: relative clocks +3 only; no native Engine call here."""
    plan_binding(parent)
    if scenario != S09:
        if order_delay != 1:
            stop("UNAUTHORIZED_DELAY_TEMPLATE")
        return parent
    plan, spec = parent['execution_plan'], parent['spec']
    if order_delay != 4 or spec.delay != 1 or plan.horizon != 126:
        stop("S09_CLOCK_AUTHORITY_INVALID")
    original = json.loads(plan.payload)
    if D.canonical(spec) != original['spec'] or any(o.issue_session != 0 for o in parent['orders']):
        stop("S09_SPEC_OR_ISSUE_ORIGIN_MISMATCH")
    data = deepcopy(original)
    for stage in data['ordered_stages']:
        day = stage['day']
        if type(day) is not int or not 0 <= day + 3 <= 126:
            stop("S09_STAGE_OUTSIDE_HORIZON")
        stage['day'] = day + 3
    payload = json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    child = D.ExecutionPlan(payload, bytes_hash(payload.encode()), plan.decision_origin_nav,
                            plan.last_stage_day + 3, plan.horizon, plan.regime, plan.stage_count)
    if child.execution_plan_id == plan.execution_plan_id:
        stop("S09_CHILD_ID_NOT_DISTINCT")
    orders = [replace(o, day=o.day + 3, plan=child,
                      qf=None if o.qf is None else D.frozen(o.qf),
                      qd=None if o.qd is None else D.frozen(o.qd)) for o in parent['orders']]
    # Copy authority fields only. Never carry parent's evaluated snapshot/cache/result.
    keys = ('reserve_min', 'wf_rank', 'restore_rank', 'authorized_qf', 'authorized_qd', 'spec', 'role')
    out = {k: deepcopy(parent[k]) for k in keys if k in parent}
    out.update(orders=orders, execution_plan=child, execution_plan_id=child.execution_plan_id,
               target_published=False, reference_plan_state='NOT_EVALUATED',
               delayed_lineage=dict(parent_plan_id=plan.execution_plan_id,
                                    parent_payload_sha256=bytes_hash(plan.payload.encode()),
                                    scenario=S09, Delta=3, child_payload_sha256=child.execution_plan_id,
                                    parent_reference_snapshot_id=parent.get('reference_state_snapshot_id')))
    if plan_binding(out) != child.execution_plan_id:
        stop("S09_NATIVE_BINDING_FAILED")
    unchanged = deepcopy(data)
    for stage in unchanged['ordered_stages']:
        stage['day'] -= 3
    if unchanged != original or D.serialise(parent['spec']) != D.serialise(out['spec']):
        stop("S09_NONCLOCK_PAYLOAD_CHANGE")
    return out


def certify_reference(su, ex):
    """Single native invocation and immediate capture; never mint from supplied caches."""
    plan_binding(ex)
    out = dict(ex)
    # REF.evaluate itself returns cached values when timeline is present. Any
    # prior native marker must therefore validate, never enter that producer.
    prior = ('_headroom_native_reference', 'reference_timeline', 'reference_result',
             'reference_identity', 'reference_state_snapshot_id', 'reference_plan_id',
             'reference_snapshot_plan_id', 'reference_completion_day', 'reference_snapshot_day',
             'reference_eligible', 'REFERENCE_TARGET_MANDATE_PASS')
    if any(k in out for k in prior):
        require_native_reference(out)
        _order_callbacks(out, producer_setup=su)
        return out
    callbacks = _order_callbacks(out, producer_setup=su)
    native = REF.evaluate(su, out, out['spec'])  # exactly one existing producer call
    if callbacks != _order_callbacks(out, producer_setup=su):
        stop('NATIVE_CALLBACK_CHANGED_DURING_REFERENCE')
    out.update(native)
    plan_binding(out)
    # Only this lexical producer return supplies receipt fields. No helper
    # accepting a caller-provided native object may register authority.
    value = dict(plan_id=out['execution_plan_id'], reference=_reference_values(native),
        callbacks=callbacks, lineage=deepcopy(out.get('delayed_lineage')),
        context=dict(fin_names=list(su.fin_names), unit_price=dict(su.unit_price), nav0=su.nav0,
                     floor_set=list(out['spec'].mandate['floor_set'])))
    payload = D.serialise(value)
    identity = bytes_hash(payload.encode())
    # Strongly retain the actual factory callbacks: object IDs cannot be recycled
    # while this receipt authorizes them. Validate their referents again on read.
    _NATIVE_REFERENCE_RECEIPTS[identity] = (payload, tuple(o.stage_check for o in out['orders']))
    out['_headroom_native_reference'] = NativeReferenceEvidence(payload, identity)
    require_native_reference(out)
    return out


def case_provider(su, sealed_cases=None):
    def provide(name, scenario, parent, spec):
        if D.identity(spec) != D.identity(parent['spec']):
            stop("CASE_SPEC_MISMATCH")
        delay = int(scenario.get('order_delay_sessions', 1))
        if sealed_cases is not None and name == S09:
            child = sealed_cases.get(parent['execution_plan_id'])
            if child is None or delay != 4:
                stop('SEALED_DELAYED_CASE_MISSING')
            require_delayed_reference(child, parent)
            return child
        child = delayed_control_case(parent, name, delay)
        if child is parent:
            require_matched_reference(parent)
            return parent
        child = certify_reference(su, child)
        require_delayed_reference(child, parent)
        return child
    return provide


def plan_descriptor(ex):
    pid = plan_binding(ex)
    return dict(plan_id=pid, payload=ex['execution_plan'].payload,
                orders=D.serialise([dict(day=o.day, issue_session=o.issue_session, stage_sequence=o.stage_sequence,
                                         qf=o.qf, qd=o.qd, tag=o.tag, delta=o.delta) for o in ex['orders']]),
                spec=D.serialise(ex['spec']), reference_state=ex.get('reference_plan_state'),
                reference_snapshot=ex.get('reference_state_snapshot_id'),
                role=ex.get('role'), lineage=ex.get('delayed_lineage'),
                custody=ex.get('original_record_sha256'),
                mechanics=D.canonical({k: ex[k] for k in ('reserve_min','wf_rank','restore_rank')}),
                native_evidence=ex.get('_headroom_native_reference').identity if isinstance(ex.get('_headroom_native_reference'), NativeReferenceEvidence) else None,
                consumed_reference=D.identity(_reference_values(ex)) if 'reference_identity' in ex else None,
                callbacks=_order_callbacks(ex))


@dataclass(frozen=True)
class ListSeal:
    payload: str
    identity: str
    bindings: tuple
    live_bindings: tuple = field(default=(), repr=False)

    def read(self):
        if bytes_hash(self.payload.encode()) != self.identity:
            stop("LIST_SEAL_MUTATED")
        data = json.loads(self.payload)
        for bindings in (self.bindings, self.live_bindings):
            if [plan_descriptor(ex) for ex in bindings] != data['plans'] + data['delayed_cases']:
                stop('POSTSEAL_PLAN_OR_REFERENCE_CHANGED')
            count = len(data['plans'])
            for ex in bindings[:count]:
                require_matched_reference(ex)
            for parent, child in zip(bindings[:count], bindings[count:]):
                require_delayed_reference(child, parent)
        return data


def seal_plan_list(su, slots, plans, controls, delayed_cases, commitment):
    if (len(slots) != 6 or [(r['objective'], r['es_target']) for r in slots] != list(SLOTS)
            or len(plans) > 6 or tuple(c['control_id'] for c in controls) != CONTROL_IDS
            or any(r['status'] not in ('INFEASIBLE', 'DUPLICATE', 'INVALID', 'INELIGIBLE', 'ELIGIBLE') for r in slots)):
        stop("SIX_SLOT_OR_CONTROL_LIST_INVALID")
    descriptors = [plan_descriptor(p) for p in [*plans, *controls]]
    for ex in [*plans, *controls]:
        require_matched_reference(ex)
    if len(delayed_cases) != len(descriptors):
        stop('INCOMPLETE_DELAYED_LIST')
    for parent, child in zip([*plans, *controls], delayed_cases):
        require_delayed_reference(child, parent)
        expected = delayed_control_case(parent, S09, 4)
        if expected['execution_plan'].payload != child['execution_plan'].payload:
            stop('DELAYED_PAYLOAD_NOT_CLOCK_ONLY')
    value = dict(commitment=commitment.identity, generation_identity=generation_identity(su),
                 slots=slots, plans=descriptors, delayed_cases=[plan_descriptor(ex) for ex in delayed_cases],
                 models=MODELS, admission_seeds=ADMISSION_SEEDS, cache=None,
                 planned_max_stochastic_batches=144, planned_max_deterministic_cells=288)
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'))
    live = tuple([*plans, *controls, *delayed_cases])
    return ListSeal(payload, bytes_hash(payload.encode()), tuple(deepcopy(ex) for ex in live), live)


def verify_phase_b(su, seal, commitment):
    c, s = commitment.read(), seal.read()
    require_execution_authority(commitment, synthetic_callbacks=getattr(su, '_headroom_synthetic', False))
    if (s['commitment'] != commitment.identity or s['generation_identity'] != generation_identity(su)
            or getattr(su, '_headroom_generation_identity', None) != s['generation_identity']
            or getattr(su, '_headroom_commitment', None) != commitment.identity
            or getattr(su, '_headroom_admission_used', True) or su.adm
            or s['cache'] is not None or len(s['slots']) != 6
            or s['admission_seeds'] != list(ADMISSION_SEEDS) or s['models'] != list(MODELS)
            or c['scope']['scope_reconciliation_sha256'] != RECONCILIATION):
        stop("PHASE_B_PREIMAGE_OR_ROLE_MISMATCH")


def admission_context(su, seal, commitment):
    if len(su.adm) != 4 or [m.name for m in su.adm] != [m + '_ADMISSION' for m in MODELS]:
        stop("ADMISSION_MODEL_IDENTITY_MISMATCH")
    tensors = []
    for k, mp in enumerate(su.adm):
        producer = getattr(mp, 'headroom_producer_identity', None)
        if commitment.read()['mode'] == 'FUTURE_ECONOMIC' and producer is None:
            stop('ADMISSION_PRODUCER_IDENTITY_MISSING')
        if commitment.read()['mode'] == 'FUTURE_ECONOMIC' and not isinstance(producer.get('raw_indices'), dict):
            stop('ADMISSION_PATH_INDEX_IDENTITY_MISSING')
        if producer is not None and (producer['model'] != MODELS[k] or producer['seed'] != ADMISSION_SEEDS[k]
                or producer['role'] != 'HEADROOM_A_ADMISSION' or producer['runtime'] != commitment.read()['runtime']):
            stop('ADMISSION_PRODUCER_MODEL_SEED_ROLE_MISMATCH')
        if mp.mkt.P.shape[:2] != (2000, 127):
            stop("ADMISSION_SHAPE_INVALID")
        if any(np.shares_memory(mp.mkt.P, g[4].P) for g in su.gen):
            stop("ADMISSION_REUSES_GENERATION_TENSOR")
        tensors.append(dict(model=mp.name, producer=producer or {'fixture':'SYNTHETIC_ONLY'}, P=tensor_identity(mp.mkt.P), Fn=tensor_identity(mp.mkt.Fn),
                            funded=D.identity(mp.fin), derivatives=D.identity(mp.din)))
    value = dict(role='HEADROOM_A_ADMISSION', commitment=commitment.identity, list_seal=seal.identity,
                 origin=generation_identity(su), runtime=commitment.read()['runtime'],
                 seed_mapping=list(zip(MODELS, ADMISSION_SEEDS)), tensors=tensors,
                 plans=seal.read()['plans'], cache=None)
    return dict(identity=D.identity(value), descriptor=value)


def packed_mask(mask):
    a = np.asarray(mask, dtype=bool)
    if a.ndim != 1:
        stop("MASK_DIMENSION_INVALID")
    return dict(N=len(a), bitorder='little', hex=np.packbits(a, bitorder='little').tobytes().hex())


def unpacked_mask(record):
    b = bytes.fromhex(record['hex'])
    if record['bitorder'] != 'little' or len(b) != (record['N'] + 7) // 8:
        stop("PACKED_MASK_IDENTITY_INVALID")
    bits = np.unpackbits(np.frombuffer(b, dtype=np.uint8), bitorder='little')
    if bits[record['N']:].any():
        stop("PACKED_MASK_PADDING_NONZERO")
    return bits[:record['N']].astype(bool)


class StageObserver:
    """Passive copied receipt values; no verdict, output mutation or feedback."""
    def __init__(self, context_id, path_index_maps=None):
        if not context_id:
            stop("OBSERVER_CONTEXT_MISSING")
        self.context_id, self.records, self.seen = context_id, [], set()
        self.path_index_maps = {m: tuple(v) for m, v in (path_index_maps or {}).items()}
        self.unions, self.lifecycle, self.run_count = {}, [], 0

    def observe(self, result, model, mult):
        self.run_count += 1
        for stage in result['execution_records']:
            key = (self.context_id, model, mult, stage.keys)
            if key in self.seen:
                stop("DUPLICATE_NATIVE_STAGE_RECORD")
            self.seen.add(key)
            # Native IDs hash trajectory content; identical draw slots can share them.
            # Preserve native IDs and explicitly bind the ordered ensemble index.
            # Chunked/permuted consumers must supply their original index map.
            index_map = self.path_index_maps.get(model, tuple(range(len(stage.path_ids))))
            if len(index_map) != len(stage.path_ids) or len(set(index_map)) != len(index_map):
                stop('PATH_INDEX_MAP_INVALID')
            path_ids = tuple(D.identity((self.context_id, model, index)) for index in index_map)
            details = {row[0]: row[1:] for row in stage.rule_details}
            rules = list(stage.rule_results)
            if not any(row[0] == 'funded_investment_floor' for row in rules):
                rules.append(('funded_investment_floor', np.where(stage.floor_predicate, 'PASS', 'FAIL'), stage.floor_applicable))
            for name, values, mask in rules:
                applicable = np.asarray(mask, bool)
                values = np.asarray(values)
                failed = applicable & (values == 'FAIL')
                unknown = applicable & ~np.isin(values, ['PASS', 'FAIL'])
                stratum = (self.context_id, model, mult, name)
                union = self.unions.setdefault(stratum, {'applicable': set(), 'failed': set(), 'not_evaluable': set()})
                for label, bits in [('applicable', applicable), ('failed', failed), ('not_evaluable', unknown)]:
                    union[label].update(pid for pid, yes in zip(path_ids, bits) if yes)
                witness = None
                indices = np.flatnonzero(failed | unknown)
                if len(indices):
                    n = int(indices[0])
                    witness = dict(path_id=path_ids[n], native_path_id=stage.path_ids[n], session=stage.session,
                                   plan_id=stage.plan_ids[n], evaluation_id=stage.evaluation_ids[n],
                                   intent=stage.intent, native_detail_missing=name not in details)
                    if name in details:
                        num, den, limit, epsilon, allowance = details[name]
                        witness.update(numerator=float(num[n]), denominator=float(den[n]), limit=limit,
                                       arithmetic_epsilon=epsilon, economic_allowance=allowance)
                    witness.update(qf=stage.qf[n].tolist(), qd=stage.qd[n].tolist(), prices=stage.prices[n].tolist(),
                                   cash=float(stage.cash[n]), fees=float(stage.fees[n]), IM=float(stage.im[n]),
                                   NAV_stage=float(stage.nav[n]), N0=float(stage.decision_origin_nav[n]),
                                   dated_pending=stage.pending[n].tolist())
                self.records.append(dict(context_id=self.context_id, model=model, multiplier=mult,
                                         keys=stage.keys, path_ids=path_ids, native_path_ids=stage.path_ids, path_index_map=index_map, rule=name,
                                         applicable=packed_mask(applicable), failed=packed_mask(failed),
                                         not_evaluable=packed_mask(unknown), witness=witness))
        self.lifecycle.extend(deepcopy(result.get('plan_history', [])))

    def receipt(self):
        return dict(context_id=self.context_id, native_runs_observed=self.run_count, records=deepcopy(self.records), lifecycle=deepcopy(self.lifecycle),
                    path_unions=[dict(context=k[0], model=k[1], multiplier=k[2], rule=k[3],
                                      **{label: sorted(ids) for label, ids in masks.items()})
                                 for k, masks in sorted(self.unions.items())], verdict_authority='NATIVE_ENGINE_ONLY')


@dataclass
class AttemptJournal:
    attempt_id: str
    predecessor_id: str | None = None
    events: list | None = None
    state: str = 'STARTED'

    def __post_init__(self):
        if self.events is None:
            self.events = []

    def record(self, kind, **data):
        self.events.append(dict(attempt_id=self.attempt_id, predecessor_id=self.predecessor_id, kind=kind, **data))


def validate_retry(previous, commitment, *, checkpoint_identity, uncertain_draw=False, lost_solve=False):
    if (previous['commitment'] != commitment.identity or uncertain_draw or lost_solve
            or not checkpoint_identity or previous.get('economic_fail')
            or previous.get('checkpoint_identity') != checkpoint_identity):
        stop("RETRY_NOT_EXACT_PRECOMMITTED_CHECKPOINT")
    return True  # Permission is checked separately; this function never executes retry.


def run_six_slots(su, spec, u, *, prepare, solver=FR.solve, journal, commitment, synthetic=False):
    """Prospective PhaseA: no admission/candidate(), exactly six planned calls."""
    require_execution_authority(commitment, synthetic_callbacks=synthetic and solver is not FR.solve)
    if su.adm:
        stop("PHASE_A_ADMISSION_ALREADY_PRESENT")
    if not np.isfinite(u) or u < 1:
        stop('CONSTRUCTION_ENVELOPE_INVALID')
    slots, plans, keys = [], [], set()
    journal.record('PHASE_A_START', commitment=commitment.identity, generation=generation_identity(su))
    try:
        for index, (objective, e) in enumerate(SLOTS):
            journal.record('SOLVE_CALL', index=index, objective=objective, es_target=e)
            solved = solver(su, spec, e, objective, ofz_upper_bound=.40 / u)
            row = dict(slot=index, objective=objective, es_target=e, solve_status=solved['status'])
            if solved['x'] is None:
                row['status'] = 'INFEASIBLE'
            else:
                ex = prepare(solved['x'], index)
                ex['role'] = 'NEW'
                plan_binding(ex)
                row['descriptor'] = plan_descriptor(ex)
                row['quantity_proof'] = validate_quantity_envelope(ex, fin_names=su.fin_names,
                    inherited_qf=su.book0.qf, nav0=su.nav0, price0=su.unit_price['OFZ_FUNDED'], u=u)
                cert = REF.certify_cached(ex)
                key = D.identity(row['descriptor'])
                if cert or not ex.get('reference_eligible') or ex.get('reference_plan_state') != 'COMPLETED':
                    row['status'] = 'INELIGIBLE'
                elif key in keys:
                    row['status'] = 'DUPLICATE'
                else:
                    row['status'] = 'ELIGIBLE'
                    ex['role'] = 'NEW'
                    plans.append(ex)
                    keys.add(key)
            slots.append(row)
            journal.record('SLOT', **row)
        journal.state = 'PHASE_A_COMPLETE'
        return slots, plans
    except Exception as exc:
        journal.state = 'PARTIAL_STOP_FOR_MASTER'
        journal.record('INVALID_SLOT', slot=len(slots), status='INVALID', reason=str(exc))
        journal.record('STOP', reason=str(exc), completed_slots=len(slots))
        raise


def evaluate_plan(su, ex, observer=None):
    """Native full admission of a pre-bound plan; no candidate regeneration."""
    require_matched_reference(ex)
    spec = ex['spec']
    out = dict(ex)
    out['rows'] = FR._engine_rows(su, out, out['orders'], bool(np.any(out['authorized_qd']) or np.any(su.book0.qd)), spec, observer=observer)
    out['det'] = FR.det_survival(su, None, spec, out, execution_case_provider=case_provider(su, getattr(su, "_headroom_cases", None)), observer=observer)
    completion = REF._completion(out, out['execution_plan_id'])['stage']
    out['w_exec'] = {r: float(out['qf'][su.fin_names.index(r)] * su.unit_price[r] / su.nav0)
                     if r in FR.FUNDED else float(out['reference_notional'][su.din_names.index(r)] / su.nav0) for r in su.routes}
    out['hard'] = FR.hard_checks(su, out['w_exec'], spec, out['qf'], out['qd'],
        funded_fraction=out['funded_fraction_at_execution'], net_issuer=out['net_issuer'],
        issuer_units=out['net_issuer_units'], derivative_gross=float(completion.derivative_gross[0] / su.nav0))
    out['hard'].update(reference_plan_eligibility=out['reference_eligible'],
                       deterministic_stress_survival=all(r['FULL_DETERMINISTIC_ADMISSION_PASS'] == 'PASS' for r in out['det']))
    verdict = FR.verdict(su, out, spec.budget)
    xe = np.array([out['w_exec'][r] for r in su.routes])
    return dict(id=out.get('control_id', out['execution_plan_id']), role=out['role'], ex=out,
                stage_receipt=observer.receipt() if observer is not None else None,
                passed=verdict.passed, admission_status=verdict.status, reasons=verdict.reasons,
                expected_net=FR.expected_net(su, xe, spec.kappa), es95=FR.es_of(su, out),
                turnover_cost=FR.turnover_cost(su, xe), execution_plan_id=out['execution_plan_id'])


def validate_origin_case(su, spec):
    qf, qd = np.asarray(su.book0.qf), np.asarray(su.book0.qd)
    expected = np.array([40000. if n == 'SMLT' else 0. for n in su.fin_names])
    if (str(su.t.date()) != '2026-09-25' or su.nav0 != 20_000_000.
            or su.unit_price.get('SMLT') != 250. or su.book0.cash != 10_000_000.
            or su.book0.pending or not np.array_equal(qf, expected) or np.any(qd)
            or spec.delay != 1 or spec.budget != .15 or spec.cap_net != .10
            or spec.floor_hard != .50 or spec.floor_target != .525 or spec.floor_suspended
            or spec.fixed or spec.kappa != 0 or not spec.mandate['overlay_rule']):
        stop('ONE_CUTOFF_ORIGIN_CASE_MISMATCH')


def run_bounded_experiment(su, spec, controls, commitment, *, journal, envelope=None,
                           solver=FR.solve, prepare=None, evaluate=evaluate_plan, synthetic=False):
    """Explicit future opt-in driver. No call by default policy/reporting code."""
    if commitment.read()['mode'] == 'FUTURE_ECONOMIC' and (synthetic or envelope is not None
            or solver is not FR.solve or prepare is not None or evaluate is not evaluate_plan):
        stop('FIXTURE_OVERRIDE_FORBIDDEN_IN_ECONOMIC_MODE')
    require_execution_authority(commitment, synthetic_callbacks=synthetic and solver is not FR.solve and evaluate is not evaluate_plan)
    try:
        if commitment.read()['mode'] == 'FUTURE_ECONOMIC':
            if any(not isinstance(c, ControlRecord) for c in controls):
                stop('FROZEN_CONTROL_RECORDS_REQUIRED')
            pinned = commitment.read()['controls']
            if [(c.id, c.member_hash, c.plan_id) for c in controls] != [(r['id'], r['member_sha256'], r['plan_id']) for r in pinned]:
                stop('CONTROL_COMMITMENT_MISMATCH')
        controls = [certify_reference(su, c.execution(su)) if isinstance(c, ControlRecord) else c for c in controls]
        if commitment.read()['mode'] == 'FUTURE_ECONOMIC':
            validate_origin_case(su, spec)
        support = envelope if envelope is not None else envelope_from_setup(su)
        slots, plans = run_six_slots(su, spec, support['U'], prepare=prepare or
            (lambda x, _: FR.executable(su, x, 1, spec)), solver=solver, journal=journal,
            commitment=commitment, synthetic=synthetic)
        delayed = []
        for parent in [*plans, *controls]:
            child = delayed_control_case(parent, S09, 4)
            # Authority bytes are sealed; native reference is independently bound in PhaseA.
            child = certify_reference(su, child)
            delayed.append(child)
        seal = seal_plan_list(su, slots, plans, controls, delayed, commitment)
        journal.record('LIST_SEALED', identity=seal.identity, payload=seal.payload)
        admission = FR.attach_admission(su, seal, commitment)
        admission._headroom_cases = {ex['delayed_lineage']['parent_plan_id']: ex for ex in delayed}
        results = []
        for index, ex in enumerate([*plans, *controls]):
            if plan_descriptor(ex) != seal.read()['plans'][index]:
                stop("POSTSEAL_PLAN_CHANGED")
            journal.record('CERTIFICATION_START', index=index, plan_id=ex['execution_plan_id'])
            results.append(evaluate(admission, ex, StageObserver(D.identity((admission.headroom_context['identity'], plan_descriptor(ex))))))
            journal.record('CERTIFICATION_RESULT', index=index, status=results[-1]['admission_status'])
        journal.state = 'COMPLETE'
        chosen = FR.select([r for r in results if r['role'] == 'NEW'], su.cfg)
        return dict(slots=slots, seal=seal, results=results, selected=chosen, state='COMPLETE',
                    scope='FINITE_EXPLORATORY_SEARCH_NOT_MARKET_OOS')
    except Exception as exc:
        journal.state = 'PARTIAL_STOP_FOR_MASTER'
        journal.record('STOP', reason=str(exc))
        raise
