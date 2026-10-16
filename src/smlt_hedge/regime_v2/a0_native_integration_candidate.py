"""Task133 bounded native preparation; no authority activation or build call.

The archival adapters own their pinned bytes. This module joins their native
values and the unchanged data consumers. It does not turn a comparison receipt
into an expected REAL root. A source root may remain explicitly incomplete.
"""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import a0_remaining_native_candidate as R
from . import a0_native_book_choice_candidate as N
from .a0_input_authority import REQUEST_FIELDS

REAL_TYPED_ROOTS = ()
SEMANTICS_SHA256 = 'ff40fd5b3d60b207b75f775f83afddb921c2c53fdb816bb839188e1f67313cf4'
MODELS = ('GLOBAL_FORWARD', 'GLOBAL_FORWARD_FHS', 'RECENT_FORWARD',
          'STATE_CONDITIONAL_FORWARD')


def semantics():
    raw = Path(__file__).with_name('a0_native_integration_semantics.json').read_bytes()
    if sha256(raw).hexdigest() != SEMANTICS_SHA256:
        R.stop('TASK133_SEMANTIC_REGISTER_SUBSTITUTION')
    return R.load_json(raw)


def check_root(root, *, require_complete=False):
    """Pinned finite filesystem preimages, checked before cold readers.

    Missing root files are not repaired with decoded snapshots. Partial staging
    may serve the four independently complete authority readers only.
    """
    if type(root) is not type(Path()) or root.is_symlink() or not root.is_dir():
        R.stop('TASK133_EXACT_DETACHED_ROOT')
    root = root.resolve()
    register = semantics()
    expected = {r['path']: r for r in register['root_read_set']}
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != set(expected):
        R.stop('TASK133_ROOT_MISSING_OR_EXTRA_LEAF')
    for name, row in expected.items():
        p = root / name
        if p.is_symlink() or any(parent.is_symlink() for parent in p.parents):
            R.stop('TASK133_ROOT_ALIAS:' + name)
        raw = p.read_bytes()
        if len(raw) != row['bytes'] or sha256(raw).hexdigest() != row['sha256']:
            R.stop('TASK133_ROOT_SOURCE_SUBSTITUTION:' + name)
    if require_complete and register['missing_root_files']:
        R.stop('TASK133_REQUIRED_READER_SOURCE:' + register['missing_root_files'][0])
    return register


def _same(actual, expected, name):
    # Numerical preparation comparison ONLY: native_wire omits attrs/flags.
    # Excluded from canonical equality/copy authority; immutable gate codec owns
    # full metadata comparison. A mutable prepared object is never authority.
    if R.canonical(R.native_wire(actual)) != R.canonical(R.native_wire(expected)):
        R.stop('TASK133_SOURCE_CONSUMER_BITS_OR_TYPES:' + name)


@dataclass(frozen=True)
class NativeCandidate:
    remaining: R.RemainingCandidate
    request: dict
    conf: pd.DataFrame
    panel_now: pd.DataFrame
    calendar: object
    cold_authorities: dict
    root_consumer_values: dict
    preparation_status: str = 'COMPLETE_NATIVE_A0_CANDIDATE_FOR_INDEPENDENT_REVIEW'


def prepare_native(sources132, sources130, root):
    """Passive source transforms plus authorized zero-draw inactive precommit.

    No Engine/Model/recal/ForwardLibrary, forecast recipe, solve or build. The
    original SimpleNamespace preparation type is preserved; P is the consumer's
    actual dict contract. Raw source membership and typed values remain distinct.
    """
    register = check_root(root)
    remaining = R.prepare_remaining(sources132)
    book = N.materialize_archival_book({n: sources130[n] for n in N.BOOK_SOURCES})
    conf, archived_panel, choice = N.materialize_combined_conf_choice_panel(
        {n: sources130[n] for n in N.COMBINED_SOURCES})
    # Only passive imported constructors and data consumers, inspected beforehand.
    from ..adaptive.scenarios import history_panel
    from .session_calendar import ForecastCalendar
    from .execution_inputs import tariffs
    from . import clocks, lk, headroom_a as HA
    from .cfg import load

    root = root.resolve()
    ctx = remaining.ctx_phase2
    if type(ctx) is not SimpleNamespace:
        R.stop('TASK133_RETAINED_PHASE2_CONTEXT_TYPE')
    for phase in (remaining.ctx_phase1, ctx):
        for name in ('root', 'proc', 'interim'):
            setattr(phase, name, root / register['root_mapping'][name])
    _same(load(root), remaining.cfg, 'native_cfg_load')
    for reader in (tariffs, clocks.authority, lk.authority):
        if reader.cache_info().currsize != 0:
            R.stop('TASK133_AMBIENT_WARMED_LRU')
    cold = {'fee_authority_table': tariffs(str(root)),
            'margin_authority_daily_and_transitions': clocks.authority(str(root)),
            'LK_authority': lk.authority(str(root))}
    for name, value in cold.items():
        _same(value, remaining.snapshots[name], name)
    # Input adapters only. These closures parse retained files and select origin
    # tape/terms; they do not reach a scenario/model/optimization producer.
    from ..adaptive import frontier as F, market as M
    from .execution_inputs import funded_tape
    check_root(root, require_complete=True)
    now = pd.DatetimeIndex([pd.Timestamp('2026-09-25')])
    broker = float(ctx.cfg['costs']['broker_fee_futures_rub_per_contract'])
    consumed = {'prices_csv': pd.read_csv(ctx.proc / 'prices.csv', index_col=0, parse_dates=True)}
    _same(consumed['prices_csv'], remaining.snapshots['fresh_cache_v2_prices'], 'prices_csv')
    for name in R.FUNDED:
        consumed['funded_capacity_origin_' + name] = funded_tape(ctx, remaining.cfg, name, now)
        _same(consumed['funded_capacity_origin_' + name],
              remaining.snapshots['funded_capacity_origin_' + name], 'funded_tape.' + name)
    for family in R.FAMILIES:
        value = M.deriv_history(ctx, family, now, remaining.P['usd'], broker)
        consumed['combined_terms_origin_' + family] = value
        _same(value, remaining.snapshots['combined_terms_origin_' + family], 'deriv_history.' + family)
    for name in ('SMLT', 'EQ_FUNDED', 'OFZ_FUNDED', 'GOLD_FUNDED', 'MMF'):
        consumed['price_at_' + name] = F._price_at(ctx, remaining.P, name, now[0])
    leaves = remaining.snapshots['forecast_session_calendar126_leaves']
    if tuple(leaves) != ('sessions', 'dates', 'evidence'):
        R.stop('TASK133_FORECAST_CALENDAR_FIELDS')
    calendar = ForecastCalendar(**leaves)
    if (type(calendar.sessions) is not pd.DatetimeIndex or
            type(calendar.dates) is not pd.DatetimeIndex or
            type(calendar.evidence) is not pd.DataFrame or
            len(calendar.dates) != 127 or not calendar.dates.is_monotonic_increasing or
            not calendar.dates.is_unique or calendar.dates[0] != pd.Timestamp('2026-09-25') or
            not calendar.dates.equals(calendar.evidence.index)):
        R.stop('TASK133_FORECAST_CALENDAR_AXES')
    # Actual panel_now closure, with the source's existing proxy rules unchanged.
    panel_now = history_panel(remaining.P, pd.Timestamp('2026-09-25'),
                              remaining.cfg['r3']['frontier']['estimation_start'])
    # The accepted Task130 adapter explicitly uses an object column axis. The
    # unchanged history_panel DataFrame constructor in pinned pandas3 produces
    # a str column axis. This candidate schema transform changes no cell/row
    # bytes and does not modify the accepted archival adapter or consumer.
    panel_expected = archived_panel.copy(deep=True)
    panel_expected.columns = pd.Index(N.PANEL_COLUMNS, dtype='str', name=None)
    _same(panel_now, panel_expected, 'panel_now_source_transform_str_column_axis')
    store = R.Sources(sources132)
    observed = HA.runtime_identity()  # one external observation, no draws
    commitment = HA.precommit(store.json('protocol/manifest.json'),
        scope_bytes=store.read('protocol/scope.json'), observed_runtime=observed,
        mode='FUTURE_ECONOMIC', authority={})  # one internal observation
    request = dict(ctx=ctx, cfg=remaining.cfg, P=remaining.P,
        t=pd.Timestamp('2026-09-25'), universe='MAIN', views=remaining.views,
        book0=book, capital=None, n_gen=2000, n_adm=2000, models=MODELS,
        state_choice=choice, seed_offset=0, panel_end=None, px=None,
        available0=None, cost_mult=1.0, generation_only=True,
        headroom_commitment=commitment)
    check_request_structure(request)
    check_root(root)  # post-read source preservation, not completion of the root
    return NativeCandidate(remaining, request, conf, panel_now, calendar, cold, consumed)


def check_request_structure(request):
    """Comparison-only bounded type/branch preflight. Not full A0 validation."""
    if type(request) is not dict or tuple(request) != REQUEST_FIELDS:
        R.stop('TASK133_EXACT_19_ARGUMENTS')
    for name, value in [('n_gen', 2000), ('n_adm', 2000), ('seed_offset', 0)]:
        if type(request[name]) is not int or request[name] != value:
            R.stop('TASK133_EXACT_INT_NOT_BOOL:' + name)
    if (type(request['models']) is not tuple or request['models'] != MODELS or
            request['generation_only'] is not True or
            type(request['cost_mult']) is not float or request['cost_mult'] != 1.0 or
            any(request[n] is not None for n in ('capital', 'panel_end', 'px', 'available0'))):
        R.stop('TASK133_METHOD_ORDER_OR_BRANCH')
    if (type(request['t']) is not pd.Timestamp or
            request['t'].isoformat() != '2026-09-25T00:00:00' or
            type(request['universe']) is not str or request['universe'] != 'MAIN' or
            type(request['P']) is not dict or type(request['ctx']) is not SimpleNamespace):
        R.stop('TASK133_NATIVE_REQUEST_TYPES')
    from ..adaptive.engine import Book
    from .headroom_a import Commitment
    if type(request['book0']) is not Book or type(request['headroom_commitment']) is not Commitment:
        R.stop('TASK133_NATIVE_BOOK_OR_COMMITMENT')
    c = request['headroom_commitment'].read()
    if (c['mode'] != 'FUTURE_ECONOMIC' or c['authority'] != {} or
            c['base_head'] != '142e3d6136636e4ef2a38b3edbf9cf691dc7a62a' or
            c['base_tree'] != '8151ca6be3315ae2f3dac7e8f80662bb8e067b20' or
            c['scope']['scope_reconciliation_sha256'] != R.SCOPE_SHA or
            c['cutoff'] != '2026-09-25' or type(c['H']) is not int or c['H'] != 126 or
            type(c['n']) is not int or c['n'] != 2000 or
            c['models'] != list(MODELS) or c['generation_seeds'] != [20260925+101*k for k in range(4)] or
            c['admission_seeds'] != [20960925+101*k for k in range(4)]):
        R.stop('TASK133_INACTIVE_METHOD_SEED_ROLE')
    book = request['book0']
    if (set(vars(book)) != {'qf', 'qd', 'cash', 'pending', 'series', 'capacity', 'regime'} or
            type(book.qf) is not np.ndarray or book.qf.dtype.str != '<f8' or
            book.qf.tobytes() != np.array([40000.,0.,0.,0.,0.,0.], dtype='<f8').tobytes() or
            type(book.qd) is not np.ndarray or book.qd.dtype.str != '<f8' or
            book.qd.tobytes() != np.zeros(5,dtype='<f8').tobytes() or
            type(book.cash) is not float or book.cash != 10000000. or
            book.pending != {} or book.series is not None or book.capacity is not None or
            book.regime != 'ORDINARY'):
        R.stop('TASK133_ARCHIVAL_SEVEN_FIELD_BOOK')
    return 'STRUCTURE_ONLY_NOT_A0_ADMISSION'


def compare_preparation(request, prepared):
    """Compare a request with an archival-source preparation, never admit it.

    Runtime/method/seed/authority bytes are immutable in the inactive reference.
    A later authority receipt must produce a new reference, independent review
    and expected binding; no silent mutation is accepted here. This helper does
    not make its supplied reference independently accepted global authority.
    """
    if type(prepared) is not NativeCandidate:
        R.stop('TASK133_REFERENCE_CANDIDATE_TYPE')
    check_request_structure(request)
    if request['headroom_commitment'] != prepared.request['headroom_commitment']:
        R.stop('TASK133_COMMITMENT_RUNTIME_METHOD_OR_ROLE_CHANGED')
    _same(vars(request['book0']), vars(prepared.request['book0']), 'book0')
    for name in REQUEST_FIELDS:
        if name not in ('book0', 'headroom_commitment'):
            _same(request[name], prepared.request[name], 'request.' + name)
    return 'MATCH_SOURCE_PREPARATION_ONLY_NOT_ACCEPTED_AUTHORITY'


def detach_candidate_request(request, prepared):
    """Inert numerical diagnostic copy; never full-profile equality authority.

    Canonical validate_entry uses independently frozen expected bytes and never
    calls this helper or compare_preparation. Mutable prepared co-mutation cannot
    affect the canonical gate. Ownership uses the existing approved native copy.
    """
    from . import a0_input_authority as A0
    if type(prepared) is not NativeCandidate:
        R.stop('TASK133_REFERENCE_CANDIDATE_TYPE')
    A0._preflight(request, native=True)
    A0._preflight(prepared.request, native=True)
    compare_preparation(request, prepared)
    owned = A0._detach(request, native=True)
    compare_preparation(owned, prepared)
    return owned
