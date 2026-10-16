"""Task132 finite retained-byte adapters. Provisional values, never admission.

No filesystem reader, project producer, live request, LRU, pickle, or RNG call.
Task117/120 descriptor vocabulary is decoded only inside fixed source membership.
Snapshots are data; installing them in an ambient module cache is not authorized.
The two phase namespaces are detached from source readers and remain incomplete.
"""
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import PurePosixPath, PosixPath
import struct
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

REAL_TYPED_ROOTS = ()
REGISTRY_SHA = 'be9af5197555b0f76145d8990c1c43e83e19c06bf72ed0085c2628bb5d9bb90f'
MANIFEST_SHA = '93e7b54b8e003f595e3cba53f1823b4b3d5716f91b61d97ab1b62a81c550970e'
SCOPE_SHA = 'da3f69bb2d3e335dfc1f85feab0b9130e7806e59fcb7a66fe3be1cfb26498fd2'
EXTRAS = {
    'configs/project.yaml': ('c39701b03a53f327b8e56b342aef5d91010972c7ea0e9e81403c939675a308fe', 20378),
    'configs/adaptive.yaml': ('61e7f1445d9ca33c7fed0856bbc3f74a3c938f5994e40d036a27e8b09138b3e4', 37941),
    'configs/regime_v2.yaml': ('7aa1c1955d3de490e39ba8a2ee5f332f45703a35d773554e189b6b9972a5f92d', 23363),
    'protocol/manifest.json': ('0b36758145be4b9b0f9216a41b300152d2930f8eba590d909efc241009180410', 72117),
    'protocol/scope.json': (SCOPE_SHA, 641987),
}
FUTURES = ('SMLT_FUT', 'EQ_HEDGE_FUT', 'RATES_HEDGE_FUT', 'OIL_FUT',
           'BTC_MOEX_FUT', 'GOLD_PERP', 'CNY_PERP')
FAMILIES = ('SMLT_FUT', 'MXI_FUT', 'RGBI_FUT', 'CNYRUBF', 'BR_FUT')
FUNDED = ('SMLT', 'EQ_FUNDED', 'OFZ_FUNDED', 'GOLD_FUNDED', 'BTC_RUB_EXPERIMENTAL', 'MMF')
T117 = 'Task117/review/RunA/'
T120 = 'Task120/review/RunA/Phase2/'
SNAPSHOTS = (
    'fee_authority_table', 'margin_authority_daily_and_transitions', 'LK_authority',
    'historical_capacity_margin_LK_currency_conversion',
    'fresh_cache_v2_prices', 'fresh_cache_v2r2_btc_tape',
    *(kind + '_' + name for kind in ('combined_terms_origin', 'combined_terms_history',
                                    'forecast_metadata', 'legacy_roll_plan') for name in FAMILIES),
    *(kind + '_' + name for kind in ('funded_capacity_origin', 'funded_capacity_history') for name in FUNDED),
)
ROOTS = (
    T117 + 'P.cal.json', T117 + 'P.R.json', T117 + 'P.B.json',
    T117 + 'P.usd.json', T117 + 'P.usd_lag0.json', T117 + 'P.btc_rub_direct.json',
    T117 + 'P.meta.json', *(T117 + 'P.fut.' + n + '.json' for n in FUTURES),
    *(T117 + 'ctx' + n + '.json' for n in ('_adaptive_pit', '_adaptive_fut', '_adaptive_mr')),
    T117 + 'ctx_PHASE2_CONTRACT.json', T117 + 'CLOCK_AND_UNITS.json',
    T120 + 'forecast_session_calendar126.json',
    'Task120/review/RunA/PHASE_ORDER_AND_PARENT_WITNESS.json',
    'Task120/review/UNITS_CLOCKS_AND_ASSUMPTION_QUALIFICATIONS.json',
    'Task120/review/UNRESOLVED_GAPS_AND_EXACT_TYPES.json',
    *(T120 + n + '.json' for n in SNAPSHOTS),
    'Retained/views_provisional.json', 'Retained/full_config_wire.json',
    'Retained/setup_authority.json', 'Retained/expected_return_inputs.csv',
    'Task93/request_design.json', *EXTRAS,
)
CLOCK = 'R24_ASSUMPTION_QUALIFIED; NONCAUSAL_AT_ORIGIN where post-cutoff or publication-unverified; NO_STRICT_PIT_OR_MARKET_OOS'


class EvidenceMissing(RuntimeError):
    pass


def stop(reason):
    raise EvidenceMissing('EVIDENCE_MISSING:' + reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def _pairs(rows):
    result = {}
    for k, v in rows:
        if k in result:
            stop('DUPLICATE_JSON_KEY:' + k)
        result[k] = v
    return result


def load_json(raw):
    if type(raw) is not bytes:
        stop('EXACT_BYTES_REQUIRED')
    try:
        return json.loads(raw, object_pairs_hook=_pairs,
                          parse_constant=lambda x: stop('UNTAGGED_NONFINITE:' + x))
    except (ValueError, UnicodeError):
        stop('INVALID_NONEXECUTABLE_JSON')


def _relative(name):
    if (type(name) is not str or not name or '\\' in name or
            PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts or
            str(PurePosixPath(name)) != name):
        stop('UNSUPPORTED_DATA_PATH:' + str(name))
    return name


class Sources:
    """Fixed Task126 pins plus five independently recovered supplementary pins.

    Capture bytes before conversion. A caller cannot supply new expected hashes.
    Missing protocol bytes stop field19 alone; missing native dependencies stop
    their family, without inventing values. Extra provided paths are rejected.
    """
    def __init__(self, sources):
        if type(sources) is not dict or any(type(v) is not bytes for v in sources.values()):
            stop('EXACT_SOURCE_BYTE_DICT')
        self.owned = dict(sources)
        raw = self.owned.get('expected_registry.json', b'')
        if sha256(raw).hexdigest() != REGISTRY_SHA:
            stop('FIXED_TASK126_REGISTRY')
        self.registry = load_json(raw)
        self.pins = {r['path']: (r['sha256'], r['bytes']) for r in self.registry['read_set']}
        self.pins.update(EXTRAS)
        self.reads = {'expected_registry.json'}
        for name, value in self.owned.items():
            _relative(name)
            if name == 'expected_registry.json':
                continue
            if name not in self.pins:
                stop('UNSUPPORTED_DATA_PATH:' + name)
            self._check(name, value)

    def _check(self, name, value):
        pin = self.pins[name]
        if sha256(value).hexdigest() != pin[0] or len(value) != pin[1]:
            stop('SOURCE_PREIMAGE_SUBSTITUTION:' + name)

    def read(self, name):
        _relative(name)
        if name not in self.pins:
            stop('UNSUPPORTED_DATA_PATH:' + name)
        if name not in self.owned:
            stop('MISSING_SOURCE:' + name)
        self._check(name, self.owned[name])
        self.reads.add(name)
        return self.owned[name]

    def json(self, name):
        return load_json(self.read(name))

    def sibling(self, parent, name):
        _relative(name)
        if '/' in name:
            stop('UNSUPPORTED_DESCRIPTOR_PATH:' + name)
        return str(PurePosixPath(parent).parent / name)

    def receipt(self):
        return [{'path': n, 'sha256': sha256(self.owned[n]).hexdigest(),
                 'bytes': len(self.owned[n])} for n in sorted(self.reads)]


def logical(node):
    """Closed scalar tags from the retained exporters; no class import/eval."""
    t = node['type']
    if t == 'None':
        return None
    if t == 'pd.NA':
        return pd.NA
    if t == 'pd.NaT':
        return pd.NaT
    if t in ('float', 'float64'):
        value = struct.unpack('<d', bytes.fromhex(node['float64_le_hex']))[0]
        return value if t == 'float' else np.float64(value)
    if t in ('int', 'int64', 'int32'):
        if type(node['value']) is not int:
            stop('EXACT_INT_NOT_BOOL')
        return {'int': int, 'int64': np.int64, 'int32': np.int32}[t](node['value'])
    if t in ('str', 'bool'):
        if type(node['value']) is not {'str': str, 'bool': bool}[t]:
            stop('EXACT_LOGICAL_TYPE:' + t)
        return node['value']
    if t == 'Timestamp':
        a = np.frombuffer(bytes.fromhex(node['bytes_hex']), dtype=node['asm8_dtype'])
        value = pd.Timestamp(a[0])
        if node['iso'].endswith('+00:00'):
            value = value.tz_localize('UTC')
        if value.isoformat() != node['iso'] or value.asm8.tobytes().hex() != node['bytes_hex']:
            stop('TIMESTAMP_CLOCK_OR_RESOLUTION')
        return value
    stop('UNSUPPORTED_LOGICAL_TAG:' + t)


def vector(store, parent, desc):
    dtype = np.dtype(desc['numpy_dtype'])
    if dtype.str not in ('<f8', '<i8', '<i4', '|b1', '|O', '<M8[us]', '<M8[ns]', '<M8[ms]'):
        stop('UNSUPPORTED_RETAINED_DTYPE:' + dtype.str)
    if dtype.hasobject:
        vals = desc['values']
        a = np.empty(desc['shape'], dtype=object)
        a.flat[:] = [logical(n) for n in vals]
    else:
        raw = store.read(store.sibling(parent, desc['file']))
        if (len(raw) != desc['size_bytes'] or sha256(raw).hexdigest() != desc['sha256'] or
                desc['order'] != 'C'):
            stop('VECTOR_SOURCE_IDENTITY:' + parent)
        a = np.frombuffer(raw, dtype=dtype).copy().reshape(desc['shape'])
    _compare_vector(a, desc)
    return a


def _scalar_wire(x):
    if x is None:
        return {'type': 'None'}
    if x is pd.NA:
        return {'type': 'pd.NA'}
    if x is pd.NaT:
        return {'type': 'pd.NaT'}
    if isinstance(x, (float, np.floating)):
        return {'type': type(x).__name__, 'float64_le_hex': struct.pack('<d', float(x)).hex()}
    if isinstance(x, (bool, np.bool_)):
        return {'type': 'bool', 'value': bool(x)}
    if isinstance(x, (int, np.integer)):
        return {'type': type(x).__name__, 'value': int(x)}
    if type(x) is str:
        return {'type': 'str', 'value': x}
    if type(x) is pd.Timestamp:
        return {'type': 'Timestamp', 'iso': x.isoformat(), 'asm8_dtype': x.asm8.dtype.str,
                'bytes_hex': x.asm8.tobytes().hex()}
    stop('UNSUPPORTED_NATIVE_SCALAR:' + type(x).__name__)


def _compare_vector(a, desc):
    if a.dtype.str != desc['numpy_dtype'] or list(a.shape) != desc['shape']:
        stop('NATIVE_VECTOR_DTYPE_OR_SHAPE')
    if a.dtype.hasobject:
        if [_scalar_wire(v) for v in a.flat] != desc['values']:
            stop('NATIVE_TAGGED_VALUES')
    else:
        if sha256(a.tobytes(order='C')).hexdigest() != desc['sha256']:
            stop('NATIVE_NUMERIC_BITS')
        if a.dtype.kind == 'f':
            for key, mask in [('NaN_mask_u1_hex', np.isnan(a)),
                              ('signed_zero_mask_u1_hex', (a == 0) & np.signbit(a))]:
                if mask.astype('u1').tobytes().hex() != desc[key]:
                    stop('NATIVE_MASK:' + key)


def index(store, parent, desc):
    a = vector(store, parent, desc['values'])
    name = logical(desc['name'])
    if desc['timezone'] not in ('None', 'UTC'):
        stop('UNSUPPORTED_TIMEZONE:' + parent)
    kind = desc['class']
    if kind == 'DatetimeIndex':
        result = pd.DatetimeIndex(a, name=name)
        if str(result.tz) != desc['timezone']:
            stop('NATIVE_INDEX_TIMEZONE:' + parent)
    elif kind == 'RangeIndex':
        if not np.array_equal(a, np.arange(len(a), dtype=a.dtype)):
            stop('UNSUPPORTED_RANGE_AXIS:' + parent)
        result = pd.RangeIndex(len(a), name=name)
    elif kind == 'Index':
        result = pd.Index(a, name=name, dtype=a.dtype)
    else:
        stop('UNSUPPORTED_INDEX_CLASS:' + kind)
    _compare_vector(result.to_numpy(), desc['values'])
    if [str(x) for x in result] != desc['labels'] or type(result).__name__ != kind:
        stop('NATIVE_AXIS_ORDER_OR_TYPE:' + parent)
    return result


def frame(store, name):
    desc = store.json(name)
    if desc['kind'] != 'DataFrame' or set(desc['vectors']) != set(desc['columns']):
        stop('EXACT_FRAME_SCHEMA:' + name)
    ix = index(store, name, desc['index'])
    columns = {}
    for c in desc['columns']:
        v = desc['vectors'][c]
        columns[c] = pd.Series(vector(store, name, v), index=ix, dtype=v['pandas_dtype'], copy=True)
    result = pd.DataFrame(columns, index=ix, copy=True)
    result.columns.name = logical(desc['column_index_name'])
    compare_frame(result, desc)
    return result


def compare_frame(value, desc):
    if (type(value) is not pd.DataFrame or list(value.shape) != desc['shape'] or
            list(value.columns) != desc['columns'] or
            _scalar_wire(value.columns.name) != desc['column_index_name']):
        stop('NATIVE_FRAME_AXES_OR_SHAPE')
    ix = desc['index']
    if (type(value.index).__name__ != ix['class'] or [str(x) for x in value.index] != ix['labels'] or
            _scalar_wire(value.index.name) != ix['name'] or str(getattr(value.index, 'tz', None)) != ix['timezone']):
        stop('NATIVE_FRAME_INDEX')
    _compare_vector(value.index.to_numpy(), ix['values'])
    for c in desc['columns']:
        if str(value[c].dtype) != desc['vectors'][c]['pandas_dtype']:
            stop('NATIVE_PANDAS_DTYPE:' + c)
        _compare_vector(value[c].to_numpy(), desc['vectors'][c])


def series(store, name):
    d = store.json(name)
    result = pd.Series(vector(store, name, d['values']), index=index(store, name, d['index']),
                       name=logical(d['name']), dtype=d['values']['pandas_dtype'], copy=True)
    _compare_vector(result.to_numpy(), d['values'])
    return result


def retained_value(store, parent, n):
    """Task120's fixed data-only snapshot vocabulary; dataclasses are refused."""
    t = n['type']
    if t == 'DataFrame':
        return frame(store, store.sibling(parent, n['descriptor']))
    if t == 'Index':
        return index(store, parent, n['descriptor'])
    if t == 'ndarray':
        if 'descriptor' in n:
            return vector(store, parent, n['descriptor'])
        if n['numpy_dtype'] != '|O':
            stop('UNSUPPORTED_OBJECT_SNAPSHOT')
        vals = [retained_value(store, parent, x) for x in n['logical_values_flat_C']]
        a = np.empty(n['shape'], dtype=object)
        a.flat[:] = vals
        return a
    if t == 'mapping':
        keys = n['ordered_keys']
        if (any(type(k) is not str for k in keys) or len(set(keys)) != len(keys) or
                set(keys) != set(n['fields'])):
            stop('EXACT_SNAPSHOT_MAP_ORDER')
        return {k: retained_value(store, parent, n['fields'][k]) for k in keys}
    if t in ('tuple', 'list'):
        items = [retained_value(store, parent, x) for x in n['items']]
        return tuple(items) if t == 'tuple' else items
    return logical(n)


def _wire_config(n):
    k = n['kind']
    if k == 'dict':
        return {key: _wire_config(v) for key, v in n['fields']}
    if k == 'list':
        return [_wire_config(v) for v in n['items']]
    if k == 'float64':
        return struct.unpack('>d', bytes.fromhex(n['bits_be']))[0]
    if k == 'none':
        return None
    if k in ('str', 'bool'):
        return n['value']
    if k == 'int':
        return int(n['decimal'])
    stop('UNSUPPORTED_CONFIG_WIRE:' + k)


def config_comparison_wire(value):
    """Compare all cfg leaves; historical sorted JSON lost dict insertion order.

    Native dictionary order remains that of the pinned YAML/merge. Lists and
    IEEE scalar words remain ordered/exact in this source-value comparison.
    """
    if type(value) is dict:
        return {'type': 'dict', 'sorted_fields': [[k, config_comparison_wire(value[k])] for k in sorted(value)]}
    if type(value) is list:
        return {'type': 'list', 'items': [config_comparison_wire(v) for v in value]}
    return _scalar_wire(value)


def config(store):
    """Exact cfg.load merge semantics at pinned Task128, without path readers.

    Copy of the reviewed pure merge closure, including its existing `why` default.
    No new defaults. Complete retained config is an independent comparison only:
    a mismatch is reported rather than patched with the historical wire.
    """
    base = yaml.safe_load(store.read('configs/adaptive.yaml'))
    v2 = yaml.safe_load(store.read('configs/regime_v2.yaml'))
    if base['cutoff'] != v2['cutoff']:
        stop('SOURCE_CONFIG_CUTOFF')
    cfg = deepcopy(base)
    cfg['v2'] = v2
    def repaired_name(name):
        return name.replace('ADAPTIVE_REGIME_PIF_V2', 'ADAPTIVE_REGIME_PIF_V2R1').replace('STATIC_INVESTED_PIF_V2', 'STATIC_INVESTED_PIF_V2R1')
    cfg['v2']['policy']['name'] = repaired_name(v2['policy']['name'])
    cfg['v2']['replay']['ledgers'] = [{**s, 'name': repaired_name(s['name'])} for s in v2['replay']['ledgers']]
    cfg['v2']['validation']['incremental_value']['pair'] = [repaired_name(n) for n in v2['validation']['incremental_value']['pair']]
    cfg['v2r1'] = {'output_namespace': 'regime_v2r1_', 'derivative_terms_repair': True,
                   'historical_cost_evidence': 'CUTOFF_RATE_ASSUMPTION_DOWNGRADED',
                   'checkpoint_resume': 'RESTART_FROM_START'}
    cfg['r3']['universes'] = {k: list(v) for k, v in v2['universe'].items()}
    cfg['r3']['bounds'] = {r: {'lo': float(b['lo']), 'hi': float(b['hi']), 'why': b.get('why', '')} for r, b in v2['bounds'].items()}
    cfg['r3']['mandates']['V2_HARD_CHECKS_IN_REGIME_V2'] = {'issuer_net_min': -10.0, 'issuer_net_max': 10.0}
    cfg['stress']['scenarios'] = {**cfg['stress']['scenarios'], **v2['scenarios']['extra_paths']}
    return cfg


def views(store):
    d = store.json('Retained/views_provisional.json')['payload']
    rows = [dict(row) for row in d['setup_rows']]
    cols = d['view_field_order']
    if cols != ['central', 'd_long', 'd_short', 'centre_risk']:
        stop('VIEW_FIELD_ORDER')
    routes = [_wire_config(row['route']) for row in rows]
    if routes != d['setup_route_order']:
        stop('VIEW_ROUTE_ORDER')
    a = np.array([[_wire_config(row[c]) for c in cols] for row in rows], dtype='<f8')
    # Source returns.views_frame uses .set_index('route') and preserves these fields.
    return pd.DataFrame(a, index=pd.Index(routes, dtype=object, name='route'), columns=cols)


def begin_phase2(phase1, flags):
    keys = {'root', 'proc', 'interim', 'cfg', 'cutoff', '_adaptive_pit', '_adaptive_fut', '_adaptive_mr'}
    if type(phase1) is not SimpleNamespace or set(vars(phase1)) != keys:
        stop('CACHE_INHERITANCE_OR_PREMATURE_PHASE2')
    if flags != {'_regime_v2r1_terms': True, '_regime_v2r2_combined': True}:
        stop('SOURCE_PHASE2_FLAGS')
    ctx = SimpleNamespace(**deepcopy(vars(phase1)))
    ctx._regime_v2r1_terms = flags['_regime_v2r1_terms']
    ctx._regime_v2r2_combined = flags['_regime_v2r2_combined']
    return ctx


def phase2_namespace(phase1, prices, btc, flags):
    ctx = begin_phase2(phase1, flags)
    ctx._v2_prices = prices.copy(deep=True)
    ctx._v2r2_btc_tape = btc.copy(deep=True)
    return ctx


def inactive_field19(store):
    missing = ['EVIDENCE_MISSING:ACCEPTED_METHOD_BINDING',
               'EVIDENCE_MISSING:PROTOCOL_RUNTIME_PRNG_BINDING_NO_INITIALIZATION_AUTHORIZED',
               'EVIDENCE_MISSING:GENERATION_ADMISSION_PRODUCER_ROLE_BINDING',
               'EVIDENCE_MISSING:INDEPENDENT_REVIEW_NEW_FREEZE_SEPARATE_ECONOMIC_AUTHORIZATION']
    record = {'status': 'INACTIVE', 'authority_active': False, 'accepted_commitment': False,
              'missing': missing, 'expected_manifest_canonical_sha256': MANIFEST_SHA,
              'expected_scope_raw_sha256': SCOPE_SHA}
    try:
        manifest_raw = store.read('protocol/manifest.json')
        scope_raw = store.read('protocol/scope.json')
        canon = canonical(load_json(manifest_raw))
        if sha256(canon).hexdigest() != MANIFEST_SHA or sha256(scope_raw).hexdigest() != SCOPE_SHA:
            stop('INACTIVE_FIELD19_SOURCE_IDENTITY')
        record.update(manifest_raw_sha256=sha256(manifest_raw).hexdigest(),
                      manifest_canonical_sha256=sha256(canon).hexdigest(),
                      scope_raw_sha256=sha256(scope_raw).hexdigest(), source_bytes_recovered=True)
    except EvidenceMissing as e:
        record.update(source_bytes_recovered=False, component_stop=str(e))
    return record


def native_wire(value):
    """Non-executable full bytes/type transport; dict/axis order is explicit.

    Object pointers never enter the output. pandas extension strings receive an
    explicit new storage identity; historical pointer/storage identity is absent.
    This transport is a comparison receipt, never a REAL expected-root authority.
    """
    if type(value) is pd.DataFrame:
        return {'type': 'pandas.DataFrame', 'shape': list(value.shape),
                'index': native_wire(value.index), 'columns': native_wire(value.columns),
                'vectors': [[c, native_wire(value[c])] for c in value.columns]}
    if type(value) is pd.Series:
        return {'type': 'pandas.Series', 'name': _scalar_wire(value.name),
                'pandas_dtype': str(value.dtype), 'storage': getattr(value.dtype, 'storage', None),
                'index': native_wire(value.index), 'values': native_wire(value.to_numpy())}
    if isinstance(value, pd.Index):
        return {'type': 'pandas.' + type(value).__name__, 'name': _scalar_wire(value.name),
                'timezone': str(getattr(value, 'tz', None)), 'dtype': str(value.dtype),
                'labels': [str(x) for x in value], 'values': native_wire(value.to_numpy())}
    if type(value) is np.ndarray:
        w = {'type': 'numpy.ndarray', 'dtype': value.dtype.str, 'shape': list(value.shape),
             'order': 'C', 'endian': value.dtype.byteorder}
        if value.dtype.hasobject:
            w['logical_values_flat_C'] = [native_wire(x) for x in value.flat]
        else:
            w['raw_hex'] = value.tobytes(order='C').hex()
            if value.dtype.kind == 'f':
                w['NaN_mask_u1_hex'] = np.isnan(value).astype('u1').tobytes().hex()
                w['signed_zero_mask_u1_hex'] = ((value == 0) & np.signbit(value)).astype('u1').tobytes().hex()
        return w
    if type(value) is dict:
        return {'type': 'dict', 'ordered_items': [[k, native_wire(v)] for k, v in value.items()]}
    if type(value) in (tuple, list):
        return {'type': type(value).__name__, 'items': [native_wire(x) for x in value]}
    if type(value) is SimpleNamespace:
        return {'type': 'types.SimpleNamespace', 'ordered_fields': [[k, native_wire(v)] for k, v in vars(value).items()]}
    if type(value) is PosixPath:
        return {'type': 'pathlib.PosixPath', 'relative': str(value), 'root_id': 'TASK132_DETACHED_RETAINED_SOURCE_ROOT'}
    return _scalar_wire(value)


@dataclass(frozen=True)
class RemainingCandidate:
    P: dict
    cfg: dict
    ctx_phase1: SimpleNamespace
    ctx_phase2: SimpleNamespace | None
    views: pd.DataFrame
    snapshots: dict
    receipt: bytes
    field19: bytes

    def transport(self):
        return canonical({'P': native_wire(self.P), 'cfg': native_wire(self.cfg),
                          'ctx_phase1': native_wire(self.ctx_phase1),
                          'ctx_phase2': native_wire(self.ctx_phase2),
                          'views': native_wire(self.views), 'snapshots': native_wire(self.snapshots),
                          'receipt': load_json(self.receipt), 'field19': load_json(self.field19)})


def prepare_remaining(sources):
    store = Sources(sources)
    cfg = config(store)
    P = {'cal': index(store, T117 + 'P.cal.json', store.json(T117 + 'P.cal.json')['index']),
         'R': frame(store, T117 + 'P.R.json'), 'B': frame(store, T117 + 'P.B.json'),
         'usd': series(store, T117 + 'P.usd.json'),
         'fut': {n: frame(store, T117 + 'P.fut.' + n + '.json') for n in FUTURES},
         'usd_lag0': series(store, T117 + 'P.usd_lag0.json'),
         'btc_rub_direct': series(store, T117 + 'P.btc_rub_direct.json'),
         'meta': {k: {leaf: logical(v) for leaf, v in fields.items()}
                  for k, fields in store.json(T117 + 'P.meta.json').items()}}
    if len(P['cal']) != 1178 or P['R'].shape != (1178, 14) or P['B'].shape != (1178, 14):
        stop('EXACT_NATIVE_P_DIMENSIONS')
    if not all(P['cal'].equals(v.index) for v in [P['R'], P['B'], P['usd'], *P['fut'].values()]):
        stop('EXACT_SHARED_P_DATE_AXIS')
    root = PosixPath('retained_task132_source_root')
    ctx1 = SimpleNamespace(root=root, proc=root / 'data/processed', interim=root / 'data/interim',
                           cfg=yaml.safe_load(store.read('configs/project.yaml')), cutoff=cfg['cutoff'])
    if ctx1.cutoff != ctx1.cfg['decision_date']:
        stop('EXACT_CTX_CUTOFF')
    for n in ('_adaptive_pit', '_adaptive_fut', '_adaptive_mr'):
        setattr(ctx1, n, frame(store, T117 + 'ctx' + n + '.json'))
    ctx2 = begin_phase2(ctx1, store.json(T117 + 'ctx_PHASE2_CONTRACT.json')['flags'])
    snapshots, failures = {}, []
    for n in SNAPSHOTS:
        name = T120 + n + '.json'
        try:
            snapshots[n] = retained_value(store, name, store.json(name)['value'])
        except EvidenceMissing as e:
            failures.append({'field': n, 'reason': str(e)})
    # Reuse only exported calendar leaves. Its historical scratch-clone wrapper
    # is not silently turned into an accepted native Calendar type.
    calendar = T120 + 'forecast_session_calendar126.json'
    try:
        n = store.json(calendar)['value']
        snapshots['forecast_session_calendar126_leaves'] = {
            k: retained_value(store, calendar, n['fields'][k]) for k in n['ordered_fields']}
    except EvidenceMissing as e:
        failures.append({'field': 'forecast_session_calendar126_leaves', 'reason': str(e)})
    try:
        ctx2._v2_prices = snapshots['fresh_cache_v2_prices'].copy(deep=True)
        ctx2._v2r2_btc_tape = snapshots['fresh_cache_v2r2_btc_tape'].copy(deep=True)
    except KeyError as e:
        failures.append({'field': 'ctx_phase2', 'reason': 'EVIDENCE_MISSING:' + str(e)})
    # Bind source chronology and qualifications; do not inherit historical LRUs.
    chronology = store.json('Task120/review/RunA/PHASE_ORDER_AND_PARENT_WITNESS.json')
    units = store.json('Task120/review/UNITS_CLOCKS_AND_ASSUMPTION_QUALIFICATIONS.json')
    p_units = store.json(T117 + 'CLOCK_AND_UNITS.json')
    unresolved = store.json('Task120/review/UNRESOLVED_GAPS_AND_EXACT_TYPES.json')
    view = views(store)
    old_cfg = _wire_config(store.json('Retained/full_config_wire.json')['value'])
    cfg_equal = canonical(config_comparison_wire(cfg)) == canonical(config_comparison_wire(old_cfg))
    field19 = inactive_field19(store)
    receipt = {
        'status': 'PARTIAL_NATIVE_PREPARATION_WITH_EXACT_BLOCKERS', 'authority_active': False,
        'families': {'P': 'PARTIAL_UNITS; PASS_COMPLETE_SOURCE_TO_NATIVE_VALUE_COMPARISON_PENDING_INDEPENDENT_ADMISSION',
                     'cfg': 'PASS_PINNED_MERGE' if cfg_equal else 'PARTIAL_RETAINED_CONFIG_DIFF',
                     'ctx': 'PARTIAL_DETACHED_PATHS_LRU_BINDING_AND_CACHE_UNITS',
                     'views': 'PASS_RETAINED_BINARY64_VALUES_NEW_EXPLICIT_SCHEMA',
                     'cache_snapshots': 'PARTIAL_NATIVE_VALUES_WITH_UNRESOLVED_LEAF_UNITS'},
        'cfg_retained_full_wire_equal': cfg_equal,
        'cfg_comparison_scope': 'All typed leaves and list order; native map insertion order comes from pinned YAML/merge, historical sorted JSON map order is not asserted',
        'phase_events': ['FRESH_NAMESPACE_FLAGS_ABSENT', 'P_AND_PHASE1_CACHES_COMPLETE',
                         'PHASE2_FLAGS_AFTER_P', 'RETAINED_PHASE2_SNAPSHOTS_COPIED'],
        'source_chronology': chronology, 'clock_qualification': CLOCK,
        'source_units': units, 'P_source_units': p_units, 'source_limitations': unresolved,
        'failures': failures,
        'exact_missing': ['EVIDENCE_MISSING:ROOT_KEY_LRU_CONSUMER_BINDING_NOT_INSTALLED',
                         'EVIDENCE_MISSING:DETACHED_SOURCE_ROOT_RELOCATION_READER_BINDING',
                         'EVIDENCE_MISSING:PER_LEAF_UNITS_WHERE_SOURCE_TABLE_COLUMNS_NOT_QUALIFIED',
                         'EVIDENCE_MISSING:MODEL_RECAL_DERIVED_CACHE_SNAPSHOT_NOT_RETAINED',
                         'EVIDENCE_MISSING:FORECAST_CALENDAR_NATIVE_WRAPPER_ADMISSION'],
        'Task130': {'head': '110629b3b04ddc7ed85a19c3684054730f284b47',
                    'families': ['book0', 'conf', 'history_panel/panel_now', 'state_choice'],
                    'status': 'SIBLING_CROSS_REFERENCE_ONLY_NOT_CONSUMED_OR_ACCEPTED'},
        'prospective_request_design': store.json('Task93/request_design.json'),
        'build_request_constructed': False, 'REAL_TYPED_ROOTS': [],
        'effects': {'economic_producers': 0, 'RNG_initializations_by_adapter': 0, 'RNG_draws_by_adapter': 0,
                    'pickle_execution': 0, 'source_readers': 0, 'ambient_LRU_installations': 0,
                    'admission': 0, 'precommit': 0, 'runtime_identity': 0},
        'source_receipts': store.receipt(), 'B2': 'BLOCKED', 'REAL_A0': 'NOT_ADMITTED',
        'G2': 'NOT_PASSED', 'ECONOMICS': 'NOT_EXECUTED', 'STOP_FOR_QUANT_MASTER': True,
    }
    return RemainingCandidate(P, cfg, ctx1, ctx2, view, snapshots, canonical(receipt), canonical(field19))
