"""Direct retained source observations, never native A0 or admission authority.

Only two fixed read profiles exist: pinned retained source bytes (Task93 and
explicit D26 Book records) and authored artificial test bytes. No input-derived
expected hash, filesystem reader,
config merge, economic import, default Book.regime or request activation exists.
Decimal -> binary64 bits describe this deterministic wire conversion, NOT
unobserved native dtype/NaN payloads. Unknown native schema remains unresolved.
"""
import csv
from dataclasses import dataclass
from hashlib import sha256
import io
import json
import math
import re
import struct

REAL_TYPED_ROOTS = ()
REAL = 'TASK93_RETAINED_OBSERVATIONS'
ARTIFICIAL = 'TASK108_ARTIFICIAL_ONLY'
REQUEST_FIELDS = (
    'ctx', 'cfg', 'P', 't', 'universe', 'views', 'book0', 'capital', 'n_gen',
    'n_adm', 'models', 'state_choice', 'seed_offset', 'panel_end', 'px',
    'available0', 'cost_mult', 'generation_only', 'headroom_commitment',
)
FUNDED = ('SMLT', 'EQ_FUNDED', 'OFZ_FUNDED', 'GOLD_FUNDED',
          'BTC_RUB_EXPERIMENTAL', 'MMF')
DERIVATIVES = ('SMLT_FUT', 'EQ_HEDGE_FUT', 'RATES_HEDGE_FUT', 'CNY_PERP', 'OIL_FUT')
VIEW_FIELDS = ('central', 'd_long', 'd_short', 'centre_risk')
CSV_COLUMNS = ('route', 'exposure_route', 'prior', 'carry', 'state_adj',
               'mu', 'centre_risk', 'mode')
# Finite direct build reads in accepted frontier.build, NOT transitive closure.
# Whole-config inference/merging and dependency discovery are deliberately absent.
CONFIG_LEAVES = (
    ('r3', 'frontier', 'horizon_sessions'),
    ('r3', 'frontier', 'estimation_start'),
    ('r3', 'frontier', 'seed_generation'),
    ('r3', 'frontier', 'seed_admission_shift'),
    ('r3', 'universes', 'MAIN'),
    ('r3', 'costs', 'btc_experimental'),
    ('r3', 'costs', 'smlt', 'bp'),
    ('r3', 'costs', 'smlt', 'half_tick_rub'),
    ('r3', 'costs', 'funds'),
    ('v2', 'risk', 'block_length'),
    ('v2', 'risk', 'models'),
    ('r1', 'settlement', 'btc_experimental_lag'),
    ('r1', 'liquidity', 'immediate_cash_rate'),
    ('r1', 'liquidity', 'reserve_var_level'),
    ('owner', 'capital_main_rub'),
    ('owner', 'smlt_initial_value_rub'),
)
# SHA preimages copied from Task93 retained source custody, not received inputs.
_REAL_PINS = (
    ('setup_authority.json', 'a2fed9f6ab0903b086392a1610230df3538d721ffbab6dfa3568acca18beed78', 189479),
    ('expected_return_inputs.csv', 'c3e0395abc3904f7d08b2bea07aad81fadd999fb7ee00660d06e3608a582dd35', 1263),
    ('initial_book.json', 'c2e86b9ae203828b615b9a0895b2290f82d191f1188c3c4be4bdc09bce3dbd42', 945),
    ('request_design.json', 'd4b4aa11b89a6b662f90655604757e1a01ce8b88eaafb84a9cc8727a3bb6b593', 181089),
    # Task125 independently checked these Task123 retained D26 generation_05 bytes.
    ('D26_generation_05/primary_setup.json', '767cf411c47692835aa442ca64c8135888710d2ab7ed6e9662233b0ca3fda998', 195263),
    ('D26_generation_05/primary_state.json', '7eeee966f8a94e5037d117a4e06af913c05e2f4a7b4fd0ee30b185270ac05dc3', 7539),
)
_ARTIFICIAL_PINS = (
    ('setup_authority.json', '2604c69302f1c323232e306e34cf87d75a03b28e1d3ee366ae113efccacb9518', 224),
    ('expected_return_inputs.csv', 'f55826d21d9754a28efbdb2d946700300f56d685f01a6dafa9fefe383472c7a9', 109),
    ('initial_book.json', '152b60d8919229a1bb9ad919e034f6457e0e744ebc350f2a33ea4f32eeb30758', 404),
    ('request_design.json', 'ed5a4d66cd0ab629b0f81ce89dc7c825c7af05cb31b8d343b41c770758ea9302', 4101),
    ('axis_declarations.json', 'e6affb3eaf5caff6a0fef81b0ea3e26f4e1ae1dcb3e067b6334531a46638e22d', 235),
    ('book_regime.json', '384ced99631df81e974db3638c83b6b105f4c43586a4c8f5a4900aad5ae553c0', 46),
)


class EvidenceMissing(RuntimeError):
    pass


def _stop(reason):
    raise EvidenceMissing('EVIDENCE_MISSING:' + reason)


def _keys(value, fields):
    if (type(value) is not dict or any(type(k) is not str for k in value) or
            type(fields) is not tuple or any(type(k) is not str for k in fields) or
            set(value) != set(fields)):
        _stop('EXACT_SOURCE_FIELDS')


@dataclass(frozen=True)
class _Number:
    token: str
    source_type: str


def _pairs(pairs):
    if (type(pairs) is not list or any(type(row) is not tuple or len(row) != 2 or
                                      type(row[0]) is not str for row in pairs)):
        _stop('EXACT_SOURCE_FIELDS')
    result = {}
    for key, value in pairs:
        if key in result:
            _stop('DUPLICATE_SOURCE_FIELD')
        result[key] = value
    return result


def _json(data):
    if type(data) is not bytes:
        _stop('SOURCE_BYTES_REQUIRED')
    try:
        return json.loads(data.decode('utf-8'), object_pairs_hook=_pairs,
                          parse_int=lambda s: _Number(s, 'int'),
                          parse_float=lambda s: _Number(s, 'float'),
                          parse_constant=lambda s: _Number(s, 'nonfinite'))
    except (ValueError, UnicodeError, TypeError):
        _stop('JSON_ENCODING_OR_TYPE')


def _cell(value):
    """Owned scalar wire with source lexeme; null/nonfinite never become zero."""
    if value is None:
        return {'kind': 'none'}
    if type(value) is bool:
        return {'kind': 'bool', 'value': value}
    if type(value) is str:
        return {'kind': 'str', 'value': value}
    if type(value) is not _Number:
        _stop('UNKNOWN_SOURCE_TYPE')
    if type(value.token) is not str or type(value.source_type) is not str:
        _stop('UNKNOWN_NUMERIC_TYPE')
    if value.source_type == 'int':
        if not re.fullmatch('-?(0|[1-9][0-9]*)', value.token):
            _stop('INTEGER_LEXEME')
        return {'kind': 'int', 'decimal': value.token}
    if value.source_type not in ('float', 'nonfinite'):
        _stop('UNKNOWN_NUMERIC_TYPE')
    try:
        number = float(value.token)
    except (ValueError, OverflowError):
        _stop('NUMERIC_LEXEME')
    if not math.isfinite(number):
        return {'kind': 'nonfinite_source', 'token': value.token,
                'bits_be': None, 'uncertainty': 'NATIVE_NONFINITE_BITS_UNRESOLVED'}
    return {'kind': 'float64', 'bits_be': struct.pack('>d', number).hex(),
            'source_token': value.token, 'conversion': 'DECIMAL_TO_BINARY64'}


def _value(value):
    if type(value) is list:
        return {'kind': 'list', 'items': [_value(v) for v in value]}
    if type(value) is dict:
        if any(type(k) is not str for k in value):
            _stop('EXACT_SOURCE_FIELDS')
        return {'kind': 'dict', 'fields': [[k, _value(v)] for k, v in value.items()]}
    return _cell(value)


def _check_profile(profile):
    # No comparison on a caller object or str subclass may run first.
    if type(profile) is not str or profile not in (REAL, ARTIFICIAL):
        _stop('UNAPPROVED_PROFILE_OR_EXPECTED_AUTHORITY')


def _capture_sources(sources):
    """Own exact primitive byte references before hashing or parsing.

    Dict iteration invokes no key callbacks. Validate every atom before
    building our mapping, whose keys and byte values cannot carry callbacks.
    The caller's mutable mapping is never used by a projection after this.
    """
    if type(sources) is not dict:
        _stop('EXACT_READ_SET')
    items = tuple(sources.items())
    if any(type(name) is not str for name, data in items):
        _stop('UNAPPROVED_SOURCE_PATH')
    if any(type(data) is not bytes for name, data in items):
        _stop('SOURCE_BYTES_REQUIRED')
    return dict(items)


def _validated_sources(sources, required, *, profile=REAL):
    """Return only owned bytes together with their verified SHA/size read-set."""
    _check_profile(profile)
    if (type(required) is not tuple or not required or
            any(type(name) is not str for name in required) or
            len(required) != len(set(required))):
        _stop('EXACT_READ_SET')
    owned = _capture_sources(sources)
    if set(owned) != set(required):
        _stop('EXACT_READ_SET')
    pins = _REAL_PINS if profile == REAL else _ARTIFICIAL_PINS
    expected = {name: (digest, size) for name, digest, size in pins}
    readset = []
    for name in required:
        if type(name) is not str or name not in expected:
            _stop('UNAPPROVED_SOURCE_PATH')
        data = owned[name]
        digest, size = expected[name]
        if len(data) != size or sha256(data).hexdigest() != digest:
            _stop('PINNED_SOURCE_PREIMAGE:' + name)
        readset.append({'path': name, 'sha256': digest, 'bytes': size})
    return owned, readset


def validate_sources(sources, required, *, profile=REAL):
    """Exact finite source verification; retain the original read-set-only API.

    Projections use _validated_sources so bytes consumed are the SAME owned
    bytes whose SHA/size was checked, never a reread of the caller mapping.
    """
    owned, readset = _validated_sources(sources, required, profile=profile)
    return readset


def _check_config_paths(paths):
    if (type(paths) is not tuple or not paths or
            any(type(path) is not tuple or not path or
                any(type(atom) is not str for atom in path) for path in paths)):
        _stop('UNAPPROVED_CONSUMED_CONFIG_PATH')
    # Only exact primitive strings/tuples can reach hashing or equality.
    if len(set(paths)) != len(paths) or any(path not in CONFIG_LEAVES for path in paths):
        _stop('UNAPPROVED_CONSUMED_CONFIG_PATH')


def _result(kind, readset, payload, gaps, profile):
    return json.dumps({'status': 'NOT_APPROVED_REAL_A0', 'authority_active': False,
                       'profile': profile, 'projection': kind, 'read_set': readset,
                       'payload': payload, 'schema_uncertainties': gaps},
                      ensure_ascii=True, separators=(',', ':'),
                      allow_nan=False).encode('ascii')


def _setup(data):
    value = _json(data)
    _keys(value, ('G_shapes', 'config', 'dspecs', 'fspecs', 'nav0', 'routes', 'v2', 'views'))
    return value


def _axis_declarations(data):
    schema = _json(data)
    _keys(schema, ('index', 'columns', 'value_dtype', 'native_bits'))
    if schema['value_dtype'] != '>f8' or schema['native_bits'] != 'UNRESOLVED':
        _stop('UNAPPROVED_DTYPE_OR_NATIVE_BITS')
    for key in ('index', 'columns'):
        axis = schema[key]
        _keys(axis, ('target', 'name', 'dtype', 'timezone', 'unit'))
        if (axis['target'] != 'pandas.Index' or axis['dtype'] != 'object' or
                axis['timezone'] is not None or axis['unit'] is not None or
                (axis['name'] is not None and type(axis['name']) is not str)):
            _stop('UNKNOWN_AXIS_TYPE_OR_CLOCK')
    return schema


def project_views(sources, *, profile=REAL):
    _check_profile(profile)
    required = ('setup_authority.json', 'expected_return_inputs.csv')
    if profile == ARTIFICIAL:
        required += ('axis_declarations.json',)
    owned, readset = _validated_sources(sources, required, profile=profile)
    setup = _setup(owned['setup_authority.json'])
    views = setup['views']
    if type(views) is not list or not views:
        _stop('VIEW_ROWS_REQUIRED')
    rows, routes = [], []
    for row in views:
        _keys(row, ('route', *VIEW_FIELDS))
        if type(row['route']) is not str or not row['route']:
            _stop('VIEW_ROUTE_TYPE')
        routes.append(row['route'])
        rows.append([[k, _cell(v)] for k, v in row.items()])
    if len(set(routes)) != len(routes):
        _stop('DUPLICATE_VIEW_ROUTE')
    try:
        csv_rows = list(csv.reader(io.StringIO(owned['expected_return_inputs.csv'].decode('utf-8')), strict=True))
    except (csv.Error, UnicodeError):
        _stop('CSV_ENCODING')
    if not csv_rows or tuple(csv_rows[0]) != CSV_COLUMNS:
        _stop('CSV_COLUMN_ORDER')
    retained = []
    for row in csv_rows[1:]:
        if len(row) != len(CSV_COLUMNS):
            _stop('CSV_ROW_SHAPE')
        # Empty numeric fields are retained source-missing observations.
        cells = [_cell(_Number(v, 'float')) if v else {'kind': 'source_missing', 'token': ''}
                 for v in row[2:7]]
        retained.append({'tokens': row, 'numeric_cells': cells})
    if len(retained) != len(rows):
        _stop('CSV_VIEW_ROW_COUNT')
    # Separate axes are intentional: OIL_FUT vs OIL_FUT_USD is not reindexed.
    schema = _axis_declarations(owned['axis_declarations.json']) if profile == ARTIFICIAL else None
    return _result('views', readset, {'setup_rows': rows, 'view_field_order': list(VIEW_FIELDS),
                   'setup_route_order': routes, 'csv_columns': list(CSV_COLUMNS),
                   'csv_rows': retained, 'axis_declarations': schema},
                   ['NATIVE_VIEW_DTYPE_AND_BITS_UNRESOLVED', 'SOURCE_CLOCK_NOT_HISTORICAL_PIT'] +
                   ([] if schema else ['NATIVE_AXIS_NAMES_AND_COLUMN_ORDER_UNRESOLVED']), profile)


def _book_values(book):
    expected = {'qf': [40000.] + [0.] * 5, 'qd': [0.] * 5,
                'cash': 10000000., 'capacity': None, 'pending': {}, 'series': None}
    for name, want in expected.items():
        got = book[name]
        if isinstance(want, list):
            if type(got) is not list or len(got) != len(want):
                _stop('BOOK_QUANTITY_SHAPE')
            for cell, number in zip(got, want):
                if type(cell) is not _Number or cell.source_type != 'float' or _cell(cell).get('bits_be') != struct.pack('>d', number).hex():
                    _stop('BOOK_QUANTITY_TYPE_OR_BITS')
        elif type(want) is float:
            if type(got) is not _Number or got.source_type != 'float' or _cell(got).get('bits_be') != struct.pack('>d', want).hex():
                _stop('BOOK_CASH_TYPE_OR_BITS')
        elif got != want or type(got) is not type(want):
            _stop('BOOK_PENDING_SERIES_CAPACITY')


def project_book(sources, *, profile=REAL, require_regime=False):
    _check_profile(profile)
    if type(require_regime) is not bool:
        _stop('EXACT_REQUIRE_REGIME_BOOL')
    captured = _capture_sources(sources)
    required = ('initial_book.json',)
    if profile == ARTIFICIAL and 'book_regime.json' in captured:
        required += ('book_regime.json',)
    owned, readset = _validated_sources(captured, required, profile=profile)
    value = _json(owned['initial_book.json'])
    _keys(value, ('book', 'derivative_names', 'funded_names', 'nav', 'unit_prices', 'x0'))
    book = value['book']
    _keys(book, ('capacity', 'cash', 'pending', 'qd', 'qf', 'series'))
    if value['funded_names'] != list(FUNDED) or value['derivative_names'] != list(DERIVATIVES):
        _stop('BOOK_ROUTE_ORDER')
    _book_values(book)
    if _cell(value['nav']).get('bits_be') != struct.pack('>d', 20000000.).hex():
        _stop('INITIAL_NAV_MARK')
    regime, gaps = None, ['NATIVE_BOOK_ARRAY_DTYPE_UNRESOLVED', 'SOURCE_CLOCK_NOT_HISTORICAL_PIT']
    if profile == ARTIFICIAL and 'book_regime.json' in owned:
        companion = _json(owned['book_regime.json'])
        _keys(companion, ('regime',))
        if type(companion['regime']) is not str or not companion['regime']:
            _stop('BOOK_REGIME_TYPE')
        regime = _cell(companion['regime'])
    else:
        gaps.append('Book.regime:UNRESOLVED')
        if require_regime:
            _stop('Book.regime:UNRESOLVED')
    return _result('book0', readset, {'book_fields': _value(book),
                   'funded_names': list(FUNDED), 'derivative_names': list(DERIVATIVES),
                   'initial_nav_mark': _cell(value['nav']), 'regime': regime}, gaps, profile)


def project_prospective_book(sources):
    """Explicit seven-field D26 source descriptor; no Book or A0 activation.

    New <f8 arrays are declared constructions, not recovered historical objects.
    Both persisted records must supply ORDINARY; the Engine default is not read.
    Cash is a model input, never evidence of broker liquidity or available0.
    """
    required = ('D26_generation_05/primary_setup.json',
                'D26_generation_05/primary_state.json')
    owned, readset = _validated_sources(sources, required, profile=REAL)
    setup, state = (_json(owned[name]) for name in required)
    _keys(setup, ('book0', 'config', 'dspecs', 'fspecs', 'nav0', 'routes',
                  'unit_prices', 'v2', 'views'))
    _keys(state, ('beta', 'book0', 'confirmed_state', 'cutoff', 'event_state',
                  'features', 'observable_state', 'policy', 'priors'))
    fields = ('qf', 'qd', 'cash', 'pending', 'series', 'capacity', 'regime')
    for record in (setup, state):
        _keys(record['book0'], fields)
        _book_values(record['book0'])
        if type(record['book0']['regime']) is not str or record['book0']['regime'] != 'ORDINARY':
            _stop('D26_BOOK_REGIME_SOURCE')
    if setup['book0'] != state['book0']:
        _stop('D26_BOOK_RECORDS_INCONSISTENT')
    if ([row['name'] for row in setup['fspecs']] != list(FUNDED) or
            [row['name'] for row in setup['dspecs']] != list(DERIVATIVES)):
        _stop('BOOK_ROUTE_ORDER')
    if state['cutoff'] != '2026-09-25' or setup['config']['cutoff'] != state['cutoff']:
        _stop('D26_BOOK_ORIGIN_DATE')
    if _cell(setup['nav0']).get('bits_be') != struct.pack('>d', 20000000.).hex():
        _stop('INITIAL_NAV_MARK')
    book = {name: state['book0'][name] for name in fields}
    arrays = {}
    for name in ('qf', 'qd'):
        wire = b''.join(struct.pack('<d', float(cell.token)) for cell in book[name])
        arrays[name] = {'target': 'numpy.ndarray', 'dtype': '<f8',
                        'shape': [len(book[name])], 'order': 'C',
                        'wire_hex': wire.hex(), 'wire_sha256': sha256(wire).hexdigest()}
    return _result('prospective_book0_seven_field', readset, {
        'book_fields': _value(book), 'field_order': list(fields),
        'funded_names': list(FUNDED), 'derivative_names': list(DERIVATIVES),
        'initial_nav_mark': _cell(setup['nav0']), 'origin_date': state['cutoff'],
        'native_construction': {'target': 'smlt_hedge.adaptive.engine.Book',
            'arrays': arrays, 'cash': 'builtins.float', 'pending': 'builtins.dict',
            'series': 'builtins.NoneType', 'capacity': 'builtins.NoneType',
            'regime': 'builtins.str'},
        'prospective_none_branches': {'capital': _cell(None), 'available0': _cell(None)},
        'cash_scope': 'MODEL_INPUT_NOT_LIVE_BROKER_LIQUIDITY',
    }, ['ORIGINAL_NATIVE_BOOK_IDENTITY_UNRESOLVED', 'SOURCE_CLOCK_NOT_HISTORICAL_PIT',
        'NO_LIVE_BROKER_LIQUIDITY_OBSERVATION'], REAL)


def project_config(sources, *, profile=REAL, paths=CONFIG_LEAVES):
    _check_profile(profile)
    _check_config_paths(paths)
    owned, readset = _validated_sources(sources, ('setup_authority.json',), profile=profile)
    config = _setup(owned['setup_authority.json'])['config']
    leaves = []
    for path in paths:
        value = config
        for key in path:
            if type(value) is not dict or key not in value:
                _stop('CONFIG_CONSUMED_LEAF_MISSING:' + '.'.join(path))
            value = value[key]
        leaves.append({'selector': ['config', *path], 'wire': _value(value)})
    return _result('readonly_build_config_consumed_leaves', readset, leaves,
                   ['TRANSITIVE_CONFIG_AND_CTX_CFG_CLOSURE_UNRESOLVED',
                    'NO_CONFIG_MERGE_OR_DEFAULT_AUTHORITY', 'SOURCE_CLOCK_NOT_HISTORICAL_PIT'], profile)


def project_request_design(sources, *, profile=REAL):
    owned, readset = _validated_sources(sources, ('request_design.json',), profile=profile)
    contract = _json(owned['request_design.json'])
    request = contract.get('BuildRequest') if type(contract) is dict else None
    if type(request) is not dict or _cell(request.get('count')) != {'kind': 'int', 'decimal': '19'}:
        _stop('EXACT_19_REQUEST_DESIGN')
    bindings = request.get('ordered_bindings')
    if type(bindings) is not list or len(bindings) != 19:
        _stop('EXACT_19_REQUEST_DESIGN')
    fields = []
    unresolved = ('ctx', 'cfg', 'P', 'views', 'book0', 'state_choice', 'headroom_commitment')
    for i, (name, row) in enumerate(zip(REQUEST_FIELDS, bindings), 1):
        if (type(row) is not dict or row.get('argument') != name or
                _cell(row.get('ordinal')) != {'kind': 'int', 'decimal': str(i)}):
            _stop('REQUEST_ARGUMENT_ORDER')
        for key in ('type', 'binding_kind', 'proposed_design_value', 'historical_vs_prospective_branch'):
            if key not in row:
                _stop('REQUEST_DESIGN_FIELD_MISSING')
        for key in ('type', 'binding_kind', 'historical_vs_prospective_branch'):
            if type(row[key]) is not str or not row[key]:
                _stop('REQUEST_DESIGN_TYPE_UNKNOWN')
        value = {'kind': 'unresolved_required', 'field': name} if name in unresolved else _value(row['proposed_design_value'])
        if name in ('capital', 'panel_end', 'px', 'available0') and value != {'kind': 'none'}:
            _stop('PROSPECTIVE_NONE_BRANCH_REQUIRED')
        fields.append({'argument': name, 'declared_type': row['type'], 'wire': value,
                       'branch': 'PROSPECTIVE_DISTINCT_NOT_D26_PARITY' if name in
                       ('capital', 'panel_end', 'px', 'available0', 'generation_only', 'headroom_commitment')
                       else row['historical_vs_prospective_branch']})
    return _result('BuildRequest_design_only', readset,
                   {'fields': fields, 'policy_clock': _value(contract.get('policy_clock'))},
                   ['COMPLETE_COMPOSITE_P_CTX_PANEL_CONF_STATE_CHOICE_COMMITMENT_EXPECTED_AUTHORITY_MISSING',
                    'NO_REAL_REQUEST_ADMISSION', 'NO_D26_REQUEST_PARITY'], profile)


def require_real_request_admitted(descriptor):
    """No expected real typed root has been independently accepted in this task."""
    _stop('COMPLETE_COMPOSITE_EXPECTED_AUTHORITY_UNAVAILABLE:REAL_TYPED_ROOTS=()')
