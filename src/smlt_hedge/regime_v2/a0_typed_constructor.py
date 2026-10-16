"""Dormant typed-preimage preparation; never constructs native A0 objects.

The accepted research schemas have missing real payloads. Only independently
fixed artificial source bytes are admitted here for correctness validation.
This module imports no project, numeric dependency, model or economic producer.
A serialized draft is not input authority for the Task101 pre-reader gate.
"""
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path, PosixPath, WindowsPath
import re

GROUPS = (
    'P.cal', 'P.R', 'P.B', 'P.usd', 'P.fut', 'history_panel/panel_now',
    'ctx path/cfg', 'ctx combined terms flags', 'ctx adaptive fut/roll',
    'ctx fees/term rates', 'ctx margin/MR', 'ctx BTC funded tape',
    'ctx fee/LK/currency/session capacity', 'views', 'state_choice/conf',
    'book0', 'BuildRequest', 'price/recal caches',
)
REQUEST_FIELDS = (
    'ctx', 'cfg', 'P', 't', 'universe', 'views', 'book0', 'capital', 'n_gen',
    'n_adm', 'models', 'state_choice', 'seed_offset', 'panel_end', 'px',
    'available0', 'cost_mult', 'generation_only', 'headroom_commitment',
)
REAL_TYPED_ROOTS = ()
# Authored artificial payload/schema/read-set, not a hash of passed live inputs.
SYNTHETIC_EXPECTED_ROOT = '8eda29ec9d4d173c11943cfbd93b8baf7e908e9bf9a19999f6619688f2fc92ff'
SYNTHETIC_PROFILE = 'A0_CONSTRUCTOR_ARTIFICIAL_V1'


class EvidenceMissing(RuntimeError):
    pass


def _stop(reason):
    raise EvidenceMissing('B2_INPUT_AUTHORITY_UNAVAILABLE_STOP:EVIDENCE_MISSING:' + reason)


def _keys(value, keys):
    if (type(value) is not dict or any(type(key) is not str for key in value) or
            set(value) != set(keys)):
        _stop('EXACT_SCHEMA_FIELDS')


def _text(value):
    if type(value) is not str or not value:
        _stop('EXPLICIT_TEXT_REQUIRED')


def _hex(value, size=None):
    if (type(value) is not str or len(value) % 2 or
            not re.fullmatch('[0-9a-f]*', value) or
            (size is not None and len(value) != size * 2)):
        _stop('EXACT_RAW_BITS_REQUIRED')


def _pairs(value):
    if type(value) is not list:
        _stop('ORDERED_PAIRS_REQUIRED')
    names = []
    for row in value:
        if type(row) is not list or len(row) != 2:
            _stop('ORDERED_PAIR_SCHEMA')
        _text(row[0])
        names.append(row[0])
        _node(row[1])
    if len(set(names)) != len(names):
        _stop('DUPLICATE_FIELD')
    return names


def _node(node):
    """Validate explicitly typed wire declarations, without numeric conversion.

    IEEE payloads are kept as bytes, including NaN payload/sign and signed zero.
    Ordered axes and container/record identity are declarations, never inferred.
    Native objects and abbreviated repr strings are unsupported.
    """
    if type(node) is not dict or type(node.get('kind')) is not str:
        _stop('TYPED_NODE_REQUIRED')
    kind = node['kind']
    if kind == 'none':
        _keys(node, ('kind',))
    elif kind == 'bool':
        _keys(node, ('kind', 'value'))
        if type(node['value']) is not bool:
            _stop('BOOL_TYPE')
    elif kind == 'int':
        _keys(node, ('kind', 'decimal'))
        if type(node['decimal']) is not str or not re.fullmatch('0|-?[1-9][0-9]*', node['decimal']):
            _stop('INTEGER_TYPE')
    elif kind == 'float64':
        _keys(node, ('kind', 'bits_be'))
        _hex(node['bits_be'], 8)
    elif kind == 'str':
        _keys(node, ('kind', 'value'))
        if type(node['value']) is not str:
            _stop('STRING_TYPE')
    elif kind == 'path':
        _keys(node, ('kind', 'relative', 'root_id'))
        _relative(node['relative'])
        _text(node['root_id'])
    elif kind == 'timestamp':
        _keys(node, ('kind', 'target', 'epoch_decimal', 'unit', 'timezone'))
        _text(node['target'])
        if type(node['epoch_decimal']) is not str or not re.fullmatch('0|-?[1-9][0-9]*', node['epoch_decimal']):
            _stop('EPOCH_TYPE')
        if node['unit'] not in ('ns', 'us', 'ms', 's'):
            _stop('TIMESTAMP_UNIT')
        if node['timezone'] is not None:
            _text(node['timezone'])
    elif kind == 'array':
        _keys(node, ('kind', 'target', 'dtype', 'shape', 'order', 'raw_hex'))
        _text(node['target'])
        sizes = {'<f8': 8, '>f8': 8, '<i8': 8, '>i8': 8, '|b1': 1}
        if type(node['dtype']) is not str or node['dtype'] not in sizes or node['order'] != 'C':
            _stop('UNSUPPORTED_EXACT_ARRAY_SCHEMA')
        shape = node['shape']
        if type(shape) is not list or not shape or any(type(x) is not int or x < 0 for x in shape):
            _stop('ARRAY_SHAPE')
        count = 1
        for dimension in shape:
            count *= dimension
        _hex(node['raw_hex'], count * sizes[node['dtype']])
    elif kind in ('tuple', 'list'):
        _keys(node, ('kind', 'items'))
        if type(node['items']) is not list:
            _stop('CONTAINER_TYPE')
        for item in node['items']:
            _node(item)
    elif kind in ('record', 'dict'):
        _keys(node, ('kind', 'target', 'fields'))
        _text(node['target'])
        _pairs(node['fields'])
    elif kind == 'axis':
        _keys(node, ('kind', 'target', 'name', 'dtype', 'timezone', 'unit', 'labels'))
        _text(node['target'])
        _node(node['name'])
        _text(node['dtype'])
        if node['timezone'] is not None:
            _text(node['timezone'])
        if node['unit'] is not None:
            _text(node['unit'])
        if type(node['labels']) is not list:
            _stop('AXIS_ORDER_REQUIRED')
        for label in node['labels']:
            _node(label)
    elif kind == 'series':
        _keys(node, ('kind', 'target', 'name', 'index', 'values', 'attrs'))
        _text(node['target'])
        _node(node['name'])
        _node(node['index'])
        _node(node['values'])
        _node(node['attrs'])
        if (node['index']['kind'] != 'axis' or node['values']['kind'] != 'array' or
                node['values']['shape'] != [len(node['index']['labels'])]):
            _stop('SERIES_AXIS_SHAPE')
    elif kind == 'frame':
        _keys(node, ('kind', 'target', 'index', 'columns', 'data', 'attrs'))
        _text(node['target'])
        _node(node['index'])
        _node(node['columns'])
        _node(node['attrs'])
        if node['index']['kind'] != 'axis' or node['columns']['kind'] != 'axis':
            _stop('FRAME_AXES')
        if type(node['data']) is not list or len(node['data']) != len(node['columns']['labels']):
            _stop('FRAME_COLUMNS')
        for column in node['data']:
            _node(column)
            if column['kind'] != 'array' or column['shape'] != [len(node['index']['labels'])]:
                _stop('FRAME_COLUMN_SHAPE')
    elif kind == 'group_ref':
        _keys(node, ('kind', 'group'))
        if type(node['group']) is not str or node['group'] not in GROUPS or node['group'] == 'BuildRequest':
            _stop('GROUP_REFERENCE')
    else:
        _stop('UNSUPPORTED_TYPED_NODE:' + kind)


def serialize_typed(node):
    """Deterministic declaration serialization, not an authority creation API."""
    _node(node)
    return json.dumps(node, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('ascii')


def serialize_request(fields):
    """Serialize only an explicitly complete, ordered 19-slot declaration."""
    if _pairs(fields) != list(REQUEST_FIELDS):
        _stop('EXACT_19_ARGUMENT_REQUEST')
    return serialize_typed({'kind': 'record', 'target': 'BuildRequest', 'fields': fields})


def project_series_declaration(raw, recipe):
    """One explicit raw-cell -> Series-wire transform; no native object exists.

    Raw epoch strings and IEEE bytes are copied losslessly into mandatory axis
    and value declarations. No date inference, filling, rounding or FX/model
    arithmetic is performed. This helper creates no expected authority.
    """
    _keys(raw, ('epochs_ns', 'value_bits_be'))
    _keys(recipe, ('schema', 'target', 'series_name', 'index_target',
                   'index_name', 'index_dtype', 'index_timezone', 'index_unit',
                   'value_target', 'value_dtype', 'value_order', 'units', 'attrs'))
    for key in ('schema', 'target', 'index_target', 'index_dtype', 'index_unit',
                'value_target', 'value_dtype', 'value_order'):
        _text(recipe[key])
    if (recipe['schema'] != 'EXPLICIT_SERIES_RAW_V1' or
            recipe['target'] != 'pandas.Series' or
            recipe['index_target'] != 'pandas.DatetimeIndex' or
            recipe['index_unit'] != 'ns' or recipe['value_target'] != 'numpy.ndarray' or
            recipe['value_dtype'] != '>f8' or recipe['value_order'] != 'C'):
        _stop('UNSUPPORTED_PROJECTION_RECIPE')
    _text(recipe['units'])
    epochs, cells = raw['epochs_ns'], raw['value_bits_be']
    if type(epochs) is not list or type(cells) is not list or len(epochs) != len(cells):
        _stop('RAW_CELL_AXIS_COVERAGE')
    for cell in cells:
        _hex(cell, 8)
    labels = [{'kind': 'timestamp', 'target': 'pandas.Timestamp',
               'epoch_decimal': value, 'unit': recipe['index_unit'],
               'timezone': recipe['index_timezone']} for value in epochs]
    output = {'kind': 'series', 'target': recipe['target'],
              'name': recipe['series_name'],
              'index': {'kind': 'axis', 'target': recipe['index_target'],
                        'name': recipe['index_name'], 'dtype': recipe['index_dtype'],
                        'timezone': recipe['index_timezone'], 'unit': recipe['index_unit'],
                        'labels': labels},
              'values': {'kind': 'array', 'target': recipe['value_target'],
                         'dtype': recipe['value_dtype'], 'shape': [len(cells)],
                         'order': recipe['value_order'], 'raw_hex': ''.join(cells)},
              'attrs': recipe['attrs']}
    # Serialization returns owned immutable bytes and verifies every leaf type.
    return serialize_typed(output)


def _json_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _stop('DUPLICATE_JSON_KEY')
        result[key] = value
    return result


def _load(data):
    try:
        return json.loads(data, object_pairs_hook=_json_pairs,
                          parse_constant=lambda value: _stop('UNTYPED_JSON_NUMBER'))
    except (UnicodeError, ValueError, TypeError) as exc:
        _stop('INVALID_JSON:' + type(exc).__name__)


def _relative(value):
    _text(value)
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or path.as_posix() != value:
        _stop('READSET_PATH')
    return path


def _read(root, relative):
    path = root / _relative(relative)
    for parent in (path, *path.parents):
        if parent == root.parent:
            break
        if parent.is_symlink():
            _stop('SYMLINK_SOURCE')
    if not path.is_file():
        _stop('MISSING_SOURCE:' + relative)
    return path.read_bytes()


def _clock(value):
    _text(value)
    try:
        result = datetime.fromisoformat(value)
    except ValueError:
        _stop('EXPLICIT_CLOCK')
    if result.utcoffset() is None:
        _stop('CLOCK_TIMEZONE_REQUIRED')
    return result


def _approved(root):
    data = _read(root, 'approved_fixture.json')
    if sha256(data).hexdigest() != SYNTHETIC_EXPECTED_ROOT:
        _stop('NO_INDEPENDENT_APPROVED_EXPECTED_ROOT')
    manifest = _load(data)
    _keys(manifest, ('schema', 'authority_kind', 'read_set', 'groups', 'request'))
    if manifest['schema'] != 1 or manifest['authority_kind'] != SYNTHETIC_PROFILE:
        _stop('SYNTHETIC_REAL_SEPARATION')
    readset = manifest['read_set']
    if type(readset) is not list or not readset:
        _stop('READSET_MISSING')
    names = []
    payloads = {}
    for source in readset:
        _keys(source, ('path', 'sha256', 'role', 'known_at', 'effective_from',
                       'effective_until', 'clock_basis'))
        names.append(source['path'])
        _text(source['role'])
        _hex(source['sha256'], 32)
        known = _clock(source['known_at'])
        first = _clock(source['effective_from'])
        last = _clock(source['effective_until'])
        if first > last or not first <= known <= last or source['clock_basis'] != 'DECLARED_ARTIFICIAL_NO_PIT_CLAIM':
            _stop('SOURCE_CLOCK_INTERVAL')
        raw = _read(root, source['path'])
        if sha256(raw).hexdigest() != source['sha256']:
            _stop('INDEPENDENT_SOURCE_PIN_MISMATCH:' + source['path'])
        payloads[source['path']] = raw
    if len(set(names)) != len(names):
        _stop('DUPLICATE_READSET_MEMBER')
    # The artificial fixture is intentionally a finite, flat, closed read-set.
    if set(p.name for p in root.iterdir()) != {'approved_fixture.json', *names}:
        _stop('EXTRA_OR_MISSING_READSET_MEMBER')
    return manifest, payloads


@dataclass(frozen=True)
class DraftPreimages:
    """Owned immutable wire bytes only; no native P, ctx, Book or gate token."""
    groups: tuple
    request: bytes | None
    expected_identity: str
    status: str = 'SYNTHETIC_DRAFT_ONLY_NOT_A0_AUTHORITY'


def prepare_preimages(source_root, groups=GROUPS, *, purpose='synthetic_correctness'):
    """Verify independent expected evidence then prepare deterministic wire bytes.

    Real preparation is blocked before any source read. A partial artificial
    draft cannot activate an input gate or supply a missing real typed payload.
    No live argument object or caller-supplied digest is accepted by this API.
    """
    if type(purpose) is not str or purpose != 'synthetic_correctness':
        _stop('REAL_TYPED_PAYLOAD_AND_SCHEMA_NOT_APPROVED')
    if (type(groups) is not tuple or not groups or
            any(type(g) is not str or g not in GROUPS for g in groups) or
            len(set(groups)) != len(groups)):
        _stop('EXACT_CONSUMED_GROUP_SELECTION')
    if type(source_root) not in (str, PosixPath, WindowsPath):
        _stop('EXPLICIT_SOURCE_PATH_TYPE')
    root = Path(source_root)
    if root.is_symlink() or not root.is_dir():
        _stop('SOURCE_ROOT')
    root = root.resolve()
    manifest, payloads = _approved(root)
    declarations = _load(payloads['typed_fixture.json'])
    _keys(declarations, ('groups', 'request'))
    expected = manifest['groups']
    received = declarations['groups']
    if (type(expected) is not list or type(received) is not list or
            [r.get('name') for r in expected] != list(GROUPS) or
            [r.get('name') for r in received] != list(GROUPS)):
        _stop('COMPLETE_ORDERED_GROUP_SCHEMA')
    output = []
    for fixed, supplied in zip(expected, received):
        _keys(fixed, ('name', 'target_type', 'units', 'source', 'expected'))
        _keys(supplied, ('name', 'value'))
        _text(fixed['target_type'])
        _text(fixed['units'])
        if fixed['source'] not in payloads:
            _stop('UNPINNED_GROUP_SOURCE')
        expected_bytes = serialize_typed(fixed['expected'])
        if type(supplied['value']) is dict and supplied['value'].get('kind') == 'source_projection':
            projection = supplied['value']
            _keys(projection, ('kind', 'source', 'recipe'))
            if (fixed['name'] != 'P.usd' or projection['source'] != 'synthetic_raw_series.json' or
                    projection['recipe'] != 'synthetic_series_recipe.json' or
                    fixed['source'] != projection['source']):
                _stop('UNSUPPORTED_SOURCE_TRANSFORMATION')
            raw = _load(payloads[projection['source']])
            recipe = _load(payloads[projection['recipe']])
            if recipe.get('units') != fixed['units']:
                _stop('SOURCE_TO_OBJECT_UNIT_BINDING')
            actual_bytes = project_series_declaration(raw, recipe)
        else:
            actual_bytes = serialize_typed(supplied['value'])
        if actual_bytes != expected_bytes:
            _stop('TYPED_PREIMAGE_SUBSTITUTION:' + fixed['name'])
        if fixed['name'] in groups:
            # Metadata is part of the independently approved immutable output.
            metadata = {k: fixed[k] for k in ('target_type', 'units', 'source')}
            output.append((fixed['name'], actual_bytes,
                           json.dumps(metadata, sort_keys=True, separators=(',', ':')).encode()))
    expected_request = manifest['request']
    received_request = declarations['request']
    if _pairs(expected_request) != list(REQUEST_FIELDS) or _pairs(received_request) != list(REQUEST_FIELDS):
        _stop('EXACT_19_ARGUMENT_REQUEST')
    request_bytes = serialize_request(received_request)
    if request_bytes != serialize_request(expected_request):
        _stop('REQUEST_TYPED_SUBSTITUTION')
    if dict(output_item[:2] for output_item in output).get('BuildRequest') not in (None, request_bytes):
        _stop('REQUEST_GROUP_BIJECTION')
    return DraftPreimages(tuple(output), request_bytes if groups == GROUPS else None,
                          SYNTHETIC_EXPECTED_ROOT)


def verify_typed_equality(draft, source_root):
    """Re-read fixed authority and compare immutable draft against expected bytes."""
    if type(draft) is not DraftPreimages or draft.status != 'SYNTHETIC_DRAFT_ONLY_NOT_A0_AUTHORITY':
        _stop('DRAFT_TYPE')
    if (type(draft.groups) is not tuple or not draft.groups or
            any(type(row) is not tuple or len(row) != 3 or
                type(row[0]) is not str or type(row[1]) is not bytes or
                type(row[2]) is not bytes for row in draft.groups) or
            (draft.request is not None and type(draft.request) is not bytes)):
        _stop('DRAFT_EXPLICIT_WIRE_SCHEMA')
    if type(draft.expected_identity) is not str or draft.expected_identity != SYNTHETIC_EXPECTED_ROOT:
        _stop('DRAFT_EXPECTED_IDENTITY')
    expected = prepare_preimages(source_root, tuple(row[0] for row in draft.groups))
    if draft != expected:
        _stop('DRAFT_TYPED_SUBSTITUTION')
    return True
