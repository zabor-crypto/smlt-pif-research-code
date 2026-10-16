"""Dormant, field-specific A0 equality gate; no project/producer imports.

Task93 is documentary authority only. There are NO accepted real typed roots.
The one built-in synthetic root exercises this implementation, never economics.
NumPy/pandas values may be inspected/copied only through their already-loaded
exact value classes. This module imports neither dependency nor any producer.
"""
from dataclasses import dataclass
from datetime import date, datetime, timezone, timedelta
from hashlib import sha256
import json
from pathlib import Path, PosixPath, WindowsPath
import struct
import sys
from types import SimpleNamespace
import zipfile
import functools
from zoneinfo import ZoneInfo
import weakref

REQUEST_FIELDS = ('ctx', 'cfg', 'P', 't', 'universe', 'views', 'book0', 'capital',
                  'n_gen', 'n_adm', 'models', 'state_choice', 'seed_offset',
                  'panel_end', 'px', 'available0', 'cost_mult', 'generation_only',
                  'headroom_commitment')
GROUPS = ('ctx', 'P', 'panel_now', 'panel', 'full_conf', 'state_choice',
          'views', 'book0', 'request')
MODELS = ('GLOBAL_FORWARD', 'GLOBAL_FORWARD_FHS', 'RECENT_FORWARD',
          'STATE_CONDITIONAL_FORWARD')
# Independently authored expected JSON, not a digest obtained from live inputs.
SYNTHETIC_ROOT = '160d6b942acf39348a7532dc8e798a98cb3d02ed77c65e87d551034464446d8b'
REAL_TYPED_ROOTS = ()
NATIVE_PROFILE = 'TASK135_FIXED_NATIVE_SOURCE_V1'


class InputAuthorityStop(RuntimeError):
    pass


def stop(reason):
    raise InputAuthorityStop('B2_INPUT_AUTHORITY_UNAVAILABLE_STOP:' + reason)


def _json(value):
    return json.dumps(value, ensure_ascii=True, separators=(',', ':'),
                      sort_keys=True, allow_nan=False).encode()


def _loaded_type(module, name):
    loaded = sys.modules.get(module)
    return vars(loaded).get(name) if loaded is not None else None


def _is(value, module, name):
    cls = _loaded_type(module, name)
    return cls is not None and type(value) is cls


def _record_fields(value):
    if type(value) is SimpleNamespace:
        return vars(value)
    if _is(value, 'smlt_hedge.adaptive.engine', 'Book'):
        fields = vars(value)
        if any(type(k) is not str for k in fields):
            stop('BOOK_FIELDS')
        if set(fields) != {'qf', 'qd', 'cash', 'pending', 'series', 'capacity', 'regime'}:
            stop('BOOK_FIELDS')
        return fields
    if _is(value, 'smlt_hedge.regime_v2.headroom_a', 'Commitment'):
        fields = vars(value)
        if any(type(k) is not str for k in fields):
            stop('COMMITMENT_FIELDS')
        if set(fields) != {'payload', 'identity'}:
            stop('COMMITMENT_FIELDS')
        return fields
    return None


def _instance(value, allowed):
    # Check the instance dictionary without invoking attribute/property hooks.
    fields = vars(value)
    if any(type(k) is not str for k in fields) or set(fields) - set(allowed):
        stop('INSTANCE_OVERRIDE:' + type(value).__name__)


def _object_cells(array, timestamps=False):
    primitives = (str, int, bool, float, bytes, type(None))
    for x in type(array).ravel(array, order='C').tolist():
        if type(x) in primitives:
            continue
        if not (timestamps and _is(x, 'pandas', 'Timestamp')):
            stop('NON_PRIMITIVE_FRAME_CELL:UNSUPPORTED_TYPE')
        _preflight(x, cell=True, timestamps=True)


def _string_policy(value):
    # pandas3.0.6 normalizes every constructor NaN to the exact np.nan
    # singleton. Its array methods use identity for the missing-value branch.
    # A caller-written noncanonical NaN is not a third supported policy.
    na = vars(value).get('_na_value')
    if na is vars(sys.modules['pandas'])['NA']:
        return ['pandas.NA']
    if type(na) is float and na is vars(sys.modules['numpy'])['nan']:
        raw = struct.pack('>d', na)
        word = int.from_bytes(raw, 'big')
        if word & 0x7ff0000000000000 == 0x7ff0000000000000 and word & 0xfffffffffffff:
            return ['numpy.nan', 'float64_be', raw.hex()]
    stop('UNSUPPORTED_STRING_MISSING_POLICY')


def _string_dtype_metadata(value):
    if _is(value, 'pandas', 'StringDtype'):
        return ['StringDtype', vars(value)['_storage'], _string_policy(value)]
    return None


def _dtype(value):
    # The pinned NumPy namespace supplies the trusted identities. Do not read
    # an unknown class's module/name or use its metaclass equality/hash hooks.
    if any(type(value) is _loaded_type('numpy.dtypes', name) for name in (
            'BoolDType', 'Int8DType', 'UInt8DType', 'Int16DType', 'UInt16DType',
            'Int32DType', 'UInt32DType', 'Int64DType', 'UInt64DType',
            'IntDType', 'UIntDType', 'LongDType', 'ULongDType',
            'LongLongDType', 'ULongLongDType', 'Float16DType', 'Float32DType',
            'Float64DType', 'LongDoubleDType', 'Complex64DType',
            'Complex128DType', 'CLongDoubleDType', 'ObjectDType', 'BytesDType',
            'StrDType', 'VoidDType', 'DateTime64DType', 'TimeDelta64DType',
            'StringDType')):
        if value.metadata is not None or value.fields is not None or value.subdtype is not None:
            stop('UNSUPPORTED_DTYPE_METADATA')
        return
    if _is(value, 'pandas', 'StringDtype'):
        _instance(value, ('_storage', '_na_value', '_cache'))
        fields = vars(value)
        cache = fields.get('_cache', {})
        # Only the inherited exact native index_class memo is used by this
        # dtype. Other cached fields cannot hide policy/constructor overrides.
        if type(cache) is not dict or any(type(k) is not str or
                k != 'index_class' or v is not _loaded_type('pandas', 'Index')
                for k, v in cache.items()):
            stop('UNSUPPORTED_DTYPE_CACHE')
        if type(fields.get('_storage')) is not str or fields['_storage'] != 'python':
            stop('UNSUPPORTED_STRING_DTYPE')
        _string_policy(value)
        return
    if _is(value, 'pandas', 'DatetimeTZDtype'):
        _instance(value, ('_unit', '_tz', '_cache'))
        cache = vars(value).get('_cache', {})
        if type(cache) is not dict or any(type(k) is not str or type(v) is not int
                                         for k, v in cache.items()):
            stop('UNSUPPORTED_TIMEZONE_DTYPE_CACHE')
        if type(vars(value).get('_unit')) is not str or type(vars(value).get('_tz')) not in (ZoneInfo, timezone):
            stop('UNSUPPORTED_TIMEZONE_DTYPE')
        return
    stop('UNSUPPORTED_DTYPE')


def _axis_cache(value, *, native=True):
    cache = object.__getattribute__(value, '_cache')
    if type(cache) is not dict or any(type(k) is not str for k in cache):
        stop('UNSUPPORTED_AXIS_CACHE')
    for k, x in cache.items():
        if k == 'dtype':
            _dtype(x)
        elif k == '_engine':
            cls = type(x)
            if cls.__module__ != 'pandas._libs.index' or _loaded_type(cls.__module__, cls.__name__) is not cls:
                stop('UNSUPPORTED_AXIS_ENGINE')
        elif k == 'array':
            storage = _storage(x, native=native)
            if storage.dtype.hasobject:
                _object_cells(storage)
        elif k == '_references':
            if not _is(x, 'pandas._libs.internals', 'BlockValuesRefs'):
                stop('UNSUPPORTED_AXIS_REFERENCES')
        elif k == '_constructor':
            if not any(x is _loaded_type('pandas', n) for n in ('Index', 'DatetimeIndex', 'RangeIndex')):
                stop('UNSUPPORTED_AXIS_CONSTRUCTOR')
        elif not (type(x) in (bool, int, float, str, type(None)) or _is(x, 'numpy', 'ndarray')):
            stop('UNSUPPORTED_AXIS_CACHE_VALUE:' + k)


def _extension_frequency(value):
    # The unchanged T06 oracle uses pd.bdate_range: its exact built-in B-day
    # metadata is not a caller callback or a new expected-value authority.
    if value is None:
        return
    if not _is(value, 'pandas._libs.tslibs.offsets', 'BusinessDay'):
        stop('UNSUPPORTED_EXTENSION_FREQUENCY')
    try:
        fields = object.__getattribute__(value, '__dict__')
    except AttributeError:
        fields = {}
    if type(fields) is not dict or fields:
        stop('UNSUPPORTED_EXTENSION_FREQUENCY_OVERRIDE')
    if (type(value.n) is not int or value.n != 1 or
            type(value.normalize) is not bool or value.normalize is not False or
            type(value.offset) is not timedelta or value.offset != timedelta(0)):
        stop('UNSUPPORTED_EXTENSION_FREQUENCY')


def _storage(value, *, native=True):
    if _is(value, 'numpy', 'ndarray'):
        _dtype(value.dtype)
        return value
    for name, allowed in (('StringArray', ()),
                          ('DatetimeArray', ('_freq', '_cache'))):
        if _is(value, 'pandas.core.arrays', name):
            _instance(value, allowed)
            # These exact pandas3 extension classes own native ndarray storage.
            if not _is(value._ndarray, 'numpy', 'ndarray'):
                stop('UNSUPPORTED_STORAGE_BUFFER')
            _dtype(value.dtype)
            cache = vars(value).get('_cache', {})
            if type(cache) is not dict or any(type(k) is not str or
                    type(v) not in (str, int, bool, type(None)) for k, v in cache.items()):
                stop('UNSUPPORTED_EXTENSION_CACHE')
            if name == 'DatetimeArray':
                frequency = vars(value).get('_freq')
                if native and frequency is not None:
                    stop('UNSUPPORTED_EXTENSION_FREQUENCY')
                _extension_frequency(frequency)
            return value._ndarray
    stop('UNSUPPORTED_STORAGE_TYPE')


def _indexer_buffer(value):
    # Finite pandas3 manager/placement surface; reject foreign storage before
    # any dependency accessor or deep copy can consult it.
    if not _is(value, 'numpy', 'ndarray'):
        stop('UNSUPPORTED_PLACEMENT_BUFFER')
    _dtype(value.dtype)
    if (value.ndim != 1 or value.dtype.kind != 'i' or
            value.dtype.itemsize != struct.calcsize('P') or not value.dtype.isnative):
        stop('UNSUPPORTED_PLACEMENT_BUFFER')


def _placement(value):
    if not _is(value, 'pandas._libs.internals', 'BlockPlacement'):
        stop('UNSUPPORTED_BLOCK_PLACEMENT')
    # Exact final C-extension class has no instance overrides. as_array may
    # materialize its native cache; only its approved intp storage is admitted.
    _indexer_buffer(type(value).as_array.__get__(value))


def _preflight(value, *, native=False, cell=False, timestamps=False):
    """Closed native graph inspection before any value accessor or copy.

    Caller pandas methods, managers, blocks and extension storage cannot hide
    behind an exact public class. No caller equality/deepcopy/duck constructors.
    """
    cls = type(value)
    if value is None or cls in (bool, int, float, str, bytes, date, datetime,
                               PosixPath, WindowsPath):
        if cls is datetime and value.tzinfo is not None and type(value.tzinfo) not in (timezone, ZoneInfo):
            stop('UNSUPPORTED_DATETIME_TIMEZONE')
        return
    if _is(value, 'pandas', 'Timestamp'):
        _instance(value, ())
        if value.tz is not None and type(value.tz) not in (timezone, ZoneInfo):
            stop('UNSUPPORTED_TIMESTAMP_TIMEZONE')
        if cell and not timestamps:
            stop('NON_PRIMITIVE_FRAME_CELL')
        return
    if cell:
        stop('NON_PRIMITIVE_FRAME_CELL:UNSUPPORTED_TYPE')
    if cls in (tuple, list):
        for x in value:
            _preflight(x, native=native, timestamps=timestamps)
        return
    if cls is dict:
        for k, x in value.items():
            if type(k) not in (str, int):
                stop('UNSUPPORTED_MAPPING_KEY')
            _preflight(x, native=native, timestamps=timestamps)
        return
    if _is(value, 'numpy', 'ndarray'):
        _dtype(value.dtype)
        if value.dtype.hasobject:
            _object_cells(value, timestamps)
        return
    if any(_is(value, 'pandas', n) for n in
           ('Index', 'RangeIndex', 'DatetimeIndex', 'Series', 'DataFrame')):
        if cls.__name__ in ('Series', 'DataFrame'):
            _instance(value, ('_mgr', '_attrs', '_flags', '_name'))
            mgr = object.__getattribute__(value, '_mgr')
            expected = 'SingleBlockManager' if cls.__name__ == 'Series' else 'BlockManager'
            if not _is(mgr, 'pandas.core.internals.managers', expected):
                stop('UNSUPPORTED_MANAGER')
            _instance(mgr, ('axes', 'blocks', '_cache'))
            if type(mgr.blocks) is not tuple or type(mgr.axes) is not list:
                stop('UNSUPPORTED_MANAGER_CONTAINERS')
            for name in ('_blknos', '_blklocs'):
                buffer = object.__getattribute__(mgr, name)
                if buffer is not None:
                    _indexer_buffer(buffer)
            cache = vars(mgr).get('_cache', {})
            if (type(cache) is not dict or any(type(k) is not str for k in cache) or
                    set(cache) - {'_block'} or
                    '_block' in cache and (len(mgr.blocks) != 1 or cache['_block'] is not mgr.blocks[0])):
                stop('UNSUPPORTED_MANAGER_CACHE')
            for block in mgr.blocks:
                if not any(_is(block, 'pandas.core.internals.blocks', n) for n in
                           ('NumpyBlock', 'ExtensionBlock', 'DatetimeLikeBlock')):
                    stop('UNSUPPORTED_BLOCK')
                _instance(block, ('_cache',))
                _placement(object.__getattribute__(block, '_mgr_locs'))
                storage = _storage(block.values, native=native)
                if storage.dtype.hasobject:
                    _object_cells(storage, timestamps)
                # Reject non-native cached metadata before a class method can
                # consult it. dtype caches must have exact native dtype types.
                cache = vars(block).get('_cache', {})
                if type(cache) is not dict:
                    stop('UNSUPPORTED_BLOCK_CACHE')
                for k, x in cache.items():
                    if type(k) is not str:
                        stop('UNSUPPORTED_BLOCK_CACHE')
                    dtype = block.values.dtype
                    if _is(dtype, 'pandas', 'StringDtype'):
                        if k == 'dtype' and x is not dtype:
                            stop('UNSUPPORTED_STRING_BLOCK_DTYPE_CACHE')
                        if k == 'fill_value' and x is not vars(dtype)['_na_value']:
                            stop('UNSUPPORTED_STRING_FILL_CACHE')
                        if k == 'array_values' and x is not block.values:
                            stop('UNSUPPORTED_STRING_ARRAY_CACHE')
                        if k == '_consolidate_key':
                            name = 'string' if vars(dtype)['_na_value'] is vars(sys.modules['pandas'])['NA'] else 'str'
                            if (type(x) is not tuple or len(x) != 2 or
                                    type(x[0]) is not bool or type(x[1]) is not str or
                                    x != (False, name)):
                                stop('UNSUPPORTED_STRING_CONSOLIDATE_CACHE')
                        if k in ('is_extension', '_can_consolidate') and x is not (k == 'is_extension'):
                            stop('UNSUPPORTED_STRING_BLOCK_CACHE')
                    if k == 'dtype':
                        _dtype(x)
                    elif k == 'shape':
                        storage_shape = block.values.shape
                        # A 2D single-column ExtensionBlock may own 1D string
                        # storage; its block shape adds the column dimension.
                        if (len(storage_shape) == 1 and type(block.ndim) is int and block.ndim == 2
                                and _is(block, 'pandas.core.internals.blocks', 'ExtensionBlock')):
                            storage_shape = (1, storage_shape[0])
                        if (type(x) is not tuple or any(type(n) is not int for n in x) or
                                x != storage_shape):
                            stop('UNSUPPORTED_BLOCK_SHAPE_CACHE')
                    elif not (type(x) in (bool, int, float, str, type(None)) or
                              k == '_consolidate_key' and type(x) is tuple and
                              len(x) == 2 and type(x[0]) is bool and type(x[1]) is str or
                              k == 'fill_value' and _is(x, 'numpy', 'datetime64') or
                              k == 'fill_value' and x is vars(sys.modules['pandas']).get('NaT') or
                              k == 'fill_value' and _is(dtype, 'pandas', 'StringDtype') and
                              x is vars(sys.modules['pandas'])['NA'] or
                              k == 'array_values' and _is(dtype, 'pandas', 'StringDtype') and
                              x is block.values):
                        stop('UNSUPPORTED_BLOCK_CACHE')
            flags = object.__getattribute__(value, '_flags')
            if not _is(flags, 'pandas.core.flags', 'Flags'):
                stop('UNSUPPORTED_FLAGS')
            _instance(flags, ('_allows_duplicate_labels', '_obj'))
            if type(flags._allows_duplicate_labels) is not bool:
                stop('UNSUPPORTED_FLAGS_VALUE')
            if type(flags._obj) is not weakref.ReferenceType:
                stop('UNSUPPORTED_FLAGS_OWNER')
            _preflight(value.attrs, native=native)
            _preflight(value.index, native=native)
            _preflight(value.name if cls.__name__ == 'Series' else value.columns,
                       native=native)
        else:
            _instance(value, ('_data', '_range', '_name', '_cache', '_id', '_references'))
            _axis_cache(value, native=native)
            _preflight(value.name, native=native)
            if cls.__name__ == 'RangeIndex':
                if type(object.__getattribute__(value, '_range')) is not range:
                    stop('UNSUPPORTED_RANGE')
                cache = object.__getattribute__(value, '_cache')
                if type(cache) is not dict:
                    stop('UNSUPPORTED_RANGE_CACHE')
                if '_data' in cache and not _is(cache['_data'], 'numpy', 'ndarray'):
                    stop('UNSUPPORTED_RANGE_CACHE')
            else:
                storage = _storage(object.__getattribute__(value, '_data'), native=native)
                if storage.dtype.hasobject:
                    _object_cells(storage)
            # The native storage graph is checked without to_numpy. A validated
            # DatetimeArray later boxes its own Timestamp axis cells in the
            # approved class codec; no caller object-array cells are admitted.
        return
    fields = _record_fields(value)
    if fields is not None:
        for k, x in fields.items():
            if type(k) is not str:
                stop('UNSUPPORTED_RECORD_KEY')
            if k != '_headroom_a0':
                _preflight(x, native=native,
                           timestamps=(k in ('_adaptive_pit', '_adaptive_fut', '_adaptive_mr')
                                       if native else timestamps))
        return
    stop('UNSUPPORTED_TYPE:' + cls.__module__ + '.' + cls.__name__)


def typed(value, root, *, profile=None):
    _preflight(value, native=profile is not None, timestamps=profile is None)
    return _typed(value, root, profile)


def _typed(value, root, profile=None):
    """Lossless value encoding for this contract's explicitly supported types.

    Encoding live values is comparison only; it NEVER mints expected authority.
    Unsupported classes, extension arrays and user callbacks fail closed.
    """
    cls = type(value)
    def typed(x, r):
        return _typed(x, r, profile)
    if value is None:
        return ['none']
    if cls is bool:
        return ['bool', value]
    if cls is int:
        return ['int', str(value)]
    if cls is float:
        return ['float64', struct.pack('>d', value).hex()]
    if cls is str:
        return ['str', value]
    if cls is bytes:
        return ['bytes', value.hex()]
    if cls in (PosixPath, WindowsPath):
        path = value.resolve()
        if not path.is_relative_to(root):
            stop('PATH_OUTSIDE_READSET')
        return ['path', path.relative_to(root).as_posix()]
    if cls in (date, datetime):
        return [cls.__name__, value.isoformat()]
    if cls in (tuple, list):
        return [cls.__name__, [typed(x, root) for x in value]]
    if cls is dict:
        if any(type(k) not in (str, int) for k in value):
            stop('UNSUPPORTED_MAPPING_KEY')
        return ['dict', [[k if type(k) is str else ['integer_key', str(k)],
                          typed(v, root)] for k, v in value.items()]]
    if _is(value, 'numpy', 'ndarray'):
        if value.dtype.hasobject:
            items = [typed(x, root) for x in value.ravel(order='C').tolist()]
            if profile is not None and profile['blocks'] is not None:
                raw = _json(items)
                h = sha256(raw).hexdigest()
                if profile['blocks'].get(h) != raw:
                    stop('NATIVE_LOGICAL_BLOCK_CHANGED')
                items = ['logical_block', h, len(raw)]
            return ['array_object', value.dtype.str, list(value.shape), items]
        raw = cls.tobytes(value, order='C')
        payload = raw.hex()
        if profile is not None and profile['blocks'] is not None:
            h = sha256(raw).hexdigest()
            if profile['blocks'].get(h) != raw:
                stop('NATIVE_BINARY_BLOCK_CHANGED')
            payload = ['block', h, len(raw)]
        return ['array', value.dtype.str, list(value.shape), payload]
    if _is(value, 'pandas', 'Timestamp'):
        return ['timestamp', value.isoformat(), str(value.tz),
                str(value.asm8.dtype), value.asm8.tobytes().hex()]
    if any(_is(value, 'pandas', n) for n in
           ('Index', 'RangeIndex', 'DatetimeIndex')):
        result = ['index', cls.__name__, typed(value.name, root), str(value.dtype),
                str(value.tz) if cls.__name__ == 'DatetimeIndex' else None,
                typed(cls.to_numpy(value, copy=True), root)]
        if profile is not None:
            result.append(['storage', getattr(value.dtype, 'storage', None)])
            metadata = _string_dtype_metadata(value.dtype)
            if metadata is not None:
                result.append(['dtype_metadata', metadata])
            if cls.__name__ == 'RangeIndex':
                result.append(['range', value.start, value.stop, value.step])
        return result
    if _is(value, 'pandas', 'Series'):
        result = ['series', typed(value.name, root), typed(value.index, root),
                  str(value.dtype), typed(cls.to_numpy(value, copy=True), root),
                  typed(value.attrs, root)]
        if profile is not None:
            result.append(['storage', getattr(value.dtype, 'storage', None),
                           value.flags.allows_duplicate_labels])
            metadata = _string_dtype_metadata(value.dtype)
            if metadata is not None:
                result.append(['dtype_metadata', metadata])
        return result
    if _is(value, 'pandas', 'DataFrame'):
        return ['frame', typed(value.index, root), typed(value.columns, root),
                [[typed(name, root), typed(cls.__getitem__(value, name), root)]
                 for name in value.columns], typed(value.attrs, root),
                bool(value.flags.allows_duplicate_labels)]
    fields = _record_fields(value)
    if fields is not None:
        return ['record', cls.__module__ + '.' + cls.__name__,
                typed({k: v for k, v in fields.items()
                       if k != '_headroom_a0'}, root)]
    stop('UNSUPPORTED_TYPE:' + cls.__module__ + '.' + cls.__name__)


def _detach(value, *, native=False):
    _preflight(value, native=native)
    return _copy(value, native=native)


def _fresh_storage_dtype(array):
    # pandas deep storage copies share mutable extension dtype/cache objects.
    # Rebuild only the two preflight-approved exact native metadata classes.
    if _is(array, 'pandas.core.arrays', 'StringArray'):
        dtype = array.dtype
        dtype = type(dtype)(storage=dtype.storage, na_value=dtype.na_value)
        return type(array)._simple_new(array._ndarray, dtype=dtype)
    elif _is(array, 'pandas.core.arrays', 'DatetimeArray') and _is(array.dtype, 'pandas', 'DatetimeTZDtype'):
        dtype = array.dtype
        dtype = type(dtype)(unit=dtype.unit, tz=dtype.tz)
        return type(array)._simple_new(array._ndarray, dtype=dtype)
    return array


def _copy(value, *, native=False):
    """Copy supported value containers without user deepcopy/constructor hooks."""
    cls = type(value)
    def _detach(x):
        return _copy(x, native=native)
    if value is None or cls in (bool, int, float, str, bytes, date, datetime,
                              PosixPath, WindowsPath):
        return value
    if cls in (tuple, list):
        return cls(_detach(x) for x in value)
    if cls is dict:
        return {k: _detach(v) for k, v in value.items()}
    if _is(value, 'numpy', 'ndarray'):
        if value.dtype.hasobject:
            # Validate primitive object cells before making the array copy.
            for item in value.ravel(order='C').tolist():
                if type(item) not in (str, int, bool, float, bytes, type(None)):
                    stop('NON_PRIMITIVE_OBJECT_CELL')
        return cls.copy(value, order='C')
    if _is(value, 'pandas', 'Timestamp'):
        return value  # immutable native scalar
    if any(_is(value, 'pandas', n) for n in
           ('Index', 'RangeIndex', 'DatetimeIndex', 'Series', 'DataFrame')):
        # Exact value types only; use class method, not caller instance override.
        if cls.__name__ in ('Series', 'DataFrame'):
            _detach(value.attrs)
            _detach(value.index)
            _detach(value.name if cls.__name__ == 'Series' else value.columns)
            # Native preflight checked each storage block without a mixed matrix
            # conversion, which would box datetime64 NaT words into pd.NaT cells.
            # Those ephemeral boxed cells are never an admitted object profile.
            arrays = [] if native else [cls.to_numpy(value, copy=False)]
            for array in arrays:
                if array.dtype.hasobject:
                    for item in array.ravel(order='C').tolist():
                        if (type(item) not in (str, int, bool, float, bytes, type(None))
                                and not (native and _is(item, 'pandas', 'Timestamp'))):
                            stop('NON_PRIMITIVE_FRAME_CELL')
        if cls.__name__ == 'RangeIndex':
            # pandas3 deep copies share the cached ndarray. A fresh approved
            # range reconstructs independent storage after immutable comparison.
            return cls(value.start, value.stop, value.step, name=_detach(value.name))
        detached = cls.copy(value, deep=True)
        if cls.__name__ in ('Index', 'DatetimeIndex'):
            detached._cache = {}
            detached._data = _fresh_storage_dtype(detached._data)
        # pandas deep frame/series copies can share immutable-by-convention
        # axis storage. Detach each native axis too, including timezone data.
        if cls.__name__ in ('Series', 'DataFrame'):
            for block in detached._mgr.blocks:
                placement = object.__getattribute__(block, '_mgr_locs')
                # Block.copy(deep=True) shares mutable BlockPlacement storage.
                # Rebuild both its array and slice/cache representation from a
                # fresh native buffer, never from the caller's indexer object.
                array = type(placement).as_array.__get__(placement)
                block.mgr_locs = type(placement)(type(array).copy(array))
                block.values = _fresh_storage_dtype(block.values)
                block._cache = {}
            for name in ('_blknos', '_blklocs'):
                buffer = object.__getattribute__(detached._mgr, name)
                if buffer is not None:
                    object.__setattr__(detached._mgr, name, type(buffer).copy(buffer))
            detached.index = _detach(value.index)
            if cls.__name__ == 'DataFrame':
                detached.columns = _detach(value.columns)
            detached.attrs = _detach(value.attrs)
            detached.flags.allows_duplicate_labels = value.flags.allows_duplicate_labels
        return detached
    fields = _record_fields(value)
    if fields is not None:
        detached = SimpleNamespace() if cls is SimpleNamespace else object.__new__(cls)
        for name, item in fields.items():
            if name != '_headroom_a0':
                object.__setattr__(detached, name, _detach(item))
        return detached
    stop('UNSUPPORTED_COPY_TYPE')


def _read_sources(expected, root, authority_path):
    entries = expected['sources']
    if type(entries) is not list or not entries:
        stop('READSET_ABSENT')
    declared = {authority_path.name}
    for entry in entries:
        if set(entry) != {'id', 'path', 'sha256', 'role', 'known_at_class'}:
            stop('SOURCE_SCHEMA')
        name = entry['path']
        if type(name) is not str or Path(name).name != name or name in declared:
            stop('SOURCE_MEMBER')
        declared.add(name)
        p = root / name
        if p.is_symlink() or not p.is_file():
            stop('SOURCE_ABSENT_OR_ALIAS:' + name)
        if sha256(p.read_bytes()).hexdigest() != entry['sha256']:
            stop('SOURCE_BYTES:' + entry['id'])
        if entry['known_at_class'] != 'NONCAUSAL_AT_ORIGIN':
            stop('ORIGIN_CLASS')
    if {p.name for p in root.iterdir()} != declared:
        stop('UNLISTED_READSET_MEMBER')
    roles = {e['role'] for e in entries}
    if roles != {'SOURCE_OBSERVATIONS_BUYSELLFEE_ACADEMIC_PROXY',
                 'SYNTHETIC_TRANSFORM_BR_OI_OR_STOP'}:
        stop('FEE_OR_BR_ROLE')


@dataclass(frozen=True)
class VerifiedInputs:
    """Receipt for an owned request; not an active real A0 authority."""
    expected_bytes: bytes
    root: Path
    request_bytes: bytes
    authority_name: str

    def check_panels(self, panel_now, panel):
        expected = json.loads(self.expected_bytes)['groups']
        if (typed(panel_now, self.root) != expected['panel_now'] or
                typed(panel, self.root) != expected['panel']):
            stop('CONSUMED_PANEL_CHANGED')

    def check_request(self, request):
        anchor = self.root / self.authority_name
        if anchor.is_symlink() or anchor.read_bytes() != self.expected_bytes:
            stop('EXPECTED_CHANGED_BEFORE_CONSUMER')
        _read_sources(json.loads(self.expected_bytes), self.root, anchor)
        if _json(typed(request, self.root)) != self.request_bytes:
            stop('OWNED_REQUEST_CHANGED_BEFORE_CONSUMER')


def validate_entry(request):
    """Validate all 19 actual bound arguments and return only detached inputs.

    Real roots are deliberately absent. A caller-provided hash/expected object
    is not a trust anchor; the synthetic expected file must match the fixed pin.
    """
    if (type(request) is not dict or any(type(k) is not str for k in request)
            or tuple(request) != REQUEST_FIELDS):
        stop('REQUEST_19_FIELDS')
    ctx = request['ctx']
    if type(ctx) is not SimpleNamespace:
        stop('MINIMAL_CONTEXT_TYPE')
    marker = vars(ctx).get('_headroom_a0')
    if '_headroom_a0' not in vars(ctx):
        return _validate_t06_fixture(request)
    if (type(marker) is not dict or any(type(k) is not str for k in marker)
            or set(marker) != {'mode', 'authority_path'}):
        stop('INDEPENDENT_EXPECTED_AUTHORITY_ABSENT')
    if type(marker['mode']) is not str:
        stop('INDEPENDENT_EXPECTED_AUTHORITY_ABSENT')
    if type(marker['authority_path']) not in (str, PosixPath, WindowsPath):
        stop('INDEPENDENT_EXPECTED_AUTHORITY_ABSENT')
    if marker['mode'] == NATIVE_PROFILE:
        return _validate_native(request, marker)
    if marker['mode'] != 'SYNTHETIC_CORRECTNESS':
        stop('REAL_TYPED_PREIMAGES_NOT_ACCEPTED')
    path = Path(marker['authority_path'])
    if path.is_symlink() or not path.is_file():
        stop('EXPECTED_FILE_ABSENT')
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != SYNTHETIC_ROOT:
        stop('EXPECTED_NOT_INDEPENDENTLY_PINNED')
    expected = json.loads(raw)
    if set(expected) != {'mode', 'sources', 'groups', 'request'}:
        stop('EXPECTED_SCHEMA')
    if expected['mode'] != 'SYNTHETIC_CORRECTNESS':
        stop('EXPECTED_ROLE')
    if set(expected['groups']) != set(GROUPS):
        stop('P_CTX_PANEL_CONF_TYPED_PREIMAGE_MISSING')
    root = path.parent.resolve()
    _read_sources(expected, root, path)
    if (request['generation_only'] is not True or type(request['cost_mult']) is not float
            or request['cost_mult'] != 1.0 or request['models'] != MODELS
            or type(request['n_gen']) is not int or request['n_gen'] != 2000
            or type(request['n_adm']) is not int or request['n_adm'] != 2000
            or type(request['seed_offset']) is not int or request['seed_offset'] != 0
            or any(request[k] is not None for k in
                   ('capital', 'panel_end', 'px', 'available0'))):
        stop('CONDITIONAL_METHOD_OR_NONE_BRANCH')
    owned = _detach(request)
    encoded = typed(owned, root)
    if encoded != expected['request']:
        stop('ACTUAL_19_ARGUMENT_PREIMAGE_MISMATCH')
    fields = vars(owned['ctx'])
    for name in ('_a0_panel_now', '_a0_panel', '_a0_full_conf'):
        if name not in fields:
            stop('PANEL_OR_FULL_CONF_MISSING')
    choice = owned['state_choice']
    if type(choice) is not dict or set(choice) != {
            'available', 'condition', 'level', 'starts', 'n_episodes',
            'n_eligible_block_starts', 'tried'}:
        stop('FULL_STATE_CHOICE_MISSING')
    groups = {'ctx': owned['ctx'], 'P': owned['P'],
              'panel_now': fields['_a0_panel_now'], 'panel': fields['_a0_panel'],
              'full_conf': fields['_a0_full_conf'], 'state_choice': choice,
              'views': owned['views'], 'book0': owned['book0'], 'request': owned}
    if any(typed(v, root) != expected['groups'][k] for k, v in groups.items()):
        stop('GROUP_PREIMAGE_MISMATCH')
    # Fixed synthetic commitment role prevents a synthetic root blessing a real
    # economic commitment. No native read()/producer is called for this check.
    commitment = _record_fields(owned['headroom_commitment'])
    if (commitment is None or type(commitment.get('payload')) is not str
            or json.loads(commitment['payload']).get('mode') != 'SYNTHETIC_CORRECTNESS'):
        stop('COMMITMENT_NOT_SYNTHETIC')
    return VerifiedInputs(raw, root, _json(encoded), path.name), owned


def _safe_path(path):
    if type(path) is not PosixPath or any(p.is_symlink() for p in (path, *path.parents)):
        stop('NATIVE_PATH_ALIAS')
    return path


def _native_profile(path, raw=None):
    """Read only an exact registered archive, never a caller expected object.

    The existing compact schema/block transport is custody. Expected typed
    literals were compiled offline from pinned sources, not this live request.
    """
    import io
    _safe_path(path)
    data = path.read_bytes()
    if raw is not None and data != raw:
        stop('EXPECTED_CHANGED_BEFORE_CONSUMER')
    if sha256(data).hexdigest() not in REAL_TYPED_ROOTS:
        stop('NATIVE_EXPECTED_NOT_REGISTERED')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(
                Path(n).is_absolute() or '..' in Path(n).parts for n in names):
            stop('NATIVE_ARCHIVE_MEMBERS')
        expected = json.loads(archive.read('gate_expected.json'))
        if (expected['profile'] != NATIVE_PROFILE or
                len(expected['binding']['groups']) != 18 or
                len(expected['binding']['arguments']) != 19):
            stop('NATIVE_FIXED_PROFILE_SCHEMA')
        blocks = {}
        for row in expected['blocks']:
            name = 'blocks/' + row['sha256'] + '.bin'
            block = archive.read(name)
            if len(block) != row['bytes'] or sha256(block).hexdigest() != row['sha256']:
                stop('NATIVE_AUTHORITY_BLOCK_CHANGED')
            blocks[row['sha256']] = block
        if set(names) != {'gate_expected.json'} | {'blocks/' + h + '.bin' for h in blocks}:
            stop('NATIVE_ARCHIVE_MEMBERS')
    return data, {'expected': expected, 'blocks': blocks}


def _native_sources(profile, root):
    """Fixed hierarchy raw readback, distinct from flat synthetic fee roles."""
    _safe_path(root)
    if not root.is_dir() or root != root.resolve():
        stop('NATIVE_ROOT_SUBSTITUTION')
    expected = profile['expected']
    rows = expected['root_read_set']
    names = set()
    for row in rows:
        name = row['path']
        if (type(name) is not str or Path(name).is_absolute() or
                '..' in Path(name).parts or Path(name).as_posix() != name or name in names):
            stop('NATIVE_SOURCE_MEMBER')
        names.add(name)
    if len(names) != 79:
        stop('NATIVE_ROOT79_SCHEMA')
    hierarchy = {}
    for name in names:
        parts = Path(name).parts
        for i, part in enumerate(parts):
            parent = Path(*parts[:i]).as_posix()
            hierarchy.setdefault(parent, set()).add(part)
    # Only enumerate fixed expected directories; stop on the first extra child.
    # A substituted root cannot turn this check into an unbounded filesystem scan.
    for directory, children in hierarchy.items():
        parent = _safe_path(root / directory)
        if not parent.is_dir():
            stop('NATIVE_ROOT_MISSING_OR_EXTRA')
        actual = set()
        for path in parent.iterdir():
            if path.is_symlink():
                stop('NATIVE_SOURCE_ALIAS')
            if path.name not in children:
                stop('NATIVE_ROOT_MISSING_OR_EXTRA')
            relative = path.relative_to(root).as_posix()
            if not (path.is_file() if relative in names else path.is_dir()):
                stop('NATIVE_SOURCE_TYPE')
            actual.add(path.name)
        if actual != children:
            stop('NATIVE_ROOT_MISSING_OR_EXTRA')
    for row in rows:
        path = _safe_path(root / row['path'])
        data = path.read_bytes()
        if len(data) != row['bytes'] or sha256(data).hexdigest() != row['sha256']:
            stop('NATIVE_SOURCE_BYTES:' + row['path'])
    # A separate code/profile pin covers the gate, beyond field19's two files.
    # Only the exact single registration assignment is removed from this digest
    # to avoid a hash cycle. Full current/prospective preimages accompany review.
    source = Path(__file__).read_bytes()
    lines = source.splitlines(keepends=True)
    registrations = [i for i, line in enumerate(lines) if line.startswith(b'REAL_TYPED_ROOTS = ')]
    if len(registrations) != 1:
        stop('NATIVE_GATE_REGISTRATION_SCHEMA')
    lines[registrations[0]] = b'REAL_TYPED_ROOTS = ()\n'
    if sha256(b''.join(lines)).hexdigest() != expected['gate_profile_sha256']:
        stop('NATIVE_GATE_SOURCE_CHANGED')
    project = Path(__file__).resolve().parents[3]
    for row in expected['method_sources']:
        path = _safe_path(project / row['path'])
        if sha256(path.read_bytes()).hexdigest() != row['sha256']:
            stop('NATIVE_METHOD_SOURCE_CHANGED:' + row['path'])
    # Admission starts with cold root-keyed LRUs. No producer/data-module import
    # is needed: if already loaded, ambient caches are rejected before readers.
    for module, name in (('smlt_hedge.regime_v2.execution_inputs', 'tariffs'),
                         ('smlt_hedge.regime_v2.clocks', 'authority'),
                         ('smlt_hedge.regime_v2.lk', 'authority')):
        loaded = sys.modules.get(module)
        if loaded is not None:
            reader = vars(loaded).get(name)
            if type(reader) is not functools._lru_cache_wrapper:
                stop('NATIVE_CACHE_READER_TYPE')
            if functools._lru_cache_wrapper.cache_info(reader).currsize:
                stop('NATIVE_AMBIENT_WARMED_LRU')


@dataclass(frozen=True)
class VerifiedNativeInputs:
    """Fixed immutable 18/19 receipt. Input equality grants no execution right."""
    expected_bytes: bytes
    root: Path
    request_bytes: bytes
    authority_path: Path

    def _readback(self):
        _, profile = _native_profile(self.authority_path, self.expected_bytes)
        _native_sources(profile, self.root)
        return profile

    def check_request(self, request):
        _preflight(request, native=True)
        marker = vars(request['ctx']).get('_headroom_a0')
        if (type(marker) is not dict or any(type(k) is not str for k in marker) or
                set(marker) != {'mode', 'authority_path'} or
                type(marker['mode']) is not str or type(marker['authority_path']) is not str or
                marker['mode'] != NATIVE_PROFILE or marker['authority_path'] != str(self.authority_path)):
            stop('NATIVE_OWNED_DISPATCH_CHANGED')
        profile = self._readback()
        if _json(typed(request, self.root, profile=profile)) != self.request_bytes:
            stop('OWNED_REQUEST_CHANGED_BEFORE_CONSUMER')

    def check_panels(self, panel_now, panel):
        _preflight(panel_now, native=True)
        _preflight(panel, native=True)
        profile = self._readback()
        expected = profile['expected']['panels']
        if (typed(panel_now, self.root, profile=profile) != expected['panel_now'] or
                typed(panel, self.root, profile=profile) != expected['panel']):
            stop('CONSUMED_PANEL_CHANGED')


def _validate_native(request, marker):
    # Empty production registration remains a pre-reader STOP. Test registration
    # mutates only the private module constant in a fresh isolated test process.
    if REAL_TYPED_ROOTS == ():
        stop('REAL_TYPED_PREIMAGES_NOT_ACCEPTED')
    if type(marker['authority_path']) is not str:
        stop('NATIVE_AUTHORITY_PATH_TYPE')
    if (any(type(request[k]) is not int or request[k] != v for k, v in
            (('n_gen', 2000), ('n_adm', 2000), ('seed_offset', 0))) or
            type(request['models']) is not tuple or
            any(type(x) is not str for x in request['models']) or request['models'] != MODELS or
            request['generation_only'] is not True or type(request['cost_mult']) is not float or
            request['cost_mult'] != 1.0 or any(request[k] is not None for k in
            ('capital', 'panel_end', 'px', 'available0')) or
            type(request['universe']) is not str or request['universe'] != 'MAIN'):
        stop('CONDITIONAL_METHOD_OR_NONE_BRANCH')
    _preflight(request, native=True)
    path = _safe_path(Path(marker['authority_path']))
    raw, profile = _native_profile(path)
    ctx = vars(request['ctx'])
    root = ctx.get('root')
    if type(root) is not PosixPath:
        stop('NATIVE_ROOT_TYPE')
    if typed(request, root, profile=profile) != profile['expected']['request']:
        stop('ACTUAL_19_ARGUMENT_PREIMAGE_MISMATCH')
    _native_sources(profile, root)
    owned = _detach(request, native=True)
    if typed(owned, root, profile=profile) != profile['expected']['request']:
        stop('OWNED_NATIVE_PREIMAGE_MISMATCH')
    # Keep dispatch metadata on owned ctx so any later entry still selects the
    # registered profile. It is transport metadata, never typed input authority.
    owned['ctx']._headroom_a0 = dict(marker)
    gate = VerifiedNativeInputs(raw, root, _json(profile['expected']['request']), path)
    gate.check_request(owned)
    return gate, owned


def requires_gate(ctx, commitment):
    """Every Headroom generation-only request needs independently pinned inputs.

    Commitment metadata alone is never pre-reader authority.
    """
    return True


# Independent wire literals from the accepted, unchanged T06/config fixture.
# No environment, call-stack, caller hash or runtime-generated trust exception.
T06_FIXTURE_ROOT = '2c22a269e2512f1342282a3a64647a6fb7094331d20b50033f68fe1d403c64e5'
_T06_EXPECTED = '{"commitment_projection":{"H":126,"admission_seeds":[20960925,20961026,20961127,20961228],"base_head":"142e3d6136636e4ef2a38b3edbf9cf691dc7a62a","base_tree":"8151ca6be3315ae2f3dac7e8f80662bb8e067b20","cutoff":"2026-09-25","generation_seeds":[20260925,20261026,20261127,20261228],"mode":"SYNTHETIC_CORRECTNESS","models":["GLOBAL_FORWARD","GLOBAL_FORWARD_FHS","RECENT_FORWARD","STATE_CONDITIONAL_FORWARD"],"n":2000,"v2_archive":"63de37cf507d890e435de93be812ed6f7dbaf6fbf5436f93e5172514917341ae"},"panel":["frame",["index","DatetimeIndex",["none"],"datetime64[us]","None",["array","<M8[us]",[10],"004020464847060000a0f7635c47060000c07dbd98470600002055dbac47060000802cf9c047060000e00317d54706000040db34e94706000060618e2548060000c038ac39480600002010ca4d480600"]],["index","Index",["none"],"str",null,["array_object","|O",[4],[["str","SMLT"],["str","OFZ_FUNDED"],["str","USDRUB"],["str","MMF"]]]],[[["str","SMLT"],["series",["str","SMLT"],["index","DatetimeIndex",["none"],"datetime64[us]","None",["array","<M8[us]",[10],"004020464847060000a0f7635c47060000c07dbd98470600002055dbac47060000802cf9c047060000e00317d54706000040db34e94706000060618e2548060000c038ac39480600002010ca4d480600"]],"float64",["array","<f8",[10],"0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"],["dict",[]]]],[["str","OFZ_FUNDED"],["series",["str","OFZ_FUNDED"],["index","DatetimeIndex",["none"],"datetime64[us]","None",["array","<M8[us]",[10],"004020464847060000a0f7635c47060000c07dbd98470600002055dbac47060000802cf9c047060000e00317d54706000040db34e94706000060618e2548060000c038ac39480600002010ca4d480600"]],"float64",["array","<f8",[10],"0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"],["dict",[]]]],[["str","USDRUB"],["series",["str","USDRUB"],["index","DatetimeIndex",["none"],"datetime64[us]","None",["array","<M8[us]",[10],"004020464847060000a0f7635c47060000c07dbd98470600002055dbac47060000802cf9c047060000e00317d54706000040db34e94706000060618e2548060000c038ac39480600002010ca4d480600"]],"float64",["array","<f8",[10],"0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"],["dict",[]]]],[["str","MMF"],["series",["str","MMF"],["index","DatetimeIndex",["none"],"datetime64[us]","None",["array","<M8[us]",[10],"004020464847060000a0f7635c47060000c07dbd98470600002055dbac47060000802cf9c047060000e00317d54706000040db34e94706000060618e2548060000c038ac39480600002010ca4d480600"]],"float64",["array","<f8",[10],"0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"],["dict",[]]]]],["dict",[]],true],"request":["dict",[["ctx",["record","types.SimpleNamespace",["dict",[["cfg",["dict",[["costs",["dict",[["broker_fee_futures_rub_per_contract",["float64","0000000000000000"]]]]]]]]]]]],["cfg",["dict",[["owner",["dict",[["liquidity_min_nav",["float64","0000000000000000"]],["capital_main_rub",["float64","408f400000000000"]],["smlt_initial_value_rub",["float64","4079000000000000"]],["es_alpha",["float64","3fee666666666666"]]]]],["r1",["dict",[["liquidity",["dict",[["sweep_hysteresis_nav",["float64","0000000000000000"]],["reserve_var_level",["float64","3fefae147ae147ae"]],["immediate_cash_rate",["float64","0000000000000000"]]]]],["settlement",["dict",[]]],["admission",["dict",[["max_breach_probability",["float64","3fa999999999999a"]],["margin_stress_survival",["list",[["float64","3ff4000000000000"],["float64","3ff8000000000000"]]]]]]]]]],["r3",["dict",[["frontier",["dict",[["horizon_sessions",["int","126"]],["estimation_start",["str","2020-01-01"]],["seed_generation",["int","20260925"]],["seed_admission_shift",["int","50000"]]]]],["universes",["dict",[["MAIN",["list",[["str","SMLT"],["str","OFZ_FUNDED"]]]]]]],["bounds",["dict",[["SMLT",["dict",[["lo",["float64","0000000000000000"]],["hi",["float64","3fe3333333333333"]]]]],["OFZ_FUNDED",["dict",[["lo",["float64","0000000000000000"]],["hi",["float64","3fd999999999999a"]]]]]]]],["costs",["dict",[["smlt",["dict",[["bp",["float64","0000000000000000"]],["half_tick_rub",["float64","0000000000000000"]]]]],["funds",["float64","0000000000000000"]]]]],["portfolio_constraints",["dict",[["funded_plus_margin_plus_reserve_max_nav0",["float64","3ff0000000000000"]],["long_exposure_max_nav0",["float64","3ff0000000000000"]],["derivative_gross_max_nav0",["float64","3ff0000000000000"]]]]]]]],["v2",["dict",[["risk",["dict",[["block_length",["int","10"]],["models",["list",[["str","GLOBAL_FORWARD"],["str","GLOBAL_FORWARD_FHS"],["str","RECENT_FORWARD"],["str","STATE_CONDITIONAL_FORWARD"]]]]]]],["frontier",["dict",[["lexicographic",["dict",[["step2_within_nav0",["float64","3f50624dd2f1a9fc"]],["step3_within_nav0",["float64","3f40624dd2f1a9fc"]]]]]]]]]]],["stress",["dict",[["scenarios",["dict",[["S00_SYNTHETIC",["dict",[]]],["S01_SYNTHETIC",["dict",[]]],["S02_SYNTHETIC",["dict",[]]],["S03_SYNTHETIC",["dict",[]]],["S04_SYNTHETIC",["dict",[]]],["S05_SYNTHETIC",["dict",[]]],["S06_SYNTHETIC",["dict",[]]],["S07_SYNTHETIC",["dict",[]]],["S08_SYNTHETIC",["dict",[]]],["S09_SYNTHETIC",["dict",[]]],["S10_SYNTHETIC",["dict",[]]],["S11_SYNTHETIC",["dict",[]]]]]],["route_groups",["dict",[]]],["margin_multipliers",["list",[["float64","3ff0000000000000"],["float64","3ff8000000000000"]]]]]]]]]],["P",["dict",[["usd",["series",["none"],["index","DatetimeIndex",["none"],"datetime64[us]","None",["array","<M8[us]",[1],"0060c165435c0600"]],"float64",["array","<f8",[1],"000000000000f03f"],["dict",[]]]]]]],["t",["timestamp","2026-09-25T00:00:00","None","datetime64[us]","0060c165435c0600"]],["universe",["str","MAIN"]],["views",["frame",["index","Index",["none"],"str",null,["array_object","|O",[2],[["str","SMLT"],["str","OFZ_FUNDED"]]]],["index","Index",["none"],"str",null,["array_object","|O",[4],[["str","central"],["str","d_long"],["str","d_short"],["str","centre_risk"]]]],[[["str","central"],["series",["str","central"],["index","Index",["none"],"str",null,["array_object","|O",[2],[["str","SMLT"],["str","OFZ_FUNDED"]]]],"float64",["array","<f8",[2],"00000000000000000000000000000000"],["dict",[]]]],[["str","d_long"],["series",["str","d_long"],["index","Index",["none"],"str",null,["array_object","|O",[2],[["str","SMLT"],["str","OFZ_FUNDED"]]]],"float64",["array","<f8",[2],"00000000000000000000000000000000"],["dict",[]]]],[["str","d_short"],["series",["str","d_short"],["index","Index",["none"],"str",null,["array_object","|O",[2],[["str","SMLT"],["str","OFZ_FUNDED"]]]],"float64",["array","<f8",[2],"00000000000000000000000000000000"],["dict",[]]]],[["str","centre_risk"],["series",["str","centre_risk"],["index","Index",["none"],"str",null,["array_object","|O",[2],[["str","SMLT"],["str","OFZ_FUNDED"]]]],"float64",["array","<f8",[2],"00000000000000000000000000000000"],["dict",[]]]]],["dict",[]],true]],["book0",["record","smlt_hedge.adaptive.engine.Book",["dict",[["qf",["array","<f8",[3],"000000000000104000000000000000000000000000000000"]],["qd",["array","<f8",[0],""]],["cash",["float64","4082c00000000000"]],["pending",["dict",[]]],["series",["none"]],["capacity",["none"]],["regime",["str","ORDINARY"]]]]]],["capital",["none"]],["n_gen",["int","2000"]],["n_adm",["int","2000"]],["models",["none"]],["state_choice",["none"]],["seed_offset",["int","0"]],["panel_end",["none"]],["px",["none"]],["available0",["none"]],["cost_mult",["float64","3ff0000000000000"]],["generation_only",["bool",true]],["headroom_commitment",["dict",[["mode",["str","SYNTHETIC_CORRECTNESS"]],["base_head",["str","142e3d6136636e4ef2a38b3edbf9cf691dc7a62a"]],["base_tree",["str","8151ca6be3315ae2f3dac7e8f80662bb8e067b20"]],["v2_archive",["str","63de37cf507d890e435de93be812ed6f7dbaf6fbf5436f93e5172514917341ae"]],["models",["list",[["str","GLOBAL_FORWARD"],["str","GLOBAL_FORWARD_FHS"],["str","RECENT_FORWARD"],["str","STATE_CONDITIONAL_FORWARD"]]]],["generation_seeds",["list",[["int","20260925"],["int","20261026"],["int","20261127"],["int","20261228"]]]],["admission_seeds",["list",[["int","20960925"],["int","20961026"],["int","20961127"],["int","20961228"]]]],["cutoff",["str","2026-09-25"]],["H",["int","126"]],["n",["int","2000"]]]]]]]}'

def _t06_wire(request):
    fields = _record_fields(request['headroom_commitment'])
    if fields is None or type(fields.get('payload')) is not str:
        stop('INDEPENDENT_EXPECTED_AUTHORITY_ABSENT')
    payload = fields['payload']
    try:
        metadata = json.loads(payload)
    except (ValueError, TypeError):
        stop('INDEPENDENT_EXPECTED_AUTHORITY_ABSENT')
    expected = json.loads(_T06_EXPECTED)
    if (type(metadata) is not dict or
            sha256(payload.encode()).hexdigest() != fields.get('identity')):
        stop('INDEPENDENT_EXPECTED_AUTHORITY_ABSENT')
    projection_keys = [k for k, v in expected['request'][1][-1][1][1]]
    projection = {k: metadata.get(k) for k in projection_keys}
    if projection != expected['commitment_projection']:
        stop('INDEPENDENT_EXPECTED_AUTHORITY_ABSENT')
    # All eighteen consumed value arguments plus the independently specified
    # protocol projection of argument nineteen. Dynamic runtime/method bindings
    # do not confer input authority; their existing native check is unchanged.
    values = {k: request[k] for k in REQUEST_FIELDS[:-1]}
    wire = typed(values, Path('/'))
    wire[1].append(['headroom_commitment', typed(projection, Path('/'))])
    return wire


@dataclass(frozen=True)
class SyntheticFixtureInputs:
    """Exact tiny T06 fixture only; never a real typed A0 root."""
    def check_request(self, request):
        if _t06_wire(request) != json.loads(_T06_EXPECTED)['request']:
            stop('T06_FIXED_INPUT_PREIMAGE_MISMATCH')

    def check_panels(self, panel_now, panel):
        expected = json.loads(_T06_EXPECTED)['panel']
        if typed(panel_now, Path('/')) != expected or typed(panel, Path('/')) != expected:
            stop('CONSUMED_PANEL_CHANGED')


def _validate_t06_fixture(request):
    if sha256(_T06_EXPECTED.encode()).hexdigest() != T06_FIXTURE_ROOT:
        stop('T06_FIXED_AUTHORITY_CHANGED')
    gate = SyntheticFixtureInputs()
    gate.check_request(request)
    owned = _detach(request)
    # The unchanged T06 assertion requires phase.cfg is legacy.cfg. This one
    # exact small built-in cfg remains aliased and is checked immediately before
    # and after the first reader; every other input, including axes, is detached.
    owned['cfg'] = request['cfg']
    gate.check_request(owned)
    return gate, owned
