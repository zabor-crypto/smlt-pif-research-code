"""Two pinned archival source adapters; provisional native values, no A0 authority.

Archive custody must be freshly verified before calling (Task130 review runner).
Every consumed member is independently SHA/size pinned here; no live expected
hashes, filesystem readers, class dispatch, research producers or field19 exist.
New conf object-string/us schema is explicit, not old D26 memory identity.
"""
from hashlib import sha256
import json
import math
import struct

REAL_TYPED_ROOTS = ()
AUTHORITY_STATUS = ('REAL_A0_NOT_ADMITTED', 'B2_BLOCKED',
          'G2_NOT_PASSED', 'ECONOMICS_NOT_EXECUTED')
FUNDED = ('SMLT', 'EQ_FUNDED', 'OFZ_FUNDED', 'GOLD_FUNDED', 'BTC_RUB_EXPERIMENTAL', 'MMF')
DERIVATIVES = ('SMLT_FUT', 'EQ_HEDGE_FUT', 'RATES_HEDGE_FUT', 'CNY_PERP', 'OIL_FUT')
AXES = ('ISSUER_CREDIT_RISK', 'MARKET_RISK', 'RATES_STATE', 'RUB_STATE')
PANEL_COLUMNS = ('SMLT', 'SMLT_FUT', 'EQ_FUNDED', 'EQ_HEDGE_FUT', 'OFZ_FUNDED',
    'RATES_HEDGE_FUT', 'GOLD_FUNDED', 'CNY_PERP', 'OIL_FUT_USD',
    'BTC_RUB_EXPERIMENTAL', 'USDRUB', 'CNY_SWAP_FRAC', 'MMF')
CHOICE_FIELDS = ('available', 'condition', 'level', 'starts', 'n_episodes',
                 'n_eligible_block_starts', 'tried')
STARTS_SHA256 = 'a6922c55d01bb21ba18daff9db9566f3de2ed85e14af78a47925c52c70f98915'
# Fixed reviewed source member preimages, never caller-computed expected hashes.
SOURCE_PINS = (
    ('Task123/review/sources/D26_generation_05/primary_setup.json', '767cf411c47692835aa442ca64c8135888710d2ab7ed6e9662233b0ca3fda998', 195263),
    ('Task123/review/sources/D26_generation_05/primary_state.json', '7eeee966f8a94e5037d117a4e06af913c05e2f4a7b4fd0ee30b185270ac05dc3', 7539),
    ('Task123/review/RunA/PROVISIONAL_BOOK_LEAVES.json', '92eb8912fc5536ff9d06eda177e33d0d69b65a11a8fd38c22446be60314ed7e8', 2110),
    ('Task124/RunA/four_axis_candidate.json', '32cd4d15654e69ad71e00ad7fc920e03d630ca9e1fecad590272f606f356f5ce', 4407461),
    ('Task124/RunA/seven_field_choice_candidate.json', '137a415cab17d38ae4491d8d7ee5aae569b9cc352d943b2c9be2b1101c885684', 3146),
    ('Task124/RunA/choice_eligibility_membership.json', 'b81e9733b55750569c1f3e72832ed37eab98c160791b41d3b793400cb03e8c31', 166635),
    ('Task124/RunA/vectors/candidate_starts_panel_positions.bin', 'a6922c55d01bb21ba18daff9db9566f3de2ed85e14af78a47925c52c70f98915', 1056),
    ('Task117/review/RunA/history_panel13.col00.bin', 'dbd5460dfd9733576031475669ae57bb7ce86bb3f8b9fda123a2fd49de133b26', 9096),
    ('Task117/review/RunA/history_panel13.col01.bin', 'b3d9dcba0b032c6aea50fd42457bb435955e1f55fd489ef0c1f7f669b883ff83', 9096),
    ('Task117/review/RunA/history_panel13.col02.bin', 'f183bc1d938d9ac78f2c3f528e81dfec4d11450fcb97441eea74fb0d32107685', 9096),
    ('Task117/review/RunA/history_panel13.col03.bin', '074c21f201afa2c9d6b346d22f2752f479ee207ad40411539fa3a9d90a57bb8e', 9096),
    ('Task117/review/RunA/history_panel13.col04.bin', '26944ac145498104cd96d6b94704f0c263878ff207ed4f7ae9ba58c40a7ef5de', 9096),
    ('Task117/review/RunA/history_panel13.col05.bin', 'e7ebc3e67ead6fde7324f23cde7761e9e919ee9e2bd2aa4c2e1fcfb648835f49', 9096),
    ('Task117/review/RunA/history_panel13.col06.bin', '6616587abeb653a6c810e0e23539354db7a006556b309dfb827f615694118645', 9096),
    ('Task117/review/RunA/history_panel13.col08.bin', '87e8f08ff47a5b2e8f69ad67c152279d34c3f88c8c5e7defcaacf1e59c99d757', 9096),
    ('Task117/review/RunA/history_panel13.col09.bin', '34a4be6273ac64bc1feb13d6a6bcf507a9ddd83e26857d3cf37b94f558635c65', 9096),
    ('Task117/review/RunA/history_panel13.col10.bin', 'b86587cf1926cfa5bd43080d62af2ae5ad396e68b7d4abf22bc7b59893467d99', 9096),
    ('Task117/review/RunA/history_panel13.col12.bin', '80d67eee2630a1432190829870700dd2dfb91d9231ceeecd3c8a868e30a4bc28', 9096),
    ('Task117/review/RunA/history_panel13.index.bin', '1ea32f3a7a508fd4351bc9eacecc3aec6fe96d47cd1165103f4db19512da7038', 9096),
    ('Task117/review/RunA/history_panel13.json', 'e67f9b0cd1aaf2984a5dcf23f70ed5df3b6567a80157e96f102abc5348b4085e', 304983),
    ('Task117/review/RunA/history_panel13_source_row_positions.json', '926ba77c3ff86edadf05be978df00b84404863ad9b9a330de0464672e17453ba', 10550),
)
BOOK_SOURCES = ('Task123/review/sources/D26_generation_05/primary_setup.json', 'Task123/review/sources/D26_generation_05/primary_state.json', 'Task123/review/RunA/PROVISIONAL_BOOK_LEAVES.json')
COMBINED_SOURCES = ('Task123/review/sources/D26_generation_05/primary_setup.json', 'Task124/RunA/four_axis_candidate.json', 'Task124/RunA/seven_field_choice_candidate.json', 'Task124/RunA/choice_eligibility_membership.json', 'Task124/RunA/vectors/candidate_starts_panel_positions.bin', 'Task117/review/RunA/history_panel13.col00.bin', 'Task117/review/RunA/history_panel13.col01.bin', 'Task117/review/RunA/history_panel13.col02.bin', 'Task117/review/RunA/history_panel13.col03.bin', 'Task117/review/RunA/history_panel13.col04.bin', 'Task117/review/RunA/history_panel13.col05.bin', 'Task117/review/RunA/history_panel13.col06.bin', 'Task117/review/RunA/history_panel13.col08.bin', 'Task117/review/RunA/history_panel13.col09.bin', 'Task117/review/RunA/history_panel13.col10.bin', 'Task117/review/RunA/history_panel13.col12.bin', 'Task117/review/RunA/history_panel13.index.bin', 'Task117/review/RunA/history_panel13.json', 'Task117/review/RunA/history_panel13_source_row_positions.json')


class EvidenceMissing(RuntimeError):
    pass


def _stop(field):
    raise EvidenceMissing('EVIDENCE_MISSING:' + field)


def _require(ok, field):
    if not ok:
        _stop(field)


def _keys(value, names, field):
    _require(type(value) is dict and all(type(k) is str for k in value) and
             set(value) == set(names), field + ':EXACT_FIELDS')


def _owned(source_bytes, required):
    _require(type(source_bytes) is dict, 'EXACT_READ_SET')
    items = tuple(source_bytes.items())
    _require(all(type(k) is str for k, v in items), 'SOURCE_PATH_TYPE')
    _require(all(type(v) is bytes for k, v in items), 'SOURCE_BYTES_TYPE')
    owned = dict(items)
    _require(set(owned) == set(required), 'EXACT_READ_SET')
    pins = {name: (digest, size) for name, digest, size in SOURCE_PINS}
    for name in required:
        digest, size = pins[name]
        _require(len(owned[name]) == size and sha256(owned[name]).hexdigest() == digest,
                 'SOURCE_PREIMAGE:' + name)
    return owned


def _pairs(rows):
    result = {}
    for k, v in rows:
        _require(k not in result, 'DUPLICATE_JSON_FIELD:' + k)
        result[k] = v
    return result


def _json(data):
    try:
        return json.loads(data, object_pairs_hook=_pairs,
                          parse_constant=lambda value: _stop('JSON_NONFINITE:' + value))
    except (ValueError, UnicodeError, TypeError):
        _stop('JSON_ENCODING')


def _book_record(book):
    _keys(book, ('qf', 'qd', 'cash', 'pending', 'series', 'capacity', 'regime'), 'Book')
    for name, expected in [('qf', [40000.] + [0.] * 5), ('qd', [0.] * 5)]:
        values = book[name]
        _require(type(values) is list and len(values) == len(expected), 'Book.' + name + ':SHAPE')
        _require(all(type(a) is float and struct.pack('<d', a) == struct.pack('<d', b)
                     for a, b in zip(values, expected)), 'Book.' + name + ':TYPE_OR_BITS')
    _require(type(book['cash']) is float and book['cash'] == 10000000., 'Book.cash')
    _require(type(book['pending']) is dict and book['pending'] == {}, 'Book.pending')
    _require(book['series'] is None and book['capacity'] is None, 'Book.series_capacity')
    _require(type(book['regime']) is str and book['regime'] == 'ORDINARY', 'Book.regime:SOURCE_REQUIRED')


def _book_inputs(owned):
    setup, state, descriptor = [_json(owned[name]) for name in BOOK_SOURCES]
    for book in (setup['book0'], state['book0'], descriptor['book_fields']):
        _book_record(book)
    _require(setup['book0'] == state['book0'] == descriptor['book_fields'], 'Book.SOURCE_AGREEMENT')
    _require([v['name'] for v in setup['fspecs']] == list(FUNDED) and
             [v['name'] for v in setup['dspecs']] == list(DERIVATIVES), 'Book.AXES')
    _require(descriptor['funded_axis'] == list(FUNDED) and
             descriptor['derivative_axis'] == list(DERIVATIVES), 'Book.DESCRIPTOR_AXES')
    _require(state['cutoff'] == setup['config']['cutoff'] == descriptor['origin_date'] == '2026-09-25', 'Book.ORIGIN')
    return state['book0']


def _book_type():
    # Static import of the reviewed frozen constructor closure; no producer call.
    from ..adaptive.engine import Book
    return Book


def materialize_archival_book(source_bytes):
    """Actual frozen Book; all seven constructor arguments explicitly source supplied."""
    book = _book_inputs(_owned(source_bytes, BOOK_SOURCES))
    import numpy as np
    Book = _book_type()
    return Book(qf=np.array(book['qf'], dtype='<f8', copy=True),
                qd=np.array(book['qd'], dtype='<f8', copy=True), cash=book['cash'],
                pending=dict(book['pending']), series=book['series'],
                capacity=book['capacity'], regime=book['regime'])


def _primitive_choice(choice):
    _keys(choice, CHOICE_FIELDS, 'Choice')
    _require(type(choice['available']) is bool and choice['available'] is True, 'Choice.available')
    _require(type(choice['condition']) is dict and
             all(type(k) is str and type(v) is str for k, v in choice['condition'].items()), 'Choice.condition')
    _require(type(choice['level']) is str and choice['level'] ==
             'ISSUER_CREDIT_RISK=DISTRESSED & MARKET_RISK=NORMAL', 'Choice.level')
    for key, expected in [('n_episodes', 5), ('n_eligible_block_starts', 132)]:
        _require(type(choice[key]) is int and choice[key] == expected, 'Choice.' + key)
    _require(type(choice['starts']) is list and len(choice['starts']) == 132 and
             all(type(i) is int for i in choice['starts']), 'Choice.starts')
    _require(type(choice['tried']) is list and len(choice['tried']) == 3, 'Choice.tried')
    for i, row in enumerate(choice['tried']):
        _keys(row, ('level', 'n_axes', 'n_episodes', 'n_eligible_block_starts', 'qualifies'), 'Choice.tried')
        _require(type(row['level']) is str and type(row['qualifies']) is bool and
                 row['qualifies'] is (i == 2), 'Choice.tried:LABEL_OR_BOOL')
        _require(all(type(row[k]) is int for k in ('n_axes', 'n_episodes', 'n_eligible_block_starts')),
                 'Choice.tried:EXACT_INT')


def _panel_inputs(owned):
    prefix = 'Task117/review/RunA/'
    panel = _json(owned[prefix + 'history_panel13.json'])
    _keys(panel, ('column_index_name', 'columns', 'index', 'kind', 'label', 'shape', 'vectors'), 'Panel')
    _require(panel['kind'] == 'DataFrame' and panel['shape'] == [1137, 13] and
             panel['columns'] == list(PANEL_COLUMNS), 'Panel.SHAPE_AXES')
    index = panel['index']
    _require(index['class'] == 'DatetimeIndex' and index['timezone'] == 'None' and
             index['name'] == {'type': 'str', 'value': 'date'} and
             panel['column_index_name'] == {'type': 'None'}, 'Panel.INDEX_NAMES_TZ')
    meta = index['values']; raw_index = owned[prefix + 'history_panel13.index.bin']
    _require(meta['numpy_dtype'] == '<M8[us]' and meta['shape'] == [1137] and
             meta['order'] == 'C' and len(raw_index) == 9096 and
             sha256(raw_index).hexdigest() == meta['sha256'], 'Panel.DATETIME_UNIT_OR_BITS')
    _require(len(index['labels']) == 1137 and len(set(index['labels'])) == 1137,
             'Panel.INDEX_LABELS')
    _keys(panel['vectors'], PANEL_COLUMNS, 'Panel.VECTORS')
    values = {}
    for name in PANEL_COLUMNS:
        vector = panel['vectors'][name]
        _require(vector['shape'] == [1137], 'Panel.COLUMN_SHAPE:' + name)
        if name in ('CNY_PERP', 'CNY_SWAP_FRAC'):
            _keys(vector, ('encoding', 'numpy_dtype', 'pandas_dtype', 'shape', 'values'), 'Panel.OBJECT:' + name)
            _require(vector['numpy_dtype'] == '|O' and vector['pandas_dtype'] == 'object' and
                     len(vector['values']) == 1137, 'Panel.OBJECT_DTYPE:' + name)
            cells = []
            for cell in vector['values']:
                _keys(cell, ('type', 'float64_le_hex'), 'Panel.OBJECT_CELL')
                _require(cell['type'] == 'float' and type(cell['float64_le_hex']) is str and
                         len(cell['float64_le_hex']) == 16, 'Panel.OBJECT_CELL_TYPE')
                cells.append(struct.unpack('<d', bytes.fromhex(cell['float64_le_hex']))[0])
            values[name] = cells
        else:
            _require(vector['numpy_dtype'] == '<f8' and vector['pandas_dtype'] == 'float64' and
                     vector['order'] == 'C', 'Panel.DTYPE:' + name)
            raw = owned[prefix + vector['file']]
            _require(len(raw) == 9096 and sha256(raw).hexdigest() == vector['sha256'], 'Panel.COLUMN_BITS:' + name)
            cells = struct.unpack('<1137d', raw)
            _require(bytes(math.isnan(x) for x in cells).hex() == vector['NaN_mask_u1_hex'] and
                     bytes(x == 0. and math.copysign(1., x) < 0. for x in cells).hex() == vector['signed_zero_mask_u1_hex'],
                     'Panel.NAN_SIGNED_ZERO:' + name)
            values[name] = raw
    return panel, raw_index, values


def _combined_inputs(owned):
    conf = _json(owned['Task124/RunA/four_axis_candidate.json'])
    choice = _json(owned['Task124/RunA/seven_field_choice_candidate.json'])['seven_fields']
    eligibility = _json(owned['Task124/RunA/choice_eligibility_membership.json'])
    _primitive_choice(choice)
    _require(conf['axes'] == list(AXES) and len(conf['dates']) == len(conf['rows']) == 1178 and
             conf['dates'] == sorted(set(conf['dates'])), 'Conf.DATES_AXES')
    labels = {axis: [] for axis in AXES}
    for date, row in zip(conf['dates'], conf['rows']):
        _keys(row['conf'], AXES, 'Conf.ROW')
        _require(row['date'] == date, 'Conf.ROW_DATE')
        for axis in AXES:
            value = row['conf'][axis]
            _require(type(value) is str and value in conf['historical_order'][axis], 'Conf.HISTORICAL_LABEL')
            labels[axis].append(value)
    panel, raw_index, values = _panel_inputs(owned)
    rowmap = _json(owned['Task117/review/RunA/history_panel13_source_row_positions.json'])['row_positions']
    _require(rowmap == eligibility['panel_to_source_calendar'] == list(range(41, 1178)), 'Panel.SOURCE_ROW_MAP')
    _require([date[:10] for date in panel['index']['labels']] == [conf['dates'][i] for i in rowmap], 'Panel.CONF_MEMBERSHIP')
    starts_wire = owned['Task124/RunA/vectors/candidate_starts_panel_positions.bin']
    _require(len(starts_wire) == 1056 and sha256(starts_wire).hexdigest() == STARTS_SHA256 and
             list(struct.unpack('<132q', starts_wire)) == choice['starts'], 'Choice.STARTS_BITS')
    membership = eligibility['selected_start_membership']
    _require([row['start_panel_position'] for row in membership] == choice['starts'] and
             len({row['episode_id'] for row in membership}) == 5, 'Choice.ACCEPTED_MEMBERSHIP')
    for row in membership:
        i = row['start_panel_position']
        _require(1 <= i <= 1127 and row['start_date'] == panel['index']['labels'][i][:10] and
                 row['source_calendar_start_position'] == rowmap[i], 'Choice.PANEL_COORDINATES')
    setup = _json(owned[BOOK_SOURCES[0]])
    hierarchy = setup['config']['v2']['risk']['STATE_CONDITIONAL_FORWARD']['hierarchy']
    _require(hierarchy == [list(('MARKET_RISK', 'ISSUER_CREDIT_RISK', 'RATES_STATE', 'RUB_STATE')),
                          ['MARKET_RISK', 'ISSUER_CREDIT_RISK', 'RATES_STATE'],
                          ['MARKET_RISK', 'ISSUER_CREDIT_RISK'], ['MARKET_RISK']], 'Choice.SOURCE_HIERARCHY')
    _require(set(choice['condition']) == set(hierarchy[2]) and
             choice['condition'] == {'MARKET_RISK': 'NORMAL', 'ISSUER_CREDIT_RISK': 'DISTRESSED'}, 'Choice.CONDITION')
    condition = {axis: choice['condition'][axis] for axis in hierarchy[2]}
    # Original source dict insertion order differs from canonical sorted JSON.
    tried = [{k: row[k] for k in ('level', 'n_axes', 'n_episodes', 'n_eligible_block_starts', 'qualifies')}
             for row in choice['tried']]
    return conf['dates'], labels, raw_index, values, choice, condition, tried, starts_wire


def materialize_combined_conf_choice_panel(source_bytes):
    """Return (conf, panel, Choice); copy accepted values without any research producer."""
    dates, labels, raw_index, values, choice, condition, tried, starts_wire = _combined_inputs(
        _owned(source_bytes, COMBINED_SOURCES))
    import numpy as np
    import pandas as pd
    conf_index = pd.DatetimeIndex(np.array(dates, dtype='datetime64[us]'), name='date', copy=True)
    panel_index = pd.DatetimeIndex(np.frombuffer(raw_index, dtype='<M8[us]').copy(), name='date', copy=True)
    conf = pd.DataFrame(labels, index=conf_index, dtype=object, copy=True)
    conf.columns = pd.Index(AXES, dtype=object, name=None)
    columns = {}
    for name in PANEL_COLUMNS:
        if name in ('CNY_PERP', 'CNY_SWAP_FRAC'):
            columns[name] = pd.Series(values[name], dtype=object, index=panel_index, copy=True)
        else:
            columns[name] = pd.Series(np.frombuffer(values[name], dtype='<f8').copy(), index=panel_index, copy=True)
    panel = pd.DataFrame(columns, index=panel_index, copy=True)
    panel.columns = pd.Index(PANEL_COLUMNS, dtype=object, name=None)
    _require(str(panel.index.dtype) == str(conf.index.dtype) == 'datetime64[us]', 'NATIVE_DATETIME_UNIT')
    _require([str(t) for t in panel.dtypes] == [
        'object' if name in ('CNY_PERP', 'CNY_SWAP_FRAC') else 'float64' for name in PANEL_COLUMNS], 'NATIVE_PANEL_DTYPES')
    native_choice = {'available': choice['available'], 'condition': condition, 'level': choice['level'],
        'starts': np.frombuffer(starts_wire, dtype='<i8').copy(), 'n_episodes': choice['n_episodes'],
        'n_eligible_block_starts': choice['n_eligible_block_starts'], 'tried': tried}
    return conf, panel, native_choice
