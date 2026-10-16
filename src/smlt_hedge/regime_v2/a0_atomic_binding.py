"""Task126 finite source-byte binding, never real A0 admission.

Expected membership is fixed before execution from the pinned Task117/118/120/
121/123/124 exports and Task93 contract. No live build arguments, native object
conversion, exporter execution, economic imports or gate tokens exist here.
Binary descriptors, dtype/axes/NaN metadata and source variants stay verbatim.
The new assumption-qualified generation is always PARTIAL_ATOMIC_BINDING.
"""
from dataclasses import dataclass
from hashlib import sha256
import json

from .a0_typed_constructor import GROUPS, REQUEST_FIELDS

REAL_TYPED_ROOTS = ()
PROVISIONAL = 'TASK126_PROVISIONAL_SOURCE_BINDING'
ARTIFICIAL = 'TASK126_ARTIFICIAL_ONLY'
EXPECTED_REGISTRY_SHA = 'be9af5197555b0f76145d8990c1c43e83e19c06bf72ed0085c2628bb5d9bb90f'
ARTIFICIAL_REGISTRY_SHA = '96d4d0e4366cd661286055a1959a3af43738639cc4cf53ae07887177c91fb92e'
REGISTRY_PATH = 'expected_registry.json'


class EvidenceMissing(RuntimeError):
    pass


def _stop(reason):
    raise EvidenceMissing('EVIDENCE_MISSING:' + reason)


def _pairs(rows):
    result = {}
    for key, value in rows:
        if key in result:
            _stop('DUPLICATE_FIXED_REGISTRY_FIELD')
        result[key] = value
    return result


def _load(data):
    if type(data) is not bytes:
        _stop('EXACT_BYTES_REQUIRED')
    try:
        return json.loads(data, object_pairs_hook=_pairs,
                          parse_constant=lambda s: _stop('UNTYPED_REGISTRY_NONFINITE'))
    except (ValueError, UnicodeError, TypeError):
        _stop('INVALID_FIXED_REGISTRY')


def _serialize(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('ascii')


def _capture(sources):
    if type(sources) is not dict:
        _stop('EXACT_SOURCE_DICT_REQUIRED')
    rows = tuple(sources.items())
    if any(type(name) is not str for name, raw in rows):
        _stop('EXACT_SOURCE_PATH_REQUIRED')
    if any(type(raw) is not bytes for name, raw in rows):
        _stop('EXACT_SOURCE_BYTES_REQUIRED')
    return dict(rows)


def _fixed_registry(owned, profile):
    if type(profile) is not str or profile not in (PROVISIONAL, ARTIFICIAL):
        _stop('UNAPPROVED_FIXED_PROFILE')
    expected = EXPECTED_REGISTRY_SHA if profile == PROVISIONAL else ARTIFICIAL_REGISTRY_SHA
    if REGISTRY_PATH not in owned or sha256(owned[REGISTRY_PATH]).hexdigest() != expected:
        _stop('NO_INDEPENDENT_FIXED_SOURCE_MEMBERSHIP')
    registry = _load(owned[REGISTRY_PATH])
    if (type(registry) is not dict or registry.get('schema') != 'TASK126_FINITE_ATOMIC_BINDING_V1' or
            [g['name'] for g in registry['groups']] != list(GROUPS) or
            [r['argument'] for r in registry['request']] != list(REQUEST_FIELDS) or
            not registry['roots_missing']):
        _stop('EXACT_18_GROUP_19_ARGUMENT_PARTIAL_SCHEMA')
    return registry, expected


@dataclass(frozen=True)
class AtomicBindingDraft:
    """Immutable source payloads and declarations; no native P/ctx/Book exists."""
    groups: tuple
    request: bytes
    readiness: bytes
    stop_register: bytes
    read_set: bytes
    source_payloads: tuple
    expected_source_membership: str
    status: str = 'PARTIAL_ATOMIC_BINDING'


def prepare_atomic_binding(sources, *, profile=PROVISIONAL):
    """Bind only a closed, independently fixed source corpus to a partial draft.

    SHA/size refer to the immutable bytes actually captured and returned.
    The expected registry is documentary source membership, NOT approved real
    typed authority. Missing roots remain explicit; source order is not changed.
    """
    if type(profile) is not str or profile not in (PROVISIONAL, ARTIFICIAL):
        _stop('UNAPPROVED_FIXED_PROFILE')
    owned = _capture(sources)
    registry, expected = _fixed_registry(owned, profile)
    readset = registry['read_set']
    names = [row['path'] for row in readset]
    if len(set(names)) != len(names) or set(owned) != {REGISTRY_PATH, *names}:
        _stop('EXACT_COMPLETE_SOURCE_READ_SET')
    for row in readset:
        raw = owned[row['path']]
        if len(raw) != row['bytes'] or sha256(raw).hexdigest() != row['sha256']:
            _stop('SOURCE_PREIMAGE_SUBSTITUTION:' + row['path'])
    groups = []
    for group in registry['groups']:
        if not group['members'] or any(path not in names for path in group['members']):
            _stop('UNBOUND_GROUP_MEMBER:' + group['name'])
        if not group['missing'] or any(not reason.startswith('EVIDENCE_MISSING:') for reason in group['missing']):
            _stop('MISSING_ROOT_REASON_REQUIRED')
        # References resolve ONLY to source_payloads, not native constructors.
        groups.append((group['name'], _serialize(group)))
    request = registry['request']
    for row in request:
        if row['argument'] in ('capital', 'panel_end', 'px', 'available0'):
            wire = row['wire']
            if wire.get('value', 'MISSING') is not None or wire.get('branch') != 'PROSPECTIVE_DISTINCT_NOT_D26_PARITY':
                _stop('PROSPECTIVE_NONE_BRANCH_REQUIRED')
    matrix = {'status': 'PARTIAL_ATOMIC_BINDING', 'descriptor_status': 'NOT_APPROVED_REAL_A0',
              'authority_active': False, 'generation': registry['generation'],
              'groups': registry['groups'], 'arguments': request,
              'method_constants': registry['method_constants'],
              'axes_and_units': registry['axes_and_units'],
              'qualifications': registry['qualifications'],
              'source_archives': registry['source_archives'],
              'expected_membership_is_real_authority': False}
    stops = {'status': 'PARTIAL_ATOMIC_BINDING', 'B2': 'BLOCKED',
             'REAL_TYPED_ROOTS': [], 'A0': 'NOT_CONSTRUCTED', 'STOP_FOR_MASTER': True,
             'roots_missing': ['EVIDENCE_MISSING:' + s for s in registry['roots_missing']],
             'group_missing': [{'group': g['name'], 'missing': g['missing']} for g in registry['groups']],
             'argument_missing': [{'argument': r['argument'], 'missing': r['missing']}
                                  for r in request if r['missing']]}
    return AtomicBindingDraft(tuple(groups), _serialize({'target': 'PROVISIONAL_BuildRequest_NOT_CALLABLE',
                              'status': 'NOT_APPROVED_REAL_A0', 'fields': request}),
                              _serialize(matrix), _serialize(stops), _serialize(readset),
                              tuple((path, owned[path]) for path in [REGISTRY_PATH, *names]), expected)


def verify_draft(draft, sources, *, profile=PROVISIONAL):
    """Compare a primitive immutable draft to the same fixed source preimages.

    Equality is a substitution check, never a gate/authority approval.
    """
    if type(draft) is not AtomicBindingDraft:
        _stop('EXACT_DRAFT_TYPE')
    for value in (draft.request, draft.readiness, draft.stop_register, draft.read_set):
        if type(value) is not bytes:
            _stop('EXACT_DRAFT_BYTES')
    for value in (draft.groups, draft.source_payloads):
        if (type(value) is not tuple or any(type(row) is not tuple or len(row) != 2 or
                type(row[0]) is not str or type(row[1]) is not bytes for row in value)):
            _stop('EXACT_IMMUTABLE_DRAFT_PAIRS')
    if type(draft.expected_source_membership) is not str or type(draft.status) is not str:
        _stop('EXACT_DRAFT_STATUS_TYPE')
    expected = prepare_atomic_binding(sources, profile=profile)
    if draft != expected:
        _stop('DRAFT_SUBSTITUTION')
    return 'MATCH_FIXED_PROVISIONAL_SOURCE_BYTES_NOT_ADMISSION'


def require_admission(draft):
    _stop('REAL_TYPED_ROOTS_EMPTY_ATOMIC_DRAFT_CANNOT_ADMIT_OR_CALL_BUILD')
