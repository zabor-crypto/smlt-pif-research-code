"""Exact interval certification of complete single-book LK base-unit exposure.

Triangle inequality bounds any partition into absolute signed-sum groups.
This does not bound monetary margin or positions in hidden external accounts.
No legal netting configuration or calendar-spread credit is assumed.
"""
from decimal import Decimal, InvalidOperation, localcontext
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .cfg import StopCondition

AUTHORITY_SHA256 = "33789e810ccc83c8acc08d8d8c992ccc7ecfdc7ec88e9d88a6b882152220c3e0"
AGGREGATION = 'INTERVAL_BOUNDED_LK_AGGREGATION'
SAFE = 'CERTIFIED_SAFE'
BREACH = 'CERTIFIED_BREACH'
AMBIGUOUS = 'AGGREGATION_AMBIGUOUS'
NOT_CERTIFIED = 'CANDIDATE_LK_NOT_CERTIFIED'
ACTUAL_AMBIGUOUS = 'ACTUAL_LK_AGGREGATION_AMBIGUOUS'
UNKNOWN = 'LK_UNIT_MAPPING_UNKNOWN'
FORECAST_ASSUMPTION = 'FORECAST_CONTRACT_UNIT_ASSUMPTION'
TIER = 'LK1_TIER_REACHED_MR2_MR3_NOT_IMPLEMENTED'
ALIASES = {'EQ_HEDGE_FUT':'MXI_FUT', 'RATES_HEDGE_FUT':'RGBI_FUT',
           'OIL_FUT':'BR_FUT', 'CNY_PERP':'CNYRUBF'}


@lru_cache(maxsize=8)
def authority(root):
    data = (Path(root)/'data/regime_v2r2_combined/lk_unit_authority.json').read_bytes()
    if hashlib.sha256(data).hexdigest() != AUTHORITY_SHA256:
        raise StopCondition(UNKNOWN + ':AUTHORITY_IDENTITY')
    return json.loads(data)


def terms(root, route, series, date):
    a = authority(str(root))
    rt = a['routes'].get(ALIASES.get(route, route))
    s = a['series'].get(str(series))
    if rt is None or s is None or s['base'] != rt['base'] or s['units'] != rt['units']:
        raise StopCondition(UNKNOWN + ':' + str(series))
    label = rt['classification']
    if 'before' in rt and pd.Timestamp(date) < pd.Timestamp(rt['before']):
        label = rt['earlier_classification']
    return {'lk_base':rt['base'], 'lk_units':int(s['units']),
            'lk_classification':label, 'lk_source_sha256':s['source_sha256']}


def forecast_terms(root, route, origin_series, date):
    """Conditional model continuation of a validated origin contract convention.

    Only the internal forward producer uses this authority. Actual SECIDs still
    require strict terms(), including the origin anchoring this assumption.
    This does not assert the terms of any future exchange-listed contract.
    """
    origin = terms(root, route, origin_series, date)
    return {**origin, 'lk_route':ALIASES.get(route, route),
            'lk_origin_series':str(origin_series),
            'lk_origin_classification':origin['lk_classification'],
            'lk_classification':FORECAST_ASSUMPTION}


class CandidateRejected(StopCondition):
    """Local admission failure; enumeration may continue, economics must not."""
    def __init__(self, reason, reports):
        super().__init__(reason)
        self.reports = reports


def _number(value, reason):
    try:
        v = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise StopCondition(reason) from None
    if not v.is_finite():
        raise StopCondition(reason)
    return v


def _integer(value, reason):
    v = _number(value, reason)
    if v != v.to_integral_value():
        raise StopCondition('STOP_FOR_MASTER:NON_INTEGER_LK_BASE_UNIT_INPUT')
    return int(v)


def _headroom(limit, bound):
    if isinstance(limit, int):
        return limit-bound
    with localcontext() as context:
        context.prec = max(len(str(abs(bound))), len(limit.as_tuple().digits)) + abs(limit.as_tuple().exponent) + 2
        return limit-bound


def intervals(a, positions, t, n):
    """Per base and path: |sum q*u| <= sum_groups |sum_group q*u| <= sum |q|*u.

    The first inequality applies to group sums; the second applies inside each
    group. Groups must partition the SAME complete single-book positions.
    Python integers avoid overflow and float rounding across LK1.
    """
    series, limits = {}, {}
    canonical = 'lk_bases' in a
    for j, group in enumerate(positions):
        base = a['lk_bases'][j] if canonical else j
        limit = _number(a['lk1'][t,j] if a.get('lk1') is not None else 'NaN', UNKNOWN)
        if limit <= 0:
            raise StopCondition(UNKNOWN + ':LK1')
        if limit == limit.to_integral_value():
            limit = int(limit)
        limits[base] = min(limits.get(base, limit), limit)
        series.setdefault(base, {})
        for key, quantity in group.items():
            q = np.broadcast_to(np.asarray(quantity), (n,))
            if not np.any(q != 0):
                continue
            data = a['series'][j].get(key, {})
            units = data.get('lk_units') if canonical else a.get('lk_units', [None]*len(positions))[j]
            try:
                units = _integer(units, UNKNOWN + ':' + str(key))
                if units <= 0 or (canonical and data.get('lk_base') != base):
                    raise StopCondition(UNKNOWN + ':' + str(key))
                counts = np.array([_integer(v, UNKNOWN + ':' + str(key)) for v in q], dtype=object)
            except StopCondition as exc:
                exc.reports = {base: {'LK_LOWER_BOUND': None, 'LK_UPPER_BOUND': None,
                    'LK1': limit, 'LOWER_HEADROOM': None, 'UPPER_HEADROOM': None,
                    'INTERVAL_WIDTH': None, 'LIVE_SERIES_COUNT': None,
                    'ALL_SERIES_SAME_SIGN': None, 'LK_CERTIFICATION': 'UNIT_MAPPING_UNKNOWN'}}
                raise
            # Repeated route references to the same series belong to one series.
            series[base][key] = series[base].get(key, np.zeros(n, dtype=object)) + counts*units
    reports = {}
    for base, population in series.items():
        signed = sum(population.values(), np.zeros(n, dtype=object))
        upper = sum((np.abs(q) for q in population.values()), np.zeros(n, dtype=object))
        lower = np.abs(signed)
        limit = limits[base]
        labels = np.array([SAFE if u < limit else BREACH if l >= limit else AMBIGUOUS
                           for l,u in zip(lower,upper)])
        count = sum((q != 0 for q in population.values()), np.zeros(n, dtype=int))
        positive = sum((q > 0 for q in population.values()), np.zeros(n, dtype=int))
        negative = sum((q < 0 for q in population.values()), np.zeros(n, dtype=int))
        reports[base] = {'LK_LOWER_BOUND': lower, 'LK_UPPER_BOUND': upper, 'LK1': limit,
            'LOWER_HEADROOM': np.array([_headroom(limit,v) for v in lower], dtype=object),
            'UPPER_HEADROOM': np.array([_headroom(limit,v) for v in upper], dtype=object),
            'INTERVAL_WIDTH': upper-lower, 'LIVE_SERIES_COUNT': count,
            'ALL_SERIES_SAME_SIGN': (positive == 0) | (negative == 0), 'LK_CERTIFICATION': labels}
    return reports


def guard(a, positions, t, n, *, candidate=False):
    reports = intervals(a, positions, t, n)
    # A breach on any base/path takes precedence; all reports remain attached.
    for label, reason in ((BREACH, TIER), (AMBIGUOUS, NOT_CERTIFIED if candidate else ACTUAL_AMBIGUOUS)):
        for base, row in reports.items():
            if np.any(row['LK_CERTIFICATION'] == label):
                exc = CandidateRejected(reason + ':' + str(base), reports) if candidate else StopCondition(reason + ':' + str(base))
                exc.reports = reports
                raise exc
    return reports


def actual_guard(su):
    reports = []
    markets = [(g[3], g[4]) for g in su.gen] + [(mp.din, mp.mkt) for mp in su.adm]
    for din, mkt in markets:
        a = mkt.execution
        if a is None:
            continue
        alias = a.get('series_alias', [{} for _ in din])
        actual = []
        for j, inst in enumerate(din):
            key = a['active'][0][j]
            if su.book0.series is None:
                actual.append({key:np.array([su.book0.qd[j]])})
            else:
                population = {}
                for rawkey, quantity in su.book0.series[j].items():
                    resolved = alias[j].get(rawkey,rawkey)
                    counts = np.array([_integer(v, UNKNOWN + ':' + str(rawkey)) for v in np.atleast_1d(quantity)], dtype=object)
                    population[resolved] = population.get(resolved, np.zeros(len(counts),dtype=object)) + counts
                actual.append(population)
        reports.append(guard(a, actual, 0, 1))
    return reports


def candidate_guard(su, x):
    """Validate actual book first, then the proposed active-series integer book.

    Reference/admission engines also test every resulting live-series book,
    including residuals after partial fills. No quantity is clamped to LK1.
    """
    actual_guard(su)
    reports = []
    markets = [(g[3], g[4]) for g in su.gen] + [(mp.din, mp.mkt) for mp in su.adm]
    for din, mkt in markets:
        a = mkt.execution
        if a is None:
            continue
        target = []
        for j, inst in enumerate(din):
            key = a['active'][0][j]
            q = int(np.trunc(float(x[su.routes.index(inst.name)])*su.nav0/su.notional[inst.name]))
            target.append({key:np.array([q], dtype=object)})
        reports.append(guard(a, target, 0, 1, candidate=True))
    return reports


def serialise(reports):
    """JSON-ready telemetry, preserving arbitrary integers and exact decimals."""
    if isinstance(reports, dict):
        return {str(k):serialise(v) for k,v in reports.items()}
    if isinstance(reports, (list, tuple, np.ndarray)):
        return [serialise(v) for v in reports]
    if isinstance(reports, Decimal):
        return int(reports) if reports == reports.to_integral_value() else str(reports)
    if isinstance(reports, np.generic):
        return reports.item()
    return reports
