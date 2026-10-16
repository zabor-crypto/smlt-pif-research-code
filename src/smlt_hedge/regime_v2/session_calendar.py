"""B01: frozen research dates plus causal regular-session forecast dates.

The historical projection is immutable and contains dates only. Beyond its
boundary, an unknown exchange schedule means an eligible weekday assumption,
never a read of realised market history. Date origins use the 18:50 MSK cut.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from .cfg import StopCondition

HISTORY_SHA256 = '193133b30d657d1e0cff7905320509ab9ffbf2d68a7b400636b658ff7c443b3a'
LAST_FROZEN_SESSION = pd.Timestamp('2026-09-25')


@dataclass(frozen=True)
class ForecastCalendar:
    sessions: pd.DatetimeIndex
    dates: pd.DatetimeIndex
    evidence: pd.DataFrame


def forecast(root, origin, horizon, through=None):
    """Exactly horizon sessions after origin; optional extended scheduling domain.

    `through` resolves true expiry insertion positions before subtracting roll
    offsets. It never truncates or lengthens the returned admission horizon.
    The entire frozen historical date domain is retained for relative sessions.
    """
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError('forecast horizon must be a positive session count')
    origin = pd.Timestamp(origin).normalize()
    cut = origin.tz_localize('Europe/Moscow') + pd.Timedelta(hours=18, minutes=50)
    base = Path(root)/'data/regime_v2r2_combined'
    content = (base/'research_sessions.csv').read_bytes()
    if sha256(content).hexdigest() != HISTORY_SHA256:
        raise StopCondition('B01_FROZEN_RESEARCH_CALENDAR_IDENTITY_CHANGED')
    historical = pd.DatetimeIndex(pd.to_datetime(content.decode().splitlines()[1:]))
    if origin <= LAST_FROZEN_SESSION and origin not in historical:
        raise StopCondition('B01_ORIGIN_NOT_FROZEN_RESEARCH_SESSION')
    facts = json.loads((base/'forecast_session_facts.json').read_text())
    known = [f for f in facts if pd.Timestamp(f['known_at']) <= cut]
    end = max(LAST_FROZEN_SESSION, origin) + pd.Timedelta(days=2*horizon+14)
    if through is not None:
        end = max(end, pd.Timestamp(through).normalize() + pd.Timedelta(days=7))
    while True:
        extension = pd.bdate_range(LAST_FROZEN_SESSION + pd.Timedelta(days=1), end)
        records, eligible = [], []
        for day in extension:
            applicable = [f for f in known if pd.Timestamp(f['valid_from']) <= day <= pd.Timestamp(f['valid_to'])]
            fact = max(applicable, key=lambda f: pd.Timestamp(f['known_at'])) if applicable else None
            if fact and str(day.date()) in fact['nonregular_sessions']:
                continue  # closures and DSVD-only dates are not regular sessions
            eligible.append(day)
            records.append({'session':day, 'kind':'FORECAST_REGULAR_SESSION',
                            'status':'ORIGIN_KNOWN_EXCHANGE_SCHEDULE' if fact else 'FORECAST_CALENDAR_ASSUMPTION',
                            'authority':fact['source'] if fact else 'MONDAY_FRIDAY_SYNTHETIC_ELIGIBILITY',
                            'known_at':fact['known_at'] if fact else None})
        sessions = historical.append(pd.DatetimeIndex(eligible))
        future = sessions[sessions > origin][:horizon]
        if len(future) == horizon:
            break
        end += pd.Timedelta(days=2*horizon+14)
    if origin not in sessions:
        raise StopCondition('B01_ORIGIN_NOT_REGULAR_SESSION')
    audit = pd.DataFrame([{'session':day, 'kind':'FROZEN_RESEARCH_SESSION',
                           'status':'EXACT_FROZEN_RESEARCH_CALENDAR',
                           'authority':'FROZEN_RESEARCH_DATE_PROJECTION:'+HISTORY_SHA256,
                           'known_at':None} for day in historical] + records).set_index('session')
    dates = pd.DatetimeIndex([origin]).append(future)
    return ForecastCalendar(sessions, dates, audit.reindex(dates))
