"""D22 selectors. Dated projection is an assumption, never an intraday archive."""
from functools import lru_cache
import json
from pathlib import Path

import pandas as pd

from .cfg import StopCondition

POLICY_INFORMATION_CUT = "18:50 MSK"
HELD_BOOK_CLEARING_MARGIN_MARK = "19:05 MSK"
CONSERVATIVE_CLOCK_ASSUMPTION = "CONSERVATIVE_CLOCK_ASSUMPTION"


@lru_cache(maxsize=8)
def authority(root):
    base = Path(root) / "data/regime_v2r2_combined"
    daily = pd.read_csv(base / "mr1_daily.csv", parse_dates=["requested_date"])
    pairs = json.loads((base / "mr1_transitions.json").read_text())
    return daily, pairs


def select(root, family, date, consumer="POLICY_INFORMATION_CUT", at=None):
    """Both policies call this selector. Untimestamped execution cannot assume 19:00.

    Unique dated observations follow the accepted dated as-of convention;
    actual dated margin projection uses the day's observation. The 17 transition
    decisions use the adjudicated prior unambiguous anchor explicitly.
    """
    daily, pairs = authority(str(root))
    day = pd.Timestamp(date).normalize().tz_localize(None)
    rows = daily[(daily.family == family) & (daily.requested_date <= day)]
    pair = next((r for r in pairs if r["family"] == family and r["date"] == str(day.date())), None)
    if pair:
        old = float(pair["policy_mr1"])
        boundary = pair["known_effective_boundary"]
        if consumer == "POLICY_INFORMATION_CUT":
            return old, "LAST_UNAMBIGUOUS_CAUSALLY_EFFECTIVE_MR1"
        if boundary:
            if at is None:
                raise StopCondition("D22_EXECUTION_CLOCK_UNKNOWN: exact boundary needs a timestamp")
            mark = pd.Timestamp(at)
            if mark.tzinfo is None:
                raise StopCondition("D22_MARK_TIMEZONE_REQUIRED")
            new = next(float(x) for x in pair["candidates"] if float(x) != old)
            return (new if mark >= pd.Timestamp(boundary) else old), "PROVEN_EXACT_NOTICE"
        return float(pair["conservative_transition_mr1"]), CONSERVATIVE_CLOCK_ASSUMPTION
    # Do not promote a previous ambiguous candidate into an effective anchor.
    ambiguous = {pd.Timestamp(r["date"]) for r in pairs if r["family"] == family}
    rows = rows[~rows.requested_date.isin(ambiguous)]
    if rows.empty:
        raise StopCondition(f"D22_NO_CAUSAL_MR1_ANCHOR:{family}:{day.date()}")
    return float(rows.iloc[-1].mr1), "DATED_OBSERVATION_PROJECTION_ASSUMPTION"


def apply(root, family, out):
    out = out.copy()
    out["mr1"], out["mr1_policy_label"] = zip(*(select(root, family, t) for t in out.index))
    # Owner-adjudicated daily model clock, independent of execution prices.
    marks = [select(root, family, t, "CLEARING_MARGIN_MARK",
                    pd.Timestamp(t).normalize().tz_localize("Europe/Moscow") + pd.Timedelta(hours=19, minutes=5))
             for t in out.index]
    out["mr1_clearing"], out["mr1_clearing_label"] = zip(*marks)
    return out
