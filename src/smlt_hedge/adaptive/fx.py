"""Official USD/RUB (Bank of Russia R01235) on the portfolio information clock.

CBR XML_dynamic lists one record per VALUE date. A rate is set on a business day and takes effect the next calendar day
(Friday's rate carries the Saturday date and applies through Monday). Information convention used everywhere:

    usd_known(t) = record with the latest value date <= t + 1 calendar day

i.e. the rate SET on day t is treated as known at the day-t close (15:50 UTC). Evidence: the FORTS clearing of
2026-09-25 values USD-linked contracts at 84.3414 = the CBR record dated 2026-09-26 (set 2026-09-25); the check over the
whole history is in outputs/adaptive_route_reconciliation.csv. The strictly-lagged alternative (value date <= t) is kept
for the timing sensitivity.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .evidence import MANIFEST


def cbr_usd_records(root: Path) -> pd.Series:
    m = pd.read_csv(Path(root) / "data" / MANIFEST, dtype=str)
    m = m[(m["dataset"] == "usd_official") & (m["validation_status"] == "VALID")]
    recs = {}
    for rp in sorted(m["raw_path"]):
        t = (Path(root) / rp).read_bytes().decode("windows-1251")
        for d, v in re.findall(r'<Record Date="([0-9.]+)" Id="R01235"><Nominal>1</Nominal><Value>([0-9,]+)</Value>', t):
            recs[pd.Timestamp(pd.to_datetime(d, format="%d.%m.%Y"))] = float(v.replace(",", "."))
    s = pd.Series(recs).sort_index()
    s.index.name = "value_date"
    return s


def usd_known(records: pd.Series, dates: pd.DatetimeIndex, lag_calendar_days: int = 1) -> pd.Series:
    """Rate known at the close of each trading date (as-of join on value_date <= date + lag)."""
    key = pd.DatetimeIndex(dates) + pd.Timedelta(days=lag_calendar_days)
    pos = records.index.searchsorted(key, side="right") - 1
    vals = np.where(pos >= 0, records.values[np.clip(pos, 0, None)], np.nan)
    return pd.Series(vals, index=pd.DatetimeIndex(dates), name="USDRUB_known")
