"""V3 Phase 1.1: external BTC ECONOMIC PROXY history (label PROXY_HISTORY; never MOEX execution history).

Alignment to the MOEX decision calendar (crypto trades 24/7):
  BTC price for MOEX day t  = close of the Binance BTCUSDT hour bar ending at `sync_hour` UTC on date t
                               (MOEX main-session close 18:50 MSK = 15:50 UTC; futures settlement at the evening clearing)
  BTC return for MOEX day t = P_t / P_{t_prev} - 1, t_prev = previous MOEX trading day
so every BTC return covers the SAME economic interval as the SMLT close-to-close return, and weekend / holiday
moves are aggregated into the next MOEX interval automatically. The conventional 00:00-UTC daily close (8 hours
after the MOEX close) is kept as the NON-synchronous alternative for the alignment sensitivity test.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import load

KL_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "n_trades",
           "taker_base", "taker_quote", "ignore"]


def _ts(x: pd.Series) -> pd.Series:
    """Binance archives switched from millisecond to microsecond timestamps in 2025."""
    x = x.astype("int64")
    return pd.to_datetime(np.where(x > 10**14, x // 1000, x), unit="ms", utc=True)


def binance_hourly(root: Path, cutoff: str) -> pd.DataFrame:
    m = load.manifest(root)
    m = m[m["dataset"] == "binance_klines"]
    frames = []
    for rp in sorted(m["raw_path"]):
        z = zipfile.ZipFile(io.BytesIO((Path(root) / rp).read_bytes()))
        for n in z.namelist():
            d = pd.read_csv(z.open(n), header=None, names=KL_COLS)
            if not str(d.iloc[0, 0]).isdigit():        # header row in some archives
                d = d.iloc[1:]
            frames.append(d)
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d["open_time"] = _ts(d["open_time"])
    d["close"] = d["close"].astype(float)
    d["volume"] = d["volume"].astype(float)
    d = d.drop_duplicates("open_time").sort_values("open_time").set_index("open_time")
    end = pd.Timestamp(cutoff, tz="UTC") + pd.Timedelta(days=1)
    return d[d.index < end][["open", "high", "low", "close", "volume"]].astype(float)


def coinbase_daily(root: Path, cutoff: str) -> pd.Series:
    m = load.manifest(root)
    m = m[m["dataset"] == "coinbase_candles"]
    rows = []
    for rp in sorted(m["raw_path"]):
        rows += json.loads((Path(root) / rp).read_bytes())
    if not rows:
        return pd.Series(dtype=float)
    d = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"]).drop_duplicates("time")
    d.index = pd.to_datetime(d["time"], unit="s").dt.normalize()
    d = d.sort_index()
    return d.loc[:pd.Timestamp(cutoff), "close"].astype(float)


def synced_close(hourly: pd.DataFrame, cal: pd.DatetimeIndex, hour_utc: int, max_stale_h: int = 6) -> pd.Series:
    """BTC close at `hour_utc` UTC on each MOEX date (last hour bar that CLOSED at or before that time)."""
    close_time = hourly.index + pd.Timedelta(hours=1)
    s = pd.Series(hourly["close"].values, index=close_time.tz_convert(None))
    tgt = pd.DatetimeIndex(cal) + pd.Timedelta(hours=hour_utc)
    pos = s.index.searchsorted(tgt, side="right") - 1
    out = np.full(len(cal), np.nan)
    ok = pos >= 0
    stale = np.full(len(cal), True)
    stale[ok] = (tgt[ok] - s.index[pos[ok]]) > pd.Timedelta(hours=max_stale_h)
    out[ok & ~stale] = s.values[pos[ok & ~stale]]
    return pd.Series(out, index=cal)


def utc_day_close(hourly: pd.DataFrame, cal: pd.DatetimeIndex) -> pd.Series:
    """Conventional daily close (00:00 UTC ending calendar day t): the NON-synchronous alternative."""
    return synced_close(hourly, cal, 24)
