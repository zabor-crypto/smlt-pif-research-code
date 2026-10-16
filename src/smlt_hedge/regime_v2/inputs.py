"""Regime V2 — newly fetched raw inputs (separate namespace and manifest; the parent manifests are never written).

Only one dataset is added: Binance BTCUSDT one-hour klines for 2017-08 .. 2018-12 (the archive of the parent starts in
2019-01). It extends the PRIOR-ERA BTC history used for the strategic prior of BTC_RUB_EXPERIMENTAL; nothing after 2021
is taken from it. Every archive is verified against the publisher's .CHECKSUM (SHA-256) before it is stored.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

MANIFEST = "regime_v2_source_manifest.csv"
RAW_NS = "REGIME_V2"
DATASET = "binance_klines_prior_2017_2018"
BASE = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h"
COLUMNS = ["source_id", "publisher", "dataset", "instrument", "url", "retrieval_timestamp", "information_timestamp",
           "sha256", "publisher_checksum", "bytes", "validation_status", "validation_note", "raw_path", "notes"]


def months() -> list[str]:
    out, y, m = [], 2017, 8
    while (y, m) <= (2018, 12):
        out.append(f"{y}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _rows(root: Path) -> list[dict]:
    p = Path(root) / "data" / MANIFEST
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fetch(root: Path) -> dict:
    """Bounded fetch (17 monthly archives). An existing snapshot is reused, never overwritten."""
    import requests
    root = Path(root)
    have = {r["source_id"] for r in _rows(root)}
    log = {}
    for per in months():
        fn = f"BTCUSDT-1h-{per}.zip"
        url = f"{BASE}/{fn}"
        sid = f"BINANCE:{DATASET}:{per}"
        dst = root / "data" / "raw" / RAW_NS / "BINANCE" / DATASET / fn
        if sid in have and dst.exists():
            log[per] = "EXISTS"
            continue
        body = requests.get(url, timeout=(30, 120), headers={"User-Agent": "smlt-regime-v2-research/1.0"})
        chk = requests.get(url + ".CHECKSUM", timeout=(30, 120), headers={"User-Agent": "smlt-regime-v2-research/1.0"})
        if body.status_code != 200 or chk.status_code != 200:
            log[per] = f"FETCH_FAILED HTTP {body.status_code}/{chk.status_code}"
            continue
        sha = hashlib.sha256(body.content).hexdigest()
        pub = chk.text.split()[0]
        if sha != pub:
            log[per] = "CHECKSUM_MISMATCH"
            continue
        z = zipfile.ZipFile(io.BytesIO(body.content))
        lines = sum(1 for _ in z.open(z.namelist()[0]))
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".zip.part")
        tmp.write_bytes(body.content)
        tmp.rename(dst)
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        dst.with_suffix(".zip.meta.json").write_text(json.dumps(
            {"url": url, "retrieval_timestamp": ts, "sha256": sha, "publisher_checksum": pub, "bytes": len(body.content)}, indent=1))
        mp = root / "data" / MANIFEST
        new = not mp.exists()
        with mp.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            if new:
                w.writeheader()
            w.writerow({"source_id": sid, "publisher": "BINANCE", "dataset": DATASET, "instrument": "BTCUSDT", "url": url,
                        "retrieval_timestamp": ts, "information_timestamp": per, "sha256": sha, "publisher_checksum": pub,
                        "bytes": len(body.content), "validation_status": "VALID", "validation_note": f"zip lines={lines}; sha256 equals the published .CHECKSUM",
                        "raw_path": str(dst.relative_to(root)), "notes": "PRIOR-ERA proxy history for the BTC strategic prior only"})
        log[per] = "STORED"
    return log


def binance_hourly_prior(root: Path) -> pd.DataFrame:
    """Hour bars 2017-08 .. 2018-12 from the V2 namespace (VALID rows only), same layout as data.crypto.binance_hourly."""
    from ..data import crypto
    frames = []
    for r in sorted(_rows(root), key=lambda x: x["information_timestamp"]):
        if r["dataset"] != DATASET or r["validation_status"] != "VALID":
            continue
        body = (Path(root) / r["raw_path"]).read_bytes()
        if hashlib.sha256(body).hexdigest() != r["sha256"]:
            raise RuntimeError(f"raw body does not match its manifest hash: {r['raw_path']}")
        z = zipfile.ZipFile(io.BytesIO(body))
        for n in z.namelist():
            d = pd.read_csv(z.open(n), header=None, names=crypto.KL_COLS)
            if not str(d.iloc[0, 0]).isdigit():
                d = d.iloc[1:]
            frames.append(d)
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d["open_time"] = crypto._ts(d["open_time"])
    d = d.drop_duplicates("open_time").sort_values("open_time").set_index("open_time")
    return d[["open", "high", "low", "close", "volume"]].astype(float)
