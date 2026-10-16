"""Deterministic MOEX ISS client that stores raw pages before parsing.

ISS pagination: history endpoints return ``<block>.cursor`` = [INDEX, TOTAL, PAGESIZE];
we request ``start=0,100,200,...`` until exhausted. Each page is an immutable raw snapshot.
The connection from some networks is flaky, so requests are sequential with bounded retries.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

from .raw_store import RawStore

ISS = "https://iss.moex.com/iss"
PUBLISHER = "MOEX_ISS"


class ISSError(RuntimeError):
    pass


def _get(url: str, retries: int = 6, timeout: float = 90.0) -> bytes:
    last: Exception | None = None
    for k in range(retries):
        try:
            r = requests.get(url, timeout=(30, timeout), headers={"User-Agent": "smlt-hedge-research/0.1"})
            if r.status_code == 200 and r.content.strip().startswith(b"{"):
                return r.content
            last = ISSError(f"HTTP {r.status_code} for {url}")
        except requests.RequestException as e:  # network flakiness
            last = e
        time.sleep(min(60, 3 * 2**k))
    raise ISSError(f"failed after {retries} attempts: {url}: {last}")


def iss_url(path: str, **params) -> str:
    params = {"iss.meta": "off", **{k: v for k, v in params.items() if v is not None}}
    return f"{ISS}/{path.lstrip('/')}?{urlencode(params)}"


def fetch_paged(
    store: RawStore,
    path: str,
    block: str,
    *,
    dataset: str,
    variable: str,
    instrument: str,
    publication_date: str,
    point_in_time_valid: str,
    notes: str = "",
    offline: bool = False,
    **params,
) -> list[Path]:
    """Fetch every page of ``block`` for an ISS endpoint, saving raw pages. Returns page paths."""
    pages: list[Path] = []
    start = 0
    while True:
        url = iss_url(path, start=start, **params)
        p = store.raw_path_for(PUBLISHER, dataset, url)
        if not p.exists():
            if offline:
                raise ISSError(f"offline mode and raw snapshot missing: {url}")
            content = _get(url)
            p = store.save(
                content,
                publisher=PUBLISHER,
                dataset=dataset,
                variable=variable,
                instrument=instrument,
                url=url,
                publication_date=publication_date,
                point_in_time_valid=point_in_time_valid,
                notes=notes,
            )
        pages.append(p)
        d = json.loads(p.read_bytes())
        cur = d.get(f"{block}.cursor")
        n_rows = len(d.get(block, {}).get("data", []))
        if cur and cur["data"]:
            idx, total, size = cur["data"][0]
            start = idx + size
            if start >= total:
                break
        else:
            # endpoints without cursor: stop when a page is short/empty
            if n_rows < 100:
                break
            start += n_rows
    return pages


def fetch_single(
    store: RawStore,
    path: str,
    *,
    dataset: str,
    variable: str,
    instrument: str,
    publication_date: str,
    point_in_time_valid: str,
    notes: str = "",
    offline: bool = False,
    **params,
) -> Path:
    url = iss_url(path, **params)
    p = store.raw_path_for(PUBLISHER, dataset, url)
    if p.exists():
        return p
    if offline:
        raise ISSError(f"offline mode and raw snapshot missing: {url}")
    return store.save(
        _get(url),
        publisher=PUBLISHER,
        dataset=dataset,
        variable=variable,
        instrument=instrument,
        url=url,
        publication_date=publication_date,
        point_in_time_valid=point_in_time_valid,
        notes=notes,
    )


def pages_to_frame(pages: list[Path], block: str) -> pd.DataFrame:
    frames = []
    for p in pages:
        d = json.loads(Path(p).read_bytes())
        b = d.get(block)
        if b and b["data"]:
            frames.append(pd.DataFrame(b["data"], columns=b["columns"]))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
