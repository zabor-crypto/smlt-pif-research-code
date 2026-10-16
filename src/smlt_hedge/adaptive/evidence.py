"""Adaptive redesign — bounded primary-evidence retrieval (Section I of the redesign brief).

Every fetched body is stored immutably under ``data/raw/ADAPTIVE_V1/<publisher>/<dataset>/`` BEFORE parsing, with a
``.meta.json`` sidecar, and one row in ``data/adaptive_source_manifest.csv`` (separate from the legacy
``data/source_manifest.csv`` so the frozen V1–V4.1 pipeline reproduces unchanged).

A successful HTTP status or a hash is not proof of content: every body is validated by a publisher-specific check
(``validate``) and the result is written to the manifest (``validation_status``). Only VALID bodies may be parsed.

Transport: MOEX ISS and the Bank of Russia are not reachable over HTTPS from the research machine (TLS connections
time out) but answer over plain HTTP. The transport actually used is recorded per row (``url`` keeps the scheme).
Plain-HTTP bodies are cross-checked against overlapping HTTPS-archived owner snapshots where possible.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import requests

PARSER_VERSION = "adaptive-evidence-1.0"
RAW_NS = "ADAPTIVE_V1"
MANIFEST = "adaptive_source_manifest.csv"
COLUMNS = ["source_id", "publisher", "dataset", "instrument", "url", "transport", "retrieval_timestamp",
           "information_timestamp", "sha256", "bytes", "validation_status", "validation_note", "parser_version",
           "raw_path", "notes"]


@dataclass
class Row:
    source_id: str
    publisher: str
    dataset: str
    instrument: str
    url: str
    transport: str
    retrieval_timestamp: str
    information_timestamp: str
    sha256: str
    bytes: int
    validation_status: str
    validation_note: str
    parser_version: str
    raw_path: str
    notes: str = ""


def _slug(s: str, n: int = 60) -> str:
    return re.sub(r"[^A-Za-z0-9._=-]+", "_", s)[:n].strip("_")


def validate(publisher: str, dataset: str, body: bytes) -> tuple[str, str]:
    """Publisher-specific content validation. Returns (status, note)."""
    try:
        if publisher == "MOEX_ISS":
            d = json.loads(body)
            if not isinstance(d, dict) or not d:
                return "INVALID", "empty JSON"
            blocks = [k for k, v in d.items() if isinstance(v, dict) and "columns" in v]
            if not blocks:
                return "INVALID", "no ISS data block"
            n = sum(len(d[k].get("data", [])) for k in blocks)
            return "VALID", f"ISS blocks={blocks} rows={n}"
        if publisher == "Bank_of_Russia":
            t = body.decode("windows-1251", errors="replace")
            if "<ValCurs" not in t:
                return "INVALID", "no <ValCurs> root (server error page?)"
            n = t.count("<Record ")
            return ("VALID" if n > 0 else "INVALID"), f"records={n}"
        if publisher == "BINANCE":
            z = zipfile.ZipFile(io.BytesIO(body))
            names = z.namelist()
            lines = sum(1 for _ in z.open(names[0]))
            return ("VALID" if lines > 0 else "INVALID"), f"zip members={names} lines={lines}"
        if publisher == "Bank_of_Russia_html":
            t = body.decode("utf-8", errors="replace")
            n = t.count("<td")
            ok = "data" in t and n > 20 and "Формат запроса не распознан" not in t
            return ("VALID" if ok else "INVALID"), f"html cells={n}"
        if publisher in ("ISSUER", "ACRA", "E_DISCLOSURE"):
            t = body[:4000].decode("utf-8", errors="replace").lower()
            if body[:4] == b"%PDF":
                return "VALID_PDF", "PDF body"
            bad = ("access denied", "403 forbidden", "captcha", "запрос отклон")
            if any(b in t for b in bad):
                return "INVALID", "access-denied / captcha page"
            return "UNVERIFIED_HTML", "HTML body; content must be read before any fact is extracted"
    except Exception as e:  # noqa: BLE001
        return "INVALID", f"validation error: {type(e).__name__}: {e}"
    return "UNVERIFIED", "no validator for publisher"


class Store:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.manifest = self.root / "data" / MANIFEST

    def path_for(self, publisher: str, dataset: str, url: str, suffix: str) -> Path:
        h = hashlib.sha256(url.encode()).hexdigest()[:16]
        tail = _slug(url.split("?")[0].rsplit("/", 1)[-1], 40) or "body"
        return self.root / "data" / "raw" / RAW_NS / _slug(publisher) / _slug(dataset) / f"{tail}__{h}{suffix}"

    def rows(self) -> list[dict]:
        if not self.manifest.exists():
            return []
        with self.manifest.open(newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    def _append(self, row: Row) -> None:
        if row.source_id in {r["source_id"] for r in self.rows()}:
            return
        new = not self.manifest.exists()
        with self.manifest.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            if new:
                w.writeheader()
            w.writerow(asdict(row))

    def fetch(self, url: str, *, publisher: str, dataset: str, instrument: str, suffix: str,
              information_timestamp: str = "", notes: str = "", retries: int = 4, timeout: float = 120.0) -> Path | None:
        """Fetch once and store immutably; an existing snapshot is reused (never overwritten)."""
        p = self.path_for(publisher, dataset, url, suffix)
        if p.exists():
            return p
        body, err = None, ""
        for k in range(retries):
            try:
                r = requests.get(url, timeout=(30, timeout), headers={"User-Agent": "smlt-adaptive-research/1.0"})
                if r.status_code == 200 and r.content:
                    body = r.content
                    break
                err = f"HTTP {r.status_code}"
            except requests.RequestException as e:
                err = f"{type(e).__name__}"
            time.sleep(2 * 2**k)
        if body is None:
            self._append(Row(f"{publisher}:{dataset}:{p.stem}", publisher, dataset, instrument, url, url.split(":")[0],
                             datetime.now(timezone.utc).isoformat(timespec="seconds"), information_timestamp, "", 0,
                             "FETCH_FAILED", err, PARSER_VERSION, "", notes))
            return None
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".part")
        tmp.write_bytes(body)
        tmp.rename(p)
        sha = hashlib.sha256(body).hexdigest()
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        status, note = validate(publisher, dataset, body)
        meta = {"url": url, "retrieval_timestamp": ts, "sha256": sha, "bytes": len(body), "validation_status": status,
                "validation_note": note, "parser_version": PARSER_VERSION, "information_timestamp": information_timestamp}
        p.with_suffix(p.suffix + ".meta.json").write_text(json.dumps(meta, indent=1))
        self._append(Row(f"{publisher}:{dataset}:{p.stem}", publisher, dataset, instrument, url, url.split(":")[0], ts,
                         information_timestamp, sha, len(body), status, note, PARSER_VERSION,
                         str(p.relative_to(self.root)), notes))
        return p


# ---------------------------------------------------------------- bounded fetch list
ISS = "http://iss.moex.com/iss"


def iss(path: str, **params) -> str:
    return f"{ISS}/{path.lstrip('/')}?{urlencode({'iss.meta': 'off', **params})}"


def fetch_iss_paged(st: Store, path: str, block: str, *, dataset: str, instrument: str, notes: str = "",
                    **params) -> list[Path]:
    out, start = [], 0
    while True:
        p = st.fetch(iss(path, start=start, **params), publisher="MOEX_ISS", dataset=dataset, instrument=instrument,
                     suffix=".json", information_timestamp=params.get("till", ""), notes=notes)
        if p is None:
            break
        out.append(p)
        d = json.loads(p.read_bytes())
        cur = d.get(f"{block}.cursor")
        n = len(d.get(block, {}).get("data", []))
        if cur and cur["data"]:
            idx, total, size = cur["data"][0]
            start = idx + size
            if start >= total:
                break
        elif n < 100:
            break
        else:
            start += n
    return out


FUNDED_ROUTE_ETFS = {"SBMX": "broad equity (TR fund)", "TMOS": "broad equity (IMOEX fund)",
                     "EQMX": "broad equity (IMOEX fund)", "SBGB": "OFZ (government bond fund)"}


def run(root: Path, cutoff: str = "2026-09-25") -> dict:
    st = Store(root)
    log = {}
    # 1. official USD/RUB (Bank of Russia R01235), yearly requests; value-date convention validated separately
    for y0, y1 in [("01/12/2021", "31/12/2022"), ("01/01/2023", "31/12/2023"), ("01/01/2024", "31/12/2024"),
                   ("01/01/2025", "31/12/2025"), ("01/01/2026", "30/09/2026")]:
        url = f"http://www.cbr.ru/scripts/XML_dynamic.asp?date_req1={y0}&date_req2={y1}&VAL_NM_RQ=R01235"
        log[f"cbr_usd_{y0}"] = str(st.fetch(url, publisher="Bank_of_Russia", dataset="usd_official", instrument="USDRUB",
                                            suffix=".xml", information_timestamp=y1,
                                            notes="official USD/RUB; record Date = value date (set on the prior business day)"))
    # 2. MOEX USD/RUB TOM (exchange market until the June-2024 suspension) for cross-check only
    log["usd_tom"] = len(fetch_iss_paged(st, "history/engines/currency/markets/selt/boards/CETS/securities/USD000UTSTOM.json",
                                         "history", dataset="currency_history", instrument="USD000UTSTOM",
                                         **{"from": "2021-12-01", "till": cutoff}))
    # 3. funded broad-equity / OFZ fund routes (TQTF until 2026-06-19, TQBR after the board migration)
    for s, note in FUNDED_ROUTE_ETFS.items():
        st.fetch(iss(f"securities/{s}.json"), publisher="MOEX_ISS", dataset="security_spec", instrument=s, suffix=".json",
                 notes=note)
        for board in ("TQTF", "TQBR"):
            log[f"{s}_{board}"] = len(fetch_iss_paged(
                st, f"history/engines/stock/markets/shares/boards/{board}/securities/{s}.json", "history",
                dataset="shares_history", instrument=s, notes=note, **{"from": "2021-12-01", "till": cutoff}))
    # 4. longer auxiliary priors for broad factors (used only as labelled auxiliary risk/carry priors)
    for s in ("IMOEX", "MCFTR", "RGBITR", "RGBI"):
        log[f"{s}_long"] = len(fetch_iss_paged(st, f"history/engines/stock/markets/index/securities/{s}.json", "history",
                                               dataset="index_history_long", instrument=s,
                                               **{"from": "2013-01-01", "till": "2021-12-31"}))
    # 5. direct BTC/RUB exchange history (Binance BTCRUB, delisted 2024-01) for the overlap cross-check
    for y in range(2022, 2025):
        for mth in range(1, 13):
            if (y, mth) > (2024, 1):
                break
            fn = f"BTCRUB-1h-{y}-{mth:02d}.zip"
            url = f"https://data.binance.vision/data/spot/monthly/klines/BTCRUB/1h/{fn}"
            st.fetch(url, publisher="BINANCE", dataset="btcrub_klines", instrument="BTCRUB", suffix=".zip",
                     information_timestamp=f"{y}-{mth:02d}", notes="direct BTC/RUB exchange history (delisted 2024-01)")
    return log


def issuer_probe(root: Path) -> dict:
    """Bounded retry of issuer / rating primary sources (D4 M1, M2). Bodies are stored and validated; no fact is
    extracted automatically."""
    import yaml
    st = Store(root)
    cfg = yaml.safe_load((Path(root) / "configs" / "owner_fetch.yaml").read_text())   # owner-declared URLs only
    targets = [("ISSUER" if "samolet" in i["url"] else "E_DISCLOSURE", i["id"], i["url"]) for i in cfg["issuer"]]
    targets += [("ACRA", i["id"], i["url"]) for i in cfg["ratings"] if "acra" in i["url"]]
    out = {}
    for pub, ds, url in targets:
        for u in (url, url.replace("https://", "http://", 1)):
            suffix = ".pdf" if u.lower().endswith(".pdf") else ".html"
            p = st.fetch(u, publisher=pub, dataset=ds, instrument="SMLT", suffix=suffix, retries=2, timeout=60)
            out[u] = str(p) if p else "FETCH_FAILED"
    return out


def run_priors(root: Path) -> dict:
    """Bounded pre-2022 AUXILIARY prior histories (era-separated; never pooled with the post-2022 sample):
    RUONIA (CBR), official USD/RUB (CBR), GLDRUB_TOM and CNYRUB_TOM (MOEX currency market), 2013-2021."""
    st = Store(root)
    log = {}
    for y in range(2013, 2022):
        url = f"http://www.cbr.ru/scripts/XML_dynamic.asp?date_req1=01/01/{y}&date_req2=31/12/{y}&VAL_NM_RQ=R01235"
        log[f"usd_{y}"] = str(st.fetch(url, publisher="Bank_of_Russia", dataset="usd_official_prior", instrument="USDRUB",
                                       suffix=".xml", information_timestamp=f"{y}-12-31", notes="auxiliary prior era"))
        url = ("http://www.cbr.ru/hd_base/ruonia/dynamics/?UniDbQuery.Posted=True"
               f"&UniDbQuery.From=01.01.{y}&UniDbQuery.To=31.12.{y}")
        log[f"ruonia_{y}"] = str(st.fetch(url, publisher="Bank_of_Russia_html", dataset="ruonia_prior", instrument="RUONIA",
                                          suffix=".html", information_timestamp=f"{y}-12-31", notes="auxiliary prior era"))
    for s in ("GLDRUB_TOM", "CNYRUB_TOM"):
        log[s] = len(fetch_iss_paged(st, f"history/engines/currency/markets/selt/boards/CETS/securities/{s}.json",
                                     "history", dataset="currency_history_prior", instrument=s,
                                     **{"from": "2013-01-01", "till": "2021-12-31"}))
    return log
