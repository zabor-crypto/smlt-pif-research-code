"""Immutable raw-response store with a provenance manifest.

Every byte fetched from an external source is written to ``data/raw/<publisher>/<dataset>/``
*before* any transformation, under a file name derived from the request URL. Existing raw
files are never overwritten: a repeated fetch of the same URL reuses the snapshot (unless the
caller explicitly asks for a new, separately-named snapshot). Each snapshot gets one row in
``data/source_manifest.csv``.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_COLUMNS = [
    "source_id",
    "dataset",
    "variable",
    "instrument",
    "url",
    "publisher",
    "publication_date",
    "retrieval_timestamp",
    "point_in_time_valid",
    "checksum",
    "notes",
    "raw_path",
]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return sha256_bytes(Path(p).read_bytes())


def _slug(s: str, n: int = 80) -> str:
    return re.sub(r"[^A-Za-z0-9._=-]+", "_", s)[:n].strip("_")


@dataclass
class ManifestRow:
    source_id: str
    dataset: str
    variable: str
    instrument: str
    url: str
    publisher: str
    publication_date: str
    retrieval_timestamp: str
    point_in_time_valid: str  # "TRUE" | "FALSE" | "PARTIAL:<reason>"
    checksum: str
    notes: str = ""
    raw_path: str = ""


@dataclass
class RawStore:
    root: Path  # project root
    manifest_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.manifest_path = self.root / "data" / "source_manifest.csv"

    # ------------------------------------------------------------------ paths
    def raw_path_for(self, publisher: str, dataset: str, url: str, suffix: str = ".json") -> Path:
        h = hashlib.sha256(url.encode()).hexdigest()[:16]
        tail = _slug(url.split("?")[0].rsplit("/", 1)[-1].replace(".json", ""), 40)
        return self.root / "data" / "raw" / _slug(publisher) / _slug(dataset) / f"{tail}__{h}{suffix}"

    # --------------------------------------------------------------- manifest
    def _existing_ids(self) -> set[str]:
        if not self.manifest_path.exists():
            return set()
        with self.manifest_path.open(newline="", encoding="utf-8") as fh:
            return {r["source_id"] for r in csv.DictReader(fh)}

    def append_manifest(self, row: ManifestRow) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if row.source_id in self._existing_ids():
            return
        new = not self.manifest_path.exists()
        with self.manifest_path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
            if new:
                w.writeheader()
            w.writerow(asdict(row))

    # ------------------------------------------------------------------ write
    def save(
        self,
        content: bytes,
        *,
        publisher: str,
        dataset: str,
        variable: str,
        instrument: str,
        url: str,
        publication_date: str,
        point_in_time_valid: str,
        notes: str = "",
        suffix: str = ".json",
    ) -> Path:
        path = self.raw_path_for(publisher, dataset, url, suffix)
        if path.exists():
            # immutable: never overwrite an existing snapshot
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(content)
        tmp.rename(path)
        checksum = sha256_bytes(content)
        meta = {
            "url": url,
            "retrieval_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sha256": checksum,
            "bytes": len(content),
        }
        path.with_suffix(path.suffix + ".meta.json").write_text(json.dumps(meta, indent=1))
        self.append_manifest(
            ManifestRow(
                source_id=f"{_slug(publisher)}:{_slug(dataset)}:{path.stem}",
                dataset=dataset,
                variable=variable,
                instrument=instrument,
                url=url,
                publisher=publisher,
                publication_date=publication_date,
                retrieval_timestamp=meta["retrieval_timestamp"],
                point_in_time_valid=point_in_time_valid,
                checksum=checksum,
                notes=notes,
                raw_path=str(path.relative_to(self.root)),
            )
        )
        return path
