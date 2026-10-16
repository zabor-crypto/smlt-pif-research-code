"""Owner-fetched primary sources (Bank of Russia, rating agencies): validity check of the saved raw bodies and
standard-library parsers (no new dependencies). Raw bytes are never modified; parsed series go to data/interim."""
from __future__ import annotations

import html as _html
import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd

# A server error page saved as if it were data (e.g. the CBR DailyInfo web service answered "Error on server").
_ERROR_MARKERS = (b"Error on server", b"Server Error", b"Runtime Error", "Ошибка сервера".encode())


def is_error_body(content: bytes) -> bool:
    """True for short bodies that are server error pages rather than the requested document."""
    return len(content) < 4096 and any(m in content for m in _ERROR_MARKERS)


def owner_item(root: Path, item_id: str) -> tuple[str, Path | None]:
    """(status, raw path) of an owner-fetch item: OBTAINED / INVALID_RESPONSE / UNAVAILABLE (latest retrieval wins)."""
    man = pd.read_csv(root / "data" / "source_manifest.csv")
    got = man[man["dataset"] == item_id].sort_values("retrieval_timestamp")
    for _, r in got.iloc[::-1].iterrows():
        p = root / str(r["raw_path"])
        if p.exists():
            return ("INVALID_RESPONSE" if is_error_body(p.read_bytes()) else "OBTAINED"), p
    return ("UNAVAILABLE" if got.empty else "RAW_FILE_MISSING"), None


class _DataTable(HTMLParser):
    """Rows of the first <table class="data"> (the CBR statistics table layout)."""

    def __init__(self):
        super().__init__()
        self.rows, self._row, self._cell, self._in, self._done = [], None, None, False, False

    def handle_starttag(self, tag, attrs):
        if tag == "table" and not self._done and "data" in (dict(attrs).get("class") or "").split():
            self._in = True
        elif self._in and tag == "tr":
            self._row = []
        elif self._in and tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if not self._in:
            return
        if tag in ("td", "th") and self._cell is not None:
            self._row.append(re.sub(r"\s+", " ", "".join(self._cell)).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None
        elif tag == "table":
            self._in, self._done = False, True

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def html_table(path: Path) -> list[list[str]]:
    p = _DataTable()
    p.feed(path.read_text(encoding="utf-8", errors="replace"))
    return p.rows


def _num(s: str) -> float:
    return float(s.replace("\xa0", "").replace(" ", "").replace(",", "."))


def key_rate(path: Path) -> pd.Series:
    """Bank of Russia key rate (% p.a.) by effective date."""
    rows = [r for r in html_table(path) if len(r) >= 2 and re.fullmatch(r"\d\d\.\d\d\.\d{4}", r[0])]
    s = pd.Series({pd.to_datetime(r[0], dayfirst=True): _num(r[1]) for r in rows}, name="key_rate_pct").sort_index()
    s.index.name = "date"
    return s


def ruonia(path: Path) -> pd.DataFrame:
    """RUONIA (% p.a.) by rate date with its publication date (published the next business day)."""
    rows = [r for r in html_table(path) if len(r) >= 11 and re.fullmatch(r"\d\d\.\d\d\.\d{4}", r[0])]
    df = pd.DataFrame({"date": [pd.to_datetime(r[0], dayfirst=True) for r in rows], "ruonia_pct": [_num(r[1]) for r in rows],
                       "publication_date": [pd.to_datetime(r[10], dayfirst=True) for r in rows]})
    return df.sort_values("date").set_index("date")


def cbr_fx(path: Path) -> pd.Series:
    """Official Bank of Russia rate (RUB per unit) from XML_dynamic.asp, by the date the rate applies."""
    root = ET.parse(path).getroot()
    vals = {}
    for rec in root.iter("Record"):
        nominal = _num(rec.findtext("Nominal", "1"))
        vals[pd.to_datetime(rec.get("Date"), dayfirst=True)] = _num(rec.findtext("Value")) / nominal
    s = pd.Series(vals, name=f"official_{root.get('ID', '')}").sort_index()
    s.index.name = "date"
    return s


def html_text(path: Path) -> str:
    t = path.read_text(encoding="utf-8", errors="replace")
    t = re.sub(r"<script.*?</script>|<style.*?</style>", " ", t, flags=re.S)
    return re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", t)))
