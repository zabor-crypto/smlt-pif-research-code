import json

import numpy as np
import pandas as pd
import pytest

from smlt_hedge.data.processed import ex_dates, total_return_prices
from smlt_hedge.data.raw_store import RawStore, sha256_bytes
from smlt_hedge.validation.checks import validate_panel


def test_ex_date_rule_t1_and_t2():
    cal = pd.bdate_range("2023-01-02", "2024-12-31")
    # T+1 era: record Wednesday 2024-05-15 -> ex-date same day; record Sunday 2024-05-19 -> ex Friday 17th
    r = pd.Series(pd.to_datetime(["2024-05-15", "2024-05-19"]))
    assert list(ex_dates(r, cal)) == [pd.Timestamp("2024-05-15"), pd.Timestamp("2024-05-17")]
    # T+2 era: record Wednesday 2023-05-17 -> ex Tuesday 16th
    assert ex_dates(pd.Series([pd.Timestamp("2023-05-17")]), cal).iloc[0] == pd.Timestamp("2023-05-16")


def test_total_return_includes_dividend_on_ex_date():
    cal = pd.bdate_range("2024-05-13", periods=5)
    px = pd.Series([100.0, 100.0, 90.0, 90.0, 91.0], index=cal)
    divs = pd.DataFrame({"registryclosedate": [pd.Timestamp("2024-05-15")], "value": [10.0], "currencyid": ["RUB"]})
    tr, d = total_return_prices(px, divs, cal)
    assert tr.loc["2024-05-15"] == pytest.approx(0.0)
    assert d.loc["2024-05-15"] == 10.0


def test_raw_store_immutable_and_checksummed(tmp_path):
    st = RawStore(tmp_path)
    kw = dict(publisher="X", dataset="d", variable="v", instrument="i", url="https://x/a.json?b=1",
              publication_date="2026-09-25", point_in_time_valid="TRUE")
    p1 = st.save(b'{"a":1}', **kw)
    p2 = st.save(b'{"a":2}', **kw)  # same URL -> must NOT overwrite
    assert p1 == p2 and p1.read_bytes() == b'{"a":1}'
    man = pd.read_csv(tmp_path / "data" / "source_manifest.csv")
    assert len(man) == 1 and man["checksum"].iloc[0] == sha256_bytes(b'{"a":1}')
    meta = json.loads(p1.with_suffix(p1.suffix + ".meta.json").read_text())
    assert meta["sha256"] == sha256_bytes(b'{"a":1}')


def test_validation_flags_lookahead_and_jumps():
    idx = pd.bdate_range("2026-09-21", "2026-09-29")
    P = pd.DataFrame({"A": np.linspace(100, 110, len(idx))}, index=idx)
    P.iloc[3, 0] = 200
    R = P.pct_change(fill_method=None)
    v = validate_panel(P, R, "2026-09-25").set_index("series")
    assert not v.loc["A", "PASS_no_lookahead"]
    assert v.loc["A", "n_abs_return_gt_jump"] >= 1
