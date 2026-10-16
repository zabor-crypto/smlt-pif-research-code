"""Raw ISS pages -> tidy interim tables (data/interim/*.csv).

Rules
- Only rows with TRADEDATE <= decision_date are kept (hard point-in-time filter; asserted).
- Weekend/holiday exchange sessions (MOEX runs weekend sessions since 2025) are dropped from
  the daily research calendar; their information is carried by the next weekday close.
  The number of dropped rows is reported by validation.
- Prices are never forward-filled here. Alignment happens in processed.py with explicit rules.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


def manifest(root: Path) -> pd.DataFrame:
    return pd.read_csv(Path(root) / "data" / "source_manifest.csv", dtype=str).drop_duplicates("source_id")


def _frames(root: Path, dataset: str, instrument: str | None = None, block: str = "history") -> pd.DataFrame:
    m = manifest(root)
    m = m[m["dataset"] == dataset]
    if instrument is not None:
        m = m[m["instrument"] == instrument]
    out = []
    for rp in sorted(m["raw_path"]):
        d = json.loads((Path(root) / rp).read_bytes())
        b = d.get(block)
        if b and b["data"]:
            out.append(pd.DataFrame(b["data"], columns=b["columns"]))
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


def _pit(df: pd.DataFrame, cutoff: str, col: str = "TRADEDATE") -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df[col] = pd.to_datetime(df[col])
    df = df[df[col] <= pd.Timestamp(cutoff)]
    assert (df[col] <= pd.Timestamp(cutoff)).all()
    return df


def _weekday(df: pd.DataFrame, col: str = "TRADEDATE") -> tuple[pd.DataFrame, int]:
    m = df[col].dt.dayofweek < 5
    return df[m], int((~m).sum())


def shares(root: Path, secid: str, cutoff: str, secid_is_calendar: bool = False) -> tuple[pd.DataFrame, dict]:
    df = _pit(_frames(root, "shares_history", secid), cutoff)
    if df.empty:
        return df, {"rows": 0}
    df = df.drop_duplicates(["TRADEDATE", "BOARDID"]).sort_values("TRADEDATE")
    df, n_weekend = _weekday(df)
    keep = ["TRADEDATE", "OPEN", "HIGH", "LOW", "CLOSE", "LEGALCLOSEPRICE", "WAPRICE", "VOLUME", "VALUE", "NUMTRADES"]
    df = df[[c for c in keep if c in df.columns]].set_index("TRADEDATE")
    # LEGALCLOSEPRICE is the exchange's official close, but ISS carries it forward on days WITHOUT trades
    # (e.g. the 2022-02-28..03-25 closure). A price counts only on days with at least one trade.
    df["PX"] = df["LEGALCLOSEPRICE"].where(df["LEGALCLOSEPRICE"] > 0, df["CLOSE"]) if "LEGALCLOSEPRICE" in df else df["CLOSE"]
    no_trade = df["NUMTRADES"].fillna(0) <= 0 if "NUMTRADES" in df else pd.Series(False, index=df.index)
    df.loc[no_trade, "PX"] = np.nan
    df = df[~no_trade] if secid_is_calendar else df
    return df, {"rows": len(df), "weekend_rows_dropped": n_weekend, "no_trade_rows_blanked": int(no_trade.sum())}


def dividends(root: Path, secid: str, cutoff: str) -> pd.DataFrame:
    df = _frames(root, "dividends", secid, block="dividends")
    if df.empty:
        return df
    df["registryclosedate"] = pd.to_datetime(df["registryclosedate"])
    return df[df["registryclosedate"] <= pd.Timestamp(cutoff)][["registryclosedate", "value", "currencyid"]]


def index(root: Path, secid: str, cutoff: str) -> tuple[pd.DataFrame, dict]:
    df = _pit(_frames(root, "index_history", secid), cutoff)
    if df.empty:
        return df, {"rows": 0}
    df = df.drop_duplicates("TRADEDATE").sort_values("TRADEDATE")
    df, n_weekend = _weekday(df)
    df = df.set_index("TRADEDATE")[["CLOSE", "VALUE"] + [c for c in ["DURATION", "YIELD"] if c in df]]
    df["PX"] = df["CLOSE"]
    return df, {"rows": len(df), "weekend_rows_dropped": n_weekend}


def currency(root: Path, secid: str, cutoff: str) -> tuple[pd.DataFrame, dict]:
    df = _pit(_frames(root, "currency_history", secid), cutoff)
    if df.empty:
        return df, {"rows": 0}
    df = df.drop_duplicates("TRADEDATE").sort_values("TRADEDATE")
    df, n_weekend = _weekday(df)
    px = "CLOSE" if "CLOSE" in df else "WAPRICE"
    df = df.set_index("TRADEDATE")
    df["PX"] = df[px].where(df[px] > 0, df.get("WAPRICE"))
    # the CETS history endpoint carries no traded-value column -> VALUE is DATA_NOT_AVAILABLE (NaN)
    df["VALUE"] = df["VOLRUR"] if "VOLRUR" in df else (df["VALUE"] if "VALUE" in df else np.nan)
    keep = ["PX", "VALUE"] + (["NUMTRADES"] if "NUMTRADES" in df else [])
    return df[keep], {"rows": len(df), "weekend_rows_dropped": n_weekend}


def futures_contracts(root: Path, cutoff: str) -> pd.DataFrame:
    """All futures daily rows (every contract) with contract terms joined."""
    df = _pit(_frames(root, "futures_history"), cutoff)
    if df.empty:
        return df
    df = df.drop_duplicates(["SECID", "TRADEDATE"]).sort_values(["SECID", "TRADEDATE"])
    df["weekend"] = df["TRADEDATE"].dt.dayofweek >= 5
    spec = futures_specs(root)
    return df.merge(spec, on="SECID", how="left")


def futures_specs(root: Path) -> pd.DataFrame:
    """Static contract terms from ISS securities/<secid>.json description blocks."""
    m = manifest(root)
    m = m[m["dataset"] == "futures_spec"]
    rows = []
    for _, r in m.iterrows():
        d = json.loads((Path(root) / r["raw_path"]).read_bytes())
        desc = d.get("description", {})
        if not desc.get("data"):
            continue
        kv = {row[0]: row[2] for row in desc["data"]}
        rows.append({
            "SECID": kv.get("SECID", r["instrument"]),
            "FRSTTRADE": kv.get("FRSTTRADE"),
            "LSTTRADE": kv.get("LSTTRADE"),
            "LSTDELDATE": kv.get("LSTDELDATE"),
            "LOTSIZE": kv.get("LOTSIZE"),
            "ASSETCODE_SPEC": kv.get("ASSETCODE"),
            "CONTRACTNAME": kv.get("CONTRACTNAME") or kv.get("NAME"),
            "EXECTYPE": kv.get("EXECTYPE"),
            "UNIT": kv.get("UNIT"),
            "DELIVERYTYPE": kv.get("DELIVERYTYPE"),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        for c in ("FRSTTRADE", "LSTTRADE", "LSTDELDATE"):
            out[c] = pd.to_datetime(out[c], errors="coerce")
        out["LOTSIZE"] = pd.to_numeric(out["LOTSIZE"], errors="coerce")
    return out


def bonds(root: Path, cutoff: str) -> pd.DataFrame:
    df = _pit(_frames(root, "bonds_history"), cutoff)
    if df.empty:
        return df
    df = df.drop_duplicates(["SECID", "TRADEDATE", "BOARDID"]).sort_values(["SECID", "TRADEDATE"])
    df = df[df["TRADEDATE"].dt.dayofweek < 5]
    keep = ["TRADEDATE", "SECID", "SHORTNAME", "BOARDID", "CLOSE", "LEGALCLOSEPRICE", "WAPRICE", "YIELDCLOSE",
            "YIELDATWAP", "DURATION", "ZSPREAD", "ZSPREADATWAPRICE", "VOLUME", "VALUE", "NUMTRADES", "MATDATE",
            "OFFERDATE", "FACEVALUE", "COUPONPERCENT", "ACCINT", "BONDTYPE"]
    return df[[c for c in keep if c in df]]


def zcyc(root: Path, cutoff: str) -> pd.DataFrame:
    """End-of-day G-curve yields at standard tenors (last intraday record of each sampled date)."""
    m = manifest(root)
    m = m[m["dataset"] == "zcyc"]
    rows = []
    for _, r in m.iterrows():
        d = json.loads((Path(root) / r["raw_path"]).read_bytes())
        yy = d.get("yearyields")
        if yy and yy["data"]:
            y = pd.DataFrame(yy["data"], columns=yy["columns"])
            y["tradedate"] = pd.to_datetime(y["tradedate"])
            y = y[y["tradedate"] <= pd.Timestamp(cutoff)]
            if y.empty:
                continue
            last_t = y["tradetime"].max()
            y = y[y["tradetime"] == last_t]
            for _, q in y.iterrows():
                rows.append({"date": q["tradedate"], "tenor_y": float(q["period"]), "yield_pct": float(q["value"])})
    out = pd.DataFrame(rows).drop_duplicates(["date", "tenor_y"])
    return out.pivot(index="date", columns="tenor_y", values="yield_pct").sort_index() if not out.empty else out


def options_snapshot(root: Path, date: str) -> pd.DataFrame:
    m = manifest(root)
    m = m[(m["dataset"] == "options_history") & (m["publication_date"] == date)]
    out = []
    for rp in sorted(m["raw_path"]):
        d = json.loads((Path(root) / rp).read_bytes())["history"]
        out.append(pd.DataFrame(d["data"], columns=d["columns"]))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, quoting=csv.QUOTE_MINIMAL, float_format="%.10g")


def rms_limits(root: Path, cutoff: str) -> pd.DataFrame:
    """NCC futures market-risk rates (mr1 = base IM rate per unit notional) by date and asset."""
    df = _frames(root, "rms_futures_limits", block="limits")
    if df.empty:
        return df
    df["tradedate"] = pd.to_datetime(df["tradedate"])
    df = df[df["tradedate"] <= pd.Timestamp(cutoff)]
    return df.drop_duplicates(["tradedate", "assetcode"]).sort_values(["assetcode", "tradedate"])
