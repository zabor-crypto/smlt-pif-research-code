"""Interim tables -> processed daily research panel (data/processed/).

Calendar: SMLT TQBR weekday trading days in [history_start, decision_date] (the "master calendar").
Alignment: each instrument's prices are reindexed to the master calendar WITHOUT filling; returns use
pct_change(fill_method=None), so a missing day yields NaN (never a stale zero return).

Equity total return: on the ex-dividend date the return is (P_t + D) / P_{t-1} - 1.
Ex-date rule (Russian settlement cycle): the first trading day d whose settlement date is after the
record date R; settlement is T+2 before 2023-07-31 and T+1 from 2023-07-31 (MOEX transition).

Perpetual futures (IMOEXF, CNYRUBF, GLDRUBF): long return = (S_t - S_{t-1} - SWAPRATE_t) / S_{t-1}
(the swap rate is the daily funding transfer from longs to shorts when positive).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import load
from ..strategies.futures import continuous

T1_SWITCH = pd.Timestamp("2023-07-31")

FUT_ASSETS = {  # research name -> (ASSETCODE, kind)
    "SMLT_FUT": ("SMLT", "dated"),
    "MIX_FUT": ("MIX", "dated"),
    "MXI_FUT": ("MXI", "dated"),
    "RGBI_FUT": ("RGBI", "dated"),
    "GOLD_FUT": ("GOLD", "dated"),
    "CNY_FUT": ("CNY", "dated"),
    "BR_FUT": ("BR", "dated"),
    "HOME_FUT": ("HOME", "dated"),
    "IMOEXF": ("IMOEXF", "perpetual"),
    "CNYRUBF": ("CNYRUBF", "perpetual"),
    "GLDRUBF": ("GLDRUBF", "perpetual"),
    # V3 Phase 1: MOEX Bitcoin instruments (EXECUTION history; the external BTC proxy is loaded separately)
    "BTC_FUT": ("BTC", "dated"),          # monthly futures on the MOEX Bitcoin Index (MOEXBTC)
    "BTCUSDF": ("BTCUSDF", "perpetual"),  # one-day auto-prolonged (perpetual) contract on MOEXBTC, first trade 2026-09-22
    "IBIT_FUT": ("IBIT", "dated"),        # futures on the iShares Bitcoin Trust ETF
}


def ex_dates(record_dates: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    cal = pd.DatetimeIndex(sorted(calendar))
    out = []
    for R in record_dates:
        lag = 2 if R < T1_SWITCH else 1
        ex = None
        for i in range(len(cal) - lag):
            if cal[i + lag] > R:
                ex = cal[i]
                break
        out.append(ex)
    return pd.Series(out, index=record_dates.index)


def total_return_prices(px: pd.Series, divs: pd.DataFrame, calendar: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """Returns (daily total return, dividend-per-share series on ex-dates)."""
    r = px.pct_change(fill_method=None)
    d = pd.Series(0.0, index=px.index)
    if divs is not None and not divs.empty:
        divs = divs[divs["currencyid"].isin(["RUB", "SUR"])].copy()
        divs["ex"] = ex_dates(divs["registryclosedate"], calendar)
        for _, row in divs.dropna(subset=["ex"]).iterrows():
            if row["ex"] in d.index:
                d.loc[row["ex"]] += float(row["value"])
    prev = px.shift(1)
    tr = (px + d) / prev - 1.0
    tr[r.isna()] = np.nan
    return tr, d


def build(root: Path, cfg: dict, src: dict) -> dict[str, pd.DataFrame]:
    cutoff = cfg["decision_date"]
    start = pd.Timestamp(cfg["history_start"])
    info = {}

    smlt, info["SMLT"] = load.shares(root, "SMLT", cutoff, secid_is_calendar=True)
    cal = pd.DatetimeIndex(smlt.index[(smlt.index >= start)]).sort_values()

    prices, rets, volume_rub, meta = {}, {}, {}, []
    # --- equities & funds (total return)
    for board, secs in {**src["shares"], **src["funds"]}.items():
        for s in secs:
            df, inf = load.shares(root, s, cutoff)
            info[s] = inf
            if df.empty:
                meta.append({"series": s, "source": f"TQ:{board}", "status": "DATA_NOT_AVAILABLE"})
                continue
            px = df["PX"].reindex(cal)
            divs = load.dividends(root, s, cutoff)
            tr, dps = total_return_prices(px, divs, cal)
            prices[s], rets[s] = px, tr
            volume_rub[s] = df["VALUE"].reindex(cal)
            meta.append({"series": s, "source": f"MOEX:{board}", "status": "OK",
                         "n_div_events": int((dps > 0).sum())})
            if s == "SMLT":
                prices["SMLT_DPS"] = dps
    # --- indices (price; TR indices are separate series)
    for s in src["indices"]:
        df, inf = load.index(root, s, cutoff)
        info[s] = inf
        if df.empty:
            meta.append({"series": s, "source": "MOEX:index", "status": "DATA_NOT_AVAILABLE"})
            continue
        prices[s] = df["PX"].reindex(cal)
        # rate indices (RUSFAR etc.) are LEVELS in %, not prices: store level, no return
        if s.startswith(("RUSFAR", "MOEXREPO", "RUONIA", "RVI")):
            meta.append({"series": s, "source": "MOEX:index", "status": "OK_LEVEL"})
            continue
        rets[s] = prices[s].pct_change(fill_method=None)
        volume_rub[s] = df["VALUE"].reindex(cal) if "VALUE" in df else np.nan
        meta.append({"series": s, "source": "MOEX:index", "status": "OK"})
    # --- currency market
    for board, secs in src["currency"].items():
        for s in secs:
            df, inf = load.currency(root, s, cutoff)
            info[s] = inf
            if df.empty:
                meta.append({"series": s, "source": "MOEX:CETS", "status": "DATA_NOT_AVAILABLE"})
                continue
            prices[s] = df["PX"].reindex(cal)
            rets[s] = prices[s].pct_change(fill_method=None)
            volume_rub[s] = df["VALUE"].reindex(cal)
            meta.append({"series": s, "source": "MOEX:CETS", "status": "OK", "note": "no traded value in ISS CETS history"})

    # --- futures
    fut_all = load.futures_contracts(root, cutoff)
    fut_tables = {}
    for name, (asset, kind) in FUT_ASSETS.items():
        # perpetual contracts carry the UNDERLYING as ASSETCODE (e.g. IMOEXF -> IMOEX): select them by SECID
        key = "SECID" if kind == "perpetual" else "ASSETCODE"
        rows = fut_all[(fut_all[key] == asset)] if not fut_all.empty else fut_all
        if rows is None or rows.empty:
            meta.append({"series": name, "source": "MOEX:FORTS", "status": "DATA_NOT_AVAILABLE"})
            continue
        rows = rows[rows["TRADEDATE"] >= start]
        if kind == "dated":
            c = continuous(rows, cal, roll_days=cfg.get("futures", {}).get("roll_days", 5))
            fut_tables[name] = c
            rets[name] = c["ret"].reindex(cal)
            prices[name] = c["F_held"].reindex(cal)
            volume_rub[name] = c["value_rub_held"].reindex(cal)   # exchange RUB VALUE (A4 repair)
        else:
            r = rows[~rows["TRADEDATE"].dt.dayofweek.ge(5)].set_index("TRADEDATE").sort_index()
            s_ = r["SETTLEPRICE"].reindex(cal)
            sw = r["SWAPRATE"].reindex(cal).fillna(0.0) if "SWAPRATE" in r else 0.0
            rets[name] = (s_ - s_.shift(1) - sw) / s_.shift(1)
            prices[name] = s_
            volume_rub[name] = r["VALUE"].reindex(cal)             # exchange RUB VALUE (A4 repair)
            fut_tables[name] = r
        meta.append({"series": name, "source": "MOEX:FORTS", "status": "OK", "kind": kind})

    # --- V3: external BTC economic proxy (PROXY_HISTORY, synchronised to the MOEX close; data/crypto.py)
    from . import crypto
    ccfg = src.get("crypto", {})
    bh = crypto.binance_hourly(root, cutoff)
    if not bh.empty:
        hr = int(ccfg.get("sync_hour_utc", 16))
        for name, px in (("BTC_PROXY", crypto.synced_close(bh, cal, hr)), ("BTC_PROXY_UTC0", crypto.utc_day_close(bh, cal))):
            prices[name], rets[name] = px, px.pct_change(fill_method=None)
        meta.append({"series": "BTC_PROXY", "source": "BINANCE:BTCUSDT 1h", "status": "PROXY_HISTORY",
                     "note": f"close of the hour bar ending {hr}:00 UTC on each MOEX day; USD; not MOEX execution history"})
        meta.append({"series": "BTC_PROXY_UTC0", "source": "BINANCE:BTCUSDT 1h", "status": "PROXY_HISTORY",
                     "note": "00:00 UTC daily close (non-synchronous alternative)"})
    if "BTC_PROXY" in rets and "CNYRUB_TOM" in rets:
        # V4: RUB return of the INSTRUCTOR-AUTHORISED EXPERIMENTAL spot BTC holding. USD/RUB is not traded on MOEX since
        # 2024-06, so the RUB translation uses CNY/RUB (a managed-currency proxy for USD/RUB) - documented approximation.
        rets["BTC_SPOT_RUB"] = (1 + rets["BTC_PROXY"]) * (1 + rets["CNYRUB_TOM"]) - 1
        prices["BTC_SPOT_RUB"] = prices["BTC_PROXY"] * prices["CNYRUB_TOM"]  # level index consistent with the returns (units: USD x RUB/CNY, not RUB per BTC)
        meta.append({"series": "BTC_SPOT_RUB", "source": "BTC_PROXY x CNYRUB_TOM", "status": "PROXY_HISTORY",
                     "note": "INSTRUCTOR_AUTHORIZED_EXPERIMENTAL spot BTC in RUB; FX via CNY/RUB proxy"})
    cb = crypto.coinbase_daily(root, cutoff)
    if not cb.empty:
        prices["BTC_COINBASE_UTC0"] = cb.reindex(cal)
        rets["BTC_COINBASE_UTC0"] = prices["BTC_COINBASE_UTC0"].pct_change(fill_method=None)
        meta.append({"series": "BTC_COINBASE_UTC0", "source": "COINBASE:BTC-USD 1d", "status": "PROXY_HISTORY",
                     "note": "cross-check venue, 00:00 UTC daily close"})

    P = pd.DataFrame(prices).reindex(cal)
    R = pd.DataFrame(rets).reindex(cal)
    # --- explicit exclusions (configs/corporate_actions.yaml)
    ca = yaml.safe_load((Path(root) / "configs" / "corporate_actions.yaml").read_text())
    excluded = []
    for e in ca.get("exclude_returns", []):
        d = pd.Timestamp(e["date"])
        if e["series"] in R and d in R.index:
            R.loc[d, e["series"]] = np.nan
            excluded.append({**e, "kind": "corporate_action"})
    gap = pd.Series(cal, index=cal).diff().dt.days
    closure = gap[gap > ca["market_closures"]["max_gap_calendar_days"]].index
    for d in closure:
        R.loc[d, :] = np.nan
        excluded.append({"series": "ALL", "date": str(d.date()), "reason": f"return spans {int(gap[d])} calendar days (market closure)",
                         "kind": "market_closure"})
    meta = meta + [{"series": x["series"], "source": "exclusion", "status": f"{x['kind']}:{x['date']}"} for x in excluded]
    V = pd.DataFrame(volume_rub).reindex(cal)
    R.index.name = P.index.name = V.index.name = "date"
    assert R.index.max() <= pd.Timestamp(cutoff), "look-ahead: data after decision date"
    return {"prices": P, "returns": R, "value_rub": V, "meta": pd.DataFrame(meta), "futures": fut_tables,
            "info": info, "calendar": cal}


def credit_state(root: Path, cutoff: str, cal: pd.DatetimeIndex, min_value_rub: float = 1e6) -> pd.DataFrame:
    """Daily SMLT credit-stress state from exchange bond data.

    For each day: median Z-spread (ISS ZSPREAD, percentage points) and median YTM across Samolet bonds that traded with value
    >= min_value_rub and have > 90 days to maturity. Also a price-based measure (median clean price, % par).
    """
    b = load.bonds(root, cutoff)
    if b.empty:
        return pd.DataFrame(index=cal)
    b = b.copy()
    b["MATDATE"] = pd.to_datetime(b["MATDATE"], errors="coerce")
    b = b[(b["VALUE"].fillna(0) >= min_value_rub) & ((b["MATDATE"] - b["TRADEDATE"]).dt.days > 90)]
    px = b["CLOSE"].where(b["CLOSE"].notna(), b["LEGALCLOSEPRICE"])
    b["px"] = px
    # constant-composition credit FACTOR: median of per-bond day-on-day Z-spread changes (pp) and price returns,
    # using only bonds that traded (above the value filter) on BOTH days -> no composition jumps.
    b = b.sort_values(["SECID", "TRADEDATE"])
    b["prev_date"] = b.groupby("SECID")["TRADEDATE"].shift(1)
    b["dz"] = b.groupby("SECID")["ZSPREAD"].diff()
    b["dpx"] = b.groupby("SECID")["px"].pct_change(fill_method=None)
    cal_pos = pd.Series(np.arange(len(cal)), index=cal)
    consecutive = (b["TRADEDATE"].map(cal_pos) - b["prev_date"].map(cal_pos)) == 1
    b.loc[~consecutive, ["dz", "dpx"]] = np.nan
    g = b.groupby("TRADEDATE")
    out = pd.DataFrame({
        "dz_median_pp": g["dz"].median(),
        "bond_price_ret_median": g["dpx"].median(),
        "bond_duration_median_days": g["DURATION"].median(),
        "zspread_median_pp": g["ZSPREAD"].median(),
        "ytm_median_pct": g["YIELDCLOSE"].median(),
        "price_median_pct_par": g["px"].median(),
        "n_bonds": g["SECID"].nunique(),
    }).reindex(cal)
    rg, _ = load.index(root, "RGBI", cutoff)
    if "DURATION" in rg:
        out["rgbi_duration_days"] = rg["DURATION"].reindex(cal)
    return out
