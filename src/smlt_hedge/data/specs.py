"""Futures contract-specification audit (A4).

Point value (RUB per 1.0 of quoted price per contract) = STEPPRICE / MINSTEP from the MOEX FORTS board (static terms).
RUB notional of one contract = settlement price x point value.
Verification: the exchange-reported RUB traded VALUE must equal VOLUME x SETTLE x point value. For USD-linked contracts
(Brent, gold) the point value moves with USD/RUB every day, so the check uses a +-15% band and the exchange VALUE
is used for liquidity; for RUB-linked contracts the band is +-3%.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import load

FAMILIES = {  # research name -> (ASSETCODE on board, selector)
    "SMLT": ("SMLT", "dated"), "MIX": ("MIX", "dated"), "MXI": ("MXI", "dated"), "IMOEXF": ("IMOEXF", "secid"),
    "RGBI": ("RGBI", "dated"), "CNY": ("CNY", "dated"), "CNYRUBF": ("CNYRUBF", "secid"), "GOLD": ("GOLD", "dated"),
    "GLDRUBF": ("GLDRUBF", "secid"), "BR": ("BR", "dated"), "HOME": ("HOME", "dated"), "RUONIA": ("RUONIA", "dated"),
    "BTC": ("BTC", "dated"), "BTCUSDF": ("BTCUSDF", "secid"), "IBIT": ("IBIT", "dated"),
}
USD_LINKED = {"BR", "GOLD", "BTC", "BTCUSDF", "IBIT"}
# family -> interim continuous/perpetual table written by stage s2
INTERIM = {"SMLT": "SMLT_FUT", "MIX": "MIX_FUT", "MXI": "MXI_FUT", "IMOEXF": "IMOEXF", "RGBI": "RGBI_FUT", "CNY": "CNY_FUT",
           "CNYRUBF": "CNYRUBF", "GOLD": "GOLD_FUT", "GLDRUBF": "GLDRUBF", "BR": "BR_FUT", "HOME": "HOME_FUT",
           "BTC": "BTC_FUT", "BTCUSDF": "BTCUSDF", "IBIT": "IBIT_FUT"}


def _board(root: Path) -> pd.DataFrame:
    m = load.manifest(root)
    snap = m[m["dataset"] == "forts_board_snapshot"]["raw_path"].iloc[0]
    d = json.loads((Path(root) / snap).read_bytes())
    return pd.DataFrame(d["securities"]["data"], columns=d["securities"]["columns"])


def pit_terms(root: Path, cutoff: str) -> pd.DataFrame:
    """V3 Phase 0.6: point-in-time execution terms at the decision date for every family.

    Only genuinely STATIC terms come from the post-cutoff board snapshot: MINSTEP, the static RUB point value of
    RUB-quoted contracts (STEPPRICE/MINSTEP), and the exchange tariff expressed as a RATE (BUYSELLFEE / snapshot
    notional). Everything price-dependent is taken from the exchange history ON the decision date:
      front contract   = the contract held by the research continuous series at the cutoff (roll rule of s2);
                         the perpetual itself for perpetual contracts
      settle_pit       = SETTLEPRICE of that contract on the cutoff date
      point_value_pit  = OPENPOSITIONVALUE / (OPENPOSITION * SETTLEPRICE) on the cutoff date (exact exchange RUB
                         valuation; moves with USD/RUB for USD-quoted contracts); static STEPPRICE/MINSTEP if no OI
      notional_pit     = settle_pit * point_value_pit (RUB per contract)
    """
    board = _board(root)
    hist = load.futures_contracts(root, cutoff)
    t = pd.Timestamp(cutoff)
    hist = hist[hist["TRADEDATE"] <= t]
    rows = []
    interim = Path(root) / "data" / "interim"
    for fam, (code, sel) in FAMILIES.items():
        b = board[board["SECID"] == code] if sel == "secid" else board[board["ASSETCODE"] == code].sort_values("LASTTRADEDATE")
        f = interim / f"futures_{INTERIM.get(fam, fam)}.csv"
        if b.empty or not f.exists():
            rows.append({"family": fam, "status": "DATA_NOT_AVAILABLE"})
            continue
        tab = pd.read_csv(f, index_col=0, parse_dates=True)
        tab = tab[tab.index <= t]
        secid = code if sel == "secid" else (tab["held"].dropna().iloc[-1] if "held" in tab and tab["held"].notna().any() else None)
        h = hist[hist["SECID"] == secid].sort_values("TRADEDATE")
        h = h[h["SETTLEPRICE"].notna()]
        if h.empty:
            rows.append({"family": fam, "status": "NO_PIT_SETTLE", "secid": secid})
            continue
        last = h.iloc[-1]
        bb = b[b["SECID"] == secid]
        bb = bb.iloc[0] if len(bb) else b.iloc[0]
        pv_static = float(bb["STEPPRICE"]) / float(bb["MINSTEP"])
        oiv, oi, st = (float(last.get("OPENPOSITIONVALUE") or np.nan), float(last.get("OPENPOSITION") or np.nan),
                       float(last["SETTLEPRICE"]))
        pv_pit = oiv / (oi * st) if (np.isfinite(oiv) and np.isfinite(oi) and oi > 0 and st > 0) else np.nan
        pv = pv_pit if (np.isfinite(pv_pit) and fam in USD_LINKED) else pv_static
        snap_notional = float(bb["PREVSETTLEPRICE"]) * pv_static
        fee_rate = float(bb["BUYSELLFEE"]) / snap_notional if snap_notional > 0 else np.nan
        rows.append({"family": fam, "status": "OK", "secid": secid, "pit_date": str(last["TRADEDATE"].date()),
                     "settle_pit": st, "point_value_static": pv_static, "point_value_oi_implied": pv_pit,
                     "point_value_used": pv, "fx_linked": fam in USD_LINKED, "min_step": float(bb["MINSTEP"]),
                     "notional_pit": st * pv, "exchange_fee_rate": fee_rate,
                     "exchange_fee_rub_pit": fee_rate * st * pv, "half_tick_rub_pit": 0.5 * float(bb["MINSTEP"]) * pv,
                     "open_interest_pit": oi, "volume_pit": float(last.get("VOLUME") or np.nan),
                     "value_rub_pit": float(last.get("VALUE") or np.nan),
                     "source": "settle/OI/OI-value: MOEX ISS futures history on the decision date; MINSTEP, static RUB "
                               "point value and fee tariff rate: post-cutoff board snapshot (static terms only)"})
    return pd.DataFrame(rows)


def audit(root: Path, cutoff: str) -> pd.DataFrame:
    m = load.manifest(root)
    snap = m[m["dataset"] == "forts_board_snapshot"]["raw_path"].iloc[0]
    d = json.loads((Path(root) / snap).read_bytes())
    board = pd.DataFrame(d["securities"]["data"], columns=d["securities"]["columns"])
    hist = load.futures_contracts(root, cutoff)
    hist = hist[(hist["TRADEDATE"] >= pd.Timestamp(cutoff) - pd.Timedelta(days=120)) & (hist["VOLUME"] > 0)]
    specs = load.futures_specs(root).set_index("SECID")
    rows = []
    for fam, (code, sel) in FAMILIES.items():
        b = board[board["SECID"] == code] if sel == "secid" else board[board["ASSETCODE"] == code].sort_values("LASTTRADEDATE")
        if b.empty:
            rows.append({"instrument": fam, "verified": "NOT_ON_BOARD"})
            continue
        r = b.iloc[0]
        pv = float(r["STEPPRICE"]) / float(r["MINSTEP"])
        h = hist[hist["SECID"] == code] if sel == "secid" else hist[hist["ASSETCODE"] == code]
        ratio = (h["VALUE"] / (h["VOLUME"] * h["SETTLEPRICE"])).replace([np.inf, -np.inf], np.nan).dropna()
        med = float(ratio.median()) if len(ratio) else np.nan
        band = 0.15 if fam in USD_LINKED else 0.03
        ok = bool(np.isfinite(med) and abs(med / pv - 1) <= band)
        unit = specs["UNIT"].get(r["SECID"]) if r["SECID"] in specs.index else None
        rows.append({
            "instrument": fam, "contract": r["SECID"], "price_unit": unit, "contract_multiplier": pv,
            "point_value": pv, "step_price_rub": float(r["STEPPRICE"]), "min_step": float(r["MINSTEP"]),
            "lot": float(r["LOTVOLUME"]), "settlement_currency": "RUB",
            "fx_linked_point_value": fam in USD_LINKED,
            "notional_formula": "settle_price * STEPPRICE / MINSTEP (RUB per contract)",
            "notional_rub_at_snapshot": float(r["PREVSETTLEPRICE"]) * pv,   # audit only; never used in calculations (V3 0.6)
            "exchange_value_over_vol_settle_median_120d": med, "n_obs": int(len(ratio)),
            "v1_formula_error_factor": med,  # v1 used VOLUME*SETTLE -> understated RUB value by this factor
            "source": "MOEX ISS forts board snapshot (static terms) + futures history VALUE/VOLUME/SETTLEPRICE",
            "verified": "YES" if ok else "CHECK",
        })
    return pd.DataFrame(rows)
