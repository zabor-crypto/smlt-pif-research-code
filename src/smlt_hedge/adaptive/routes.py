"""Gate R2 — fair instrument economics: one executable route per exposure and direction, historical RUB return
streams built from the EXECUTED instrument, reconciliation to total-return benchmarks, route parity.

Route selection is rule-based and pre-declared (configs/adaptive.yaml r2.route_rules); no route is chosen by
historical performance.

Return conventions (daily, MOEX trading calendar, close to close):
  funded   r_t = P_t / P_{t-1} - 1 on the traded price (funds are total-return by construction; share dividends are
           added on the ex-date by the legacy TR series)
  futures  r_t = (F^{held since t-1}_t - F_{t-1}) * pv_t / (F_{t-1} * pv_{t-1})  (VM per RUB of yesterday's notional:
           USD-linked contracts carry the as-of RUB point value; perpetuals subtract the day's swap)
  excess   funded: r_t - r^MMF_t ; futures + collateral: r_t (the collateral earns the MMF return, which cancels)
BTC_RUB_EXPERIMENTAL = BTCUSDT close of the Binance hour bar ending 15:00 UTC (last complete bar at or before the
15:50 UTC decision time) x official USD/RUB known at the close (fx.usd_known). Never x CNY/RUB.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import fx
from .evidence import MANIFEST

ROUTE_COLUMNS = ["route", "exposure", "kind", "direction", "role", "instrument", "benchmark", "board_or_family",
                 "availability_from", "settlement", "lot", "cost_one_way", "fee_model", "carry_components",
                 "margin", "fx", "evidence", "notes"]


# ---------------------------------------------------------------------------------------------- loaders
def adaptive_frames(root: Path, dataset: str, instrument: str, block: str = "history") -> pd.DataFrame:
    m = pd.read_csv(Path(root) / "data" / MANIFEST, dtype=str)
    m = m[(m["dataset"] == dataset) & (m["instrument"] == instrument) & (m["validation_status"] == "VALID")]
    out = []
    for rp in sorted(m["raw_path"]):
        d = json.loads((Path(root) / rp).read_bytes())
        b = d.get(block)
        if b and b["data"]:
            out.append(pd.DataFrame(b["data"], columns=b["columns"]))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def fund_prices(root: Path, secid: str, cutoff: str) -> pd.DataFrame:
    """Exchange-fund daily history from the adaptive fetch (TQTF until 2026-06-19, TQBR after the board migration).
    Price counts only on days with trades (LEGALCLOSEPRICE, else CLOSE)."""
    f = adaptive_frames(root, "shares_history", secid)
    f["TRADEDATE"] = pd.to_datetime(f["TRADEDATE"])
    f = f[(f["TRADEDATE"] <= pd.Timestamp(cutoff)) & (f["TRADEDATE"].dt.dayofweek < 5)]
    f = f[f["NUMTRADES"].fillna(0) > 0].sort_values(["TRADEDATE", "VALUE"]).drop_duplicates("TRADEDATE", keep="last")
    px = f["LEGALCLOSEPRICE"].where(f["LEGALCLOSEPRICE"] > 0, f["CLOSE"])
    return pd.DataFrame({"PX": px.values, "VALUE": f["VALUE"].values, "BOARD": f["BOARDID"].values},
                        index=pd.DatetimeIndex(f["TRADEDATE"].values, name="date"))


def btc_usd_close(root: Path, cutoff: str, cal: pd.DatetimeIndex, hour_utc: int) -> pd.Series:
    from ..data import crypto
    bh = crypto.binance_hourly(root, cutoff)
    return crypto.synced_close(bh, cal, hour_utc)


def btcrub_direct(root: Path, cal: pd.DatetimeIndex, hour_utc: int) -> pd.Series:
    """Binance BTCRUB (delisted 2024-01) on the same clock — cross-check only."""
    import io
    import zipfile
    from ..data import crypto
    m = pd.read_csv(Path(root) / "data" / MANIFEST, dtype=str)
    m = m[(m["dataset"] == "btcrub_klines") & (m["validation_status"] == "VALID")]
    frames = []
    for rp in sorted(m["raw_path"]):
        z = zipfile.ZipFile(io.BytesIO((Path(root) / rp).read_bytes()))
        for n in z.namelist():
            d = pd.read_csv(z.open(n), header=None, names=crypto.KL_COLS)
            if not str(d.iloc[0, 0]).isdigit():
                d = d.iloc[1:]
            frames.append(d)
    if not frames:
        return pd.Series(dtype=float)
    d = pd.concat(frames, ignore_index=True)
    d["open_time"] = crypto._ts(d["open_time"])
    d["close"] = d["close"].astype(float)
    d = d.drop_duplicates("open_time").sort_values("open_time").set_index("open_time")[["close"]]
    return crypto.synced_close(d.assign(open=d["close"], high=d["close"], low=d["close"], volume=0.0), cal, hour_utc)


# ---------------------------------------------------------------------------------------------- futures streams
def futures_stream(ctx, route: str, cal: pd.DatetimeIndex, usd: pd.Series | None) -> pd.DataFrame:
    """Daily VM return per RUB of prior-day notional of the held contract, with as-of terms."""
    from . import market as M
    h = M.deriv_history(ctx, route, cal, usd, float(ctx.cfg["costs"]["broker_fee_futures_rub_per_contract"]))
    pv_prev = h["pv"].shift(1)
    Fn_prev = h["Fn"].shift(1)
    r = (h["Fp"] - Fn_prev) * h["pv"] / (Fn_prev * pv_prev)
    # frozen-FX variant (legacy V4 economics) for the reconciliation of USD-linked contracts
    r_frozen = (h["Fp"] - Fn_prev) / Fn_prev
    out = pd.DataFrame({"ret": r, "ret_price_only": r_frozen, "held": h["held"], "notional": h["notional"],
                        "mr1": h["mr1"], "fee": h["fee"], "is_roll": h["is_roll"], "pv": h["pv"],
                        "swap_pts": h["swap"], "Fn": h["Fn"], "Fp": h["Fp"]}, index=cal)
    return out


# ---------------------------------------------------------------------------------------------- panel
def build_panel(ctx, cfg: dict) -> dict:
    """Historical daily RUB return streams of every route, benchmarks and the money-market path (cutoff-bounded)."""
    cutoff = cfg["cutoff"]
    px = pd.read_csv(ctx.proc / "prices.csv", index_col=0, parse_dates=True)
    rt = pd.read_csv(ctx.proc / "returns.csv", index_col=0, parse_dates=True)
    cal = px.index[px.index <= pd.Timestamp(cutoff)]
    rec = fx.cbr_usd_records(ctx.root)
    usd = fx.usd_known(rec, cal, 1)
    usd_lag0 = fx.usd_known(rec, cal, 0)
    rr = cfg["r2"]
    R, B, meta = {}, {}, {}
    # --- funded
    R["SMLT"] = rt["SMLT"]
    for route, secid in (("EQ_FUNDED", rr["funds"]["broad_equity"]), ("OFZ_FUNDED", rr["funds"]["ofz"])):
        f = fund_prices(ctx.root, secid, cutoff).reindex(cal)
        R[route] = f["PX"].pct_change(fill_method=None)
        meta[route] = {"adv_median": float(f["VALUE"].loc["2023-01-03":].median())}
    R["GOLD_FUNDED"] = rt[rr["funds"]["gold"]]
    R["CNY_FUNDED"] = rt["CNYRUB_TOM"]
    R["MMF"] = rt["LQDT"]
    # BTC experimental (owner-authorised academic funded sleeve)
    btc = btc_usd_close(ctx.root, cutoff, cal, rr["btc"]["bar_close_hour_utc"])
    btc_rub = btc * usd
    R["BTC_RUB_EXPERIMENTAL"] = btc_rub.pct_change(fill_method=None)
    B["BTC_USD"] = btc.pct_change(fill_method=None)
    B["USDRUB_CBR"] = usd.pct_change(fill_method=None)
    B["BTC_RUB_LEVEL"] = btc_rub
    B["BTC_RUB_USD_LAG0"] = (btc * usd_lag0).pct_change(fill_method=None)
    # --- derivatives
    fut = {}
    for route, fam in (("SMLT_FUT", "SMLT_FUT"), ("EQ_HEDGE_FUT", rr["futures"]["equity_hedge"]),
                       ("RATES_HEDGE_FUT", "RGBI_FUT"), ("OIL_FUT", "BR_FUT"), ("BTC_MOEX_FUT", "BTC_FUT"),
                       ("GOLD_PERP", "GLDRUBF"), ("CNY_PERP", "CNYRUBF")):
        s = futures_stream(ctx, fam, cal, usd)
        fut[route] = s
        R[route] = s["ret"]
    # --- benchmarks
    for b in ("MCFTR", "IMOEX", "RGBITR", "RGBI", "GLDRUB_TOM", "CNYRUB_TOM", "SMLT"):
        B[b] = rt[b] if b in rt else px[b].pct_change(fill_method=None)
    ru = pd.read_csv(ctx.interim / "cbr_ruonia.csv", parse_dates=["date", "publication_date"])
    ru = ru[ru["publication_date"] <= pd.Timestamp(cutoff)].set_index("date")["ruonia_pct"].sort_index()
    rate = ru.reindex(ru.index.union(cal)).ffill().reindex(cal)            # rate of the last fixing date <= t
    gap = pd.Series(cal, index=cal).diff().dt.days
    B["RUONIA_ACCRUAL"] = rate.shift(1) / 100 * gap / 365                    # accrues the t-1 fixing over (t-1, t]
    # price-only USD-linked futures returns and USD/RUB for forward simulation (engine applies the RUB point value)
    B["OIL_FUT_USD"] = fut["OIL_FUT"]["ret_price_only"]
    B["BTC_MOEX_FUT_USD"] = fut["BTC_MOEX_FUT"]["ret_price_only"]
    Rdf = pd.DataFrame(R).reindex(cal)
    Rdf.loc[Rdf.index[0]] = np.nan
    return {"R": Rdf, "B": pd.DataFrame(B).reindex(cal), "fut": fut, "usd": usd, "usd_lag0": usd_lag0,
            "cal": cal, "meta": meta, "btc_rub_direct": btcrub_direct(ctx.root, cal, rr["btc"]["bar_close_hour_utc"])}


# ---------------------------------------------------------------------------------------------- reconciliation
def _ann(x: pd.Series) -> float:
    x = x.dropna()
    return float((1 + x).prod() ** (252 / len(x)) - 1) if len(x) else np.nan


def _stats(a: pd.Series, b: pd.Series, label: str, mechanics: str, window=("2023-01-03", None)) -> dict:
    d = pd.concat([a, b], axis=1).loc[window[0]:window[1]].dropna()
    if d.empty:
        return {"comparison": label, "status": "NO_OVERLAP"}
    x, y = d.iloc[:, 0], d.iloc[:, 1]
    wk_x = (1 + x).resample("W-FRI").prod() - 1
    wk_y = (1 + y).resample("W-FRI").prod() - 1
    gap = x - y
    n_years = len(d) / 252
    return {"comparison": label, "from": str(d.index.min().date()), "to": str(d.index.max().date()), "days": len(d),
            "ann_return_route": _ann(x), "ann_return_benchmark": _ann(y), "ann_gap": _ann(x) - _ann(y),
            "mean_daily_gap_ann": float(gap.mean() * 252), "se_mean_gap_ann": float(gap.std(ddof=1) * np.sqrt(252) / np.sqrt(n_years)),
            "tracking_error_ann": float(gap.std(ddof=1) * np.sqrt(252)), "corr_daily": float(x.corr(y)),
            "corr_weekly": float(wk_x.corr(wk_y)), "mechanics_in_route": mechanics}


def _br_vm_row(f: pd.DataFrame) -> dict:
    """D4-02: RUB VM of one BR contract at the as-of exchange point value vs the cutoff-frozen point value."""
    d = f.loc["2023-01-03":].copy()
    dF = d["Fn"].diff().where(~d["is_roll"].astype(bool), np.nan)
    pv_frozen = float(d["pv"].iloc[-1])
    vm_asof = dF * d["pv"]
    vm_frozen = dF * pv_frozen
    notional_prev = (d["Fn"] * d["pv"]).shift(1)
    rel = ((vm_asof - vm_frozen) / notional_prev).dropna()
    return {"comparison": "BR 1-contract RUB VM: as-of exchange point value vs cutoff-frozen point value (D4-02)",
            "from": str(d.index.min().date()), "to": str(d.index.max().date()), "days": int(rel.size),
            "tracking_error_ann": float(rel.std(ddof=1) * np.sqrt(252)),
            "mean_abs_point_value_error": float((d["pv"] / pv_frozen - 1).abs().mean()),
            "max_abs_point_value_error": float((d["pv"] / pv_frozen - 1).abs().max()),
            "cum_vm_asof_rub_per_contract": float(vm_asof.sum()), "cum_vm_frozen_rub_per_contract": float(vm_frozen.sum()),
            "mechanics_in_route": "engine VM uses the day's RUB point value (exchange-implied; CBR fallback)"}


def reconcile(P: dict, cfg: dict) -> pd.DataFrame:
    R, B = P["R"], P["B"]
    mm = R["MMF"]
    rows = [
        _stats(R["SMLT_FUT"] + mm, R["SMLT"], "SMLT_FUT + collateral(MMF) vs SMLT funded (TR)",
               "held-contract settles, roll at listed calendar, as-of notional; collateral earns MMF"),
        _stats(R["EQ_FUNDED"], B["MCFTR"], f"EQ_FUNDED ({cfg['r2']['funds']['broad_equity']}) vs MCFTR",
               "fund traded price (dividends reinvested inside the fund, TER embedded)"),
        _stats(R["EQ_HEDGE_FUT"] + mm, B["MCFTR"], "EQ_HEDGE_FUT (mini-MIX) + collateral vs MCFTR",
               "price-index futures: dividend expectation and financing are in the basis"),
        _stats(R["EQ_HEDGE_FUT"], B["IMOEX"], "EQ_HEDGE_FUT price return vs IMOEX price index (NOT a TR comparison)",
               "diagnostic: shows why MIX price futures must not be equated with funded TR"),
        _stats(R["OFZ_FUNDED"], B["RGBITR"], f"OFZ_FUNDED ({cfg['r2']['funds']['ofz']}) vs RGBITR",
               "fund NAV-based price: coupons, accrued interest and roll-down inside the fund"),
        _stats(R["RATES_HEDGE_FUT"] + mm, B["RGBITR"], "RATES_HEDGE_FUT (RGBI) + collateral vs RGBITR",
               "price-index futures: coupon carry and financing are in the basis"),
        _stats(R["GOLD_FUNDED"], B["GLDRUB_TOM"], f"GOLD_FUNDED ({cfg['r2']['funds']['gold']}) vs GLDRUB_TOM spot",
               "fund traded price, TER embedded"),
        _stats(R["GOLD_PERP"] + mm, B["GLDRUB_TOM"], "GLDRUBF perpetual (net of daily swap) + collateral vs GLDRUB_TOM spot",
               "daily swap transferred via VM (D4-01)"),
        _stats(R["CNY_PERP"] + mm, R["CNY_FUNDED"], "CNYRUBF perpetual (net of swap) + collateral vs CNYRUB_TOM funded",
               "daily swap transferred via VM (D4-01); funded CNY earns 0% CNY interest (main case)"),
        _br_vm_row(P["fut"]["OIL_FUT"]),
        _stats(R["BTC_RUB_EXPERIMENTAL"], P["btc_rub_direct"].pct_change(fill_method=None),
               "BTC_RUB_EXPERIMENTAL (BTCUSDT x CBR USD/RUB) vs direct Binance BTCRUB (same clock)",
               "construction check on the overlap 2022-2024", window=("2022-03-28", None)),
        _stats(R["BTC_RUB_EXPERIMENTAL"], B["BTC_RUB_USD_LAG0"], "BTC_RUB_EXPERIMENTAL vs same with USD/RUB value date <= t (strict lag)",
               "timing sensitivity of the FX leg"),
        _stats(R["BTC_MOEX_FUT"] + mm, R["BTC_RUB_EXPERIMENTAL"], "MOEX BTC futures + collateral vs BTC_RUB_EXPERIMENTAL",
               "separate real derivative route; never merged with the funded sleeve", window=("2025-11-18", None)),
        _stats(R["MMF"], B["RUONIA_ACCRUAL"] if "RUONIA_ACCRUAL" in B else pd.Series(dtype=float),
               "MMF (LQDT) vs RUONIA accrual", "fund price (fee-net)"),
    ]
    return pd.DataFrame(rows)


def usd_checks(P: dict, fut_br: pd.DataFrame, usd_tom: pd.Series | None) -> pd.DataFrame:
    """USD/RUB timing: CBR record dated t+1 vs exchange-implied BR point value / 10 USD, and vs MOEX USD TOM."""
    imp = fut_br["pv"] / 10.0
    rows = []
    for nm, s in (("CBR value date <= t+1 (canonical)", P["usd"]), ("CBR value date <= t", P["usd_lag0"])):
        d = (imp / s - 1).dropna()
        rows.append({"comparison": f"FORTS BR implied USD/RUB vs {nm}", "days": len(d),
                     "median_abs_rel_diff": float(d.abs().median()),
                     "share_within_0.01pct_from_2024-07": float((d.loc["2024-07":].abs() < 1e-4).mean()),
                     "share_within_0.1pct_all": float((d.abs() < 1e-3).mean())})
    if usd_tom is not None and len(usd_tom):
        d = (usd_tom.reindex(P["cal"]) / P["usd"] - 1).dropna()
        rows.append({"comparison": "MOEX USD000UTSTOM close vs CBR (value date <= t+1)", "days": len(d),
                     "median_abs_rel_diff": float(d.abs().median()), "share_within_0.1pct_all": float((d.abs() < 1e-3).mean())})
    return pd.DataFrame(rows)
