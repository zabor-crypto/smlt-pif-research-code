"""Adaptive redesign R1 — market inputs for the quantity engine.

`forward_market`  : scenario returns -> Market (admission authority). Contract terms are those known at the decision
                    date (last NCC mr1, last settle, RUB point value, fee tariff); rolls at the listed expiry schedule;
                    USD-linked point values move with the simulated USD/RUB path.
`historical_market`: exchange history -> Market for one path (replay). Every term is AS OF the date:
                    contract identity and roll (continuous-series roll rule: 5 sessions before LSTTRADE), settle of the
                    held contract, NCC mr1 on the date (last published value carried forward — causal), RUB point value
                    (exchange-implied OPENPOSITIONVALUE / (OI x SETTLE) for USD-linked contracts, CBR fallback), fee =
                    cutoff tariff RATE x as-of notional + broker fee + half tick (the tariff history is not archived;
                    D3-07 showed the RUB fee must not be the cutoff RUB amount), settlement lag T+2 / T+1 by date.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import engine as E


# ---------------------------------------------------------------------------------------------- forward
@dataclass
class DerivSpec:
    name: str
    F0: float                 # held-contract settle at the decision date (price points)
    pv0: float                # RUB per point at the decision date
    usd_linked: bool
    mr: float                 # NCC mr1 at the decision date
    fee_rate: float           # exchange tariff as a fraction of notional
    fee_fixed_rub: float      # broker fee + half tick (RUB per contract per side) at the decision date
    roll_days: tuple = ()     # engine days (1..H) on which the position is rolled
    roll_gap: float = 0.0     # log gap next/front at the decision date (term structure), applied at each roll
    execution_info: dict | None = None  # V2R2 decision-known conditional authority


@dataclass
class FundedSpec:
    name: str
    kind: str                 # FUNDED | MMF
    P0: float
    lag: int
    lot: float = 1.0
    cost_prop: float = 0.0
    cost_unit: float = 0.0
    issuer: str = ""
    factor: str = ""
    execution_capacity: float | None = None
    execution_key: str | None = None
    execution_dates: object = None


def forward_market(fspecs: list[FundedSpec], dspecs: list[DerivSpec], Rf: np.ndarray, Rd: np.ndarray,
                   Rfx: np.ndarray | None, mmf_daily: np.ndarray, cash_rate: float = 0.0,
                   cny_swap_frac: np.ndarray | None = None) -> tuple:
    """Rf (N, H, Af) daily funded returns (MMF column ignored: priced from `mmf_daily`), Rd (N, H, Ad) daily futures
    price returns (CNY: NET VM factor), cny_swap_frac (N,H) raw contractual auxiliary, Rfx (N, H) daily USD/RUB returns for USD-linked point values."""
    N, H = (Rf.shape[0], Rf.shape[1]) if Rf.size else (Rd.shape[0], Rd.shape[1])
    T = H + 1
    Af, Ad = len(fspecs), len(dspecs)
    P = np.empty((N, T, Af))
    for j, f in enumerate(fspecs):
        if f.kind == E.MMF:
            P[:, :, j] = f.P0 * np.concatenate([[1.0], np.cumprod(1 + np.asarray(mmf_daily, float))])[None, :]
        else:
            P[:, 0, j] = f.P0
            P[:, 1:, j] = f.P0 * np.cumprod(1 + Rf[:, :, j], axis=1)
    Fp = np.empty((N, T, Ad))
    Fn = np.empty((N, T, Ad))
    pv = np.empty((N, T, Ad))
    fee = np.empty((N, T, Ad))
    roll = np.zeros((T, Ad), bool)
    fx = np.ones((N, T)) if Rfx is None else np.concatenate([np.ones((N, 1)), np.cumprod(1 + Rfx, axis=1)], axis=1)
    for j, d in enumerate(dspecs):
        perpetual = d.name == "CNY_PERP"
        if perpetual:
            if cny_swap_frac is None:
                raise ValueError("CNY NET alone cannot identify settlement: supply SWAP_FRAC explicitly")
            swap = np.asarray(cny_swap_frac, float)
            if swap.shape != (N, H) or not np.isfinite(swap).all():
                raise ValueError("invalid CNY SWAP_FRAC shape or values")
            if d.roll_days:
                raise ValueError("CNY perpetual has no roll")
        F = np.empty((N, T))
        Fprev = np.full(N, d.F0)
        F[:, 0] = d.F0
        Fp[:, 0, j] = d.F0
        for t in range(1, T):
            # NET is the daily VM factor; SWAP_FRAC transports the next settlement separately.
            f_old = Fprev * (1 + Rd[:, t - 1, j] + (swap[:, t - 1] if perpetual else 0.0))
            Fp[:, t, j] = f_old - (swap[:, t - 1] * Fprev if perpetual else 0.0)
            if t in d.roll_days:
                roll[t, j] = True
                f_new = f_old * np.exp(d.roll_gap)
            else:
                f_new = f_old
            F[:, t] = f_new
            Fprev = f_new
        Fn[:, :, j] = F
        pv[:, :, j] = d.pv0 * (fx if d.usd_linked else 1.0)
        fee[:, :, j] = d.fee_rate * Fn[:, :, j] * pv[:, :, j] + d.fee_fixed_rub
    mkt = E.Market(P=P, Fp=Fp, Fn=Fn, pv=pv, mr=np.tile([d.mr for d in dspecs], (T, 1)) if Ad else np.zeros((T, 0)),
                   fee=fee, roll=roll, lag=np.tile([f.lag for f in fspecs], (T, 1)).astype(int),
                   tradable_f=np.ones((T, Af), bool), tradable_d=np.ones((T, Ad), bool),
                   cash_rate=np.full(T, cash_rate))
    if any(f.execution_capacity is not None for f in fspecs) or any(d.execution_info is not None for d in dspecs):
        from ..regime_v2.execution_inputs import forward
        mkt.execution = forward(fspecs, dspecs, mkt)
    fin = [E.Instrument(f.name, f.kind, f.lot, f.cost_prop, f.cost_unit, f.issuer, f.factor) for f in fspecs]
    din = [E.Instrument(d.name, E.DERIV) for d in dspecs]
    return fin, din, mkt


# ---------------------------------------------------------------------------------------------- history
FAMILY = {  # route -> (interim continuous table, NCC asset code, usd_linked, perpetual)
    "SMLT_FUT": ("SMLT_FUT", "SMLT", False, False),
    "MIX_FUT": ("MIX_FUT", "MIX", False, False),
    "MXI_FUT": ("MXI_FUT", "MXI", False, False),
    "RGBI_FUT": ("RGBI_FUT", "RGBI", False, False),
    "BR_FUT": ("BR_FUT", "BR", True, False),
    "BTC_FUT": ("BTC_FUT", "BTC", True, False),
    "GLDRUBF": ("GLDRUBF", "GLDRUBTOM", False, True),
    "CNYRUBF": ("CNYRUBF", "CNYRUBTOM", False, True),
}
PIT_FAMILY = {"SMLT_FUT": "SMLT", "MIX_FUT": "MIX", "MXI_FUT": "MXI", "RGBI_FUT": "RGBI", "BR_FUT": "BR", "BTC_FUT": "BTC",
              "GLDRUBF": "GLDRUBF", "CNYRUBF": "CNYRUBF"}


def settlement_lag(dates: pd.DatetimeIndex, cfg: dict) -> np.ndarray:
    s = cfg["r1"]["settlement"]
    return np.where(dates >= pd.Timestamp(s["equity_t1_from"]), s["lag_after"], s["lag_before"]).astype(int)


def deriv_history(ctx, route: str, dates: pd.DatetimeIndex, usdrub: pd.Series | None, broker_fee_smlt: float) -> pd.DataFrame:
    """As-of per-date terms of the contract held by the continuous series (causal: every column uses data <= date)."""
    from ..data import load
    from ..data.specs import pit_terms
    table, ncc, usd, perp = FAMILY[route]
    if not hasattr(ctx, "_adaptive_pit"):
        ctx._adaptive_pit = pit_terms(ctx.root, ctx.cutoff).set_index("family")
        ctx._adaptive_fut = load.futures_contracts(ctx.root, ctx.cutoff)
        ctx._adaptive_mr = pd.read_csv(ctx.proc / "im_rate_mr1.csv", index_col=0, parse_dates=True)
    pit = ctx._adaptive_pit.loc[PIT_FAMILY[route]]
    allc = ctx._adaptive_fut
    mr = ctx._adaptive_mr[ncc].reindex(ctx._adaptive_mr.index.union(dates)).ffill().reindex(dates)
    if perp:
        r = allc[(allc["SECID"] == table) & (~allc["weekend"])].set_index("TRADEDATE").sort_index()
        s = r["SETTLEPRICE"].reindex(dates)
        sw = r["SWAPRATE"].reindex(dates).fillna(0.0) if "SWAPRATE" in r else pd.Series(0.0, index=dates)
        out = pd.DataFrame({"held": np.where(s.notna(), table, None), "Fn": s, "Fp": s - sw, "swap": sw,
                            "is_roll": False, "oi_value": r["OPENPOSITIONVALUE"].reindex(dates),
                            "oi": r["OPENPOSITION"].reindex(dates)}, index=dates)
    else:
        c = pd.read_csv(ctx.interim / f"futures_{table}.csv", index_col=0, parse_dates=True).reindex(dates)
        h = allc[allc["ASSETCODE"] == ncc][["SECID", "TRADEDATE", "OPENPOSITIONVALUE", "OPENPOSITION", "SETTLEPRICE"]]
        key = pd.MultiIndex.from_arrays([c["held"].values, dates])
        hv = h.set_index(["SECID", "TRADEDATE"])
        out = pd.DataFrame({"held": c["held"].values, "Fn": c["F_held"].values, "Fp": c["F_held_prev_today"].values,
                            "swap": 0.0, "is_roll": c["is_roll"].fillna(False).astype(bool).values,
                            "oi_value": hv["OPENPOSITIONVALUE"].reindex(key).values,
                            "oi": hv["OPENPOSITION"].reindex(key).values}, index=dates)
    out["mr1"] = mr.values
    pv_static = float(pit["point_value_static"])
    if usd:
        fx_cut = float(pit["point_value_used"]) / pv_static if pv_static else np.nan    # = 1 at the cutoff snapshot
        pv_usd = float(pit["point_value_used"]) / (usdrub.loc[:pd.Timestamp(ctx.cutoff)].iloc[-1] if usdrub is not None else np.nan)
        implied = out["oi_value"] / (out["oi"] * out["Fn"])
        cbr = pv_usd * usdrub.reindex(dates) if usdrub is not None else pd.Series(np.nan, index=dates)
        ok = np.isfinite(implied) & (out["oi"] > 0)
        out["pv"] = np.where(ok, implied, cbr)
        out["pv_source"] = np.where(ok, "EXCHANGE_OI_IMPLIED", "CBR_x_USD_POINT_VALUE")
        out["pv_frozen_cutoff"] = pv_static * fx_cut
    else:
        out["pv"] = pv_static
        out["pv_source"] = "STATIC_RUB"
        out["pv_frozen_cutoff"] = pv_static
    notional = out["Fn"] * out["pv"]
    broker = broker_fee_smlt if route == "SMLT_FUT" else float(pit["exchange_fee_rate"]) * notional
    out["fee"] = float(pit["exchange_fee_rate"]) * notional + broker + 0.5 * float(pit["min_step"]) * out["pv"]
    out["notional"] = notional
    out["notional_cutoff"] = float(pit["notional_pit"])
    out["fee_cutoff_rub"] = float(pit["exchange_fee_rub_pit"]) + (broker_fee_smlt if route == "SMLT_FUT" else
                                                                   float(pit["exchange_fee_rub_pit"])) + float(pit["half_tick_rub_pit"])
    out["mr1_cutoff"] = float(ctx._adaptive_mr[ncc].dropna().loc[:pd.Timestamp(ctx.cutoff)].iloc[-1])
    if getattr(ctx, "_regime_v2r1_terms", False) or getattr(ctx, "_regime_v2r2_combined", False):
        from ..regime_v2.terms import repair
        out = repair(ctx, route, out, pit, broker_fee_smlt)
    if getattr(ctx, "_regime_v2r2_combined", False):
        from ..regime_v2.execution_inputs import repair_terms
        out = repair_terms(ctx, route, out, pit, broker_fee_smlt)
    return out


def historical_market(ctx, cfg: dict, funded: list[FundedSpec], routes: list[str], prices: pd.DataFrame,
                      dates: pd.DatetimeIndex, usdrub: pd.Series | None, cash_rate: float = 0.0) -> tuple:
    """One-path Market over `dates` (history). `prices[name]` holds the unit price of every funded/MMF instrument."""
    T = len(dates)
    Af, Ad = len(funded), len(routes)
    P = np.empty((1, T, Af))
    trad_f = np.ones((T, Af), bool)
    for j, f in enumerate(funded):
        px = prices[f.name].reindex(dates)
        trad_f[:, j] = px.notna().values
        P[0, :, j] = px.ffill().bfill().values        # untradable days: price carried only for valuation continuity
    Fp = np.empty((1, T, Ad))
    Fn = np.empty((1, T, Ad))
    pv = np.empty((1, T, Ad))
    fee = np.empty((1, T, Ad))
    mr = np.empty((T, Ad))
    roll = np.zeros((T, Ad), bool)
    trad_d = np.ones((T, Ad), bool)
    contract = [[None] * Ad for _ in range(T)]
    terms = {}
    broker = float(ctx.cfg["costs"]["broker_fee_futures_rub_per_contract"])
    for j, r in enumerate(routes):
        h = deriv_history(ctx, r, dates, usdrub, broker)
        terms[r] = h
        ok = h["Fn"].notna() & h["mr1"].notna() & np.isfinite(h["pv"])
        trad_d[:, j] = ok.values
        Fn[0, :, j] = h["Fn"].ffill().bfill().values
        fp = h["Fp"].where(h["Fp"].notna(), h["Fn"])
        Fp[0, :, j] = fp.ffill().bfill().values
        pv[0, :, j] = pd.Series(h["pv"]).ffill().bfill().values
        fee[0, :, j] = h["fee"].ffill().bfill().values
        mr[:, j] = h["mr1"].ffill().bfill().values
        roll[:, j] = h["is_roll"].values
        for t in range(T):
            contract[t][j] = h["held"].iat[t]
    lag = np.tile(settlement_lag(dates, cfg)[:, None], (1, Af))
    for j, f in enumerate(funded):
        if f.lag == 0:
            lag[:, j] = 0                                   # owner-authorised academic BTC sleeve
        elif f.lag < 0:
            lag[:, j] = -f.lag                              # fixed cycle (e.g. currency TOM = 1)
    mkt = E.Market(P=P, Fp=Fp, Fn=Fn, pv=pv, mr=mr, fee=fee, roll=roll, lag=lag, tradable_f=trad_f,
                   tradable_d=trad_d, cash_rate=np.full(T, cash_rate), contract=contract)
    if getattr(ctx, "_regime_v2r2_combined", False):
        from ..regime_v2.execution_inputs import historical
        mkt.execution = historical(ctx, cfg, funded, routes, dates, mkt, terms)
    fin = [E.Instrument(f.name, f.kind, f.lot, f.cost_prop, f.cost_unit, f.issuer, f.factor) for f in funded]
    din = [E.Instrument(r, E.DERIV) for r in routes]
    return fin, din, mkt, terms
