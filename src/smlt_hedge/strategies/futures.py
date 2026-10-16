"""Futures mechanics: continuous-contract construction and a daily VM/IM/collateral simulator.

Continuous contract (documented in METHODOLOGY.md)
- Contracts of one ASSETCODE are ordered by last trading day (LSTTRADE).
- Roll rule: the position is moved from the front contract to the next one at the settlement
  of the trading day that is ``roll_days`` trading days before the front's LSTTRADE.
- Daily return on day t is computed WITHIN the contract held at the close of t-1:
      r_t = F_{c,t} / F_{c,t-1} - 1,   c = held(t-1)
  i.e. there is never a return across two different contracts (no back-adjustment artefacts).
  The roll price gap F_next/F_front is recorded separately as the roll yield.
- Prices are exchange settlement prices (SETTLEPRICE), the basis of variation margin.

Simulator (fixed share count, signed contracts q, short < 0)
    VM_t     = q_{t-1} * (F_t - F_{t-1}) * point_value        (RUB, credited/debited daily)
    IM_t     = |q_t| * im_per_contract_t
    fees     = |dq| * (exchange_fee + broker_fee) + |dq| * half_spread_rub
    collateral cash is taken from the liquidity sleeve; the sleeve earns m_t on its balance,
    while cash posted as IM earns 0 in the conservative main case.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import ledger
from .collateral import CASH_ONLY, Collateral


def roll_schedule(contracts: pd.DataFrame, calendar: pd.DatetimeIndex, roll_days: int = 5) -> pd.DataFrame:
    """contracts: one row per SECID with LSTTRADE. Returns SECID, LSTTRADE, roll_date."""
    c = contracts.dropna(subset=["LSTTRADE"]).drop_duplicates("SECID").sort_values("LSTTRADE").copy()
    cal = pd.DatetimeIndex(sorted(calendar))
    rd = []
    for lt in c["LSTTRADE"]:
        if lt > cal[-1]:
            # expiry beyond the available calendar (e.g. after the decision date): roll date lies in the future
            rd.append(lt - pd.offsets.BDay(roll_days))
            continue
        pos = cal.searchsorted(lt, side="right") - 1  # last calendar day <= LSTTRADE
        rd.append(cal[max(pos - roll_days, 0)])
    c["roll_date"] = rd
    return c[["SECID", "LSTTRADE", "roll_date"]].reset_index(drop=True)


def continuous(rows: pd.DataFrame, calendar: pd.DatetimeIndex, roll_days: int = 5) -> pd.DataFrame:
    """rows: futures history of ONE asset (SECID, TRADEDATE, SETTLEPRICE, OPENPOSITION, VOLUME, LSTTRADE)."""
    rows = rows[~rows["TRADEDATE"].dt.dayofweek.ge(5)]
    px = rows.pivot_table(index="TRADEDATE", columns="SECID", values="SETTLEPRICE", aggfunc="last")
    # pivot_table drops all-NaN rows/cols -> align OI/volume explicitly to the price grid
    oi = rows.pivot_table(index="TRADEDATE", columns="SECID", values="OPENPOSITION", aggfunc="last").reindex(
        index=px.index, columns=px.columns)
    vol = rows.pivot_table(index="TRADEDATE", columns="SECID", values="VOLUME", aggfunc="last").reindex(
        index=px.index, columns=px.columns)
    # exchange-reported RUB traded value (correct for every contract unit; volume*settle is NOT, see specs.audit)
    val = (rows.pivot_table(index="TRADEDATE", columns="SECID", values="VALUE", aggfunc="last").reindex(
        index=px.index, columns=px.columns) if "VALUE" in rows else vol * np.nan)
    cal = pd.DatetimeIndex(sorted(set(calendar) & set(px.index)))
    sched = roll_schedule(rows[["SECID", "LSTTRADE"]], cal, roll_days)
    held = []
    for t in cal:
        cand = sched[(sched["roll_date"] > t)]
        # first contract (by expiry) not yet rolled out of that has a settlement price today
        sec = None
        for s in cand["SECID"]:
            if s in px.columns and not np.isnan(px.at[t, s]):
                sec = s
                break
        held.append(sec)
    out = pd.DataFrame({"held": held}, index=cal)
    out["held_prev"] = out["held"].shift(1)
    f_t, f_prev, f_new = [], [], []
    for i, t in enumerate(cal):
        hp = out["held_prev"].iat[i]
        h = out["held"].iat[i]
        if i == 0 or hp is None or (isinstance(hp, float) and np.isnan(hp)):
            f_t.append(np.nan); f_prev.append(np.nan)
        else:
            f_t.append(px.at[t, hp] if hp in px.columns else np.nan)
            f_prev.append(px.at[cal[i - 1], hp] if hp in px.columns else np.nan)
        f_new.append(px.at[t, h] if h is not None and h in px.columns else np.nan)
    out["F_held_prev_today"] = f_t        # settle today of contract held since yesterday
    out["F_held_prev_yday"] = f_prev
    out["F_held"] = f_new                 # settle today of contract held from today's close
    out["ret"] = out["F_held_prev_today"] / out["F_held_prev_yday"] - 1.0
    out["is_roll"] = out["held"] != out["held_prev"]
    out.loc[out.index[0], "is_roll"] = False
    out["roll_gap"] = np.where(out["is_roll"], out["F_held"] / out["F_held_prev_today"] - 1.0, np.nan)
    out["oi_held"] = [oi.at[t, h] if h in oi.columns else np.nan for t, h in zip(cal, out["held"])]
    out["vol_held"] = [vol.at[t, h] if h in vol.columns else np.nan for t, h in zip(cal, out["held"])]
    out["value_rub_held"] = [val.at[t, h] if h in val.columns else np.nan for t, h in zip(cal, out["held"])]
    lt = sched.set_index("SECID")["LSTTRADE"]
    out["days_to_expiry"] = [(lt.get(h) - t).days if h in lt.index else np.nan for t, h in zip(cal, out["held"])]
    return out


def annualised_basis(F: pd.Series, S: pd.Series, days_to_expiry: pd.Series) -> pd.Series:
    """Implied carry ln(F/S) * 365/days (calendar-day convention)."""
    tau = days_to_expiry.clip(lower=1) / 365.0
    return np.log(F / S) / tau


@dataclass
class FuturesCosts:
    exchange_fee_rub: float = 0.05     # per contract per side (ISS BUYSELLFEE, SMLT)
    broker_fee_rub: float = 0.05       # per contract per side (assumption: = exchange fee)
    half_spread_rub: float = 0.5       # half of one 1-RUB tick (observed bid/ask 261/262 on SSZ6)

    def per_contract(self) -> float:
        return self.exchange_fee_rub + self.broker_fee_rub + self.half_spread_rub


def simulate_hedge(
    spot: pd.Series,               # SMLT close (RUB), research calendar
    fut: pd.DataFrame,             # output of continuous()
    target_ratio: pd.Series,       # desired short hedge ratio in SHARE units (0..1), decided at close t
    n_shares: float,               # fixed SMLT share count
    hedge_capital: float,          # RUB, liquidity sleeve starting balance
    mmf_ret: pd.Series,            # daily liquidity-sleeve return
    im_rate: pd.Series,            # IM per RUB of futures notional (fraction), known at close t
    costs: FuturesCosts,
    threshold: float = 0.0,        # trade only if |target - current| > threshold
    point_value: float = 1.0,      # RUB per 1.0 price unit per contract (SMLT: lot 1 share, 1 RUB)
    lot: float = 1.0,
    dividends: pd.Series | None = None,  # RUB per share, on ex-date
    im_cash_earns: bool = False,   # legacy flag: True == Collateral(yield_fraction=1, haircut=0)
    margin_budget_frac: float | None = None,  # max share of (free sleeve + posted capital) usable as collateral
    collateral: Collateral = CASH_ONLY,
    on_breach: str = "cut_futures",  # 'cut_futures' | 'sell_shares' (DE_RISKING, recorded separately)
    min_buffer: float = 0.0,
) -> pd.DataFrame:
    idx = fut.index.intersection(spot.index)
    spot, fut = spot.reindex(idx), fut.reindex(idx)
    tr = target_ratio.reindex(idx).ffill().fillna(0.0)
    m = mmf_ret.reindex(idx).fillna(0.0)
    imr = im_rate.reindex(idx).ffill()
    div = (dividends.reindex(idx).fillna(0.0) if dividends is not None else pd.Series(0.0, idx))

    if im_cash_earns:
        collateral = Collateral("legacy_im_earns", 1.0, 0.0, collateral.margin_multiplier)
    cpm, epm, mult = collateral.capital_per_margin(), collateral.earning_per_margin(), collateral.margin_multiplier
    q = 0.0                      # signed contracts held after close
    sleeve = float(hedge_capital)
    im_posted = 0.0              # posted collateral CAPITAL (= IM * capital_per_margin)
    shares_sold = 0.0
    rec = []
    for i, t in enumerate(idx):
        vm = 0.0
        roll_cost = 0.0
        if i > 0:
            # variation margin on contracts held since yesterday's close
            if not np.isnan(fut["ret"].iat[i]):
                dF = fut["F_held_prev_today"].iat[i] - fut["F_held_prev_yday"].iat[i]
                vm = q * dF * point_value
            # free cash earns the money-market return; productive posted collateral earns it too
            sleeve += ledger.interest(sleeve, im_posted / cpm if cpm else 0.0, m.iat[i], collateral)  # unified ledger
        sleeve += vm
        # contract roll at today's settlement: close old, open same size in new contract
        if fut["is_roll"].iat[i] and q != 0:
            roll_cost = 2 * abs(q) * costs.per_contract()
        # rebalance decision at close t with info <= t
        F_now = fut["F_held"].iat[i]
        imr_now = imr.iat[i] if not np.isnan(imr.iat[i]) else 0.0
        im_per_contract = (F_now if not np.isnan(F_now) else 0.0) * lot * point_value * imr_now * mult
        cap_per_contract = im_per_contract * cpm
        target_q = -round(tr.iat[i] * n_shares / lot)
        capped = forced = False
        sold_today = 0.0
        if on_breach == "sell_shares" and sleeve - roll_cost < min_buffer and i > 0 and n_shares > 0:
            # DE_RISKING waterfall step: sell shares to restore the buffer, keep the hedge ratio on remaining shares
            need = min_buffer - (sleeve - roll_cost)
            sold_today = min(n_shares, need / spot.iat[i])
            n_shares -= sold_today
            shares_sold += sold_today
            sleeve += sold_today * spot.iat[i]
            target_q = -round(tr.iat[i] * n_shares / lot)   # hedge follows the remaining shares; traded at cost below
        if margin_budget_frac is not None and cap_per_contract > 0:
            avail = max(0.0, (sleeve - roll_cost + im_posted) * margin_budget_frac)
            max_q = np.floor(avail / cap_per_contract)
            if abs(target_q) > max_q:
                target_q, capped = -max_q, True
            if abs(q) > max_q and sleeve - roll_cost < min_buffer:
                forced = True          # VM drained the free sleeve: position must be cut now
        if sold_today > 0:
            forced = True
        cur_ratio = -q * lot / n_shares if n_shares else 0.0
        if not n_shares:
            target_q = 0.0
        trade = 0.0
        tgt_ratio = -target_q * lot / n_shares if n_shares else 0.0
        if forced or abs(tgt_ratio - cur_ratio) > threshold + 1e-12 or (threshold == 0 and target_q != q):
            trade = target_q - q
        tc = abs(trade) * costs.per_contract()
        q += trade
        sleeve -= tc + roll_cost
        # initial margin requirement; capital moves between free sleeve and posted collateral
        im_req = abs(q) * im_per_contract
        posted_req = im_req * cpm
        sleeve -= posted_req - im_posted
        im_posted = posted_req
        spot_pnl = (n_shares + sold_today) * ((spot.iat[i] - spot.iat[i - 1]) + div.iat[i]) if i > 0 else 0.0
        rec.append({
            "date": t, "spot": spot.iat[i], "F": F_now, "held": fut["held"].iat[i], "q": q, "trade": trade,
            "hedge_ratio": -q * lot / n_shares if n_shares else 0.0, "vm": vm, "tc": tc, "roll_cost": roll_cost,
            "im_req": im_req, "posted_capital": im_posted, "sleeve_free": sleeve, "spot_pnl": spot_pnl,
            "capped": capped, "forced": forced, "n_shares": n_shares, "shares_sold": sold_today,
        })
    df = pd.DataFrame(rec).set_index("date")
    df["sleeve_total"] = df["sleeve_free"] + df["posted_capital"]
    df["spot_value"] = df["n_shares"] * df["spot"]
    df["total_value"] = df["spot_value"] + df["sleeve_total"]
    initial = float(df["n_shares"].iat[0] + df["shares_sold"].iat[0]) * spot.iat[0] + float(hedge_capital)
    df["daily_pnl"] = df["total_value"].diff()
    df.iloc[0, df.columns.get_loc("daily_pnl")] = df["total_value"].iat[0] - initial  # day-0 entry costs
    df["hedge_pnl"] = df["daily_pnl"] - df["spot_pnl"]
    df["cum_vm"] = df["vm"].cumsum()
    return df


def liquidity_at_risk(sim: pd.DataFrame) -> dict:
    """Interim liquidity metrics of a futures hedge path (RUB, positive = cash need)."""
    vm = sim["vm"]
    cum = vm.cumsum()
    drain = (cum.cummax() - cum).max()   # worst peak-to-trough cumulative VM outflow
    return {
        "max_daily_vm_outflow": float(max(0.0, -vm.min())),
        "max_cum_vm_drain": float(max(0.0, drain)),
        "worst_cum_vm_from_start": float(max(0.0, -cum.min())),
        "peak_im": float(sim["im_req"].max()),
        "peak_total_collateral": float((sim["im_req"] + (cum.cummax() - cum)).max()),
        "min_free_sleeve": float(sim["sleeve_free"].min()),
        "sleeve_breach": bool((sim["sleeve_free"] < 0).any()),
        "n_forced_reductions": int(sim.get("forced", pd.Series(False)).sum()),
        "share_days_capped": float(sim.get("capped", pd.Series(False)).mean()),
    }


# ---------------------------------------------------------------- engine-level headline quantities
def contract_position(h: float, n_shares: float, F0: float, point_value: float = 1.0, lot: float = 1.0) -> dict:
    """Exact contract-level position for a share-unit hedge ratio h (the SAME rounding as simulate_hedge)."""
    q = -round(h * n_shares / lot)
    notional = abs(q) * F0 * point_value * lot
    return {"contracts": int(abs(q)), "signed_contracts": int(q), "notional_rub": float(notional),
            "notional_over_spot_exposure": float(notional / (n_shares * 1.0)) if n_shares else float("nan")}


def engine_jump_state(h: float, n_shares: float, S0: float, F0: float, x: float, hedge_capital: float, im_rate: float,
                      collateral: Collateral = CASH_ONLY, point_value: float = 1.0, lot: float = 1.0,
                      costs: FuturesCosts | None = None) -> pd.Series:
    """Run simulate_hedge on a synthetic two-day path (spot and futures both move by x) and return day-1 state."""
    idx = pd.bdate_range("2030-01-01", periods=2)
    fut = pd.DataFrame({"held": ["X", "X"], "F_held": [F0, F0 * (1 + x)], "F_held_prev_today": [np.nan, F0 * (1 + x)],
                        "F_held_prev_yday": [np.nan, F0], "ret": [np.nan, x], "is_roll": [False, False]}, index=idx)
    sim = simulate_hedge(pd.Series([S0, S0 * (1 + x)], index=idx), fut, pd.Series(h, index=idx), n_shares=n_shares,
                         hedge_capital=hedge_capital, mmf_ret=pd.Series(0.0, index=idx), im_rate=pd.Series(im_rate, index=idx),
                         costs=costs or FuturesCosts(0.0, 0.0, 0.0), point_value=point_value, lot=lot, collateral=collateral)
    return sim.iloc[-1]


def rally_exhausting_sleeve(h: float, n_shares: float, S0: float, F0: float, hedge_capital: float, im_rate: float,
                            collateral: Collateral = CASH_ONLY, buffer: float = 0.0, point_value: float = 1.0,
                            lot: float = 1.0, costs: FuturesCosts | None = None, hi: float = 20.0) -> float:
    """Smallest one-day joint up-move x at which the ENGINE's free sleeve falls below `buffer` (bisection)."""
    if h <= 0:
        return float("inf")
    f = lambda x: engine_jump_state(h, n_shares, S0, F0, x, hedge_capital, im_rate, collateral, point_value, lot,  # noqa: E731
                                    costs)["sleeve_free"] - buffer
    if f(0.0) < 0:
        return 0.0
    if f(hi) > 0:
        return float("inf")
    lo_, hi_ = 0.0, hi
    for _ in range(60):
        mid = 0.5 * (lo_ + hi_)
        if f(mid) > 0:
            lo_ = mid
        else:
            hi_ = mid
    return 0.5 * (lo_ + hi_)
