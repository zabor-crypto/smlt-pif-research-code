"""Walk-forward evaluation of hedge policies without look-ahead.

Two complementary views:

1. DAILY CONSTANT-EXPOSURE series (hedge-effectiveness view). Each day the investor holds RUB E of
   SMLT and a short futures notional h_{t-1} * E decided at the previous close:
       pnl_t = E * (r_S,t - h_{t-1} * r_F,t) - costs_t + C * m_t
   h is re-estimated every ``rebalance_every`` days from the trailing ``window`` returns ending at the
   decision date and traded only if |h_target - h_held| > threshold.

2. EPISODES (investor view, horizons 6m / 12m). Each episode starts with RUB 10m of SMLT (fixed share
   count) + hedge capital C, and runs the full futures simulator (contracts, VM, IM from the NCC
   market-risk rate, rolls, fees, margin budget, forced deleveraging). Start dates advance monthly.
   Episodes overlap -> they are dependent draws; reported as a distribution, not as i.i.d. evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ..risk.metrics import es_hist, max_drawdown_rub, var_hist
from ..strategies.futures import FuturesCosts, liquidity_at_risk, simulate_hedge

# policy(t_index, history_df) -> target short hedge ratio in [0, 1] using rows < = t only
Policy = Callable[[pd.DataFrame], float]


@dataclass
class WFConfig:
    window: int = 120
    rebalance_every: int = 5
    threshold: float = 0.0
    cost_rate: float | pd.Series = 0.002      # one-way cost per unit notional (fees + half spread); a Series
    roll_cost_rate: float | pd.Series = 0.004  # makes it date-specific (per-contract RUB cost / price at t)


def target_path(hist: pd.DataFrame, policy: Policy, cfg: WFConfig, start: int) -> pd.Series:
    """Decision-date targets. hist rows are daily; the target at row i uses hist.iloc[:i+1] only."""
    out = pd.Series(np.nan, index=hist.index)
    for i in range(start, len(hist), cfg.rebalance_every):
        out.iloc[i] = policy(hist.iloc[: i + 1])
    return out.ffill()


def daily_walk_forward(R: pd.DataFrame, is_roll: pd.Series, policy: Policy, cfg: WFConfig, exposure: float,
                       capital: float, mmf: pd.Series, start_date: str) -> pd.DataFrame:
    hist = R.dropna(subset=["SMLT", "SMLT_FUT"])
    start = max(int(hist.index.searchsorted(pd.Timestamp(start_date))), cfg.window)
    tgt = target_path(hist, policy, cfg, start)
    held = 0.0
    rec = []
    for i in range(start, len(hist)):
        t = hist.index[i]
        rS, rF = hist["SMLT"].iat[i], hist["SMLT_FUT"].iat[i]
        pnl_spot = exposure * rS
        pnl_hedge = -held * exposure * rF
        cr = float(cfg.cost_rate.get(t, np.nan)) if isinstance(cfg.cost_rate, pd.Series) else cfg.cost_rate
        rr = float(cfg.roll_cost_rate.get(t, np.nan)) if isinstance(cfg.roll_cost_rate, pd.Series) else cfg.roll_cost_rate
        cr = 0.0 if np.isnan(cr) else cr
        rr = 0.0 if np.isnan(rr) else rr
        roll = rr * held * exposure if bool(is_roll.get(t, False)) else 0.0
        # decision at close t (after observing t) for tomorrow
        h_new = float(np.clip(tgt.iat[i], 0.0, 1.0))
        trade = 0.0
        if abs(h_new - held) > cfg.threshold + 1e-12:
            trade = h_new - held
        tc = abs(trade) * exposure * cr
        carry = capital * float(mmf.get(t, 0.0) if not np.isnan(mmf.get(t, np.nan)) else 0.0)
        rec.append({"date": t, "h_held": held, "h_target": h_new, "trade": trade, "pnl_spot": pnl_spot,
                    "pnl_hedge": pnl_hedge, "costs": tc + roll, "carry": carry,
                    "pnl_total": pnl_spot + pnl_hedge - tc - roll + carry, "rS": rS, "rF": rF})
        held += trade
    return pd.DataFrame(rec).set_index("date")


def run_episodes(spot: pd.Series, fut: pd.DataFrame, R: pd.DataFrame, policy: Policy, *, horizon: int,
                 exposure: float, capital: float, mmf: pd.Series, im_rate: pd.Series, costs: FuturesCosts,
                 threshold: float, window: int, rebalance_every: int, step: int = 21, first_start: str,
                 margin_budget_frac: float | None, label: str) -> tuple[pd.DataFrame, dict]:
    """Rolling-start episodes. Returns (summary per episode, dict of full sims keyed by start date)."""
    cal = fut.index.intersection(spot.index)
    cal = cal[cal >= pd.Timestamp(first_start)]
    hist_all = R.dropna(subset=["SMLT", "SMLT_FUT"])
    rows, sims = [], {}
    for k in range(0, len(cal) - horizon, step):
        t0, t1 = cal[k], cal[k + horizon]
        idx = cal[(cal >= t0) & (cal <= t1)]
        n_shares = exposure / spot.loc[t0]
        # targets: recompute every rebalance_every days using returns up to that close
        targets = pd.Series(np.nan, index=idx)
        for j in range(0, len(idx), rebalance_every):
            h = hist_all.loc[: idx[j]]
            targets.iloc[j] = float(np.clip(policy(h.tail(max(window, 260))), 0.0, 1.0))
        targets = targets.ffill()
        sim = simulate_hedge(spot.loc[idx], fut.loc[idx], targets, n_shares=n_shares, hedge_capital=capital,
                             mmf_ret=mmf.reindex(idx).fillna(0.0), im_rate=im_rate.reindex(idx).ffill().bfill(),
                             costs=costs, threshold=threshold, margin_budget_frac=margin_budget_frac)
        lar = liquidity_at_risk(sim)
        pnl = sim["daily_pnl"]
        rows.append({
            "strategy": label, "start": t0, "end": t1, "horizon_days": horizon, "capital": capital,
            "threshold": threshold, "terminal_pnl": float(pnl.sum()), "spot_pnl": float(sim["spot_pnl"].sum()),
            "hedge_pnl": float(sim["hedge_pnl"].sum()), "costs": float((sim["tc"] + sim["roll_cost"]).sum()),
            "n_trades": int((sim["trade"] != 0).sum()), "turnover_contracts": float(sim["trade"].abs().sum()),
            "avg_hedge_ratio": float(sim["hedge_ratio"].mean()), "max_dd_rub": max_drawdown_rub(pnl.cumsum().values),
            "daily_ES95_rub": es_hist(pnl.values, 0.95), "daily_VaR95_rub": var_hist(pnl.values, 0.95),
            **lar,
        })
        sims[t0] = sim
    return pd.DataFrame(rows), sims


# ------------------------------------------------------------------ policies
def fixed(h: float) -> Policy:
    return lambda hist: h


def rolling_hstar(window: int = 120, cap: float = 1.0) -> Policy:
    def f(hist: pd.DataFrame) -> float:
        x = hist[["SMLT", "SMLT_FUT"]].dropna().tail(window)
        if len(x) < window // 2:
            return 0.0
        c = np.cov(x["SMLT"], x["SMLT_FUT"], ddof=1)
        return float(np.clip(c[0, 1] / c[1, 1], 0.0, cap))
    return f
