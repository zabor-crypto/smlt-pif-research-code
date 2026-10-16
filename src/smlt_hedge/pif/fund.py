"""V4 fund-level NAV engine (vectorised over paths) on the unified recursive ledger (strategies/ledger.py).

Balance sheet per path at every close t (RUB):
    V_f   market value of each FUNDED asset (SMLT, experimental BTC spot)       -> consumes NAV
    D_i   notional of each DERIVATIVE (signed)                                   -> consumes margin only
    cash  liquidity sleeve (earns the money-market return); posted IM is taken from cash
    NAV   = sum_f V_f + cash      (derivative P&L is settled into cash daily as variation margin)
    IM    = sum_i mr_i * mult * |D_i|;  free = cash - IM * capital_per_margin
    gross exposure = sum_f V_f + sum_i |D_i|;   net market exposure = sum_f V_f + sum_i D_i
Every target is a vector of NAV weights x (funded weights and derivative notionals per NAV) - never a hedge ratio
relative to the SMLT holding. Sale proceeds of ANY asset go to cash and are reusable by every other sleeve.

Rebalancing: observable state -> candidate state -> confirmation (`confirm` days) -> minimum holding (`hold` days) ->
trade the WHOLE portfolio to the new state's target only if max |x_target - x_current| > threshold. Emergency override:
an immediate switch to a MORE protective state after a one-day SMLT move >= emergency_shock or when free liquidity is
below the buffer.

Liquidity waterfall (free < min_buffer), deterministic and documented:
  1. free cash is used automatically (it is what `free` measures);
  2. the liquidity sleeve IS cash in this model (money-market units are redeemed first);
  3./4. then positions are cut in the order `order` computed ONCE per state from the scenario covariance: actions sorted
     by (change in portfolio variance per RUB of liquidity released) ascending, i.e. the action that removes the most
     risk per RUB released goes first; transaction cost per RUB released breaks ties. Derivative cuts release
     margin * capital_per_margin per RUB of notional; funded sales release 100% (less costs). SMLT is therefore sold
     neither first nor last by construction - its place follows from risk contribution and liquidity released.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..risk.metrics import es_hist, var_hist
from ..strategies import ledger
from ..strategies.collateral import CASH_ONLY, Collateral


@dataclass
class FundPolicy:
    name: str
    targets: np.ndarray                 # (S, A) NAV weights per state code
    confirm: int = 0
    hold: int = 0
    threshold: float = 0.0
    protective_rank: np.ndarray | None = None
    emergency_shock: float = 0.10
    rebalance: bool = True               # False = buy-and-hold after t0 (STATIC portfolios)
    orders: np.ndarray | None = None     # (S, A) waterfall order per state (asset indices)


def waterfall_order(x: np.ndarray, cov: np.ndarray, kinds: np.ndarray, mr: np.ndarray, cost: np.ndarray, cpm: float) -> np.ndarray:
    """Deterministic waterfall ranking for target vector x (see module docstring)."""
    mrc = cov @ x                                            # marginal variance contribution
    scores = []
    for j in range(len(x)):
        if abs(x[j]) < 1e-12:
            scores.append((np.inf, np.inf, j))
            continue
        dvar = -2.0 * np.sign(x[j]) * mrc[j]                 # d variance per unit reduction of |x_j|
        release = 1.0 - cost[j] if kinds[j] == "FUNDED" else mr[j] * cpm - cost[j]
        release = max(release, 1e-9)
        scores.append((dvar / release, cost[j] / release, j))
    return np.array([j for *_, j in sorted(scores)])


def run(P: dict, mmf_daily, assets, pol: FundPolicy, states: np.ndarray, *, E: float, C: float, x0_initial: np.ndarray,
        coll: Collateral = CASH_ONLY, min_buffer_nav: float = 0.0, shock: np.ndarray | None = None, log: bool = False,
        smlt_series: str = "SMLT", targets_t: np.ndarray | None = None, force: np.ndarray | None = None) -> dict:
    """targets_t: optional (H+1, S, A) target tables in force at each close (causal re-estimation); force: optional (H+1,)
    days on which the portfolio is re-traded to the table in force for its current state (if the change is material),
    also for buy-and-hold policies (scheduled re-estimation)."""
    N, H = P[smlt_series].shape
    A = len(assets)
    kinds = np.array([a.kind for a in assets])
    fund = kinds == "FUNDED"
    R = np.stack([np.nan_to_num(np.asarray(P[a.series], float)) for a in assets], axis=2)   # (N, H, A)
    rS = np.nan_to_num(np.asarray(P[smlt_series], float))
    md = np.broadcast_to(np.asarray(mmf_daily, float), (H,))
    mr = np.array([a.mr for a in assets])
    cost = np.array([a.cost_one_way for a in assets])
    rolls = np.zeros((H, A), bool)
    for j, a in enumerate(assets):
        n = int(np.ceil(a.rolls_per_year * H / 252 - 1e-9)) if a.kind == "DERIVATIVE" else 0
        for k in range(n):
            rolls[int(round(H * (k + 1) / (n + 1))) - 1, j] = True
    mult, cpm = coll.margin_multiplier, coll.capital_per_margin()
    NAV0 = E + C
    # ---- t0: initial portfolio, then trade to the target of the observed state
    V = np.tile(x0_initial * NAV0, (N, 1))                  # funded values / derivative notionals
    cash = np.full(N, NAV0 - float((x0_initial * fund).sum() * NAV0))
    cur = states[:, 0].copy()
    table = lambda t: pol.targets if targets_t is None else targets_t[t]  # noqa: E731
    tgt = table(0)[cur]
    dV = tgt * NAV0 - V
    c = (np.abs(dV) * cost[None, :]).sum(axis=1)
    cash -= (dV * fund[None, :]).sum(axis=1) + c
    V = V + dV
    costs = c.copy()
    turnover = np.abs(dV).sum(axis=1) / NAV0
    IM = (np.abs(V) * (~fund)[None, :] * mr[None, :]).sum(axis=1) * mult
    free = cash - IM * cpm
    buf = min_buffer_nav * NAV0
    breach = free < buf
    cand, cand_n, since = cur.copy(), np.zeros(N, int), np.zeros(N, int)
    switches = np.zeros(N, int)
    wf_events, forced_funded = np.zeros(N, int), np.zeros(N)
    nav_path = np.empty((N, H))
    free_path = np.empty((N, H))
    im_path = np.empty((N, H))
    wsum = np.zeros((N, A))
    rows = []
    for t in range(H):
        r = R[:, t, :]
        vm = (V * r * (~fund)[None, :]).sum(axis=1)
        intr = ledger.interest(free, IM, md[t], coll)
        V = V * (1 + r)
        rc = (2 * np.abs(V) * cost[None, :] * rolls[t][None, :] * (~fund)[None, :]).sum(axis=1)
        cash = cash + intr + vm - rc
        costs += rc
        nav = (V * fund[None, :]).sum(axis=1) + cash
        # ---- state machine
        obs = states[:, t + 1]
        cand_n = np.where(obs == cand, cand_n + 1, 1)
        cand = obs
        since += 1
        want = (cand != cur) & (cand_n >= max(pol.confirm, 1)) & (since >= pol.hold) & pol.rebalance
        if pol.protective_rank is not None and pol.rebalance:
            sh = (np.abs(rS[:, t]) >= pol.emergency_shock) if shock is None else shock[:, t]
            emerg = (cand != cur) & (pol.protective_rank[cand] > pol.protective_rank[cur]) & (sh | (free < buf))
            want |= emerg
        new = np.where(want, cand, cur)
        x_cur = V / np.maximum(nav, 1e-9)[:, None]
        tgt = table(t + 1)[new]
        go = want | (bool(force[t + 1]) if force is not None else False)
        trade = go & (np.abs(tgt - x_cur).max(axis=1) > pol.threshold + 1e-12)
        switches += (new != cur).astype(int)
        since = np.where(new != cur, 0, since)
        cur = new
        dV = np.where(trade[:, None], tgt * nav[:, None] - V, 0.0)
        c = (np.abs(dV) * cost[None, :]).sum(axis=1)
        cash -= (dV * fund[None, :]).sum(axis=1) + c
        V = V + dV
        costs += c
        turnover += np.abs(dV).sum(axis=1) / NAV0
        IM = (np.abs(V) * (~fund)[None, :] * mr[None, :]).sum(axis=1) * mult
        free = cash - IM * cpm
        bad = free < buf
        if bad.any():
            breach |= bad
            deficit = np.where(bad, buf - free, 0.0)
            order = pol.orders[cur] if pol.orders is not None else np.tile(np.arange(A), (N, 1))
            for step in range(A):
                j = order[:, step]
                idx = np.arange(N)
                vj = V[idx, j]
                is_f = fund[j]
                rel = np.where(is_f, 1.0 - cost[j], mr[j] * mult * cpm - cost[j])
                cut = np.where(deficit > 0, np.minimum(np.abs(vj), deficit / np.maximum(rel, 1e-9)), 0.0)
                cut = np.where(rel > 0, cut, 0.0)
                V[idx, j] = vj - np.sign(vj) * cut
                cash += np.where(is_f, cut * (1 - cost[j]), -cut * cost[j])
                costs += cut * cost[j]
                forced_funded += np.where(is_f, cut, 0.0)
                wf_events += (cut > 0).astype(int)
                IM = (np.abs(V) * (~fund)[None, :] * mr[None, :]).sum(axis=1) * mult
                free = cash - IM * cpm
                deficit = np.where(free < buf, buf - free, 0.0)
        nav = (V * fund[None, :]).sum(axis=1) + cash
        nav_path[:, t], free_path[:, t], im_path[:, t] = nav, free, IM
        wsum += V / np.maximum(nav, 1e-9)[:, None]
        if log and N == 1:
            rows.append({"t": t + 1, "state": int(cur[0]), "switched": bool(trade[0]), "nav": float(nav[0]), "cash": float(cash[0]),
                         "free": float(free[0]), "im": float(IM[0]), **{f"x_{a.name}": float(V[0, j] / nav[0]) for j, a in enumerate(assets)}})
    pnl = nav_path[:, -1] - NAV0
    ctrl = E * (np.prod(1 + rS, axis=1) - 1) + C * (np.prod(1 + md) - 1)
    up = np.prod(1 + rS, axis=1) > 1
    part = float(pnl[up].mean() / ctrl[up].mean()) if up.any() and ctrl[up].mean() != 0 else np.nan
    dnc = float(pnl[~up].mean() / ctrl[~up].mean()) if (~up).any() and ctrl[~up].mean() != 0 else np.nan
    peak = np.maximum.accumulate(np.concatenate([np.full((N, 1), NAV0), nav_path], axis=1), axis=1)[:, 1:]
    mdd = ((peak - nav_path) / peak).max(axis=1)
    wbar = wsum / H
    out = {"mean_pnl": float(pnl.mean()), "median_pnl": float(np.median(pnl)), "ES95": es_hist(pnl, 0.95), "ES99": es_hist(pnl, 0.99),
           "VaR95": var_hist(pnl, 0.95), "std_pnl": float(pnl.std(ddof=1)) if N > 1 else 0.0, "min_pnl": float(pnl.min()),
           "p95_mdd_nav": float(np.quantile(mdd, 0.95)), "mean_mdd_nav": float(mdd.mean()), "participation": part,
           "downside_capture": dnc, "p_liquidity_breach": float(breach.mean()), "p_waterfall": float((wf_events > 0).mean()),
           "forced_funded_sales_mean": float(forced_funded.mean()), "transaction_costs": float(costs.mean()),
           "turnover_nav": float(turnover.mean()), "mean_switches": float(switches.mean()),
           "initial_margin_nav": float(((np.abs(table(0)[states[:, 0]]) * (~fund)[None, :] * mr[None, :]).sum(axis=1) * mult).mean()),
           "peak_margin_nav_p95": float(np.quantile(im_path.max(axis=1) / NAV0, 0.95)),
           "min_free_nav_p05": float(np.quantile(free_path.min(axis=1) / NAV0, 0.05)),
           **{f"avg_w_{a.name}": float(wbar[:, j].mean()) for j, a in enumerate(assets)},
           "_pnl": pnl, "_ctrl": ctrl, "_nav": nav_path if N <= 50 else None}
    if log:
        out["_log"] = rows
    return out
