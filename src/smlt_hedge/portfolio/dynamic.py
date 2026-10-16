"""V3 Phases 4-6: path-dependent policy engine (vectorised over paths) on the unified recursive ledger.

One engine runs every policy on any set of paths: bootstrap / forward-scenario paths, or HISTORICAL episodes
stacked as paths. Positions at the close of day t are decided with information <= t and earn day t+1 returns.

State per path (N paths, H days):
    shares fraction e_t (of the initial share count; 0.25 .. 1.0; never above 1.0),
    futures notionals N_i,t (RUB, signed; drift with the contract price between trades),
    account A_t (ledger), IM_t, free_t = A_t - IM_t * capital_per_margin.
Daily: spot P&L = e * E * S_{t-1} * r_S;  VM = sum_i N_i r_i;  interest = ledger.interest(free, IM);
       roll costs 2 c_i |N_i| on each sleeve's roll days; exit costs at the horizon.
Policy output each day: (portfolio-state code, target hedge fractions h (of CURRENT SMLT exposure value),
target share fraction). Trades happen only when the policy changes its portfolio state (after confirmation /
minimum holding, or an emergency override) AND the change is material (max |dh| > threshold), or when the share
fraction changes. Liquidity: if free_t < 0 the WATERFALL releases margin (least valuable non-direct sleeves, then
direct SMLT futures, then SELL SHARES = DE_RISKING, always counted).
P&L_t = e_t E S_t + (cash from share sales) ... all inside A_t:  P&L_t = e_t E S_t + A_t - E - C.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..risk.metrics import es_hist, var_hist
from ..strategies import ledger
from ..strategies.collateral import CASH_ONLY, Collateral


@dataclass
class Sleeves:
    series: list[str]
    mr: np.ndarray             # IM rate per unit notional
    cost: np.ndarray           # one-way cost per unit notional
    rolls_per_year: np.ndarray
    direct: int                # index of the DIRECT (SMLT futures) sleeve


@dataclass
class Policy:
    """Target table per portfolio-state code: hedge fractions (S x m) and share fractions (S,)."""
    name: str
    h_by_state: np.ndarray
    e_by_state: np.ndarray
    confirm: int = 0
    hold: int = 0
    threshold: float = 0.0
    protective_rank: np.ndarray | None = None     # higher = more protective (emergency may skip confirmation)
    emergency_shock: float = 0.10                 # |1-day SMLT move| that allows an immediate protective switch
    step_exposure: bool = False                   # active branch: move the share fraction one 25% step at a time
    meta: dict = field(default_factory=dict)


def _roll_mask(H: int, rpy: float) -> np.ndarray:
    n = int(np.ceil(rpy * H / 252 - 1e-9))
    m = np.zeros(H, bool)
    for k in range(n):
        m[int(round(H * (k + 1) / (n + 1))) - 1] = True
    return m


def run(P: dict, states: np.ndarray, pol: Policy, sl: Sleeves, *, E: float, C: float, mmf_daily, coll: Collateral = CASH_ONLY,
        equity_cost: float = 0.0006, min_buffer: float = 0.0, init_state: np.ndarray | None = None, shock: np.ndarray | None = None) -> dict:
    """states: (N, H+1) int codes observed at the close of days 0..H (day 0 = decision date)."""
    rS = np.asarray(P["SMLT"], float)
    N, H = rS.shape
    R = np.stack([np.asarray(P[s], float) for s in sl.series], axis=2) if sl.series else np.zeros((N, H, 0))
    m_ = len(sl.series)
    md = np.broadcast_to(np.asarray(mmf_daily, float), (H,))
    mult, cpm = coll.margin_multiplier, coll.capital_per_margin()
    rolls = np.stack([_roll_mask(H, r) for r in sl.rolls_per_year], axis=1) if m_ else np.zeros((H, 0), bool)
    # ---- day 0: initial state = observed state (no confirmation needed at inception)
    cur = states[:, 0].copy() if init_state is None else init_state.copy()
    cand, cand_n = cur.copy(), np.zeros(N, int)
    since = np.zeros(N, int)
    since_e = np.zeros(N, int)
    S = np.ones(N)
    e = pol.e_by_state[cur].astype(float)
    e = np.where(np.isnan(e), 1.0, e)                                 # HOLD at inception = keep the full position
    h = pol.h_by_state[cur].astype(float)                     # (N, m) fraction of current exposure value
    Nn = h * (E * e * S)[:, None]
    A = C - (np.abs(Nn) * sl.cost[None, :]).sum(axis=1) - E * (1 - e) * equity_cost + E * (1 - e)
    costs_cum = (np.abs(Nn) * sl.cost[None, :]).sum(axis=1) + E * (1 - e) * equity_cost
    turnover = np.abs(Nn).sum(axis=1) / E
    IM = (np.abs(Nn) * sl.mr[None, :]).sum(axis=1) * mult
    free = A - IM * cpm
    breach = free < min_buffer
    switches = np.zeros(N, int)
    exp_changes = (e < 1.0).astype(int)
    shares_sold_forced = np.zeros(N)
    step5 = np.zeros(N, int)
    step34 = np.zeros(N, int)
    total = np.empty((N, H))
    free_path = np.empty((N, H))
    im_path = np.empty((N, H))
    e_path = np.empty((N, H))
    for t in range(H):
        r_s, r = rS[:, t], R[:, t, :]
        # ---- P&L of positions held since yesterday's close
        vm = (Nn * r).sum(axis=1)
        intr = ledger.interest(free, IM, md[t], coll)
        S = S * (1 + r_s)
        Nn = Nn * (1 + r)
        rc = (2 * np.abs(Nn) * sl.cost[None, :] * rolls[t][None, :]).sum(axis=1)
        A = A + intr + vm - rc
        costs_cum += rc
        # ---- observe state, confirmation / holding / emergency
        obs = states[:, t + 1]
        cand_n = np.where(obs == cand, cand_n + 1, 1)
        cand = obs
        since += 1
        since_e += 1
        want = (cand != cur) & (cand_n >= max(pol.confirm, 1)) & (since >= pol.hold)
        if pol.protective_rank is not None:
            sh = np.abs(r_s) >= pol.emergency_shock if shock is None else shock[:, t]
            more_prot = pol.protective_rank[cand] > pol.protective_rank[cur]
            emergency = (cand != cur) & more_prot & (sh | (free < min_buffer))
            want = want | emergency
        new = np.where(want, cand, cur)
        # ---- targets of the (possibly new) portfolio state
        e_tgt = pol.e_by_state[new].astype(float)
        e_tgt = np.where(np.isnan(e_tgt), e, e_tgt)                   # NaN = HOLD the current share fraction
        if pol.step_exposure:
            # one 25% step at a time, at most one step per `hold` days (also while the state persists)
            step_ok = (since_e >= max(pol.hold, 1)) | want
            e_tgt = np.where(e_tgt < e, np.maximum(e - 0.25, e_tgt), np.where(e_tgt > e, np.minimum(e + 0.25, e_tgt), e))
            e_tgt = np.where(step_ok, e_tgt, e)
        e_tgt = np.minimum(e_tgt, 1.0)
        h_tgt = pol.h_by_state[new].astype(float)
        expo = E * S
        cur_h = Nn / np.maximum((e * expo)[:, None], 1e-9)
        material = (np.abs(h_tgt - cur_h).max(axis=1) > pol.threshold + 1e-12) if m_ else np.zeros(N, bool)
        trade_h = want & material
        trade_e = np.abs(e_tgt - e) > 1e-12
        switches += (new != cur).astype(int)
        since = np.where(new != cur, 0, since)
        cur = new
        # shares: sell (DE_RISKING) or buy back previously sold (RE_RISKING), cash through the account
        de = np.where(trade_e, e_tgt - e, 0.0)
        A = A - de * expo - np.abs(de) * expo * equity_cost
        costs_cum += np.abs(de) * expo * equity_cost
        exp_changes += trade_e.astype(int)
        since_e = np.where(trade_e, 0, since_e)
        e = e + de
        # hedge sleeves re-sized to the new exposure only when the policy trades
        tgt_N = h_tgt * (e * expo)[:, None]
        dN = np.where((trade_h | trade_e)[:, None], tgt_N - Nn, 0.0)
        c = (np.abs(dN) * sl.cost[None, :]).sum(axis=1)
        A -= c
        costs_cum += c
        turnover += np.abs(dN).sum(axis=1) / E
        Nn = Nn + dN
        if t == H - 1:
            c = (np.abs(Nn) * sl.cost[None, :]).sum(axis=1)
            A -= c
            costs_cum += c
        IM = (np.abs(Nn) * sl.mr[None, :]).sum(axis=1) * mult
        free = A - IM * cpm
        # ---- liquidity waterfall (steps 3-4: cut sleeves, non-direct first; step 5: sell shares)
        bad = free < min_buffer
        if bad.any():
            breach |= bad
            order = [j for j in range(m_) if j != sl.direct] + ([sl.direct] if m_ else [])
            deficit = np.where(bad, min_buffer - free, 0.0)
            for j in order:
                rel_per = sl.mr[j] * mult * cpm - sl.cost[j]
                cut = np.minimum(np.abs(Nn[:, j]), np.where(rel_per > 0, deficit / max(rel_per, 1e-12), 0.0))
                cut = np.where(bad, cut, 0.0)
                step34 += (cut > 0).astype(int)
                Nn[:, j] -= np.sign(Nn[:, j]) * cut
                A -= cut * sl.cost[j]
                costs_cum += cut * sl.cost[j]
                deficit = np.maximum(deficit - cut * rel_per, 0.0)
            IM = (np.abs(Nn) * sl.mr[None, :]).sum(axis=1) * mult
            free = A - IM * cpm
            need = np.where(free < min_buffer - 1.0, min_buffer - free, 0.0)
            sell = np.minimum(need / np.maximum(E * S * (1 - equity_cost), 1e-9), e)
            step5 += (sell > 0).astype(int)
            shares_sold_forced += sell * E * S
            A += sell * E * S * (1 - equity_cost)
            costs_cum += sell * E * S * equity_cost
            e = e - sell
            free = A - IM * cpm
        total[:, t] = e * E * S + A - E - C
        free_path[:, t], im_path[:, t], e_path[:, t] = free, IM, e
    term = total[:, -1]
    ctrl = E * (S - 1) + C * (np.prod(1 + md) - 1)
    up = S > 1
    upc = float(term[up].mean() / ctrl[up].mean()) if up.any() else np.nan
    dnc = float(term[~up].mean() / ctrl[~up].mean()) if (~up).any() else np.nan
    peak = np.maximum.accumulate(np.concatenate([np.zeros((N, 1)), total], axis=1), axis=1)[:, 1:]
    mdd = (peak - total).max(axis=1)
    return {"mean_pnl": float(term.mean()), "ES95": es_hist(term, 0.95), "ES99": es_hist(term, 0.99), "VaR95": var_hist(term, 0.95),
            "std_pnl": float(term.std(ddof=1)) if N > 1 else 0.0, "p05_pnl": float(np.quantile(term, 0.05)), "min_pnl": float(term.min()),
            "p95_max_dd": float(np.quantile(mdd, 0.95)), "mean_max_dd": float(mdd.mean()), "upside_capture": upc, "downside_capture": dnc,
            "p_sleeve_breach": float(breach.mean()), "transaction_costs": float(costs_cum.mean()), "turnover": float(turnover.mean()),
            "mean_switches": float(switches.mean()), "mean_exposure_changes": float(exp_changes.mean()),
            "mean_final_share_fraction": float(e.mean()), "mean_share_fraction": float(e_path.mean()),
            "forced_share_sales_rub_mean": float(shares_sold_forced.mean()), "p_forced_share_sale": float((step5 > 0).mean()),
            "p_waterfall_cut": float((step34 > 0).mean()),
            "initial_im": float(np.mean((np.abs(pol.h_by_state[states[:, 0]] * E) * sl.mr[None, :]).sum(axis=1) * mult)) if m_ else 0.0,
            "peak_im_p95": float(np.quantile(im_path.max(axis=1), 0.95)),
            "peak_collateral_p95": float(np.quantile((im_path * cpm).max(axis=1), 0.95)),
            "_term": term, "_ctrl": ctrl}
