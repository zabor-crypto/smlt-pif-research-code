"""B7/B10/B12: adaptive multi-asset hedge policy, walk-forward engine and liquidity waterfall.

Policy architecture (no return forecasting):
    observable state (regime, collateral usage)  ->  constraints (upside floor, margin budget)
    ->  min-ES95 LP on the trailing `estimation_window` daily returns (data strictly BEFORE the decision day,
        demeaned: no drift view)  ->  target sleeve notionals  ->  trade only if the target moved by more than
        a threshold (or the regime changed).
Target paths are computed once per policy at the weekly decision dates and reused by the daily walk-forward and
by the episode engine (decision at close of t, position effective for t+1).

Episode engine (fixed SMLT share count, hedge capital C, notional-level futures accounting):
    VM_t = sum_i N_i,t-1 * r_i,t      (N drifts with the contract price: fixed contracts)
    IM_t = sum_i mr_i,t * mult * |N_i,t| ;  posted capital = IM * capital_per_margin
    free_t = free_t-1 * (1+m_t) + VM_t + posted_{t-1}*earning - (posted_t - posted_{t-1}) - costs_t
Liquidity waterfall when free_t < 0 (restore free to `restore_frac * C`):
    1 free cash buffer (already exhausted)  2 productive-collateral switch (only in regimes B/C: already reflected
    in capital_per_margin; recorded)  3 cut the least-valuable non-direct sleeve(s) (lowest |corr with SMLT|
    on the last estimation window)  4 cut direct SMLT futures  5 LAST RESORT: sell SMLT shares -> DE_RISKING
    (counted separately, never silent).
IMMEDIATE_CUT comparator (v1 logic): cut direct SMLT futures first, then the other sleeves, then shares.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..optimization.overlay import Problem, Sleeve, min_es
from ..strategies import ledger
from ..strategies.collateral import CASH_ONLY, Collateral
from ..strategies.state_policy import state_policy


@dataclass
class Env:
    R: pd.DataFrame            # daily returns (SMLT + sleeve series), real history
    specs: list                # SleeveSpec list
    imr: pd.DataFrame          # daily PIT NCC mr1 by asset code
    mmf: pd.Series             # liquidity sleeve daily return
    fs_ratio: pd.Series        # F/S (front futures settle / spot): notional per share-unit hedge ratio
    regimes: pd.DataFrame
    E: float
    C: float
    cfg: dict                  # smart_portfolio config


# ---------------------------------------------------------------- target generators (weekly, causal)
def _lp_target(env: Env, t_pos: int, min_upside: float, margin_budget: float, coll: Collateral) -> np.ndarray:
    win = env.cfg["estimation_window"]
    cols = [s.series for s in env.specs]
    hist = env.R.iloc[max(0, t_pos - win):t_pos][["SMLT"] + cols].dropna()      # strictly before the decision day
    if len(hist) < win // 2:
        return np.zeros(len(env.specs))
    H = hist - hist.mean()                                                    # demeaned: no drift / return view
    sl = [Sleeve(s.name, s.lo, s.hi, cost=2 * s.cost_one_way,
                 margin=s.mr * coll.margin_multiplier * coll.capital_per_margin()) for s in env.specs]
    prob = Problem(r_s=H["SMLT"].values, R=H[cols].values, sleeves=sl, margin_budget=margin_budget * env.C / env.E,
                   gross_cap=env.cfg["gross_cap"], min_upside_capture=min_upside)
    sol = min_es(prob)
    return sol.h if sol.status.startswith("optimal") else np.zeros(len(env.specs))


def target_path(env: Env, policy: str, dates: pd.DatetimeIndex, every: int = 5, coll: Collateral = CASH_ONLY,
                regimes: pd.DataFrame | None = None, fixed_h: np.ndarray | None = None, x: float = 0.0) -> pd.DataFrame:
    """Targets (notional fraction of the SMLT exposure per sleeve) decided at the close of each decision date."""
    names = [s.name for s in env.specs]
    di = names.index("DIRECT")
    reg = env.regimes if regimes is None else regimes
    rows = {}
    pos = env.R.index.get_indexer(dates)
    sp = state_policy()
    for k, (t, p) in enumerate(zip(dates, pos)):
        if k % every:
            continue
        h = np.zeros(len(names))
        fs = float(env.fs_ratio.iloc[:p + 1].dropna().iloc[-1]) if env.fs_ratio.iloc[:p + 1].notna().any() else 1.0
        if policy == "FIXED":
            h[di] = -x * fs
        elif policy == "ROLLING_HSTAR":
            w = env.R.iloc[max(0, p - 119):p + 1][["SMLT", "SMLT_FUT"]].dropna()
            hs = np.cov(w["SMLT"], w["SMLT_FUT"])[0, 1] / w["SMLT_FUT"].var() if len(w) > 60 else 0.0
            h[di] = -float(np.clip(hs, 0, 1)) * fs
        elif policy == "CURRENT_STATE":
            hist = env.R.iloc[:p + 1][["SMLT", "SMLT_FUT"]].copy()
            h[di] = -sp(hist) * fs
        elif policy == "FIXED_SMART":
            h = fixed_h.copy()
        elif policy == "ADAPTIVE":
            rg = str(reg["regime"].iloc[p]) if t in reg.index else "NORMAL"
            pol = env.cfg["regime_policy"][rg]
            h = _lp_target(env, p + 1, pol["min_upside"], pol["margin_budget"], coll)   # uses rows <= t
        rows[t] = h
    out = pd.DataFrame.from_dict(rows, orient="index", columns=names)
    out["regime"] = [str(reg["regime"].get(t, "NORMAL")) for t in out.index]
    return out


# ---------------------------------------------------------------- daily constant-exposure walk-forward
def daily_walk(env: Env, targets: pd.DataFrame, dates: pd.DatetimeIndex, threshold: float) -> pd.DataFrame:
    names = [s.name for s in env.specs]
    cols = [s.series for s in env.specs]
    cost = np.array([s.cost_one_way for s in env.specs])
    roll = np.array([2 * s.cost_one_way * s.rolls_per_year / 252 for s in env.specs])
    tg = targets[names].reindex(dates).ffill().fillna(0.0).values
    regs = targets["regime"].reindex(dates).ffill().fillna("NORMAL").values
    R = env.R.reindex(dates)
    h = np.zeros(len(names))
    last_reg = None
    rec = []
    for i, t in enumerate(dates):
        r = np.nan_to_num(R[cols].iloc[i].values)
        rS = float(np.nan_to_num(R["SMLT"].iloc[i]))
        pnl_h = env.E * float(h @ r)
        costs = env.E * float(roll @ np.abs(h))
        new = tg[i]
        trade = np.zeros_like(h)
        if np.max(np.abs(new - h)) > threshold + 1e-12 or (regs[i] != last_reg and last_reg is not None):
            trade = new - h
        costs += env.E * float(cost @ np.abs(trade))
        carry = env.C * float(np.nan_to_num(env.mmf.get(t, 0.0)))
        rec.append({"date": t, "pnl_spot": env.E * rS, "pnl_hedge": pnl_h, "costs": costs, "carry": carry,
                    "pnl_total": env.E * rS + pnl_h - costs + carry, "turnover": float(np.abs(trade).sum()),
                    "regime": regs[i], **{f"h_{n}": v for n, v in zip(names, h)}})
        h = h + trade
        last_reg = regs[i]
    return pd.DataFrame(rec).set_index("date")


# ---------------------------------------------------------------- episode engine with liquidity waterfall
def episode(env: Env, targets: pd.DataFrame, dates: pd.DatetimeIndex, threshold: float, coll: Collateral = CASH_ONLY,
            liquidity: str = "WATERFALL", restore_frac: float = 0.05) -> dict:
    names = [s.name for s in env.specs]
    cols = [s.series for s in env.specs]
    di = names.index("DIRECT")
    cost = np.array([s.cost_one_way for s in env.specs])
    roll = np.array([2 * s.cost_one_way * s.rolls_per_year / 252 for s in env.specs])
    assets = [s.asset for s in env.specs]
    mult, cpm, epm = coll.margin_multiplier, coll.capital_per_margin(), coll.earning_per_margin()
    R = env.R.reindex(dates)
    S = (1 + R["SMLT"].fillna(0)).cumprod()
    # PIT NCC IM rate per sleeve (last published <= t); before a contract's history starts: cutoff rate (documented)
    mr_frame = pd.DataFrame({a: (env.imr[a] if a in env.imr else pd.Series(np.nan, index=env.imr.index)) for a in assets})
    mr_frame = mr_frame.reindex(mr_frame.index.union(dates)).ffill().reindex(dates)
    mr_rows = mr_frame.fillna(pd.Series({a: s.mr for a, s in zip(assets, env.specs)})).values
    tg = targets[names].reindex(dates).ffill().fillna(0.0).values
    regs = targets["regime"].reindex(dates).ffill().fillna("NORMAL").values
    E0, C = env.E, env.C
    n_sh = 1.0                                   # share count in units of the initial position
    N = np.zeros(len(names))                     # RUB notionals
    free, posted = float(C), 0.0
    actions = {"step3_cut_other": 0, "step4_cut_direct": 0, "step5_sell_shares": 0, "immediate_cut": 0}
    shares_sold_value, total_pnl, costs_sum, turnover = 0.0, 0.0, 0.0, 0.0
    daily, im_path, coll_path, free_path = [], [], [], []
    last_reg = None
    corr_rank = None
    for i, t in enumerate(dates):
        r = np.nan_to_num(R[cols].iloc[i].values)
        rS = float(np.nan_to_num(R["SMLT"].iloc[i]))
        m = float(np.nan_to_num(env.mmf.get(t, 0.0)))
        if i > 0:
            vm = float(N @ r)
            spot = n_sh * E0 * S.iloc[i - 1] * rS
            interest = ledger.interest(free, posted / cpm if cpm else 0.0, m, coll)   # unified ledger (V3 0.4)
            free += vm + interest
            N = N * (1 + r)
            rc = E0 * float(roll @ np.abs(N)) / E0
            free -= rc
            costs_sum += rc
            day = spot + vm + interest - rc
        else:
            day = 0.0
        mr = mr_rows[i]
        # --- scheduled target & threshold trading
        Et = n_sh * E0 * S.iloc[i]
        new = tg[i] * Et
        if i == 0 or np.max(np.abs(tg[i] - N / max(Et, 1e-9))) > threshold + 1e-12 or (regs[i] != last_reg and last_reg is not None):
            tr = new - N
            c = float(cost @ np.abs(tr))
            free -= c
            day -= c
            costs_sum += c
            turnover += float(np.abs(tr).sum()) / E0
            N = new.copy()
        last_reg = regs[i]
        # --- margin and liquidity
        im = float(mr @ np.abs(N)) * mult
        need = im * cpm
        free -= need - posted
        posted = need
        if free < 0:
            deficit = restore_frac * C - free
            if corr_rank is None or i % 5 == 0:
                w = env.R.iloc[max(0, env.R.index.get_loc(t) - 250):env.R.index.get_loc(t)][["SMLT"] + cols].dropna()
                corr_rank = np.argsort([abs(w["SMLT"].corr(w[c])) if len(w) > 30 else 0.0 for c in cols])
            order = ([j for j in corr_rank if j != di] + [di]) if liquidity == "WATERFALL" else ([di] + [j for j in corr_rank if j != di])
            for j in order:
                if deficit <= 0 or N[j] == 0:
                    continue
                release_per = mr[j] * mult * cpm
                cut = min(abs(N[j]), deficit / max(release_per, 1e-9))
                N[j] -= np.sign(N[j]) * cut
                rel = cut * release_per
                c = cut * cost[j]
                free += rel - c
                posted -= rel
                costs_sum += c
                day -= c
                deficit -= rel - c
                key = ("step4_cut_direct" if j == di else "step3_cut_other") if liquidity == "WATERFALL" else "immediate_cut"
                actions[key] += 1
            if deficit > 1.0:                                            # step 5: DE_RISKING, never silent (>1 RUB)
                sell = deficit / (E0 * S.iloc[i])
                sell = min(sell, n_sh)
                n_sh -= sell
                free += sell * E0 * S.iloc[i]
                shares_sold_value += sell * E0 * S.iloc[i]
                actions["step5_sell_shares"] += 1
        total_pnl += day
        daily.append(day)
        im_path.append(im)
        coll_path.append(posted)
        free_path.append(free)
    daily = np.array(daily)
    cum = np.cumsum(daily)
    return {"terminal_pnl": float(total_pnl), "max_dd": float((np.maximum.accumulate(np.concatenate([[0], cum])) - np.concatenate([[0], cum])).max()),
            "costs": costs_sum, "turnover": turnover, "peak_im": float(max(im_path)), "peak_posted": float(max(coll_path)),
            "min_free": float(min(free_path)), "breach": bool(min(free_path) < 0) or sum(actions.values()) > 0,
            "shares_sold_value": shares_sold_value, "final_share_fraction": n_sh, **actions}


def select_threshold_causally(series: dict[float, pd.DataFrame], es_fn) -> pd.DataFrame:
    """At each quarter start pick the threshold with the lowest realised daily ES95 over PRIOR OOS days."""
    keys = sorted(series)
    idx = series[keys[0]].index
    rows = []
    for qs in pd.date_range(idx.min(), idx.max(), freq="QS"):
        prior = idx[idx < qs]
        best = keys[len(keys) // 2]
        if len(prior) >= 60:
            best = min(keys, key=lambda k: es_fn(series[k].loc[prior, "pnl_total"].values) + 1e-12 * k)
        rows.append({"quarter_start": qs, "threshold": best})
    return pd.DataFrame(rows).set_index("quarter_start")
