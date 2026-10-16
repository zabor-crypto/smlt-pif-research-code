"""Stochastic risk engine: circular block bootstrap of JOINT daily returns (cross-sectional dependence
and short-range serial dependence preserved within blocks). Seeded; no return forecasting.

Path P&L for static overlays (positions set at t0 and held; contracts fixed):
  spot:     E * (S_t/S_0 - 1)
  futures:  -h * E * (F_t/F_0 - 1)   (h in share units; F follows the continuous-contract returns;
                                       carry/basis enters through the sampled futures returns)
  linear sleeve i: w_i * E * (P_i,t/P_i,0 - 1)
  liquidity sleeve: (C - IM_t - options premium) earns m_t; IM earns 0 (conservative)
Liquidity metrics per path: max daily VM outflow, max cumulative VM drain, peak IM, peak collateral,
and SLEEVE_BREACH = free sleeve < 0 at any day (the hedge would have to be cut).
The circular block bootstrap follows the pattern in
_autonomous_strategy_discovery_20260905/lib/stats.py::block_bootstrap_mean (re-implemented).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..risk.metrics import es_hist, var_hist
from ..strategies.options import black76


def block_bootstrap_indices(n_obs: int, horizon: int, n_paths: int, block: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(horizon / block))
    starts = rng.integers(0, n_obs, size=(n_paths, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_paths, nb * block) % n_obs
    return idx[:, :horizon]


def paths(R: pd.DataFrame, cols: list[str], horizon: int, n_paths: int, block: int, seed: int) -> dict[str, np.ndarray]:
    X = R[cols].dropna()
    idx = block_bootstrap_indices(len(X), horizon, n_paths, block, seed)
    return {c: X[c].values[idx] for c in cols} | {"_n_obs": len(X), "_first": X.index.min(), "_last": X.index.max(),
                                                  "_idx": idx, "_dates": X.index}


def structural_futures(P: dict, basis_resid: np.ndarray, carry_ann: float, e0: float) -> np.ndarray:
    """Futures daily returns implied by F_t/F_0 = (S_t/S_0) * exp(-b t/252) * exp(e_t - e0).

    The log-basis is STATIONARY (empirical residual sampled at the same bootstrap dates) instead of the random walk
    that arises when spot and futures returns are compounded separately. basis_resid is aligned to the sample rows.
    """
    S = cum_level(P["SMLT"])
    N, H = S.shape
    e = basis_resid[P["_idx"]]
    t = np.arange(1, H + 1)[None, :]
    F = S * np.exp(-carry_ann * t / 252.0) * np.exp(e - e0)
    prev = np.concatenate([np.ones((N, 1)), F[:, :-1]], axis=1)
    return F / prev - 1.0


def cum_level(r: np.ndarray) -> np.ndarray:
    """(N,H) daily returns -> (N,H) level relative to t0 (=1 at t0, excluded)."""
    return np.cumprod(1.0 + r, axis=1)


from dataclasses import dataclass

from ..strategies.collateral import CASH_ONLY, Collateral  # noqa: F401  (re-exported)




@dataclass
class SleevePos:
    """One overlay position. notional = signed RUB exposure at t0 (short < 0).
    kind: 'future' (margin + VM, no capital outlay) or 'cash' (bought with sleeve cash, no margin)."""
    series: str
    notional: float
    im_rate: float = 0.0
    cost_one_way: float = 0.0
    n_rolls: int = 0
    kind: str = "future"


def _roll_days(H: int, n_rolls: int) -> list[int]:
    """Deterministic roll dates inside the horizon (0-based day index), equally spaced."""
    return [int(round(H * (k + 1) / (n_rolls + 1))) - 1 for k in range(n_rolls)] if n_rolls > 0 else []


def control_pnl(P: dict, *, E: float, C: float, mmf_daily: float | np.ndarray = 0.0) -> np.ndarray:
    """Same-capital CONTROL terminal P&L per path: unhedged SMLT + hedge capital in the money market."""
    S = cum_level(P["SMLT"])
    H = S.shape[1]
    m = np.broadcast_to(np.asarray(mmf_daily, float), (H,))
    return E * (S[:, -1] - 1.0) + C * (np.prod(1.0 + m) - 1.0)


def capture(term: np.ndarray, ctrl: np.ndarray, s_up: np.ndarray) -> tuple[float, float]:
    """Upside/downside capture vs the SAME-CAPITAL CONTROL (both include liquidity carry). States are split by the
    SMLT price move over the horizon. CONTROL has upside_capture == downside_capture == 1 by construction."""
    dn = ~s_up
    upc = float(term[s_up].mean() / ctrl[s_up].mean()) if s_up.any() and ctrl[s_up].mean() != 0 else np.nan
    dnc = float(term[dn].mean() / ctrl[dn].mean()) if dn.any() and ctrl[dn].mean() != 0 else np.nan
    return upc, dnc


def evaluate_portfolio(P: dict, *, E: float, C: float, sleeves: list[SleevePos] | None = None,
                       mmf_daily: float | np.ndarray = 0.0, derisk: float = 0.0, derisk_cost: float = 0.0,
                       put: dict | None = None, collateral: Collateral = CASH_ONLY, min_buffer: float = 0.0,
                       return_paths: bool = False, attribution: bool = False) -> dict:
    """Self-financing static overlay on scenario paths, through the unified recursive ledger (strategies/ledger.py).

    Account at t0: A0 = C + E*d - premium - cash-asset purchases - entry costs - de-risking costs.
    Daily: VM of futures sleeves, IM_t = sum_i mr_i * mult * |N_i| * L_i,t, roll costs on roll days, exit costs on
    the last day; interest on free cash and on productive collateral is credited recursively (it changes future
    free liquidity). SLEEVE_BREACH: free_t < min_buffer on any day (t0 included).
    P&L_t = E(1-d)(S_t - 1) + A_t + option value_t + cash-asset value_t - C - E*d.
    """
    from ..strategies import ledger
    sleeves = sleeves or []
    S = cum_level(P["SMLT"])
    N, H = S.shape
    E_eff = E * (1.0 - derisk)
    spot_pnl = E_eff * (S - 1.0)
    m = np.broadcast_to(np.asarray(mmf_daily, float), (H,)) if np.ndim(mmf_daily) == 0 else np.asarray(mmf_daily)
    im = np.zeros_like(S)
    fut_only = np.zeros_like(S)
    cash_val = np.zeros_like(S)
    costs = np.zeros_like(S)
    cash_cost0, entry, later, im0 = 0.0, 0.0, 0.0, 0.0
    per: dict[str, np.ndarray] = {}
    for sp in sleeves:
        L = cum_level(P[sp.series])
        pnl = sp.notional * (L - 1.0)
        per[sp.series] = per.get(sp.series, 0.0) + pnl
        c1 = abs(sp.notional) * sp.cost_one_way
        entry += c1
        costs[:, -1] += c1                                   # exit
        for d in _roll_days(H, sp.n_rolls):
            costs[:, d] += 2 * c1                            # close + reopen at the roll
        later += c1 * (1 + 2 * sp.n_rolls)
        if sp.kind == "future":
            fut_only += pnl
            im += sp.im_rate * collateral.margin_multiplier * abs(sp.notional) * L
            im0 += sp.im_rate * collateral.margin_multiplier * abs(sp.notional)
        else:
            cash_val += sp.notional * L
            cash_cost0 += sp.notional
    vm = np.diff(np.concatenate([np.zeros((N, 1)), fut_only], axis=1), axis=1)
    premium, opt_val = 0.0, np.zeros_like(S)
    if put:
        K, T, iv, r = put["K"], put["T"], put["iv"], put.get("r", 0.14)
        hr = put.get("hedge_ratio", 1.0)
        premium = black76(1.0, K, T, iv, r, "P") * E_eff * hr
        if put.get("short_call_K"):
            premium -= black76(1.0, put["short_call_K"], T, iv, r, "C") * E_eff * hr
        from scipy.stats import norm
        tau_left = np.maximum(T - np.arange(1, H + 1) / 252.0, 0.0)
        sd = iv * np.sqrt(np.maximum(tau_left, 1e-12))[None, :]
        d1 = (np.log(S / K) + 0.5 * sd**2) / sd
        disc = np.exp(-r * tau_left)[None, :]
        pv = disc * (K * norm.cdf(-(d1 - sd)) - S * norm.cdf(-d1))
        pv = np.where(tau_left[None, :] <= 1e-9, np.maximum(K - S, 0.0), pv)
        if put.get("short_call_K"):
            Kc = put["short_call_K"]
            d1c = (np.log(S / Kc) + 0.5 * sd**2) / sd
            cv = disc * (S * norm.cdf(d1c) - Kc * norm.cdf(d1c - sd))
            cv = np.where(tau_left[None, :] <= 1e-9, np.maximum(S - Kc, 0.0), cv)
            pv = pv - cv
        opt_val = pv * E_eff * hr
    d_cost = E * derisk * derisk_cost
    A0 = C + E * derisk - premium - cash_cost0 - entry - d_cost
    led = ledger.run(A0=A0, im0=im0, vm=vm, im=im, m=m, coll=collateral, costs=costs)
    A, free = led["A"], led["free"]
    total = spot_pnl + A + opt_val + cash_val - C - E * derisk
    cumvm = np.cumsum(vm, axis=1)
    interest = np.cumsum(led["interest"], axis=1)
    tc_cum = entry + d_cost + np.cumsum(costs, axis=1)
    daily = np.diff(np.concatenate([np.full((N, 1), -(entry + d_cost)), total], axis=1), axis=1)
    term = total[:, -1]
    peak = np.maximum.accumulate(np.concatenate([np.zeros((N, 1)), total], axis=1), axis=1)[:, 1:]
    mdd = (peak - total).max(axis=1)
    drain = (np.maximum.accumulate(np.concatenate([np.zeros((N, 1)), cumvm], axis=1), axis=1)[:, 1:] - cumvm).max(axis=1)
    ctrl = control_pnl(P, E=E, C=C, mmf_daily=m)
    upc, dnc = capture(term, ctrl, S[:, -1] > 1.0)
    posted = led["posted"]
    min_free = np.minimum(free.min(axis=1), led["free0"])
    out = {
        "mean_pnl": float(term.mean()), "median_pnl": float(np.median(term)),
        "VaR95": var_hist(term, 0.95), "ES95": es_hist(term, 0.95), "VaR99": var_hist(term, 0.99),
        "ES99": es_hist(term, 0.99), "std_pnl": float(term.std(ddof=1)) if N > 1 else 0.0,
        "p_loss": float((term < 0).mean()), "p05_pnl": float(np.quantile(term, 0.05)),
        "p95_pnl": float(np.quantile(term, 0.95)), "min_pnl": float(term.min()),
        "mean_max_dd": float(mdd.mean()), "p95_max_dd": float(np.quantile(mdd, 0.95)),
        "upside_capture": upc, "downside_capture": dnc,
        "max_daily_vm_outflow_p95": float(np.quantile(np.maximum(-vm.min(axis=1), 0), 0.95)),
        "max_daily_vm_outflow_max": float(np.maximum(-vm.min(axis=1), 0).max()),
        "max_cum_vm_drain_p95": float(np.quantile(drain, 0.95)),
        "max_cum_vm_drain_p99": float(np.quantile(drain, 0.99)),
        "peak_im_p95": float(np.quantile(im.max(axis=1), 0.95)),
        "peak_collateral_p95": float(np.quantile((posted + (np.maximum.accumulate(cumvm, axis=1) - cumvm)).max(axis=1), 0.95)),
        "initial_im": float(im0),
        "initial_capital_used": float(im0 * collateral.capital_per_margin() + cash_cost0 + max(premium, 0.0)),
        "p_sleeve_breach": float((min_free < min_buffer).mean()),
        "min_free_p05": float(np.quantile(min_free, 0.05)),
        "option_premium": float(premium),
        "transaction_costs": float(entry + later + d_cost),
        "collateral_income_mean": float(interest[:, -1].mean()),
        "gross_derivatives_notional": float(sum(abs(sp.notional) for sp in sleeves if sp.kind == "future")),
        "n_active_sleeves": int(sum(1 for sp in sleeves if abs(sp.notional) > 1e-9)),
        "worst_1d_p95": float(np.quantile(-daily.min(axis=1), 0.95)),
        "n_paths": int(N), "horizon_days": int(H),
    }
    if attribution:
        tail = term <= np.quantile(term, 0.05)
        var_t = float(np.var(term, ddof=1))
        up, dn = S[:, -1] > 1.0, S[:, -1] <= 1.0
        comps = {"SMLT_SPOT": spot_pnl[:, -1], "LIQUIDITY_INTEREST": interest[:, -1], "COSTS": -tc_cum[:, -1]}
        comps.update({k: v[:, -1] for k, v in per.items()})
        if put:
            comps["OPTIONS"] = opt_val[:, -1] - premium
        out["attribution"] = {k: {"mean": float(v.mean()), "var_contrib": float(np.cov(v, term, ddof=1)[0, 1] / var_t) if var_t > 0 else np.nan,
                                  "es95_contrib": float(-v[tail].mean()), "up_state_mean": float(v[up].mean()) if up.any() else np.nan,
                                  "down_state_mean": float(v[dn].mean()) if dn.any() else np.nan} for k, v in comps.items()}
    if return_paths:
        out["_free"], out["_total"], out["_im"], out["_term"], out["_ctrl"] = free, total, im, term, ctrl
    return out


def evaluate_static(P: dict, *, E: float, C: float, h_fut: float = 0.0, lin: dict | None = None,
                    im_rate_fut: float = 0.5, im_rate_lin: dict | None = None, mmf_daily: float | np.ndarray = 0.0,
                    derisk: float = 0.0, put: dict | None = None, alpha: float = 0.95,
                    fut_cost_one_way: float = 0.0, n_rolls: int = 0, derisk_cost: float = 0.0,
                    lin_cost_one_way: dict | None = None, fut_notional: float | None = None,
                    collateral: Collateral = CASH_ONLY) -> dict:
    """Backward-compatible wrapper. fut_notional = exact contract notional q*F0*point_value (RUB); if None the
    legacy approximation h*E*(1-d) is used (kept only for the v1 tests)."""
    lin = lin or {}
    im_rate_lin = im_rate_lin or {}
    lin_cost_one_way = lin_cost_one_way or {}
    Nf = -(fut_notional if fut_notional is not None else h_fut * E * (1.0 - derisk))
    sl = []
    if h_fut:
        sl.append(SleevePos("SMLT_FUT", Nf, im_rate_fut, fut_cost_one_way, n_rolls, "future"))
    for k, w in lin.items():
        sl.append(SleevePos(k, w * E, im_rate_lin.get(k, 0.15), lin_cost_one_way.get(k, 0.0), 0, "future"))
    return evaluate_portfolio(P, E=E, C=C, sleeves=sl, mmf_daily=mmf_daily, derisk=derisk, derisk_cost=derisk_cost,
                              put=put, collateral=collateral)


def horizon_returns(P: dict, cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: np.prod(1.0 + P[c], axis=1) - 1.0 for c in cols})
