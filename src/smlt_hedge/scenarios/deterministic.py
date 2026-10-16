"""Deterministic scenario engine: SCENARIOS, not forecasts.

Axes
  x      : terminal SMLT price return over the horizon (grid from config)
  rates  : DOWN / STABLE / UP   -> money-market yield path, RGBI yield change, index drift
  credit : IMPROVING / STABLE / WORSENING -> futures IM rate (NCC market-risk rate), execution spread
Event overlays apply a one-day SMLT jump j followed by a terminal outcome, to measure the interim
variation-margin shock separately from the terminal P&L.

First-order payoff algebra per strategy (E = SMLT exposure, C = hedge capital, tau = years):
  spot            E(1-d) x
  short futures   -h E(1-d) x  + h E(1-d) b tau                (b = annualised futures basis at cutoff)
                  - h E(1-d) (entry + rolls) cost              (costs scale with the credit state)
  IM drag         the IM (= mr * h * E(1-d) * (1 + x/2) on average) earns 0 instead of y
  index hedge     -h_I E r_I,  r_I = beta_I|S x + delta_rates  (beta from trailing 250d; delta = assumption)
  RGBI leg        w E r_RGBI,  r_RGBI = -D_mod dy + y_rgbi tau (D_mod from MOEX RGBI DURATION field)
  put (OTC)       E(1-d) hr max(K - (1+x), 0) - premium(IV scenario) (premium financed from the sleeve)
  sleeve          (C + E d - premium - IM) y tau
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..strategies.collateral import CASH_ONLY, Collateral
from ..strategies.options import black76


@dataclass
class Market:
    S0: float
    basis_ann: float           # ln(F/S)/tau of the front SMLT future at cutoff
    y_mmf: float               # liquidity-sleeve annual yield (trailing, PIT)
    mr_base: float             # NCC market-risk rate for SMLT futures at cutoff
    mr_stress: float           # next NCC level (mr2) used in the WORSENING credit state
    mr_relief: float           # IM rate in the IMPROVING state (historical median, PIT)
    beta_idx: float            # beta of index return on SMLT return (for conditional index scenario)
    index_basis_ann: float
    rgbi_dmod: float
    rgbi_yield: float
    listed_iv: float           # IV backed out from listed SMLT options at cutoff (short-dated)
    realised_vol_250: float
    fut_cost_rate: float       # one-way cost per unit notional (fees + half spread)
    F0: float = 0.0            # front futures settlement at cutoff (exact contract notional = contracts * F0 * pv)
    point_value: float = 1.0
    collateral: Collateral = CASH_ONLY
    rates_shift: dict = field(default_factory=lambda: {"DOWN": -0.02, "STABLE": 0.0, "UP": 0.02})
    rgbi_dy: dict = field(default_factory=lambda: {"DOWN": -0.015, "STABLE": 0.0, "UP": 0.015})
    index_delta: dict = field(default_factory=lambda: {"DOWN": 0.05, "STABLE": 0.0, "UP": -0.05})
    credit_cost_mult: dict = field(default_factory=lambda: {"IMPROVING": 1.0, "STABLE": 1.0, "WORSENING": 2.0})


@dataclass
class Strategy:
    name: str
    klass: str                 # CONTROL | HEDGE | CONVEX | CROSS_HEDGE | DE_RISKING | COMBO
    h_fut: float = 0.0
    h_idx: float = 0.0
    w_rgbi: float = 0.0
    derisk: float = 0.0
    put_K: float | None = None      # strike as fraction of S0
    put_low_K: float | None = None  # put-spread lower strike
    call_K: float | None = None     # collar short call strike
    put_hr: float = 1.0
    iv_source: str = "listed"       # listed | realised
    executability: str = "EXECUTABLE"


def fut_notional(h: float, Ee: float, m: Market) -> float:
    """Exact RUB notional of round(h * shares) contracts at settlement F0 (falls back to h*Ee if F0 unknown)."""
    if h == 0:
        return 0.0
    if not m.F0:
        return h * Ee
    return round(h * Ee / m.S0) * m.F0 * m.point_value


def payoff(st: Strategy, m: Market, x: float, tau: float, rates: str, credit: str, E: float, C: float) -> dict:
    """Terminal payoff of one strategy in one scenario cell. Liquidity is run through the unified recursive
    ledger (strategies/ledger.py, V3 0.4) along a deterministic geometric path from 1 to (1+x): variation margin,
    IM (moving with the futures price), roll/exit costs and money-market income on free cash and productive
    collateral all accrue day by day, so collateral income affects intermediate free liquidity."""
    from ..strategies import ledger
    Ee = E * (1.0 - st.derisk)
    y = max(m.y_mmf + m.rates_shift[rates], 0.0)
    mr = {"IMPROVING": m.mr_relief, "STABLE": m.mr_base, "WORSENING": m.mr_stress}[credit]
    cm = m.credit_cost_mult[credit]
    mult = m.collateral.margin_multiplier
    H = max(int(round(tau * 252)), 1)
    u = np.arange(1, H + 1) / H
    md = np.full(H, (1.0 + y) ** (1.0 / 252.0) - 1.0)
    spot = Ee * x
    rolls = int(np.ceil(tau / 0.25))
    N0 = fut_notional(st.h_fut, Ee, m)
    # continuously rolled short: F_t/F_0 = S_t/S_0 * exp(-b t)   (carry earned through convergence/rolls)
    f_lvl = (1.0 + x) ** u * np.exp(-m.basis_ann * tau * u)
    fut_path = -N0 * (f_lvl - 1.0)
    fut = float(fut_path[-1])
    c1 = N0 * m.fut_cost_rate * cm
    fut_cost = c1 * (2 + 2 * rolls)                       # entry + exit + two legs per roll
    r_idx = m.beta_idx * x + m.index_delta[rates] * (tau / 0.5)
    idx = -st.h_idx * E * r_idx + st.h_idx * E * m.index_basis_ann * tau
    idx_cost = st.h_idx * E * 0.0005 * (1 + 2 * rolls)
    r_rgbi = -m.rgbi_dmod * m.rgbi_dy[rates] + m.rgbi_yield * tau
    rg = st.w_rgbi * E * r_rgbi
    lin_path = (idx + rg) * u                              # index / RGBI legs accrue linearly (first-order)
    im_path = mult * (mr * N0 * f_lvl + 0.12 * st.h_idx * E + 0.11 * abs(st.w_rgbi) * E)
    im0 = mult * (mr * N0 + 0.12 * st.h_idx * E + 0.11 * abs(st.w_rgbi) * E)
    prem, opt = 0.0, 0.0
    if st.put_K is not None:
        iv = m.listed_iv if st.iv_source == "listed" else m.realised_vol_250
        prem = black76(1.0, st.put_K, tau, iv, y, "P")
        pay = max(st.put_K - (1 + x), 0.0)
        if st.put_low_K is not None:
            prem -= black76(1.0, st.put_low_K, tau, iv, y, "P")
            pay -= max(st.put_low_K - (1 + x), 0.0)
        if st.call_K is not None:
            prem -= black76(1.0, st.call_K, tau, iv, y, "C")
            pay -= max((1 + x) - st.call_K, 0.0)
        prem *= Ee * st.put_hr
        opt = pay * Ee * st.put_hr - prem
    derisk_cost = E * st.derisk * 0.0010
    costs = np.zeros(H)
    costs[-1] += c1 + st.h_idx * E * 0.0005
    for k in range(rolls):
        costs[int(round(H * (k + 1) / (rolls + 1))) - 1] += 2 * c1 + 2 * st.h_idx * E * 0.0005
    vm = np.diff(np.concatenate([[0.0], fut_path + lin_path]))
    A0 = C + E * st.derisk - max(prem, 0.0) - c1 - derisk_cost
    led = ledger.run(A0=A0, im0=im0, vm=vm[None, :], im=im_path[None, :], m=md, coll=m.collateral, costs=costs[None, :])
    sleeve = float(led["interest"].sum())
    min_free = float(min(led["free"].min(), led["free0"][0]))
    total = spot + fut - fut_cost + idx - idx_cost + rg + opt + sleeve - derisk_cost
    return {"pnl": total, "spot": spot, "futures": fut - fut_cost, "index": idx - idx_cost, "rgbi": rg,
            "options": opt, "sleeve": sleeve, "im_avg": float(im_path.mean()), "premium": prem, "futures_notional": N0,
            "collateral_shortfall": max(-min_free, 0.0), "min_free_liquidity": min_free}


def scenario_matrix(strats: list[Strategy], m: Market, grid: list[float], horizons: dict, E: float, C: float) -> pd.DataFrame:
    rows = []
    for hname, tau in horizons.items():
        for st in strats:
            for x in grid:
                for rates in ("DOWN", "STABLE", "UP"):
                    for credit in ("IMPROVING", "STABLE", "WORSENING"):
                        p = payoff(st, m, x, tau, rates, credit, E, C)
                        rows.append({"horizon": hname, "strategy": st.name, "class": st.klass,
                                     "executability": st.executability, "smlt_terminal_return": x,
                                     "rates": rates, "credit": credit, "pnl_rub": p["pnl"],
                                     "pnl_pct_of_smlt_10m": p["pnl"] / E, "pnl_pct_of_total_capital": p["pnl"] / (E + C),
                                     **{f"comp_{k}": v for k, v in p.items() if k != "pnl"}})
    return pd.DataFrame(rows)


EVENTS = {
    # name: (one-day jump, terminal return over horizon) -- magnitudes are ASSUMPTIONS anchored to
    # observed SMLT daily extremes (e.g. +21% on 2026-07-21, -13.5% on 2026-09-15) and credit history.
    "strategic_investor": (0.40, 0.60),
    "state_support": (0.25, 0.35),
    "m_and_a_tender": (0.30, 0.40),
    "restructuring": (-0.20, -0.50),
    "equity_dilution_spo": (-0.25, -0.30),
    "adverse_credit_event_default": (-0.50, -0.85),
}


def event_overlay(strats: list[Strategy], m: Market, horizons: dict, E: float, C: float) -> pd.DataFrame:
    rows = []
    for hname, tau in horizons.items():
        for ev, (j, xT) in EVENTS.items():
            credit = "WORSENING" if xT < 0 else "STABLE"
            for st in strats:
                Ee = E * (1 - st.derisk)
                N0 = fut_notional(st.h_fut, Ee, m)
                mult, cpm = m.collateral.margin_multiplier, m.collateral.capital_per_margin()
                vm_day1 = -N0 * j - st.h_idx * E * m.beta_idx * j
                im0 = mult * (m.mr_base * N0 + 0.12 * st.h_idx * E + 0.11 * abs(st.w_rgbi) * E)
                # after a jump NCC can move the IM rate to the stress level (mr2)
                im1 = mult * (m.mr_stress * N0 * (1 + j) + 0.12 * st.h_idx * E + 0.11 * abs(st.w_rgbi) * E)
                prem = payoff(st, m, xT, tau, "STABLE", credit, E, C)["premium"]
                free_after = C + E * st.derisk - max(prem, 0.0) - im1 * cpm + vm_day1
                p = payoff(st, m, xT, tau, "STABLE", credit, E, C)
                rows.append({"horizon": hname, "event": ev, "jump": j, "terminal_return": xT, "strategy": st.name,
                             "class": st.klass, "day1_vm_rub": vm_day1, "im_before": im0, "im_after_stress": im1,
                             "free_sleeve_after_day1": free_after, "liquidity_breach": free_after < 0,
                             "terminal_pnl_rub": p["pnl"], "terminal_pnl_pct_smlt": p["pnl"] / E})
    return pd.DataFrame(rows)
