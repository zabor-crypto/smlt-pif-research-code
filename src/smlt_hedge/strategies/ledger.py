"""V3 Phase 0.4: ONE recursive self-financing collateral ledger shared by every engine.

State of the hedge account (RUB), decisions at the close of each day:
    A_t      = total hedge-account value (free cash + posted collateral capital), own money only
    IM_t     = exchange/broker initial-margin requirement after the close of t (multiplier included)
    posted_t = IM_t * capital_per_margin          (capital locked as collateral)
    free_t   = A_t - posted_t                      (free liquidity; < min_buffer == SLEEVE_BREACH)
Recursion (t = 1..H):
    interest_t = free_{t-1} * m_t  +  IM_{t-1} * earning_per_margin * m_t
    A_t        = A_{t-1} + interest_t + VM_t - costs_t + cash_in_t
Negative free liquidity is financed at the same money-market rate (a shortfall is funded, never ignored); every
such day is a breach. Productive-collateral income therefore feeds FUTURE free liquidity and future breach
probability - no engine may credit collateral income to terminal P&L without passing it through this recursion.

Users: scenarios/bootstrap.evaluate_portfolio (vectorised over paths), scenarios/deterministic (a single
deterministic path through `run`), portfolio/adaptive.episode and strategies/futures.simulate_hedge (daily loops
through `interest`).
"""
from __future__ import annotations

import numpy as np

from .collateral import CASH_ONLY, Collateral


def interest(free_prev, im_prev, m, coll: Collateral = CASH_ONLY):
    """Money-market income earned over one day on free cash and on productive posted collateral."""
    return free_prev * m + im_prev * coll.earning_per_margin() * m


def run(*, A0, im0, vm, im, m, coll: Collateral = CASH_ONLY, costs=None, cash_in=None) -> dict:
    """Vectorised recursion. vm, im, costs, cash_in: (N, H) arrays for days 1..H; m: (H,) daily rates;
    A0: scalar or (N,) account value at t0 after t0 outlays; im0: scalar or (N,) IM at t0."""
    vm = np.atleast_2d(np.asarray(vm, float))
    N, H = vm.shape
    im = np.broadcast_to(np.asarray(im, float), (N, H))
    m = np.broadcast_to(np.asarray(m, float), (H,))
    costs = np.zeros((N, H)) if costs is None else np.broadcast_to(np.asarray(costs, float), (N, H))
    cash_in = np.zeros((N, H)) if cash_in is None else np.broadcast_to(np.asarray(cash_in, float), (N, H))
    cpm = coll.capital_per_margin()
    A = np.empty((N, H))
    free = np.empty((N, H))
    intr = np.empty((N, H))
    a = np.broadcast_to(np.asarray(A0, float), (N,)).astype(float).copy()
    im_prev = np.broadcast_to(np.asarray(im0, float), (N,)).astype(float).copy()
    f_prev = a - im_prev * cpm
    free0 = f_prev.copy()
    for t in range(H):
        r = interest(f_prev, im_prev, m[t], coll)
        a = a + r + vm[:, t] - costs[:, t] + cash_in[:, t]
        A[:, t] = a
        intr[:, t] = r
        im_prev = im[:, t]
        f_prev = a - im_prev * cpm
        free[:, t] = f_prev
    return {"A": A, "free": free, "interest": intr, "posted": im * cpm, "free0": free0}
