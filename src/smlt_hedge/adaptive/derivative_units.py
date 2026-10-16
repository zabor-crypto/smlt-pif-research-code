"""D18 authoritative gross fixed-count derivative horizon units. Fees are separate.

Market Fp/Fn/pv is the economic authority; the recursive engine remains final admission authority.
History never compresses missing sessions. A missing origin or clearing invalidates its whole window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SERIES = {"OIL_FUT": "OIL_FUT_USD"}
DERIVATIVES = ("SMLT_FUT", "EQ_HEDGE_FUT", "RATES_HEDGE_FUT", "CNY_PERP", "OIL_FUT")


def vm_increments(Fp: np.ndarray, Fn: np.ndarray, pv: np.ndarray) -> np.ndarray:
    """RUB clearing per one fixed long contract, arrays (..., T, Ad). No roll-gap VM or fee."""
    return (Fp[..., 1:, :] - Fn[..., :-1, :]) * pv[..., 1:, :]


def horizon(market) -> np.ndarray:
    """(..., Ad) gross cumulative VM / positive initial derivative notional, q=+1."""
    initial = market.Fn[..., 0, :] * market.pv[..., 0, :]
    if not np.isfinite(initial).all() or (initial <= 0).any():
        raise ValueError("invalid initial derivative notional")
    return vm_increments(market.Fp, market.Fn, market.pv).sum(axis=-2) / initial


def historical_forward(terms: pd.DataFrame, H: int) -> pd.Series:
    """Origin t -> exact t+1..t+H clearing; every session including origin must be observed."""
    a = terms[["Fp", "Fn", "pv"]].to_numpy(float)[None, :, :]
    vm = vm_increments(a[:, :, 0:1], a[:, :, 1:2], a[:, :, 2:3])[0, :, 0]
    daily = pd.Series(np.r_[np.nan, vm], index=terms.index)
    valid = terms[["Fn", "pv"]].notna().all(axis=1) & (terms["Fn"] > 0) & (terms["pv"] > 0)
    # Nonavailable days are never filled; missing day t terms invalidate the t+1 clearing too.
    daily = daily.where(valid & valid.shift(1, fill_value=False))
    sums = daily.iloc[::-1].rolling(H, min_periods=H).sum().iloc[::-1].shift(-1)
    return (sums / (terms["Fn"] * terms["pv"])).where(valid)


def path_market(ps, dspecs):
    """Derivative-only Market using the same forward transport as full admission."""
    from . import market as M
    idx = {r: k for k, r in enumerate(ps.routes)}
    N, H = ps.R.shape[:2]
    rd = np.stack([ps.R[:, :, idx[SERIES.get(d.name, d.name)]] for d in dspecs], axis=2)
    fx = ps.R[:, :, idx["USDRUB"]] if "USDRUB" in idx else None
    return M.forward_market([], dspecs, np.zeros((N, H, 0)), rd, fx, ps.mmf_daily,
                            cny_swap_frac=ps.cny_swap_frac)[2]


def default_specs(ps):
    """Explicit normalized synthetic convention: F0=pv0=1, no rolls. Canonical callers supply actual specs."""
    from . import market as M
    return [M.DerivSpec(r, 1., 1., r == "OIL_FUT", 0., 0., 0.)
            for r in DERIVATIVES if SERIES.get(r, r) in ps.routes]


def centre_route(ps, spec, column: int, target: float, tol: float = 1e-12):
    """One bounded log shift, shared by all paths; auxiliary and FX are immutable.

    Deterministic bracket [-0.05,0.05]. All nine fixed nodes must be finite, with one target
    interval and positive analytic slope at its endpoints and root. Positive settlement is required.
    Slopes outside the target interval are diagnostic; failures never trigger a pathwise fit.
    """
    from scipy.optimize import brentq
    from . import scenarios as SC
    raw = ps.R[:, :, column].copy()
    if not np.isfinite(raw).all() or (raw <= -1).any():
        raise ValueError("invalid daily factor for derivative centring")
    lo, hi = -.05, .05
    evaluations = 0
    def evaluate(a):
        nonlocal evaluations
        evaluations += 1
        rr = (1 + raw) * np.exp(a) - 1
        sub = SC.PathSet(ps.model, rr[:, :, None], ps.mmf_daily, [SERIES.get(spec.name, spec.name)], {},
                         ps.cny_swap_frac)
        if spec.usd_linked and "USDRUB" in ps.routes:
            sub.R = np.concatenate([sub.R, ps.R[:, :, ps.routes.index("USDRUB")][:, :, None]], axis=2)
            sub.routes.append("USDRUB")
        m = path_market(sub, [spec])
        if not np.isfinite(m.Fn).all() or (m.Fn <= 0).any():
            raise ValueError("nonpositive settlement in centring bracket")
        # Exact derivative of the same transport recurrence, with raw auxiliary held fixed.
        base = np.full(len(raw), spec.F0); tangent = np.zeros(len(raw)); deriv = np.zeros(len(raw))
        for t in range(raw.shape[1]):
            dr = 1 + rr[:, t]
            deriv += (tangent * rr[:, t] + base * dr) * m.pv[:, t + 1, 0]
            swap = ps.cny_swap_frac[:, t] if spec.name == "CNY_PERP" else 0.
            growth = 1 + rr[:, t] + swap
            roll = np.exp(spec.roll_gap) if t + 1 in spec.roll_days else 1.
            tangent, base = (tangent * growth + base * dr) * roll, base * growth * roll
        slope = float((deriv / (spec.F0 * spec.pv0)).mean())
        return float(horizon(m)[:, 0].mean()), slope
    nodes = [evaluate(a) for a in np.linspace(lo, hi, 9)]
    slopes = [s for _, s in nodes]
    if not np.isfinite(nodes).all():
        raise ValueError("derivative centring fixed nodes are nonfinite")
    grid = np.linspace(lo, hi, 9)
    residuals = [mean - target for mean, _ in nodes]
    # An exact root shared by adjacent cells is one target root, counted once.
    intervals = [(i, i) for i, residual in enumerate(residuals) if residual == 0]
    intervals += [(i, i + 1) for i in range(8) if residuals[i] * residuals[i + 1] < 0]
    if not intervals:
        raise ValueError("derivative centring has no bracket")
    if len(intervals) != 1:
        raise ValueError("derivative centring has multiple target brackets")
    left, right = intervals[0]
    if slopes[left] <= 0 or slopes[right] <= 0:
        raise ValueError("derivative centring target-bracket slope is nonpositive")
    fl, fh = nodes[0][0] - target, nodes[-1][0] - target
    if fl > 0 or fh < 0:
        raise ValueError("derivative centring has no bracket")
    shift, result = brentq(lambda a: evaluate(a)[0] - target, lo, hi, xtol=5e-16, rtol=1e-14,
                           maxiter=100, full_output=True)
    achieved, slope = evaluate(shift)
    if not np.isfinite([achieved, slope]).all() or slope <= 0:
        raise ValueError("derivative centring root slope is nonpositive or nonfinite")
    if not grid[left] - 1e-15 <= shift <= grid[right] + 1e-15:
        raise ValueError("derivative centring root outside target bracket")
    if abs(achieved - target) > tol:
        raise ValueError("derivative centring engine-unit tolerance failed")
    return (1 + raw) * np.exp(shift) - 1, shift, {
        "model": ps.model, "route": spec.name, "target_engine_unit": float(target), "shift": shift,
        "achieved_mean_engine_unit": achieved, "error": achieved - target, "bracket": [lo, hi],
        "bracket_residuals": [fl, fh], "iterations": result.iterations, "evaluations": evaluations,
        "monotonicity": "positive analytic mean transport derivative at unique target endpoints and root",
        "target_bracketing_interval_count": len(intervals),
        "target_bracketing_interval": [float(grid[left]), float(grid[right])],
        "fixed_node_diagnostics": [{"a": float(a), "engine_mean": mean, "target_residual": mean - target,
                                    "analytic_derivative": derivative}
                                   for a, (mean, derivative) in zip(grid, nodes)],
        "minimum_mean_derivative_at_nodes": min(slopes), "mean_derivative_at_root": slope, "status": "PASS"}
