"""Settlement-aware conversion of a target into executable orders (shared by admission, policy and replay).

Rules (EXECUTION_ACCOUNTING_SPEC.md §4):
  * targets are integer / lot quantities for RISKY funded assets and derivatives; money-market (MMF) units are never
    bought by a target order — excess settled cash is swept into MMF by the engine's cash rule;
  * net funded purchases and additional initial margin are funded, in this order, by (a) immediately available cash
    above the target reserve, (b) sale proceeds of the same order haircut by the asset's 1-session 99% loss
    (decision-time scenarios), (c) MMF sales;
  * funded legs and MMF sales settle on the same exchange cycle, so they net at settlement; additional IM, however,
    must be posted from IMMEDIATE cash at the entry clearing — if (a) does not cover it, derivative entries are
    scheduled one settlement cycle after the MMF sale (two-step order);
  * if (a)+(b)+(c) cannot fund the purchases, all risky purchases are scaled down pro rata (recorded PARTIAL).
"""
from __future__ import annotations

import numpy as np

from . import engine as E


def build(day: int, qf: np.ndarray, qd: np.ndarray, tgt_f: np.ndarray, tgt_d: np.ndarray, P: np.ndarray,
          notional: np.ndarray, mr: np.ndarray, mult: float, fin: list, available: float, nav: float,
          target_frac: float, haircut: np.ndarray, lag_cycle: int) -> tuple[list, dict]:
    """All arrays for ONE path at decision time. Returns (orders, info)."""
    kinds = np.array([i.kind for i in fin])
    mm = kinds == E.MMF
    cprop = np.array([i.cost_prop for i in fin])
    cunit = np.array([i.cost_unit for i in fin])
    tf = np.array(tgt_f, float).copy()
    tf[mm] = qf[mm]
    dq = np.where(mm, 0.0, tf - qf)
    buys = float((np.maximum(dq, 0) * (P * (1 + cprop) + cunit)).sum())
    sells = float((np.maximum(-dq, 0) * (P * (1 - haircut - cprop) - cunit)).sum())
    dim = float(mult * ((np.abs(tgt_d) - np.abs(qd)) * notional * mr).sum())
    free = available - target_frac * nav
    info = {"buys": buys, "sells_haircut": sells, "delta_im": dim, "free_above_target": free, "partial": False,
            "two_step": False, "mmf_sold_units": 0.0}
    need = buys - sells + max(dim, 0.0) - max(free, 0.0)
    mmf_sell = np.zeros_like(qf)
    if need > 0 and mm.any():
        j = int(np.flatnonzero(mm)[0])
        unit = P[j] * (1 - haircut[j])
        units = min(qf[j], np.ceil(need / max(unit, 1e-12)))
        mmf_sell[j] = units
        need -= units * unit
        info["mmf_sold_units"] = float(units)
    if need > 1e-6 and buys > 0:
        scale = max(0.0, 1 - need / buys)
        lots = np.array([i.lot for i in fin])
        add = np.maximum(dq, 0) * scale
        add = np.where(lots > 0, np.floor(add / np.where(lots > 0, lots, 1)) * np.where(lots > 0, lots, 1), add)
        tf = np.where(dq > 0, qf + add, tf)
        info["partial"] = True
        info["purchase_scale"] = scale
    tf_mm = tf.copy()
    tf_mm[mm] = qf[mm] - mmf_sell[mm]
    immediate_ok = max(dim, 0.0) <= max(free, 0.0) + 1e-6
    if immediate_ok:
        return [E.Order(day, tf_mm, np.array(tgt_d, float))], info
    # two-step: funded legs + MMF sale now; derivative entries after the MMF proceeds settle
    info["two_step"] = True
    red = np.where(np.abs(tgt_d) < np.abs(qd), tgt_d, qd)          # reductions execute immediately
    return [E.Order(day, tf_mm, red), E.Order(day + lag_cycle, np.full_like(qf, np.nan), np.array(tgt_d, float))], info
