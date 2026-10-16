"""V4 fund-level frontier LPs (all quantities per unit of NAV0; horizon scenarios from several models).

Position vector x = funded weights w_f (share of NAV, >= 0) and derivative notionals d_i (share of NAV, signed).
Horizon P&L per NAV in scenario s (money-market horizon return m_s):
    pnl_s = m_s + sum_f w_f (R_fs - m_s) + sum_i d_i R_is
            - sum_i |d_i| (c_i^H + mr_i m_s)            (derivative costs; posted margin earns nothing)
            - sum_f c_f |w_f - w0_f|                      (trading from the INITIAL portfolio w0)
Constraints (no hidden leverage): sum_f w_f + sum_i mr_i |d_i| <= 1 - buffer;  sum_f w_f + sum_i max(d_i,0) <= L_max;
sum_i |d_i| <= G_max;  per-model ES95_k(pnl) <= budget (Rockafellar-Uryasev);  optional turnover and fixed weights.
Participation (recovery participation) of model k: P_k = E_k[pnl | SMLT up] / E_k[pnl_CONTROL | SMLT up], where CONTROL
is the unchanged initial portfolio (so CONTROL has P = 1). Objectives:
  max_participation : maximise min_k P_k  s.t. ES budget            (frontier, risk budget -> participation)
  min_worst_es      : minimise max_k ES_k  s.t. P_k >= floor for all k (frontier, participation floor -> risk)
  max_premium       : maximise mean_k E_k[pnl] + sum_j x_j * premium_j * tau  s.t. ES budget   (SENSITIVITY ONLY:
                      premia are ASSUMPTIONS, never estimated from history)
No scenario probabilities are invented: models enter through worst-case constraints only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from ..risk.metrics import es_hist


@dataclass
class Model:
    R: np.ndarray        # (N, A) horizon returns of the asset series
    rs: np.ndarray       # (N,) SMLT horizon return (defines recovery states)
    m: float             # money-market horizon return


@dataclass
class FundLP:
    assets: list
    w0: np.ndarray
    tau: float
    buffer: float
    long_max: float
    gross_max: float
    turnover_max: float = np.inf
    fixed: dict | None = None          # name -> fixed weight (e.g. SELL_AND_REALLOCATE)


def _coef(lp: FundLP, M: Model):
    """Per-scenario linear coefficients on [pos(A), neg(A), a(A), b(A)] and the constant term."""
    fund = np.array([a.kind == "FUNDED" for a in lp.assets])
    Rp = M.R - np.where(fund, M.m, 0.0)[None, :]
    dc = np.array([0.0 if a.kind == "FUNDED" else a.horizon_cost(lp.tau) + a.mr * M.m for a in lp.assets])
    fc = np.array([a.cost_one_way if a.kind == "FUNDED" else 0.0 for a in lp.assets])
    G = np.hstack([Rp - dc[None, :], -Rp - dc[None, :], np.tile(-fc, (len(Rp), 1)), np.tile(-fc, (len(Rp), 1))])
    return G, np.full(len(Rp), M.m)


def control_pnl(lp: FundLP, M: Model) -> np.ndarray:
    fund = np.array([a.kind == "FUNDED" for a in lp.assets])
    return M.m + (M.R - np.where(fund, M.m, 0.0)[None, :]) @ lp.w0


def solve(lp: FundLP, models: list[Model], objective: str, budget: float | None = None, floor: float | None = None,
          premia: np.ndarray | None = None, alpha: float = 0.95) -> dict:
    A = len(lp.assets)
    K = len(models)
    Ns = [len(M.rs) for M in models]
    nx = 4 * A
    # variable layout: x(4A) | t | zeta(K) | u(sum N)
    it, iz, iu = nx, nx + 1, nx + 1 + K
    nv = iu + sum(Ns)
    fund = np.array([a.kind == "FUNDED" for a in lp.assets])
    mr = np.array([a.mr for a in lp.assets])
    rows, rhs = [], []
    eq_rows, eq_rhs = [], []
    off = iu
    parts = []
    for k, M in enumerate(models):
        G, c0 = _coef(lp, M)
        N = len(c0)
        # u_ks >= -(c0 + G x) - zeta_k   <=>   -G x - zeta_k - u_ks <= c0
        blk = sp.hstack([sp.csr_matrix(-G), sp.csr_matrix((N, 1)),
                         sp.csr_matrix((np.full(N, -1.0), (np.arange(N), np.full(N, k))), shape=(N, K)),
                         sp.csr_matrix((N, off - iu)), -sp.identity(N, format="csr"), sp.csr_matrix((N, nv - off - N))], format="csr")
        rows.append(blk)
        rhs.append(c0)
        es_row = np.zeros(nv)
        es_row[iz + k] = 1.0
        es_row[off:off + N] = 1.0 / ((1 - alpha) * N)
        if objective == "min_worst_es":
            es_row[it] = -1.0                                 # ES_k <= t
            rows.append(sp.csr_matrix(es_row[None, :])); rhs.append(np.array([0.0]))
        else:
            rows.append(sp.csr_matrix(es_row[None, :])); rhs.append(np.array([budget]))
        up = M.rs > 0
        D = float(control_pnl(lp, M)[up].mean())
        gu = G[up].mean(axis=0)
        cu = float(c0[up].mean())
        parts.append((gu, cu, D))
        if objective == "max_participation":                 # t * D_k <= cu + gu.x
            r = np.zeros(nv); r[:nx] = -gu; r[it] = D
            rows.append(sp.csr_matrix(r[None, :])); rhs.append(np.array([cu]))
        elif objective == "min_worst_es" and floor is not None:   # cu + gu.x >= floor * D
            r = np.zeros(nv); r[:nx] = -gu
            rows.append(sp.csr_matrix(r[None, :])); rhs.append(np.array([cu - floor * D]))
        off += N
    # fund constraints
    r = np.zeros(nv); r[:A] = fund.astype(float); r[:A] += np.where(fund, 0.0, mr); r[A:2 * A] = np.where(fund, 0.0, mr)
    rows.append(sp.csr_matrix(r[None, :])); rhs.append(np.array([1.0 - lp.buffer]))
    r = np.zeros(nv); r[:A] = 1.0                            # funded + long derivatives
    rows.append(sp.csr_matrix(r[None, :])); rhs.append(np.array([lp.long_max]))
    r = np.zeros(nv); r[:A] = (~fund).astype(float); r[A:2 * A] = (~fund).astype(float)
    rows.append(sp.csr_matrix(r[None, :])); rhs.append(np.array([lp.gross_max]))
    if np.isfinite(lp.turnover_max):
        r = np.zeros(nv); r[2 * A:4 * A] = np.tile(fund.astype(float), 2); r[:2 * A] += np.tile((~fund).astype(float), 2)
        rows.append(sp.csr_matrix(r[None, :])); rhs.append(np.array([lp.turnover_max]))
    # funded weight linkage: pos - neg - a + b = w0 (neg of funded is bounded at 0 below)
    for j in range(A):
        if fund[j]:
            e = np.zeros(nv); e[j] = 1; e[A + j] = -1; e[2 * A + j] = -1; e[3 * A + j] = 1
            eq_rows.append(e); eq_rhs.append(lp.w0[j])
    pos_b, neg_b = [], []
    for j, a in enumerate(lp.assets):
        lo, hi = a.lo, a.hi
        if lp.fixed and a.name in lp.fixed:
            lo = hi = lp.fixed[a.name]
        if fund[j]:
            pos_b.append((max(lo, 0.0), max(hi, 0.0)))
            neg_b.append((0.0, 0.0))
        else:
            pos_b.append((max(lo, 0.0), max(hi, 0.0)))
            neg_b.append((max(-hi, 0.0), max(-lo, 0.0)))
    bounds = pos_b + neg_b
    bounds += [(0.0, None)] * (2 * A)
    bounds += [(None, None)] + [(None, None)] * K + [(0.0, None)] * sum(Ns)
    obj = np.zeros(nv)
    if objective == "max_participation":
        obj[it] = -1.0
    elif objective == "min_worst_es":
        obj[it] = 1.0
    elif objective == "max_premium":
        for k, M in enumerate(models):
            G, c0 = _coef(lp, M)
            obj[:nx] -= G.mean(axis=0) / K
        if premia is not None:
            obj[:A] -= premia * lp.tau
            obj[A:2 * A] += premia * lp.tau
    else:
        raise ValueError(objective)
    res = linprog(obj, A_ub=sp.vstack(rows, format="csr"), b_ub=np.concatenate(rhs),
                  A_eq=np.array(eq_rows) if eq_rows else None, b_eq=np.array(eq_rhs) if eq_rows else None,
                  bounds=bounds, method="highs")
    if res.status != 0:
        return {"status": f"infeasible:{res.message[:50]}"}
    x = res.x[:A] - res.x[A:2 * A]
    return {"status": "optimal", "x": x, "t": float(res.x[it]), **diagnostics(lp, models, x, alpha)}


def diagnostics(lp: FundLP, models: list[Model], x: np.ndarray, alpha: float = 0.95) -> dict:
    """ES, participation and mean of a given position vector under every model (exact, not LP relaxation)."""
    fund = np.array([a.kind == "FUNDED" for a in lp.assets])
    pos = np.concatenate([np.maximum(x, 0), np.maximum(-x, 0), np.maximum(np.where(fund, x - lp.w0, 0), 0),
                          np.maximum(np.where(fund, lp.w0 - x, 0), 0)])
    es, part, mean = [], [], []
    for M in models:
        G, c0 = _coef(lp, M)
        pnl = c0 + G @ pos
        up = M.rs > 0
        es.append(es_hist(pnl, alpha))
        part.append(float(pnl[up].mean() / control_pnl(lp, M)[up].mean()))
        mean.append(float(pnl.mean()))
    return {"worst_es": max(es), "es_by_model": es, "min_participation": min(part), "participation_by_model": part,
            "mean_by_model": mean, "margin_used": float(np.sum(np.abs(x) * np.array([0.0 if f else a.mr for f, a in zip(fund, lp.assets)]))),
            "funded_risky": float(np.sum(np.where(fund, x, 0.0))), "deriv_gross": float(np.sum(np.where(fund, 0.0, np.abs(x))))}
