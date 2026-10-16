"""Hedge-overlay optimisation around a FIXED SMLT exposure.

Decision variables: h_i = signed notional of hedge sleeve i as a fraction of the fixed SMLT
exposure E (h < 0 = short). The SMLT position itself is NOT a decision variable here; the
DE_RISKING branch lives in strategies/derisking.py.

Scenario P&L per unit of E (scenario s):
    pnl_s = r_S,s + sum_i h_i r_i,s + k * m_s  - sum_i c_i |h_i|
where m_s is the liquidity-sleeve return and k = hedge_capital / E (identical for every
strategy with the same capital, so it never manufactures risk reduction).

Primary objective: min ES_alpha (Rockafellar-Uryasev LP, solved with HiGHS).
Benchmarks: min variance (QP), target-vol / target-ES scaling. Diagnostic: max Sharpe.
Cardinality K is enforced EXACTLY by enumerating sleeve subsets of size <= K (the sleeve
universe after G1-G5 screening is small, so enumeration is cheap and avoids MIQP heuristics).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog, minimize

from ..risk.metrics import es_hist, var_hist


@dataclass
class Sleeve:
    name: str
    lo: float                  # lower bound on h (e.g. -1.0 for up to 100% short)
    hi: float                  # upper bound on h
    cost: float = 0.0          # one-way cost per unit |h| over the horizon (fees+spread+expected roll costs)
    margin: float = 0.0        # initial-margin (or capital) usage per unit |h| as fraction of notional
    kind: str = "linear"       # linear | premium (capital consumed up-front)
    group: str = ""            # economic sleeve label (DIRECT, INDEX, SECTOR, RATES, DIVERSIFIER)


@dataclass
class Problem:
    r_s: np.ndarray            # (N,) SMLT return per scenario
    R: np.ndarray              # (N, m) sleeve returns per unit notional
    sleeves: list[Sleeve]
    k_liq: float = 0.0         # hedge capital / E
    m_liq: np.ndarray | None = None  # (N,) liquidity-sleeve return
    alpha: float = 0.95
    margin_budget: float = np.inf    # max sum margin_i |h_i| (fraction of E)
    gross_cap: float = np.inf        # max sum |h_i|
    min_upside_capture: float | None = None  # E[pnl | r_S>0] >= u * E[r_S + k*m | r_S>0] (same-capital CONTROL)
    weights: np.ndarray | None = None       # scenario probabilities (default uniform)

    def base(self) -> np.ndarray:
        b = self.r_s.astype(float).copy()
        if self.m_liq is not None:
            b = b + self.k_liq * self.m_liq
        return b


@dataclass
class Solution:
    h: np.ndarray
    names: list[str]
    es: float
    var: float
    std: float
    mean: float
    cost: float
    margin: float
    status: str
    active: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = {f"h_{n}": float(v) for n, v in zip(self.names, self.h)}
        d.update(es=self.es, var=self.var, std=self.std, mean=self.mean, cost=self.cost,
                 margin=self.margin, status=self.status, active="|".join(self.active), K=len(self.active))
        return d


def portfolio_pnl(p: Problem, h: np.ndarray) -> np.ndarray:
    c = np.array([s.cost for s in p.sleeves])
    return p.base() + p.R @ h - float(c @ np.abs(h))


def _evaluate(p: Problem, h: np.ndarray, status: str, tol: float = 1e-6) -> Solution:
    pnl = portfolio_pnl(p, h)
    c = np.array([s.cost for s in p.sleeves])
    mg = np.array([s.margin for s in p.sleeves])
    names = [s.name for s in p.sleeves]
    return Solution(h=h, names=names, es=es_hist(pnl, p.alpha), var=var_hist(pnl, p.alpha),
                    std=float(np.std(pnl, ddof=1)), mean=float(np.mean(pnl)), cost=float(c @ np.abs(h)),
                    margin=float(mg @ np.abs(h)), status=status,
                    active=[n for n, v in zip(names, h) if abs(v) > tol])


def _subset(p: Problem, idx: list[int]) -> Problem:
    return Problem(r_s=p.r_s, R=p.R[:, idx], sleeves=[p.sleeves[i] for i in idx], k_liq=p.k_liq,
                   m_liq=p.m_liq, alpha=p.alpha, margin_budget=p.margin_budget, gross_cap=p.gross_cap,
                   min_upside_capture=p.min_upside_capture, weights=p.weights)


def _embed(p: Problem, idx: list[int], sol_sub: Solution) -> Solution:
    h = np.zeros(len(p.sleeves))
    h[idx] = sol_sub.h
    return _evaluate(p, h, sol_sub.status)


# ------------------------------------------------------------------ min ES (RU LP)
def min_es(p: Problem) -> Solution:
    """Rockafellar-Uryasev LP. Variables x = [pos (m), neg (m), zeta, u (N)], h = pos - neg."""
    N, m = p.R.shape
    w = np.full(N, 1.0 / N) if p.weights is None else p.weights / p.weights.sum()
    c = np.array([s.cost for s in p.sleeves])
    mg = np.array([s.margin for s in p.sleeves])
    base = p.base()
    nv = 2 * m + 1 + N
    obj = np.zeros(nv)
    obj[2 * m] = 1.0
    obj[2 * m + 1:] = w / (1.0 - p.alpha)
    # u_s >= L_s - zeta,  L_s = -(base_s + R_s (pos-neg)) + c.(pos+neg)
    # => -R_s pos + R_s neg + c pos + c neg - zeta - u_s <= base_s
    # sparse constraint matrix: [ -R+c | R+c | -1 | -I_N ]  (the -I_N block dominates the size)
    A = sp.hstack([sp.csr_matrix(-p.R + c), sp.csr_matrix(p.R + c), sp.csr_matrix(-np.ones((N, 1))),
                   -sp.identity(N, format="csr")], format="csr")
    b = base.copy()
    rows, rhs = [A], [b]
    if np.isfinite(p.margin_budget):
        r = np.zeros(nv); r[:m] = mg; r[m:2 * m] = mg
        rows.append(sp.csr_matrix(r[None, :])); rhs.append(np.array([p.margin_budget]))
    if np.isfinite(p.gross_cap):
        r = np.zeros(nv); r[:2 * m] = 1.0
        rows.append(sp.csr_matrix(r[None, :])); rhs.append(np.array([p.gross_cap]))
    if p.min_upside_capture is not None:
        up = p.r_s > 0
        if up.any():
            # mean over up-states of pnl >= u * mean(r_S | up)
            wu = w[up] / w[up].sum()
            Ru = wu @ p.R[up]
            r = np.zeros(nv)
            r[:m] = -(Ru - c)
            r[m:2 * m] = -(-Ru - c)
            rows.append(sp.csr_matrix(r[None, :]))
            # V3: the denominator is the same-capital CONTROL (SMLT + liquidity carry), like the numerator
            rhs.append(np.array([float(wu @ base[up]) - p.min_upside_capture * float(wu @ base[up])]))
    bounds = ([(0.0, max(s.hi, 0.0)) for s in p.sleeves] + [(0.0, max(-s.lo, 0.0)) for s in p.sleeves]
              + [(None, None)] + [(0.0, None)] * N)
    res = linprog(obj, A_ub=sp.vstack(rows, format="csr"), b_ub=np.concatenate(rhs), bounds=bounds, method="highs")
    if res.status != 0:
        return _evaluate(p, np.zeros(m), f"infeasible:{res.message[:60]}")
    h = res.x[:m] - res.x[m:2 * m]
    sol = _evaluate(p, h, "optimal")
    return sol


# ------------------------------------------------------------------ min variance (QP)
def min_variance(p: Problem) -> Solution:
    """Min Var(pnl) over h with bounds, margin and gross caps (SLSQP; convex QP)."""
    m = p.R.shape[1]
    mg = np.array([s.margin for s in p.sleeves])
    X = np.column_stack([p.base(), p.R])
    S = np.cov(X, rowvar=False, ddof=1)
    s_bb, s_bR, S_RR = S[0, 0], S[0, 1:], S[1:, 1:]
    sab = lambda h: np.sqrt(h**2 + 1e-12)  # noqa: E731  smooth |h| for constraint gradients
    cons = []
    if np.isfinite(p.margin_budget):
        cons.append({"type": "ineq", "fun": lambda h: p.margin_budget - mg @ sab(h)})
    if np.isfinite(p.gross_cap):
        cons.append({"type": "ineq", "fun": lambda h: p.gross_cap - sab(h).sum()})
    res = minimize(lambda h: s_bb + 2 * s_bR @ h + h @ S_RR @ h, np.zeros(m),
                   jac=lambda h: 2 * s_bR + 2 * S_RR @ h, method="SLSQP",
                   bounds=[(s.lo, s.hi) for s in p.sleeves], constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-15})
    h = np.where(np.abs(res.x) < 1e-7, 0.0, res.x)
    return _evaluate(p, h, "optimal" if res.success else f"slsqp:{res.message}")


# ------------------------------------------------------------------ diagnostics
def max_sharpe(p: Problem) -> Solution:
    """DIAGNOSTIC ONLY. Unstable because it depends on estimated means."""
    m = p.R.shape[1]

    def neg_sr(h):
        x = portfolio_pnl(p, h)
        sd = np.std(x, ddof=1)
        return -np.mean(x) / sd if sd > 0 else 0.0

    res = minimize(neg_sr, np.zeros(m), method="SLSQP", bounds=[(s.lo, s.hi) for s in p.sleeves],
                   options={"maxiter": 500})
    return _evaluate(p, res.x, "diagnostic")


def scale_to_target(p: Problem, h_dir: np.ndarray, target: float, metric: str = "es") -> Solution:
    """Scale a hedge direction h_dir by t in [0,1] to reach target ES (or std); returns closest feasible."""
    ts = np.linspace(0, 1, 201)
    best = None
    for t in ts:
        sol = _evaluate(p, t * h_dir, "scaled")
        val = sol.es if metric == "es" else sol.std
        if val <= target:
            return sol
        best = sol if best is None or (sol.es if metric == "es" else sol.std) < (best.es if metric == "es" else best.std) else best
    best.status = "target_not_reached"
    return best


# ------------------------------------------------------------------ cardinality
def cardinality_path(p: Problem, K_grid: list[int], solver=min_es, mandatory: list[str] | None = None,
                     key=lambda s: s.es) -> dict:
    """Exact best solution for each K by enumerating sleeve subsets of size <= K."""
    m = len(p.sleeves)
    names = [s.name for s in p.sleeves]
    mand = [names.index(n) for n in (mandatory or [])]
    cache: dict[tuple, Solution] = {}
    out = {}
    for K in sorted(K_grid):
        best = None
        for k in range(1, min(K, m) + 1):
            for comb in combinations(range(m), k):
                if not set(mand) <= set(comb):
                    continue
                if comb not in cache:
                    cache[comb] = _embed(p, list(comb), solver(_subset(p, list(comb))))
                sol = cache[comb]
                if not sol.status.startswith(("optimal", "diagnostic")):
                    continue
                if best is None or key(sol) < key(best) - 1e-12:
                    best = sol
        out[K] = best
    return out


# ------------------------------------------------------------------ V3 0.2: true min-max ES across models
def min_max_es(problems: list[Problem]) -> Solution:
    """min_h max_k ES_alpha,k(h) over several scenario models (joint Rockafellar-Uryasev LP).

    Variables x = [pos (m), neg (m), t, zeta_1..zeta_K, u_1 (N_1), ..., u_K (N_K)].
    minimise t  s.t.  t >= zeta_k + sum_s w_ks u_ks / (1-alpha)   and   u_ks >= L_ks(h) - zeta_k,  u >= 0.
    Sleeves, bounds, margin budget, gross cap and the upside floor are taken from problems[0]; the upside floor is
    imposed in EVERY model (robust upside). The returned Solution's es/var/std/mean are those of the WORST model.
    """
    p0 = problems[0]
    m, K = len(p0.sleeves), len(problems)
    c = np.array([s.cost for s in p0.sleeves])
    mg = np.array([s.margin for s in p0.sleeves])
    Ns = [p.R.shape[0] for p in problems]
    nv = 2 * m + 1 + K + sum(Ns)
    obj = np.zeros(nv)
    obj[2 * m] = 1.0
    rows, rhs = [], []
    off = 2 * m + 1 + K
    for k, p in enumerate(problems):
        N = Ns[k]
        w = np.full(N, 1.0 / N) if p.weights is None else p.weights / p.weights.sum()
        zc = np.zeros((N, 1 + K)); zc[:, 1 + k] = -1.0
        left = sp.csr_matrix(np.hstack([-p.R + c, p.R + c, zc]))
        blocks = [left]
        pre, post = off - (2 * m + 1 + K), nv - off - N
        if pre:
            blocks.append(sp.csr_matrix((N, pre)))
        blocks.append(-sp.identity(N, format="csr"))
        if post:
            blocks.append(sp.csr_matrix((N, post)))
        rows.append(sp.hstack(blocks, format="csr"))
        rhs.append(p.base().astype(float))
        r = np.zeros(nv); r[2 * m] = -1.0; r[2 * m + 1 + k] = 1.0; r[off:off + N] = w / (1.0 - p.alpha)
        rows.append(sp.csr_matrix(r[None, :])); rhs.append(np.array([0.0]))
        if p0.min_upside_capture is not None:
            up = p.r_s > 0
            if up.any():
                wu = w[up] / w[up].sum()
                Ru = wu @ p.R[up]
                base_up = float(wu @ p.base()[up])
                r = np.zeros(nv); r[:m] = -(Ru - c); r[m:2 * m] = -(-Ru - c)
                rows.append(sp.csr_matrix(r[None, :])); rhs.append(np.array([base_up * (1.0 - p0.min_upside_capture)]))
        off += N
    if np.isfinite(p0.margin_budget):
        r = np.zeros(nv); r[:m] = mg; r[m:2 * m] = mg
        rows.append(sp.csr_matrix(r[None, :])); rhs.append(np.array([p0.margin_budget]))
    if np.isfinite(p0.gross_cap):
        r = np.zeros(nv); r[:2 * m] = 1.0
        rows.append(sp.csr_matrix(r[None, :])); rhs.append(np.array([p0.gross_cap]))
    bounds = ([(0.0, max(s.hi, 0.0)) for s in p0.sleeves] + [(0.0, max(-s.lo, 0.0)) for s in p0.sleeves]
              + [(None, None)] * (1 + K) + [(0.0, None)] * sum(Ns))
    res = linprog(obj, A_ub=sp.vstack(rows, format="csr"), b_ub=np.concatenate(rhs), bounds=bounds, method="highs")
    if res.status != 0:
        return _evaluate(p0, np.zeros(m), f"infeasible:{res.message[:60]}")
    h = res.x[:m] - res.x[m:2 * m]
    sols = [_evaluate(p, h, "optimal") for p in problems]
    worst = max(sols, key=lambda s: s.es)
    worst.status = "optimal_minmax"
    return worst
