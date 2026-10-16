import numpy as np
import pandas as pd
import pytest

from smlt_hedge.features.stats import mvhr, rolling_beta, simple_returns
from smlt_hedge.optimization.overlay import Problem, Sleeve, cardinality_path, min_es, min_variance, portfolio_pnl
from smlt_hedge.risk.covariance import cov_ewma, cov_lw, cov_oas, cov_sample, min_var_hedge_weights
from smlt_hedge.risk.metrics import (capture_ratios, es_hist, he_pnl, he_var, max_drawdown,
                                     max_drawdown_rub, var_hist, worst_k_day)


# ------------------------------------------------------------------ returns
def test_simple_returns_basic():
    p = pd.Series([100.0, 110.0, 99.0])
    r = simple_returns(p)
    assert np.isnan(r.iloc[0])
    assert r.iloc[1] == pytest.approx(0.10)
    assert r.iloc[2] == pytest.approx(-0.10)


def test_returns_do_not_fill_gaps():
    p = pd.Series([100.0, np.nan, 121.0])
    r = simple_returns(p)
    assert np.isnan(r.iloc[1]) and np.isnan(r.iloc[2])  # no silent forward fill across a gap


# ------------------------------------------------------------------ VaR / ES
def test_var_es_known_sample():
    pnl = -np.arange(1, 101, dtype=float)  # losses 1..100
    assert var_hist(pnl, 0.95) == 95.0
    # RU ES = 95 + mean(max(L-95,0))/0.05 = 95 + (1+2+3+4+5)/100/0.05 = 98
    assert es_hist(pnl, 0.95) == pytest.approx(98.0)
    assert es_hist(pnl, 0.99) >= var_hist(pnl, 0.99)


def test_es_is_coherent_on_scaling_and_translation():
    rng = np.random.default_rng(0)
    x = rng.standard_t(4, 5000)
    assert es_hist(2 * x) == pytest.approx(2 * es_hist(x))
    assert es_hist(x + 1.0) == pytest.approx(es_hist(x) - 1.0)


def test_drawdown_and_worst_k():
    assert max_drawdown([1, 2, 1, 3, 1.5]) == pytest.approx(0.5)
    assert max_drawdown_rub([1, 3, 0, 2]) == pytest.approx(3.0)
    assert worst_k_day([1, -2, -3, 4], 2) == pytest.approx(5.0)


def test_capture_and_course_metrics():
    b = np.array([0.1, -0.1, 0.2, -0.2])
    u, d = capture_ratios(0.5 * b, b)
    assert u == pytest.approx(0.5) and d == pytest.approx(0.5)
    ds = np.array([-100.0])
    dh = np.array([80.0])
    assert he_pnl(dh, ds) == pytest.approx(0.8)
    rng = np.random.default_rng(1)
    s = rng.normal(size=1000)
    assert he_var(0.0 * s + 1e-9 * rng.normal(size=1000), s) == pytest.approx(1.0, abs=1e-6)


# ------------------------------------------------------------------ hedge ratio
def test_mvhr_identity_and_recovery():
    rng = np.random.default_rng(2)
    h = rng.normal(0, 0.02, 20000)
    s = 0.7 * h + rng.normal(0, 0.01, 20000)
    out = mvhr(s, h)
    assert out["h_star"] == pytest.approx(0.7, abs=0.01)
    assert out["identity_abs_err"] < 1e-12


def test_rolling_beta_is_trailing():
    rng = np.random.default_rng(3)
    x = pd.Series(rng.normal(size=300))
    y = 2 * x + pd.Series(rng.normal(scale=0.01, size=300))
    b = rolling_beta(y, x, 50)
    assert b.iloc[:49].isna().all()
    # perturbing a FUTURE observation must not change today's beta
    y2 = y.copy(); y2.iloc[200] += 100
    b2 = rolling_beta(y2, x, 50)
    assert b.iloc[199] == b2.iloc[199]


# ------------------------------------------------------------------ covariance
def test_covariance_estimators_psd_and_close():
    rng = np.random.default_rng(4)
    X = rng.multivariate_normal([0, 0, 0], [[1, .5, .2], [.5, 1, .3], [.2, .3, 1]], 2000) * 0.01
    for f in (cov_sample, cov_lw, cov_oas):
        S = f(X)
        assert np.all(np.linalg.eigvalsh(S) > 0)
        assert np.allclose(S, cov_sample(X), atol=2e-6)
    E = cov_ewma(X, 0.94)
    assert np.all(np.linalg.eigvalsh(E) > 0)


def test_min_var_hedge_matches_mvhr_two_assets():
    rng = np.random.default_rng(5)
    h = rng.normal(0, 0.02, 5000)
    s = 0.6 * h + rng.normal(0, 0.01, 5000)
    S = cov_sample(np.column_stack([s, h]))
    w = min_var_hedge_weights(S, 0)
    assert -w[1] == pytest.approx(mvhr(s, h)["h_star"], rel=1e-7)  # 1e-12 ridge in solver


# ------------------------------------------------------------------ optimisation
def _toy(n=3000, seed=6):
    rng = np.random.default_rng(seed)
    f = rng.standard_t(4, n) * 0.02
    r_s = f + rng.normal(0, 0.01, n)                       # SMLT
    fut = f + rng.normal(0, 0.002, n)                      # direct future
    idx = 0.5 * f + rng.normal(0, 0.01, n)                 # index
    noise = rng.normal(0, 0.01, n)                         # useless diversifier
    R = np.column_stack([fut, idx, noise])
    sl = [Sleeve("FUT", -1.0, 0.0, cost=1e-4, margin=0.3), Sleeve("IDX", -1.5, 0.0, cost=1e-4, margin=0.15),
          Sleeve("NOISE", -0.5, 0.5, cost=1e-4, margin=0.1)]
    return Problem(r_s=r_s, R=R, sleeves=sl)


def test_min_es_lp_objective_equals_reported_es():
    p = _toy()
    sol = min_es(p)
    assert sol.status == "optimal"
    assert sol.es == pytest.approx(es_hist(portfolio_pnl(p, sol.h), p.alpha), rel=1e-9)
    assert sol.h[0] < -0.8        # direct hedge dominates
    assert abs(sol.h[2]) < 0.05   # noise sleeve not used materially
    assert sol.es < min_es(Problem(p.r_s, p.R[:, 2:], p.sleeves[2:])).es


def test_bounds_margin_and_gross_constraints_respected():
    p = _toy()
    p.margin_budget = 0.15
    p.gross_cap = 0.6
    for solver in (min_es, min_variance):
        sol = solver(p)
        assert sol.margin <= 0.15 + 1e-6
        assert np.abs(sol.h).sum() <= 0.6 + 1e-6
        for v, s in zip(sol.h, p.sleeves):
            assert s.lo - 1e-9 <= v <= s.hi + 1e-9


def test_fixed_smlt_cannot_be_sold_by_optimizer():
    # The SMLT exposure enters as a constant term; no decision variable scales it.
    p = _toy()
    sol = min_es(p)
    pnl0 = portfolio_pnl(p, np.zeros(3))
    assert np.allclose(pnl0, p.r_s)
    assert len(sol.h) == len(p.sleeves)


def test_cardinality_monotone_and_exact():
    p = _toy()
    path = cardinality_path(p, [1, 2, 3])
    es = [path[k].es for k in (1, 2, 3)]
    assert es[0] >= es[1] - 1e-12 >= es[2] - 2e-12
    for k in (1, 2, 3):
        assert len(path[k].active) <= k


def test_upside_capture_constraint_binds():
    p = _toy()
    free = min_es(p)
    p.min_upside_capture = 0.6
    con = min_es(p)
    up = p.r_s > 0
    pnl = portfolio_pnl(p, con.h)
    assert pnl[up].mean() >= 0.6 * p.r_s[up].mean() - 1e-9
    assert con.es >= free.es - 1e-12


def test_liquidity_sleeve_same_for_all_strategies():
    p = _toy()
    p.k_liq, p.m_liq = 1.0, np.full(len(p.r_s), 0.0005)
    a = portfolio_pnl(p, np.zeros(3))
    assert np.allclose(a - p.r_s, 0.0005)
