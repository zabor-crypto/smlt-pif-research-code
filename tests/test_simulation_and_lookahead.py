import numpy as np
import pandas as pd
import pytest

from smlt_hedge.backtest.walk_forward import WFConfig, daily_walk_forward, fixed, rolling_hstar
from smlt_hedge.scenarios.bootstrap import block_bootstrap_indices, evaluate_static, paths
from smlt_hedge.strategies.state_policy import state_policy


def _R(n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    f = rng.standard_t(4, n) * 0.02
    return pd.DataFrame({"SMLT": f + rng.normal(0, 0.005, n), "SMLT_FUT": f, "IMOEX": 0.3 * f + rng.normal(0, 0.01, n)},
                        index=idx)


def test_bootstrap_seeded_and_blocky():
    a = block_bootstrap_indices(100, 30, 50, 10, seed=1)
    b = block_bootstrap_indices(100, 30, 50, 10, seed=1)
    assert np.array_equal(a, b)
    d = np.diff(a[0, :10]) % 100
    assert np.all(d == 1)  # consecutive within a block (circular)


def test_evaluate_static_full_hedge_reduces_es_and_zero_hedge_is_control():
    R = _R()
    P = paths(R, ["SMLT", "SMLT_FUT"], 126, 4000, 10, seed=2)
    ctrl = evaluate_static(P, E=1e7, C=1e7, h_fut=0.0)
    full = evaluate_static(P, E=1e7, C=1e7, h_fut=1.0, im_rate_fut=0.5)
    assert full["ES95"] < 0.5 * ctrl["ES95"]
    assert ctrl["initial_im"] == 0.0 and full["initial_im"] == pytest.approx(0.5e7)


def test_daily_walk_forward_has_no_lookahead():
    R = _R()
    is_roll = pd.Series(False, index=R.index)
    mmf = pd.Series(0.0, index=R.index)
    cfg = WFConfig(window=120, rebalance_every=5)
    a = daily_walk_forward(R, is_roll, rolling_hstar(120), cfg, 1e7, 1e7, mmf, "2023-07-03")
    R2 = R.copy()
    R2.iloc[400:, :] *= 5.0          # change the FUTURE only
    b = daily_walk_forward(R2, is_roll, rolling_hstar(120), cfg, 1e7, 1e7, mmf, "2023-07-03")
    cut = R.index[400]
    assert np.allclose(a.loc[:cut, "h_held"], b.loc[:cut, "h_held"])
    # the hedge held on day t was decided at t-1: first held ratio is zero
    assert a["h_held"].iloc[0] == 0.0


def test_fixed_policy_ratio_and_costs():
    R = _R()
    out = daily_walk_forward(R, pd.Series(False, index=R.index), fixed(0.5), WFConfig(window=120, cost_rate=0.001),
                             1e7, 0.0, pd.Series(0.0, index=R.index), "2023-07-03")
    assert out["h_held"].iloc[5:].eq(0.5).all()
    assert out["costs"].iloc[0] == pytest.approx(0.5 * 1e7 * 0.001)


def test_state_policy_uses_only_supplied_history():
    R = _R()
    f = state_policy()
    h1 = f(R.iloc[:300])
    R2 = R.copy(); R2.iloc[300:] = 0.5
    assert f(R2.iloc[:300]) == h1
    assert 0.0 <= h1 <= 1.0


def test_neutral_drift_removes_directional_view():
    from smlt_hedge.pipeline import _neutral_drift
    R = _R()
    R["SMLT"] -= 0.01  # strong bear drift in the sample
    Y = _neutral_drift(R, ["SMLT_FUT"], 0.0005)
    assert Y["SMLT"].mean() == pytest.approx(0.0005)
    assert Y["SMLT_FUT"].mean() == pytest.approx(0.0, abs=1e-15)
    assert Y["SMLT"].std() == pytest.approx(R["SMLT"].std())
    assert np.corrcoef(Y["SMLT"], Y["SMLT_FUT"])[0, 1] == pytest.approx(np.corrcoef(R["SMLT"], R["SMLT_FUT"])[0, 1])


def test_structural_futures_basis_is_stationary():
    from smlt_hedge.scenarios.bootstrap import structural_futures, cum_level
    R = _R()
    P = paths(R, ["SMLT", "SMLT_FUT"], 252, 2000, 10, seed=3)
    e = np.random.default_rng(4).normal(0, 0.01, len(R))
    P["SMLT_FUT"] = structural_futures(P, e, carry_ann=0.12, e0=0.0)
    S, F = cum_level(P["SMLT"]), cum_level(P["SMLT_FUT"])
    lb = np.log(F / S) + 0.12 * np.arange(1, 253) / 252
    # log-basis net of carry does not grow with the horizon (stationary), unlike a random walk
    assert lb[:, -1].std() == pytest.approx(lb[:, 20].std(), rel=0.2)
    assert lb[:, -1].std() < 0.02


def test_fhs_never_produces_returns_below_minus_100pct():
    from smlt_hedge.pipeline import _fhs
    R = _R()
    R.iloc[300, 0] = -0.45                       # regime-break day at low prior volatility
    R.iloc[-30:, 0] *= 6                         # high current volatility
    Y = _fhs(R, ["SMLT", "SMLT_FUT"], 0.94, None, groups={"SMLT": ["SMLT", "SMLT_FUT"]})
    assert (Y > -1).all().all()
