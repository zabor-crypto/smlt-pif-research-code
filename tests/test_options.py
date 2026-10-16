import math

import numpy as np
import pytest

from smlt_hedge.strategies.options import (black76, collar_payoff, greeks, implied_vol, put_payoff,
                                           put_spread_payoff)


def test_put_call_parity():
    F, K, tau, s, r = 250.0, 240.0, 0.5, 0.8, 0.14
    c, p = black76(F, K, tau, s, r, "C"), black76(F, K, tau, s, r, "P")
    assert c - p == pytest.approx(math.exp(-r * tau) * (F - K), abs=1e-10)


def test_greeks_match_finite_differences():
    F, K, tau, s, r = 250.0, 240.0, 0.5, 0.8, 0.14
    g = greeks(F, K, tau, s, r, "P")
    h = 1e-4
    assert g["delta"] == pytest.approx((black76(F + h, K, tau, s, r, "P") - black76(F - h, K, tau, s, r, "P")) / (2 * h), rel=1e-5)
    assert g["gamma"] == pytest.approx((black76(F + h, K, tau, s, r, "P") - 2 * black76(F, K, tau, s, r, "P")
                                        + black76(F - h, K, tau, s, r, "P")) / h**2, rel=1e-3)
    assert g["vega"] == pytest.approx((black76(F, K, tau, s + h, r, "P") - black76(F, K, tau, s - h, r, "P")) / (2 * h), rel=1e-5)
    dt = 1e-5
    theta = (black76(F, K, tau - dt, s, r, "P") - black76(F, K, tau, s, r, "P")) / dt / 365.0
    assert g["theta_per_day"] == pytest.approx(theta, rel=1e-3)
    assert g["rho"] == pytest.approx((black76(F, K, tau, s, r + h, "P") - black76(F, K, tau, s, r - h, "P")) / (2 * h), rel=1e-5)


def test_implied_vol_roundtrip_and_below_intrinsic():
    p = black76(250, 300, 26 / 365, 1.1, 0.14, "P")
    assert implied_vol(p, 250, 300, 26 / 365, 0.14, "P") == pytest.approx(1.1, abs=1e-7)
    assert math.isnan(implied_vol(10.0, 250, 300, 26 / 365, 0.14, "P"))  # below intrinsic -> no IV


def test_payoffs():
    S = np.array([100.0, 200.0, 300.0])
    assert np.allclose(put_payoff(S, 200), [100, 0, 0])
    assert np.allclose(put_spread_payoff(S, 250, 150), [100, 50, 0])
    assert np.allclose(collar_payoff(S, 200, 250), [100, 0, -50])
