"""Correctness tests. A fast engine producing wrong prices is worthless,
so these run before any performance work."""
import math

import numpy as np
import pytest

from engine import greeks, implied_vol, price

S, K, T, r, q, SIG = 100.0, 100.0, 1.0, 0.05, 0.0, 0.2


def test_known_value():
    # Textbook value (Hull): ATM 1y call, r=5%, vol=20% -> 10.4506
    assert float(price(S, K, T, r, SIG, True)) == pytest.approx(10.4506, abs=1e-4)


@pytest.mark.parametrize("q_", [0.0, 0.03])
@pytest.mark.parametrize("k", [80.0, 100.0, 120.0])
def test_put_call_parity(k, q_):
    c = float(price(S, k, T, r, SIG, True, q_))
    p = float(price(S, k, T, r, SIG, False, q_))
    assert c - p == pytest.approx(S * math.exp(-q_ * T) - k * math.exp(-r * T), abs=1e-10)


@pytest.mark.parametrize("name,bump_arg", [("delta", "S"), ("vega", "sigma"),
                                           ("rho", "r")])
@pytest.mark.parametrize("is_call", [True, False])
def test_greeks_match_finite_differences(name, bump_arg, is_call):
    args = dict(S=S, K=K, T=T, r=r, sigma=SIG)
    h = 1e-4
    up, dn = dict(args), dict(args)
    up[bump_arg] += h
    dn[bump_arg] -= h
    fd = (float(price(**up, is_call=is_call)) - float(price(**dn, is_call=is_call))) / (2 * h)
    assert float(greeks(**args, is_call=is_call)[name]) == pytest.approx(fd, rel=1e-5)


def test_gamma_and_theta_finite_differences():
    h = 1e-3
    g = greeks(S, K, T, r, SIG, True)
    fd_gamma = (float(price(S + h, K, T, r, SIG)) - 2 * float(price(S, K, T, r, SIG))
                + float(price(S - h, K, T, r, SIG))) / h**2
    assert float(g["gamma"]) == pytest.approx(fd_gamma, rel=1e-4)
    # theta = dV/dt = -dV/dT
    fd_theta = -(float(price(S, K, T + h, r, SIG)) - float(price(S, K, T - h, r, SIG))) / (2 * h)
    assert float(g["theta"]) == pytest.approx(fd_theta, rel=1e-5)


@pytest.mark.parametrize("true_sigma", [0.05, 0.2, 0.6, 1.5])
@pytest.mark.parametrize("k", [50.0, 90.0, 100.0, 110.0, 200.0])
@pytest.mark.parametrize("is_call", [True, False])
def test_implied_vol_round_trip(true_sigma, k, is_call):
    mkt = float(price(S, k, T, r, true_sigma, is_call))
    sigma, _, method = implied_vol(mkt, S, k, T, r, is_call)
    if method in ("out_of_bounds", "unidentifiable"):
        # Time value is ~0 (deep ITM/OTM, low vol): the solver must say so
        # instead of returning a confident wrong number.
        assert math.isnan(sigma)
        return
    assert sigma == pytest.approx(true_sigma, rel=1e-5)


def test_implied_vol_rejects_arbitrage_prices():
    assert math.isnan(implied_vol(150.0, S, K, T, r, True)[0])  # above spot
    assert math.isnan(implied_vol(0.0, S, K, T, r, True)[0])    # at/below intrinsic


def test_vectorized_matches_scalar():
    ks = np.array([80.0, 100.0, 120.0])
    vec = price(S, ks, T, r, SIG, True)
    for i, k in enumerate(ks):
        assert vec[i] == pytest.approx(float(price(S, k, T, r, SIG, True)))
