"""Black-Scholes-Merton pricing and Greeks.

All functions are vectorized: pass scalars or NumPy arrays of equal shape.
Conventions:
    S     spot price of the underlying
    K     strike
    T     time to expiry in years (must be > 0)
    r     continuously compounded risk-free rate
    q     continuous dividend yield
    sigma volatility (annualized)
    is_call  True for calls, False for puts
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr

# Standard normal CDF and PDF. scipy.special.ndtr gives the same values as
# scipy.stats.norm.cdf without its per-call argument checking, which made up
# ~85% of the implied-vol solver's run time when pricing one quote at a time.
_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _cdf(x):
    return ndtr(x)


def _pdf(x):
    return np.exp(-0.5 * np.square(x)) / _SQRT_2PI


def _d1_d2(S, K, T, r, q, sigma):
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    sqrt_t = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return d1, d2


def price(S, K, T, r, sigma, is_call=True, q=0.0):
    """Black-Scholes-Merton option price."""
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    disc_s = S * np.exp(-q * np.asarray(T))
    disc_k = K * np.exp(-r * np.asarray(T))
    call = disc_s * _cdf(d1) - disc_k * _cdf(d2)
    put = disc_k * _cdf(-d2) - disc_s * _cdf(-d1)
    return np.where(is_call, call, put)


def greeks(S, K, T, r, sigma, is_call=True, q=0.0):
    """Return a dict of delta, gamma, vega, theta, rho.

    vega and rho are per 1.00 change (divide by 100 for per-1% figures);
    theta is per year (divide by 365 for per-day).
    """
    T = np.asarray(T, dtype=float)
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    sqrt_t = np.sqrt(T)
    pdf_d1 = _pdf(d1)
    eq, er = np.exp(-q * T), np.exp(-r * T)

    gamma = eq * pdf_d1 / (S * sigma * sqrt_t)
    vega = S * eq * pdf_d1 * sqrt_t

    call_delta = eq * _cdf(d1)
    put_delta = call_delta - eq

    common_theta = -S * eq * pdf_d1 * sigma / (2 * sqrt_t)
    call_theta = common_theta - r * K * er * _cdf(d2) + q * S * eq * _cdf(d1)
    put_theta = common_theta + r * K * er * _cdf(-d2) - q * S * eq * _cdf(-d1)

    call_rho = K * T * er * _cdf(d2)
    put_rho = -K * T * er * _cdf(-d2)

    return {
        "delta": np.where(is_call, call_delta, put_delta),
        "gamma": gamma,
        "vega": vega,
        "theta": np.where(is_call, call_theta, put_theta),
        "rho": np.where(is_call, call_rho, put_rho),
    }
