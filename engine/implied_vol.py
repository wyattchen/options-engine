"""Implied volatility: invert Black-Scholes to find sigma from a market price.

Strategy: Newton-Raphson (fast, uses vega) with a bisection fallback when
Newton misbehaves (vega near zero for deep ITM/OTM options, or a step that
leaves the bracket). Prices outside no-arbitrage bounds return NaN.
"""
from __future__ import annotations

import math

import numpy as np

from .pricing import greeks, price

SIGMA_LO, SIGMA_HI = 1e-6, 5.0


def _bounds(S, K, T, r, q, is_call):
    """No-arbitrage price bounds for a European option."""
    fwd_s = S * math.exp(-q * T)
    disc_k = K * math.exp(-r * T)
    if is_call:
        return max(fwd_s - disc_k, 0.0), fwd_s
    return max(disc_k - fwd_s, 0.0), disc_k


def implied_vol(market_price, S, K, T, r, is_call=True, q=0.0,
                tol=1e-8, max_iter=100):
    """Solve for sigma. Returns (sigma, iterations, method) or (nan, ...)."""
    lo_b, hi_b = _bounds(S, K, T, r, q, is_call)
    if not (lo_b < market_price < hi_b):
        return float("nan"), 0, "out_of_bounds"
    # Time value too small to pin down sigma: many vols fit within tolerance.
    # (Typical for far out-of-the-money, illiquid strikes.)
    if market_price - lo_b < tol * 10:
        return float("nan"), 0, "unidentifiable"

    def f(sig):
        return float(price(S, K, T, r, sig, is_call, q)) - market_price

    # Newton-Raphson, starting from a reasonable guess
    sigma = 0.2
    lo, hi = SIGMA_LO, SIGMA_HI
    for i in range(1, max_iter + 1):
        diff = f(sigma)
        if abs(diff) < tol:
            return sigma, i, "newton"
        # keep a bracket so we can fall back safely
        if diff > 0:
            hi = sigma
        else:
            lo = sigma
        vega = float(greeks(S, K, T, r, sigma, is_call, q)["vega"])
        if vega < 1e-10:
            break
        step = sigma - diff / vega
        if not (lo < step < hi):
            break  # Newton left the bracket; let bisection finish
        sigma = step

    # Bisection fallback on the bracket we've narrowed so far
    for j in range(1, 200):
        mid = 0.5 * (lo + hi)
        diff = f(mid)
        if abs(diff) < tol or (hi - lo) < 1e-12:
            return mid, max_iter + j, "bisection"
        if diff > 0:
            hi = mid
        else:
            lo = mid
    return float("nan"), max_iter + 200, "no_convergence"


def implied_vol_vec(prices, S, K, T, r, is_call, q=0.0):
    """Convenience loop over arrays. (Phase 2 will vectorize/parallelize this.)"""
    out = np.empty(len(prices))
    for i in range(len(prices)):
        out[i] = implied_vol(prices[i], S[i], K[i], T[i], r, bool(is_call[i]), q)[0]
    return out
