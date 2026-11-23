"""Offline tests for the vol surface: synthetic chains with a known smile."""
import math

import numpy as np
import pandas as pd
import pytest

from engine import price
from engine.surface import build_surface, clean, fit_smile, forward, select_otm, smile_iv

S, r, q = 125.0, 0.04, 0.004
TRUE = {"a": 0.50, "b": -0.30, "c": 0.40}  # ATM 50%, downside skew, some smile


def true_iv(k):
    return TRUE["a"] + TRUE["b"] * k + TRUE["c"] * k**2


def synthetic_chain(T, strikes, last_trade=None):
    F = forward(S, T, r, q)
    rows = []
    for K in strikes:
        v = true_iv(math.log(K / F))
        for kind in ("call", "put"):
            p = float(price(S, K, T, r, v, kind == "call", q))
            rows.append(dict(type=kind, strike=K, bid=p - 0.005, ask=p + 0.005, last=0.0,
                             last_trade=last_trade or pd.Timestamp("2026-11-20 21:00", tz="UTC")))
    return pd.DataFrame(rows)


def test_fit_smile_recovers_exact_quadratic():
    k = np.linspace(-0.3, 0.3, 15)
    coef = fit_smile(k, true_iv(k), weights=np.ones_like(k))
    assert coef == pytest.approx([TRUE["a"], TRUE["b"], TRUE["c"]], abs=1e-10)
    assert smile_iv(coef, 0.1) == pytest.approx(true_iv(0.1))


def test_select_otm_keeps_puts_below_and_calls_above_forward():
    res = pd.DataFrame({"type": ["put", "put", "call", "call"],
                        "strike": [100.0, 130.0, 100.0, 130.0]})
    out = select_otm(res, F=125.0)
    assert list(zip(out.type, out.strike)) == [("put", 100.0), ("call", 130.0)]


def test_clean_drops_stale_cheap_and_unsolved_quotes():
    now = pd.Timestamp("2026-11-20 21:00", tz="UTC")
    res = pd.DataFrame({
        "iv":    [0.5, np.nan, 0.5, 0.5, 0.5],
        "price": [2.0, 2.0, 0.01, 2.0, 2.0],
        "delta": [0.4, 0.4, 0.4, 0.4, 0.4],
        "src":   ["last", "last", "last", "last", "mid"],
        "last_trade": [now, now, now, now - pd.Timedelta(hours=3), now - pd.Timedelta(hours=3)],
    })
    kept = clean(res, latest_trade=now, max_stale_min=60)
    # row 0 good; 1 no IV; 2 too cheap; 3 stale last trade; 4 old but a live mid quote
    assert list(kept.index) == [0, 4]


def test_build_surface_recovers_known_smile_per_expiry():
    strikes = np.arange(80.0, 175.0, 5.0)
    chains = {"2026-12-18": (30 / 365, synthetic_chain(30 / 365, strikes)),
              "2027-03-19": (120 / 365, synthetic_chain(120 / 365, strikes))}
    points, fits = build_surface(chains, S, r, q)
    assert len(fits) == 2 and (fits.n >= 5).all()
    for row in fits.itertuples():
        # bid/ask is +-half a cent, so recovered IVs are near-exact
        assert row.atm_iv == pytest.approx(TRUE["a"], abs=2e-3)
        assert row.b == pytest.approx(TRUE["b"], abs=2e-2)
        assert row.skew_1sd > 0          # downside vol is higher
        assert row.rmse < 2e-3
    assert set(points.expiry) == set(chains)


def test_build_surface_skips_expiry_with_too_few_quotes():
    chains = {"2026-12-18": (30 / 365, synthetic_chain(30 / 365, [120.0, 125.0, 130.0]))}
    points, fits = build_surface(chains, S, r, q)
    assert points.empty
    assert math.isnan(fits.iloc[0].get("atm_iv", float("nan")))
