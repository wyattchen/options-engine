"""Offline tests for the chain pipeline: no network, synthetic market data."""
import math
from datetime import date, datetime

import pandas as pd
import pytest

from engine import price
from engine.chain import NY, analyze, market_price, pick_expiry, years_to_expiry

S, T, r, q = 125.0, 30 / 365, 0.04, 0.004


def test_market_price_prefers_mid_then_last():
    assert market_price(1.0, 1.2, 5.0) == pytest.approx((1.1, "mid"))
    assert market_price(0.0, 0.0, 5.0) == (5.0, "last")     # market closed
    assert market_price(1.2, 1.0, 5.0) == (5.0, "last")     # crossed quote
    assert math.isnan(market_price(0.0, 0.0, 0.0)[0])


def test_years_to_expiry_uses_4pm_new_york():
    now = datetime(2026, 12, 17, 16, 0, tzinfo=NY)
    assert years_to_expiry(date(2026, 12, 18), now) == pytest.approx(1 / 365)
    assert years_to_expiry(date(2026, 12, 1), now) == 0.0   # already expired


def test_pick_expiry_skips_very_short_dated():
    exps = ["2026-11-23", "2026-11-27", "2026-12-04"]
    assert pick_expiry(exps, date(2026, 11, 22)) == "2026-12-04"
    assert pick_expiry(exps, date(2026, 12, 20)) == "2026-12-04"  # none qualify


def test_analyze_recovers_known_vols():
    # Build a chain priced with a volatility skew, then check analyze() reads it back.
    vols = {110.0: 0.55, 125.0: 0.45, 140.0: 0.40}
    rows = []
    for k, v in vols.items():
        for kind in ("call", "put"):
            p = float(price(S, k, T, r, v, kind == "call", q))
            rows.append(dict(type=kind, strike=k, bid=p - 0.01, ask=p + 0.01, last=0.0))
    res = analyze(pd.DataFrame(rows), S, T, r, q)
    for row in res.itertuples():
        assert row.src == "mid"
        assert row.iv == pytest.approx(vols[row.strike], abs=1e-6)
    assert ((res.type == "call") == (res.delta > 0)).all()


def test_analyze_flags_bad_prices_instead_of_guessing():
    rows = [dict(type="call", strike=125.0, bid=0, ask=0, last=0),          # no price
            dict(type="call", strike=125.0, bid=0, ask=0, last=200.0)]      # above spot
    res = analyze(pd.DataFrame(rows), S, T, r, q)
    assert list(res.iv_status) == ["no_price", "out_of_bounds"]
    assert res.iv.isna().all()
