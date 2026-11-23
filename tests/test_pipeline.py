"""Streaming pipeline tests.

Unit tests run offline. The integration test needs a local Redis and skips
itself otherwise; it uses database 15 so it never touches the live pipeline.
"""
import json
import math

import numpy as np
import pandas as pd
import pytest

from engine import bus, price
from engine.dashboard.state import State
from engine.producer import Simulator
from engine.worker import process, work_once

S, r, q, T = 125.0, 0.04, 0.004, 30 / 365


def quote(kind="call", K=125.0, sigma=0.5, **kw):
    p = float(price(S, K, T, r, sigma, kind == "call", q))
    base = dict(ticker="NVDA", expiry="2026-12-18", T=T, type=kind, strike=K,
                bid=round(p - 0.01, 4), ask=round(p + 0.01, 4), last=0.0,
                spot=S, r=r, q=q, feed="sim", seq=1, ts=bus.now_ms())
    return base | kw


def fits_frame():
    return pd.DataFrame([{"expiry": "2026-12-18", "T": T, "forward": S * math.exp((r - q) * T),
                          "a": 0.5, "b": -0.2, "c": 0.3, "atm_iv": 0.5,
                          "k_min": -0.3, "k_max": 0.3}])


# ---------------------------------------------------------------- unit

def test_encode_turns_nan_into_null_for_browsers():
    out = bus.encode({"iv": float("nan"), "x": 1.5})
    assert json.loads(out["data"]) == {"iv": None, "x": 1.5}
    assert bus.decode(out) == {"iv": None, "x": 1.5}


def test_process_recovers_vol_and_greeks():
    out = process(quote(sigma=0.45))
    assert out["src"] == "mid"
    assert out["iv"] == pytest.approx(0.45, abs=1e-4)
    assert 0.4 < out["delta"] < 0.7 and out["theta_day"] < 0
    assert out["ts_priced"] >= out["ts"]


def test_process_handles_missing_and_expired_quotes():
    assert process(quote(bid=0.0, ask=0.0, last=0.0))["iv_status"] == "no_price"
    assert process(quote(T=0.0))["iv_status"] == "expired"


def test_simulator_quotes_price_back_to_the_fitted_smile():
    sim = Simulator("NVDA", S, r, q, fits_frame(), {"2026-12-18": [110.0, 125.0, 140.0]}, seed=1)
    msgs = sim.quotes(seq=1)
    assert len(msgs) == 6
    for m in msgs:
        assert 0 <= m["bid"] < m["ask"]
        k = math.log(m["strike"] / fits_frame().forward[0])
        expected = 0.5 - 0.2 * k + 0.3 * k**2
        # quotes are rounded to cents, so allow a little IV slack
        assert process(m)["iv"] == pytest.approx(expected, abs=0.01)


def test_simulator_random_walk_is_seeded_and_moves():
    a = Simulator("NVDA", S, r, q, fits_frame(), {"2026-12-18": [125.0]}, seed=7)
    b = Simulator("NVDA", S, r, q, fits_frame(), {"2026-12-18": [125.0]}, seed=7)
    for _ in range(20):
        a.step(1.0), b.step(1.0)
    assert a.spot == b.spot != S


def test_dashboard_state_tracks_latest_quote_and_metrics():
    st = State()
    now = bus.now_ms()
    for i, K in enumerate([120.0, 125.0, 130.0]):
        st.ingest(process(quote(K=K, ts=now - 50)) | {"worker": "w1"}, now)
    st.ingest(process(quote(K=125.0, ts=now - 10)) | {"worker": "w2"}, now)  # replaces
    snap = st.snapshot(now)
    assert len(snap["rows"]) == 3
    assert snap["metrics"]["workers_active"] == 2
    assert snap["metrics"]["e2e_p50_ms"] == pytest.approx(50, abs=1)
    assert snap["atm_iv"]["2026-12-18"] == pytest.approx(0.5, abs=1e-3)
    json.dumps(snap)  # must be serialisable for the browser


# ---------------------------------------------------------------- Redis

@pytest.fixture
def redis_db():
    try:
        conn = bus.connect("redis://localhost:6379/15")
        conn.ping()
    except Exception:
        pytest.skip("Redis not running")
    conn.flushdb()
    yield conn
    conn.flushdb()


def test_quotes_flow_through_redis_and_are_acknowledged(redis_db):
    bus.ensure_group(redis_db, bus.QUOTES, bus.PRICERS)
    sent = [quote(K=K) for K in (115.0, 125.0, 135.0)]
    bus.publish(redis_db, bus.QUOTES, sent)

    # two workers share the stream: each message goes to exactly one
    n1 = work_once(redis_db, "w1", batch=2, block_ms=100)
    n2 = work_once(redis_db, "w2", batch=2, block_ms=100)
    assert (n1, n2) == (2, 1)

    out = [bus.decode(f) for _, f in redis_db.xrange(bus.ANALYTICS)]
    assert sorted(m["strike"] for m in out) == [115.0, 125.0, 135.0]
    assert {m["worker"] for m in out} == {"w1", "w2"}
    assert all(m["iv"] == pytest.approx(0.5, abs=1e-3) for m in out)
    assert redis_db.xpending(bus.QUOTES, bus.PRICERS)["pending"] == 0
