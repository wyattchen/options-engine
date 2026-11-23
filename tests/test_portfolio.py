"""Paper-trading account: fills, cash, P&L, risk totals, persistence, API."""
import pytest

from engine import bus
from engine.portfolio import Account, OrderError, contract_key

K = contract_key("NVDA", "2026-12-18", "call", 125)
P = contract_key("NVDA", "2026-12-18", "put", 120)


def acct(cash=10_000.0, fee=0.65):
    return Account(starting_cash=cash, cash=cash, fee_per_contract=fee)


def test_buy_pays_ask_times_100_plus_fees():
    a = acct()
    a.buy(K, 2, ask=5.00, ts=0)
    assert a.cash == pytest.approx(10_000 - 1_000 - 1.30)
    assert a.positions[K].qty == 2 and a.positions[K].avg_price == 5.00


def test_average_cost_across_buys():
    a = acct()
    a.buy(K, 1, ask=4.00, ts=0)
    a.buy(K, 3, ask=6.00, ts=1)
    assert a.positions[K].avg_price == pytest.approx(5.50)


def test_sell_realizes_pnl_and_closes_position():
    a = acct(fee=0)
    a.buy(K, 2, ask=5.00, ts=0)
    t = a.sell(K, 2, bid=6.50, ts=1)
    assert t["realized"] == pytest.approx(300.0)   # (6.50 - 5.00) * 100 * 2
    assert K not in a.positions
    assert a.cash == pytest.approx(10_300.0)


@pytest.mark.parametrize("bad", [
    lambda a: a.buy(K, 1, ask=200.0, ts=0),         # $20,000 > $10,000 cash
    lambda a: a.sell(K, 1, bid=5.0, ts=0),          # don't own it (no writing yet)
    lambda a: a.buy(K, 0, ask=5.0, ts=0),
    lambda a: a.buy(K, 1.5, ask=5.0, ts=0),
    lambda a: a.buy(K, 1, ask=0.0, ts=0),           # no ask
])
def test_bad_orders_are_rejected_and_change_nothing(bad):
    a = acct()
    with pytest.raises(OrderError):
        bad(a)
    assert a.cash == 10_000 and not a.positions and not a.trades


def test_valuation_marks_at_mid_and_scales_greeks_by_position():
    a = acct(fee=0)
    a.buy(K, 2, ask=5.00, ts=0)
    a.buy(P, 1, ask=3.00, ts=0)
    quotes = {K: dict(bid=5.90, ask=6.10, delta=0.55, gamma=0.02, vega_1pct=0.15, theta_day=-0.10),
              P: dict(bid=1.90, ask=2.10, delta=-0.40, gamma=0.03, vega_1pct=0.12, theta_day=-0.08)}
    v = a.valuation(quotes)
    assert v["market_value"] == pytest.approx(6.00 * 200 + 2.00 * 100)
    assert v["unrealized_pnl"] == pytest.approx(200 - 100)          # +$1 x 200, -$1 x 100
    assert v["account_value"] == pytest.approx(v["cash"] + v["market_value"])
    assert v["total_pnl"] == pytest.approx(100)
    assert v["greeks"]["delta"] == pytest.approx(0.55 * 200 - 0.40 * 100)   # 70 shares
    assert v["greeks"]["theta"] == pytest.approx(-0.10 * 200 - 0.08 * 100)


def test_valuation_without_a_quote_falls_back_to_cost():
    a = acct(fee=0)
    a.buy(K, 1, ask=5.0, ts=0)
    v = a.valuation({})
    assert v["positions"][0]["stale"] and v["total_pnl"] == 0


def test_save_and_load_round_trip(tmp_path):
    a = acct()
    a.buy(K, 3, ask=2.5, ts=0)
    a.save(tmp_path / "acct.json")
    b = Account.load(tmp_path / "acct.json")
    assert b.to_dict() == a.to_dict()
    assert Account.load(tmp_path / "missing.json").cash == 100_000


# ---------------------------------------------------------------- HTTP API

@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient
    from engine.dashboard.server import create_app

    app = create_app(account_path=tmp_path / "acct.json", start_tasks=False)
    st = app.state.dashboard
    for kind, bid, ask in (("call", 4.90, 5.10), ("put", 2.90, 3.10)):
        st.ingest(dict(ticker="NVDA", expiry="2026-12-18", type=kind, strike=125.0, bid=bid,
                       ask=ask, spot=125.0, ts=bus.now_ms(), iv=0.5, delta=0.5), bus.now_ms())
    with TestClient(app) as c:
        yield c, app, tmp_path


def test_api_fills_at_server_side_quote_and_saves(client):
    c, app, tmp = client
    r = c.post("/api/order", json=dict(side="buy", expiry="2026-12-18", type="call", strike=125, qty=2))
    assert r.status_code == 200, r.text
    assert r.json()["trade"]["price"] == 5.10                  # the ask, from the stream
    assert Account.load(tmp / "acct.json").positions[K].qty == 2
    r = c.post("/api/order", json=dict(side="sell", expiry="2026-12-18", type="call", strike=125, qty=2))
    assert r.json()["trade"]["price"] == 4.90                  # the bid


def test_api_rejects_unknown_contract_and_stale_prices(client):
    c, app, _ = client
    r = c.post("/api/order", json=dict(side="buy", expiry="2026-12-18", type="call", strike=999, qty=1))
    assert r.status_code == 400 and "No live quote" in r.json()["detail"]
    for row in app.state.dashboard.rows.values():
        row["ts"] -= 60_000
    r = c.post("/api/order", json=dict(side="buy", expiry="2026-12-18", type="put", strike=125, qty=1))
    assert r.status_code == 400 and "stale" in r.json()["detail"]


def test_api_reset_sets_new_starting_cash(client):
    c, app, _ = client
    c.post("/api/order", json=dict(side="buy", expiry="2026-12-18", type="call", strike=125, qty=1))
    r = c.post("/api/reset", json={"cash": 1_000_000})
    acc = r.json()["account"]
    assert acc["cash"] == acc["starting_cash"] == 1_000_000 and acc["positions"] == []
    assert c.get("/api/snapshot").json()["account"]["cash"] == 1_000_000
