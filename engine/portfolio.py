"""Paper-trading account and portfolio risk.

Rules (basic version):
  - one contract = 100 shares; prices are quoted per share
  - buys fill at the ask, sells at the bid (you pay the spread, as in a real market)
  - a flat commission per contract (default $0.65, a typical retail rate)
  - long only: you can sell what you hold, but not sell to open (no margin yet)
  - positions are marked at the bid/ask midpoint for P&L

Pure logic: no Redis, no web. The dashboard server owns one Account, feeds it
live quotes and saves it to disk after every trade.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

MULTIPLIER = 100
DEFAULT_CASH = 100_000.0
DEFAULT_FEE = 0.65


class OrderError(ValueError):
    """An order that can't be filled; the message is shown to the user."""


def contract_key(ticker: str, expiry: str, kind: str, strike: float) -> str:
    return f"{ticker}|{expiry}|{kind}|{float(strike):g}"


def parse_key(key: str) -> dict:
    ticker, expiry, kind, strike = key.split("|")
    return {"ticker": ticker, "expiry": expiry, "type": kind, "strike": float(strike)}


def describe(key: str) -> str:
    c = parse_key(key)
    return f"{c['ticker']} {c['expiry']} {c['strike']:g} {c['type']}"


@dataclass
class Position:
    qty: int = 0
    avg_price: float = 0.0          # per share, fees excluded


@dataclass
class Account:
    starting_cash: float = DEFAULT_CASH
    cash: float = DEFAULT_CASH
    fee_per_contract: float = DEFAULT_FEE
    realized_pnl: float = 0.0       # closed trades, after fees
    fees_paid: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    trades: list[dict] = field(default_factory=list)

    # ------------------------------------------------------------ trading

    def buy(self, key: str, qty: int, ask: float | None, ts: float) -> dict:
        qty = _check_qty(qty)
        if not ask or ask <= 0:
            raise OrderError("No ask price for this contract right now, so it can't be bought.")
        fees = self.fee_per_contract * qty
        cost = ask * MULTIPLIER * qty + fees
        if cost > self.cash + 1e-9:
            raise OrderError(f"Not enough cash: this costs ${cost:,.2f} and you have ${self.cash:,.2f}.")
        pos = self.positions.setdefault(key, Position())
        pos.avg_price = (pos.avg_price * pos.qty + ask * qty) / (pos.qty + qty)
        pos.qty += qty
        self.cash -= cost
        self.fees_paid += fees
        self.realized_pnl -= fees
        return self._record("buy", key, qty, ask, fees, ts, realized=-fees)

    def sell(self, key: str, qty: int, bid: float | None, ts: float) -> dict:
        qty = _check_qty(qty)
        pos = self.positions.get(key)
        held = pos.qty if pos else 0
        if qty > held:
            raise OrderError(f"You hold {held} of this contract, so you can't sell {qty}. "
                             "Selling options you don't own (writing) needs margin, "
                             "which isn't supported yet.")
        if bid is None or bid < 0:
            raise OrderError("No bid price for this contract right now, so it can't be sold.")
        fees = self.fee_per_contract * qty
        proceeds = bid * MULTIPLIER * qty - fees
        realized = (bid - pos.avg_price) * MULTIPLIER * qty - fees
        pos.qty -= qty
        if pos.qty == 0:
            del self.positions[key]
        self.cash += proceeds
        self.fees_paid += fees
        self.realized_pnl += realized
        return self._record("sell", key, qty, bid, fees, ts, realized=realized)

    def _record(self, side, key, qty, px, fees, ts, realized) -> dict:
        t = {"ts": ts, "side": side, "contract": key, "label": describe(key), "qty": qty,
             "price": px, "fees": fees, "realized": realized, "cash_after": self.cash}
        self.trades.append(t)
        return t

    # ------------------------------------------------------------ valuation

    def valuation(self, quotes: dict[str, dict]) -> dict:
        """Mark every position to market and add up its risk.

        quotes maps contract key -> latest worker result (bid, ask, iv, Greeks).
        Greeks are scaled to the position: delta is in share-equivalents
        (+250 = moves like owning 250 shares), vega and theta are in dollars.
        """
        rows, mv_total = [], 0.0
        tot = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
        for key, pos in self.positions.items():
            q = quotes.get(key) or {}
            bid, ask = q.get("bid"), q.get("ask")
            mark = (bid + ask) / 2 if bid is not None and ask and ask >= bid else None
            size = pos.qty * MULTIPLIER
            mv = mark * size if mark is not None else pos.avg_price * size  # fallback: cost
            cost = pos.avg_price * size
            row = {"contract": key, "label": describe(key), **parse_key(key),
                   "qty": pos.qty, "avg_price": pos.avg_price, "mark": mark,
                   "bid": bid, "ask": ask, "iv": q.get("iv"),
                   "market_value": mv, "unrealized": mv - cost,
                   "unrealized_pct": (mv - cost) / cost if cost else None,
                   "stale": mark is None}
            for g, src in (("delta", "delta"), ("gamma", "gamma"),
                           ("vega", "vega_1pct"), ("theta", "theta_day")):
                v = q.get(src)
                row[g] = v * size if v is not None else None
                if v is not None:
                    tot[g] += v * size
            rows.append(row)
            mv_total += mv
        value = self.cash + mv_total
        return {
            "starting_cash": self.starting_cash, "cash": self.cash,
            "market_value": mv_total, "account_value": value,
            "total_pnl": value - self.starting_cash,
            "total_pnl_pct": (value - self.starting_cash) / self.starting_cash,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": sum(r["unrealized"] for r in rows),
            "fees_paid": self.fees_paid,
            "fee_per_contract": self.fee_per_contract,
            "greeks": tot, "positions": sorted(rows, key=lambda r: r["contract"]),
            "trades": self.trades[-50:][::-1],
        }

    # ------------------------------------------------------------ persistence

    def to_dict(self) -> dict:
        d = asdict(self)
        d["positions"] = {k: asdict(p) for k, p in self.positions.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Account":
        d = dict(d)
        d["positions"] = {k: Position(**p) for k, p in d.get("positions", {}).items()}
        return cls(**d)

    def save(self, path: Path) -> None:
        """Atomic write: a crash mid-save can't leave a half-written file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(self.to_dict(), f, indent=1)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "Account":
        if path.exists():
            return cls.from_dict(json.loads(path.read_text()))
        return cls()


def _check_qty(qty) -> int:
    try:
        n = int(qty)
    except (TypeError, ValueError):
        raise OrderError("Quantity must be a whole number of contracts.")
    if n != qty or n <= 0:
        raise OrderError("Quantity must be a whole number of contracts, 1 or more.")
    return n
