"""Dashboard state: folds the `analytics` stream into a live snapshot.

Kept free of Redis and web code so it can be unit-tested directly.
"""
from __future__ import annotations

from collections import deque

import numpy as np

WINDOW_MS = 5_000          # throughput / latency window
SPOT_HISTORY = 600         # points kept for the spot chart


class State:
    def __init__(self):
        self.rows: dict[tuple, dict] = {}            # latest result per contract
        self.spot_hist: deque = deque(maxlen=SPOT_HISTORY)
        self.recent: deque = deque()                 # (arrived_ms, e2e_ms, compute_ms)
        self.ticker = self.feed = None
        self.tickers: set[str] = set()               # every ticker seen on the stream
        self.spot = self.spot_open = None
        self.workers: dict[str, float] = {}          # worker -> last seen ms
        self.total = 0
        self.backlog = self.pending = None           # filled from Redis by the server

    def ingest(self, msg: dict, now: float) -> None:
        key = (msg["expiry"], msg["type"], msg["strike"])
        self.rows[key] = msg
        self.total += 1
        self.ticker, self.feed = msg.get("ticker"), msg.get("feed")
        if self.ticker:
            self.tickers.add(self.ticker)
        spot = msg["spot"]
        if self.spot_open is None:
            self.spot_open = spot
        if spot != self.spot or not self.spot_hist:
            self.spot_hist.append((msg["ts"], spot))
        self.spot = spot
        self.recent.append((now, now - msg["ts"], msg.get("ts_priced", now) - msg["ts"]))
        if msg.get("worker"):
            self.workers[msg["worker"]] = now

    def _trim(self, now: float) -> None:
        while self.recent and self.recent[0][0] < now - WINDOW_MS:
            self.recent.popleft()

    def metrics(self, now: float) -> dict:
        self._trim(now)
        e2e = np.array([x[1] for x in self.recent]) if self.recent else np.array([])
        comp = np.array([x[2] for x in self.recent]) if self.recent else np.array([])
        pct = lambda a, p: float(np.percentile(a, p)) if len(a) else None
        return {
            "msgs_per_s": len(self.recent) / (WINDOW_MS / 1000),
            "e2e_p50_ms": pct(e2e, 50), "e2e_p99_ms": pct(e2e, 99),
            "worker_p50_ms": pct(comp, 50),
            "workers_active": sum(1 for t in self.workers.values() if now - t < WINDOW_MS),
            "backlog": self.backlog, "pending": self.pending, "total": self.total,
        }

    def atm_iv(self) -> dict:
        """Per expiry: mean call/put IV at the strike closest to spot."""
        out = {}
        if self.spot is None:
            return out
        by_exp: dict[str, list] = {}
        for (e, _, k), row in self.rows.items():
            by_exp.setdefault(e, []).append(row)
        for e, rows in by_exp.items():
            k = min({r["strike"] for r in rows}, key=lambda s: abs(s - self.spot))
            ivs = [r["iv"] for r in rows if r["strike"] == k and r.get("iv") is not None]
            out[e] = sum(ivs) / len(ivs) if ivs else None
        return out

    def snapshot(self, now: float) -> dict:
        keep = ("expiry", "type", "strike", "bid", "ask", "price", "iv",
                "delta", "gamma", "vega_1pct", "theta_day", "T")
        return {
            "ticker": self.ticker, "tickers": sorted(self.tickers), "feed": self.feed,
            "spot": self.spot, "spot_open": self.spot_open,
            "spot_hist": list(self.spot_hist),
            "expiries": sorted({k[0] for k in self.rows}),
            "atm_iv": self.atm_iv(),
            "rows": [{f: row.get(f) for f in keep} for row in self.rows.values()],
            "metrics": self.metrics(now),
            "now": now,
        }
