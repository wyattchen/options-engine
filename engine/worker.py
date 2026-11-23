"""Pricing worker: reads `quotes`, solves IV and Greeks, writes `analytics`.

    python -m engine.worker                 # name defaults to worker-<pid>
    python -m engine.worker --name w2       # run several to share the load

Workers join the `pricers` consumer group, so Redis gives each quote to exactly
one of them. A message is acknowledged only after its result is published:
if a worker dies mid-batch, those quotes stay pending rather than vanish
(reclaiming them automatically is Phase 3 fault-tolerance work).
"""
from __future__ import annotations

import argparse
import math
import os
import signal

from . import bus
from .chain import market_price
from .implied_vol import implied_vol
from .pricing import greeks


def process(quote: dict) -> dict:
    """Price one quote. Pure function: no Redis, easy to test."""
    is_call = quote["type"] == "call"
    S, K, T = quote["spot"], quote["strike"], quote["T"]
    r, q = quote["r"], quote["q"]
    px, src = market_price(quote["bid"] or 0.0, quote["ask"] or 0.0, quote["last"] or 0.0)
    out = dict(quote, price=px, src=src, iv=float("nan"), iv_status="no_price",
               delta=None, gamma=None, vega_1pct=None, theta_day=None)
    if T <= 0:
        out["iv_status"] = "expired"
    elif not math.isnan(px):
        sigma, _, status = implied_vol(px, S, K, T, r, is_call, q)
        out.update(iv=sigma, iv_status=status)
        if not math.isnan(sigma):
            g = greeks(S, K, T, r, sigma, is_call, q)
            out.update(delta=float(g["delta"]), gamma=float(g["gamma"]),
                       vega_1pct=float(g["vega"]) / 100, theta_day=float(g["theta"]) / 365)
    out["ts_priced"] = bus.now_ms()
    return out


def work_once(r, name: str, batch: int = 100, block_ms: int = 1000) -> int:
    """Read up to `batch` quotes, price them, publish, acknowledge. Returns count."""
    resp = r.xreadgroup(bus.PRICERS, name, {bus.QUOTES: ">"}, count=batch, block=block_ms)
    if not resp:
        return 0
    _, entries = resp[0]
    ids = [mid for mid, _ in entries]
    results = [process(bus.decode(fields)) | {"worker": name} for _, fields in entries]
    bus.publish(r, bus.ANALYTICS, results)
    r.xack(bus.QUOTES, bus.PRICERS, *ids)      # only after results are safe
    return len(ids)


def run(url: str, name: str, batch: int = 100, block_ms: int = 1000) -> None:
    r = bus.connect(url)
    bus.ensure_group(r, bus.QUOTES, bus.PRICERS)
    stop = False

    def _stop(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    done = 0
    print(f"{name}: waiting for quotes on '{bus.QUOTES}'", flush=True)
    while not stop:
        n = work_once(r, name, batch, block_ms)
        done += n
        if n and done % 5000 < n:
            print(f"{name}: priced {done} quotes", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default=f"worker-{os.getpid()}")
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--redis", default=bus.DEFAULT_URL)
    args = ap.parse_args(argv)
    run(args.redis, args.name, args.batch)


if __name__ == "__main__":
    main()
