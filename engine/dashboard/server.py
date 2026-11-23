"""Live dashboard: python -m engine.dashboard  ->  http://localhost:8000

A background task tails the `analytics` stream into `State`; the browser
holds one Server-Sent Events connection and gets a fresh snapshot each second.
The dashboard only *observes* (plain XREAD, not the workers' consumer group),
so opening or closing it never affects pricing.

It also hosts the paper-trading account: orders fill against the latest
quote the server has seen (never a price sent by the browser), and the
account is saved to data/account.json after every change.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from .. import bus
from ..portfolio import DEFAULT_CASH, Account, OrderError, contract_key
from .state import State

STATIC = Path(__file__).parent / "static"
ACCOUNT_PATH = Path("data") / "account.json"
MAX_QUOTE_AGE_MS = 15_000      # refuse to trade on prices older than this


class Order(BaseModel):
    expiry: str
    type: str                  # "call" | "put"
    strike: float
    side: str                  # "buy" | "sell"
    qty: int
    ticker: str | None = None  # defaults to the ticker on the stream


class Reset(BaseModel):
    cash: float = DEFAULT_CASH


def create_app(redis_url: str = bus.DEFAULT_URL, account_path: Path = ACCOUNT_PATH,
               start_tasks: bool = True) -> FastAPI:
    import redis.asyncio as aioredis

    state = State()
    account = Account.load(account_path)
    lock = asyncio.Lock()      # one order at a time

    def quotes() -> dict[str, dict]:
        return {contract_key(r.get("ticker") or state.ticker, r["expiry"], r["type"], r["strike"]): r
                for r in state.rows.values()}

    def full_snapshot() -> dict:
        snap = state.snapshot(bus.now_ms())
        snap["account"] = account.valuation(quotes())
        return snap

    async def tail():
        r = aioredis.Redis.from_url(redis_url, decode_responses=True)
        last_id = "$"            # only new results; history isn't needed for a live view
        while True:
            try:
                resp = await r.xread({bus.ANALYTICS: last_id}, count=2000, block=1000)
                now = bus.now_ms()
                for _, entries in resp or []:
                    for mid, fields in entries:
                        state.ingest(bus.decode(fields), now)
                        last_id = mid
            except Exception as e:  # Redis restarting etc.: keep the page alive
                print("dashboard: stream read failed:", e)
                await asyncio.sleep(1)

    async def poll_backlog():
        r = aioredis.Redis.from_url(redis_url, decode_responses=True)
        while True:
            try:
                for g in await r.xinfo_groups(bus.QUOTES):
                    if g["name"] == bus.PRICERS:
                        state.backlog = g.get("lag")       # published, not yet read
                        state.pending = g.get("pending")   # read, not yet acknowledged
            except Exception:
                state.backlog = state.pending = None
            await asyncio.sleep(1)

    @asynccontextmanager
    async def lifespan(app):
        tasks = [asyncio.create_task(tail()), asyncio.create_task(poll_backlog())] \
            if start_tasks else []
        yield
        for t in tasks:
            t.cancel()

    app = FastAPI(title="Options engine dashboard", lifespan=lifespan)
    app.state.dashboard = state          # exposed for tests
    app.state.account = account

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/snapshot")
    async def snapshot():
        return full_snapshot()

    @app.post("/api/order")
    async def order(o: Order):
        if o.side not in ("buy", "sell") or o.type not in ("call", "put"):
            raise HTTPException(400, "side must be buy/sell and type call/put")
        key = contract_key(o.ticker or state.ticker or "", o.expiry, o.type, o.strike)
        async with lock:
            q = quotes().get(key)
            now = bus.now_ms()
            if q is None:
                raise HTTPException(400, "No live quote for that contract. Is the pipeline running?")
            age = now - q["ts"]
            if age > MAX_QUOTE_AGE_MS:
                raise HTTPException(400, f"The last price is {age / 1000:.0f}s old, too stale to "
                                         "trade on. Is the producer running?")
            try:
                if o.side == "buy":
                    trade = account.buy(key, o.qty, q.get("ask"), now)
                else:
                    trade = account.sell(key, o.qty, q.get("bid"), now)
            except OrderError as e:
                raise HTTPException(400, str(e))
            account.save(account_path)
        return {"trade": trade, "account": account.valuation(quotes())}

    @app.post("/api/reset")
    async def reset(r: Reset):
        if not 0 < r.cash <= 1e12:
            raise HTTPException(400, "Starting cash must be more than $0.")
        async with lock:
            fresh = Account(starting_cash=r.cash, cash=r.cash)
            account.__dict__.update(fresh.__dict__)   # keep the same object
            account.save(account_path)
        return {"account": account.valuation(quotes())}

    @app.get("/events")
    async def events():
        async def gen():
            while True:
                yield f"data: {json.dumps(full_snapshot())}\n\n"
                await asyncio.sleep(1)
        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    return app


def main(argv=None):
    import uvicorn

    ap = argparse.ArgumentParser(description="Live options engine dashboard")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--redis", default=bus.DEFAULT_URL)
    args = ap.parse_args(argv)
    print(f"Dashboard on http://localhost:{args.port}")
    uvicorn.run(create_app(args.redis), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
