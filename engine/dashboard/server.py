"""Live dashboard: python -m engine.dashboard  ->  http://localhost:8000

A background task tails the `analytics` stream into `State`; the browser
holds one Server-Sent Events connection and gets a fresh snapshot each second.
The dashboard only *observes* (plain XREAD, not the workers' consumer group),
so opening or closing it never affects pricing.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

from .. import bus
from .state import State

STATIC = Path(__file__).parent / "static"


def create_app(redis_url: str = bus.DEFAULT_URL) -> FastAPI:
    import redis.asyncio as aioredis

    state = State()

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
        tasks = [asyncio.create_task(tail()), asyncio.create_task(poll_backlog())]
        yield
        for t in tasks:
            t.cancel()

    app = FastAPI(title="Options engine dashboard", lifespan=lifespan)

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/snapshot")
    async def snapshot():
        return state.snapshot(bus.now_ms())

    @app.get("/events")
    async def events():
        async def gen():
            while True:
                yield f"data: {json.dumps(state.snapshot(bus.now_ms()))}\n\n"
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
