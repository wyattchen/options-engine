"""Message bus: a thin layer over Redis Streams.

Everything that talks to Redis goes through here, so swapping in Kafka later
means rewriting this file, not the producer, workers or dashboard.

Streams:
    quotes     producer -> pricing workers   (one message per contract quote)
    analytics  workers  -> dashboard, risk   (the quote plus IV and Greeks)

A stream is an append-only log. Workers read `quotes` through a *consumer
group*: Redis hands each message to exactly one worker in the group and
remembers which ones were acknowledged, so adding workers splits the load and
a crashed worker's unacknowledged messages can be reclaimed.
"""
from __future__ import annotations

import json
import math
import time

QUOTES = "quotes"
ANALYTICS = "analytics"
PRICERS = "pricers"                  # consumer group of pricing workers
DEFAULT_URL = "redis://localhost:6379/0"
MAXLEN = 200_000                     # keep the logs bounded (approximate trim)


def connect(url: str = DEFAULT_URL):
    import redis
    return redis.Redis.from_url(url, decode_responses=True)


def now_ms() -> float:
    return time.time() * 1000.0


def _jsonable(v):
    """JSON has no NaN; send null so browsers can parse it."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def encode(msg: dict) -> dict:
    return {"data": json.dumps({k: _jsonable(v) for k, v in msg.items()},
                               separators=(",", ":"))}


def decode(fields: dict) -> dict:
    return json.loads(fields["data"])


def publish(r, stream: str, msgs: list[dict], maxlen: int = MAXLEN) -> int:
    """Append many messages in one round trip."""
    pipe = r.pipeline(transaction=False)
    for m in msgs:
        pipe.xadd(stream, encode(m), maxlen=maxlen, approximate=True)
    pipe.execute()
    return len(msgs)


def ensure_group(r, stream: str, group: str) -> None:
    """Create the consumer group (and stream) if missing. Starts at the
    beginning ("0") so nothing published before the first worker is skipped."""
    import redis
    try:
        r.xgroup_create(stream, group, id="0", mkstream=True)
    except redis.ResponseError as e:
        if "BUSYGROUP" not in str(e):
            raise
