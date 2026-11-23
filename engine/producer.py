"""Market-data producer: publishes option quotes to the `quotes` stream.

    python -m engine.producer NVDA --source sim      # simulated, moves every second
    python -m engine.producer NVDA --source yahoo    # real quotes, polled

Two sources:

  yahoo  Polls Yahoo Finance every --interval seconds (default 30). Real data,
         but ~15 min delayed and frozen when the market is closed.
  sim    Calibrates to the real NVDA surface once, then moves the stock price
         as a random walk and re-prices every contract from the fitted smile.
         Lets the pipeline and dashboard run live at any hour. --speed 60
         means one wall-clock second = one minute of simulated market time.

Each message is one contract quote and is self-contained (it carries the spot,
rate and time to expiry), so any worker can price it without shared state.
"""
from __future__ import annotations

import argparse
import math
import signal
import time
from datetime import date, datetime

import numpy as np

from . import bus
from .chain import NY, fetch_chain, fetch_inputs, years_to_expiry
from .pricing import price
from .surface import fetch_surface, forward, smile_iv

TRADING_SECONDS_PER_YEAR = 252 * 6.5 * 3600


class Simulator:
    """Geometric Brownian motion spot + quotes priced off a fitted smile.

    Uses a "sticky strike" assumption: each strike keeps its fitted IV as
    spot moves. Simple, and good enough to exercise the pipeline.
    """

    def __init__(self, ticker, spot, r, q, fits, strikes_by_expiry, speed=60.0,
                 spread=0.02, seed=None):
        self.ticker, self.spot, self.r, self.q = ticker, spot, r, q
        self.speed, self.spread = speed, spread
        self.rng = np.random.default_rng(seed)
        f = fits.dropna(subset=["atm_iv"]).set_index("expiry")
        exp, typ, K, T, iv = [], [], [], [], []
        for e, strikes in strikes_by_expiry.items():
            if e not in f.index:
                continue
            row = f.loc[e]
            coef = np.array([row.a, row.b, row.c])
            for k_ in strikes:
                # clamp to the quoted range: never extrapolate the quadratic
                k = min(max(math.log(k_ / row.forward), row.k_min), row.k_max)
                for t in ("call", "put"):
                    exp.append(e); typ.append(t); K.append(k_); T.append(row["T"])
                    iv.append(float(smile_iv(coef, k)))
        self.expiry = np.array(exp)
        self.type = np.array(typ)
        self.K, self.T, self.iv = map(np.array, (K, T, iv))
        nearest = f["T"].idxmin()
        self.vol = float(f.loc[nearest, "atm_iv"])   # drives the random walk

    def step(self, wall_dt: float) -> None:
        dt = wall_dt * self.speed / TRADING_SECONDS_PER_YEAR
        z = self.rng.standard_normal()
        self.spot *= math.exp((self.r - self.q - 0.5 * self.vol**2) * dt
                              + self.vol * math.sqrt(dt) * z)

    def quotes(self, seq: int) -> list[dict]:
        is_call = self.type == "call"
        fair = price(self.spot, self.K, self.T, self.r, self.iv, is_call, self.q)
        half = np.maximum(0.01, fair * self.spread / 2)
        bid = np.maximum(np.round(fair - half, 2), 0.0)
        ask = np.round(fair + half, 2)
        ts = bus.now_ms()
        return [{"ticker": self.ticker, "expiry": e, "T": float(T), "type": t,
                 "strike": float(k), "bid": float(b), "ask": float(a), "last": 0.0,
                 "spot": self.spot, "r": self.r, "q": self.q,
                 "feed": "sim", "seq": seq, "ts": ts}
                for e, T, t, k, b, a in zip(self.expiry, self.T, self.type, self.K, bid, ask)]


def pick_expiries(expiries, today: date, n: int, min_days: int = 7) -> list[str]:
    return [e for e in expiries if (date.fromisoformat(e) - today).days >= min_days][:n]


def near_strikes(chain, spot: float, width: float) -> list[float]:
    ks = sorted(set(chain.strike))
    return [k for k in ks if abs(k - spot) <= width * spot]


def build_simulator(ticker, n_expiries, width, speed, seed=None) -> Simulator:
    import yfinance as yf

    tk = yf.Ticker(ticker)
    exps = pick_expiries(tk.options, datetime.now(NY).date(), n_expiries)
    max_days = (date.fromisoformat(exps[-1]) - datetime.now(NY).date()).days
    print(f"Calibrating to the live {ticker} surface ({len(exps)} expiries)...")
    s = fetch_surface(ticker, min_days=7, max_days=max_days)
    strikes = {e: near_strikes(fetch_chain(tk, e), s.spot, width) for e in exps}
    return Simulator(s.ticker, s.spot, s.r, s.q, s.fits, strikes, speed=speed, seed=seed)


def yahoo_quotes(tk, ticker, expiries, width, seq) -> list[dict]:
    spot, r, q = fetch_inputs(tk)
    now = datetime.now(NY)
    ts = bus.now_ms()
    out = []
    for e in expiries:
        T = years_to_expiry(date.fromisoformat(e), now)
        ch = fetch_chain(tk, e)
        ch = ch[(ch.strike - spot).abs() <= width * spot]
        for c in ch.itertuples(index=False):
            out.append({"ticker": ticker, "expiry": e, "T": T, "type": c.type,
                        "strike": float(c.strike), "bid": float(c.bid), "ask": float(c.ask),
                        "last": float(c.last), "spot": spot, "r": r, "q": q,
                        "feed": "yahoo", "seq": seq, "ts": ts})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ticker")
    ap.add_argument("--source", choices=["sim", "yahoo"], default="sim")
    ap.add_argument("--interval", type=float, help="seconds between updates "
                    "(default 1 for sim, 30 for yahoo)")
    ap.add_argument("--expiries", type=int, default=4, help="how many expiries (default 4)")
    ap.add_argument("--width", type=float, default=0.15, help="strikes within this "
                    "fraction of spot (default 0.15)")
    ap.add_argument("--speed", type=float, default=60.0, help="sim only: market "
                    "seconds per wall second (default 60)")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--redis", default=bus.DEFAULT_URL)
    args = ap.parse_args(argv)
    interval = args.interval or (1.0 if args.source == "sim" else 30.0)

    r = bus.connect(args.redis)
    r.ping()
    stop = False

    def _stop(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    if args.source == "sim":
        sim = build_simulator(args.ticker, args.expiries, args.width, args.speed, args.seed)
        print(f"Simulating {sim.ticker} from {sim.spot:.2f}, {len(sim.K)} contracts, "
              f"vol {sim.vol:.0%}, every {interval}s, speed x{args.speed:g}")
    else:
        import yfinance as yf
        tk = yf.Ticker(args.ticker)
        expiries = pick_expiries(tk.options, datetime.now(NY).date(), args.expiries)
        print(f"Polling Yahoo for {args.ticker.upper()} {', '.join(expiries)} every {interval}s")

    seq, last = 0, time.monotonic()
    while not stop:
        seq += 1
        if args.source == "sim":
            now = time.monotonic()
            sim.step(now - last)
            last = now
            msgs = sim.quotes(seq)
        else:
            msgs = yahoo_quotes(tk, args.ticker.upper(), expiries, args.width, seq)
        bus.publish(r, bus.QUOTES, msgs)
        print(f"tick {seq}: published {len(msgs)} quotes, spot {msgs[0]['spot']:.2f}"
              if msgs else f"tick {seq}: no quotes", flush=True)
        deadline = time.monotonic() + interval
        while not stop and time.monotonic() < deadline:
            time.sleep(0.05)


if __name__ == "__main__":
    main()
