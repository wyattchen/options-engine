"""Price a real options chain: fetch market data, solve IV, compute Greeks.

    python -m engine.chain NVDA                     # first expiry >= 7 days out
    python -m engine.chain NVDA --expiry 2026-12-18 --width 0.1

Data comes from Yahoo Finance via yfinance (free, ~15 min delayed). The fetching
(`fetch_snapshot`) is kept separate from the maths (`analyze`) so the maths can
be tested offline and the data source swapped out later (e.g. a live feed).
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .implied_vol import implied_vol
from .pricing import greeks

NY = ZoneInfo("America/New_York")
SECONDS_PER_YEAR = 365.0 * 24 * 3600


@dataclass
class Snapshot:
    ticker: str
    spot: float
    r: float          # risk-free rate, decimal
    q: float          # dividend yield, decimal
    expiry: date
    T: float          # years to expiry
    asof: datetime
    chain: pd.DataFrame  # columns: type, strike, bid, ask, last, volume, open_interest, yahoo_iv


def years_to_expiry(expiry: date, now: datetime) -> float:
    """US equity options stop trading at 4pm New York time on expiry day."""
    close = datetime(expiry.year, expiry.month, expiry.day, 16, 0, tzinfo=NY)
    return max((close - now).total_seconds(), 0.0) / SECONDS_PER_YEAR


def pick_expiry(expiries: list[str], today: date, min_days: int = 7) -> str:
    """Nearest expiry at least `min_days` away. Very short-dated options have
    tiny time values, which makes IV noisy and Greeks extreme."""
    for e in expiries:
        if (date.fromisoformat(e) - today).days >= min_days:
            return e
    return expiries[-1]


def market_price(bid: float, ask: float, last: float) -> tuple[float, str]:
    """Mid of bid/ask when there's a live two-sided quote, else last trade.

    Last trade is a fallback only: it can be hours old and out of sync with
    the spot price (common on weekends, when Yahoo shows bid = ask = 0).
    """
    if bid > 0 and ask > 0 and ask >= bid:
        return 0.5 * (bid + ask), "mid"
    if last > 0:
        return last, "last"
    return float("nan"), "none"


def analyze(chain: pd.DataFrame, S: float, T: float, r: float, q: float) -> pd.DataFrame:
    """Solve IV and Greeks for every contract. Pure function: no network."""
    rows = []
    for c in chain.itertuples(index=False):
        is_call = c.type == "call"
        px, src = market_price(c.bid, c.ask, c.last)
        sigma, method = float("nan"), "no_price"
        if not math.isnan(px):
            sigma, _, method = implied_vol(px, S, c.strike, T, r, is_call, q)
        row = {"type": c.type, "strike": c.strike, "bid": c.bid, "ask": c.ask,
               "price": px, "src": src, "iv": sigma, "iv_status": method,
               "yahoo_iv": getattr(c, "yahoo_iv", float("nan")),
               "last_trade": getattr(c, "last_trade", pd.NaT)}
        if not math.isnan(sigma):
            g = greeks(S, c.strike, T, r, sigma, is_call, q)
            row.update(delta=float(g["delta"]), gamma=float(g["gamma"]),
                       vega_1pct=float(g["vega"]) / 100,      # $ per 1 vol point
                       theta_day=float(g["theta"]) / 365)     # $ per calendar day
        rows.append(row)
    return pd.DataFrame(rows)


def fetch_inputs(tk) -> tuple[float, float, float]:
    """Spot, risk-free rate and dividend yield for a yfinance Ticker."""
    import yfinance as yf

    spot = float(tk.fast_info["last_price"])
    # 13-week T-bill yield, quoted in percent. A reasonable short-rate proxy.
    r = float(yf.Ticker("^IRX").fast_info["last_price"]) / 100
    # yfinance's `dividendYield` is in percent; the trailing one is a decimal.
    q = float(tk.info.get("trailingAnnualDividendYield") or 0.0)
    return spot, r, q


def fetch_chain(tk, expiry: str) -> pd.DataFrame:
    """Calls and puts for one expiry, in this module's column layout."""
    oc = tk.option_chain(expiry)
    parts = []
    for kind, df in (("call", oc.calls), ("put", oc.puts)):
        parts.append(pd.DataFrame({
            "type": kind, "strike": df["strike"], "bid": df["bid"].fillna(0),
            "ask": df["ask"].fillna(0), "last": df["lastPrice"].fillna(0),
            "last_trade": pd.to_datetime(df["lastTradeDate"], utc=True),
            "volume": df["volume"].fillna(0), "open_interest": df["openInterest"].fillna(0),
            "yahoo_iv": df["impliedVolatility"],
        }))
    return pd.concat(parts, ignore_index=True)


def fetch_snapshot(ticker: str, expiry: str | None = None) -> Snapshot:
    import yfinance as yf  # imported here so `analyze` works without it

    tk = yf.Ticker(ticker)
    expiries = list(tk.options)
    if not expiries:
        raise SystemExit(f"No listed options for {ticker}")
    now = datetime.now(NY)
    expiry = expiry or pick_expiry(expiries, now.date())
    if expiry not in expiries:
        raise SystemExit(f"{expiry} not listed. Available: {', '.join(expiries[:12])} ...")

    spot, r, q = fetch_inputs(tk)
    exp = date.fromisoformat(expiry)
    return Snapshot(ticker.upper(), spot, r, q, exp, years_to_expiry(exp, now), now,
                    fetch_chain(tk, expiry))


def format_table(res: pd.DataFrame) -> str:
    out = res.copy()
    pct = lambda x: "" if pd.isna(x) else f"{x:6.1%}"
    num = lambda d: (lambda x: "" if pd.isna(x) else f"{x:.{d}f}")
    fmt = {"strike": num(1), "bid": num(2), "ask": num(2), "price": num(2),
           "iv": pct, "yahoo_iv": pct, "delta": num(3), "gamma": num(4),
           "vega_1pct": num(3), "theta_day": num(3)}
    cols = ["strike", "bid", "ask", "price", "src", "iv", "yahoo_iv",
            "delta", "gamma", "vega_1pct", "theta_day", "iv_status"]
    for c in cols:
        if c not in out:
            out[c] = np.nan
    return out[cols].to_string(index=False, formatters=fmt)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ticker")
    ap.add_argument("--expiry", help="YYYY-MM-DD (default: first expiry >= 7 days out)")
    ap.add_argument("--width", type=float, default=0.15,
                    help="show strikes within this fraction of spot (default 0.15)")
    args = ap.parse_args(argv)

    snap = fetch_snapshot(args.ticker, args.expiry)
    near = snap.chain[(snap.chain.strike - snap.spot).abs() <= args.width * snap.spot]
    res = analyze(near, snap.spot, snap.T, snap.r, snap.q)

    days = snap.T * 365
    print(f"{snap.ticker}  spot {snap.spot:.2f}   expiry {snap.expiry} ({days:.1f} days)   "
          f"r {snap.r:.2%}   q {snap.q:.2%}   as of {snap.asof:%Y-%m-%d %H:%M %Z}")
    if (res.src == "last").any():
        print("Note: some prices are LAST TRADES (no live bid/ask, e.g. market closed). "
              "They may be stale relative to spot, so their IVs are approximate.")
    for kind in ("call", "put"):
        part = res[res.type == kind]
        if len(part):
            print(f"\n{kind.upper()}S")
            print(format_table(part))


if __name__ == "__main__":
    main()
