"""Implied-volatility surface: IV across every strike and expiry.

    python -m engine.surface NVDA                  # expiries 7-365 days out
    python -m engine.surface NVDA --max-days 120 --out out

Black-Scholes assumes one volatility per stock. Real option prices imply a
different volatility at each strike (the *skew* or *smile*) and each expiry
(the *term structure*). This module measures that pattern and fits a smooth
curve per expiry, so you can read off a sensible IV for any strike.

Pipeline:
  1. fetch every expiry's chain and solve IV + Greeks (reuses engine.chain)
  2. keep only out-of-the-money, liquid, non-stale quotes
  3. fit IV as a quadratic in log-moneyness per expiry, weighted by vega
  4. summarise (ATM IV, skew) and draw charts
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .chain import NY, analyze, fetch_chain, fetch_inputs, years_to_expiry

MIN_POINTS = 5  # fewer quotes than this and a 3-parameter fit is meaningless


# ---------------------------------------------------------------- the maths

def forward(S: float, T: float, r: float, q: float) -> float:
    """Expected stock price at expiry under the model: spot grown at r - q."""
    return S * math.exp((r - q) * T)


def select_otm(res: pd.DataFrame, F: float) -> pd.DataFrame:
    """Puts below the forward, calls at or above it.

    Out-of-the-money options are almost pure time value, so their price is
    almost entirely a statement about volatility. In-the-money options are
    mostly intrinsic value, trade less, and give noisier IVs. By put-call
    parity both sides imply the same vol in theory, so we lose nothing.
    """
    otm = ((res.type == "put") & (res.strike < F)) | ((res.type == "call") & (res.strike >= F))
    return res[otm]


def clean(res: pd.DataFrame, latest_trade=None, max_stale_min: float = 60,
          min_price: float = 0.05, min_abs_delta: float = 0.03) -> pd.DataFrame:
    """Drop quotes that would distort the surface.

    - no solvable IV (NaN)
    - price under 5 cents: tick size dominates, IV is mostly rounding noise
    - |delta| under 0.03: far wings, illiquid, wildly unstable IV
    - last-trade prices more than `max_stale_min` minutes older than the most
      recent trade: the stock has moved since, so the IV is wrong
    """
    keep = res.iv.notna() & (res.price >= min_price) & (res.delta.abs() >= min_abs_delta)
    if latest_trade is not None and "last_trade" in res:
        stale = (res.src == "last") & (
            res.last_trade < latest_trade - pd.Timedelta(minutes=max_stale_min))
        keep &= ~stale
    return res[keep]


def fit_smile(k, iv, weights=None) -> np.ndarray:
    """Weighted least squares fit of IV = a + b*k + c*k^2.

    k is log-moneyness ln(K/F): 0 at the money, negative for low strikes.
    a = ATM vol, b = skew (slope; negative means low strikes are pricier),
    c = curvature (the "smile"). Vega weights trust near-the-money quotes more,
    since a small price error there moves IV least.
    Returns [a, b, c].
    """
    w = None if weights is None else np.sqrt(np.asarray(weights, float))
    c, b, a = np.polyfit(np.asarray(k, float), np.asarray(iv, float), 2, w=w)
    return np.array([a, b, c])


def smile_iv(coef: np.ndarray, k) -> np.ndarray:
    a, b, c = coef
    k = np.asarray(k, float)
    return a + b * k + c * k**2


@dataclass
class Surface:
    ticker: str
    spot: float
    r: float
    q: float
    asof: datetime
    points: pd.DataFrame   # every cleaned quote used in a fit
    fits: pd.DataFrame     # one row per expiry: coefficients and summary stats


def build_surface(chains: dict[str, tuple[float, pd.DataFrame]], S: float, r: float,
                  q: float, max_stale_min: float = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    """chains maps expiry -> (T, raw chain). Returns (points, fits). No network."""
    analysed = {e: (T, analyze(raw, S, T, r, q)) for e, (T, raw) in chains.items()}
    # Most recent trade anywhere on the surface ~ "now" for the data we have.
    trades = [raw["last_trade"].max() for _, raw in chains.values() if "last_trade" in raw]
    latest = max(trades) if trades else None

    points, fits = [], []
    for expiry, (T, res) in sorted(analysed.items()):
        F = forward(S, T, r, q)
        pts = clean(select_otm(res, F), latest, max_stale_min).copy()
        pts["expiry"], pts["T"], pts["days"] = expiry, T, T * 365
        pts["k"] = np.log(pts.strike / F)
        pts["moneyness"] = pts.strike / S
        row = {"expiry": expiry, "T": T, "days": T * 365, "forward": F, "n": len(pts)}
        if len(pts) >= MIN_POINTS:
            coef = fit_smile(pts.k, pts.iv, pts.vega_1pct)
            atm = coef[0]
            sd = atm * math.sqrt(T)  # one standard deviation in log-price terms
            resid = pts.iv - smile_iv(coef, pts.k)
            row.update(a=coef[0], b=coef[1], c=coef[2], atm_iv=atm,
                       skew_1sd=float(smile_iv(coef, -sd) - smile_iv(coef, sd)),
                       rmse=float(np.sqrt(np.mean(resid**2))),
                       k_min=float(pts.k.min()), k_max=float(pts.k.max()))
            points.append(pts)
        fits.append(row)
    points = pd.concat(points, ignore_index=True) if points else pd.DataFrame()
    return points, pd.DataFrame(fits)


# ---------------------------------------------------------------- data

def fetch_surface(ticker: str, min_days: int = 7, max_days: int = 365,
                  max_stale_min: float = 60) -> Surface:
    import yfinance as yf

    tk = yf.Ticker(ticker)
    now = datetime.now(NY)
    spot, r, q = fetch_inputs(tk)
    chains = {}
    for e in tk.options:
        days = (date.fromisoformat(e) - now.date()).days
        if min_days <= days <= max_days:
            chains[e] = (years_to_expiry(date.fromisoformat(e), now), fetch_chain(tk, e))
    if not chains:
        raise SystemExit(f"No {ticker} expiries between {min_days} and {max_days} days")
    points, fits = build_surface(chains, spot, r, q, max_stale_min)
    return Surface(ticker.upper(), spot, r, q, now, points, fits)


# ---------------------------------------------------------------- charts
# Static PNGs (matplotlib). Colours: fixed categorical order, blue sequential
# ramp for magnitude, recessive grey chrome.

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#6250d6", "#e34948"]
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e",
                                         "#898781", "#e1e0d9", "#c3c2b7")


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def _pct(ax, axis="y"):
    from matplotlib.ticker import PercentFormatter
    getattr(ax, f"{axis}axis").set_major_formatter(PercentFormatter(1.0, decimals=0))


def _pick_expiries(fits: pd.DataFrame, n: int = 5) -> list[str]:
    """Spread n expiries across the available range (short, medium, long)."""
    ok = fits.dropna(subset=["atm_iv"]).reset_index(drop=True)
    if len(ok) <= n:
        return list(ok.expiry)
    idx = np.unique(np.linspace(0, len(ok) - 1, n).round().astype(int))
    return list(ok.expiry.iloc[idx])


def plot_smiles(s: Surface, path: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=SURFACE)
    _style(ax)
    for i, e in enumerate(_pick_expiries(s.fits)):
        f = s.fits.set_index("expiry").loc[e]
        pts = s.points[s.points.expiry == e]
        col = SERIES[i]
        label = f"{e} ({f.days:.0f}d)"
        ax.scatter(pts.moneyness, pts.iv, s=30, color=col, edgecolor=SURFACE,
                   linewidth=1, zorder=3, label=label)
        k = np.linspace(f.k_min, f.k_max, 200)
        m = np.exp(k) * f.forward / s.spot
        iv = smile_iv(np.array([f.a, f.b, f.c]), k)
        ax.plot(m, iv, color=col, linewidth=2, zorder=2)
        ax.annotate(f"{f.days:.0f}d", (m[-1], iv[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK2, fontsize=9)
    ax.axvline(1.0, color=MUTED, linewidth=1, linestyle="--", zorder=1)
    ax.annotate("spot", (1.0, 1), xycoords=("data", "axes fraction"), xytext=(4, -12),
                textcoords="offset points", color=MUTED, fontsize=9)
    ax.set_xlabel("Strike / spot")
    ax.set_ylabel("Implied volatility")
    _pct(ax, "x")
    _pct(ax, "y")
    ax.set_title(f"{s.ticker} volatility smile by expiry  (dots = market, lines = fit)",
                 loc="left", fontsize=12)
    leg = ax.legend(frameon=False, loc="upper right", fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_term(s: Surface, path: Path):
    import matplotlib.pyplot as plt

    f = s.fits.dropna(subset=["atm_iv"])
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=SURFACE)
    _style(ax)
    ax.plot(f.days, f.atm_iv, color=SERIES[0], linewidth=2, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    hi, lo = f.loc[f.atm_iv.idxmax()], f.loc[f.atm_iv.idxmin()]
    for row in {hi.expiry: hi, lo.expiry: lo}.values():
        ax.annotate(f"{row.atm_iv:.0%}  {row.expiry}", (row.days, row.atm_iv),
                    xytext=(8, 6), textcoords="offset points", color=INK2, fontsize=9)
    ax.set_xlabel("Days to expiry")
    ax.set_ylabel("At-the-money implied volatility")
    _pct(ax)
    ax.set_title(f"{s.ticker} ATM implied volatility term structure", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_surface(s: Surface, path: Path):
    """Heatmap of the fitted surface: rows = expiries, columns = strike/spot.
    Cells outside each expiry's quoted strike range are left blank, not
    extrapolated."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    f = s.fits.dropna(subset=["atm_iv"]).reset_index(drop=True)
    grid = np.linspace(0.6, 1.4, 161)
    Z = np.full((len(f), len(grid)), np.nan)
    for i, row in f.iterrows():
        k = np.log(grid * s.spot / row.forward)
        inside = (k >= row.k_min) & (k <= row.k_max)
        Z[i, inside] = smile_iv(np.array([row.a, row.b, row.c]), k[inside])

    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    cmap.set_bad(SURFACE)
    fig, ax = plt.subplots(figsize=(10, 0.32 * len(f) + 2.2), facecolor=SURFACE)
    _style(ax)
    ax.grid(False)
    im = ax.imshow(np.ma.masked_invalid(Z), aspect="auto", cmap=cmap, origin="lower",
                   extent=(grid[0], grid[-1], -0.5, len(f) - 0.5), interpolation="nearest")
    ax.set_yticks(range(len(f)))
    ax.set_yticklabels([f"{e}  ({d:.0f}d)" for e, d in zip(f.expiry, f.days)], fontsize=8)
    ax.axvline(1.0, color=MUTED, linewidth=1, linestyle="--")
    ax.set_xlabel("Strike / spot")
    _pct(ax, "x")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("Implied volatility", color=INK2)
    cb.ax.yaxis.set_major_formatter(__import__("matplotlib.ticker", fromlist=["x"]).PercentFormatter(1.0, 0))
    cb.ax.tick_params(colors=MUTED, labelcolor=INK2)
    cb.outline.set_visible(False)
    ax.set_title(f"{s.ticker} implied volatility surface (fitted)", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ticker")
    ap.add_argument("--min-days", type=int, default=7)
    ap.add_argument("--max-days", type=int, default=365)
    ap.add_argument("--max-stale", type=float, default=60,
                    help="drop last-trade prices older than this many minutes (default 60)")
    ap.add_argument("--out", default="out", help="folder for charts and CSVs (default out/)")
    args = ap.parse_args(argv)

    s = fetch_surface(args.ticker, args.min_days, args.max_days, args.max_stale)
    out = Path(args.out) / s.ticker
    out.mkdir(parents=True, exist_ok=True)

    print(f"{s.ticker}  spot {s.spot:.2f}   r {s.r:.2%}   q {s.q:.2%}   "
          f"as of {s.asof:%Y-%m-%d %H:%M %Z}")
    if len(s.points) and (s.points.src == "last").any():
        print("Note: using LAST TRADES (no live bid/ask, market likely closed). "
              "Stale trades are filtered out, but treat the surface as approximate.")
    print()
    table = s.fits.copy()
    fmt = lambda x: "" if pd.isna(x) else f"{x:.1%}"
    print(table[["expiry", "days", "n", "atm_iv", "skew_1sd", "rmse"]].to_string(
        index=False, formatters={"days": "{:.0f}".format, "atm_iv": fmt,
                                 "skew_1sd": fmt, "rmse": fmt}))
    print("\natm_iv   = fitted IV at the forward price (at the money)")
    print("skew_1sd = IV one standard deviation below the forward minus one above;"
          " positive = downside protection costs more")
    print("rmse     = typical gap between market IVs and the fitted curve")

    if s.points.empty:
        print("\nNot enough clean quotes to fit any expiry.")
        return
    s.points.to_csv(out / "points.csv", index=False)
    s.fits.to_csv(out / "fits.csv", index=False)
    plot_smiles(s, out / "smile.png")
    plot_term(s, out / "term.png")
    plot_surface(s, out / "surface.png")
    print(f"\nWrote {out}/smile.png, term.png, surface.png, points.csv, fits.csv")


if __name__ == "__main__":
    main()
