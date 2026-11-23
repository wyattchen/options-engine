# Options Pricing & Risk Engine

A real-time options pricing and risk engine: streams market data, solves implied
volatility, computes Black-Scholes Greeks across thousands of contracts in
parallel, and aggregates live portfolio risk.

## Status

- [x] **Phase 1** – Black-Scholes-Merton pricing, Greeks, implied-vol solver, correctness tests
- [ ] **Phase 2** – *(in progress: live options chain and IV surface done)* streaming feed (Redis Streams/Kafka), partitioned pricing workers, risk aggregator
- [ ] **Phase 3** – fault tolerance, benchmarks (throughput, p50/p99 latency), observability dashboard

## Phase 1

| Module | What it does |
|---|---|
| `engine/pricing.py` | Vectorized BSM price + delta, gamma, vega, theta, rho (with dividend yield) |
| `engine/implied_vol.py` | Newton-Raphson IV solver with bisection fallback and no-arbitrage bound checks |
| `tests/test_engine.py` | Known values, put-call parity, Greeks vs. finite differences, IV round-trips |

```bash
pip install numpy scipy pytest
pytest -q
```

## Live options chain

`engine/chain.py` pulls a real chain from Yahoo Finance (free, ~15 min delayed)
and fills in every model input automatically: spot, strikes, expiry (to 4pm New
York time), risk-free rate (13-week T-bill, `^IRX`) and dividend yield. It then
solves IV and Greeks for each contract.

```bash
pip install pandas yfinance
python -m engine.chain NVDA                          # first expiry >= 7 days out
python -m engine.chain NVDA --expiry 2026-12-18 --width 0.1
```

Prices use the bid/ask midpoint. When the market is closed Yahoo reports
bid = ask = 0, so the tool falls back to the last trade and says so. Those IVs
are approximate, because the last trade can be hours older than the spot price.
Yahoo's own `impliedVolatility` column is shown for comparison, but it is
unreliable outside market hours.

## Implied-volatility surface

`engine/surface.py` runs the chain pipeline across every expiry (7–365 days by
default), keeps clean quotes, and fits a smile curve per expiry.

```bash
pip install pandas yfinance matplotlib
python -m engine.surface NVDA          # charts + CSVs written to out/NVDA/
```

It prints ATM IV, skew and fit error per expiry, and saves `smile.png` (IV vs
strike for 5 expiries), `term.png` (ATM IV vs time) and `surface.png` (heatmap).

- **OTM only.** Puts below the forward, calls above. Out-of-the-money prices
  are nearly all time value, so they say the most about volatility.
- **Cleaning.** Drops quotes under 5 cents, |delta| under 0.03, unsolvable
  IVs, and last-trade prices more than 60 minutes older than the newest trade.
- **Fit.** IV = a + b·k + c·k² in log-moneyness k = ln(K/F), weighted by vega.
  `a` is ATM vol, `b` the skew, `c` the curvature. A quadratic can't follow
  the far wings when they turn up sharply; an SVI fit is the standard upgrade.

### Design notes

- **Why a bisection fallback?** Newton uses vega as its derivative. For deep
  in/out-of-the-money options vega is near zero, so Newton steps explode. The
  solver keeps a bracket while iterating and hands off to bisection if a step
  leaves it.
- **Why bound checks?** A price outside no-arbitrage bounds has no valid implied
  vol. Returning NaN (instead of a garbage number) matters once real, noisy
  market data flows through.
- **Why an "unidentifiable" result?** A far out-of-the-money option with low
  vol can be worth ~1e-20 dollars, so many volatilities reproduce that price
  within tolerance. The first test run caught the solver returning 13% vol for
  a 5% vol option in exactly this case. It now returns NaN with a reason code.
- **Known model limits.** Black-Scholes assumes constant volatility and no
  jumps. Real markets show a volatility skew, which is why Phase 2 builds an
  implied-vol surface rather than assuming one sigma.
