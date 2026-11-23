# Options Pricing & Risk Engine

A real-time options pricing and risk engine: streams market data, solves implied
volatility, computes Black-Scholes Greeks across thousands of contracts in
parallel, and aggregates live portfolio risk.

## Status

- [x] **Phase 1** – Black-Scholes-Merton pricing, Greeks, implied-vol solver, correctness tests
- [ ] **Phase 2** – streaming feed (Redis Streams/Kafka), partitioned pricing workers, risk aggregator
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
