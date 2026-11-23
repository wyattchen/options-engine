"""Start the whole pipeline with one command; Ctrl+C stops everything.

    python -m engine.launch NVDA                  # simulated feed, 2 workers
    python -m engine.launch NVDA --source yahoo --workers 1

Starts (in order): Redis if it isn't already running, the pricing workers,
the dashboard (http://localhost:8000), then the producer.
"""
from __future__ import annotations

import argparse
import shutil
import signal
import subprocess
import sys
import time

from . import bus


def redis_up(url: str) -> bool:
    try:
        return bus.connect(url).ping()
    except Exception:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ticker", nargs="?", default="NVDA")
    ap.add_argument("--source", choices=["sim", "yahoo"], default="sim")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--fresh", action="store_true",
                    help="delete old messages in the quotes/analytics streams first")
    args, producer_args = ap.parse_known_args(argv)   # extra flags go to the producer
    py = sys.executable
    procs: list[subprocess.Popen] = []

    if not redis_up(bus.DEFAULT_URL):
        if not shutil.which("redis-server"):
            sys.exit("Redis isn't running and redis-server isn't installed (brew install redis).")
        print("Starting Redis (in-memory only, nothing saved to disk)")
        procs.append(subprocess.Popen(["redis-server", "--save", "", "--appendonly", "no"],
                                      stdout=subprocess.DEVNULL))
        for _ in range(50):
            if redis_up(bus.DEFAULT_URL):
                break
            time.sleep(0.1)
        else:
            sys.exit("Redis did not start")

    if args.fresh:
        bus.connect().delete(bus.QUOTES, bus.ANALYTICS)

    for i in range(args.workers):
        procs.append(subprocess.Popen([py, "-m", "engine.worker", "--name", f"worker-{i + 1}"]))
    procs.append(subprocess.Popen([py, "-m", "engine.dashboard", "--port", str(args.port)]))
    procs.append(subprocess.Popen([py, "-m", "engine.producer", args.ticker,
                                   "--source", args.source, *producer_args]))
    print(f"\n  Dashboard: http://localhost:{args.port}   (Ctrl+C to stop)\n", flush=True)

    def shutdown(*_):
        # One at a time, in reverse start order: producer, dashboard, workers,
        # and Redis last, so nothing loses its connection while still running.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        for p in reversed(procs):
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
        print("Stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    while True:
        for p in procs:
            if p.poll() is not None and p.returncode not in (0, -signal.SIGTERM):
                print(f"A process exited with code {p.returncode}: {p.args}. Stopping.")
                shutdown()
        time.sleep(0.5)


if __name__ == "__main__":
    main()
