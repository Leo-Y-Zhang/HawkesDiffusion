"""Fetch real trade event times from Binance's public aggregate-trade endpoint.

Aggregate trades carry a millisecond timestamp and a flag saying whether the
buyer was the maker, which splits the tape into buyer-initiated and
seller-initiated streams. No key is required.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np

URL = "https://api.binance.com/api/v3/aggTrades"
UA = {"User-Agent": "hawkesdiffusion/0.1 (research)"}
CACHE_DIR = os.environ.get(
    "HAWKES_CACHE",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "cache"))


def _get(url, timeout=45, retries=3):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed: {url} ({last})")


def fetch_trades(symbol="BTCUSDT", n=6000, use_cache=True):
    """Most recent ``n`` aggregate trades, paged backwards from the tape."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"trades_{symbol}_{n}.json")
    if use_cache and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    rows, from_id = [], None
    while len(rows) < n:
        params = {"symbol": symbol, "limit": min(1000, n - len(rows))}
        if from_id is not None:
            params["fromId"] = max(from_id - min(1000, n - len(rows)), 0)
        batch = _get(f"{URL}?{urllib.parse.urlencode(params)}")
        if not batch:
            break
        rows = batch + rows if from_id is not None else batch
        from_id = batch[0]["a"]
        if from_id <= 0:
            break
        time.sleep(0.25)

    rows = sorted({r["a"]: r for r in rows}.values(), key=lambda r: r["T"])
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh)
    return rows


def event_streams(symbol="BTCUSDT", n=6000, use_cache=True):
    """Split the tape into buyer-initiated and seller-initiated event times.

    Times are seconds since the first trade in the window. Binance's ``m`` flag
    is true when the *buyer* was the maker, meaning the trade was initiated by
    the seller.
    """
    rows = fetch_trades(symbol, n, use_cache)
    if not rows:
        raise RuntimeError("no trades returned")
    t0 = rows[0]["T"]
    buys, sells = [], []
    for r in rows:
        t = (r["T"] - t0) / 1000.0
        (sells if r["m"] else buys).append(t)

    # Ties matter, and how they are broken changes the answer.
    #
    # Many aggregate trades share a millisecond, and a point process cannot
    # have simultaneous events. Nudging duplicates apart by a fixed microsecond
    # was tried first and is wrong: it manufactures clusters of events a
    # microsecond apart, the fitter explains them with an enormous alpha and
    # beta, and the reported half-life collapses to zero. The structure being
    # measured is then the tie-breaking rule, not the market.
    #
    # The timestamps are interval-censored: the true time is known only to lie
    # somewhere inside its millisecond. Sampling uniformly within that
    # millisecond is the honest treatment -- it respects the known resolution
    # and adds no structure the data does not have.
    def jitter(ts, rng):
        arr = np.asarray(ts, dtype=float)
        arr = arr + rng.uniform(0.0, 0.001, size=len(arr))
        return np.sort(arr)

    rng = np.random.default_rng(0)
    b, s = jitter(buys, rng), jitter(sells, rng)
    horizon = float(max(b[-1] if len(b) else 0.0, s[-1] if len(s) else 0.0)) + 1e-3
    meta = {
        "n_raw_trades": len(rows),
        "first_timestamp_ms": rows[0]["T"],
        "last_timestamp_ms": rows[-1]["T"],
        "horizon_seconds": horizon,
        "n_tied_timestamps": int(len(rows) - len(set(r["T"] for r in rows))),
    }
    return b, s, meta
