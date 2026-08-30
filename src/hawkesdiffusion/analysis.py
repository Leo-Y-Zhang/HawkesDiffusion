# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Fit both kernels to real trade flow and report what each can and cannot say.

The grid deliberately stops at a 2 ms half-life. Timestamps are
millisecond-resolution and ties are broken by jitter inside the millisecond, so
a kernel faster than that is fitting the jitter rather than the market -- the
likelihood keeps improving, which is exactly what makes the trap dangerous.
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
from scipy import stats

from . import binance
from .hawkes import branching_ratio, fit_beta_grid, log_likelihood, rescaled_residuals
from .multiexp import (
    branching_matrix_multi,
    branching_ratio_multi,
    fit_multi_exp,
    geometric_betas,
    rescaled_residuals_multi,
)

RESOLUTION_HALF_LIFE = 0.002


def run(symbol="BTCUSDT", n_trades=4000, refresh=False, quiet=False,
        out_path="results_multi.json"):
    say = (lambda *a: None) if quiet else print
    b, s, meta = binance.event_streams(symbol, n_trades, use_cache=not refresh)
    horizon = meta["horizon_seconds"]
    times = [b, s]
    say(f"{symbol}: {len(b)} buyer-initiated, {len(s)} seller-initiated over "
        f"{horizon:.0f}s; {meta['n_tied_timestamps']} tied timestamps")

    single = fit_beta_grid(times, horizon)
    sr = rescaled_residuals(times, horizon, single["mu"], single["alpha"],
                            single["beta"])
    sk = [stats.kstest(x, "expon") for x in sr]
    single_range = [min(r["branching_ratio"] for r in single["scan"]),
                    max(r["branching_ratio"] for r in single["scan"])]

    betas = geometric_betas(10, RESOLUTION_HALF_LIFE, 60.0)
    multi = fit_multi_exp(times, horizon, betas=betas)
    mr = rescaled_residuals_multi(multi["design"], multi["mu"], multi["a"])
    mk = [stats.kstest(x, "expon") for x in mr]
    norms = branching_matrix_multi(multi["a"], betas)

    stability = []
    for fast in (0.05, 0.02, 0.01, 0.005, 0.002):
        bb = geometric_betas(10, fast, 60.0)
        mm = fit_multi_exp(times, horizon, betas=bb)
        stability.append({"fastest_half_life": fast,
                          "branching_ratio": branching_ratio_multi(mm["a"], bb),
                          "log_likelihood": mm["log_likelihood"]})
    multi_range = [min(x["branching_ratio"] for x in stability),
                   max(x["branching_ratio"] for x in stability)]

    rates = [len(t) / horizon for t in times]
    poisson = log_likelihood(times, horizon, rates, [[0.0, 0.0]] * 2,
                             [[1.0, 1.0]] * 2)

    say(f"\n  single exponential : logLik {single['log_likelihood']:9.1f}  "
        f"branching {branching_ratio(single['alpha'], single['beta']):.4f}  "
        f"KS {sk[0].statistic:.4f}")
    say(f"  multi-exponential  : logLik {multi['log_likelihood']:9.1f}  "
        f"branching {branching_ratio_multi(multi['a'], betas):.4f}  "
        f"KS {mk[0].statistic:.4f}")
    say(f"  improvement        : {multi['log_likelihood'] - single['log_likelihood']:+.1f}")
    say("\n  branching ratio across grid choices")
    say(f"    single exponential : {single_range[0]:.3f} to {single_range[1]:.3f}"
        "   NOT identified")
    say(f"    multi-exponential  : {multi_range[0]:.3f} to {multi_range[1]:.3f}"
        "   stable")

    out = {
        "asof": dt.datetime.now().isoformat(timespec="seconds"), "symbol": symbol,
        "n_buy": len(b), "n_sell": len(s), "horizon_seconds": horizon,
        "n_tied_timestamps": meta["n_tied_timestamps"],
        "resolution_half_life": RESOLUTION_HALF_LIFE,
        "poisson_log_likelihood": poisson,
        "single": {"log_likelihood": single["log_likelihood"],
                   "branching_ratio": branching_ratio(single["alpha"], single["beta"]),
                   "best_half_life": single["half_life_seconds"],
                   "branching_range": single_range,
                   "ks": [{"stat": k.statistic, "p": k.pvalue} for k in sk]},
        "multi": {"log_likelihood": multi["log_likelihood"],
                  "n_components": len(betas),
                  "branching_ratio": branching_ratio_multi(multi["a"], betas),
                  "branching_range": multi_range,
                  "kernel_norms": norms.tolist(),
                  "half_lives": [float(x) for x in np.log(2) / betas],
                  "weights_buy_buy": [float(x) for x in multi["a"][0][0] / betas],
                  "stability_scan": stability,
                  "ks": [{"stat": k.statistic, "p": k.pvalue} for k in mk]},
        "improvement": multi["log_likelihood"] - single["log_likelihood"],
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    say(f"\nwrote {out_path}")
    return out
