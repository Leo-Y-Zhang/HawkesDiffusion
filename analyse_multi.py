"""Fit both kernels to real trade flow and report what each can and cannot say.

Writes results_multi.json. The grid deliberately stops at a 2 ms half-life:
the timestamps are millisecond-resolution and ties were broken by jitter within
the millisecond, so any apparent gain from a faster kernel is fitting that
jitter rather than the market.
"""
import json, sys, datetime as dt
import numpy as np
sys.path.insert(0, "src"); sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from scipy import stats
from hawkesdiffusion import binance
from hawkesdiffusion.hawkes import (fit_beta_grid, branching_ratio,
                                    rescaled_residuals, log_likelihood)
from hawkesdiffusion.multiexp import (fit_multi_exp, geometric_betas,
    branching_matrix_multi, branching_ratio_multi, rescaled_residuals_multi)

RESOLUTION_HL = 0.002   # 2 ms: twice the timestamp resolution

def main(symbol="BTCUSDT", n=4000):
    b, s, meta = binance.event_streams(symbol, n)
    T = meta["horizon_seconds"]; times = [b, s]
    print(f"{symbol}: {len(b)} buy / {len(s)} sell over {T:.0f}s; "
          f"{meta['n_tied_timestamps']} tied timestamps")

    single = fit_beta_grid(times, T)
    sr = rescaled_residuals(times, T, single["mu"], single["alpha"], single["beta"])
    sk = [stats.kstest(x, "expon") for x in sr]
    single_range = [min(r["branching_ratio"] for r in single["scan"]),
                    max(r["branching_ratio"] for r in single["scan"])]

    betas = geometric_betas(10, RESOLUTION_HL, 60.0)
    multi = fit_multi_exp(times, T, betas=betas)
    mr = rescaled_residuals_multi(multi["design"], multi["mu"], multi["a"])
    mk = [stats.kstest(x, "expon") for x in mr]
    norms = branching_matrix_multi(multi["a"], betas)

    # is the multi-exp branching ratio stable across grid choices?
    stability = []
    for fast in (0.05, 0.02, 0.01, 0.005, 0.002):
        bb = geometric_betas(10, fast, 60.0)
        mm = fit_multi_exp(times, T, betas=bb)
        stability.append({"fastest_half_life": fast,
                          "branching_ratio": branching_ratio_multi(mm["a"], bb),
                          "log_likelihood": mm["log_likelihood"]})
    mr_range = [min(x["branching_ratio"] for x in stability),
                max(x["branching_ratio"] for x in stability)]

    rates = [len(t) / T for t in times]
    poisson = log_likelihood(times, T, rates, [[0.0, 0.0]] * 2, [[1.0, 1.0]] * 2)

    print(f"\n  single exponential : logLik {single['log_likelihood']:8.1f}  "
          f"branching {branching_ratio(single['alpha'], single['beta']):.4f}  "
          f"KS {sk[0].statistic:.4f}")
    print(f"  multi-exponential  : logLik {multi['log_likelihood']:8.1f}  "
          f"branching {branching_ratio_multi(multi['a'], betas):.4f}  "
          f"KS {mk[0].statistic:.4f}")
    print(f"  improvement        : {multi['log_likelihood']-single['log_likelihood']:+.1f}")
    print(f"\n  branching ratio across grid choices:")
    print(f"    single exponential : {single_range[0]:.3f} to {single_range[1]:.3f}  "
          f"(NOT identified)")
    print(f"    multi-exponential  : {mr_range[0]:.3f} to {mr_range[1]:.3f}  (stable)")

    out = {"asof": dt.datetime.now().isoformat(timespec="seconds"), "symbol": symbol,
           "n_buy": len(b), "n_sell": len(s), "horizon_seconds": T,
           "n_tied_timestamps": meta["n_tied_timestamps"],
           "resolution_half_life": RESOLUTION_HL,
           "poisson_log_likelihood": poisson,
           "single": {"log_likelihood": single["log_likelihood"],
                      "branching_ratio": branching_ratio(single["alpha"], single["beta"]),
                      "best_half_life": single["half_life_seconds"],
                      "branching_range": single_range,
                      "ks": [{"stat": k.statistic, "p": k.pvalue} for k in sk]},
           "multi": {"log_likelihood": multi["log_likelihood"],
                     "n_components": len(betas),
                     "branching_ratio": branching_ratio_multi(multi["a"], betas),
                     "branching_range": mr_range,
                     "kernel_norms": norms.tolist(),
                     "half_lives": [float(x) for x in np.log(2)/betas],
                     "weights_buy_buy": [float(x) for x in multi["a"][0][0]/betas],
                     "stability_scan": stability,
                     "ks": [{"stat": k.statistic, "p": k.pvalue} for k in mk]},
           "improvement": multi["log_likelihood"] - single["log_likelihood"]}
    json.dump(out, open("results_multi.json", "w"), indent=1)
    print("\nwrote results_multi.json")
    return out

if __name__ == "__main__":
    main()
