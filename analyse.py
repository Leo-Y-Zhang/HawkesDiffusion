"""Fit a bivariate Hawkes model to real trade flow and read off the diffusion.

The two streams are buyer-initiated and seller-initiated trades in the same
instrument. That is the canonical Hawkes application: each side excites itself
(order splitting, momentum) and the other (response, liquidity replenishment),
and the branching ratio says how much of the activity is the market reacting to
itself rather than to outside information.

Macro news timestamps would be the other natural stream, but no free feed
publishes them at the millisecond resolution this needs, so the model is fitted
to a pair of streams that are actually obtainable. The mathematics is identical.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from hawkesdiffusion import binance                    # noqa: E402
from hawkesdiffusion.hawkes import (                   # noqa: E402
    branching_matrix, branching_ratio, fit_beta_grid, half_life,
    log_likelihood, rescaled_residuals)
from scipy import stats                                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--n-trades", type=int, default=6000)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    buys, sells, meta = binance.event_streams(
        args.symbol, args.n_trades, use_cache=not args.refresh)
    horizon = meta["horizon_seconds"]
    times = [buys, sells]

    print(f"{args.symbol}: {len(buys)} buyer-initiated, {len(sells)} "
          f"seller-initiated trades over {horizon:.1f}s")
    print(f"  rates: {len(buys)/horizon:.2f}/s and {len(sells)/horizon:.2f}/s")

    res = fit_beta_grid(times, horizon)
    mu, alpha, beta = res["mu"], res["alpha"], res["beta"]
    norms = branching_matrix(alpha, beta)
    n = branching_ratio(alpha, beta)
    hl = half_life(beta)

    # a Poisson model at the observed rates, as the thing to beat
    rates = [len(t) / horizon for t in times]
    poisson_ll = log_likelihood(times, horizon, rates,
                                [[0.0, 0.0], [0.0, 0.0]],
                                [[1.0, 1.0], [1.0, 1.0]])
    n_params = 2 + 4 + 4
    lr = 2.0 * (res["log_likelihood"] - poisson_ll)

    resid = rescaled_residuals(times, horizon, mu, alpha, beta)
    ks = []
    for r in resid:
        if len(r) > 20:
            k = stats.kstest(r, "expon")
            ks.append({"n": int(len(r)), "mean": float(np.mean(r)),
                       "ks_stat": float(k.statistic), "p_value": float(k.pvalue)})
        else:
            ks.append(None)

    labels = ["buy", "sell"]
    print(f"\n  branching ratio (endogeneity): {n:.4f}")
    print("  kernel norms  ||phi_ij|| = alpha/beta  (row = excited, col = exciter)")
    for i in range(2):
        print(f"    {labels[i]:>4} <- " +
              "  ".join(f"{labels[j]}:{norms[i][j]:.4f}" for j in range(2)))
    print("  half-lives (seconds)")
    for i in range(2):
        print(f"    {labels[i]:>4} <- " +
              "  ".join(f"{labels[j]}:{hl[i][j]:.3f}" for j in range(2)))
    print(f"\n  log-likelihood      {res['log_likelihood']:.1f}")
    print(f"  Poisson benchmark   {poisson_ll:.1f}")
    print(f"  likelihood ratio    {lr:.1f} on {n_params - 2} extra parameters")
    for lab, k in zip(labels, ks):
        if k:
            print(f"  residuals {lab:>4}: mean {k['mean']:.3f}  "
                  f"KS {k['ks_stat']:.4f}  p {k['p_value']:.3g}")

    out = {
        "asof": dt.datetime.now().isoformat(timespec="seconds"),
        "symbol": args.symbol,
        "meta": meta,
        "n_buy": int(len(buys)),
        "n_sell": int(len(sells)),
        "horizon_seconds": horizon,
        "mu": [float(v) for v in mu],
        "alpha": [[float(v) for v in row] for row in alpha],
        "beta": [[float(v) for v in row] for row in beta],
        "kernel_norms": [[float(v) for v in row] for row in norms],
        "half_life_seconds": [[float(v) for v in row] for row in hl],
        "branching_ratio": float(n),
        "best_half_life_seconds": float(res["half_life_seconds"]),
        "timescale_scan": res["scan"],
        "branching_ratio_range": [
            min(r["branching_ratio"] for r in res["scan"]),
            max(r["branching_ratio"] for r in res["scan"])],
        "log_likelihood": float(res["log_likelihood"]),
        "poisson_log_likelihood": float(poisson_ll),
        "likelihood_ratio": float(lr),
        "residual_tests": ks,
        "self_excitation_share": float(np.trace(norms) / np.sum(norms))
        if float(np.sum(norms)) > 0 else None,
    }
    with open("results.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("\nwrote results.json")

    try:
        plot(out)
    except Exception as exc:  # pragma: no cover
        print(f"(figure skipped: {exc})")


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "figure.dpi": 160,
                         "axes.spines.top": False, "axes.spines.right": False})
    norms = np.array(out["kernel_norms"])
    labels = ["buy", "sell"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.2),
                                  gridspec_kw={"width_ratios": [1, 1.25]})

    vmax = float(np.max(np.abs(norms))) or 1.0
    im = ax.imshow(norms, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("exciting stream")
    ax.set_ylabel("excited stream")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{norms[i][j]:.3f}", ha="center", va="center",
                    fontsize=9,
                    color="white" if abs(norms[i][j]) > 0.55 * vmax else "#15181d")
    ax.set_title(f"kernel norms, branching ratio {out['branching_ratio']:.3f}",
                 fontsize=9, loc="left")
    fig.colorbar(im, ax=ax, shrink=0.8)

    t = np.linspace(0, 2.0, 400)
    colours = {"buy": "#b3202c", "sell": "#2b5d8a"}
    for i in range(2):
        for j in range(2):
            a = out["alpha"][i][j]
            b = out["beta"][i][j]
            ax2.plot(t, a * np.exp(-b * t),
                     color=colours[labels[i]],
                     ls="-" if i == j else (0, (4, 3)), lw=1.5,
                     label=f"{labels[i]} <- {labels[j]}  "
                           f"t½={np.log(2)/b:.2f}s")
    ax2.set_xlabel("seconds since the exciting trade")
    ax2.set_ylabel("added intensity")
    ax2.set_title("impulse response of one trade", fontsize=9, loc="left")
    ax2.legend(fontsize=7, frameon=False)
    fig.suptitle(f"{out['symbol']} trade flow, {out['n_buy'] + out['n_sell']} "
                 f"trades over {out['horizon_seconds']:.0f}s", fontsize=9.5)
    fig.tight_layout()
    fig.savefig("kernels.png")
    print("wrote kernels.png")


if __name__ == "__main__":
    main()
