# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""HawkesDiffusion CLI.

  hawkesdiffusion fit [--symbol BTCUSDT] [--n-trades 4000] [--refresh]
      Fit both kernels to real trade flow and report the comparison: the
      single exponential, the multi-exponential that approximates a power law,
      and how far the branching ratio moves when the assumed timescale changes.
      Writes results_multi.json.

  hawkesdiffusion recover [--seed N] [--horizon T]
      Simulate a process whose parameters are known exactly, fit it, and print
      what came back. Fully offline. This is the check that decides whether any
      fitted number in this repository can be believed, so it is a first-class
      command rather than a hidden test.

  hawkesdiffusion residuals [--symbol BTCUSDT]
      Time-rescaling goodness of fit. Under the model the compensator
      differences are i.i.d. Exp(1); this prints how far they are from that.
      A model that fails here should not have its parameters quoted.

  hawkesdiffusion verify
      Run the offline suite, including the residual test in both directions:
      it must fail to reject a correct model and must reject a wrong one.

Reads public, unauthenticated market data. Holds no credentials, connects to no
account, places no orders.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def cmd_fit(args) -> int:
    from .analysis import run
    run(symbol=args.symbol, n_trades=args.n_trades, refresh=args.refresh)
    return 0


def cmd_recover(args) -> int:
    from .hawkes import branching_ratio, fit_beta_grid, simulate
    mu = [0.5, 1e-8]
    alpha = [[0.9, 0.0], [0.0, 0.0]]
    beta = [[2.0, 5.0], [5.0, 5.0]]
    truth_n = branching_ratio(alpha, beta)

    print(f"simulating {args.horizon:.0f}s with mu={mu[0]}, alpha={alpha[0][0]}, "
          f"beta={beta[0][0]}  (branching ratio {truth_n:.4f})")
    times = simulate(mu, alpha, beta, horizon=args.horizon, seed=args.seed)
    print(f"  {len(times[0])} events generated")

    got = fit_beta_grid(times, args.horizon)
    n = branching_ratio(got["alpha"], got["beta"])
    print("\n  parameter      truth     fitted")
    print(f"  mu           {mu[0]:8.4f} {got['mu'][0]:10.4f}")
    print(f"  branching    {truth_n:8.4f} {n:10.4f}")
    err = abs(n - truth_n)
    print(f"\n  branching-ratio error {err:.4f} "
          f"({'within tolerance' if err < 0.12 else 'OUT OF TOLERANCE'})")
    return 0 if err < 0.12 else 1


def cmd_residuals(args) -> int:
    from scipy import stats

    from . import binance
    from .multiexp import fit_multi_exp, geometric_betas, rescaled_residuals_multi
    b, s, meta = binance.event_streams(args.symbol, args.n_trades,
                                       use_cache=not args.refresh)
    horizon = meta["horizon_seconds"]
    betas = geometric_betas(10, 0.002, 60.0)
    fit = fit_multi_exp([b, s], horizon, betas=betas)
    resid = rescaled_residuals_multi(fit["design"], fit["mu"], fit["a"])
    print(f"{args.symbol}: time-rescaling residuals should be i.i.d. Exp(1)")
    print(f"{'stream':>8} {'n':>7} {'mean':>8} {'KS':>8} {'p':>12}  verdict")
    rc = 0
    for name, r in zip(("buy", "sell"), resid):
        if len(r) < 20:
            continue
        k = stats.kstest(r, "expon")
        verdict = "not rejected" if k.pvalue > 0.01 else "REJECTED"
        if k.pvalue <= 0.01:
            rc = 1
        print(f"{name:>8} {len(r):>7} {float(np.mean(r)):>8.3f} "
              f"{k.statistic:>8.4f} {k.pvalue:>12.2e}  {verdict}")
    print("\nA rejected model may still be the best available, but its "
          "parameters should be quoted with that stated.")
    return rc


def cmd_verify(args) -> int:
    import unittest
    suite = unittest.TestLoader().discover("tests")
    ok = unittest.TextTestRunner(verbosity=2 if args.verbose else 1).run(suite)
    return 0 if ok.wasSuccessful() else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hawkesdiffusion", description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = p.add_subparsers(dest="command")

    f = sub.add_parser("fit", help="fit both kernels to real trade flow")
    f.add_argument("--symbol", default="BTCUSDT")
    f.add_argument("--n-trades", type=int, default=4000)
    f.add_argument("--refresh", action="store_true")
    f.set_defaults(func=cmd_fit)

    r = sub.add_parser("recover", help="simulate with known parameters and fit them back")
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--horizon", type=float, default=4000.0)
    r.set_defaults(func=cmd_recover)

    d = sub.add_parser("residuals", help="time-rescaling goodness of fit")
    d.add_argument("--symbol", default="BTCUSDT")
    d.add_argument("--n-trades", type=int, default=4000)
    d.add_argument("--refresh", action="store_true")
    d.set_defaults(func=cmd_residuals)

    v = sub.add_parser("verify", help="run the offline suite")
    v.add_argument("--verbose", action="store_true")
    v.set_defaults(func=cmd_verify)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
