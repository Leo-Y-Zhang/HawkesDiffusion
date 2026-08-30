# Architecture

A Hawkes process is a point process that excites itself. The code is
organised around the two things that decide whether a fitted number may
be quoted: whether the fitter recovers known parameters, and whether the
residuals pass the time-rescaling test.

## Module map

| module | responsibility |
|---|---|
| `hawkes.py` | O(n) likelihood, MLE, timescale scan, simulation, residuals |
| `multiexp.py` | multi-exponential kernel, precomputed design, exact gradient |
| `binance.py` | real event times, tie handling, caching |
| `analysis.py` | both kernels compared; writes `results_multi.json` |
| `__main__.py` | CLI |


## Why it is shaped this way

**The recursive state is precomputed.** It depends only on the decay
rates, never on the weights, so it is computed once and the likelihood
becomes linear in the weights inside a log: concave, with an exact
gradient. That is what makes a ten-component fit take under a second.

**Parameters are kernel norms, not raw amplitudes.** Decay rates span
four orders of magnitude, so amplitudes do too, and L-BFGS-B stalls on
them -- it returned a fit *worse* than the single-exponential model it
contains as a special case, which is always a numerical bug rather than a
modelling result.

**Beta is scanned, not fitted freely.** Left free it runs away chasing
sub-millisecond bursts and the cross terms collapse to zero.

## What would break it

- The grid stops at a 2 ms half-life. Timestamps are millisecond
  resolution and ties are broken by jitter within the millisecond, so a
  faster kernel fits that jitter. Going further needs microsecond data,
  not a better kernel.
- The model is still rejected by the residual test. It is reported that
  way rather than quietly quoting its parameters.

