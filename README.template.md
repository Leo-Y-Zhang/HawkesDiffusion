# HawkesDiffusion

A Hawkes process is a point process that excites itself: every event raises the
chance of the next, and the effect decays. With two streams it also
cross-excites, so you can ask how hard a shock in one stream hits the other and
how long the reaction lasts.

```
λ_i(t) = μ_i + Σ_j Σ_{t_k^j < t} α_ij · exp(−β_ij (t − t_k^j))
```

Two quantities carry the meaning. The **half-life** ln2/β is how fast the
reaction to one event decays. The **branching ratio** — the spectral radius of
[α_ij/β_ij] — is the share of activity that is the market reacting to itself
rather than to outside information.

## What it found, and the caveat that matters more

Fitted to <<n_total>> live Binance <<symbol>> trades over <<horizon>> seconds
(<<n_buy>> buyer-initiated, <<n_sell>> seller-initiated), on <<asof>>.

The headline number would be a branching ratio of **<<branching>>**. It should
not be quoted on its own, because **it is not identified without the kernel**:

| assumed half-life | log-likelihood | branching ratio |
|---|---|---|
<<scan_rows>>

The same data yields an endogeneity anywhere from **<<br_lo>> to <<br_hi>>**
depending purely on the decay timescale you assume. Any paper reporting "the
market is X% endogenous" without stating its kernel is reporting a choice, not
a measurement.

Worse for the model: the likelihood **increases monotonically as the kernel gets
faster**, right to the edge of the grid. There is no interior optimum. That is
the known signature of a process with no single characteristic timescale, and it
is why the literature on trade flow uses power-law rather than exponential
kernels.

And the goodness-of-fit test rejects it outright. Under the time-rescaling
theorem the compensator differences should be i.i.d. Exp(1); they are not:

| stream | KS statistic | p |
|---|---|---|
| buy | <<ks_buy>> | <<p_buy>> |
| sell | <<ks_sell>> | <<p_sell>> |

So the honest conclusion is a rejection: **a single-exponential bivariate Hawkes
process does not describe this trade flow at any timescale tested**, and the
branching ratio it produces is an artefact of the assumed kernel. The model
still beats a Poisson process by a likelihood ratio of <<lr>>, so the clustering
is real — it simply is not exponential.

![Kernel norms and impulse responses](kernels.png)

## Why the fitter is trustworthy anyway

A rejection is only worth reading if the machinery works. It is checked by
simulating processes whose parameters are known exactly and requiring them back:

- **Parameter recovery** — simulate 4,000 seconds of a self-exciting process
  with μ=0.5, α=0.9, β=2.0 via Ogata thinning, fit it, and require μ, α, β and
  the branching ratio back within tolerance.
- **Analytic likelihood** — with α=0 the process is Poisson, whose
  log-likelihood is `n·log μ − μT` in closed form; the general code must match
  it exactly.
- **The residual test in both directions** — it must *fail to reject* a
  correctly specified Poisson process, and must decisively reject one fitted
  with the wrong rate. A goodness-of-fit test that never fires proves nothing.
- **The O(n) recursion** — the recursive intensity state is checked against the
  direct O(n²) double sum to 1e-10.

## Two bugs worth recording

**Tie-breaking invented the answer.** Binance timestamps are millisecond
resolution and many trades share one. A point process cannot have simultaneous
events, so the first version nudged duplicates a fixed microsecond apart. The
fitter then found enormous excitation on a microsecond timescale — it was
measuring the tie-breaking rule. The timestamps are interval-censored, so the
right treatment is uniform jitter *within* the known millisecond. That moved the
residual KS p-value from 1e-32 to 1e-3.

**Fitting all ten parameters at once is ill-conditioned.** Free β runs away
chasing sub-millisecond bursts and the cross-excitation terms collapse to
exactly zero. Holding β fixed and scanning a grid of timescales leaves six
well-behaved parameters and turns the awkward dimension into a visible model
choice, which is what produced the table above.

## Running it

Requires `numpy` and `scipy`.

```
python analyse.py --n-trades 4000            # cached tape
python analyse.py --n-trades 4000 --refresh  # pull fresh trades
python -m unittest discover -s tests -v      # <<n_tests>> tests, offline
```

## A note on the streams

The brief for this was macro news against price moves. No free feed publishes
news at the millisecond resolution the model needs, so the two streams here are
buyer-initiated and seller-initiated trades in the same instrument — the
canonical Hawkes application, and the mathematics is identical. `hawkes.py`
takes any pair of event-time arrays.

## Layout

| file | purpose |
|---|---|
| `src/hawkesdiffusion/hawkes.py` | likelihood, MLE, timescale scan, simulation, residuals |
| `src/hawkesdiffusion/binance.py` | real trade event times, tie handling, caching |
| `analyse.py` | end-to-end fit, writes `results.json` and `kernels.png` |
| `make_readme.py` | renders this file from `results.json` |

Every number above is injected from `results.json`; the generator fails if one
is missing.

## Reference

Rambaldi, Pennesi & Lillo, *Modeling FX market activity around macroeconomic
news: a Hawkes process approach*. Bacry, Mastromatteo & Muzy, *Hawkes processes
in finance*, on why power-law kernels are used for trade flow.
