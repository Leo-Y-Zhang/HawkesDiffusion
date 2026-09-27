# Changelog

Notable changes to this project, in the format of
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

The project has never been tagged for release, so there are no version headings
yet, only *Unreleased*. Back-filling release notes for work that shipped without
them would be writing history after the fact, which is the thing this
repository's documents are meant not to do.

## [Unreleased]

### Added

- **Multi-exponential kernel** approximating a power law while keeping the O(n)
  recursion. On live trade flow, at the same 10 ms fastest half-life as the
  single exponential, it gained +13.7 log-likelihood and, more importantly,
  made the branching ratio identified:
  0.572-0.605 across a 25x range of grids, against 0.531-0.928 before.
- **CLI** (`hawkesdiffusion fit|recover|residuals|verify`), with the
  simulate-and-recover check promoted to a first-class command.

### Fixed

- **Headline likelihood gain was not like for like.** The README quoted
  +1,533.5 log-likelihood for the multi-exponential kernel, but that fit had a
  2 ms component while the single exponential's fastest half-life was 10 ms.
  At a matched 10 ms the gain is +13.7; the other +1,519.8 comes from the
  faster components. The README now leads with the matched figure, and
  `results_multi.json` records it as `improvement_matched`.
- **Tie count wording.** The README said 1,367 trades "share a millisecond";
  the number counts trades that share a millisecond with an *earlier* trade
  (each tied group's first trade is not counted).
- **Silent streams in the multi-exponential fit.** The same unidentified-column
  problem existed in `fit_multi_exp`: for an empty stream the weights it would
  excite with have zero gradient, so L-BFGS-B returned them at their starting
  value of 0.02 each, a phantom kernel norm of 0.2 per entry with ten
  components. A self-exciting stream with a true branching ratio of 0.15 was
  reported as 0.20. Those weights are now pinned at zero. The real-data fit,
  where both streams are busy, takes exactly the same path as before.
- **Likelihood reported for a nearly silent stream.** `fit` and
  `fit_fixed_beta` zeroed the columns of streams with fewer than five events
  only after the optimiser finished. With one to four events those parameters
  do enter the likelihood, so the reported log-likelihood belonged to
  parameters that were not returned (10 units too high in the regression test),
  and the timescale scan compared such numbers. The columns are now zeroed
  inside the objective too.
- **Unidentified kernel columns.** `alpha[i][j]` multiplies a sum over the
  events of stream `j`, so a silent stream leaves that parameter absent from
  the likelihood entirely; the optimiser left it wherever it started and the
  spectral radius reported the noise. Observed as a branching ratio of 4e8 on a
  simulated pair whose second stream was deliberately empty. Columns of silent
  streams are now zeroed, which is the correct normalisation rather than a
  patch. Results on real data, where both streams are busy, are unchanged.
- **Tie-breaking.** Nudging tied millisecond timestamps a fixed microsecond
  apart made the fitter measure the tie-breaking rule rather than the market.
  Timestamps are interval-censored, so ties are now broken by uniform jitter
  within the known millisecond.

