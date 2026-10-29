# Mutation-guided unit tests

One unittest file, `tests/test_edges.py` (11 tests), and a reproducible
mutation runner, `tests/mutation_check.py`, with its results in
`tests/mutation_results.json`. Production code and CI are unchanged: CI already
runs `python -m unittest discover -s tests`.

## Measurement

`python tests/mutation_check.py before` runs the CI gate (SPDX check, ruff,
unittest) against each mutant with only the original `test_hawkes.py` and
`test_multiexp.py`; `after` runs it with every test file. Each mutant is
applied alone in a temporary copy; the original sources are never edited.

| Module | Mutants tried | Killed before | Killed after |
| --- | ---: | ---: | ---: |
| `src/hawkesdiffusion/hawkes.py` | 17 | 11 | 17 |
| `src/hawkesdiffusion/multiexp.py` | 6 | 5 | 6 |
| `src/hawkesdiffusion/binance.py` | 10 | 0 | 10 |
| **Total** | **33** | **16** | **33** |

The recovery tests fit simulated data back, which exercises the model as a
whole but not each cross term, the silent-stream boundary, or the Binance tape
at all: every `binance.py` mutant survived, including swapping the maker flag,
which would label every buy as a sell. This is a targeted sample, not an
exhaustive mutation score.

## What each test kills

| Test | Mutant killed (line) |
| --- | --- |
| `test_intensity_matches_the_direct_sum_with_asymmetric_cross_terms` | `hawkes.py:58` alpha[j][i] for alpha[i][j] |
| `test_residuals_are_differences_of_the_direct_compensator` | `hawkes.py:258` compensator missing 1/beta; `:255` cross stream read from the wrong side |
| `test_a_stream_is_silent_below_min_events_and_active_at_it` | `hawkes.py:162` boundary `<=`; `:160` caller's matrix modified in place |
| `test_multi_exp_pins_exactly_the_silent_columns` | `multiexp.py:168` boundary `<=` (bounds captured from the optimiser call) |
| `test_the_best_likelihood_wins_and_every_timescale_is_reported` | `hawkes.py:316` the grid keeps the worst fit |
| `test_the_maker_flag_splits_buyers_from_sellers` | `binance.py:81` maker flag swapped; `:80` timestamps scaled by 100; `:98` 10 ms jitter |
| `test_jitter_stays_inside_the_millisecond_and_keeps_order` | `binance.py:99` jittered times left unsorted |
| `test_horizon_and_tie_count` | `binance.py:103` horizon without its 1 ms margin; `:112` tie count |
| `test_an_empty_tape_is_an_error` | `binance.py:75` an empty tape accepted |
| `test_trades_are_deduplicated_by_id_and_sorted_by_time` | `binance.py:61` duplicates kept; trades sorted newest first |
| `test_get_retries_before_giving_up` | `binance.py:28` only one attempt |

## Commands and results

Python 3.13 on a real checkout:

| Command | Result |
| --- | --- |
| `python tests/mutation_check.py before` | 16/33 killed |
| `python tests/mutation_check.py after` | 33/33 killed |
| `python -m unittest discover -s tests` | 47 tests, OK |
| `ruff check .` | All checks passed |
| `python scripts/check_spdx.py` | all source files carry the identifier |

No test makes a network request: `urlopen`, `_get` and `fetch_trades` are
replaced with stubs and the cache goes to a temporary directory.
