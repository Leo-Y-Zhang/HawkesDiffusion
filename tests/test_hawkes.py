# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Tests for the Hawkes fitter.

The one that matters is parameter recovery: simulate a process whose parameters
are known exactly, fit it, and require the fit to find them back. A fitter that
cannot recover ground truth tells you nothing about a real trade stream.
"""
from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from hawkesdiffusion.hawkes import (  # noqa: E402
    _recursive_state,
    branching_matrix,
    branching_ratio,
    fit,
    fit_fixed_beta,
    half_life,
    intensity_at_events,
    log_likelihood,
    rescaled_residuals,
    simulate,
)


class TestRecursiveState(unittest.TestCase):
    def test_matches_the_direct_double_sum(self):
        rng = np.random.default_rng(0)
        target = np.sort(rng.uniform(0, 10, 40))
        source = np.sort(rng.uniform(0, 10, 60))
        beta = 1.7
        fast = _recursive_state(target, source, beta)
        slow = np.array([np.sum(np.exp(-beta * (t - source[source < t])))
                         for t in target])
        np.testing.assert_allclose(fast, slow, rtol=1e-10, atol=1e-12)

    def test_empty_source_gives_zeros(self):
        out = _recursive_state(np.array([1.0, 2.0]), np.array([]), 1.0)
        np.testing.assert_array_equal(out, np.zeros(2))

    def test_only_strictly_earlier_events_count(self):
        out = _recursive_state(np.array([1.0]), np.array([1.0, 2.0]), 1.0)
        self.assertAlmostEqual(float(out[0]), 0.0)


class TestBranchingAndHalfLife(unittest.TestCase):
    def test_branching_matrix_is_alpha_over_beta(self):
        m = branching_matrix([[1.0, 2.0], [3.0, 4.0]], [[2.0, 4.0], [6.0, 8.0]])
        np.testing.assert_allclose(m, [[0.5, 0.5], [0.5, 0.5]])

    def test_branching_ratio_of_a_diagonal_system(self):
        r = branching_ratio([[0.6, 0.0], [0.0, 0.3]], [[1.0, 1.0], [1.0, 1.0]])
        self.assertAlmostEqual(r, 0.6)

    def test_cross_excitation_raises_the_branching_ratio(self):
        weak = branching_ratio([[0.3, 0.0], [0.0, 0.3]], [[1.0, 1.0], [1.0, 1.0]])
        strong = branching_ratio([[0.3, 0.4], [0.4, 0.3]], [[1.0, 1.0], [1.0, 1.0]])
        self.assertGreater(strong, weak)

    def test_zero_beta_does_not_divide_by_zero(self):
        m = branching_matrix([[1.0]], [[0.0]])
        self.assertEqual(float(m[0][0]), 0.0)

    def test_half_life_is_ln2_over_beta(self):
        self.assertAlmostEqual(float(half_life(2.0)), math.log(2) / 2.0)

    def test_faster_decay_means_shorter_half_life(self):
        self.assertLess(float(half_life(10.0)), float(half_life(1.0)))


class TestLogLikelihood(unittest.TestCase):
    def test_poisson_case_matches_the_closed_form(self):
        """With alpha = 0 the process is Poisson, whose log-likelihood is
        n*log(mu) - mu*T per stream."""
        times = [np.array([1.0, 2.5, 4.0]), np.array([0.5, 3.0])]
        mu = [0.3, 0.2]
        zero = [[0.0, 0.0], [0.0, 0.0]]
        beta = [[1.0, 1.0], [1.0, 1.0]]
        got = log_likelihood(times, 5.0, mu, zero, beta)
        want = (3 * math.log(0.3) - 0.3 * 5.0) + (2 * math.log(0.2) - 0.2 * 5.0)
        self.assertAlmostEqual(got, want, places=9)

    def test_excitation_raises_the_likelihood_of_clustered_data(self):
        """Tightly clustered events should be better explained by a self-
        exciting model than by a Poisson one at the same base rate."""
        clustered = [np.array([1.0, 1.02, 1.05, 1.08, 5.0, 5.03, 5.06]),
                     np.array([])]
        beta = [[10.0, 10.0], [10.0, 10.0]]
        poisson = log_likelihood(clustered, 10.0, [0.7, 1e-9],
                                 [[0.0, 0.0], [0.0, 0.0]], beta)
        exciting = log_likelihood(clustered, 10.0, [0.2, 1e-9],
                                  [[6.0, 0.0], [0.0, 0.0]], beta)
        self.assertGreater(exciting, poisson)

    def test_negative_intensity_is_rejected(self):
        times = [np.array([1.0]), np.array([])]
        got = log_likelihood(times, 2.0, [-1.0, 0.1],
                             [[0.0, 0.0], [0.0, 0.0]],
                             [[1.0, 1.0], [1.0, 1.0]])
        self.assertEqual(got, -np.inf)


class TestIntensity(unittest.TestCase):
    def test_intensity_starts_at_the_baseline(self):
        times = [np.array([2.0]), np.array([])]
        lam = intensity_at_events(times, [0.4, 0.1],
                                  [[0.5, 0.0], [0.0, 0.0]],
                                  [[1.0, 1.0], [1.0, 1.0]])
        self.assertAlmostEqual(float(lam[0][0]), 0.4)

    def test_a_recent_event_raises_the_intensity(self):
        times = [np.array([1.0, 1.1]), np.array([])]
        lam = intensity_at_events(times, [0.4, 0.1],
                                  [[0.5, 0.0], [0.0, 0.0]],
                                  [[1.0, 1.0], [1.0, 1.0]])
        self.assertGreater(float(lam[0][1]), float(lam[0][0]))
        self.assertAlmostEqual(float(lam[0][1]),
                               0.4 + 0.5 * math.exp(-1.0 * 0.1), places=9)


class TestSimulation(unittest.TestCase):
    def test_poisson_simulation_has_about_the_right_count(self):
        out = simulate([0.5, 0.5], [[0.0, 0.0], [0.0, 0.0]],
                       [[1.0, 1.0], [1.0, 1.0]], horizon=2000.0, seed=1)
        for stream in out:
            self.assertAlmostEqual(len(stream) / 2000.0, 0.5, delta=0.06)

    def test_self_excitation_produces_more_events_than_the_baseline(self):
        base = simulate([0.3, 0.0], [[0.0, 0.0], [0.0, 0.0]],
                        [[2.0, 2.0], [2.0, 2.0]], horizon=1000.0, seed=2)
        exc = simulate([0.3, 0.0], [[1.0, 0.0], [0.0, 0.0]],
                       [[2.0, 2.0], [2.0, 2.0]], horizon=1000.0, seed=2)
        self.assertGreater(len(exc[0]), len(base[0]))

    def test_events_are_sorted_and_inside_the_horizon(self):
        out = simulate([0.4, 0.4], [[0.3, 0.1], [0.1, 0.3]],
                       [[2.0, 2.0], [2.0, 2.0]], horizon=300.0, seed=3)
        for stream in out:
            self.assertTrue(np.all(np.diff(stream) > 0))
            self.assertTrue(np.all(stream < 300.0))


class TestParameterRecovery(unittest.TestCase):
    """Simulate with known parameters, fit, and require the truth back."""

    def test_recovers_a_univariate_self_exciting_process(self):
        mu = [0.5, 1e-8]
        alpha = [[0.9, 0.0], [0.0, 0.0]]
        beta = [[2.0, 5.0], [5.0, 5.0]]
        times = simulate(mu, alpha, beta, horizon=4000.0, seed=7)
        got = fit(times, 4000.0)

        self.assertAlmostEqual(got["mu"][0], 0.5, delta=0.12)
        self.assertAlmostEqual(got["alpha"][0][0], 0.9, delta=0.35)
        self.assertAlmostEqual(got["beta"][0][0], 2.0, delta=0.8)
        # the branching ratio is the quantity that carries the interpretation
        self.assertAlmostEqual(
            branching_ratio(got["alpha"], got["beta"]), 0.45, delta=0.12)

    def test_fit_beats_a_poisson_model_on_excited_data(self):
        mu = [0.4, 1e-8]
        alpha = [[1.2, 0.0], [0.0, 0.0]]
        beta = [[3.0, 5.0], [5.0, 5.0]]
        times = simulate(mu, alpha, beta, horizon=2000.0, seed=11)
        got = fit(times, 2000.0)
        rate = len(times[0]) / 2000.0
        poisson_ll = log_likelihood(times, 2000.0, [rate, 1e-8],
                                    [[0.0, 0.0], [0.0, 0.0]],
                                    [[1.0, 1.0], [1.0, 1.0]])
        self.assertGreater(got["log_likelihood"], poisson_ll)


class TestGoodnessOfFit(unittest.TestCase):
    def test_residuals_of_a_poisson_process_are_exponential(self):
        """Time rescaling: with the true parameters the compensator
        differences must look like Exp(1)."""
        mu = [0.8, 1e-9]
        zero = [[0.0, 0.0], [0.0, 0.0]]
        beta = [[1.0, 1.0], [1.0, 1.0]]
        times = simulate(mu, zero, beta, horizon=3000.0, seed=5)
        resid = rescaled_residuals(times, 3000.0, mu, zero, beta)[0]
        self.assertGreater(len(resid), 500)
        self.assertAlmostEqual(float(np.mean(resid)), 1.0, delta=0.1)
        p = stats.kstest(resid, "expon").pvalue
        self.assertGreater(p, 0.01, "residuals should not be rejected as Exp(1)")

    def test_residuals_reject_a_wrong_model(self):
        """Fit the wrong baseline and the test must notice."""
        mu = [0.8, 1e-9]
        zero = [[0.0, 0.0], [0.0, 0.0]]
        beta = [[1.0, 1.0], [1.0, 1.0]]
        times = simulate(mu, zero, beta, horizon=3000.0, seed=5)
        wrong = rescaled_residuals(times, 3000.0, [3.0, 1e-9], zero, beta)[0]
        p = stats.kstest(wrong, "expon").pvalue
        self.assertLess(p, 1e-6, "a badly wrong rate must be rejected")



class TestUnidentifiedStreams(unittest.TestCase):
    """A silent stream must not poison the branching ratio.

    alpha[i][j] multiplies a sum over the events of stream j, so an empty
    stream j leaves that parameter absent from the likelihood entirely. Before
    this was handled the fitter returned a branching ratio of 4e8 on a pair
    whose second stream was deliberately silent.
    """

    def test_empty_second_stream_gives_a_sane_branching_ratio(self):
        mu = [0.5, 1e-8]
        alpha = [[0.9, 0.0], [0.0, 0.0]]
        beta = [[2.0, 5.0], [5.0, 5.0]]
        times = simulate(mu, alpha, beta, horizon=1500.0, seed=7)
        self.assertLess(len(times[1]), 5, "second stream should be silent")
        got = fit(times, 1500.0)
        n = branching_ratio(got["alpha"], got["beta"])
        self.assertLess(n, 1.0, f"branching ratio ran away to {n}")
        self.assertGreater(n, 0.0)

    def test_columns_of_silent_streams_are_zeroed(self):
        times = [np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), np.array([])]
        got = fit(times, 10.0)
        np.testing.assert_allclose(np.asarray(got["alpha"])[:, 1], 0.0)

    def _nearly_silent_second_stream(self):
        times = simulate([0.5, 0.0], [[0.3, 0.0], [0.0, 0.0]],
                         [[2.0, 2.0], [2.0, 2.0]], horizon=500.0, seed=2)
        # three events: below the threshold, but they do enter the likelihood
        times[1] = np.array([100.0, 100.05, 250.0])
        return times

    def test_reported_likelihood_belongs_to_the_returned_parameters(self):
        """A stream with one to four events has its column zeroed. If that
        happens only after the optimiser finishes, the reported likelihood is
        that of the unzeroed parameters: it came out 10 units too high here."""
        times = self._nearly_silent_second_stream()
        got = fit(times, 500.0)
        np.testing.assert_array_equal(np.asarray(got["alpha"])[:, 1], 0.0)
        self.assertAlmostEqual(
            got["log_likelihood"],
            log_likelihood(times, 500.0, got["mu"], got["alpha"], got["beta"]),
            places=6)

    def test_fixed_beta_likelihood_belongs_to_the_returned_parameters(self):
        times = self._nearly_silent_second_stream()
        got = fit_fixed_beta(times, 500.0, np.full((2, 2), 2.0))
        np.testing.assert_array_equal(np.asarray(got["alpha"])[:, 1], 0.0)
        self.assertAlmostEqual(
            got["log_likelihood"],
            log_likelihood(times, 500.0, got["mu"], got["alpha"], got["beta"]),
            places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
