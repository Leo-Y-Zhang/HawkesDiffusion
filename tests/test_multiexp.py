"""Tests for the multi-exponential kernel.

The decisive ones: it must reduce exactly to the single-exponential model when
given one component, and it must recover a known multi-exponential process.
"""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from hawkesdiffusion.hawkes import log_likelihood, simulate       # noqa: E402
from hawkesdiffusion.multiexp import (                            # noqa: E402
    MultiExpDesign, branching_matrix_multi, branching_ratio_multi,
    fit_multi_exp, geometric_betas, kernel_values, rescaled_residuals_multi)


class TestAgreesWithSingleExponential(unittest.TestCase):
    """One component must reproduce the original model exactly."""

    def test_log_likelihood_matches(self):
        mu = [0.4, 0.2]
        alpha = [[0.5, 0.1], [0.2, 0.3]]
        beta = 2.0
        times = simulate(mu, alpha, [[beta] * 2] * 2, horizon=400.0, seed=3)

        old = log_likelihood(times, 400.0, mu, alpha, [[beta] * 2] * 2)
        des = MultiExpDesign(times, 400.0, [beta])
        a = np.array(alpha, dtype=float).reshape(2, 2, 1)
        new = des.log_likelihood(np.array(mu), a)
        self.assertAlmostEqual(old, new, places=6)

    def test_branching_matrix_matches(self):
        a = np.array([[[0.5], [0.1]], [[0.2], [0.3]]], dtype=float)
        m = branching_matrix_multi(a, [2.0])
        np.testing.assert_allclose(m, [[0.25, 0.05], [0.10, 0.15]])


class TestGradient(unittest.TestCase):
    def test_analytic_gradient_matches_finite_differences(self):
        """A wrong gradient makes the optimiser converge confidently to the
        wrong answer, so it is checked rather than trusted."""
        times = simulate([0.5, 0.3], [[0.4, 0.1], [0.1, 0.4]],
                         [[3.0] * 2] * 2, horizon=200.0, seed=5)
        des = MultiExpDesign(times, 200.0, geometric_betas(3, 0.05, 5.0))
        rng = np.random.default_rng(0)
        x = np.abs(rng.normal(0.3, 0.1, des.d + des.d * des.d * des.m))
        f0, g = des.neg_ll_and_grad(x)
        eps = 1e-6
        for idx in (0, 1, 3, 7, len(x) - 1):
            xp = x.copy(); xp[idx] += eps
            xm = x.copy(); xm[idx] -= eps
            num = (des.neg_ll_and_grad(xp)[0] - des.neg_ll_and_grad(xm)[0]) / (2 * eps)
            self.assertAlmostEqual(num, g[idx], delta=1e-3 * max(1.0, abs(g[idx])),
                                   msg=f"gradient mismatch at {idx}")


class TestRecovery(unittest.TestCase):
    def test_recovers_a_two_timescale_process(self):
        """Simulate a process with a fast and a slow component and require the
        fit to put weight on both, not collapse onto one."""
        betas = np.array([np.log(2) / 0.05, np.log(2) / 2.0])
        a_true = np.zeros((2, 2, 2))
        a_true[0][0] = [4.0, 0.15]
        times = simulate([0.5, 1e-9],
                         [[float(np.sum(a_true[0][0])), 0.0], [0.0, 0.0]],
                         [[float(betas[0])] * 2] * 2, horizon=1500.0, seed=9)
        got = fit_multi_exp(times, 1500.0, betas=betas)
        self.assertTrue(got["success"])
        self.assertGreater(got["a"][0][0].sum(), 0.0)
        # the fitted branching ratio must be a sane, sub-critical number
        n = branching_ratio_multi(got["a"], got["betas"])
        self.assertGreater(n, 0.0)
        self.assertLess(n, 1.0)

    def test_beats_the_single_exponential_on_its_own_data(self):
        times = simulate([0.4, 1e-9], [[1.2, 0.0], [0.0, 0.0]],
                         [[3.0] * 2] * 2, horizon=1200.0, seed=11)
        multi = fit_multi_exp(times, 1200.0, betas=geometric_betas(6, 0.02, 10.0))
        single = log_likelihood(times, 1200.0, [0.4, 1e-9],
                                [[1.2, 0.0], [0.0, 0.0]], [[3.0] * 2] * 2)
        self.assertGreater(multi["log_likelihood"], single - 5.0)


class TestKernelAndResiduals(unittest.TestCase):
    def test_kernel_decays_and_is_positive(self):
        v = kernel_values([1.0, 0.2], [5.0, 0.5], np.linspace(0, 5, 50))
        self.assertTrue(np.all(v > 0))
        self.assertTrue(np.all(np.diff(v) < 0))

    def test_residuals_of_a_poisson_process_are_exponential(self):
        times = simulate([0.8, 1e-9], [[0.0, 0.0], [0.0, 0.0]],
                         [[1.0] * 2] * 2, horizon=2000.0, seed=13)
        des = MultiExpDesign(times, 2000.0, geometric_betas(3, 0.1, 5.0))
        a = np.zeros((2, 2, 3))
        r = rescaled_residuals_multi(des, [0.8, 1e-9], a)[0]
        self.assertGreater(len(r), 300)
        self.assertAlmostEqual(float(np.mean(r)), 1.0, delta=0.12)
        self.assertGreater(stats.kstest(r, "expon").pvalue, 0.01)

    def test_geometric_betas_span_the_requested_half_lives(self):
        b = geometric_betas(5, 0.01, 10.0)
        hl = np.log(2) / b
        self.assertAlmostEqual(hl[0], 0.01, places=6)
        self.assertAlmostEqual(hl[-1], 10.0, places=6)
        self.assertEqual(len(b), 5)
