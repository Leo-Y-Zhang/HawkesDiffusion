# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Edges the recovery tests cannot see.

The recovery tests simulate a process and fit it back, which exercises the
model as a whole but not its cross terms one by one, the boundary at which a
stream counts as silent, or the code that turns a Binance tape into event
times. These tests pin each of those against a direct computation or a tape
built by hand.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from hawkesdiffusion import binance, hawkes, multiexp  # noqa: E402

# Two streams with deliberately asymmetric cross terms, so that any transposed
# index (alpha[j][i] for alpha[i][j]) changes the answer.
TIMES = [np.array([0.3, 1.1, 1.15, 2.0, 2.7]), np.array([0.5, 0.9, 1.6, 2.2])]
MU = [0.4, 0.7]
ALPHA = [[0.8, 0.1], [0.5, 0.3]]
BETA = [[2.0, 1.0], [3.0, 1.5]]


def direct_intensity(i, t):
    lam = MU[i]
    for j in range(2):
        for s in TIMES[j][TIMES[j] < t]:
            lam += ALPHA[i][j] * np.exp(-BETA[i][j] * (t - s))
    return lam


def direct_compensator(i, t):
    total = MU[i] * t
    for j in range(2):
        for s in TIMES[j][TIMES[j] < t]:
            total += ALPHA[i][j] / BETA[i][j] * (1 - np.exp(-BETA[i][j] * (t - s)))
    return total


class TestCrossTerms(unittest.TestCase):
    def test_intensity_matches_the_direct_sum_with_asymmetric_cross_terms(self):
        got = hawkes.intensity_at_events(TIMES, MU, ALPHA, BETA)
        for i in range(2):
            want = [direct_intensity(i, t) for t in TIMES[i]]
            np.testing.assert_allclose(got[i], want, rtol=1e-12)

    def test_residuals_are_differences_of_the_direct_compensator(self):
        got = hawkes.rescaled_residuals(TIMES, 3.0, MU, ALPHA, BETA)
        for i in range(2):
            want = np.diff([direct_compensator(i, t) for t in TIMES[i]])
            np.testing.assert_allclose(got[i], want, rtol=1e-12)


class TestSilentStreams(unittest.TestCase):
    def test_a_stream_is_silent_below_min_events_and_active_at_it(self):
        n = hawkes.MIN_EVENTS
        times = [np.arange(n - 1, dtype=float), np.arange(n, dtype=float)]
        alpha = np.array([[1.0, 2.0], [3.0, 4.0]])
        got = hawkes._zero_unidentified(alpha, times)
        np.testing.assert_array_equal(got, [[0.0, 2.0], [0.0, 4.0]])
        # The caller's matrix is left alone.
        np.testing.assert_array_equal(alpha, [[1.0, 2.0], [3.0, 4.0]])

    def test_multi_exp_pins_exactly_the_silent_columns(self):
        n = multiexp.MIN_EVENTS
        times = [np.linspace(0.1, 9.0, 30), np.linspace(0.5, 8.0, n),
                 np.linspace(1.0, 7.0, n - 1)]
        captured = {}

        def fake_minimize(fun, x0, jac, method, bounds, options):
            captured["bounds"] = bounds
            return mock.Mock(x=np.asarray(x0), fun=0.0, success=True)

        betas = np.array([1.0, 4.0])
        with mock.patch.object(multiexp, "minimize", fake_minimize):
            multiexp.fit_multi_exp(times, 10.0, betas=betas)
        weight_bounds = captured["bounds"][3:]
        for flat, bound in enumerate(weight_bounds):
            j = (flat // len(betas)) % 3
            self.assertEqual(bound, (0.0, 0.0) if j == 2 else (0.0, None))


class TestBetaGrid(unittest.TestCase):
    def test_the_best_likelihood_wins_and_every_timescale_is_reported(self):
        def fake_fit(times, horizon, beta, maxiter):
            hl = float(np.log(2.0) / beta[0][0])
            return {"mu": np.ones(2), "alpha": np.zeros((2, 2)), "beta": beta,
                    "log_likelihood": -(hl - 0.5) ** 2, "success": True}

        grid = [0.1, 0.5, 2.0]
        with mock.patch.object(hawkes, "fit_fixed_beta", fake_fit):
            best = hawkes.fit_beta_grid(TIMES, 3.0, half_lives_seconds=grid)
        self.assertEqual(best["half_life_seconds"], 0.5)
        self.assertEqual([r["half_life_seconds"] for r in best["scan"]], grid)


def tape(rows):
    return [{"a": a, "T": t, "m": m} for a, t, m in rows]


class TestEventStreams(unittest.TestCase):
    ROWS = tape([(1, 1000, False), (2, 1000, True), (3, 1500, False),
                 (4, 1500, False), (5, 1500, False), (6, 1500, False),
                 (7, 1500, False), (8, 3000, True)])

    def streams(self, rows):
        with mock.patch.object(binance, "fetch_trades", return_value=rows):
            return binance.event_streams("BTCUSDT", len(rows))

    def test_the_maker_flag_splits_buyers_from_sellers(self):
        b, s, meta = self.streams(self.ROWS)
        # m=True means the buyer was the maker: the trade was seller-initiated.
        self.assertEqual(len(b), 6)
        self.assertEqual(len(s), 2)
        np.testing.assert_array_equal(np.floor(s * 1000), [0, 2000])
        np.testing.assert_array_equal(np.floor(b * 1000), [0, 500, 500, 500, 500, 500])

    def test_jitter_stays_inside_the_millisecond_and_keeps_order(self):
        b, s, _ = self.streams(self.ROWS)
        for stream in (b, s):
            frac = stream * 1000 - np.floor(stream * 1000)
            self.assertTrue(np.all(frac < 1.0))
            self.assertTrue(np.all(np.diff(stream) >= 0))
        self.assertTrue(np.any(b > 0.0))

    def test_horizon_and_tie_count(self):
        b, s, meta = self.streams(self.ROWS)
        self.assertAlmostEqual(meta["horizon_seconds"], max(b[-1], s[-1]) + 1e-3, places=12)
        self.assertEqual(meta["n_tied_timestamps"], 8 - 3)
        self.assertEqual(meta["n_raw_trades"], 8)

    def test_an_empty_tape_is_an_error(self):
        with self.assertRaises(RuntimeError):
            self.streams([])


class TestFetch(unittest.TestCase):
    def test_trades_are_deduplicated_by_id_and_sorted_by_time(self):
        batch = tape([(5, 300, False), (3, 100, True), (5, 300, False)])
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(binance, "CACHE_DIR", tmp), \
                mock.patch.object(binance, "_get", return_value=batch), \
                mock.patch.object(binance.time, "sleep"):
            rows = binance.fetch_trades("BTCUSDT", 3, use_cache=False)
        self.assertEqual([r["a"] for r in rows], [3, 5])

    def test_get_retries_before_giving_up(self):
        body = io.BytesIO(json.dumps([{"a": 1}]).encode("utf-8"))
        flaky = mock.Mock(side_effect=[OSError("reset"), OSError("reset"), body])
        with mock.patch.object(binance.urllib.request, "urlopen", flaky), \
                mock.patch.object(binance.time, "sleep"):
            self.assertEqual(binance._get("https://example.invalid", retries=3), [{"a": 1}])
        dead = mock.Mock(side_effect=OSError("down"))
        with mock.patch.object(binance.urllib.request, "urlopen", dead), \
                mock.patch.object(binance.time, "sleep"), \
                self.assertRaises(RuntimeError):
            binance._get("https://example.invalid", retries=2)
        self.assertEqual(dead.call_count, 2)


if __name__ == "__main__":
    unittest.main()
