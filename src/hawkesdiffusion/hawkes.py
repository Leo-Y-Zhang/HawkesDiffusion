# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Bivariate Hawkes process with exponential kernels.

A Hawkes process is a point process that excites itself: every event raises the
chance of the next one, and the effect decays. With two streams it also
cross-excites, so you can ask how hard a shock in one stream hits the other and
how long the reaction lasts.

    lambda_i(t) = mu_i + sum_j sum_{t_k^j < t} alpha_ij * exp(-beta_ij (t - t_k^j))

Two quantities carry the interpretation:

* the **half-life** ln(2)/beta_ij, how fast the reaction to one event decays;
* the **branching ratio**, the spectral radius of the matrix [alpha_ij/beta_ij],
  which is the fraction of events that are the market reacting to itself rather
  than to outside information. Below 1 the process is stationary; approaching 1
  means nearly everything is echo.

The likelihood is written recursively so it costs O(n) rather than O(n^2),
which is what makes fitting a real trade stream feasible.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _recursive_state(target_times, source_times, beta):
    """R[k] = sum over source events strictly before target_times[k] of
    exp(-beta * (target_times[k] - source_time)), computed in one pass."""
    n = len(target_times)
    out = np.zeros(n)
    if len(source_times) == 0 or n == 0:
        return out
    j = 0
    acc = 0.0
    prev_t = None
    for k in range(n):
        t = target_times[k]
        if prev_t is not None:
            acc *= np.exp(-beta * (t - prev_t))
        while j < len(source_times) and source_times[j] < t:
            acc += np.exp(-beta * (t - source_times[j]))
            j += 1
        out[k] = acc
        prev_t = t
    return out


def intensity_at_events(times_by_type, mu, alpha, beta):
    """lambda_i evaluated at each event of type i."""
    d = len(times_by_type)
    out = []
    for i in range(d):
        lam = np.full(len(times_by_type[i]), float(mu[i]))
        for j in range(d):
            r = _recursive_state(times_by_type[i], times_by_type[j], beta[i][j])
            lam = lam + alpha[i][j] * r
        out.append(lam)
    return out


def log_likelihood(times_by_type, horizon, mu, alpha, beta):
    """Exact log-likelihood of a bivariate exponential-kernel Hawkes process."""
    d = len(times_by_type)
    total = 0.0
    for i in range(d):
        ti = times_by_type[i]
        total -= mu[i] * horizon
        for j in range(d):
            tj = times_by_type[j]
            if len(tj) == 0 or beta[i][j] <= 0:
                continue
            decayed = 1.0 - np.exp(-beta[i][j] * (horizon - tj))
            total -= (alpha[i][j] / beta[i][j]) * float(np.sum(decayed))
        if len(ti):
            lam = np.full(len(ti), float(mu[i]))
            for j in range(d):
                lam = lam + alpha[i][j] * _recursive_state(ti, times_by_type[j],
                                                           beta[i][j])
            if np.any(lam <= 0):
                return -np.inf
            total += float(np.sum(np.log(lam)))
    return total


def _pack(mu, alpha, beta):
    return np.concatenate([np.log(mu), np.log(np.ravel(alpha)),
                           np.log(np.ravel(beta))])


def _unpack(x, d=2):
    # Nelder-Mead explores freely in log space, so clip the exponent before
    # exponentiating. Without this the search overflows to inf, the likelihood
    # becomes nan rather than a large penalty, and the simplex can wander
    # instead of being pushed back.
    x = np.clip(np.asarray(x, dtype=float), -60.0, 60.0)
    mu = np.exp(x[:d])
    alpha = np.exp(x[d:d + d * d]).reshape(d, d)
    beta = np.exp(x[d + d * d:]).reshape(d, d)
    return mu, alpha, beta


def fit(times_by_type, horizon, x0=None, maxiter=600):
    """Maximum-likelihood fit.

    Parameters are optimised in log space so positivity is automatic and no
    bound can be hit exactly, which is what makes the optimiser well behaved
    here.
    """
    d = len(times_by_type)
    if x0 is None:
        rates = np.array([max(len(t), 1) / horizon for t in times_by_type])
        mu0 = np.maximum(rates * 0.5, 1e-6)
        alpha0 = np.full((d, d), float(np.mean(rates)) * 0.5 + 1e-3)
        beta0 = np.full((d, d), max(float(np.mean(rates)) * 2.0, 1.0))
        x0 = _pack(mu0, alpha0, beta0)

    def neg(x):
        mu, alpha, beta = _unpack(x, d)
        ll = log_likelihood(times_by_type, horizon, mu, alpha, beta)
        return -ll if np.isfinite(ll) else 1e12

    res = minimize(neg, x0, method="Nelder-Mead",
                   options={"maxiter": maxiter * len(x0), "fatol": 1e-6,
                            "xatol": 1e-6, "adaptive": True})
    mu, alpha, beta = _unpack(res.x, d)
    alpha = _zero_unidentified(alpha, times_by_type)
    return {"mu": mu, "alpha": alpha, "beta": beta,
            "log_likelihood": -res.fun, "success": bool(res.success),
            "n_iter": int(res.nit)}


def _zero_unidentified(alpha, times_by_type, min_events=5):
    """Zero the kernel columns of streams that have (almost) no events.

    alpha[i][j] multiplies a sum over the events of stream j. If stream j is
    empty that sum is zero for every t, so alpha[i][j] does not appear in the
    likelihood at all: it is unidentified, the optimiser leaves it wherever it
    started, and the spectral radius then reports whatever noise it landed on.
    Observed in practice as a branching ratio of 4e8 on a simulated pair whose
    second stream was deliberately silent.

    Zero is the right normalisation, not a fudge: a stream with no events
    excites nothing, so its kernel norm genuinely is zero.
    """
    alpha = np.array(alpha, dtype=float, copy=True)
    for j, t in enumerate(times_by_type):
        if len(t) < min_events:
            alpha[:, j] = 0.0
    return alpha


def branching_matrix(alpha, beta):
    """Integrated kernel norms, ||phi_ij|| = alpha_ij / beta_ij."""
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(beta > 0, alpha / beta, 0.0)


def branching_ratio(alpha, beta):
    """Spectral radius of the integrated kernel matrix.

    The share of activity that is the system reacting to itself. Stationary
    only if this is below 1.
    """
    return float(np.max(np.abs(np.linalg.eigvals(branching_matrix(alpha, beta)))))


def half_life(beta):
    """ln(2)/beta: how long the reaction to a single event takes to halve."""
    beta = np.asarray(beta, dtype=float)
    with np.errstate(divide="ignore"):
        return np.where(beta > 0, np.log(2.0) / beta, np.inf)


def simulate(mu, alpha, beta, horizon, seed=0, max_events=200000):
    """Ogata thinning.

    Used to generate data with known parameters, which is the only way to check
    that the fitter recovers the truth.
    """
    rng = np.random.default_rng(seed)
    mu = np.asarray(mu, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    d = len(mu)
    times = [[] for _ in range(d)]
    t = 0.0

    while t < horizon:
        # upper bound on total intensity at t+
        lam = mu.copy()
        for i in range(d):
            for j in range(d):
                if times[j]:
                    tj = np.asarray(times[j])
                    lam[i] += alpha[i][j] * float(
                        np.sum(np.exp(-beta[i][j] * (t - tj))))
        lam_bar = float(np.sum(lam))
        if lam_bar <= 0:
            break
        t += rng.exponential(1.0 / lam_bar)
        if t >= horizon:
            break

        lam_new = mu.copy()
        for i in range(d):
            for j in range(d):
                if times[j]:
                    tj = np.asarray(times[j])
                    lam_new[i] += alpha[i][j] * float(
                        np.sum(np.exp(-beta[i][j] * (t - tj))))
        total_new = float(np.sum(lam_new))
        if rng.random() <= total_new / lam_bar:
            which = rng.choice(d, p=lam_new / total_new)
            times[which].append(t)
            if sum(len(x) for x in times) >= max_events:
                break
    return [np.asarray(x) for x in times]


def rescaled_residuals(times_by_type, horizon, mu, alpha, beta):
    """Time-rescaling theorem residuals.

    Integrate the fitted intensity between consecutive events of a type. If the
    model is right, those integrals are i.i.d. Exp(1). This is the goodness-of-
    fit check that stops a fitted number being taken on faith.
    """
    d = len(times_by_type)
    out = []
    for i in range(d):
        ti = np.asarray(times_by_type[i])
        if len(ti) < 2:
            out.append(np.array([]))
            continue
        compensator = np.zeros(len(ti))
        for k, t in enumerate(ti):
            total = mu[i] * t
            for j in range(d):
                tj = np.asarray(times_by_type[j])
                tj = tj[tj < t]
                if len(tj) and beta[i][j] > 0:
                    total += (alpha[i][j] / beta[i][j]) * float(
                        np.sum(1.0 - np.exp(-beta[i][j] * (t - tj))))
            compensator[k] = total
        out.append(np.diff(compensator))
    return out


def fit_fixed_beta(times_by_type, horizon, beta, maxiter=400):
    """Fit mu and alpha with the decay rates held fixed.

    Fitting all ten parameters at once is badly conditioned on real trade data:
    the likelihood rewards ever-faster decay to explain sub-millisecond bursts,
    so beta runs away and the cross terms collapse to zero. Holding beta fixed
    leaves six well-behaved parameters, and scanning a grid of candidate
    timescales turns the awkward dimension into an explicit model choice.
    """
    d = len(times_by_type)
    beta = np.asarray(beta, dtype=float)
    rates = np.array([max(len(t), 1) / horizon for t in times_by_type])
    x0 = np.concatenate([np.log(np.maximum(rates * 0.5, 1e-9)),
                         np.log(np.full(d * d, 0.1))])

    def neg(x):
        x = np.clip(x, -60.0, 60.0)
        mu = np.exp(x[:d])
        alpha = np.exp(x[d:]).reshape(d, d)
        ll = log_likelihood(times_by_type, horizon, mu, alpha, beta)
        return -ll if np.isfinite(ll) else 1e12

    res = minimize(neg, x0, method="Nelder-Mead",
                   options={"maxiter": maxiter * len(x0), "fatol": 1e-6,
                            "xatol": 1e-6, "adaptive": True})
    x = np.clip(res.x, -60.0, 60.0)
    mu = np.exp(x[:d])
    alpha = _zero_unidentified(np.exp(x[d:]).reshape(d, d), times_by_type)
    return {"mu": mu, "alpha": alpha, "beta": beta,
            "log_likelihood": -res.fun, "success": bool(res.success)}


def fit_beta_grid(times_by_type, horizon, half_lives_seconds=None, maxiter=400):
    """Scan candidate decay timescales and keep the best by likelihood.

    Returns the winning fit plus the whole scan, so the choice of timescale is
    visible rather than buried in an optimiser trace.
    """
    if half_lives_seconds is None:
        half_lives_seconds = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 15.0]
    d = len(times_by_type)
    scan = []
    best = None
    for hl in half_lives_seconds:
        b = np.full((d, d), np.log(2.0) / hl)
        got = fit_fixed_beta(times_by_type, horizon, b, maxiter=maxiter)
        got["half_life_seconds"] = float(hl)
        got["branching_ratio"] = branching_ratio(got["alpha"], got["beta"])
        scan.append({"half_life_seconds": float(hl),
                     "log_likelihood": float(got["log_likelihood"]),
                     "branching_ratio": float(got["branching_ratio"])})
        if best is None or got["log_likelihood"] > best["log_likelihood"]:
            best = got
    best["scan"] = scan
    return best
