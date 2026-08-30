# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Multi-exponential (approximately power-law) kernels.

The single-exponential model was rejected on real trade flow: the likelihood
rose monotonically as the decay rate grew, with no interior optimum. That is
the signature of a process with no single characteristic timescale, which is
why the market-microstructure literature uses power-law kernels.

A true power law costs O(n^2) because it has no recursive form. The standard
way round it, following Bochud & Challet and Hardiman, Bercot & Bouchaud, is to
approximate the power law by a sum of exponentials with geometrically spaced
decay rates:

    phi_ij(t) = sum_k a_ijk * exp(-beta_k t),   beta_k = beta_0 * gamma^k

which keeps the O(n) recursion, one state per component.

The reason this fits quickly is that the recursive state R_ijk(t) depends only
on beta_k and the event times -- never on the weights. So every R is computed
once, and the log-likelihood becomes linear in the weights inside a log:

    lambda_i(t) = mu_i + sum_j sum_k a_ijk R_ijk(t)

which is concave in (mu, a), has an exact gradient, and solves in seconds.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .hawkes import _recursive_state


def geometric_betas(n_components=8, half_life_fast=0.01, half_life_slow=30.0):
    """Decay rates spanning fast to slow, geometrically spaced.

    Expressed as half-lives because that is the interpretable end: the grid
    below spans roughly a hundredth of a second to half a minute, which brackets
    both order-splitting bursts and slow news response.
    """
    hl = np.geomspace(half_life_fast, half_life_slow, n_components)
    return np.log(2.0) / hl


class MultiExpDesign:
    """Precomputed quantities that do not depend on the fitted weights."""

    def __init__(self, times_by_type, horizon, betas):
        self.times = [np.asarray(t, dtype=float) for t in times_by_type]
        self.horizon = float(horizon)
        self.betas = np.asarray(betas, dtype=float)
        self.d = len(self.times)
        self.m = len(self.betas)

        # R[i][j][k] : the k-th recursive state of stream j evaluated at the
        # event times of stream i. Computed once; reused by every likelihood
        # evaluation, which is what makes the fit fast.
        self.R = [[[_recursive_state(self.times[i], self.times[j], b)
                    for b in self.betas]
                   for j in range(self.d)]
                  for i in range(self.d)]

        # S[j][k] : integral of the k-th kernel over the observation window,
        # summed across the events of stream j.
        self.S = np.array([[float(np.sum(1.0 - np.exp(-b * (self.horizon - self.times[j]))))
                            if len(self.times[j]) else 0.0
                            for b in self.betas]
                           for j in range(self.d)])

    def intensities(self, mu, a):
        """lambda_i at the events of stream i, for weights a[i][j][k]."""
        out = []
        for i in range(self.d):
            lam = np.full(len(self.times[i]), float(mu[i]))
            for j in range(self.d):
                for k in range(self.m):
                    if a[i][j][k]:
                        lam = lam + a[i][j][k] * self.R[i][j][k]
            out.append(lam)
        return out

    def log_likelihood(self, mu, a):
        total = 0.0
        lams = self.intensities(mu, a)
        for i in range(self.d):
            total -= mu[i] * self.horizon
            for j in range(self.d):
                total -= float(np.sum(a[i][j] * self.S[j] / self.betas))
            lam = lams[i]
            if len(lam):
                if np.any(lam <= 0):
                    return -np.inf
                total += float(np.sum(np.log(lam)))
        return total

    # ---- packing helpers -------------------------------------------------
    def pack(self, mu, a):
        w = np.asarray(a, float) / self.betas
        return np.concatenate([np.asarray(mu, float).ravel(), w.ravel()])

    def unpack(self, x):
        """Parameters are kernel NORMS w = a/beta, not raw amplitudes.

        The decay rates span four orders of magnitude, so the amplitudes a do
        too, and L-BFGS-B on them is badly conditioned -- it stalls and returns
        a fit worse than the single-exponential model it contains as a special
        case. Norms are all branching contributions on the same scale, which
        fixes the conditioning. a = w * beta recovers the amplitudes.
        """
        mu = x[:self.d]
        w = x[self.d:].reshape(self.d, self.d, self.m)
        return mu, w * self.betas

    def neg_ll_and_grad(self, x):
        """Negative log-likelihood and its exact gradient.

        d/dmu_i  = -T      + sum_events 1/lambda_i
        d/da_ijk = -S_jk/beta_k + sum_events R_ijk/lambda_i
        """
        mu, a = self.unpack(x)
        lams = self.intensities(mu, a)
        ll = 0.0
        g_mu = np.zeros(self.d)
        g_a = np.zeros((self.d, self.d, self.m))

        for i in range(self.d):
            lam = lams[i]
            if len(lam) and np.any(lam <= 0):
                return 1e12, np.zeros_like(x)
            ll -= mu[i] * self.horizon
            g_mu[i] -= self.horizon
            inv = 1.0 / lam if len(lam) else np.zeros(0)
            if len(lam):
                ll += float(np.sum(np.log(lam)))
                g_mu[i] += float(np.sum(inv))
            for j in range(self.d):
                # in norm space the compensator is simply sum_k w_ijk * S_jk
                ll -= float(np.sum((a[i][j] / self.betas) * self.S[j]))
                g_a[i][j] -= self.S[j]
                if len(lam):
                    for k in range(self.m):
                        g_a[i][j][k] += self.betas[k] * float(
                            np.dot(self.R[i][j][k], inv))
        return -ll, -np.concatenate([g_mu.ravel(), g_a.ravel()])


def fit_multi_exp(times_by_type, horizon, betas=None, n_components=8,
                  max_iter=500, design=None):
    """Fit mu and the kernel weights with the decay grid held fixed.

    Non-negativity is imposed as a bound rather than by a log transform, so a
    component the data does not want can sit exactly at zero instead of being
    pushed to minus infinity.
    """
    if betas is None:
        betas = geometric_betas(n_components)
    des = design or MultiExpDesign(times_by_type, horizon, betas)

    rates = np.array([max(len(t), 1) / horizon for t in des.times])
    # start every component at a small equal share of the branching budget
    x0 = np.concatenate([np.maximum(rates * 0.5, 1e-6),
                         np.full(des.d * des.d * des.m, 0.02)])
    bounds = [(1e-9, None)] * des.d + [(0.0, None)] * (des.d * des.d * des.m)

    res = minimize(des.neg_ll_and_grad, x0, jac=True, method="L-BFGS-B",
                   bounds=bounds, options={"maxiter": max_iter, "ftol": 1e-12})
    mu, a = des.unpack(res.x)
    return {"mu": mu, "a": a, "betas": des.betas,
            "log_likelihood": -res.fun, "success": bool(res.success),
            "design": des}


def branching_matrix_multi(a, betas):
    """||phi_ij|| = sum_k a_ijk / beta_k."""
    a = np.asarray(a, dtype=float)
    betas = np.asarray(betas, dtype=float)
    return np.sum(a / betas, axis=2)


def branching_ratio_multi(a, betas):
    return float(np.max(np.abs(np.linalg.eigvals(branching_matrix_multi(a, betas)))))


def kernel_values(a_ij, betas, t):
    """Evaluate a fitted kernel phi_ij(t) on a time grid."""
    t = np.asarray(t, dtype=float)
    return np.sum(np.asarray(a_ij)[:, None] *
                  np.exp(-np.asarray(betas)[:, None] * t[None, :]), axis=0)


def rescaled_residuals_multi(design, mu, a):
    """Time-rescaling residuals for the multi-exponential fit.

    The compensator between consecutive events of stream i is integrated
    component by component, so this is exact rather than a quadrature.
    """
    out = []
    for i in range(design.d):
        ti = design.times[i]
        if len(ti) < 2:
            out.append(np.array([]))
            continue
        comp = np.zeros(len(ti))
        for idx, t in enumerate(ti):
            total = mu[i] * t
            for j in range(design.d):
                tj = design.times[j]
                tj = tj[tj < t]
                if len(tj):
                    for k, b in enumerate(design.betas):
                        if a[i][j][k]:
                            total += (a[i][j][k] / b) * float(
                                np.sum(1.0 - np.exp(-b * (t - tj))))
            comp[idx] = total
        out.append(np.diff(comp))
    return out
