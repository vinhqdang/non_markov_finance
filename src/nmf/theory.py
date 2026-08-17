"""Closed-form quantities for the common-shock staleness model.

Everything here is derived in the paper; nothing is fitted.  Each routine is
written so that it can be checked against the exact Monte Carlo simulator in
`nmf.clocks`.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate
from scipy.special import gamma as _gamma, hyp1f1
from scipy.stats import beta as beta_dist


# ---------------------------------------------------------------------------
# Marginal undershoot law and its transform
# ---------------------------------------------------------------------------


def undershoot_law(alpha: float):
    """Law of H(t)/t for a standard alpha-stable clock: Beta(alpha, 1-alpha)."""
    return beta_dist(alpha, 1.0 - alpha)


def undershoot_laplace(c, t, alpha):
    """E[exp(-c H(t))] = 1F1(alpha; 1; -c t)."""
    return hyp1f1(alpha, 1.0, -np.asarray(c, dtype=float) * t)


def undershoot_laplace_quad(c, t, alpha):
    """Same quantity by direct quadrature of the Beta integral (independent check)."""
    f = lambda s: np.exp(-c * t * s) * beta_dist(alpha, 1.0 - alpha).pdf(s)
    val, err = integrate.quad(f, 0.0, 1.0, points=[0.0, 1.0], limit=400)
    return val, err


def levy_tail(z, alpha):
    """Tail Pibar(z) of the standard alpha-stable Levy measure."""
    return np.asarray(z, dtype=float) ** (-alpha) / _gamma(1.0 - alpha)


# ---------------------------------------------------------------------------
# Bivariate potential density at alpha = 1/2  (Theorem: closed form)
# ---------------------------------------------------------------------------


def _potential_integrand(z, s1, s2, a):
    x1 = s1 - a * z
    x2 = s2 - a * z
    den = x1 * x2 + z * (x1 + x2)
    return np.sqrt(np.maximum(z * x1 * x2, 0.0)) / den ** 2


def potential_density_half(s1, s2, a=1.0, n_gl=400):
    """u(s1, s2) for alpha = 1/2, common-shock clocks S_j = X_j + a Z.

        u = pi^{-3/2} int_0^{min(s1,s2)/a} sqrt(z x1 x2) / (x1 x2 + z(x1+x2))^2 dz

    The substitution z = (m/a)(1 - w^2) absorbs the square-root vanishing at the
    upper endpoint, after which fixed-order Gauss-Legendre is accurate.
    """
    m = min(s1, s2) / a
    w, wt = np.polynomial.legendre.leggauss(n_gl)
    w = 0.5 * (w + 1.0)                      # w in (0,1)
    wt = 0.5 * wt
    z = m * (1.0 - w ** 2)
    jac = m * 2.0 * w
    vals = _potential_integrand(z, s1, s2, a) * jac
    return float(np.pi ** -1.5 * np.sum(wt * vals))


def potential_density_half_quad(s1, s2, a=1.0):
    """Same quantity via adaptive quadrature, for cross-checking `potential_density_half`."""
    m = min(s1, s2) / a
    f = lambda z: _potential_integrand(z, s1, s2, a)
    val, err = integrate.quad(f, 0.0, m, limit=500, points=[0.0, m])
    return np.pi ** -1.5 * val, np.pi ** -1.5 * err


def potential_density_half_grid(S1, S2, a=1.0, n_gl=200):
    """Vectorized evaluation of `potential_density_half` on broadcastable arrays."""
    S1 = np.asarray(S1, dtype=float)
    S2 = np.asarray(S2, dtype=float)
    m = np.minimum(S1, S2) / a
    w, wt = np.polynomial.legendre.leggauss(n_gl)
    w = 0.5 * (w + 1.0)
    wt = 0.5 * wt
    shape = np.broadcast(S1, S2).shape
    z = m[..., None] * (1.0 - w ** 2)
    jac = m[..., None] * 2.0 * w
    x1 = S1[..., None] - a * z
    x2 = S2[..., None] - a * z
    den = x1 * x2 + z * (x1 + x2)
    vals = np.sqrt(np.maximum(z * x1 * x2, 0.0)) / den ** 2 * jac
    out = np.pi ** -1.5 * np.sum(wt * vals, axis=-1)
    return out.reshape(shape)


def potential_density_marginal_half(s, a_unused=None):
    """One-dimensional potential density of a standard 1/2-stable subordinator."""
    return 1.0 / np.sqrt(np.pi * np.asarray(s, dtype=float))


# ---------------------------------------------------------------------------
# Simultaneous-staleness atom via the compensation formula
# ---------------------------------------------------------------------------


def atom_compensation_half(t=1.0, a=1.0, n_outer=400, n_inner=400):
    """P[L_1(t) = L_2(t)] from the compensation-formula representation

        pi = int_{[0,t]^2} Pibar_{alpha_c}( (t-s1)/a1 v (t-s2)/a2 ) U(ds1, ds2),

    specialized to alpha = alpha_c = 1/2 and a1 = a2 = a, where the closed-form
    bivariate potential density of the theorem is available.

    By symmetry the integral is twice the contribution of {s1 < s2}, on which
    min(s1, s2) = s1 and the Levy tail factor is sqrt(a / (pi (t - s1))).
    The diagonal is approached from one side only, which is exactly the
    quadrature split the paper's appendix identifies as necessary.
    """
    # outer variable: s1 = t(1 - v^2) absorbs the (t - s1)^{-1/2} endpoint
    v, wv = np.polynomial.legendre.leggauss(n_outer)
    v = 0.5 * (v + 1.0)
    wv = 0.5 * wv
    s1 = t * (1.0 - v ** 2)
    jac1 = t * 2.0 * v
    weight = np.sqrt(a / np.pi) / np.sqrt(np.maximum(t - s1, 1e-300))

    # inner variable: s2 in (s1, t); the log singularity at s2 = s1 is absorbed
    # by s2 = s1 + (t - s1) * y^2 (clustering nodes at the diagonal).
    y, wy = np.polynomial.legendre.leggauss(n_inner)
    y = 0.5 * (y + 1.0)
    wy = 0.5 * wy

    s2 = s1[:, None] + (t - s1)[:, None] * y ** 2
    jac2 = (t - s1)[:, None] * 2.0 * y
    u = potential_density_half_grid(np.broadcast_to(s1[:, None], s2.shape), s2, a=a)
    inner = np.sum(wy * u * jac2, axis=1)

    return float(2.0 * np.sum(wv * weight * inner * jac1))


# ---------------------------------------------------------------------------
# Short-horizon correlation (Proposition)
# ---------------------------------------------------------------------------


def rho_zero(a_i, a_h, sigma_i, sigma_h, alpha_i, alpha_h, alpha_c):
    """rho_{ih}(0+) = a_i a_h alpha_c / sqrt((s_i^2 al_i + a_i^2 al_c)(s_h^2 al_h + a_h^2 al_c))."""
    num = a_i * a_h * alpha_c
    den = np.sqrt(
        (sigma_i ** 2 * alpha_i + a_i ** 2 * alpha_c)
        * (sigma_h ** 2 * alpha_h + a_h ** 2 * alpha_c)
    )
    return num / den
