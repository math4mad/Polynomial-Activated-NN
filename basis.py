"""
Orthogonal polynomial bases for learnable activations — exp8, PolyNN bench.

The Jacobi part is *ported* from the JacobiGP bench rather than reimplemented:
``JacobiGP@f181923:src/jacobigp/basis.py`` (blob 597d1fea1f824415be35d3cf6eeaee369bdf0135),
cited per CHORA law ("hashes over histories"). Unit tests in ``tests/test_basis.py``
verify the ported recurrence against ``scipy.special`` before any run consumes it.

Conventions (unchanged from the source):
    weight         w(x) = (1-x)**alpha (1+x)**beta,   alpha, beta > -1
    orthonormal    int  hat P_n hat P_m w dx = delta_nm
    recurrence     x hat P_n = a_n hat P_{n+1} + b_n hat P_n + a_{n-1} hat P_{n-1}
                   (Gautschi, Orthogonal Polynomials, 3.2.4-3.2.7)

Everything is built from torch primitives so autograd differentiates through it
with respect to x and the parameters alpha, beta — that is the point of the bench:
the *shape* knob, learned inside the neuron.

Also here: physicist Hermite (H_{n+1} = 2x H_n - 2n H_{n-1}, as in the project
brief) and Chebyshev first kind (T_{n+1} = 2x T_n - T_{n-1}), unnormalized —
their explosion behaviour on wide pre-activations is part of the measurement.
"""

from __future__ import annotations

import math

import torch


# --------------------------------------------------------------------------- #
# Jacobi: recurrence coefficients and norms (ported from JacobiGP basis.py)
# --------------------------------------------------------------------------- #
def recurrence_coeffs(N: int, alpha: torch.Tensor, beta: torch.Tensor):
    """a_n = sqrt(lam_{n+1}) (n=0..N-1) and b_n (n=0..N-1) of the orthonormal recurrence."""
    dtype, device = alpha.dtype, alpha.device
    ab = alpha + beta
    if float(ab) <= -2 + 1e-12:  # outside the admissible range
        raise ValueError(f"need alpha+beta > -2 (got {float(ab)})")

    # lam_1 stored in cancelled form: the generic branch is 0/0 at n=1 when
    # alpha+beta+1=0 and autograd would poison d/dalpha through the discarded entry.
    lam1 = 4.0 * (1 + alpha) * (1 + beta) / ((2 + ab) ** 2 * (3 + ab))
    if N == 1:
        lam = lam1.reshape(1)
    else:
        n = torch.arange(2, N + 1, dtype=dtype, device=device)
        num = 4.0 * n * (n + alpha) * (n + beta) * (n + ab)
        den = (2 * n + ab) ** 2 * (2 * n + ab + 1) * (2 * n + ab - 1)
        lam = torch.cat([lam1.reshape(1), num / den])

    a = lam.sqrt()[:N]
    n_ge1 = torch.arange(1, N, dtype=dtype, device=device)
    b = torch.cat([
        ((beta - alpha) / (ab + 2)).reshape(1),
        (beta**2 - alpha**2) / ((2 * n_ge1 + ab) * (2 * n_ge1 + ab + 2)),
    ])[:N]
    return a, b


def jacobi_mass(alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """mu0 = int_{-1}^{1} (1-x)^alpha (1+x)^beta dx."""
    ab = alpha + beta
    return torch.exp(
        (ab + 1) * math.log(2.0)
        + torch.lgamma(alpha + 1)
        + torch.lgamma(beta + 1)
        - torch.lgamma(ab + 2)
    )


def _edge_clamp(x: torch.Tensor) -> torch.Tensor:
    """Polynomials are continuous, so clamping |x| to 1-1e-15 is numerically exact."""
    return x.clamp(-1.0 + 1e-15, 1.0 - 1e-15)


def jacobi_orthonormal(x: torch.Tensor, N: int, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """hat P_n^{(alpha,beta)}(x), n = 0..N-1;  x of shape (M,) -> output (M, N).

    Differentiable in x, alpha and beta. ``x`` is clamped to the basis domain
    [-1, 1]: for the squashed activation the input is tanh(pre-activation) and
    already lives inside; the clamp is a numerical belt, not a data change.
    """
    x = _edge_clamp(torch.as_tensor(x, dtype=alpha.dtype, device=alpha.device)).reshape(-1)
    a, b = recurrence_coeffs(max(N, 1), alpha, beta)

    out = torch.empty(x.numel(), N, dtype=alpha.dtype, device=alpha.device)
    p_prev = torch.zeros_like(x)
    p = torch.ones_like(x) / torch.sqrt(jacobi_mass(alpha, beta))
    for n in range(N):
        out[:, n] = p
        rest = a[n - 1] * p_prev if n else torch.zeros_like(p)
        p_next = ((x - b[n]) * p - rest) / a[n]
        p_prev, p = p, p_next
    return out


# --------------------------------------------------------------------------- #
# Hermite (physicist') and Chebyshev T — raw arms, unnormalized (brief's recursions)
# --------------------------------------------------------------------------- #
def hermite_physicist(x: torch.Tensor, N: int) -> torch.Tensor:
    """H_0..H_{N-1}(x), physicist' recursion from the project brief: (M, N)."""
    x = x.reshape(-1)
    cols = [torch.ones_like(x)]
    if N > 1:
        cols.append(2.0 * x)
    for n in range(1, N - 1):
        cols.append(2.0 * x * cols[-1] - 2.0 * n * cols[-2])
    return torch.stack(cols[:N], dim=1)


def chebyshev_t(x: torch.Tensor, N: int) -> torch.Tensor:
    """T_0..T_{N-1}(x), T_{n+1} = 2x T_n - T_{n-1}; (M, N). NOT clamped — the raw
    arm must feel its own explosion outside [-1, 1]; that is the measurement."""
    x = x.reshape(-1)
    cols = [torch.ones_like(x)]
    if N > 1:
        cols.append(x.clone())
    for n in range(1, N - 1):
        cols.append(2.0 * x * cols[-1] - cols[-2])
    return torch.stack(cols[:N], dim=1)


# --------------------------------------------------------------------------- #
# unified interface used by models.PolynomialActivation
# --------------------------------------------------------------------------- #
def basis_table(kind: str, x: torch.Tensor, N: int,
                alpha: torch.Tensor | None = None,
                beta: torch.Tensor | None = None) -> torch.Tensor:
    """(M, N) design matrix of the requested basis, orthonormal Jacobi or raw."""
    if kind == "jacobi":
        assert alpha is not None and beta is not None
        return jacobi_orthonormal(x, N, alpha, beta)
    if kind == "hermite":
        return hermite_physicist(x, N)
    if kind == "cheby":
        return chebyshev_t(x, N)
    raise ValueError(f"unknown basis kind {kind!r}")
