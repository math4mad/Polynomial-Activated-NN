"""
Unit tests for the ported Jacobi recurrence and the raw bases — the promise in
Letter 005: `basis.py` output is verified against scipy.special on a grid
BEFORE any run consumes it. Run:  python -m unittest tests.test_basis -v
"""

from __future__ import annotations

import unittest

import numpy as np
import torch
from scipy import special

from basis import (basis_table, chebyshev_t, hermite_physicist,
                   jacobi_mass, jacobi_orthonormal)

TOL = 1e-9


def classic_log_hn(n, a, b):
    """Textbook Gamma-ratio norm h_n (valid off the Chebyshev pole line)."""
    return (special.gammaln(n + a + 1) + special.gammaln(n + b + 1)
            - special.gammaln(n + 1) - np.log(2 * n + a + b + 1)
            + (a + b + 1) * np.log(2.0) - special.gammaln(n + a + b + 1))


class TestJacobiPort(unittest.TestCase):
    PARAM_SETS = [(0.0, 0.0), (0.5, 0.5), (-0.5, -0.5), (2.0, -0.3), (3.0, 3.0)]

    def test_orthonormal_matches_scipy_eval_jacobi(self):
        """hat P_n(x) == P_n^{(a,b)}(x) / sqrt(h_n) for scipy's Szego convention."""
        x = np.linspace(-1 + 1e-6, 1 - 1e-6, 211)
        a = torch.tensor(0.0, dtype=torch.float64); b = torch.tensor(0.0, dtype=torch.float64)
        got = jacobi_orthonormal(torch.tensor(x, dtype=torch.float64), 8, a, b).numpy()
        for n in range(8):
            want = special.eval_jacobi(n, 0.0, 0.0, x) / np.sqrt(np.exp(classic_log_hn(n, 0.0, 0.0)))
            np.testing.assert_allclose(got[:, n], want, rtol=1e-7, atol=1e-8,
                                       err_msg=f"Legendre n={n}")

    def test_gram_is_identity(self):
        """Gauss-Jacobi quadrature: int hatP_n hatP_m w dx = delta_nm."""
        N, nq = 7, 300
        for (al, be) in self.PARAM_SETS:
            nodes, wj = special.roots_jacobi(nq, al, be)      # weights include (1-x)^a(1+x)^b
            a = torch.tensor(al, dtype=torch.float64); b = torch.tensor(be, dtype=torch.float64)
            P = jacobi_orthonormal(torch.tensor(nodes, dtype=torch.float64), N, a, b).numpy()
            G = P.T @ (P * wj[:, None])
            np.testing.assert_allclose(G, np.eye(N), atol=2e-6,
                                       err_msg=f"not orthonormal at alpha={al}, beta={be}")

    def test_pole_free_chebyshev_line(self):
        """alpha+beta+1 = 0 (Chebyshev 1st kind) is where textbook gammas die;
        the cancelled recurrence must not notice."""
        a = torch.tensor(-0.5, dtype=torch.float64); b = torch.tensor(-0.5, dtype=torch.float64)
        x = torch.linspace(-0.999, 0.999, 97, dtype=torch.float64)
        P = jacobi_orthonormal(x, 6, a, b)
        self.assertTrue(torch.isfinite(P).all())
        # on this line hat P_0 = 1/sqrt(mu0) = 1/sqrt(pi); hat P_1 is odd, so
        # hat P_1(x)/x is constant on the grid (checked below).
        self.assertAlmostEqual(float(P[:, 0].mean()), 1.0 / np.sqrt(np.pi), delta=1e-6)
        np.testing.assert_allclose(P[:, 1].numpy() / x.numpy(),
                                   float(P[40, 1] / x[40]), rtol=1e-10)   # linear: ratio const

    def test_gradients_finite_wrt_ab(self):
        a = torch.tensor(0.4, requires_grad=True); b = torch.tensor(-0.1, requires_grad=True)
        x = torch.linspace(-0.99, 0.99, 50, dtype=torch.float64)
        loss = jacobi_orthonormal(x, 6, a.double(), b.double()).square().sum()
        loss.backward()
        self.assertTrue(torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all())

    def test_mass_against_scipy(self):
        for al, be in self.PARAM_SETS:
            a = torch.tensor(al, dtype=torch.float64); b = torch.tensor(be, dtype=torch.float64)
            mu0 = float(jacobi_mass(a, b))
            # integral of w itself under plain quadrature weights would reuse wj; simpler:
            # mu0 closed form 2^(a+b+1) G(a+1)G(b+1)/G(a+b+2)
            want = 2 ** (al + be + 1) * np.exp(special.gammaln(al + 1) + special.gammaln(be + 1)
                                               - special.gammaln(al + be + 2))
            self.assertAlmostEqual(mu0, want, places=8)


class TestRawArms(unittest.TestCase):
    def test_hermite_matches_scipy(self):
        x = np.linspace(-3, 3, 101)
        got = hermite_physicist(torch.tensor(x, dtype=torch.float64), 6).numpy()
        for n in range(6):
            np.testing.assert_allclose(got[:, n], special.eval_hermite(n, x), rtol=1e-10)

    def test_chebyshev_matches_scipy(self):
        x = np.linspace(-3, 3, 101)     # deliberately outside [-1,1]: raw is raw
        got = chebyshev_t(torch.tensor(x, dtype=torch.float64), 6).numpy()
        for n in range(6):
            np.testing.assert_allclose(got[:, n], special.eval_chebyt(n, x), rtol=1e-10)

    def test_basis_table_dispatch(self):
        x = torch.linspace(-0.5, 0.5, 11, dtype=torch.float64)
        a = torch.tensor(0.2, dtype=torch.float64); b = torch.tensor(0.7, dtype=torch.float64)
        self.assertEqual(basis_table("jacobi", x, 5, a, b).shape, (11, 5))
        self.assertEqual(basis_table("hermite", x, 5).shape, (11, 5))
        with self.assertRaises(ValueError):
            basis_table("bernstein", x, 5)


class TestActivation(unittest.TestCase):
    def test_warmstart_identity_regime(self):
        from models import PolynomialActivation
        for kind, squash, target in [("jacobi", True, torch.tanh),
                                     ("hermite", False, lambda t: t),
                                     ("cheby", False, lambda t: t)]:
            act = PolynomialActivation(kind, 8, degree=4, squash=squash)
            x = torch.linspace(-3, 3, 601, dtype=torch.float32).unsqueeze(1).repeat(1, 8)
            with torch.no_grad():
                err = (act(x) - target(x)).abs().max()
            self.assertLess(float(err), 0.35, f"{kind}: warm start not near identity ({err})")

    def test_jacobi_alpha_admissible_after_walk(self):
        from models import PolynomialActivation
        act = PolynomialActivation("jacobi", 4, degree=4)
        with torch.no_grad():   # adversarial raw values toward the boundary
            act.raw_alpha.fill_(-20.0); act.raw_beta.fill_(-20.0)
        self.assertGreater(float(act.alpha), -1.0)
        self.assertGreater(float(act.alpha + act.beta), -2.0)
        _ = act(torch.randn(3, 4))     # must not raise


if __name__ == "__main__":
    unittest.main()
