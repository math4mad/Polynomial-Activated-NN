"""
Models for exp8 — one hidden layer 784 -> h -> 10, four arms (docs/PREREG.md §2):

  A-relu     plain ReLU MLP, width matched to <= 1% of the poly arm's TOTAL params
  A-jacobi   phi(x) = <c, hatP^{(alpha,beta)}(tanh x)>, c per neuron, (alpha,beta) per layer
  A-hermite  phi(x) = <c, H_k(x)>          (raw, no squash, no rescue)
  A-cheby    phi(x) = <c, T_k(x)>          (raw, no squash, no rescue)

Coefficient budget is billed as architecture: d+1 params per neuron (regime law).
For h in {128, 256} and d = 4 the SAME width satisfies the matching rule:
  poly total = 784h + h + (d+1)h + 10h + 10 (+2 jacobi) = 800h + 12
  relu total(h) = 795h + 10   ->   |diff| / poly = 0.63% < 1%.
`matched_relu_width()` keeps this honest for any (h, d) and refuses to bill silently.

Warm start: coefficients initialized so phi ~ identity (squash: tanh(x),
raw: x) by least squares on [-3, 3] — every arm starts in ReLU's near-linear
regime; the README's promise is that the container then bends usefully.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from basis import basis_table

KINDS = ("jacobi", "hermite", "cheby")


class PolynomialActivation(nn.Module):
    """Learnable truncated orthogonal-poly activation, one coefficient vector
    per neuron, optional tanh squashing onto the basis' finite domain.

    Args:
        kind:   "jacobi" | "hermite" | "cheby"
        dim:    number of neurons (coefficient vectors) in the layer
        degree: polynomial degree d  ->  d+1 coefficients per neuron
        squash: tanh-squash the pre-activation before the basis (jacobi arm: True,
                raw arms: False — pre-registered, no normalization rescue)
    """

    def __init__(self, kind: str, dim: int, degree: int = 4, squash: bool = True,
                 grid: int = 2001, halfwidth: float = 3.0):
        super().__init__()
        assert kind in KINDS, kind
        self.kind, self.dim, self.degree, self.squash = kind, dim, degree, squash

        if kind == "jacobi":
            # alpha = softplus(raw) - 0.5 keeps alpha > -0.5 > -1 for all raw;
            # raw_0 = log(expm1(0.5)) initializes Legendre: alpha = beta = 0.
            raw0 = math.log(math.expm1(0.5))
            self.raw_alpha = nn.Parameter(torch.tensor(raw0))
            self.raw_beta = nn.Parameter(torch.tensor(raw0))
        else:
            self.raw_alpha = self.raw_beta = None

        self.coeffs = nn.Parameter(torch.zeros(dim, degree + 1))
        self._warm_start(grid, halfwidth)

    # -- parameter views ------------------------------------------------------ #
    @property
    def alpha(self) -> torch.Tensor | None:
        return None if self.raw_alpha is None else torch.nn.functional.softplus(self.raw_alpha) - 0.5

    @property
    def beta(self) -> torch.Tensor | None:
        return None if self.raw_beta is None else torch.nn.functional.softplus(self.raw_beta) - 0.5

    # -- init ----------------------------------------------------------------- #
    @torch.no_grad()
    def _warm_start(self, grid: int, halfwidth: float) -> None:
        x = torch.linspace(-halfwidth, halfwidth, grid, dtype=torch.float64)
        u = torch.tanh(x) if self.squash else x
        a = self.alpha.double() if self.alpha is not None else None
        b = self.beta.double() if self.beta is not None else None
        B = basis_table(self.kind, u, self.degree + 1, a, b)          # (G, d+1)
        target = torch.tanh(x) if self.squash else x                  # identity start
        c, _, _, _ = torch.linalg.lstsq(B, target.unsqueeze(1))       # (d+1, 1)
        self.coeffs.copy_(c.reshape(1, -1).to(self.coeffs.dtype))     # shared across neurons

    # -- forward -------------------------------------------------------------- #
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, dim) pre-activations -> (B, dim) activations."""
        u = torch.tanh(z) if self.squash else z
        a = self.alpha if self.alpha is not None else None
        b = self.beta if self.beta is not None else None
        flat = u.reshape(-1)
        table = basis_table(self.kind, flat, self.degree + 1, a, b)   # (B*dim, d+1)
        table = table.view(*u.shape, self.degree + 1)
        return (table * self.coeffs).sum(-1)

    def extra_repr(self) -> str:
        ab = "" if self.alpha is None else f", alpha={float(self.alpha):.3f}, beta={float(self.beta):.3f}"
        return f"kind={self.kind}, dim={self.dim}, degree={self.degree}, squash={self.squash}{ab}"


class MLP(nn.Module):
    """784 -> [h -> h] x (depth-1) -> h -> 10 plain stack; activation = ReLU or poly arm.

    depth = number of hidden layers (no BN, no residual — exp9 pre-reg §1).
    """

    def __init__(self, hidden: int = 128, activation: str = "relu", degree: int = 4,
                 depth: int = 1):
        super().__init__()
        assert depth >= 1
        self.depth = depth
        self.fc_in = nn.Linear(784, hidden)
        self.act_in: nn.Module = _make_act(activation, hidden, degree)
        self.mid = nn.ModuleList()
        self.mid_acts = nn.ModuleList()
        for _ in range(depth - 1):
            self.mid.append(nn.Linear(hidden, hidden))
            self.mid_acts.append(_make_act(activation, hidden, degree))
        self.fc_out = nn.Linear(hidden, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.act_in(self.fc_in(x))
        for lin, act in zip(self.mid, self.mid_acts):
            z = act(lin(z))
        return self.fc_out(z)


def _make_act(activation: str, hidden: int, degree: int) -> nn.Module:
    if activation == "relu":
        return nn.ReLU()
    if activation in KINDS:
        return PolynomialActivation(activation, hidden, degree,
                                    squash=(activation == "jacobi"))
    raise ValueError(activation)


# --------------------------------------------------------------------------- #
# parameter billing (pre-registered demand 3: equal TOTAL parameters)
# --------------------------------------------------------------------------- #
def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def relu_total(hidden: int, depth: int = 1) -> int:
    """784h + h + (depth-1)(h^2 + h) + 10h + 10 (biases on every Linear)."""
    return 784 * hidden + hidden + max(depth - 1, 0) * (hidden * hidden + hidden) + 10 * hidden + 10


def poly_total(hidden: int, degree: int, activation: str, depth: int = 1) -> int:
    per_neuron = (degree + 1) * depth if activation in KINDS else 0
    extra = 2 * depth if activation == "jacobi" else 0     # (alpha, beta) per layer
    return relu_total(hidden, depth) + per_neuron * hidden + extra


def matched_relu_width(hidden: int, degree: int, activation: str, depth: int = 1,
                       grid=None) -> int:
    """ReLU width whose total params is closest to the poly arm's; must be <= 1%
    or the comparison is refused — the bill may not be cooked silently."""
    if grid is None:   # step-2 grid: at L>=4 the coarse {64..512} grid cannot land inside 1%
        grid = tuple(range(64, 513, 2))
    target = poly_total(hidden, degree, activation, depth)
    best = min(grid, key=lambda h: abs(relu_total(h, depth) - target))
    rel = abs(relu_total(best, depth) - target) / target
    if rel > 0.01:
        raise ValueError(f"no ReLU width within 1% of {activation} h={hidden} d={degree} "
                         f"L={depth} (closest {best} at {rel:.2%}) — pre-register a new grid, do not fudge")
    return best
