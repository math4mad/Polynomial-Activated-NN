"""
exp8 visualisation + the GPU-free spectral row (P-relu-tail, PREREG §3).

    python visualize.py                 # from runs/*.json + figures/ outputs
    python visualize.py --tails-only    # analytic row only (fast, no runs needed)

Produces:
  figures/activations.png   learned phi(x) on [-3,3] per arm (mean coeffs) + ReLU ghost
  figures/basis_shapes.png  the three bases at their init (alpha,beta)=(0,0), d=4
  figures/curves.png        loss histories per arm (h=128, all seeds)
  results/relu_tail.json    min-type error E_d(ReLU|[1-3,3]) of best degree-d poly,
                            and the SAME quantity for a trained activation:
                            every accuracy cell's "retained energy" companion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from basis import basis_table
from models import PolynomialActivation

ROOT = Path(__file__).resolve().parent
FIGS = ROOT / "figures"
RESULTS = ROOT / "results"


# --------------------------------------------------------------------------- #
# P-relu-tail: how non-bandlimited is ReLU? E_d ~ 0.28/d (classical |x| asympotics)
# --------------------------------------------------------------------------- #
def relu_tail(dmax: int = 16, npts: int = 20001) -> dict:
    x = np.linspace(-3.0, 3.0, npts)
    f = np.maximum(x, 0.0)
    xs = x / 3.0                                   # map to [-1,1] for chebfit
    out = {"relu_abs_interval": 3.0, "rows": []}
    for d in range(1, dmax + 1):
        c = np.polynomial.chebyshev.chebfit(xs, f, d)
        err = np.abs(np.polynomial.chebyshev.chebval(xs, c) - f)
        rel = float(err.max() / np.abs(f).max())
        out["rows"].append({"degree": d, "max_abs_err_pp": float(err.max()),
                            "rel_err": rel, "predicted_0p28_over_d": 0.28 / d,
                            "ratio_to_prediction": rel / (0.28 / d)})
    return out


# --------------------------------------------------------------------------- #
# activation shapes from recorded runs
# --------------------------------------------------------------------------- #
def learned_shapes() -> dict:
    """Rebuild phi(x) on [-3,3] per arm from runs/*.json (mean coeffs + alpha,beta)."""
    shapes, seen = {}, set()
    for p in sorted((ROOT / "runs").glob("*.json")) if (ROOT / "runs").is_dir() else []:
        r = json.loads(p.read_text())
        if r["smoke"] or not r.get("activation"):
            continue
        key = (r["arm"], r["hidden"])
        if key in seen:
            continue
        seen.add(key)
        act_info = r["activation"]
        act = PolynomialActivation("jacobi" if r["arm"] == "jacobi" else r["arm"],
                                   1, degree=r["degree"],
                                   squash=(r["arm"] == "jacobi"))
        if act_info.get("alpha") is not None:
            with torch.no_grad():   # invert softplus-0.5 to place learned shape
                ra = torch.log(torch.expm1(torch.tensor(act_info["alpha"] + 0.5)))
                rb = torch.log(torch.expm1(torch.tensor(act_info["beta"] + 0.5)))
                act.raw_alpha.copy_(ra); act.raw_beta.copy_(rb)
        with torch.no_grad():
            act.coeffs.copy_(torch.tensor(act_info["coeffs_mean"], dtype=torch.float32).unsqueeze(0))
        x = torch.linspace(-3, 3, 601)
        shapes[f"{r['arm']}_h{r['hidden']}"] = {
            "x": x.tolist(), "phi": act(x.unsqueeze(1)).detach().numpy()[:, 0].tolist(),
            "alpha": act_info.get("alpha"), "beta": act_info.get("beta"),
            "coeffs_mean": act_info["coeffs_mean"], "coeffs_std": act_info["coeffs_std"],
        }
    return shapes


def plots(shapes: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGS.mkdir(exist_ok=True)
    made = []

    x = torch.linspace(-3, 3, 601)
    plt.figure(figsize=(7, 5))
    plt.axhline(0, color="k", lw=0.5)
    plt.plot(x.numpy(), np.maximum(x.numpy(), 0), "k--", lw=1, label="ReLU (ghost)")
    if shapes:
        for name, s in shapes.items():
            plt.plot(s["x"], s["phi"], lw=1.5, label=name)
    plt.legend(fontsize=8); plt.xlabel("pre-activation x"); plt.ylabel("phi(x)")
    plt.title("learned activations on [-3,3] — where the story is (Next §2c)")
    plt.savefig(FIGS / "activations.png", dpi=140); plt.close(); made.append("activations.png")

    plt.figure(figsize=(7, 5))
    a = torch.tensor(0.0, dtype=torch.float64); b = torch.tensor(0.0, dtype=torch.float64)
    uu = torch.linspace(-1, 1, 400, dtype=torch.float64)
    for k, (kind, xx) in enumerate([("jacobi(Legendre,alpha=beta=0)", uu),
                                    ("hermite(tanh-u)", torch.tanh(torch.linspace(-3, 3, 400, dtype=torch.float64))),
                                    ("cheby(tanh-u)", torch.tanh(torch.linspace(-3, 3, 400, dtype=torch.float64)))]):
        T = basis_table("jacobi" if k == 0 else kind.split("(")[0], xx, 5, a, b)
        for n in range(5):
            plt.plot(xx.numpy(), T[:, n].numpy(), lw=1, label=f"{kind} n={n}" if n < 2 else "")
    plt.legend(fontsize=6, ncol=3); plt.title("degree<=4 basis atoms at init")
    plt.savefig(FIGS / "basis_shapes.png", dpi=140); plt.close(); made.append("basis_shapes.png")

    hist = {}
    for p in sorted((ROOT / "runs").glob("*_h128_*.json")) if (ROOT / "runs").is_dir() else []:
        r = json.loads(p.read_text())
        if r["smoke"]:      # engineering runs must not bend the recorded curves
            continue
        hist.setdefault(r["arm"], []).append(r["history"])
    if hist:
        plt.figure(figsize=(7, 4))
        for arm, hs in hist.items():
            m = np.mean([[row.get("loss", np.nan) for row in h] for h in hs], axis=0)
            plt.plot(range(1, len(m) + 1), m, label=f"{arm} (seeds mean)")
        plt.legend(fontsize=8); plt.xlabel("epoch"); plt.ylabel("train loss")
        plt.savefig(FIGS / "curves.png", dpi=140); plt.close(); made.append("curves.png")
    return made


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tails-only", action="store_true")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    tails = relu_tail()
    (RESULTS / "relu_tail.json").write_text(json.dumps(tails, indent=1) + "\n")
    print("P-relu-tail (rel. best-degree-d error of ReLU on [-3,3], vs ~0.28/d):")
    for row in tails["rows"]:
        if row["degree"] in (2, 4, 6, 8, 12, 16):
            print(f"  d={row['degree']:2d}  rel_err={row['rel_err']:.4f}  0.28/d={row['predicted_0p28_over_d']:.4f}"
                  f"  ratio={row['ratio_to_prediction']:.2f}")
    if not args.tails_only:
        made = plots(learned_shapes())
        print("figures:", made or "(no recorded runs yet)")
