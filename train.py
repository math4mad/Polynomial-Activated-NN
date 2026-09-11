"""
exp8 single-run trainer — protocol frozen in docs/PREREG.md §1:
Adam lr 1e-3, batch 64, CE, final-epoch model (no checkpoint selection),
5 paired seeds {1000..1004}. One command = one cell of the 40-run matrix.

    python train.py --arm jacobi --hidden 128 --seed 1000 --epochs 20   # recorded run
    python train.py --arm jacobi --hidden 128 --seed 1000 --smoke       # engineering

`--smoke` scores on the 5k train carve-out (val) and NEVER touches the test
set: engineering checks cannot contaminate the pre-registered comparison.
Recorded runs score the untouched canonical 10k test at the final epoch.

Results -> runs/<arm>_h<hidden>_d<degree>_s<seed>[_smoke].json (git-ignored);
wall-clock per epoch is recorded — the compute estimate becomes measured.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
import torch.nn as nn

from data import loaders, SPLIT_SEED
from models import MLP, n_params, matched_relu_width, poly_total

RUNS = Path(__file__).resolve().parent / "runs"
SEEDS = [1000, 1001, 1002, 1003, 1004]      # pre-registered seed list
ARM_ACT = {"relu": "relu", "jacobi": "jacobi", "hermite": "hermite", "cheby": "cheby"}


def make_model(arm: str, hidden: int, degree: int) -> tuple[nn.Module, dict]:
    """Build the arm; ReLU arm is width-MATCHED to the poly bill (demand 3)."""
    if arm == "relu":
        h_relu = matched_relu_width(hidden, degree, "jacobi")  # jacobi = +2 over hermite/cheby: tightest
        return MLP(h_relu, "relu"), {"relu_width": h_relu}
    return MLP(hidden, ARM_ACT[arm], degree), {}


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> tuple[float, float]:
    model.eval()
    tot_l, tot_n, correct = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        tot_l += float(nn.functional.cross_entropy(out, y, reduction="sum"))
        tot_n += y.numel()
        correct += int((out.argmax(1) == y).sum())
    return tot_l / tot_n, correct / tot_n


def run(arm: str, hidden: int = 128, degree: int = 4, seed: int = 1000,
        epochs: int = 20, smoke: bool = False, device: str = "auto",
        lr: float = 1e-3, batch: int = 64, verbose: bool = True) -> dict:
    dev = torch.device(device) if device != "auto" else \
        torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    torch.manual_seed(seed)                       # model init stream
    g_data = torch.Generator().manual_seed(SPLIT_SEED + seed)  # loader shuffle, per-run

    train_dl, val_dl, test_dl = loaders(batch)
    # re-shuffle with the run's own generator: data ORDER is per-seed, SPLIT is not
    train_dl.generator = g_data

    model, meta = make_model(arm, hidden, degree)
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()

    history, t0 = [], time.perf_counter()
    steps = 0
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss, ep_n = 0.0, 0
        te = time.perf_counter()
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev)
            loss = lossf(model(x), y)
            if not torch.isfinite(loss):
                history.append({"epoch": ep, "loss": float("nan"), "status": "diverged"})
                if verbose:
                    print(f"epoch {ep}: DIVERGED ({loss.item()})")
                break
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss) * y.shape[0]
            ep_n += y.shape[0]
            steps += 1
        else:
            row = {"epoch": ep, "loss": ep_loss / max(ep_n, 1), "epoch_s": round(time.perf_counter() - te, 2)}
            history.append(row)
            if verbose:
                print(f"epoch {ep:2d} | loss {row['loss']:.4f} | {row['epoch_s']}s")
            continue
        break  # diverged: leave the loop

    total_s = time.perf_counter() - t0
    if smoke:
        loss, acc = evaluate(model, val_dl, dev)
        scored = "val5k_train_carveout"
    else:
        loss, acc = evaluate(model, test_dl, dev)
        scored = "test10k_canonical"

    act_info = {}
    act = model.act
    if hasattr(act, "coeffs"):
        act_info = {
            "alpha": None if act.alpha is None else float(act.alpha),
            "beta": None if act.beta is None else float(act.beta),
            "coeffs_mean": act.coeffs.data.mean(0).tolist(),   # per-degree, averaged over neurons
            "coeffs_std": act.coeffs.data.std(0).tolist(),
            "max_abs_act": float(act(torch.linspace(-5, 5, 401, device=dev).unsqueeze(1)
                                      .repeat(1, hidden)).abs().max()),
        }

    result = {
        "bench": "PolyNN", "experiment": "exp8",
        "regime": "learned-unbounded-poly-activation / from-scratch",
        "arm": arm, "hidden": hidden, "degree": degree, "seed": seed,
        "epochs": epochs, "lr": lr, "batch": batch, "smoke": smoke,
        "scored_on": scored, "final_loss": loss, "final_acc": acc,
        "params": n_params(model), "poly_total_formula": poly_total(hidden, degree, arm if arm != "relu" else "jacobi"),
        **meta,
        "diverged": any(r.get("status") == "diverged" for r in history),
        "total_s": round(total_s, 1), "secs_per_epoch": round(total_s / max(len(history), 1), 2),
        "tflops_note": "fwd+bwd ~0.6 MFLOP/sample (Letter 004); measured = "
                       f"{(0.6e6 * 55000 * epochs / total_s / 1e12):.2f} TFLOP/s effective",
        "activation": act_info, "history": history,
        "env": {"device": str(dev), "torch": str(torch.__version__), "python": platform.python_version()},
    }
    return result


def save(result: dict, outdir: Path = RUNS) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{result['arm']}_h{result['hidden']}_d{result['degree']}_s{result['seed']}"
    tag += "_smoke" if result["smoke"] else ""
    path = outdir / f"{tag}.json"
    path.write_text(json.dumps(result, indent=1))
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=["relu", "jacobi", "hermite", "cheby"])
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--degree", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--smoke", action="store_true", help="score on val carve-out, never on test")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    res = run(args.arm, args.hidden, args.degree, args.seed, args.epochs,
              args.smoke, args.device, args.lr, args.batch)
    p = save(res)
    print(f"[{args.arm} h={args.hidden} s={args.seed}] acc={res['final_acc']:.4f} "
          f"params={res['params']} {res['total_s']}s -> {p.name}")
