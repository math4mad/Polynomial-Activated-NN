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


def act_pair(model: nn.Module) -> dict:
    """(alpha, beta) of every poly activation layer, read at this instant.

    Added for H6c's per-seed walks (`h6c_dumps.py --walks`): a LOG, not a knob.
    Nothing in the training loop consumes it, so the trajectory it describes is
    the registered exp8 trajectory — the walk is the same run, witnessed.
    ReLU arms have no exponents and report None.
    """
    acts = [model.act_in] + list(getattr(model, "mid_acts", []))
    a = [None if getattr(x, "alpha", None) is None else float(x.alpha) for x in acts]
    b = [None if getattr(x, "beta", None) is None else float(x.beta) for x in acts]
    return {"alpha": a[0], "beta": b[0], "alpha_layers": a, "beta_layers": b}


def make_model(arm: str, hidden: int, degree: int, depth: int = 1) -> tuple[nn.Module, dict]:
    """Build the arm; ReLU arm is width-MATCHED to the poly bill (demand 3)."""
    if arm == "relu":
        h_relu = matched_relu_width(hidden, degree, "jacobi", depth)  # jacobi billing is the tightest
        return MLP(h_relu, "relu", degree, depth), {"relu_width": h_relu}
    return MLP(hidden, ARM_ACT[arm], degree, depth), {}


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
        lr: float = 1e-3, batch: int = 64, verbose: bool = True,
        depth: int = 1, exp: str = "exp8", walk: bool = False) -> dict:
    dev = torch.device(device) if device != "auto" else \
        torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    torch.manual_seed(seed)                       # model init stream
    g_data = torch.Generator().manual_seed(SPLIT_SEED + seed)  # loader shuffle, per-run

    train_dl, val_dl, test_dl = loaders(batch)
    # re-shuffle with the run's own generator: data ORDER is per-seed, SPLIT is not
    train_dl.generator = g_data

    model, meta = make_model(arm, hidden, degree, depth)
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
            ep_loss += float(loss.detach()) * y.shape[0]
            ep_n += y.shape[0]
            steps += 1
        else:
            row = {"epoch": ep, "loss": ep_loss / max(ep_n, 1), "epoch_s": round(time.perf_counter() - te, 2)}
            if walk:
                row.update(act_pair(model))          # log after the epoch's last step
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
    acts = [model.act_in] + list(model.mid_acts)
    if hasattr(acts[0], "coeffs"):
        layers = []
        for a in acts:
            layers.append({
                "alpha": None if a.alpha is None else float(a.alpha),
                "beta": None if a.beta is None else float(a.beta),
                "coeffs_mean": a.coeffs.data.mean(0).tolist(),   # per-degree, over neurons
                "coeffs_std": a.coeffs.data.std(0).tolist(),
                "max_abs_act": float(a(torch.linspace(-5, 5, 401, device=dev).unsqueeze(1)
                                     .repeat(1, hidden)).abs().max()),
            })
        act_info = {**layers[0], "layers": layers}   # layer-0 keys kept for exp8 readers

    result = {
        "bench": "PolyNN",
        "experiment": exp,
        "regime": "learned-unbounded-poly-activation / from-scratch",
        "arm": arm, "hidden": hidden, "degree": degree, "seed": seed, "depth": depth,
        "epochs": epochs, "lr": lr, "batch": batch, "smoke": smoke,
        "scored_on": scored, "final_loss": loss, "final_acc": acc,
        "params": n_params(model), "poly_total_formula": poly_total(hidden, degree, arm if arm != "relu" else "jacobi", depth),
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
    tag = f"{result['arm']}_h{result['hidden']}_d{result['degree']}"
    tag += f"_l{result.get('depth', 1)}" if result.get("depth", 1) > 1 else ""
    tag += f"_s{result['seed']}"
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
    ap.add_argument("--walk", action="store_true",
                    help="log (alpha, beta) into every history row (H6c walks; a log, not a knob)")
    ap.add_argument("--depth", type=int, default=1, help="hidden layers (exp9)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    res = run(args.arm, args.hidden, args.degree, args.seed, args.epochs,
              args.smoke, args.device, args.lr, args.batch, depth=args.depth,
              walk=args.walk)
    p = save(res)
    print(f"[{args.arm} L={args.depth} h={args.hidden} s={args.seed}] acc={res['final_acc']:.4f} "
          f"params={res['params']} {res['total_s']}s -> {p.name}")
