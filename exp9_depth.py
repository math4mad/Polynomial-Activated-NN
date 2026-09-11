"""
exp9 runner — depth ablation, protocol in docs/PREREG-exp9.md (committed first).

    python exp9_depth.py --calibration     # six L=1 cells -> exp9_noise_band.json
    python exp9_depth.py --deep            # L in {2,3,4,6,8} x 2 arms x 3 seeds
    python exp9_depth.py --report          # verdicts: P-degrade / P-tide / P-nocode

Widths: for each depth L the jacobi arm's h is chosen so total params ~= P0 =
150,000 (+-3%); the ReLU arm is width-matched within 1% at the SAME depth
(models.matched_relu_width refuses the cell otherwise). Test numbers only in
the recorded phase.
"""

from __future__ import annotations

import argparse
import json
import statistics
from itertools import product
from pathlib import Path

import train as tr
from models import matched_relu_width, poly_total, relu_total

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
BAND_FILE = RESULTS / "exp9_noise_band.json"
P0 = 150_000
DEPTHS = (1, 2, 3, 4, 6, 8)
ARMS = ("relu", "jacobi")
SEEDS = (1000, 1001, 1002)


def jacobi_width(depth: int, degree: int = 4) -> int:
    """Even grid; minimize |poly_total - P0|; must stay within +-3% or abort."""
    best = min(range(64, 513, 2), key=lambda h: abs(poly_total(h, degree, "jacobi", depth) - P0))
    rel = abs(poly_total(best, degree, "jacobi", depth) - P0) / P0
    if rel > 0.03:
        raise ValueError(f"L={depth}: no width within 3% of P0 (h={best} off {rel:.1%})")
    return best


def plan() -> list[dict]:
    out = []
    for L, arm in product(DEPTHS, ARMS):
        h_j = jacobi_width(L)
        h = h_j if arm == "jacobi" else matched_relu_width(h_j, 4, "jacobi", L)
        out.append({"depth": L, "arm": arm, "hidden": h,
                    "params": poly_total(h_j, 4, "jacobi", L) if arm == "jacobi" else relu_total(h, L)})
    return out


def cell(arm: str, depth: int, hidden: int, seed: int, epochs: int, device: str) -> dict:
    tag = f"{arm}_h{hidden}_d4" + (f"_l{depth}" if depth > 1 else "") + f"_s{seed}"
    path = tr.RUNS / f"{tag}.json"
    if path.exists():
        return json.loads(path.read_text())
    res = tr.run(arm, hidden, 4, seed, epochs, smoke=False, device=device, depth=depth,
                 exp="exp9")
    tr.save(res)
    return res


def calibration(epochs: int, device: str) -> None:
    rows = [cell(a, 1, jacobi_width(1) if a == "jacobi" else
                 matched_relu_width(jacobi_width(1), 4, "jacobi", 1), s, epochs, device)
            for a, s in product(ARMS, SEEDS)]
    sds = []
    for a in ARMS:
        acc = [r["final_acc"] * 100 for r in rows if r["arm"] == a]
        sds.append(statistics.stdev(acc))
    band = max(0.3, 2 * max(sds) * 100)
    RESULTS.mkdir(exist_ok=True)
    BAND_FILE.write_text(json.dumps({
        "calibration_cells": "L=1 x 2 arms x 3 seeds", "max_seed_sd_pp": round(max(sds) * 100, 4),
        "adopted_band_pp": round(band, 4), "fixed_before_deep_cells": True,
    }, indent=1) + "\n")
    print(f"L={jacobi_width(1)}(j)/{matched_relu_width(jacobi_width(1),4,'jacobi',1)}(r) calibration: "
          f"max sd {max(sds)*100:.3f}pp -> band ±{band:.3f}pp")


def deep(epochs: int, device: str) -> None:
    if not BAND_FILE.exists():
        raise SystemExit("run --calibration first (PREREG-exp9 §3): band before deep cells")
    for L, arm, seed in product((2, 3, 4, 6, 8), ARMS, SEEDS):
        spec = next(p for p in plan() if p["depth"] == L and p["arm"] == arm)
        cell(arm, L, spec["hidden"], seed, epochs, device)
    print("30 deep cells present; now: python exp9_depth.py --report")


def report() -> None:
    rows = []
    for p in sorted(tr.RUNS.glob("*.json")):
        if "_smoke" in p.name:
            continue
        r = json.loads(p.read_text())
        if r.get("experiment") == "exp9" and not r["smoke"]:
            rows.append(r)
    band = json.loads(BAND_FILE.read_text())["adopted_band_pp"]
    means: dict[tuple, float] = {}
    sds: dict[tuple, float] = {}
    for L, arm in product(DEPTHS, ARMS):
        acc = [r["final_acc"] * 100 for r in rows if r["depth"] == L and r["arm"] == arm]
        if acc:
            means[(L, arm)] = sum(acc) / len(acc)
            sds[(L, arm)] = statistics.stdev(acc) if len(acc) > 1 else float("nan")

    degrade_at = [L for L in DEPTHS[1:]
                  if (L, "relu") in means and (1, "relu") in means
                  and means[(1, "relu")] - means[(L, "relu")] > band]
    tide_at = [L for L in DEPTHS[1:]
               if (L, "jacobi") in means and (L, "relu") in means
               and means[(L, "jacobi")] - means[(L, "relu")] > band]
    nocode_survives = not tide_at
    out = {
        "band_pp": band, "P0": P0,
        "means_pct": {f"L{L}_{arm}": round(v, 3) for (L, arm), v in means.items()},
        "seed_sds_pp": {f"L{L}_{arm}": round(v, 3) for (L, arm), v in sds.items()},
        "widths_params": plan(),
        "P-degrade": {"L*_exists": bool(degrade_at), "first_L": degrade_at[0] if degrade_at else None,
                       "all_beyond_band": degrade_at},
        "P-tide(力挽狂澜)": {"survives": bool(tide_at), "at_depths": tide_at},
        "P-nocode(counter-prediction)": {"survives": nocode_survives},
        "P-shape-port(bounded sd <= relu sd at each L)":
            {f"L{L}": (sds.get((L, 'jacobi'), float('nan')) <= sds.get((L, 'relu'), float('inf')))
             for L in DEPTHS},
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "exp9_summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true")
    g.add_argument("--calibration", action="store_true")
    g.add_argument("--deep", action="store_true")
    g.add_argument("--report", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)   # pre-registered §1
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    if args.plan:
        print(json.dumps(plan(), indent=1))
    elif args.calibration:
        calibration(args.epochs, args.device)
    elif args.deep:
        deep(args.epochs, args.device)
    else:
        report()
