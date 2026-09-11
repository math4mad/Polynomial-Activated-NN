"""
exp8 matrix runner + pre-registered hypothesis check (docs/PREREG.md).

    python compare.py --calibration          # first 20 runs (relu+jacobi), fix noise band
    python compare.py --phase remaining      # the other 20, uses the committed band
    python compare.py --report               # rebuild summary from runs/ only

Cells: 4 arms x widths {128, 256} x 5 paired seeds = 40 recorded runs.
Every cell lands in the table, diverged or not (negative results are
first-class). Test-set numbers appear ONLY here — recorded phase only.
Writes results/exp8_summary.{json,parquet} and prints the P-shape verdict.
"""

from __future__ import annotations

import argparse
import json
import statistics
from itertools import product
from pathlib import Path

import train as tr
from models import matched_relu_width, poly_total

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
WIDTHS = (128, 256)
ARMS = ("relu", "jacobi", "hermite", "cheby")
CALIB_ARMS = ("relu", "jacobi")
BAND_FILE = RESULTS / "noise_band.json"   # committed band, fixed after calibration


def cell(arm: str, hidden: int, seed: int, epochs: int = 20, device: str = "auto") -> dict:
    path = tr.RUNS / f"{arm}_h{hidden}_d4_s{seed}.json"
    if path.exists():                                  # idempotent: never rerun a cell
        return json.loads(path.read_text())
    res = tr.run(arm, hidden, 4, seed, epochs, smoke=False, device=device)
    tr.save(res)
    return res


def calibration(epochs: int, device: str) -> None:
    rows = [cell(a, h, s, epochs, device) for a, h, s in product(CALIB_ARMS, WIDTHS, tr.SEEDS)]
    # paired-seed sd of accuracy, within arm x width, over the 5 seeds
    sds = []
    for a, h in product(CALIB_ARMS, WIDTHS):
        accs = [r["final_acc"] for r in rows if r["arm"] == a and r["hidden"] == h]
        sds.append(statistics.stdev(accs) * 100)       # percentage points
    sd = max(sds)
    band = 2 * sd if sd > 0.15 else 0.3
    RESULTS.mkdir(exist_ok=True)
    BAND_FILE.write_text(json.dumps({
        "provisional_pp": 0.3, "observed_max_seed_sd_pp": round(sd, 4),
        "adopted_band_pp": round(band, 4), "rule": "band = 0.3pp unless sd>0.15pp then 2*sd",
        "fixed_before_remaining_runs": True,
    }, indent=1) + "\n")
    print(f"calibration done: max seed sd = {sd:.3f}pp -> claim band ±{band:.3f}pp -> {BAND_FILE}")


def pshape_verdict(rows: list[dict], band_pp: float) -> dict:
    """P-shape: jacobi >= hermite >= cheby in seed-paired mean test accuracy,
    per width, never averaged across widths. Killed if an adjacent ordering
    reverses beyond the band."""
    out = {}
    for h in WIDTHS:
        means = {}
        for arm in ("jacobi", "hermite", "cheby"):
            accs = [r["final_acc"] * 100 for r in rows if r["arm"] == arm and r["hidden"] == h]
            means[arm] = sum(accs) / len(accs) if accs else None
        checks = {}
        for lo, hi in (("hermite", "jacobi"), ("cheby", "hermite")):
            if means[lo] is None or means[hi] is None:
                checks[f"{hi}>={lo}"] = "pending"
            else:
                diff = means[hi] - means[lo]
                checks[f"{hi}>={lo}"] = ("hold" if diff >= -band_pp else
                                          "KILLED" if diff < -band_pp else "hold")
        out[f"h={h}"] = {"means_pct": {k: (round(v, 3) if v else None) for k, v in means.items()},
                          "checks": checks}
    return out


def report(device: str = "cpu") -> None:
    rows = []
    for p in sorted(tr.RUNS.glob("*.json")):
        if "_smoke" in p.name:
            continue
        r = json.loads(p.read_text())
        if r.get("depth", 1) != 1:     # exp9 cells belong to exp9's report, not this one
            continue
        rows.append(r)
    if not rows:
        print("no recorded runs yet"); return
    band = json.loads(BAND_FILE.read_text())["adopted_band_pp"] if BAND_FILE.exists() else None

    import polars as pl
    flat = [{
        "arm": r["arm"], "regime": r["regime"], "hidden": r["hidden"], "seed": r["seed"],
        "params": r["params"], "relu_width": r.get("relu_width", r["hidden"]),
        "acc_pct": round(r["final_acc"] * 100, 3), "diverged": r["diverged"],
        "secs_per_epoch": r["secs_per_epoch"], "alpha": (r["activation"] or {}).get("alpha"),
        "beta": (r["activation"] or {}).get("beta"), "smoke": r["smoke"],
    } for r in rows]
    df = pl.DataFrame(flat)
    RESULTS.mkdir(exist_ok=True)
    df.write_parquet(RESULTS / "exp8_summary.parquet")
    verdict = {
        "n_cells": len(flat) / 5, "n_runs": len(flat),
        "noise_band_pp": band,
        "billing": {f"{a}_h{h}": {"poly_total": poly_total(h, 4, a),
                                    "matched_relu": (None if a == "relu" else matched_relu_width(h, 4, "jacobi"))}
                    for a, h in product(ARMS, WIDTHS) if a != "relu"},
        "P-shape": pshape_verdict(rows, band or 0.3),
        "wall_clock": {"mean_secs_per_epoch": df["secs_per_epoch"].mean()},
    }
    (RESULTS / "exp8_summary.json").write_text(json.dumps(verdict, indent=1) + "\n")
    print(df.sort(["hidden", "arm", "seed"]))
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--calibration", action="store_true")
    g.add_argument("--phase", choices=["remaining"])
    g.add_argument("--report", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.report:
        report()
    elif args.calibration:
        calibration(args.epochs, args.device)
    else:
        rows = [cell(a, h, s, args.epochs, args.device)
                for a, h, s in product(ARMS, WIDTHS, tr.SEEDS)]
        print(f"{len(rows)} cells present; now: python compare.py --report")
