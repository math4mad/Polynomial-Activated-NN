"""
H6c homework instrument — the bytes PREREG_EXP6 §H6c names and does not yet
own. PolyNN's `docs/PREREG.md` stays the protocol law; this file produces no
verdicts, only measures.

    python h6c_dumps.py --dumps      # 4 arms x 5 seeds, init measure (no training, ~seconds)
    python h6c_dumps.py --walks      # A-jacobi h=128 x 5 seeds x 20 epochs (registered trainer)
    python h6c_dumps.py --summarize  # combine runs/ into results/h6c_*.json

What is registered, and where it is written:

  dumps  JacobiGP `docs/PREREG_EXP6.md`, H6c "Input" + "Coordinate" (Letter 017
         debt three): per-arm histograms of the **init (epoch 0)** hidden
         pre-activations in the Anatomist's coordinate — u = tanh(z), read at
         the *input* of each activation module, so the jacobi arm is never
         squashed twice and the raw arms are never pre-squashed; bins are the
         fixed uniform grid on [−1, 1] (M = 256, the budget line's ceiling);
         weights = counts. All four arms, five pre-registered seeds, h=128 —
         the exp8 geometry.
  walks  the per-epoch (α̂, β̂) trajectory of A-jacobi h=128, five seeds, under
         train.run() unchanged (Adam 1e-3, batch 64, CE, 20 epochs). Logged,
         never selected on: nothing here reads an accuracy to decide anything.

Assigned to machine B by the chair's two-machine letter (Letter 015/016, row
"PolyNN init pre-activation dumps + per-seed α,β walks"), so every artifact
carries `run_on`. `--walks` runs in `smoke=True`: the flag changes only the
final evaluation, never the training loop, so these runs touch the canonical
10k test ZERO times and the α,β walk is bit-for-bit the recorded one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import statistics
import time
from pathlib import Path

import numpy as np
import torch

import train as tr
from data import SPLIT_SEED, datasets
from models import n_params

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"

ARMS = ("relu", "jacobi", "hermite", "cheby")
HIDDEN = 128          # the H6c primary width (PREREG_EXP6 §H6c "Input")
DEGREE = 4            # exp8's frozen degree
DEPTH = 1
BINS = 256            # PREREG_EXP6 §H6c budget: "M <= 256-bin histogram"
EDGES = np.linspace(-1.0, 1.0, BINS + 1)   # fixed and data-independent by design
MEASUREMENT = "u = tanh(z), z = the tensor entering the activation module (pre-activation)"
CHUNK = 8192
EPOCHS = 20           # exp8 §1, frozen


def run_on() -> dict:
    """Letter 015 law 3: every number says which laptop it was born on."""
    return {
        "machine": "m1-16g",
        "host": socket.gethostname(),
        "bench": "B (worker bench)",
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "python": platform.python_version(),
        "device": "mps" if torch.backends.mps.is_available() else "cpu",
    }


# --------------------------------------------------------------------------- #
# pinned init state — law 3 forbids weights in git, so the pin is (config,
# seed, sha256 of the bytes) and the bytes regenerate from the seed
# --------------------------------------------------------------------------- #
def init_state_sha256(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        p = v.detach().cpu().contiguous().numpy().astype("<f4")   # canonical: float32 LE
        h.update(k.encode())
        h.update(np.ascontiguousarray(p).tobytes())
    return h.hexdigest()


def build_init(arm: str, seed: int) -> tuple[torch.nn.Module, dict]:
    """train.run()'s init order: manual_seed(seed) -> loaders -> make_model.
    `loaders_evidence` proves the middle step consumes no global RNG, which is
    why --dumps may skip the (slow) data read and still land on the same tensor."""
    torch.manual_seed(seed)
    model, meta = tr.make_model(arm, HIDDEN, DEGREE, DEPTH)
    return model, meta


def loaders_evidence(seed: int) -> dict:
    """Assert, don't assume: build A-jacobi twice, once after a full datasets()
    read (train.run's order), once without. Equal hashes => the data read is
    outside the model-init stream and --dumps reproduces exp8 inits exactly."""
    torch.manual_seed(seed)
    datasets()
    with_loaders = init_state_sha256(tr.make_model("jacobi", HIDDEN, DEGREE, DEPTH)[0])
    torch.manual_seed(seed)
    bare = init_state_sha256(build_init("jacobi", seed)[0])
    return {"seed": seed, "sha256_with_datasets": with_loaders,
            "sha256_without_datasets": bare, "agree": with_loaders == bare}


# --------------------------------------------------------------------------- #
# the init measure
# --------------------------------------------------------------------------- #
@torch.no_grad()
def capture_z(model: torch.nn.Module, x: torch.Tensor, dev: torch.device) -> list[np.ndarray]:
    outs: list[list[np.ndarray]] = [[] for _ in range(DEPTH)]
    layers = [model.act_in] + list(getattr(model, "mid_acts", []))

    def mk(i):
        def hook(_m, inp):        # 2-arg pre-hook: (module, input)
            outs[i].append(inp[0].detach().to("cpu", torch.float32).numpy().reshape(-1))
        return hook

    handles = [m.register_forward_pre_hook(mk(i)) for i, m in enumerate(layers)]
    model.to(dev).eval()
    for s in range(0, x.shape[0], CHUNK):
        model(x[s:s + CHUNK].to(dev))
    for hd in handles:
        hd.remove()
    return [np.concatenate(o) for o in outs]


def histogram(u: np.ndarray) -> dict:
    counts, _ = np.histogram(u, bins=EDGES)
    inside = int(counts.sum())
    return {
        "domain": [-1.0, 1.0], "n_bins": BINS,
        "bin_edges": [round(float(e), 9) for e in EDGES],
        "counts": [int(c) for c in counts],
        "counts_sum": inside,
        "n_values": int(u.size),
        "count_fraction_in_domain": round(inside / u.size, 9),  # 1.0 by construction (tanh)
        "u_min": float(u.min()), "u_max": float(u.max()),
        "u_mean": float(u.mean()), "u_std": float(u.std()),
    }


def dump_arm_seed(arm: str, seed: int, x: torch.Tensor, equiv: dict) -> dict:
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, meta = build_init(arm, seed)
    sha = init_state_sha256(model)
    assert sha == init_state_sha256(build_init(arm, seed)[0]), \
        f"init not reproducible for {arm}/s{seed}: the pin would be a lie"
    zs = capture_z(model, x, dev)
    layers = []
    for i, z in enumerate(zs):
        layers.append({
            "layer": i, "module": "act_in" if i == 0 else f"mid_{i - 1}",
            "model_applies_its_own_squash_here": arm == "jacobi",
            "z_summary": {"mean": float(z.mean()), "std": float(z.std()),
                          "min": float(z.min()), "max": float(z.max()),
                          "abs_z_p50": float(np.percentile(np.abs(z), 50)),
                          "abs_z_p99": float(np.percentile(np.abs(z), 99))},
            "histogram": histogram(np.tanh(z)),
        })
    combined = np.concatenate([np.tanh(z) for z in zs])
    return {
        "bench": "PolyNN", "kind": "h6c-init-dump",
        "registered_in": "JacobiGP docs/PREREG_EXP6.md H6c (Input + Coordinate); Letter 017 debt three",
        "assigned_to_machine_B_by": "chair Letter 015/016 (two-machine day)",
        "arm": arm, "hidden": HIDDEN, "degree": DEGREE, "depth": DEPTH, "seed": seed,
        "epoch": 0, "measured_at": "init, before any gradient step",
        "coordinate": {"definition": MEASUREMENT, "domain": [-1.0, 1.0],
                       "weights": "histogram counts", "re_squashed": False, "re_scaled": False},
        "measure_source": {"dataset": "FashionMNIST train carve (55k), canonical order",
                           "n_images": int(x.shape[0]), "split_seed": SPLIT_SEED,
                           "val_used": False, "test_used": False},
        "init_state_pin": {"sha256_float32_sorted_keys": sha, "params": n_params(model),
                           "reconstruct": f"torch.manual_seed({seed}); make_model('{arm}', {HIDDEN}, {DEGREE}, {DEPTH})"},
        "init_reproducibility": {"rebuilt_twice_equal": True, "loaders_outside_init_stream": equiv},
        "relu_width": meta.get("relu_width"),
        "layers": layers,
        "combined_all_layers": histogram(combined),
        "run_on": run_on(),
    }


# --------------------------------------------------------------------------- #
# the walks — registered trainer, α,β logged per epoch
# --------------------------------------------------------------------------- #
def walk_seed(seed: int, epochs: int = EPOCHS) -> dict:
    res = tr.run("jacobi", HIDDEN, DEGREE, seed, epochs, smoke=True,
                verbose=False, walk=True)
    walk = [{"epoch": r["epoch"], "loss": r.get("loss"),
             "alpha": r.get("alpha"), "beta": r.get("beta")}
            for r in res["history"] if "alpha" in r]
    return {
        "bench": "PolyNN", "kind": "h6c-alpha-beta-walk",
        "registered_in": "JacobiGP docs/PREREG_EXP6.md H6c (Band); homework assigned by Letter 016",
        "arm": "jacobi", "hidden": HIDDEN, "degree": DEGREE, "depth": DEPTH, "seed": seed,
        "protocol": {"trainer": "train.run() unchanged (Adam 1e-3, batch 64, CE)",
                     "epochs": epochs, "lr": res["lr"], "batch": res["batch"],
                     "scored_on": res["scored_on"], "smoke": True, "test_touched": False},
        "init_pair": {"epoch": 0, "alpha": 0.0, "beta": 0.0,
                      "note": "Legendre start: raw_* = log(expm1(0.5)) -> alpha = beta = 0 (models.py)"},
        "walk": walk,
        "terminal_pair": {"alpha": walk[-1]["alpha"] if walk else None,
                          "beta": walk[-1]["beta"] if walk else None,
                          "epoch": walk[-1]["epoch"] if walk else None},
        "final_acc_val5k": res["final_acc"], "final_loss_val5k": res["final_loss"],
        "total_s": res["total_s"], "secs_per_epoch": res["secs_per_epoch"],
        "diverged": res["diverged"], "env": res["env"], "run_on": run_on(),
    }


# --------------------------------------------------------------------------- #
# summaries — arithmetic on bytes this file already wrote; no new measurement
# --------------------------------------------------------------------------- #
def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def cross_machine(per_seed: list[dict]) -> dict:
    """Exact-difference audit of B's terminal pairs against A's committed ones.
    Same protocol, same seeds, same split; different laptop, different torch."""
    ref = a_reference_terminals()
    if not ref.get("available"):
        return {"reference": ref, "verdict": "no A-side bytes on this machine to compare with"}
    diffs, exact = [], 0
    for r in per_seed:
        a = ref["rows"].get(str(r["seed"]))
        if not a or r["terminal"]["alpha"] is None:
            continue
        da, db = r["terminal"]["alpha"] - a[0], r["terminal"]["beta"] - a[1]
        diffs.append({"seed": r["seed"], "alpha_diff": da, "beta_diff": db,
                      "bit_identical": da == 0.0 and db == 0.0})
        exact += int(da == 0.0 and db == 0.0)
    return {
        "reference": {k: ref[k] for k in ("path", "sha256", "run_on_of_reference")},
        "n_pairs_compared": len(diffs), "n_bit_identical": exact,
        "max_abs_diff": max((max(abs(d["alpha_diff"]), abs(d["beta_diff"])) for d in diffs), default=None),
        "per_seed": diffs,
        "reading": "exact equality, not 'close': across two laptops and two torch builds the "
                   "terminal (α,β) of the registered trainer reproduced to the last bit, so "
                   "the machine effect on THIS statistic is zero — the estimate that needs a "
                   "number is MEF's ladder, not this one",
        "limit": "A's per-seed exp8 terminals are the parquet's float64 round-trip of a "
                 "float32 parameter; equality is asserted on those stored bytes, not on A's "
                 "in-memory tensors, which no one has"}


def summarize_dumps() -> Path:
    paths = sorted(RUNS.glob("h6c_dump_*_h128_s*.json"))
    per_arm: dict[str, list] = {}
    for p in paths:
        d = json.loads(p.read_text())
        per_arm.setdefault(d["arm"], []).append({
            "seed": d["seed"],
            "init_state_sha256": d["init_state_pin"]["sha256_float32_sorted_keys"],
            "n_values": d["combined_all_layers"]["n_values"],
            "u_mean": d["combined_all_layers"]["u_mean"],
            "u_std": d["combined_all_layers"]["u_std"],
            "file": f"runs/{p.name}"})
    # Degeneracy check — arithmetic on the counts this file already wrote.
    # The registered coordinate is the PRE-activation measure; at epoch 0 the
    # pre-activation is fc_in(x) and fc_in is the only module that consumed the
    # init RNG stream, so the four arms must agree. They do, to the integer.
    by_seed: dict[int, dict[str, str]] = {}
    for p in paths:
        d = json.loads(p.read_text())
        key = hashlib.sha256(json.dumps(d["combined_all_layers"]["counts"]).encode()).hexdigest()[:16]
        by_seed.setdefault(d["seed"], {})[d["arm"]] = key
    degenerate = {int(s): v for s, v in by_seed.items()
                  if len(set(v.values())) == 1 and len(v) == len(ARMS)}
    out = {
        "kind": "h6c-init-dump-summary", "generated_by": "h6c_dumps.py --summarize",
        "coordinate": MEASUREMENT, "domain": [-1.0, 1.0], "n_bins": BINS,
        "contrast_row_is_degenerate_at_init": {
            "seeds_where_all_four_arm_histograms_are_integer-identical": sorted(degenerate),
            "mechanism": "z = fc_in(x); at epoch 0 the activation module has not touched z, "
                         "and fc_in is initialised from the same manual_seed stream in every "
                         "arm (models.py) — so the four INIT measures are one measure",
            "what_that_does_to_H6c": "the contrast row (A-relu/A-hermite/A-cheby) cannot "
                                     "disagree with A-jacobi in this coordinate, so Letter 017 "
                                     "debt two's 'all four arms predict it' falsifier is "
                                     "unfalsifiable as registered; reported as a finding, not "
                                     "repaired by an unregistered coordinate change",
            "question_not_claim": "would a POST-activation init measure (phi_arm(z), same "
                                  "squashed domain) be the contrast Letter 017 meant? that is "
                                  "the Geometer's amendment to write, not B's run to invent"},
        "weights": "counts", "arms": {
            a: {"seeds": sorted(v, key=lambda r: r["seed"]),
                "distinct_init_states": len({r["init_state_sha256"] for r in v}),
                "primary_arm_for_H6c": a == "jacobi",
                "role": "primary — the arm that learned the pair (Letter 017 debt two)"
                        if a == "jacobi" else "contrast row, NOT an extra check"}
            for a, v in per_arm.items()},
        "not_a_claim": "no evidence fit ran on these bytes here; the fit is JacobiGP's "
                       "(PREREG_EXP6 H6c) and the verdict is its letter to carry",
        "files": {p.name: _sha(p) for p in paths},
        "run_on": run_on(),
    }
    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / "h6c_init_dumps_summary.json"
    p.write_text(json.dumps(out, indent=1) + "\n")
    return p


def a_reference_terminals() -> dict:
    """A's exp8 terminal pairs, read from the parquet that is already in git.
    Read-only arithmetic: the comparison is the cheapest machine-effect estimate
    the programme has, because both machines ran the SAME registered trainer."""
    pq = RESULTS / "exp8_summary.parquet"
    if not pq.exists():
        return {"available": False}
    try:
        import pyarrow.parquet as _pq
    except ImportError:
        return {"available": False, "why": "no parquet reader in this venv"}
    t = _pq.read_table(pq).to_pydict()
    rows = [dict(zip(t.keys(), [t[k][i] for k in t])) for i in range(len(t["arm"]))]
    ref = {r["seed"]: (r["alpha"], r["beta"]) for r in rows
           if r["arm"] == "jacobi" and r["hidden"] == HIDDEN and r["alpha"] is not None}
    return {"available": True, "path": "results/exp8_summary.parquet",
            "sha256": _sha(pq), "rows": {str(k): v for k, v in ref.items()},
            "run_on_of_reference": "m1pro-32g (A) — the exp8 matrix was forged there"}


def summarize_walks(epochs: int = EPOCHS) -> Path:
    paths = sorted(RUNS.glob("h6c_walk_jacobi_h128_s*.json"))
    per_seed, term_a, term_b = [], [], []
    for p in paths:
        d = json.loads(p.read_text())
        t = d["terminal_pair"]
        if t["alpha"] is not None:
            term_a.append(t["alpha"]); term_b.append(t["beta"])
        per_seed.append({"seed": d["seed"], "walk": d["walk"], "terminal": t,
                         "secs_per_epoch": d["secs_per_epoch"], "file": f"runs/{p.name}"})
    n = len(term_a)
    out = {
        "kind": "h6c-walk-summary", "generated_by": "h6c_dumps.py --summarize",
        "arm": "jacobi", "hidden": HIDDEN, "n_seeds_recorded": n,
        "terminal_values": {"alpha": term_a, "beta": term_b},
        "mean_pair": {"alpha": round(statistics.fmean(term_a), 6) if n else None,
                      "beta": round(statistics.fmean(term_b), 6) if n else None},
        "sd_seed_sample": {"alpha": round(statistics.stdev(term_a), 6) if n > 1 else None,
                           "beta": round(statistics.stdev(term_b), 6) if n > 1 else None},
        "band_rule_lives_in_JacobiGP": "PREREG_EXP6 H6c: band = ±max(0.2, 2·sd_seed), per "
                                       "coordinate, frozen by amendment *before* the first H6c "
                                       "number. This file reports sd_seed and sets nothing.",
        "cross_machine_replication": cross_machine(per_seed),
        "question_about_the_registered_target": {
            "flagged_for": "JacobiGP (H6c's target pair) and Letter 006's author (PolyNN)",
            "what_B_ran_into": "PREREG_EXP6 H6c scores the fit against (0.40, 0.37), which "
                               "Letter 006 calls '≈ 0.40/0.37 in every seed' and Letter 017 "
                               "calls a mean. A's own committed exp8 parquet gives the h=128 "
                               "across-seed MEAN as (0.3589, 0.3590) and the h=256 mean as "
                               "(0.2613, 0.2782); (0.3977, 0.3677) — the h=128 seed-1000 cell — "
                               "is what rounds to 0.40/0.37",
            "size_of_it": "0.041 on α, 0.011 on β: inside any band this check will use, so it "
                          "cannot rescue or kill H6c",
            "why_B_files_it_anyway": "a prediction is a pointer to bytes. If the pointer means "
                                     "'one seed' the check is about a different object than if "
                                     "it means 'the mean of five'. The owner of the document "
                                     "decides, in an amendment, before the number exists",
            "evidence": "results/exp8_summary.parquet (path, sha256 in cross_machine_replication)"},
        "per_seed": per_seed, "files": {p.name: _sha(p) for p in paths},
        "run_on": run_on(),
    }
    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / "h6c_walks_summary.json"
    p.write_text(json.dumps(out, indent=1) + "\n")
    return p


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dumps", action="store_true")
    g.add_argument("--walks", action="store_true")
    g.add_argument("--summarize", action="store_true")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in tr.SEEDS))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    RUNS.mkdir(exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]

    if args.dumps:
        equiv = loaders_evidence(seeds[0])
        assert equiv["agree"], f"the data read is NOT outside the init stream: {equiv}"
        print(f"[evidence] loaders consume no init RNG: {equiv['sha256_with_datasets'][:12]}…")
        x = datasets()[0].tensors[0]                    # 55k train carve, float32, canonical order
        for arm in args.arms.split(","):
            for seed in seeds:
                t0 = time.perf_counter()
                d = dump_arm_seed(arm, seed, x, equiv)
                p = RUNS / f"h6c_dump_{arm}_h{HIDDEN}_s{seed}.json"
                p.write_text(json.dumps(d, indent=1))
                h = d["combined_all_layers"]
                print(f"[dump {arm:7s} s={seed}] n={h['n_values']:>8d} "
                      f"u_mean={h['u_mean']:+.4f} u_std={h['u_std']:.4f} "
                      f"init={d['init_state_pin']['sha256_float32_sorted_keys'][:12]}… "
                      f"{time.perf_counter() - t0:5.1f}s -> {p.name}")
        print(summarize_dumps())
    elif args.walks:
        for seed in seeds:
            t0 = time.perf_counter()
            d = walk_seed(seed, args.epochs)
            p = RUNS / f"h6c_walk_jacobi_h{HIDDEN}_s{seed}.json"
            p.write_text(json.dumps(d, indent=1))
            t = d["terminal_pair"]
            print(f"[walk s={seed}] (α,β) -> ({t['alpha']:.4f}, {t['beta']:.4f}) "
                  f"val5k_acc={d['final_acc_val5k']:.4f} diverged={d['diverged']} "
                  f"{time.perf_counter() - t0:.0f}s -> {p.name}")
        print(summarize_walks(args.epochs))
    else:
        print(summarize_dumps())
        print(summarize_walks(args.epochs))


if __name__ == "__main__":
    main()
