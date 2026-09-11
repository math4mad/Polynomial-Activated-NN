"""
Canonical Fashion-MNIST split module for exp8 — one module, pinned, no other
file may build a split (Sarcos `data.py` pattern, adopted as programme law).

Bytes live in the shared CHORA store: ``data/`` is a symlink to ``chora/data``,
and the dataset lands in ``chora/data/FashionMNIST/``. On first fetch the
files are hashed and appended to ``chora/data/manifest.json`` (law 2); see
``make_manifest_entry()`` — run it once, paste nothing from memory.

Split contract (pre-registered in docs/PREREG.md §1):
  * torchvision canonical train (60k) / test (10k);
  * a 5k VALIDATION carve-out from TRAIN, permutation of seed SPLIT_SEED —
    used only by ``--smoke`` engineering runs; hypothesis-test runs score on
    the untouched 10k test at the final epoch, never for selection;
  * transform: ToTensor + Normalize((0.5,), (0.5,)) — as pre-registered.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision
import torchvision.transforms as T

BENCH_ROOT = Path(__file__).resolve().parent
STORE_ROOT = BENCH_ROOT / "data"                    # -> chora/data (symlink)
DATA_ROOT = STORE_ROOT                              # torchvision creates FashionMNIST/ here
MANIFEST = STORE_ROOT / "manifest.json"

SPLIT_SEED = 20260911
N_VAL = 5_000
MEAN, STD = (0.5,), (0.5,)
_batch = 64


def _transform() -> T.Compose:
    return T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])


def datasets(train_root: Path = DATA_ROOT) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """(train_carved 55k, val 5k, test 10k). Download only if absent.

    Tensors are materialised once (784-d float, uint8-free) so every run of
    every arm sees bit-identical inputs and the DataLoader worker cost stays
    out of the wall-clock comparison.
    """
    tf = _transform()
    tr = torchvision.datasets.FashionMNIST(train_root, train=True, download=True, transform=tf)
    te = torchvision.datasets.FashionMNIST(train_root, train=False, download=True, transform=tf)

    x_tr = tr.data.float().div_(255.0).view(-1, 784).sub_(0.5).div_(0.5)
    y_tr = tr.targets.long()
    x_te = te.data.float().div_(255.0).view(-1, 784).sub_(0.5).div_(0.5)
    y_te = te.targets.long()

    perm = torch.randperm(x_tr.shape[0], generator=torch.Generator().manual_seed(SPLIT_SEED))
    val_idx, tr_idx = perm[:N_VAL], perm[N_VAL:]
    train = TensorDataset(x_tr[tr_idx], y_tr[tr_idx])
    val = TensorDataset(x_tr[val_idx], y_tr[val_idx])
    test = TensorDataset(x_te, y_te)
    return train, val, test


def loaders(batch_size: int = _batch, num_workers: int = 0, root: Path = DATA_ROOT):
    train, val, test = datasets(root)
    g = torch.Generator().manual_seed(SPLIT_SEED)  # epoch-shuffle stream, pinned
    return (DataLoader(train, batch_size=batch_size, shuffle=True, generator=g,
                       num_workers=num_workers, drop_last=False),
            DataLoader(val, batch_size=256, shuffle=False, num_workers=num_workers),
            DataLoader(test, batch_size=256, shuffle=False, num_workers=num_workers))


# --------------------------------------------------------------------------- #
# manifest machinery (CHORA law 2: nothing crosses benches without a hash)
# --------------------------------------------------------------------------- #
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_manifest_entry(by: str, script: str = "data.py::datasets",
                        notes: str = "FashionMNIST for exp8; torchvision MD5-pinned upstream gz layout") -> list[dict]:
    """Return manifest entries for every file under DATA_ROOT (run once after
    first fetch; caller appends to chora/data/manifest.json and commits there)."""
    entries = []
    for p in sorted((DATA_ROOT / "FashionMNIST").rglob("*")):
        if p.is_file() and p.name != ".DS_Store":
            entries.append({
                "path": str(Path("data") / p.relative_to(DATA_ROOT)),
                "sha256": sha256_of(p),
                "bytes": p.stat().st_size,
                "source": "torchvision FashionMNIST (mirrors: "
                          "https://storage.googleapis.com/cvdf-datasets/mnist/)",
                "obtained": "2026-09-11",
                "by": by,
                "script": script,
                "notes": notes,
            })
    return entries


if __name__ == "__main__":
    tr, va, te = datasets()
    print(json.dumps({
        "train_carved": len(tr), "val": len(va), "test": len(te),
        "split_seed": SPLIT_SEED,
        "manifest": [
            {"path": e["path"], "sha256": e["sha256"][:12] + "…", "bytes": e["bytes"]}
            for e in make_manifest_entry(by="Polynomial-Activated-NN@<sha>")
        ],
    }, indent=1))
