"""Build the Quarto site and stage it on the ``gh-pages`` branch (PolyNN bench).

The site is rendered from local artifacts — it never re-runs exp8. The numbers
on the page must be the numbers in ``results/`` (produced by the pinned,
pre-registered protocol of docs/PREREG.md), not a fresh run elsewhere.

    python publish.py             # render + commit onto gh-pages locally
    python publish.py --push      # ...and push it
    python publish.py --skip-render

Why a branch, not docs/ on main: generated HTML lives on gh-pages; docs/ is
git-ignored except docs/LETTERS and docs/PREREG.md, which stay on main.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DOCS = REPO_ROOT / "docs"
WORKTREE = REPO_ROOT / ".gh-pages-worktree"
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"  # git hash-object -t tree /dev/null


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc


def _env() -> dict[str, str]:
    """PATH carrying this venv (pandas/polars/great_tables/ipykernel) and quarto.

    Quarto resolves `python3` from PATH; without the venv first it falls back to
    a system Python that cannot execute the note's chunks.
    """
    env = dict(os.environ)
    parts = [str(Path(sys.prefix) / "bin")]
    q = Path.home() / ".local" / "quarto" / "bin"
    if q.is_dir():
        parts.append(str(q))
    parts.append(env.get("PATH", ""))
    env["VIRTUAL_ENV"] = sys.prefix
    env["PATH"] = os.pathsep.join(p for p in parts if p)
    return env


def run(cmd: list[str]) -> None:
    print("  $", " ".join(cmd))
    if subprocess.run(cmd, cwd=REPO_ROOT, env=_env()).returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}")


def quarto_bin() -> str:
    found = shutil.which("quarto") or shutil.which("quarto", path=_env()["PATH"])
    if not found:
        raise SystemExit("quarto not found (expected ~/.local/quarto/bin); install or --skip-render")
    return found


def render() -> None:
    for need in ("exp8_summary.json", "exp8_summary.parquet", "noise_band.json", "relu_tail.json"):
        if not (REPO_ROOT / "results" / need).exists():
            raise SystemExit(f"results/{need} missing — run compare.py/visualize.py first")
    run([quarto_bin(), "render"])
    (DOCS / ".nojekyll").write_text("")
    if not (DOCS / "index.html").exists():
        raise SystemExit("render produced no docs/index.html")


SKIP = {".DS_Store", "Thumbs.db"}


def _copy_clean(src: Path, dst: Path) -> None:
    """docs/ onto the worktree, dropping macOS clutter — no .DS_Store on the site."""
    if src.name in SKIP:
        return
    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        for child in src.iterdir():
            _copy_clean(child, dst / child.name)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def stage(branch: str, message: str) -> str:
    """Copy docs/ onto `branch` as the repo root via a scratch worktree; the main
    working tree is never switched, so uncommitted source work is safe."""
    if WORKTREE.exists():
        raise SystemExit(f"refusing to touch {WORKTREE.name}: remove it first")
    had_branch = git("rev-parse", "--verify", "--quiet", branch, check=False).returncode == 0
    base = branch if had_branch else git(
        "commit-tree", EMPTY_TREE, "-m", "empty base for gh-pages").stdout.strip()
    git("worktree", "add", "--force", "-B", branch, str(WORKTREE), base)
    try:
        if had_branch:
            git("-C", str(WORKTREE), "rm", "-r", "-q", "--cached", "--ignore-unmatch", ".")
        for child in WORKTREE.iterdir():
            if child.name == ".git":
                continue
            shutil.rmtree(child, ignore_errors=True) if child.is_dir() else child.unlink(missing_ok=True)
        for item in DOCS.iterdir():
            _copy_clean(item, WORKTREE / item.name)
        git("-C", str(WORKTREE), "add", "-A")
        sha = ""
        if git("-C", str(WORKTREE), "diff", "--cached", "--quiet", check=False).returncode != 0:
            git("-C", str(WORKTREE), "commit", "-q", "-m", message)
            sha = git("-C", str(WORKTREE), "rev-parse", "--short", "HEAD").stdout.strip()
        return sha
    finally:
        git("worktree", "remove", "--force", str(WORKTREE))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--branch", default="gh-pages")
    p.add_argument("--skip-render", action="store_true", help="re-stage whatever is in docs/")
    p.add_argument("--push", action="store_true")
    p.add_argument("--message", default=None)
    args = p.parse_args(argv)

    if not args.skip_render:
        render()
    if not DOCS.exists():
        raise SystemExit("docs/ missing — run without --skip-render first")

    message = args.message or f"Publish exp8 site {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}"
    sha = stage(args.branch, message)
    print(f"\ngh-pages branch: {sha or 'unchanged (nothing new to commit)'}")
    if args.push:
        run(["git", "push", "--force-with-lease", "origin", args.branch])
        print("site: https://math4mad.github.io/Polynomial-Activated-NN/")
    else:
        print(f"to publish:  git push -u origin main {args.branch}   (or re-run with --push)")


if __name__ == "__main__":
    main()
