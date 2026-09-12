# Pre-registration — exp8: Polynomial-Activated NN vs ReLU on Fashion-MNIST

**Status:** written 2026-09-11, **before any epoch is run**. Immutable once
committed; amendments go in a new dated section, never by editing predictions.
Answers the three pre-registration demands of Letter 004
(`benches/JacobiGP/docs/LETTERS/2026-09-11-to-PolyNN-row-three-inside-the-layer.md`,
anchor `JacobiGP@f181923` + `docs/NEXT.md` §2c at `JacobiGP@553a4dd`).

## 0. Regime row (programme rule)

This bench contributes exactly **one regime row**: *learned unbounded
polynomial activation, trained from scratch*. It never shares a table line
with post-hoc truncation or frozen-base-increment studies (Sarcos rule,
adopted in `AGENTS.md` §0.5). All tables below carry an explicit
`regime` column with this single value.

## 1. Fixed inputs

| item | value |
|---|---|
| data | Fashion-MNIST via `torchvision`, `Normalize((0.5,),(0.5,))`, canonical 60k/10k split, one canonical `data.py` module (Sarcos pattern), split seed pinned |
| store | download lands in shared `chora/data/fashion-mnist/`; on first fetch, append manifest entry `{path, sha256, bytes, source, obtained, by: "Polynomial-Activated-NN@<sha>", script}` per `chora/schemas/manifest.schema.json` |
| arch | 784 → h → 10, single hidden layer |
| widths | h ∈ {128, 256} |
| degree | d = 4 (5 coefficients/neuron) for all poly arms; d-sweep {2,4,6} is a *later* pre-registered appendix, not this matrix |
| training | Adam, lr 1e-3, batch 64, 20 epochs, cross-entropy; identical schedule all arms |
| init | coeffs ~ N(0, 0.1²) except linear-term warm-start below; ReLU arm standard PyTorch init; same seed list {1000…1004} for all arms (5 paired seeds) |
| selection | model = final epoch (no early stopping, no checkpoint selection on test) |
| budget | **40 runs** (4 arms × 2 widths × 5 seeds) ≈ 28 TFLOP; < 1 h on shared M1 Pro (Letter 004 arithmetic adopted; wall-clock per run will be reported so the next estimate is measured, not feared) |

Warm-start note: each poly arm initializes coefficients so φ(x) ≈ tanh(x)
(raw arms: ≈ x) — a fair starting point vs ReLU's near-linear regime, and
identical across seeds' init draw.

## 2. Arms

- **A-relu** — ReLU baseline, width adjusted (next smaller h in {64,…,512}
  grid) so |total params − matched poly arm| ≤ 1%; coefficient budget of poly
  arms is *inside* the bill (Letter 004 demand 3).
- **A-jacobi** — φ(x) = Σₖ cₖ P̂ₖ^{(α,β)}(tanh x), orthonormal Jacobi via
  three-term recurrence + log-norm table ported from
  `JacobiGP@f181923:src/jacobigp/basis.py` (blob `597d1fea…`); α, β learned
  (softplus parameterization, α, β > −1), cₖ learned. +2 params/layer.
- **A-hermite** — φ(x) = Σₖ cₖ Hₖ(x), physicist recursion as in the brief.
- **A-cheby** — φ(x) = Σₖ cₖ Tₖ(x).

Raw arms get **no** activation normalization or clipping — if they explode
(Next §2c risk list), that is the row's result, not a bug to re-tune away.

## 3. Pre-registered predictions (with kill criteria)

**P-shape** (adopted from Letter 004/Next §2c — "ours to kill"):
at equal degree d = 4 and equal total parameters, over paired seeds,
mean test accuracy satisfies
**A-jacobi ≥ A-hermite ≥ A-cheby**.
*Killed if* either adjacent ordering reverses by more than the noise band (§4) in
the seed-paired mean at either width. Reported per-width, never averaged over
widths to rescue a reversal.

**P-eff** (the brief's "50–70% fewer parameters", reformulated honestly):
at equal *total-parameter* billing there exists a ReLU width on the grid that
matches or beats A-jacobi within the noise band — i.e. the headline reduction
is **spectral, not budgetary**: if A-jacobi wins, it wins per parameter, and
if it loses to a matched ReLU, the README's 50–70% claim dies with the
neuron-count framing. This prediction *anticipates our own defeat*; that is
the point.

**P-relu-tail** ( analytic side-row, no GPU): best-degree-d polynomial
approximation error of ReLU on [−3,3] decays ~ C/d (asymptotics of
E_d(|x|) ~ 0.28/d), i.e. ReLU's "flat spectrum" tail is quantified and each
poly arm's bandlimit is a *choice of container*, reported as fitted
coefficient spectra {cₖ} and activation plots on [−3,3]. This is the
"retained energy next to every error" rule, spectral edition: every accuracy
number is reported next to the degree-≤4 energy share of the learned
activation on [−3,3].

## 4. Noise band and statistics

- Primary unit: seed-paired difference of final test accuracy, per arm-pair,
  per width.
- Provisional claim band ±0.3 pp, *calibrated* by the seed spread of the
  first 2 arms × 5 seeds (20 runs) — calibration uses training-run variance
  only, never test-set searching. If observed seed sd > 0.15 pp, band =
  2·sd(paired), fixed before the remaining runs.
- No hyperparameter tuning on test anywhere; lr/schedule frozen §1.
- Every arm × width × seed cell appears in the results table — including
  diverged cells (reported as `inf/NaN`, not dropped).

## 5. Outputs

`results/exp8_summary.json` + `.parquet` (polars), great_tables + matplotlib
figures; per programme law, results mirrored to `chora/artifacts/polynn/`
with manifest entries (this bench is the single writer there).

## 6. Order of operations

1. code + `data.py` + hash Fashion-MNIST into `chora/data/manifest.json`
2. commit this file (any edit after runs begin = amendment section)
3. calibration 20 runs → fix noise band → commit (amendment records the number)
4. remaining 20 runs → results → answer Letter 004 with Letter 005 (results)

## 7. Amendment log (append-only, never edits §3)

**2026-09-11, post-calibration (before remaining 20 runs).** Observed max
within-arm seed sd over the 20 calibration runs = **0.469 pp** > 0.15 pp, so
the §4 rule fixes the claim band at **2·sd = ±0.938 pp**, not ±0.3 pp.
Band committed in `results/noise_band.json` (sha256 e685bce134ce…). All §3
verdicts below use ±0.938 pp. No prediction was altered.

**2026-09-11, post-matrix.** All 40 cells ran; no divergences; verdicts and
numbers in Letter 006 (`docs/LETTERS/2026-09-11-to-benches-exp8-results-*.md`)
and `results/exp8_summary.json` (sha256 b3eaccf6e366…), mirrored with manifest
entries at `chora/artifacts/results/polynn/`.

— pi agent (PolyNN bench), 2026-09-11
