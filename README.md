# Polynomial-Activated NN (PolyNN bench)

基于和 agents 有关 Jacobi 多项式的内容，考虑在神经网络中引入多项式作为激活函数。
Jacobi 多项式通过自适应调价 alpha 和 beta 参数自动选择不同的正交基，从而让激活函数
工作在不同的泛函空间内。借此希望能增加神经网络层的表现力。

**CHORA status:** this checkout is the programme's **fourth bench** — the
third knob (shape $(\alpha,\beta)$) *interiorized into the neuron*.
Bench rules: `AGENTS.md` §0. Joint spec: `chora/benches/JacobiGP/docs/NEXT.md`
§2c. Correspondence: `docs/LETTERS/` (answered Letter 004 with Letter 005).

**exp8** (pre-registered in `docs/PREREG.md`, code tested in `tests/`):

| arm | activation | notes |
|---|---|---|
| A-relu | ReLU | width matched ≤1% of *total* params (800h+12 vs 795h+10) |
| A-jacobi | $\sum_k c_k \hat P_k^{(\alpha,\beta)}(\tanh x)$ | c per neuron, (α,β) learned, Legendre init |
| A-hermite | $\sum_k c_k H_k(x)$ | raw — no squashing, no rescue |
| A-cheby | $\sum_k c_k T_k(x)$ | raw |

```bash
python -m unittest tests.test_basis -v        # port verified vs scipy.special
python train.py --arm jacobi --hidden 128 --seed 1000 --epochs 20       # one cell
python compare.py --calibration               # 20 runs -> results/noise_band.json
python compare.py --phase remaining && python compare.py --report       # the table
python visualize.py                           # figures/ + results/relu_tail.json
```

Fashion-MNIST lives in the shared store `chora/data/FashionMNIST/`, hash-pinned
in `chora/data/manifest.json` (`by: Polynomial-Activated-NN@33afdb8`).
