# Static Tokens Are Colder

**Averaged noise for frame-shared video generation — errata and a follow-up to
"Entropy-Continuum Mechanics for Video" (K. S. Garlapati, technical report, Nov 2026).**

Paper: [`paper/paper.pdf`](paper/paper.pdf) · Errata list: [`ERRATA.md`](ERRATA.md) ·
Where every number comes from: [`FINDINGS.md`](FINDINGS.md)

Everything here is toy-scale and CPU-only, like the report it builds on. The claims are about
mechanisms, not about production video models.

## What this is

The base report proposes *frame sharing* for video diffusion transformers: compute the tokens of a
static background once and reuse them in every frame. Sharing needs the copies to be identical, so
the report overwrites the starting noise of a static position in all frames with the noise of one
"leader" frame.

This repository contains

1. **an errata list** for the base report (17 items, most of them checked numerically), and
2. **new work on its open question** — "is the shared-noise sample as good?" — with theory, a
   one-line fix, and experiments.

## Findings

### 1. Errata for the base paper ([`ERRATA.md`](ERRATA.md))

Seventeen items. The ones that matter most:

- **The headline result of frame sharing uses a moved reference.** Its 41.5-43.8 dB is fidelity to
  full compute from noise the method has already tied; every baseline it is compared with is scored
  against the original output. Against the original, the base paper's own number is 11-15 dB, below
  every baseline in its table. We reproduce both numbers on our model:
  61.2 dB and 7.9 dB.
- **"Noise alone costs nothing" (Eq. 3) is false**: the clamp leaves 1/(e ln 2) =
  0.531 bits/pixel under the paper's own noise model.
- **The score is unbounded and depends on grain**: one fixed lighting change scores
  0.14 / 0.46 / 1.46 at three noise levels.
- **The clip score grows with clip length**: two event-free sets of different length are separated
  with AUC 0.89.
- **Proportional attention is exact only without temporal position embeddings**
  (relative error 4e-10 without, 0.28 with, on a random transformer).

### 2. Why copied noise is too hot (theory)

The F copies of a static token are F noisy measurements of one quantity, and the network pools
them. The pooled measurement is their **mean**, whose noise is √F weaker than any single copy.

- *Averaging preserves the sample.* For an ideal denoiser and a static region, replacing the copies
  by their frame mean leaves the generated sample exactly unchanged, at the start or at any later
  step.
- *Copying heats the static content by √F.* For Gaussian content the output variance is inflated by
  exactly F.
- *Static tokens are colder.* A token shared by F frames has the signal-to-noise ratio of an
  ordinary token at t' = t / (t + √F (1 - t)), which predicts the base paper's observation that
  static tokens settle earlier.

### 3. Checks that need no network

| | copy (base paper) | mean (ours) |
|---|---|---|
| Exact Gaussian model, variance inflation at F = 8 | 7.98 (predicted 8) | 1.000, sample identical to 1e-15 |
| Exact nonlinear denoiser: sampler returns the original clip | 0.1 % | 90.2 % |
| ... collapsing at step 6 instead of step 0 | 84.1 % | 100.0 % |
| ... spread of the produced backgrounds (1 is correct) | 1.73 | 1.01 |

### 4. A trained video diffusion transformer, everything on the same footing

PSNR is against the **original** sample (20-step full compute from the untouched noise), median over
1,024 seeds. "bg" and "tex." are background spread and texture relative to full compute (1 is correct).

| | standard model | | | | regularised model | | | |
|---|---|---|---|---|---|---|---|---|
| Method | speedup | PSNR (dB) | bg | tex. | speedup | PSNR (dB) | bg | tex. |
| Fewer steps, 20→10 | 2.00× | 26.9 | 0.75 | 0.8 | 2.00× | 27.3 | 0.74 | 0.9 |
| Fewer steps, 20→7 | 2.86× | 23.2 | 0.59 | 0.7 | 2.86× | 23.6 | 0.59 | 0.8 |
| Copy-tied sharing, oracle map (base paper) | 2.31× | 7.9 | 2.60 | 5.2 | 3.20× | 7.3 | 2.77 | 8.1 |
| Copy-tied sharing, 2-step draft map (base paper) | 1.24× | 11.4 | 2.05 | 3.1 | 1.99× | 8.1 | 2.63 | 7.1 |
| **Mean-tied sharing, oracle map (ours)** | 2.31× | 20.9 | 1.03 | 2.4 | 3.20× | 43.8 | 1.00 | 1.0 |
| **Mean-tied sharing, 2-step draft map (ours)** | 1.24× | 23.2 | 1.07 | 2.0 | 1.99× | 43.9 | 1.00 | 1.0 |
| **Mean-collapse at step 8, no draft (ours)** | 1.21× | 26.9 | 1.04 | 1.7 | 1.44× | 47.1 | 1.00 | 1.0 |

- **Standard model.** Averaging removes the over-dispersion that copying causes and is 13 dB closer
  to the original sample. It is still not sample-preserving: the network has never seen tied tokens
  and answers with too much texture. On an equal footing, frame sharing loses to simply taking
  fewer steps, with either way of tying. *The base paper's central comparison does not survive.*
- **Regularised model.** A 3k-step fine-tune with a regulariser that encodes the sufficiency of the
  frame mean makes mean-tied sharing reproduce the original sample at
  43.8 dB with 3.2× fewer FLOPs, 15-20 dB above the
  fewer-steps baseline. Copying stays at 7.3 dB on the same network.
- **Classifier two-sample test against full compute** (0.5 = indistinguishable): mean-tied sharing
  0.86 on the standard model and 0.50 on the regularised one; copy-tied sharing
  0.96 and 1.00; 7-step sampling 0.85 and 0.84.

### 5. What the regulariser costs

Full-compute samples of each model variant against the training data (1 = same as data):

| Model | consistency error ×1e3 | mean-tied PSNR vs. original (dB) | texture | motion |
|---|---|---|---|---|
| standard, 12k steps | 8.8 | 20.9 | 0.85 | 1.46 |
| + 3k steps, no regulariser (control) | 8.7 | 21.3 | 0.85 | 1.47 |
| + 3k steps, λ = 5 | 0.5 | 37.1 | 0.74 | 1.24 |
| + 3k steps, λ = 50 | 0.1 | 43.5 | 0.54 | 0.71 |

The control shows the change is the regulariser's, not that of extra training. Its price is texture
in the model's own samples; a gentler weight (λ = 5) keeps most of the benefit at a smaller cost.
We tried only these two weights, so where the best trade-off lies is open.

### What this does not show

- Nothing here was run on a real video model. The theory does not depend on scale; the experiments do.
- One training run per model variant.
- The permanence-score errata come from our re-implementation of the printed equations, and we
  propose no replacement score.
- Speedups are FLOP counts.

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest -q tests          # 9 sanity tests, a few seconds
./run_all.sh                       # everything: checks, training, evaluation, figures, paper
```

`run_all.sh` takes roughly three hours on 4 CPU cores. The parts that need no training take
about 15 minutes and already contain the core argument:

```bash
export PYTHONPATH=$PWD
python experiments/e0_errata_checks.py     # errata for the base paper
python experiments/e1_gaussian_law.py      # variance-inflation law, exact Gaussian model
python experiments/e2_ideal_denoiser.py    # exact nonlinear denoiser, no network
```

Trained weights are included in `checkpoints/`, so the evaluation scripts (`e3`–`e5`) can be run
without retraining. Building the PDF needs `pdflatex` and `bibtex`.

## Layout

| Path | Content |
|---|---|
| `meantie/data.py` | synthetic clips: Gaussian-field background + moving sprites |
| `meantie/model.py` | 1.27M-parameter video diffusion transformer (rectified flow) |
| `meantie/sampling.py` | Euler sampler, `copy` / `mean` tying, frame sharing with proportional attention |
| `meantie/flops.py` | analytic FLOP model, including the base paper's large-model projection |
| `meantie/c2st.py` | classifier two-sample test |
| `experiments/e0_errata_checks.py` | numerical checks of the errata |
| `experiments/e1_gaussian_law.py` | exact Gaussian model |
| `experiments/e2_ideal_denoiser.py` | exact nonlinear (empirical-data) denoiser |
| `experiments/train.py` | training, incl. the sufficiency regulariser (`--consistency`) |
| `experiments/e3_trained_model.py` | all sampling methods on a trained model, same-footing metrics |
| `experiments/e4_distribution_test.py` | does an accelerated sampler still draw from the model's distribution? |
| `experiments/e5_wallclock.py` | FLOP model vs. measured CPU time |
| `experiments/make_figures.py`, `make_paper_inputs.py` | figures, LaTeX macros and tables from `results/*.json` |
| `results/` | every number in the paper, as JSON |
| `paper/` | LaTeX source and PDF |

## The fix in one line

```python
# base paper: every copy of a static token starts from the leader frame's noise
rep = noise[:, :1]
# this work: every copy starts from the frame mean of the noise
rep = noise.mean(dim=1, keepdim=True)
```

(`meantie/sampling.py`, function `tie`.)

## License

Apache-2.0, see `LICENSE`.
