# Where every number comes from

All numbers that come from our own experiments are written into the paper by
`experiments/make_paper_inputs.py`, which reads `results/*.json` and emits
`paper/generated/numbers.tex` (LaTeX macros) and `paper/generated/table_*.tex`.
Figures are drawn from the same files by `experiments/make_figures.py`.
Numbers quoted from the base paper (e.g. 41.5-43.8 dB, 11-15 dB, 0.86 AUC) are copied from its text.

| Paper item | Script | Result file |
|---|---|---|
| Table 1 (errata), Fig. 1, §2 numbers | `experiments/e0_errata_checks.py` | `results/e0_errata_checks.json` |
| §5 "Exact Gaussian model", Fig. 2a | `experiments/e1_gaussian_law.py` | `results/e1_gaussian_law.json` |
| Table 2 (exact nonlinear denoiser) | `experiments/e2_ideal_denoiser.py` | `results/e2_ideal_denoiser.json` |
| Table 3 left, Fig. 2b, Fig. 3a, Fig. 4 top (standard model) | `experiments/e3_trained_model.py --ckpt checkpoints/dit.pt` | `results/e3_trained_model.json`, `results/e3_examples.pt` |
| Table 3 right, Fig. 3b, Fig. 4 bottom (regularised model) | `experiments/e3_trained_model.py --ckpt checkpoints/dit_regularised.pt` | `results/e3_regularised_model.json`, `results/e3_examples_regularised.pt` |
| Table 4 (model variants: consistency error, alpha sweep, sample statistics) | `experiments/e6_sufficiency.py` | `results/e6_sufficiency.json` |
| Table 5 (classifier two-sample test) | `experiments/e4_distribution_test.py` | `results/e4_distribution_test.json`, `results/e4_distribution_test_regularised.json` |
| Settling steps, proportional-attention error on the trained model | `experiments/e3_trained_model.py` | key `settling`, `proportional_attention_trained_model_rel_error` |
| FLOPs vs. wall-clock | `experiments/e5_wallclock.py` | `results/e5_wallclock.json` |
| Training curves | `experiments/train.py` | `results/train_log*.json` |

## Models

| Checkpoint | How it was trained |
|---|---|
| `checkpoints/dit.pt` | standard rectified flow, 12k steps, batch 64, seed 0 |
| `checkpoints/dit_step3000.pt`, `dit_step6000.pt` | snapshots of the same run |
| `checkpoints/dit_regularised.pt` | `dit.pt` + 3k steps with the sufficiency regulariser, lambda = 50 |
| `checkpoints/dit_regularised_w5.pt` | same with lambda = 5 |
| `checkpoints/dit_control.pt` | `dit.pt` + 3k steps without the regulariser (control for "more training") |

All checkpoints hold the EMA weights used for sampling; `dit.pt` also holds the raw weights that the
fine-tunes start from.

## Things to know before trusting a number

- **Toy scale.** 1.27M parameters, 8x32x32 grayscale synthetic clips, static camera, CPU. One
  training run per model variant.
- **PSNR** uses a peak-to-peak range of 2 and is the median over 1,024 noise seeds. A value of 206 dB
  means "identical" (the MSE is clamped at 1e-20).
- **Speedups are FLOP counts**, not timings. They include every draft or warm-up forward call.
  Token removal is executed through an attention bias, which `tests/test_equivalence.py` shows to be
  identical to physically removing the tokens.
- **Static-map thresholds** are tuned on separate seeds (1000) from the evaluation seeds (2000).
  "Oracle" maps are read from the original sample itself and are not available in practice.
- **The PVS errata** use our re-implementation of the base paper's printed equations with zero flow
  on synthetic two-frame scenes. We do not have the authors' code.
- **The classifier two-sample test** is a lower bound on distinguishability; white noise (std 0.03)
  is added to both sides so that jitter on static positions is not a cue.
