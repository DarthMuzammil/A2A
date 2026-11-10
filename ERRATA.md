# Errata for "Entropy-Continuum Mechanics for Video"

Base paper: K. S. Garlapati, *Entropy-Continuum Mechanics for Video: Permanence Scoring and
Frame-Shared Generation. Preliminary Results and Failure Modes*, technical report, November 2026.

We re-derived or re-implemented what could be checked without the authors' code. Every number
quoted below is produced by `experiments/e0_errata_checks.py` (results in
`results/e0_errata_checks.json`) unless another script is named.

Status codes: **A** shown analytically · **N** verified numerically here · **T** follows from the
base paper's own text or tables · **C** a concern that needs the authors' code to quantify.

The base paper is candid about its scale and its failures; several items below sharpen negative
results it already states.

## Permanence violation score (PVS)

| # | Where | Statement or practice | Finding | Status |
|---|---|---|---|---|
| 1 | Eq. 3 | "Motion explained by noise alone costs nothing." | False. The clamp at zero leaves E[s] = 1/(e ln 2) = **0.531 bits/px** under the paper's own Laplacian noise model (Monte Carlo: 0.530); 2^-δ times that with dead zone δ. At 4 bits/px this alone implies PVS_t ≈ 0.13 for a clip in which nothing happens, 20-40× the 0.003-0.007 shown for legal clips in Fig. 1. The printed equations do not determine the printed numbers; an unreported floor on `b` (which is zero on noise-free synthetic clips) is the likely reason. | A, N |
| 2 | Eq. 3 vs. Eq. 2 | Eq. 3 is called the Lagrangian form of the balance law. | It is not. The Lagrangian form of Eq. 2 is a statement about the scalar density, which the paper itself shows fails (0.57 AUC). Eq. 3 is a noise-normalised motion-compensated residual, i.e. warping error in other units. Only the mask comes from the balance law, and the comparison "PVS 0.86 vs. warping error 0.63" changes four things at once (normalisation, mask, max over frames, division by total density) without an ablation. | A |
| 3 | Eq. 3-4 | PVS is read as a share of the frame's information. | Eq. 3 charges a pixel without bound, so PVS is not a fraction (the paper reports values up to 2.22). In our re-implementation a legal soft spotlight scores **8.6** under the paper's σ=2 prefilter, with single pixels charged 91 bits (a raw pixel costs 8). A code can always fall back to intra coding; capping the charge by the intra cost gives 0.33. | N, T |
| 4 | Eq. 3-4 | One threshold for all footage. | The numerator is relative to the clip's noise level, the denominator is absolute bits. One fixed event scores **0.14 / 0.46 / 1.46** at noise σ = 3 / 1.5 / 0.75 grey levels. Clean (generated, denoised, blurred) footage is scored far more harshly than grainy footage, which confounds the real-vs-generated test and matches the paper's remark that compression doubled the peak score. | N |
| 5 | §4, §4.3 | Clip score is max over frames. | Grows with clip length. Two event-free sets with 127 and 15 frame pairs are separated with AUC T1/(T1+T2) = **0.89** (simulated with frame-to-frame correlation 0-0.9: 0.86-0.90). The synthetic clips have 16 frames, the real ones are several times longer. | A, N |
| 6 | §4.3 | "There was no separation" between real and generated clips. | 4 real and 15 generated clips, no violation labels, mixed lengths and noise levels (items 4, 5). The test can show neither separation nor its absence. §7 calls the same set "19 clips". | T |
| 7 | Table 1 | AUC differences of 0.01-0.03 are interpreted. | No uncertainty is reported. With about 150 clips per class the standard error of an AUC near 0.8 is about 0.026 (Hanley-McNeil). | A |
| 8 | §4.1-4.2 | Headline 0.86 (clean set) and 0.80 (stress set). | As far as the text specifies, the two use different pipelines (prefilter off / on); no single configuration is reported on both sets. | C |
| 9 | §4.2 | Occlusion boundaries from the neighbouring frame pairs. | Fixes one-frame events only. A melt, fade or boil spans the neighbouring pairs too, so the spurious boundaries return. The first and last pair have only one neighbour. | C |
| 10 | §3-4 | Sources are legal at occlusion boundaries. | *Any* creation there is free: an object that changes identity behind an occluder, or never re-emerges, costs nothing. The balance law has no storage term, so the classical object-permanence violation is invisible by construction. | A |

## Frame-shared generation

| # | Where | Statement or practice | Finding | Status |
|---|---|---|---|---|
| 11 | Eq. 5 | Proportional attention "reproduces full attention exactly". | Only without temporal position information, which a video model needs. Random transformer: relative error **4e-10** without a temporal embedding, **0.28** with one (`tests/test_equivalence.py` asserts both). The paper's own oracle result (43.6 dB, falling as more frames share) is not exact either. | N, T |
| 12 | Table 2, Fig. 4 | Sharing "lies clearly above the fewer-steps baseline" (41.5-43.8 dB vs. 27-30 dB). | **Different references.** Baselines are scored against the original output; sharing is scored against full compute from noise the method has already tied. Against the original, sharing scores 11-15 dB (the paper's §5), below every baseline in the table. The tied reference also rewards over-sharing, so it cannot see the method's main failure mode. Reproduced in `experiments/e3_trained_model.py`. | T, N |
| 13 | §5 | Whether the shared-noise sample is as good is "untested". | It has an answer: copying one frame's noise to F frames is not distribution-preserving. Static content is driven by noise √F too strong; for Gaussian content the output variance is inflated by exactly F (`experiments/e1_gaussian_law.py`, `e2_ideal_denoiser.py`, `e3_trained_model.py`). | A, N |
| 14 | §5 | Static map from a 2-step draft. | The draft runs on independent noise, the final sample on tied noise 11-15 dB away. The map describes a video that is never generated. | T |
| 15 | §5 | 41.5-43.8 dB fidelity. | Finer than the reference itself: 20 and 40 steps differ by 39.2 dB. Above about 39 dB the metric measures agreement with a discretisation error. | T |
| 16 | Table 2 | Projected speedups 3.7-5.7×. | Include a fixed draft cost of 2 of 20 full steps. At the 90 % setting the draft is 56 % of the projected cost (our FLOP model reproduces the projections: 3.6× and 5.6×) and caps any speedup at 10×. | A |
| 17 | §5 | Static tokens settle earlier, taken as support for the temperature view. | Predicted by pooling alone: a token shared by F frames has the signal-to-noise ratio of an ordinary token at t' = t / (t + √F (1 - t)). If moving tokens settle at step 7 of 20, static ones should settle at about step 3; the paper measured 4. | A |

## Minor

- Eq. 4 writes |s| although s ≥ 0 by Eq. 3. The score measures created content only; "destroyed"
  information is seen only as whatever replaces it.
- The legal set uses ‖∇v‖ > τ, which includes divergence. Under a zoom every pixel can become
  "legal" (not checked; needs the authors' τ).
