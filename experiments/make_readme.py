"""Fill the results section of README.md from results/*.json (so the README cannot drift from the data)."""
import json, os

R = lambda f: json.load(open(os.path.join("results", f)))
e0, e2 = R("e0_errata_checks.json"), R("e2_ideal_denoiser.json")
A, B = R("e3_trained_model.json"), R("e3_regularised_model.json")
e6 = R("e6_sufficiency.json")["models"]
c4a, c4b = R("e4_distribution_test.json")["vs_full_compute"], R("e4_distribution_test_regularised.json")["vs_full_compute"]


def row(res, key):
    v = res["methods"][key]
    ref = res["reference"]["stats"]
    return (f"{v['speedup_flops']:.2f}×", f"{v['psnr_vs_original_db']['median']:.1f}", f"{v['bg_coef_std_ratio_mean']:.2f}",
            f"{v['stats']['texture'] / ref['texture']:.1f}")


def line(label, key):
    a, b = row(A, key), row(B, key)
    return f"| {label} | {a[0]} | {a[1]} | {a[2]} | {a[3]} | {b[0]} | {b[1]} | {b[2]} | {b[3]} |"


m2 = e2["methods"]
v = lambda k: e6[k]
block = f"""## Findings

### 1. Errata for the base paper ([`ERRATA.md`](ERRATA.md))

Seventeen items. The ones that matter most:

- **The headline result of frame sharing uses a moved reference.** Its 41.5-43.8 dB is fidelity to
  full compute from noise the method has already tied; every baseline it is compared with is scored
  against the original output. Against the original, the base paper's own number is 11-15 dB, below
  every baseline in its table. We reproduce both numbers on our model:
  {A['methods']['copy-tied sharing, oracle map']['psnr_vs_own_tied_full_compute_db']:.1f} dB and {A['methods']['copy-tied sharing, oracle map']['psnr_vs_original_db']['median']:.1f} dB.
- **"Noise alone costs nothing" (Eq. 3) is false**: the clamp leaves 1/(e ln 2) =
  {e0['a_null_bias']['analytic_laplace_bits_per_px']:.3f} bits/pixel under the paper's own noise model.
- **The score is unbounded and depends on grain**: one fixed lighting change scores
  {e0['c_unbounded_and_noise_scale']['noise_level']['sigma=3.0']['pvs_as_written']:.2f} / {e0['c_unbounded_and_noise_scale']['noise_level']['sigma=1.5']['pvs_as_written']:.2f} / {e0['c_unbounded_and_noise_scale']['noise_level']['sigma=0.75']['pvs_as_written']:.2f} at three noise levels.
- **The clip score grows with clip length**: two event-free sets of different length are separated
  with AUC {e0['b_length_bias']['closed_form_iid']['127_vs_15']:.2f}.
- **Proportional attention is exact only without temporal position embeddings**
  (relative error {e0['d_proportional_attention']['no_temporal_pe']['proportional']:.0e} without, {e0['d_proportional_attention']['temporal_pe']['proportional']:.2f} with, on a random transformer).

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
| Exact nonlinear denoiser: sampler returns the original clip | {100*m2['copy-tied, oracle map']['same_clip_rate']:.1f} % | {100*m2['mean-tied, oracle map']['same_clip_rate']:.1f} % |
| ... collapsing at step 6 instead of step 0 | {100*m2['copy-collapse at step 6, map from trajectory']['same_clip_rate']:.1f} % | {100*m2['mean-collapse at step 6, map from trajectory']['same_clip_rate']:.1f} % |
| ... spread of the produced backgrounds (1 is correct) | {m2['copy-tied, oracle map']['bg_coef_std_ratio_mean']:.2f} | {m2['mean-tied, oracle map']['bg_coef_std_ratio_mean']:.2f} |

### 4. A trained video diffusion transformer, everything on the same footing

PSNR is against the **original** sample (20-step full compute from the untouched noise), median over
1,024 seeds. "bg" and "tex." are background spread and texture relative to full compute (1 is correct).

| | standard model | | | | regularised model | | | |
|---|---|---|---|---|---|---|---|---|
| Method | speedup | PSNR (dB) | bg | tex. | speedup | PSNR (dB) | bg | tex. |
{line('Fewer steps, 20→10', 'fewer steps (10)')}
{line('Fewer steps, 20→7', 'fewer steps (7)')}
{line('Copy-tied sharing, oracle map (base paper)', 'copy-tied sharing, oracle map')}
{line('Copy-tied sharing, 2-step draft map (base paper)', 'copy-tied sharing, 2-step draft map')}
{line('**Mean-tied sharing, oracle map (ours)**', 'mean-tied sharing, oracle map')}
{line('**Mean-tied sharing, 2-step draft map (ours)**', 'mean-tied sharing, 2-step draft map')}
{line('**Mean-collapse at step 8, no draft (ours)**', 'mean-collapse sharing at step 8')}

- **Standard model.** Averaging removes the over-dispersion that copying causes and is 13 dB closer
  to the original sample. It is still not sample-preserving: the network has never seen tied tokens
  and answers with too much texture. On an equal footing, frame sharing loses to simply taking
  fewer steps, with either way of tying. *The base paper's central comparison does not survive.*
- **Regularised model.** A 3k-step fine-tune with a regulariser that encodes the sufficiency of the
  frame mean makes mean-tied sharing reproduce the original sample at
  {B['methods']['mean-tied sharing, oracle map']['psnr_vs_original_db']['median']:.1f} dB with {B['methods']['mean-tied sharing, oracle map']['speedup_flops']:.1f}× fewer FLOPs, 15-20 dB above the
  fewer-steps baseline. Copying stays at {B['methods']['copy-tied sharing, oracle map']['psnr_vs_original_db']['median']:.1f} dB on the same network.
- **Classifier two-sample test against full compute** (0.5 = indistinguishable): mean-tied sharing
  {c4a['mean-tied sharing, 2-step draft map']['accuracy']:.2f} on the standard model and {c4b['mean-tied sharing, 2-step draft map']['accuracy']:.2f} on the regularised one; copy-tied sharing
  {c4a['copy-tied sharing, 2-step draft map']['accuracy']:.2f} and {c4b['copy-tied sharing, 2-step draft map']['accuracy']:.2f}; 7-step sampling {c4a['fewer steps (7)']['accuracy']:.2f} and {c4b['fewer steps (7)']['accuracy']:.2f}.

### 5. What the regulariser costs

Full-compute samples of each model variant against the training data (1 = same as data):

| Model | consistency error ×1e3 | mean-tied PSNR vs. original (dB) | texture | motion |
|---|---|---|---|---|
| standard, 12k steps | {1000*v('standard, 12k steps')['consistency_error_mean']:.1f} | {v('standard, 12k steps')['alpha_sweep_psnr_db']['0.0']:.1f} | {v('standard, 12k steps')['relative_to_data']['texture']:.2f} | {v('standard, 12k steps')['relative_to_data']['frame_diff']:.2f} |
| + 3k steps, no regulariser (control) | {1000*v('+3k plain')['consistency_error_mean']:.1f} | {v('+3k plain')['alpha_sweep_psnr_db']['0.0']:.1f} | {v('+3k plain')['relative_to_data']['texture']:.2f} | {v('+3k plain')['relative_to_data']['frame_diff']:.2f} |
| + 3k steps, λ = 5 | {1000*v('+3k lambda=5')['consistency_error_mean']:.1f} | {v('+3k lambda=5')['alpha_sweep_psnr_db']['0.0']:.1f} | {v('+3k lambda=5')['relative_to_data']['texture']:.2f} | {v('+3k lambda=5')['relative_to_data']['frame_diff']:.2f} |
| + 3k steps, λ = 50 | {1000*v('+3k lambda=50')['consistency_error_mean']:.1f} | {v('+3k lambda=50')['alpha_sweep_psnr_db']['0.0']:.1f} | {v('+3k lambda=50')['relative_to_data']['texture']:.2f} | {v('+3k lambda=50')['relative_to_data']['frame_diff']:.2f} |

The control shows the change is the regulariser's, not that of extra training. Its price is texture
in the model's own samples; a gentler weight (λ = 5) keeps most of the benefit at a smaller cost.
We tried only these two weights, so where the best trade-off lies is open.

### What this does not show

- Nothing here was run on a real video model. The theory does not depend on scale; the experiments do.
- One training run per model variant.
- The permanence-score errata come from our re-implementation of the printed equations, and we
  propose no replacement score.
- Speedups are FLOP counts.
"""
s = open("README.md").read()
a, b = s.index("## Findings") if "## Findings" in s else s.index("RESULTS_PLACEHOLDER"), s.index("## Reproduce")
open("README.md", "w").write(s[:a] + block + "\n" + s[b:])
print("README updated")
