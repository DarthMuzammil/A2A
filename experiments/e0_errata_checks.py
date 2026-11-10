"""E0 -- numerical checks of the errata we raise against the base paper.

Every check below is self-contained: it uses only the equations printed in the base paper
(re-implemented here from the text; we do not have the authors' code) or a random
transformer.  Checks:

 (a) Eq. 3 null bias: "motion explained by noise alone costs nothing" is false.
 (b) Clip score max_t PVS_t grows with clip length: null-vs-null AUC = T1/(T1+T2).
 (c) Eq. 3 charges a pixel without bound, so Eq. 4 is not a fraction (PVS_t > 1 for a legal
     lighting change once the paper's prefilter is on), and the score of one and the same
     event scales inversely with the clip's noise level.
 (d) Eq. 5 (proportional attention is exact) holds only without temporal position
     information.
 (e) The projected speedups are capped by the fixed cost of the draft pass.
 (f) Standard error of the reported AUCs (Hanley-McNeil).
 (g) The settling step that pooling alone predicts for static tokens.
"""
import json, math, os, sys
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from meantie import data as D
from meantie.model import VideoDiT
from meantie.sampling import share_bias, tie

rng = np.random.default_rng(0)
res = {}
LOG2E = 1 / math.log(2)

# ---------------------------------------------------------------- (a) null bias of Eq. 3
def created_content(r, b=None, dead=0.0):
    """Eq. 3 of the base paper, plus the dead zone of Eq. 4."""
    if b is None:
        b = np.median(np.abs(r)) / math.log(2)
    s = np.maximum(np.abs(r) / (b * math.log(2)) - LOG2E, 0.0)
    return np.maximum(s - dead, 0.0)

n = 4_000_000
lap = rng.laplace(0.0, 1.0, n)
gau = rng.normal(0.0, 1.0, n)
a = {"analytic_laplace_bits_per_px": 1 / (math.e * math.log(2)), "by_dead_zone": {}}
for dead in (0.0, 0.5, 1.0, 2.0):
    a["by_dead_zone"][str(dead)] = {
        "laplace_monte_carlo": float(created_content(lap, dead=dead).mean()),
        "laplace_analytic": 2 ** (-dead) / (math.e * math.log(2)),
        "gaussian_monte_carlo": float(created_content(gau, dead=dead).mean()),
    }
# What PVS_t does this floor imply for a clip with no event at all?
a["implied_pvs_floor_at_4_bits_per_px"] = a["analytic_laplace_bits_per_px"] / 4.0
a["dead_zone_needed_for_floor_0.005"] = math.log2(a["implied_pvs_floor_at_4_bits_per_px"] / 0.005)
res["a_null_bias"] = a
print("(a) E[s] under the paper's own Laplacian null: analytic %.4f, MC %.4f bits/px (claimed: 0)"
      % (a["analytic_laplace_bits_per_px"], a["by_dead_zone"]["0.0"]["laplace_monte_carlo"]))

# ---------------------------------------------------------------- (b) length bias of the max
def null_auc(T1, T2, rho, n=20000):
    """P(max of T1 null frame scores > max of T2 null frame scores), AR(1)-correlated log-normal scores."""
    def clip_max(T):
        z = np.empty((n, T))
        z[:, 0] = rng.normal(size=n)
        for t in range(1, T):
            z[:, t] = rho * z[:, t - 1] + math.sqrt(1 - rho ** 2) * rng.normal(size=n)
        return np.exp(0.5 * z).max(axis=1)
    return float((clip_max(T1) > clip_max(T2)).mean())

b = {"closed_form_iid": {}, "simulated": {}}
for T1, T2 in ((127, 15), (63, 15), (31, 15)):
    b["closed_form_iid"][f"{T1}_vs_{T2}"] = T1 / (T1 + T2)
    b["simulated"][f"{T1}_vs_{T2}"] = {f"rho={rho}": null_auc(T1, T2, rho) for rho in (0.0, 0.5, 0.9)}
res["b_length_bias"] = b
print("(b) null-vs-null AUC, 127 vs 15 frame pairs: closed form %.3f, simulated %s"
      % (b["closed_form_iid"]["127_vs_15"], b["simulated"]["127_vs_15"]))

# ---------------------------------------------------------------- (c) PVS is not a bounded fraction
def loco_density(img):
    """Eq. 1: bits/pixel of the LOCO-I MED residual under a discretised Laplacian whose scale
    is the local mean absolute residual (7x7 box)."""
    I = img.astype(np.float64)
    a_ = np.roll(I, 1, axis=1); b_ = np.roll(I, 1, axis=0); c_ = np.roll(np.roll(I, 1, axis=0), 1, axis=1)
    pred = np.where(c_ >= np.maximum(a_, b_), np.minimum(a_, b_),
                    np.where(c_ <= np.minimum(a_, b_), np.maximum(a_, b_), a_ + b_ - c_))
    e = I - pred
    k = 7
    pad = np.pad(np.abs(e), k // 2, mode="reflect")
    cs = np.cumsum(np.cumsum(np.pad(pad, ((1, 0), (1, 0))), axis=0), axis=1)
    lam = (cs[k:, k:] - cs[:-k, k:] - cs[k:, :-k] + cs[:-k, :-k]) / (k * k)
    lam = np.maximum(lam, 0.05)
    cdf = lambda u: np.where(u < 0, 0.5 * np.exp(u / lam), 1 - 0.5 * np.exp(-u / lam))
    p = np.clip(cdf(e + 0.5) - cdf(e - 0.5), 1e-12, 1.0)
    return -np.log2(p)[1:, 1:]            # drop the wrap-around row/column

def pvs_pair(f0, f1, dead, cap=False, prefilter=0.0):
    """Eq. 3-4 for a static camera (zero flow, no motion boundary, so the legal set is empty).

    prefilter: sigma of the Gaussian blur the base paper applies to compressed footage.
    Returns (PVS_t, largest per-pixel charge in bits).
    """
    f0 = f0.astype(np.float64); f1 = f1.astype(np.float64)
    if prefilter > 0:
        f0 = gaussian_filter(f0, prefilter, mode="reflect"); f1 = gaussian_filter(f1, prefilter, mode="reflect")
    r = f1 - f0
    s = created_content(r, dead=0.0)[1:, 1:]
    if cap:                               # a code may always fall back to intra-coding the pixel
        s = np.minimum(s, loco_density(np.rint(f1)))
    return float(np.maximum(s - dead, 0).sum() / loco_density(np.rint(f0)).sum()), float(s.max())

def scene(light=0.0, noise=1.5, seed=0):
    """A smooth 128x128 night-like scene (two frames), sensor noise, optional soft spotlight in frame 1."""
    r_ = np.random.default_rng(seed)
    y, x = np.mgrid[0:128, 0:128] / 128.0
    base = 40 + 25 * x + 15 * np.sin(4 * y) + 10 * np.cos(7 * x * y)
    spot = light * np.exp(-((x - 0.6) ** 2 + (y - 0.4) ** 2) / (2 * 0.12 ** 2))
    f0 = base + r_.normal(0, noise, base.shape)
    f1 = base + spot + r_.normal(0, noise, base.shape)
    q = lambda f: np.clip(np.rint(f), 0, 255)
    return q(f0), q(f1)

DEAD = 0.5
c = {"dead_zone": DEAD, "lighting_change": {}, "noise_level": {}}
for prefilter in (0.0, 2.0):
    f0, f1 = scene(light=45.0)
    g0, g1 = scene(light=0.0)
    written, peak = pvs_pair(f0, f1, DEAD, prefilter=prefilter)
    c["lighting_change"][f"prefilter={prefilter}"] = {
        "pvs_as_written": written,
        "largest_per_pixel_charge_bits": peak,
        "pvs_capped_by_intra_cost": pvs_pair(f0, f1, DEAD, cap=True, prefilter=prefilter)[0],
        "pvs_no_event": pvs_pair(g0, g1, DEAD, prefilter=prefilter)[0],
        "mean_density_bits_per_px": float(loco_density(np.rint(gaussian_filter(f0, prefilter)) if prefilter else f0).mean()),
    }
# The same event (same amplitude, same scene) under different sensor-noise levels.
for noise in (3.0, 1.5, 0.75):
    f0, f1 = scene(light=45.0, noise=noise)
    c["noise_level"][f"sigma={noise}"] = {"pvs_as_written": pvs_pair(f0, f1, DEAD)[0],
                                          "pvs_capped_by_intra_cost": pvs_pair(f0, f1, DEAD, cap=True)[0]}
res["c_unbounded_and_noise_scale"] = c
for k, v in c["lighting_change"].items():
    print("(c) legal spotlight, %s: PVS_t = %.2f as written (peak charge %.0f bits/px), capped %.2f, no event %.3f"
          % (k, v["pvs_as_written"], v["largest_per_pixel_charge_bits"], v["pvs_capped_by_intra_cost"], v["pvs_no_event"]))
print("    same event at noise sigma 3, 1.5, 0.75: PVS_t = %s"
      % [round(c["noise_level"][k]["pvs_as_written"], 3) for k in c["noise_level"]])

# ---------------------------------------------------------------- (d) proportional attention vs position embeddings
torch.manual_seed(0)
d_res = {}
for temporal_pe in (False, True):
    m = VideoDiT(temporal_pe=temporal_pe).double()
    with torch.no_grad():
        for p in m.parameters():                     # random, non-degenerate weights everywhere
            p.copy_(torch.randn_like(p) * (0.5 if p.dim() == 1 else 1.0 / math.sqrt(p.shape[-1])))
        m.pos_s.mul_(8.0); m.pos_t.mul_(8.0)
    B = 16
    static = torch.rand(B, D.P_POS) < 0.7
    x = tie(torch.randn(B, D.F_FRAMES, D.P_POS, D.D_TOK, dtype=torch.float64), static, "copy")
    t = torch.full((B,), 0.6, dtype=torch.float64)
    with torch.no_grad():
        full = m(x, t)
        out = {}
        for name, log_m in (("proportional", True), ("plain", False)):
            y = m(x, t, key_bias=share_bias(static, log_m).double())
            y = torch.where(static[:, None, :, None], y[:, :1].expand_as(y), y)
            out[name] = float((y - full).pow(2).mean().sqrt() / full.pow(2).mean().sqrt())
    d_res["temporal_pe" if temporal_pe else "no_temporal_pe"] = out
    print("(d) temporal PE %-5s relative error of merged vs full: proportional %.2e, plain %.2e"
          % (temporal_pe, out["proportional"], out["plain"]))
res["d_proportional_attention"] = d_res

# ---------------------------------------------------------------- (e) what the draft pass costs at scale
from meantie import flops as FL
e_res = {}
for static_fraction in (0.73, 0.90):
    frac = 1 - 7 / 8 * static_fraction                       # tokens left when 8 frames share
    full = FL.forward_flops(75600.0, **FL.BIG)
    red = FL.forward_flops(frac * 75600.0, **FL.BIG)
    e_res[f"static={static_fraction}"] = {
        "projected_speedup_with_2_step_draft": 20 * full / (2 * full + 20 * red),
        "projected_speedup_without_draft": full / red,
        "draft_share_of_cost": 2 * full / (2 * full + 20 * red),
    }
res["e_draft_cost_projection"] = e_res
print("(e) projected speedup with / without the 2-step draft, and the draft's share of the cost:",
      {k: [round(v["projected_speedup_with_2_step_draft"], 2), round(v["projected_speedup_without_draft"], 2),
           round(v["draft_share_of_cost"], 2)] for k, v in e_res.items()})

# ---------------------------------------------------------------- (f) how uncertain is an AUC of 0.8?
def hanley_mcneil_se(auc, n_pos, n_neg):
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    return math.sqrt((auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg))

res["f_auc_standard_error"] = {"auc_0.80_150_per_class": hanley_mcneil_se(0.80, 150, 150),
                               "auc_0.86_180_per_class": hanley_mcneil_se(0.86, 180, 180)}
print("(f) standard error of an AUC of 0.80 with 150 clips per class: %.3f" % res["f_auc_standard_error"]["auc_0.80_150_per_class"])

# ---------------------------------------------------------------- (g) the settling prediction of the sqrt(F) clock
def static_step_from_moving(step_moving, frames=8, steps=20):
    t = 1 - step_moving / steps
    r = math.sqrt(frames)
    return (1 - t * r / ((1 - t) + t * r)) * steps

res["g_clock_prediction"] = {"moving_settles_at_step_7_static_predicted": static_step_from_moving(7)}
print("(g) if moving tokens settle at step 7 of 20, static tokens of an 8-frame clip should settle at step %.1f"
      % res["g_clock_prediction"]["moving_settles_at_step_7_static_predicted"])

os.makedirs("results", exist_ok=True)
json.dump(res, open("results/e0_errata_checks.json", "w"), indent=1)
