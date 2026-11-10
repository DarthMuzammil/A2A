"""E2 -- noise tying under an exact, nonlinear ideal denoiser (no network, no training).

For a finite data set the ideal denoiser is a softmax-weighted average of the data
(posterior mean under the empirical distribution).  Sampling with it lands on one data
clip, so we can ask a sharp question: after tying the noise on the static positions of
the clip that independent noise would have produced, does the sampler still produce
*that* clip?  The data set mixes clips with different static sets, so this also probes
the mixture case that the exactness proposition does not cover.
"""
import json, os, sys, time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from meantie import data as D
from meantie.sampling import tie, psnr

torch.set_num_threads(int(os.environ.get("E2_THREADS", "1")))
N_DATA, N_SEEDS, BATCH, STEPS = 4096, 1024, 256, 20
g = torch.Generator().manual_seed(123)
clips, meta = D.sample_clips(N_DATA, g, return_meta=True)
X0 = D.patchify(clips).reshape(N_DATA, -1)
X0_sq = X0.pow(2).sum(1)
F = D.F_FRAMES


def denoise(x, t):
    """Posterior mean of the clean clip and the posterior weights. x [B,F,P,D]."""
    xf = x.reshape(x.shape[0], -1)
    if t >= 1.0:
        w = torch.full((x.shape[0], N_DATA), 1.0 / N_DATA)
    else:
        logits = ((1 - t) / t ** 2) * (xf @ X0.T) - ((1 - t) ** 2 / (2 * t ** 2)) * X0_sq
        w = torch.softmax(logits, dim=1)
    return (w @ X0).view_as(x), w


def run(eps, static=None, how=None, switch=0, map_thresh=None):
    """Euler sampler.  With static=None the map is read at the switch step from the clean
    prediction of the previous step, so it costs no extra denoiser call."""
    x = eps.clone()
    w = x0h = None
    for i in range(STEPS):
        t = 1.0 - i / STEPS
        if how is not None and i == switch:
            if static is None:
                static = D.static_map(x0h, map_thresh)
            x = tie(x, static, how)
        x0h, w = denoise(x, t)
        x = x - (x - x0h) / t / STEPS
    return x, w.argmax(dim=1), static


def summarize(x, idx, ref_x, ref_idx, static):
    p = psnr(x, ref_x)
    coef = meta["coef"][idx]
    ref_coef = meta["coef"][ref_idx]
    covered = (meta["static_pos"][idx] | ~static).all(dim=1)       # is the produced clip static on S?
    return {
        "same_clip_rate": float((idx == ref_idx).float().mean()),
        "median_psnr_vs_independent_db": float(p.median()),
        "bg_coef_std_ratio": (coef.std(0) / ref_coef.std(0)).tolist(),
        "bg_coef_std_ratio_mean": float((coef.std(0) / ref_coef.std(0)).mean()),
        "output_static_on_S_rate": float(covered.float().mean()),
        "static_fraction": float(static.float().mean()),
    }


def x0_hat_traj(eps, k):
    """Clean prediction made at step k-1 of the independent-noise sampler (what a switch at step k reads)."""
    x = eps.clone()
    for i in range(k):
        t = 1.0 - i / STEPS
        x0h = denoise(x, t)[0]
        x = x - (x - x0h) / t / STEPS
    return x0h


def draft2(eps):
    """2-step Euler draft from independent noise (the base paper's draft pass)."""
    x = eps - 0.5 * (eps - denoise(eps, 1.0)[0])
    return denoise(x, 0.5)[0]


def pick_thresholds(change, target, min_precision=0.95):
    """Two operating points for a static map read from `change`, scored against the oracle map.

    conservative: the largest threshold whose precision is at least min_precision (a position
        wrongly marked static freezes motion, a missed one only costs speed);
    aggressive:   the threshold with the best F1.
    """
    rows = []
    for q in torch.linspace(0.02, 0.98, 97):
        th = float(torch.quantile(change.flatten(), q))
        pred = change < th
        tp = (pred & target).sum().item()
        prec, rec = tp / max(pred.sum().item(), 1), tp / max(target.sum().item(), 1)
        rows.append({"threshold": th, "precision": prec, "recall": rec,
                     "f1": 2 * prec * rec / max(prec + rec, 1e-9)})
    ok = [r for r in rows if r["precision"] >= min_precision]
    cons = max(ok, key=lambda r: r["recall"]) if ok else max(rows, key=lambda r: r["precision"])
    return {**cons, "aggressive": max(rows, key=lambda r: r["f1"])}


# thresholds for the draft map and the in-trajectory maps, tuned on separate dev seeds
WARM = (2, 3, 4, 6)
dev = torch.randn(BATCH, F, D.P_POS, D.D_TOK, generator=torch.Generator().manual_seed(99))
_, dev_idx, _ = run(dev)
dev_S = meta["static_pos"][dev_idx]
calib = {"draft2": pick_thresholds(D.position_change(draft2(dev)), dev_S)}
for k in WARM:
    calib[f"traj{k}"] = pick_thresholds(D.position_change(x0_hat_traj(dev, k)), dev_S)
print("calibration", {k: {a: (round(b, 3) if not isinstance(b, dict) else {c: round(d, 3) for c, d in b.items()})
                             for a, b in v.items()} for k, v in calib.items()}, flush=True)

gs = torch.Generator().manual_seed(7)
acc = {}
t0 = time.time()
for b in range(N_SEEDS // BATCH):
    eps = torch.randn(BATCH, F, D.P_POS, D.D_TOK, generator=gs)
    ref_x, ref_idx, _ = run(eps)
    S = meta["static_pos"][ref_idx]                                # exact static set of the reference clip
    dr = draft2(eps)
    S_draft = D.static_map(dr, calib["draft2"]["threshold"])
    S_draft_aggr = D.static_map(dr, calib["draft2"]["aggressive"]["threshold"])
    cold = torch.where(S[:, None, :, None], (eps[:, :1] / F ** 0.5).expand_as(eps), eps)
    cx, cidx, _ = run(cold)
    runs = {
        "copy-tied, oracle map": run(eps, S, "copy"),
        "mean-tied, oracle map": run(eps, S, "mean"),
        "cold copy (leader/sqrt F), oracle map": (cx, cidx, S),
        "copy-tied, 2-step draft map": run(eps, S_draft, "copy"),
        "mean-tied, 2-step draft map": run(eps, S_draft, "mean"),
        "copy-tied, 2-step draft map (aggressive)": run(eps, S_draft_aggr, "copy"),
        "mean-tied, 2-step draft map (aggressive)": run(eps, S_draft_aggr, "mean"),
    }
    for k in WARM:
        th = calib[f"traj{k}"]["threshold"]
        runs[f"copy-collapse at step {k}, map from trajectory"] = run(eps, None, "copy", k, th)
        runs[f"mean-collapse at step {k}, map from trajectory"] = run(eps, None, "mean", k, th)
    for name, (x, idx, st) in runs.items():
        acc.setdefault(name, []).append((x, idx, st, ref_x, ref_idx))
    acc.setdefault("_ref", []).append((ref_x, ref_idx, S, ref_x, ref_idx))
    print(f"batch {b} done {time.time()-t0:.0f}s", flush=True)

res = {"n_data": N_DATA, "n_seeds": N_SEEDS, "steps": STEPS, "calibration": calib, "methods": {}}
for name, parts in acc.items():
    x, idx, st, rx, ri = (torch.cat([p[j] for p in parts]) for j in range(5))
    if name == "_ref":
        dc = meta["coef"][idx]
        res["reference"] = {"static_fraction": float(st.float().mean()),
                            "bg_coef_std_over_data_sigma": (dc.std(0) / torch.tensor(D.BG_SIGMAS)).tolist()}
        continue
    res["methods"][name] = summarize(x, idx, rx, ri, st)
    m = res["methods"][name]
    print(f"{name:48s} same-clip {m['same_clip_rate']:.3f}  PSNR {m['median_psnr_vs_independent_db']:6.1f} dB  "
          f"bg std ratio {m['bg_coef_std_ratio_mean']:.2f}  static frac {m['static_fraction']:.2f}")
json.dump(res, open("results/e2_ideal_denoiser.json", "w"), indent=1)
