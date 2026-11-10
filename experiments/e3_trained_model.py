"""E3 -- frame sharing on a trained video diffusion transformer: copy-tied vs mean-tied noise.

Every method is scored on the same footing: fidelity to the *original* output, i.e. the
20-step full-compute sample from the untouched independent noise.  We also report the base
paper's metric (fidelity to full compute from the method's own tied noise), background
over-dispersion and analytic FLOP speedups (which include every draft or warm-up call).
The distribution-level test lives in e4_distribution_test.py.
"""
import argparse, json, os, sys, time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from meantie import data as D
from meantie import flops as FL
from meantie.model import VideoDiT
from meantie.sampling import sample, tie, psnr, share_bias

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints/dit.pt")
ap.add_argument("--out", default="results/e3_trained_model.json")
ap.add_argument("--n", type=int, default=1024)
ap.add_argument("--n_dev", type=int, default=512)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--save_examples", default="results/e3_examples.pt")
ap.add_argument("--cache", default="", help="file to store/reuse the generated outputs of every method")
ap.add_argument("--lite", action="store_true", help="only the methods of the main table (for extra model variants)")
args = ap.parse_args()
torch.set_num_threads(args.threads)
K = 20
F = D.F_FRAMES

model = VideoDiT()
ck = torch.load(args.ckpt, map_location="cpu")
model.load_state_dict(ck["ema"])
model.eval()
print("checkpoint step", ck["step"], flush=True)


def noise(n, seed):
    return torch.randn(n, F, D.P_POS, D.D_TOK, generator=torch.Generator().manual_seed(seed))


def batched(fn, eps, *extra):
    outs = []
    for i in range(0, len(eps), args.batch):
        outs.append(fn(eps[i:i + args.batch], *[e[i:i + args.batch] for e in extra]))
    if isinstance(outs[0], tuple):
        return tuple(torch.cat([o[j] for o in outs]) for j in range(len(outs[0])))
    return torch.cat(outs)


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


# ----------------------------------------------------------------------------- calibration on dev seeds
t0 = time.time()
WARM = (2, 3, 4, 6, 8)
dev = noise(args.n_dev, 1000)


def ref_with_predictions(e):
    """Full-compute sample plus the clean predictions a switch at step k would read (made at step k-1)."""
    x, info = sample(model, e, K, collect_x0=[k - 1 for k in WARM])
    return (x,) + tuple(info["x0_hats"][k - 1] for k in WARM)


dev_out = batched(ref_with_predictions, dev)
dev_ref = dev_out[0]
# Oracle threshold: in real data static positions have exactly zero change; generated ones jitter
# slightly.  The log-change histogram of full-compute samples is bimodal (static vs moving) and we
# split it with Otsu's threshold.
ch = D.position_change(dev_ref).flatten().clamp(1e-4, 3.0).log10()
edges = torch.linspace(-4.0, 0.5, 91)
hist = torch.histc(ch, bins=90, min=-4.0, max=0.5)
centers = (edges[:-1] + edges[1:]) / 2
w0 = hist.cumsum(0); w1 = hist.sum() - w0
m0 = (hist * centers).cumsum(0) / w0.clamp_min(1); m1 = ((hist * centers).sum() - (hist * centers).cumsum(0)) / w1.clamp_min(1)
TAU_ORACLE = float(10 ** edges[1:][(w0 * w1 * (m0 - m1) ** 2).argmax()])
dev_oracle = D.static_map(dev_ref, TAU_ORACLE)
calib = {"tau_oracle": TAU_ORACLE, "oracle_static_fraction_dev": float(dev_oracle.float().mean()),
         "log10_change_hist": hist.tolist(), "log10_change_centers": centers.tolist()}
dev_draft = batched(lambda e: sample(model, e, 2)[0], dev)
calib["draft2"] = pick_thresholds(D.position_change(dev_draft), dev_oracle)
for j, k in enumerate(WARM):
    calib[f"traj{k}"] = pick_thresholds(D.position_change(dev_out[1 + j]), dev_oracle)
print("calibration", json.dumps({k: (v if not isinstance(v, dict) else
                                       {a: round(b, 4) for a, b in v.items() if not isinstance(b, dict)})
                                   for k, v in calib.items() if "hist" not in k and "centers" not in k}), flush=True)

# ----------------------------------------------------------------------------- methods
eps = noise(args.n, 2000)
ref = batched(lambda e: sample(model, e, K)[0], eps)
S_oracle = D.static_map(ref, TAU_ORACLE)
draft_out = batched(lambda e: sample(model, e, 2)[0], eps)
S_draft = D.static_map(draft_out, calib["draft2"]["threshold"])
S_draft_aggr = D.static_map(draft_out, calib["draft2"]["aggressive"]["threshold"])
print(f"reference done {time.time()-t0:.0f}s; oracle static {S_oracle.float().mean():.3f}, "
      f"draft static {S_draft.float().mean():.3f}", flush=True)


def traj_map(k):
    return lambda x0h: D.static_map(x0h, calib[f"traj{k}"]["threshold"])


def run(eps, S=None, how=None, share=True, log_m=True, switch=0, k_map=None, steps=K):
    def fn(e, *s):
        x, info = sample(model, e, steps, static=s[0] if s else None, tie_how=how, share=share, log_m=log_m,
                         switch_step=switch, map_fn=traj_map(k_map) if k_map else None)
        st = info["static"] if info["static"] is not None else torch.zeros(len(e), D.P_POS, dtype=torch.bool)
        return x, st
    return batched(fn, eps, *([S] if S is not None else []))


def cold_copy(eps, S):
    e = torch.where(S[:, None, :, None], (eps[:, :1] / F ** 0.5).expand_as(eps), eps)
    x = batched(lambda a, s: sample(model, a, K, static=s, tie_how="copy", share=True)[0], e, S)
    return x, S


if args.cache and os.path.exists(args.cache):
    METHODS = torch.load(args.cache)
    print("reusing cached outputs from", args.cache, flush=True)
else:
    METHODS = {}                      # name -> (output, static map used, warm full steps, extra full steps, steps)
    for k in (14, 10, 7, 5):
        METHODS[f"fewer steps ({k})"] = run(eps, steps=k) + (0, 0, k)
    for how in ("copy", "mean"):
        METHODS[f"{how}-tied sharing, oracle map"] = run(eps, S_oracle, how) + (0, 0, K)
        METHODS[f"{how}-tied sharing, 2-step draft map"] = run(eps, S_draft, how) + (0, 2, K)
        METHODS[f"{how}-tied sharing, 2-step draft map (aggressive)"] = run(eps, S_draft_aggr, how) + (0, 2, K)
        for k in ((8,) if args.lite else WARM):
            METHODS[f"{how}-collapse sharing at step {k}"] = run(eps, None, how, switch=k, k_map=k) + (k, 0, K)
        METHODS[f"{how}-tied full compute, oracle map (no sharing)"] = run(eps, S_oracle, how, share=False) + (K, 0, K)
        if not args.lite:
            METHODS[f"{how}-tied full compute, draft map (no sharing)"] = run(eps, S_draft, how, share=False) + (K, 0, K)
    if not args.lite:
        METHODS["mean-tied sharing, oracle map, plain attention"] = run(eps, S_oracle, "mean", log_m=False) + (0, 0, K)
        METHODS["cold-copy sharing (leader noise / sqrt F), oracle map"] = cold_copy(eps, S_oracle) + (0, 0, K)

    if args.cache:
        os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
        torch.save(METHODS, args.cache)
print(f"methods done {time.time()-t0:.0f}s", flush=True)

# ----------------------------------------------------------------------------- metrics
ref_coef, ref_valid = D.fit_bg_coef(ref, S_oracle)
res_ref_motion = float((ref[:, 1:] - ref[:, :-1]).abs().mean())
full_cost = float(FL.forward_flops(D.N_TOK))


def metrics(name):
    x, S, warm, extra, steps = METHODS[name]
    p = psnr(x, ref)
    out_static = D.static_map(x, TAU_ORACLE)
    coef, valid = D.fit_bg_coef(x, out_static)
    v = valid & ref_valid
    ratio = coef[v].std(0) / ref_coef[v].std(0)
    n_tok = FL.shared_tokens(S.sum(1)).float()
    shares = "sharing" in name and "no sharing" not in name
    if shares:
        red = torch.tensor([float(FL.forward_flops(float(n))) for n in n_tok], dtype=torch.float64)
        cost = (warm + extra) * full_cost + (steps - warm) * red
        big_full = float(FL.forward_flops(75600.0, **FL.BIG))
        big_red = torch.tensor([float(FL.forward_flops(float(n) / D.N_TOK * 75600.0, **FL.BIG)) for n in n_tok],
                               dtype=torch.float64)
        projected = float(K * big_full * len(x) / float(((warm + extra) * big_full + (steps - warm) * big_red).sum()))
    else:
        cost = torch.full((len(x),), float(steps * full_cost), dtype=torch.float64)
        projected = K / steps
    m = {
        "psnr_vs_original_db": {"median": float(p.median()), "mean": float(p.mean()),
                                "p10": float(torch.quantile(p, 0.1)), "p90": float(torch.quantile(p, 0.9))},
        "speedup_flops": float(K * full_cost * len(x) / float(cost.sum())),
        "projected_speedup_big_model": projected,
        "static_fraction_used": float(S.float().mean()),
        "map_precision_vs_oracle": float((S & S_oracle).sum() / max(S.sum().item(), 1)),
        "map_recall_vs_oracle": float((S & S_oracle).sum() / S_oracle.sum()),
        "bg_coef_std_ratio": ratio.tolist(),
        "bg_coef_std_ratio_mean": float(ratio.mean()),
        "output_static_fraction": float(out_static.float().mean()),
        "stats": D.clip_statistics(x, TAU_ORACLE),
    }
    if shares and S.any() and not S.all():
        # motion restricted to positions that were NOT shared, and the reference's jitter on the shared ones
        fd = lambda t, mask: float((t[:, 1:] - t[:, :-1]).abs().mean(-1)[mask[:, None, :].expand(-1, F - 1, -1)].mean())
        m["motion_ratio_on_unshared_positions"] = fd(x, ~S) / fd(ref, ~S)
        m["reference_jitter_share_of_motion"] = fd(ref, S) * float(S.float().mean()) / res_ref_motion
    return m


res = {"checkpoint_step": int(ck["step"]), "n": args.n, "steps": K, "calibration": calib,
       "reference": {"oracle_static_fraction": float(S_oracle.float().mean()),
                     "bg_coef_std_over_data_sigma": (ref_coef[ref_valid].std(0) / torch.tensor(D.BG_SIGMAS)).tolist(),
                     "stats": D.clip_statistics(ref, TAU_ORACLE),
                     "training_data_stats": D.clip_statistics(
                         D.patchify(D.sample_clips(args.n, torch.Generator().manual_seed(4000))), TAU_ORACLE)},
       "methods": {n: metrics(n) for n in METHODS}}

# The base paper's metric: fidelity to full compute started from the method's own tied noise.
for how in ("copy", "mean"):
    for tag, key in (("oracle map", "oracle map"), ("2-step draft map", "draft map")):
        if f"{how}-tied full compute, {key} (no sharing)" not in METHODS:
            continue
        a = METHODS[f"{how}-tied sharing, {tag}"][0]
        b = METHODS[f"{how}-tied full compute, {key} (no sharing)"][0]
        res["methods"][f"{how}-tied sharing, {tag}"]["psnr_vs_own_tied_full_compute_db"] = float(psnr(a, b).median())

for n, m in res["methods"].items():
    print(f"{n:58s} x{m['speedup_flops']:.2f}  PSNR(orig) {m['psnr_vs_original_db']['median']:5.1f}  "
          f"bg-std {m['bg_coef_std_ratio_mean']:.2f}  static {m['static_fraction_used']:.2f}  "
          f"texture x{m['stats']['texture'] / res['reference']['stats']['texture']:.2f}  "
          f"motion x{m['stats']['frame_diff'] / res['reference']['stats']['frame_diff']:.2f}"
          + (f"  PSNR(own tied) {m['psnr_vs_own_tied_full_compute_db']:.1f}" if "psnr_vs_own_tied_full_compute_db" in m else ""),
          flush=True)

# ----------------------------------------------------------------------------- settling: are static tokens colder?
# Step at which a token's clean prediction comes within 0.05 RMS of its final value, in the full-compute run.
sub = eps[:min(256, args.n)]
x = sub.clone(); preds = []
with torch.no_grad():
    for i in range(K):
        t = 1 - i / K
        v = model(x, torch.full((len(x),), t))
        preds.append(x - t * v)
        x = x - v / K
preds = torch.stack(preds)                                              # [K,B,F,P,D]
err = (preds - x[None]).pow(2).mean(-1).sqrt()                          # [K,B,F,P]
settled = (err < 0.05).flip(0).cummin(0).values.flip(0)                # stays below from this step on
settle_step = (~settled).sum(0).float()                                 # first step index at which it is settled
st = S_oracle[:len(sub)][:, None, :].expand(-1, F, -1)
res["settling"] = {
    "median_step_static": float(settle_step[st].median()), "median_step_moving": float(settle_step[~st].median()),
    "mean_step_static": float(settle_step[st].mean()), "mean_step_moving": float(settle_step[~st].mean()),
}
print("settling", res["settling"], flush=True)

# ----------------------------------------------------------------------------- proportional attention on the trained model
nb = min(256, args.n)
with torch.no_grad():
    xt = tie(0.4 * ref[:nb] + 0.6 * eps[:nb], S_oracle[:nb], "copy")
    tt = torch.full((nb,), 0.6)
    full = model(xt, tt)
    pa = {}
    for nm, lm in (("proportional", True), ("plain", False)):
        y = model(xt, tt, key_bias=share_bias(S_oracle[:nb], lm))
        y = torch.where(S_oracle[:nb][:, None, :, None], y[:, :1].expand_as(y), y)
        pa[nm] = float((y - full).pow(2).mean().sqrt() / full.pow(2).mean().sqrt())
res["proportional_attention_trained_model_rel_error"] = pa
print("prop. attention rel. error on trained model", pa, flush=True)
json.dump(res, open(args.out, "w"), indent=1)

json.dump(res, open(args.out, "w"), indent=1)
if args.save_examples:
    keep = ["copy-tied sharing, oracle map", "mean-tied sharing, oracle map",
            "copy-tied sharing, 2-step draft map", "mean-tied sharing, 2-step draft map",
            "mean-collapse sharing at step 8", "fewer steps (7)"]
    keep = [k for k in keep if k in METHODS]
    torch.save({"ref": ref[:8].clone(), "eps": eps[:8].clone(), "S_oracle": S_oracle[:8].clone(),
                **{k: METHODS[k][0][:8].clone() for k in keep},
                **{k + " | map": METHODS[k][1][:8].clone() for k in keep}}, args.save_examples)
print(f"total {time.time()-t0:.0f}s")
