"""E6 -- how far is a trained network from the ideal denoiser's invariance, and what does fixing it cost?

For every checkpoint we report
  consistency error : mean squared difference between the frame-mean of the velocity on an untouched
                      input and the velocity on the mean-tied input (zero for an ideal denoiser);
  alpha sweep       : PSNR against the original sample when the frame deviations of the static
                      starting noise are scaled by alpha (alpha=1 untouched, alpha=0 mean-tied);
                      an ideal denoiser gives the same sample for every alpha;
  generator quality : texture, motion and background spread of full-compute samples relative to
                      the training data.
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from meantie import data as D
from meantie.model import VideoDiT
from meantie.sampling import sample, tie, psnr

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", nargs="+", required=True, help="name=path pairs")
ap.add_argument("--out", default="results/e6_sufficiency.json")
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--threads", type=int, default=4)
args = ap.parse_args()
torch.set_num_threads(args.threads)
K, TAU = 20, 0.03

g = torch.Generator().manual_seed(77)
clips, meta = D.sample_clips(1024, g, return_meta=True)
x0 = D.patchify(clips)
eps_d = torch.randn(x0.shape, generator=g)
S_d = meta["static_pos"]
data_stats = D.clip_statistics(x0, TAU)
sigma = torch.tensor(D.BG_SIGMAS)
eps = torch.randn(args.n, D.F_FRAMES, D.P_POS, D.D_TOK, generator=torch.Generator().manual_seed(78))

res = {"training_data_stats": data_stats, "models": {}}
if os.path.exists(args.out):
    res["models"] = json.load(open(args.out)).get("models", {})
for item in args.ckpts:
    name, path = item.rsplit("=", 1)
    model = VideoDiT()
    model.load_state_dict(torch.load(path, map_location="cpu")["ema"])
    model.eval()
    r = {}
    with torch.no_grad():
        cons, fit = {}, {}
        for t in (0.95, 0.8, 0.6, 0.4, 0.2):
            c = f = 0.0
            for i in range(0, 1024, 256):
                sl = slice(i, i + 256)
                tt = torch.full((256,), t)
                e, ec = eps_d[sl], tie(eps_d[sl], S_d[sl], "mean")
                vi = model((1 - t) * x0[sl] + t * e, tt)
                vc = model((1 - t) * x0[sl] + t * ec, tt)
                c += (tie(vi, S_d[sl], "mean") - vc).pow(2).mean().item() / 4
                f += (vi - (e - x0[sl])).pow(2).mean().item() / 4
            cons[str(t)], fit[str(t)] = c, f
        r["consistency_error"] = cons
        r["consistency_error_mean"] = sum(cons.values()) / len(cons)
        r["fit_loss_mean"] = sum(fit.values()) / len(fit)
    run = lambda e: torch.cat([sample(model, e[i:i + 256], K)[0] for i in range(0, len(e), 256)])
    ref = run(eps)
    S = D.static_map(ref, TAU)
    mean = tie(eps, S, "mean")
    r["alpha_sweep_psnr_db"] = {str(a): float(psnr(run(mean + a * (eps - mean)), ref).median()) for a in (0.9, 0.5, 0.0)}
    r["copy_tied_psnr_db"] = float(psnr(run(tie(eps, S, "copy")), ref).median())
    st = D.clip_statistics(ref, TAU)
    coef, valid = D.fit_bg_coef(ref, S)
    r["stats"] = st
    r["relative_to_data"] = {k: st[k] / data_stats[k] for k in st}
    r["bg_spread_over_data"] = float((coef[valid].std(0) / sigma).mean())
    res["models"][name] = r
    print(f"{name:28s} cons {r['consistency_error_mean']:.5f}  fit {r['fit_loss_mean']:.4f} | alpha 0.9/0.5/0: "
          f"{r['alpha_sweep_psnr_db']['0.9']:.1f}/{r['alpha_sweep_psnr_db']['0.5']:.1f}/{r['alpha_sweep_psnr_db']['0.0']:.1f} dB"
          f" | copy {r['copy_tied_psnr_db']:.1f} dB | texture x{r['relative_to_data']['texture']:.2f} "
          f"motion x{r['relative_to_data']['frame_diff']:.2f} moving x{r['relative_to_data']['moving_frac']:.2f} "
          f"bg x{r['bg_spread_over_data']:.2f}", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
