"""E4 -- does an accelerated sampler still draw from the model's distribution?

Classifier two-sample test (C2ST) of each sampler against 20-step full compute, on fresh
seeds.  0.5 = the classifier cannot tell the two sets apart.  Thresholds for the static
maps are read from the E3 calibration.
"""
import argparse, json, os, sys, time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from meantie import data as D
from meantie.c2st import c2st
from meantie.model import VideoDiT
from meantie.sampling import sample

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints/dit.pt")
ap.add_argument("--calib", default="results/e3_trained_model.json")
ap.add_argument("--out", default="results/e4_distribution_test.json")
ap.add_argument("--n", type=int, default=2048)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--clf_seeds", type=int, default=3)
ap.add_argument("--noise_floor", type=float, default=0.03)
args = ap.parse_args()
torch.set_num_threads(args.threads)
K, F = 20, D.F_FRAMES
model = VideoDiT()
model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["ema"])
model.eval()
calib = json.load(open(args.calib))["calibration"]


def gen(seed, fn):
    eps = torch.randn(args.n, F, D.P_POS, D.D_TOK, generator=torch.Generator().manual_seed(seed))
    return D.unpatchify(torch.cat([fn(eps[i:i + args.batch]) for i in range(0, args.n, args.batch)]))


def full(steps):
    return lambda e: sample(model, e, steps)[0]


def draft(how):
    def fn(e):
        S = D.static_map(sample(model, e, 2)[0], calib["draft2"]["threshold"])
        return sample(model, e, K, static=S, tie_how=how, share=True)[0]
    return fn


def collapse(how, k):
    th = calib[f"traj{k}"]["threshold"]
    return lambda e: sample(model, e, K, tie_how=how, share=True, switch_step=k,
                            map_fn=lambda x0h: D.static_map(x0h, th))[0]


t0 = time.time()
base = gen(3000, full(K))
sets = {
    "full compute, fresh seeds (control)": (3001, full(K)),
    "fewer steps (10)": (3002, full(10)),
    "fewer steps (7)": (3003, full(7)),
    "copy-tied sharing, 2-step draft map": (3004, draft("copy")),
    "mean-tied sharing, 2-step draft map": (3005, draft("mean")),
    "copy-collapse sharing at step 2": (3006, collapse("copy", 2)),
    "mean-collapse sharing at step 2": (3007, collapse("mean", 2)),
    "mean-collapse sharing at step 4": (3008, collapse("mean", 4)),
}
res = {"n_per_class": args.n, "classifier_seeds": args.clf_seeds, "noise_floor": args.noise_floor, "vs_full_compute": {}}
for name, (seed, fn) in sets.items():
    xs = gen(seed, fn)
    accs = [c2st(base, xs, seed=s, noise_floor=args.noise_floor) for s in range(args.clf_seeds)]
    mean = sum(a for a, _ in accs) / len(accs)
    res["vs_full_compute"][name] = {"accuracy": mean, "per_seed": [a for a, _ in accs], "binomial_se": accs[0][1]}
    print(f"C2ST {name:42s} acc {mean:.3f} {[round(a, 3) for a, _ in accs]}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
real = D.sample_clips(args.n, torch.Generator().manual_seed(4000))
a, se = c2st(base, real, seed=0, noise_floor=args.noise_floor)
res["model_vs_training_data"] = {"accuracy": a, "binomial_se": se}
print("C2ST model vs training data", round(a, 3), flush=True)
json.dump(res, open(args.out, "w"), indent=1)
