"""E5 -- does the FLOP model predict wall-clock time?  (CPU, tiny model, real token removal.)

The sharing experiments execute token removal through an attention bias, which is exact
but saves no time.  Here the shortened sequences are built physically and timed.
Run this on an otherwise idle machine.
"""
import argparse, json, math, os, sys, time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from meantie import data as D
from meantie import flops as FL
from meantie.model import VideoDiT

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints/dit.pt")
ap.add_argument("--out", default="results/e5_wallclock.json")
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--reps", type=int, default=30)
ap.add_argument("--threads", type=int, default=4)
args = ap.parse_args()
torch.set_num_threads(args.threads)
model = VideoDiT()
if os.path.exists(args.ckpt):
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["ema"])
model.eval()

frame = torch.arange(D.F_FRAMES).repeat_interleave(D.P_POS)
pos = torch.arange(D.P_POS).repeat(D.F_FRAMES)
x = torch.randn(args.batch, D.N_TOK, D.D_TOK)
t = torch.full((args.batch,), 0.5)
rows = []
with torch.no_grad():
    for n_static in (0, 4, 8, 10, 12, 14, 16):
        S = torch.zeros(D.P_POS, dtype=torch.bool)
        S[:n_static] = True
        keep = ~(S[pos] & (frame > 0))
        bias = torch.where(S[pos[keep]], math.log(D.F_FRAMES), 0.0)[None].expand(args.batch, -1)
        fi, pi = frame[keep][None].expand(args.batch, -1), pos[keep][None].expand(args.batch, -1)
        xs = x[:, keep]
        for _ in range(3):
            model(xs, t, key_bias=bias, frame_idx=fi, pos_idx=pi)
        times = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            model(xs, t, key_bias=bias, frame_idx=fi, pos_idx=pi)
            times.append(time.perf_counter() - t0)
        times.sort()
        rows.append({"static_positions": n_static, "tokens": int(keep.sum()),
                     "median_ms": 1000 * times[len(times) // 2],
                     "flops": FL.forward_flops(int(keep.sum()))})
base = rows[0]
for r in rows:
    r["speedup_wallclock"] = base["median_ms"] / r["median_ms"]
    r["speedup_flops"] = base["flops"] / r["flops"]
    print(f"static {r['static_positions']:2d}/16  tokens {r['tokens']:3d}  {r['median_ms']:.1f} ms  "
          f"wall-clock x{r['speedup_wallclock']:.2f}  FLOPs x{r['speedup_flops']:.2f}")
json.dump({"batch": args.batch, "threads": args.threads, "rows": rows}, open(args.out, "w"), indent=1)
