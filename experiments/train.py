"""Train the tiny video DiT with rectified flow on synthetic static-camera clips (CPU).

--collapse_aug p  turns on the sharing-aware augmentation: for a fraction p of the batch, the
noise at a random subset of the clip's static positions is replaced by its frame mean, so the
network also sees the tied states that frame sharing produces.  The regression target stays
"noise minus data" for whatever noise the token actually carries.

--consistency w   adds the sufficiency regulariser.  Every clip is presented twice with the same
noise: once untouched and once with a random subset S of its static positions mean-tied.  For an
ideal denoiser the frame mean of the first velocity on S equals the second velocity (Lemma 1 of
the paper), and elsewhere the two are equal.  The regulariser penalises the squared difference.
"""
import argparse, copy, json, math, os, sys, time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from meantie import data as D
from meantie.model import VideoDiT, n_params
from meantie.sampling import tie


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/dit.pt")
    ap.add_argument("--log", default="results/train_log.json")
    ap.add_argument("--init", default=None, help="checkpoint to fine-tune from")
    ap.add_argument("--snapshots", type=int, nargs="*", default=[], help="steps at which to keep a copy")
    ap.add_argument("--collapse_aug", type=float, default=0.0)
    ap.add_argument("--const_lr", action="store_true", help="constant learning rate after warm-up")
    ap.add_argument("--consistency", type=float, default=0.0,
                    help="weight of the sufficiency regulariser (see the module docstring)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed + 1)
    model = VideoDiT()
    if args.init:
        ck = torch.load(args.init, map_location="cpu")
        model.load_state_dict(ck["model"])
    ema = copy.deepcopy(model).requires_grad_(False)
    if args.init:
        ema.load_state_dict(ck["ema"])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99))
    warm = 300
    if args.const_lr:
        sched = lambda s: min(1.0, (s + 1) / warm)
    else:
        sched = lambda s: min(1.0, (s + 1) / warm) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * s / args.steps)))
    print(f"params: {n_params(model)/1e6:.3f}M", flush=True)
    log, t0, run = [], time.time(), 0.0
    for step in range(args.steps):
        clips, meta = D.sample_clips(args.batch, g, return_meta=True)
        x0 = D.patchify(clips)
        t = torch.rand(args.batch, generator=g)
        eps = torch.randn(x0.shape, generator=g)
        if args.collapse_aug > 0:
            keep = torch.rand(args.batch, 1, generator=g)                       # share of static positions tied
            sub = meta["static_pos"] & (torch.rand(args.batch, D.P_POS, generator=g) < keep)
            sub = sub & (torch.rand(args.batch, 1, generator=g) < args.collapse_aug)
            eps = tie(eps, sub, "mean")
        tb = t[:, None, None, None]
        if args.consistency > 0:
            keep = torch.rand(args.batch, 1, generator=g)
            sub = meta["static_pos"] & (torch.rand(args.batch, D.P_POS, generator=g) < keep)
            eps_c = tie(eps, sub, "mean")
            v = model(torch.cat([(1 - tb) * x0 + tb * eps, (1 - tb) * x0 + tb * eps_c]), torch.cat([t, t]))
            v_ind, v_col = v[:args.batch], v[args.batch:]
            fit = 0.5 * ((v_ind - (eps - x0)).pow(2).mean() + (v_col - (eps_c - x0)).pow(2).mean())
            cons = (tie(v_ind, sub, "mean") - v_col).pow(2).mean()
            loss = fit + args.consistency * cons
        else:
            xt = (1 - tb) * x0 + tb * eps
            loss = (model(xt, t) - (eps - x0)).pow(2).mean()
        for pg in opt.param_groups:
            pg["lr"] = args.lr * sched(step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        decay = 0.999 if args.init else min(0.999, (1 + step) / (10 + step))
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(decay).add_(pm, alpha=1 - decay)
        run = loss.item() if step == 0 else 0.98 * run + 0.02 * loss.item()
        if step % 250 == 0 or step == args.steps - 1:
            el = time.time() - t0
            print(f"step {step:6d}  loss {run:.4f}  {el/60:.1f} min", flush=True)
            log.append({"step": step, "loss_ema": run, "minutes": el / 60})
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save({"ema": ema.state_dict(), "model": model.state_dict(), "step": step}, args.out)
            if step in args.snapshots:
                torch.save({"ema": ema.state_dict(), "step": step}, args.out.replace(".pt", f"_step{step}.pt"))
            os.makedirs(os.path.dirname(args.log), exist_ok=True)
            json.dump({"args": vars(args), "params": n_params(model), "log": log}, open(args.log, "w"), indent=1)


if __name__ == "__main__":
    main()
