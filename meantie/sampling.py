"""Euler samplers: full compute, noise tying (copy or mean) and frame sharing.

Convention: x_t = (1-t) x_0 + t eps, t=1 is pure noise.  The model predicts
v = eps - x_0, and one Euler step is x <- x - v / K.

Frame sharing keeps one leader token (frame 0) for each static position and drops the
other F-1 copies.  It is executed here by an attention key bias (-inf on dropped keys,
+log F on the leader key), which is numerically identical to physically removing the
tokens (see tests/test_equivalence.py); FLOPs are counted for the reduced sequence.
"""
import math
import torch

from . import data as D


def tie(x, static, how):
    """Collapse the F copies of every static position to one value.

    how='copy': every copy takes the leader's (frame 0) value  -- the base paper's shared noise.
    how='mean': every copy takes the frame mean                 -- this work.
    x [B,F,P,D], static [B,P] bool.
    """
    if how == "copy":
        rep = x[:, :1]
    elif how == "mean":
        rep = x.mean(dim=1, keepdim=True)
    else:
        raise ValueError(how)
    return torch.where(static[:, None, :, None], rep.expand_as(x), x)


def share_bias(static, log_m=True):
    """Key bias [B,N] that removes non-leader static tokens and (optionally) adds log F to leaders."""
    B = static.shape[0]
    bias = torch.zeros(B, D.F_FRAMES, D.P_POS)
    if log_m:
        bias[:, 0] = torch.where(static, torch.full_like(bias[:, 0], math.log(D.F_FRAMES)), bias[:, 0])
    bias[:, 1:] = torch.where(static[:, None, :].expand(-1, D.F_FRAMES - 1, -1),
                              torch.full_like(bias[:, 1:], float("-inf")), bias[:, 1:])
    return bias.reshape(B, D.N_TOK)


@torch.no_grad()
def sample(model, eps, steps=20, static=None, tie_how=None, share=False, log_m=True,
           switch_step=0, map_fn=None, collect_x0=()):
    """Generate from noise eps [B,F,P,D].

    static: [B,P] bool map, or None.  If None and map_fn is given, the map is read from the
        model's own clean-sample prediction at the last full step before switch_step.
    tie_how: None (independent noise), 'copy' or 'mean'; applied at switch_step
        (switch_step=0 means the starting noise itself is tied).
    share: if True, static positions are computed once (token removal + proportional attention)
        from switch_step on.  If False the full sequence is computed (reference runs).
    collect_x0: step indices at which to keep the clean prediction x - t*v (in info["x0_hats"]).
    Returns (x0 [B,F,P,D], info dict).
    """
    x = eps.clone()
    B = x.shape[0]
    info = {"static": static, "x0_hats": {}}
    bias = None
    for i in range(steps):
        t = 1.0 - i / steps
        if i == switch_step and tie_how is not None:
            if static is None:
                raise ValueError("static map required at switch")
            x = tie(x, static, tie_how)
            if share:
                bias = share_bias(static, log_m)
        tt = torch.full((B,), t)
        v = model(x, tt, key_bias=bias)
        if bias is not None:
            v = torch.where(static[:, None, :, None], v[:, :1].expand_as(v), v)
        if i in collect_x0:
            info["x0_hats"][i] = x - t * v
        if map_fn is not None and static is None and i == switch_step - 1:
            static = map_fn(x - t * v)
            info["static"] = static
        x = x - v / steps
    return x, info


@torch.no_grad()
def draft_map(model, eps, thresh, draft_steps=2):
    """Static map from a short independent-noise draft (the base paper's 2-step draft)."""
    x, _ = sample(model, eps, steps=draft_steps)
    return D.static_map(x, thresh), x


def psnr(a, b, peak=2.0):
    """Per-clip PSNR in dB for clips in [-1,1]-like range (peak-to-peak 2)."""
    mse = (a - b).pow(2).flatten(1).mean(dim=1).clamp_min(1e-20)
    return 10 * torch.log10(peak ** 2 / mse)
