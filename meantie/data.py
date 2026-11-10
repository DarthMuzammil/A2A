"""Synthetic static-camera clips: smooth Gaussian-field background + 1-2 textured moving sprites.

The background is a Gaussian random field in a 6-dimensional smooth basis with known
coefficient variances.  That makes the over-dispersion predicted for copy-tied noise
directly measurable: project a generated background on the basis and compare the
coefficient spread with the reference.
"""
import math
import torch

F_FRAMES, H, W, PATCH = 8, 32, 32, 8
P_POS = (H // PATCH) * (W // PATCH)          # 16 spatial positions
N_TOK = F_FRAMES * P_POS                     # 128 tokens
D_TOK = PATCH * PATCH                        # 64 values per token (grayscale)
BG_SIGMAS = (0.30, 0.20, 0.20, 0.15, 0.12, 0.12)
BG_NAMES = ("DC", "x", "y", "xy", "cos x", "cos y")


def bg_basis(h=H, w=W):
    """[6,h,w] orthogonal smooth basis, every element scaled to unit RMS."""
    ys = (torch.arange(h) + 0.5) / h * 2 - 1
    xs = (torch.arange(w) + 0.5) / w * 2 - 1
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    B = torch.stack([torch.ones_like(X), X, Y, X * Y, torch.cos(math.pi * X), torch.cos(math.pi * Y)])
    return B / B.pow(2).mean(dim=(1, 2), keepdim=True).sqrt()


def sample_clips(n, gen=None, return_meta=False):
    """Return clips [n,F,H,W] (float32, roughly in [-1.5,1.5])."""
    g = gen
    basis = bg_basis()
    coef = torch.randn(n, len(BG_SIGMAS), generator=g) * torch.tensor(BG_SIGMAS)
    bg = torch.einsum("nk,khw->nhw", coef, basis)
    clips = bg[:, None].repeat(1, F_FRAMES, 1, 1)
    touched = torch.zeros(n, F_FRAMES, H, W)
    n_sprites = torch.randint(1, 3, (n,), generator=g)
    xs = torch.arange(W).float()
    ys = torch.arange(H).float()
    fr = torch.arange(F_FRAMES).float()
    for k in range(2):
        active = (n_sprites > k).float()
        size = torch.randint(5, 9, (n,), generator=g).float()
        vel = (torch.rand(n, 2, generator=g) * 2 - 1) * 1.0
        span = vel * (F_FRAMES - 1)
        lo = torch.clamp(-span, min=0.0)
        hi = torch.minimum(torch.tensor(float(W)) - size[:, None], float(W) - size[:, None] - span)
        p0 = lo + torch.rand(n, 2, generator=g) * (hi - lo)
        pos = p0[:, None, :] + vel[:, None, :] * fr[None, :, None]            # [n,F,2] (x,y)
        G = 4
        tex = (torch.randint(0, 2, (n, G, G), generator=g).float() * 2 - 1)
        amp = 0.6 + 0.4 * torch.rand(n, generator=g)
        x0 = pos[..., 0][..., None]                                           # [n,F,1]
        y0 = pos[..., 1][..., None]
        s = size[:, None, None]
        ax = (torch.minimum(xs + 1, x0 + s) - torch.maximum(xs, x0)).clamp(0, 1)   # [n,F,W]
        ay = (torch.minimum(ys + 1, y0 + s) - torch.maximum(ys, y0)).clamp(0, 1)   # [n,F,H]
        alpha = ay[..., :, None] * ax[..., None, :] * active[:, None, None, None]  # [n,F,H,W]
        u = ((xs + 0.5 - x0) / s * G).floor().clamp(0, G - 1).long()               # [n,F,W]
        v = ((ys + 0.5 - y0) / s * G).floor().clamp(0, G - 1).long()               # [n,F,H]
        idx = v[..., :, None] * G + u[..., None, :]                                # [n,F,H,W]
        val = torch.gather(tex.view(n, 1, G * G).expand(n, F_FRAMES, G * G), 2,
                           idx.view(n, F_FRAMES, H * W)).view(n, F_FRAMES, H, W)
        val = val * amp[:, None, None, None]
        clips = clips * (1 - alpha) + val * alpha
        touched = torch.maximum(touched, (alpha > 0).float())
    if return_meta:
        static_px = touched.amax(dim=1) == 0                                       # [n,H,W]
        static_pos = static_px.view(n, H // PATCH, PATCH, W // PATCH, PATCH).permute(0, 1, 3, 2, 4) \
            .reshape(n, P_POS, D_TOK).all(dim=-1)
        return clips, {"coef": coef, "static_pos": static_pos, "n_sprites": n_sprites}
    return clips


def patchify(clips):
    """[n,F,H,W] -> [n,F,P,D]"""
    n = clips.shape[0]
    x = clips.view(n, F_FRAMES, H // PATCH, PATCH, W // PATCH, PATCH)
    return x.permute(0, 1, 2, 4, 3, 5).reshape(n, F_FRAMES, P_POS, D_TOK)


def unpatchify(tok):
    """[n,F,P,D] -> [n,F,H,W]"""
    n = tok.shape[0]
    x = tok.view(n, F_FRAMES, H // PATCH, W // PATCH, PATCH, PATCH)
    return x.permute(0, 1, 2, 4, 3, 5).reshape(n, F_FRAMES, H, W)


def position_change(tok):
    """Largest frame-to-frame mean absolute change per spatial position. [n,F,P,D] -> [n,P]"""
    return (tok[:, 1:] - tok[:, :-1]).abs().mean(dim=-1).amax(dim=1)


def static_map(tok, thresh):
    """Boolean [n,P]: positions whose largest frame-to-frame change is below thresh."""
    return position_change(tok) < thresh


def fit_bg_coef(tok, static):
    """Least-squares background coefficients from the static positions of each clip.

    tok [n,F,P,D], static [n,P] bool.  Uses the frame-mean image restricted to static
    positions.  Returns coef [n,6] and a validity mask (enough static area for a stable fit).
    """
    n = tok.shape[0]
    img = unpatchify(tok).mean(dim=1).reshape(n, H * W)
    px = static[:, :, None].expand(n, P_POS, D_TOK).reshape(n, 1, H // PATCH, W // PATCH, PATCH, PATCH)
    px = px.permute(0, 1, 2, 4, 3, 5).reshape(n, H * W).float()
    A = bg_basis().reshape(len(BG_SIGMAS), H * W).T                               # [HW,6]
    Aw = A[None] * px[:, :, None]
    AtA = torch.einsum("npk,npl->nkl", Aw, A[None].expand(n, -1, -1))
    Atb = torch.einsum("npk,np->nk", Aw, img)
    valid = static.sum(dim=1) >= 6
    AtA = AtA + 1e-6 * torch.eye(len(BG_SIGMAS))
    coef = torch.linalg.solve(AtA, Atb)
    return coef, valid


def clip_statistics(tok, static_thresh):
    """Interpretable per-set statistics of generated clips (tokens [n,F,P,D]).

    texture:        mean |I - 3x3 box blur|, the fine detail carried mostly by the sprites
    frame_diff:     mean absolute frame-to-frame difference, i.e. how much moves
    moving_frac:    share of positions whose largest frame-to-frame change exceeds 0.1
    static_frac:    share of positions whose change is below static_thresh
    pixel_std:      overall contrast
    """
    import torch.nn.functional as Fn
    clips = unpatchify(tok)
    f = clips.reshape(-1, 1, H, W)
    blur = Fn.avg_pool2d(Fn.pad(f, (1, 1, 1, 1), mode="replicate"), 3, 1)
    ch = position_change(tok)
    return {
        "texture": float((f - blur).abs().mean()),
        "frame_diff": float((clips[:, 1:] - clips[:, :-1]).abs().mean()),
        "moving_frac": float((ch > 0.1).float().mean()),
        "static_frac": float((ch < static_thresh).float().mean()),
        "pixel_std": float(clips.std()),
    }
