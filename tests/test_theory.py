"""Theorem 1 on a tiny exact denoiser: averaging the noise over the frames of a static region
leaves the Euler sample unchanged; copying one frame's noise does not."""
import os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from meantie.sampling import tie

F, P, Dm, N, K = 4, 6, 3, 40, 12


def _setup(seed=0):
    g = torch.Generator().manual_seed(seed)
    data = torch.randn(N, F, P, Dm, generator=g, dtype=torch.float64)
    S = torch.zeros(P, dtype=torch.bool)
    S[:4] = True                                             # the same static set for every clip
    data[:, :, S] = data[:, :1, S]                           # static there: identical in all frames
    X0 = data.reshape(N, -1)
    return g, X0, S


def _sample(x, X0):
    for i in range(K):
        t = 1.0 - i / K
        xf = x.reshape(x.shape[0], -1)
        if t >= 1.0:
            w = torch.full((x.shape[0], N), 1.0 / N, dtype=torch.float64)
        else:
            w = torch.softmax(((1 - t) / t ** 2) * (xf @ X0.T) - ((1 - t) ** 2 / (2 * t ** 2)) * X0.pow(2).sum(1), dim=1)
        x = x - (x - (w @ X0).view_as(x)) / t / K
    return x


def test_mean_tying_preserves_the_sample_and_copying_does_not():
    g, X0, S = _setup()
    eps = torch.randn(16, F, P, Dm, generator=g, dtype=torch.float64)
    Sb = S[None].expand(16, -1)
    ref = _sample(eps, X0)
    mean = _sample(tie(eps, Sb, "mean"), X0)
    copy = _sample(tie(eps, Sb, "copy"), X0)
    assert (mean - ref).abs().max().item() < 1e-9
    assert (copy - ref).abs().max().item() > 1e-2


def test_mid_generation_collapse_is_lossless():
    g, X0, S = _setup(1)
    x = torch.randn(8, F, P, Dm, generator=g, dtype=torch.float64)
    Sb = S[None].expand(8, -1)
    ref = _sample(x, X0)
    y = x.clone()
    for i in range(K):                                       # same sampler, collapse inserted at step 5
        t = 1.0 - i / K
        if i == 5:
            y = tie(y, Sb, "mean")
        yf = y.reshape(8, -1)
        w = torch.full((8, N), 1.0 / N, dtype=torch.float64) if t >= 1.0 else torch.softmax(
            ((1 - t) / t ** 2) * (yf @ X0.T) - ((1 - t) ** 2 / (2 * t ** 2)) * X0.pow(2).sum(1), dim=1)
        y = y - (y - (w @ X0).view_as(y)) / t / K
    assert (y - ref).abs().max().item() < 1e-9
