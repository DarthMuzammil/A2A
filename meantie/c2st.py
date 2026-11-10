"""Classifier two-sample test: can a small 3D CNN tell two sets of clips apart?

Held-out accuracy 0.5 means the classifier found no difference.  It is a lower bound on
distinguishability (a better classifier could find more), so it is reported next to a
same-distribution control.

Noise floor.  Frame sharing makes shared positions bit-identical across frames, whereas full
compute leaves a faint frame-to-frame jitter there (a model imperfection: the training data are
exactly static).  A classifier could separate the two sets by that jitter alone.  We therefore
add white noise of standard deviation `noise_floor` to every clip, on both sides, so that only
differences above the floor count.
"""
import torch
import torch.nn as nn


def _net():
    return nn.Sequential(
        nn.Conv3d(2, 16, 3, stride=(1, 2, 2), padding=1), nn.ReLU(),
        nn.Conv3d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(64, 1))


def _feat(x):
    """[n,F,H,W] -> [n,2,F,H,W]: the clip and its frame-to-frame difference."""
    d = torch.zeros_like(x)
    d[:, 1:] = (x[:, 1:] - x[:, :-1]) * 5.0
    return torch.stack([x, d], dim=1)


def c2st(xa, xb, seed=0, epochs=8, test_frac=0.25, batch=128, noise_floor=0.03):
    """Return held-out accuracy and its binomial standard error."""
    g = torch.Generator().manual_seed(seed)
    if noise_floor > 0:
        xa = xa + noise_floor * torch.randn(xa.shape, generator=g)
        xb = xb + noise_floor * torch.randn(xb.shape, generator=g)
    n = min(len(xa), len(xb))
    xa, xb = xa[torch.randperm(len(xa), generator=g)[:n]], xb[torch.randperm(len(xb), generator=g)[:n]]
    n_te = int(n * test_frac)
    xtr = torch.cat([xa[n_te:], xb[n_te:]]); ytr = torch.cat([torch.zeros(n - n_te), torch.ones(n - n_te)])
    xte = torch.cat([xa[:n_te], xb[:n_te]]); yte = torch.cat([torch.zeros(n_te), torch.ones(n_te)])
    torch.manual_seed(seed)
    net = _net()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    steps = epochs * (len(xtr) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    for _ in range(epochs):
        perm = torch.randperm(len(xtr), generator=g)
        for i in range(len(xtr) // batch):
            idx = perm[i * batch:(i + 1) * batch]
            loss = nn.functional.binary_cross_entropy_with_logits(net(_feat(xtr[idx])).squeeze(1), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        pred = torch.cat([net(_feat(xte[i:i + 512])).squeeze(1) for i in range(0, len(xte), 512)]) > 0
    acc = float((pred.float() == yte).float().mean())
    return acc, (acc * (1 - acc) / len(yte)) ** 0.5
