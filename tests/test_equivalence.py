"""Sanity tests.  Run with:  python -m pytest -q tests"""
import math, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from meantie import data as D
from meantie import flops as FL
from meantie.model import VideoDiT
from meantie.sampling import sample, share_bias, tie


def _random_model(temporal_pe=True, seed=0):
    torch.manual_seed(seed)
    m = VideoDiT(temporal_pe=temporal_pe).double()
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(torch.randn_like(p) * (0.5 if p.dim() == 1 else 1.0 / math.sqrt(p.shape[-1])))
    return m.eval()


def test_patchify_roundtrip():
    c = D.sample_clips(8, torch.Generator().manual_seed(0))
    assert torch.equal(D.unpatchify(D.patchify(c)), c)


def test_static_map_matches_generator():
    c, meta = D.sample_clips(256, torch.Generator().manual_seed(1), return_meta=True)
    assert torch.equal(D.static_map(D.patchify(c), 1e-6), meta["static_pos"])


def test_tie_statistics():
    eps = torch.randn(4000, D.F_FRAMES, D.P_POS, D.D_TOK)
    S = torch.ones(4000, D.P_POS, dtype=torch.bool)
    assert abs(tie(eps, S, "copy").var().item() - 1.0) < 0.02
    assert abs(tie(eps, S, "mean").var().item() - 1.0 / D.F_FRAMES) < 0.01


def test_key_bias_equals_physical_token_removal():
    """-inf / +log F key bias on the full grid == running the model on the gathered short sequence."""
    m = _random_model()
    S = torch.zeros(1, D.P_POS, dtype=torch.bool)
    S[0, [0, 3, 4, 9, 15]] = True
    x = tie(torch.randn(1, D.F_FRAMES, D.P_POS, D.D_TOK, dtype=torch.float64), S, "mean")
    t = torch.tensor([0.7], dtype=torch.float64)
    with torch.no_grad():
        a = m(x, t, key_bias=share_bias(S).double())
        frame = torch.arange(D.F_FRAMES).repeat_interleave(D.P_POS)
        pos = torch.arange(D.P_POS).repeat(D.F_FRAMES)
        keep = ~(S[0][pos] & (frame > 0))
        bias = torch.where(S[0][pos[keep]], math.log(D.F_FRAMES), 0.0).double()[None]
        b = m(x.reshape(1, D.N_TOK, D.D_TOK)[:, keep], t, key_bias=bias,
              frame_idx=frame[keep][None], pos_idx=pos[keep][None])
    assert keep.sum().item() == FL.shared_tokens(5)
    assert torch.allclose(a.reshape(1, D.N_TOK, D.D_TOK)[:, keep], b, atol=1e-9)


def test_proportional_attention_exact_only_without_temporal_pe():
    S = torch.rand(4, D.P_POS) < 0.6
    x = tie(torch.randn(4, D.F_FRAMES, D.P_POS, D.D_TOK, dtype=torch.float64), S, "copy")
    t = torch.full((4,), 0.5, dtype=torch.float64)
    errs = {}
    for pe in (False, True):
        m = _random_model(temporal_pe=pe)
        with torch.no_grad():
            m.pos_t.mul_(8.0)
            full = m(x, t)
            y = m(x, t, key_bias=share_bias(S).double())
            y = torch.where(S[:, None, :, None], y[:, :1].expand_as(y), y)
        errs[pe] = (y - full).abs().max().item()
    assert errs[False] < 1e-8 and errs[True] > 1e-3


def test_sampler_keeps_shared_positions_tied():
    m = _random_model().float()
    S = torch.rand(3, D.P_POS) < 0.5
    x, _ = sample(m, torch.randn(3, D.F_FRAMES, D.P_POS, D.D_TOK), steps=4, static=S, tie_how="mean", share=True)
    spread = (x - x[:, :1]).abs().amax(dim=(1, 3))
    assert spread[S].max().item() < 1e-5


def test_flops_speedup_monotone():
    assert FL.speedup(FL.shared_tokens(16)) > FL.speedup(FL.shared_tokens(8)) > 1.0
    assert abs(FL.speedup(D.N_TOK) - 1.0) < 1e-9
