"""Analytic multiply-accumulate counts for one forward call of a DiT with n tokens."""


def forward_flops(n, d=128, depth=4, mlp_ratio=4, d_tok=64):
    per_block = 4 * n * d * d + 2 * mlp_ratio * n * d * d + 2 * n * n * d   # qkv+proj, mlp, QK^T + AV
    return depth * per_block + 2 * n * d_tok * d


def shared_tokens(n_static_pos, frames=8, positions=16):
    """Tokens left after every static position keeps one leader token."""
    return frames * positions - (frames - 1) * n_static_pos


def speedup(n_shared_tokens, steps=20, warm_full_steps=0, extra_full_steps=0, steps_ref=20,
            frames=8, positions=16, **arch):
    """Speedup over steps_ref full-compute steps.

    warm_full_steps: steps of the trajectory itself that run on all tokens (mid-generation switch).
    extra_full_steps: full-compute steps that are *not* part of the trajectory (a separate draft pass).
    """
    n = frames * positions
    full = forward_flops(n, **arch)
    red = forward_flops(n_shared_tokens, **arch)
    cost = (warm_full_steps + extra_full_steps) * full + (steps - warm_full_steps) * red
    return steps_ref * full / cost


# The projection used in the base paper: a 75.6k-token, 40-layer, width-3072 video DiT.
BIG = dict(d=3072, depth=40, mlp_ratio=4, d_tok=64)


def projected_speedup(token_fraction, steps=20, warm_full_steps=0, extra_full_steps=0, n_big=75600):
    full = forward_flops(n_big, **BIG)
    red = forward_flops(token_fraction * n_big, **BIG)
    cost = (warm_full_steps + extra_full_steps) * full + (steps - warm_full_steps) * red
    return steps * full / cost
