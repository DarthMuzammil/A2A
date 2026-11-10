"""E1 -- the variance-inflation law for tied noise, checked on an exactly solvable model.

Data: a static Gaussian-field background repeated over F frames.  The ideal denoiser is
linear and known in closed form, so the 20-step Euler sampler is exact up to float error.
We tie the starting noise over a set S of positions, by copying the leader frame's noise
(base paper) or by averaging the noise over frames (this work), and measure the variance
of every background coefficient in the output relative to independent noise.

Prediction for copy-tying:  inflation_k = rho_k * F + (1 - rho_k),
where rho_k is the share of basis function k's energy that lies on S.
Prediction for mean-tying:  inflation_k = 1, and the sample itself is unchanged.
"""
import json, os, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from meantie import data as D

torch.set_num_threads(1)
torch.manual_seed(0)
Hs = Ws = 16           # a smaller grid keeps this instant; the law does not depend on it
PATCH = 4
basis = D.bg_basis(Hs, Ws).reshape(6, Hs * Ws).double()          # unit RMS rows
sig = torch.tensor(D.BG_SIGMAS).double()


def run(F, static_px, n=20000, steps=20):
    """static_px: bool [Hs*Ws].  Returns per-coefficient variance ratios and PSNR."""
    npx = Hs * Ws
    U = basis.T / (npx * F) ** 0.5                                # [npx,6]: per-frame part of unit eigenvectors
    lam = sig ** 2 * npx * F                                      # eigenvalues of the data covariance
    eps = torch.randn(n, F, npx, dtype=torch.float64)

    def sampler(x):
        for i in range(steps):
            t = 1.0 - i / steps
            a = torch.einsum("nfp,pk->nk", x, U)                  # projection on the 6 data eigenvectors
            gain = (1 - t) * lam / ((1 - t) ** 2 * lam + t ** 2)
            x0_hat = torch.einsum("nk,pk->np", a * gain, U)[:, None, :].expand(-1, F, -1)
            x = x - (x - x0_hat) / t / steps
        return x

    def coef(x):                                                  # background coefficients of the output
        return torch.einsum("nfp,kp->nk", x, basis) / (npx * F)

    out = {}
    x_ind = sampler(eps)
    c_ind = coef(x_ind)
    for how in ("copy", "mean"):
        rep = eps[:, :1] if how == "copy" else eps.mean(dim=1, keepdim=True)
        e = torch.where(static_px[None, None, :], rep.expand_as(eps), eps)
        x = sampler(e)
        c = coef(x)
        mse = (x - x_ind).pow(2).flatten(1).mean(1)
        out[how] = {
            "var_ratio": (c.var(0) / c_ind.var(0)).tolist(),
            "median_psnr_vs_independent": float(torch.median(10 * torch.log10(4.0 / mse.clamp_min(1e-30)))),
            "max_abs_diff_vs_independent": float((x - x_ind).abs().max()),
        }
    rho = (basis ** 2 * static_px[None].double()).sum(1) / (basis ** 2).sum(1)
    out["rho"] = rho.tolist()
    out["predicted_copy_var_ratio"] = (rho * F + 1 - rho).tolist()
    out["data_var_recovered"] = (c_ind.var(0) / sig ** 2).tolist()
    return out


def region(frac_rows):
    m = torch.zeros(Hs, Ws, dtype=torch.bool)
    m[: int(round(frac_rows * Hs))] = True
    return m.reshape(-1)


res = {"names": list(D.BG_NAMES), "runs": []}
for F in (2, 4, 8, 16):
    for frac in (1.0, 0.75, 0.5):
        r = run(F, region(frac))
        r.update({"F": F, "static_fraction": frac})
        res["runs"].append(r)
        print(f"F={F:2d} static={frac:.2f}  copy var ratio DC: measured {r['copy']['var_ratio'][0]:.3f} "
              f"predicted {r['predicted_copy_var_ratio'][0]:.3f} | mean-tied: {r['mean']['var_ratio'][0]:.6f}, "
              f"max|diff| {r['mean']['max_abs_diff_vs_independent']:.1e}")
os.makedirs("results", exist_ok=True)
json.dump(res, open("results/e1_gaussian_law.json", "w"), indent=1)
