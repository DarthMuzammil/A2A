"""Render every figure of the paper from results/*.json (and results/e3_examples.pt)."""
import json, math, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3 (validated all-pairs)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})
OUT = "figures"
os.makedirs(OUT, exist_ok=True)


def load(name):
    p = os.path.join("results", name)
    return json.load(open(p)) if os.path.exists(p) else None


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=220)
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ errata checks
def fig_errata():
    r = load("e0_errata_checks.json")
    if r is None:
        return
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.0))
    a = r["a_null_bias"]["by_dead_zone"]
    dz = sorted(float(k) for k in a)
    xs = np.linspace(0, 2, 100)
    ax[0].plot(xs, 2 ** (-xs) / (math.e * math.log(2)), color=BLUE, lw=1.6, label="Laplacian noise")
    ax[0].plot(dz, [a[str(d)]["laplace_monte_carlo"] for d in dz], "o", color=BLUE, ms=4.5, mec="white", mew=0.8)
    ax[0].plot(dz, [a[str(d)]["gaussian_monte_carlo"] for d in dz], "s-", color=ORANGE, ms=4, lw=1.2,
               mec="white", mew=0.8, label="Gaussian noise")
    ax[0].axhline(0, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax[0].text(2.0, -0.012, "claimed: 0", ha="right", va="top", color=MUTED, fontsize=6.5)
    ax[0].set(xlabel="dead zone δ (bits)", ylabel="E[s] with no event (bits/px)", title="(a) Noise alone is charged")
    ax[0].set_ylim(-0.06, 0.6)
    ax[0].legend(loc="upper right", handlelength=1.4)

    b = r["b_length_bias"]
    keys = list(b["closed_form_iid"])
    x = np.arange(len(keys))
    ax[1].bar(x, [b["closed_form_iid"][k] for k in keys], width=0.55, color=BLUE)
    for i, k in enumerate(keys):
        ax[1].text(i, b["closed_form_iid"][k] + 0.015, f"{b['closed_form_iid'][k]:.2f}", ha="center", fontsize=7)
    ax[1].axhline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax[1].set_xticks(x, [k.replace("_vs_", " vs ") for k in keys])
    ax[1].set(xlabel="frame pairs per clip", ylabel="AUC, two event-free sets", ylim=(0.4, 1.0),
              title="(b) Longer clips score higher")

    c = r["c_unbounded_and_noise_scale"]["noise_level"]
    ks = list(c)
    x = np.arange(len(ks))
    w = 0.36
    ax[2].bar(x - w / 2 - 0.01, [c[k]["pvs_as_written"] for k in ks], width=w, color=ORANGE, label="as printed")
    ax[2].bar(x + w / 2 + 0.01, [c[k]["pvs_capped_by_intra_cost"] for k in ks], width=w, color=BLUE,
              label="capped by intra cost")
    for i, k in enumerate(ks):
        ax[2].text(i - w / 2 - 0.01, c[k]["pvs_as_written"] + 0.03, f"{c[k]['pvs_as_written']:.2f}", ha="center", fontsize=7)
        ax[2].text(i + w / 2 + 0.01, c[k]["pvs_capped_by_intra_cost"] + 0.03, f"{c[k]['pvs_capped_by_intra_cost']:.2f}",
                   ha="center", fontsize=7)
    ax[2].axhline(1.0, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax[2].text(-0.45, 0.97, "all bits of the frame", va="top", color=MUTED, fontsize=6.5)
    ax[2].set_xticks(x, [k.replace("sigma=", "σ = ") for k in ks])
    ax[2].set(xlabel="sensor noise (grey levels)", ylabel="PVS$_t$, one fixed lighting change",
              title="(c) Same event, different noise")
    ax[2].set_ylim(0, 2.3)
    ax[2].legend(loc="upper left", handlelength=1.2)
    for a_ in ax:
        a_.grid(axis="y")
        a_.set_axisbelow(True)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_errata")


# ------------------------------------------------------------------ the variance law
def fig_law():
    r = load("e1_gaussian_law.json")
    e3 = load("e3_trained_model.json")
    if r is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.35), gridspec_kw={"width_ratios": [1, 1.25]})
    pred, meas, mean = [], [], []
    for run in r["runs"]:
        pred += run["predicted_copy_var_ratio"]
        meas += run["copy"]["var_ratio"]
        mean += run["mean"]["var_ratio"]
    lim = (0.8, 20)
    ax[0].plot(lim, lim, color=MUTED, lw=0.8, ls=(0, (3, 2)))
    ax[0].plot(pred, meas, "o", color=ORANGE, ms=4.5, mec="white", mew=0.7, label="copy-tied noise (base paper)")
    ax[0].plot(pred, mean, "s", color=BLUE, ms=4, mec="white", mew=0.7, label="mean-tied noise (ours)")
    ax[0].set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="predicted  ρF + 1 − ρ",
              ylabel="measured variance ratio", title="(a) Exact Gaussian model, F = 2…16")
    ax[0].set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
    ax[0].set_yticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
    ax[0].minorticks_off()
    ax[0].legend(loc="upper left", handletextpad=0.3)
    ax[0].grid(True)
    if e3 is not None:
        names = ["DC", "x", "y", "xy", "cos x", "cos y"]
        m = e3["methods"]
        rows = [("copy-tied sharing, oracle map", ORANGE, "copy-tied (base paper)"),
                ("mean-tied sharing, oracle map", BLUE, "mean-tied (ours)")]
        x = np.arange(len(names))
        w = 0.36
        top = 0
        for j, (key, col, lab) in enumerate(rows):
            vals = m[key]["bg_coef_std_ratio"]
            top = max(top, max(vals))
            ax[1].bar(x + (j - 0.5) * (w + 0.02), vals, width=w, color=col, label=lab)
            for xi, v in zip(x, vals):
                ax[1].text(xi + (j - 0.5) * (w + 0.02), v + 0.04, f"{v:.1f}", ha="center", fontsize=6.5)
        rho = m["copy-tied sharing, oracle map"]["static_fraction_used"]
        law = math.sqrt(rho * 8 + 1 - rho)
        ax[1].axhline(law, color=MUTED, lw=0.8, ls=(0, (3, 2)))
        ax[1].text(len(names) - 0.42, law - 0.07, "Gaussian law", ha="right", va="top", color=MUTED, fontsize=6.5)
        ax[1].axhline(1.0, color=MUTED, lw=0.8)
        ax[1].set_xticks(x, names)
        ax[1].set(xlabel="background coefficient", ylabel="spread relative to full compute",
                  title="(b) Trained transformer, F = 8")
        ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.2, columnspacing=1.0)
        ax[1].grid(axis="y")
        ax[1].set_ylim(0, top * 1.32)
    for a_ in ax:
        a_.set_axisbelow(True)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_law")


# ------------------------------------------------------------------ fidelity vs speedup
def fig_tradeoff():
    panels = [(load("e3_trained_model.json"), "(a) Standard model"),
              (load("e3_regularised_model.json"), "(b) Sufficiency-regularised model")]
    panels = [(r, t) for r, t in panels if r is not None]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.5 * len(panels), 2.5), sharey=True, squeeze=False)
    for ax, (e3, title) in zip(axes[0], panels):
        m = e3["methods"]
        fs = sorted((m[k]["speedup_flops"], m[k]["psnr_vs_original_db"]["median"]) for k in m if k.startswith("fewer steps"))
        ax.plot([a for a, _ in fs], [b for _, b in fs], "-o", color=MUTED, lw=1.4, ms=4, mec="white", mew=0.7,
                label="fewer steps (baseline)", zorder=3)
        for how, col, lab in (("copy", ORANGE, "copy-tied (base paper)"), ("mean", BLUE, "mean-tied (ours)")):
            pts = []
            for k in m:
                if k in (f"{how}-collapse sharing at step 6", f"{how}-collapse sharing at step 8"):
                    pts.append((m[k]["speedup_flops"], m[k]["psnr_vs_original_db"]["median"]))
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "D", color=col, ms=4.2, mec="white", mew=0.7, zorder=4)
            for tag, mk in (("oracle map", "o"), ("2-step draft map", "s"), ("2-step draft map (aggressive)", "s")):
                v = m[f"{how}-tied sharing, {tag}"]
                ax.plot(v["speedup_flops"], v["psnr_vs_original_db"]["median"], mk, color=col, ms=6, mec=INK, mew=0.6, zorder=5)
        ax.set(xlabel="speedup (FLOPs, incl. draft and warm-up calls)", title=title, xlim=(0.9, 4.15), ylim=(0, 53))
        ax.grid(True)
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel("PSNR vs. original sample (dB)")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=MUTED, marker="o", ms=4, mec="white", lw=1.4, label="fewer steps (baseline)"),
               Line2D([], [], color=ORANGE, marker="o", ms=5.5, mec=INK, mew=0.6, lw=0, label="copy-tied sharing (base paper)"),
               Line2D([], [], color=BLUE, marker="o", ms=5.5, mec=INK, mew=0.6, lw=0, label="mean-tied sharing (ours)"),
               Line2D([], [], color="white", marker="o", ms=5.5, mec=INK, mew=0.6, lw=0, label="oracle map"),
               Line2D([], [], color="white", marker="s", ms=5.5, mec=INK, mew=0.6, lw=0, label="2-step draft map"),
               Line2D([], [], color="white", marker="D", ms=4.2, mec=INK, mew=0.6, lw=0, label="switch at step 6 or 8")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.17), handlelength=1.4, columnspacing=1.2)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_tradeoff")


# ------------------------------------------------------------------ qualitative examples
def fig_examples():
    import torch
    from meantie import data as D
    blocks = []
    if os.path.exists("results/e3_examples.pt"):
        ex = torch.load("results/e3_examples.pt")
        blocks += [(ex, "ref", "standard model\nfull compute (original)"),
                   (ex, "copy-tied sharing, oracle map", "copy-tied sharing\n(base paper)"),
                   (ex, "mean-tied sharing, oracle map", "mean-tied sharing\n(ours)")]
    if os.path.exists("results/e3_examples_regularised.pt"):
        exb = torch.load("results/e3_examples_regularised.pt")
        blocks += [(exb, "ref", "regularised model\nfull compute (original)"),
                   (exb, "copy-tied sharing, oracle map", "copy-tied sharing\n(base paper)"),
                   (exb, "mean-tied sharing, oracle map", "mean-tied sharing\n(ours)")]
    if not blocks:
        return
    seeds, frames = [0, 1, 2, 3], [0, 7]
    fig, ax = plt.subplots(len(blocks), len(seeds) * len(frames), figsize=(7.0, 0.93 * len(blocks) + 0.2))
    for i, (ex, key, lab) in enumerate(blocks):
        clip = D.unpatchify(ex[key])
        for j, s_ in enumerate(seeds):
            for k, f in enumerate(frames):
                a = ax[i, j * len(frames) + k]
                a.imshow(clip[s_, f].numpy(), cmap="gray", vmin=-1.6, vmax=1.6, interpolation="nearest")
                a.set_xticks([]); a.set_yticks([])
                for sp in a.spines.values():
                    sp.set_visible(False)
                if i == 0:
                    a.set_title(f"seed {j + 1}\nframe {f + 1}", fontsize=6, pad=2, linespacing=0.95)
                if key + " | map" in ex and k == 0:
                    S = ex[key + " | map"][s_].view(4, 4).numpy()
                    for yy in range(4):
                        for xx in range(4):
                            if S[yy, xx]:
                                a.add_patch(plt.Rectangle((xx * 8 - 0.5, yy * 8 - 0.5), 8, 8, fill=False,
                                                          ec=AQUA, lw=0.5, alpha=0.9))
        ax[i, 0].set_ylabel(lab, fontsize=6.5, rotation=0, ha="right", va="center", labelpad=4)
    fig.subplots_adjust(wspace=0.04, hspace=0.07)
    if len(blocks) == 6:                       # a little air between the two models
        for i in range(3, 6):
            for a in ax[i]:
                pos = a.get_position()
                a.set_position([pos.x0, pos.y0 - 0.012, pos.width, pos.height])
    save(fig, "fig_examples")


if __name__ == "__main__":
    fig_errata(); fig_law(); fig_tradeoff(); fig_examples()
