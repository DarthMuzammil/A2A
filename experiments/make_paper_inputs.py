"""Turn results/*.json into LaTeX macros and tables, so no number in the paper is typed by hand."""
import json, math, os

R = "results"
OUT = "paper/generated"
os.makedirs(OUT, exist_ok=True)


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


macros = {}


def put(name, value, fmt="{:.2f}"):
    macros[name] = fmt.format(value) if not isinstance(value, str) else value


def sci(x):
    m, e = f"{x:.1e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


# ------------------------------------------------------------------ E0
e0 = load("e0_errata_checks.json")
if e0:
    a = e0["a_null_bias"]
    put("NullBias", a["analytic_laplace_bits_per_px"], "{:.3f}")
    put("NullBiasGauss", a["by_dead_zone"]["0.0"]["gaussian_monte_carlo"])
    put("PvsFloor", a["implied_pvs_floor_at_4_bits_per_px"])
    put("LenAUC", e0["b_length_bias"]["closed_form_iid"]["127_vs_15"])
    sim = e0["b_length_bias"]["simulated"]["127_vs_15"]
    put("LenAUCsimLo", min(sim.values())); put("LenAUCsimHi", max(sim.values()))
    c = e0["c_unbounded_and_noise_scale"]
    pre = c["lighting_change"]["prefilter=2.0"]; raw = c["lighting_change"]["prefilter=0.0"]
    put("SpotPVS", pre["pvs_as_written"], "{:.1f}"); put("SpotPeak", pre["largest_per_pixel_charge_bits"], "{:.0f}")
    put("SpotCapped", pre["pvs_capped_by_intra_cost"]); put("SpotNoEvent", raw["pvs_no_event"])
    put("SpotRaw", raw["pvs_as_written"]); put("SpotRawPeak", raw["largest_per_pixel_charge_bits"], "{:.0f}")
    n = c["noise_level"]
    put("NoiseA", n["sigma=3.0"]["pvs_as_written"]); put("NoiseB", n["sigma=1.5"]["pvs_as_written"])
    put("NoiseC", n["sigma=0.75"]["pvs_as_written"])
    pr = e0["e_draft_cost_projection"]
    put("DraftShare", 100 * pr["static=0.9"]["draft_share_of_cost"], "{:.0f}")
    put("ProjLo", pr["static=0.73"]["projected_speedup_with_2_step_draft"], "{:.1f}")
    put("ProjHi", pr["static=0.9"]["projected_speedup_with_2_step_draft"], "{:.1f}")
    d = e0["d_proportional_attention"]
    put("PropExact", "$" + sci(d["no_temporal_pe"]["proportional"]) + "$")
    put("PropPE", d["temporal_pe"]["proportional"])

# ------------------------------------------------------------------ E1
e1 = load("e1_gaussian_law.json")
if e1:
    err = max(abs(m / p - 1) for r in e1["runs"] for m, p in zip(r["copy"]["var_ratio"], r["predicted_copy_var_ratio"]))
    put("LawMaxRelErr", 100 * err, "{:.1f}")
    put("LawMeanMaxDiff", "$" + sci(max(r["mean"]["max_abs_diff_vs_independent"] for r in e1["runs"])) + "$")
    r8 = next(r for r in e1["runs"] if r["F"] == 8 and r["static_fraction"] == 1.0)
    put("LawFEight", r8["copy"]["var_ratio"][0])
    put("LawCopyPsnr", r8["copy"]["median_psnr_vs_independent"], "{:.1f}")

# ------------------------------------------------------------------ E2
e2 = load("e2_ideal_denoiser.json")
if e2:
    m = e2["methods"]
    rows = [("oracle & 0", "copy-tied, oracle map", "mean-tied, oracle map"),
            ("2-step draft & 0", "copy-tied, 2-step draft map", "mean-tied, 2-step draft map")]
    for k in (3, 4, 6):
        rows.append((f"trajectory & {k}", f"copy-collapse at step {k}, map from trajectory",
                     f"mean-collapse at step {k}, map from trajectory"))
    lines = []
    for label, kc, km in rows:
        c, mm = m[kc], m[km]
        lines.append(f"{label} & {100*mm['static_fraction']:.0f} & {100*c['same_clip_rate']:.1f} & "
                     f"{c['bg_coef_std_ratio_mean']:.2f} & \\textbf{{{100*mm['same_clip_rate']:.1f}}} & "
                     f"{mm['bg_coef_std_ratio_mean']:.2f} \\\\")
    cold = m["cold copy (leader/sqrt F), oracle map"]
    open(os.path.join(OUT, "table_ideal.tex"), "w").write("\n".join(lines) + "\n")
    put("IdealCopySame", 100 * m["copy-tied, oracle map"]["same_clip_rate"], "{:.1f}")
    put("IdealMeanSame", 100 * m["mean-tied, oracle map"]["same_clip_rate"], "{:.1f}")
    put("IdealCopyPsnr", m["copy-tied, oracle map"]["median_psnr_vs_independent_db"], "{:.1f}")
    put("IdealCopyStd", m["copy-tied, oracle map"]["bg_coef_std_ratio_mean"])
    put("IdealMeanStd", m["mean-tied, oracle map"]["bg_coef_std_ratio_mean"])
    put("IdealColdSame", 100 * cold["same_clip_rate"], "{:.1f}")
    put("IdealColdStd", cold["bg_coef_std_ratio_mean"])
    put("IdealColdPsnr", cold["median_psnr_vs_independent_db"], "{:.1f}")
    put("IdealMeanSixSame", 100 * m["mean-collapse at step 6, map from trajectory"]["same_clip_rate"], "{:.1f}")
    put("IdealCopySixSame", 100 * m["copy-collapse at step 6, map from trajectory"]["same_clip_rate"], "{:.1f}")
    put("IdealDraftPrecision", 100 * e2["calibration"]["draft2"]["precision"], "{:.0f}")
    put("IdealNData", str(e2["n_data"])); put("IdealNSeeds", str(e2["n_seeds"]))

# ------------------------------------------------------------------ E3 (trained transformer)
A, B = load("e3_trained_model.json"), load("e3_regularised_model.json")
WORD = {14: "Fourteen", 10: "Ten", 7: "Seven", 5: "Five"}


def cells(res, key, bold):
    if res is None or key not in res["methods"]:
        return ["--"] * 6
    v = res["methods"][key]
    ref = res["reference"]["stats"]
    p = f"{v['psnr_vs_original_db']['median']:.1f}"
    return [f"{100*v['static_fraction_used']:.0f}" if v["static_fraction_used"] > 0 else "--",
            f"{v['speedup_flops']:.2f}", f"{v['projected_speedup_big_model']:.1f}",
            f"\\textbf{{{p}}}" if bold else p, f"{v['bg_coef_std_ratio_mean']:.2f}",
            f"{v['stats']['texture']/ref['texture']:.1f}\\,/\\,{v['stats']['frame_diff']/ref['frame_diff']:.1f}"]


if A:
    L = []

    def row(label, key, bold=False):
        L.append(" & ".join([label] + cells(A, key, bold) + [""] + cells(B, key, bold)) + " \\\\")

    L.append("\\multicolumn{14}{@{}l}{\\emph{Baseline}}\\\\")
    for k in (14, 10, 7, 5):
        row(f"Fewer steps, 20$\\to${k}", f"fewer steps ({k})")
    for how, title in (("copy", "Frame sharing, copy-tied noise (base paper)"),
                       ("mean", "Frame sharing, mean-tied noise (ours)")):
        L.append("\\midrule\n\\multicolumn{14}{@{}l}{\\emph{" + title + "}}\\\\")
        row("\\quad oracle map", f"{how}-tied sharing, oracle map", how == "mean")
        row("\\quad 2-step draft map", f"{how}-tied sharing, 2-step draft map", how == "mean")
        row("\\quad 2-step draft map, aggressive", f"{how}-tied sharing, 2-step draft map (aggressive)", how == "mean")
        row("\\quad switch at step 8, no draft", f"{how}-collapse sharing at step 8", how == "mean")
    open(os.path.join(OUT, "table_main.tex"), "w").write("\n".join(L) + "\n")

W = load("e3_regularised_w5_model.json")
if W:
    mw = W["methods"]
    for name, key in (("Oracle", "mean-tied sharing, oracle map"), ("Draft", "mean-tied sharing, 2-step draft map"),
                      ("Aggr", "mean-tied sharing, 2-step draft map (aggressive)")):
        put(f"W{name}Orig", mw[key]["psnr_vs_original_db"]["median"], "{:.1f}")
        put(f"W{name}Speedup", mw[key]["speedup_flops"], "{:.1f}")
    put("WCopyOrig", mw["copy-tied sharing, oracle map"]["psnr_vs_original_db"]["median"], "{:.1f}")
    put("WOracleStatic", 100 * W["reference"]["oracle_static_fraction"], "{:.0f}")
    put("WFewerTen", mw["fewer steps (10)"]["psnr_vs_original_db"]["median"], "{:.1f}")
    put("WFewerSeven", mw["fewer steps (7)"]["psnr_vs_original_db"]["median"], "{:.1f}")
    put("WMeanTex", mw["mean-tied sharing, oracle map"]["stats"]["texture"] / W["reference"]["stats"]["texture"])

for tag, e3 in (("A", A), ("B", B)):
    if not e3:
        continue
    m = e3["methods"]; ref = e3["reference"]["stats"]
    med = lambda key: m[key]["psnr_vs_original_db"]["median"]
    tex = lambda key: m[key]["stats"]["texture"] / ref["texture"]
    mot = lambda key: m[key]["stats"]["frame_diff"] / ref["frame_diff"]
    co, me = "copy-tied sharing, oracle map", "mean-tied sharing, oracle map"
    put(f"{tag}CopyOrig", med(co), "{:.1f}"); put(f"{tag}MeanOrig", med(me), "{:.1f}")
    put(f"{tag}MeanOrigLo", m[me]["psnr_vs_original_db"]["p10"], "{:.1f}")
    put(f"{tag}CopyOwnRef", m[co]["psnr_vs_own_tied_full_compute_db"], "{:.1f}")
    put(f"{tag}MeanOwnRef", m[me]["psnr_vs_own_tied_full_compute_db"], "{:.1f}")
    put(f"{tag}CopyStd", m[co]["bg_coef_std_ratio_mean"], "{:.1f}"); put(f"{tag}MeanStd", m[me]["bg_coef_std_ratio_mean"])
    put(f"{tag}CopyTex", tex(co), "{:.1f}"); put(f"{tag}MeanTex", tex(me), "{:.1f}"); put(f"{tag}MeanMot", mot(me), "{:.1f}")
    put(f"{tag}MeanTexFine", tex(me)); put(f"{tag}MeanMotFine", mot(me))
    if "reference_jitter_share_of_motion" in m[me]:
        put(f"{tag}JitterShare", 100 * m[me]["reference_jitter_share_of_motion"], "{:.0f}")
        put(f"{tag}MeanMotUnshared", m[me]["motion_ratio_on_unshared_positions"])
    put(f"{tag}CopyDraftOrig", med("copy-tied sharing, 2-step draft map"), "{:.1f}")
    put(f"{tag}MeanDraftOrig", med("mean-tied sharing, 2-step draft map"), "{:.1f}")
    put(f"{tag}DraftSpeedup", m["mean-tied sharing, 2-step draft map"]["speedup_flops"], "{:.1f}")
    ag = "mean-tied sharing, 2-step draft map (aggressive)"
    put(f"{tag}MeanAggrOrig", med(ag), "{:.1f}"); put(f"{tag}AggrSpeedup", m[ag]["speedup_flops"], "{:.1f}")
    put(f"{tag}AggrProj", m[ag]["projected_speedup_big_model"], "{:.1f}")
    put(f"{tag}DraftProj", m["mean-tied sharing, 2-step draft map"]["projected_speedup_big_model"], "{:.1f}")
    put(f"{tag}MeanEightOrig", med("mean-collapse sharing at step 8"), "{:.1f}")
    put(f"{tag}CopyEightOrig", med("copy-collapse sharing at step 8"), "{:.1f}")
    put(f"{tag}OracleStatic", 100 * e3["reference"]["oracle_static_fraction"], "{:.0f}")
    put(f"{tag}OracleSpeedup", m[me]["speedup_flops"], "{:.1f}")
    put(f"{tag}OracleProj", m[me]["projected_speedup_big_model"], "{:.1f}")
    put(f"{tag}ColdOrig", med("cold-copy sharing (leader noise / sqrt F), oracle map"), "{:.1f}")
    put(f"{tag}ColdStd", m["cold-copy sharing (leader noise / sqrt F), oracle map"]["bg_coef_std_ratio_mean"])
    put(f"{tag}PlainOrig", med("mean-tied sharing, oracle map, plain attention"), "{:.1f}")
    put(f"{tag}MeanFullOrig", med("mean-tied full compute, oracle map (no sharing)"), "{:.1f}")
    for k in (14, 10, 7, 5):
        put(f"{tag}Fewer{WORD[k]}", med(f"fewer steps ({k})"), "{:.1f}")
    rho = m[co]["static_fraction_used"]
    put(f"{tag}LawPred", math.sqrt(rho * 8 + 1 - rho), "{:.1f}")
    st = e3["settling"]
    put(f"{tag}SettleStatic", st["median_step_static"], "{:.0f}"); put(f"{tag}SettleMoving", st["median_step_moving"], "{:.0f}")
    tm = 1 - st["median_step_moving"] / 20
    ts = tm * math.sqrt(8) / ((1 - tm) + tm * math.sqrt(8))
    put(f"{tag}SettlePred", (1 - ts) * 20, "{:.1f}")
    pa = e3["proportional_attention_trained_model_rel_error"]
    put(f"{tag}PropTrained", 100 * pa["proportional"], "{:.2f}"); put(f"{tag}PlainTrained", 100 * pa["plain"], "{:.0f}")
    put(f"{tag}N", f"{e3['n']:,}".replace(",", "{,}"))
    put(f"{tag}BgData", sum(e3["reference"]["bg_coef_std_over_data_sigma"]) / 6)
    put(f"{tag}DraftPrecision", 100 * e3["calibration"]["draft2"]["precision"], "{:.0f}")
    put(f"{tag}DraftRecall", 100 * e3["calibration"]["draft2"]["recall"], "{:.0f}")

# ------------------------------------------------------------------ E4 (distribution test)
c4 = {"A": load("e4_distribution_test.json"), "B": load("e4_distribution_test_regularised.json")}
rows = [("Full compute, fresh seeds (control)", "full compute, fresh seeds (control)"),
        ("Fewer steps, 20$\\to$10", "fewer steps (10)"), ("Fewer steps, 20$\\to$7", "fewer steps (7)"),
        ("Copy-tied sharing, 2-step draft", "copy-tied sharing, 2-step draft map"),
        ("Mean-tied sharing, 2-step draft", "mean-tied sharing, 2-step draft map")]
if c4["A"]:
    L = []
    for label, key in rows:
        vals = [f"{c4[t]['vs_full_compute'][key]['accuracy']:.2f}" if c4[t] and key in c4[t]["vs_full_compute"] else "--"
                for t in ("A", "B")]
        L.append(f"{label} & {vals[0]} & {vals[1]} \\\\")
    open(os.path.join(OUT, "table_c2st.tex"), "w").write("\n".join(L) + "\n")
    for t in ("A", "B"):
        if not c4[t]:
            continue
        v = c4[t]["vs_full_compute"]
        put(f"{t}CtControl", v["full compute, fresh seeds (control)"]["accuracy"])
        put(f"{t}CtFewerTen", v["fewer steps (10)"]["accuracy"]); put(f"{t}CtFewerSeven", v["fewer steps (7)"]["accuracy"])
        put(f"{t}CtCopy", v["copy-tied sharing, 2-step draft map"]["accuracy"])
        put(f"{t}CtMean", v["mean-tied sharing, 2-step draft map"]["accuracy"])
        put(f"{t}CtData", c4[t]["model_vs_training_data"]["accuracy"])

# ------------------------------------------------------------------ E6 (model variants)
e6 = load("e6_sufficiency.json")
if e6:
    order = [("standard, 3k steps", "Standard, 3k", "Three"), ("standard, 6k steps", "Standard, 6k", "Six"),
             ("standard, 12k steps", "Standard, 12k", "A"), ("+3k plain", "\\;\\;+3k, $\\lambda{=}0$", "C"),
             ("+3k lambda=5", "\\;\\;+3k, $\\lambda{=}5$", "Five"), ("+3k lambda=50", "\\;\\;+3k, $\\lambda{=}50$", "B")]
    L = []
    for key, label, tag in order:
        if key not in e6["models"]:
            continue
        r = e6["models"][key]; rel = r["relative_to_data"]
        L.append(f"{label} & {1000*r['consistency_error_mean']:.1f} & {r['alpha_sweep_psnr_db']['0.9']:.1f} & "
                 f"{r['alpha_sweep_psnr_db']['0.0']:.1f} & {r['copy_tied_psnr_db']:.1f} & {rel['texture']:.2f} & "
                 f"{rel['frame_diff']:.2f} & {r['bg_spread_over_data']:.2f} \\\\")
        put(f"Var{tag}Cons", 1000 * r["consistency_error_mean"], "{:.1f}")
        put(f"Var{tag}AlphaNine", r["alpha_sweep_psnr_db"]["0.9"], "{:.1f}")
        put(f"Var{tag}AlphaZero", r["alpha_sweep_psnr_db"]["0.0"], "{:.1f}")
        put(f"Var{tag}Tex", rel["texture"]); put(f"Var{tag}Mot", rel["frame_diff"]); put(f"Var{tag}Moving", rel["moving_frac"])
        put(f"Var{tag}Bg", r["bg_spread_over_data"]); put(f"Var{tag}Fit", r["fit_loss_mean"], "{:.3f}")
    open(os.path.join(OUT, "table_variants.tex"), "w").write("\n".join(L) + "\n")
    m6 = e6["models"]
    if "standard, 12k steps" in m6 and "+3k lambda=50" in m6:
        put("VarConsDrop", m6["standard, 12k steps"]["consistency_error_mean"] / m6["+3k lambda=50"]["consistency_error_mean"], "{:.0f}")
    if "+3k lambda=5" in m6 and "+3k plain" in m6:
        put("VarFiveConsDrop", m6["standard, 12k steps"]["consistency_error_mean"] / m6["+3k lambda=5"]["consistency_error_mean"], "{:.0f}")
        put("VarBFitRise", 100 * (m6["+3k lambda=50"]["fit_loss_mean"] / m6["+3k plain"]["fit_loss_mean"] - 1), "{:.1f}")
        put("VarFiveFitRise", 100 * (m6["+3k lambda=5"]["fit_loss_mean"] / m6["+3k plain"]["fit_loss_mean"] - 1), "{:.1f}")

# ------------------------------------------------------------------ E5 (wall clock)
e5 = load("e5_wallclock.json")
if e5:
    r = {x["static_positions"]: x for x in e5["rows"]}
    for k, name in ((8, "Eight"), (12, "Twelve"), (16, "Sixteen")):
        put(f"Wall{name}", r[k]["speedup_wallclock"], "{:.1f}"); put(f"Flop{name}", r[k]["speedup_flops"], "{:.1f}")

with open(os.path.join(OUT, "numbers.tex"), "w") as f:
    for k in sorted(macros):
        f.write(f"\\newcommand{{\\{k}}}{{{macros[k]}}}\n")
print(f"wrote {len(macros)} macros to {OUT}/numbers.tex")
