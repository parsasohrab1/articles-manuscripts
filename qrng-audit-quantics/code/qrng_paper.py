"""Figures and Word manuscript for the QRNG study. Usage: python qrng_paper.py <dir S_2023_trials> <results dir> <paper dir>"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mslib import GOLD, GREY, NAVY, ORANGE, RED, TEAL, flow, write_docx

ROOT, RES, OUT = (Path(a) for a in sys.argv[1:4])
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
R = json.loads((RES / "results.json").read_text(encoding="utf-8"))
IBM = ["Belem", "Jakarta", "Lagos", "Lima", "Manila", "Nairobi", "Oslo", "Perth", "Quito"]
REF = {"PRNG_Python": "Python", "PRNG_Secrets_module": "secrets", "Random_org": "Random.org", "Pi": "digits of pi", "PRNG_biased_5_0_5_1": "fair",
       "PRNG_biased_505_0_495_1": "P(1)=.495", "PRNG_biased_51_0_49_1": "P(1)=.49", "PRNG_biased_52_0_48_1": "P(1)=.48",
       "PRNG_biased_55_0_45_1": "P(1)=.45", "PRNG_biased_58_0_42_1": "P(1)=.42", "PRNG_biased_59_0_41_1": "P(1)=.41"}
TYPES = ["Basic", "Type 1", "Type 2", "Type 3"]


def ctype(c):
    return "Basic" if c.startswith("Basic") else "Type " + c.split("Type_")[1][0]


def save(fig, name):
    p = FIG / name
    fig.savefig(p)
    plt.close(fig)
    return p


def csv(name):
    return pd.read_csv(RES / name)


f1 = FIG / "fig1_pipeline.png"
flow(f1, [("Public archive", "Nine IBM devices, four circuit types, eleven reference sources"),
          ("Data screen", "Remove stuck strings and repeated files; keep a list of both"),
          ("Audit", "Five tests per string; uniformity per circuit and test; false discovery rate per source"),
          ("Diagnosis", "Min-entropy, bias by bit position, sample-size sensitivity"),
          ("Conditioning", "Von Neumann extractor, plain and by position stream; audit again")], "Study pipeline", height=2.6)

# fig 2: screen
fig, axes = plt.subplots(1, 3, figsize=(10, 2.9), sharey=True)
for ax, (d, c) in zip(axes, [("Belem", "Basic_QRNG_Belem"), ("Nairobi", "QRNG_Type_1_Nairobi"), ("Lima", "Basic_QRNG_Lima")]):
    files = sorted((ROOT / d / c).glob("*.txt"), key=lambda f: int(f.stem.rsplit("_", 1)[1]))
    s = [f.read_text().strip() for f in files]
    x = np.array([int(f.stem.rsplit("_", 1)[1]) for f in files])
    y = np.array([v.count("1") / len(v) for v in s])
    seen, rep = set(), []
    for v in s:
        rep.append(v in seen)
        seen.add(v)
    stuck = np.abs(y - 0.5) > 0.25
    rep = np.array(rep) & ~stuck
    ax.scatter(x[~stuck & ~rep], y[~stuck & ~rep], s=7, color=NAVY, label="kept")
    ax.scatter(x[stuck], y[stuck], s=16, color=RED, marker="x", label="stuck")
    ax.scatter(x[rep], y[rep], s=45, facecolors="none", edgecolors=ORANGE, lw=1.4, label="repeat")
    ax.axhspan(0.25, 0.75, color=TEAL, alpha=0.07)
    ax.set_title(f"{d}, {ctype(c)}"), ax.set_xlabel("File number")
axes[0].set_ylabel("Share of ones in the file")
h, l = axes[0].get_legend_handles_labels()
h2, l2 = axes[1].get_legend_handles_labels()
keys = dict(zip(l + l2, h + h2))
fig.legend([keys[k] for k in ("kept", "stuck", "repeat")], ["kept", "stuck", "repeated file"], loc="upper center", ncol=3)
fig.tight_layout(rect=(0, 0, 1, 0.91))
f2 = save(fig, "fig2_screen.png")

# fig 3: rejected tests before and after conditioning
sets = [("Raw output", "groups_raw.csv"), ("Von Neumann", "groups_vn.csv"), ("Von Neumann by position stream", "groups_vn4.csv")]
fig, axes = plt.subplots(1, 3, figsize=(10, 4.6), sharey=True)
rows = IBM + list(REF)
for ax, (label, name) in zip(axes, sets):
    g = csv(name)
    m = np.full((len(rows), 4), np.nan)
    for (d, c), v in g.groupby(["device_id", "circuit_id"]):
        m[rows.index(d), TYPES.index(ctype(c)) if d in IBM else 0] = v.reject.sum()
    im = ax.imshow(m, cmap="YlOrRd", vmin=0, vmax=5, aspect="auto")
    for i in range(len(rows)):
        for j in range(4):
            if not np.isnan(m[i, j]):
                ax.text(j, i, int(m[i, j]), ha="center", va="center", fontsize=7, color="white" if m[i, j] > 3 else "black")
    ax.set_title(label, fontsize=9), ax.set_xticks(range(4), TYPES, fontsize=7.5), ax.grid(False)
    ax.axhline(len(IBM) - 0.5, color=NAVY, lw=1.5)
axes[0].set_yticks(range(len(rows)), IBM + list(REF.values()), fontsize=7.5)
fig.colorbar(im, ax=axes, shrink=0.75, label="Tests rejected (of 5)")
fig.text(0.5, 0.0, "Reference sources (below the line) have one data set each, shown in the first column", ha="center", fontsize=8)
f3 = save(fig, "fig3_audit.png")

# fig 4: position bias
pos = csv("position_bias.csv")
fig, axes = plt.subplots(3, 3, figsize=(10, 6), sharex=True, sharey=True)
for ax, d in zip(axes.ravel(), IBM):
    v = pos[(pos.device_id == d) & (pos.m == 8)]
    for _, r in v.iterrows():
        t = ctype(r.circuit_id)
        y = np.array([float(a) for a in r.fractions.split()])
        ax.plot(range(8), y, marker="o", ms=3.5, lw=1.8 if t == "Type 3" else 0.9, color={"Basic": GREY, "Type 1": TEAL, "Type 2": GOLD, "Type 3": RED}[t], label=t)
    ax.set_title(d), ax.axhline(0.5, color=NAVY, ls=":", lw=0.8)
for ax in axes[-1]:
    ax.set_xlabel("Bit position modulo 8")
for ax in axes[:, 0]:
    ax.set_ylabel("Share of ones")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4)
fig.tight_layout(rect=(0, 0, 1, 0.95))
f4 = save(fig, "fig4_position.png")

# fig 5: min-entropy
fig, axes = plt.subplots(2, 1, figsize=(10, 5.4), sharex=True)
order = None
for ax, (col, label) in zip(axes, (("h_mcv", "Most common value estimate"), ("h_markov", "Markov estimate"))):
    for name, mk, c, lab in (("circuits_raw.csv", "o", RED, "raw"), ("circuits_vn.csv", "s", GOLD, "von Neumann"), ("circuits_vn4.csv", "^", TEAL, "von Neumann by position stream")):
        d = csv(name)
        d = d[d.device_id.isin(IBM)].reset_index(drop=True)
        ax.scatter(range(len(d)), d[col], marker=mk, s=18, color=c, label=lab)
    ref = csv("circuits_raw.csv")
    ref = ref[ref.device_id.isin(["PRNG_Python", "PRNG_Secrets_module", "Random_org", "Pi", "PRNG_biased_5_0_5_1"])]
    ax.axhspan(ref[col].min(), ref[col].max(), color=NAVY, alpha=0.15, label="range of clean references (raw)")
    ax.set_ylabel("Bits per bit"), ax.set_title(label, loc="left")
for k in range(1, 9):
    for ax in axes:
        ax.axvline(4 * k - 0.5, color=GREY, lw=0.6)
axes[1].set_xticks([4 * k + 1.5 for k in range(9)], IBM), axes[0].legend(ncol=4, fontsize=7.5, loc="lower center")
axes[0].set_ylim(0.855, 1.003)
fig.tight_layout()
f5 = save(fig, "fig5_entropy.png")

# fig 6: bias and lag-1 autocorrelation
fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharex=True, sharey=True)
for ax, (label, name) in zip(axes, (("Raw output", "circuits_raw.csv"), ("Von Neumann", "circuits_vn.csv"), ("Von Neumann by position stream", "circuits_vn4.csv"))):
    d = csv(name)
    for sel, c, lab in ((d.device_id.isin(IBM) & ~d.circuit_id.str.contains("Type_3"), NAVY, "IBM, other types"), (d.circuit_id.str.contains("Type_3"), RED, "IBM, Type 3"),
                        (~d.device_id.isin(IBM), TEAL, "references")):
        ax.scatter(d[sel].frac_ones, d[sel].lag1, s=18, color=c, label=lab, alpha=0.85)
    ax.axvline(0.5, color=GREY, ls=":"), ax.axhline(0, color=GREY, ls=":"), ax.set_title(label, fontsize=9), ax.set_xlabel("Share of ones")
axes[0].set_ylabel("Mean lag-1 autocorrelation"), axes[0].legend(fontsize=7.5, loc="lower left")
fig.tight_layout()
f6 = save(fig, "fig6_bias.png")

# fig 7: sensitivity
S = csv("sensitivity.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
for cls, c in (("clean", TEAL), ("ibm", NAVY), ("biased", ORANGE)):
    g = S[S["class"] == cls].groupby("n")[["audit", "naive"]].mean()
    axes[0].plot(g.index, g.audit, marker="o", color=c, label=f"{cls if cls != 'ibm' else 'IBM circuits'}, audit")
    axes[0].plot(g.index, g.naive, marker="x", color=c, ls="--", label=f"{cls if cls != 'ibm' else 'IBM circuits'}, any p < 0.01")
for name, c in zip(list(REF)[5:], (RED, ORANGE, GOLD, TEAL, NAVY, GREY)):
    g = S[S.source == name]
    axes[1].plot(g.n, g.audit, marker="o", color=c, label=REF[name])
for ax in axes:
    ax.set_xscale("log", base=2), ax.set_xticks([4, 8, 16, 32, 64, 128], [4, 8, 16, 32, 64, 128]), ax.set_xlabel("Strings drawn per data set")
axes[0].set_ylabel("Share of draws flagged"), axes[0].set_title("Audit and per-string rule"), axes[1].set_title("Audit on biased references")
axes[0].legend(fontsize=6.8, loc="center right"), axes[1].legend(fontsize=7.5, loc="lower right")
fig.tight_layout()
f7 = save(fig, "fig7_sensitivity.png")

raw, pub, vn, vn4, sc, ps = (R[k] for k in ("raw", "as_published", "von_neumann", "von_neumann_4streams", "screen", "position_significant"))
S1, SB = R["sensitivity_by_class"], R["sensitivity_biased"]
pc = lambda x: f"{100 * x:.1f}"
frac = R["ibm_device_frac_ones"]
frac_rng = f"{min(frac.values()):.3f} to {max(frac.values()):.3f}"
clean_max = max(S1[f"clean|{n}"][0] for n in (4, 8, 16, 32, 64))
refs_after = vn["clean_unhealthy"] + vn["biased_unhealthy"]
title = "Position-dependent bias limits randomness extraction in public bit sequences from nine IBM quantum computers: a screened, false-discovery-controlled audit"
abstract = (f"Random bits taken from cloud quantum computers need health checks before use. We audited a public archive of {R['ibm_strings']} bit strings from four circuit types on nine IBM devices, with reference sources from the same archive. "
 f"A data screen removed {sc['stuck']} nearly constant strings from one circuit and {sc['repeat']} repeated files. "
 f"Five statistical tests per string were combined per circuit with false-discovery-rate control. All nine devices failed: {raw['ibm_circuits_rejected']} of 36 circuits had at least one rejection, while five clean references passed and six biased generators were detected. "
 f"The von Neumann extractor left {vn['ibm_circuits_rejected']} circuits rejected, {R['vn_split']['type3_circuits_rejected']} of them of the third circuit type. All nine circuits of that type, and none of the other 27, showed a significant dependence of the share of ones on bit position, with a period of four. "
 f"Applying the extractor separately to the four position streams removed every rejection in all 36 circuits, keeping about {pc(R['ibm_vn_yield'][1])} percent of the bits. This last step was chosen after seeing the data, and passing five basic tests does not certify randomness. "
 "Position structure should be checked before an extractor is trusted.")

by_dev = []
g = csv("groups_raw.csv")
cr = csv("circuits_raw.csv")
for d in IBM:
    a = g[g.device_id == d]
    by_dev.append([d, int(cr[cr.device_id == d].strings.sum()), f"{cr[cr.device_id == d].bits.sum() / 1e6:.1f}", f"{frac[d]:.4f}", f"{int(a.groupby('circuit_id').reject.any().sum())} of 4", f"{int(a.reject.sum())} of 20"])
cond = [("Raw, all files as published", pub), ("Raw, screened", raw), ("Raw, Kolmogorov-Smirnov uniformity", R["variant_ks"]), ("Raw, one correction over all sources", R["variant_global_bh"]),
        ("Von Neumann", vn), ("Von Neumann, Kolmogorov-Smirnov", R["von_neumann_ks"]), ("Von Neumann by position stream", vn4)]
cond_rows = [[k, f"{v['ibm_devices_unhealthy']} of 9", f"{v['ibm_circuits_rejected']} of 36", f"{v['ibm_groups_rejected']} of 180", f"{v['clean_unhealthy']} of 5", f"{v['biased_unhealthy']} of 6"] for k, v in cond]
p4 = R["type3_period4_spread"]
pos_rows = [[d] + p4[d][1].split() + [f"{p4[d][0]:.4f}"] for d in IBM]
sens_rows = [[n] + [pc(S1[f"{c}|{n}"][i]) for c in ("clean", "biased", "ibm") for i in (0, 1)] for n in (4, 8, 16, 32, 64, 128)]

blocks = [("title", title), ("small", "Authors and affiliations: [to be completed]. Draft for author review; not submitted."),
 ("h", "Abstract"), ("p", abstract), ("p", "Keywords: quantum random number generator; randomness testing; false discovery rate; von Neumann extractor; readout bias; IBM Quantum"),
 ("h", "1. Introduction"),
 ("p", "Cloud quantum computers let a user prepare superposition states, measure them and read the outcomes as random bits. Whether those bits are uniform and independent is an empirical matter. Random numbers produced on a cloud superconducting quantum computer have been studied before [1]. Root and Becker tested four generator circuits on nine IBM systems with eighteen statistical tests and judged only one combination of device and circuit statistically random [2]. The sequences analysed here were published by the first of those authors [3]."),
 ("p", "We therefore make no claim that finding non-randomness in these data is new. Our questions are different: whether the public files are sound; how a device-level verdict with false-discovery-rate control behaves on reference sources of known status and at smaller sample sizes; whether a classical extractor repairs the output; and, where it does not, what structure in the bits explains the failure."),
 ("h", "2. Data"),
 ("p", f"The archive S_2023_trials.zip (Zenodo record 10542216, licence CC BY 4.0, checksum verified) holds one text file per bit string. We used all strings of at most 40,000 bits and left out the long concatenated sequences: {R['strings']} strings and {R['bits'] / 1e6:.1f} million bits, of which {R['ibm_strings']} strings and {R['ibm_bits'] / 1e6:.1f} million bits come from the devices Belem, Jakarta, Lagos, Lima, Manila, Nairobi, Oslo, Perth and Quito. Each device has a basic generator circuit and three further types. The archive description does not define the circuits, so we use their labels only. Eleven reference sources are included: Python and secrets-module generators, Random.org, digits of pi, a fair generator, and six generators with a probability of one between 0.495 and 0.41."),
 ("p", f"A screen marked a string as stuck when its share of ones differed from one half by more than 0.25, and as a repeat when it was identical to an earlier file. It removed {sc['stuck']} stuck strings, all from the basic circuit of Belem, and {sc['repeat']} further repeated files, all from the Type 1 circuit of Nairobi (Figure 2). The largest deviation among kept strings was {sc['max_abs_dev_kept']:.3f} and the smallest among removed ones {sc['min_abs_dev_removed']:.3f}. The cause of these files is unknown; they may reflect a fault in execution or in assembling the archive. The threshold was set after inspecting the files, so results for the unscreened data are also reported. {R['screened_strings']} strings remained."),
 ("fig", f2, "Share of ones per file for two circuits with removed files and, for comparison, one without. The shaded band is the range accepted by the screen.", "three scatter panels with shared vertical axis; markers for kept, stuck and repeated files; 300 dpi PNG."),
 ("h", "3. Methods"),
 ("fig", f1, "Study pipeline.", "five-step left-to-right flow diagram; graphical abstract; 300 dpi PNG."),
 ("p", "For each string five p-values were computed: monobit frequency and runs as defined in NIST SP 800-22 [4], a longest-run-of-ones test with 8-bit blocks, a poker test on non-overlapping 4-bit patterns, and a lag-1 autocorrelation test. For each data set (a circuit of a device, or a reference source) and each test, a one-sided binomial test asked whether more p-values fell below 0.05 than expected. These p-values were corrected within each device or reference source by the Benjamini-Hochberg procedure at a false discovery rate of 0.05 [5]. A circuit or source is rejected when any of its tests is rejected. A Kolmogorov-Smirnov uniformity test and a single correction across all sources were run as variants. The comparison rule flags a source when any single p-value is below 0.01."),
 ("p", "Min-entropy per bit was estimated with the most-common-value and Markov estimators of NIST SP 800-90B [6]; the other estimators of that standard were not applied. Position dependence was tested by grouping bits by their index modulo m, for m from 2 to 16, and testing equality of the shares of ones by chi-square, with a Bonferroni threshold over all data sets and values of m."),
 ("p", "The von Neumann extractor [7] maps non-overlapping bit pairs 01 to 0 and 10 to 1 and discards equal pairs. It gives unbiased independent bits when input bits are independent with the same probability. In a second variant, chosen after the position analysis, each string was split into the four streams of positions equal to 0, 1, 2 and 3 modulo 4; the extractor was applied to each stream and the outputs were concatenated. Both outputs were audited with the same procedure."),
 ("p", "Sample-size sensitivity was estimated by drawing 4 to 128 strings without replacement from each data set 500 times and repeating the five-test decision. Draws come from a finite pool, so values at large sample sizes describe that pool rather than new data; a data set with fewer strings than the draw size was left out."),
 ("h", "4. Results"),
 ("h2", "4.1 Audit of the raw output"),
 ("p", f"All nine devices were rejected (Table 1): {raw['ibm_circuits_rejected']} of 36 circuits and {raw['ibm_groups_rejected']} of 180 circuit-test pairs. By test, the numbers of rejected circuits were {raw['by_test']['frequency']} for frequency, {raw['by_test']['poker']} for poker, {raw['by_test']['longest_run']} for longest run, {raw['by_test']['runs']} for runs and {raw['by_test']['autocorrelation']} for autocorrelation. The pooled share of ones ranged from {frac_rng} across devices, so most but not all devices produced too few ones. All five clean references passed and all six biased generators were rejected. The per-string rule flagged all {R['naive_sources_flagged']} sources, including the clean ones."),
 ("p", f"Screening changed the counts little ({pub['ibm_groups_rejected']} rejected pairs before and {raw['ibm_groups_rejected']} after), because the other failures of the affected circuits remained. The screen matters nevertheless: a file of identical bits should never reach a user, and an audit that only reports pass or fail would not show it."),
 ("table", "Screened raw output by device", ["Device", "Strings", "Million bits", "Share of ones", "Circuits rejected", "Circuit-test pairs rejected"], by_dev),
 ("fig", f3, "Number of tests rejected per circuit before and after conditioning.", "three annotated heat maps with shared rows; devices above and reference sources below the dividing line; 300 dpi PNG."),
 ("h2", "4.2 Conditioning"),
 ("p", f"The von Neumann extractor retained a median of {pc(R['ibm_vn_yield'][1])} % of the bits. It reduced the rejected circuits from {raw['ibm_circuits_rejected']} to {vn['ibm_circuits_rejected']}, but {vn['ibm_devices_unhealthy']} devices still had a rejection (Table 2). Of the nine Type 3 circuits, {R['vn_split']['type3_circuits_rejected']} remained rejected, against {R['vn_split']['other_circuits_rejected']} of the other 27. After conditioning, {refs_after} of the 11 reference sources was rejected (the generator with probability 0.49, on runs and autocorrelation) by the binomial variant and none by the Kolmogorov-Smirnov variant; for an independent biased input this is best read as a chance rejection."),
 ("table", "Audit results under the variants examined", ["Data and procedure", "Devices rejected", "Circuits rejected", "Pairs rejected", "Clean references rejected", "Biased references rejected"], cond_rows),
 ("h2", "4.3 Position dependence explains the remaining failures"),
 ("p", f"The share of ones depended significantly on bit position in all nine Type 3 circuits, in none of the other 27 circuits and in no reference source. The pattern repeats with period four (Figure 4, Table 3); the spread between the four positions ranged from {min(v[0] for v in p4.values()):.4f} to {max(v[0] for v in p4.values()):.4f}. Such a pattern would arise if four qubits with unequal bias were written in turn into one string, but the archive does not describe the circuit and this explanation is untested. Whatever the cause, neighbouring bits then have different probabilities, which breaks the assumption of the extractor: the two orders of an unequal pair are no longer equally likely."),
 ("fig", f4, "Share of ones by bit position modulo 8 for each device and circuit type.", "three-by-three line panels with shared axes; Type 3 emphasised; 300 dpi PNG."),
 ("table", "Type 3 circuits: share of ones by position modulo 4", ["Device", "Position 0", "Position 1", "Position 2", "Position 3", "Spread"], pos_rows),
 ("p", f"Applying the extractor within position streams, so that both bits of a pair come from the same position, left {vn4['ibm_circuits_rejected']} of 36 circuits and {vn4['ibm_groups_rejected']} of 180 pairs rejected, and no reference source was rejected. Min-entropy estimates after conditioning moved towards the range of the clean references (Figure 5), and the share of ones and lag-1 autocorrelation clustered around their ideal values (Figure 6)."),
 ("fig", f5, "Min-entropy estimates per IBM circuit before and after conditioning; four points per device (Basic, Types 1 to 3).", "two stacked scatter panels with three marker types and a reference band; 300 dpi PNG."),
 ("fig", f6, "Share of ones and mean lag-1 autocorrelation per data set.", "three scatter panels with shared axes and dotted ideal values; 300 dpi PNG."),
 ("h2", "4.4 Sample size"),
 ("p", f"On draws of 4 to 64 strings the audit flagged clean references in at most {pc(clean_max)} % of cases, whereas the per-string rule rose from {pc(S1['clean|4'][1])} % to {pc(S1['clean|64'][1])} % (Table 4). Sensitivity to a probability of one of 0.495 was weak with few strings: {pc(SB['PRNG_biased_505_0_495_1|16'])} % of draws at 16 strings and {pc(SB['PRNG_biased_505_0_495_1|64'])} % at 64 strings of 8192 bits. A bias of 0.49 was detected in {pc(SB['PRNG_biased_51_0_49_1|16'])} % of draws at 16 strings. A pass on a small sample therefore says little about small biases."),
 ("table", "Share of draws flagged, percent (A: audit; P: any p-value below 0.01)", ["Strings", "Clean A", "Clean P", "Biased A", "Biased P", "IBM A", "IBM P"], sens_rows),
 ("fig", f7, "Flagged share against the number of strings drawn.", "two line panels on a base-2 logarithmic axis; solid lines for the audit, dashed for the per-string rule; 300 dpi PNG."),
 ("h", "5. Discussion and limitations"),
 ("ul", ["All data come from one archive and one set of runs. Device calibration data and circuit definitions are not part of it, so causes are not established and nothing is concluded about the devices at other times.",
         "Only five basic tests and two min-entropy estimators were used. The full SP 800-22 and SP 800-90B batteries were not run. Passing here does not certify output for cryptographic use.",
         "The position-stream extractor and the screen threshold were chosen after looking at these data. Both need confirmation on independent runs.",
         "The extractor is only justified if bits within a stream are independent with a stable probability; drift over time was not tested.",
         "Sample-size results are conditional on finite pools of strings.",
         "Concatenating the four streams changes the order of the output bits."]),
 ("h", "6. Conclusions"),
 ("p", f"The raw output of all nine devices failed a device-level audit that passed the clean references. The public files include {sc['stuck']} nearly constant and {sc['repeat']} repeated strings. A standard extractor was not enough for the Type 3 circuits, whose bias changes with bit position with period four; respecting that structure removed all rejections under the five tests used. A health check for such sources should screen files, test position structure and then audit the conditioned output."),
 ("h", "Data and code availability"),
 ("p", "The sequences are public (Zenodo record 10542216, CC BY 4.0) and are not redistributed. Analysis code and result tables accompany this manuscript. An earlier version of the audit software is in the repository articles-qrng-audit; the availability statement and repository visibility are [to be confirmed before submission]."),
 ("h", "Declarations"),
 ("p", "Funding, competing interests (including any related patent filings) and author contributions: [to be confirmed]. Use of AI tools in analysis and drafting: [to be declared according to the venue policy]."),
 ("h", "References"),
 ("ref", ["Tamura K, Shikano Y. Quantum random numbers generated by a cloud superconducting quantum computer. In: International Symposium on Mathematics, Quantum Theory, and Cryptography. Mathematics for Industry, vol. 33. Springer; 2021. p. 17-37. (arXiv:1906.04410)",
          "Root O, Becker M. Does true randomness exist? Efficacy testing IBM quantum computers via statistical randomness. arXiv:2401.12250; 2024.",
          "Root O. QRNG Trial Sequences. Zenodo. https://doi.org/10.5281/zenodo.10542216",
          "Rukhin A, et al. A Statistical Test Suite for Random and Pseudorandom Number Generators for Cryptographic Applications. NIST SP 800-22 Rev. 1a; 2010.",
          "Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. Journal of the Royal Statistical Society B 1995;57:289-300.",
          "Turan MS, et al. Recommendation for the Entropy Sources Used for Random Bit Generation. NIST SP 800-90B; 2018.",
          "von Neumann J. Various techniques used in connection with random digits. National Bureau of Standards Applied Mathematics Series 1951;12:36-38."])]
info = write_docx(OUT / "qrng-position-bias-manuscript.docx", blocks)
(OUT / "abstract.txt").write_text(title + "\n\n" + abstract, encoding="utf-8")
(OUT / "figure_specs.json").write_text(json.dumps([{"file": Path(b[1]).name, "caption": b[2], "spec": b[3]} for b in blocks if b[0] == "fig"], indent=1), encoding="utf-8")
print(info, "abstract words", len(abstract.split()))
print(abstract)
