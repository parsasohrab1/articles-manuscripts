"""Draw the figures and write the Word manuscript from results/.

    python -I code/isci_paper.py results .
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from mslib import GREY, NAVY, ORANGE, PAL, RED, TEAL, flow, plt, write_docx

MODELS = ["LR", "NB", "KNN", "RF", "GB"]
LONG = {"LR": "logistic regression", "NB": "Gaussian naive Bayes", "KNN": "15 nearest neighbours", "RF": "random forest", "GB": "gradient boosting"}
METHODS = ["none", "platt", "isotonic", "beta", "temperature"]
LAB = {"none": "None", "platt": "Platt", "isotonic": "Isotonic", "beta": "Beta", "temperature": "Temperature"}
CAL = METHODS[1:]

res, out = Path(sys.argv[1]), Path(sys.argv[2])
figs = out / "figures"
figs.mkdir(exist_ok=True)
info = pd.read_csv(res / "datasets.csv")
means = pd.read_csv(res / "means.csv").set_index(["dataset", "model", "method"])
an = json.loads((res / "analysis.json").read_text())
diag = json.loads((res / "diagnostics.json").read_text())
summ = pd.DataFrame(an["summary"]).set_index(["metric", "model", "method"])
cmp_ = pd.DataFrame(an["comparisons"]).set_index(["metric", "model", "method"])
order = list(info.sort_values("events").dataset)
specs = []


def save(fig, name, caption, spec):
    fig.savefig(figs / name)
    plt.close(fig)
    specs.append({"file": name, "caption": caption, "spec": spec})
    return ("fig", figs / name, caption, spec)


def flowfig(name, steps, title, caption, spec, height=2.4):
    flow(figs / name, steps, title, height)
    specs.append({"file": name, "caption": caption, "spec": spec})
    return ("fig", figs / name, caption, spec)


def f4(x):
    return f"{x:.4f}"


def fp(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


g_abs = flowfig("fig1_graphical_abstract.png", [
    ("15 public data sets", f"{info.n.min()} to {info.n.max()} records, {info.events.min()} to {info.events.max()} events"),
    ("Design", "5 classifiers, 4 calibrators; calibrator fitted on inner cross-validated scores of the training folds"),
    ("Evaluation", "10 x 5-fold; Brier score (primary), log loss, ECE, AUC"),
    ("Finding", "large gains for naive Bayes and gradient boosting; none detected for logistic regression and nearest neighbours on the Brier score")],
    "Graphical abstract", "Graphical abstract of the study.",
    "Vector-style flow diagram, four boxes left to right, 10 x 2.6 in, 300 dpi, navy/teal/orange/red borders.", 2.6)
f_flow = flowfig("fig2_protocol.png", [
    ("Outer split", "stratified 5-fold, repeated 10 times"),
    ("Pre-processing", "median imputation and standardisation fitted on training data only"),
    ("Calibration scores", "inner stratified 5-fold predictions of the training part"),
    ("Refit", "base classifier refitted on the whole training part"),
    ("Test", "calibrator applied to outer test predictions")],
    "Evaluation protocol", "Nested evaluation protocol used for every data set, classifier and calibrator.",
    "Flow diagram, five boxes left to right, 10 x 2.4 in, 300 dpi.")

# forest plot of mean paired differences
fig, axes = plt.subplots(1, 5, figsize=(10, 2.8), sharey=True)
for ax, mn in zip(axes, MODELS):
    for i, m in enumerate(CAL):
        r = cmp_.loc[("brier", mn, m)]
        c = TEAL if r.p_holm < 0.05 else GREY
        ax.plot(r.ci, [i, i], color=c, lw=2)
        ax.plot(r.mean_diff, i, "o", color=c)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(mn)
    ax.tick_params(axis="x", labelsize=7)
    ax.locator_params(axis="x", nbins=4)
axes[0].set_yticks(range(4), [LAB[m] for m in CAL])
axes[0].invert_yaxis()
fig.supxlabel("Mean change in Brier score against no calibration (negative is better)", fontsize=9)
fig.tight_layout()
f_forest = save(fig, "fig3_brier_change.png", "Mean paired change in Brier score over 15 data sets with 95% bootstrap intervals. Green: Holm-adjusted p < 0.05; grey: not significant. Horizontal scales differ between panels.",
                "Five-panel forest plot (one panel per classifier), 10 x 2.8 in, 300 dpi; points are means, lines are percentile bootstrap intervals; zero line in black.")


def heat(metric, name, caption, vmax):
    cols = [(mn, m) for mn in MODELS for m in CAL]
    M = np.array([[means.loc[(d, mn, m), metric] - means.loc[(d, mn, "none"), metric] for mn, m in cols] for d in order])
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(np.clip(M, -vmax, vmax), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.grid(False)
    ax.set_yticks(range(len(order)), [f"{d} ({int(info.set_index('dataset').events[d])})" for d in order])
    ax.set_xticks(range(len(cols)), [LAB[m] for _, m in cols], rotation=90, fontsize=7)
    for k, mn in enumerate(MODELS):
        ax.text(4 * k + 1.5, -0.9, mn, ha="center", fontweight="bold")
        if k:
            ax.axvline(4 * k - 0.5, color="black", lw=1)
    fig.colorbar(im, ax=ax, label=f"change (clipped at ±{vmax})", fraction=0.025)
    return save(fig, name, caption, f"Heat map, 15 data sets (sorted by number of events, in brackets) by 20 classifier-calibrator pairs, diverging red-blue map clipped at ±{vmax}, 10 x 5 in, 300 dpi.")


f_heat = heat("brier", "fig4_brier_by_dataset.png", "Change in Brier score against no calibration for every data set (blue: calibration helps).", 0.03)
f_heatll = heat("logloss", "fig5_logloss_by_dataset.png", "Change in log loss against no calibration for every data set (blue: calibration helps).", 0.15)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
for ax, metric, ttl in zip(axes, ["brier", "logloss"], ["Brier score", "Log loss"]):
    for j, m in enumerate(METHODS):
        ax.bar(np.arange(5) + (j - 2) * 0.16, [summ.loc[(metric, mn, m), "rank"] for mn in MODELS], 0.16, color=PAL[j], label=LAB[m])
    ax.set_xticks(range(5), MODELS)
    ax.set_title(ttl)
axes[0].set_ylabel("Mean rank (1 = best)")
fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06))
f_rank = save(fig, "fig6_ranks.png", "Mean rank of the five options over 15 data sets (lower is better).",
              "Grouped bar chart, two panels (Brier score, log loss), five classifiers by five options, 10 x 3.2 in, 300 dpi.")

fig, axes = plt.subplots(1, 5, figsize=(10, 2.6), sharey=False)
ev = info.set_index("dataset").events
for ax, mn in zip(axes, MODELS):
    d = [means.loc[(k, mn, "isotonic"), "brier"] - means.loc[(k, mn, "platt"), "brier"] for k in order]
    ax.scatter(ev[order], d, color=NAVY, s=18)
    ax.axhline(0, color=RED, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([50, 100, 200, 500], ["50", "100", "200", "500"])
    ax.minorticks_off()
    rho, p, _ = an["spearman"][f"brier|{mn}"]
    ax.set_title(f"{mn}: rho={rho:.2f}", fontsize=9)
    ax.tick_params(labelsize=7)
axes[0].set_ylabel("Brier: isotonic − Platt", fontsize=8)
fig.supxlabel("Number of events (log scale)", fontsize=9)
fig.tight_layout()
f_sc = save(fig, "fig7_events.png", "Difference between isotonic and Platt Brier scores against the number of events; points below the red line favour isotonic regression.",
            "Five scatter panels, logarithmic horizontal axis, Spearman correlation in each title, 10 x 2.6 in, 300 dpi.")

c = lambda metric, mn, m: cmp_.loc[(metric, mn, m)]
m_ = lambda metric, mn, m: summ.loc[(metric, mn, m), "mean"]
sig = lambda metric: int((cmp_.loc[metric].p_holm < 0.05).sum())
pct = lambda mn, k, t: 100 * diag[mn][k] / diag[mn][t]

TITLE = "Post-hoc probability calibration on small clinical tabular data: a repeated cross-validation study of four calibrators, five classifiers and fifteen public data sets"
ABSTRACT = (
    "Risk estimates are useful for decisions only if the stated probabilities agree with observed frequencies, yet many clinical tabular data sets are too small to set aside a separate calibration sample. "
    f"We compare no calibration with Platt scaling, isotonic regression, beta calibration and temperature scaling for five classifiers on 15 public clinical and health-related data sets ({info.n.min()} to {info.n.max()} records, {info.events.min()} to {info.events.max()} events). "
    "Calibrators were fitted on inner cross-validated scores and evaluated by ten repetitions of stratified five-fold cross-validation; comparisons used data-set means, Wilcoxon tests with Holm correction and bootstrap intervals. "
    f"On the Brier score, all four calibrators improved Gaussian naive Bayes (mean change {c('brier','NB','temperature').mean_diff:.3f} to {c('brier','NB','isotonic').mean_diff:.3f}, driven partly by a few severe cases) and gradient boosting ({c('brier','GB','isotonic').mean_diff:.4f} to {c('brier','GB','platt').mean_diff:.4f}). "
    "No calibrator changed the Brier score significantly for logistic regression, nearest neighbours or random forest after correction. "
    f"Isotonic regression raised the log loss of logistic regression on all 15 data sets (mean {c('logloss','LR','isotonic').mean_diff:+.3f}). "
    "Beta calibration and Platt scaling were the most frequent log-loss winners. The benefit of calibration therefore depends on the classifier and the metric, and it should not be applied to small clinical data by default."
)

summary_rows = lambda metric: [[LONG[mn]] + [f4(m_(metric, mn, m)) for m in METHODS] for mn in MODELS]
cmp_rows = lambda metric: [[mn, LAB[m], f"{r.mean_diff:+.4f}", f"[{r.ci[0]:+.4f}, {r.ci[1]:+.4f}]", f"{r.better}/{r.worse}", fp(r.p_holm)]
                           for mn in MODELS for m in CAL for r in [c(metric, mn, m)]]
HEAD = ["Classifier"] + [LAB[m] for m in METHODS]
CH = ["Model", "Calibrator", "Mean change", "95% interval", "Better/worse", "Holm p"]

blocks = [
    ("title", TITLE),
    ("p", "Authors and affiliations: [to be completed]. Draft for author review; not submitted."),
    ("h", "Abstract"), ("p", ABSTRACT),
    ("p", "Keywords: probability calibration; clinical prediction; small data; Brier score; isotonic regression; beta calibration"),
    g_abs,
    ("h", "1. Introduction"),
    ("p", "A clinical risk model is usually read as a probability: a stated risk of 20% should correspond to an event in about one of five similar patients. Poor calibration can mislead decisions even when discrimination is good [9]. Post-hoc calibration maps the scores of a fitted classifier to probabilities. Platt scaling fits a logistic curve [2]; isotonic regression fits a non-decreasing step function [3]; beta calibration uses a three-parameter family that includes the identity map [5]; temperature scaling rescales the logit by one parameter and was popularised for neural networks [6]."),
    ("p", "Earlier comparisons showed that the shape and the size of the miscalibration depend on the learner and that isotonic regression needs more data than Platt scaling [4,5]. Clinical tabular data sets are often small, so the calibration sample has to be obtained by cross-validation from the same records used for fitting. This study asks, under that constraint: (i) for which classifiers does post-hoc calibration improve probability accuracy; (ii) which calibrator performs best; and (iii) does the comparison between isotonic regression and Platt scaling change with the number of events."),
    ("p", "The contribution is empirical. No new calibrator is proposed. The study offers a fully scripted comparison with data-set level inference, reports metrics on which calibration fails as well as those on which it helps, and documents an implementation effect, probability clipping, that changes discrimination."),
    ("h", "2. Materials and methods"),
    ("h2", "2.1 Data sets"),
    ("p", "Fifteen binary data sets were downloaded from the UCI Machine Learning Repository [11]; the archive addresses, sizes and SHA-256 hashes are recorded in the manifest supplied with the code. Exact duplicate rows (identical predictors and outcome) were removed. Identifier columns were dropped, as were the follow-up time of the heart-failure data (it is only known after the outcome period) and the BI-RADS assessment of the mammographic data (an expert judgement of the outcome). Categorical predictors were one-hot encoded. Not every set is a diagnostic cohort: the transfusion set records donation behaviour."),
    ("table", "Data sets after duplicate removal.", ["Data set", "UCI id", "Records (raw)", "Predictors", "Events", "Missing cells", "Event"],
     [[r.dataset, r.uci, f"{r.n} ({r.n_raw})", r.features, r.events, r.missing, r.event] for r in info.itertuples()]),
    ("h2", "2.2 Classifiers and calibrators"),
    ("p", "The classifiers were L2-regularised logistic regression (LR), Gaussian naive Bayes (NB), a 15-nearest-neighbour classifier (KNN), a random forest of 200 trees (RF) and gradient boosting (GB), implemented in scikit-learn with otherwise default settings [10]. No hyper-parameter search was performed. Median imputation and standardisation were fitted inside each training part."),
    ("p", "Platt scaling was a practically unpenalised logistic regression on the logit of the score with Platt's smoothed targets. Beta calibration used the features ln q and -ln(1-q) with non-negative coefficients, refitting with one feature when a coefficient was negative; it was fitted with the same smoothed targets, a choice that differs slightly from the original description. Isotonic regression was clipped outside the calibration range. Temperature scaling estimated one positive inverse temperature on the logit by minimising the negative log-likelihood. Logits and logarithms used probabilities clipped to [10^-6, 1-10^-6]."),
    ("h2", "2.3 Evaluation protocol"),
    ("p", "Each data set was evaluated with stratified five-fold cross-validation repeated ten times (Fig. 2). Within each outer training part, out-of-fold scores from an inner stratified five-fold cross-validation were used to fit each calibrator; the classifier was then refitted on the whole training part and its test scores were calibrated. Test records never influenced the pre-processing, the classifier or the calibrator. All five options were compared on identical splits."),
    f_flow,
    ("p", "The Brier score [1] was the primary metric and log loss the secondary one. The expected calibration error (ECE, ten equal-count bins, computed per test fold) [7] and the area under the ROC curve (AUC) were descriptive. Because folds of small data sets are noisy and ECE with ten bins is biased upward in small samples, ECE is reported without tests."),
    ("h2", "2.4 Statistical analysis"),
    ("p", "The unit of analysis was the data set: metrics were averaged over the 50 test folds. For each classifier, a Friedman test compared the five options [8]. Each calibrator was compared with no calibration by a two-sided Wilcoxon signed-rank test; Holm's correction was applied separately to the 20 tests of each metric. Percentile bootstrap intervals (10,000 resamples of data sets) accompany mean differences. The association between the number of events and the isotonic-minus-Platt difference was measured by Spearman correlation, without multiplicity correction. This plan was fixed before the results were inspected; it was not registered."),
    ("h", "3. Results"),
    ("h2", "3.1 Brier score"),
    ("table", "Mean Brier score over 15 data sets (lower is better).", HEAD, summary_rows("brier")),
    ("p", f"Friedman tests indicated differences among the five options for every classifier (p from {min(v[1] for k, v in an['friedman'].items() if k.startswith('brier')):.1e} to {max(v[1] for k, v in an['friedman'].items() if k.startswith('brier')):.3f}). In the pairwise comparisons, {sig('brier')} of 20 reached Holm-adjusted p < 0.05, all for NB and GB (Table 3, Fig. 3). For NB the mean gain was large ({c('brier','NB','beta').mean_diff:.3f} with beta calibration) but the median was much smaller ({c('brier','NB','beta').median_diff:.3f}): the thoracic-surgery, liver, hepatitis and SPECTF sets, on which raw NB probabilities were very poor, dominate the mean. For GB, Platt scaling and beta calibration lowered the Brier score on 14 of 15 sets, by about 0.01 on average. For RF the intervals of Platt, beta and temperature scaling lay below zero but none survived correction. LR and KNN showed mean changes of about 0.003 or less."),
    ("table", "Brier score: paired comparisons with no calibration.", CH, cmp_rows("brier")),
    f_forest, f_heat,
    ("h2", "3.2 Log loss"),
    ("table", "Mean log loss over 15 data sets (lower is better).", HEAD, summary_rows("logloss")),
    ("p", f"On log loss, {sig('logloss')} of 20 comparisons were significant after correction (Table 5). Platt and beta calibration improved NB, KNN (14 and 15 of 15 sets) and GB. Isotonic regression was significantly worse than no calibration for LR, raising the log loss on all 15 sets (mean {c('logloss','LR','isotonic').mean_diff:+.3f}). A step function can return exactly 0 or 1, and one wrong record then costs -ln(10^-6), so the size of this penalty depends on the clipping constant."),
    ("table", "Log loss: paired comparisons with no calibration.", CH, cmp_rows("logloss")),
    f_heatll,
    ("h2", "3.3 Ranking, calibration error and discrimination"),
    ("p", "Over the 75 data set and classifier combinations, the lowest Brier score was obtained by Platt scaling in 19, beta calibration in 16, isotonic regression in 15, no calibration in 13 and temperature scaling in 12. For log loss the counts were 27 (beta), 24 (Platt), 11 (temperature), 8 (none) and 5 (isotonic). Figure 6 shows mean ranks."),
    f_rank,
    ("table", "Mean expected calibration error (descriptive).", HEAD, summary_rows("ece")),
    ("table", "Mean area under the ROC curve (descriptive).", HEAD, summary_rows("auc")),
    ("p", f"Isotonic regression had the worst mean AUC rank for every classifier, as expected from the ties it creates. The parametric calibrators also lowered the mean AUC of NB (from {m_('auc','NB','none'):.3f} to {m_('auc','NB','platt'):.3f} with Platt scaling) although they are monotone in principle. A diagnostic rerun on identical splits showed that {pct('NB','clipped','total'):.1f}% of NB test probabilities and {pct('KNN','clipped','total'):.1f}% of KNN test probabilities lay beyond the clipping limits and became tied ({pct('LR','clipped','total'):.1f}% for LR), and that the fitted Platt slope was negative in {diag['NB']['negative_slope']} of {diag['NB']['folds']} NB folds, which reverses the ranking in those folds. The relative contribution of the two causes was not separated."),
    ("h2", "3.4 Number of events"),
    ("p", "The isotonic-minus-Platt difference in Brier score correlated negatively with the number of events for GB (rho = {:.2f}, p = {}) and RF (rho = {:.2f}, p = {}), but not for LR, NB or KNN (Fig. 7). Even for GB, isotonic regression beat Platt scaling on only {} of 15 sets. With 15 data sets that differ in much more than size, this is a weak, exploratory association and gives no threshold number of events.".format(
        an["spearman"]["brier|GB"][0], fp(an["spearman"]["brier|GB"][1]), an["spearman"]["brier|RF"][0], fp(an["spearman"]["brier|RF"][1]), an["spearman"]["brier|GB"][2])),
    f_sc,
    ("h", "4. Discussion"),
    ("p", "Calibration helped where the raw classifier was known to be distorted: naive Bayes, which pushes probabilities towards 0 and 1, and boosted trees with default settings. For logistic regression, whose fit already optimises a proper scoring rule, no Brier gain was detected. This agrees in direction with earlier work on larger data [4,5]. A non-significant result with 15 data sets is not evidence of equivalence."),
    ("p", "In practice, the results suggest measuring calibration for the chosen classifier before adding a calibrator, preferring a parametric map (Platt or beta) when the sample is small, and avoiding unsmoothed isotonic outputs when log loss or extreme probabilities matter. A calibrator should not be used to repair a classifier that is badly misspecified for the data; choosing a better base model may be preferable, which this study did not test as a competing strategy."),
    ("h2", "4.1 Limitations"),
    ("ul", ["Public benchmark data sets are old, heterogeneous and without external validation cohorts; nothing here establishes clinical usefulness.",
            "Removing exact duplicates can delete genuine repeated patients when predictors are few or discrete (Haberman, transfusion, mammographic, original breast-cancer and early-diabetes sets), which changes their prevalence. A sensitivity analysis with duplicates retained was not run.",
            "Classifiers used default hyper-parameters; tuning may reduce or enlarge the miscalibration.",
            "Several sets share a clinical domain and two concern breast cancer, so they are not fully independent units.",
            "Repeated cross-validation reduces split noise but does not add patients.",
            "Log loss depends on the clipping constant; only one value was used.",
            "Decision-analytic measures such as net benefit and recalibration curves were not assessed."]),
    ("h", "5. Conclusion"),
    ("p", "On 15 small public data sets, post-hoc calibration clearly improved naive Bayes and gradient boosting, showed no corrected Brier improvement for logistic regression, nearest neighbours or random forest, and could harm log loss when isotonic regression was used. The choice to calibrate should be made per classifier and per metric."),
    ("h", "Data and code availability"),
    ("p", "All data sets are public at the UCI Machine Learning Repository; each has its own licence and citation requirements. Scripts, file hashes, fold-level results and the figure code accompany this manuscript. The code repository address is [to be completed]."),
    ("h", "Declarations"),
    ("p", "Funding: [to be completed]. Competing interests: [to be completed]. Ethics: only public, de-identified data were analysed; [authors to confirm local requirements]. Use of generative AI: an AI assistant was used to write the analysis code and a first draft of the text; [authors to review, verify and state according to journal policy]."),
    ("h", "References"),
    ("ref", ["G.W. Brier, Verification of forecasts expressed in terms of probability, Monthly Weather Review 78 (1950) 1-3.",
             "J. Platt, Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods, in: Advances in Large Margin Classifiers, MIT Press, 1999, pp. 61-74.",
             "B. Zadrozny, C. Elkan, Transforming classifier scores into accurate multiclass probability estimates, in: Proc. 8th ACM SIGKDD, 2002, pp. 694-699.",
             "A. Niculescu-Mizil, R. Caruana, Predicting good probabilities with supervised learning, in: Proc. 22nd ICML, 2005, pp. 625-632.",
             "M. Kull, T. Silva Filho, P. Flach, Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers, in: Proc. AISTATS, PMLR 54, 2017, pp. 623-631.",
             "C. Guo, G. Pleiss, Y. Sun, K.Q. Weinberger, On calibration of modern neural networks, in: Proc. 34th ICML, 2017, pp. 1321-1330.",
             "M.P. Naeini, G.F. Cooper, M. Hauskrecht, Obtaining well calibrated probabilities using Bayesian binning, in: Proc. AAAI, 2015, pp. 2901-2907.",
             "J. Demsar, Statistical comparisons of classifiers over multiple data sets, J. Mach. Learn. Res. 7 (2006) 1-30.",
             "B. Van Calster, D.J. McLernon, M. van Smeden, L. Wynants, E.W. Steyerberg, Calibration: the Achilles heel of predictive analytics, BMC Medicine 17 (2019) 230.",
             "F. Pedregosa et al., Scikit-learn: machine learning in Python, J. Mach. Learn. Res. 12 (2011) 2825-2830.",
             "M. Kelly, R. Longjohn, K. Nottingham, The UCI Machine Learning Repository, https://archive.ics.uci.edu."]),
]
stats = write_docx(out / "isci-clinical-calibration-manuscript.docx", blocks)
(out / "abstract.txt").write_text(ABSTRACT, encoding="utf-8")
(out / "figure_specs.json").write_text(json.dumps(specs, indent=1, ensure_ascii=False), encoding="utf-8")
(out / "README.md").write_text(f"""# {TITLE}

Target venue: Information Sciences (Elsevier). Draft for author review; not submitted.

Data: 15 UCI Machine Learning Repository archives; URLs, sizes and SHA-256 hashes in `data/manifest.json`.

Run from this folder (Python 3.10; numpy 2.1, pandas 2.2, scikit-learn 1.5, scipy 1.14, joblib, matplotlib, python-docx, xlrd):

    python -I code/download.py data
    python -I code/study.py data results
    python -I code/analyse.py results
    python -I code/diagnostics.py data results
    python -I code/isci_paper.py results .

Open points before submission: authors, affiliations, funding and declarations; verify every reference and the UCI licences; no sensitivity analysis for duplicate removal; default hyper-parameters only. The study proposes no new method, so its fit and novelty for Information Sciences are weak and rejection at screening is a real risk. The ClinCal tracking row on the dashboard says an older submission could not be verified: confirm that no calibration paper on these data is already under review. ReadmitCal (diabetes readmission calibration) is a separate, already submitted paper on a different data set.
""", encoding="utf-8")
print(stats, len(ABSTRACT.split()), "abstract words")
