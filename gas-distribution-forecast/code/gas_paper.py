"""Figures and Word manuscript for the gas study. Usage: python gas_paper.py <results dir> <paper dir>"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mslib import GOLD, GREY, NAVY, ORANGE, RED, TEAL, flow, write_docx

RES, OUT = (Path(a) for a in sys.argv[1:3])
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
R = json.loads((RES / "results.json").read_text(encoding="utf-8"))["zones"]
Z = list(R)
SHORT = {"NL_L": "Netherlands (L-gas)", "BE_H": "Belgium (H-gas)", "IT": "Italy"}
COL = {"NL_L": NAVY, "BE_H": TEAL, "IT": ORANGE}
MOD = ["previous week", "previous day", "ridge", "gradient boosting", "random forest"]
MC = [GREY, GOLD, TEAL, ORANGE, NAVY]


def save(fig, name):
    p = FIG / name
    fig.savefig(p)
    plt.close(fig)
    return p


def read(name):
    return pd.read_csv(RES / name, index_col=0, parse_dates=True)


f1 = FIG / "fig1_pipeline.png"
flow(f1, [("ENTSOG flows", "Daily physical exit flow of three distribution points, 2022 to 2025"),
          ("Quality checks", "One record per gas day; unit and direction asserted; gap and outlier rules"),
          ("Features", "Demand lags, calendar terms, NASA POWER temperature"),
          ("Monthly refit", "Ridge, random forest, gradient boosting; test 2024 to 2025"),
          ("Decision metric", "Balancing margin from the last 90 errors and its observed coverage")], "Study pipeline", height=2.6)

fig, axes = plt.subplots(3, 1, figsize=(10, 6.4), sharex=True)
for ax, z in zip(axes, Z):
    d = read(f"series_{z}.csv")
    ax.plot(d.index, d.y, color=COL[z], lw=0.9)
    ax.set_ylabel("GWh/day"), ax.set_title(SHORT[z], loc="left")
    ax.axvline(pd.Timestamp("2024-01-01"), color=RED, ls="--", lw=1)
    tw = ax.twinx()
    tw.plot(d.index, d.t, color=GREY, lw=0.6, alpha=0.6), tw.invert_yaxis(), tw.set_ylabel("°C (inverted)"), tw.grid(False)
fig.tight_layout()
f2 = save(fig, "fig2_series.png")

fig, axes = plt.subplots(1, 3, figsize=(10, 3.1))
for ax, z in zip(axes, Z):
    d = read(f"series_{z}.csv").dropna()
    ax.scatter(d.t, d.y, s=5, alpha=0.45, c=np.where(d.index.dayofweek >= 5, ORANGE, NAVY))
    ax.set_title(f"{SHORT[z]}  r = {R[z]['temperature_correlation']}"), ax.set_xlabel("Daily mean temperature, °C")
axes[0].set_ylabel("Offtake, GWh/day")
axes[0].scatter([], [], c=NAVY, s=12, label="weekday"), axes[0].scatter([], [], c=ORANGE, s=12, label="weekend"), axes[0].legend()
fig.tight_layout()
f3 = save(fig, "fig3_temperature.png")

fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
for ax, z in zip(axes, Z):
    v = [R[z]["metrics"][m]["mape_pct"] for m in MOD]
    ax.barh(MOD, v, color=MC)
    for i, x in enumerate(v):
        ax.text(x + max(v) * 0.02, i, f"{x:.1f}", va="center", fontsize=8)
    ax.set_title(SHORT[z]), ax.set_xlabel("Mean absolute percentage error, %"), ax.set_xlim(0, max(v) * 1.18)
for ax in axes[1:]:
    ax.set_yticklabels([])
fig.tight_layout()
f4 = save(fig, "fig4_mape.png")

fig, axes = plt.subplots(3, 1, figsize=(10, 6.2), sharex=True)
for ax, z in zip(axes, Z):
    p = read(f"predictions_{z}.csv")
    b = R[z]["best_model"]
    ax.plot(p.index, p.y, color="black", lw=1, label="measured"), ax.plot(p.index, p[b], color=COL[z], lw=0.9, label=b)
    ax.set_title(SHORT[z], loc="left"), ax.set_ylabel("GWh/day"), ax.legend(ncol=2, loc="upper center"), ax.set_ylim(0, p.y.max() * 1.22)
fig.tight_layout()
f5 = save(fig, "fig5_forecast.png")

labels = list(R[Z[0]]["ablation_mae_gwh"])
fig, ax = plt.subplots(figsize=(10, 3.2))
for i, z in enumerate(Z):
    a = R[z]["ablation_mae_gwh"]
    v = [a[k] / a["full"] for k in labels]
    x = np.arange(len(labels)) + (i - 1) * 0.27
    ax.bar(x, v, 0.25, color=COL[z], label=SHORT[z])
    for xx, vv, k in zip(x, v, labels):
        ax.text(xx, vv + 0.02, f"{a[k]:.0f}", ha="center", fontsize=7.5)
ax.axhline(1, color=RED, ls="--", lw=1), ax.set_xticks(range(len(labels)), labels), ax.set_ylabel("Error relative to full inputs"), ax.legend(ncol=3), ax.set_ylim(0, 2.2)
ax.set_title("Effect of the inputs (numbers: mean absolute error, GWh/day)")
f6 = save(fig, "fig6_ablation.png")

fig, axes = plt.subplots(2, 3, figsize=(10, 5.2))
for j, z in enumerate(Z):
    p = read(f"predictions_{z}.csv")
    for m, c in (("previous day", GOLD), (R[z]["best_model"], COL[z])):
        a = (p[m] - p.y).abs()
        q = a.rolling(90, min_periods=90).quantile(0.95).shift(1)
        axes[0, j].plot(q.index, q, color=c, lw=1.2, label=m)
        cov = (a <= q).where(q.notna()).astype(float).rolling(90, min_periods=90).mean() * 100
        axes[1, j].plot(cov.index, cov, color=c, lw=1.2)
    axes[0, j].set_title(SHORT[z]), axes[1, j].axhline(95, color=RED, ls="--", lw=1), axes[1, j].set_ylim(70, 101)
    for k in (0, 1):
        axes[k, j].tick_params(axis="x", rotation=45, labelsize=7)
axes[0, 0].set_ylabel("Margin, GWh/day"), axes[1, 0].set_ylabel("Coverage, last 90 days, %"), axes[0, 0].legend(fontsize=8)
fig.tight_layout()
f7 = save(fig, "fig7_margin.png")

best = {z: R[z]["metrics"][R[z]["best_model"]] for z in Z}
rng_ = lambda xs: f"{min(xs):.1f} to {max(xs):.1f}"
mape_b = rng_([best[z]["mape_pct"] for z in Z])
skill = rng_([best[z]["skill_vs_previous_day_pct"] for z in Z])
mshare_b = rng_([R[z]["margin"][R[z]["best_model"]]["margin_share_of_mean_demand_pct"] for z in Z])
mshare_p = rng_([R[z]["margin"]["previous day"]["margin_share_of_mean_demand_pct"] for z in Z])
covs = rng_([R[z]["margin"][R[z]["best_model"]]["coverage_pct"] for z in Z])
title = "Day-ahead forecasting of gas distribution offtake and the balancing margin it implies: an open-data study of three European zones"
abstract = ("Balancing a gas network requires a forecast of distribution offtake and a margin for its error. Using only open data, this study forecasts daily physical exit flow at distribution points in the Netherlands (low-calorific gas), Belgium (high-calorific gas) and Italy, with data from the ENTSOG Transparency Platform and temperature from NASA POWER. "
 f"Ridge regression, random forest and gradient boosting were refitted every month and tested on 2024 and 2025. The random forest had the lowest error in all three zones, with mean absolute percentage errors of {mape_b} % and mean absolute errors {skill} % below those of repeating the previous day. "
 f"A margin set at the 95th percentile of the last 90 absolute errors averaged {mshare_b} % of mean demand for the random forest against {mshare_p} % for the previous-day rule, but covered only {covs} % of days, short of the nominal 95 %. "
 "The accuracy relies on same-day observed temperature standing in for a weather forecast: using only past temperature, or learning demand one day later, raised the error substantially. The results are a benchmark on public data, not a validation in an operating control room.")

rows_d = [[SHORT[z], R[z]["data"]["first_day"], R[z]["data"]["valid_days"], f"{R[z]['data']['calendar_days'] - R[z]['data']['valid_days']}", R[z]["data"]["mean_gwh"], R[z]["data"]["min_gwh"], R[z]["data"]["max_gwh"], R[z]["data"]["provisional_share_pct"]] for z in Z]
rows_m = [[SHORT[z], m, f"{v['mae_gwh']:.2f}", f"{v['rmse_gwh']:.2f}", f"{v['mape_pct']:.2f}", f"{v['bias_gwh']:.2f}", "reference" if m == "previous day" else f"{v['mae_minus_previous_day_ci'][0]:.1f} to {v['mae_minus_previous_day_ci'][1]:.1f}"] for z in Z for m, v in R[z]["metrics"].items()]
rows_a = [[k] + [f"{R[z]['ablation_mae_gwh'][k]:.2f}" for z in Z] for k in labels]
rows_g = [[SHORT[z], m, v["mean_margin_gwh"], v["margin_share_of_mean_demand_pct"], v["coverage_pct"], v["days"]] for z in Z for m, v in R[z]["margin"].items()]
A = lambda z, k: R[z]["ablation_mae_gwh"][k]

blocks = [("title", title), ("small", "Authors and affiliations: [to be completed]. Draft for author review; not submitted."),
 ("h", "Abstract"), ("p", abstract), ("p", "Keywords: natural gas demand; day-ahead forecasting; distribution network; balancing margin; random forest; open data; ENTSOG"),
 ("h", "1. Introduction"),
 ("p", "Short-term forecasts of gas offtake support nominations, line-pack management and balancing. The literature on consumption forecasting is large and covers statistical and machine-learning methods at levels from households to countries [1]. Two practical questions are less often answered together on fully open data: how much a learned model gains over naive rules when it is refitted and tested forward in time, and what that gain means for the margin an operator would have to hold against forecast error."),
 ("p", "This study answers both for three distribution aggregates published on the ENTSOG Transparency Platform [2]. All data and code can be re-run, the comparison includes two naive rules, and the sensitivity to the information actually available at forecast time is reported."),
 ("h", "2. Data"),
 ("p", "Daily physical exit flow (kWh per gas day) was retrieved from the ENTSOG public interface for the point Local Distribution Companies L-gas (Netherlands) and the distribution points of Belgium (H-gas) and Italy. For the Dutch point the interface returned data from July 2022; the Belgian and Italian series start in January 2022. All series end in December 2025. Daily mean air temperature at one representative location per zone (52.10 N 5.18 E, 50.85 N 4.35 E and 45.46 N 9.19 E) came from NASA POWER [3]."),
 ("table", "Series after quality checks (GWh per day)", ["Zone", "Start", "Valid days", "Missing days", "Mean", "Minimum", "Maximum", "Provisional records, %"], rows_d),
 ("fig", f2, "Daily offtake (colour, left axis) and temperature (grey, inverted right axis). The dashed line marks the start of the test period.", "three stacked time-series panels with twin axes; 300 dpi PNG."),
 ("fig", f3, "Offtake against temperature by zone; the title gives the Pearson correlation.", "three scatter panels; weekdays and weekends in two colours; 300 dpi PNG."),
 ("h", "3. Methods"),
 ("fig", f1, "Study pipeline.", "five-step left-to-right flow diagram; graphical abstract; 300 dpi PNG."),
 ("p", "Records were reduced to the latest update per gas day. Non-positive values and values below 20 % or above three times a centred 15-day median were to be set to missing; no value met either rule. Missing days were not imputed."),
 ("p", "The inputs were demand one, two and seven days earlier, the mean of the previous seven days, weekday indicators, annual sine and cosine terms, and temperature of the target day, of the previous day, its three-day mean, heating degrees below 16 °C and the one-day change. Models learned the change from the previous day. Ridge regression [4] used standardised inputs with a cross-validated penalty; the random forest [5] used 200 trees, a minimum leaf size of three and 60 % of the inputs per split; histogram gradient boosting [6] used a learning rate of 0.05 and 300 iterations, all in scikit-learn [7]. Hyperparameters were fixed in advance and not tuned on the test period. Each month of 2024 and 2025 was predicted by a model fitted on all earlier days."),
 ("p", "The references were the previous-day and previous-week values. Mean absolute error differences from the previous-day rule were given 95 % intervals by resampling seven-day blocks 3000 times [8]. For the balancing margin, the margin of each day was the 95th percentile of the absolute errors of the preceding 90 test days, and coverage was the share of days whose error stayed within it. For the input analysis, the model with the lowest test error was refitted without temperature, with previous-day temperature only, and with demand known only up to two days earlier. Because that model was chosen on the test period, its advantage over the other learned models should be read as descriptive."),
 ("h", "4. Results"),
 ("h2", "4.1 Forecast accuracy"),
 ("p", f"All learned models improved on both naive rules, and every interval of the error difference excluded zero (Table 2). The random forest gave mean absolute errors of {best['NL_L']['mae_gwh']:.2f}, {best['BE_H']['mae_gwh']:.2f} and {best['IT']['mae_gwh']:.2f} GWh per day in the Dutch, Belgian and Italian series. Gradient boosting was close behind. The previous-week rule was the weakest, which is consistent with demand driven mainly by temperature."),
 ("table", "Test results, 2024 to 2025 (errors in GWh per day)", ["Zone", "Model", "MAE", "RMSE", "MAPE, %", "Bias", "MAE difference, 95 % interval"], rows_m),
 ("fig", f4, "Mean absolute percentage error by model and zone.", "three horizontal bar panels with shared model order and value labels; 300 dpi PNG."),
 ("fig", f5, "Measured offtake and random forest forecast in the test period.", "three stacked line panels; measured in black; 300 dpi PNG."),
 ("h2", "4.2 What the forecast needs"),
 ("p", f"Removing the same-day temperature raised the error from {A('NL_L','full')} to {A('NL_L','past temperature only')} GWh per day in the Netherlands, from {A('BE_H','full')} to {A('BE_H','past temperature only')} in Belgium and from {A('IT','full')} to {A('IT','past temperature only')} in Italy. Past temperature added little beyond demand lags. Learning demand only up to two days earlier had a comparable or larger cost (Table 3). An operational system would use a weather forecast, so its error should lie between the full and the past-temperature cases; that value is [to be measured] with archived forecasts."),
 ("table", "Mean absolute error of the random forest by available inputs, GWh per day", ["Inputs"] + [SHORT[z] for z in Z], rows_a),
 ("fig", f6, "Error relative to the full input set.", "grouped bar chart, one colour per zone, dashed reference at 1; 300 dpi PNG."),
 ("h2", "4.3 Balancing margin"),
 ("p", f"The random forest reduced the average margin to {mshare_b} % of mean demand, from {mshare_p} % under the previous-day rule (Table 4). Observed coverage stayed below the nominal 95 % in all zones for the forest, because errors rise when the heating season begins and a trailing window reacts late. A seasonal or temperature-conditional margin is a natural next step and is [to be measured]."),
 ("table", "Trailing-window 95 % balancing margin", ["Zone", "Forecast", "Mean margin, GWh/day", "Share of mean demand, %", "Coverage, %", "Days"], rows_g),
 ("fig", f7, "Margin through time (top) and coverage over the last 90 days (bottom). The dashed line marks 95 %.", "two-row, three-column line panels; previous-day rule and random forest; 300 dpi PNG."),
 ("h", "5. Discussion and limitations"),
 ("ul", ["Same-day reanalysis temperature is used in the full model; this is optimistic compared with operations.",
         "One temperature point represents each national zone.",
         "Published values may be revised after the gas day, and part of the Dutch series is still marked provisional; the data available to an operator at forecast time may differ.",
         "The test covers two years and three aggregates; results need not transfer to city gates or individual networks.",
         "The margin concerns forecast error only. It does not model line pack, storage, contracts or imbalance prices, and no cost saving is claimed.",
         "No holiday calendar was used."]),
 ("h", "6. Conclusions"),
 ("p", f"On three open distribution series, a monthly refitted random forest reached {mape_b} % mean absolute percentage error and roughly halved the error of the previous-day rule, with a smaller balancing margin whose coverage was nevertheless below its nominal level. Weather information at the target day and timely demand data are the main requirements for keeping that accuracy."),
 ("h", "Data and code availability"),
 ("p", "ENTSOG Transparency Platform and NASA POWER (open access; the terms of both sources should be checked before redistribution). Download script, analysis code and result files accompany this manuscript; repository visibility is [to be confirmed before submission]."),
 ("h", "Declarations"),
 ("p", "Funding, competing interests and author contributions: [to be completed]. Use of AI tools in analysis and drafting: [to be declared according to the journal policy]."),
 ("h", "References"),
 ("ref", ["Soldo B. Forecasting natural gas consumption. Applied Energy 2012;92:26-37.",
          "European Network of Transmission System Operators for Gas. ENTSOG Transparency Platform. https://transparency.entsog.eu",
          "NASA Langley Research Center. Prediction Of Worldwide Energy Resources (POWER) project. https://power.larc.nasa.gov",
          "Hoerl AE, Kennard RW. Ridge regression: biased estimation for nonorthogonal problems. Technometrics 1970;12:55-67.",
          "Breiman L. Random forests. Machine Learning 2001;45:5-32.",
          "Friedman JH. Greedy function approximation: a gradient boosting machine. Annals of Statistics 2001;29:1189-1232.",
          "Pedregosa F, et al. Scikit-learn: machine learning in Python. Journal of Machine Learning Research 2011;12:2825-2830.",
          "Kunsch HR. The jackknife and the bootstrap for general stationary observations. Annals of Statistics 1989;17:1217-1241."])]
info = write_docx(OUT / "gas-offtake-forecasting-manuscript.docx", blocks)
(OUT / "abstract.txt").write_text(title + "\n\n" + abstract, encoding="utf-8")
(OUT / "figure_specs.json").write_text(json.dumps([{"file": Path(b[1]).name, "caption": b[2], "spec": b[3]} for b in blocks if b[0] == "fig"], indent=1), encoding="utf-8")
print(info, "abstract words", len(abstract.split()))
print(abstract)
