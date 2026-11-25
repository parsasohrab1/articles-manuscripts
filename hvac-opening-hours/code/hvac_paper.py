"""Figures and Word manuscript for the HVAC study. Usage: python hvac_paper.py <data dir> <results dir> <paper dir>"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mslib import GREY, NAVY, ORANGE, PAL, RED, TEAL, flow, write_docx

DATA, RES, OUT = (Path(a) for a in sys.argv[1:4])
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
R = json.loads((RES / "results.json").read_text(encoding="utf-8"))
TH, NU = R["buildings"]["Town hall"], R["buildings"]["Nursery"]
FT, FN = R["forecast"]["Town hall"], R["forecast"]["Nursery"]
names = ["Town hall", "Nursery"]
specs = []


def save(fig, name):
    p = FIG / name
    fig.savefig(p)
    plt.close(fig)
    return p


f1 = FIG / "fig1_pipeline.png"
flow(f1, [("Measured data", "Smart-switch power and energy, two public buildings, Jun to Dec 2025"),
          ("Cleaning", "0 to 6000 W limit; 5-min grid; 30-min maximum hold; days with at least 80 % coverage"),
          ("Decomposition", "Standby below 20 W; weekend; weekday outside three opening windows"),
          ("Savings bounds", "Standby cut-off; all load off outside the window; weekly block bootstrap"),
          ("Daily baseline", "Month-forward test: weekday mean, last week, ridge, random forest")], "Study pipeline", height=2.6)

raw = {n: pd.read_csv(RES / f"power5min_{n.replace(' ', '_')}.csv.gz", index_col=0, parse_dates=True) for n in names}
daily = pd.read_csv(RES / "daily_kwh.csv", index_col=0, parse_dates=True)
txt = (DATA / "nasa_t2m_fornes_hourly_2025.csv").read_text().split("-END HEADER-\n")[1]
from io import StringIO

w = pd.read_csv(StringIO(txt))
w["d"] = pd.to_datetime(dict(year=w.YEAR, month=w.MO, day=w.DY))
T = w[w.T2M > -90].groupby("d").T2M.mean()["2025-06-01":"2025-12-31"]

M = R["meters"]
fig, ax = plt.subplots(figsize=(10, 3))
used = list(M)
vals = [M[m]["valid_days"] for m in used]
ax.barh(range(len(used)), vals, color=[NAVY if M[m]["building"] == "Town hall" else TEAL for m in used])
ax.set_yticks(range(len(used)), [f"{M[m]['building']} {m[-5:]}" for m in used])
for i, v in enumerate(vals):
    ax.text(v + 2, i, str(v), va="center", fontsize=8)
ax.axvline(100, color=RED, ls="--", lw=1), ax.text(102, len(used) - 0.6, "inclusion limit: 100 days", color=RED, fontsize=8)
ax.set_xlabel("Valid days out of 214"), ax.set_title("Data coverage per meter"), ax.invert_yaxis()
f2 = save(fig, "fig2_coverage.png")

fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
for ax, n, c in zip(axes, names, (NAVY, TEAL)):
    s = daily[n].dropna()
    ax.bar(s.index, s.values, width=0.9, color=c)
    ax.set_ylabel("kWh per day"), ax.set_title(n, loc="left")
axes[2].plot(T.index, T.values, color=ORANGE), axes[2].set_ylabel("°C"), axes[2].set_title("Outdoor temperature (NASA POWER, daily mean)", loc="left")
fig.tight_layout()
f3 = save(fig, "fig3_daily.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for ax, n in zip(axes, names):
    p = pd.read_csv(RES / f"profile_{n.replace(' ', '_')}.csv", index_col=0)
    im = ax.imshow(p.values * 1000, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][:len(p)]), ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Hour of day"), ax.set_title(n), ax.grid(False)
    for h in (7, 15, 18, 22):
        ax.axvline(h - 0.5, color=NAVY, lw=0.7, ls=":")
    fig.colorbar(im, ax=ax, label="mean power, W", shrink=0.9)
f4 = save(fig, "fig4_heatmap.png")

wins = list(TH["windows"])
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for ax, key, title in zip(axes, ("standby_cut", "all_off_outside"), ("Standby cut-off outside the window", "All load off outside the window (upper bound)")):
    for i, (n, b, c) in enumerate(((names[0], TH, NAVY), (names[1], NU, TEAL))):
        v = np.array([b["windows"][k][key + "_pct"] for k in wins])
        ci = np.array([b["windows"][k][key + "_ci"] for k in wins])
        x = np.arange(len(wins)) + (i - 0.5) * 0.38
        ax.bar(x, v, 0.36, color=c, label=n, yerr=np.abs(ci.T - v), capsize=3, error_kw={"lw": 1})
        for xx, vv, hi in zip(x, v, ci[:, 1]):
            ax.text(xx, hi + 1.2, f"{vv}", ha="center", color=c, fontsize=8, fontweight="bold")
    ax.axhline(10, color=RED, ls="--", lw=1), ax.set_xticks(range(len(wins)), wins), ax.set_title(title), ax.set_xlabel("Assumed weekday opening window")
axes[0].set_ylabel("Share of measured electricity, %"), axes[0].legend()
f5 = save(fig, "fig5_bounds.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
bins = np.logspace(-1, np.log10(8000), 60)
for n, c in zip(names, (NAVY, TEAL)):
    v = raw[n].sum(axis=1).clip(lower=0.1)
    axes[0].hist(v, bins=bins, histtype="step", lw=1.6, color=c, label=n, density=False, weights=np.ones(len(v)) / len(v) * 100)
    b = R["buildings"][n]["standby_threshold_sensitivity"]
    axes[1].plot([int(k) for k in b], list(b.values()), "o-", color=c, label=n)
axes[0].set_xscale("log"), axes[0].set_xlabel("Building power on HVAC meters, W"), axes[0].set_ylabel("Share of intervals, %"), axes[0].set_title("Power distribution"), axes[0].legend()
axes[1].set_xlabel("Standby threshold, W"), axes[1].set_ylabel("Standby share of energy, %"), axes[1].set_title("Sensitivity to the threshold"), axes[1].set_ylim(0, None)
fig.tight_layout()
f6 = save(fig, "fig6_standby.png")

models = list(FT["mae_kwh"])
fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), gridspec_kw={"width_ratios": [1, 1.5]})
for i, (n, ff, c) in enumerate(((names[0], FT, NAVY), (names[1], FN, TEAL))):
    axes[0].bar(np.arange(len(models)) + (i - 0.5) * 0.38, [ff["mae_kwh"][m] for m in models], 0.36, color=c, label=n)
axes[0].set_xticks(range(len(models)), models, rotation=25, ha="right"), axes[0].set_ylabel("Mean absolute error, kWh/day"), axes[0].set_title("Test error"), axes[0].legend()
p = pd.read_csv(RES / "forecast_Nursery.csv", index_col=0, parse_dates=True)
axes[1].plot(p.index, p.y, color="black", lw=1.2, label="measured"), axes[1].plot(p.index, p["last week"], color=ORANGE, lw=1, label="last week")
axes[1].plot(p.index, p["random forest"], color=RED, lw=1, label="random forest"), axes[1].set_title("Nursery, test months"), axes[1].set_ylabel("kWh per day"), axes[1].legend(ncol=3, fontsize=8)
fig.autofmt_xdate()
f7 = save(fig, "fig7_forecast.png")

W = lambda b, k, key: b["windows"][k][key + "_pct"]
C = lambda b, k, key: f"{b['windows'][k][key + '_ci'][0]} to {b['windows'][k][key + '_ci'][1]}"
title = "How much HVAC electricity in small public buildings falls outside opening hours? A seven-month measurement study with schedule-based upper bounds"
abstract = (f"Programmes that promise a fixed percentage saving from smarter HVAC scheduling need to know how much electricity is actually used when a building is closed. "
 f"This study analyses open smart-switch measurements of HVAC units in a town hall and a public nursery in Fornes (Spain) from June to December 2025: {TH['days']} and {NU['days']} complete days, {TH['energy_kwh']} and {NU['energy_kwh']} kWh. "
 f"Standby consumption below 20 W was {TH['standby_share_pct']} % and {NU['standby_share_pct']} % of the measured energy, and removing it outside a 07:00 to 15:00 weekday window would save only {W(TH,'07-15','standby_cut')} % and {W(NU,'07-15','standby_cut')} %. "
 f"Switching off all load outside that window gives upper bounds of {W(TH,'07-15','all_off_outside')} % and {W(NU,'07-15','all_off_outside')} %, which fall to {W(TH,'07-22','all_off_outside')} % and {W(NU,'07-22','all_off_outside')} % if the window ends at 22:00. "
 f"The result therefore depends mainly on the true opening hours, which the dataset does not record. For daily consumption, ridge degree-day and random forest models did not beat repeating the value of the previous week "
 f"(test error {FT['mae_kwh']['random forest']} against {FT['mae_kwh']['last week']} kWh/day in the town hall and {FN['mae_kwh']['random forest']} against {FN['mae_kwh']['last week']} in the nursery). "
 "No control action was applied, so no realised saving is claimed.")

rows_m = [[M[m]["building"], m, M[m]["records"], M[m]["valid_days"], M[m]["energy_kwh_valid_days"], M[m]["integrated_over_counter"] if M[m]["integrated_over_counter"] is not None else "not computed", "yes" if m in R["settings"]["meters_used"] else "no"] for m in M]
rows_w = [[n, k, W(b, k, "standby_cut"), C(b, k, "standby_cut"), W(b, k, "all_off_outside"), C(b, k, "all_off_outside")] for n, b in ((names[0], TH), (names[1], NU)) for k in wins]
rows_f = [[m, FT["mae_kwh"][m], FT["bias_kwh"][m], FN["mae_kwh"][m], FN["bias_kwh"][m]] for m in models]

blocks = [("title", title), ("small", "Authors and affiliations: [to be completed]. Draft for author review; not submitted."),
 ("h", "Abstract"), ("p", abstract), ("p", "Keywords: HVAC; public buildings; standby consumption; opening hours; measured data; demand-side management; baseline forecasting"),
 ("h", "1. Introduction"),
 ("p", "Heating, ventilation and air-conditioning (HVAC) control products are often specified with a target saving, for example at least 10 % of HVAC electricity. Whether such a target is reachable by scheduling depends on how much energy is used outside occupied hours. Field work has shown that energy use outside occupancy can be large in commercial buildings [1], but small public buildings with split units are less often documented with open data."),
 ("p", "Low-cost wireless sensor networks now make unit-level measurements available. This paper asks three questions on an open dataset: (i) what share of HVAC electricity is standby consumption; (ii) how large is the upper bound of schedule-based savings, and how sensitive is it to the assumed opening hours; and (iii) can daily consumption be predicted well enough to serve as a baseline for verifying a later intervention."),
 ("h", "2. Data"),
 ("p", "The measurements come from the WeForming Demo 5 wireless sensor network of the town hall (which also houses the municipal theatre) [2] and the public nursery of Fornes [3], published under CC-BY-4.0. Smart switches report active power, current and cumulative energy of HVAC units. According to the dataset description, metering was reported about once per day until late May 2025 and about every five minutes from June, so the analysis starts on 1 June 2025. Outdoor air temperature at the village coordinates was taken from NASA POWER [4]."),
 ("table", "Meters, coverage and consistency between integrated power and the energy counter", ["Building", "Meter", "Records", "Valid days", "kWh", "Integrated / counter", "Used"], rows_m),
 ("fig", f2, "Valid days per meter. One nursery meter has no day above the coverage limit and is excluded.", "horizontal bar chart; one bar per meter; red dashed line at the 100-day inclusion limit; 300 dpi PNG."),
 ("h", "3. Methods"),
 ("fig", f1, "Study pipeline from raw measurements to savings bounds and baseline models.", "five-step left-to-right flow diagram; graphical abstract; 300 dpi PNG."),
 ("p", "A plausibility limit of 0 to 6000 W was applied to power readings; none fell outside it. Readings were averaged on a five-minute grid and held for at most 30 minutes. A day was valid for a meter when at least 80 % of its intervals had a value, and a building day was analysed when all included meters reported in at least 80 % of intervals. Integrated power agreed with the cumulative energy counter within the ratios listed in Table 1."),
 ("p", "An interval of one meter was classed as standby when its power was below 20 W; thresholds of 5, 10 and 50 W were also tested. Because opening hours are not part of the dataset, three weekday windows were assumed: 07:00 to 15:00, 07:00 to 18:00 and 07:00 to 22:00; weekends were treated as closed. Two schedule rules were replayed on the measured series. The first removes standby energy outside the window. The second removes all energy outside the window; it is an upper bound, since it ignores pre-conditioning, frost protection, events in the theatre and rebound after restart. Intervals of 95 % for each share were obtained by resampling seven-day blocks 2000 times [5]."),
 ("p", "Daily building energy was predicted for September to December where enough earlier data existed, training on all days before each test month. Four predictors were compared: the training mean for the weekday, the value seven days earlier, ridge regression [6] on temperature, heating and cooling degree terms and weekday indicators, and a random forest [7] with the same inputs and the seven-day lag, implemented in scikit-learn [8]. Observed same-day temperature was used, which favours the weather-based models compared with operational forecasting."),
 ("h", "4. Results"),
 ("h2", "4.1 Consumption pattern"),
 ("p", f"The included meters used on average {TH['mean_kwh_per_day']} kWh per day in the town hall and {NU['mean_kwh_per_day']} kWh per day in the nursery. These are small loads, and consumption is intermittent: the median uninterrupted run above 100 W lasted {TH['runs']['median_h']} h and {NU['runs']['median_h']} h. Only {TH['runs']['over_12h_count']} and {NU['runs']['over_12h_count']} runs exceeded 12 hours."),
 ("fig", f3, "Daily HVAC electricity of the two buildings and daily mean outdoor temperature.", "three stacked panels with shared date axis; bars for energy, line for temperature; 300 dpi PNG."),
 ("fig", f4, "Mean HVAC power by weekday and hour. Dotted lines mark the assumed window limits.", "two heat maps, 7 rows by 24 columns, sequential yellow-orange-red scale in watts; 300 dpi PNG."),
 ("h2", "4.2 Standby and schedule-based bounds"),
 ("p", f"Standby accounted for {TH['standby_share_pct']} % (95 % interval {TH['standby_share_ci'][0]} to {TH['standby_share_ci'][1]}) of town-hall energy and {NU['standby_share_pct']} % ({NU['standby_share_ci'][0]} to {NU['standby_share_ci'][1]}) of nursery energy. Weekends accounted for {TH['weekend_share_pct']} % and {NU['weekend_share_pct']} %, and active load between midnight and 06:00 for {TH['night_00_06_active_share_pct']} % and {NU['night_00_06_active_share_pct']} %."),
 ("table", "Savings bounds in percent of measured energy, with 95 % block-bootstrap intervals", ["Building", "Window", "Standby cut", "Interval", "All off outside", "Interval"], rows_w),
 ("fig", f5, "Savings bounds by assumed opening window. The red dashed line marks a 10 % target.", "paired bar charts with asymmetric 95 % error bars; shared percentage axis; 300 dpi PNG."),
 ("fig", f6, "Distribution of HVAC power and sensitivity of the standby share to the threshold.", "log-scale step histogram and line chart for 5, 10, 20 and 50 W thresholds; 300 dpi PNG."),
 ("h2", "4.3 Daily baseline"),
 ("p", f"The test sets contain {FT['test_days']} town-hall days and {FN['test_days']} nursery days. Repeating the value of the previous week gave the lowest error in both buildings (Table 3). The errors are of the same order as the mean daily energy ({FT['test_mean_kwh']} and {FN['test_mean_kwh']} kWh), so none of these baselines could verify a saving of about 10 % on single days."),
 ("table", "Test error and bias of daily predictions, kWh per day", ["Model", "Town hall error", "Town hall bias", "Nursery error", "Nursery bias"], rows_f),
 ("fig", f7, "Test error of the four predictors and the nursery test series.", "grouped bar chart of mean absolute error beside a time series of measured and predicted daily energy; 300 dpi PNG."),
 ("h", "5. Discussion"),
 ("p", "Cutting standby power is the least disputable measure but is worth only a few percent here. A target of 10 % or more rests on removing active load outside opening hours. Under a short window the upper bound is well above 10 % in both buildings; under a window ending at 22:00 the point estimates are close to or above 10 % but the intervals extend below it. The weekly profiles (Figure 4) show regular weekday load until about 18:00 in the town hall and on some evenings in the nursery, so the shortest window probably counts occupied use as savable and its bound is the least credible. A product claim should therefore be tied to a recorded opening schedule, and the evening load should first be checked against actual use, because the town hall also serves as a theatre."),
 ("p", "For measurement and verification, daily prediction from weather and calendar was not informative. A later trial would need occupancy or booking records, a longer baseline covering a full year, or an on/off experimental design with alternating weeks."),
 ("h", "6. Limitations"),
 ("ul", ["Two buildings in one village and seven months without a full heating season; the results do not generalise to other climates or building types.",
         "Only the metered HVAC units are covered; one nursery meter was excluded for lack of coverage.",
         "Opening hours are assumed. The town-hall description does not state clearly whether timestamps are in local time or UTC; the nursery description states local time. A shift of one or two hours would change the window results.",
         "Power is held for up to 30 minutes between reports and invalid days are dropped; energy totals are not annual totals.",
         "Savings are replayed bounds on past data. Realised savings, comfort effects and rebound are [to be measured] in an intervention trial."]),
 ("h", "7. Conclusions"),
 ("p", f"In two small public buildings, standby consumption was a few percent of HVAC electricity. Schedule-based upper bounds ranged from {min(W(b,k,'all_off_outside') for b in (TH,NU) for k in wins)} % to {max(W(b,k,'all_off_outside') for b in (TH,NU) for k in wins)} % depending mainly on the assumed opening hours. Reliable claims require recorded schedules and a controlled trial."),
 ("h", "Data and code availability"),
 ("p", "Measurements: Zenodo records 10.5281/zenodo.23099351 and 10.5281/zenodo.23099489 (CC-BY-4.0). Temperature: NASA POWER. Analysis code and result files accompany this manuscript; repository visibility is [to be confirmed before submission]."),
 ("h", "Declarations"),
 ("p", "Funding, competing interests and author contributions: [to be completed]. Use of AI tools in analysis and drafting: [to be declared according to the journal policy]."),
 ("h", "References"),
 ("ref", ["Masoso OT, Grobler LJ. The dark side of occupants' behaviour on building energy use. Energy and Buildings 2010;42:173-177.",
          "Cuerva Energia. Wireless sensor network monitoring data (indoor environment and HVAC electricity use) from the Town Hall of Fornes (Granada, Spain), WeForming Demo 5. Zenodo, 2026. https://doi.org/10.5281/zenodo.23099351",
          "Cuerva Energia. Wireless sensor network monitoring data (indoor environment and HVAC electricity use) from the Public Nursery of Fornes (Granada, Spain), WeForming Demo 5. Zenodo, 2026. https://doi.org/10.5281/zenodo.23099489",
          "NASA Langley Research Center. Prediction Of Worldwide Energy Resources (POWER) project. https://power.larc.nasa.gov",
          "Kunsch HR. The jackknife and the bootstrap for general stationary observations. Annals of Statistics 1989;17:1217-1241.",
          "Hoerl AE, Kennard RW. Ridge regression: biased estimation for nonorthogonal problems. Technometrics 1970;12:55-67.",
          "Breiman L. Random forests. Machine Learning 2001;45:5-32.",
          "Pedregosa F, et al. Scikit-learn: machine learning in Python. Journal of Machine Learning Research 2011;12:2825-2830."])]
info = write_docx(OUT / "hvac-opening-hours-manuscript.docx", blocks)
(OUT / "abstract.txt").write_text(title + "\n\n" + abstract, encoding="utf-8")
(OUT / "figure_specs.json").write_text(json.dumps([{"file": Path(b[1]).name, "caption": b[2], "spec": b[3]} for b in blocks if b[0] == "fig"], indent=1), encoding="utf-8")
print(info, "abstract words", len(abstract.split()))
print(abstract)
