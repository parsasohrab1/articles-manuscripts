"""HVAC electricity outside opening hours in two public buildings (Fornes, Spain), measured data 2025-06-01..2025-12-31.

Usage: python hvac_study.py <data dir of the HVAC download> <output dir>
"""
import json
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV

DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
START, END = pd.Timestamp("2025-06-01"), pd.Timestamp("2026-01-01")
B = {"Town hall": ("23099351", ["ESP01000SS013", "ESP01002SS015", "ESP01003SS017"]),
     "Nursery": ("23099489", ["ESP03000SS010", "ESP03001SS009", "ESP03002SS008"])}
STANDBY_W, PMAX_W, HOLD = 20.0, 6000.0, 6  # standby threshold, plausibility limit, forward-fill limit (x 5 min)
WINDOWS = {"07-15": (7, 15), "07-18": (7, 18), "07-22": (7, 22)}
grid = pd.date_range(START, END, freq="5min", inclusive="left")
rng = np.random.default_rng(0)


def boot(num, den, block=7, n=2000):
    """95 % interval of a ratio of sums, resampling 7-day blocks of daily values (result in percent)."""
    v = np.c_[num, den]
    blocks = [v[i:i + block] for i in range(0, len(v), block)]
    out = []
    for _ in range(n):
        pick = rng.integers(0, len(blocks), len(blocks))
        x = np.vstack([blocks[i] for i in pick])
        out.append(x[:, 0].sum() / x[:, 1].sum())
    return [round(float(q) * 100, 1) for q in np.percentile(out, [2.5, 97.5])]


res = {"settings": {"period": ["2025-06-01", "2025-12-31"], "standby_threshold_w": STANDBY_W, "max_plausible_w": PMAX_W,
                    "max_hold_minutes": HOLD * 5, "min_day_coverage": 0.8}, "meters": {}, "buildings": {}}
power = {}
for b, (rec, meters) in B.items():
    for m in meters:
        d = pd.read_csv(DATA / rec / f"{m}_Electric_meter_watts.csv", parse_dates=["time"]).sort_values("time")
        n = len(d)
        bad = int(((d.value < 0) | (d.value > PMAX_W)).sum())
        d = d[(d.value >= 0) & (d.value <= PMAX_W)]
        s = d.set_index("time").value.resample("5min").mean().reindex(grid).ffill(limit=HOLD)
        cov = s.notna().resample("D").mean()
        good = cov[cov >= 0.8].index
        s = s.where(s.index.normalize().isin(good))
        k = pd.read_csv(DATA / rec / f"{m}_Electric_meter_kWh.csv", parse_dates=["time"]).sort_values("time").set_index("time").value
        daily_int = s.resample("D").sum(min_count=1) / 12 / 1000
        counter = k.resample("D").last().diff()
        both = pd.concat([daily_int, counter], axis=1, keys=["int", "ctr"]).dropna()
        both = both[(both.ctr >= 0) & (both.ctr < 150)]
        res["meters"][m] = {"building": b, "records": n, "implausible_removed": bad, "valid_days": int(len(good)),
                            "energy_kwh_valid_days": round(float(daily_int.sum()), 1), "counter_check_days": int(len(both)),
                            "integrated_over_counter": round(float(both["int"].sum() / both.ctr.sum()), 3) if both.ctr.sum() > 0 else None}
        power[m] = s
P = pd.DataFrame(power)
usable = [m for m in P if res["meters"][m]["valid_days"] >= 100]
res["settings"]["meters_used"] = usable

daily_all = {}
for b, (rec, meters) in B.items():
    ms = [m for m in meters if m in usable]
    if not ms:
        continue
    X = P[ms].dropna()  # intervals where every used meter reports
    full = X.groupby(X.index.normalize()).size()
    X = X[X.index.normalize().isin(full[full >= 0.8 * 288].index)]
    day, hour, wkend = X.index.normalize(), X.index.hour, X.index.dayofweek >= 5
    tot = X.sum(axis=1) / 12000  # kWh per 5-minute interval, all meters
    standby = X.where(X < STANDBY_W, 0).sum(axis=1) / 12000
    active = tot - standby

    def share(x):
        return round(float(x.sum() / tot.sum() * 100), 1)

    def ci(x):
        return boot(x.groupby(day).sum().values, tot.groupby(day).sum().values)

    r = {"meters": ms, "days": int(day.nunique()), "energy_kwh": round(float(tot.sum()), 1),
         "mean_kwh_per_day": round(float(tot.sum() / day.nunique()), 2),
         "standby_share_pct": share(standby), "standby_share_ci": ci(standby),
         "weekend_share_pct": share(tot.where(wkend, 0)), "night_00_06_active_share_pct": share(active.where(hour < 6, 0)),
         "windows": {}}
    for name, (a, z) in WINDOWS.items():
        outside = wkend | (hour < a) | (hour >= z)
        idle, off = standby.where(outside, 0), tot.where(outside, 0)
        r["windows"][name] = {"standby_cut_pct": share(idle), "standby_cut_ci": ci(idle),
                              "all_off_outside_pct": share(off), "all_off_outside_ci": ci(off),
                              "weekday_outside_pct": share(tot.where((~wkend) & ((hour < a) | (hour >= z)), 0)),
                              "weekend_pct": share(tot.where(wkend, 0))}
    # sensitivity of the standby share to the threshold
    r["standby_threshold_sensitivity"] = {str(t): share(X.where(X < t, 0).sum(axis=1) / 12000) for t in (5, 10, 20, 50)}
    runs = []
    for m in ms:
        on = (P[m] >= 100).astype(int)
        gid = (on != on.shift()).cumsum()
        for _, g in on[on == 1].groupby(gid[on == 1]):
            runs.append((m, str(g.index[0]), len(g) / 12, float(P[m].loc[g.index].sum() / 12000)))
    long = [x for x in runs if x[2] >= 12]
    r["runs"] = {"count": len(runs), "median_h": round(float(np.median([x[2] for x in runs])), 2), "over_12h_count": len(long),
                 "over_12h_energy_share_pct": round(float(sum(x[3] for x in long) / sum(x[3] for x in runs) * 100), 1),
                 "longest_h": round(max(x[2] for x in runs), 1)}
    res["buildings"][b] = r
    daily_all[b] = tot.groupby(day).sum()
    tag = b.replace(" ", "_")
    pd.DataFrame({"kw": X.sum(axis=1) / 1000, "hour": hour, "dow": X.index.dayofweek}).groupby(["dow", "hour"]).kw.mean().unstack().to_csv(OUT / f"profile_{tag}.csv")
    X.to_csv(OUT / f"power5min_{tag}.csv.gz")
    pd.DataFrame(runs, columns=["meter", "start", "hours", "kwh"]).to_csv(OUT / f"runs_{tag}.csv", index=False)

# daily forecast, forward-chaining by month (train on all earlier days)
txt = (DATA / "nasa_t2m_fornes_hourly_2025.csv").read_text().split("-END HEADER-\n")[1]
w = pd.read_csv(StringIO(txt))
w["d"] = pd.to_datetime(dict(year=w.YEAR, month=w.MO, day=w.DY))
T = w[w.T2M > -90].groupby("d").T2M.mean()
res["forecast"] = {}
for b, y in daily_all.items():
    f = pd.DataFrame({"y": y})
    f["t"] = T.reindex(f.index)
    f["hdd"], f["cdd"] = (15 - f.t).clip(lower=0), (f.t - 21).clip(lower=0)
    f["we"] = (f.index.dayofweek >= 5).astype(int)
    for k in range(5):
        f[f"dw{k}"] = (f.index.dayofweek == k).astype(int)
    f["lag7"] = f.y.reindex(f.index - pd.Timedelta(days=7)).values
    f = f.dropna()
    cols = ["t", "hdd", "cdd", "we"] + [f"dw{k}" for k in range(5)]
    preds = []
    for mo in range(9, 13):
        tr, te = f[f.index < pd.Timestamp(2025, mo, 1)], f[f.index.month == mo]
        if len(tr) < 30 or len(te) < 5:
            continue
        p = pd.DataFrame({"y": te.y, "t": te.t})
        p["weekday mean"] = te.index.dayofweek.map(tr.groupby(tr.index.dayofweek).y.mean())
        p["last week"] = te.lag7
        p["ridge degree-day"] = np.clip(RidgeCV(alphas=np.logspace(-2, 3, 20)).fit(tr[cols], tr.y).predict(te[cols]), 0, None)
        p["random forest"] = RandomForestRegressor(n_estimators=400, min_samples_leaf=3, random_state=0).fit(tr[cols + ["lag7"]], tr.y).predict(te[cols + ["lag7"]])
        preds.append(p)
    if not preds:
        continue
    p = pd.concat(preds)
    p.to_csv(OUT / f"forecast_{b.replace(' ', '_')}.csv")
    res["forecast"][b] = {"test_days": int(len(p)), "test_mean_kwh": round(float(p.y.mean()), 2),
                          "mae_kwh": {m: round(float((p[m] - p.y).abs().mean()), 2) for m in p.columns[2:]},
                          "bias_kwh": {m: round(float((p[m] - p.y).mean()), 2) for m in p.columns[2:]}}
pd.DataFrame(daily_all).to_csv(OUT / "daily_kwh.csv")
(OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1))
