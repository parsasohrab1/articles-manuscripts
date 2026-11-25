"""Day-ahead forecasting of gas distribution offtake in three European zones from ENTSOG physical flows.

Usage: python gas_study.py <data dir of the gas download> <output dir>
"""
import json
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV

DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
ZONES = {"NL_L": "Netherlands, local distribution, L-gas", "BE_H": "Belgium, distribution, H-gas", "IT": "Italy, distribution"}
TEST_START = pd.Timestamp("2024-01-01")
rng = np.random.default_rng(0)


def load(z):
    rows = []
    for f in sorted(DATA.glob(f"entsog_{z}_*.json")):
        rows += json.loads(f.read_text(encoding="utf-8"))["operationalData"]
    d = pd.DataFrame(rows)
    assert set(d.unit) == {"kWh/d"} and set(d.indicator) == {"Physical Flow"} and set(d.directionKey) == {"exit"}
    d["day"] = pd.to_datetime(d.periodFrom.str[:10])
    d["value"] = pd.to_numeric(d.value, errors="coerce")
    d["upd"] = pd.to_datetime(d.lastUpdateDateTime, errors="coerce", utc=True)
    dup = int(d.duplicated("day").sum())
    d = d.sort_values("upd").drop_duplicates("day", keep="last").set_index("day").sort_index()
    full = pd.date_range(d.index.min(), "2025-12-31")
    s = d.value.reindex(full) / 1e6  # GWh per day
    nonpos = int((s <= 0).sum())
    s = s.where(s > 0)
    med = s.rolling(15, center=True, min_periods=5).median()
    odd = (s < 0.2 * med) | (s > 3 * med)
    s = s.where(~odd)
    info = {"label": ZONES[z], "first_day": str(full[0].date()), "calendar_days": int(len(full)), "valid_days": int(s.notna().sum()),
            "duplicate_records": dup, "non_positive": nonpos, "outliers_removed": int(odd.sum()),
            "provisional_share_pct": round(float((d.flowStatus == "Provisional").mean() * 100), 1),
            "mean_gwh": round(float(s.mean()), 1), "min_gwh": round(float(s.min()), 1), "max_gwh": round(float(s.max()), 1)}
    return s, info


def temperature(z):
    txt = (DATA / f"nasa_t2m_{z}_daily.csv").read_text().split("-END HEADER-\n")[1]
    w = pd.read_csv(StringIO(txt))
    key = ["YEAR", "DOY"] if "DOY" in w else ["YEAR", "MO", "DY"]
    idx = pd.to_datetime(w.YEAR.astype(str) + w.DOY.astype(str).str.zfill(3), format="%Y%j") if "DOY" in w else pd.to_datetime(dict(year=w.YEAR, month=w.MO, day=w.DY))
    t = pd.Series(w.T2M.values, index=idx)
    return t.where(t > -90)


def frame(s, t):
    f = pd.DataFrame({"y": s})
    f["lag1"], f["lag2"], f["lag7"] = s.shift(1), s.shift(2), s.shift(7)
    f["mean7"] = s.shift(1).rolling(7, min_periods=5).mean()
    t = t.reindex(s.index)
    f["t"], f["t1"], f["t3"] = t, t.shift(1), t.rolling(3).mean()
    f["hdd"], f["dt"] = (16 - t).clip(lower=0), t - t.shift(1)
    doy = s.index.dayofyear
    f["sin"], f["cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    for k in range(7):
        f[f"dw{k}"] = (s.index.dayofweek == k).astype(int)
    return f


CAL = ["sin", "cos"] + [f"dw{k}" for k in range(7)]
LAGS = ["lag1", "lag2", "lag7", "mean7"]
FEATURES = {"full": LAGS + CAL + ["t", "t1", "t3", "hdd", "dt"], "past temperature only": LAGS + CAL + ["t1"], "no temperature": LAGS + CAL}


def models():
    return {"ridge": RidgeCV(alphas=np.logspace(-3, 3, 25)),
            "random forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=3, max_features=0.6, random_state=0, n_jobs=-1),
            "gradient boosting": HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, min_samples_leaf=15, random_state=0)}


def forward(f, cols, name, change=True):
    """Monthly refit on all earlier days. The learned target is the change from the previous day."""
    out = []
    for start in pd.date_range(TEST_START, "2025-12-01", freq="MS"):
        d = f.dropna(subset=cols + ["y", "lag1"])
        tr, te = d[d.index < start], d[(d.index >= start) & (d.index < start + pd.offsets.MonthBegin(1))]
        if len(te) == 0:
            continue
        m = models()[name]
        base_tr, base_te = (tr.lag1, te.lag1) if change else (0, 0)
        if name == "ridge":
            mu, sd = tr[cols].mean(), tr[cols].std().replace(0, 1)
            m.fit((tr[cols] - mu) / sd, tr.y - base_tr)
            pred = m.predict((te[cols] - mu) / sd)
        else:
            m.fit(tr[cols], tr.y - base_tr)
            pred = m.predict(te[cols])
        out.append(pd.Series(pred + base_te, index=te.index))
    return pd.concat(out)


def boot_diff(a, b, block=7, n=3000):
    """95 % interval of mean(|a|) - mean(|b|) using 7-day blocks."""
    d = (np.abs(a) - np.abs(b)).values
    blocks = [d[i:i + block] for i in range(0, len(d), block)]
    out = [np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]).mean() for _ in range(n)]
    return [round(float(q), 3) for q in np.percentile(out, [2.5, 97.5])]


res = {"test_period": ["2024-01-01", "2025-12-31"], "zones": {}}
for z in ZONES:
    s, info = load(z)
    t = temperature(z)
    f = frame(s, t)
    preds = pd.DataFrame({"y": f.y, "previous day": f.lag1, "previous week": f.lag7})
    for name in models():
        preds[name] = forward(f, FEATURES["full"], name)
    preds = preds.loc[TEST_START:].dropna()
    err = preds.drop(columns="y").sub(preds.y, axis=0)
    base = float(err["previous day"].abs().mean())
    metrics = {m: {"mae_gwh": round(float(err[m].abs().mean()), 2), "rmse_gwh": round(float(np.sqrt((err[m] ** 2).mean())), 2),
                   "mape_pct": round(float((err[m].abs() / preds.y).mean() * 100), 2), "bias_gwh": round(float(err[m].mean()), 2),
                   "skill_vs_previous_day_pct": round(float((1 - err[m].abs().mean() / base) * 100), 1),
                   "mae_minus_previous_day_ci": boot_diff(err[m], err["previous day"])} for m in err}
    best = min(models(), key=lambda m: metrics[m]["mae_gwh"])
    ablation = {}
    for label, cols in FEATURES.items():
        p = forward(f, cols, best).reindex(preds.index)
        ablation[label] = round(float((p - preds.y).abs().mean()), 2)
    # one-day later publication: only data up to d-2 known
    g = f.copy()
    g["lag1"], g["mean7"] = s.shift(2), s.shift(2).rolling(7, min_periods=5).mean()
    late = forward(g, [c for c in FEATURES["full"] if c != "lag2"], best).reindex(preds.index)
    ablation["demand known only to two days before"] = round(float((late - preds.y).abs().mean()), 2)
    # balancing margin: 95th percentile of the previous 90 absolute errors, applied to the next day
    margin = {}
    for m in ("previous day", best):
        a = err[m].abs()
        q = a.rolling(90, min_periods=90).quantile(0.95).shift(1)
        ok = q.notna()
        margin[m] = {"mean_margin_gwh": round(float(q[ok].mean()), 1), "coverage_pct": round(float((a[ok] <= q[ok]).mean() * 100), 1),
                     "margin_share_of_mean_demand_pct": round(float(q[ok].mean() / preds.y[ok].mean() * 100), 1), "days": int(ok.sum())}
    monthly = err.abs().groupby(preds.index.month).mean()
    res["zones"][z] = {"data": info, "test_days": int(len(preds)), "test_mean_gwh": round(float(preds.y.mean()), 1), "best_model": best,
                       "metrics": metrics, "ablation_mae_gwh": ablation, "margin": margin,
                       "temperature_correlation": round(float(f.y.corr(f.t)), 3)}
    preds.join(t.rename("temp")).to_csv(OUT / f"predictions_{z}.csv")
    f[["y", "t"]].to_csv(OUT / f"series_{z}.csv")
    monthly.to_csv(OUT / f"monthly_{z}.csv")
    print(z, best, {m: v["mae_gwh"] for m, v in metrics.items()}, flush=True)
(OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1))
