"""Pre-planned analysis of results/splits.csv.

    python -I code/analyse.py results

Unit of analysis: the mean over the 50 test folds of one dataset, model and method.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, spearmanr, wilcoxon

MODELS = ["LR", "NB", "KNN", "RF", "GB"]
METHODS = ["none", "platt", "isotonic", "beta", "temperature"]
CAL = METHODS[1:]


def holm(p):
    order = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (len(p) - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def boot_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


if __name__ == "__main__":
    res = Path(sys.argv[1])
    s = pd.read_csv(res / "splits.csv")
    info = pd.read_csv(res / "datasets.csv").set_index("dataset")
    assert len(s) == len(info) * 50 * len(MODELS) * len(METHODS) and not s[["brier", "logloss", "ece", "auc"]].isna().any().any()
    m = s.groupby(["dataset", "model", "method"])[["brier", "logloss", "ece", "auc"]].mean()
    m.to_csv(res / "means.csv")
    out = {"summary": [], "friedman": {}, "comparisons": [], "spearman": {}}
    for metric in ["brier", "logloss", "ece", "auc"]:
        for mn in MODELS:
            t = m[metric].xs(mn, level="model").unstack()[METHODS]
            ranks = np.vstack([rankdata(-r if metric == "auc" else r) for r in t.to_numpy()]).mean(0)
            for j, meth in enumerate(METHODS):
                out["summary"].append(dict(metric=metric, model=mn, method=meth, mean=float(t[meth].mean()), rank=float(ranks[j])))
            if metric in ("brier", "logloss"):
                st, p = friedmanchisquare(*[t[c] for c in METHODS])
                out["friedman"][f"{metric}|{mn}"] = [float(st), float(p)]
    for metric in ["brier", "logloss"]:
        fam = []
        for mn in MODELS:
            t = m[metric].xs(mn, level="model").unstack()[METHODS]
            for meth in CAL:
                d = (t[meth] - t["none"]).to_numpy()
                fam.append(dict(metric=metric, model=mn, method=meth, mean_diff=float(d.mean()), median_diff=float(np.median(d)),
                                ci=boot_ci(d), better=int((d < 0).sum()), worse=int((d > 0).sum()), p=float(wilcoxon(d).pvalue)))
        for row, a in zip(fam, holm(np.array([r["p"] for r in fam]))):
            row["p_holm"] = float(a)
        out["comparisons"] += fam
    for metric in ["brier", "logloss"]:
        for mn in MODELS:
            t = m[metric].xs(mn, level="model").unstack()
            d = (t["isotonic"] - t["platt"]).reindex(info.index)
            r = spearmanr(info.events, d)
            out["spearman"][f"{metric}|{mn}"] = [float(r.statistic), float(r.pvalue), int((d < 0).sum())]
    (res / "analysis.json").write_text(json.dumps(out, indent=1))
    q = pd.DataFrame(out["summary"])
    for metric in ["brier", "logloss", "ece", "auc"]:
        x = q[q.metric == metric]
        print("\n", metric, "mean\n", x.pivot(index="model", columns="method", values="mean").loc[MODELS, METHODS].round(4))
        print("rank\n", x.pivot(index="model", columns="method", values="rank").loc[MODELS, METHODS].round(2))
    print({k: [round(v[0], 2), float(f"{v[1]:.2g}")] for k, v in out["friedman"].items()})
    c = pd.DataFrame(out["comparisons"])
    c["ci"] = c.ci.map(lambda v: [round(a, 4) for a in v])
    print(c.round(4).to_string())
    print({k: [round(v[0], 2), round(v[1], 3), v[2]] for k, v in out["spearman"].items()})
