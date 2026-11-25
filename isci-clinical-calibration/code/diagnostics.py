"""Why can a monotone calibrator change AUC? Counts clipped base probabilities and negative Platt slopes (same splits as study.py).

    python -I code/diagnostics.py data results
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).parent))
import study

MODELS = ["LR", "NB", "KNN"]  # the cheap models; tree ensembles are not needed to explain the AUC change

if __name__ == "__main__":
    sets = study.load(Path(sys.argv[1]))
    out = {m: dict(clipped=0, total=0, negative_slope=0, folds=0) for m in MODELS}
    for name, (X, y, _) in sets.items():
        for rep in range(study.REPEATS):
            for fold, (tr, te) in enumerate(StratifiedKFold(study.FOLDS, shuffle=True, random_state=rep).split(X, y)):
                seed = 1000 * rep + fold
                inner = StratifiedKFold(5, shuffle=True, random_state=seed)
                for mn in MODELS:
                    est = study.model(mn, seed)
                    p_cal = cross_val_predict(est, X[tr], y[tr], cv=inner, method="predict_proba")[:, 1]
                    p = est.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
                    o = out[mn]
                    o["clipped"] += int(((p < study.EPS) | (p > 1 - study.EPS)).sum())
                    o["total"] += len(p)
                    o["folds"] += 1
                    o["negative_slope"] += int(study.smoothed_logistic(study.logit(p_cal)[:, None], y[tr]).coef_[0, 0] < 0)
        print(name, flush=True)
    (Path(sys.argv[2]) / "diagnostics.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))
