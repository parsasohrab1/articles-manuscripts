"""Post-hoc calibration on small clinical tabular data: repeated stratified cross-validation on 15 UCI datasets.

    python -I code/study.py data results
"""
import io
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
EPS = 1e-6
REPEATS, FOLDS = 10, 5
MODELS = ["LR", "NB", "KNN", "RF", "GB"]
METHODS = ["none", "platt", "isotonic", "beta", "temperature"]


def read(zf, name, **kw):
    return pd.read_csv(io.BytesIO(zf.read(name)), **kw)


def arff(zf, name):
    lines = zf.read(name).decode("latin-1").splitlines()
    cols = [l.split()[1] for l in lines if l.lower().startswith("@attribute")]
    start = next(i for i, l in enumerate(lines) if l.lower().startswith("@data")) + 1
    body = "\n".join(l for l in lines[start:] if l.strip() and not l.startswith("%"))
    return pd.read_csv(io.StringIO(body), header=None, names=cols)


def load(data):
    """Return {name: (X, y, info)}; y = 1 is the event named in info."""
    def z(i):
        return zipfile.ZipFile(Path(data) / f"{i}.zip")
    out = {}

    def add(key, uci, X, y, event):
        df = pd.get_dummies(X, drop_first=True).apply(pd.to_numeric, errors="coerce").astype(float)
        df.columns = [str(c) for c in df.columns]
        df["__y"] = np.asarray(y).astype(int)
        n0 = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        yy = df.pop("__y").to_numpy()
        out[key] = (df.to_numpy(), yy, dict(uci=uci, n_raw=n0, n=len(yy), features=df.shape[1], events=int(yy.sum()),
                                            missing=int(df.isna().sum().sum()), event=event))

    d = read(z(17), "wdbc.data", header=None)
    add("WDBC", 17, d.iloc[:, 2:], d[1] == "M", "malignant tumour")
    d = read(z(45), "processed.cleveland.data", header=None, na_values="?")
    add("Cleveland", 45, d.iloc[:, :13], d[13] > 0, "coronary disease")
    d = read(z(519), "heart_failure_clinical_records_dataset.csv")
    add("HeartFailure", 519, d.drop(columns=["DEATH_EVENT", "time"]), d.DEATH_EVENT == 1, "death during follow-up")
    d = read(z(225), "Indian Liver Patient Dataset (ILPD).csv", header=None)
    add("ILPD", 225, d.iloc[:, :10], d[10] == 1, "liver patient")
    d = read(z(43), "haberman.data", header=None)
    add("Haberman", 43, d.iloc[:, :3], d[3] == 2, "death within five years")
    d = read(z(176), "transfusion.data")
    add("Transfusion", 176, d.iloc[:, :4], d.iloc[:, 4] == 1, "donated in March 2007")
    d = read(z(161), "mammographic_masses.data", header=None, na_values="?")
    add("Mammographic", 161, d.iloc[:, 1:5], d[5] == 1, "malignant mass")
    d = read(z(212), "column_2C.dat", header=None, sep=r"\s+")
    add("Vertebral", 212, d.iloc[:, :6], d[6] == "AB", "abnormal spine")
    d = arff(z(329), "messidor_features.arff")
    add("Retinopathy", 329, d.iloc[:, :-1], d.iloc[:, -1] == 1, "signs of retinopathy")
    d = pd.read_excel(io.BytesIO(z(193).read("CTG.xls")), sheet_name="Raw Data").dropna(subset=["NSP", "LB"])
    feats = ["LB", "AC", "FM", "UC", "DL", "DS", "DP", "ASTV", "MSTV", "ALTV", "MLTV", "Width", "Min", "Max", "Nmax",
             "Nzeros", "Mode", "Mean", "Median", "Variance", "Tendency"]
    add("CTG", 193, d[feats], d.NSP > 1, "suspect or pathological trace")
    d = read(z(529), "diabetes_data_upload.csv")
    add("EarlyDiabetes", 529, d.iloc[:, :-1], d["class"] == "Positive", "diabetes")
    d = pd.concat([read(z(96), "SPECTF.train", header=None), read(z(96), "SPECTF.test", header=None)])
    add("SPECTF", 96, d.iloc[:, 1:], d[0] == 1, "class 1 of the source file")
    d = read(z(15), "breast-cancer-wisconsin.data", header=None, na_values="?")
    add("WBC-Original", 15, d.iloc[:, 1:10], d[10] == 4, "malignant")
    d = arff(z(277), "ThoraricSurgery.arff")
    add("Thoracic", 277, d.iloc[:, :-1].replace({"T": 1, "F": 0}), d.iloc[:, -1] == "T", "death within one year")
    d = read(z(46), "hepatitis.data", header=None, na_values="?")
    add("Hepatitis", 46, d.iloc[:, 1:], d[0] == 1, "death")
    return out


def model(name, seed):
    clf = {"LR": lambda: LogisticRegression(max_iter=5000), "NB": GaussianNB,
           "KNN": lambda: KNeighborsClassifier(15), "RF": lambda: RandomForestClassifier(200, random_state=seed),
           "GB": lambda: GradientBoostingClassifier(random_state=seed)}[name]()
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), clf)


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def smoothed_logistic(F, y):
    """Almost unpenalised logistic fit with Platt's smoothed targets, implemented with sample weights."""
    n1 = y.sum()
    n0 = len(y) - n1
    t = np.where(y == 1, (n1 + 1) / (n1 + 2), 1 / (n0 + 2))
    m = LogisticRegression(C=1e8, max_iter=5000)
    m.fit(np.vstack([F, F]), np.r_[np.ones(len(y)), np.zeros(len(y))], sample_weight=np.r_[t, 1 - t])
    return m


def beta_features(q):
    q = np.clip(q, EPS, 1 - EPS)
    return np.c_[np.log(q), -np.log(1 - q)]


def calibrate(method, p_cal, y, p):
    if method == "none":
        return p
    if method == "platt":
        m = smoothed_logistic(logit(p_cal)[:, None], y)
        return m.predict_proba(logit(p)[:, None])[:, 1]
    if method == "isotonic":
        return IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(p_cal, y).predict(p)
    if method == "beta":
        keep = [0, 1]
        while True:
            m = smoothed_logistic(beta_features(p_cal)[:, keep], y)
            if (m.coef_[0] >= 0).all() or len(keep) == 1:
                break
            keep = [keep[int(np.argmax(m.coef_[0]))]]
        return m.predict_proba(beta_features(p)[:, keep])[:, 1]
    s = logit(p_cal)

    def loss(a):
        return np.mean(np.logaddexp(0, s * np.exp(a)) - y * s * np.exp(a))

    a = minimize_scalar(loss, bounds=(-6, 4), method="bounded").x
    return 1 / (1 + np.exp(-logit(p) * np.exp(a)))


def ece(y, p, bins=10):
    order = np.argsort(p, kind="stable")
    return sum(len(i) * abs(y[i].mean() - p[i].mean()) for i in np.array_split(order, bins) if len(i)) / len(y)


def one_repeat(name, X, y, rep):
    rows = []
    outer = StratifiedKFold(FOLDS, shuffle=True, random_state=rep)
    for fold, (tr, te) in enumerate(outer.split(X, y)):
        seed = 1000 * rep + fold
        inner = StratifiedKFold(5, shuffle=True, random_state=seed)
        for mn in MODELS:
            est = model(mn, seed)
            p_cal = cross_val_predict(est, X[tr], y[tr], cv=inner, method="predict_proba")[:, 1]
            p_te = est.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            for meth in METHODS:
                p = np.clip(calibrate(meth, p_cal, y[tr], p_te), 0, 1)
                rows.append((name, mn, meth, rep, fold, brier_score_loss(y[te], p),
                             log_loss(y[te], np.clip(p, EPS, 1 - EPS), labels=[0, 1]), ece(y[te], p),
                             roc_auc_score(y[te], p)))
    return rows


if __name__ == "__main__":
    data, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(exist_ok=True)
    sets = load(data)
    info = pd.DataFrame([{"dataset": k, **v[2]} for k, v in sets.items()])
    info.to_csv(out / "datasets.csv", index=False)
    print(info.to_string(), flush=True)
    sys.path.insert(0, str(Path(__file__).parent))
    import study  # import by module name so that worker processes can unpickle the function
    jobs = [delayed(study.one_repeat)(k, v[0], v[1], r) for k, v in sets.items() for r in range(REPEATS)]
    rows = [r for chunk in Parallel(n_jobs=-1, verbose=5)(jobs) for r in chunk]
    cols = ["dataset", "model", "method", "repeat", "fold", "brier", "logloss", "ece", "auc"]
    pd.DataFrame(rows, columns=cols).to_csv(out / "splits.csv", index=False)
    print("rows", len(rows))
