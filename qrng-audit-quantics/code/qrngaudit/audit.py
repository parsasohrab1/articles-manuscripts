"""Device-level audit: per-string tests -> per-(device, circuit, test) uniformity tests of the p-values -> Benjamini-Hochberg at the FDR target."""
import numpy as np
import pandas as pd
from scipy.stats import binom, kstest

from . import data, tests_stat


def bh_adjust(p):
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / (np.arange(n) + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def string_pvalues(table):
    rows = []
    for r in table.itertuples():
        for k, p in tests_stat.run_all(data.to_bits(r.bitstring)).items():
            rows.append({"device_id": r.device_id, "circuit_id": r.circuit_id, "test": k, "p": p})
    return pd.DataFrame(rows)


def uniformity_p(p, method="binom", cut=0.05):
    """One p-value per group saying whether the string-level p-values look uniform.

    `ks`: Kolmogorov-Smirnov against U(0,1). `binom`: one-sided binomial test for too many p-values below `cut`; discrete tests give
    conservative (stochastically larger) p-values, so the lower tail is the informative one and the false-alarm rate stays controlled.
    """
    p = np.asarray(p)
    if method == "ks":
        return float(kstest(p, "uniform").pvalue)
    k = int(np.sum(p < cut))
    return float(binom.sf(k - 1, len(p), cut))


def audit(table, q=0.05, method="binom", scope="device"):
    """Returns (per-group table with adjusted p-values, per-device verdict with a calibrated health score).

    scope="device": Benjamini-Hochberg inside each device (its circuits x tests), so a faulty neighbour cannot raise the false-alarm
    rate of a healthy device. scope="global": one BH over all devices (controls the false discovery rate of the whole fleet report).
    """
    sp = string_pvalues(table)
    g = sp.groupby(["device_id", "circuit_id", "test"])["p"].apply(lambda s: uniformity_p(s.values, method)).rename("p_uniform").reset_index()
    if scope == "global":
        g["q_value"] = bh_adjust(g["p_uniform"].values)
    else:
        g["q_value"] = g.groupby("device_id")["p_uniform"].transform(lambda x: bh_adjust(x.values))
    g["reject"] = g["q_value"] <= q
    dev = g.groupby("device_id").agg(min_q=("q_value", "min"), failed_tests=("reject", "sum")).reset_index()
    dev["verdict"] = np.where(dev.failed_tests > 0, "unhealthy", "healthy")
    dev["health_score"] = dev["min_q"]
    circ = g.groupby(["device_id", "circuit_id"]).agg(min_q=("q_value", "min"), failed_tests=("reject", "sum")).reset_index()
    circ["verdict"] = np.where(circ.failed_tests > 0, "unhealthy", "healthy")
    return g, dev, circ


def empirical_false_alarm_rate(n_devices=60, n_strings=40, length=1024, q=0.05, method="binom", seed=0):
    """Share of ideal (fault-free) devices that the audit calls unhealthy; to be reported with every verdict."""
    spec = {f"ideal{i}": "ideal" for i in range(n_devices)}
    _, dev, _ = audit(data.make_table(spec, n_strings, length, seed), q, method)
    return float((dev.verdict == "unhealthy").mean())


def detection_rate(kind, n_devices=20, n_strings=40, length=1024, q=0.05, method="binom", seed=500, **kw):
    spec = {f"{kind}{i}": kind for i in range(n_devices)}
    _, dev, _ = audit(data.make_table(spec, n_strings, length, seed), q, method)
    return float((dev.verdict == "unhealthy").mean())


def naive_false_alarm_rate(n_devices=30, n_strings=40, length=1024, cut=0.01, seed=3):
    """Baseline the framework replaces: flag a device when any single string has p < cut. Shows why device-level control is needed."""
    sp = string_pvalues(data.make_table({f"i{i}": "ideal" for i in range(n_devices)}, n_strings, length, seed))
    return float((sp.groupby("device_id")["p"].min() < cut).mean())
