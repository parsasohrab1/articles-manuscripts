"""Audit, min-entropy, von Neumann conditioning and sample-size sensitivity on the public IBM QRNG archive (Zenodo 10542216).

Usage: python qrng_study.py <dir S_2023_trials> <package src dir> <output dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT, SRC, OUT = (Path(a) for a in sys.argv[1:4])
sys.path.insert(0, str(SRC))
from qrngaudit import audit, data  # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
CLEAN = {"Python generator": "PRNG_Python", "secrets module": "PRNG_Secrets_module", "Random.org": "Random_org", "digits of pi": "Pi", "fair generator": "PRNG_biased_5_0_5_1"}
BIASED = ["PRNG_biased_505_0_495_1", "PRNG_biased_51_0_49_1", "PRNG_biased_52_0_48_1", "PRNG_biased_55_0_45_1", "PRNG_biased_58_0_42_1", "PRNG_biased_59_0_41_1"]
IBM = ["Belem", "Jakarta", "Lagos", "Lima", "Manila", "Nairobi", "Oslo", "Perth", "Quito"]
TESTS = ["frequency", "runs", "longest_run", "poker", "autocorrelation"]


def load():
    rows = []
    for f in sorted(ROOT.rglob("*.txt")):
        rel = f.relative_to(ROOT).parts
        if "combined_sequences" in rel or any("checkpoint" in p or p.startswith(".") for p in rel):
            continue
        s = f.read_text().strip()
        if not s or set(s) - {"0", "1"} or len(s) > 40000:
            continue
        d = rel[:-1]
        device = d[0] if d[0] != "PRNGs" else d[1]
        circuit = d[1] if len(d) > 1 and d[0] != "PRNGs" else "all"
        rows.append((device, circuit, s))
    return pd.DataFrame(rows, columns=["device_id", "circuit_id", "bitstring"])


def von_neumann(s):
    b = data.to_bits(s)
    n = len(b) // 2
    a, c = b[:2 * n:2], b[1:2 * n:2]
    return "".join(map(str, a[a != c]))


def min_entropy(strings):
    """SP 800-90B most-common-value (6.3.1) and Markov (6.3.3) estimates in bits per bit; transitions counted inside strings."""
    arr = [data.to_bits(s).astype(int) for s in strings]
    n = sum(len(a) for a in arr)
    ones = sum(int(a.sum()) for a in arr)
    p = max(ones, n - ones) / n
    mcv = -np.log2(min(1.0, p + 2.576 * np.sqrt(p * (1 - p) / (n - 1))))
    tr = sum(np.bincount(2 * a[:-1] + a[1:], minlength=4) for a in arr).reshape(2, 2).astype(float)
    P = tr / tr.sum(1, keepdims=True)
    p0, p1 = (n - ones) / n, ones / n
    with np.errstate(divide="ignore"):
        L = np.log2
        paths = [L(p0) + 127 * L(P[0, 0]), L(p0) + 64 * L(P[0, 1]) + 63 * L(P[1, 0]), L(p0) + L(P[0, 1]) + 126 * L(P[1, 1]),
                 L(p1) + L(P[1, 0]) + 126 * L(P[0, 0]), L(p1) + 64 * L(P[1, 0]) + 63 * L(P[0, 1]), L(p1) + 127 * L(P[1, 1])]
    markov = min(-max(paths) / 128, 1.0)
    return float(mcv), float(markov)


def describe(table):
    rows = []
    for (d, c), g in table.groupby(["device_id", "circuit_id"]):
        arr = [data.to_bits(s).astype(float) for s in g.bitstring]
        n = sum(len(a) for a in arr)
        r1 = np.mean([np.corrcoef(a[:-1], a[1:])[0, 1] for a in arr])
        mcv, mk = min_entropy(g.bitstring)
        rows.append({"device_id": d, "circuit_id": c, "strings": len(g), "bits": n, "frac_ones": sum(a.sum() for a in arr) / n, "lag1": r1, "h_mcv": mcv, "h_markov": mk})
    return pd.DataFrame(rows)


def verdicts(table, **kw):
    g, dev, circ = audit.audit(table, 0.05, **kw)
    return g, dev, circ


def summary(g):
    ibm = g[g.device_id.isin(IBM)]
    c = ibm.groupby(["device_id", "circuit_id"]).reject.any()
    return {"ibm_devices_unhealthy": int(ibm.groupby("device_id").reject.any().sum()), "ibm_circuits_rejected": int(c.sum()), "ibm_circuits": int(len(c)),
            "ibm_groups_rejected": int(ibm.reject.sum()), "ibm_groups": int(len(ibm)),
            "by_test": {t: int(ibm[ibm.test == t].reject.sum()) for t in TESTS},
            "clean_unhealthy": int(g[g.device_id.isin(CLEAN.values())].groupby("device_id").reject.any().sum()),
            "biased_unhealthy": int(g[g.device_id.isin(BIASED)].groupby("device_id").reject.any().sum())}


table = load()
assert set(table.device_id) == set(IBM) | set(CLEAN.values()) | set(BIASED), sorted(set(table.device_id))
res = {"strings": len(table), "bits": int(table.bitstring.str.len().sum()),
       "ibm_strings": int(table.device_id.isin(IBM).sum()), "ibm_bits": int(table[table.device_id.isin(IBM)].bitstring.str.len().sum())}

# 0. data screen: stuck strings (fraction of ones outside 0.25-0.75) and exact repeats of an earlier file
published = table
frac = published.bitstring.map(lambda s: s.count("1") / len(s))
stuck = (frac - 0.5).abs() > 0.25
dup = published.bitstring.duplicated() & ~stuck
flags = published[stuck | dup][["device_id", "circuit_id"]].assign(reason=np.where(stuck[stuck | dup], "stuck", "repeat"), length=published.bitstring.str.len()[stuck | dup], frac_ones=frac[stuck | dup])
flags.to_csv(OUT / "screened_out.csv", index=False)
res["screen"] = {"stuck": int(stuck.sum()), "repeat": int(dup.sum()),
                 "by_circuit": {f"{d}|{c}|{r}": int(len(g)) for (d, c, r), g in flags.groupby(["device_id", "circuit_id", "reason"])},
                 "max_abs_dev_kept": round(float((frac[~stuck] - 0.5).abs().max()), 4), "min_abs_dev_removed": round(float((frac[stuck] - 0.5).abs().min()), 4)}
table = published[~(stuck | dup)].reset_index(drop=True)
res["screened_strings"] = len(table)
res["screened_ibm_strings"] = int(table.device_id.isin(IBM).sum())
res["screened_ibm_bits"] = int(table[table.device_id.isin(IBM)].bitstring.str.len().sum())

# 1. raw audit, plus KS and fleet-wide variants (string p-values computed once)
spp = audit.string_pvalues(published)
sp = spp[np.repeat((~(stuck | dup)).values, 5)].reset_index(drop=True)
sp.to_csv(OUT / "string_pvalues.csv.gz", index=False)


def groups(sp_, method="binom", scope="device"):
    g = sp_.groupby(["device_id", "circuit_id", "test"])["p"].apply(lambda s: audit.uniformity_p(s.values, method)).rename("p_uniform").reset_index()
    if scope == "global":
        g["q_value"] = audit.bh_adjust(g.p_uniform.values)
    else:
        g["q_value"] = g.groupby("device_id")["p_uniform"].transform(lambda x: audit.bh_adjust(x.values))
    g["reject"] = g.q_value <= 0.05
    return g


raw = groups(sp)
raw.to_csv(OUT / "groups_raw.csv", index=False)
res["raw"] = summary(raw)
pub = groups(spp)
pub.to_csv(OUT / "groups_published.csv", index=False)
res["as_published"] = summary(pub)
res["as_published_naive"] = int((spp.groupby("device_id").p.min() < 0.01).sum())
dp = describe(published)
dp.to_csv(OUT / "circuits_published.csv", index=False)
res["as_published_device_frac_ones"] = {d: round(float(np.average(g.frac_ones, weights=g.bits)), 4) for d, g in dp[dp.device_id.isin(IBM)].groupby("device_id")}
res["variant_ks"] = summary(groups(sp, "ks"))
res["variant_global_bh"] = summary(groups(sp, "binom", "global"))
res["naive_sources_flagged"] = int((sp.groupby("device_id").p.min() < 0.01).sum())
res["sources"] = int(sp.device_id.nunique())

# 2. description and min-entropy
des = describe(table)
des.to_csv(OUT / "circuits_raw.csv", index=False)

# 3. von Neumann conditioning, then the same audit
vn = table.assign(bitstring=table.bitstring.map(von_neumann))
short = vn.bitstring.str.len() < 128
res["vn_short_strings_dropped"] = int(short.sum())
vn = vn[~short].reset_index(drop=True)
spv = audit.string_pvalues(vn)
gv = groups(spv)
gv.to_csv(OUT / "groups_vn.csv", index=False)
res["von_neumann"] = summary(gv)
res["von_neumann_ks"] = summary(groups(spv, "ks"))
dv = describe(vn)
dv["yield"] = dv.bits / des.bits
dv.to_csv(OUT / "circuits_vn.csv", index=False)
ib = des.device_id.isin(IBM)
for name, d in (("raw", des), ("vn", dv)):
    x = d[ib]
    res[f"ibm_{name}"] = {k: [round(float(x[k].min()), 4), round(float(x[k].median()), 4), round(float(x[k].max()), 4)] for k in ("frac_ones", "lag1", "h_mcv", "h_markov")}
    c = d[d.device_id.isin(CLEAN.values())]
    res[f"clean_{name}"] = {k: [round(float(c[k].min()), 4), round(float(c[k].max()), 4)] for k in ("frac_ones", "h_mcv", "h_markov")}
y = dv[ib]["yield"]
res["ibm_vn_yield"] = [round(float(y.min()), 4), round(float(y.median()), 4), round(float(y.max()), 4)]
res["ibm_vn_bits"] = int(dv[ib].bits.sum())
res["ibm_device_frac_ones"] = {d: round(float(np.average(g.frac_ones, weights=g.bits)), 4) for d, g in des[ib].groupby("device_id")}

# 4. sample-size sensitivity: draw n strings without replacement from each pool
rng = np.random.default_rng(0)
pools = {k: v.pivot_table(index=v.groupby("test").cumcount(), columns="test", values="p")[TESTS].values for k, v in sp[sp.circuit_id == "all"].groupby("device_id")}
for (d, c), v in sp[sp.device_id.isin(IBM)].groupby(["device_id", "circuit_id"]):
    pools[f"{d}|{c}"] = np.column_stack([v[v.test == t].p.values for t in TESTS])
sens = []
for name, m in pools.items():
    for n in (4, 8, 16, 32, 64, 128):
        if n > len(m):
            continue
        flag = naive = 0
        for _ in range(500):
            s = m[rng.choice(len(m), n, replace=False)]
            pu = [audit.uniformity_p(s[:, j]) for j in range(5)]
            flag += audit.bh_adjust(pu).min() <= 0.05
            naive += s.min() < 0.01
        sens.append({"source": name, "n": n, "pool": len(m), "audit": flag / 500, "naive": naive / 500})
sens = pd.DataFrame(sens)
sens["class"] = np.where(sens.source.isin(CLEAN.values()), "clean", np.where(sens.source.isin(BIASED), "biased", "ibm"))
sens.to_csv(OUT / "sensitivity.csv", index=False)
res["sensitivity_by_class"] = {f"{c}|{n}": [round(float(g.audit.mean()), 3), round(float(g.naive.mean()), 3)] for (c, n), g in sens.groupby(["class", "n"])}
res["sensitivity_biased"] = {f"{s}|{n}": round(float(a), 3) for s, n, a in sens[sens["class"] == "biased"][["source", "n", "audit"]].values}
(OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1))

# 5. dependence of the share of ones on the bit position modulo m (diagnostic for interleaved outputs of unequal bias)
from scipy.stats import chi2_contingency  # noqa: E402

pos = []
for (d, c), g in table.groupby(["device_id", "circuit_id"]):
    for m in range(2, 17):
        ones, tot = np.zeros(m), np.zeros(m)
        for s in g.bitstring:
            b = data.to_bits(s)
            k = len(b) // m * m
            ones += b[:k].reshape(-1, m).sum(0)
            tot += k // m
        f = ones / tot
        p = chi2_contingency(np.array([ones, tot - ones])).pvalue
        pos.append({"device_id": d, "circuit_id": c, "m": m, "spread": float(f.max() - f.min()), "p": float(p), "fractions": " ".join(f"{x:.4f}" for x in f)})
pos = pd.DataFrame(pos)
pos.to_csv(OUT / "position_bias.csv", index=False)
best = pos.loc[pos.groupby(["device_id", "circuit_id"]).spread.idxmax()]
vnrej = gv.groupby(["device_id", "circuit_id"]).reject.any()
best = best.assign(vn_rejected=[bool(vnrej[(d, c)]) for d, c in zip(best.device_id, best.circuit_id)])
best.drop(columns="fractions").to_csv(OUT / "position_summary.csv", index=False)
x = best[best.device_id.isin(IBM)]
res["position"] = {f"{'rejected' if k else 'passed'}_after_vn": {"circuits": int(len(g)), "median_spread": round(float(g.spread.median()), 4), "min": round(float(g.spread.min()), 4), "max": round(float(g.spread.max()), 4)} for k, g in x.groupby("vn_rejected")}
res["position_reference_max_spread"] = round(float(best[~best.device_id.isin(IBM)].spread.max()), 4)
(OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(best.drop(columns="fractions").to_string(index=False))
print(res["position"], res["position_reference_max_spread"])

# 6. post-hoc check: von Neumann applied separately to the four position streams of every circuit
bon = 0.05 / len(pos)
sig = pos.assign(s=pos.p < bon).groupby(["device_id", "circuit_id"]).s.any().reset_index()
sig = sig[sig.device_id.isin(IBM)]
t3 = sig.circuit_id.str.contains("Type_3")
res["position_significant"] = {"tests": int(len(pos)), "type3": f"{int(sig[t3].s.sum())} of {int(t3.sum())}", "other_ibm": f"{int(sig[~t3].s.sum())} of {int((~t3).sum())}",
                               "references": int((pos[~pos.device_id.isin(IBM)].p < bon).sum())}
four = pos[(pos.m == 4) & pos.circuit_id.str.contains("Type_3")]
res["type3_period4_spread"] = {r.device_id: [r.spread, r.fractions] for r in four.itertuples()}
st = table.assign(bitstring=table.bitstring.map(lambda s: "".join(von_neumann(s[k::4]) for k in range(4))))
gs = groups(audit.string_pvalues(st))
gs.to_csv(OUT / "groups_vn4.csv", index=False)
res["von_neumann_4streams"] = summary(gs)
ds = describe(st)
ds.to_csv(OUT / "circuits_vn4.csv", index=False)
for nm, g_ in (("vn", gv), ("vn4", gs)):
    c3 = g_[g_.circuit_id.str.contains("Type_3")]
    co = g_[g_.device_id.isin(IBM) & ~g_.circuit_id.str.contains("Type_3")]
    res[f"{nm}_split"] = {"type3_circuits_rejected": int(c3.groupby("circuit_id").reject.any().sum()), "type3_groups": int(c3.reject.sum()),
                          "other_circuits_rejected": int(co.groupby("circuit_id").reject.any().sum()), "other_groups": int(co.reject.sum())}
    res[f"{nm}_rejected_list"] = [f"{r.device_id}|{r.circuit_id}|{r.test}" for r in g_[g_.reject].itertuples()]
(OUT / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps({k: res[k] for k in ("position_significant", "von_neumann_4streams", "vn_split", "vn4_split", "vn4_rejected_list")}, indent=1))
