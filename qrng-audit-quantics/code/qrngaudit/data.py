"""Bit-string loader (CSV: device_id,circuit_id,bitstring) and synthetic sources: ideal, biased, correlated."""
import numpy as np
import pandas as pd


def load_csv(path):
    return pd.read_csv(path, dtype={"bitstring": str})[["device_id", "circuit_id", "bitstring"]]


def to_bits(s):
    return np.frombuffer(s.encode(), dtype=np.uint8) - ord("0")


def source(kind="ideal", n_strings=50, length=1024, seed=0, bias=0.55, stick=0.2):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_strings):
        if kind == "ideal":
            bits = rng.integers(0, 2, length)
        elif kind == "biased":
            bits = (rng.random(length) < bias).astype(int)
        elif kind == "correlated":                     # with probability `stick` repeat the previous bit
            bits = np.zeros(length, int)
            bits[0] = rng.integers(0, 2)
            r = rng.random(length)
            fresh = rng.integers(0, 2, length)
            for i in range(1, length):
                bits[i] = bits[i - 1] if r[i] < stick else fresh[i]
        else:
            raise ValueError(kind)
        out.append("".join(map(str, bits)))
    return out


def make_table(spec, n_strings=50, length=1024, seed=0):
    """spec: {device_id: kind or (kind, circuit_count)}. Returns the CSV-shaped table."""
    rows = []
    for j, (dev, kind) in enumerate(spec.items()):
        for c in range(2):
            for s in source(kind, n_strings, length, seed + 17 * j + c):
                rows.append((dev, f"c{c}", s))
    return pd.DataFrame(rows, columns=["device_id", "circuit_id", "bitstring"])
