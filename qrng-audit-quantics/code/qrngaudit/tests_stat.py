"""Five NIST SP 800-22 style tests, implemented with numpy/scipy only. Every function takes a 0/1 array and returns a p-value."""
import numpy as np
from scipy.special import erfc, gammaincc

_LR_PI = np.array([0.2148, 0.3672, 0.2305, 0.1875])    # longest run of ones in 8-bit blocks: <=1, 2, 3, >=4


def frequency(b):
    n = len(b)
    s = abs(np.sum(2 * b.astype(int) - 1)) / np.sqrt(n)
    return float(erfc(s / np.sqrt(2)))


def runs(b):
    n = len(b)
    pi = b.mean()
    if abs(pi - 0.5) >= 2 / np.sqrt(n):
        return 0.0
    v = 1 + int(np.sum(b[1:] != b[:-1]))
    return float(erfc(abs(v - 2 * n * pi * (1 - pi)) / (2 * np.sqrt(2 * n) * pi * (1 - pi))))


def _longest_ones(block):
    best = cur = 0
    for x in block:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best


def longest_run(b):
    nb = len(b) // 8
    L = np.array([_longest_ones(b[i * 8:(i + 1) * 8]) for i in range(nb)])
    v = np.array([np.sum(L <= 1), np.sum(L == 2), np.sum(L == 3), np.sum(L >= 4)])
    chi = np.sum((v - nb * _LR_PI) ** 2 / (nb * _LR_PI))
    return float(gammaincc(3 / 2, chi / 2))


def poker(b, m=4):
    k = len(b) // m
    w = b[:k * m].reshape(k, m) @ (1 << np.arange(m))
    counts = np.bincount(w, minlength=2 ** m)
    chi = (2 ** m / k) * np.sum(counts.astype(float) ** 2) - k
    return float(gammaincc((2 ** m - 1) / 2, chi / 2))


def autocorrelation(b, d=1):
    n = len(b) - d
    a = int(np.sum(b[:n] != b[d:]))
    z = 2 * (a - n / 2) / np.sqrt(n)
    return float(erfc(abs(z) / np.sqrt(2)))


TESTS = {"frequency": frequency, "runs": runs, "longest_run": longest_run, "poker": poker, "autocorrelation": autocorrelation}


def run_all(bits):
    b = np.asarray(bits, dtype=np.uint8)
    return {k: f(b) for k, f in TESTS.items()}
