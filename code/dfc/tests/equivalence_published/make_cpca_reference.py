#!/usr/bin/env python3
"""
Reference outputs for test_cpca_vs_published.m: the complex PCA of Bolt et al.
(2022) as published in github.com/tsb46/complex_pca (cpca.py), on simulated
band-passed travelling waves (10 scans x 570 volumes x 30 ROIs, TR 2 s).

cpca.py takes scipy.signal.hilbert, conjugates it, centres the data and takes
a truncated SVD (fbpca.pca, a randomised SVD). fbpca did not build here, so the
exact SVD of the same centred matrix is used (fbpca approximates it). Two
versions are written: Hilbert transform after concatenating the scans (as the
published script does) and per scan (as dfc_analytic_signal.m does).

    python make_cpca_reference.py [out_dir]
"""
import os
import sys

import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import zscore

out = sys.argv[1] if len(sys.argv) > 1 else "cpca_reference"
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(1)
S, T, N, TR = 10, 570, 30, 2.0
b, a = butter(4, [0.01, 0.1], btype="band", fs=1 / TR)
t = np.arange(T) * TR
scans = []
for s in range(S):
    lag1 = np.linspace(0, 6, N)
    lag2 = np.linspace(4, 0, N)[::-1] ** 1.2
    w1 = np.sin(2 * np.pi * 0.05 * (t[:, None] - lag1[None, :]) + rng.uniform(0, 2 * np.pi))
    w2 = np.sin(2 * np.pi * 0.03 * (t[:, None] - lag2[None, :]) + rng.uniform(0, 2 * np.pi))
    amp = 1 + 0.5 * np.sin(2 * np.pi * t / 400 + s)
    X = zscore(filtfilt(b, a, amp[:, None] * w1 + 0.6 * w2 + 1.5 * rng.standard_normal((T, N)), axis=0), axis=0)
    scans.append(X)
    np.savetxt(os.path.join(out, f"scan{s + 1:02d}.txt"), X)


def published_pca(Z, k=3):
    Zc = Z - Z.mean(axis=0)                       # fbpca.pca centres (raw=False)
    _, _, Va = np.linalg.svd(Zc, full_matrices=False)
    return Va[:k]


for tag, Z in (("concatenated", hilbert(np.vstack(scans), axis=0).conj()),
               ("per_scan", np.vstack([hilbert(x, axis=0) for x in scans]).conj())):
    Va = published_pca(Z)
    np.savetxt(os.path.join(out, f"Va_{tag}_real.txt"), Va.real)
    np.savetxt(os.path.join(out, f"Va_{tag}_imag.txt"), Va.imag)
np.savetxt(os.path.join(out, "scipy_hilbert_scan01_imag.txt"), hilbert(scans[0], axis=0).imag)
print(f"Reference written to {out}/")
