#!/usr/bin/env python3
"""
compute_lag_covariate.py
-------------------------------------------------------------------
Per-mouse coupling magnitude: the covariate for the complex-PCA lag index.

The lag index (dfc_cpca_subject_stats.m) is
    sqrt( sum Im(C)^2 / sum |C|^2 )   over all off-diagonal ROI pairs,
estimated split-half from the analytic signal of each mouse's standardised
ROI time series. When coupling is weak, the ratio is inflated (noise in the
denominator), so group comparisons adjust for the denominator itself: the
lag-invariant coupling magnitude over ALL pairs.

Do not use zero-lag FC strength for this. Zero-lag correlation of a pair is
about |coherence| x cos(phase lag), so it falls as lags grow: adjusting the
lag index for it would remove part of the group difference it is meant to
test. A thresholded measure (top 15% of edges) also covers a different set
of pairs than the lag index does.

Output (one row per mouse)
    subject_id
    coupling_magnitude  RMS |complex coherence| over all pairs, split-half
                        (noise-debiased). THE covariate.
    coupling_zero_lag   RMS of the real (zero-lag) part, split-half
    coupling_lagged     RMS of the imaginary (lagged) part, split-half
    lag_index           recomputed exactly as in MATLAB (should match
                        cpca_roi_lag_index in dfc_subject_metrics.csv)

Usage
    python compute_lag_covariate.py --data_dir hmm_signals --n_subjects 95 \
        --out lag_covariate.csv
    python plot_dfc_results.py --in_dir dfc_dcc_cpca --lag_covariate lag_covariate.csv
"""

import argparse
import os

import numpy as np
import pandas as pd


def analytic_signal(X):
    """Same as dfc_analytic_signal.m (FFT Hilbert transform, per column)."""
    T = X.shape[0]
    h = np.zeros(T)
    h[0] = 1
    if T % 2 == 0:
        h[1:T // 2] = 2
        h[T // 2] = 1
    else:
        h[1:(T + 1) // 2] = 2
    return np.fft.ifft(np.fft.fft(X, axis=0) * h[:, None], axis=0)


def standardise(X):
    """Same as local_standardise in run_dcc_cpca_null.m."""
    Xs = X - X.mean(axis=0)
    sd = Xs.std(axis=0, ddof=1)
    sd[sd <= 0] = 1
    return Xs / sd


def subject_measures(X):
    Z = analytic_signal(standardise(X))
    T, N = Z.shape
    h = T // 2
    C1 = Z[:h].conj().T @ Z[:h] / h
    C2 = Z[h:].conj().T @ Z[h:] / (T - h)
    off = ~np.eye(N, dtype=bool)

    # lag index, identical to dfc_cpca_subject_stats.m
    num = np.sum(C1.imag[off] * C2.imag[off])
    den = np.sum((C1[off] * C2[off].conj()).real)
    lag_index = np.sqrt(max(num, 0) / max(den, np.finfo(float).eps))

    # complex coherence (unit diagonal), split-half cross-products
    def coherence(C):
        d = np.sqrt(np.real(np.diag(C)))
        return C / np.outer(d, d)
    K1, K2 = coherence(C1), coherence(C2)
    n = off.sum()
    mag2 = np.sum((K1[off] * K2[off].conj()).real) / n
    re2 = np.sum(K1.real[off] * K2.real[off]) / n
    im2 = np.sum(K1.imag[off] * K2.imag[off]) / n
    root = lambda v: np.sqrt(max(v, 0.0))
    return dict(coupling_magnitude=root(mag2), coupling_zero_lag=root(re2),
                coupling_lagged=root(im2), lag_index=lag_index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="hmm_signals")
    ap.add_argument("--file_pattern", default="signal{}.txt")
    ap.add_argument("--n_subjects", type=int, default=95)
    ap.add_argument("--out", default="lag_covariate.csv")
    args = ap.parse_args()

    rows = []
    for s in range(1, args.n_subjects + 1):
        fn = os.path.join(args.data_dir, args.file_pattern.format(s))
        if not os.path.isfile(fn):
            print(f"missing {fn}")
            continue
        X = np.loadtxt(fn, delimiter=None if open(fn).readline().count(",") == 0 else ",")
        if X.ndim != 2 or X.shape[0] < X.shape[1]:
            print(f"subject {s}: unexpected shape {X.shape} (expected time x ROI)")
            continue
        rows.append(dict(subject_id=s, **subject_measures(X.astype(float))))
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(out.describe().T[["mean", "std", "min", "max"]].to_string())
    print(f"\nCorrelation of lag index with coupling magnitude across mice: "
          f"r = {out.lag_index.corr(out.coupling_magnitude):.2f}")
    print(f"Wrote {args.out} ({len(out)} subjects)")


if __name__ == "__main__":
    main()
