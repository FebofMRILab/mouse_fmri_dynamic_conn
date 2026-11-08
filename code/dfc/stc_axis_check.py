#!/usr/bin/env python3
"""
stc_axis_check.py
Which slice order was acquired, and did slice-timing correction remove it?

Status: NEW lab diagnostic (no analysis step changed).

Idea. In 2D multislice EPI a slice acquired tau seconds later in the TR
records the signal tau seconds ahead of its time stamp, so its time series
LEADS: lag = const - tau (Sladky et al., 2011). With interleaved acquisition
adjacent slices differ by about TR/2 (1 s at TR 2 s), a zig-zag that brain
physiology cannot produce over 0.9 mm. The global BOLD fluctuation is shared
by all slices, so each slice's lag relative to the whole-brain mean shows the
acquisition timing directly.

For every mouse:
  - brain mask = voxels whose temporal mean exceeds --mask_frac x the 99th
    percentile of the mean image;
  - linear detrend, band-pass 0.01-0.1 Hz (FFT), global mean g(t);
  - lag of each slice (mean over its brain voxels) relative to g: weighted
    least-squares slope of the cross-spectral phase on angular frequency over
    0.01-0.1 Hz (weights |cross-spectrum|); positive = later;
  - the same for each row along --row_axis (to see what an STC applied along
    the wrong axis did).
Across mice: per-mouse lags are regressed on each candidate pattern's slice
times (mouse means removed); slope -1 = that pattern is the acquisition and
it is uncorrected, 0 = not present (or corrected), +1 for the reversed
interleaved order (for an even slice count alt-z = const - alt+z). CI by
bootstrap over mice. The even-minus-odd contrast is the specific statistic:
for interleaved orders it is about +TR/2 when even slices were acquired first
(alt+z for an even slice count) and -TR/2 when odd slices were (alt-z), and a
real lag cannot alternate between adjacent slices. The slopes can absorb part
of a smooth real gradient along the axis (simulation: a 0.5-s gradient moved
the alt+z slope from -1.00 to -0.84; even minus odd stayed at 0.96 s).

Run it on the raw input (bold_datasink/input_bold) to learn the true order,
and on RABIES's slice-timing-correction output to see what the correction
did: along the slice axis an uncorrected order keeps slope -1; along the row
axis a correction applied there shows slope +1 for the pattern it assumed.

TR: give --TR (the NIfTI header of these data says 1.5 s; the TR is 2 s).

Usage
    python stc_axis_check.py --inputs "preprocess_outputs/bold_datasink/input_bold/*.nii.gz" \
        --TR 2 --slice_axis 2 --row_axis 1 --out_dir stc_check_raw
"""
import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import nibabel as nib
except ImportError:                      # pragma: no cover
    raise SystemExit("nibabel is required: pip install nibabel")

INK, MUTED, BASELINE = "#0b0b0b", "#898781", "#c3c2b7"
PATTERNS = ["seq+z", "seq-z", "alt+z", "alt-z", "alt+z2", "alt-z2"]


def pattern_times(name, n, TR):
    """AFNI 3dTshift -tpattern; '+z' = increasing index along the axis."""
    up = list(range(n))
    dn = up[::-1]
    order = {"seq+z": up, "seq-z": dn, "alt+z": up[0::2] + up[1::2],
             "alt+z2": up[1::2] + up[0::2], "alt-z": dn[0::2] + dn[1::2],
             "alt-z2": dn[1::2] + dn[0::2]}[name]
    t = np.empty(n)
    t[np.asarray(order)] = np.arange(n) * TR / n
    return t


def bandpass(X, TR, lo=0.01, hi=0.1):
    T = X.shape[0]
    t = np.arange(T)
    A = np.column_stack([np.ones(T), t - t.mean()])
    X = X - A @ np.linalg.lstsq(A, X, rcond=None)[0]
    F = np.fft.rfft(X, axis=0)
    f = np.fft.rfftfreq(T, TR)
    F[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(F, n=T, axis=0)


def lags_vs(Xparts, g, TR, lo=0.01, hi=0.1):
    """Lag (s, positive = later) of each column of Xparts relative to g."""
    T = len(g)
    f = np.fft.rfftfreq(T, TR)
    b = (f >= lo) & (f <= hi)
    G = np.fft.rfft(g)[b]
    X = np.fft.rfft(Xparts, axis=0)[b]
    S = X * np.conj(G)[:, None]
    w = np.abs(S)
    om = 2 * np.pi * f[b][:, None]
    return -(w * om * np.angle(S)).sum(0) / (w * om ** 2).sum(0)


def mouse_profile(path, TR, slice_axis, row_axis, mask_frac, min_vox):
    img = nib.load(path)
    D = np.asarray(img.dataobj, dtype=np.float32)
    if D.ndim != 4:
        raise SystemExit(f"{path}: not 4-D")
    mean = D.mean(-1)
    mask = mean > mask_frac * np.percentile(mean, 99)
    X = bandpass(D[mask].T.astype(np.float64), TR)            # T x V
    g = X.mean(1)
    idx = np.argwhere(mask)
    out = {}
    for name, ax in (("slice", slice_axis), ("row", row_axis)):
        if ax is None:
            continue
        n = D.shape[ax]
        parts, ok = np.zeros((X.shape[0], n)), np.zeros(n, bool)
        for k in range(n):
            sel = idx[:, ax] == k
            if sel.sum() >= min_vox:
                parts[:, k] = X[:, sel].mean(1)
                ok[k] = True
        lag = np.full(n, np.nan)
        lag[ok] = lags_vs(parts[:, ok], g, TR)
        out[name] = lag
    return out, D.shape, img.header.get_zooms()


def fit(L, tau, n_boot, rng):
    """Slope of lag on tau with mouse means removed; bootstrap over mice."""
    def slope(Lm):
        y = Lm - np.nanmean(Lm, 1, keepdims=True)
        x = np.where(np.isfinite(Lm), tau[None, :], np.nan)
        x = x - np.nanmean(x, 1, keepdims=True)
        return np.nansum(x * y) / np.nansum(x * x)
    b = slope(L)
    bb = np.array([slope(L[rng.integers(0, len(L), len(L))]) for _ in range(n_boot)])
    return b, np.percentile(bb, 2.5), np.percentile(bb, 97.5)


def main():
    ap = argparse.ArgumentParser(description="Slice order and STC check from per-slice timing")
    ap.add_argument("--inputs", required=True, nargs="+", help="4-D NIfTIs (globs allowed)")
    ap.add_argument("--TR", type=float, required=True)
    ap.add_argument("--slice_axis", type=int, default=2)
    ap.add_argument("--row_axis", type=int, default=1, help="-1 to skip")
    ap.add_argument("--order", default=None,
                    help="scanner acquisition order (PVM_ObjOrderList), comma-separated")
    ap.add_argument("--mask_frac", type=float, default=0.25)
    ap.add_argument("--min_vox", type=int, default=20)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--out_dir", default="stc_check")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    files = sorted({f for g in a.inputs for f in glob.glob(g)})
    if not files:
        raise SystemExit("no input files matched")
    row_axis = None if a.row_axis < 0 else a.row_axis
    prof, shapes = {"slice": [], "row": []}, []
    for i, f in enumerate(files):
        p, shape, zooms = mouse_profile(f, a.TR, a.slice_axis, row_axis, a.mask_frac, a.min_vox)
        shapes.append(shape)
        for k in p:
            prof[k].append(p[k])
        if i == 0:
            print(f"{os.path.basename(f)}: shape {shape}, voxel {zooms[:3]}, header TR {zooms[3]:g} "
                  f"(using --TR {a.TR:g})")
    if len({s[:3] for s in shapes}) > 1:
        raise SystemExit("inputs differ in grid size")
    print(f"{len(files)} scans")

    rows = []
    for name in prof:
        if not prof[name]:
            continue
        L = np.vstack(prof[name])
        n = L.shape[1]
        pd.DataFrame(L, index=[os.path.basename(f) for f in files],
                     columns=[f"{name}_{k}" for k in range(n)]).to_csv(
            os.path.join(a.out_dir, f"lags_per_{name}.csv"))
        cands = {p: pattern_times(p, n, a.TR) for p in PATTERNS}
        if a.order and name == "slice":
            o = [int(v) for v in a.order.split(",")]
            if len(o) == n:
                t = np.empty(n); t[np.asarray(o)] = np.arange(n) * a.TR / n
                cands["scanner"] = t
                t = np.empty(n); t[np.asarray([n - 1 - s for s in o])] = np.arange(n) * a.TR / n
                cands["scanner, reversed"] = t
        for p, tau in cands.items():
            b, lo, hi = fit(L, tau, a.n_boot, rng)
            rows.append(dict(axis=name, n=n, pattern=p, slope=b, ci_lo=lo, ci_hi=hi))
        ev = np.nanmean(L[:, 0::2], 1) - np.nanmean(L[:, 1::2], 1)
        bb = [np.nanmean(ev[rng.integers(0, len(ev), len(ev))]) for _ in range(a.n_boot)]
        rows.append(dict(axis=name, n=n, pattern="even minus odd (s)", slope=np.nanmean(ev),
                         ci_lo=np.percentile(bb, 2.5), ci_hi=np.percentile(bb, 97.5)))
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(a.out_dir, "stc_axis_check.csv"), index=False)
    print("\nslope of lag on candidate slice times (mouse means removed); "
          "-1 = acquired that way and uncorrected, +1 = reversed order, 0 = absent")
    with pd.option_context("display.width", 140, "display.float_format", "{:.3f}".format):
        print(S.to_string(index=False))

    warnings.filterwarnings("ignore", category=RuntimeWarning)   # rows outside the brain are NaN
    names = [k for k in prof if prof[k]]
    fig, axes = plt.subplots(1, len(names), figsize=(5.2 * len(names), 3.6), squeeze=False)
    for ax, name in zip(axes[0], names):
        L = np.vstack(prof[name])
        L = L - np.nanmean(L, 1, keepdims=True)
        n = L.shape[1]
        m = np.nanmean(L, 0)
        se = np.nanstd(L, 0) / np.sqrt(np.sum(np.isfinite(L), 0))
        ax.errorbar(np.arange(n), m, yerr=se, fmt="o-", color=INK, ms=3.5, lw=1, label="data")
        for p, st in (("alt+z", "--"), ("alt-z", ":")):
            tau = pattern_times(p, n, a.TR)
            ax.plot(np.arange(n), -(tau - tau.mean()), st, color=MUTED, lw=1.2,
                    label=f"{p}, uncorrected")
        ax.axhline(0, color=BASELINE, lw=0.8)
        ax.set_xlabel(f"{name} index (axis {a.slice_axis if name == 'slice' else row_axis})")
        ax.set_ylabel("lag vs whole-brain mean (s)")
        ax.legend(frameon=False, fontsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, "stc_axis_check.png"), dpi=200)
    print(f"\nWrote {a.out_dir}/stc_axis_check.csv, lags_per_slice.csv, lags_per_row.csv, stc_axis_check.png")


if __name__ == "__main__":
    main()
