#!/usr/bin/env python3
"""
slice_timing_lag_check.py
Are the complex-PCA lags explained by when each slice was acquired?

Status: NEW lab diagnostic, not taken from a published package. It reads the
pipeline's outputs (cpca_loadings_roi.csv, the ROI signal files, RABIES
native-space labels) and changes no analysis step.

Why slice timing can create lags (Sladky et al., 2011, NeuroImage 58:588-594)
    In 2D multislice EPI each slice s is acquired at its own time tau_s within
    the TR, but every slice of a volume gets the same time stamp. A slice
    acquired tau_s late records the signal tau_s ahead of its label:
    y_n = x(n*TR + tau_s). Without slice-timing correction (STC) a region
    therefore looks EARLIER by its acquisition time: lag = const - t (slope -1).
    After a correct STC acquisition time should not predict lag (slope 0).

Why the test is done within ROIs, across mice
    Averaged over the mice, an ROI's acquisition time is close to a straight
    line in its position along the slice axis (the odd/even alternation of
    interleaved orders averages out). A real front-to-back lag would then look
    exactly like slice timing. But each mouse sits slightly differently in the
    slice package, so the same ROI is acquired at different times in different
    mice while its anatomy, and any real lag, stays the same. Comparing mice
    within each ROI separates the two.

Steps
    1. Each mouse's ROI x slice voxel fractions from its RABIES native-space
       label map (bold_datasink/native_labels): the grid the slices were
       acquired on. ROI acquisition time t[m, r] = fractions @ slice times.
    2. Each mouse's lag of every ROI on the group components (unchanged), by
       dual regression (Beckmann et al., 2009, NeuroImage 47:S148; Nickerson
       et al., 2017, Front Neurosci 11:115): the mouse's analytic signal Z is
       projected on all group components, c = Z V, and Z is regressed on all
       their time courses jointly, B = (c'c)^-1 c'Z. Joint regression keeps
       component 1 from leaking into the weaker components (projecting on one
       component at a time gave component-2 slopes of +1.9 instead of -1 in
       simulation). The ROI phase is angle(conj(B)), which equals the group
       phase convention of dfc_cpca_fit.m (positive = later). Phase folded
       onto (-pi/2, pi/2] (an ROI at phase ~pi is an opposite-sign member, not
       a half-cycle lag; slice timing shifts both signs alike), lag = phase /
       (2*pi*f), f from the loadings file. For component 1 folding changes
       nothing.
    3. lag[m, r] = a_r + g_m + b * t[m, r] + e, weights |v_r|^2 (the phase of
       a weak loading is noisy). ROI effects a_r absorb every lag that belongs
       to the anatomy; mouse effects g_m absorb a constant phase offset per
       mouse. b uses only how acquisition time varies between mice within an
       ROI. Expected b: -1 with no or failed STC, 0 after a correct STC.
    4. CI: bootstrap over mice. p: the mice's acquisition-time maps are
       shuffled between mice (within --strata, e.g. group, if given), so each
       mouse's lags are paired with another mouse's slice placement; this
       keeps the spatial structure of both and breaks only their pairing.
       p = (1 + #|b_perm| >= |b|) / (k + 1). (Permuting slice times across
       slices instead was anticonservative in simulation: p < 0.05 in 23-46%
       of null runs, because it destroys the spatial structure that lag noise
       shares with acquisition time.)
    Also reported, descriptive only: group lag vs group-mean acquisition time
    (confounded with position along the slice axis, as explained above).

Candidate patterns (AFNI 3dTshift -tpattern, which RABIES --tpattern uses);
'+z' = increasing voxel index along the slice axis:
    seq+z  0,1,...,n-1            seq-z  n-1,...,0
    alt+z  0,2,4,...,1,3,...      alt-z  n-1,n-3,...,n-2,n-4,...
    alt+z2 1,3,...,0,2,...        alt-z2 n-2,n-4,...,n-1,n-3,...
    slice time = (place in the acquisition order) * TR / n
Optional: the order actually used by the scanner (ParaVision method file,
PVM_ObjOrderList, tested in both index directions because the Bruker slice
index may run either way in the NIfTI), or a BIDS SliceTiming list.

RABIES 0.6.1 writes no native-space labels: then build the slice fractions
with make_slice_maps.py (per-mouse slice-number maps pushed into commonspace
with RABIES's transforms, counted per ROI of the time-series atlas) and pass
--fractions slice_fractions.csv with the pairs file it writes.

Usage (HiPerGator)
    python slice_timing_lag_check.py \
        --loadings dfc_dcc_cpca/cpca_loadings_roi.csv \
        --pairs slice_timing_pairs.csv --TR 2 --rabies_tpattern alt-z \
        --roi_labels roi_labels.csv [--method /path/to/one/scan/method] \
        [--expected_T 570] [--out_dir slice_timing_check]

    slice_timing_pairs.csv: one row per mouse, columns signal_file (the ROI
        time series used by run_dcc_cpca_null.m, T x 82) and label_file (that
        mouse's RABIES native_labels NIfTI).
    roi_labels.csv: region (1..82, column order of the signal files), label
        and, if the atlas integer values are not 1..82 in that order,
        atlas_value.
"""
import argparse
import json
import os
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import nibabel as nib
except ImportError:                      # pragma: no cover
    raise SystemExit("nibabel is required: pip install nibabel")

INK, SECONDARY, MUTED, GRID, BASELINE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({"font.family": "Arial", "font.size": 12, "axes.titlesize": 13,
                     "axes.titleweight": "bold", "axes.labelsize": 12,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.linewidth": 1.2,
                     "savefig.dpi": 350, "savefig.facecolor": "white",
                     "figure.facecolor": "white"})
sns.set_theme(style="ticks", context="paper")
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})
warnings.filterwarnings("ignore", message=".*findfont.*")

PATTERNS = ["seq+z", "seq-z", "alt+z", "alt-z", "alt+z2", "alt-z2"]


# ------------------------------------------------------------------ timing
def pattern_order(name, n):
    """Slice indices in the order they are acquired (AFNI 3dTshift)."""
    up = list(range(n))
    dn = up[::-1]
    return {"seq+z": up, "seq-z": dn,
            "alt+z": up[0::2] + up[1::2], "alt+z2": up[1::2] + up[0::2],
            "alt-z": dn[0::2] + dn[1::2], "alt-z2": dn[1::2] + dn[0::2]}[name]


def times_from_order(order, n, TR):
    order = np.asarray(order, int)
    if sorted(order.tolist()) != list(range(n)):
        raise SystemExit(f"acquisition order is not a permutation of 0..{n - 1}")
    t = np.empty(n)
    t[order] = np.arange(n) * TR / n
    return t


def read_bruker_order(path):
    """PVM_ObjOrderList from a ParaVision method file."""
    txt = open(path, errors="replace").read()
    m = re.search(r"##\$PVM_ObjOrderList=\(\s*(\d+)\s*\)\s*\n(.*?)(?=\n##|\n\$\$|\Z)", txt, re.S)
    if not m:
        raise SystemExit(f"PVM_ObjOrderList not found in {path}")
    vals = [int(v) for v in m.group(2).split()]
    if len(vals) != int(m.group(1)):
        raise SystemExit(f"PVM_ObjOrderList: expected {m.group(1)} values, found {len(vals)}")
    return vals


def read_bids_timing(path):
    js = json.load(open(path))
    if "SliceTiming" not in js:
        raise SystemExit(f"no SliceTiming in {path}")
    t = np.asarray(js["SliceTiming"], float)
    if js.get("SliceEncodingDirection", "k").endswith("-"):   # first entry = highest index
        t = t[::-1]
    return t


# ------------------------------------------------------------------ labels
def slice_axis_of(img, requested):
    """Voxel axis the slices were acquired along: --slice_axis, else the
    thickest voxel dimension, else the anterior-posterior axis (RABIES
    --stc_axis Y)."""
    zooms = np.asarray(img.header.get_zooms()[:3], float)
    codes = nib.aff2axcodes(img.affine)
    if requested != "auto":
        return int(requested), zooms, codes
    thick = int(np.argmax(zooms))
    if zooms[thick] > 1.2 * np.median(zooms):
        return thick, zooms, codes
    ap = [i for i, c in enumerate(codes) if c in "AP"]
    return (ap[0] if ap else 2), zooms, codes


def roi_slice_fractions(path, values, requested_axis):
    """ROI x slice voxel fractions (rows sum to 1; NaN if the ROI is absent)."""
    img = nib.load(path)
    ax, zooms, codes = slice_axis_of(img, requested_axis)
    L = np.rint(np.asarray(img.dataobj)).astype(np.int64)
    if L.ndim == 4:
        L = L[..., 0]
    flat = np.moveaxis(L, ax, -1).reshape(-1, L.shape[ax])
    n = flat.shape[1]
    top = max(int(flat.max()), int(max(values))) + 1
    counts = np.stack([np.bincount(np.clip(flat[:, s], 0, None), minlength=top)
                       for s in range(n)], axis=1)
    H = counts[np.asarray(values)].astype(float)
    tot = H.sum(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        F = np.where(tot > 0, H / tot, np.nan)
    return F, dict(label_file=path, n_slices=n, slice_axis=ax, axcodes="".join(codes),
                   voxel_mm="x".join(f"{z:g}" for z in zooms),
                   n_roi_present=int((tot[:, 0] > 0).sum()))


def fractions_from_table(FT, bids_id, n_roi):
    """ROI x slice fractions for one mouse from make_slice_maps.py output
    (commonspace voxels of each ROI coming from each acquired slice)."""
    t = FT[FT.bids_id == str(bids_id)].sort_values("region")
    if len(t) != n_roi:
        raise SystemExit(f"slice_fractions: {len(t)} regions for bids_id {bids_id}, expected {n_roi}")
    scols = sorted([c for c in FT.columns if re.fullmatch(r"s\d+", c)], key=lambda c: int(c[1:]))
    H = t[scols].to_numpy(float)
    tot = H.sum(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        F = np.where(tot > 0, H / tot, np.nan)
    return F, dict(label_file="slice_fractions:" + str(bids_id), n_slices=len(scols),
                   slice_axis="native", axcodes="commonspace", voxel_mm="",
                   n_roi_present=int((tot[:, 0] > 0).sum()))


# ------------------------------------------------------------------ lags
def analytic_signal(X):
    """Same as dfc_analytic_signal.m (FFT; real part = X)."""
    T = X.shape[0]
    h = np.zeros(T)
    h[0] = 1
    if T % 2 == 0:
        h[1:T // 2] = 2
        h[T // 2] = 1
    else:
        h[1:(T + 1) // 2] = 2
    return np.fft.ifft(np.fft.fft(X, axis=0) * h[:, None], axis=0)


def fold(ph):
    ph = np.angle(np.exp(1j * ph))
    return np.where(ph > np.pi / 2, ph - np.pi, np.where(ph <= -np.pi / 2, ph + np.pi, ph))


def group_components(L, comps):
    V, f = {}, {}
    for k in comps:
        lk = L[L.component == k].sort_values("region")
        ph, lag = lk.phase_rad.to_numpy(float), lk.lag_sec.to_numpy(float)
        ok = np.abs(ph) > 1e-9
        f[k] = float(np.median(ph[ok] / (2 * np.pi * lag[ok])))
        V[k] = lk.magnitude.to_numpy(float) * np.exp(1j * ph)
    return V, f


def mouse_lags(signal_file, V, f, expected_T):
    """Dual regression of one mouse on all group components (see step 2)."""
    X = np.loadtxt(signal_file, ndmin=2)
    if expected_T and X.shape[0] != expected_T:
        print(f"  note: {os.path.basename(signal_file)} has {X.shape[0]} rows")
    X = X - X.mean(0)
    sd = X.std(0, ddof=1)
    X = X / np.where(sd > 0, sd, 1)
    Z = analytic_signal(X)
    ks = sorted(V)
    c = Z @ np.column_stack([V[k] for k in ks])          # T x K time courses
    B = np.linalg.lstsq(c, Z, rcond=None)[0]             # K x N maps
    return {k: fold(np.angle(np.conj(B[i]))) / (2 * np.pi * f[k]) for i, k in enumerate(ks)}


# ------------------------------------------------------------------ stats
def demean2(A, mask, w, iters=None):
    """Two-way (mouse x ROI) demeaning of A[m, r, ...] over observed cells;
    ROI weights w for the mouse means. Same transform for every column.
    One pass is exact when every ROI is present in every mouse; otherwise
    alternate until converged."""
    if iters is None:
        iters = 1 if mask.all() else 50
    A = np.where(mask[..., None], A, 0.0) if A.ndim == 3 else np.where(mask, A, 0.0)
    W = mask * w[None, :]
    W3 = W[..., None] if A.ndim == 3 else W
    M3 = mask[..., None] if A.ndim == 3 else mask
    for _ in range(iters):
        rm = (W3 * A).sum(1, keepdims=True) / np.maximum(W3.sum(1, keepdims=True), 1e-12)
        if iters > 1 and np.max(np.abs(rm)) < 1e-10:
            break
        A = np.where(M3, A - rm, 0.0)
        A = A - (M3 * A).sum(0, keepdims=True) / np.maximum(M3.sum(0, keepdims=True), 1)
        A = np.where(M3, A, 0.0)
    return A


def fe_slope(Y, T, mask, w):
    """Within (two-way FE) WLS slope; T may carry extra trailing columns."""
    W = mask * w[None, :]
    y = demean2(Y, mask, w)
    t = demean2(T, mask, w)
    if t.ndim == 3:
        num = np.einsum("mr,mr,mrp->p", W, y, t)
        den = np.einsum("mr,mrp,mrp->p", W, t, t)
    else:
        num, den = (W * y * t).sum(), (W * t * t).sum()
    return num / np.where(den > 0, den, np.nan), t


def wslope(y, t, w):
    w = w / w.sum()
    dt, dy = t - w @ t, y - w @ y
    return (w @ (dt * dy)) / (w @ (dt * dt))


def wcorr(a, b, w):
    w = w / w.sum()
    da, db = a - w @ a, b - w @ b
    return (w @ (da * db)) / np.sqrt((w @ da ** 2) * (w @ db ** 2))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Slice acquisition time vs complex-PCA lag")
    ap.add_argument("--loadings", required=True, help="cpca_loadings_roi.csv")
    ap.add_argument("--pairs", required=True,
                    help="CSV: signal_file and label_file, or signal_file and bids_id with --fractions")
    ap.add_argument("--fractions", default=None,
                    help="slice_fractions.csv from make_slice_maps.py (instead of native label files)")
    ap.add_argument("--roi_labels", default=None, help="CSV: region, label[, atlas_value]")
    ap.add_argument("--TR", type=float, default=2.0)
    ap.add_argument("--expected_T", type=int, default=570)
    ap.add_argument("--slice_axis", default="auto", help="voxel axis of the slices or auto")
    ap.add_argument("--rabies_tpattern", default="alt-z",
                    help="--tpattern given to RABIES (alt-z is the RABIES default)")
    ap.add_argument("--method", default=None, help="ParaVision method file")
    ap.add_argument("--order", default=None, help="acquisition order, comma-separated")
    ap.add_argument("--bids_json", default=None, help="BOLD sidecar with SliceTiming")
    ap.add_argument("--strata", default=None,
                    help="column of --pairs within which mice are shuffled (e.g. group)")
    ap.add_argument("--components", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--n_perm", type=int, default=10000)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--out_dir", default="slice_timing_check")
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"])
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    L = pd.read_csv(a.loadings)
    n_roi = int(L.region.max())
    names = [f"ROI {i}" for i in range(1, n_roi + 1)]
    values = list(range(1, n_roi + 1))
    if a.roi_labels:
        R = pd.read_csv(a.roi_labels).set_index("region").reindex(range(1, n_roi + 1))
        if "label" in R:
            names = R["label"].astype(str).tolist()
        if "atlas_value" in R:
            values = R["atlas_value"].astype(int).tolist()
    V, f = group_components(L, sorted(L.component.unique()))   # all, for dual regression
    a.components = [k for k in a.components if k in V]
    wv = {k: np.abs(V[k]) ** 2 for k in a.components}

    # ---- per mouse: slice fractions and lags
    P = pd.read_csv(a.pairs, dtype=str)
    FT = pd.read_csv(a.fractions, dtype={"bids_id": str}) if a.fractions else None
    Fs, info, lags = [], [], {k: [] for k in a.components}
    for _, row in P.iterrows():
        if FT is not None:
            F, d = fractions_from_table(FT, row.bids_id, n_roi)
        else:
            F, d = roi_slice_fractions(row.label_file, values, a.slice_axis)
        if F.shape[0] != n_roi:
            raise SystemExit("ROI count mismatch")
        d["signal_file"] = row.signal_file
        Fs.append(F)
        info.append(d)
        lm = mouse_lags(row.signal_file, V, f, a.expected_T)
        for k in a.components:
            lags[k].append(lm[k])
    I = pd.DataFrame(info)
    n = int(I.n_slices.mode().iloc[0])
    keep = np.where(I.n_slices == n)[0]
    if len(keep) < len(I):
        print(f"WARNING: {len(I) - len(keep)} mouse/mice with a different slice count "
              f"left out (kept n_slices = {n})")
    if I.slice_axis.nunique() > 1:
        print("WARNING: the slice axis differs between label files; set --slice_axis")
    I.to_csv(os.path.join(a.out_dir, "slice_timing_label_files.csv"), index=False)
    Fall = np.stack([Fs[i] for i in keep])                       # mouse x ROI x slice
    M = len(keep)
    present = np.isfinite(Fall[:, :, 0])
    if FT is not None:
        print(f"{M} mice, {n} slices (from {os.path.basename(a.fractions)})")
    else:
        print(f"{M} mice, {n} slices along voxel axis {I.slice_axis.iloc[keep[0]]} "
              f"(orientation {I.axcodes.iloc[keep[0]]}, voxel {I.voxel_mm.iloc[keep[0]]} mm)")
    Ffill = np.nan_to_num(Fall)

    # ---- candidate acquisition patterns
    cands = {p: times_from_order(pattern_order(p, n), n, a.TR) for p in PATTERNS}
    order = (read_bruker_order(a.method) if a.method else
             [int(v) for v in a.order.split(",")] if a.order else None)
    if order is not None:
        if len(order) != n:
            raise SystemExit(f"scanner order has {len(order)} slices, label maps have {n}")
        cands["scanner"] = times_from_order(order, n, a.TR)
        cands["scanner, reversed"] = times_from_order([n - 1 - s for s in order], n, a.TR)
    if a.bids_json:
        tj = read_bids_timing(a.bids_json)
        if len(tj) == n:
            cands["BIDS SliceTiming"] = tj
        else:
            print(f"WARNING: SliceTiming has {len(tj)} entries, label maps have {n}; skipped")
    if a.rabies_tpattern not in PATTERNS:
        raise SystemExit(f"--rabies_tpattern must be one of {PATTERNS}")

    # ---- long table: mouse x ROI
    long = []
    for j, i in enumerate(keep):
        d = pd.DataFrame({"signal_file": I.signal_file.iloc[i], "region": np.arange(1, n_roi + 1),
                          "label": names, "present": present[j],
                          "slice_position": Ffill[j] @ np.arange(n)})
        for p, tau in cands.items():
            d[f"t_{p}"] = np.where(present[j], Ffill[j] @ tau, np.nan)
        for k in a.components:
            d[f"lag_c{k}_s"] = lags[k][i]
        long.append(d)
    pd.concat(long).to_csv(os.path.join(a.out_dir, "slice_timing_mouse_roi.csv"), index=False)

    # ---- regressions
    strata = P[a.strata].to_numpy()[keep] if a.strata else np.zeros(M, int)

    def shuffled():
        idx = np.arange(M)
        for g in pd.unique(strata):
            s_ = np.where(strata == g)[0]
            idx[s_] = rng.permutation(s_)
        return idx

    perm = np.stack([shuffled() for _ in range(a.n_perm)])       # n_perm x mouse
    boot = rng.integers(0, M, size=(a.n_boot, M))
    rows, plotdat = [], {}
    for k in a.components:
        Y = np.stack([lags[k][i] for i in keep])
        ymask = np.isfinite(Y)
        mask = present & ymask
        balanced = bool(mask.all())
        w = wv[k]
        gl = np.angle(V[k])
        glag = fold(gl) / (2 * np.pi * f[k])
        for p, tau in cands.items():
            T = Ffill @ tau                                       # mouse x ROI
            b, t_dm = fe_slope(Y, T, mask, w)
            Wm = mask * w[None, :]
            lev = np.sqrt((Wm * t_dm ** 2).sum() / Wm.sum())
            if balanced:
                bp = np.concatenate([fe_slope(Y, np.moveaxis(T[perm[c:c + 250]], 0, -1),
                                              mask, w)[0]
                                     for c in range(0, a.n_perm, 250)])
            else:
                bp = np.array([fe_slope(Y, T[ix], ymask & present[ix], w)[0] for ix in perm])
            pval = (1 + np.sum(np.abs(bp) >= abs(b))) / (a.n_perm + 1)
            bb = np.array([fe_slope(Y[ix], T[ix], mask[ix], w)[0] for ix in boot])
            tg = np.nanmean(np.where(mask, T, np.nan), axis=0)
            ok = np.isfinite(tg)
            posg = np.nanmean(np.where(mask, Ffill @ np.arange(n), np.nan), axis=0)
            rows.append(dict(
                component=k, pattern=p, rabies_pattern=(p == a.rabies_tpattern),
                n_mice=M, n_obs=int(mask.sum()), freq_hz=f[k],
                within_roi_sd_t=lev, slope_within=b,
                ci_lo=np.nanpercentile(bb, 2.5), ci_hi=np.nanpercentile(bb, 97.5),
                p_perm=pval,
                group_slope_descriptive=wslope(glag[ok], tg[ok], w[ok]),
                group_corr_t_position=wcorr(tg[ok], posg[ok], w[ok])))
            if k == a.components[0]:
                plotdat[p] = (t_dm, demean2(Y, mask, w), mask, w)
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(a.out_dir, "slice_timing_lag_slopes.csv"), index=False)

    print("\nslope_within: lag on acquisition time, within ROI across mice "
          "(expected -1 without STC, 0 after a correct STC)")
    with pd.option_context("display.width", 170, "display.float_format", "{:.3f}".format):
        print(S[["component", "pattern", "within_roi_sd_t", "slope_within", "ci_lo", "ci_hi",
                 "p_perm", "group_slope_descriptive", "group_corr_t_position"]]
              .to_string(index=False))
    low = S[(S.component == a.components[0]) & (S.within_roi_sd_t < 0.05)]
    if len(low):
        print("\nNOTE: acquisition time varies little within ROIs (SD < 0.05 s) for "
              + ", ".join(low.pattern) + "; slopes there are poorly determined.")

    # ---- figure
    k1 = a.components[0]
    p0 = "scanner" if "scanner" in cands else a.rabies_tpattern
    t_dm, y_dm, mask, w = plotdat[p0]
    x, y = t_dm[mask], y_dm[mask]
    ww = (mask * w[None, :])[mask]
    fig = plt.figure(figsize=(10.5, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.6, left=0.08, right=0.97,
                          bottom=0.15, top=0.88)
    ax = fig.add_subplot(gs[0])
    edges = np.quantile(x, np.linspace(0, 1, 13))
    bi = np.clip(np.digitize(x, edges[1:-1]), 0, 11)
    xm, ym, ye = [], [], []
    for j in range(12):
        s = bi == j
        if s.sum() > 5:
            wj = ww[s] / ww[s].sum()
            xm.append(wj @ x[s])
            ym.append(wj @ y[s])
            ye.append(np.sqrt(wj @ (y[s] - wj @ y[s]) ** 2) / np.sqrt(s.sum()))
    ax.errorbar(xm, ym, yerr=ye, fmt="o", color=INK, ms=4, lw=1, capsize=0)
    b = S[(S.component == k1) & (S.pattern == p0)].iloc[0]
    xx = np.array([min(xm), max(xm)])
    ax.plot(xx, b.slope_within * xx, color=INK, lw=1.6, label=f"fit, slope {b.slope_within:.2f}")
    ax.plot(xx, -xx, color=MUTED, lw=1.4, ls="--", label="no STC (slope −1)")
    ax.axhline(0, color=BASELINE, lw=0.8, zorder=0)
    ax.set_xlabel(f"Acquisition time vs ROI mean (s), {p0}")
    ax.set_ylabel(f"Component {k1} lag vs ROI mean (s)")
    ax.set_title("A", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="best")
    sns.despine(ax=ax)

    ax = fig.add_subplot(gs[1])
    pats = list(cands)
    style = [("o", INK, INK), ("o", "white", INK), ("s", MUTED, MUTED)]
    off = np.linspace(0.18, -0.18, len(a.components)) if len(a.components) > 1 else [0.0]
    for j, k in enumerate(a.components):
        sk = S[S.component == k].set_index("pattern").loc[pats]
        yy = np.arange(len(pats))[::-1] + off[j]
        m, fc, ec = style[min(j, 2)]
        ax.hlines(yy, sk.ci_lo, sk.ci_hi, color=ec, lw=1.2)
        ax.scatter(sk.slope_within, yy, marker=m, s=34, facecolor=fc, edgecolor=ec, lw=1.0,
                   zorder=3, label=f"component {k}")
    for yb in np.arange(len(pats) - 1) + 0.5:
        ax.axhline(yb, color=GRID, lw=0.6, zorder=0)
    ax.axvline(0, color=BASELINE, lw=1.0)
    ax.axvline(-1, color=MUTED, lw=1.0, ls="--")
    ax.set_ylim(-0.5, len(pats) - 0.5)
    ax.set_yticks(np.arange(len(pats))[::-1])
    ax.set_yticklabels([p + (" (RABIES)" if p == a.rabies_tpattern else "") for p in pats])
    ax.set_xlabel("Within-ROI slope of lag on acquisition time (95% CI)")
    ax.set_title("B", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 1.0),
              ncol=len(a.components), handletextpad=0.2, columnspacing=1.0)
    sns.despine(ax=ax)
    for ext in a.formats:
        fig.savefig(os.path.join(a.out_dir, f"FigS_slice_timing_lag.{ext}"),
                    dpi=600 if ext in ("png", "tiff") else None)
    print(f"\nWrote {a.out_dir}/slice_timing_lag_slopes.csv, slice_timing_mouse_roi.csv, "
          f"FigS_slice_timing_lag.*")


if __name__ == "__main__":
    main()
