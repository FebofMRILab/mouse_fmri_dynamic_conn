#!/usr/bin/env python3
"""
make_slice_maps.py  (input for slice_timing_lag_check.py)

RABIES 0.6.1 writes no native-space labels. This gives the lag check what it
needs without them: for every mouse, how each ROI's commonspace voxels are
distributed over the slices that were acquired.

Status: NEW lab code. It uses RABIES's own transforms with ANTs
antsApplyTransforms and changes no analysis step.

Per mouse
  1. A slice-number map on the mouse's native EPI grid (bold_datasink/
     input_bold): every voxel holds its slice index + 1 along --slice_axis.
  2. The map is pushed into commonspace with the transforms RABIES used for
     commonspace_bold (transforms_datasink), in the antsApplyTransforms order
     unbiased_to_atlas warp, affine, anat_to_unbiased warp, affine,
     bold_to_anat warp, affine; GenericLabel interpolation; reference grid =
     that of commonspace_bold. Value 0 = outside the acquired slab.
  3. For every ROI of the commonspace atlas used for the ROI time series
     (run3: --ROI_labels_file), the number of its commonspace voxels coming
     from each slice. This mirrors how the ROI signals were made: averages
     over commonspace voxels.
  Region order: region j = the j-th smallest nonzero atlas value (the order
  RABIES uses for parcellated time series), unless --roi_labels gives
  atlas_value. The count must equal the number of columns of the signal files.

Checks (written to the report; the lag check should not be run if they fail)
  - Registration chain: the mean of the first --qc_vols native volumes,
    pushed through the same chain (linear interpolation), is correlated with
    the mean of the same volumes of RABIES's commonspace_bold, for --qc_n mice
    (0 = all). Expected r > 0.9; a wrong order or file gives a low r.
  - Atlas, region order and mouse mapping (optional, --cleaned_glob): ROI
    means of RABIES's confound-corrected commonspace time series, with the
    resampled atlas, are correlated with the columns of signal<N>.txt.
    Expected r near 1 for every ROI. This also verifies that signal<N>.txt
    belongs to the bids_id given in --meta.

Usage (in rabies/, after `module load ants`)
    python signals/make_slice_maps.py --meta signals/meta_data_with_bids.csv \
        --rabies_preproc preprocess_outputs \
        --atlas /blue/febo/share/civm_atlas/BrainNetCoord.nii \
        --signals_dir signals/hmm_signals --out_dir signals/slice_maps \
        [--cleaned_glob "confound_correction_outputs/**/*sub-{id}_*cleaned*.nii*"]
Then
    python signals/slice_timing_lag_check.py --loadings <dfc>/cpca_loadings_roi.csv \
        --pairs signals/slice_maps/slice_timing_pairs.csv \
        --fractions signals/slice_maps/slice_fractions.csv --strata group \
        --method <EPI method file> --TR 2 ...
"""
import argparse
import glob
import os
import re
import shlex
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

try:
    import nibabel as nib
except ImportError:                      # pragma: no cover
    raise SystemExit("nibabel is required: pip install nibabel")


def one(pattern, what):
    hits = sorted(set(glob.glob(pattern, recursive=True)))
    if len(hits) != 1:
        raise FileNotFoundError(f"{what}: {len(hits)} files match {pattern}"
                                + (": " + "; ".join(os.path.relpath(h) for h in hits[:6]) if hits else ""))
    return hits[0]


def clean_id(v):
    s = re.sub(r"^sub-", "", str(v).strip())
    return re.sub(r"\.0$", "", s)


def save_like(data, header, path, dtype):
    h = header.copy()
    h.set_data_dtype(dtype)
    img = nib.Nifti1Image(np.asarray(data, dtype=dtype), None, header=h)
    img.header.set_slope_inter(1, 0)
    nib.save(img, path)


def run(cmd, log):
    log.write(" ".join(shlex.quote(c) for c in cmd) + "\n")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(cmd)}\n{r.stderr[-2000:]}")


def mean_first(path, n):
    img = nib.load(path)
    k = min(n, img.shape[3]) if len(img.shape) == 4 else None
    d = np.asarray(img.dataobj[..., :k] if k else img.dataobj, dtype=np.float64)
    return (d.mean(-1) if k else d), img.header


def main():
    ap = argparse.ArgumentParser(description="Per-mouse slice maps in commonspace for the lag check")
    ap.add_argument("--meta", required=True, help="CSV: subject_id, group, bids_id")
    ap.add_argument("--rabies_preproc", required=True, help="RABIES preprocess output folder")
    ap.add_argument("--atlas", required=True, help="commonspace ROI atlas used for the time series")
    ap.add_argument("--roi_labels", default=None, help="optional CSV: region, label, atlas_value")
    ap.add_argument("--signals_dir", default="hmm_signals")
    ap.add_argument("--file_pattern", default="signal{}.txt")
    ap.add_argument("--slice_axis", type=int, default=2, help="native voxel axis of the slices")
    ap.add_argument("--ants", default="antsApplyTransforms",
                    help="antsApplyTransforms command (may include a container prefix)")
    ap.add_argument("--interp_labels", default="GenericLabel", help="or NearestNeighbor")
    ap.add_argument("--qc_n", type=int, default=5, help="mice for the registration check (0 = all)")
    ap.add_argument("--qc_vols", type=int, default=50)
    ap.add_argument("--cleaned_glob", default=None,
                    help="confound-corrected commonspace time series, with {id} for the bids_id")
    ap.add_argument("--edge_crop", type=int, default=15,
                    help="volumes removed at each end between cleaned data and signal files")
    ap.add_argument("--out_dir", default="slice_maps")
    a = ap.parse_args()

    ants = shlex.split(a.ants)
    if shutil.which(ants[0]) is None:
        sys.exit(f"'{ants[0]}' not found: run `module load ants` or give --ants")
    os.makedirs(os.path.join(a.out_dir, "maps"), exist_ok=True)
    log = open(os.path.join(a.out_dir, "ants_commands.log"), "w")
    rep = open(os.path.join(a.out_dir, "slice_maps_report.txt"), "w")

    def say(s):
        print(s)
        rep.write(s + "\n")

    m = pd.read_csv(a.meta, dtype=str, encoding="utf-8-sig", sep=None, engine="python")
    m.columns = [re.sub(r"[^a-z0-9_]", "", c.strip().lower()) for c in m.columns]
    if not {"subject_id", "group", "bids_id"} <= set(m.columns):
        sys.exit(f"{a.meta} needs subject_id, group, bids_id; found {list(m.columns)}")
    m = m[m.subject_id.notna() & m.bids_id.notna()].copy()
    m["subject_id"] = m.subject_id.map(clean_id)
    m["bids_id"] = m.bids_id.map(clean_id)

    P = a.rabies_preproc
    T = os.path.join(P, "transforms_datasink")
    u2a_w = one(f"{T}/unbiased_to_atlas_warp/**/*1Warp.nii*", "unbiased_to_atlas warp")
    u2a_a = one(f"{T}/unbiased_to_atlas_affine/**/*.mat", "unbiased_to_atlas affine")

    def files(bid):
        return dict(
            native=one(f"{P}/bold_datasink/input_bold/**/sub-{bid}_*bold.nii*", f"sub-{bid} input_bold"),
            cs_bold=one(f"{P}/bold_datasink/commonspace_bold/**/sub-{bid}_*.nii*", f"sub-{bid} commonspace_bold"),
            chain=[u2a_w, u2a_a,
                   one(f"{T}/anat_to_unbiased_warp/**/sub-{bid}_*1Warp.nii*", f"sub-{bid} anat_to_unbiased warp"),
                   one(f"{T}/anat_to_unbiased_affine/**/sub-{bid}_*.mat", f"sub-{bid} anat_to_unbiased affine"),
                   one(f"{T}/bold_to_anat_warp/**/sub-{bid}_*1Warp.nii*", f"sub-{bid} bold_to_anat warp"),
                   one(f"{T}/bold_to_anat_affine/**/sub-{bid}_*.mat", f"sub-{bid} bold_to_anat affine")])

    # ---- reference grid and atlas on it
    first = files(m.bids_id.iloc[0])
    csh = nib.load(first["cs_bold"]).header
    ref = os.path.join(a.out_dir, "commonspace_ref.nii.gz")
    h = csh.copy()
    h.set_data_shape(csh.get_data_shape()[:3])
    save_like(np.zeros(csh.get_data_shape()[:3]), h, ref, np.uint8)
    atlas_cs = os.path.join(a.out_dir, "atlas_commonspace.nii.gz")
    run(ants + ["-d", "3", "-i", a.atlas, "-r", ref, "-o", atlas_cs, "-n", a.interp_labels,
                "-t", "identity"], log)
    A = np.rint(np.asarray(nib.load(atlas_cs).dataobj)).astype(np.int64)
    if a.roi_labels:
        R = pd.read_csv(a.roi_labels).sort_values("region")
        values = R.atlas_value.astype(int).tolist()
    else:
        values = sorted(int(v) for v in np.unique(A) if v != 0)
    n_cols = None
    sig0 = os.path.join(a.signals_dir, a.file_pattern.format(m.subject_id.iloc[0]))
    if os.path.exists(sig0):
        n_cols = np.loadtxt(sig0, ndmin=2).shape[1]
    say(f"Atlas on the commonspace grid: {len(values)} regions"
        + (f"; signal files have {n_cols} columns" if n_cols else ""))
    if n_cols is not None and n_cols != len(values):
        sys.exit("Region count differs from the signal files: give --roi_labels with atlas_value.")
    size = {v: int((A == v).sum()) for v in values}

    # ---- per mouse
    rows, pairs, qc = [], [], []
    qc_ids = set(m.bids_id if a.qc_n == 0 else m.bids_id.iloc[:a.qc_n])
    for i, r in m.iterrows():
        bid = r.bids_id
        f = files(bid)
        img = nib.load(f["native"])
        shp = img.shape[:3]
        ax = a.slice_axis
        n = shp[ax]
        idx = np.indices(shp)[ax] + 1
        nat = os.path.join(a.out_dir, "maps", f"sub-{bid}_slicemap_native.nii.gz")
        h = img.header.copy()
        h.set_data_shape(shp)
        save_like(idx, h, nat, np.int16)
        cs = os.path.join(a.out_dir, "maps", f"sub-{bid}_slicemap_commonspace.nii.gz")
        cmd = ants + ["-d", "3", "-i", nat, "-r", ref, "-o", cs, "-n", a.interp_labels]
        for t in f["chain"]:
            cmd += ["-t", t]
        run(cmd, log)
        S = np.rint(np.asarray(nib.load(cs).dataobj)).astype(np.int64)
        for j, v in enumerate(values, start=1):
            c = np.bincount(S[A == v], minlength=n + 1)
            rows.append(dict(bids_id=bid, region=j, atlas_value=v, outside=int(c[0]),
                             **{f"s{k}": int(c[k + 1]) for k in range(n)}))
        pairs.append(dict(signal_file=os.path.join(a.signals_dir, a.file_pattern.format(r.subject_id)),
                          bids_id=bid, subject_id=r.subject_id, group=r.group))
        q = dict(bids_id=bid, n_slices=n)
        if bid in qc_ids:
            mean_nat, hn = mean_first(f["native"], a.qc_vols)
            hn = hn.copy()
            hn.set_data_shape(shp)
            mn = os.path.join(a.out_dir, "maps", f"sub-{bid}_mean_native.nii.gz")
            save_like(mean_nat, hn, mn, np.float32)
            mw = os.path.join(a.out_dir, "maps", f"sub-{bid}_mean_pushed.nii.gz")
            cmd = ants + ["-d", "3", "-i", mn, "-r", ref, "-o", mw, "-n", "Linear"]
            for t in f["chain"]:
                cmd += ["-t", t]
            run(cmd, log)
            x = np.asarray(nib.load(mw).dataobj, dtype=np.float64)
            y, _ = mean_first(f["cs_bold"], a.qc_vols)
            ok = (A > 0) & (x > 0) & (y > 0)
            q["r_registration"] = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 100 else np.nan
            if a.cleaned_glob:
                try:
                    cl = one(a.cleaned_glob.replace("{id}", bid), f"sub-{bid} cleaned")
                    D = np.asarray(nib.load(cl).dataobj, dtype=np.float32)
                    X = np.column_stack([D[A == v].mean(0) for v in values])
                    sig = np.loadtxt(pairs[-1]["signal_file"], ndmin=2)
                    if X.shape[0] == sig.shape[0] + 2 * a.edge_crop:
                        X = X[a.edge_crop:X.shape[0] - a.edge_crop]
                    if X.shape[0] == sig.shape[0]:
                        rr = [np.corrcoef(X[:, j], sig[:, j])[0, 1] for j in range(sig.shape[1])]
                        q["roi_r_median"] = float(np.nanmedian(rr))
                        q["roi_r_min"] = float(np.nanmin(rr))
                    else:
                        q["roi_r_note"] = f"lengths {X.shape[0]} vs {sig.shape[0]}"
                except Exception as e:      # report, do not stop the maps
                    q["roi_r_note"] = str(e)[:1200]
        qc.append(q)
        print(f"  sub-{bid}: done" + (f", registration r = {q['r_registration']:.3f}"
                                      if "r_registration" in q else ""))

    F = pd.DataFrame(rows)
    F.to_csv(os.path.join(a.out_dir, "slice_fractions.csv"), index=False)
    pd.DataFrame(pairs).to_csv(os.path.join(a.out_dir, "slice_timing_pairs.csv"), index=False)
    Q = pd.DataFrame(qc)
    Q.to_csv(os.path.join(a.out_dir, "qc.csv"), index=False)

    scols = [c for c in F.columns if re.fullmatch(r"s\d+", c)]
    inside = F[scols].sum(axis=1)
    frac_out = F.outside / (F.outside + inside).replace(0, np.nan)
    say(f"{F.bids_id.nunique()} mice, {len(values)} regions, {len(scols)} slices")
    say(f"ROI voxels outside the acquired slab: median {np.nanmedian(frac_out):.1%}; "
        f"ROI-mouse pairs with >50% outside: {(frac_out > 0.5).sum()}")
    if "r_registration" in Q:
        rr = Q.r_registration.dropna()
        say(f"Registration check ({len(rr)} mice): r median {rr.median():.3f}, min {rr.min():.3f} "
            f"(expected > 0.9)")
        if (rr < 0.8).any():
            say("WARNING: low r for " + ", ".join(Q.loc[Q.r_registration < 0.8, "bids_id"]))
    if "roi_r_median" in Q:
        say(f"ROI signal check: median r {Q.roi_r_median.median():.3f}, lowest ROI r "
            f"{Q.roi_r_min.min():.3f} (expected near 1)")
    if "roi_r_note" in Q:
        bad = Q[Q.roi_r_note.notna()]
        say(f"ROI signal check not done for {len(bad)} mice (details in qc.csv); first:")
        say("  sub-" + str(bad.bids_id.iloc[0]) + ": " + str(bad.roi_r_note.iloc[0]))
    say(f"Wrote {a.out_dir}/slice_fractions.csv, slice_timing_pairs.csv, qc.csv")
    rep.close()
    log.close()


if __name__ == "__main__":
    main()
