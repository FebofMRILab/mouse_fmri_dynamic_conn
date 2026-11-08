#!/usr/bin/env python3
"""
check_roi_signals.py  (diagnostic for make_slice_maps.py's ROI signal check)

For each mouse: ROI means of RABIES's confound-corrected commonspace time
series with the resampled atlas (slice_maps/atlas_commonspace.nii.gz), against
every column of signal<N>.txt. For each signal column j it reports
  r_same   correlation with atlas region j (should be near 1)
  best     the atlas region k with the highest correlation, and that r
so that the cause of a low r can be told apart:
  - one mouse low across all ROIs        -> signal file of another mouse
  - the same ROIs low in every mouse,
    best match a fixed other region      -> region order or a mirrored atlas
                                            (best = the homologue, with names)
  - small ROIs low, best match = itself  -> resampling of small regions

Status: NEW lab diagnostic; reads files only.

Usage (in signals/)
    python check_roi_signals.py --slice_maps slice_maps \
        --cleaned_glob "../confound_correction_outputs/confound_correction_datasink/cleaned_timeseries/**/*sub-{id}_*.nii*" \
        --roi_labels roi_labels.csv [--n_mice 10]
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import nibabel as nib


def one(pattern):
    hits = sorted(set(glob.glob(pattern, recursive=True)))
    if len(hits) != 1:
        raise FileNotFoundError(f"{len(hits)} files match {pattern}")
    return hits[0]


def corr_cols(A, B):
    A = (A - A.mean(0)) / A.std(0)
    B = (B - B.mean(0)) / B.std(0)
    return (A.T @ B) / len(A)                    # A-columns x B-columns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice_maps", default="slice_maps", help="make_slice_maps.py output folder")
    ap.add_argument("--cleaned_glob", required=True, help="with {id} for the bids_id")
    ap.add_argument("--roi_labels", default=None, help="CSV: region, label[, atlas_value]")
    ap.add_argument("--masks", default=None,
                    help="text file with one ROI mask per line, in region order (e.g. seeds_list.txt): "
                         "ROI means from these masks instead of the atlas")
    ap.add_argument("--voxel_z", action="store_true",
                    help="z-score every voxel's time series before averaging (tests whether the signal "
                         "files were made from variance-normalised voxels)")
    ap.add_argument("--edge_crop", type=int, default=15)
    ap.add_argument("--n_mice", type=int, default=0, help="0 = all")
    ap.add_argument("--out_dir", default=None)
    a = ap.parse_args()
    out = a.out_dir or a.slice_maps
    os.makedirs(out, exist_ok=True)
    A = np.rint(np.asarray(nib.load(os.path.join(a.slice_maps, "atlas_commonspace.nii.gz")).dataobj)).astype(int)
    P = pd.read_csv(os.path.join(a.slice_maps, "slice_timing_pairs.csv"), dtype=str)
    if a.n_mice:
        P = P.iloc[:a.n_mice]
    names, values = None, sorted(int(v) for v in np.unique(A) if v != 0)
    if a.roi_labels:
        R = pd.read_csv(a.roi_labels).sort_values("region")
        names = R["label"].astype(str).tolist() if "label" in R else None
        if "atlas_value" in R:
            values = R.atlas_value.astype(int).tolist()
    n = len(values)
    names = names or [f"ROI{j}" for j in range(1, n + 1)]
    size = np.array([(A == v).sum() for v in values])
    mask_paths = [q.strip() for q in open(a.masks) if q.strip()] if a.masks else None
    if mask_paths and len(mask_paths) != n:
        raise SystemExit(f"{len(mask_paths)} masks listed, {n} regions")
    sel = None
    same, best, bestr = [], [], []
    for _, p in P.iterrows():
        ci = nib.load(one(a.cleaned_glob.replace("{id}", p.bids_id)))
        D = np.asarray(ci.dataobj, dtype=np.float32)
        if sel is None:
            if mask_paths:
                from nibabel.processing import resample_from_to
                sel = []
                for q in mask_paths:
                    mi = nib.load(q)
                    if mi.shape[:3] != D.shape[:3] or not np.allclose(mi.affine, ci.affine, atol=1e-4):
                        mi = resample_from_to(mi, (D.shape[:3], ci.affine), order=0)
                    sel.append(np.asarray(mi.dataobj) > 0)
                size = np.array([m.sum() for m in sel])
                print(f"ROI definition: {len(sel)} masks from {a.masks}")
            else:
                sel = [A == v for v in values]
                print("ROI definition: atlas_commonspace.nii.gz")
        if a.voxel_z:
            def zmean(V):
                V = V - V.mean(1, keepdims=True)
                sd = V.std(1, keepdims=True)
                return (V / np.where(sd > 0, sd, np.inf)).mean(0)
            X = np.column_stack([zmean(D[m]) for m in sel])
        else:
            X = np.column_stack([D[m].mean(0) for m in sel])
        S = np.loadtxt(p.signal_file, ndmin=2)
        if X.shape[0] == S.shape[0] + 2 * a.edge_crop:
            X = X[a.edge_crop:X.shape[0] - a.edge_crop]
        if X.shape[0] != S.shape[0] or S.shape[1] != n:
            print(f"sub-{p.bids_id}: shapes {X.shape} vs {S.shape}, skipped")
            continue
        C = corr_cols(X, S)                      # atlas region k x signal column j
        same.append(np.diag(C))
        best.append(C.argmax(0))
        bestr.append(C.max(0))
        print(f"sub-{p.bids_id}: median r(same region) {np.median(np.diag(C)):.3f}", flush=True)
    ids = P.bids_id.iloc[:len(same)].tolist()
    same, best, bestr = np.array(same), np.array(best), np.array(bestr)
    pd.DataFrame(same, index=ids, columns=names).to_csv(os.path.join(out, "roi_signal_r_same.csv"))
    pd.DataFrame(best + 1, index=ids, columns=names).to_csv(os.path.join(out, "roi_signal_best_region.csv"))

    per_roi = pd.DataFrame({
        "region": np.arange(1, n + 1), "label": names, "atlas_value": values, "n_voxels": size,
        "r_same_median": np.median(same, 0), "r_same_min": same.min(0),
        "best_region_mode": [np.bincount(best[:, j], minlength=n).argmax() + 1 for j in range(n)],
        "best_is_same_frac": (best == np.arange(n)[None, :]).mean(0),
        "r_best_median": np.median(bestr, 0)})
    per_roi["best_label"] = [names[k - 1] for k in per_roi.best_region_mode]
    per_roi.to_csv(os.path.join(out, "roi_signal_check_by_roi.csv"), index=False)
    per_mouse = pd.DataFrame({"bids_id": ids, "r_same_median": np.median(same, 1),
                              "frac_best_is_same": (best == np.arange(n)[None, :]).mean(1)})
    per_mouse.to_csv(os.path.join(out, "roi_signal_check_by_mouse.csv"), index=False)

    print(f"\n{len(ids)} mice. Median r(same region): {np.median(same):.3f}; "
          f"columns whose best match is their own region: {(best == np.arange(n)[None, :]).mean():.1%}")
    lowm = per_mouse[per_mouse.r_same_median < 0.8]
    print(f"Mice with median r < 0.8: {len(lowm)}" + (" -> " + ", ".join(lowm.bids_id) if len(lowm) else ""))
    lowr = per_roi[(per_roi.r_same_median < 0.9) | (per_roi.best_is_same_frac < 0.5)]
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print(f"ROIs with median r < 0.9 or best match mostly elsewhere: {len(lowr)}")
        if len(lowr):
            print(lowr.sort_values("r_same_median")[["region", "label", "n_voxels", "r_same_median",
                  "best_is_same_frac", "best_label", "r_best_median"]].head(25).to_string(index=False))
            if len(lowr) > 25:
                print(f"  ... {len(lowr) - 25} more in roi_signal_check_by_roi.csv")
    if names and all(re.search(r"_[LR]$", s) for s in names):
        homol = {j: names.index(re.sub(r"_L$", "_R", s) if s.endswith("_L") else re.sub(r"_R$", "_L", s))
                 if (re.sub(r"_L$", "_R", s) if s.endswith("_L") else re.sub(r"_R$", "_L", s)) in names else -1
                 for j, s in enumerate(names)}
        h = np.mean([best[m, j] == homol[j] for m in range(len(ids)) for j in range(n) if homol[j] >= 0])
        print(f"Best match is the opposite-hemisphere homologue in {h:.1%} of cases "
              f"(near 100% = atlas mirrored left-right relative to the signal files)")
    print(f"Wrote {out}/roi_signal_check_by_roi.csv, roi_signal_check_by_mouse.csv, roi_signal_r_same.csv")


if __name__ == "__main__":
    main()
