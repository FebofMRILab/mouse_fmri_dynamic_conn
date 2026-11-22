#!/usr/bin/env python3
"""
check_parcellation.py  -- is BrainNetCoord.nii the same parcellation as the seed masks?

BrainNetCoord.nii is built (civm roi_extraction.bash) by summing the 82 masks
ROIs_Lo/roi<i>.nii.gz with weight i and keeping values <= 82. The masks are
hole-filled (fslmaths -fillh), so they can overlap; a voxel in masks i and j
then gets label i + j (another region if <= 82, removed if > 82). This
script measures that, region by region, and writes a names table from the
curated list (regions.txt order = region index 1..82).

Status: NEW lab diagnostic; reads files only.

Per region i
  n_mask             voxels in roi<i>
  n_atlas            voxels labelled i in BrainNetCoord
  dice               overlap of the two
  mask_relabelled    voxels of roi<i> that carry another nonzero label
  mask_dropped       voxels of roi<i> labelled 0 (sum > 82)
  atlas_foreign      voxels labelled i that are not in roi<i> (sums of other masks)
Global: voxels covered by 2 or more masks, and whether their label equals the
sum of the covering indices (the summing artefact).

Usage
    python check_parcellation.py --atlas /blue/febo/share/civm_atlas/BrainNetCoord.nii \
        --seeds seeds_list.txt --regions regions.txt \
        [--ontology RCCF_Ontology_2022-08-19.xls] --out_dir parcellation_check
"""
import argparse
import os

import numpy as np
import pandas as pd
import nibabel as nib


def curated_names(ontology, regions):
    d = pd.read_excel(ontology, sheet_name="July2026_ROIcurationBySize", header=None)
    cur = d.iloc[1:, 15:19].dropna(subset=[15])
    cur.columns = ["code", "structure", "acronym", "region"]
    cur = cur.astype({"code": int, "region": int})
    if regions is not None and cur.code.tolist() != regions:
        raise SystemExit("curated list order differs from regions.txt")
    cur["label"] = cur.acronym.astype(str) + np.where(cur.code >= 1000, "_R", "_L")
    return cur[["region", "label", "code", "acronym", "structure"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--seeds", required=True, help="text file with the 82 mask paths in region order")
    ap.add_argument("--regions", default=None, help="regions.txt (original codes, region order)")
    ap.add_argument("--ontology", default=None, help="RCCF ontology .xls with the curated list")
    ap.add_argument("--out_dir", default="parcellation_check")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    img = nib.load(a.atlas)
    A = np.rint(np.asarray(img.dataobj)).astype(np.int64)
    paths = [p.strip() for p in open(a.seeds) if p.strip()]
    n = len(paths)
    cover = np.zeros(A.shape, np.int16)
    idx_sum = np.zeros(A.shape, np.int32)
    masks = []
    for i, p in enumerate(paths, start=1):
        mi = nib.load(p)
        if mi.shape[:3] != A.shape[:3] or not np.allclose(mi.affine, img.affine, atol=1e-4):
            raise SystemExit(f"{p}: grid differs from the atlas")
        M = np.asarray(mi.dataobj) > 0
        masks.append(M)
        cover += M
        idx_sum += M * i
    rows = []
    for i, M in enumerate(masks, start=1):
        L = A == i
        rows.append(dict(region=i, n_mask=int(M.sum()), n_atlas=int(L.sum()),
                         dice=2 * (M & L).sum() / max(M.sum() + L.sum(), 1),
                         mask_relabelled=int((M & (A != i) & (A != 0)).sum()),
                         mask_dropped=int((M & (A == 0)).sum()),
                         atlas_foreign=int((L & ~M).sum()),
                         overlaps_with=",".join(str(j) for j, Mj in enumerate(masks, start=1)
                                                if j != i and (M & Mj).any())))
    R = pd.DataFrame(rows)
    regions = [int(x) for x in open(a.regions).read().split()] if a.regions else None
    if a.ontology:
        names = curated_names(a.ontology, regions)
        R = names.merge(R, on="region")
        names.assign(atlas_value=names.region).to_csv(os.path.join(a.out_dir, "roi_labels.csv"), index=False)
    elif regions:
        R.insert(1, "code", regions)
    R.to_csv(os.path.join(a.out_dir, "parcellation_check.csv"), index=False)

    multi = cover >= 2
    expect = np.where(idx_sum <= n, idx_sum, 0)
    print(f"{n} masks; atlas labels present: {len(np.unique(A[A > 0]))}")
    print(f"voxels in 2+ masks: {int(multi.sum())} "
          f"({multi.sum() / max((cover > 0).sum(), 1):.2%} of masked voxels); "
          f"of these, label = sum of indices: {np.mean(A[multi] == expect[multi]) if multi.any() else float('nan'):.1%}")
    bad = R[(R.dice < 0.98) | (R.atlas_foreign > 0) | (R.mask_relabelled > 0) | (R.mask_dropped > 0)]
    print(f"regions where BrainNetCoord differs from the mask: {len(bad)}")
    cols = [c for c in ["region", "label", "n_mask", "n_atlas", "dice", "mask_relabelled", "mask_dropped",
                        "atlas_foreign", "overlaps_with"] if c in R]
    with pd.option_context("display.width", 180, "display.float_format", "{:.3f}".format):
        if len(bad):
            print(bad.sort_values("dice")[cols].to_string(index=False))
    print(f"Wrote {a.out_dir}/parcellation_check.csv" + (", roi_labels.csv" if a.ontology else ""))


if __name__ == "__main__":
    main()
