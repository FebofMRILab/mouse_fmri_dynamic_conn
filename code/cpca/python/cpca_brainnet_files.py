#!/usr/bin/env python3
"""
cpca_brainnet_files.py
-------------------------------------------------------------------
Writes BrainNet Viewer node files (Xia et al., 2013, PLoS One 8:e68910) for the
complex-PCA components of the all-mouse fit (Bolt et al., 2022), so they can be
drawn on the same glass mouse brain as the study's connectome figure.
Nothing is recomputed: the values are the loadings written by
run_dcc_cpca_null.m (cpca_loadings_roi.csv: region, component, magnitude,
phase_rad, lag_sec; unit-norm complex loadings with the largest-magnitude ROI
at phase 0), which match those of Bolt's published code.

Coordinates and ROI names come from --node_template: the study's BrainNet .node
file (x y z colour size label) or a CSV with roi_name, x, y, z
(brainnet_node_coords_highres.csv), in atlas order (region 1..N; a roi_index
column, if present, sets the order).

Outputs (in --out_dir), for each component k:
  cpca_C{k}.node   colour 1 = strong ROI (|loading| >= the component's median)
                   with phase near 0, 2 = strong ROI with phase near pi (peaks
                   half a cycle later), 3 = weak ROI; size = |loading| x
                   COMP_SIZE; the four largest ROIs of each sign are labelled.
Colours and the strong/weak split are for display only; no test is involved.

BrainNet Viewer settings: load the mouse surface used for the connectome
figure, then the .node file. Node > Colour: Modular (1 red, 2 blue, 3 light
grey); Node > Size: Value.

The exploratory group-contrast maps of the dual-regression analysis (not in the
paper) are written by cpca_brainnet_files_dualreg.py.

Usage
    python cpca_brainnet_files.py --loadings dfc_dcc_cpca/cpca_loadings_roi.csv \
        --node_template brainnet_node_coords_highres.csv --out_dir dfc_dcc_cpca/cpca_brainnet
"""

import argparse
import os

import numpy as np
import pandas as pd

N_LABEL = 4
COMP_SIZE = 30.0     # node size per unit |loading| (unit-norm loadings)


def read_node(path):
    """BrainNet .node file (x y z colour size label), or a CSV with x, y, z and an
    ROI-name column (roi_name, label or name), in atlas order."""
    if path.lower().endswith(".csv"):
        t = pd.read_csv(path)
        if "roi_index" in t.columns:
            t = t.sort_values("roi_index")
        name = next((c for c in ("roi_name", "label", "name") if c in t.columns), None)
        return pd.DataFrame(dict(x=t.x.astype(float).to_numpy(), y=t.y.astype(float).to_numpy(),
                                 z=t.z.astype(float).to_numpy(),
                                 label=t[name].astype(str).to_numpy() if name else None))
    rows = []
    for line in open(path):
        p = line.split()
        if len(p) >= 3 and not line.lstrip().startswith(("#", "%")):
            rows.append(dict(x=float(p[0]), y=float(p[1]), z=float(p[2]),
                             label=p[5] if len(p) >= 6 else None))
    return pd.DataFrame(rows)


def write_node(path, xyz, colour, size, labels):
    with open(path, "w") as f:
        for (x, y, z), c, s, lab in zip(xyz, colour, size, labels):
            f.write(f"{x:.4f}\t{y:.4f}\t{z:.4f}\t{int(c)}\t{s:.4f}\t{lab}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--loadings", required=True,
                    help="cpca_loadings_roi.csv written by run_dcc_cpca_null.m")
    ap.add_argument("--node_template", required=True,
                    help="the study's BrainNet .node file (x y z colour size label), "
                         "or a CSV with roi_name, x, y, z")
    ap.add_argument("--out_dir", default=os.path.join("dfc_dcc_cpca", "cpca_brainnet"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    L = pd.read_csv(args.loadings)
    n = int(L.region.max())
    T = read_node(args.node_template)
    if len(T) != n:
        raise SystemExit(f"{args.node_template}: {len(T)} nodes, loadings have {n} ROIs")
    xyz = T[["x", "y", "z"]].to_numpy()
    names = (T.label.to_numpy() if T.label.notna().all()
             else np.array([f"ROI{r}" for r in range(1, n + 1)], dtype=object))
    print("ROIs matched by atlas order (region 1..N = template rows)")

    for k in sorted(L.component.unique()):
        a = L[L.component == k].sort_values("region")
        if not (a.region.to_numpy() == np.arange(1, n + 1)).all():
            raise SystemExit(f"component {k}: regions are not 1..{n}")
        mag, ph = a.magnitude.to_numpy(float), a.phase_rad.to_numpy(float)
        pole = np.where(np.cos(ph) >= 0, 1, 2)
        strong = mag >= np.median(mag)
        colour = np.where(strong, pole, 3)
        signed = np.where(pole == 1, 1.0, -1.0) * mag
        lab = np.array(["-"] * n, dtype=object)
        for idx in list(np.argsort(-signed)[:N_LABEL]) + list(np.argsort(signed)[:N_LABEL]):
            if strong[idx]:
                lab[idx] = names[idx]
        write_node(os.path.join(args.out_dir, f"cpca_C{k}.node"), xyz, colour,
                   COMP_SIZE * mag, lab)
        dev = np.degrees(np.abs(np.angle(np.exp(2j * ph[strong]))) / 2)
        print(f"  C{k}: {int((colour == 1).sum())} + / {int((colour == 2).sum())} - strong ROIs; "
              f"strong ROIs within {dev.max():.1f} deg of 0 or pi")
    print(f"Done. Files in {args.out_dir}")


if __name__ == "__main__":
    main()
