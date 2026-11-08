"""
BrainNet Viewer node files (.node) for the static-FC maps (28 Oct 2026).

Each .node file sits next to the .edge file it belongs to, so the 3D map
(edges + spheres on the mouse shell) loads as one pair in BrainNet Viewer.

Format: one row per ROI, tab-delimited, no header,
    x  y  z  colour  size  label
as BrainNet Viewer reads it (BrainNet_LoadFiles.m and BrainNet_MapCfg.m:
textscan(fid, '%f %f %f %f %f %s')). BrainNet draws no label for '-'.
  x, y, z  ROI coordinates on the mouse shell: the lab's high-resolution CIVM
           atlas coordinates (brainnet_node_coords_highres.csv, taken from the
           lab's .node files), in the order of the FC matrices
  size     node strength of the network in the matching .edge file
  colour   the strength rounded to the nearest integer, as in the lab's
           earlier .node files (MATLAB round: halves away from zero)
  label    '-'

Coordinates file: BRAINNET_NODE_COORDS=<csv> (columns roi_name, x, y, z) points
to another one, e.g. full-precision centroids of the same atlas.
"""
import os

import numpy as np
import pandas as pd

DEFAULT_COORDS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "brainnet_node_coords_highres.csv")


def load_node_coordinates(roi_names, path=None):
    """Coordinates (n x 3) for roi_names, in their order. The coordinates file
    must list the same ROI names in the same order (case and surrounding spaces
    ignored); otherwise a ValueError lists the first differences."""
    path = path or os.environ.get("BRAINNET_NODE_COORDS") or DEFAULT_COORDS
    table = pd.read_csv(path)
    missing = {"roi_name", "x", "y", "z"}.difference(table.columns)
    if missing:
        raise ValueError(f"{path}: missing column(s) {', '.join(sorted(missing))}.")
    names = [str(n).strip() for n in table["roi_name"]]
    want = [str(n).strip() for n in roi_names]
    if len(names) != len(want):
        raise ValueError(f"{path} has {len(names)} ROIs; the FC matrices have {len(want)}.")
    diff = [(i + 1, a, b) for i, (a, b) in enumerate(zip(want, names)) if a.lower() != b.lower()]
    if diff:
        shown = "; ".join(f"ROI {i}: matrices '{a}' vs coordinates '{b}'" for i, a, b in diff[:5])
        raise ValueError(f"ROI names differ between roi_names_order.csv and {path} "
                         f"({len(diff)} of {len(want)}): {shown}. Fix the order or the names "
                         "before writing .node files.")
    xyz = table[["x", "y", "z"]].to_numpy(dtype=float)
    if not np.isfinite(xyz).all():
        raise ValueError(f"{path}: non-numeric or missing coordinates.")
    return xyz, path


def matlab_round(values):
    """MATLAB's round: to the nearest integer, halves away from zero
    (numpy.round sends halves to the even integer)."""
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.floor(np.abs(values) + 0.5)


def write_brainnet_node(path, strength, xyz):
    """Write one BrainNet .node file: coordinates, colour = rounded strength,
    size = strength, label '-'. NaN strength is written as 0."""
    strength = np.nan_to_num(np.asarray(strength, dtype=float), nan=0.0)
    xyz = np.asarray(xyz, dtype=float)
    if strength.shape != (xyz.shape[0],):
        raise ValueError(f"{len(strength)} strengths for {xyz.shape[0]} nodes.")
    colour = matlab_round(strength).astype(int)
    with open(path, "w", encoding="ascii") as handle:
        for (x, y, z), c, s in zip(xyz, colour, strength):
            handle.write(f"{x:.6g}\t{y:.6g}\t{z:.6g}\t{c:d}\t{s:.6g}\t-\n")
    return path
