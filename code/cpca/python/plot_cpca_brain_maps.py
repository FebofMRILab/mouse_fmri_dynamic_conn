#!/usr/bin/env python3
"""
plot_cpca_brain_maps.py
-------------------------------------------------------------------
Dorsal glass-brain maps of the complex-PCA results, drawn on the study's CIVM
brain surface (the BYU .g mesh exported from ITK-SNAP and used in BrainNet
Viewer), from the .node files written by cpca_brainnet_files.py (component maps,
all-mouse loadings) and, if present, cpca_brainnet_files_dualreg.py (exploratory
group-contrast maps, not in the paper).
A matplotlib alternative to rendering the same files in BrainNet Viewer.

View: dorsal, orthographic projection on x (left-right) and z (anterior-
posterior); anterior at the top, left hemisphere on the left. The surface is
drawn as a glass brain: shading is the summed area of mesh triangles seen
edge-on (outline and internal boundaries), on a light silhouette.

Nodes: colour 1 red, 2 blue, 3 grey (see cpca_brainnet_files.py for what the
colours mean in each file). Node area grows with |loading| (component maps) or
with the group difference in |loading| (contrast maps); a size key under each
set of panels gives the scale.

Usage
    python plot_cpca_brain_maps.py --mesh civm.g --bn_dir dfc_dcc_cpca/cpca_brainnet \
        --out Fig_cpca_brain_maps.png
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy import ndimage

COLOURS = {1: "#e34948", 2: "#2a78d6", 3: "#b9b7af"}
# Node-size scales used by cpca_brainnet_files.py and cpca_brainnet_files_dualreg.py
# (keep them in step):
COMP_SIZE = 30.0     # node size per unit |loading| (component maps, unit-norm loadings)
MAG_SIZE = 30.0      # node size per unit |difference in |loading|| (contrast maps)
WEAK_SIZE = 0.5      # node size of weak, untested ROIs (contrast maps)
SILHOUETTE = "#f3f2ef"
COMP_KEY = (0.05, 0.10, 0.20)       # |loading| values shown in the size key
MAG_KEY = (0.05, 0.10, 0.15)        # |difference in |loading|| values shown


def read_byu(path):
    """Movie.BYU mesh: header (parts, vertices, polygons, edges), part line,
    vertices (x y z), then 1-based polygon indices with the last one negative."""
    with open(path) as f:
        h = f.readline().split()
        f.readline()
        nv, nt = int(h[1]), int(h[2])
        tok = np.array(f.read().split())
    v = tok[:nv * 3].astype(float).reshape(nv, 3)
    tri = np.abs(tok[nv * 3:nv * 3 + nt * 3].astype(int).reshape(nt, 3)) - 1
    return v, tri


def glass_dorsal(v, tri, res=0.025, pad=0.6):
    a, b, c = v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]
    n = np.cross(b - a, c - a)
    area = np.linalg.norm(n, axis=1) / 2
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    cen = (a + b + c) / 3
    x0, x1 = v[:, 0].min() - pad, v[:, 0].max() + pad
    z0, z1 = v[:, 2].min() - pad, v[:, 2].max() + pad
    bx, bz = np.arange(x0, x1 + res, res), np.arange(z0, z1 + res, res)
    rim = area * (1 - np.abs(n[:, 1])) ** 3
    H, _, _ = np.histogram2d(cen[:, 0], cen[:, 2], bins=[bx, bz], weights=rim)
    occ, _, _ = np.histogram2d(v[:, 0], v[:, 2], bins=[bx, bz])
    H = ndimage.gaussian_filter(H, 1.2)
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(occ > 0, iterations=3))
    return H.T, mask.T, (x0, x1, z0, z1)


def read_node(path):
    rows = [l.rstrip("\n").split("\t") for l in open(path) if l.strip()]
    xyz = np.array([[float(r[0]), float(r[1]), float(r[2])] for r in rows])
    return xyz, np.array([int(r[3]) for r in rows]), np.array([float(r[4]) for r in rows]), \
        [r[5] for r in rows]


def marker_area(size):
    """Scatter area (pt^2) for a node-file size value."""
    return 4 + 12 * np.asarray(size, float) ** 1.15


def draw(ax, glass, stem, title):
    H, mask, (x0, x1, z0, z1) = glass
    ext = [x0, x1, z0, z1]
    ax.imshow(np.where(mask, 1.0, np.nan), extent=ext, origin="lower",
              cmap=LinearSegmentedColormap.from_list("m", [SILHOUETTE, SILHOUETTE]), zorder=0)
    ax.imshow(H, extent=ext, origin="lower", zorder=1,
              cmap=LinearSegmentedColormap.from_list("g", [(1, 1, 1, 0), (0.55, 0.55, 0.55, 0.9)]),
              vmax=np.percentile(H[H > 0], 99.3))
    xyz, col, size, lab = read_node(stem + ".node")
    X, Z = xyz[:, 0], xyz[:, 2]
    o = np.argsort(size)
    ax.scatter(X[o], Z[o], s=marker_area(size[o]), c=[COLOURS[k] for k in col[o]],
               edgecolor="white", linewidth=0.4, zorder=3)
    for i, l in enumerate(lab):
        if l != "-":
            ax.annotate(l, (X[i], Z[i]), xytext=(3, 2), textcoords="offset points", fontsize=6,
                        zorder=4)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(x0, x1)
    ax.set_ylim(z0, z1)
    ax.set_aspect("equal")
    ax.axis("off")


def dot(colour, size, label):
    return Line2D([], [], ls="", marker="o", markersize=np.sqrt(marker_area(size)),
                  markerfacecolor=colour, markeredgecolor="white", markeredgewidth=0.4,
                  label=label)


def add_key(fig, axes, top, colour_items, size_items, size_title):
    """Colour key and size key, stacked, centred under the given panels."""
    boxes = [ax.get_position() for ax in axes]
    xc = (min(b.x0 for b in boxes) + max(b.x1 for b in boxes)) / 2
    kw = dict(loc="upper center", frameon=False, fontsize=6.5, handletextpad=0.3,
              columnspacing=1.2, borderaxespad=0)
    leg = fig.legend(handles=[dot(c, 2.5, l) for c, l in colour_items],
                     ncol=len(colour_items), bbox_to_anchor=(xc, top), **kw)
    fig.canvas.draw()
    bb = leg.get_window_extent().transformed(fig.transFigure.inverted())
    fig.legend(handles=[dot(COLOURS[3], s, l) for s, l in size_items], ncol=len(size_items),
               bbox_to_anchor=(xc, bb.y0 - 0.015), title=size_title, title_fontsize=6.5,
               alignment="center", **kw)
    fig.add_artist(leg)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--mesh", required=True, help="BYU surface (.g), e.g. civm.g")
    ap.add_argument("--bn_dir", default=os.path.join("dfc_dcc_cpca", "cpca_brainnet"))
    ap.add_argument("--out", default="Fig_cpca_brain_maps.png", help=".png, .pdf or .tif")
    ap.add_argument("--dpi", type=int, default=300, help="e.g. 600 for the journal TIFF")
    args = ap.parse_args()
    glass = glass_dorsal(*read_byu(args.mesh))
    stems = sorted(f[:-5] for f in os.listdir(args.bn_dir) if f.endswith(".node"))
    comps = [s for s in stems if s.startswith("cpca_C")]
    mags = [s for s in stems if s.startswith("cpca_mag_")]
    panels = [(s, f"Component {s[6:]}") for s in comps]
    for s in mags:
        con, k = s[len("cpca_mag_"):].rsplit("_C", 1)
        panels.append((s, con.replace("_vs_", " vs ") + f"\ncomponent {k}"))
    fig, axes = plt.subplots(1, len(panels), figsize=(2.4 * len(panels), 4.6), squeeze=False)
    axes = axes[0]
    for ax, (s, t) in zip(axes, panels):
        draw(ax, glass, os.path.join(args.bn_dir, s), t)
    note = ("Dorsal view, anterior at top, left hemisphere on the left. Components (all-mouse "
            "fit): red and blue = strong ROIs at the two poles (peak together within a pole, "
            "half a cycle apart between poles); grey = weak ROIs (|loading| below the median).")
    if mags:
        note += " Contrasts: red or blue = ROI passes the per-ROI FWE test."
    fig.text(0.5, 0.01, note, ha="center", fontsize=6.5, color="#52514e", wrap=True)
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    fig.canvas.draw()
    top = min(ax.get_position().y0 for ax in axes) - 0.01
    if comps:
        add_key(fig, axes[:len(comps)], top,
                [(COLOURS[1], "phase ≈ 0"), (COLOURS[2], "phase ≈ π"),
                 (COLOURS[3], "weak ROI")],
                [(COMP_SIZE * v, f"{v:.2f}") for v in COMP_KEY], "|loading|")
    if mags:
        add_key(fig, axes[len(comps):], top,
                [(COLOURS[1], "larger in first group"), (COLOURS[2], "larger in second group"),
                 (COLOURS[3], "not significant")],
                [(MAG_SIZE * v, f"{v:.2f}") for v in MAG_KEY]
                + [(WEAK_SIZE, "weak ROI, not tested")],
                "difference in |loading|")
    kw = ({"pil_kwargs": {"compression": "tiff_lzw"}}
          if args.out.lower().endswith((".tif", ".tiff")) else {})
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", **kw)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
