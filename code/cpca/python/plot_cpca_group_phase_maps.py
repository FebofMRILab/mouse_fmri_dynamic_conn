#!/usr/bin/env python3
"""
plot_cpca_group_phase_maps.py
-------------------------------------------------------------------
Phase delay maps of the complex-PCA components, one column per group fit, set
side by side in the manner of Bolt et al. (2022), Supplementary Fig. 1 (phase
delay maps of the first three components in their independent sample, in
radians, on a cyclic colour map). Drawn on a dorsal glass view of the study's
CIVM brain surface (the BYU .g mesh used in BrainNet Viewer).

Nothing is recomputed or tested. The values are the loadings written by
cpca_group_lags.py (cpca_group_loadings.csv: group, n_mice, component, region,
label, magnitude, phase_rad, match_similarity), where each group was fitted
separately with Bolt's code and each group component was matched to the
all-mouse component by |cosine similarity| and turned by one common phase to
line up with it. Phases are therefore relative to the all-mouse fit, whose
largest ROI is at phase 0.

Each panel: node colour = phase (0 to 2*pi, cyclic); node area = |loading| after
scaling the component to unit length, so weak ROIs, whose phase is poorly
defined, are small. Under each panel: the traveling index (Bolt et al., 2022,
after Feeny, 2008; computed here from the same loadings, identical to
cpca_group_traveling_index.csv) and the match to the all-mouse component;
"weak match" marks |cosine| < MATCH_MIN, as the open circles of
Fig_cpca_patterns. Region names are in Fig_cpca_brain_maps and Fig_cpca_patterns.

The figure is laid out at its printed size (FIG_W = 7.2 in, double column), so
the font sizes are the printed sizes; the explanation of colours and symbols
belongs in the legend.

The script is self-contained (numpy, pandas, scipy, matplotlib).

Usage
    python plot_cpca_group_phase_maps.py \
        --loadings dfc_dcc_cpca/cpca_group_lags/cpca_group_loadings.csv \
        --mesh civm.g --node_template brainnet_node_coords_highres.csv \
        --out Fig_cpca_group_phase_maps.png
    Journal TIFF: --out Fig_cpca_group_phase_maps.tif --dpi 600
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy import ndimage

# Weak-match cutoff on |cosine| between a group component and the all-mouse
# component: 0.85, the lower bound of 'fair similarity' for Tucker's congruence
# coefficient (Lorenzo-Seva and ten Berge, 2006, Methodology 2:57-64).
MATCH_MIN = 0.85
SILHOUETTE = "#f3f2ef"
INK, INK2 = "#1f1e1c", "#52514e"
SIZE_KEY = (0.05, 0.10, 0.20)          # unit-norm |loading| values in the size key
# Print layout (inches): drawn at its final size for a double-column page, so
# the font sizes below are the printed sizes.
FIG_W = 7.2
LEFT, RIGHT = 0.24, 0.02         # margin for the row labels; right margin
GAP_ALL, GAP = 0.14, 0.02        # gap after the all-mouse column; between group columns
HEAD, NOTE, KEY = 0.34, 0.27, 1.0   # column headers; TI text under a row; key strip
# Cyclic phase colours: red at 0 and blue at pi, as in Fig_cpca_brain_maps
# (red = phase near 0, blue = phase near pi), with the hues in between marking
# phases off the 0 / pi axis.
PHASE_ANCHORS = ["#e34948", "#ec7f2d", "#c99a06", "#4a9e4f", "#2a78d6",
                 "#5a4fcf", "#9b3fb5", "#cc3f86", "#e34948"]
PHASE_CMAP = LinearSegmentedColormap.from_list("phase", PHASE_ANCHORS, N=512)


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
    """Dorsal glass view: summed area of triangles seen edge-on, on a silhouette."""
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


def read_coords(path):
    """BrainNet .node file (x y z colour size label) or a CSV with x, y, z and
    roi_name (sorted by roi_index if present), in atlas order."""
    if path.lower().endswith(".csv"):
        t = pd.read_csv(path)
        if "roi_index" in t.columns:
            t = t.sort_values("roi_index")
        name = next((c for c in ("roi_name", "label", "name") if c in t.columns), None)
        return t[["x", "y", "z"]].to_numpy(float), (t[name].astype(str).to_numpy()
                                                    if name else None)
    rows = [l.split() for l in open(path)
            if len(l.split()) >= 3 and not l.lstrip().startswith(("#", "%"))]
    xyz = np.array([[float(r[0]), float(r[1]), float(r[2])] for r in rows])
    lab = np.array([r[5] for r in rows]) if all(len(r) >= 6 for r in rows) else None
    return xyz, lab


def traveling_index(v):
    """Bolt et al. (2022) traveling index (Feeny, 2008): smaller over larger
    singular value of [Re(v), Im(v)] (ROI x 2)."""
    s = np.linalg.svd(np.column_stack([v.real, v.imag]), compute_uv=False)
    return s[1] / s[0]


def node_area(m):
    """Scatter area (pt^2) for a unit-norm |loading|, at print size."""
    return 1 + 80 * np.asarray(m, float)


def draw_brain(ax, glass):
    H, mask, (x0, x1, z0, z1) = glass
    ext = [x0, x1, z0, z1]
    ax.imshow(np.where(mask, 1.0, np.nan), extent=ext, origin="lower",
              cmap=LinearSegmentedColormap.from_list("m", [SILHOUETTE, SILHOUETTE]), zorder=0)
    ax.imshow(H, extent=ext, origin="lower", zorder=1,
              cmap=LinearSegmentedColormap.from_list("g", [(1, 1, 1, 0), (0.55, 0.55, 0.55, 0.9)]),
              vmax=np.percentile(H[H > 0], 99.3))
    ax.set_xlim(x0, x1)
    ax.set_ylim(z0, z1)
    ax.set_aspect("equal")
    ax.axis("off")


def phase_ring(ax):
    th = np.linspace(0, 2 * np.pi, 361)
    r = np.array([0.62, 1.0])
    T, R = np.meshgrid(th, r)
    ax.pcolormesh(T, R, np.tile(th, (2, 1)), cmap=PHASE_CMAP, vmin=0, vmax=2 * np.pi,
                  shading="gouraud")
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax.set_xticklabels(["0", "π/2", "π", "3π/2"], fontsize=6, color=INK)
    ax.tick_params(pad=-1)
    ax.spines["polar"].set_visible(False)
    ax.grid(False)
    ax.set_title("Phase (rad)", fontsize=6.5, color=INK, pad=4)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--loadings", required=True,
                    help="cpca_group_loadings.csv written by cpca_group_lags.py")
    ap.add_argument("--mesh", required=True, help="BYU surface (.g), e.g. civm.g")
    ap.add_argument("--node_template", required=True,
                    help="ROI coordinates: the study's BrainNet .node file or a CSV "
                         "with roi_name, x, y, z, in atlas order")
    ap.add_argument("--out", default="Fig_cpca_group_phase_maps.png",
                    help=".png, .pdf or .tif")
    ap.add_argument("--dpi", type=int, default=300, help="e.g. 600 for the journal TIFF")
    args = ap.parse_args()

    L = pd.read_csv(args.loadings)
    groups = list(dict.fromkeys(L.group))           # file order; ALL first
    groups = ["ALL"] + [g for g in groups if g != "ALL"]
    comps = sorted(L.component.unique())
    n = int(L.region.max())
    xyz, names = read_coords(args.node_template)
    if len(xyz) != n:
        raise SystemExit(f"{args.node_template}: {len(xyz)} ROIs, loadings have {n}")
    ref = L[(L.group == "ALL") & (L.component == comps[0])].sort_values("region")
    labels = ref.label.astype(str).to_numpy()
    if names is not None and (names == labels).all():
        print("Coordinates matched to the loadings by ROI name")
    elif names is not None and set(names) == set(labels):
        xyz = xyz[[list(names).index(l) for l in labels]]
        print("Coordinates reordered to the loadings by ROI name")
    else:
        print("Coordinates matched by row order (region 1..N); check they are in atlas order")
    X, Z = xyz[:, 0], xyz[:, 2]
    glass = glass_dorsal(*read_byu(args.mesh))

    ncol, nrow = len(groups), len(comps)
    H, _, (x0, x1, z0, z1) = glass
    cw = (FIG_W - LEFT - RIGHT - GAP_ALL - GAP * (ncol - 2)) / ncol      # column width
    bh = cw * (z1 - z0) / (x1 - x0)                                      # brain height
    fig_h = HEAD + nrow * (bh + NOTE) + KEY
    fig = plt.figure(figsize=(FIG_W, fig_h))

    def box(r, c):
        """Axes rectangle (figure fraction) of row r, column c."""
        x = LEFT + c * cw + (GAP_ALL + (c - 1) * GAP if c else 0)
        y = fig_h - HEAD - (r + 1) * bh - r * NOTE
        return [x / FIG_W, y / fig_h, cw / FIG_W, bh / fig_h]

    for r, k in enumerate(comps):
        for c, g in enumerate(groups):
            d = L[(L.group == g) & (L.component == k)].sort_values("region")
            if len(d) != n:
                raise SystemExit(f"{g} component {k}: {len(d)} ROIs, expected {n}")
            v = d.magnitude.to_numpy(float) * np.exp(1j * d.phase_rad.to_numpy(float))
            m = np.abs(v) / np.linalg.norm(v)
            ph = np.mod(np.angle(v), 2 * np.pi)
            ax = fig.add_axes(box(r, c))
            draw_brain(ax, glass)
            o = np.argsort(-m)                       # large nodes first, small on top
            ax.scatter(X[o], Z[o], s=node_area(m[o]), c=ph[o], cmap=PHASE_CMAP, vmin=0,
                       vmax=2 * np.pi, edgecolor="white", linewidth=0.2, zorder=3)
            sim = float(d.match_similarity.iloc[0])
            note = f"TI {traveling_index(v):.2f}"
            if g != "ALL":
                note += f"\nmatch {sim:.2f}" + ("*" if sim < MATCH_MIN else "")
            ax.text(0.5, -0.01, note, transform=ax.transAxes, ha="center", va="top",
                    fontsize=5.5, color=INK2, linespacing=1.15)
            if r == 0:
                nm = int(d.n_mice.iloc[0])
                ax.text(0.5, 1.0 + 0.04 / bh, f"{'All mice' if g == 'ALL' else g}\nn = {nm}",
                        transform=ax.transAxes, ha="center", va="bottom", fontsize=6.5,
                        color=INK, linespacing=1.15,
                        fontweight="bold" if g == "ALL" else "normal")
            if c == 0:
                ax.text(-0.04, 0.5, f"Component {k}", transform=ax.transAxes, rotation=90,
                        ha="right", va="center", fontsize=7, color=INK)
    # thin rule between the all-mouse column and the group fits
    xr = (LEFT + cw + GAP_ALL / 2) / FIG_W
    fig.add_artist(Line2D([xr, xr], [(KEY + 0.05) / fig_h, 1 - 0.03 / fig_h],
                          color="#d8d6d0", lw=0.6))

    # key strip: phase ring, size key, one line of reading notes
    ring_d = 0.42
    ring = fig.add_axes([(LEFT + 0.25) / FIG_W, 0.24 / fig_h, ring_d / FIG_W, ring_d / fig_h],
                        projection="polar")
    phase_ring(ring)
    handles = [Line2D([], [], ls="", marker="o", markersize=np.sqrt(node_area(s)),
                      markerfacecolor="#b9b7af", markeredgecolor="white",
                      markeredgewidth=0.2, label=f"{s:.2f}") for s in SIZE_KEY]
    fig.legend(handles=handles, ncol=len(SIZE_KEY), frameon=False, fontsize=6,
               title="|loading| (unit-length component)", title_fontsize=6,
               loc="center left", bbox_to_anchor=((LEFT + 1.25) / FIG_W, 0.48 / fig_h),
               handletextpad=0.3, columnspacing=1.0, borderaxespad=0)
    fig.text((LEFT + 3.2) / FIG_W, 0.48 / fig_h,
             "Dorsal view; anterior at top, left hemisphere on the left.\n"
             "TI = traveling index (0 standing, 1 traveling).\n"
             f"match = |cosine| with the all-mouse component; * below {MATCH_MIN:g}.",
             ha="left", va="center", fontsize=6, color=INK2, linespacing=1.4)
    kw = ({"pil_kwargs": {"compression": "tiff_lzw"}}
          if args.out.lower().endswith((".tif", ".tiff")) else {})
    fig.savefig(args.out, dpi=args.dpi, **kw)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
