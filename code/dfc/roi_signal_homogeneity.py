#!/usr/bin/env python3
"""
roi_signal_homogeneity.py  -- how well one time series represents each ROI (Supplement)

For every mouse and ROI, from RABIES's confound-corrected commonspace time
series and the atlas on the same grid (slice_maps/atlas_commonspace.nii.gz:
BrainNetCoord.nii, region j = atlas value j):

  r2_mean     share of the ROI's voxel variance explained by the ROI mean time
              series, i.e. the signal used in every analysis (D29): the mean
              over voxels of r(voxel, ROI mean)^2. Voxels are standardised, so
              each voxel counts once whatever its variance.
  r2_pc1      the most that ANY single time series can explain: the largest
              eigenvalue of the voxel correlation matrix divided by the number
              of voxels (first principal component; Eckart & Young 1936,
              Jolliffe 2002). r2_mean <= r2_pc1 always; the gap is what using
              the mean costs relative to the best single time series.
  r2_pc2      the same for the second component (structure the first leaves).
  mean_r      mean correlation over all voxel pairs: resting-state
              homogeneity as defined by Schaefer et al. 2018.
  r_mean_eig  correlation of the ROI mean with the first eigenvariate computed
              as FSL fslmeants --eig does (fslmeants.cc: first eigenvector of
              the voxel-demeaned time x time covariance, scaled by sqrt(T),
              sign set to agree with the mean); the E47 comparison, all mice.
  n_vox       atlas voxels; n_vox_used: voxels with non-zero variance (voxels
              outside RABIES's brain mask are constant and are left out of the
              voxel measures; the ROI mean uses all atlas voxels, as RABIES and
              fslmeants do, which changes only its scale).

Per mouse, the measures are averaged over ROIs weighted by n_vox_used (the
share of all ROI voxel variance explained, as Schaefer et al. weight by parcel
size). Group tests (study's Welch code, welch_tests_scipy.py): Welch's F with a
permutation null for the per-mouse r2_mean and for each ROI's r2_mean (BH over
the 82 ROIs), and the study's 8 planned contrasts (Welch t, BH over the 8).

Optional checks on real data
  --eig_dir      FSL's own eigenvariates (e.g. hmm_signals_eig/signal<N>.txt):
                 this script's eigenvariate must correlate 1.000 with them.
  --signals_dir  the ROI-mean signal files (hmm_signals after step 0b): this
                 script's ROI means must correlate 1.000 with them.

Status: NEW lab diagnostic (descriptive measures from standard linear algebra;
group tests with the study's Welch functions).

Usage (in signals/)
    python roi_signal_homogeneity.py --meta meta_data_with_bids.csv \
        --atlas_cs slice_maps/atlas_commonspace.nii.gz \
        --cleaned_glob "../confound_correction_outputs/confound_correction_datasink/cleaned_timeseries/**/*sub-{id}_*.nii*" \
        --roi_labels roi_labels.csv --eig_dir hmm_signals_eig --signals_dir hmm_signals \
        --out_dir roi_homogeneity
"""
import argparse
import glob
import logging
import os
import re
import warnings

import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats

warnings.filterwarnings("ignore", message=".*findfont.*")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # Arial missing -> DejaVu Sans

GROUP_ORDER_PREF = ["NTG_Y", "NTG_M", "NTG_A", "CRND8_Y", "CRND8_M", "5XFAD_Y", "5XFAD_M"]
GROUP_COLORS = {"NTG_Y": "#e06666", "NTG_M": "#a73dd9", "NTG_A": "#a64d79", "CRND8_Y": "#3d85c6",
                "CRND8_M": "#38761d", "5XFAD_Y": "#8fce00", "5XFAD_M": "#d4b75d"}   # Figure 6
FALLBACK_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
PAIRWISE_COMPARISONS = [("NTG_Y", "NTG_M"), ("NTG_Y", "NTG_A"), ("NTG_Y", "CRND8_Y"), ("NTG_M", "CRND8_M"),
                        ("NTG_Y", "5XFAD_Y"), ("NTG_M", "5XFAD_M"), ("CRND8_Y", "CRND8_M"),
                        ("5XFAD_Y", "5XFAD_M")]                                    # as plot_dfc_results.py
RANDOM_SEED = 20260920
INK, SECONDARY, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
C_MEAN, C_PC1 = "#2a78d6", "#eb6834"
MEASURES = ["r2_mean", "r2_pc1", "r2_pc2", "mean_r", "r_mean_eig"]


def one(pattern):
    hits = sorted(set(glob.glob(pattern, recursive=True)))
    if len(hits) != 1:
        raise FileNotFoundError(f"{len(hits)} files match {pattern}")
    return hits[0]


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else np.nan


def fsl_eigenvariate(X):
    """First eigenvariate as fslmeants --eig (FSL avwutils fslmeants.cc): voxel
    means removed, first eigenvector of the time x time covariance, times
    sqrt(T), sign flipped if the voxel-mean of the demeaned data projects
    negatively on it."""
    Xd = X - X.mean(0)
    U, _, _ = np.linalg.svd(Xd, full_matrices=False)
    e = U[:, 0] * np.sqrt(X.shape[0])
    if Xd.mean(1) @ e < 0:
        e = -e
    return e


def roi_measures(X, m):
    """X: time x voxels (all atlas voxels of one ROI), m: ROI mean time series."""
    T = X.shape[0]
    sd = X.std(0)
    use = sd > 0
    n = int(use.sum())
    out = dict(n_vox=X.shape[1], n_vox_used=n)
    if n < 2 or m.std() == 0:
        return out | {k: np.nan for k in MEASURES}
    Z = (X[:, use] - X[:, use].mean(0)) / sd[use]
    mz = (m - m.mean()) / m.std()
    r = Z.T @ mz / T                                   # r(voxel, ROI mean)
    lam = np.linalg.svd(Z, compute_uv=False) ** 2 / T  # eigenvalues of the voxel correlation matrix
    zs = Z.sum(1)
    out.update(r2_mean=float(np.mean(r ** 2)), r2_pc1=float(lam[0] / n),
               r2_pc2=float(lam[1] / n), mean_r=float(((zs @ zs) / T - n) / (n * (n - 1))))
    out["eig"] = fsl_eigenvariate(X)
    out["r_mean_eig"] = corr(m, out["eig"])
    return out


def read_meta(path):
    m = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    m.columns = [re.sub(r"[^a-z0-9_]", "", c.strip().lower()) for c in m.columns]
    for c in ("subject_id", "group", "bids_id"):
        if c not in m:
            raise SystemExit(f"{path}: column {c} missing")
    m["bids_id"] = m.bids_id.str.replace(r"^sub-", "", regex=True).str.strip()
    m["group"] = m.group.str.strip()
    m["subject_id"] = m.subject_id.str.strip()
    return m


def group_order(groups):
    present = list(dict.fromkeys(groups))
    pref = [g for g in GROUP_ORDER_PREF if g in present]
    return pref + sorted(g for g in present if g not in pref)


def palette_for(groups):
    extra = iter(FALLBACK_COLORS)
    return {g: GROUP_COLORS.get(g) or next(extra) for g in groups}


def bh(p):
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        q[ok] = stats.false_discovery_control(p[ok], method="bh")
    return q


def group_tests(values, groups, order, n_perm):
    """Welch omnibus (permutation p, Welch df) and the planned contrasts."""
    from welch_tests_scipy import welch_anova_permutation, welch_anova_df, welch_t_test_df
    samples = [values[groups == g] for g in order]
    F, p = welch_anova_permutation(samples, n_permutations=n_perm, random_seed=RANDOM_SEED)
    df1, df2 = welch_anova_df(samples)
    rows = [dict(test="omnibus (Welch F, permutation p)", groups=" / ".join(order), statistic=F,
                 df1=df1, df2=df2, p=p, q=np.nan)]
    con = []
    for a, b in PAIRWISE_COMPARISONS:
        if a in order and b in order:
            t, df, pp = welch_t_test_df(values[groups == a], values[groups == b])
            con.append(dict(test="contrast (Welch t)", groups=f"{a} - {b}", statistic=t, df1=np.nan,
                            df2=df, p=pp))
    for r, q in zip(con, bh([r["p"] for r in con])):
        r["q"] = q
    return pd.DataFrame(rows + con)


def make_figure(byroi, mouse, mr, order, pal, out_png, rho, omni):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.linewidth": 0.8,
                         "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
                         "savefig.facecolor": "white", "figure.facecolor": "white",
                         "axes.edgecolor": SECONDARY, "xtick.color": SECONDARY, "ytick.color": SECONDARY,
                         "axes.labelcolor": INK, "text.color": INK})
    b = byroi.sort_values("r2_mean_median").reset_index(drop=True)
    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.15], wspace=0.42, hspace=0.45)
    axA = fig.add_subplot(gs[:, 0])
    y = np.arange(len(b))
    # the mean is drawn first and the ceiling as a larger ring on top, so the ring stays visible
    # around the dot where the two coincide
    for col, c, lab, face, size, z in [("r2_mean", C_MEAN, "ROI mean (used in all analyses)", C_MEAN, 20, 3),
                                       ("r2_pc1", C_PC1, "First principal component (ceiling)", "none", 46, 4)]:
        axA.hlines(y, b[f"{col}_q1"], b[f"{col}_q3"], color=c, lw=1.6, alpha=0.55)
        axA.scatter(b[f"{col}_median"], y, s=size, facecolor=face, edgecolor=c, lw=1.1, zorder=z, label=lab)
    axA.set_yticks(y)
    axA.set_yticklabels(b.label, fontsize=6.5)
    axA.set_ylim(-1, len(b))
    axA.set_xlim(0, 1)
    axA.grid(axis="x", color=GRID, lw=0.6)
    axA.set_axisbelow(True)
    axA.set_xlabel("Share of voxel variance explained (median, IQR over mice)")
    axA.legend(loc="upper left", frameon=False, fontsize=7.5)   # ROIs sorted ascending: upper left stays empty
    axA.set_title("A  One time series per ROI", loc="left", fontweight="bold", fontsize=10)

    axB = fig.add_subplot(gs[0, 1])
    axB.scatter(byroi.n_vox_used_median, byroi.r2_mean_median, s=18, color=C_MEAN, edgecolor="white", lw=0.4,
                label="ROI mean")
    axB.scatter(byroi.n_vox_used_median, byroi.r2_pc1_median, s=34, facecolor="none", edgecolor=C_PC1, lw=1.0,
                label="First principal component")
    axB.set_xscale("log")
    axB.set_ylim(0, 1)
    axB.grid(color=GRID, lw=0.6)
    axB.set_axisbelow(True)
    axB.set_xlabel("ROI size (voxels, log scale)")
    axB.set_ylabel("Share of voxel variance explained")
    axB.legend(frameon=False, fontsize=7.5, loc="lower left")
    axB.set_title(f"B  Size dependence (ROI mean: Spearman ρ = {rho:.2f})", loc="left",
                  fontweight="bold", fontsize=10)

    axC = fig.add_subplot(gs[1, 1])
    rng = np.random.default_rng(RANDOM_SEED)
    for i, g in enumerate(order):
        v = mouse.loc[mouse.group == g, "r2_mean_w"].to_numpy(float)
        v = v[np.isfinite(v)]
        if v.size:
            axC.boxplot(v, positions=[i], widths=0.56, showfliers=False, patch_artist=True,
                        medianprops=dict(color=INK, lw=1.1),
                        boxprops=dict(facecolor="white", edgecolor=SECONDARY, lw=0.8),
                        whiskerprops=dict(color=SECONDARY, lw=0.8), capprops=dict(color=SECONDARY, lw=0.8))
            axC.scatter(i + rng.uniform(-0.14, 0.14, v.size), v, s=20, facecolor=pal[g], edgecolor="white",
                        lw=0.35, alpha=0.85, zorder=3)
    axC.set_xticks(range(len(order)))
    axC.set_xticklabels(order, rotation=45, ha="right", rotation_mode="anchor")
    axC.set_ylabel("Share explained by ROI means\n(voxel-weighted over ROIs)")
    axC.grid(axis="y", color=GRID, lw=0.6)
    axC.set_axisbelow(True)
    ptxt = "< 0.001" if omni.p < 0.001 else f"= {omni.p:.3f}"
    axC.set_title(f"C  By group (Welch F = {omni.statistic:.2f}, permutation p {ptxt})",
                  loc="left", fontweight="bold", fontsize=10)

    axD = fig.add_subplot(gs[2, 1])
    axD.scatter(mr.n_vox_used, mr.r_mean_eig, s=5, color=MUTED, alpha=0.35, lw=0, label="mouse × ROI")
    axD.scatter(byroi.n_vox_used_median, byroi.r_mean_eig_median, s=18, color=INK, edgecolor="white", lw=0.4,
                label="ROI median")
    axD.set_xscale("log")
    axD.set_ylim(min(-0.05, float(np.nanmin(mr.r_mean_eig)) - 0.05), 1.02)
    axD.axhline(0, color=MUTED, lw=0.6)
    axD.grid(color=GRID, lw=0.6)
    axD.set_axisbelow(True)
    axD.set_xlabel("ROI size (voxels, log scale)")
    axD.set_ylabel("r(ROI mean, first eigenvariate)")
    axD.legend(frameon=False, fontsize=7.5, loc="lower left")
    axD.set_title("D  Mean vs first eigenvariate (fslmeants --eig)", loc="left", fontweight="bold", fontsize=10)
    import matplotlib.ticker as mticker
    for ax in (axB, axD):                                   # plain numbers on the log axes
        ax.xaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    for ax in (axA, axB, axC, axD):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="CSV: subject_id, group, bids_id")
    ap.add_argument("--atlas_cs", required=True)
    ap.add_argument("--cleaned_glob", required=True, help="with {id} for the bids_id")
    ap.add_argument("--roi_labels", default=None, help="CSV with atlas_value (or region) and label")
    ap.add_argument("--eig_dir", default=None, help="folder of FSL eigenvariate signal<N>.txt files")
    ap.add_argument("--signals_dir", default=None, help="folder of ROI-mean signal<N>.txt files")
    ap.add_argument("--n_perm", type=int, default=10000)
    ap.add_argument("--n_mice", type=int, default=0, help="0 = all")
    ap.add_argument("--out_dir", default="roi_homogeneity")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    meta = read_meta(a.meta)
    if a.n_mice:
        meta = meta.iloc[:a.n_mice]
    A = np.rint(np.asarray(nib.load(a.atlas_cs).dataobj)).astype(int)
    values = sorted(int(v) for v in np.unique(A) if v != 0)
    labels = {v: f"ROI{v}" for v in values}
    if a.roi_labels:
        L = pd.read_csv(a.roi_labels)
        key = "atlas_value" if "atlas_value" in L else "region"
        labels.update({int(k): str(s) for k, s in zip(L[key], L["label"])})
    inroi = A > 0
    lab_in = A[inroi]
    idx = {v: np.flatnonzero(lab_in == v) for v in values}

    rows, mrows = [], []
    for _, r in meta.iterrows():
        mrow = dict(subject_id=r.subject_id, bids_id=r.bids_id, group=r.group)
        try:
            img = nib.load(one(a.cleaned_glob.replace("{id}", r.bids_id)))
            if img.shape[:3] != A.shape:
                raise ValueError(f"grid {img.shape[:3]} differs from atlas {A.shape}")
            V = np.asarray(img.dataobj, dtype=np.float32)[inroi]          # voxels x time
            eig_cols, mean_cols = [], []
            for v in values:
                X = V[idx[v]].T.astype(np.float64)                       # time x voxels
                mvec = X.mean(1)                                         # ROI mean (RABIES, fslmeants)
                res = roi_measures(X, mvec)
                eig_cols.append(res.pop("eig", np.full(X.shape[0], np.nan)))
                mean_cols.append(mvec)
                rows.append(dict(subject_id=r.subject_id, bids_id=r.bids_id, group=r.group, region=v,
                                 label=labels[v]) | res)
            for name, folder, cols in [("eig_check_min_r", a.eig_dir, eig_cols),
                                       ("signals_check_min_r", a.signals_dir, mean_cols)]:
                if folder:
                    S = np.loadtxt(os.path.join(folder, f"signal{r.subject_id}.txt"), ndmin=2)
                    if S.shape != (len(cols[0]), len(values)):
                        raise ValueError(f"{name}: signal file shape {S.shape}")
                    mrow[name] = float(np.nanmin([corr(c, S[:, j]) for j, c in enumerate(cols)]))
            mrow["n_vol"] = V.shape[1]
        except Exception as e:
            mrow["note"] = str(e)[:300]
        mrows.append(mrow)
        print(f"sub-{r.bids_id}: " + (mrow["note"] if "note" in mrow else
              f"{len(values)} ROIs" + "".join(f", {k} {mrow[k]:.6f}" for k in
                                              ("eig_check_min_r", "signals_check_min_r") if k in mrow)),
              flush=True)

    mr = pd.DataFrame(rows)
    if mr.empty:
        raise SystemExit("no mouse processed")
    mr.to_csv(os.path.join(a.out_dir, "roi_homogeneity_mouse_roi.csv"), index=False)
    order = group_order(mr.group)
    pal = palette_for(order)

    # per mouse: voxel-weighted averages over ROIs (share of all ROI voxel variance)
    mouse = pd.DataFrame(mrows)
    w = mr.n_vox_used.astype(float)
    agg = {}
    for k in ["r2_mean", "r2_pc1", "r2_pc2", "mean_r"]:
        agg[f"{k}_w"] = (mr[k] * w).groupby(mr.subject_id).sum() / w.where(mr[k].notna()).groupby(mr.subject_id).sum()
    agg["r2_mean_unweighted"] = mr.groupby("subject_id").r2_mean.mean()
    agg["r_mean_eig_min"] = mr.groupby("subject_id").r_mean_eig.min()
    mouse = mouse.merge(pd.DataFrame(agg).reset_index(), on="subject_id", how="left")
    mouse.to_csv(os.path.join(a.out_dir, "roi_homogeneity_by_mouse.csv"), index=False)

    # per ROI across mice
    g = mr.groupby("region")
    byroi = pd.DataFrame({"region": values, "label": [labels[v] for v in values]}).set_index("region")
    byroi["n_vox_median"] = g.n_vox.median()
    byroi["n_vox_used_median"] = g.n_vox_used.median()
    for k in MEASURES:
        byroi[f"{k}_median"] = g[k].median()
        byroi[f"{k}_q1"] = g[k].quantile(0.25)
        byroi[f"{k}_q3"] = g[k].quantile(0.75)
    byroi["mean_over_pc1_median"] = (mr.r2_mean / mr.r2_pc1).groupby(mr.region).median()
    byroi["frac_mice_r_mean_eig_lt_0.5"] = (mr.r_mean_eig < 0.5).groupby(mr.region).mean()
    for gr in order:
        byroi[f"r2_mean_median_{gr}"] = mr[mr.group == gr].groupby("region").r2_mean.median()

    # group tests
    tests = []
    ok = mouse.r2_mean_w.notna().to_numpy()
    mg = mouse.group.to_numpy()[ok]
    t1 = group_tests(mouse.r2_mean_w.to_numpy(float)[ok], mg, order, a.n_perm)
    t1.insert(0, "measure", "r2_mean_w (per mouse, voxel-weighted over ROIs)")
    tests.append(t1)
    from welch_tests_scipy import welch_anova_edgewise_permutation
    W = mr.pivot(index="subject_id", columns="region", values="r2_mean").loc[mouse.subject_id[ok]]
    gm = [(mg == gr) for gr in order]
    F, p = welch_anova_edgewise_permutation(W[values].to_numpy(float), gm, n_permutations=a.n_perm,
                                            random_seed=RANDOM_SEED)
    byroi["welch_F_r2_mean"] = F
    byroi["welch_p_r2_mean"] = p
    byroi["welch_q_r2_mean"] = bh(p)
    byroi.reset_index().to_csv(os.path.join(a.out_dir, "roi_homogeneity_by_roi.csv"), index=False)
    T = pd.concat(tests)
    T.to_csv(os.path.join(a.out_dir, "roi_homogeneity_group_tests.csv"), index=False)

    rho, rho_p = stats.spearmanr(byroi.n_vox_used_median, byroi.r2_mean_median)
    omni = t1.iloc[0]
    make_figure(byroi.reset_index(), mouse, mr, order, pal,
                os.path.join(a.out_dir, "FigS_roi_homogeneity.png"), rho, omni)

    # report
    q = lambda s: f"{np.nanmedian(s):.2f} [{np.nanpercentile(s, 25):.2f}, {np.nanpercentile(s, 75):.2f}]"
    lines = [f"ROI signal homogeneity: {mouse.r2_mean_w.notna().sum()} mice, {len(values)} ROIs",
             "",
             "Per mouse x ROI, median [IQR] over all pairs:",
             f"  share explained by the ROI mean (r2_mean)         {q(mr.r2_mean)}",
             f"  ceiling, first principal component (r2_pc1)        {q(mr.r2_pc1)}",
             f"  second component (r2_pc2)                          {q(mr.r2_pc2)}",
             f"  ROI mean / ceiling                                 {q(mr.r2_mean / mr.r2_pc1)}",
             f"  mean pairwise voxel r (homogeneity)                {q(mr.mean_r)}",
             f"  r(ROI mean, first eigenvariate)                    {q(mr.r_mean_eig)}; "
             f"< 0.5 in {(mr.r_mean_eig < 0.5).sum()} of {mr.r_mean_eig.notna().sum()} pairs",
             f"Per mouse, voxel-weighted share explained by ROI means: {q(mouse.r2_mean_w)}",
             f"ROI size vs r2_mean (ROI medians): Spearman rho = {rho:.2f}, p = {rho_p:.1e}",
             f"Voxels left out (zero variance), median per mouse x ROI: {np.median(mr.n_vox - mr.n_vox_used):.0f}",
             ""]
    for k, what in [("eig_check_min_r", "FSL eigenvariate files (--eig_dir)"),
                    ("signals_check_min_r", "ROI-mean signal files (--signals_dir)")]:
        if k in mouse:
            badm = mouse[mouse[k] < 0.999]
            lines.append(f"Check vs {what}: lowest r over mice and ROIs {mouse[k].min():.6f}; mice below 0.999: "
                         + (", ".join(f"sub-{x}" for x in badm.bids_id) if len(badm) else "none"))
    lines += ["", "Group tests, per-mouse r2_mean_w:",
              T.to_string(index=False, float_format=lambda x: f"{x:.4g}"), "",
              f"ROIs with a group effect on r2_mean (BH q < 0.05 over {len(values)}): "
              f"{int((byroi.welch_q_r2_mean < 0.05).sum())}"]
    sig = byroi[byroi.welch_q_r2_mean < 0.05]
    if len(sig):
        lines.append(sig[["label", "welch_F_r2_mean", "welch_p_r2_mean", "welch_q_r2_mean"]].to_string())
    lines += ["", "Lowest ROIs (median r2_mean):",
              byroi.sort_values("r2_mean_median")[["label", "n_vox_used_median", "r2_mean_median", "r2_pc1_median",
                                                   "r2_pc2_median", "mean_r_median", "r_mean_eig_median"]]
              .head(15).to_string(float_format=lambda x: f"{x:.3f}")]
    bad = mouse[mouse.note.notna()] if "note" in mouse else mouse.iloc[:0]
    if len(bad):
        lines += ["", "NOT processed: " + ", ".join(f"sub-{b.bids_id} ({b.note})" for b in bad.itertuples())]
    rep = "\n".join(lines)
    open(os.path.join(a.out_dir, "roi_homogeneity_report.txt"), "w").write(rep + "\n")
    print("\n" + rep)


if __name__ == "__main__":
    main()
