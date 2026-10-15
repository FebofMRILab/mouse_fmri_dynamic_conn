#!/usr/bin/env python3
"""
mp_signal_rank_group.py
-------------------------------------------------------------------
Group-level signal rank: the same two noise thresholds as mp_signal_rank.py
(which works per subject), applied to the pooled data the group PCA and the
HMM-PCA are estimated from (every subject's standardised ROI series,
concatenated). Functions are imported unchanged from mp_signal_rank.py.

  Marchenko-Pastur  edge (1 + sqrt(N / T_eff))^2 with T_eff = the sum of the
                    subjects' Bartlett effective sample sizes (subjects are
                    independent).
  Parallel analysis circular shifts within each subject (every ROI by its own
                    offset), then pooled; the i-th observed eigenvalue must
                    exceed the 95th percentile of the i-th surrogate
                    eigenvalue (leading eigenvalues counted until the first
                    failure).

With ~95 x 208 effective samples both thresholds sit close to 1, so almost
any structure in any subject counts as "above noise", including structure
that only some subjects share. To read the pooled rank, the script also
reports how reproducible the leading group subspace is across subjects:
  Split-half overlap  subjects split at random into two halves (stratified by
                    group when meta_data.csv is present); for each k, the
                    overlap of the two halves' top-k eigenvector subspaces,
                    ||U_A' U_B||_F^2 / k (1 = same subspace; k / N expected
                    for unrelated subspaces). Median and 5-95 % over splits.

Usage (as mp_signal_rank.py; one or more datasets):
    python mp_signal_rank_group.py --dataset mouse_SE11T hmm_signals 95 2
    python mp_signal_rank_group.py --dataset mouse_SE11T hmm_signals 95 2 \
        --dataset human_GE3T ../human/hmm_signals 54 3
Outputs in --out_dir (default signal_rank_group): signal_rank_group.csv,
signal_rank_group_spectrum.csv, FigS_signal_rank_group.{pdf,png}.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.lines import Line2D                                    # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mp_signal_rank import (standardise, effective_samples, circular_shift,  # noqa: E402
                            eigvals_corr, INK, SECONDARY, BASELINE, DATASET_COLORS)
import seaborn as sns                                                  # noqa: E402

SEED = 20261015
MARK_K = (15, 25)            # the HMM-PCA rank and the DCC group-PCA dimension


def load_dataset(ddir, n_sub, pattern):
    Xs, ids = [], []
    for s in range(1, n_sub + 1):
        fn = os.path.join(ddir, pattern.format(s))
        if not os.path.isfile(fn):
            continue
        sep = "," if "," in open(fn).readline() else None
        X = np.loadtxt(fn, delimiter=sep).astype(float)
        if X.shape[0] < X.shape[1]:
            X = X.T
        Xs.append(standardise(X))
        ids.append(s)
    groups = None
    meta = os.path.join(ddir, "meta_data.csv")
    if os.path.isfile(meta):
        m = pd.read_csv(meta, encoding="utf-8-sig")        # tolerates a byte-order mark
        m.columns = [c.strip().lower() for c in m.columns]
        if {"subject_id", "group"} <= set(m.columns):
            g = dict(zip(m.subject_id.astype(int), m.group.astype(str).str.strip()))
            groups = [g.get(i, "NA") for i in ids]
    return Xs, ids, groups


def top_vectors(Xs_list):
    C = np.corrcoef(np.vstack(Xs_list), rowvar=False)
    w, V = np.linalg.eigh((C + C.T) / 2)
    return V[:, np.argsort(w)[::-1]]


def split_halves(n, groups, rng):
    """Random half of the subjects, stratified by group when groups are known."""
    if groups is None:
        p = rng.permutation(n)
        return p[: n // 2], p[n // 2:]
    a, b = [], []
    for g in sorted(set(groups)):
        idx = rng.permutation([i for i in range(n) if groups[i] == g])
        h = len(idx) // 2 + (rng.random() < 0.5 if len(idx) % 2 else 0)
        a += list(idx[:h])
        b += list(idx[h:])
    return np.array(a), np.array(b)


def group_rank(Xs, groups, rng, n_shift, n_split, k_max):
    N = Xs[0].shape[1]
    lam = eigvals_corr(np.vstack(Xs))
    teff = float(sum(effective_samples(x) for x in Xs))
    edge = (1 + np.sqrt(N / teff)) ** 2
    rank_mp = int((lam > edge).sum())
    S = np.array([eigvals_corr(np.vstack([circular_shift(x, rng) for x in Xs]))
                  for _ in range(n_shift)])
    thr = np.percentile(S, 95, axis=0)
    above = lam > thr
    rank_pa = int(np.argmin(above)) if not above.all() else N
    k_max = min(k_max, N)
    ov = np.zeros((n_split, k_max))
    for b in range(n_split):
        ia, ib = split_halves(len(Xs), groups, rng)
        UA = top_vectors([Xs[i] for i in ia])
        UB = top_vectors([Xs[i] for i in ib])
        M = UA[:, :k_max].T @ UB[:, :k_max]
        for k in range(1, k_max + 1):
            ov[b, k - 1] = np.sum(M[:k, :k] ** 2) / k
    return dict(lam=lam, thr=thr, edge=edge, teff=teff, rank_mp=rank_mp, rank_pa=rank_pa,
                overlap=ov, N=N, T=int(sum(x.shape[0] for x in Xs)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("--dataset", nargs=4, action="append", required=True,
                    metavar=("LABEL", "DATA_DIR", "N_SUBJECTS", "TR"))
    ap.add_argument("--file_pattern", default="signal{}.txt")
    ap.add_argument("--n_shift", type=int, default=50)
    ap.add_argument("--n_split", type=int, default=50)
    ap.add_argument("--k_max", type=int, default=40)
    ap.add_argument("--out_dir", default="signal_rank_group")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    rows, spec, results = [], [], {}
    for label, ddir, n_sub, tr in args.dataset:
        Xs, ids, groups = load_dataset(ddir, int(n_sub), args.file_pattern)
        if not Xs:
            print(f"{label}: no subjects in {ddir}")
            continue
        r = group_rank(Xs, groups, rng, args.n_shift, args.n_split, args.k_max)
        results[label] = r
        lam, N = r["lam"], r["N"]
        ov_med = np.median(r["overlap"], axis=0)
        row = dict(dataset=label, n_subjects=len(Xs), n_roi=N, T_total=r["T"],
                   T_eff_total=round(r["teff"], 1), mp_edge=r["edge"],
                   rank_mp=r["rank_mp"], rank_pa=r["rank_pa"],
                   lambda1_share=lam[0] / lam.sum(),
                   var_signal_mp=lam[:r["rank_mp"]].sum() / lam.sum(),
                   var_signal_pa=lam[:r["rank_pa"]].sum() / lam.sum(),
                   stratified_by_group=groups is not None)
        for k in MARK_K:
            if k <= N:
                row[f"var_top{k}"] = lam[:k].sum() / lam.sum()
            if k <= len(ov_med):
                row[f"splithalf_overlap_k{k}"] = ov_med[k - 1]
                row[f"random_overlap_k{k}"] = k / N
        rows.append(row)
        p05, p95 = np.percentile(r["overlap"], [5, 95], axis=0)
        for i in range(N):
            spec.append(dict(dataset=label, rank=i + 1, eigenvalue=lam[i],
                             var_share=lam[i] / lam.sum(), cum_share=lam[: i + 1].sum() / lam.sum(),
                             pa_threshold_95=r["thr"][i], mp_edge=r["edge"],
                             splithalf_overlap_median=ov_med[i] if i < len(ov_med) else np.nan,
                             splithalf_overlap_p05=p05[i] if i < len(ov_med) else np.nan,
                             splithalf_overlap_p95=p95[i] if i < len(ov_med) else np.nan,
                             random_overlap=(i + 1) / N))
        print(f"{label}: {len(Xs)} subjects pooled; MP rank {r['rank_mp']}, PA rank {r['rank_pa']}")
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(args.out_dir, "signal_rank_group.csv"), index=False)
    pd.DataFrame(spec).to_csv(os.path.join(args.out_dir, "signal_rank_group_spectrum.csv"), index=False)

    # ---------------- figure ----------------
    plt.rcParams.update({"font.family": "Arial", "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), gridspec_kw=dict(wspace=0.38))
    handles = []
    for i, (lab, r) in enumerate(results.items()):
        c = DATASET_COLORS[i % 3]
        kk = np.arange(1, min(60, r["N"]) + 1)
        ax = axes[0]
        ax.plot(kk, r["lam"][: kk.size], "-", color=c, lw=1.3)
        ax.plot(kk, r["thr"][: kk.size], "--", color=c, lw=0.9)
        ax.axhline(r["edge"], color=c, lw=0.7, ls=":")
        ax = axes[1]
        k2 = np.arange(1, r["overlap"].shape[1] + 1)
        p05, med, p95 = np.percentile(r["overlap"], [5, 50, 95], axis=0)
        ax.fill_between(k2, p05, p95, color=c, alpha=0.18, lw=0)
        ax.plot(k2, med, "-", color=c, lw=1.3)
        ax.plot(k2, k2 / r["N"], ":", color=c, lw=0.9)
        handles.append(Line2D([], [], color=c, lw=1.3,
                              label=f"{lab}: MP {r['rank_mp']}, PA {r['rank_pa']}"))
    ax = axes[0]
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Eigenvalue rank", fontsize=8.5)
    ax.set_ylabel("Eigenvalue, pooled correlation", fontsize=8.5)
    ax.set_title("Pooled spectrum and noise thresholds", fontsize=8, loc="left")
    leg = handles + [Line2D([], [], color=SECONDARY, lw=0.9, ls="--", label="Parallel analysis, 95th pct"),
                     Line2D([], [], color=SECONDARY, lw=0.7, ls=":", label="Marchenko–Pastur edge")]
    ax.legend(handles=leg, frameon=False, fontsize=6, loc="upper right")
    ax = axes[1]
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Number of components k", fontsize=8.5)
    ax.set_ylabel("Split-half subspace overlap", fontsize=8.5)
    ax.set_title("Top-k group subspace: two halves of the subjects", fontsize=8, loc="left")
    ax.legend(handles=[Line2D([], [], color=SECONDARY, lw=1.3, label="median (band: 5-95 %)"),
                       Line2D([], [], color=SECONDARY, lw=0.9, ls=":", label="unrelated subspaces (k/N)")],
              frameon=False, fontsize=6, loc="lower right")
    for ax, letter in zip(axes, "DE"):
        for k in MARK_K:
            ax.axvline(k, color=BASELINE, lw=0.8, zorder=0)
        ax.tick_params(labelsize=7)
        sns.despine(ax=ax)
        ax.text(-0.18, 1.07, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", color=INK)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out_dir, f"FigS_signal_rank_group.{ext}"), bbox_inches="tight",
                    dpi=600 if ext == "png" else None)
    plt.close(fig)
    pd.set_option("display.width", 220)
    print("\n" + R.round(4).to_string(index=False))
    print(f"\nVertical lines: k = {MARK_K}. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()