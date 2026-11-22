#!/usr/bin/env python3
"""
cpca_group_lags.py
-------------------------------------------------------------------
Between-group comparison of complex-PCA phase (lag) maps.

Published code, used unchanged
    Complex PCA of Bolt et al. (2022), github.com/tsb46/complex_pca
    (tested against commit d5ec42ca6dfb2b43c93d124d535f0410439d9d7b).
    For every fit this script calls that repository's own functions, in the
    same order as cpca.py does for a list of input files:
        utils.load_write.load_scans   per-scan z-score, then concatenation
        cpca.hilbert_transform        analytic signal of the concatenated data
        cpca.pca                      randomized SVD (fbpca), loadings
    Nothing in the repository is modified.

Standard method around it
    Planned contrasts are tested by permutation: the mice of the two groups
    are pooled, relabelled at random (group sizes kept), and both groups are
    refitted (Nichols and Holmes, 2002). p = (1 + #{perm >= observed}) /
    (n_perm + 1) (Phipson and Smyth, 2010). Per-ROI tests use the maximum
    statistic over ROIs for family-wise error control (Nichols and Holmes,
    2002); the global test is BH-corrected over contrasts x components
    (Benjamini and Hochberg, 1995), as for the other planned contrasts.

Published measure of propagation
    Traveling index of each component (Bolt et al., 2022, Methods, after Feeny,
    2008): the reciprocal of the condition number of the ROI x 2 matrix whose
    columns are the real and imaginary parts of the component's loadings
    (0 = pure standing wave, 1 = pure traveling wave). It is computed for every
    group fit and every refit, and contrasts are tested on dTI = TI_a - TI_b
    (two-sided permutation p, BH over contrasts x components). A global phase
    rotation leaves it unchanged, so it needs only the matching below, not the
    phase alignment.

Our additions (to review)
    1. Matching: each fit's components are matched to the fit on all mice by
       |cosine similarity| of the complex loadings, and each matched component
       is rotated by its overall phase relative to the all-mouse component
       (complex components are only defined up to a global phase). Without
       this, phases from separate fits are not comparable.
    2. Statistics, per contrast and component (computed and written to the
       tables, but not reported in the paper: on 11 Nov 2026 the traveling
       index above was chosen as the paper's only group propagation test):
       G   = sum_r w_r |dphi_r| / sum_r w_r, w_r = |all-mouse loading of ROI r|,
             dphi_r = angle(v_a,r * conj(v_b,r)) (radians, -pi..pi).
             G is the loading-weighted mean absolute phase difference.
       ROI = |dphi_r| for ROIs with |all-mouse loading| >= median (the same
             "strong" ROIs as in Fig_cpca_propagation_roi); weak ROIs have
             poorly defined phases and are not tested.
       Lag differences in seconds use dphi / (2 pi f), f = median
       instantaneous frequency of the all-mouse component.

Component time courses (used only for the component frequency that turns
phase into seconds) are U*s from cpca.pca's outputs, not its 'pc_scores'. At
commit d5ec42c, cpca.pca computes pc_scores = input_data @ Va.T; fbpca returns
A ~ U diag(s) Va, so the projection on the components is input_data @ Va.conj().T
= U diag(s). The two agree only for real data. On 30 synthetic scans the
published pc_scores correlated 0.09-0.16 with U*s; input_data @ Va.conj().T
correlated 1.000. The loadings (Va.T @ diag(s) / sqrt(n-1)) are not affected.

Read a component only where every group's match_similarity is high (close to
1): a low value means that group has no counterpart of the all-mouse component,
and its phases are then meaningless. On synthetic data a noise component
matched at 0.1-0.2 and gave G near 85 degrees in every contrast.

Phase randomisation preserves these phases exactly, so a group difference here
describes the stationary lag structure of each group, not dynamics.

Synthetic check (10 Nov 2026; 95 mice in the study's group sizes, 40
permutations): a phase gradient added to CRND8_M only was detected in both of
its contrasts (p = 0.024, the minimum possible with 40 permutations), and no
other contrast reached p < 0.05; with no gradient no contrast reached p < 0.1.

Inputs
    --cpca_dir    clone of tsb46/complex_pca (needs fbpca, xmca, nibabel, scipy)
    --data_dir    one text file per mouse, volumes x ROIs (default hmm_signals)
    --groups      CSV with subject_id and group (default: the DFC run's
                  dfc_subject_metrics.csv); file of mouse N = --file_pattern
Outputs (in --out_dir)
    cpca_group_loadings.csv       per group (and 'ALL'): magnitude and phase per ROI
    cpca_group_traveling_index.csv  traveling index per group (and 'ALL') x component
    cpca_lag_contrasts.csv        G and traveling-index difference, permutation p,
                                  BH q per contrast x component
    cpca_lag_contrasts_roi.csv    per-ROI phase / lag difference, FWE p (strong ROIs)
    FigS_cpca_lags_by_group.pdf/.png  aligned loadings per group (polar)

Usage
    python cpca_group_lags.py --cpca_dir complex_pca --data_dir hmm_signals \
        --groups dfc_dcc_cpca/dfc_subject_metrics.csv --roi_labels roi_labels.csv
    Cost: 2 x n_perm fits per contrast (about 1 s each on one core), so 8
    contrasts x 1000 permutations is about 16,000 fits; use --n_jobs (e.g. 16)
    in a SLURM job. --n_perm 200 for a quick look. Results do not depend on
    --n_jobs: permutation b of contrast c always uses seed (seed, c, b).
"""

import argparse
import contextlib
import io
import itertools
import multiprocessing as mp
import os
import sys

# one BLAS thread per process; parallelism comes from --n_jobs
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

GROUP_ORDER = ["NTG_Y", "NTG_M", "NTG_A", "CRND8_Y", "CRND8_M", "5XFAD_Y", "5XFAD_M"]
# The same 8 planned contrasts as every other analysis in the study.
PAIRWISE_COMPARISONS = [
    ("NTG_Y", "NTG_M"), ("NTG_Y", "NTG_A"), ("NTG_Y", "CRND8_Y"),
    ("NTG_M", "CRND8_M"), ("NTG_Y", "5XFAD_Y"), ("NTG_M", "5XFAD_M"),
    ("CRND8_Y", "CRND8_M"), ("5XFAD_Y", "5XFAD_M"),
]
N_COMPS = 3          # components compared (as in the DFC run)
N_FIT_EXTRA = 2      # extra components fitted per group so a swap in order can be matched
FIT_SEED = 20261110  # fbpca uses numpy's global RNG; reset before every fit
N_PERM = 1000


# ------------------------------------------------------------------
# Published code (imported, not modified)
# ------------------------------------------------------------------
def import_cpca(cpca_dir):
    sys.path.insert(0, os.path.abspath(cpca_dir))
    import cpca                                   # noqa: E402
    from utils.load_write import load_scans       # noqa: E402
    return cpca, load_scans


def load_mouse(load_scans, path):
    """cpca's own loader on one scan: z-score per ROI (normalize='zscore')."""
    with contextlib.redirect_stdout(io.StringIO()):
        data, _, _ = load_scans([path], "txt", None, "zscore", False,
                                None, None, None, False)
    return data


def fit(cpca, blocks, n_comps):
    """Concatenate z-scored scans, Hilbert transform, PCA: as cpca.run_cpca."""
    np.random.seed(FIT_SEED)
    X = np.concatenate(blocks, axis=0)
    out = cpca.pca(cpca.hilbert_transform(X, False), n_comps, False)
    return out["loadings"], out["exp_var"], component_scores(out)


def anchor(L):
    """Rotate each component so the ROI with the largest |loading| has phase 0, the
    convention of Fig_cpca_propagation_roi. One common rotation per component:
    phase differences between ROIs, mice and groups are unchanged."""
    i = np.abs(L).argmax(axis=1)
    return L * np.exp(-1j * np.angle(L[np.arange(L.shape[0]), i]))[:, None]


def component_scores(out):
    """Component time courses U*s (= analytic data @ Va^H), from cpca.pca's own
    outputs U and s. cpca.pca's 'pc_scores' (input_data @ Va.T) omits the complex
    conjugate, so for complex data it is not the projection on the components;
    see the header. Loadings are unaffected."""
    return out["U"] * out["s"]


# ------------------------------------------------------------------
# Our additions: matching and statistics
# ------------------------------------------------------------------
def align(L, ref):
    """Match the rows of L (fit) to the rows of ref (all mice) by |cosine|;
    rotate each matched row by its overall phase relative to ref."""
    k = ref.shape[0]
    C = np.abs(ref.conj() @ L.T) / (np.linalg.norm(ref, axis=1)[:, None]
                                    * np.linalg.norm(L, axis=1)[None, :])
    best = max(itertools.permutations(range(L.shape[0]), k),
               key=lambda p: sum(C[i, p[i]] for i in range(k)))
    A = np.empty((k, L.shape[1]), complex)
    sim = np.empty(k)
    for i, j in enumerate(best):
        A[i] = L[j] * np.exp(-1j * np.angle(ref[i].conj() @ L[j]))
        sim[i] = C[i, j]
    return A, sim


def traveling_index(L):
    """Bolt et al. (2022) traveling index (Feeny, 2008) of each row of L: the
    smaller over the larger singular value of [Re(v), Im(v)] (ROI x 2)."""
    out = []
    for v in L:
        s = np.linalg.svd(np.column_stack([v.real, v.imag]), compute_uv=False)
        out.append(s[1] / s[0])
    return np.array(out)


def phase_diff(Aa, Ab):
    return np.angle(Aa * Ab.conj())               # k x ROI, -pi..pi


def stats(Aa, Ab, w, strong):
    d = np.abs(phase_diff(Aa, Ab))
    G = (w * d).sum(axis=1) / w.sum(axis=1)
    roi_max = np.array([d[i, strong[i]].max() for i in range(d.shape[0])])
    return G, roi_max, d


def component_frequency(pc_scores, n_per_scan, tr):
    """Median instantaneous frequency (Hz) of each component, within scans."""
    f = []
    for i in range(pc_scores.shape[1]):
        vals = []
        for s in range(0, pc_scores.shape[0], n_per_scan):
            ph = np.unwrap(np.angle(pc_scores[s:s + n_per_scan, i]))
            vals.append(np.abs(np.diff(ph)))
        f.append(np.median(np.concatenate(vals)) / (2 * np.pi * tr))
    return np.array(f)


def bh(p):
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        pv = p[ok]
        o = np.argsort(pv)
        r = pv[o] * len(pv) / np.arange(1, len(pv) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]
        out = np.empty_like(pv)
        out[o] = np.minimum(r, 1)
        q[ok] = out
    return q


# ------------------------------------------------------------------
_W = {}   # shared with forked workers


def _perm_chunk(task):
    """Null statistics for permutations bs of contrast c (runs in a worker)."""
    c, bs = task
    W = _W
    ga, gb = W["contrasts"][c]
    pool = W["members"][ga] + W["members"][gb]
    na = len(W["members"][ga])
    out = []
    for b in bs:
        perm = np.random.default_rng([W["seed"], c, b]).permutation(pool)
        Pa, _ = W["group_fit"](list(perm[:na]))
        Pb, _ = W["group_fit"](list(perm[na:]))
        Gp, Mp, _ = stats(Pa, Pb, W["w"], W["strong"])
        out.append((b, Gp, Mp, traveling_index(Pa) - traveling_index(Pb)))
    return c, out


def read_groups(path):
    g = pd.read_csv(path)
    g = g[["subject_id", "group"]].drop_duplicates()
    if g.subject_id.duplicated().any():
        raise SystemExit(f"{path}: a subject_id has more than one group")
    return g


def roi_names(path, n):
    names = {r: f"ROI {r}" for r in range(1, n + 1)}
    if path:
        lab = pd.read_csv(path)
        names.update({int(r): str(l) for r, l in zip(lab.region, lab.label)})
    return names


def fig_polar(load, out_dir, k, groups, stem="FigS_cpca_lags_by_group"):
    """Aligned loadings per group; one radial scale per component."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["ALL"] + groups
    fig = plt.figure(figsize=(1.9 * len(cols), 2.0 * k))
    for i in range(k):
        for j, g in enumerate(cols):
            ax = fig.add_subplot(k, len(cols), i * len(cols) + j + 1, projection="polar")
            s = load[(load.group == g) & (load.component == i + 1)]
            strong = s.strong.to_numpy(bool)
            ax.scatter(s.phase_rad[~strong], s.magnitude[~strong], s=5, color="#b5b3ab", lw=0)
            ax.scatter(s.phase_rad[strong], s.magnitude[strong], s=7, color="#0b0b0b", lw=0)
            rmax = 1.05 * load.magnitude[load.component == i + 1].max()
            ax.set_ylim(0, rmax)
            ax.set_yticks([round(rmax / 2, 2), round(rmax, 2)])
            ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
            ax.set_xticklabels(["0", "π/2", "π", "−π/2"], fontsize=6)
            ax.tick_params(axis="y", labelsize=5)
            if i == 0:
                n = int(s.n_mice.iloc[0]) if len(s) else 0
                ax.set_title(f"{g} (n = {n})", fontsize=8, pad=10)
            if j == 0:
                ax.text(-0.45, 0.5, f"Component {i + 1}", transform=ax.transAxes,
                        rotation=90, va="center", fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{stem}.{ext}"), dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--cpca_dir", required=True, help="clone of tsb46/complex_pca")
    ap.add_argument("--data_dir", default="hmm_signals")
    ap.add_argument("--file_pattern", default="signal{}.txt")
    ap.add_argument("--groups", default=os.path.join("dfc_dcc_cpca", "dfc_subject_metrics.csv"))
    ap.add_argument("--roi_labels", default=None)
    ap.add_argument("--tr", type=float, default=2.0)
    ap.add_argument("--n_comps", type=int, default=N_COMPS)
    ap.add_argument("--n_perm", type=int, default=N_PERM)
    ap.add_argument("--seed", type=int, default=20261110, help="permutation seed")
    ap.add_argument("--n_jobs", type=int, default=1, help="worker processes")
    ap.add_argument("--out_dir", default=os.path.join("dfc_dcc_cpca", "cpca_group_lags"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cpca, load_scans = import_cpca(args.cpca_dir)

    G = read_groups(args.groups)
    groups = [g for g in GROUP_ORDER if g in set(G.group)] + \
             sorted(set(G.group) - set(GROUP_ORDER))
    data = {}
    for sid in G.subject_id:
        fp = os.path.join(args.data_dir, args.file_pattern.format(int(sid)))
        if not os.path.exists(fp):
            raise SystemExit(f"missing {fp}")
        data[int(sid)] = load_mouse(load_scans, fp)
    T = {x.shape[0] for x in data.values()}
    n_roi = next(iter(data.values())).shape[1]
    print(f"{len(data)} mice, {n_roi} ROIs, volumes per scan: {sorted(T)}")
    members = {g: G.subject_id[G.group == g].astype(int).tolist() for g in groups}
    print("Groups: " + ", ".join(f"{g} ({len(members[g])})" for g in groups))
    names = roi_names(args.roi_labels, n_roi)
    k = args.n_comps
    n_fit = k + N_FIT_EXTRA

    # reference: all mice, k components
    ids = [int(s) for s in G.subject_id]
    ref, ref_var, ref_scores = fit(cpca, [data[s] for s in ids], k)
    ref = anchor(ref)
    w = np.abs(ref)
    strong = w >= np.median(w, axis=1, keepdims=True)
    freq = (component_frequency(ref_scores, next(iter(T)), args.tr) if len(T) == 1
            else np.full(k, np.nan))
    print("All mice: variance " + ", ".join(f"C{i + 1} {100 * v:.1f}%" for i, v in enumerate(ref_var))
          + "; frequency " + ", ".join(f"C{i + 1} {f:.3f} Hz" for i, f in enumerate(freq)))

    def group_fit(sids):
        L, _, _ = fit(cpca, [data[s] for s in sids], n_fit)
        return align(L, ref)

    # per-group fits
    rows, aligned = [], {}
    for i in range(k):
        for r in range(n_roi):
            rows.append(dict(group="ALL", n_mice=len(ids), component=i + 1, region=r + 1,
                             label=names[r + 1], magnitude=abs(ref[i, r]),
                             phase_rad=np.angle(ref[i, r]), strong=bool(strong[i, r]),
                             match_similarity=1.0))
    for g in groups:
        A, sim = group_fit(members[g])
        aligned[g] = (A, sim)
        for i in range(k):
            for r in range(n_roi):
                rows.append(dict(group=g, n_mice=len(members[g]), component=i + 1,
                                 region=r + 1, label=names[r + 1], magnitude=abs(A[i, r]),
                                 phase_rad=np.angle(A[i, r]), strong=bool(strong[i, r]),
                                 match_similarity=sim[i]))
        print(f"  {g}: match to all-mouse components " + ", ".join(f"{s:.2f}" for s in sim))
    print("  (read a component only where every group matches closely; see the header)")
    load = pd.DataFrame(rows)
    load.to_csv(os.path.join(args.out_dir, "cpca_group_loadings.csv"), index=False)
    ti = {"ALL": traveling_index(ref)}
    ti.update({g: traveling_index(aligned[g][0]) for g in groups})
    pd.DataFrame([dict(group=g, n_mice=len(ids) if g == "ALL" else len(members[g]),
                       component=i + 1, traveling_index=ti[g][i],
                       match_similarity=1.0 if g == "ALL" else aligned[g][1][i])
                  for g in ti for i in range(k)]).to_csv(
        os.path.join(args.out_dir, "cpca_group_traveling_index.csv"), index=False)
    print("Traveling index (Bolt et al., 2022): " + "; ".join(
        f"{g} " + ", ".join(f"{v:.3f}" for v in ti[g]) for g in ti))
    fig_polar(load, args.out_dir, k, groups)

    # planned contrasts, permutation test (parallel over chunks of permutations)
    contrasts = [(a, b) for a, b in PAIRWISE_COMPARISONS if a in members and b in members]
    _W.update(contrasts=contrasts, members=members, seed=args.seed, group_fit=group_fit,
              w=w, strong=strong)
    G_null = np.empty((len(contrasts), args.n_perm, k))
    M_null = np.empty((len(contrasts), args.n_perm, k))
    T_null = np.empty((len(contrasts), args.n_perm, k))
    step = max(1, args.n_perm // max(1, 4 * args.n_jobs // max(1, len(contrasts))))
    tasks = [(c, list(range(s0, min(s0 + step, args.n_perm))))
             for c in range(len(contrasts)) for s0 in range(0, args.n_perm, step)]
    print(f"Permutations: {len(contrasts)} contrasts x {args.n_perm} "
          f"({2 * len(contrasts) * args.n_perm} fits, {args.n_jobs} job(s))")
    if args.n_jobs > 1:
        with mp.get_context("fork").Pool(args.n_jobs) as P:
            results = P.imap_unordered(_perm_chunk, tasks)
            for c, out in results:
                for b, Gp, Mp, Tp in out:
                    G_null[c, b], M_null[c, b], T_null[c, b] = Gp, Mp, Tp
    else:
        for t in tasks:
            c, out = _perm_chunk(t)
            for b, Gp, Mp, Tp in out:
                G_null[c, b], M_null[c, b], T_null[c, b] = Gp, Mp, Tp

    crow, rrow = [], []
    for c, (ga, gb) in enumerate(contrasts):
        Aa, sa = aligned[ga]
        Ab, sb = aligned[gb]
        G_obs, _, d_obs = stats(Aa, Ab, w, strong)
        na = len(members[ga])
        dph = phase_diff(Aa, Ab)
        for i in range(k):
            p = (1 + np.sum(G_null[c, :, i] >= G_obs[i] - 1e-12)) / (args.n_perm + 1)
            dT = ti[ga][i] - ti[gb][i]
            pT = (1 + np.sum(np.abs(T_null[c, :, i]) >= abs(dT) - 1e-12)) / (args.n_perm + 1)
            crow.append(dict(contrast=f"{ga} vs {gb}", component=i + 1, n_a=na,
                             n_b=len(members[gb]), G_rad=G_obs[i], G_deg=np.degrees(G_obs[i]),
                             G_null_median_rad=np.median(G_null[c, :, i]), p_perm=p,
                             TI_a=ti[ga][i], TI_b=ti[gb][i], dTI=dT, p_TI=pT,
                             match_similarity_a=sa[i], match_similarity_b=sb[i]))
            for r in range(n_roi):
                pr = ((1 + np.sum(M_null[c, :, i] >= d_obs[i, r] - 1e-12)) / (args.n_perm + 1)
                      if strong[i, r] else np.nan)
                rrow.append(dict(contrast=f"{ga} vs {gb}", component=i + 1, region=r + 1,
                                 label=names[r + 1], strong=bool(strong[i, r]),
                                 dphase_rad=dph[i, r], dphase_deg=np.degrees(dph[i, r]),
                                 dlag_s=dph[i, r] / (2 * np.pi * freq[i]), p_FWE=pr))
        print(f"  {ga} vs {gb}: G = " + ", ".join(f"C{i + 1} {np.degrees(G_obs[i]):.1f}°"
                                                  for i in range(k)))
    C = pd.DataFrame(crow)
    C["q_BH"] = bh(C.p_perm)
    C["q_TI_BH"] = bh(C.p_TI)
    C.to_csv(os.path.join(args.out_dir, "cpca_lag_contrasts.csv"), index=False)
    pd.DataFrame(rrow).to_csv(os.path.join(args.out_dir, "cpca_lag_contrasts_roi.csv"), index=False)
    print("\nPlanned contrasts (G = loading-weighted mean |phase difference|):")
    print(C[["contrast", "component", "n_a", "n_b", "G_deg", "p_perm", "q_BH", "TI_a", "TI_b",
             "p_TI", "q_TI_BH", "match_similarity_a", "match_similarity_b"]]
          .round(4).to_string(index=False))
    print(f"Done. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
