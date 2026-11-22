#!/usr/bin/env python3
"""
cpca_ti_bootstrap.py
-------------------------------------------------------------------
Bootstrap 95% intervals for each group's complex-PCA traveling index.

Published code, used unchanged
    Every fit is the one in cpca_group_lags.py (imported from the same folder):
    Bolt et al. (2022) complex PCA, github.com/tsb46/complex_pca, i.e.
    utils.load_write.load_scans (per-scan z-score), concatenation,
    cpca.hilbert_transform and cpca.pca, followed by the same matching to the
    all-mouse components and the same traveling index (Bolt et al., 2022,
    after Feeny, 2008).

Standard method around it
    Nonparametric bootstrap over mice with percentile intervals (Efron and
    Tibshirani, 1993, ch. 6 and 13): for each group of n mice, draw n mice
    from that group with replacement, refit on their concatenated scans (a
    mouse drawn twice contributes its scan twice), match the components to
    the all-mouse fit and compute the traveling index. The interval is the
    2.5th to 97.5th percentile over --n_boot resamples (1000, the number
    Efron and Tibshirani recommend for percentile intervals).

How to read the intervals
    They show how much each group's index would move with a different sample
    of mice of the same size. Group differences are tested by the permutation
    test in cpca_group_lags.py (cpca_lag_contrasts.csv), not by whether
    intervals overlap.
    With 5 to 13 mice per group the intervals are approximate. The index is
    0 to 1 and noise in the loadings raises it, so resampled fits (which
    repeat mice and so carry fewer distinct mice) tend to give higher values
    than the group's own fit; the interval can then sit partly above the
    observed dot. boot_median is written so this can be seen; the intervals
    are reported as they are, without a bias correction.
    frac_match_below is the fraction of resamples whose matched component had
    |cosine| < MATCH_MIN (0.85) with the all-mouse component; where it is large, the
    resamples often had no counterpart of that component and the interval
    mixes different patterns.

Inputs
    Same as cpca_group_lags.py: --cpca_dir, --data_dir, --file_pattern, --groups.
Outputs (in --out_dir, by default the same folder as cpca_group_lags.py)
    cpca_ti_bootstrap.csv          one row per group x resample x component:
                                   traveling_index, match_similarity
    cpca_ti_bootstrap_summary.csv  per group x component: traveling_index of
                                   the group's own fit, n_boot, boot_median,
                                   ci_lo, ci_hi, frac_match_below
    plot_cpca_patterns.py and plot_cpca_traveling_index.py draw ci_lo..ci_hi
    as a bar behind each group's dot when the summary file is in --in_dir.

Usage
    python cpca_ti_bootstrap.py --cpca_dir complex_pca --data_dir hmm_signals \
        --groups dfc_dcc_cpca/dfc_subject_metrics.csv --n_jobs 16
    Cost: 7 groups x 1000 resamples = 7000 fits (about 1 s each on one core).
    Results do not depend on --n_jobs: resample b of group g always uses seed
    (seed, index of g in GROUP_ORDER, b).
"""

import argparse
import multiprocessing as mp
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cpca_group_lags as L   # noqa: E402  (fit, align, anchor, traveling_index, ...)

N_BOOT = 1000
# Weak-match cutoff on |cosine| between a group component and the all-mouse
# component: 0.85, the lower bound of 'fair similarity' for Tucker's congruence
# coefficient (Lorenzo-Seva and ten Berge, 2006, Methodology 2:57-64).
MATCH_MIN = 0.85
SEED = 20261111

_W = {}   # shared with forked workers


def _boot_chunk(task):
    g, bs = task
    W = _W
    ids = W["members"][g]
    out = []
    for b in bs:
        draw = np.random.default_rng([W["seed"], W["gidx"][g], b]).choice(
            ids, len(ids), replace=True)
        A, sim = W["group_fit"](list(draw))
        out.append((g, b, L.traveling_index(A), sim))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--cpca_dir", required=True, help="clone of tsb46/complex_pca")
    ap.add_argument("--data_dir", default="hmm_signals")
    ap.add_argument("--file_pattern", default="signal{}.txt")
    ap.add_argument("--groups", default=os.path.join("dfc_dcc_cpca", "dfc_subject_metrics.csv"))
    ap.add_argument("--n_comps", type=int, default=L.N_COMPS)
    ap.add_argument("--n_boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED, help="bootstrap seed")
    ap.add_argument("--n_jobs", type=int, default=1, help="worker processes")
    ap.add_argument("--out_dir", default=os.path.join("dfc_dcc_cpca", "cpca_group_lags"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    cpca, load_scans = L.import_cpca(args.cpca_dir)

    G = L.read_groups(args.groups)
    groups = [g for g in L.GROUP_ORDER if g in set(G.group)] + \
             sorted(set(G.group) - set(L.GROUP_ORDER))
    data = {}
    for sid in G.subject_id:
        fp = os.path.join(args.data_dir, args.file_pattern.format(int(sid)))
        if not os.path.exists(fp):
            raise SystemExit(f"missing {fp}")
        data[int(sid)] = L.load_mouse(load_scans, fp)
    members = {g: G.subject_id[G.group == g].astype(int).tolist() for g in groups}
    print(f"{len(data)} mice; groups: " + ", ".join(f"{g} ({len(members[g])})" for g in groups))
    k = args.n_comps

    # all-mouse reference and group fits, exactly as in cpca_group_lags.py
    ids = [int(s) for s in G.subject_id]
    ref = L.anchor(L.fit(cpca, [data[s] for s in ids], k)[0])

    def group_fit(sids):
        Lg, _, _ = L.fit(cpca, [data[s] for s in sids], k + L.N_FIT_EXTRA)
        return L.align(Lg, ref)

    obs = {}
    for g in groups:
        A, sim = group_fit(members[g])
        obs[g] = (L.traveling_index(A), sim)
        print(f"  {g}: traveling index " + ", ".join(f"{v:.3f}" for v in obs[g][0])
              + "; match " + ", ".join(f"{s:.2f}" for s in sim))

    gidx = {g: (L.GROUP_ORDER.index(g) if g in L.GROUP_ORDER
                else len(L.GROUP_ORDER) + sorted(set(groups) - set(L.GROUP_ORDER)).index(g))
            for g in groups}
    _W.update(members=members, seed=args.seed, gidx=gidx, group_fit=group_fit)
    step = max(1, args.n_boot // max(1, 4 * args.n_jobs // max(1, len(groups))))
    tasks = [(g, list(range(s0, min(s0 + step, args.n_boot))))
             for g in groups for s0 in range(0, args.n_boot, step)]
    print(f"Bootstrap: {len(groups)} groups x {args.n_boot} resamples "
          f"({len(groups) * args.n_boot} fits, {args.n_jobs} job(s))")
    rows = []
    if args.n_jobs > 1:
        with mp.get_context("fork").Pool(args.n_jobs) as P:
            chunks = list(P.imap_unordered(_boot_chunk, tasks))
    else:
        chunks = [_boot_chunk(t) for t in tasks]
    for chunk in chunks:
        for g, b, tiv, sim in chunk:
            for i in range(k):
                rows.append(dict(group=g, n_mice=len(members[g]), draw=b, component=i + 1,
                                 traveling_index=tiv[i], match_similarity=sim[i]))
    B = (pd.DataFrame(rows).sort_values(["group", "component", "draw"],
                                        key=lambda s: s.map(gidx) if s.name == "group" else s)
         .reset_index(drop=True))
    B.to_csv(os.path.join(args.out_dir, "cpca_ti_bootstrap.csv"), index=False)

    summ = []
    for g in groups:
        for i in range(k):
            v = B[(B.group == g) & (B.component == i + 1)]
            summ.append(dict(group=g, n_mice=len(members[g]), component=i + 1,
                             traveling_index=obs[g][0][i], match_similarity=obs[g][1][i],
                             n_boot=len(v), boot_median=v.traveling_index.median(),
                             ci_lo=v.traveling_index.quantile(0.025),
                             ci_hi=v.traveling_index.quantile(0.975),
                             frac_match_below=(v.match_similarity < MATCH_MIN).mean()))
    S = pd.DataFrame(summ)
    S.to_csv(os.path.join(args.out_dir, "cpca_ti_bootstrap_summary.csv"), index=False)
    print("\nTraveling index, 95% bootstrap percentile interval:")
    print(S.round(3).to_string(index=False))
    print(f"Done. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
