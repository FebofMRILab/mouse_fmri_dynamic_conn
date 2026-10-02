"""
run_nbs: the network-based statistic for one planned comparison, as two
one-sided contrasts, computed by bctpy's nbs_bct (Zalesky et al. 2010).

The NBS itself is bctpy 0.6.1's code (bctpy_nbs_welch.py) with the three
documented changes F01-F03 (Welch t, (1 + b)/(k + 1) p-values, no error when
no edge passes the threshold). This module only reshapes the inputs and
outputs to what the analysis scripts use:

    observed_t, components, null_max_extents = run_nbs(A, B, threshold, k, seed)

    A, B               (subjects, ROI, ROI) Fisher-z matrices of the two groups
    observed_t         edgewise Welch t (A - B), upper triangle in
                       np.triu_indices(n, 1) order (scipy.stats.ttest_ind)
    components         list of dicts: nodes, edge_indices, extent (edges),
                       tail ('positive' = A > B, 'negative' = A < B), p_FWER
    null_max_extents   {'positive': array, 'negative': array} of the maximum
                       component extent per permutation; with
                       return_supra_counts=True also 'supra_counts'

Both contrasts use the same seed, so they see the same relabelings (bctpy draws
one permutation per iteration from RandomState(seed)).
"""

import numpy as np
from scipy import stats

from bctpy_nbs_welch import nbs_bct

NBS_TAILS = {"positive": "right", "negative": "left"}


def run_nbs(
    matrices_group_a,
    matrices_group_b,
    primary_threshold=3.1,
    n_permutations=5000,
    random_seed=20260911,
    return_supra_counts=False,
):
    a = np.asarray(matrices_group_a, dtype=float)
    b = np.asarray(matrices_group_b, dtype=float)
    n = a.shape[1]
    iu = np.triu_indices(n, k=1)
    edge_a, edge_b = a[:, iu[0], iu[1]], b[:, iu[0], iu[1]]

    observed_t = stats.ttest_ind(edge_a, edge_b, axis=0, equal_var=False).statistic
    observed_t = np.nan_to_num(observed_t, nan=0.0, posinf=0.0, neginf=0.0)

    x = np.moveaxis(a, 0, -1)          # bctpy layout: ROI x ROI x subjects
    y = np.moveaxis(b, 0, -1)

    components, null_max_extents = [], {}
    for name, tail in NBS_TAILS.items():
        pvals, adj, null = nbs_bct(x, y, primary_threshold, k=n_permutations,
                                   tail=tail, seed=random_seed)
        null_max_extents[name] = np.asarray(null, dtype=int)
        labels = adj[iu]
        for c, p in enumerate(np.atleast_1d(pvals)):
            edge_indices = np.flatnonzero(labels == c + 1)
            nodes = np.unique(np.concatenate([iu[0][edge_indices], iu[1][edge_indices]]))
            components.append({
                "nodes": nodes.tolist(),
                "edge_indices": edge_indices,
                "extent": int(edge_indices.size),
                "tail": name,
                "p_FWER": float(p),
            })
    components.sort(key=lambda comp: comp["extent"], reverse=True)

    if return_supra_counts:
        null_max_extents["supra_counts"] = _replay_supra_counts(
            edge_a, edge_b, primary_threshold, n_permutations, random_seed)

    return observed_t, components, null_max_extents


def _replay_supra_counts(edge_a, edge_b, threshold, n_permutations, seed):
    """Number of suprathreshold edges per permutation, for the chance baseline.
    Replays nbs_bct's relabelings exactly: RandomState(seed), one
    permutation(n_a + n_b) per iteration, group A = first n_a columns."""
    rng = np.random.RandomState(seed)
    combined = np.vstack([edge_a, edge_b])
    n_a, n_total = edge_a.shape[0], combined.shape[0]
    counts = {name: np.zeros(n_permutations, dtype=int) for name in NBS_TAILS}
    for u in range(n_permutations):
        order = rng.permutation(n_total)
        t = stats.ttest_ind(combined[order[:n_a]], combined[order[n_a:]],
                            axis=0, equal_var=False).statistic
        t = np.nan_to_num(t, nan=0.0)
        counts["positive"][u] = int(np.sum(t > threshold))
        counts["negative"][u] = int(np.sum(-t > threshold))
    return counts
