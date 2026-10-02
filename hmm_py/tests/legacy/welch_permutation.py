"""
Permutation versions of Welch's ANOVA and Welch's t-test.

The test statistic is unchanged (Welch's F or t, so unequal variances are
still handled); only the null distribution changes, from the theoretical F or
t distribution to the distribution obtained by permuting group labels. This
removes the normality assumption while keeping the heteroscedasticity
correction, which matters when a group is small (e.g. n = 5) and the
asymptotic argument for the parametric version is thin.

Studentized permutation tests of this kind are asymptotically valid under
unequal variances (Janssen, 1997), unlike permutation tests on unstudentized
mean differences.

Functions
---------
welch_f_statistic        Welch's F for a list of samples.
welch_anova_permutation  Permutation p-value for Welch's F (one omnibus test).
welch_t_permutation      Permutation p-value for Welch's t (two groups).
welch_anova_edgewise_permutation
                         Vectorized version for many edges or ROIs at once.
n_distinct_labelings     Number of distinct relabelings, for reporting the
                         smallest attainable p-value with small groups.
"""

from math import comb

import numpy as np


def welch_f_statistic(samples):
    """Welch's F. NaN if any group has < 2 observations or zero variance."""
    n = np.array([len(x) for x in samples], dtype=float)
    m = np.array([np.mean(x) for x in samples], dtype=float)
    v = np.array([np.var(x, ddof=1) for x in samples], dtype=float)

    if np.any(n < 2) or np.any(v <= 0) or not np.all(np.isfinite(v)):
        return np.nan

    k = len(samples)
    w = n / v
    total_w = w.sum()
    grand_mean = (w * m).sum() / total_w

    numerator = (w * (m - grand_mean) ** 2).sum() / (k - 1)
    lam = (((1 - w / total_w) ** 2) / (n - 1)).sum()
    denominator = 1 + 2 * (k - 2) / (k ** 2 - 1) * lam

    return float(numerator / denominator)


def welch_anova_permutation(samples, n_permutations=10000, random_seed=0):
    """
    Permutation p-value for Welch's F across >= 2 groups.

    Returns (F, p). The p-value is (1 + #{F* >= F}) / (n_permutations + 1),
    so it is never exactly zero. Permutations in which the statistic is
    undefined (a group with zero variance) are counted as not exceeding F.
    """
    samples = [np.asarray(x, dtype=float) for x in samples]
    samples = [x[np.isfinite(x)] for x in samples]

    observed = welch_f_statistic(samples)
    if not np.isfinite(observed):
        return observed, np.nan

    sizes = np.array([len(x) for x in samples])
    pooled = np.concatenate(samples)
    split_points = np.cumsum(sizes)[:-1]

    rng = np.random.default_rng(random_seed)
    exceed = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(pooled)
        f_null = welch_f_statistic(np.split(shuffled, split_points))
        if np.isfinite(f_null) and f_null >= observed:
            exceed += 1

    return observed, (1 + exceed) / (n_permutations + 1)


def welch_t_permutation(a, b, n_permutations=10000, random_seed=0):
    """
    Two-sided permutation p-value for Welch's t (group a minus group b).

    Returns (t, p). Equivalent to welch_anova_permutation for two groups
    (F = t^2), but returns the signed statistic.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]

    def statistic(x, y):
        n1, n2 = len(x), len(y)
        v1, v2 = np.var(x, ddof=1), np.var(y, ddof=1)
        denom = np.sqrt(v1 / n1 + v2 / n2)
        return np.nan if denom <= 0 else (np.mean(x) - np.mean(y)) / denom

    observed = statistic(a, b)
    if not np.isfinite(observed):
        return observed, np.nan

    pooled = np.concatenate([a, b])
    n_a = len(a)

    rng = np.random.default_rng(random_seed)
    exceed = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(pooled)
        t_null = statistic(shuffled[:n_a], shuffled[n_a:])
        if np.isfinite(t_null) and abs(t_null) >= abs(observed):
            exceed += 1

    return observed, (1 + exceed) / (n_permutations + 1)


def welch_anova_edgewise_permutation(
    data, group_masks, n_permutations=10000, random_seed=0
):
    """
    Welch's F and permutation p-values for every column of `data` at once.

    data         (subjects, variables) array; variables are edges or ROIs.
    group_masks  list of boolean masks over subjects, one per group.

    The same relabeling is applied to every variable in a permutation, which
    preserves the correlation between edges. Returns (F, p) arrays.
    """
    data = np.asarray(data, dtype=float)
    n_subjects = data.shape[0]
    labels = np.zeros(n_subjects, dtype=int)
    for index, mask in enumerate(group_masks):
        labels[np.asarray(mask, dtype=bool)] = index

    k = len(group_masks)

    def f_all(label_vector):
        n = np.empty((k, data.shape[1]))
        m = np.empty_like(n)
        v = np.empty_like(n)
        for index in range(k):
            block = data[label_vector == index]
            n[index] = np.sum(np.isfinite(block), axis=0)
            m[index] = np.nanmean(block, axis=0)
            v[index] = np.nanvar(block, axis=0, ddof=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            w = n / v
            total_w = w.sum(axis=0)
            grand_mean = (w * m).sum(axis=0) / total_w
            numerator = (w * (m - grand_mean) ** 2).sum(axis=0) / (k - 1)
            lam = (((1 - w / total_w) ** 2) / (n - 1)).sum(axis=0)
            denominator = 1 + 2 * (k - 2) / (k ** 2 - 1) * lam
            return numerator / denominator

    observed = f_all(labels)
    rng = np.random.default_rng(random_seed)
    exceed = np.zeros(data.shape[1])

    for _ in range(n_permutations):
        f_null = f_all(rng.permutation(labels))
        exceed += np.nan_to_num(f_null, nan=-np.inf) >= np.nan_to_num(
            observed, nan=np.inf
        )

    p_values = (1 + exceed) / (n_permutations + 1)
    p_values[~np.isfinite(observed)] = np.nan
    return np.nan_to_num(observed, nan=np.nan), p_values


def welch_t_edgewise_permutation(
    group_a, group_b, n_permutations=10000, random_seed=0
):
    """
    Welch's t and two-sided permutation p-values for every column at once.

    group_a, group_b  (subjects, variables) arrays for the two groups; the
                      variables are edges, ROIs or metrics.

    Only the two groups being compared are relabeled, which is the null the
    contrast tests. The same relabeling is applied to every variable in a
    permutation, preserving the correlation between edges.

    Returns (t, p) arrays. p is (1 + #{|t*| >= |t|}) / (n_permutations + 1).
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    n_a, n_b = a.shape[0], b.shape[0]
    pooled = np.vstack([a, b])

    def t_all(matrix):
        x, y = matrix[:n_a], matrix[n_a:]
        with np.errstate(divide="ignore", invalid="ignore"):
            denominator = np.sqrt(
                np.nanvar(x, axis=0, ddof=1) / n_a
                + np.nanvar(y, axis=0, ddof=1) / n_b
            )
            return (np.nanmean(x, axis=0) - np.nanmean(y, axis=0)) / denominator

    observed = t_all(pooled)
    rng = np.random.default_rng(random_seed)
    exceed = np.zeros(pooled.shape[1])

    for _ in range(n_permutations):
        t_null = t_all(pooled[rng.permutation(n_a + n_b)])
        exceed += np.nan_to_num(np.abs(t_null), nan=-np.inf) >= np.nan_to_num(
            np.abs(observed), nan=np.inf
        )

    p_values = (1 + exceed) / (n_permutations + 1)
    p_values[~np.isfinite(observed)] = np.nan
    return observed, p_values

def welch_df(samples):
    """Numerator and Welch-approximated denominator df for Welch's ANOVA."""
    n = np.array([len(x) for x in samples], dtype=float)
    v = np.array([np.var(x, ddof=1) for x in samples], dtype=float)
    k = len(samples)
    w = n / v
    lam = (((1 - w / w.sum()) ** 2) / (n - 1)).sum()
    return k - 1, (k ** 2 - 1) / (3 * lam)
    
   
def n_distinct_labelings(group_sizes):
    """
    Number of distinct assignments of subjects to groups of these sizes.
    The smallest attainable exact p-value is 1 / this number; a permutation
    p-value below it is an artefact of sampling with replacement.
    """
    total = int(sum(group_sizes))
    count = 1
    remaining = total
    for size in group_sizes[:-1]:
        count *= comb(remaining, int(size))
        remaining -= int(size)
    return count