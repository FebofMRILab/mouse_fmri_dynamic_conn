"""
Welch tests computed by SciPy (SciPy >= 1.16). Replaces welch_permutation.py.

Omnibus tests (7 groups; human: 2 groups), Welch's F with a permutation null:
  welch_anova_permutation            scipy.stats.permutation_test on
  welch_anova_edgewise_permutation   scipy.stats.f_oneway(..., equal_var=False)
                                     (labels permuted between groups;
                                     p = (1 + b) / (k + 1))
Pairwise contrasts, ordinary Welch t-test (Welch-Satterthwaite df, two-sided):
  welch_t_test                       scipy.stats.ttest_ind(a, b, equal_var=False)
  welch_t_test_df                    the same, also returning the df
  welch_t_edgewise                   the same for every column at once
Degrees of freedom of Welch's F, for reporting next to its permutation p:
  welch_anova_df                     statsmodels anova_oneway(use_var='unequal')
Within-animal comparisons (e.g. two HMM states in the same mice; not Welch):
  paired_signflip_test               scipy.stats.permutation_test, one sample of
                                     paired differences, sign flips
                                     (permutation_type='samples'), paired t
                                     statistic (scipy.stats.ttest_1samp); exact
                                     when 2**n <= n_permutations

Why Welch statistics: group sizes range from 5 to 25 and variances differ.
Why the ordinary (parametric) Welch t for pairwise contrasts rather than a
permutation of it: at our group sizes, with the 5-mouse group the more
variable, permuting Welch's t rejected 10-13% of true nulls at alpha = 0.05,
while the ordinary Welch t stayed near 5% (tests/simulate_type1_two_groups.py).
Permuting Welch's F for the 7-group omnibus stayed at 5-7%
(tests/simulate_type1_seven_groups.py).
"""

import numpy as np
import scipy
from scipy import stats

if tuple(int(v) for v in scipy.__version__.split(".")[:2]) < (1, 16):
    raise ImportError("SciPy >= 1.16 is needed (f_oneway with equal_var=False).")

_BATCH = 100        # permutations per vectorised batch (memory: batch x subjects x edges)


def _welch_f(*samples, axis=0):
    return stats.f_oneway(*samples, axis=axis, equal_var=False).statistic


def welch_anova_permutation(samples, n_permutations=10000, random_seed=0):
    """Welch's F and its permutation p-value (one test). Returns (F, p)."""
    samples = [np.asarray(x, dtype=float) for x in samples]
    samples = [x[np.isfinite(x)] for x in samples]
    with np.errstate(divide="ignore", invalid="ignore"):
        res = stats.permutation_test(
            samples, _welch_f, permutation_type="independent", vectorized=True,
            alternative="greater", n_resamples=n_permutations, rng=random_seed)
    f = float(res.statistic)
    return (f, float(res.pvalue)) if np.isfinite(f) else (f, np.nan)


def welch_anova_edgewise_permutation(data, group_masks, n_permutations=10000, random_seed=0):
    """Welch's F and permutation p for every column of data (subjects x edges).
    Returns (F, p) arrays."""
    data = np.asarray(data, dtype=float)
    samples = [data[np.asarray(mask, dtype=bool)] for mask in group_masks]
    with np.errstate(divide="ignore", invalid="ignore"):
        res = stats.permutation_test(
            samples, _welch_f, permutation_type="independent", vectorized=True,
            axis=0, alternative="greater", n_resamples=n_permutations,
            rng=random_seed, batch=_BATCH)
    f = np.asarray(res.statistic, dtype=float)
    p = np.asarray(res.pvalue, dtype=float)
    p[~np.isfinite(f)] = np.nan
    return f, p


def welch_t_test(a, b):
    """Welch's t (a minus b) and its two-sided p. Returns (t, p)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    with np.errstate(divide="ignore", invalid="ignore"):
        res = stats.ttest_ind(a, b, equal_var=False)
    t = float(res.statistic)
    return (t, float(res.pvalue)) if np.isfinite(t) else (t, np.nan)


def welch_t_test_df(a, b):
    """Welch's t (a minus b), its Welch-Satterthwaite df and two-sided p.
    Returns (t, df, p)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    with np.errstate(divide="ignore", invalid="ignore"):
        res = stats.ttest_ind(a, b, equal_var=False)
    t = float(res.statistic)
    if not np.isfinite(t):
        return t, np.nan, np.nan
    return t, float(res.df), float(res.pvalue)


def welch_anova_df(samples):
    """Numerator and denominator df of Welch's F (statsmodels). Returns (df1, df2)."""
    from statsmodels.stats.oneway import anova_oneway
    samples = [np.asarray(x, dtype=float) for x in samples]
    samples = [x[np.isfinite(x)] for x in samples]
    with np.errstate(divide="ignore", invalid="ignore"):
        res = anova_oneway(samples, use_var="unequal", welch_correction=True)
    return float(res.df_num), float(res.df_denom)


def welch_t_edgewise(group_a, group_b):
    """Welch's t (a minus b) and two-sided p for every column of
    (subjects x variables) arrays. Returns (t, p) arrays."""
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        res = stats.ttest_ind(a, b, axis=0, equal_var=False)
    t = np.asarray(res.statistic, dtype=float)
    p = np.asarray(res.pvalue, dtype=float)
    p[~np.isfinite(t)] = np.nan
    return t, p


def paired_signflip_test(differences, n_permutations=10000, random_seed=0):
    """Two-sided sign-flip permutation test that the mean within-animal
    difference is 0, with the paired t statistic. Returns (t, p, n);
    NaN for fewer than 3 animals or no spread."""
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 3 or np.std(d, ddof=1) == 0:
        return np.nan, np.nan, n

    def paired_t(x, axis=-1):
        return stats.ttest_1samp(x, 0.0, axis=axis).statistic

    with np.errstate(divide="ignore", invalid="ignore"):
        res = stats.permutation_test(
            (d,), paired_t, permutation_type="samples", vectorized=True,
            alternative="two-sided", n_resamples=n_permutations, rng=random_seed)
    return float(res.statistic), float(res.pvalue), n

