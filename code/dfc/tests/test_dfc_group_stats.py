#!/usr/bin/env python3
"""
Group statistics of plot_dfc_results.py: updated (welch_tests_scipy.py +
statsmodels) vs previous (the script's own code), on simulated 7-group data
with the study's group sizes and unequal variances.

    python tests/test_dfc_group_stats.py        # from the dfc folder

Must be equal:  Holm and BH adjustments; Welch F and its df; contrast Welch t,
                df and mean difference; NaN for groups with < 2 values or no
                spread.
Reported:       omnibus permutation p (both permutation tests, different
                random draws: Monte Carlo difference only); contrast p, which
                changes from a permutation of Welch's t to the ordinary Welch t
                (the study's decision; see static_v2/tests/simulate_type1_*).
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE, "legacy"))
warnings.filterwarnings("ignore")
import plot_dfc_results as new                       # noqa: E402
import plot_dfc_results_previous as old              # noqa: E402

rng = np.random.default_rng(7)
sizes = {"NTG_Y": 19, "NTG_M": 9, "NTG_A": 25, "CRND8_Y": 13, "CRND8_M": 5,
         "5XFAD_Y": 12, "5XFAD_M": 12}
sds = {"NTG_Y": 1, "NTG_M": 1.5, "NTG_A": 0.8, "CRND8_Y": 1.2, "CRND8_M": 3,
       "5XFAD_Y": 1, "5XFAD_M": 2}
groups = list(sizes)
rows = []
for m in range(6):
    shift = {g: (0.8 if (m % 2 and g.startswith("5XFAD")) else 0.0) for g in groups}
    for g in groups:
        for v in rng.normal(shift[g], sds[g], sizes[g]):
            rows.append(dict(metric=f"m{m}", group=g, value=v))
d = pd.DataFrame(rows)
# a metric with one constant group (both versions must return NaN)
d.loc[(d.metric == "m5") & (d.group == "CRND8_M"), "value"] = 1.0
metrics = [f"m{m}" for m in range(6)]
n_perm = 10000

o_new, c_new = new.group_tests(d, metrics, "value", groups, "test", n_perm)
o_old, c_old = old.group_tests(d, metrics, "value", groups, "test", n_perm)

res = []
def check(name, value, ok):
    res.append((name, value, "PASS" if ok else "FAIL"))
def report(name, value):
    res.append((name, value, "report"))

p = rng.uniform(0, 0.2, 40)
p[[3, 17]] = np.nan
d_holm = np.nanmax(np.abs(new.holm(p) - old.holm(p)))
d_bh = np.nanmax(np.abs(new.bh(p) - old.bh(p)))
check("Holm: statsmodels vs previous code (max diff)", d_holm, d_holm < 1e-14)
check("BH: statsmodels vs previous code (max diff)", d_bh, d_bh < 1e-14)

ok = o_old.welch_F.notna()
dF = np.max(np.abs(o_new.welch_F[ok] - o_old.welch_F[ok]) / o_old.welch_F[ok])
dd2 = np.max(np.abs(o_new.df2[ok] - o_old.df2[ok]))
check("Welch F, scipy vs previous code (max rel. diff)", dF, dF < 1e-12)
check("Welch F denominator df, statsmodels vs previous (max diff)", dd2, dd2 < 1e-9)
same_nan = (o_new.welch_F.isna() == o_old.welch_F.isna()).all() and o_new.welch_F.isna().sum() == 1
check("NaN for the metric with a constant group (omnibus)", int(o_new.welch_F.isna().sum()), same_nan)
dp = np.max(np.abs(o_new.p_perm[ok] - o_old.p_perm[ok]))
report("omnibus permutation p, scipy vs previous (max |dp|; Monte Carlo)", dp)

okc = c_old.welch_t.notna()
dt = np.max(np.abs(c_new.welch_t[okc] - c_old.welch_t[okc]))
ddf = np.max(np.abs(c_new.df[okc] - c_old.df[okc]))
dm = np.max(np.abs(c_new.mean_diff[okc] - c_old.mean_diff[okc]))
check("contrast Welch t, scipy vs previous (max diff)", dt, dt < 1e-12)
check("contrast Welch df, scipy vs previous (max diff)", ddf, ddf < 1e-9)
check("contrast mean difference (max diff)", dm, dm < 1e-12)
check("contrast NaN pattern unchanged", int(c_new.welch_t.isna().sum()),
      (c_new.welch_t.isna() == c_old.welch_t.isna()).all())
dpc = c_new.p_welch[okc] - c_old.p_perm[okc]
report("contrast p: ordinary Welch t minus permutation of Welch t (median)", float(np.median(dpc)))
report("contrast p: largest |difference|", float(np.max(np.abs(dpc))))
small = c_new.n_b[okc].eq(5) | c_new.n_a[okc].eq(5)
report("  ... in contrasts with the 5-mouse group (median difference)", float(np.median(dpc[small])))

w = max(len(r[0]) for r in res)
print(f"{'Check':<{w}}  {'Value':>12}  Result")
for name, v, r in res:
    print(f"{name:<{w}}  {v:>12.4g}  {r}")
n_fail = sum(r[2] == "FAIL" for r in res)
print(f"\n{sum(r[2] == 'PASS' for r in res)} passed, {n_fail} failed.")
sys.exit(1 if n_fail else 0)
