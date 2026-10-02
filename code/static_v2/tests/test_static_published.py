#!/usr/bin/env python3
"""
Equivalence and validity tests for the static-FC changes (see CHANGES.md).

    python tests/test_static_published.py            # from the static_v2 folder

S1  Welch's F: scipy f_oneway(equal_var=False) vs the previous code vs
    statsmodels anova_oneway(use_var='unequal')                 must be equal
S2  Welch's t: scipy ttest_ind(equal_var=False) vs the previous code   equal
S3  Permutation p (omnibus), previous vs SciPy; Welch t-test wrapper   report/equal
S4  Type I error under equal means and unequal variances, our group
    sizes: permutation of pooled t vs permutation of Welch t           report
S5  Proportional threshold: previous code vs bctpy threshold_proportional
S6  Nodal efficiency: previous code vs networkx harmonic centrality   equal
S7  NBS (patched bctpy): observed components, null and p-values
    reproduced independently from scipy + bctpy get_components        equal
S8  NBS F03: nothing above threshold -> no component, null returned   pass
S9  NBS: previous custom vs patched bctpy on the same data            report
"""

import os
import sys
import time
import warnings

import networkx as nx
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE, "legacy"))

import bct                                             # noqa: E402
import welch_permutation as old                        # noqa: E402  previous module
import welch_tests_scipy as new                        # noqa: E402
import legacy_static as leg                            # noqa: E402
from bctpy_nbs_welch import nbs_bct                    # noqa: E402
from nbs_published import run_nbs                      # noqa: E402
from statsmodels.stats.oneway import anova_oneway      # noqa: E402

warnings.filterwarnings("ignore")
TOL = 1e-10
SIZES = [19, 9, 25, 13, 5, 12, 12]                     # mouse groups
rows = []


def report(name, value, passed, note=""):
    tag = "report" if passed is None else ("PASS" if passed else "FAIL")
    rows.append((name, value, tag, note))
    print(f"{name:<70s} {value:>12.4g}  {tag}  {note}", flush=True)


rng = np.random.default_rng(1)

# ---------- S1 Welch F ----------
d = 0.0
for _ in range(200):
    groups = [rng.normal(rng.normal(), rng.uniform(0.5, 3), n) for n in SIZES]
    f_new = stats.f_oneway(*groups, equal_var=False).statistic
    f_old = old.welch_f_statistic(groups)
    f_sm = anova_oneway(groups, use_var="unequal", welch_correction=True).statistic
    d = max(d, abs(f_new - f_old) / f_old, abs(f_sm - f_old) / f_old)
report("S1 Welch F: scipy / statsmodels vs previous (max rel. diff)", d, d < TOL)

# ---------- S2 Welch t ----------
a = rng.normal(0, 1, (19, 300)); b = rng.normal(0.2, 2, (5, 300))
t_new = stats.ttest_ind(a, b, axis=0, equal_var=False).statistic
t_old = np.array([(x.mean() - y.mean()) / np.sqrt(x.var(ddof=1) / 19 + y.var(ddof=1) / 5)
                  for x, y in zip(a.T, b.T)])
t_nbs = leg.welch_t_vectorized(a, b)
d = max(np.max(np.abs(t_new - t_old)), np.max(np.abs(t_new - t_nbs)))
report("S2 Welch t: scipy vs previous (max diff)", d, d < TOL)

# ---------- S3 permutation p-values, previous vs SciPy ----------
X = np.vstack([rng.normal(0.15 * i, 1 + 0.3 * i, (n, 60)) for i, n in enumerate(SIZES)])
masks = [np.repeat(np.arange(7), SIZES) == g for g in range(7)]
t0 = time.time(); f1, p1 = old.welch_anova_edgewise_permutation(X, masks, 5000, 3); t_old = time.time() - t0
t0 = time.time(); f2, p2 = new.welch_anova_edgewise_permutation(X, masks, 5000, 3); t_new = time.time() - t0
report("S3a edgewise Welch F: max |F diff|", np.max(np.abs(f1 - f2)), np.max(np.abs(f1 - f2)) < 1e-8)
mc = np.max(np.abs(p1 - p2) / np.sqrt(2 * np.maximum(p1 * (1 - p1), 1e-4) / 5000))
report("S3b edgewise Welch F perm. p: max |dp| in SDs of a difference", mc, None,
       f"max |dp| = {np.max(np.abs(p1 - p2)):.4f}; time old {t_old:.1f}s, scipy {t_new:.1f}s")
A, B = X[masks[1]], X[masks[4]]                        # 9 vs 5: 2002 relabelings
t1, q1 = old.welch_t_edgewise_permutation(A, B, 5000, 3)
t2, q2 = new.welch_t_edgewise(A, B)
report("S3c 9 vs 5 Welch t: max |t diff|", np.max(np.abs(t1 - t2)), np.max(np.abs(t1 - t2)) < 1e-8)
report("S3d 9 vs 5: previous permutation p vs Welch t-test p: max |dp|", np.max(np.abs(q1 - q2)),
       None, "the pairwise contrasts now use the ordinary Welch t-test (see S4)")
tt, pp = new.welch_t_test(A[:, 0], B[:, 0])
ref = stats.ttest_ind(A[:, 0], B[:, 0], equal_var=False)
report("S3e welch_t_test equals scipy.stats.ttest_ind(equal_var=False)",
       abs(tt - ref.statistic) + abs(pp - ref.pvalue), abs(tt - ref.statistic) + abs(pp - ref.pvalue) < 1e-12)

# ---------- S4 type I error: pooled vs Welch permutation, equal means ----------
n_sim, k = 2000, 999
for (na, nb), sd_ratio in [((9, 5), 1), ((9, 5), 3), ((9, 5), 1 / 3), ((19, 5), 3), ((19, 5), 1 / 3)]:
    xa = rng.normal(0, 1, (na, n_sim)); xb = rng.normal(0, sd_ratio, (nb, n_sim))
    pm = stats.PermutationMethod(n_resamples=k, rng=11)
    p_pool = stats.ttest_ind(xa, xb, axis=0, equal_var=True, method=pm).pvalue
    p_welch = stats.ttest_ind(xa, xb, axis=0, equal_var=False, method=pm).pvalue
    p_param = stats.ttest_ind(xa, xb, axis=0, equal_var=False).pvalue
    report(f"S4 n={na} vs {nb}, SD of n={nb} group x{sd_ratio:.2g}: type I, pooled t perm.",
           np.mean(p_pool < 0.05), None, f"Welch t perm. {np.mean(p_welch < 0.05):.3f}; "
           f"Welch t parametric {np.mean(p_param < 0.05):.3f} (nominal 0.05; {n_sim} null datasets)")

# ---------- S5 threshold ----------
n = 82
leg.set_size(n)
diff_edges, n_old, n_new = 0, None, None
for _ in range(20):
    L = rng.normal(size=(n, 6)); C = np.corrcoef(rng.normal(size=(200, n)) + rng.normal(size=(200, 6)) @ L.T, rowvar=False)
    for dens in (0.10, 0.15, 0.20):
        w_old, b_old = leg.proportional_threshold_positive(C, dens)
        w_new = bct.threshold_proportional(np.where(C > 0, C, 0), dens)
        b_new = bct.binarize(w_new)
        n_old, n_new = int(np.triu(b_old, 1).sum()), int(np.triu(b_new, 1).sum())
        extra = np.triu(b_old, 1).astype(bool) & ~np.triu(b_new, 1).astype(bool)
        missing = np.triu(b_new, 1).astype(bool) & ~np.triu(b_old, 1).astype(bool)
        assert missing.sum() == 0 and extra.sum() == n_old - n_new
        diff_edges = max(diff_edges, n_old - n_new)
report("S5 threshold: edges kept by previous code but not by BCT (max)", diff_edges, None,
       "previous keeps ceil(p*E), BCT round(p*E); every BCT edge is also in the previous set")

# ---------- S6 nodal efficiency ----------
d = 0.0
for dens in (0.10, 0.15, 0.20):
    _, b_old = leg.proportional_threshold_positive(C, dens)
    e_old = leg.nodal_efficiency(b_old)
    G = nx.from_numpy_array(b_old)
    hc = nx.harmonic_centrality(G)
    e_new = np.array([hc[i] for i in range(n)]) / (n - 1)
    d = max(d, np.max(np.abs(e_old - e_new)))
report("S6 nodal efficiency: networkx harmonic/(n-1) vs previous (max diff)", d, d < TOL)

# ---------- S7 NBS patched bctpy, reproduced independently ----------
nn, na, nb, k = 30, 9, 5, 300
leg.set_size(nn)
iu = np.triu_indices(nn, 1)
base = rng.normal(size=(na + nb, nn, nn)); base = (base + base.transpose(0, 2, 1)) / 2
base[:na, :8, :8] += 1.2                                # group A higher in an 8-node block
Ma, Mb = base[:na], base[na:]
thr = 2.5
obs_t, comps, nulls = run_nbs(Ma, Mb, thr, k, 77, return_supra_counts=True)
# independent reconstruction
ea, eb = Ma[:, iu[0], iu[1]], Mb[:, iu[0], iu[1]]
ok = True
for name, sign in (("positive", 1), ("negative", -1)):
    rs = np.random.RandomState(77)
    comb_ = np.vstack([ea, eb]); null = np.zeros(k)
    for u in range(k):
        o = rs.permutation(na + nb)
        t = np.nan_to_num(stats.ttest_ind(comb_[o[:na]], comb_[o[na:]], axis=0, equal_var=False).statistic)
        null[u] = leg.max_component_extent(t, thr, sign)
    ok &= np.array_equal(null.astype(int), nulls[name])
    obs = leg.suprathreshold_components(obs_t, thr, sign)
    mine = sorted([c["extent"] for c in comps if c["tail"] == name], reverse=True)
    ok &= mine == sorted([c["extent"] for c in obs], reverse=True)
    for c in [c for c in comps if c["tail"] == name]:
        ok &= abs(c["p_FWER"] - (1 + np.sum(null >= c["extent"])) / (k + 1)) < 1e-12
report("S7 NBS: null, components and p reproduced from scipy t + extents", 0 if ok else 1, ok,
       f"components found: {[(c['tail'], c['extent'], round(c['p_FWER'], 4)) for c in comps][:4]}")

# ---------- S8 F03 ----------
try:
    _, comps8, nulls8 = run_nbs(Ma, Mb, 50.0, 50, 1)
    ok8 = (comps8 == []) and nulls8["positive"].size == 50
except Exception as e:                                  # noqa: BLE001
    ok8 = False
report("S8 NBS with nothing above threshold: no component, null returned", 0 if ok8 else 1, ok8)

# ---------- S9 previous custom NBS vs patched bctpy ----------
o_t, o_comps, o_null = leg.run_nbs(Ma, Mb, thr, 2000, 5)
n_t, n_comps, n_null = run_nbs(Ma, Mb, thr, 2000, 5)
same_obs = sorted((c["tail"], c["extent"]) for c in o_comps) == sorted((c["tail"], c["extent"]) for c in n_comps)
report("S9 NBS previous vs patched bctpy: identical observed components", 0 if same_obs else 1, same_obs,
       "nulls differ only by random stream: 95th pct extent previous "
       f"{np.percentile(o_null['positive'], 95):.0f} vs bctpy {np.percentile(n_null['positive'], 95):.0f}; "
       f"top p {o_comps[0]['p_FWER']:.4f} vs {n_comps[0]['p_FWER']:.4f}" if o_comps and n_comps else "")

n_fail = sum(r[2] == "FAIL" for r in rows)
print(f"\n{len(rows)} checks, {n_fail} failed")
sys.exit(1 if n_fail else 0)
