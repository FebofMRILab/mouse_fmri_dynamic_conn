#!/usr/bin/env python3
"""
Tests of the Hodges-Lehmann estimate and CI, the rank-biserial correlation and
the K = 2 drift fix in analyze_confirm_lowrank.py (27 Oct 2026; E50, E51).

    python tests/hodges_lehmann/test_hodges_lehmann.py      # from hmm_null/

  H1  exact null of the signed-rank statistic: n = 30 lower two-sided 0.05
      critical value 137 (published tables), and the distribution matches
      scipy's exact Wilcoxon p for every attainable statistic
  H2  the CI is the set of shifts the two-sided exact Wilcoxon test does not
      reject: just inside each end p > 0.05, just outside p <= 0.05
      (n = 8, 15, 30, 50; 200 samples each)
  H3  estimate = median of all Walsh averages (brute force)
  H4  coverage of the true centre over 4000 simulated samples (n = 30,
      symmetric t3 errors) within Monte Carlo error of the achieved coverage
  H5  n > 50: the normal-approximation critical value is within 1 of the exact
      one (n = 51, 60, 80, 120)
  H6  rank-biserial: +1 when all positive, -1 when all negative, equals
      2 T+/M - 1 without zeros
  H7  one-sided exact p < 0.025 (i.e. Holm p < 0.05 at m >= 2) <=> CI lower
      end > 0, on 500 samples
  H8  drift: K = 2 counted once (both rows carry state 1's p, no Holm
      doubling) in analyze_confirm_lowrank.py and hmm_occupancy_drift.py
      alike; K = 3 unchanged from the previous code (Holm within K)
  H9  K = 2 occupancies that are not complements (state 2 perturbed): both
      tested with Holm, as before, and flagged
"""
import os
import sys
import tempfile
import types

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
NULLDIR = os.path.dirname(os.path.dirname(HERE))
CODE = os.path.dirname(NULLDIR)
sys.path.insert(0, NULLDIR)
sys.path.insert(0, os.path.join(CODE, "hmm_v2"))
import analyze_confirm_lowrank as acl          # noqa: E402
import hmm_occupancy_drift as hod              # noqa: E402

res = []


def check(name, value, ok):
    res.append((name, value, "PASS" if ok else "FAIL"))


rng = np.random.default_rng(20261027)

# H1
cdf = acl._signed_rank_null_cdf(30)
crit = int(np.nonzero(cdf <= 0.025)[0][-1])
maxd = 0.0
for n in (6, 10, 20, 30):
    c = acl._signed_rank_null_cdf(n)
    x = rng.normal(0.3, 1, n)
    r = stats.rankdata(np.abs(x))
    t = int(r[x > 0].sum())
    p_scipy = stats.wilcoxon(x, alternative="less", method="exact").pvalue   # P(T+ <= t)
    maxd = max(maxd, abs(c[t] - p_scipy))
check(f"H1 n = 30 critical value {crit} (table: 137); P(T+ <= t) vs scipy exact (max diff)",
      maxd, crit == 137 and maxd < 1e-12)

# H2
bad = 0
for n in (8, 15, 30, 50):
    for _ in range(200):
        x = rng.standard_t(3, n) + rng.normal(0, 2)
        est, lo, hi, cov = acl.hodges_lehmann(x)
        eps = 1e-9 * max(1.0, np.abs(x).max())
        p = lambda th: stats.wilcoxon(x - th, method="exact").pvalue     # noqa: E731
        if not (p(lo + eps) > 0.05 and p(hi - eps) > 0.05
                and p(lo - eps) <= 0.05 and p(hi + eps) <= 0.05):
            bad += 1
check("H2 CI = shifts not rejected by the exact two-sided Wilcoxon test (failures of 800)",
      bad, bad == 0)

# H3
dmax = 0.0
for n in (6, 9, 30):
    x = rng.normal(size=n)
    w = [(x[i] + x[j]) / 2 for i in range(n) for j in range(i, n)]
    dmax = max(dmax, abs(acl.hodges_lehmann(x)[0] - np.median(w)))
check("H3 estimate = median of Walsh averages (max diff)", dmax, dmax < 1e-14)

# H4
hits, covs = 0, []
for _ in range(4000):
    x = 0.7 + rng.standard_t(3, 30)
    _, lo, hi, cov = acl.hodges_lehmann(x)
    hits += lo <= 0.7 <= hi
    covs.append(cov)
cover, target = hits / 4000, float(np.mean(covs))
se = np.sqrt(target * (1 - target) / 4000)
check(f"H4 coverage {cover:.4f} vs achieved {target:.4f} (|z|)", abs(cover - target) / se,
      abs(cover - target) < 3 * se)

# H5
diffs = []
for n in (51, 60, 80, 120):
    m = n * (n + 1) // 2
    d_exact = int(np.nonzero(acl._signed_rank_null_cdf(n) <= 0.025)[0][-1]) + 1
    x = rng.normal(size=n)
    i, j = np.triu_indices(n)
    w = np.sort((x[i] + x[j]) / 2)
    _, lo, _, _ = acl.hodges_lehmann(x)
    d_approx = int(np.searchsorted(w, lo)) + 1
    diffs.append(abs(d_exact - d_approx))
check(f"H5 n > 50: normal-approximation critical index vs exact (max |diff|, n 51-120)",
      max(diffs), max(diffs) <= 1)

# H6
x = rng.normal(size=25)
r = stats.rankdata(np.abs(x))
m = 25 * 26 / 2
ok6 = (acl.rank_biserial(np.abs(x) + 0.1) == 1.0 and acl.rank_biserial(-np.abs(x) - 0.1) == -1.0
       and abs(acl.rank_biserial(x) - (2 * r[x > 0].sum() / m - 1)) < 1e-14)
check("H6 rank-biserial: +1, -1, and 2 T+/M - 1", float(ok6), ok6)

# H7
mism = 0
for _ in range(500):
    x = rng.normal(rng.uniform(-0.2, 0.8), 1, 30)
    p1 = stats.wilcoxon(x, alternative="greater", method="exact").pvalue
    lo = acl.hodges_lehmann(x)[1]
    mism += (p1 < 0.025) != (lo > 0)
check("H7 one-sided p < 0.025 <=> CI lower end > 0 (mismatches of 500)", mism, mism == 0)

# H8
rows = []
for s in range(1, 41):
    for q in range(1, 5):
        f1 = np.clip(0.5 + 0.03 * (q - 2.5) + rng.normal(0, 0.08), 0.01, 0.99)
        for k, f in ((1, f1), (2, 1 - f1)):
            rows.append(dict(subject_id=s, group="g", K=2, state=k, quarter=q,
                             fo_quarter=f, fo_total=np.nan))
occ2 = pd.DataFrame(rows)
occ2["fo_total"] = occ2.groupby(["subject_id", "state"]).fo_quarter.transform("mean")
rows = []
for s in range(1, 41):
    fo = rng.dirichlet(np.ones(3) * 3)
    for q in range(1, 5):
        f = np.clip(fo + np.array([0.03, -0.03, 0]) * (q - 2.5) + rng.normal(0, 0.05, 3), 1e-3, None)
        f /= f.sum()
        for k in range(3):
            rows.append(dict(subject_id=s, group="g", K=3, state=k + 1, quarter=q,
                             fo_quarter=f[k], fo_total=fo[k]))
occ3 = pd.DataFrame(rows)


def drift_block(path, occ):
    """The drift block of an analyze_confirm_lowrank.py version, run verbatim."""
    src = open(path).read()
    block = src[src.index("    # ---------------- drift from the full fits"):
                src.index("    # ---------------- figure")]
    block = "\n".join(line[4:] for line in block.splitlines())
    tmp = tempfile.mkdtemp(prefix="hl_drift_")
    os.makedirs(os.path.join(tmp, "observed"), exist_ok=True)
    occ.to_csv(os.path.join(tmp, "observed", "occupancy_lowrank_15.csv"), index=False)
    ns = dict(np=np, pd=pd, stats=stats, os=os, labels=["lowrank_15"], out=tmp,
              args=types.SimpleNamespace(root=tmp), holm=acl.holm,
              hodges_lehmann=acl.hodges_lehmann, complements=acl.complements)
    exec(block, ns)
    return ns["D"]


cur = os.path.join(NULLDIR, "analyze_confirm_lowrank.py")
leg = os.path.join(CODE, "hmm_v2", "tests", "legacy", "analyze_confirm_lowrank.py")
D2 = drift_block(cur, occ2)
N2 = hod.occupancy_drift(occ2, "lowrank_15")
d1 = occ2[occ2.state == 1].pivot_table(index="subject_id", columns="quarter", values="fo_quarter")
p_once = float(stats.wilcoxon((d1[3] + d1[4]) / 2 - (d1[1] + d1[2]) / 2).pvalue)
cols = ["K", "state", "n", "median_change", "frac_increase", "p_two_sided", "p_holm_within_K"]
ok8a = (np.allclose(D2.p_two_sided, p_once, rtol=1e-12) and np.allclose(D2.p_holm_within_K, p_once, rtol=1e-12)
        and np.allclose(N2[cols].to_numpy(float), D2[cols].to_numpy(float), rtol=1e-12, equal_nan=True))
D2_old = drift_block(leg, occ2)
D3 = drift_block(cur, occ3)
D3_old = drift_block(leg, occ3)
N3 = hod.occupancy_drift(occ3, "lowrank_15")
d3 = float(np.nanmax(np.abs(D3[cols].to_numpy(float) - D3_old[cols].to_numpy(float))))
d3n = float(np.nanmax(np.abs(N3[cols].to_numpy(float) - D3_old[cols].to_numpy(float))))
ok8 = ok8a and abs(D2_old.p_holm_within_K.iloc[0] - min(1, 2 * p_once)) < 1e-12 and max(d3, d3n) < 1e-12
check(f"H8 drift K = 2: p {p_once:.4g} once (previous code: Holm {D2_old.p_holm_within_K.iloc[0]:.4g}); "
      "K = 3 vs previous (max diff)", max(d3, d3n), ok8)

# H9
occ2b = occ2.copy()
m2 = occ2b.state == 2
occ2b.loc[m2, "fo_quarter"] = occ2b.loc[m2, "fo_quarter"] + rng.normal(0, 0.02, m2.sum())
D9 = drift_block(cur, occ2b)
N9 = hod.occupancy_drift(occ2b, "lowrank_15")
ok9 = (np.allclose(D9.p_holm_within_K, acl.holm(D9.p_two_sided)) and D9.p_two_sided.nunique() == 2
       and "not complements" in D9.note.iloc[0]
       and np.allclose(N9.p_holm_within_K, D9.p_holm_within_K))
check("H9 K = 2, not complements: Holm over both, flagged", float(ok9), ok9)

w_ = max(len(r[0]) for r in res)
print(f"{'Check':<{w_}}  {'Value':>10}  Result")
for name, v, r in res:
    print(f"{name:<{w_}}  {v:>10.4g}  {r}")
n_fail = sum(r[2] == "FAIL" for r in res)
print(f"\n{sum(r[2] == 'PASS' for r in res)} passed, {n_fail} failed.")
sys.exit(1 if n_fail else 0)
