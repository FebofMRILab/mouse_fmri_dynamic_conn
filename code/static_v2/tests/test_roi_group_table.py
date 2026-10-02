#!/usr/bin/env python3
"""
roi_group_table.py on a simulated wide ROI file (95 mice, 7 groups, 82 ROIs).

    python tests/test_roi_group_table.py        # from the static_v2 folder

Checks
  R1  Welch F and df of every ROI equal statsmodels anova_oneway(use_var='unequal')
  R2  contrast t and df equal scipy ttest_ind(equal_var=False)
  R3  BH across ROIs and across contrasts x ROIs equal statsmodels fdr_bh
  R4  the CSV, Word and PDF tables are written
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
work = tempfile.mkdtemp(prefix="roi_table_test_")
sizes = {"NTG_Y": 19, "NTG_M": 9, "NTG_A": 25, "CRND8_Y": 13, "CRND8_M": 5,
         "5XFAD_Y": 12, "5XFAD_M": 12}
rng = np.random.default_rng(4)
rows, sid = [], 0
for g, n in sizes.items():
    for _ in range(n):
        sid += 1
        shift = 0.6 if g.startswith("5XFAD") else 0.0
        rows.append([sid, g] + list(rng.normal(10 + shift, 3.0 if g == "CRND8_M" else 1.5, 82)))
src = os.path.join(work, "mice_NodePosStrength_PCorr.csv")
pd.DataFrame(rows, columns=["index", "group"] + [f"ROI_{i + 1:02d}" for i in range(82)]).to_csv(
    src, index=False)
subprocess.run([sys.executable, os.path.join(PKG, "roi_group_table.py"), src,
                "--out_dir", work, "--n_perm", "500"], check=True, capture_output=True)

dat = pd.read_csv(src)
dat["group"] = dat.group.str.replace("_", "", regex=False)
order = ["NTGY", "NTGM", "NTGA", "CRND8Y", "CRND8M", "5XFADY", "5XFADM"]
omni = pd.read_csv(os.path.join(work, "Strength_roi_tests_numeric.csv"))
cons = pd.read_csv(os.path.join(work, "Strength_planned_contrasts_numeric.csv"))
res = []
dF = dd = 0.0
for _, r in omni.iterrows():
    a = anova_oneway([dat.loc[dat.group == g, r.ROI].to_numpy() for g in order],
                     use_var="unequal", welch_correction=True)
    dF = max(dF, abs(r.F - a.statistic) / a.statistic)
    dd = max(dd, abs(r.df2 - a.df_denom))
res.append(("R1 Welch F (rel.) and df2 vs statsmodels", max(dF, dd), max(dF, dd) < 1e-10))
dt = 0.0
for _, r in cons.iterrows():
    a, b = r.Contrast.split(" vs ")
    t = stats.ttest_ind(dat.loc[dat.group == a, r.ROI], dat.loc[dat.group == b, r.ROI],
                        equal_var=False)
    dt = max(dt, abs(r.t - t.statistic), abs(r.df - t.df), abs(r.p_raw - t.pvalue))
res.append(("R2 contrast t, df, p vs scipy ttest_ind", dt, dt < 1e-10))
q1 = np.max(np.abs(omni.p_adjusted - multipletests(omni.p_raw, method="fdr_bh")[1]))
q2 = np.max(np.abs(cons.p_adjusted - multipletests(cons.p_raw, method="fdr_bh")[1]))
res.append(("R3 BH across ROIs and across contrasts x ROIs", max(q1, q2), max(q1, q2) < 1e-12))
outs = [f"Strength_{n}" for n in ("summary_table.csv", "summary_table.docx", "summary_table.pdf",
                                  "planned_contrasts_table.csv", "planned_contrasts_table.docx",
                                  "planned_contrasts_table.pdf")]
missing = [o for o in outs if not os.path.isfile(os.path.join(work, o))]
res.append(("R4 tables written (missing)", len(missing), not missing))
for name, v, ok in res:
    print(f"{name:<48} {v:>10.3g}  {'PASS' if ok else 'FAIL'}")
print(f"{sum(r[2] for r in res)} of {len(res)} passed")
sys.exit(0 if all(r[2] for r in res) else 1)
