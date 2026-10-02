#!/usr/bin/env python3
"""
Checks that hmm_occupancy_drift.py gives the same drift table as the
confirmatory analysis (analyze_confirm_lowrank.py), by running the drift block
of the confirmatory script verbatim on the same simulated occupancy table.

    python tests/test_occupancy_drift.py        # from the hmm_v2 folder
"""
import os
import sys
import tempfile
import types

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import hmm_occupancy_drift as new                      # noqa: E402

# simulated occupancy: 40 mice, K = 3, drift in states 1 and 2
rng = np.random.default_rng(2)
rows = []
for s in range(1, 41):
    fo = rng.dirichlet(np.ones(3) * 3)
    for q in range(1, 5):
        f = np.clip(fo + np.array([0.03, -0.03, 0]) * (q - 2.5) + rng.normal(0, 0.05, 3), 1e-3, None)
        f /= f.sum()
        for k in range(3):
            rows.append(dict(subject_id=s, group="g", K=3, state=k + 1, quarter=q,
                             fo_quarter=f[k], fo_total=fo[k]))
occ = pd.DataFrame(rows)

# previous code, verbatim: the drift block of analyze_confirm_lowrank.py
src = open(os.path.join(HERE, "legacy", "analyze_confirm_lowrank.py")).read()
block = src[src.index("    # ---------------- drift from the full fits"):
            src.index("    # ---------------- figure")]
block = "\n".join(line[4:] for line in block.splitlines())     # dedent from main()
tmp = tempfile.mkdtemp()
os.makedirs(os.path.join(tmp, "observed"))
occ.to_csv(os.path.join(tmp, "observed", "occupancy_lowrank_15.csv"), index=False)
legacy_holm = src[src.index("def holm(p):"):src.index("\n\n\n", src.index("def holm(p):"))]
ns = dict(np=np, pd=pd, stats=stats, os=os, labels=["lowrank_15"], out=tmp,
          args=types.SimpleNamespace(root=tmp))
exec(legacy_holm, ns)
exec(block, ns)
old = ns["D"]

D = new.occupancy_drift(occ, "lowrank_15")
cols = ["K", "state", "n", "fo_total", "median_change", "frac_increase", "p_two_sided",
        "p_holm_within_K"]
diff = np.nanmax(np.abs(old[cols].to_numpy(float) - D[cols].to_numpy(float)))
print(D[cols].round(5).to_string(index=False))
print(f"\nmax |difference| vs the confirmatory analysis code: {diff:.2e}")
print("PASS" if diff < 1e-12 else "FAIL")
sys.exit(0 if diff < 1e-12 else 1)
