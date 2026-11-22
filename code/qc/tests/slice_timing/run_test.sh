#!/bin/bash
# Simulation check of slice_timing_lag_check.py (Octave + Python, ~3 min).
# 20 mice, 20 coronal slices acquired interleaved (alt+z), TR 2 s, 570 volumes;
# a global source (component 1), a seesaw source (component 2) and ROI noise,
# band-passed 0.01-0.1 Hz. Group complex PCA by the pipeline's own
# dfc_cpca_fit.m. Four scenarios: no STC, correct STC, STC in the wrong
# direction (alt-z), and a real 1-s front-to-back lag with correct STC.
set -euo pipefail
cd "$(dirname "$0")"
python3 make_labels.py
octave --no-gui -q sim_stc.m
ORDER=$(python3 -c "u=list(range(20)); print(','.join(map(str,u[0::2]+u[1::2])))")
for s in none correct wrongdir gradient; do
  python3 - "$s" <<'PY'
import sys, pandas as pd
s = sys.argv[1]
pd.DataFrame({"signal_file": [f"sig_{s}/signal{m}.txt" for m in range(1, 21)],
              "label_file": [f"labels/sub-{m:02d}_native_labels.nii.gz" for m in range(1, 21)]}
             ).to_csv(f"pairs_{s}.csv", index=False)
PY
  python3 ../../slice_timing_lag_check.py --loadings loadings_$s.csv --pairs pairs_$s.csv \
      --order "$ORDER" --n_perm 2000 --n_boot 500 --out_dir out_$s --formats png > log_$s.txt
done
python3 - <<'PY'
import pandas as pd
exp = {"none": "-1 (attenuated a little)", "correct": "0", "wrongdir": "< -1",
       "gradient": "0 (group-level slope would not be)"}
for s in exp:
    d = pd.read_csv(f"out_{s}/slice_timing_lag_slopes.csv")
    d = d[(d.pattern == "scanner") & (d.component <= 2)]
    for _, r in d.iterrows():
        print(f"{s:9s} C{r.component}: within-ROI slope {r.slope_within:6.2f} "
              f"[{r.ci_lo:5.2f}, {r.ci_hi:5.2f}] p={r.p_perm:.4f}  "
              f"group-level {r.group_slope_descriptive:5.2f}   expected {exp[s]}")
PY
