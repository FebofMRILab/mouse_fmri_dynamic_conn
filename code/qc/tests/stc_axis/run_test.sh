#!/bin/bash
# Simulation check of stc_axis_check.py (~1 min). 8 mice per case, 14 slices
# along axis 2, TR 2 s, 600 volumes. Expected (even minus odd, s; slope for
# alt+z): raw_altp +1.00, -1.00 | raw_altm -1.00, +1.00 | raw_altp_grad +0.96,
# -0.84 | stcY slices as raw_altp, rows +0.92 and slope +1.00 for alt-z.
set -euo pipefail
cd "$(dirname "$0")"
python3 sim.py
for c in raw_altp raw_altm raw_altp_grad stcY; do
  echo "== $c"
  python3 ../../stc_axis_check.py --inputs "data/${c}_m*.nii.gz" --TR 2 --n_boot 300 \
      --out_dir "out_$c" | grep -E "^ *(slice|row) +[0-9]+ +(alt\+z|alt-z|even)"
done
