#!/bin/bash
# Test of select_screen_candidate.py on synthetic step 5 tables with known answers.
# Run: bash run_test.sh [python]
set -e
PY=${1:-python}
HERE=$(cd "$(dirname "$0")" && pwd)
S="$HERE/../../select_screen_candidate.py"
W=$(mktemp -d); cd "$W"
$PY - <<'PYEOF'
import pandas as pd
models = ['gaussian_full', 'gaussian_pca_15', 'gaussian_pca_25', 'gaussian_pca_50', 'lowrank_15', 'lowrank_25', 'lowrank_50']
def table(lead, n_null=10):
    rows = []
    for m in models:
        for K in range(2, 11):
            for null in ['phase', 'iaaft']:
                for metric in ['raw', 'demeaned']:
                    d, p = 0.001, 0.4
                    if (m, K) in lead and null == 'phase' and metric == 'demeaned':
                        d, p = lead[(m, K)]
                    rows.append(dict(model=m, null=null, K=K, metric=metric, n_obs=10, n_null=n_null,
                                     median_observed=.3, median_null=.3 - d, delta_median=d, p_mw=p))
    return pd.DataFrame(rows)
table({('gaussian_pca_25', 6): (0.05, 0.001), ('gaussian_full', 2): (0.04, 0.002), ('lowrank_15', 3): (0.03, 0.004)}).to_csv('a.csv', index=False)
table({('lowrank_15', 3): (0.08, 0.001), ('gaussian_pca_25', 3): (0.05, 0.001)}).to_csv('b.csv', index=False)
table({('gaussian_pca_25', 3): (0.05, 0.01)}).to_csv('c.csv', index=False)
table({('gaussian_pca_25', 5): (0.0501, 0.001), ('gaussian_pca_15', 5): (0.0504, 0.001), ('gaussian_full', 7): (0.0498, 0.001)}).to_csv('d.csv', index=False)
table({('gaussian_pca_25', 3): (0.05, 0.001)}, n_null=5).to_csv('e.csv', index=False)
table({('gaussian_pca_25', 10): (0.05, 0.001)}).to_csv('f.csv', index=False)
pd.DataFrame({'subject_id': range(1, 96), 'group': 'x'}).to_csv('m95.csv', index=False)
pd.DataFrame({'subject_id': range(1, 59), 'group': 'x'}).to_csv('m58.csv', index=False)
PYEOF
fail=0
expect() { if grep -q "$2" "$1.txt"; then echo "PASS  $3"; else echo "FAIL  $3"; fail=1; fi; }
for f in a b c d f; do $PY "$S" --table $f.csv --metadata m95.csv --out $f.txt > /dev/null; done
expect a "CANDIDATE  gaussian_pca_25, K = 6" "largest delta among leads; K outside 2-4"
expect a "K \[5, 6, 7\]" "confirmation cells K-1, K, K+1"
expect b "step 6 (already run" "HMM-PCA rank 15 at K 3: covered by step 6"
expect c "NO LEAD" "p above 0.05/9: no lead"
expect d "CANDIDATE  gaussian_pca_15, K = 5" "tie at 3 decimals: smaller K, then smaller dimension"
expect f "K \[9, 10, 11\]" "K = 10: cells 9-11"
if $PY "$S" --table e.csv --metadata m95.csv > /dev/null 2>&1; then echo "FAIL  5 null fits refused"; fail=1; else echo "PASS  5 null fits refused"; fi
if $PY "$S" --table a.csv --metadata m58.csv > /dev/null 2>&1; then echo "FAIL  58 mice refused"; fail=1; else echo "PASS  58 mice refused (E49)"; fi
[[ $fail == 0 ]] && echo "ALL PASS" || { echo "FAILED"; exit 1; }
