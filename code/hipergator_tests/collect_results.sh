#!/bin/bash
# Bundle the results to send back after the K = 4 fits and pipeline, the tests
# and the DCC cross-check: tables, logs and figures only (no .mat, .svg or .tiff).
# Run from code/:
#   bash hipergator_tests/collect_results.sh            # -> abeta_results_<date>.zip
set -uo pipefail
SIG=${HMM_OUT_BASE:-/blue/febo/share/Abeta_Age/rabies/signals}
FITS=${HMM_FITS_DIR:-$SIG/hmm_k4_hmmpca15_fits}
# After the fit-selection rule (FIT_SELECTION_RULE.md) the reported fit and all
# its outputs sit in <fits>/fit_NN/; before it, in the data folder itself.
if [[ -f "$FITS/selection/selected_fit.txt" ]]; then
    BASE=$(sed -n 's/^selected_base: //p' "$FITS/selection/selected_fit.txt")
    echo "Reported fit (fit-selection rule): $BASE"
else
    BASE=$SIG
fi
M="$BASE/hmmmar_simple_k4_hmmpca15"
OUT=${1:-abeta_results_$(date +%Y%m%d).zip}
paths=(
  "$FITS/selection" "$FITS/rule_sha256.txt" "$M/inference_scope.csv"
  "$M/fit_id.txt" "$M/occupancy_drift.csv" "$M/occupancy_by_quarter.csv" "$M/metadata_aligned.csv"
  "$M/check_against_confirmatory"
  "$M/hmm_raw_posthoc_graph_metrics" "$M/hmm_bootstrap_stability"
  "$M/hmm_graph_figures" "$M/bootstrap_supplement"
  "$BASE/hmm_dynamics_stats_hmmpca15_k4" "$BASE/hmm_dynamics_figures_hmmpca15_k4"
  "$BASE/hmm_switching_rates_all_K_all_lowrank.csv" "$BASE/hmm_maxFO_all_K_all_lowrank.csv"
  "$BASE/hmm_dwell_times_all_K_all_lowrank.csv"
  "$SIG/dfc_dcc_cpca/dcc_example" "$SIG/dfc_dcc_cpca/figures/FigS_dfc_raw_units.png"
  "$SIG/dfc_dcc_cpca/figures/FigS_dfc_raw_units.pdf" "$SIG/dfc_dcc_cpca/figures/table_raw_units.csv"
  hmm_v2_test_results static_test_results dcc_lindquist_check logs
)
found=()
for p in "${paths[@]}"; do
    if [[ -e "$p" ]]; then found+=("$p"); else echo "Missing (skipped): $p"; fi
done
[[ ${#found[@]} -gt 0 ]] || { echo "Nothing to collect."; exit 1; }
if command -v zip >/dev/null; then
    zip -qr "$OUT" "${found[@]}" -x '*.mat' '*.svg' '*.tiff' '*/__pycache__/*'
else
    OUT=${OUT%.zip}.tar.gz
    tar czf "$OUT" --exclude='*.mat' --exclude='*.svg' --exclude='*.tiff' "${found[@]}"
fi
echo "Written: $OUT ($(du -h "$OUT" | cut -f1)). Send this file."
