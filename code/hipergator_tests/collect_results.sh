#!/bin/bash
# Bundle the results to send back after the fits and pipelines of both reported
# models (primary and secondary; hmm_v2/MODEL_*.env), the tests and the DCC
# cross-check: tables, logs and figures only (no .mat, .svg or .tiff).
# Run from code/:
#   bash hipergator_tests/collect_results.sh            # -> abeta_results_<date>.zip
set -uo pipefail
SIG=${HMM_OUT_BASE:-/blue/febo/share/Abeta_Age/rabies/signals}
OUT=${1:-abeta_results_$(date +%Y%m%d).zip}
paths=()
for role in primary secondary; do
    # model folder, tag and fits folder of this model (from its model file)
    . hipergator_tests/reported_model.sh "$role" || exit 1
    FITS=$SIG/$FITS_NAME
    # After the fit-selection rule (FIT_SELECTION_RULE.md) the reported fit and all
    # its outputs sit in <fits>/fit_NN/.
    if [[ -f "$FITS/selection/selected_fit.txt" ]]; then
        BASE=$(sed -n 's/^selected_base: //p' "$FITS/selection/selected_fit.txt")
        echo "$MODEL_LABEL: reported fit $BASE"
    else
        echo "$MODEL_LABEL: no selection yet in $FITS (skipped)"
        continue
    fi
    M="$BASE/$MODEL_FOLDER"
    paths+=(
      "$FITS/selection" "$FITS/rule_sha256.txt" "$FITS/model.env" "$M/inference_scope.csv"
      "$M/fit_id.txt" "$M/occupancy_drift.csv" "$M/occupancy_by_quarter.csv" "$M/metadata_aligned.csv"
      "$M/check_against_confirmatory" "$M/state_descriptors"
      "$M/hmm_raw_posthoc_graph_metrics" "$M/hmm_bootstrap_stability"
      "$M/hmm_graph_figures" "$M/bootstrap_supplement"
      "$BASE/hmm_dynamics_stats_$MODEL_TAG" "$BASE/hmm_dynamics_figures_$MODEL_TAG"
      "$BASE/hmm_switching_rates_$TABLE_SUFFIX.csv" "$BASE/hmm_maxFO_$TABLE_SUFFIX.csv"
      "$BASE/hmm_dwell_times_$TABLE_SUFFIX.csv"
    )
done
paths+=(
  "$SIG/hmm_model_agreement"
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
