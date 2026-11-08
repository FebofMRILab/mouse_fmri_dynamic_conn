#!/bin/bash
# Test: the two reported models reach every job script from their model files
# (hmm_v2/MODEL_PRIMARY.env, MODEL_SECONDARY.env; FIT_SELECTION_RULE.md
# amendment 1). matlab, sbatch, module and sleep are replaced by stubs that
# record what they were given; nothing is fitted. Seconds; any Linux shell.
#   bash hipergator_tests/tests/test_model_files.sh        (from code/)
#
#   M1  reported_model.sh: primary -> K 3, pca 25, folders hmm_k3_pca25_fits,
#       hmmmar_simple_k3_pca25, tag pca25_k3, PCA tables, step 6b gamma file
#   M2  reported_model.sh: secondary -> K 2, lowrank 15, hmm_k2_hmmpca15_fits,
#       tag hmmpca15_k2, lowrank tables, step 6 gamma file
#   M3  a missing role, an unknown role, a malformed file and a file setting
#       both lowrank and pca are refused
#   M4  fits array, primary: run4e gets K 3 / lowrank 0 / pca 25 even when the
#       caller exported other HMM_* values; rule SHA-256 and model.env written
#   M5  fits array: a model file changed after the first task is refused, also
#       when the change (K 3 -> 4) would point to a new fits folder
#   M6  selection job, secondary: MATLAB gets K 2 / lowrank 15 / pca 0 and the
#       secondary fits folder; refused if model.env is another model
#   M7  run_selected_fit.sh primary: pipeline submitted with K 3 / pca 25, the
#       chosen fit's folders and the step 6b gamma file; refused when the
#       chosen fit is not of that model
#   M8  collect_results.sh: both models' selections, scopes, stats folders and
#       run4d tables are collected
set -u
CODE=$(cd "$(dirname "$0")/../.." && pwd)
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
pass=0; fail=0
check() { if [[ "$2" == 1 ]]; then echo "PASS  $1"; pass=$((pass + 1)); else echo "FAIL  $1"; fail=$((fail + 1)); fi; }

# a code/ copy with stubs on PATH
mkdir -p "$W/code/hmm_v2" "$W/code/hipergator_tests" "$W/bin" "$W/sig/hmm_signals"
cp "$CODE"/hmm_v2/MODEL_*.env "$CODE"/hmm_v2/FIT_SELECTION_RULE.md "$W/code/hmm_v2/"
cp "$CODE"/hipergator_tests/*.sh "$CODE"/hipergator_tests/*.sbatch "$W/code/hipergator_tests/"
cat > "$W/bin/matlab" <<'EOF'
#!/bin/bash
env | grep '^HMM_' | sort > "$STUB_LOG/matlab_env_$(date +%s%N).txt"
echo "$*" >> "$STUB_LOG/matlab_calls.txt"
EOF
cat > "$W/bin/sbatch" <<'EOF'
#!/bin/bash
echo "$*" >> "$STUB_LOG/sbatch_calls.txt"
EOF
printf '#!/bin/bash\nexit 0\n' > "$W/bin/module"
printf '#!/bin/bash\nexit 0\n' > "$W/bin/sleep"
chmod +x "$W"/bin/*
export PATH="$W/bin:$PATH" STUB_LOG="$W/log"
# On HiPerGator, Lmod defines `module` (and `ml`) as shell functions, exported
# and also re-defined in every non-interactive bash through BASH_ENV (Lmod's init
# script). Either way they take precedence over the stub on PATH, and
# `module load matlab` would put the real MATLAB in front of the stub. Unset
# BASH_ENV/ENV and export no-op functions, so the job scripts started below see
# the stubs only.
unset BASH_ENV ENV
module() { return 0; }
ml() { return 0; }
export -f module ml
mkdir -p "$STUB_LOG"
cd "$W/code"

# ---- M1-M3: the helper ----
out=$( . hipergator_tests/reported_model.sh primary && echo "$HMM_K $HMM_LOWRANK $HMM_PCA $FITS_NAME $MODEL_FOLDER $MODEL_TAG $TABLE_SUFFIX $CONFIRM_GAMMA_REL" )
check "M1 primary: $out" "$([[ "$out" == "3 0 25 hmm_k3_pca25_fits hmmmar_simple_k3_pca25 pca25_k3 all_K_all_PCA hmm_confirm_candidate/observed/gamma_gaussian_pca_25_K3.mat" ]] && echo 1)"
out=$( . hipergator_tests/reported_model.sh secondary && echo "$HMM_K $HMM_LOWRANK $HMM_PCA $FITS_NAME $MODEL_FOLDER $MODEL_TAG $TABLE_SUFFIX $CONFIRM_GAMMA_REL" )
check "M2 secondary: $out" "$([[ "$out" == "2 15 0 hmm_k2_hmmpca15_fits hmmmar_simple_k2_hmmpca15 hmmpca15_k2 all_K_all_lowrank hmm_confirm_lowrank/observed/gamma_lowrank_15_K2.mat" ]] && echo 1)"
r1=$( ( . hipergator_tests/reported_model.sh ) 2>&1 >/dev/null; echo "rc=$?" )
r2=$( ( . hipergator_tests/reported_model.sh tertiary ) 2>&1 >/dev/null; echo "rc=$?" )
cp hmm_v2/MODEL_SECONDARY.env "$W/sec.bak"
printf 'HMM_K=2\nHMM_LOWRANK=15\nHMM_PCA=0\nK=7\n' > hmm_v2/MODEL_SECONDARY.env
r3=$( ( . hipergator_tests/reported_model.sh secondary ) 2>&1 >/dev/null; echo "rc=$?" )
printf 'HMM_K=2\nHMM_LOWRANK=15\nHMM_PCA=25\n' > hmm_v2/MODEL_SECONDARY.env
r4=$( ( . hipergator_tests/reported_model.sh secondary ) 2>&1 >/dev/null; echo "rc=$?" )
cp "$W/sec.bak" hmm_v2/MODEL_SECONDARY.env
ok=$([[ $r1 == *"rc=1" && $r2 == *"rc=1" && $r3 == *"Unreadable line"*"rc=1" && $r4 == *"both"*"rc=1" ]] && echo 1)
check "M3 no role / unknown role / malformed line / both lowrank and pca refused" "$ok"

# ---- M4: fits array, primary, with stray HMM_* in the caller ----
rm -f "$STUB_LOG"/*
( export SLURM_SUBMIT_DIR="$W/code" SLURM_ARRAY_TASK_ID=1 HMM_OUT_BASE="$W/sig" HMM_K=9 HMM_LOWRANK=7 HMM_INITREP=1
  bash hipergator_tests/run_hmm_fits_array.sbatch primary > "$W/m4.txt" 2>&1 )
envf=$(ls "$STUB_LOG"/matlab_env_*.txt 2>/dev/null | head -1)
F="$W/sig/hmm_k3_pca25_fits"
ok=$([[ -n "$envf" ]] && grep -qx 'HMM_K=3' "$envf" && grep -qx 'HMM_LOWRANK=0' "$envf" && grep -qx 'HMM_PCA=25' "$envf" \
     && grep -qx "HMM_OUT_BASE=$F/fit_01" "$envf" && ! grep -q '^HMM_INITREP' "$envf" \
     && cmp -s "$F/model.env" hmm_v2/MODEL_PRIMARY.env && cmp -s "$W/sig/hmm_fits_primary_model.env" hmm_v2/MODEL_PRIMARY.env && grep -q "^sha256: $(sha256sum hmm_v2/FIT_SELECTION_RULE.md | cut -d' ' -f1)" "$F/rule_sha256.txt" && echo 1)
check "M4 fits array primary: run4e gets K 3, pca 25 (caller's HMM_K=9 ignored); rule hash and model.env recorded" "$ok"
[[ "$ok" == 1 ]] || { echo "      --- M4 job output (last 15 lines):"; tail -15 "$W/m4.txt" | sed 's/^/      /';
                      echo "      --- matlab env recorded: ${envf:-none}"; [[ -n "$envf" ]] && sed 's/^/      /' "$envf"; }

# ---- M5: model file changed after the first task ----
cp hmm_v2/MODEL_PRIMARY.env "$W/pri.bak"
sed -i 's/^HMM_K=3$/HMM_K=4/' hmm_v2/MODEL_PRIMARY.env
rm -f "$STUB_LOG"/*
( export SLURM_SUBMIT_DIR="$W/code" SLURM_ARRAY_TASK_ID=2 HMM_OUT_BASE="$W/sig"
  bash hipergator_tests/run_hmm_fits_array.sbatch primary > "$W/m5.txt" 2>&1; echo "rc=$?" >> "$W/m5.txt" )
cp "$W/pri.bak" hmm_v2/MODEL_PRIMARY.env
ok=$(grep -q "differs from the primary model recorded" "$W/m5.txt" && grep -q "rc=1" "$W/m5.txt" \
     && ! ls "$STUB_LOG"/matlab_env_* >/dev/null 2>&1 && [[ ! -e "$W/sig/hmm_k4_pca25_fits" ]] && echo 1)
check "M5 model file changed after the first task (K 3 -> 4): refused, no new folder, nothing fitted" "$ok"

# ---- M6: selection job, secondary ----
S="$W/sig/hmm_k2_hmmpca15_fits"; mkdir -p "$S"
cp hmm_v2/MODEL_SECONDARY.env "$S/model.env"
rm -f "$STUB_LOG"/*
( export SLURM_SUBMIT_DIR="$W/code" HMM_OUT_BASE="$W/sig" HMM_PCA=25
  bash hipergator_tests/run_select_fit.sbatch secondary > "$W/m6.txt" 2>&1 )
envf=$(ls "$STUB_LOG"/matlab_env_*.txt 2>/dev/null | head -1)
ok6a=$([[ -n "$envf" ]] && grep -qx 'HMM_K=2' "$envf" && grep -qx 'HMM_LOWRANK=15' "$envf" && grep -qx 'HMM_PCA=0' "$envf" \
       && grep -q "select_reported_fit('$S')" "$STUB_LOG/matlab_calls.txt" && echo 1)
cp hmm_v2/MODEL_PRIMARY.env "$S/model.env"
rm -f "$STUB_LOG"/*
( export SLURM_SUBMIT_DIR="$W/code" HMM_OUT_BASE="$W/sig"
  bash hipergator_tests/run_select_fit.sbatch secondary > "$W/m6b.txt" 2>&1 )
cp hmm_v2/MODEL_SECONDARY.env "$S/model.env"
ok=$([[ $ok6a == 1 ]] && grep -q "differs from the copy" "$W/m6b.txt" && ! ls "$STUB_LOG"/matlab_env_* >/dev/null 2>&1 && echo 1)
check "M6 selection secondary: K 2, lowrank 15, its fits folder; another model's model.env refused" "$ok"
[[ "$ok" == 1 ]] || { echo "      --- M6 job output (last 15 lines):"; tail -15 "$W/m6.txt" | sed 's/^/      /';
                      echo "      --- second run:"; tail -5 "$W/m6b.txt" | sed 's/^/      /'; }

# ---- M7: run_selected_fit.sh primary ----
B="$F/fit_04"; MD="$B/hmmmar_simple_k3_pca25"; mkdir -p "$MD" "$F/selection"
echo "x" > "$MD/inference_scope.csv"
printf 'reported_fit: fit_04\nselected_base: %s\nmodel_dir: %s\nfit_id: abcdef0123456789\n' "$B" "$MD" > "$F/selection/selected_fit.txt"
rm -f "$STUB_LOG"/*
( export HMM_OUT_BASE="$W/sig" HMM_K=7; bash hipergator_tests/run_selected_fit.sh primary > "$W/m7.txt" 2>&1 )
call=$(cat "$STUB_LOG/sbatch_calls.txt" 2>/dev/null)
ok7a=$([[ "$call" == *"HMM_K=3,HMM_LOWRANK=0,HMM_PCA=25,"* && "$call" == *"HMM_MODEL_DIR=$MD"* \
         && "$call" == *"HMM_OUT_BASE=$B"* && "$call" == *"HMM_CONFIRM_GAMMA=$W/sig/hmm_confirm_candidate/observed/gamma_gaussian_pca_25_K3.mat"* \
         && "$call" == *"SKIP_FIT=1"* ]] && echo 1)
printf 'reported_fit: fit_04\nselected_base: %s\nmodel_dir: %s/hmmmar_simple_k2_hmmpca15\nfit_id: abc\n' "$B" "$B" > "$W/wrongsel.txt"
cp "$F/selection/selected_fit.txt" "$W/sel.bak"; cp "$W/wrongsel.txt" "$F/selection/selected_fit.txt"
mkdir -p "$B/hmmmar_simple_k2_hmmpca15"; echo x > "$B/hmmmar_simple_k2_hmmpca15/inference_scope.csv"
rm -f "$STUB_LOG"/*
( export HMM_OUT_BASE="$W/sig"; bash hipergator_tests/run_selected_fit.sh primary > "$W/m7b.txt" 2>&1 )
cp "$W/sel.bak" "$F/selection/selected_fit.txt"
ok=$([[ $ok7a == 1 ]] && grep -q "is not a fit of" "$W/m7b.txt" && [[ ! -s "$STUB_LOG/sbatch_calls.txt" ]] && echo 1)
check "M7 run_selected_fit primary: K 3 / pca 25, chosen fit, step 6b gamma; wrong model refused" "$ok"

# ---- M8: collect both models ----
for f in inference_scope.csv fit_id.txt occupancy_drift.csv; do echo x > "$MD/$f"; done
mkdir -p "$B/hmm_dynamics_stats_pca25_k3" && echo x > "$B/hmm_dynamics_stats_pca25_k3/omnibus_tests.csv"
echo x > "$B/hmm_switching_rates_all_K_all_PCA.csv"
B2="$S/fit_07"; MD2="$B2/hmmmar_simple_k2_hmmpca15"; mkdir -p "$MD2" "$S/selection" "$B2/hmm_dynamics_stats_hmmpca15_k2"
echo x > "$MD2/inference_scope.csv"; echo x > "$B2/hmm_dynamics_stats_hmmpca15_k2/omnibus_tests.csv"
echo x > "$B2/hmm_maxFO_all_K_all_lowrank.csv"
printf 'reported_fit: fit_07\nselected_base: %s\nmodel_dir: %s\nfit_id: abc\n' "$B2" "$MD2" > "$S/selection/selected_fit.txt"
echo "sha256: x" > "$S/rule_sha256.txt"
( export HMM_OUT_BASE="$W/sig"; bash hipergator_tests/collect_results.sh "$W/out.zip" > "$W/m8.txt" 2>&1 )
if [[ -f "$W/out.zip" ]]; then lst=$(python3 -c "import zipfile,sys; print('\n'.join(zipfile.ZipFile(sys.argv[1]).namelist()))" "$W/out.zip")
else lst=$(tar tzf "$W/out.tar.gz" 2>/dev/null); fi
ok=1
for want in hmm_k3_pca25_fits/model.env hmm_k3_pca25_fits/fit_04/hmmmar_simple_k3_pca25/inference_scope.csv \
            hmm_k3_pca25_fits/fit_04/hmm_dynamics_stats_pca25_k3/omnibus_tests.csv \
            hmm_k3_pca25_fits/fit_04/hmm_switching_rates_all_K_all_PCA.csv \
            hmm_k2_hmmpca15_fits/fit_07/hmmmar_simple_k2_hmmpca15/inference_scope.csv \
            hmm_k2_hmmpca15_fits/fit_07/hmm_dynamics_stats_hmmpca15_k2/omnibus_tests.csv \
            hmm_k2_hmmpca15_fits/fit_07/hmm_maxFO_all_K_all_lowrank.csv hmm_k2_hmmpca15_fits/selection/selected_fit.txt; do
    grep -q "$want" <<< "$lst" || { ok=0; echo "      missing in the bundle: $want"; }
done
check "M8 collect_results: both models' selection, scope, stats and run4d tables" "$ok"

echo; echo "$pass passed, $fail failed."
[[ $fail == 0 ]]
