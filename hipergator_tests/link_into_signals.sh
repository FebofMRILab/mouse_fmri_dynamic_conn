#!/bin/bash
# Three packages were written to run inside the data folder (the one that holds
# hmm_signals/): hmm_null, dfc and positive_control. Instead of copying their
# files there, this links them, so the code stays in one place (code/) and the
# results stay with the data (signals/).
#
# Run once from the code folder (re-run after updating the code):
#   bash hipergator_tests/link_into_signals.sh            # /blue/febo/share/Abeta_Age/rabies/signals
#   bash hipergator_tests/link_into_signals.sh /path/to/signals
#
# Each package's slurm/*.sbatch files are linked into <signals>/slurm/. The three
# packages share three helper files (dfc_stationary_replica.m,
# hmm_iaaft_surrogate.m, hmm_phase_randomise.m); the copies are identical.
set -euo pipefail
CODE=$(pwd -P)
SIG=$(cd "${1:-${HMM_OUT_BASE:-/blue/febo/share/Abeta_Age/rabies/signals}}" && pwd -P)
[[ -d "$SIG/hmm_signals" ]] || { echo "No hmm_signals/ in $SIG"; exit 1; }
mkdir -p "$SIG/slurm"
n=0
for pkg in hmm_null dfc positive_control; do
    [[ -d "$CODE/$pkg" ]] || { echo "Skipped (not found): $pkg"; continue; }
    for f in "$CODE/$pkg"/*; do
        b=$(basename "$f")
        case "$b" in slurm|tests|__pycache__|*.md) continue ;; esac
        if [[ -e "$SIG/$b" && ! -L "$SIG/$b" ]]; then
            echo "Not linked (a real file of that name is already there): $SIG/$b"
            continue
        fi
        ln -sfn "$f" "$SIG/$b"; n=$((n + 1))
    done
    if [[ -d "$CODE/$pkg/slurm" ]]; then
        for f in "$CODE/$pkg/slurm"/*; do
            ln -sfn "$f" "$SIG/slurm/$(basename "$f")"; n=$((n + 1))
        done
    fi
done
echo "Linked $n files into $SIG (and $SIG/slurm)."
