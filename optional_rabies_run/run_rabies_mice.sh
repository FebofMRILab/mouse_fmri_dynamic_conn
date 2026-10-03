#!/usr/bin/env bash
#SBATCH --job-name=mouse_fmri
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=febo@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --time=124:00:00
#SBATCH --account=febo
#SBATCH --output=output_rabies_%j.log

set -Eeuo pipefail

trap 'printf "ERROR at line %s: %s\n" "$LINENO" "$BASH_COMMAND" >&2' ERR

# -------------------- Configuration --------------------
data="${DATA_NAME:-Abeta_Age}"
main_dir="/blue/febo/share"
project_dir="${main_dir}/${data}"
atlas_dir="${main_dir}/civm_atlas"

input_BIDS="${project_dir}/bids_converted"
preprocess_outputs="${project_dir}/rabies/preprocess_outputs"
confound_correction_outputs="${project_dir}/rabies/confound_correction_outputs"
analysis_outputs="${project_dir}/rabies/analysis_outputs"

rabies_container="${main_dir}/rabies_0.6.1.sif"
rois="${atlas_dir}/BrainNetCoord.nii"
seeds_dir="${atlas_dir}/ROIs_Lo"

# -------------------- Prerequisites --------------------
for path in "$input_BIDS" "$atlas_dir" "$seeds_dir" \
            "$rabies_container" "$rois"; do
    [[ -e "$path" ]] || {
        printf 'Required input not found: %s\n' "$path" >&2
        exit 1
    }
done

module load singularity
module load afni/25.2.12
module load fsl/6.0.7

: "${FSLDIR:?FSLDIR was not set by the FSL module}"
: "${HPC_AFNI_DIR:?HPC_AFNI_DIR was not set by the AFNI module}"

mkdir -p "$preprocess_outputs" \
         "$confound_correction_outputs" \
         "$analysis_outputs"

# Match the container PATH used for the existing preprocessing and
# confound-correction jobs.
container_path="$(singularity exec "$rabies_container" bash -c 'echo "$PATH"')"
rabies_path="/opt/fsl/bin:/opt/afni:/opt/afni_local/bin:${container_path}"

# Pass the seed files as separate arguments, in version-number order.
mapfile -t seeds < <(
    find "$seeds_dir" -maxdepth 1 -type f -name 'roi*' | sort -V
)
((${#seeds[@]} > 0)) || {
    printf 'No roi* seed files found in %s\n' "$seeds_dir" >&2
    exit 1
}

# Reuse the same container mounts for all three stages.
container_args=(
    -B "${FSLDIR}:/opt/fsl:ro"
    -B "${HPC_AFNI_DIR}:/opt/afni_local:ro"
    -B "${input_BIDS}:/input_BIDS:ro"
    -B "${preprocess_outputs}:/preprocess_outputs"
    -B "${confound_correction_outputs}:/confound_correction_outputs"
    -B "${analysis_outputs}:/analysis_outputs"
    -B "${atlas_dir}:${atlas_dir}:ro"
    --env "PATH=${rabies_path}"
)

run_stage() {
    local stage="$1"
    shift
    printf '\n===== Starting %s: %s =====\n' "$stage" "$(date)"
    singularity run "${container_args[@]}" "$rabies_container" \
        -f -p MultiProc "$@"
    printf '===== Finished %s: %s =====\n' "$stage" "$(date)"
}

# -------------------- 1. Preprocessing --------------------
run_stage "preprocessing" \
    --local_threads 15 --scale_min_memory 4.0 \
    preprocess /input_BIDS/ /preprocess_outputs/ \
    --apply_despiking \
    --apply_STC \
    --TR 2 \
    --anat_inho_cor otsu_thresh=2 \
    --anat_robust_inho_cor apply=true \
    --bold_inho_cor otsu_thresh=3 \
    --bold_robust_inho_cor apply=true \
    --commonspace_reg \
        stages=rigid-affine-nlin,masking=true,brain_extraction=true,keep_mask_after_extract=true \
    --bold2anat_coreg \
        masking=true,brain_extraction=true,keep_mask_after_extract=true \
    --hmc_qc_report apply=true,fps=10 \
    --commonspace_resampling 0.2x0.2x0.4 \
    --anatomical_resampling 0.1x0.1x0.4 \
    --anat_template "${atlas_dir}/symmetric45um_DH.nii.gz" \
    --brain_mask "${atlas_dir}/symmetric45um_mask.nii.gz" \
    --WM_mask "${atlas_dir}/symmetric45um_WM.nii.gz" \
    --CSF_mask "${atlas_dir}/symmetric45um_CSF.nii.gz" \
    --vascular_mask "${atlas_dir}/symmetric45um_empty.nii.gz"

# -------------------- 2. Confound correction --------------------
run_stage "confound correction" \
    --local_threads 7 --scale_min_memory 4.0 \
    confound_correction \
    /preprocess_outputs/ /confound_correction_outputs/ \
    --nuisance_regressors WM_signal CSF_signal mot_24 \
    --image_scaling grand_mean_scaling \
    --detrending order=2 \
    --highpass 0.01 \
    --lowpass 0.1 \
    --edge_cutoff 30 \
    --smoothing_filter 0.5 \
    --TR 2 \
    --ica_aroma apply=true \
    --keep_EPI_average

# -------------------- 3. Analysis --------------------
run_stage "analysis" \
    --local_threads 7 --scale_min_memory 4.0 \
    analysis \
    /confound_correction_outputs/ /analysis_outputs/ \
    --group_ica apply=true,dim=30,random_seed=1,disableMigp=true \
    --seed_list "${seeds[@]}" \
    --FC_matrix \
    --ROI_labels_file "$rois" \
    --ROI_type parcellated

printf '\nAll three RABIES stages completed: %s\n' "$(date)"