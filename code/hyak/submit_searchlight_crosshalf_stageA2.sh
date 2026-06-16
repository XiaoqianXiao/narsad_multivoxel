#!/usr/bin/env bash
set -euo pipefail

# ---- User-configurable settings ----
PROJECT_ROOT="/gscratch/fang/NARSAD"
CONTAINER_SIF="/gscratch/fang/images/jupyter.sif"
APP_PATH="/gscratch/scrubbed/fanglab/xiaoqian/repo/narsad_multivoxel/hyak"
OUT_BASE="/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/LSS/searchlight"
REFERENCE_LSS="/gscratch/fang/NARSAD/MRI/derivatives/fMRI_analysis/LSS/firstLevel/all_subjects/subjects/sub-N101_task-phase2_contrast1.nii.gz"
GLASSER_ATLAS="/gscratch/fang/NARSAD/ROI/Glasser/HCP-MMP1_2mm.nii"
TIAN_ATLAS="/gscratch/fang/NARSAD/ROI/Tian/3T/Subcortex-Only/Tian_Subcortex_S4_3T_2009cAsym.nii.gz"

LOG_DIR="/gscratch/fang/NARSAD/logs/searchlight_crosshalf"
PARTITION="ckpt-all"
ACCOUNT="fang"
TIME="4:00:00"
MEM="120G"
CPUS=32
CHUNKS=512 #384 for others
N_PERM=5000
MODE="rst"  # ext | rst | dyn | crossphase | all

mkdir -p "$LOG_DIR"

module load apptainer 2>/dev/null || true

submit_array() {
  local name="$1"
  local script="$2"
  local out_dir="$3"
  local extra_args="$4"
  sbatch \
    --partition="$PARTITION" \
    --account="$ACCOUNT" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$TIME" \
    --job-name="$name" \
    --array=0-$((CHUNKS - 1)) \
    --output="$LOG_DIR/${name}_%A_%a.out" \
    --error="$LOG_DIR/${name}_%A_%a.err" \
    --wrap="apptainer exec -B ${PROJECT_ROOT}:${PROJECT_ROOT} -B ${APP_PATH}:/app -B ${OUT_BASE}:/output_dir ${CONTAINER_SIF} python3 /app/${script} --project_root ${PROJECT_ROOT} --out_dir /output_dir/${out_dir} --reference_lss ${REFERENCE_LSS} --glasser_atlas ${GLASSER_ATLAS} --tian_atlas ${TIAN_ATLAS} --n_jobs ${CPUS} --n_perm ${N_PERM} --chunk_idx \$SLURM_ARRAY_TASK_ID --chunk_count ${CHUNKS} --cross_half_stage ${extra_args}"
}

if [[ "${MODE}" == "all" || "${MODE}" == "ext" ]]; then
  submit_array "crosshalf_ext" "mvpa_searchlight_wholeBrain_ext.py" "ext" ""
fi

if [[ "${MODE}" == "all" || "${MODE}" == "rst" ]]; then
  submit_array "crosshalf_rst" "mvpa_searchlight_wholeBrain_rst.py" "rst" ""
fi

if [[ "${MODE}" == "all" || "${MODE}" == "dyn" ]]; then
  submit_array "crosshalf_dyn_ext" "mvpa_searchlight_wholeBrain_dyn_ext.py" "dyn_ext" ""
  submit_array "crosshalf_dyn_rst" "mvpa_searchlight_wholeBrain_dyn_rst.py" "dyn_rst" ""
fi

if [[ "${MODE}" == "all" || "${MODE}" == "crossphase" ]]; then
  submit_array "crosshalf_crossphase" "mvpa_searchlight_wholeBrain_crossphase.py" "crossphase" ""
fi

echo "Submitted cross_half_stage jobs (chunked)."
