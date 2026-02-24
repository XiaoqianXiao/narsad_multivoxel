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

LOG_DIR="/gscratch/fang/NARSAD/logs/searchlight_merge"
PARTITION="ckpt-all"
ACCOUNT="fang"
TIME="2:00:00"
MEM="32G"
CPUS=8

mkdir -p "$LOG_DIR"
module load apptainer 2>/dev/null || true

submit_merge() {
  local name="$1"
  local subdir="$2"
  sbatch \
    --partition="$PARTITION" \
    --account="$ACCOUNT" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$TIME" \
    --job-name="$name" \
    --output="$LOG_DIR/${name}_%j.out" \
    --error="$LOG_DIR/${name}_%j.err" \
    --wrap="apptainer exec -B ${PROJECT_ROOT}:${PROJECT_ROOT} -B ${APP_PATH}:/app -B ${OUT_BASE}:/output_dir ${CONTAINER_SIF} python3 /app/merge_searchlight_chunks.py --in_dir /output_dir/${subdir} --out_dir /output_dir/${subdir}/merged --reference_lss ${REFERENCE_LSS} --glasser_atlas ${GLASSER_ATLAS} --tian_atlas ${TIAN_ATLAS}"
}

submit_merge "merge_ext" "ext"
submit_merge "merge_rst" "rst"
submit_merge "merge_dyn_ext" "dyn_ext"
submit_merge "merge_dyn_rst" "dyn_rst"
submit_merge "merge_crossphase" "crossphase"

echo "Submitted Stage B merge jobs in parallel."
