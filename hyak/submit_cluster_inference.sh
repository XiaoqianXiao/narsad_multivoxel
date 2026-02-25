#!/usr/bin/env bash
set -euo pipefail

# ---- User-configurable settings (aligned with submit_searchlight_stageA.sh) ----
PROJECT_ROOT="/gscratch/fang/NARSAD"
CONTAINER_SIF="/gscratch/fang/images/jupyter.sif"
APP_PATH="/gscratch/scrubbed/fanglab/xiaoqian/repo/narsad_multivoxel/hyak"
OUT_BASE="/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/LSS/searchlight"
REFERENCE_LSS="/gscratch/fang/NARSAD/MRI/derivatives/fMRI_analysis/LSS/firstLevel/all_subjects/subjects/sub-N101_task-phase2_contrast1.nii.gz"
GLASSER_ATLAS="/gscratch/fang/NARSAD/ROI/Glasser/HCP-MMP1_2mm.nii"
TIAN_ATLAS="/gscratch/fang/NARSAD/ROI/Tian/3T/Subcortex-Only/Tian_Subcortex_S4_3T_2009cAsym.nii.gz"

LOG_DIR="/gscratch/fang/NARSAD/logs/searchlight"
PARTITION="ckpt-all"
ACCOUNT="fang"
TIME="4:00:00"
MEM="120G"
CPUS=16

mkdir -p "$LOG_DIR"
mkdir -p "$OUT_BASE"/{ext,rst,dyn_ext,dyn_rst,crossphase}

# 1. Load the container module
module load apptainer 2>/dev/null || true

submit_job() {
  local name="$1"
  local script="$2"
  local out_dir="$3"
  sbatch \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${CPUS}" \
    --mem="${MEM}" \
    --time="${TIME}" \
    --job-name="${name}" \
    --output="${LOG_DIR}/cluster_${name}_%j.out" \
    --error="${LOG_DIR}/cluster_${name}_%j.err" \
    --wrap="mkdir -p ${out_dir} && apptainer exec -B ${PROJECT_ROOT}:${PROJECT_ROOT} -B ${APP_PATH}:/app -B ${out_dir}:/output_dir ${CONTAINER_SIF} python3 /app/${script} --project_root ${PROJECT_ROOT} --out_dir /output_dir --reference_lss ${REFERENCE_LSS} --glasser_atlas ${GLASSER_ATLAS} --tian_atlas ${TIAN_ATLAS} --n_jobs ${CPUS}"
}

submit_job "ext" "cluster_inference_ext.py" "${OUT_BASE}/ext/merged"
submit_job "rst" "cluster_inference_rst.py" "${OUT_BASE}/rst/merged"
submit_job "dyn_ext" "cluster_inference_dyn_ext.py" "${OUT_BASE}/dyn_ext/merged"
submit_job "dyn_rst" "cluster_inference_dyn_rst.py" "${OUT_BASE}/dyn_rst/merged"
submit_job "crossphase" "cluster_inference_crossphase.py" "${OUT_BASE}/crossphase/merged"
