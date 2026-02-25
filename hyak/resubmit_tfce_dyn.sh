#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/gscratch/fang/NARSAD"
CONTAINER_SIF="/gscratch/fang/images/jupyter.sif"
APP_PATH="/gscratch/scrubbed/fanglab/xiaoqian/repo/narsad_multivoxel/hyak"
OUT_BASE="/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/LSS/searchlight"
REFERENCE_LSS="/gscratch/fang/NARSAD/MRI/derivatives/fMRI_analysis/LSS/firstLevel/all_subjects/subjects/sub-N101_task-phase2_contrast1.nii.gz"
GLASSER_ATLAS="/gscratch/fang/NARSAD/ROI/Glasser/HCP-MMP1_2mm.nii"
TIAN_ATLAS="/gscratch/fang/NARSAD/ROI/Tian/3T/Subcortex-Only/Tian_Subcortex_S4_3T_2009cAsym.nii.gz"

LOG_DIR="/gscratch/fang/NARSAD/logs/searchlight_tfce"
PARTITION="cpu-g2"
ACCOUNT="fang"
TIME="24:00:00"
MEM="120G"
CPUS=32
N_PERM=5000

mkdir -p "$LOG_DIR"
module load apptainer 2>/dev/null || true

submit() {
  local name="$1"
  local script="$2"
  local out_dir="$3"
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
    --wrap="apptainer exec -B ${PROJECT_ROOT}:${PROJECT_ROOT} -B ${APP_PATH}:/app -B ${OUT_BASE}:/output_dir ${CONTAINER_SIF} \
      python3 /app/${script} \
      --project_root ${PROJECT_ROOT} \
      --out_dir /output_dir/${out_dir}/merged \
      --reference_lss ${REFERENCE_LSS} \
      --glasser_atlas ${GLASSER_ATLAS} \
      --tian_atlas ${TIAN_ATLAS} \
      --n_jobs ${CPUS} \
      --n_perm ${N_PERM} \
      --post_merge_tfce"
}

submit "tfce_dyn_ext_8h" "mvpa_searchlight_wholeBrain_dyn_ext.py" "dyn_ext"
submit "tfce_dyn_rst_8h" "mvpa_searchlight_wholeBrain_dyn_rst.py" "dyn_rst"

echo "Submitted dyn TFCE re-run jobs with 8h time limit."
