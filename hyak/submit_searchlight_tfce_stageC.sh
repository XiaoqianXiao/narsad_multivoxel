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

LOG_DIR="/gscratch/fang/NARSAD/logs/searchlight_tfce"
PARTITION="ckpt-all"
ACCOUNT="fang"
TIME="4:00:00"
MEM="120G"
CPUS=32
N_PERM=5000

mkdir -p "$LOG_DIR"
module load apptainer 2>/dev/null || true

submit() {
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
    --output="$LOG_DIR/${name}_%j.out" \
    --error="$LOG_DIR/${name}_%j.err" \
    --wrap="apptainer exec -B ${PROJECT_ROOT}:${PROJECT_ROOT} -B ${APP_PATH}:/app -B ${OUT_BASE}:/output_dir ${CONTAINER_SIF} python3 /app/${script} --project_root ${PROJECT_ROOT} --out_dir /output_dir/${out_dir}/merged --reference_lss ${REFERENCE_LSS} --glasser_atlas ${GLASSER_ATLAS} --tian_atlas ${TIAN_ATLAS} --n_jobs ${CPUS} --n_perm ${N_PERM} --post_merge_tfce ${extra_args}"
}

# Extinction (per condition)
submit "tfce_ext_csminus" "mvpa_searchlight_wholeBrain_ext.py" "ext" "--cond 'CS-'"
submit "tfce_ext_css"     "mvpa_searchlight_wholeBrain_ext.py" "ext" "--cond CSS"
submit "tfce_ext_csr"     "mvpa_searchlight_wholeBrain_ext.py" "ext" "--cond CSR"

# Reinstatement (per condition)
submit "tfce_rst_csminus" "mvpa_searchlight_wholeBrain_rst.py" "rst" "--cond 'CS-'"
submit "tfce_rst_css"     "mvpa_searchlight_wholeBrain_rst.py" "rst" "--cond CSS"
submit "tfce_rst_csr"     "mvpa_searchlight_wholeBrain_rst.py" "rst" "--cond CSR"

# Dynamic
submit "tfce_dyn_ext"     "mvpa_searchlight_wholeBrain_dyn_ext.py" "dyn_ext" ""
submit "tfce_dyn_rst"     "mvpa_searchlight_wholeBrain_dyn_rst.py" "dyn_rst" ""

# Crossphase
submit "tfce_crossphase"  "mvpa_searchlight_wholeBrain_crossphase.py" "crossphase" ""

echo "Submitted Stage C TFCE jobs."
