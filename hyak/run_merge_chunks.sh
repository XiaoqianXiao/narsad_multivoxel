#!/usr/bin/env bash
#SBATCH --partition=ckpt-all
#SBATCH --account=fang
#SBATCH --time=4:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=32
#SBATCH --job-name=merge_searchlight
#SBATCH --output=/gscratch/fang/NARSAD/logs/searchlight/merge_%j.out
#SBATCH --error=/gscratch/fang/NARSAD/logs/searchlight/merge_%j.err
set -euo pipefail

# Run merge_searchlight_chunks.py for all merged folders on Hyak.

PROJECT_ROOT="/gscratch/fang/NARSAD"
CONTAINER_SIF="/gscratch/fang/images/jupyter.sif"
APP_PATH="/gscratch/scrubbed/fanglab/xiaoqian/repo/narsad_multivoxel/hyak"
OUT_BASE="/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/LSS/searchlight"

REFERENCE_LSS="/gscratch/fang/NARSAD/MRI/derivatives/fMRI_analysis/LSS/firstLevel/all_subjects/subjects/sub-N101_task-phase2_contrast1.nii.gz"
GLASSER_ATLAS="/gscratch/fang/NARSAD/ROI/Glasser/HCP-MMP1_2mm.nii"
TIAN_ATLAS="/gscratch/fang/NARSAD/ROI/Tian/3T/Subcortex-Only/Tian_Subcortex_S4_3T_2009cAsym.nii.gz"

module load apptainer 2>/dev/null || true

run_merge() {
  local name="$1"
  local in_dir="${OUT_BASE}/${name}"
  local out_dir="${OUT_BASE}/${name}/merged"
  mkdir -p "$out_dir"
  apptainer exec \
    -B "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    -B "${APP_PATH}:/app" \
    -B "${OUT_BASE}:/output_dir" \
    "${CONTAINER_SIF}" \
    python3 /app/merge_searchlight_chunks.py \
      --in_dir "/output_dir/${name}" \
      --out_dir "/output_dir/${name}/merged" \
      --reference_lss "${REFERENCE_LSS}" \
      --glasser_atlas "${GLASSER_ATLAS}" \
      --tian_atlas "${TIAN_ATLAS}"
}

run_merge "ext"
run_merge "rst"
run_merge "dyn_ext"
run_merge "dyn_rst"
run_merge "crossphase"

echo "All merges complete."
