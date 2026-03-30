#!/usr/bin/env bash
set -euo pipefail

# ---- User-configurable settings ----
PROJECT_ROOT="/gscratch/fang/NARSAD"
CONTAINER_SIF="/gscratch/fang/images/jupyter.sif"
APP_PATH="/gscratch/scrubbed/fanglab/xiaoqian/repo/narsad_multivoxel/hyak"
OUT_BASE="/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/LSS/results"
NPZ_DIR="/gscratch/fang/NARSAD/MRI/derivatives/fMRI_analysis/LSS/firstLevel/all_subjects/group_level"

LOG_DIR="/gscratch/fang/NARSAD/logs/mvpa_l2_schaefer"
PARTITION="cpu-g2"
ACCOUNT="fang"
TIME="96:00:00"
MEM="100G"
CPUS=16

# Parallel settings
N_JOBS=16
N_JOBS_CV=1

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <stage_id 6-17> [--resume]"
  exit 1
fi
STAGE="$1"
shift
EXTRA_ARGS="$*"

mkdir -p "$LOG_DIR"
mkdir -p "$OUT_BASE"
module load apptainer 2>/dev/null || true

sbatch \
  --partition="$PARTITION" \
  --account="$ACCOUNT" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="$CPUS" \
  --mem="$MEM" \
  --time="$TIME" \
  --job-name="mvpa_l2_schaefer_s${STAGE}" \
  --output="$LOG_DIR/mvpa_l2_schaefer_s${STAGE}_%j.out" \
  --error="$LOG_DIR/mvpa_l2_schaefer_s${STAGE}_%j.err" \
  --wrap="export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PHASE2_NPZ=${NPZ_DIR}/phase2_X_ext_y_ext_voxels_schaefer_tian.npz PHASE3_NPZ=${NPZ_DIR}/phase3_X_reinst_y_reinst_voxels_schaefer_tian.npz; apptainer exec -B ${PROJECT_ROOT}:${PROJECT_ROOT} -B ${APP_PATH}:/app -B ${OUT_BASE}:/output_dir ${CONTAINER_SIF} python3 /app/mvpa_L2_voxel_WholeBrain_Schaefer.py --project_root ${PROJECT_ROOT} --output_dir /output_dir/wholebrain_parcellation_schaefer --n_jobs ${N_JOBS} --n_jobs_cv ${N_JOBS_CV} --stage ${STAGE} ${EXTRA_ARGS}"

echo "Submitted mvpa_L2_schaefer stage ${STAGE}."
