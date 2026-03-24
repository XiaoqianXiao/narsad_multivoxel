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
MEM="120G"
CPUS=32

# Parallel settings (script will auto-tune if not set)
N_JOBS=8
N_JOBS_CV=1

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
  --job-name="mvpa_l2_schaefer" \
  --output="$LOG_DIR/mvpa_l2_schaefer_%j.out" \
  --error="$LOG_DIR/mvpa_l2_schaefer_%j.err" \
  --wrap="export OMP_NUM_THREADS=${CPUS} MKL_NUM_THREADS=${CPUS} OPENBLAS_NUM_THREADS=${CPUS} NUMEXPR_NUM_THREADS=${CPUS} PHASE2_NPZ=${NPZ_DIR}/phase2_X_ext_y_ext_voxels_schaefer_tian.npz PHASE3_NPZ=${NPZ_DIR}/phase3_X_reinst_y_reinst_voxels_schaefer_tian.npz; apptainer exec -B ${PROJECT_ROOT}:${PROJECT_ROOT} -B ${APP_PATH}:/app -B ${OUT_BASE}:/output_dir ${CONTAINER_SIF} python3 /app/mvpa_L2_voxel_WholeBrain_Schaefer.py --project_root ${PROJECT_ROOT} --output_dir /output_dir/wholebrain_parcellation_schaefer --n_jobs ${N_JOBS} --n_jobs_cv ${N_JOBS_CV}"


echo "Submitted mvpa_L2_voxel_WholeBrain_Schaefer job."
