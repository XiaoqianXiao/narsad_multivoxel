#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/gscratch/fang/NARSAD"
CONTAINER_SIF="/gscratch/fang/images/jupyter.sif"
APP_PATH="/gscratch/scrubbed/fanglab/xiaoqian/repo/narsad_multivoxel/hyak"
OUT_BASE="/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/LSS/searchlight"

LOG_DIR="/gscratch/fang/NARSAD/logs/searchlight_crosshalf_merge"
PARTITION="ckpt-all"
ACCOUNT="fang"
TIME="4:00:00"
MEM="64G"
CPUS=8

MODE="dyn"  # all | ext | rst | dyn | crossphase

mkdir -p "$LOG_DIR"
module load apptainer 2>/dev/null || true

submit_mode() {
  local name="$1"
  local mode="$2"
  sbatch \
    --partition="$PARTITION" \
    --account="$ACCOUNT" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$TIME" \
    --job-name="${name}" \
    --output="$LOG_DIR/${name}_%j.out" \
    --error="$LOG_DIR/${name}_%j.err" \
    --wrap="apptainer exec -B ${PROJECT_ROOT}:${PROJECT_ROOT} -B ${APP_PATH}:/app -B ${OUT_BASE}:/output_dir ${CONTAINER_SIF} python3 /app/merge_crosshalf_chunks.py --root /output_dir --mode ${mode}"
}

if [[ "${MODE}" == "all" || "${MODE}" == "ext" ]]; then
  submit_mode "merge_crosshalf_ext" "ext"
fi

if [[ "${MODE}" == "all" || "${MODE}" == "rst" ]]; then
  submit_mode "merge_crosshalf_rst" "rst"
fi

if [[ "${MODE}" == "all" || "${MODE}" == "dyn" ]]; then
  submit_mode "merge_crosshalf_dyn_ext" "dyn_ext"
  submit_mode "merge_crosshalf_dyn_rst" "dyn_rst"
fi

if [[ "${MODE}" == "all" || "${MODE}" == "crossphase" ]]; then
  submit_mode "merge_crosshalf_crossphase" "crossphase"
fi

echo "Submitted per-mode cross-half merge jobs."
