#!/bin/bash
set -euo pipefail

PARTITION="ckpt-all"
ACCOUNT="fang"
TIME="4:00:00"
MEM="120G"
CPUS=16

SCRIPT_DIR="/Users/xiaoqianxiao/PycharmProjects/narsad_multivoxel/hyak"
CONTAINER="/gscratch/fang/images/jupyter.sif"
BIND1="/gscratch/fang/NARSAD:/gscratch/fang/NARSAD"
BIND2="/gscratch/scrubbed/fanglab/xiaoqian/NARSAD:/gscratch/scrubbed/fanglab/xiaoqian/NARSAD"

submit_job() {
  local name="$1"
  local script="$2"
  sbatch --job-name="${name}" \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --time="${TIME}" \
    --mem="${MEM}" \
    --cpus-per-task="${CPUS}" \
    --output="cluster_${name}_%j.out" \
    --error="cluster_${name}_%j.err" \
    --wrap="apptainer exec -B ${BIND1} -B ${BIND2} ${CONTAINER} python3 ${script} --n_jobs ${CPUS}"
}

submit_job "ext" "${SCRIPT_DIR}/cluster_inference_ext.py"
submit_job "rst" "${SCRIPT_DIR}/cluster_inference_rst.py"
submit_job "dyn_ext" "${SCRIPT_DIR}/cluster_inference_dyn_ext.py"
submit_job "dyn_rst" "${SCRIPT_DIR}/cluster_inference_dyn_rst.py"
submit_job "crossphase" "${SCRIPT_DIR}/cluster_inference_crossphase.py"
