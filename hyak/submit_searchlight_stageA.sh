#!/usr/bin/env bash
set -euo pipefail

# ---- User-configurable settings (aligned with run_mvpa.sh) ----
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
CPUS=32
CHUNKS=384
N_PERM=5000
MODE="ext"  # all | ext | rst | dyn | crossphase
POST_MERGE_FLAG="--post_merge_tfce"

mkdir -p "$LOG_DIR"
mkdir -p "$OUT_BASE"/{ext,rst,dyn_ext,dyn_rst,crossphase}

# 1. Load the container module
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
    --wrap="mkdir -p ${out_dir} && apptainer exec -B ${PROJECT_ROOT}:${PROJECT_ROOT} -B ${APP_PATH}:/app -B ${out_dir}:/output_dir ${CONTAINER_SIF} python3 /app/${script} --project_root ${PROJECT_ROOT} --out_dir /output_dir --reference_lss ${REFERENCE_LSS} --glasser_atlas ${GLASSER_ATLAS} --tian_atlas ${TIAN_ATLAS} --n_jobs ${CPUS} --batch_size 256 --save_trial_npz ${POST_MERGE_FLAG} ${extra_args}"
}

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
    --wrap="mkdir -p ${out_dir} && apptainer exec -B ${PROJECT_ROOT}:${PROJECT_ROOT} -B ${APP_PATH}:/app -B ${out_dir}:/output_dir ${CONTAINER_SIF} python3 /app/${script} --project_root ${PROJECT_ROOT} --out_dir /output_dir --reference_lss ${REFERENCE_LSS} --glasser_atlas ${GLASSER_ATLAS} --tian_atlas ${TIAN_ATLAS} --n_jobs ${CPUS} --batch_size 256 --chunk_idx \$SLURM_ARRAY_TASK_ID --chunk_count ${CHUNKS} --save_trial_npz ${POST_MERGE_FLAG} ${extra_args}"
}

# ---- Scripts ----
if [[ "${MODE}" == "all" || "${MODE}" == "ext" ]]; then
  submit_array "sl_ext_csminus" "mvpa_searchlight_wholeBrain_ext.py" "${OUT_BASE}/ext" "--cond 'CS-' --n_perm ${N_PERM}"
  submit_array "sl_ext_css"     "mvpa_searchlight_wholeBrain_ext.py" "${OUT_BASE}/ext" "--cond CSS --n_perm ${N_PERM}"
  submit_array "sl_ext_csr"     "mvpa_searchlight_wholeBrain_ext.py" "${OUT_BASE}/ext" "--cond CSR --n_perm ${N_PERM}"
fi

if [[ "${MODE}" == "all" || "${MODE}" == "rst" ]]; then
  submit_array "sl_rst_csminus" "mvpa_searchlight_wholeBrain_rst.py" "${OUT_BASE}/rst" "--cond 'CS-' --n_perm ${N_PERM}"
  submit_array "sl_rst_css"     "mvpa_searchlight_wholeBrain_rst.py" "${OUT_BASE}/rst" "--cond CSS --n_perm ${N_PERM}"
  submit_array "sl_rst_csr"     "mvpa_searchlight_wholeBrain_rst.py" "${OUT_BASE}/rst" "--cond CSR --n_perm ${N_PERM}"
fi

if [[ "${MODE}" == "all" || "${MODE}" == "dyn" ]]; then
  submit_array "dyn_ext"     "mvpa_searchlight_wholeBrain_dyn_ext.py" "${OUT_BASE}/dyn_ext" "--n_perm ${N_PERM}"
  submit_array "dyn_rst"     "mvpa_searchlight_wholeBrain_dyn_rst.py" "${OUT_BASE}/dyn_rst" "--n_perm ${N_PERM}"
fi

if [[ "${MODE}" == "all" || "${MODE}" == "crossphase" ]]; then
  submit_array "crossphase"  "mvpa_searchlight_wholeBrain_crossphase.py" "${OUT_BASE}/crossphase" "--n_perm ${N_PERM}"
fi

echo "Submitted all searchlight jobs."
echo "After jobs finish, merge chunk outputs:"
echo "  apptainer exec -B ${PROJECT_ROOT}:${PROJECT_ROOT} -B ${APP_PATH}:/app -B ${OUT_BASE}:/output_dir ${CONTAINER_SIF} python3 /app/merge_searchlight_chunks.py --in_dir /output_dir/ext --out_dir /output_dir/ext/merged"
echo "  apptainer exec -B ${PROJECT_ROOT}:${PROJECT_ROOT} -B ${APP_PATH}:/app -B ${OUT_BASE}:/output_dir ${CONTAINER_SIF} python3 /app/merge_searchlight_chunks.py --in_dir /output_dir/rst --out_dir /output_dir/rst/merged"
echo "  apptainer exec -B ${PROJECT_ROOT}:${PROJECT_ROOT} -B ${APP_PATH}:/app -B ${OUT_BASE}:/output_dir ${CONTAINER_SIF} python3 /app/merge_searchlight_chunks.py --in_dir /output_dir/dyn_ext --out_dir /output_dir/dyn_ext/merged"
echo "  apptainer exec -B ${PROJECT_ROOT}:${PROJECT_ROOT} -B ${APP_PATH}:/app -B ${OUT_BASE}:/output_dir ${CONTAINER_SIF} python3 /app/merge_searchlight_chunks.py --in_dir /output_dir/dyn_rst --out_dir /output_dir/dyn_rst/merged"
echo "  apptainer exec -B ${PROJECT_ROOT}:${PROJECT_ROOT} -B ${APP_PATH}:/app -B ${OUT_BASE}:/output_dir ${CONTAINER_SIF} python3 /app/merge_searchlight_chunks.py --in_dir /output_dir/crossphase --out_dir /output_dir/crossphase/merged"
