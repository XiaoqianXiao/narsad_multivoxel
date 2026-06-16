#!/usr/bin/env bash
set -euo pipefail

# ---- Match submit_searchlight_jobs.sh settings ----
PROJECT_ROOT="/gscratch/fang/NARSAD"
CONTAINER_SIF="/gscratch/fang/images/jupyter.sif"
APP_PATH="/gscratch/scrubbed/fanglab/xiaoqian/repo/narsad_multivoxel/hyak"
OUT_BASE="/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/LSS/searchlight"
REFERENCE_LSS="/gscratch/fang/NARSAD/MRI/derivatives/fMRI_analysis/LSS/firstLevel/all_subjects/subjects/sub-N101_task-phase2_contrast1.nii.gz"
GLASSER_ATLAS="/gscratch/fang/NARSAD/ROI/Glasser/HCP-MMP1_2mm.nii"
TIAN_ATLAS="/gscratch/fang/NARSAD/ROI/Tian/3T/Subcortex-Only/Tian_Subcortex_S4_3T_2009cAsym.nii.gz"

LOG_DIR="/gscratch/fang/NARSAD/logs/searchlight"
PARTITION="cpu-g2"
ACCOUNT="fang"
TIME="24:00:00"
MEM="120G"
CPUS=32
CHUNKS=384
N_PERM=5000

mkdir -p "$LOG_DIR"
mkdir -p "$OUT_BASE"/{ext,rst,dyn_ext,dyn_rst,crossphase}

module load apptainer 2>/dev/null || true

declare -A SCRIPT_MAP
declare -A OUT_MAP
declare -A EXTRA_MAP

SCRIPT_MAP["sl_ext_csminus"]="mvpa_searchlight_wholeBrain_ext.py"
OUT_MAP["sl_ext_csminus"]="${OUT_BASE}/ext"
EXTRA_MAP["sl_ext_csminus"]="--cond 'CS-' --n_perm ${N_PERM}"

SCRIPT_MAP["sl_ext_css"]="mvpa_searchlight_wholeBrain_ext.py"
OUT_MAP["sl_ext_css"]="${OUT_BASE}/ext"
EXTRA_MAP["sl_ext_css"]="--cond CSS --n_perm ${N_PERM}"

SCRIPT_MAP["sl_ext_csr"]="mvpa_searchlight_wholeBrain_ext.py"
OUT_MAP["sl_ext_csr"]="${OUT_BASE}/ext"
EXTRA_MAP["sl_ext_csr"]="--cond CSR --n_perm ${N_PERM}"

SCRIPT_MAP["sl_rst_csminus"]="mvpa_searchlight_wholeBrain_rst.py"
OUT_MAP["sl_rst_csminus"]="${OUT_BASE}/rst"
EXTRA_MAP["sl_rst_csminus"]="--cond 'CS-' --n_perm ${N_PERM}"

SCRIPT_MAP["sl_rst_css"]="mvpa_searchlight_wholeBrain_rst.py"
OUT_MAP["sl_rst_css"]="${OUT_BASE}/rst"
EXTRA_MAP["sl_rst_css"]="--cond CSS --n_perm ${N_PERM}"

SCRIPT_MAP["sl_rst_csr"]="mvpa_searchlight_wholeBrain_rst.py"
OUT_MAP["sl_rst_csr"]="${OUT_BASE}/rst"
EXTRA_MAP["sl_rst_csr"]="--cond CSR --n_perm ${N_PERM}"

SCRIPT_MAP["dyn_ext"]="mvpa_searchlight_wholeBrain_dyn_ext.py"
OUT_MAP["dyn_ext"]="${OUT_BASE}/dyn_ext"
EXTRA_MAP["dyn_ext"]="--n_perm ${N_PERM}"

SCRIPT_MAP["dyn_rst"]="mvpa_searchlight_wholeBrain_dyn_rst.py"
OUT_MAP["dyn_rst"]="${OUT_BASE}/dyn_rst"
EXTRA_MAP["dyn_rst"]="--n_perm ${N_PERM}"

SCRIPT_MAP["crossphase"]="mvpa_searchlight_wholeBrain_crossphase.py"
OUT_MAP["crossphase"]="${OUT_BASE}/crossphase"
EXTRA_MAP["crossphase"]="--n_perm ${N_PERM}"

submit_indices() {
  local name="$1"
  local indices="$2"
  local script="${SCRIPT_MAP[$name]}"
  local out_dir="${OUT_MAP[$name]}"
  local extra_args="${EXTRA_MAP[$name]}"

  if [[ -z "${script:-}" ]]; then
    echo "Unknown job name: $name"
    return
  fi
  if [[ -z "$indices" ]]; then
    return
  fi

  sbatch \
    --partition="$PARTITION" \
    --account="$ACCOUNT" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$TIME" \
    --job-name="$name" \
    --array="$indices" \
    --output="$LOG_DIR/${name}_%A_%a.out" \
    --error="$LOG_DIR/${name}_%A_%a.err" \
    --wrap="mkdir -p ${out_dir} && apptainer exec -B ${PROJECT_ROOT}:${PROJECT_ROOT} -B ${APP_PATH}:/app -B ${out_dir}:/output_dir ${CONTAINER_SIF} python3 /app/${script} --project_root ${PROJECT_ROOT} --out_dir /output_dir --reference_lss ${REFERENCE_LSS} --glasser_atlas ${GLASSER_ATLAS} --tian_atlas ${TIAN_ATLAS} --n_jobs ${CPUS} --batch_size 256 --chunk_idx \$SLURM_ARRAY_TASK_ID --chunk_count ${CHUNKS} ${extra_args}"
}

extract_failed_indices() {
  local prefix="$1"
  local pattern="${LOG_DIR}/${prefix}_*_*\.err"
  local idxs
  idxs=$(ls ${pattern} 2>/dev/null | awk -F'_' '{print $NF}' | sed 's/\\.err$//' | sort -n | uniq)
  if [[ -z "$idxs" ]]; then
    return
  fi
  local list
  list=$(echo "$idxs" | paste -sd, -)
  echo "$list"
}

if [[ "$#" -gt 0 ]]; then
  # Usage: bash resubmit_failed_searchlight.sh sl_ext_css:8,306,333 sl_ext_csminus:314
  for item in "$@"; do
    name="${item%%:*}"
    indices="${item#*:}"
    if [[ -z "${indices}" || "${indices}" == "${name}" ]]; then
      echo "Invalid argument: ${item}. Use name:idx,idx"
      continue
    fi
    echo "Resubmitting ${name} indices: ${indices}"
    submit_indices "$name" "$indices"
  done
else
  # Auto-resubmit all indices that have .err files
  for name in "${!SCRIPT_MAP[@]}"; do
    indices=$(extract_failed_indices "$name")
    if [[ -n "$indices" ]]; then
      echo "Resubmitting ${name} indices: ${indices}"
      submit_indices "$name" "$indices"
    fi
  done
fi
