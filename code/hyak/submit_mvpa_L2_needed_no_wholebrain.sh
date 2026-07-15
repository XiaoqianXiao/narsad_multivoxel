#!/usr/bin/env bash
set -euo pipefail

# Submit the executable MVPA L2 workflow without the whole-brain/Schaefer
# sensitivity analysis:
#   1. FearNetwork primary feature space.
#   2. MemoryFearNetwork mask-sensitivity feature space.
#   3. Post-Hyak harmonization, primary models, SCR sensitivity, and summary.
#
# All jobs run inside /gscratch/fang/images/jupyter.sif through the underlying
# stage submitters.
#
# Usage:
#   hyak/submit_mvpa_L2_needed_no_wholebrain.sh
#
# Optional examples:
#   N_PERMUTATION=1000 N_NULL_PERMS=1000 hyak/submit_mvpa_L2_needed_no_wholebrain.sh
#   SUBMIT_MEMORY=0 hyak/submit_mvpa_L2_needed_no_wholebrain.sh

PROJECT_ROOT="${PROJECT_ROOT:-/gscratch/fang/NARSAD}"
CONTAINER_SIF="${CONTAINER_SIF:-/gscratch/fang/images/jupyter.sif}"
REPO_ROOT="${REPO_ROOT:-/gscratch/scrubbed/fanglab/xiaoqian/repo/narsad_multivoxel/code}"
APP_PATH="${APP_PATH:-${REPO_ROOT}/hyak}"
OUT_BASE="${OUT_BASE:-/gscratch/fang/NARSAD/MRI/derivatives/fMRI_analysis/LSS/results}"
STAGE11_MASK_MODE="${STAGE11_MASK_MODE:-${MVPA_L2_MASK_MODE:-current}}"

case "$STAGE11_MASK_MODE" in
  current)
    FEAR_DEFAULT_NAME="FearNetwork"
    MEMORY_DEFAULT_NAME="MemoryFearNetwork"
    OUT_DEFAULT_NAME="mvpa_l2"
    ;;
  original_notebook)
    FEAR_DEFAULT_NAME="FearNetwork_originalMask"
    MEMORY_DEFAULT_NAME="MemoryFearNetwork_originalMask"
    OUT_DEFAULT_NAME="mvpa_l2_originalMask"
    ;;
  *)
    echo "ERROR: STAGE11_MASK_MODE must be 'current' or 'original_notebook'." >&2
    exit 1
    ;;
esac

SUBMIT_FEAR="${SUBMIT_FEAR:-1}"
SUBMIT_MEMORY="${SUBMIT_MEMORY:-1}"
SUBMIT_POSTHYAK="${SUBMIT_POSTHYAK:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FEAR_SUBMIT="${SCRIPT_DIR}/submit_mvpa_L2_fearnetwork_stage.sh"
MEMORY_SUBMIT="${SCRIPT_DIR}/submit_mvpa_L2_memoryfearnetwork_stage.sh"
POSTHYAK_SUBMIT="${SCRIPT_DIR}/submit_mvpa_L2_posthyak.sh"

print_usage() {
  cat <<'EOF'
Usage:
  submit_mvpa_L2_needed_no_wholebrain.sh

This submits:
  FearNetwork all stages
  MemoryFearNetwork all stages
  post-Hyak harmonization/statistics after both finish

Environment overrides:
  PROJECT_ROOT, CONTAINER_SIF, REPO_ROOT, APP_PATH, OUT_BASE
  PARTITION, ACCOUNT, TIME, MEM, CPUS
  N_JOBS, N_JOBS_CV, N_PERMUTATION, N_NULL_PERMS
  STAGE11_MASK_MODE=current|original_notebook
  STAGE11_ACTUAL_REPEATS, STAGE11_CHUNKS, STAGE11_ARRAY_MAX_RUNNING
  SUBMIT_FEAR=0 to skip FearNetwork submission
  SUBMIT_MEMORY=0 to skip MemoryFearNetwork submission
  SUBMIT_POSTHYAK=0 to skip post-Hyak submission

Whole-brain/Schaefer is intentionally not submitted here.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_usage
  exit 0
fi

extract_final_stage30_job() {
  awk '/stage 30: job / {job=$NF} END {if (job != "") print job}'
}

submit_chain() {
  local label="$1"
  local submit_script="$2"
  local output final_job

  echo "Submitting ${label} full stage chain..."
  output="$("${submit_script}" all)"
  echo "$output"
  final_job="$(printf '%s\n' "$output" | extract_final_stage30_job)"
  if [[ -z "$final_job" ]]; then
    echo "ERROR: Could not parse final stage-30 job ID for ${label}." >&2
    exit 1
  fi
  echo "${label} final dependency job: ${final_job}"
  printf '%s\n' "$final_job"
}

dependency_jobs=()

export PROJECT_ROOT CONTAINER_SIF REPO_ROOT APP_PATH OUT_BASE STAGE11_MASK_MODE

if [[ "$SUBMIT_FEAR" == "1" ]]; then
  fear_final="$(submit_chain "FearNetwork" "$FEAR_SUBMIT" | tail -n 1)"
  dependency_jobs+=("$fear_final")
fi

if [[ "$SUBMIT_MEMORY" == "1" ]]; then
  memory_final="$(submit_chain "MemoryFearNetwork" "$MEMORY_SUBMIT" | tail -n 1)"
  dependency_jobs+=("$memory_final")
fi

if [[ "$SUBMIT_POSTHYAK" == "1" ]]; then
  if [[ ${#dependency_jobs[@]} -eq 0 ]]; then
    dependency_arg=()
  else
    dependency="$(IFS=:; echo "${dependency_jobs[*]}")"
    dependency_arg=(--dependency "$dependency")
  fi

  echo "Submitting post-Hyak job after dependencies: ${dependency_jobs[*]:-none}"
  STAGE11_MASK_MODE="$STAGE11_MASK_MODE" REPO_ROOT="$REPO_ROOT" OUT_BASE="$OUT_BASE" CONTAINER_SIF="$CONTAINER_SIF" \
    FEAR_DIR="/output_dir/$FEAR_DEFAULT_NAME" \
    MEMORY_DIR="/output_dir/$MEMORY_DEFAULT_NAME" \
    SCHAEFER_DIR="" \
    "$POSTHYAK_SUBMIT" "${dependency_arg[@]}"
fi

cat <<EOF

Submitted MVPA L2 workflow without whole-brain/Schaefer sensitivity.
Stage 11 mask mode: ${STAGE11_MASK_MODE}
Container: ${CONTAINER_SIF}
Feature outputs expected under: ${OUT_BASE}/${FEAR_DEFAULT_NAME} and ${OUT_BASE}/${MEMORY_DEFAULT_NAME}
Post-Hyak outputs expected under: ${OUT_BASE}/${OUT_DEFAULT_NAME}
EOF
