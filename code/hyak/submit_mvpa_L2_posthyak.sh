#!/usr/bin/env bash
set -euo pipefail

# Submit the lightweight post-Hyak MVPA L2 harmonization and statistics job.
#
# This runs inside /gscratch/fang/images/jupyter.sif and does not require the
# whole-brain Schaefer + Tian sensitivity output unless SCHAEFER_DIR is explicitly set.
#
# Usage:
#   hyak/submit_mvpa_L2_posthyak.sh
#   hyak/submit_mvpa_L2_posthyak.sh --dependency 12345:12346
#   FEAR_DIR=/output_dir/FearNetwork MEMORY_DIR=/output_dir/MemoryFearNetwork hyak/submit_mvpa_L2_posthyak.sh

PROJECT_ROOT="${PROJECT_ROOT:-/gscratch/fang/NARSAD}"
CONTAINER_SIF="${CONTAINER_SIF:-/gscratch/fang/images/jupyter.sif}"
REPO_ROOT="${REPO_ROOT:-/gscratch/scrubbed/fanglab/xiaoqian/repo/narsad_multivoxel/code}"
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

FEAR_DIR="${FEAR_DIR:-/output_dir/$FEAR_DEFAULT_NAME}"
MEMORY_DIR="${MEMORY_DIR:-/output_dir/$MEMORY_DEFAULT_NAME}"
SCHAEFER_DIR="${SCHAEFER_DIR:-}"
SCR_FLAGS="${SCR_FLAGS:-/output_dir/$OUT_DEFAULT_NAME/harmonized/scr_sensitivity_groups.csv}"
SCR_DIR="${SCR_DIR:-/app/scr_analysis_outputs}"
OUT_ROOT="${OUT_ROOT:-/output_dir/$OUT_DEFAULT_NAME}"

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/mvpa_l2_posthyak}"
PARTITION="${PARTITION:-ckpt-all}"
ACCOUNT="${ACCOUNT:-fang}"
TIME="${TIME:-08:00:00}"
MEM="${MEM:-32G}"
CPUS="${CPUS:-4}"
DEPENDENCY=""

print_usage() {
  cat <<'EOF'
Usage:
  submit_mvpa_L2_posthyak.sh [--dependency JOBID[:JOBID...]]

Environment overrides:
  PROJECT_ROOT, CONTAINER_SIF, REPO_ROOT, OUT_BASE
  STAGE11_MASK_MODE, FEAR_DIR, MEMORY_DIR, SCHAEFER_DIR, SCR_FLAGS, SCR_DIR, OUT_ROOT
  LOG_DIR, PARTITION, ACCOUNT, TIME, MEM, CPUS

Defaults:
  CONTAINER_SIF=/gscratch/fang/images/jupyter.sif
  FEAR_DIR=/output_dir/FearNetwork or /output_dir/FearNetwork_originalMask when STAGE11_MASK_MODE=original_notebook
  MEMORY_DIR=/output_dir/MemoryFearNetwork or /output_dir/MemoryFearNetwork_originalMask when STAGE11_MASK_MODE=original_notebook
  SCR_FLAGS=/output_dir/mvpa_l2/harmonized/scr_sensitivity_groups.csv or /output_dir/mvpa_l2_originalMask/harmonized/scr_sensitivity_groups.csv
  SCHAEFER_DIR is empty, so whole-brain/parcellation sensitivity is skipped.
  SCR_DIR is only used as a fallback when SCR_FLAGS is missing.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dependency)
      DEPENDENCY="${2:-}"
      shift 2
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      print_usage >&2
      exit 1
      ;;
  esac
done

mkdir -p "$LOG_DIR"
module load apptainer 2>/dev/null || true

dependency_args=()
if [[ -n "$DEPENDENCY" ]]; then
  dependency_args=(--dependency="afterok:${DEPENDENCY}")
fi

bind_args=(-B "${PROJECT_ROOT}:${PROJECT_ROOT}" -B "${REPO_ROOT}:/app" -B "${OUT_BASE}:/output_dir")
if [[ -n "$SCR_DIR" && "$SCR_DIR" = /* && -d "$SCR_DIR" ]]; then
  bind_args+=(-B "${SCR_DIR}:${SCR_DIR}")
elif [[ -n "$SCR_DIR" && "$SCR_DIR" = /* ]]; then
  echo "WARNING: SCR_DIR does not exist; skipping fallback SCR bind: $SCR_DIR" >&2
fi

inner_cmd=$(
  cat <<EOF
set -euo pipefail
cd /app
export STAGE11_MASK_MODE='${STAGE11_MASK_MODE}'
export FEAR_DIR='${FEAR_DIR}'
export MEMORY_DIR='${MEMORY_DIR}'
export SCHAEFER_DIR='${SCHAEFER_DIR}'
export SCR_FLAGS='${SCR_FLAGS}'
export SCR_DIR='${SCR_DIR}'
export OUT_ROOT='${OUT_ROOT}'
export PYTHON_BIN='python3'
bash scripts/run_mvpa_l2_posthyak.sh
EOF
)
printf -v inner_cmd_q "%q" "$inner_cmd"

job_id=$(
  sbatch --parsable \
    "${dependency_args[@]}" \
    --partition="$PARTITION" \
    --account="$ACCOUNT" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$TIME" \
    --job-name="mvpa_l2_posthyak" \
    --output="$LOG_DIR/mvpa_l2_posthyak_%j.out" \
    --error="$LOG_DIR/mvpa_l2_posthyak_%j.err" \
    --wrap="apptainer exec ${bind_args[*]} ${CONTAINER_SIF} bash -lc ${inner_cmd_q}"
)

echo "Submitted MVPA L2 post-Hyak job: ${job_id}"
echo "Container: ${CONTAINER_SIF}"
echo "Output root: ${OUT_ROOT}"
echo "Whole-brain Schaefer+Tian sensitivity: ${SCHAEFER_DIR:-skipped}"
