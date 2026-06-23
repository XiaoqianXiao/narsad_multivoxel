#!/usr/bin/env bash
set -euo pipefail

# Submit the lightweight exporter for Analysis 1 SCR-subgroup sensitivity jobs.
#
# This should be run after hyak/submit_mvpa_L2_aim1_scr_sensitivity.sh jobs
# have finished and written labeled cell_06_aim1_*.joblib checkpoints.
#
# Usage:
#   hyak/submit_export_aim1_scr_sensitivity.sh
#   hyak/submit_export_aim1_scr_sensitivity.sh --dependency 12345:12346

PROJECT_ROOT="${PROJECT_ROOT:-/gscratch/fang/NARSAD}"
CONTAINER_SIF="${CONTAINER_SIF:-/gscratch/fang/images/jupyter.sif}"
REPO_ROOT="${REPO_ROOT:-/gscratch/scrubbed/fanglab/xiaoqian/repo/narsad_multivoxel/code}"
OUT_BASE="${OUT_BASE:-/gscratch/fang/NARSAD/MRI/derivatives/fMRI_analysis/LSS/results}"

FEATURE_DIR="${FEATURE_DIR:-/output_dir/FearNetwork}"
OUT_CSV="${OUT_CSV:-/output_dir/mvpa_l2/stats/aim1_scr_sensitivity.csv}"
FEATURE_SPACE="${FEATURE_SPACE:-FearNetwork}"

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/mvpa_l2_aim1_scr_export}"
PARTITION="${PARTITION:-ckpt-all}"
ACCOUNT="${ACCOUNT:-fang}"
TIME="${TIME:-01:00:00}"
MEM="${MEM:-8G}"
CPUS="${CPUS:-1}"
DEPENDENCY=""

print_usage() {
  cat <<'EOF'
Usage:
  submit_export_aim1_scr_sensitivity.sh [--dependency JOBID[:JOBID...]]

Environment overrides:
  PROJECT_ROOT, CONTAINER_SIF, REPO_ROOT, OUT_BASE
  FEATURE_DIR, OUT_CSV, FEATURE_SPACE
  LOG_DIR, PARTITION, ACCOUNT, TIME, MEM, CPUS

Defaults:
  FEATURE_DIR=/output_dir/FearNetwork
  OUT_CSV=/output_dir/mvpa_l2/stats/aim1_scr_sensitivity.csv
  CONTAINER_SIF=/gscratch/fang/images/jupyter.sif
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

inner_cmd=$(
  cat <<EOF
set -euo pipefail
cd /app
export FEATURE_DIR='${FEATURE_DIR}'
export OUT_CSV='${OUT_CSV}'
export FEATURE_SPACE='${FEATURE_SPACE}'
python3 scripts/export_aim1_scr_sensitivity.py --feature-dir "\${FEATURE_DIR}" --out "\${OUT_CSV}" --feature-space "\${FEATURE_SPACE}"
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
    --job-name="mvpa_a1_scr_export" \
    --output="$LOG_DIR/mvpa_a1_scr_export_%j.out" \
    --error="$LOG_DIR/mvpa_a1_scr_export_%j.err" \
    --wrap="apptainer exec -B ${PROJECT_ROOT}:${PROJECT_ROOT} -B ${REPO_ROOT}:/app -B ${OUT_BASE}:/output_dir ${CONTAINER_SIF} bash -lc ${inner_cmd_q}"
)

echo "Submitted Analysis 1 SCR sensitivity export job: ${job_id}"
echo "Feature dir: ${FEATURE_DIR}"
echo "Output CSV: ${OUT_CSV}"
