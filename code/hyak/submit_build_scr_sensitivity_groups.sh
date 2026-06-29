#!/usr/bin/env bash
set -euo pipefail

# Submit a lightweight Hyak job to rebuild the SCR sensitivity group CSV inside
# the project container. This is useful before rerunning Aim 1 SCR sensitivity
# Stage 6 jobs.
#
# Usage:
#   hyak/submit_build_scr_sensitivity_groups.sh
#   SCR_DIR=/path/to/scr_analysis_outputs hyak/submit_build_scr_sensitivity_groups.sh
#   OUT_CSV=/path/to/scr_sensitivity_groups.csv hyak/submit_build_scr_sensitivity_groups.sh

PROJECT_ROOT="${PROJECT_ROOT:-/gscratch/fang/NARSAD}"
CONTAINER_SIF="${CONTAINER_SIF:-/gscratch/fang/images/jupyter.sif}"
REPO_ROOT="${REPO_ROOT:-/gscratch/scrubbed/fanglab/xiaoqian/repo/narsad_multivoxel/code}"
OUT_BASE="${OUT_BASE:-/gscratch/fang/NARSAD/MRI/derivatives/fMRI_analysis/LSS/results}"

SCR_DIR="${SCR_DIR:-/gscratch/scrubbed/fanglab/xiaoqian/repo/narsad_multivoxel/results/scr_analysis_outputs}"
OUT_CSV="${OUT_CSV:-${OUT_BASE}/mvpa_l2/harmonized/scr_sensitivity_groups.csv}"

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/mvpa_l2_build_scr_groups}"
PARTITION="${PARTITION:-ckpt-all}"
ACCOUNT="${ACCOUNT:-fang}"
TIME="${TIME:-00:30:00}"
MEM="${MEM:-8G}"
CPUS="${CPUS:-1}"

if [[ ! -d "$SCR_DIR" ]]; then
  cat >&2 <<EOF
ERROR: SCR_DIR does not exist:
  $SCR_DIR

Set SCR_DIR to the directory containing files such as:
  physiological_responder_subjects.txt
  simple_CSplus_learner_subjects.txt
  habituation_adjusted_CSplus_learner_subjects.txt
  late_phase_sensitivity_CSplus_learner_subjects.txt
  scr_acquisition_learner_subjects.csv
EOF
  exit 1
fi

mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$OUT_CSV")"
module load apptainer 2>/dev/null || true

inner_cmd=$(
  cat <<EOF
set -euo pipefail
cd /app
SCR_GROUPS_OUT="/output_dir/mvpa_l2/harmonized/scr_sensitivity_groups.csv"
python3 scripts/build_scr_sensitivity_groups.py --scr-dir /scr_results --out "\${SCR_GROUPS_OUT}"
python3 - <<'PY'
from pathlib import Path
path = Path("/output_dir/mvpa_l2/harmonized/scr_sensitivity_groups.csv")
n_lines = sum(1 for _ in path.open())
if n_lines < 2:
    raise SystemExit(f"ERROR: Built SCR group CSV has no subject rows: {path}")
print(f"Validated SCR group CSV with {n_lines - 1} subject rows -> {path}")
PY
EOF
)
printf -v inner_cmd_q "%q" "$inner_cmd"

job_id=$(
  sbatch --parsable \
    --partition="$PARTITION" \
    --account="$ACCOUNT" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$TIME" \
    --job-name="mvpa_build_scr_groups" \
    --output="$LOG_DIR/mvpa_build_scr_groups_%j.out" \
    --error="$LOG_DIR/mvpa_build_scr_groups_%j.err" \
    --wrap="apptainer exec -B ${PROJECT_ROOT}:${PROJECT_ROOT} -B ${REPO_ROOT}:/app -B ${SCR_DIR}:/scr_results -B ${OUT_BASE}:/output_dir ${CONTAINER_SIF} bash -lc ${inner_cmd_q}"
)

echo "Submitted SCR sensitivity group build job: ${job_id}"
echo "SCR input dir: ${SCR_DIR}"
echo "Output CSV: ${OUT_CSV}"
echo "Logs: ${LOG_DIR}"
