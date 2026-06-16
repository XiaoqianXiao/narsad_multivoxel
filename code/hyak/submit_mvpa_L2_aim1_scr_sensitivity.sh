#!/usr/bin/env bash
set -euo pipefail

# Submit Analysis 1 / Stage 6 SCR-subgroup sensitivity runs for the FearNetwork
# feature space. These jobs rerun only the CSR-vs-CSS decoding analysis within
# SCR-defined responder/learner cohorts and save labeled checkpoints, leaving
# the primary cell_06.joblib untouched.
#
# Usage:
#   hyak/submit_mvpa_L2_aim1_scr_sensitivity.sh
#   N_PERMUTATION=1000 hyak/submit_mvpa_L2_aim1_scr_sensitivity.sh
#   SCR_FLAGS=/path/to/scr_sensitivity_groups.csv hyak/submit_mvpa_L2_aim1_scr_sensitivity.sh

PROJECT_ROOT="${PROJECT_ROOT:-/gscratch/fang/NARSAD}"
CONTAINER_SIF="${CONTAINER_SIF:-/gscratch/fang/images/jupyter.sif}"
REPO_ROOT="${REPO_ROOT:-/gscratch/scrubbed/fanglab/xiaoqian/repo/narsad_multivoxel}"
APP_PATH="${APP_PATH:-${REPO_ROOT}/hyak}"
OUT_BASE="${OUT_BASE:-/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/LSS/results}"
OUT_DIR="${OUT_DIR:-/output_dir/FearNetwork}"
ROI_DIR="${FEAR_ROI_DIR:-${PROJECT_ROOT}/ROI/Gillian_anatomically_constrained}"

SCR_FLAGS_HOST="${SCR_FLAGS:-${REPO_ROOT}/outputs/mvpa_l2/harmonized/scr_sensitivity_groups.csv}"
SCR_FLAGS_CONTAINER="/repo/outputs/mvpa_l2/harmonized/scr_sensitivity_groups.csv"

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/mvpa_l2_aim1_scr_sensitivity}"
PARTITION="${PARTITION:-ckpt-all}"
ACCOUNT="${ACCOUNT:-fang}"
TIME="${TIME:-48:00:00}"
MEM="${MEM:-100G}"
CPUS="${CPUS:-16}"

N_JOBS="${N_JOBS:-16}"
N_JOBS_CV="${N_JOBS_CV:-1}"
N_PERMUTATION="${N_PERMUTATION:-1000}"
N_NULL_PERMS="${N_NULL_PERMS:-5000}"

SCR_FLAGS=(
  SCR_Physiological_Responder
  SCR_Simple_Acquisition_Differential_Learner
  SCR_Habituation_Adjusted_Learner
  SCR_Late_Phase_Sensitivity_Learner
)

if [[ ! -f "$SCR_FLAGS_HOST" ]]; then
  cat >&2 <<EOF
ERROR: SCR subgroup CSV not found:
  $SCR_FLAGS_HOST

Build it first, for example:
  bash scripts/run_mvpa_l2_posthyak.sh

or set SCR_FLAGS=/path/to/scr_sensitivity_groups.csv.
EOF
  exit 1
fi

mkdir -p "$LOG_DIR"
mkdir -p "$OUT_BASE"
module load apptainer 2>/dev/null || true

submit_flag() {
  local flag="$1"
  local label="aim1_${flag}"
  label="$(printf '%s' "$label" | tr '[:upper:]' '[:lower:]')"

  local job_id
  job_id=$(
    sbatch --parsable \
      --partition="$PARTITION" \
      --account="$ACCOUNT" \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task="$CPUS" \
      --mem="$MEM" \
      --time="$TIME" \
      --job-name="mvpa_a1_${flag}" \
      --output="$LOG_DIR/mvpa_a1_${flag}_%j.out" \
      --error="$LOG_DIR/mvpa_a1_${flag}_%j.err" \
      --wrap="export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 N_PERMUTATION=${N_PERMUTATION} N_NULL_PERMS=${N_NULL_PERMS}; apptainer exec -B ${PROJECT_ROOT}:${PROJECT_ROOT} -B ${REPO_ROOT}:/repo -B ${APP_PATH}:/app -B ${OUT_BASE}:/output_dir ${CONTAINER_SIF} python3 /app/mvpa_L2_voxel_FearNetwork.py --project_root ${PROJECT_ROOT} --output_dir ${OUT_DIR} --roi_dir ${ROI_DIR} --n_jobs ${N_JOBS} --n_jobs_cv ${N_JOBS_CV} --n_permutation ${N_PERMUTATION} --n_null_perms ${N_NULL_PERMS} --stage 6 --include_subjects_csv ${SCR_FLAGS_CONTAINER} --include_subjects_flag ${flag} --include_subjects_column sub_ID --analysis_label ${label}"
  )
  echo "Submitted Analysis 1 SCR sensitivity ${flag}: job ${job_id}"
}

for flag in "${SCR_FLAGS[@]}"; do
  submit_flag "$flag"
done

cat <<EOF

All SCR subgroup sensitivity jobs submitted.
Outputs will be labeled checkpoints in:
  ${OUT_BASE}/FearNetwork/checkpoints/cell_06_aim1_<scr_flag>.joblib

After jobs finish, summarize with:
  python3 scripts/export_aim1_scr_sensitivity.py \\
    --feature-dir ${OUT_BASE}/FearNetwork \\
    --out ${OUT_BASE}/mvpa_l2/stats/aim1_scr_sensitivity.csv
EOF
