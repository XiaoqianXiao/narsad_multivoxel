#!/usr/bin/env bash
set -euo pipefail

# Submit mvpa_L2_voxel_WholeBrain_Schaefer.py by logical stage.
# Usage:
#   ./submit_mvpa_L2_schaefer_stage.sh all
#   ./submit_mvpa_L2_schaefer_stage.sh 1
#   ./submit_mvpa_L2_schaefer_stage.sh 2 --resume
#   ./submit_mvpa_L2_schaefer_stage.sh --help

PROJECT_ROOT="${PROJECT_ROOT:-/gscratch/fang/NARSAD}"
CONTAINER_SIF="${CONTAINER_SIF:-/gscratch/fang/images/jupyter.sif}"
APP_PATH="${APP_PATH:-/gscratch/scrubbed/fanglab/xiaoqian/repo/narsad_multivoxel/hyak}"
OUT_BASE="${OUT_BASE:-/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/LSS/results}"
OUT_DIR="${OUT_DIR:-/output_dir/wholebrain_parcellation_schaefer}"
NPZ_DIR="${NPZ_DIR:-${PROJECT_ROOT}/MRI/derivatives/fMRI_analysis/LSS/firstLevel/all_subjects/group_level}"

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/mvpa_l2_schaefer}"
PARTITION="${PARTITION:-ckpt-all}"
ACCOUNT="${ACCOUNT:-fang}"
TIME="${TIME:-96:00:00}"
MEM="${MEM:-120G}"
CPUS="${CPUS:-32}"

N_JOBS="${N_JOBS:-16}"
N_JOBS_CV="${N_JOBS_CV:-1}"
N_PERMUTATION="${N_PERMUTATION:-5000}"
N_NULL_PERMS="${N_NULL_PERMS:-5000}"
STAGE1_ACTUAL_REPEATS="${STAGE1_ACTUAL_REPEATS:-${N_NULL_PERMS}}"
STAGE1_CHUNKS="${STAGE1_CHUNKS:-1}"
STAGE1_ARRAY_MAX_RUNNING="${STAGE1_ARRAY_MAX_RUNNING:-10}"
STAGE1_GROUPS=(SAD HC)

ALL_STAGES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)

print_usage() {
  cat <<'EOF'
Usage:
  submit_mvpa_L2_schaefer_stage.sh all
      Submit stages 1 -> 16 as a dependency chain.

  submit_mvpa_L2_schaefer_stage.sh <stage_id> [extra python args]
      Submit one logical stage. Extra args are passed through to
      mvpa_L2_voxel_WholeBrain_Schaefer.py.

  submit_mvpa_L2_schaefer_stage.sh 1:SAD
  submit_mvpa_L2_schaefer_stage.sh 1:HC
      Submit one group for Stage 1 empirical permutation importance. This is
      mainly useful after Analysis 1.1 models already exist and you intentionally
      want to compute/merge split importance chunks for one group.

Stage reference:
  1   Analysis 1.1 models/permutations plus empirical permutation-importance masks.
  2   Analysis 1.2 static representational topology.
  3   Analysis 1.3 dynamic drift and single-trial trajectories.
  4   Analysis 1.4 decision-boundary/self-network statistics.
  5   Analysis 2.1 safety restoration/threat discrimination.
  6   Analysis 2.2 drift efficiency.
  7   Analysis 2.3 probabilistic opening.
  8   Analysis 2.4 spatial re-alignment.
  9   Analysis 2.5 reverse cross-decoding.
  10  Clinical score loading.
  11  Neural clinical index generation.
  12  Clinical-neural master merge.
  13  Group-wise neural-clinical Pearson correlations.
  14  Partial neural-clinical correlations.
  15  Outlier removal and z-scoring.
  16  Z-scored OLS neural-clinical associations.

Environment overrides:
  PROJECT_ROOT, CONTAINER_SIF, APP_PATH, OUT_BASE, OUT_DIR, NPZ_DIR
  LOG_DIR, PARTITION, ACCOUNT, TIME, MEM, CPUS
  N_JOBS, N_JOBS_CV, N_PERMUTATION, N_NULL_PERMS
  STAGE1_ACTUAL_REPEATS, STAGE1_CHUNKS, STAGE1_ARRAY_MAX_RUNNING

Notes:
  The Schaefer Python script supports Stage 1 chunk files and merge mode, but
  Stage 1 also fits Analysis 1.1 models. For most runs, keep STAGE1_CHUNKS=1.
  If you use STAGE1_CHUNKS>1, submit group chunks only after the stage-1 model
  outputs are already available, then submit the merge job before stages 2+.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  print_usage
  exit 1
fi

REQUESTED="$1"
shift
EXTRA_ARGS="$*"
REQUESTED="${REQUESTED^^}"

mkdir -p "$LOG_DIR"
mkdir -p "$OUT_BASE"
module load apptainer 2>/dev/null || true

base_wrap_prefix() {
  printf 'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 N_PERMUTATION=%q N_NULL_PERMS=%q PHASE2_NPZ=%q PHASE3_NPZ=%q; ' \
    "$N_PERMUTATION" \
    "$N_NULL_PERMS" \
    "${NPZ_DIR}/phase2_X_ext_y_ext_voxels_schaefer_tian.npz" \
    "${NPZ_DIR}/phase3_X_reinst_y_reinst_voxels_schaefer_tian.npz"
}

stage_output_suffix() {
  case "$1" in
    1)  printf 'a11_models_importance' ;;
    2)  printf 'a12_topology' ;;
    3)  printf 'a13_drift_trajectories' ;;
    4)  printf 'a14_decision_stats' ;;
    5)  printf 'a21_safety_restoration' ;;
    6)  printf 'a22_drift_efficiency' ;;
    7)  printf 'a23_probabilistic_opening' ;;
    8)  printf 'a24_spatial_realignment' ;;
    9)  printf 'a25_reverse_cross_decoding' ;;
    10) printf 'c23_clinical_scores' ;;
    11) printf 'c24_neural_clinical_indices' ;;
    12) printf 'c26_master_clinical_neural' ;;
    13) printf 'c27_neural_clinical_pearson' ;;
    14) printf 'c28_neural_clinical_partial' ;;
    15) printf 'c29_neural_clinical_zscore' ;;
    16) printf 'c30_neural_clinical_ols' ;;
    *)  printf 's%s' "$1" ;;
  esac
}

python_cmd() {
  local stage="$1"
  local group="${2:-ALL}"
  local extra_stage_args="${3:-}"
  printf 'apptainer exec -B %q:%q -B %q:/app -B %q:/output_dir %q python3 /app/mvpa_L2_voxel_WholeBrain_Schaefer.py --project_root %q --output_dir %q --n_jobs %q --n_jobs_cv %q --n_permutation %q --n_null_perms %q --stage %q --stage1_group %q --stage1_actual_repeats %q --stage1_chunk_count %q %s %s' \
    "$PROJECT_ROOT" "$PROJECT_ROOT" \
    "$APP_PATH" \
    "$OUT_BASE" \
    "$CONTAINER_SIF" \
    "$PROJECT_ROOT" \
    "$OUT_DIR" \
    "$N_JOBS" \
    "$N_JOBS_CV" \
    "$N_PERMUTATION" \
    "$N_NULL_PERMS" \
    "$stage" \
    "$group" \
    "$STAGE1_ACTUAL_REPEATS" \
    "$STAGE1_CHUNKS" \
    "$extra_stage_args" \
    "$EXTRA_ARGS"
}

submit_stage() {
  local stage_spec="$1"
  local dependency="${2:-}"
  local stage="$stage_spec"
  local group="ALL"
  local job_suffix="$stage_spec"
  local dependency_args=()

  if [[ "$stage_spec" =~ ^1[:_-](SAD|HC|ALL)$ ]]; then
    stage="1"
    group="${BASH_REMATCH[1]}"
  fi
  job_suffix="$(stage_output_suffix "$stage")"
  if [[ "$stage" == "1" && "$group" != "ALL" ]]; then
    job_suffix="${job_suffix}_${group}"
  fi

  if [[ -n "$dependency" ]]; then
    dependency_args=(--dependency="afterok:${dependency}")
  fi

  local wrap
  wrap="$(base_wrap_prefix)$(python_cmd "$stage" "$group")"

  sbatch --parsable \
    "${dependency_args[@]}" \
    --partition="$PARTITION" \
    --account="$ACCOUNT" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$TIME" \
    --job-name="mvpa_schaefer_${job_suffix}" \
    --output="$LOG_DIR/mvpa_schaefer_${job_suffix}_%j.out" \
    --error="$LOG_DIR/mvpa_schaefer_${job_suffix}_%j.err" \
    --wrap="$wrap"
}

submit_stage1_array() {
  local group_name="$1"
  local dependency="${2:-}"
  local dependency_args=()
  if [[ -n "$dependency" ]]; then
    dependency_args=(--dependency="afterok:${dependency}")
  fi

  local max_task=$((STAGE1_CHUNKS - 1))
  local wrap
  wrap="$(base_wrap_prefix)$(python_cmd 1 "$group_name" '--stage1_chunk_idx $SLURM_ARRAY_TASK_ID')"

  sbatch --parsable \
    "${dependency_args[@]}" \
    --array="0-${max_task}%${STAGE1_ARRAY_MAX_RUNNING}" \
    --partition="$PARTITION" \
    --account="$ACCOUNT" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$TIME" \
    --job-name="mvpa_schaefer_a11_importance_${group_name}" \
    --output="$LOG_DIR/mvpa_schaefer_a11_importance_${group_name}_%A_%a.out" \
    --error="$LOG_DIR/mvpa_schaefer_a11_importance_${group_name}_%A_%a.err" \
    --wrap="$wrap"
}

submit_stage1_merge() {
  local group_name="$1"
  local dependency="$2"
  local wrap
  wrap="$(base_wrap_prefix)$(python_cmd 1 "$group_name" '--stage1_merge')"

  sbatch --parsable \
    --dependency="afterok:${dependency}" \
    --partition="$PARTITION" \
    --account="$ACCOUNT" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$TIME" \
    --job-name="mvpa_schaefer_a11_importance_${group_name}_merge" \
    --output="$LOG_DIR/mvpa_schaefer_a11_importance_${group_name}_merge_%j.out" \
    --error="$LOG_DIR/mvpa_schaefer_a11_importance_${group_name}_merge_%j.err" \
    --wrap="$wrap"
}

if [[ "$REQUESTED" == "ALL" ]]; then
  if [[ "$STAGE1_CHUNKS" -ne 1 ]]; then
    echo "ERROR: Schaefer 'all' currently expects STAGE1_CHUNKS=1 because stage 1 also fits Analysis 1.1 models." >&2
    echo "Set STAGE1_CHUNKS=1 for the full chain, or submit stage/group chunks manually after confirming model outputs exist." >&2
    exit 1
  fi
  prev_job=""
  for stage in "${ALL_STAGES[@]}"; do
    job_id="$(submit_stage "$stage" "$prev_job")"
    echo "Submitted Schaefer stage ${stage}: job ${job_id}"
    prev_job="$job_id"
  done
  echo "Submitted chained Schaefer stages: ${ALL_STAGES[*]}"
elif [[ "$REQUESTED" =~ ^1[:_-](SAD|HC)$ ]]; then
  group_name="${BASH_REMATCH[1]}"
  if [[ "$STAGE1_CHUNKS" -le 1 ]]; then
    job_id="$(submit_stage "1:${group_name}")"
    echo "Submitted Schaefer stage 1:${group_name}: job ${job_id}"
  else
    array_job_id="$(submit_stage1_array "$group_name")"
    echo "Submitted Schaefer stage 1:${group_name} array: job ${array_job_id}"
    merge_job_id="$(submit_stage1_merge "$group_name" "$array_job_id")"
    echo "Submitted Schaefer stage 1:${group_name} merge: job ${merge_job_id}"
  fi
else
  job_id="$(submit_stage "$REQUESTED")"
  echo "Submitted Schaefer stage ${REQUESTED}: job ${job_id}"
fi
