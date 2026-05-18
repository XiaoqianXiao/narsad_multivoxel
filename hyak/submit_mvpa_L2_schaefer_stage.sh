#!/usr/bin/env bash
set -euo pipefail

# Submit mvpa_L2_voxel_WholeBrain_Schaefer.py by logical stage.
# Usage:
#   ./submit_mvpa_L2_schaefer_stage.sh all
#   ./submit_mvpa_L2_schaefer_stage.sh 11:SAD
#   ./submit_mvpa_L2_schaefer_stage.sh 12 --resume
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
STAGE11_ACTUAL_REPEATS="${STAGE11_ACTUAL_REPEATS:-${N_NULL_PERMS}}"
STAGE11_CHUNKS="${STAGE11_CHUNKS:-100}"
STAGE11_ARRAY_MAX_RUNNING="${STAGE11_ARRAY_MAX_RUNNING:-20}"
STAGE11_CHUNK_IDX="${STAGE11_CHUNK_IDX:-}"
STAGE11_GROUPS=(SAD HC)

# User-facing stages mirror MemoryFearNetwork/FearNetwork.
PRE_STAGE11_STAGES=(6)
POST_STAGE11_STAGES=(12 13 15 16 18 19 20 21 23 24 26 27 28 29 30)

print_usage() {
  cat <<'EOF'
Usage:
  submit_mvpa_L2_schaefer_stage.sh all
      Submit Schaefer stages as a dependency chain using the same user-facing
      stage numbers as MemoryFearNetwork/FearNetwork.

  submit_mvpa_L2_schaefer_stage.sh <stage_id> [extra python args]
      Submit one user-facing stage. Extra args are passed through to
      mvpa_L2_voxel_WholeBrain_Schaefer.py.

  submit_mvpa_L2_schaefer_stage.sh 11:SAD
  submit_mvpa_L2_schaefer_stage.sh 11:HC
      Submit only one half of the slow Stage 11 permutation-importance job.
      This submits a Slurm array of STAGE11_CHUNKS chunk jobs plus one merge
      job for that group. Stage 12+ loads the final split outputs automatically.

      If STAGE11_CHUNK_IDX is set, submit only that single chunk. This is for
      recovering one failed array task without recomputing the full array.

Examples:
  submit_mvpa_L2_schaefer_stage.sh all
  submit_mvpa_L2_schaefer_stage.sh 12 --resume
  STAGE11_CHUNKS=100 submit_mvpa_L2_schaefer_stage.sh 11:SAD
  STAGE11_CHUNKS=100 STAGE11_CHUNK_IDX=80 submit_mvpa_L2_schaefer_stage.sh 11:SAD

Analysis structure:
  Main prerequisite chain:
    6 -> 11:SAD + 11:HC -> 12 -> 13 -> 15 -> 16 -> 18 -> 19 -> 20 -> 21 -> 23 -> 24 -> 26 -> 27 -> 28 -> 29 -> 30

  Parallel part:
    Stage 11 is split into two independent Slurm arrays after Stage 6 finishes:
      11:SAD array computes SAD permutation-importance chunks.
      11:HC  array computes HC permutation-importance chunks.
    Each array is followed by one merge job. Stage 12 waits for both merge jobs.

Stage reference:
   6  Analysis 1.1 models/permutations and spatial/Haufe model maps.
  11  Empirical permutation-importance masks.
  12  Analysis 1.2 static representational topology.
  13  Analysis 1.3 dynamic drift and single-trial trajectories.
  15  Analysis 1.4 decision-boundary/self-network statistics.
  16  Analysis 2.1 safety restoration/threat discrimination.
  18  Analysis 2.2 drift efficiency.
  19  Analysis 2.3 probabilistic opening.
  20  Analysis 2.4 spatial re-alignment.
  21  Analysis 2.5 reverse cross-decoding.
  23  Clinical score loading.
  24  Neural clinical index generation.
  26  Clinical-neural master merge.
  27  Group-wise neural-clinical Pearson correlations.
  28  Partial neural-clinical correlations.
  29  Outlier removal and z-scoring.
  30  Z-scored OLS neural-clinical associations.

Environment overrides:
  PROJECT_ROOT, CONTAINER_SIF, APP_PATH, OUT_BASE, OUT_DIR, NPZ_DIR
  LOG_DIR, PARTITION, ACCOUNT, TIME, MEM, CPUS
  N_JOBS, N_JOBS_CV, N_PERMUTATION, N_NULL_PERMS
  STAGE11_ACTUAL_REPEATS, STAGE11_CHUNKS, STAGE11_ARRAY_MAX_RUNNING
  STAGE11_CHUNK_IDX for single-chunk recovery

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
    6)  printf 'a11_models' ;;
    11) printf 'a11_importance' ;;
    12) printf 'a12_topology' ;;
    13|14) printf 'a13_drift_trajectories' ;;
    15) printf 'a14_decision_stats' ;;
    16) printf 'a21_safety_restoration' ;;
    18) printf 'a22_drift_efficiency' ;;
    19) printf 'a23_probabilistic_opening' ;;
    20) printf 'a24_spatial_realignment' ;;
    21) printf 'a25_reverse_cross_decoding' ;;
    23) printf 'c23_clinical_scores' ;;
    24) printf 'c24_neural_clinical_indices' ;;
    26) printf 'c26_master_clinical_neural' ;;
    27) printf 'c27_neural_clinical_pearson' ;;
    28) printf 'c28_neural_clinical_partial' ;;
    29) printf 'c29_neural_clinical_zscore' ;;
    30) printf 'c30_neural_clinical_ols' ;;
    *)  printf 's%s' "$1" ;;
  esac
}

python_cmd() {
  local stage="$1"
  local group="${2:-ALL}"
  local extra_stage_args="${3:-}"
  printf 'apptainer exec -B %q:%q -B %q:/app -B %q:/output_dir %q python3 /app/mvpa_L2_voxel_WholeBrain_Schaefer.py --project_root %q --output_dir %q --n_jobs %q --n_jobs_cv %q --n_permutation %q --n_null_perms %q --stage %q --stage11_group %q --stage11_actual_repeats %q --stage11_chunk_count %q %s %s' \
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
    "$STAGE11_ACTUAL_REPEATS" \
    "$STAGE11_CHUNKS" \
    "$extra_stage_args" \
    "$EXTRA_ARGS"
}

submit_stage() {
  local stage_spec="$1"
  local dependency="${2:-}"
  local stage="$stage_spec"
  local requested_stage="$stage_spec"
  local group="ALL"
  local job_suffix="$stage_spec"
  local dependency_args=()

  if [[ "$stage_spec" =~ ^11[:_-](SAD|HC|ALL)$ ]]; then
    stage="11"
    requested_stage="11"
    group="${BASH_REMATCH[1]}"
  fi
  job_suffix="$(stage_output_suffix "$stage")"
  if [[ "$stage" == "11" && "$group" != "ALL" ]]; then
    job_suffix="${job_suffix}_${group}"
  fi

  if [[ -n "$dependency" ]]; then
    dependency_args=(--dependency="afterok:${dependency}")
  fi

  local stage_extra=""
  if [[ "$stage" != "6" ]]; then
    stage_extra="--resume"
  fi
  local wrap
  wrap="$(base_wrap_prefix)$(python_cmd "$stage" "$group" "$stage_extra")"

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

submit_stage11_array() {
  local group_name="$1"
  local dependency="${2:-}"
  local dependency_args=()
  if [[ -n "$dependency" ]]; then
    dependency_args=(--dependency="afterok:${dependency}")
  fi

  local max_task=$((STAGE11_CHUNKS - 1))
  local wrap
  wrap="$(base_wrap_prefix)$(python_cmd 11 "$group_name" '--resume --stage11_chunk_idx $SLURM_ARRAY_TASK_ID')"

  sbatch --parsable \
    "${dependency_args[@]}" \
    --array="0-${max_task}%${STAGE11_ARRAY_MAX_RUNNING}" \
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

submit_stage11_chunk() {
  local group_name="$1"
  local chunk_idx="$2"
  local dependency="${3:-}"
  local dependency_args=()
  if [[ -n "$dependency" ]]; then
    dependency_args=(--dependency="afterok:${dependency}")
  fi

  local wrap
  wrap="$(base_wrap_prefix)$(python_cmd 11 "$group_name" "--resume --stage11_chunk_idx ${chunk_idx}")"

  sbatch --parsable \
    "${dependency_args[@]}" \
    --partition="$PARTITION" \
    --account="$ACCOUNT" \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$TIME" \
    --job-name="mvpa_schaefer_a11_importance_${group_name}_chunk${chunk_idx}" \
    --output="$LOG_DIR/mvpa_schaefer_a11_importance_${group_name}_chunk${chunk_idx}_%j.out" \
    --error="$LOG_DIR/mvpa_schaefer_a11_importance_${group_name}_chunk${chunk_idx}_%j.err" \
    --wrap="$wrap"
}

submit_stage11_merge() {
  local group_name="$1"
  local dependency="${2:-}"
  local dependency_args=()
  if [[ -n "$dependency" ]]; then
    dependency_args=(--dependency="afterok:${dependency}")
  fi
  local wrap
  wrap="$(base_wrap_prefix)$(python_cmd 11 "$group_name" '--resume --stage11_merge')"

  sbatch --parsable \
    "${dependency_args[@]}" \
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
  prev_job=""
  for stage in "${PRE_STAGE11_STAGES[@]}"; do
    job_id="$(submit_stage "$stage" "$prev_job")"
    echo "Submitted Schaefer stage ${stage}: job ${job_id}"
    prev_job="$job_id"
  done

  stage11_merge_jobs=()
  for group_name in "${STAGE11_GROUPS[@]}"; do
    if [[ "$STAGE11_CHUNKS" -le 1 ]]; then
      job_id="$(submit_stage "11:${group_name}" "$prev_job")"
      echo "Submitted Schaefer stage 11:${group_name}: job ${job_id}"
      stage11_merge_jobs+=("$job_id")
    else
      array_job_id="$(submit_stage11_array "$group_name" "$prev_job")"
      echo "Submitted Schaefer stage 11:${group_name} array: job ${array_job_id}"
      merge_job_id="$(submit_stage11_merge "$group_name" "$array_job_id")"
      echo "Submitted Schaefer stage 11:${group_name} merge: job ${merge_job_id}"
      stage11_merge_jobs+=("$merge_job_id")
    fi
  done
  prev_job="$(IFS=:; echo "${stage11_merge_jobs[*]}")"

  for stage in "${POST_STAGE11_STAGES[@]}"; do
    job_id="$(submit_stage "$stage" "$prev_job")"
    echo "Submitted Schaefer stage ${stage}: job ${job_id}"
    prev_job="$job_id"
  done

  echo "Submitted chained Schaefer stages with split stage 11 SAD/HC jobs."
elif [[ "$REQUESTED" =~ ^11[:_-](SAD|HC)$ ]]; then
  requested_stage="11"
  group_name="${BASH_REMATCH[1]}"
  if [[ -n "$STAGE11_CHUNK_IDX" ]]; then
    if (( STAGE11_CHUNK_IDX < 0 || STAGE11_CHUNK_IDX >= STAGE11_CHUNKS )); then
      echo "ERROR: STAGE11_CHUNK_IDX=${STAGE11_CHUNK_IDX} must be between 0 and $((STAGE11_CHUNKS - 1))." >&2
      exit 1
    fi
    chunk_job_id="$(submit_stage11_chunk "$group_name" "$STAGE11_CHUNK_IDX")"
    echo "Submitted Schaefer stage ${requested_stage}:${group_name} chunk ${STAGE11_CHUNK_IDX}/${STAGE11_CHUNKS}: job ${chunk_job_id}"
    echo "After this chunk completes, run: STAGE11_CHUNKS=${STAGE11_CHUNKS} hyak/submit_mvpa_L2_schaefer_stage.sh 11:${group_name}:merge"
  elif [[ "$STAGE11_CHUNKS" -le 1 ]]; then
    job_id="$(submit_stage "${requested_stage}:${group_name}")"
    echo "Submitted Schaefer stage ${requested_stage}:${group_name}: job ${job_id}"
  else
    array_job_id="$(submit_stage11_array "$group_name")"
    echo "Submitted Schaefer stage ${requested_stage}:${group_name} array: job ${array_job_id}"
    merge_job_id="$(submit_stage11_merge "$group_name" "$array_job_id")"
    echo "Submitted Schaefer stage ${requested_stage}:${group_name} merge: job ${merge_job_id}"
  fi
elif [[ "$REQUESTED" =~ ^11[:_-](SAD|HC)[:_-]MERGE$ ]]; then
  requested_stage="11"
  group_name="${BASH_REMATCH[1]}"
  merge_job_id="$(submit_stage11_merge "$group_name" "")"
  echo "Submitted Schaefer stage ${requested_stage}:${group_name} merge: job ${merge_job_id}"
else
  if [[ "$REQUESTED" == "11" && "$STAGE11_CHUNKS" -gt 1 ]]; then
    echo "ERROR: Stage 11 is split by group when STAGE11_CHUNKS > 1." >&2
    echo "Submit 11:SAD and 11:HC, or set STAGE11_CHUNKS=1 for a single combined Stage 11 run." >&2
    exit 1
  fi
  job_id="$(submit_stage "$REQUESTED")"
  echo "Submitted Schaefer stage ${REQUESTED}: job ${job_id}"
fi
