#!/usr/bin/env bash
set -euo pipefail

# Submit mvpa_L2_voxel_MemoryFearNetwork.py by notebook code-cell stage.
# Usage:
#   ./submit_mvpa_L2_memoryfearnetwork_stage.sh all
#   ./submit_mvpa_L2_memoryfearnetwork_stage.sh 11:SAD
#   ./submit_mvpa_L2_memoryfearnetwork_stage.sh 12
#   ./submit_mvpa_L2_memoryfearnetwork_stage.sh 12 --n_permutation 1000
#   ./submit_mvpa_L2_memoryfearnetwork_stage.sh --help

# ---- User-configurable settings ----
PROJECT_ROOT="${PROJECT_ROOT:-/gscratch/fang/NARSAD}"
CONTAINER_SIF="${CONTAINER_SIF:-/gscratch/fang/images/jupyter.sif}"
APP_PATH="${APP_PATH:-/gscratch/scrubbed/fanglab/xiaoqian/repo/narsad_multivoxel/hyak}"
OUT_BASE="${OUT_BASE:-/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/LSS/results}"
OUT_DIR="${OUT_DIR:-/output_dir/MemoryFearNetwork}"
ROI_DIR="${MEMORY_FEAR_ROI_DIR:-${PROJECT_ROOT}/tool/parcellation/ROIs/MemoryFearNetwork}"

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/mvpa_l2_memoryfearnetwork}"
PARTITION="${PARTITION:-cpu-g2}"
ACCOUNT="${ACCOUNT:-fang}"
TIME="${TIME:-96:00:00}"
MEM="${MEM:-100G}"
CPUS="${CPUS:-16}"

# Parallel settings. Keep CV serial by default to avoid nested oversubscription.
N_JOBS="${N_JOBS:-16}"
N_JOBS_CV="${N_JOBS_CV:-1}"
N_PERMUTATION="${N_PERMUTATION:-5000}"
N_NULL_PERMS="${N_NULL_PERMS:-5000}"
STAGE11_ACTUAL_REPEATS="${STAGE11_ACTUAL_REPEATS:-${N_NULL_PERMS}}"
STAGE11_CHUNKS="${STAGE11_CHUNKS:-100}"
STAGE11_ARRAY_MAX_RUNNING="${STAGE11_ARRAY_MAX_RUNNING:-20}"

# Analysis-bearing notebook cells in execution order.
PRE_STAGE11_STAGES=(6 7 10)
POST_STAGE11_STAGES=(12 13 14 15 16 17 18 19 20 21 23 24 26 27 28 29 30)
STAGE11_GROUPS=(SAD HC)

print_usage() {
  cat <<'EOF'
Usage:
  submit_mvpa_L2_memoryfearnetwork_stage.sh all
      Submit every analysis stage below with the dependency graph shown under
      "Analysis structure".

  submit_mvpa_L2_memoryfearnetwork_stage.sh <stage_cell> [extra python args]
      Submit one notebook code-cell stage. Extra args are passed through to
      mvpa_L2_voxel_MemoryFearNetwork.py.

  submit_mvpa_L2_memoryfearnetwork_stage.sh 11:SAD
  submit_mvpa_L2_memoryfearnetwork_stage.sh 11:HC
      Submit only one half of the slow Stage 11 permutation-importance job.
      This submits a Slurm array of STAGE11_CHUNKS chunk jobs plus one merge
      job for that group. Stage 12+ automatically merges/loads the final
      cell_11_SAD.joblib and cell_11_HC.joblib outputs.

Examples:
  submit_mvpa_L2_memoryfearnetwork_stage.sh 6
  submit_mvpa_L2_memoryfearnetwork_stage.sh 11:SAD
  submit_mvpa_L2_memoryfearnetwork_stage.sh 11:HC
  submit_mvpa_L2_memoryfearnetwork_stage.sh 12 --n_permutation 1000
  N_PERMUTATION=1000 CPUS=8 N_JOBS=8 submit_mvpa_L2_memoryfearnetwork_stage.sh all
  STAGE11_CHUNKS=200 submit_mvpa_L2_memoryfearnetwork_stage.sh 11:SAD

Analysis structure:
  Main prerequisite chain:
    6 -> 7 -> 10 -> 11:SAD + 11:HC -> 12 -> 13 -> 14 -> 15 -> 16 -> 17 -> 18 -> 19 -> 20 -> 21 -> 23 -> 24 -> 26 -> 27 -> 28 -> 29 -> 30

  Parallel part:
    Stage 11 is split into two independent Slurm arrays after Stage 10 finishes:
      11:SAD array computes SAD permutation-importance chunks.
      11:HC  array computes HC permutation-importance chunks.
    Each array is followed by one merge job. Stage 12 waits for both merge jobs
    and then loads/combines the final split outputs automatically.

  Sequential parts:
    Stages 6, 7, and 10 must run in order because they prepare models,
    cross-phase nulls, and voxel-wise spatial/Haufe inputs.
    Stages 12-30 are submitted sequentially by "all" because later cells
    consume saved outputs from earlier analysis and clinical-merge cells.

  Manual/resume usage:
    You may run one stage manually after its prerequisites have completed.
    For example, after both 11:SAD and 11:HC finish, stage 12 can be submitted
    by itself and will load/merge the split Stage 11 checkpoints.

Stage / cell reference:
   6  Analysis 1.1: Neural dissociation, self-decoding, cross-decoding, spatial specificity.
   7  Corrected cross-phase permutation nulls for reinstatement generalization.
  10  Voxel-wise spatial/Haufe analysis in the MemoryFearNetwork ROI set.
  11  Empirical permutation-importance masks. Slow stage; can split as 11:SAD and 11:HC.
  12  Analysis 1.2: Static representational topology with crossnobis RDMs.
  13  Analysis 1.3: Dynamic representational drift.
  14  Analysis 1.3 part 2: Single-trial safety/threat trajectories.
  15  Analysis 1.4: Decision boundary/self-network uncertainty statistics.
  16  Analysis 2.1: Safety restoration and threat discrimination, per-voxel normalized.
  17  Analysis 2.1: Safety restoration and threat discrimination, raw mixed-effects metrics.
  18  Analysis 2.2: Drift efficiency for safety and threat maintenance.
  19  Analysis 2.3: Probabilistic opening / decision-probability extraction.
  20  Analysis 2.4: Spatial re-alignment, applying the HC model to SAD PLC/OXT.
  21  Analysis 2.5: Reverse cross-decoding, applying the SAD model to HC PLC/OXT.
  23  Load and score clinical LSAS, ECR, and DASS data.
  24  Build neural topology, trajectory, and uncertainty indices for clinical merge.
  26  Merge clinical and neural indices into the master analysis dataframe.
  27  Group-wise neural-clinical Pearson correlations.
  28  Group-wise partial correlations adjusted for covariates.
  29  Outlier removal and z-scoring for neural, clinical, and covariate variables.
  30  Z-scored OLS/regression plots for neural-clinical associations.

Environment overrides:
  PROJECT_ROOT, CONTAINER_SIF, APP_PATH, OUT_BASE, OUT_DIR, MEMORY_FEAR_ROI_DIR
  LOG_DIR, PARTITION, ACCOUNT, TIME, MEM, CPUS
  N_JOBS, N_JOBS_CV, N_PERMUTATION, N_NULL_PERMS
  STAGE11_ACTUAL_REPEATS, STAGE11_CHUNKS, STAGE11_ARRAY_MAX_RUNNING
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

submit_stage() {
  local stage_spec="$1"
  local dependency="${2:-}"
  local stage="$stage_spec"
  local stage11_group="ALL"
  local job_suffix="$stage_spec"
  local dependency_args=()

  if [[ "$stage_spec" =~ ^11[:_-](SAD|HC|ALL)$ ]]; then
    stage="11"
    stage11_group="${BASH_REMATCH[1]}"
    job_suffix="11_${stage11_group}"
  fi

  if [[ -n "$dependency" ]]; then
    dependency_args=(--dependency="afterok:${dependency}")
  fi

  local job_id
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
      --job-name="mvpa_memfear_c${job_suffix}" \
      --output="$LOG_DIR/mvpa_memfear_c${job_suffix}_%j.out" \
      --error="$LOG_DIR/mvpa_memfear_c${job_suffix}_%j.err" \
      --wrap="export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 N_PERMUTATION=${N_PERMUTATION} N_NULL_PERMS=${N_NULL_PERMS}; apptainer exec -B ${PROJECT_ROOT}:${PROJECT_ROOT} -B ${APP_PATH}:/app -B ${OUT_BASE}:/output_dir ${CONTAINER_SIF} python3 /app/mvpa_L2_voxel_MemoryFearNetwork.py --project_root ${PROJECT_ROOT} --output_dir ${OUT_DIR} --roi_dir ${ROI_DIR} --n_jobs ${N_JOBS} --n_jobs_cv ${N_JOBS_CV} --n_permutation ${N_PERMUTATION} --n_null_perms ${N_NULL_PERMS} --stage ${stage} --stage11_group ${stage11_group} ${EXTRA_ARGS}"
  )
  echo "$job_id"
}

submit_stage11_array() {
  local group_name="$1"
  local dependency="${2:-}"
  local dependency_args=()
  if [[ -n "$dependency" ]]; then
    dependency_args=(--dependency="afterok:${dependency}")
  fi

  local max_task=$((STAGE11_CHUNKS - 1))
  local job_id
  job_id=$(
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
      --job-name="mvpa_memfear_c11_${group_name}" \
      --output="$LOG_DIR/mvpa_memfear_c11_${group_name}_%A_%a.out" \
      --error="$LOG_DIR/mvpa_memfear_c11_${group_name}_%A_%a.err" \
      --wrap="export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 N_PERMUTATION=${N_PERMUTATION} N_NULL_PERMS=${N_NULL_PERMS} STAGE11_ACTUAL_REPEATS=${STAGE11_ACTUAL_REPEATS} STAGE11_CHUNK_COUNT=${STAGE11_CHUNKS}; apptainer exec -B ${PROJECT_ROOT}:${PROJECT_ROOT} -B ${APP_PATH}:/app -B ${OUT_BASE}:/output_dir ${CONTAINER_SIF} python3 /app/mvpa_L2_voxel_MemoryFearNetwork.py --project_root ${PROJECT_ROOT} --output_dir ${OUT_DIR} --roi_dir ${ROI_DIR} --n_jobs ${N_JOBS} --n_jobs_cv ${N_JOBS_CV} --n_permutation ${N_PERMUTATION} --n_null_perms ${N_NULL_PERMS} --stage11_actual_repeats ${STAGE11_ACTUAL_REPEATS} --stage11_chunk_count ${STAGE11_CHUNKS} --stage11_chunk_idx \$SLURM_ARRAY_TASK_ID --stage 11 --stage11_group ${group_name} ${EXTRA_ARGS}"
  )
  echo "$job_id"
}

submit_stage11_merge() {
  local group_name="$1"
  local dependency="$2"
  local job_id
  job_id=$(
    sbatch --parsable \
      --dependency="afterok:${dependency}" \
      --partition="$PARTITION" \
      --account="$ACCOUNT" \
      --nodes=1 \
      --ntasks=1 \
      --cpus-per-task="$CPUS" \
      --mem="$MEM" \
      --time="$TIME" \
      --job-name="mvpa_memfear_c11_${group_name}_merge" \
      --output="$LOG_DIR/mvpa_memfear_c11_${group_name}_merge_%j.out" \
      --error="$LOG_DIR/mvpa_memfear_c11_${group_name}_merge_%j.err" \
      --wrap="export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 N_PERMUTATION=${N_PERMUTATION} N_NULL_PERMS=${N_NULL_PERMS} STAGE11_ACTUAL_REPEATS=${STAGE11_ACTUAL_REPEATS} STAGE11_CHUNK_COUNT=${STAGE11_CHUNKS}; apptainer exec -B ${PROJECT_ROOT}:${PROJECT_ROOT} -B ${APP_PATH}:/app -B ${OUT_BASE}:/output_dir ${CONTAINER_SIF} python3 /app/mvpa_L2_voxel_MemoryFearNetwork.py --project_root ${PROJECT_ROOT} --output_dir ${OUT_DIR} --roi_dir ${ROI_DIR} --n_jobs ${N_JOBS} --n_jobs_cv ${N_JOBS_CV} --n_permutation ${N_PERMUTATION} --n_null_perms ${N_NULL_PERMS} --stage11_actual_repeats ${STAGE11_ACTUAL_REPEATS} --stage11_chunk_count ${STAGE11_CHUNKS} --stage11_merge --stage 11 --stage11_group ${group_name} ${EXTRA_ARGS}"
  )
  echo "$job_id"
}

if [[ "$REQUESTED" == "ALL" ]]; then
  prev_job=""
  for stage in "${PRE_STAGE11_STAGES[@]}"; do
    job_id="$(submit_stage "$stage" "$prev_job")"
    echo "Submitted MemoryFearNetwork cell/stage ${stage}: job ${job_id}"
    prev_job="$job_id"
  done

  stage11_jobs=()
  stage11_merge_jobs=()
  for group_name in "${STAGE11_GROUPS[@]}"; do
    job_id="$(submit_stage11_array "$group_name" "$prev_job")"
    echo "Submitted MemoryFearNetwork cell/stage 11:${group_name} array: job ${job_id}"
    stage11_jobs+=("$job_id")
    merge_job_id="$(submit_stage11_merge "$group_name" "$job_id")"
    echo "Submitted MemoryFearNetwork cell/stage 11:${group_name} merge: job ${merge_job_id}"
    stage11_merge_jobs+=("$merge_job_id")
  done
  prev_job="$(IFS=:; echo "${stage11_merge_jobs[*]}")"

  for stage in "${POST_STAGE11_STAGES[@]}"; do
    job_id="$(submit_stage "$stage" "$prev_job")"
    echo "Submitted MemoryFearNetwork cell/stage ${stage}: job ${job_id}"
    prev_job="$job_id"
  done

  echo "Submitted chained MemoryFearNetwork stages with split stage 11 SAD/HC jobs."
else
  if [[ "$REQUESTED" =~ ^11[:_-](SAD|HC)$ ]]; then
    group_name="${BASH_REMATCH[1]}"
    array_job_id="$(submit_stage11_array "$group_name")"
    echo "Submitted MemoryFearNetwork cell/stage 11:${group_name} array: job ${array_job_id}"
    merge_job_id="$(submit_stage11_merge "$group_name" "$array_job_id")"
    echo "Submitted MemoryFearNetwork cell/stage 11:${group_name} merge: job ${merge_job_id}"
  else
    job_id="$(submit_stage "$REQUESTED")"
    echo "Submitted MemoryFearNetwork cell/stage ${REQUESTED}: job ${job_id}"
  fi
fi
