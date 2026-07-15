#!/usr/bin/env bash
set -euo pipefail

# Fast post-Hyak MVPA L2 workflow.
#
# Run this after the feature-space Hyak jobs have finished. Override the
# feature-space directories if your Hyak outputs live somewhere else:
#
#   FEAR_DIR=/path/FearNetwork \
#   MEMORY_DIR=/path/MemoryFearNetwork \
#   bash scripts/run_mvpa_l2_posthyak.sh
#
# Optional whole-brain/parcellation sensitivity can be added later:
#
#   SCHAEFER_DIR=/path/wholebrain_parcellation_schaefer \
#   bash scripts/run_mvpa_l2_posthyak.sh

PROJECT_ROOT="${PROJECT_ROOT:-/gscratch/fang/NARSAD}"
CONTAINER_SIF="${CONTAINER_SIF:-/gscratch/fang/images/jupyter.sif}"
OUT_BASE="${OUT_BASE:-/gscratch/fang/NARSAD/MRI/derivatives/fMRI_analysis/LSS/results}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

if [[ -d "$OUT_BASE" || "$REPO_ROOT" == /gscratch/* ]]; then
  FEAR_DIR="${FEAR_DIR:-$OUT_BASE/FearNetwork}"
  MEMORY_DIR="${MEMORY_DIR:-$OUT_BASE/MemoryFearNetwork}"
  SCHAEFER_DIR="${SCHAEFER_DIR:-$OUT_BASE/WholeBrain_Schaefer}"
  OUT_ROOT="${OUT_ROOT:-$OUT_BASE/mvpa_l2}"
else
  FEAR_DIR="${FEAR_DIR:-outputs/mvpa_l2/FearNetwork}"
  MEMORY_DIR="${MEMORY_DIR:-outputs/mvpa_l2/MemoryFearNetwork}"
  SCHAEFER_DIR="${SCHAEFER_DIR:-}"
  OUT_ROOT="${OUT_ROOT:-outputs/mvpa_l2}"
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
SCR_FLAGS="${SCR_FLAGS:-}"
SCR_FLAGS_OUT="$OUT_ROOT/harmonized/scr_sensitivity_groups.csv"
CLINICAL_OUTLIER_Z="${CLINICAL_OUTLIER_Z:-3.0}"
RUN_AIM1_SCR="${RUN_AIM1_SCR:-1}"

pick_default_scr_dir() {
  local candidates=(
    "$OUT_BASE/scr_analysis_outputs"
    "/output_dir/scr_analysis_outputs"
    "/app/scr_analysis_outputs"
    "/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/scr_analysis_outputs"
    "/gscratch/scrubbed/fanglab/xiaoqian/repo/narsad_multivoxel/results/scr_analysis_outputs"
    "scr_analysis_outputs"
  )
  local path
  for path in "${candidates[@]}"; do
    if [[ -d "$path" ]]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
  printf '%s\n' "$OUT_BASE/scr_analysis_outputs"
}

SCR_DIR="${SCR_DIR:-$(pick_default_scr_dir)}"

pick_existing_scr_flags() {
  local candidates=(
    "$SCR_FLAGS_OUT"
    "/output_dir/mvpa_l2/harmonized/scr_sensitivity_groups.csv"
    "$OUT_BASE/mvpa_l2/harmonized/scr_sensitivity_groups.csv"
    "outputs/mvpa_l2/harmonized/scr_sensitivity_groups.csv"
  )
  local path
  for path in "${candidates[@]}"; do
    if [[ -s "$path" ]]; then
      printf '%s\n' "$path"
      return 0
    fi
  done
  return 1
}

python_is_modern() {
  "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 7) else 1)
PY
}

running_in_container() {
  [[ -n "${APPTAINER_CONTAINER:-}" || -n "${SINGULARITY_CONTAINER:-}" || "${MVPA_L2_IN_CONTAINER:-}" == "1" ]]
}

on_hyak_filesystem() {
  [[ -d /gscratch || "$REPO_ROOT" == /gscratch/* || "$OUT_BASE" == /gscratch/* ]]
}

if ! python_is_modern && ! running_in_container && on_hyak_filesystem; then
  if ! command -v apptainer >/dev/null 2>&1; then
    module load apptainer 2>/dev/null || true
  fi
  if command -v apptainer >/dev/null 2>&1 && [[ -f "$CONTAINER_SIF" ]]; then
    echo "Python $("$PYTHON_BIN" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo unknown) is too old; relaunching post-Hyak workflow inside ${CONTAINER_SIF}."
    bind_args=(-B "${REPO_ROOT}:/app")
    [[ -d "$PROJECT_ROOT" ]] && bind_args+=(-B "${PROJECT_ROOT}:${PROJECT_ROOT}")
    [[ -d "$OUT_BASE" ]] && bind_args+=(-B "${OUT_BASE}:/output_dir")
    if [[ -n "$SCR_DIR" && "$SCR_DIR" = /* && -d "$SCR_DIR" && "$SCR_DIR" != "$PROJECT_ROOT"* && "$SCR_DIR" != "$OUT_BASE"* ]]; then
      bind_args+=(-B "${SCR_DIR}:${SCR_DIR}")
    fi
    exec env \
      FEAR_DIR="$FEAR_DIR" \
      MEMORY_DIR="$MEMORY_DIR" \
      SCHAEFER_DIR="$SCHAEFER_DIR" \
      SCR_DIR="$SCR_DIR" \
      OUT_ROOT="$OUT_ROOT" \
      SCR_FLAGS="$SCR_FLAGS" \
      CLINICAL_OUTLIER_Z="$CLINICAL_OUTLIER_Z" \
      RUN_AIM1_SCR="$RUN_AIM1_SCR" \
      PROJECT_ROOT="$PROJECT_ROOT" \
      CONTAINER_SIF="$CONTAINER_SIF" \
      OUT_BASE="$OUT_BASE" \
      REPO_ROOT="$REPO_ROOT" \
      apptainer exec "${bind_args[@]}" "$CONTAINER_SIF" bash -lc '
      cd /app
      export MVPA_L2_IN_CONTAINER=1
      export PYTHON_BIN=python3
      bash scripts/run_mvpa_l2_posthyak.sh
    '
  fi
fi

"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 7):
    raise SystemExit(
        "ERROR: MVPA L2 post-Hyak scripts require Python >= 3.7. "
        f"Current interpreter is {sys.executable} ({sys.version.split()[0]}). "
        "Run inside /gscratch/fang/images/jupyter.sif or set PYTHON_BIN to a newer Python."
    )
print(f"Using Python: {sys.executable} ({sys.version.split()[0]})")
PY

validate_scr_flags() {
  local path="$1"
  local n_lines
  if [[ ! -s "$path" ]]; then
    echo "ERROR: SCR sensitivity flags are missing or empty: $path" >&2
    exit 1
  fi
  n_lines="$(wc -l < "$path" | tr -d '[:space:]')"
  if [[ "$n_lines" -lt 2 ]]; then
    echo "ERROR: SCR sensitivity flags contain no subject rows: $path" >&2
    echo "Provide a valid SCR_FLAGS file or set SCR_DIR to the directory containing the SCR subject-list outputs." >&2
    exit 1
  fi
}

mkdir -p "$(dirname "$SCR_FLAGS_OUT")"
if [[ -n "$SCR_FLAGS" && -f "$SCR_FLAGS" ]]; then
  validate_scr_flags "$SCR_FLAGS"
  if [[ "$SCR_FLAGS" != "$SCR_FLAGS_OUT" ]]; then
    cp "$SCR_FLAGS" "$SCR_FLAGS_OUT"
  fi
  echo "Using prebuilt SCR sensitivity flags -> $SCR_FLAGS_OUT"
elif [[ -n "$SCR_FLAGS" ]]; then
  echo "ERROR: SCR_FLAGS was set but the file does not exist: $SCR_FLAGS" >&2
  echo "Expected a prebuilt CSV such as /output_dir/mvpa_l2/harmonized/scr_sensitivity_groups.csv inside the container." >&2
  exit 1
elif existing_scr_flags="$(pick_existing_scr_flags)"; then
  validate_scr_flags "$existing_scr_flags"
  if [[ "$existing_scr_flags" != "$SCR_FLAGS_OUT" ]]; then
    cp "$existing_scr_flags" "$SCR_FLAGS_OUT"
  fi
  echo "Using existing SCR sensitivity flags -> $SCR_FLAGS_OUT"
else
  "$PYTHON_BIN" scripts/build_scr_sensitivity_groups.py \
    --scr-dir "$SCR_DIR" \
    --out "$SCR_FLAGS_OUT"
  validate_scr_flags "$SCR_FLAGS_OUT"
fi

FEATURE_ARGS=(
  --feature-dir "FearNetwork=$FEAR_DIR"
  --feature-dir "MemoryFearNetwork=$MEMORY_DIR"
)

if [[ -n "$SCHAEFER_DIR" ]]; then
  FEATURE_ARGS+=(--feature-dir "Schaefer=$SCHAEFER_DIR")
fi

"$PYTHON_BIN" scripts/export_mvpa_l2_metrics.py \
  "${FEATURE_ARGS[@]}" \
  --scr-flags "$SCR_FLAGS_OUT" \
  --out "$OUT_ROOT/harmonized/mvpa_l2_subject_metrics.csv" \
  --stats-out-dir "$OUT_ROOT/stats"

"$PYTHON_BIN" scripts/export_aim1_decoding_primary.py \
  --feature-dir "$FEAR_DIR" \
  --out "$OUT_ROOT/stats/aim1_decoding_primary.csv" \
  --feature-space "FearNetwork"

FEATURE_AIM1_ARGS=(
  --feature-dir "FearNetwork=$FEAR_DIR"
  --feature-dir "MemoryFearNetwork=$MEMORY_DIR"
)

if [[ -n "$SCHAEFER_DIR" ]]; then
  FEATURE_AIM1_ARGS+=(--feature-dir "Schaefer=$SCHAEFER_DIR")
fi

"$PYTHON_BIN" scripts/export_aim1_feature_sensitivity.py \
  "${FEATURE_AIM1_ARGS[@]}" \
  --out "$OUT_ROOT/stats/aim1_mask_feature_sensitivity.csv" \
  --wide-out "$OUT_ROOT/stats/aim1_mask_feature_sensitivity_wide.csv" \
  --raincloud-out "$OUT_ROOT/stats/aim1_mask_feature_sensitivity_raincloud.csv" \
  --drop-tests-out "$OUT_ROOT/stats/aim1_mask_feature_sensitivity_functional_drop_tests.csv" \
  --drop-nulls-out "$OUT_ROOT/stats/aim1_mask_feature_sensitivity_functional_drop_nulls.csv"

if [[ "$RUN_AIM1_SCR" == "1" ]]; then
  "$PYTHON_BIN" scripts/export_aim1_scr_sensitivity.py \
    --feature-dir "$FEAR_DIR" \
    --out "$OUT_ROOT/stats/aim1_scr_sensitivity.csv" \
    --feature-space "FearNetwork"
fi

"$PYTHON_BIN" scripts/run_mvpa_l2_primary_models.py \
  --input "$OUT_ROOT/harmonized/mvpa_l2_subject_metrics.csv" \
  --clinical-outlier-z "$CLINICAL_OUTLIER_Z" \
  --out-dir "$OUT_ROOT/stats"

"$PYTHON_BIN" scripts/run_mvpa_l2_sensitivity_models.py \
  --input "$OUT_ROOT/harmonized/mvpa_l2_subject_metrics.csv" \
  --out "$OUT_ROOT/stats/sensitivity_models_all.csv"

"$PYTHON_BIN" scripts/plot_figure_s2_aim2_sensitivity.py \
  --input "$OUT_ROOT/stats/sensitivity_models_all.csv" \
  --figure-dir "$OUT_ROOT/stats/figures" \
  --table-dir "$OUT_ROOT/stats"

"$PYTHON_BIN" scripts/export_mvpa_l2_manuscript_artifacts.py \
  --input "$OUT_ROOT/harmonized/mvpa_l2_subject_metrics.csv" \
  --stats-dir "$OUT_ROOT/stats" \
  --repo-root "."

"$PYTHON_BIN" scripts/summarize_mvpa_l2_results.py \
  --stats-dir "$OUT_ROOT/stats" \
  --out "$OUT_ROOT/stats/mvpa_l2_results_summary.md"
