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

FEAR_DIR="${FEAR_DIR:-outputs/mvpa_l2/FearNetwork}"
MEMORY_DIR="${MEMORY_DIR:-outputs/mvpa_l2/MemoryFearNetwork}"
SCHAEFER_DIR="${SCHAEFER_DIR:-}"
SCR_DIR="${SCR_DIR:-scr_analysis_outputs}"
OUT_ROOT="${OUT_ROOT:-outputs/mvpa_l2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SCR_FLAGS="${SCR_FLAGS:-}"
SCR_FLAGS_OUT="$OUT_ROOT/harmonized/scr_sensitivity_groups.csv"
CLINICAL_OUTLIER_Z="${CLINICAL_OUTLIER_Z:-3.0}"

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

mkdir -p "$(dirname "$SCR_FLAGS_OUT")"
if [[ -n "$SCR_FLAGS" && -f "$SCR_FLAGS" ]]; then
  if [[ "$SCR_FLAGS" != "$SCR_FLAGS_OUT" ]]; then
    cp "$SCR_FLAGS" "$SCR_FLAGS_OUT"
  fi
  echo "Using prebuilt SCR sensitivity flags -> $SCR_FLAGS_OUT"
elif [[ -n "$SCR_FLAGS" ]]; then
  echo "ERROR: SCR_FLAGS was set but the file does not exist: $SCR_FLAGS" >&2
  echo "Expected a prebuilt CSV such as /app/outputs/mvpa_l2/harmonized/scr_sensitivity_groups.csv" >&2
  exit 1
else
  "$PYTHON_BIN" scripts/build_scr_sensitivity_groups.py \
    --scr-dir "$SCR_DIR" \
    --out "$SCR_FLAGS_OUT"
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
  --out "$OUT_ROOT/harmonized/mvpa_l2_subject_metrics.csv"

"$PYTHON_BIN" scripts/run_mvpa_l2_primary_models.py \
  --input "$OUT_ROOT/harmonized/mvpa_l2_subject_metrics.csv" \
  --clinical-outlier-z "$CLINICAL_OUTLIER_Z" \
  --out-dir "$OUT_ROOT/stats"

"$PYTHON_BIN" scripts/run_mvpa_l2_sensitivity_models.py \
  --input "$OUT_ROOT/harmonized/mvpa_l2_subject_metrics.csv" \
  --out "$OUT_ROOT/stats/sensitivity_models_all.csv"

"$PYTHON_BIN" scripts/export_mvpa_l2_manuscript_artifacts.py \
  --input "$OUT_ROOT/harmonized/mvpa_l2_subject_metrics.csv" \
  --stats-dir "$OUT_ROOT/stats" \
  --repo-root "."

"$PYTHON_BIN" scripts/summarize_mvpa_l2_results.py \
  --stats-dir "$OUT_ROOT/stats" \
  --out "$OUT_ROOT/stats/mvpa_l2_results_summary.md"
