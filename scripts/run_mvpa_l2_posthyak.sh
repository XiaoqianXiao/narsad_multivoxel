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

python scripts/build_scr_sensitivity_groups.py \
  --scr-dir "$SCR_DIR" \
  --out "$OUT_ROOT/harmonized/scr_sensitivity_groups.csv"

FEATURE_ARGS=(
  --feature-dir "FearNetwork=$FEAR_DIR"
  --feature-dir "MemoryFearNetwork=$MEMORY_DIR"
)

if [[ -n "$SCHAEFER_DIR" ]]; then
  FEATURE_ARGS+=(--feature-dir "Schaefer=$SCHAEFER_DIR")
fi

python scripts/export_mvpa_l2_metrics.py \
  "${FEATURE_ARGS[@]}" \
  --scr-flags "$OUT_ROOT/harmonized/scr_sensitivity_groups.csv" \
  --out "$OUT_ROOT/harmonized/mvpa_l2_subject_metrics.csv"

python scripts/run_mvpa_l2_primary_models.py \
  --input "$OUT_ROOT/harmonized/mvpa_l2_subject_metrics.csv" \
  --out-dir "$OUT_ROOT/stats"

python scripts/run_mvpa_l2_sensitivity_models.py \
  --input "$OUT_ROOT/harmonized/mvpa_l2_subject_metrics.csv" \
  --out "$OUT_ROOT/stats/sensitivity_models_all.csv"

python scripts/summarize_mvpa_l2_results.py \
  --stats-dir "$OUT_ROOT/stats" \
  --out "$OUT_ROOT/stats/mvpa_l2_results_summary.md"
