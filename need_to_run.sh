#!/usr/bin/env bash
set -euo pipefail

# Commands to refresh the outputs used by code/mvpa_l2.ipynb after the current
# Hyak/script updates.
#
# Default behavior prints the commands without running them:
#   bash need_to_run.sh
#
# Actually submit/run them:
#   DRY_RUN=0 bash need_to_run.sh
#
# Optional controls:
#   RUN_OPTIONAL_CORR=1   include Hyak stages 27 and 28
#   RUN_SCHAEFER=1        include WholeBrain/Schaefer stages
#   RUN_AIM1_SCR=1        export Aim 1 SCR sensitivity table
#   RUN_HAUFE_SCR=1       export Aim 2 Haufe/SCR sensitivity tables
#
# Usual Hyak path overrides:
#   FEAR_DIR=/gscratch/.../FearNetwork
#   MEMORY_DIR=/gscratch/.../MemoryFearNetwork
#   SCHAEFER_DIR=/gscratch/.../wholebrain_parcellation_schaefer
#   OUT_ROOT=/gscratch/.../mvpa_l2
#   SCR_DIR=/gscratch/.../scr_analysis_outputs

DRY_RUN="${DRY_RUN:-1}"
RUN_OPTIONAL_CORR="${RUN_OPTIONAL_CORR:-0}"
RUN_SCHAEFER="${RUN_SCHAEFER:-0}"
RUN_AIM1_SCR="${RUN_AIM1_SCR:-0}"
RUN_HAUFE_SCR="${RUN_HAUFE_SCR:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="${CODE_DIR:-$REPO_ROOT/code}"

FEAR_DIR="${FEAR_DIR:-/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/LSS/results/FearNetwork}"
MEMORY_DIR="${MEMORY_DIR:-/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/LSS/results/MemoryFearNetwork}"
SCHAEFER_DIR="${SCHAEFER_DIR:-/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/LSS/results/wholebrain_parcellation_schaefer}"
OUT_ROOT="${OUT_ROOT:-/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/LSS/results/mvpa_l2}"
SCR_DIR="${SCR_DIR:-/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/scr_analysis_outputs}"

run_cmd() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  if [[ "$DRY_RUN" == "0" ]]; then
    "$@"
  fi
}

cd "$CODE_DIR"

echo "Step 1: refresh late Hyak stage bundles used by mvpa_l2.ipynb"
for stage in 23 24 26 29 30; do
  run_cmd bash hyak/submit_mvpa_L2_fearnetwork_stage.sh "$stage" --resume
  run_cmd bash hyak/submit_mvpa_L2_memoryfearnetwork_stage.sh "$stage" --resume
  if [[ "$RUN_SCHAEFER" == "1" ]]; then
    run_cmd bash hyak/submit_mvpa_L2_schaefer_stage.sh "$stage" --resume
  fi
done

if [[ "$RUN_OPTIONAL_CORR" == "1" ]]; then
  echo "Step 1b: optional Pearson/partial correlation stage bundles"
  for stage in 27 28; do
    run_cmd bash hyak/submit_mvpa_L2_fearnetwork_stage.sh "$stage" --resume
    run_cmd bash hyak/submit_mvpa_L2_memoryfearnetwork_stage.sh "$stage" --resume
    if [[ "$RUN_SCHAEFER" == "1" ]]; then
      run_cmd bash hyak/submit_mvpa_L2_schaefer_stage.sh "$stage" --resume
    fi
  done
fi

echo "Step 2: rebuild notebook-facing harmonized/stat/model/QC tables"
POSTHYAK_ENV=(
  FEAR_DIR="$FEAR_DIR"
  MEMORY_DIR="$MEMORY_DIR"
  OUT_ROOT="$OUT_ROOT"
  SCR_DIR="$SCR_DIR"
)
if [[ "$RUN_SCHAEFER" == "1" ]]; then
  POSTHYAK_ENV+=(SCHAEFER_DIR="$SCHAEFER_DIR")
fi
run_cmd env "${POSTHYAK_ENV[@]}" bash scripts/run_mvpa_l2_posthyak.sh

if [[ "$RUN_AIM1_SCR" == "1" ]]; then
  echo "Step 3: optional Aim 1 SCR subgroup sensitivity table"
  run_cmd python3 scripts/export_aim1_scr_sensitivity.py \
    --feature-dir "$FEAR_DIR" \
    --out "$OUT_ROOT/stats/aim1_scr_sensitivity.csv" \
    --feature-space FearNetwork
fi

if [[ "$RUN_HAUFE_SCR" == "1" ]]; then
  echo "Step 4: optional Aim 2 Haufe/SCR sensitivity tables"
  run_cmd python3 scripts/export_haufe_scr_sensitivity.py \
    --feature-dir "$FEAR_DIR" \
    --scr-flags "$OUT_ROOT/harmonized/scr_sensitivity_groups.csv" \
    --out-summary "$OUT_ROOT/stats/aim2_haufe_scr_sensitivity.csv" \
    --out-roi "$OUT_ROOT/stats/aim2_haufe_scr_sensitivity_roi_distribution.csv"
fi

if [[ "$DRY_RUN" != "0" ]]; then
  echo
  echo "Dry run only. Re-run with DRY_RUN=0 to execute these commands."
fi
