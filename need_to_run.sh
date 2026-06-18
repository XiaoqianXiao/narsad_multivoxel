#!/usr/bin/env bash
set -euo pipefail

# Commands to refresh the outputs used by code/mvpa_l2.ipynb after the current
# Hyak/script updates.
#
# Default behavior prints the commands without running them:
#   bash need_to_run.sh
#
# Submit the Hyak late-stage jobs:
#   DRY_RUN=0 bash need_to_run.sh
#
# After those Hyak jobs finish, rebuild the notebook-facing tables:
#   DRY_RUN=0 RUN_HYAK=0 RUN_POSTHYAK_NOW=1 bash need_to_run.sh
#
# Optional controls:
#   RUN_HYAK=0            skip Hyak submission and only print/run post-Hyak commands
#   RUN_POSTHYAK_NOW=1    run post-Hyak commands now; default prints them for after Hyak jobs finish
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
RUN_HYAK="${RUN_HYAK:-1}"
RUN_POSTHYAK_NOW="${RUN_POSTHYAK_NOW:-0}"
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

print_cmd() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
}

cd "$CODE_DIR"

LATE_STAGE_SPEC="23,24,26,29,30"

if [[ "$RUN_OPTIONAL_CORR" == "1" ]]; then
  LATE_STAGE_SPEC="23,24,26,27,28,29,30"
fi

if [[ "$RUN_HYAK" == "1" ]]; then
  echo "Step 1: refresh late Hyak stage bundles used by mvpa_l2.ipynb"
  echo "Submitting stages ${LATE_STAGE_SPEC} as one ordered job per feature space."
  run_cmd bash hyak/submit_mvpa_L2_fearnetwork_stage.sh "$LATE_STAGE_SPEC"
  run_cmd bash hyak/submit_mvpa_L2_memoryfearnetwork_stage.sh "$LATE_STAGE_SPEC"
  if [[ "$RUN_SCHAEFER" == "1" ]]; then
    run_cmd bash hyak/submit_mvpa_L2_schaefer_stage.sh "$LATE_STAGE_SPEC"
  fi
else
  echo "Step 1 skipped because RUN_HYAK=0"
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
if [[ "$RUN_POSTHYAK_NOW" == "1" ]]; then
  run_cmd env "${POSTHYAK_ENV[@]}" bash scripts/run_mvpa_l2_posthyak.sh
else
  echo "Run this after the Hyak jobs above finish:"
  print_cmd env "${POSTHYAK_ENV[@]}" bash scripts/run_mvpa_l2_posthyak.sh
fi

if [[ "$RUN_AIM1_SCR" == "1" ]]; then
  echo "Step 3: optional Aim 1 SCR subgroup sensitivity table"
  if [[ "$RUN_POSTHYAK_NOW" == "1" ]]; then
    run_cmd python3 scripts/export_aim1_scr_sensitivity.py \
      --feature-dir "$FEAR_DIR" \
      --out "$OUT_ROOT/stats/aim1_scr_sensitivity.csv" \
      --feature-space FearNetwork
  else
    print_cmd python3 scripts/export_aim1_scr_sensitivity.py \
      --feature-dir "$FEAR_DIR" \
      --out "$OUT_ROOT/stats/aim1_scr_sensitivity.csv" \
      --feature-space FearNetwork
  fi
fi

if [[ "$RUN_HAUFE_SCR" == "1" ]]; then
  echo "Step 4: optional Aim 2 Haufe/SCR sensitivity tables"
  if [[ "$RUN_POSTHYAK_NOW" == "1" ]]; then
    run_cmd python3 scripts/export_haufe_scr_sensitivity.py \
      --feature-dir "$FEAR_DIR" \
      --scr-flags "$OUT_ROOT/harmonized/scr_sensitivity_groups.csv" \
      --out-summary "$OUT_ROOT/stats/aim2_haufe_scr_sensitivity.csv" \
      --out-roi "$OUT_ROOT/stats/aim2_haufe_scr_sensitivity_roi_distribution.csv"
  else
    print_cmd python3 scripts/export_haufe_scr_sensitivity.py \
      --feature-dir "$FEAR_DIR" \
      --scr-flags "$OUT_ROOT/harmonized/scr_sensitivity_groups.csv" \
      --out-summary "$OUT_ROOT/stats/aim2_haufe_scr_sensitivity.csv" \
      --out-roi "$OUT_ROOT/stats/aim2_haufe_scr_sensitivity_roi_distribution.csv"
  fi
fi

if [[ "$DRY_RUN" != "0" ]]; then
  echo
  echo "Dry run only. Re-run with DRY_RUN=0 to execute these commands."
fi
