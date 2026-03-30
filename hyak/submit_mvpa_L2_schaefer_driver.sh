#!/usr/bin/env bash
set -euo pipefail

# Driver: submit all stages (6-17) with resume
STAGES=(6 7 8 9 10 11 12 13 14 15 16 17)

for s in "${STAGES[@]}"; do
  /Users/xiaoqianxiao/PycharmProjects/narsad_multivoxel/hyak/submit_mvpa_L2_schaefer_stage.sh "$s" --resume
  sleep 0.5
 done

echo "Submitted all mvpa_L2_schaefer stages 6-17 (with --resume)."
