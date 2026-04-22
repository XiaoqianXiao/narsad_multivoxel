#!/usr/bin/env bash
set -euo pipefail

# Driver: submit all logical stages (1-9, 17) with resume
STAGES=(1 2 3 4 5 6 7 8 9 17)

for s in "${STAGES[@]}"; do
  /Users/xiaoqianxiao/PycharmProjects/narsad_multivoxel/hyak/submit_mvpa_L2_schaefer_stage.sh "$s" --resume
  sleep 0.5
 done

echo "Submitted all mvpa_L2_schaefer stages 1-9 and 17 (with --resume)."
