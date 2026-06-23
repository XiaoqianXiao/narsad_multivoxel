#!/usr/bin/env bash
set -euo pipefail

# Recovery helper for zero-byte Stage 11 SAD permutation-importance chunks
# reported by mvpa_schaefer_a11_importance_SAD_merge_36328527.err.
#
# Run from the repository root on Hyak, for example:
#   bash hyak/resubmit_voxel_WholeBrain_Schaefer.sh
#
# The submit wrapper will submit the sparse array and a dependent merge job.
STAGE11_CHUNKS=500 \
STAGE11_ARRAY_SPEC=309,310,320-359 \
  bash hyak/submit_mvpa_L2_schaefer_stage.sh 11:SAD
