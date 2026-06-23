#!/usr/bin/env bash
set -euo pipefail

# Recovery helper for zero-byte Stage 11 SAD permutation-importance chunks
# reported by mvpa_schaefer_a11_importance_SAD_merge_36294213.err.
#
# Run from the repository root on Hyak, for example:
#   bash hyak/resubmit_voxel_WholeBrain_Schaefer.sh
#
# After all submitted chunk jobs finish successfully, rerun the merge:
#   STAGE11_CHUNKS=500 bash hyak/submit_mvpa_L2_schaefer_stage.sh 11:SAD:merge

FAILED_STAGE11_SAD_CHUNKS=(
  309 310
  320 321 322 323 324 325 326 327 328 329
  330 331 332 333 334 335 336 337 338 339
  340 341 342 343 344 345 346 347 348 349
  350 351 352 353 354 355 356 357 358 359
)

for chunk_idx in "${FAILED_STAGE11_SAD_CHUNKS[@]}"; do
  echo "Submitting Stage 11 SAD chunk ${chunk_idx}/500"
  STAGE11_CHUNKS=500 STAGE11_CHUNK_IDX="$chunk_idx" \
    bash hyak/submit_mvpa_L2_schaefer_stage.sh 11:SAD
done

cat <<'EOF'

Submitted the failed Stage 11 SAD chunk jobs.

Once they finish successfully, run:
  STAGE11_CHUNKS=500 bash hyak/submit_mvpa_L2_schaefer_stage.sh 11:SAD:merge

EOF
