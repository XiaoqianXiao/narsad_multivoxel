#!/bin/bash
#SBATCH --partition=cpu-g2
#SBATCH --account=fang
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --time=20:00:00
#SBATCH --output=mvpa_output_%j.out
#SBATCH --error=mvpa_output_%j.err

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'USAGE'
Usage:
  ./run_mvpa.sh

Behavior:
  - Submits two jobs (WholeBrain + FearNetwork) with separate output directories.
  - When running inside SLURM, uses ANALYSIS to choose which script to run.

Environment variables (optional):
  ANALYSIS=WholeBrain|FearNetwork
  OUT_PATH=/path/to/output_dir

Examples:
  ./run_mvpa.sh
  # WholeBrain only:
  # ANALYSIS=WholeBrain sbatch run_mvpa.sh
USAGE
    exit 0
fi

project_ROOT=/gscratch/fang/NARSAD
CONTAINER_SIF=/gscratch/fang/images/jupyter.sif
APP_PATH=/gscratch/scrubbed/fanglab/xiaoqian/repo/narsad_multivoxel/hyak
OUT_PATH_WHOLEBRAIN=/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/LSS/WholeBrain_Parcellation
OUT_PATH_FEARNETWORK=/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/LSS/FearNetwork

# 1. Load the container module
module load apptainer 2>/dev/null || true


# If not running under SLURM, submit two jobs (WholeBrain + FearNetwork)
if [[ -z "${SLURM_JOB_ID}" ]]; then
    sbatch --job-name=mvpa_wholebrain --output=mvpa_wholebrain_%j.out --error=mvpa_wholebrain_%j.err \
        --export=ALL,ANALYSIS=WholeBrain,OUT_PATH=${OUT_PATH_WHOLEBRAIN} "$0"
    sbatch --job-name=mvpa_fearnetwork --output=mvpa_fearnetwork_%j.out --error=mvpa_fearnetwork_%j.err \
        --export=ALL,ANALYSIS=FearNetwork,OUT_PATH=${OUT_PATH_FEARNETWORK} "$0"
    exit 0
fi

# Default to WholeBrain if not specified
ANALYSIS=${ANALYSIS:-WholeBrain}
OUT_PATH=${OUT_PATH:-${OUT_PATH_WHOLEBRAIN}}

# 3. Run the analysis
# We bind both the data root and the code directory
if [[ "${ANALYSIS}" == "FearNetwork" ]]; then
    SCRIPT=/app/mvpa_L2_voxel_FearNetwork_All.py
else
    SCRIPT=/app/mvpa_L2_voxel_WholeBrain_Parcellation.py
fi

apptainer exec \
    -B "${project_ROOT}:/gscratch/fang/NARSAD" \
    -B "${APP_PATH}:/app" \
    -B "${OUT_PATH}:/output_dir" \
    "${CONTAINER_SIF}" python3 "${SCRIPT}" --output_dir /output_dir --project_root /gscratch/fang/NARSAD
