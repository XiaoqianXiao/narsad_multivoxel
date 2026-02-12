#!/bin/bash
#SBATCH --partition=cpu-g2
#SBATCH --account=fang
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --output=mvpa_output_%j.out
#SBATCH --error=mvpa_output_%j.err

project_ROOT=/gscratch/fang/NARSAD
CONTAINER_SIF=/gscratch/fang/images/jupyter.sif
APP_PATH=/gscratch/scrubbed/fanglab/xiaoqian/repo/narsad_multivoxel/hyak
OUT_PATH=/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/LSS/WholeBrain_Parcellation
#OUT_PATH=/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/LSS/FearNetwork

# 1. Load the container module
module load apptainer 2>/dev/null || true


# 3. Run the analysis
# We bind both the data root and the code directory
apptainer exec \
    -B "${project_ROOT}:/gscratch/fang/NARSAD" \
    -B "${APP_PATH}:/app" \
    -B "${OUT_PATH}:/output_dir" \
    "${CONTAINER_SIF}" python3 /app/mvpa_L2_voxel_WholeBrain_Parcellation.py --output_dir /output_dir --project_root /gscratch/fang/NARSAD
    #"${CONTAINER_SIF}" python3 /app/mvpa_L2_voxel_FearNetwork_All.py --output_dir /output_dir --project_root /gscratch/fang/NARSAD