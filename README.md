# hyak_narsad

MVPA and LSS pipeline scripts for the NARSAD dataset on Hyak and local environments.

**Core MVPA Scripts**
- `mvpa_L2_voxel_FearNetworkAll.py`: Local L2 voxel-wise MVPA (fear-network ROIs).
- `mvpa_L2_voxel_FearNetworkAll.ipynb`: Notebook version of fear-network MVPA.
- `mvpa_L2_voxel_WholeBrain_Parcellation.py`: Local L2 MVPA on whole-brain parcellation (Glasser + Tian).
- `mvpa_L2_voxel_WholeBrain_Parcellation.ipynb`: Notebook version of parcellation MVPA.
- `mvpa_L2_S4_ROI_WholeBrain.py`: ROI/parcellation MVPA (S4/whole-brain).

**Hyak MVPA Scripts (cluster)**
- `hyak/mvpa_ElasticNet_voxel_FearNetwork_All.py`: ElasticNet voxel-wise MVPA (fear-network ROIs).
- `hyak/mvpa_ElasticNet_voxel_FearNetwork_All.ipynb`: Notebook version (ElasticNet).
- `hyak/mvpa_L2_voxel_FearNetwork_All.py`: L2 voxel-wise MVPA (fear-network ROIs).
- `hyak/mvpa_L2_voxel_WholeBrain_Parcellation.py`: L2 MVPA on whole-brain parcellation.
- `run_mvpa.sh`: Example SLURM job script using Apptainer.

**LSS / First-Level Pipeline**
- `create_1st_LSS_1st_singleTrialEstimate.py`: Build single-trial LSS first-level jobs.
- `create_1st_LSS_2nd_cateAlltrials.py`: Build LSS second-stage category jobs.
- `create_1st_LSS_3rd_similarity.py`: Build LSS similarity jobs.
- `create_1st_LSS_4th_classification.py`: Build LSS classification jobs.
- `run_1st_LSS.py`: Run LSS stages locally.
- `first_LSS_2nd_cateAlltrials.py`: LSS second-stage category processing.
- `first_LSS_3rd_similarity.py`: LSS similarity processing.
- `first_LSS_4th_classification.py`: LSS classification processing.
- `first_level_workflows.py`: Nipype-style first-level workflows.
- `group_level_workflows.py`: Group-level workflows and utilities.

**Voxelwise / Group MVPA Utilities**
- `prepare_X_y_voxel_FearNetwork.py`: Build voxel-wise features for fear-network ROIs.
- `prepare_X_y_voxel_WholeBrain.py`: Build voxel-wise features for whole brain.
- `prepare_X_y_ROI_WholeBrain.py`: Build ROI/parcellation features.
- `create_1st_voxelWise.py`: Build first-level voxel-wise jobs.
- `create_pre_group_voxelWise.py`: Build pre-group voxel-wise jobs.
- `run_group_voxelWise.py`: Run group voxel-wise MVPA.
- `run_pre_group_voxelWise.py`: Run pre-group voxel-wise MVPA.
- `group_LSS_searchlight.py`: Group-level searchlight analyses.
- `create_group_LSS_searchlight.py`: Build group searchlight jobs.
- `similarity.py`: Shared similarity/RSA helpers.
- `utils.py`: General utilities.
- `other_tools.py`: Miscellaneous helper functions.

**SLURM / Container Launchers**
- `launch_1st_LSS_1st_singleTrialEstimate.sh`
- `launch_1st_LSS_2nd.sh`
- `launch_1st_LSS_2nd_cateAlltrials.sh`
- `launch_1st_LSS_3rd_similarity.sh`
- `launch_1st_voxelWise.sh`
- `launch_group_LSS.sh`
- `launch_group_voxelWise.sh`
- `launch_pre_group_voxelWise.sh`
- `create_group_voxelWise.sh`
- `check_error_files.sh`

**Environment / Containers**
- `pyproject.toml`, `poetry.lock`: Python dependencies.
- `dockerfile`: Docker build for local container runs.
- `run_1st_level.def`, `jupyter.def`: Apptainer/Singularity definitions.

**Data Layout (defaults in scripts)**
- `project_root`: `/Users/xiaoqianxiao/projects/NARSAD` (override on Hyak with `--project_root /gscratch/fang/NARSAD`)
- Fear-network voxel data:
  - `MRI/derivatives/fMRI_analysis/LSS/firstLevel/all_subjects/fear_network/phase2_X_ext_y_ext_roi_voxels.npz`
  - `MRI/derivatives/fMRI_analysis/LSS/firstLevel/all_subjects/fear_network/phase3_X_reinst_y_reinst_roi_voxels.npz`
- Whole-brain parcellation data:
  - `MRI/derivatives/fMRI_analysis/LSS/firstLevel/all_subjects/wholeBrain_S4/cope_ROI/phase2_X_ext_y_ext_glasser_tian.npz`
  - `MRI/derivatives/fMRI_analysis/LSS/firstLevel/all_subjects/wholeBrain_S4/cope_ROI/phase3_X_reinst_y_reinst_glasser_tian.npz`
- Metadata:
  - `MRI/source_data/behav/drug_order.csv`

**Running on Hyak (example)**
```bash
sbatch run_mvpa.sh
```

**Running locally (example)**
```bash
python hyak/mvpa_L2_voxel_FearNetwork_All.py \
  --project_root /Users/xiaoqianxiao/projects/NARSAD \
  --output_dir /Users/xiaoqianxiao/projects/NARSAD/MRI/derivatives/fMRI_analysis/LSS/outputs/L2_FearNetwork
```

**Outputs**
- Figures and `*.pkl` result artifacts are written to `--output_dir`.

**Workflow (Recommended Order)**
1. **First‑level LSS single‑trial estimation**
   - Generate scripts: `create_1st_LSS_1st_singleTrialEstimate.py`
   - Launch on Hyak: `launch_1st_LSS_1st_singleTrialEstimate.sh`
2. **LSS category aggregation**
   - Generate scripts: `create_1st_LSS_2nd_cateAlltrials.py`
   - Launch on Hyak: `launch_1st_LSS_2nd_cateAlltrials.sh` (or `launch_1st_LSS_2nd.sh`)
3. **LSS similarity & classification**
   - Similarity: `create_1st_LSS_3rd_similarity.py` → `launch_1st_LSS_3rd_similarity.sh`
   - Classification: `create_1st_LSS_4th_classification.py`
4. **Prepare MVPA feature matrices**
   - Fear network voxels: `prepare_X_y_voxel_FearNetwork.py`
   - Whole‑brain voxels: `prepare_X_y_voxel_WholeBrain.py`
   - Parcellation/ROI: `prepare_X_y_ROI_WholeBrain.py`
5. **Run MVPA analyses**
   - Local: `mvpa_L2_voxel_FearNetworkAll.py`, `mvpa_L2_voxel_WholeBrain_Parcellation.py`
   - Hyak: `run_mvpa.sh` (calls `hyak/mvpa_ElasticNet_voxel_FearNetwork_All.py` by default)
6. **Group / searchlight analyses (optional)**
   - Group voxel‑wise: `run_group_voxelWise.py`
   - Pre‑group: `run_pre_group_voxelWise.py`
   - Searchlight: `group_LSS_searchlight.py` (or generate with `create_group_LSS_searchlight.py`)

**Notes**
- Many scripts assume `project_root` points to the dataset root; update per environment.
- Hyak scripts generally use `/gscratch/fang/NARSAD` as the root dataset path.
# narsad_multivoxel
