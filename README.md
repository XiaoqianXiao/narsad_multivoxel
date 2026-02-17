# NARSAD Multivoxel Analysis Pipeline

MVPA (Multi-Voxel Pattern Analysis) and LSS (Least Squares Separate) pipeline scripts for the NARSAD dataset, designed for both Hyak cluster and local environments.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Project Structure](#project-structure)
- [Complete Workflow](#complete-workflow)
- [Script Reference](#script-reference)
- [Data Layout](#data-layout)
- [Running Analyses](#running-analyses)
- [Troubleshooting](#troubleshooting)

## Overview

This repository contains a comprehensive pipeline for analyzing fMRI data using:

- **LSS (Least Squares Separate)**: Single-trial GLM analysis for high-resolution pattern analysis
- **MVPA**: Multi-voxel pattern analysis for classification and representational similarity
- **Group-level statistics**: FLAMEO and Randomise for group inference

The pipeline supports multiple analysis approaches:
- Fear network ROI analysis
- Whole-brain voxel-wise analysis
- Whole-brain parcellation analysis (Glasser + Tian atlases)
- Searchlight similarity analysis

## Quick Start

### On Hyak Cluster

```bash
# Step 1: Generate and launch first-level LSS jobs
python create_1st_LSS_1st_singleTrialEstimate.py
./launch_1st_LSS_1st_singleTrialEstimate.sh

# Step 2: After Step 1 completes, merge trials
python create_1st_LSS_2nd_cateAlltrials.py
./launch_1st_LSS_2nd_cateAlltrials.sh

# Step 3: Prepare MVPA features (can run in parallel)
python prepare_X_y_voxel_FearNetwork.py
python prepare_X_y_voxel_WholeBrain.py
python prepare_X_y_ROI_WholeBrain.py

# Step 4: Run MVPA analysis
sbatch run_mvpa.sh
```

### Locally

```bash
# Run MVPA analysis locally
python mvpa_L2_voxel_FearNetworkAll.py \
  --project_root /path/to/NARSAD \
  --output_dir /path/to/output
```

## Prerequisites

### Software Requirements

- Python 3.8+
- FSL (for GLM analysis)
- Apptainer/Singularity (for containerized runs on Hyak)
- SLURM (for cluster job submission)

### Python Dependencies

Install via Poetry (recommended):
```bash
poetry install
```

Or via pip:
```bash
pip install -r requirements.txt
```

Key dependencies:
- `nibabel`, `nilearn`: Neuroimaging data handling
- `scikit-learn`: MVPA and machine learning
- `pandas`, `numpy`: Data manipulation
- `nipype`: Workflow management
- `pybids`: BIDS data handling

### Data Requirements

- BIDS-formatted NARSAD dataset
- Preprocessed fMRI data (fMRIPrep derivatives)
- Behavioral event files (CSV format)
- ROI masks and atlas files (Glasser, Tian)

## Project Structure

```
narsad_multivoxel/
├── create_1st_LSS_*.py          # LSS workflow script generators
├── first_LSS_*.py                # LSS processing scripts
├── prepare_X_y_*.py              # Feature matrix preparation
├── mvpa_L2_*.py                  # MVPA analysis scripts
├── group_LSS_searchlight.py      # Group-level searchlight
├── launch_*.sh                   # SLURM job launchers
├── hyak/                         # Cluster-specific scripts
│   ├── mvpa_L2_voxel_FearNetwork_All.py
│   └── mvpa_L2_voxel_WholeBrain_Parcellation.py
├── first_level_workflows.py      # Nipype workflows
├── group_level_workflows.py      # Group-level utilities
├── utils.py                      # Shared utilities
└── similarity.py                 # RSA/similarity helpers
```

## Complete Workflow

This section provides a step-by-step guide through the entire LSS and MVPA analysis pipeline. Each step builds on the previous one, so follow them in order.

### Step 1: First-Level LSS Single-Trial Estimation

**Purpose**: Run LSS (Least Squares Separate) GLM analysis on each individual trial to obtain single-trial beta estimates.

**1.1 Generate SLURM scripts**:
```bash
python create_1st_LSS_1st_singleTrialEstimate.py
```

**Options**:
- `--subjects N101 N102`: Process specific subjects only
- `--tasks phase2 phase3`: Process specific tasks only
- `--trial-range 1 20`: Process only trials 1-20
- `--dry-run`: Preview what would be created
- `--account fang --partition ckpt-all --memory 40G`: Custom SLURM settings

**Output**: SLURM scripts in `/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/Lss/{task}/sub_{subject}_trial_{trial_ID}_slurm.sh`

**1.2 Launch jobs**:
```bash
./launch_1st_LSS_1st_singleTrialEstimate.sh
```

**Options**:
- `--phase phase2`: Launch only phase2 jobs
- `--subjects N101 N102`: Launch only specific subjects
- `--dry-run`: Preview what would be launched

**What it does**: Each script runs `run_1st_LSS.py` which calls `first_level_wf_LSS` from `first_level_workflows.py` to perform GLM analysis on individual trials.

**Expected outputs**: 
- Individual trial cope files: `sub-{subject}_task-{task}_trial-{trial_ID}_cope1.nii.gz`
- Located in: `MRI/derivatives/fMRI_analysis/LSS/firstLevel/sub-{subject}/task-{task}/trial-{trial_ID}/`

---

### Step 2: LSS Trial Merging (Category Aggregation)

**Purpose**: Merge individual trial outputs into 4D NIfTI images (one per subject-task-contrast).

**2.1 Generate SLURM scripts**:
```bash
python create_1st_LSS_2nd_cateAlltrials.py
```

**Options**:
- `--subjects N101 N102`: Process specific subjects only
- `--tasks phase2 phase3`: Process specific tasks only
- `--contrasts 1 2 3`: Process specific contrasts only
- `--contrast-range 1 5`: Process contrast range
- `--dry-run`: Preview what would be created

**Output**: SLURM scripts in `/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/LSS_step2/{task}/sub_{subject}_slurm.sh`

**2.2 Launch jobs**:
```bash
./launch_1st_LSS_2nd_cateAlltrials.sh
```

**Options**:
- `--task phase2`: Launch only phase2 jobs
- `--subjects N101 N102`: Launch only specific subjects
- `--max-jobs 50`: Limit concurrent jobs
- `--job-delay 1`: Delay between submissions (seconds)
- `--dry-run`: Preview what would be launched

**What it does**: Runs `first_LSS_2nd_cateAlltrials.py` to concatenate all trial cope files into 4D images.

**Expected outputs**:
- 4D images: `sub-{subject}_task-{task}_contrast{contrast}.nii.gz`
- Located in: `MRI/derivatives/fMRI_analysis/LSS/firstLevel/all_subjects/`

---

### Step 3: LSS Similarity Analysis

**Purpose**: Compute similarity matrices (RSA) for searchlight and ROI analyses.

**3.1 Generate SLURM scripts**:
```bash
python create_1st_LSS_3rd_similarity.py
```

**Options**:
- `--analysis-types searchlight roi both`: Choose analysis types (default: both)
- `--batch-size 1000`: Batch size for searchlight (default: 1000)
- `--n-jobs 12`: Number of parallel jobs (default: 12)
- `--cpus-per-task 16 --memory 20G --time 04:00:00`: Custom SLURM resources
- `--profile`: Enable profiling for debugging
- `--dry-run`: Preview what would be created

**Output**: SLURM scripts in `/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/Lss_step3/{task}/{analysis_type}/sub_{subject}_slurm.sh`

**3.2 Launch jobs**:
```bash
./launch_1st_LSS_3rd_similarity.sh
```

**Options**:
- `--dry-run`: Preview what would be launched
- `--verbose`: Show detailed output

**What it does**: Runs `first_LSS_3rd_similarity.py` to compute:
- **Searchlight**: Voxel-wise similarity maps using spherical searchlights
- **ROI**: Similarity matrices within predefined ROIs (Schaefer + Tian atlas)

**Expected outputs**:
- Searchlight maps: `sub-{subject}_task-{task}_searchlight_similarity.nii.gz`
- ROI matrices: `sub-{subject}_task-{task}_roi_similarity.npz`
- Located in: `MRI/derivatives/fMRI_analysis/LSS/firstLevel/all_subjects/`

---

### Step 4: LSS ROI Classification

**Purpose**: Perform ROI-based MVPA/classification using single-trial LSS outputs (cross-phase generalization).

**4.1 Generate SLURM scripts**:
```bash
python create_1st_LSS_4th_classification.py
```

**Options**:
- `--phase2-task phase2`: Extinction task name (default: phase2)
- `--phase3-task phase3`: Reinstatement task name (default: phase3)
- `--n-splits 4`: Number of CV folds (default: 4)
- `--cpus-per-task 8 --memory 16G --time 06:00:00`: Custom SLURM resources
- `--dry-run`: Preview what would be created

**Output**: SLURM scripts in `/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/Lss_step4/sub_{subject}_classification_slurm.sh`

**What it does**: Runs `first_LSS_4th_classification.py` to perform:
- ROI-based classification using phase2 (extinction) and phase3 (reinstatement) data
- Cross-phase generalization (PSI analysis)
- Cross-validation with subject-level grouping

**Expected outputs**:
- Classification results: `sub-{subject}_classification_results.pkl`
- Located in: `MRI/derivatives/fMRI_analysis/LSS/firstLevel/all_subjects/classification/`

---

### Step 5: Prepare MVPA Feature Matrices

**Purpose**: Extract and prepare feature matrices (X) and labels (y) for MVPA analyses from LSS outputs.

**5.1 Fear Network Voxel Features**:
```bash
python prepare_X_y_voxel_FearNetwork.py
```

**Output**: 
- `phase2_X_ext_y_ext_roi_voxels.npz`
- `phase3_X_reinst_y_reinst_roi_voxels.npz`
- Located in: `MRI/derivatives/fMRI_analysis/LSS/firstLevel/all_subjects/fear_network/`

**What it does**: Extracts voxel-wise features from anatomical ROI masks (Gillian anatomically constrained ROIs).

**5.2 Whole-Brain Voxel Features**:
```bash
python prepare_X_y_voxel_WholeBrain.py
```

**Output**:
- `phase2_X_ext_y_ext_voxels_glasser_tian.npz`
- `phase3_X_reinst_y_reinst_voxels_glasser_tian.npz`
- Located in: `MRI/derivatives/fMRI_analysis/LSS/firstLevel/all_subjects/group_level/`

**What it does**: Extracts whole-brain voxel features with Glasser & Tian atlas parcellation metadata.

**5.3 ROI/Parcellation Features**:
```bash
python prepare_X_y_ROI_WholeBrain.py
```

**Output**:
- `phase2_X_ext_y_ext_glasser_tian.npz`
- `phase3_X_reinst_y_reinst_glasser_tian.npz`
- Located in: `MRI/derivatives/fMRI_analysis/LSS/firstLevel/all_subjects/group_level/`

**What it does**: Extracts mean beta values within each parcel (Glasser cortical + Tian subcortical).

**Note**: All three preparation scripts can be run independently and in parallel. They read from the same LSS outputs but extract different feature representations.

---

### Step 6: Run MVPA Analyses

**Purpose**: Perform comprehensive MVPA analyses on prepared feature matrices.

**6.1 Local MVPA (Fear Network)**:
```bash
python mvpa_L2_voxel_FearNetworkAll.py \
  --project_root /path/to/NARSAD \
  --output_dir /path/to/output
```

**6.2 Local MVPA (Whole-Brain Parcellation)**:
```bash
python mvpa_L2_voxel_WholeBrain_Parcellation.py \
  --project_root /path/to/NARSAD \
  --output_dir /path/to/output
```

**6.3 Local MVPA (ROI Whole-Brain)**:
```bash
python mvpa_L2_S4_ROI_WholeBrain.py \
  --project_root /path/to/NARSAD \
  --output_dir /path/to/output
```

**6.4 Hyak MVPA (Cluster)**:
```bash
sbatch run_mvpa.sh
```

**Options** (via environment variables):
- `ANALYSIS=WholeBrain`: Run whole-brain analysis only
- `ANALYSIS=FearNetwork`: Run fear-network analysis only
- `OUT_PATH=/path/to/output`: Custom output directory

**What it does**: Performs comprehensive MVPA including:
- Neural dissociation (pairwise classification)
- Spatial topology & visualization (Haufe transform)
- Feature importance (permutation)
- Representational drift analysis
- Decision boundary characteristics
- Oxytocin effects analysis
- Cross-phase generalization

**Expected outputs**:
- Results pickle files: `*.pkl` or `*.joblib`
- Figures: `*.png` or `*.pdf`
- Located in: `--output_dir` or default `MRI/derivatives/fMRI_analysis/LSS/results/mvpa_outputs/`

---

### Step 7: Group-Level Searchlight Analysis

**Purpose**: Perform group-level statistical analysis on LSS searchlight similarity maps.

**7.1 Generate SLURM scripts**:
```bash
python create_group_LSS_searchlight.py --method flameo
```

**Options**:
- `--method flameo`: Parametric analysis with FLAMEO (recommended)
- `--method randomise`: Non-parametric permutation testing with Randomise
- `--time 02:00:00 --memory 20G --cpus 16`: Custom SLURM resources
- `--list-map-types`: List all map types that will be generated
- `--dry-run`: Preview what would be created

**Output**: SLURM scripts in `/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/Lss_group_searchlight/{method}/group_LSS_searchlight_{map_type}_{task}_{method}_slurm.sh`

**Map types generated**:
- **Within-condition** (6): `within-SHOCK`, `within-FIXATION`, `within-CS-_first`, `within-CS-`, `within-CSS`, `within-CSR`
- **Between-condition** (15): All pairwise combinations (e.g., `between-FIXATION-CS-_first`, `between-SHOCK-CSS`)

**7.2 Launch jobs**:
```bash
./launch_group_LSS.sh --method flameo
```

**Options**:
- `--method flameo`: Launch only FLAMEO scripts
- `--method randomise`: Launch only Randomise scripts
- `--map-type within-FIXATION`: Launch only specific map type
- `--task phase2`: Launch only specific task
- `--max-jobs 10 --job-delay 2`: Control job submission rate
- `--dry-run`: Preview what would be launched

**What it does**: Runs `group_LSS_searchlight.py` to perform:
- **FLAMEO**: Parametric group-level analysis with cluster correction
- **Randomise**: Non-parametric permutation testing with TFCE (Threshold-Free Cluster Enhancement)

**Expected outputs**:
- Group-level statistical maps: `group_{map_type}_{task}_{method}_tstat.nii.gz`
- Cluster-corrected maps: `group_{map_type}_{task}_{method}_clustered.nii.gz`
- Located in: `MRI/derivatives/fMRI_analysis/LSS/group_level/searchlight/`

---

## Workflow Summary

**Complete pipeline order**:
1. **Step 1**: First-level LSS single-trial estimation → Individual trial beta maps
2. **Step 2**: Trial merging → 4D images per subject-task-contrast
3. **Step 3**: Similarity analysis → Searchlight and ROI similarity maps
4. **Step 4**: ROI classification → Cross-phase generalization results
5. **Step 5**: Feature preparation → MVPA-ready feature matrices (can run in parallel)
6. **Step 6**: MVPA analyses → Comprehensive classification and representational analyses
7. **Step 7**: Group-level searchlight → Group statistical maps

**Typical workflow timeline**:
- Steps 1-2: ~1-2 weeks (depends on number of subjects/trials)
- Step 3: ~2-3 days per subject
- Step 4: ~1 day per subject
- Step 5: ~1-2 hours (can run in parallel)
- Step 6: ~1-2 days per analysis type
- Step 7: ~1-2 days per method

## Script Reference

### LSS Pipeline Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `create_1st_LSS_1st_singleTrialEstimate.py` | Generate SLURM scripts for single-trial LSS GLM | SLURM scripts for each trial |
| `run_1st_LSS.py` | Execute single-trial LSS analysis | Individual trial cope files |
| `create_1st_LSS_2nd_cateAlltrials.py` | Generate scripts for trial merging | SLURM scripts for merging |
| `first_LSS_2nd_cateAlltrials.py` | Merge trials into 4D images | 4D NIfTI images |
| `create_1st_LSS_3rd_similarity.py` | Generate similarity analysis scripts | SLURM scripts for similarity |
| `first_LSS_3rd_similarity.py` | Compute similarity matrices | Searchlight/ROI similarity maps |
| `create_1st_LSS_4th_classification.py` | Generate classification scripts | SLURM scripts for classification |
| `first_LSS_4th_classification.py` | ROI-based classification | Classification results |

### Feature Preparation Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `prepare_X_y_voxel_FearNetwork.py` | Extract fear network voxel features | NPZ files with X, y matrices |
| `prepare_X_y_voxel_WholeBrain.py` | Extract whole-brain voxel features | NPZ files with voxel data |
| `prepare_X_y_ROI_WholeBrain.py` | Extract ROI/parcellation features | NPZ files with parcel data |

### MVPA Analysis Scripts

| Script | Purpose | Environment |
|--------|---------|-------------|
| `mvpa_L2_voxel_FearNetworkAll.py` | Fear network MVPA | Local |
| `mvpa_L2_voxel_WholeBrain_Parcellation.py` | Whole-brain parcellation MVPA | Local |
| `mvpa_L2_S4_ROI_WholeBrain.py` | ROI whole-brain MVPA | Local |
| `hyak/mvpa_L2_voxel_FearNetwork_All.py` | Fear network MVPA | Hyak |
| `hyak/mvpa_L2_voxel_WholeBrain_Parcellation.py` | Whole-brain MVPA | Hyak |
| `hyak/mvpa_ElasticNet_voxel_FearNetwork_All.py` | ElasticNet MVPA | Hyak |

### Group-Level Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `create_group_LSS_searchlight.py` | Generate group searchlight scripts | SLURM scripts |
| `group_LSS_searchlight.py` | Group-level searchlight analysis | Statistical maps |

### Launcher Scripts

| Script | Purpose |
|--------|---------|
| `launch_1st_LSS_1st_singleTrialEstimate.sh` | Launch Step 1 jobs |
| `launch_1st_LSS_2nd_cateAlltrials.sh` | Launch Step 2 jobs |
| `launch_1st_LSS_3rd_similarity.sh` | Launch Step 3 jobs |
| `launch_group_LSS.sh` | Launch group-level jobs |
| `run_mvpa.sh` | Launch MVPA jobs on Hyak |
| `check_error_files.sh` | Check for failed jobs |

## Data Layout

### Default Paths

**On Hyak**:
- Project root: `/gscratch/fang/NARSAD` or `/data/NARSAD`
- Workflows: `/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/`
- Container: `/gscratch/scrubbed/fanglab/xiaoqian/images/narsad.sif`

**Locally**:
- Project root: `/Users/xiaoqianxiao/projects/NARSAD` (configurable)

### Key Data Directories

```
NARSAD/
├── MRI/
│   ├── source_data/
│   │   └── behav/                    # Behavioral data
│   │       ├── single_trial_task-Narsad_phase2_half_events.csv
│   │       ├── single_trial_task-Narsad_phase3_half_events.csv
│   │       └── drug_order.csv
│   ├── derivatives/
│   │   ├── fmriprep/                  # Preprocessed data
│   │   └── fMRI_analysis/
│   │       └── LSS/
│   │           └── firstLevel/
│   │               ├── sub-*/          # Individual subject outputs
│   │               └── all_subjects/   # Group-level outputs
│   │                   ├── fear_network/
│   │                   ├── group_level/
│   │                   └── classification/
│   └── ROI/                           # ROI masks and atlases
│       ├── Gillian_anatomically_constrained/
│       ├── Glasser/
│       └── Tian/
```

### Output File Locations

**LSS outputs**:
- Trial-level: `MRI/derivatives/fMRI_analysis/LSS/firstLevel/sub-{subject}/task-{task}/trial-{trial_ID}/`
- Merged: `MRI/derivatives/fMRI_analysis/LSS/firstLevel/all_subjects/`
- Similarity: `MRI/derivatives/fMRI_analysis/LSS/firstLevel/all_subjects/`

**MVPA outputs**:
- Default: `MRI/derivatives/fMRI_analysis/LSS/results/mvpa_outputs/`
- Custom: Specified via `--output_dir`

**Group-level outputs**:
- Searchlight: `MRI/derivatives/fMRI_analysis/LSS/group_level/searchlight/`

## Running Analyses

### On Hyak Cluster

Most analyses are designed to run on the Hyak cluster using SLURM:

```bash
# Generate scripts
python create_1st_LSS_1st_singleTrialEstimate.py

# Launch jobs
./launch_1st_LSS_1st_singleTrialEstimate.sh

# Monitor jobs
squeue -u $USER

# Check logs
tail -f /gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/Lss/logs/*/*.out
```

### Locally

For local development and testing:

```bash
# Run MVPA analysis
python mvpa_L2_voxel_FearNetworkAll.py \
  --project_root /path/to/NARSAD \
  --output_dir /path/to/output

# Run single-trial LSS (if data available)
python run_1st_LSS.py --subject N101 --task phase2 --trial 1
```

**Note**: Update paths in scripts to match your local environment. Most scripts use environment variables or command-line arguments to specify paths.

## Troubleshooting

### Common Issues

**1. Path errors**:
- Verify `project_root` matches your environment
- Check that data directories exist
- Update paths in scripts if needed

**2. SLURM job failures**:
```bash
# Check for failed jobs
./check_error_files.sh

# Review specific job logs
tail -f /gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/Lss/logs/{task}/sub_{subject}_trial_{trial_ID}_*.err
```

**3. Missing dependencies**:
- Ensure FSL is loaded: `module load fsl`
- Check Python environment: `python --version`
- Verify container exists: `ls /gscratch/scrubbed/fanglab/xiaoqian/images/narsad.sif`

**4. Memory/timeout errors**:
- Increase SLURM resources: `--memory 64G --time 08:00:00`
- Reduce batch sizes: `--batch-size 500`
- Process fewer subjects at once

### Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# Check job history
sacct -u $USER --starttime=2024-01-01

# Check specific job output
tail -f /gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/Lss/logs/{task}/sub_{subject}_trial_{trial_ID}_*.out

# Cancel all your jobs
scancel -u $USER
```

### Getting Help

- Check script help: `python script.py --help`
- Review SLURM logs in `work_flows/Lss/logs/` directories
- Verify data paths match your environment
- Check that required input files exist

## Notes

- Many scripts assume `project_root` points to the dataset root; update per environment
- Hyak scripts generally use `/gscratch/fang/NARSAD` or `/data/NARSAD` as the root dataset path
- Local scripts may need path adjustments for your system
- Container paths are hardcoded for Hyak; update if using different container locations
- All scripts support `--dry-run` or `--help` for preview and documentation

## License

[Add license information if applicable]

## Citation

[Add citation information if applicable]

## Contact

- Author: Xiaoqian Xiao (xiao.xiaoqian.320@gmail.com)
- Repository: [Add repository URL if applicable]
