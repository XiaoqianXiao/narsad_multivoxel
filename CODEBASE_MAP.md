# Codebase Map

This document maps the current on-disk project layout for the NARSAD multivoxel analysis repository. It is intended as a navigation guide for future coding and analysis work, not as a complete scientific analysis plan. For scientific context, see `PROJECT_CONTEXT.md`. For the current post-Hyak MVPA L2 workflow, see `code/scripts/README_mvpa_l2_pipeline.md`.

## Top-Level Layout

- `README.md`: broad pipeline overview for LSS, MVPA, group-level statistics, and Hyak/local execution.
- `PROJECT_CONTEXT.md`: compact current scientific and reproducibility context.
- `mvpa_L2.md`: detailed MVPA L2 analysis plan, but currently contains merge-conflict markers; do not treat it as canonical until resolved.
- `pyproject.toml` and `poetry.lock`: Python environment metadata. The project targets Python `>=3.11,<3.14` and uses scientific Python, neuroimaging, ML, plotting, and stats libraries.
- `dockerfile`: container/environment support.
- `AI_RULES.md`: local AI-assistant guidance, if maintained by the project.
- `code/`: active analysis scripts, notebooks, Hyak jobs, and post-Hyak statistical workflow.
- `old_version_md/`: older exported guides and notebook markdown retained for reference.
- `results/`: local or synchronized outputs, especially SCR outputs and MVPA L2 harmonized results.
- `.idea/`, `.ipython_tmp/`, `.nbconvert_out/`, `__pycache__/`: local IDE, notebook, conversion, and Python cache artifacts.

## Active Code Directory

`code/` is the active working layer. It contains first-level LSS scripts, feature-preparation scripts, MVPA notebooks, visualization notebooks, cached joblib outputs, and cluster wrappers.

### First-Level LSS Generation And Execution

These scripts generate and run subject/trial-level LSS jobs.

- `code/create_1st_LSS_1st_singleTrialEstimate.py`: generates SLURM scripts for single-trial LSS first-level GLMs.
- `code/run_1st_LSS.py`: single-trial runner that calls `first_level_wf_LSS`.
- `code/first_level_workflows.py`: Nipype/FSL first-level workflow definitions and BIDS wiring.
- `code/launch_1st_LSS_1st_singleTrialEstimate.sh`: launcher for generated first-level LSS jobs.

### LSS Trial Merging

These scripts merge individual trial outputs into subject/task-level 4D NIfTI files.

- `code/create_1st_LSS_2nd_cateAlltrials.py`: generates SLURM scripts for LSS trial concatenation.
- `code/first_LSS_2nd_cateAlltrials.py`: merges individual trial cope files.
- `code/launch_1st_LSS_2nd.sh`: legacy or alternate Step 2 launcher.
- `code/launch_1st_LSS_2nd_cateAlltrials.sh`: current Step 2 category/trial merge launcher.

### Similarity And Searchlight Preparation

These scripts compute subject-level similarity/RSA-style outputs from merged LSS data.

- `code/create_1st_LSS_3rd_similarity.py`: generates SLURM scripts for similarity analyses.
- `code/first_LSS_3rd_similarity.py`: computes searchlight and ROI similarity outputs.
- `code/launch_1st_LSS_3rd_similarity.sh`: launcher for Step 3 similarity jobs.
- `code/similarity.py`: helper functions for similarity calculations.

### ROI Classification

These scripts run ROI-level MVPA/classification from single-trial LSS estimates.

- `code/create_1st_LSS_4th_classification.py`: generates SLURM scripts for ROI classification.
- `code/first_LSS_4th_classification.py`: subject-level ROI MVPA/classification script.

### Feature Matrix Preparation

These scripts build group-level `X`, `y`, and subject-vector inputs for downstream MVPA.

- `code/prepare_X_y_voxel_FearNetwork.py`: voxel-wise features from anatomically constrained FearNetwork ROIs.
- `code/prepare_X_y_voxel_MemoryFearNetwork.py`: voxel-wise features from MemoryFearNetwork ROIs.
- `code/prepare_X_y_voxel_WholeBrain.py`: whole-brain voxel-wise extraction and atlas grid handling.
- `code/prepare_X_y_ROI_WholeBrain.py`: whole-brain parcellation features using cortical/subcortical atlas labels.

### Group-Level Searchlight

These scripts support group-level searchlight statistics on LSS similarity maps.

- `code/create_group_LSS_searchlight.py`: generates group-level searchlight SLURM scripts.
- `code/group_LSS_searchlight.py`: runs group-level FLAMEO or Randomise-style inference.
- `code/group_level_workflows.py`: Nipype group-level workflow helpers.
- `code/launch_group_LSS.sh`: launcher for group-level LSS jobs.

### Shared Utilities And Miscellaneous Scripts

- `code/utils.py`: shared utility functions used by LSS/Nipype workflow scripts.
- `code/other_tools.py`: miscellaneous helper code.
- `code/check_error_files.sh`: shell helper for checking failed or error-producing jobs.
- `code/run_mvpa.sh`: shell wrapper for MVPA execution.

## Post-Hyak MVPA L2 Layer

`code/scripts/` is the lightweight downstream layer that reads expensive Hyak caches, harmonizes metrics, runs manuscript-facing statistics, and exports summaries.

- `code/scripts/README_mvpa_l2_pipeline.md`: best current operational guide for this layer.
- `code/scripts/mvpa_l2_common.py`: shared constants, metric names, harmonization helpers, model helpers, FDR utilities, and CSV output helpers.
- `code/scripts/build_scr_sensitivity_groups.py`: converts SCR outputs into subject-level responder/learner flags for MVPA sensitivity analyses.
- `code/scripts/export_mvpa_l2_metrics.py`: reads Hyak joblib caches and writes harmonized subject-level neural metrics.
- `code/scripts/run_mvpa_l2_primary_models.py`: runs Aim 2-5 primary statistical models from the harmonized subject table.
- `code/scripts/run_mvpa_l2_sensitivity_models.py`: runs feature-space and SCR-cohort sensitivity models.
- `code/scripts/export_mvpa_l2_manuscript_artifacts.py`: exports the manuscript-ready primary results table, Aim 4 convergence matrix, and reproducibility/QC dashboard.
- `code/scripts/summarize_mvpa_l2_results.py`: writes a compact markdown summary of MVPA L2 model outputs.
- `code/scripts/export_aim1_scr_sensitivity.py`: exports labeled Aim 1 SCR-subgroup sensitivity checkpoints to CSV.
- `code/scripts/export_haufe_scr_sensitivity.py`: exports SCR-subgroup Haufe spatial-pattern stability checks.
- `code/scripts/run_mvpa_l2_posthyak.sh`: shell orchestrator for the post-Hyak harmonization/statistics workflow.

The downstream workflow expects feature-space results from Hyak MVPA jobs and writes outputs such as harmonized subject metrics, Aim 2-5 CSVs, sensitivity-model CSVs, and a markdown summary.

## Hyak Cluster Layer

`code/hyak/` contains cluster-oriented scripts for expensive feature-space MVPA, whole-brain searchlight analyses, chunk merging, and inference.

### Feature-Space MVPA Jobs

- `code/hyak/mvpa_L2_voxel_FearNetwork.py`: Hyak execution script for the primary FearNetwork voxel-wise MVPA workflow.
- `code/hyak/mvpa_L2_voxel_MemoryFearNetwork.py`: MemoryFearNetwork sensitivity feature-space workflow.
- `code/hyak/mvpa_L2_voxel_WholeBrain_Parcellation.py`: whole-brain/parcellation feature-space workflow.
- `code/hyak/mvpa_L2_voxel_WholeBrain_Schaefer.py`: Schaefer/Tian whole-brain feature-space workflow.

### MVPA Submission Wrappers

- `code/hyak/submit_mvpa_L2_needed_no_wholebrain.sh`: submits the currently executable FearNetwork, MemoryFearNetwork, and post-Hyak chain without whole-brain/parcellation sensitivity.
- `code/hyak/submit_mvpa_L2_fearnetwork_stage.sh`: submits the FearNetwork stage.
- `code/hyak/submit_mvpa_L2_memoryfearnetwork_stage.sh`: submits the MemoryFearNetwork stage.
- `code/hyak/submit_mvpa_L2_schaefer_stage.sh`: submits the Schaefer/Tian sensitivity stage.
- `code/hyak/submit_mvpa_L2_posthyak.sh`: submits the downstream post-Hyak harmonization/statistics job.
- `code/hyak/submit_mvpa_L2_aim1_scr_sensitivity.sh`: submits Aim 1 SCR-subgroup sensitivity jobs.
- `code/hyak/submit_export_aim1_scr_sensitivity.sh`: submits export of Aim 1 SCR sensitivity outputs.

### Whole-Brain Searchlight Jobs

- `code/hyak/mvpa_searchlight_wholeBrain_ext.py`: extinction-phase within/between similarity searchlight.
- `code/hyak/mvpa_searchlight_wholeBrain_rst.py`: reinstatement-phase within/between similarity searchlight.
- `code/hyak/mvpa_searchlight_wholeBrain_dyn_ext.py`: dynamic extinction searchlight.
- `code/hyak/mvpa_searchlight_wholeBrain_dyn_rst.py`: dynamic reinstatement searchlight.
- `code/hyak/mvpa_searchlight_wholeBrain_crossphase.py`: extinction-to-reinstatement cross-phase searchlight.

### Searchlight Submission, Merge, And Recovery

- `code/hyak/submit_searchlight_stageA.sh`: submits base searchlight stage A.
- `code/hyak/submit_searchlight_merge_stageB.sh`: submits merge stage B.
- `code/hyak/submit_searchlight_tfce_stageC.sh`: submits TFCE or cluster-inference stage C.
- `code/hyak/submit_searchlight_crosshalf_stageA2.sh`: submits cross-half stage A2.
- `code/hyak/submit_merge_crosshalf_stageB2.sh`: submits cross-half merge stage B2.
- `code/hyak/submit_searchlight_crosshalf_stageC2.sh`: submits cross-half inference stage C2.
- `code/hyak/merge_searchlight_chunks.py`: merges chunked searchlight outputs into whole-brain maps and summary CSVs.
- `code/hyak/merge_crosshalf_chunks.py`: merges chunked cross-half subject maps.
- `code/hyak/merge_trial_npz_chunks.py`: merges chunked trial-level `.npz` outputs.
- `code/hyak/run_merge_chunks.sh`: shell wrapper for chunk merging.
- `code/hyak/resubmit_failed_searchlight.sh`: helper for resubmitting failed searchlight jobs.
- `code/hyak/resubmit_merge_dyn.sh`, `code/hyak/resubmit_tfce_dyn.sh`, `code/hyak/resubmit_tfce_rst.sh`: recovery wrappers for merge/TFCE stages.

### Group Inference

- `code/hyak/cluster_inference_ext.py`: cluster-level FWE inference for extinction searchlight maps.
- `code/hyak/cluster_inference_rst.py`: cluster-level FWE inference for reinstatement searchlight maps.
- `code/hyak/cluster_inference_dyn_ext.py`: cluster-level FWE inference for dynamic extinction.
- `code/hyak/cluster_inference_dyn_rst.py`: cluster-level FWE inference for dynamic reinstatement.
- `code/hyak/cluster_inference_crossphase.py`: cluster-level FWE inference for cross-phase searchlight maps.
- `code/hyak/fdr_inference_ext.py`: FDR correction for extinction voxel-wise p-maps.
- `code/hyak/fdr_inference_rst.py`: FDR correction for reinstatement voxel-wise p-maps.
- `code/hyak/fdr_inference_dyn_ext.py`: FDR correction for dynamic extinction p-maps.
- `code/hyak/fdr_inference_dyn_rst.py`: FDR correction for dynamic reinstatement p-maps.
- `code/hyak/fdr_inference_crossphase.py`: FDR correction for cross-phase p-maps.
- `code/hyak/flameo_inference_ext.py`: FLAMEO mixed-effects group inference for extinction maps.
- `code/hyak/flameo_nipype_ext.py`: Nipype FLAMEO variant for extinction maps.
- `code/hyak/submit_cluster_inference.sh`: submits cluster inference jobs.
- `code/hyak/submit_fdr_inference.sh`: submits FDR inference jobs.

### Hyak Documentation

- `code/hyak/README_searchlight_workflow.md`: searchlight workflow notes.

## Notebooks

Notebooks in `code/` appear to be analysis, visualization, and SCR derivation workbooks. Treat them as analysis artifacts unless a current workflow explicitly depends on them.

- SCR and cohort derivation:
  - `code/analysis_scr.ipynb`.
  - `code/identify_fear_learning_subjects_scr.ipynb`.
- MVPA L2 development and feature-space notebooks:
  - `code/mvpa_l2.ipynb`.
  - `code/mvpa_L2_voxel_FearNetworkAll.ipynb`.
  - `code/mvpa_L2_voxel_FearNetworkAll_ori.ipynb`.
  - `code/mvpa_L2_voxel_MemoryFearNetwork.ipynb`.
  - `code/mvpa_L2_voxel_WholeBrain_Parcellation.ipynb`.
  - `code/mvpa_L2_voxel_WholeBrain_Parcellation_schaefer.ipynb`.
- Visualization notebooks:
  - `code/visualize_mvpa_L2_fearnetwork_outputs.ipynb`.
  - `code/visualize_mvpa_L2_fearnetwork_outputs_FDR.ipynb`.
  - `code/visualize_mvpa_L2_fearnetwork_outputs_Haufe.ipynb`.
  - `code/visualize_mvpa_L2_fearnetwork_outputs_pLess0.01.ipynb`.
  - `code/visualize_mvpa_L2_memoryfearnetwork_outputs.ipynb`.
  - `code/visualize_mvpa_L2_memoryfearnetwork_outputs_pLess0.01.ipynb`.
  - `code/visualize_mvpa_L2_schaefer_outputs.ipynb`.
  - `code/visualize_wholebrain_schaefer.ipynb`.
  - `code/visualize_wholebrain_schaefer_inter.ipynb`.
  - `code/searchlight_merged_visualization.ipynb`.
  - `code/searchlight_merged_visualization_sig_only.ipynb`.
  - `code/read_lss_data.ipynb`.

## Results And Cached Artifacts

### Local Results

- `results/scr_analysis_outputs/`: SCR-derived tables, learner/responder subject lists, trialwise plots, group-difference summaries, and publication figures.
- `results/outputs/mvpa_l2/harmonized/`: local or synchronized harmonized MVPA L2 outputs, including `scr_sensitivity_groups.csv`.

### Cached Joblib Files

`code/` contains several `.joblib` files such as permutation results and analysis checkpoints. These are analysis artifacts rather than source code. They can be useful for inspection or exports, but should not be edited manually.

Examples include:

- `code/perm_results_HC_2way.joblib`.
- `code/perm_results_SAD_2way.joblib`.
- `code/perm_results_HC_fear_network_2way.joblib`.
- `code/perm_results_SAD_fear_network_2way.joblib`.
- `code/results_analysis_11.joblib`.

## Data And Output Flow

The high-level data flow is:

1. BIDS/fMRIPrep data and behavioral events feed first-level LSS workflows.
2. Single-trial LSS outputs are merged into subject/task 4D images.
3. Similarity, ROI classification, and feature-preparation scripts derive analysis matrices or maps.
4. Hyak MVPA scripts run expensive feature-space decoding, topology, trajectory, decision-boundary, Haufe, and sensitivity stages.
5. `code/scripts/export_mvpa_l2_metrics.py` harmonizes Hyak caches into subject-level CSVs.
6. `code/scripts/run_mvpa_l2_primary_models.py` and `code/scripts/run_mvpa_l2_sensitivity_models.py` run manuscript-facing statistical models.
7. `code/scripts/summarize_mvpa_l2_results.py` creates compact result summaries.
8. Visualization notebooks and export scripts produce figures and sensitivity readouts.

## Environment Notes

The Poetry environment declares:

- Python `>=3.11,<3.14`.
- Core scientific stack: `numpy`, `pandas`, `scipy`, `statsmodels`, `joblib`.
- ML stack: `scikit-learn`.
- Neuroimaging stack: `nibabel`, `nilearn`.
- Plotting stack: `matplotlib`, `seaborn`, `plotly`.
- Development tools: `jupyter`, `ipykernel`, `pytest`, `black`, `flake8`, `mypy`.

Some scripts also rely on external neuroimaging or cluster tools such as FSL, Nipype, BIDS/pybids, SLURM, and Apptainer/Singularity, as described in `README.md` and Hyak wrappers.

## Current Worktree Cautions

- Many files are currently untracked under `code/`, `old_version_md/`, `results/`, and `PROJECT_CONTEXT.md`.
- Many former root-level scripts appear as deleted in `git status`, while corresponding active copies exist under `code/`.
- `mvpa_L2.md` is modified and contains visible merge-conflict markers.
- Generated caches, notebooks, and result files are mixed with source scripts. Be careful when staging or committing.
- Use `rg --files` or targeted `find` scans before editing, because the repo contains local artifacts, historical markdown exports, notebook outputs, and cache files.

## Common Entry Points

For a first-level LSS rerun:

```bash
cd code
python create_1st_LSS_1st_singleTrialEstimate.py
bash launch_1st_LSS_1st_singleTrialEstimate.sh
```

For feature extraction after LSS outputs exist:

```bash
cd code
python prepare_X_y_voxel_FearNetwork.py
python prepare_X_y_voxel_MemoryFearNetwork.py
python prepare_X_y_voxel_WholeBrain.py
python prepare_X_y_ROI_WholeBrain.py
```

For the current post-Hyak MVPA L2 pipeline:

```bash
cd code
bash scripts/run_mvpa_l2_posthyak.sh
```

For the currently documented Hyak chain without whole-brain/parcellation sensitivity:

```bash
cd code
bash hyak/submit_mvpa_L2_needed_no_wholebrain.sh
```

Check `code/scripts/README_mvpa_l2_pipeline.md` before running cluster or post-Hyak commands, because it contains the most specific current path and dependency assumptions.
