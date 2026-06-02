# NARSAD MVPA Project Guide

This repository supports a multivoxel fMRI analysis of vicarious safety/threat learning in social anxiety disorder (SAD) and healthy controls (HC), with placebo and oxytocin drug conditions. The project has two linked goals:

1. Identify neural patterns that discriminate vicarious safety from threat learning.
2. Characterize those patterns with representational topology, dynamic drift, decision-boundary geometry, SCR coupling, and clinical associations.

The guide below documents the knowledge required, the analysis logic, the input/output paths, and the rerun strategy. It is intended to help future analysis changes remain scientifically interpretable and computationally reproducible.

## Study Logic

The primary construct is the neural representation of vicarious learning, operationalized mainly as the distinction between:

- `CSS`: conditioned stimulus associated with observed safety learning
- `CSR`: conditioned stimulus associated with observed threat learning
- `CS-`: background/safe reference condition

The core scientific sequence is:

1. **Identify** a decodable neural representation of vicarious safety vs threat learning.
2. **Localize/interpret** the learned pattern using Haufe-transformed maps and permutation-importance masks.
3. **Characterize** the learned representation using independent descriptive metrics: topology, drift, trajectories, and decision-boundary geometry.
4. **Relate** neural indices to physiology and symptoms using SCR and clinical measures.

The project should avoid treating a classifier result as the endpoint. Decoding establishes that information is present; the later analyses explain what the representation looks like and how it relates to SAD-relevant learning mechanisms.

## Analysis Hierarchy

### Primary Analyses

Primary analyses should be treated as confirmatory or near-confirmatory when writing results:

- Analysis 1.1: within-group neural dissociation, especially `CSS` vs `CSR`.
- Analysis 1.2: static representational topology in the important-feature space.
- Analysis 1.3: dynamic representational drift.
- Analysis 1.3 part 2: single-trial safety and threat trajectories.
- Analysis 1.4: decision-boundary and uncertainty measures.

### Secondary Analyses

Secondary analyses extend interpretation and should be reported as mechanistic or translational follow-up:

- Safety restoration and threat discrimination.
- Drift efficiency.
- Probabilistic opening / decision probability extraction.
- Spatial re-alignment.
- Reverse cross-decoding.
- SCR-neural coupling.
- Clinical-neural correlations and covariate-adjusted partial correlations.

### Sensitivity Analyses

Sensitivity analyses should be clearly labeled and not mixed with the primary feature-space claims:

- `p < .01` permutation-importance masks.
- FDR-corrected masks.
- Haufe-map-derived masks.
- all-positive permutation-importance fallback when the primary significant mask selects too few features.
- shock/US target metrics as an alternative threat target.

## Required Expertise

### Neuroscience And Clinical Knowledge

- Social anxiety disorder and clinically relevant threat/safety learning mechanisms.
- Vicarious/social learning task structure, including extinction, reinstatement, and shock/US events.
- Interpretation of `CS-`, `CSS`, `CSR`, and shock/US target patterns.
- SCR physiology and the difference between anticipatory SCR and US-evoked response.
- Clinical symptom measures: LSAS, DASS, ECR, and relevant covariates such as age, gender, group, and drug.

### fMRI And Neuroimaging Knowledge

- BIDS/fMRIPrep-style project organization.
- LSS single-trial beta estimation and its assumptions.
- ROI voxel extraction, atlas/parcellation workflows, and vector-to-brain reconstruction.
- FearNetwork, MemoryFearNetwork, and Schaefer/Tian whole-brain feature spaces.
- Haufe transformation and the distinction between classifier weights and interpretable activation patterns.
- Glass-brain visualization, thresholded maps, and reporting of ROI voxel coverage.

### Machine Learning Knowledge

- Linear SVM-style decoding pipelines with scaling and hyperparameter tuning.
- Subject-aware cross-validation to prevent trial leakage.
- Pairwise decoding and cross-phase/cross-group generalization.
- Permutation testing for decoding accuracy.
- Cross-validated permutation importance.
- Group-specific feature masks and the consequences of unequal SAD/HC feature spaces.
- Proper interpretation of prediction accuracy versus representational structure.

### Representational Analysis Knowledge

- RSA/RDM logic.
- Crossnobis/Mahalanobis distances with shrinkage covariance.
- Split-half reliability logic for condition centroids.
- Static topology metrics:
  - threat distance: `dist(CSR, CSS)`
  - safety distance: `dist(CSS, CS-)`
  - threat bias: `dist(CSR, CS-) - dist(CSS, CS-)`
  - safety integration: `dist(CSR, CSS) - dist(CSS, CS-)`
- Dynamic drift metrics:
  - projection magnitude
  - cosine fidelity
  - initial distance
- Decision-boundary measures:
  - entropy
  - variance
  - kurtosis
  - boundary separation
  - decision margins
  - `P_CSR_CSS`
  - `P_CSR_CSR`

### Statistical Knowledge

- Permutation t-tests and one-sample tests against zero.
- Multiple-comparison correction, including ROI-FDR and whole-brain FDR.
- Mixed-effects models for repeated trial-level data.
- Interaction interpretation for Group, Drug, Domain, and Trial.
- Partial correlations and covariate adjustment.
- Z-scoring and outlier handling.
- Distinguishing primary outcomes from exploratory analyses.
- Avoiding overinterpretation of non-significant effects.

### Engineering And Reproducibility Skills

- Python scientific stack: `numpy`, `pandas`, `scipy`, `scikit-learn`, `statsmodels`, `joblib`, `matplotlib`, `seaborn`, `nibabel`, and `nilearn`.
- Large-array memory management.
- Checkpoint/resume design.
- Notebook validation and reproducible visualization.
- SLURM job arrays, dependencies, merge jobs, and failure recovery.
- Apptainer/Singularity path mapping.
- Disk quota management on Hyak.

## Analysis Workflow

### Input Flow

1. First-level LSS estimates produce trial-wise neural beta maps.
2. Feature-preparation scripts convert trial-wise maps into matrices:
   - `X`: neural features
   - `y`: trial labels
   - `subjects`: subject IDs
   - ROI or parcel metadata when applicable
3. L2 scripts split data by phase, group, drug, and feature space.
4. Analysis stages save checkpoints and intermediate `.joblib` files for downstream reuse and notebook visualization.

### Analysis 1.1: Neural Dissociation

Purpose: test whether the neural feature space contains information that distinguishes vicarious safety from threat learning.

Core procedure:

- Train group-specific classifiers for SAD and HC.
- Use subject-aware cross-validation.
- Estimate self-decoding accuracy and permutation null distributions.
- Test functional specificity across SAD/HC train-test combinations.
- Test spatial specificity between SAD and HC maps.
- Refit final models for Haufe transformation and map visualization.

Primary output:

- CV accuracy and null distribution.
- permutation p-value.
- best model settings.
- refit model.
- Haufe/Z maps.
- functional specificity matrix.
- spatial specificity matrix.

Interpretation:

- Significant decoding means information is present in the feature space.
- It does not, by itself, explain the representational geometry or mechanism.

### Stage 11: Empirical Feature-Importance Masks

Purpose: define the feature space used for characterization analyses 1.2 to 1.4.

Core procedure:

- Compute cross-validated permutation importance from Analysis 1.1 models.
- Compare observed importance with null importance.
- Save group-specific masks for SAD and HC.
- Use empirical p-thresholded positive-importance masks for ROI pipelines.
- Use whole-brain FDR masks for Schaefer/Tian, with a prespecified all-positive fallback when too few features survive.

Reporting requirement:

- Always report the number of selected features per group.
- If fallback is used, state the fallback rule and feature count.
- Do not describe fallback masks as primary FDR-significant findings.

### Analysis 1.2: Static Representational Topology

Purpose: describe the geometry of the neural learning space.

Core procedure:

- Use Stage 11 important-feature masks.
- Compute subject-level crossnobis RDMs for `CS-`, `CSS`, and `CSR`.
- Save raw, z-scored, and per-voxel-normalized RDMs when available.
- Add shock-inclusive RDMs with `CS-`, `CSS`, `CSR`, and `Shock` when shock/US trials are available.

Primary topology metrics:

- `dist(CSR, CSS)`: threat-vs-safety separation.
- `dist(CSS, CS-)`: safety-vs-background distance.
- `dist(CSR, CS-)`: threat-vs-background distance.
- safety integration: `dist(CSR, CSS) - dist(CSS, CS-)`.
- threat bias: `dist(CSR, CS-) - dist(CSS, CS-)`.

Shock-inclusive metrics:

- `dist(Shock, CS-)`
- `dist(Shock, CSS)`
- `dist(Shock, CSR)`

Interpretation:

- The 3-condition RDM is the primary topology analysis.
- The shock-inclusive RDM is a sensitivity/extension that asks whether threat representations align more closely with actual aversive/shock patterns.

### Analysis 1.3: Dynamic Representational Drift

Purpose: quantify movement from learning-state patterns toward target-state patterns.

Core procedure:

- Use Stage 11 important-feature masks.
- Compute subject-level vectors and drift metrics.
- Safety target: `CSS -> CS-`.
- Threat maintenance target: early/extinction `CSR -> reinstated CSR`.
- Alternative threat target: early/extinction `CSR -> shock/US`.

Metrics:

- projection magnitude: movement along target direction.
- cosine fidelity: alignment with target direction.
- initial distance: distance from target before drift.

Interpretation:

- Safety drift tests restoration toward a safe/background representation.
- Reinstated-CSR drift tests threat maintenance.
- Shock-target drift tests alignment with actual aversive/shock representation.

### Analysis 1.3 Part 2: Single-Trial Trajectories

Purpose: visualize and test trial-wise dynamics.

Core procedure:

- Compute trial-wise start-to-target similarity scores.
- Plot safety and threat trajectories separately or with clearly separated panels.
- Include the second threat metric, shock/US target, as a distinct trajectory panel.
- Annotate trial-wise group differences when significant.

Important interpretation rule:

- Safety and threat trajectory scores are not the same psychological measure because their targets differ.
- The y-axis should be labeled in terms of start-to-target similarity, not generic improvement.

### Analysis 1.4: Decision Boundary And Uncertainty

Purpose: characterize the classifier boundary and class evidence beyond accuracy.

Core procedure:

- Use Analysis 1.1 model logic in the important-feature space.
- Extract probabilities and decision margins.
- Compute uncertainty/distributional measures at subject level.

Key outputs:

- entropy
- variance
- kurtosis
- boundary separation
- `Decision_Margin_CSS`
- `Decision_Margin_All`
- `P_CSR_CSS`
- `P_CSR_CSR`

Interpretation:

- Entropy and variance describe uncertainty.
- Margins describe boundary distance.
- `P_CSR_CSS` and `P_CSR_CSR` describe threat-class evidence for safety and threat trials.

### Analyses 2.1 To 2.5

These are downstream characterization analyses and should be interpreted as extensions of the primary representational findings.

Analysis 2.1: Safety restoration and threat discrimination.

- Tests whether `CSS` moves closer to `CS-` and whether `CSR` remains discriminable.
- Uses mixed-effects models for group/drug comparisons.

Analysis 2.2: Drift efficiency.

- Tests whether safety and threat drift differ by Group, Drug, and Domain.
- Includes both reinstated-CSR and shock-target threat domains when available.

Analysis 2.3: Probabilistic opening.

- Tests decision-probability and margin measures.
- Should use the same measure set as Analysis 1.4 when possible.

Analysis 2.4: Spatial re-alignment.

- Tests whether SAD and HC maps align spatially across group/drug contexts.

Analysis 2.5: Reverse cross-decoding.

- Tests representational generalization in the reverse training/testing direction.

### Clinical And SCR Analyses

Purpose: connect neural indices with clinical symptoms and psychophysiology.

Core procedure:

- Merge neural indices with LSAS, DASS, ECR, and SCR summaries.
- Run group-wise Pearson correlations.
- Run covariate-adjusted partial correlations.
- Z-score neural, clinical, and covariate variables for regression plots.
- Use mixed-effects models for trial-wise SCR because repeated trials are nested within subjects.

Recommended trial-wise SCR model:

```text
z(SCR) ~ z(neural trajectory score) * Group * Domain + Drug + Trial_Z + subject random effects
```

Placebo-only follow-up models:

```text
z(SCR) ~ z(neural trajectory score) * Group + Trial_Z + subject random effects
```

Interpretation:

- Subject-level correlations test symptom relevance.
- Trial-wise SCR models test whether neural dynamics track physiological arousal within subject.
- Clinical/SCR analyses should be reported as convergent validity unless they were prespecified as primary endpoints.

## Input And Output Path Structure

The project uses three path layers:

- **Hyak host paths**: real filesystem paths such as `/gscratch/fang/NARSAD`.
- **Container paths**: paths visible inside Apptainer, especially `/app` and `/output_dir`.
- **Local notebook paths**: local copies of Hyak outputs used for visualization.

### Core Hyak Roots

```text
PROJECT_ROOT=/gscratch/fang/NARSAD
APP_PATH=/gscratch/scrubbed/fanglab/xiaoqian/repo/narsad_multivoxel/hyak
OUT_BASE=/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/LSS/results
CONTAINER_SIF=/gscratch/fang/images/jupyter.sif
```

Container binding convention:

```text
PROJECT_ROOT -> PROJECT_ROOT
APP_PATH     -> /app
OUT_BASE     -> /output_dir
```

Thus:

```text
/output_dir/FearNetwork
```

inside the container corresponds to:

```text
/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/LSS/results/FearNetwork
```

on Hyak.

### L2 Output Roots

FearNetwork:

```text
/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/LSS/results/FearNetwork
/gscratch/fang/NARSAD/ROI/Gillian_anatomically_constrained
/gscratch/fang/NARSAD/logs/mvpa_l2_fearnetwork
```

MemoryFearNetwork:

```text
/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/LSS/results/MemoryFearNetwork
/gscratch/fang/NARSAD/ROI/MemoryFearNetwork
/gscratch/fang/NARSAD/logs/mvpa_l2_memoryfearnetwork
```

WholeBrain Schaefer/Tian:

```text
/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/LSS/results/wholebrain_parcellation_schaefer
/gscratch/fang/NARSAD/MRI/derivatives/fMRI_analysis/LSS/firstLevel/all_subjects/group_level
/gscratch/fang/NARSAD/logs/mvpa_l2_schaefer
```

### Output Subdirectories

Each L2 output root contains:

```text
checkpoints/
intermediate/
root-level .joblib and .png outputs
```

Use `checkpoints/` for exact resume state and notebook loading. Use `intermediate/` for named stage bundles and reusable analysis payloads.

Common checkpoint files:

```text
cell_06.joblib
cell_11_SAD.joblib
cell_11_HC.joblib
analysis_12_topology.joblib
cell_12_trajectories.joblib
cell_13_decision_stats_opt.joblib
cell_16_opening_test.joblib
cell_17_realignment.joblib
cell_18_reverse_cross_decoding.joblib
```

Common intermediate files:

```text
stage11_importance_masks.joblib
stage11_importance_masks_SAD.joblib
stage11_importance_masks_HC.joblib
stage12_topology_stats.joblib
```

### Feature-Matrix Inputs

LSS first-level outputs:

```text
/gscratch/fang/NARSAD/MRI/derivatives/fMRI_analysis/LSS/firstLevel
```

Group-level NPZ inputs:

```text
/gscratch/fang/NARSAD/MRI/derivatives/fMRI_analysis/LSS/firstLevel/all_subjects/group_level
```

FearNetwork:

```text
phase2_X_ext_y_ext_roi_voxels.npz
phase3_X_reinst_y_reinst_roi_voxels.npz
```

MemoryFearNetwork:

```text
phase2_X_ext_y_ext_roi_voxels_MemoryFearNetwork.npz
phase3_X_reinst_y_reinst_roi_voxels_MemoryFearNetwork.npz
```

Schaefer/Tian:

```text
phase2_X_ext_y_ext_voxels_schaefer_tian.npz
phase3_X_reinst_y_reinst_voxels_schaefer_tian.npz
```

Expected NPZ fields:

```text
X_ext or X_reinst
y_ext or y_reinst
subjects
roi_names / roi_voxel_counts for ROI-voxel pipelines
parcel_names for Schaefer/Tian pipelines
```

### Behavioral, Clinical, And SCR Inputs

Drug/group metadata:

```text
/gscratch/fang/NARSAD/MRI/source_data/behav/drug_order.csv
```

Clinical default directory:

```text
/gscratch/fang/NARSAD/MRI/source_data/behav
```

Clinical file patterns:

```text
SocialSafetyLearning-LSASSubtotals_DATA_*.csv
SocialSafetyLearning-ECR_DATA_*.csv
SocialSafetyLearning-DASS_DATA_*.csv
```

Useful overrides:

```text
CLINICAL_DIR
CLINICAL_LSASSUBTOTALS_PATH
CLINICAL_ECR_PATH
CLINICAL_DASS_PATH
```

Trial-wise SCR default:

```text
/gscratch/fang/NARSAD/EDR/peak_stats_table-phase2.3.csv
```

SCR overrides:

```text
TRIAL_SCR_PATH
SCR_TRIAL_PATH
```

### Local Notebook Roots

Common local result mirrors:

```text
/Users/xiaoqianxiao/projects/NARSAD/LSS/results
/Users/xiaoqianxiao/projects/NARSAD/MRI/derivatives/fMRI_analysis/LSS/results
```

Pipeline subfolders should mirror Hyak:

```text
FearNetwork
MemoryFearNetwork
wholebrain_parcellation_schaefer
```

If a notebook figure is missing or stale, first verify that the local `.joblib` files were synced from the matching Hyak output root.

### Searchlight Outputs

Searchlight output root:

```text
/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/LSS/searchlight
```

Main subfolders:

```text
ext
rst
dyn_ext
dyn_rst
crossphase
```

Common searchlight outputs:

```text
<OUT_BASE>/<mode>/merged
<OUT_BASE>/<mode>/crosshalf_subj_maps
<OUT_BASE>/<mode>/crosshalf_permutation
*_mean.nii.gz
*_p.nii.gz
*_q.nii.gz
*_summary_contrasts.csv
*_sig_merged.csv
```

## Rerun Strategy

Use targeted reruns. Avoid recomputing permutation-heavy stages unless the upstream object they depend on has changed.

### If Analysis 1.1 Changes

Rerun:

- Analysis 1.1 model stages.
- Haufe/permutation-map stages.
- Stage 11 feature-importance masks.
- Analyses 1.2, 1.3, 1.3 part 2, and 1.4.
- downstream analyses that consume these outputs.

Reason:

- The downstream feature masks and decision-boundary metrics depend on the model learned in Analysis 1.1.

### If Feature-Mask Logic Changes

Rerun:

- Stage 11 if the mask definition changes.
- Analyses 1.2, 1.3, 1.3 part 2, and 1.4.
- Analyses 2.1, 2.2, and 2.3 if they consume the changed outputs.

Usually do not rerun:

- first-level LSS.
- feature matrix preparation.
- Analysis 1.1 model training, unless the model itself changed.

### If A New Derived Metric Is Added

Rerun only the stage that computes or saves that metric and any dependent downstream stages.

Examples:

- shock-inclusive RDM: rerun Analysis 1.2.
- shock-target trajectory: rerun Analysis 1.3, Analysis 1.3 part 2, and Drift Efficiency.
- new decision-probability metric: rerun Analysis 1.4 and Probabilistic Opening.

Do not delete unrelated checkpoints unless a stale cache prevents recomputation.

### If Only Notebook Visualization Changes

Usually no `.py` rerun is needed.

Rerun `.py` stages only when:

- the required data field is absent from saved `.joblib` files.
- the notebook needs a metric that was not previously computed.
- a sensitivity notebook must recompute downstream results from a different active mask.

### If A Stage 11 Chunk Fails

Recommended recovery:

1. Use `sacct` to identify the failed array element.
2. Rerun only the failed chunk with `STAGE11_CHUNK_IDX`.
3. Re-run the merge job for that group.
4. Continue downstream stages after both SAD and HC merged outputs exist.

Do not rerun all chunks unless multiple chunks failed or the chunk configuration changed.

## Reporting And Interpretation Guardrails

- State whether each result is primary, secondary, or sensitivity.
- Report feature counts for SAD and HC in every important-feature analysis.
- Report when fallback masks are used and avoid describing fallback voxels as FDR-significant.
- Keep the distinction between Haufe-significant voxels and permutation-importance voxels clear.
- Do not infer neural activation from raw classifier weights.
- Do not interpret decoding accuracy as a complete explanation of representation.
- For trial-wise trajectory plots, state the target pattern explicitly.
- For shock-target metrics, frame them as an alternative threat target unless prespecified as primary.
- For SCR and clinical models, report sample size after merging and missing-data exclusions.
- For mixed-effects models, report fixed effects, random-effects structure, and the meaning of each interaction term.
- For multiple tests, specify the correction family.

## Project Conventions To Preserve

- Keep the three L2 scripts aligned when analyses are conceptually identical:
  - `hyak/mvpa_L2_voxel_FearNetwork.py`
  - `hyak/mvpa_L2_voxel_MemoryFearNetwork.py`
  - `hyak/mvpa_L2_voxel_WholeBrain_Schaefer.py`
- Preserve the distinction between ROI/FearNetwork-style FDR and whole-brain Schaefer FDR.
- Save subject IDs with subject-level arrays.
- Save feature masks and feature-space metadata with downstream results.
- Save raw and per-voxel-normalized RDMs when possible.
- Add new metrics alongside existing metrics rather than replacing existing scientific definitions.
- Keep output names consistent across FearNetwork, MemoryFearNetwork, and Schaefer when analyses refer to the same concept.
- For sensitivity notebooks, recompute downstream analyses from the active mask rather than loading cached p < .05 downstream payloads.

## Editing Requirement
- for figures
  - use seaborn if possible
  - use same theme consistently
  - if doing group comparison, alwasy add standard error as error bars
