# MVPA L2 Post-Hyak Pipeline

This folder contains the lightweight project layer for executing the analysis plan in `mvpa_L2.md` after the expensive feature-space MVPA jobs have finished.

## Inputs

- Primary feature space: output cache from `hyak/mvpa_L2_voxel_FearNetwork.py`.
- Mask sensitivity feature space: output cache from `hyak/mvpa_L2_voxel_MemoryFearNetwork.py`.
- Optional whole-brain/parcellation sensitivity feature space: output cache from `hyak/mvpa_L2_voxel_WholeBrain_Schaefer.py`.
- SCR sensitivity groups: derived from `scr_analysis_outputs`, originally produced from `analysis_scr.ipynb` and `identify_fear_learning_subjects_scr.ipynb`.

## One-Command Workflow

On Hyak, submit the whole currently executable workflow without whole-brain/parcellation sensitivity:

```bash
hyak/submit_mvpa_L2_needed_no_wholebrain.sh
```

This submits the FearNetwork chain, the MemoryFearNetwork sensitivity chain, and then a dependent post-Hyak harmonization/statistics job. All three parts run inside:

```text
/gscratch/fang/images/jupyter.sif
```

The post-Hyak job can also be submitted by itself after feature-space jobs finish:

```bash
hyak/submit_mvpa_L2_posthyak.sh --dependency FEAR_FINAL_JOB_ID:MEMORY_FINAL_JOB_ID
```

Use `SCR_DIR=/path/to/scr_analysis_outputs` if the SCR notebook outputs are not under the default project SCR folder.

## Stage 11 Mask Modes

The feature-space scripts support two Stage 11 mask/scoring modes:

```bash
# Default current .py behavior: decision-margin permutation importance plus downstream fallback.
STAGE11_MASK_MODE=current STAGE11_SCORING=auto hyak/submit_mvpa_L2_fearnetwork_stage.sh all

# Original-notebook mask behavior: forced-choice permutation importance and no all-positive fallback.
STAGE11_MASK_MODE=original_notebook STAGE11_SCORING=auto hyak/submit_mvpa_L2_fearnetwork_stage.sh all
```

`STAGE11_SCORING=auto` maps `current` to `decision_margin` and `original_notebook` to `forced_choice`. You can override it with `STAGE11_SCORING=decision_margin` or `STAGE11_SCORING=forced_choice` for sensitivity checks. Use separate output directories for manuscript-grade parallel runs, for example `FearNetwork` and `FearNetwork_originalMask`, so final Stage 11 checkpoints and downstream outputs do not overwrite each other.

Post-Hyak scripts do not rerun masks or voxel models. They read whichever completed feature-space output folders are supplied through `FEAR_DIR`, `MEMORY_DIR`, and optionally `SCHAEFER_DIR`. Therefore each mask-mode run should also use a separate `OUT_ROOT` for harmonized/statistical outputs.

Current-mask post-Hyak example:

```bash
FEAR_DIR=/output_dir/FearNetwork \
MEMORY_DIR=/output_dir/MemoryFearNetwork \
OUT_ROOT=/output_dir/mvpa_l2 \
bash scripts/run_mvpa_l2_posthyak.sh
```

Original-notebook-mask post-Hyak example:

```bash
STAGE11_MASK_MODE=original_notebook \
bash scripts/run_mvpa_l2_posthyak.sh
```

You can still override `FEAR_DIR`, `MEMORY_DIR`, or `OUT_ROOT` explicitly, but `STAGE11_MASK_MODE=original_notebook` now defaults them to `/output_dir/FearNetwork_originalMask`, `/output_dir/MemoryFearNetwork_originalMask`, and `/output_dir/mvpa_l2_originalMask`.

After each post-Hyak run, check the generated files under `OUT_ROOT/stats/`, especially `aim2_trajectory_panel.csv`, `aim2_group_difference.csv`, and `manuscript_primary_results.csv`. For the original-notebook mask run, the upstream Stage 11 joblib diagnostics should report `stage11_mask_mode = original_notebook` and `importance_scoring = forced_choice_scorer`.

Run this after the Hyak jobs are complete:

```bash
salloc -A psych -p cpu-g2-mem2x --mem=60G --time=12:00:00
FEAR_DIR='/gscratch/fang/NARSAD/MRI/derivatives/fMRI_analysis/LSS/results/FearNetwork/' \
MEMORY_DIR='/gscratch/fang/NARSAD/MRI/derivatives/fMRI_analysis/LSS/results/MemoryFearNetwork/' \
bash scripts/run_mvpa_l2_posthyak.sh
```

If the outputs are already in `outputs/mvpa_l2/FearNetwork` and `outputs/mvpa_l2/MemoryFearNetwork`, run:

```bash
bash scripts/run_mvpa_l2_posthyak.sh
```

To include the whole-brain/parcellation sensitivity check later, add `SCHAEFER_DIR`:

```bash
SCHAEFER_DIR=/path/to/Schaefer/output \
bash scripts/run_mvpa_l2_posthyak.sh
```

## Analysis 1 SCR-Subgroup Sensitivity

For Analysis 1, the clean sensitivity test is to rerun only Stage 6 within the SCR-defined responder/learner cohorts. This asks whether the main `CSR` versus `CSS` discriminability, cross-group generalization, and SAD-HC spatial similarity remain visible among participants who show physiological learning. It does not rerun the full Aim 2-5 model stack.

First make sure `outputs/mvpa_l2/harmonized/scr_sensitivity_groups.csv` exists. If needed:

```bash
bash scripts/run_mvpa_l2_posthyak.sh
```

Then submit the labeled Stage 6 subgroup jobs:

```bash
hyak/submit_mvpa_L2_aim1_scr_sensitivity.sh
```

By default this runs four cohorts:

- `SCR_Physiological_Responder`
- `SCR_Simple_Acquisition_Differential_Learner`
- `SCR_Habituation_Adjusted_Learner`
- `SCR_Late_Phase_Sensitivity_Learner`

The jobs write labeled checkpoints such as:

```text
/gscratch/fang/NARSAD/MRI/derivatives/fMRI_analysis/LSS/results/FearNetwork/checkpoints/cell_06_aim1_scr_physiological_responder.joblib
```

The primary `cell_06.joblib` is not overwritten. After the subgroup jobs finish, export a tidy table:

```bash
python3 scripts/export_aim1_scr_sensitivity.py \
  --feature-dir /gscratch/fang/NARSAD/MRI/derivatives/fMRI_analysis/LSS/results/FearNetwork \
  --out /gscratch/fang/NARSAD/MRI/derivatives/fMRI_analysis/LSS/results/mvpa_l2/stats/aim1_scr_sensitivity.csv
```

## Outputs

- `outputs/mvpa_l2/harmonized/scr_sensitivity_groups.csv`: SCR responder/learner flags.
- `outputs/mvpa_l2/harmonized/mvpa_l2_subject_metrics.csv`: harmonized subject-level neural, clinical, drug, and SCR variables.
- `outputs/mvpa_l2/stats/aim2_group_difference.csv`: placebo SAD vs HC models.
- `outputs/mvpa_l2/stats/aim1_mask_feature_sensitivity.csv`: Aim 1 feature-space sensitivity rows for FearNetwork, MemoryFearNetwork, and optional Schaefer/whole-brain.
- `outputs/mvpa_l2/stats/aim1_mask_feature_sensitivity_functional_drop_tests.csv`: paired self-minus-cross sign-flip tests for Aim 1 feature-space sensitivity.
- `outputs/mvpa_l2/stats/aim1_mask_feature_sensitivity_functional_drop_nulls.csv`: paired self-minus-cross sign-flip null distributions for Aim 1 feature-space sensitivity; this is used by Figure S1 Panel B null histograms.
- `outputs/mvpa_l2/stats/aim3_clinical_relevance.csv`: anxiety-symptom association models.
- `outputs/mvpa_l2/stats/aim4_scr_convergence.csv`: neural-SCR convergence models.
- `outputs/mvpa_l2/stats/aim5_oxytocin_modulation.csv`: `Group * Drug` modulation models.
- `outputs/mvpa_l2/stats/sensitivity_models_all.csv`: available alternative-mask and SCR-cohort sensitivity models. Without `SCHAEFER_DIR`, this includes MemoryFearNetwork and SCR-cohort sensitivities only.
- `outputs/mvpa_l2/stats/manuscript_primary_results.csv`: single manuscript-ready primary results table spanning Aim 2-5.
- `outputs/mvpa_l2/stats/manuscript_primary_results.md`: compact Markdown version of the primary results table.
- `outputs/mvpa_l2/stats/aim4_convergence_matrix.csv`: long-form primary neural metric by primary SCR index convergence table.
- `outputs/mvpa_l2/stats/aim4_convergence_matrix_wide.csv`: matrix-form convergence table for manuscript review.
- `outputs/mvpa_l2/stats/aim4_convergence_matrix.md`: Markdown convergence matrix that emphasizes estimates and confidence intervals.
- `outputs/mvpa_l2/stats/mvpa_l2_qc_dashboard.md`: reproducibility/QC dashboard with subject counts, missingness, model status counts, and leakage-audit reminders.
- `outputs/mvpa_l2/stats/mvpa_l2_results_summary.md`: compact report sorted by p-value within each result table.

## Design Logic

The expensive voxel scripts remain the source of feature extraction and decoder/cache generation. The post-Hyak scripts only harmonize subject-level outputs and run manuscript-facing statistics, so sensitivity analyses do not require rerunning the full MVPA unless a feature-space script itself changes.

## Manuscript-Ready Result Hierarchy

Aim 3 clinical relevance models use an explicit clinical hierarchy:

1. `lsas_total`: primary social anxiety endpoint.
2. `lsas_fear`: secondary LSAS subscale decomposition.
3. `lsas_avoid`: secondary LSAS subscale decomposition.
4. `dass_anxiety`: convergent general anxiety endpoint.

The primary manuscript table foregrounds effect estimates and confidence intervals. Corrected p-values remain in the table for reference, but rows are ordered by scientific aim, endpoint hierarchy, group, and neural metric hierarchy rather than by p-value.

## Leakage Audit Scope

The manuscript-export scripts operate only on harmonized subject-level tables. They do not refit decoders, scalers, feature masks, or calibration models. Predictive leakage checks therefore belong primarily in the upstream Hyak feature-space scripts: scaling, mask generation, feature selection, hyperparameter tuning, and probability calibration must be fit inside the appropriate training structure with subject-aware validation.
