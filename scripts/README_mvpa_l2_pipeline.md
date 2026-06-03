# MVPA L2 Post-Hyak Pipeline

This folder contains the lightweight project layer for executing the analysis plan in `mvpa_L2.md` after the expensive feature-space MVPA jobs have finished.

## Inputs

- Primary feature space: output cache from `hyak/mvpa_L2_voxel_FearNetwork.py`.
- Mask sensitivity feature space: output cache from `hyak/mvpa_L2_voxel_MemoryFearNetwork.py`.
- Optional whole-brain/parcellation sensitivity feature space: output cache from `hyak/mvpa_L2_voxel_WholeBrain_Schaefer.py`.
- SCR sensitivity groups: derived from `scr_analysis_outputs`, originally produced from `analysis_scr.ipynb` and `identify_fear_learning_subjects_scr.ipynb`.

## One-Command Workflow

Run this after the Hyak jobs are complete:

```bash
FEAR_DIR=/path/to/FearNetwork/output \
MEMORY_DIR=/path/to/MemoryFearNetwork/output \
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

## Outputs

- `outputs/mvpa_l2/harmonized/scr_sensitivity_groups.csv`: SCR responder/learner flags.
- `outputs/mvpa_l2/harmonized/mvpa_l2_subject_metrics.csv`: harmonized subject-level neural, clinical, drug, and SCR variables.
- `outputs/mvpa_l2/stats/aim2_group_difference.csv`: placebo SAD vs HC models.
- `outputs/mvpa_l2/stats/aim3_clinical_relevance.csv`: anxiety-symptom association models.
- `outputs/mvpa_l2/stats/aim4_scr_convergence.csv`: neural-SCR convergence models.
- `outputs/mvpa_l2/stats/aim5_oxytocin_modulation.csv`: `Group * Drug` modulation models.
- `outputs/mvpa_l2/stats/sensitivity_models_all.csv`: available alternative-mask and SCR-cohort sensitivity models. Without `SCHAEFER_DIR`, this includes MemoryFearNetwork and SCR-cohort sensitivities only.
- `outputs/mvpa_l2/stats/mvpa_l2_results_summary.md`: compact report sorted by p-value within each result table.

## Design Logic

The expensive voxel scripts remain the source of feature extraction and decoder/cache generation. The post-Hyak scripts only harmonize subject-level outputs and run manuscript-facing statistics, so sensitivity analyses do not require rerunning the full MVPA unless a feature-space script itself changes.
