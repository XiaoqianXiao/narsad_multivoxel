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

Run this after the Hyak jobs are complete:

```bash
salloc -A psych -p cpu-g2-mem2x --mem=60G --time=12:00:00
FEAR_DIR='/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/LSS/results/FearNetwork/' \
MEMORY_DIR='/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/LSS/results/MemoryFearNetwork/' \
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
/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/LSS/results/FearNetwork/checkpoints/cell_06_aim1_scr_physiological_responder.joblib
```

The primary `cell_06.joblib` is not overwritten. After the subgroup jobs finish, export a tidy table:

```bash
python3 scripts/export_aim1_scr_sensitivity.py \
  --feature-dir /gscratch/scrubbed/fanglab/xiaoqian/NARSAD/LSS/results/FearNetwork \
  --out /gscratch/scrubbed/fanglab/xiaoqian/NARSAD/LSS/results/mvpa_l2/stats/aim1_scr_sensitivity.csv
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
