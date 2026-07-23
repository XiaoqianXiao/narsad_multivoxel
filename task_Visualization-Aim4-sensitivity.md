# Task: Aim 4 Sensitivity Analysis Visualization

Update the Aim 4 sensitivity-analysis section of `mvpa_l2.ipynb` to generate:

1. A publication-quality 2-panel robustness figure.
2. A companion sensitivity statistics table for Aim 4.

## Context

Aim 4 tests whether neural profiles of vicarious safety/threat learning align with SCR indices of threat and safety learning.

Primary Aim 4 model:

```text
SCR_metric ~ neural_metric + covariates
```

Aim 4 primary analyses are run separately within diagnostic group, focusing on SAD and HC participants.

The sensitivity analysis should evaluate whether Aim 4 neural–SCR convergence is robust to:

1. Alternative neural feature spaces / masks.
2. SCR-defined responder or learner cohorts.

The sensitivity analysis is about robustness, not discovery. Keep the same general model family, same core neural–SCR interpretation, and same group-specific structure as the primary Aim 4 analysis.

---

## Primary variables for sensitivity checks

### Primary SCR outcomes

Use the primary Aim 4 SCR trajectory metrics:

- `SCR_Safety_Trajectory_Slope`
- `SCR_Threat_Trajectory_Slope`

### Primary neural predictors

Use the six primary neural metrics from Aim 2:

Geometry:

- `Neural_Dist_Safety_Background`
- `Neural_Dist_Threat_Safety`

Certainty:

- `Neural_SafetyEvidence`
- `Neural_ThreatEvidence`

Trajectory:

- `Neural_Safety_Trajectory_Slope`
- `Neural_Threat_Trajectory_Slope`

### Key convergence model pairs for figure display

For the figure, emphasize the most interpretable safety- and threat-aligned model pairs:

Safety-aligned models:

- `Neural_Safety_Trajectory_Slope` predicting `SCR_Safety_Trajectory_Slope`
- `Neural_Dist_Safety_Background` predicting `SCR_Safety_Trajectory_Slope`
- `Neural_SafetyEvidence` predicting `SCR_Safety_Trajectory_Slope`

Threat-aligned models:

- `Neural_Threat_Trajectory_Slope` predicting `SCR_Threat_Trajectory_Slope`
- `Neural_Dist_Threat_Safety` predicting `SCR_Threat_Trajectory_Slope`
- `Neural_ThreatEvidence` predicting `SCR_Threat_Trajectory_Slope`

The full table should still include all six neural predictors crossed with both primary SCR outcomes, unless the existing Aim 4 primary code already defines a smaller confirmatory set.

---

## Sensitivity dimensions

### 1. Feature-space robustness

Compare the same Aim 4 neural–SCR models across available feature spaces.

Required if available:

- `FearNetwork`
- `MemoryFearNetwork`

Optional if already implemented in the notebook/data outputs:

- `Schaefer_Tian`
- `WholeBrain`
- other parcellation-based feature spaces already used elsewhere in `mvpa_l2.ipynb`

Use the same model specification as the primary Aim 4 analysis within each feature space.

### 2. SCR-cohort robustness

Compare the same Aim 4 neural–SCR models across SCR-defined cohorts.

Required cohort columns, if available:

- `All`
- `SCR_Physiological_Responder`
- `SCR_Simple_Acquisition_Differential_Learner`
- `SCR_Habituation_Adjusted_Learner`
- `SCR_Late_Phase_Sensitivity_Learner`

If some cohort flags are missing, skip them gracefully and report which were unavailable.

If cohort sample size is too small for a stable regression, do not force the model. Mark the cell/table row as `NA_insufficient_N` and preserve the reason in the table.

Recommended minimum:

- Do not estimate a model if `N < 10` within group and cohort.
- If a stricter threshold is already used in the notebook, follow the existing threshold.

---

## Figure output

Create one figure:

```text
Figure 4-S2. Sensitivity analysis of Aim 4 neural–SCR convergence
```

Use a 1 x 2 layout.

---

## Panel A. Feature-space robustness heatmap

Purpose:

Show whether the main Aim 4 neural–SCR convergence pattern is robust across neural feature spaces.

Rows:

- Safety trajectory convergence
- Safety geometry convergence
- Safety certainty convergence
- Threat trajectory convergence
- Threat geometry convergence
- Threat certainty convergence

Columns:

- Feature spaces, ordered as:
  1. `FearNetwork`
  2. `MemoryFearNetwork`
  3. `Schaefer_Tian`, if available
  4. `WholeBrain`, if available
  5. any other available feature spaces

Group handling:

- Show SAD and HC separately.
- Preferred format: two side-by-side heatmaps within Panel A, one for SAD and one for HC.
- Alternative acceptable format: grouped columns, for example `SAD_FearNetwork`, `HC_FearNetwork`, etc.

Cell encoding:

- Cell color = standardized beta coefficient, `beta_std`.
- Cell text = rounded beta value.
- Add significance marker if `q_FDR < .05`.
- Use a centered diverging color scale around zero.
- Use the same color scale for SAD and HC.

Missing or unstable models:

- Show missing/insufficient cells in light gray or leave blank.
- Add `NA` or `n<10` in the cell text when appropriate.

---

## Panel B. SCR-cohort robustness heatmap

Purpose:

Show whether Aim 4 neural–SCR convergence is robust among SCR responders and SCR learner subgroups.

Rows:

Use the same six model rows as Panel A:

- Safety trajectory convergence
- Safety geometry convergence
- Safety certainty convergence
- Threat trajectory convergence
- Threat geometry convergence
- Threat certainty convergence

Columns:

Order SCR cohorts as:

1. `All`
2. `SCR_Physiological_Responder`
3. `SCR_Simple_Acquisition_Differential_Learner`
4. `SCR_Habituation_Adjusted_Learner`
5. `SCR_Late_Phase_Sensitivity_Learner`

Group handling:

- Show SAD and HC separately.
- Preferred format: two side-by-side heatmaps within Panel B, one for SAD and one for HC.
- Do not create separate Group x Drug cells.
- If cohort sensitivity pools across drug conditions because subgroup sizes are small, include `Drug` as a covariate when available and label the panel/table clearly as pooled across drug for sensitivity only.

Cell encoding:

- Cell color = standardized beta coefficient, `beta_std`.
- Cell text = rounded beta value.
- Add significance marker if `q_FDR < .05`.
- Use the same centered diverging color scale as Panel A.

---

## Companion statistics table

Create a table named:

```text
Table S5. Sensitivity analysis of Aim 4 neural–SCR convergence
```

Save the table as CSV and, if the notebook already supports it, render a clean markdown preview.

Required columns:

- `Sensitivity_type`
- `Group`
- `Feature_space`
- `SCR_cohort`
- `SCR_outcome`
- `Neural_predictor`
- `Model_label`
- `beta_std`
- `CI_95`
- `t`
- `p`
- `q_FDR`
- `N`
- `Covariates`
- `Status`

Column notes:

- `Sensitivity_type` should be either `Feature_space` or `SCR_cohort`.
- `Feature_space` should be filled for feature-space robustness rows and can be `FearNetwork` for cohort robustness rows unless another primary feature space is used.
- `SCR_cohort` should be filled for SCR-cohort robustness rows and can be `All` for feature-space robustness rows.
- `Model_label` should use clean labels, such as `Safety trajectory convergence` or `Threat geometry convergence`.
- `Status` should be `estimated`, `missing_variable`, `insufficient_N`, or another clear reason.

The table should include all estimable sensitivity models, not only significant models.

---

## Plot labels

Use clean labels in the figure.

Suggested model-row labels:

- `Safety trajectory`
- `Safety geometry`
- `Safety certainty`
- `Threat trajectory`
- `Threat geometry`
- `Threat certainty`

Suggested SCR cohort labels:

- `All`
- `Responder`
- `Simple learner`
- `Habituation-adjusted learner`
- `Late-phase learner`

Suggested feature-space labels:

- `FearNetwork`
- `MemoryFearNetwork`
- `Schaefer/Tian`
- `Whole brain`

Preserve original variable names in the statistics table.

---

## Statistical and plotting requirements

- Use the same covariates as the Aim 4 primary analysis unless the notebook specifies otherwise.
- Standardize continuous predictors and outcomes before estimating `beta_std`, consistent with primary Aim 4.
- Apply FDR correction within each sensitivity family in a transparent way:
  - feature-space sensitivity models
  - SCR-cohort sensitivity models
- If the primary Aim 4 code already defines an FDR family, reuse that structure and document it in comments.
- Use `pandas`, `numpy`, `matplotlib`; `seaborn` is optional.
- Save the figure as both SVG and PNG.
- Save the table as CSV.
- Add clear panel labels: `A` and `B`.
- Keep typography, margins, and color scale consistent with the Aim 4 primary and Aim 3 primary figures.

---

## Output file names

Use clear output names such as:

```text
figures/Figure4_S2_Aim4_sensitivity_neural_SCR_convergence.svg
figures/Figure4_S2_Aim4_sensitivity_neural_SCR_convergence.png
tables/TableS5_Aim4_sensitivity_neural_SCR_convergence.csv
```

If the project already has a naming convention, follow the existing convention.

---

## Exclusions

Do not include:

- Secondary SCR outcomes in this sensitivity figure unless they are already part of a separate secondary-sensitivity section.
- Aim 5 oxytocin modulation tests.
- Group x Drug inference.
- New exploratory neural metrics not defined in the analysis plan.
- Scatter plots, unless one is already required elsewhere. The sensitivity figure should be a compact robustness heatmap figure.

---

## Acceptance criteria

The updated notebook section should:

1. Generate a 2-panel sensitivity figure for Aim 4.
2. Show feature-space robustness in Panel A.
3. Show SCR responder/learner cohort robustness in Panel B.
4. Preserve group-specific results for SAD and HC.
5. Use standardized beta values as the heatmap color scale.
6. Mark FDR-significant cells clearly.
7. Gracefully handle missing feature spaces, missing cohort flags, and insufficient sample sizes.
8. Save both figure and table outputs.
9. Avoid adding secondary SCR metrics, drug-modulation inference, or unrelated exploratory analyses.
