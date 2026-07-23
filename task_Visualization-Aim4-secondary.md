# Task: Aim 4 Secondary Analysis Visualization

Update the Aim 4 secondary-results visualization section in `mvpa_l2.ipynb` to generate:

1. One publication-quality 3-panel secondary-results figure.
2. One companion statistics table for Aim 4 secondary neural-SCR convergence models.

This task should follow the same overall style as the Aim 4 primary figure: association heatmap plus selected regression scatter plots. The secondary analysis should be clearly labeled as supportive evidence and should not be mixed with the primary Aim 4 results.

---

## Scientific context

Aim 4 tests whether neural profiles of vicarious safety and threat learning align with physiological SCR indices.

For secondary analysis, focus on the secondary SCR contrast metrics:

- `SCR_SafetyMinusBackground`: mean `SCR_Anticipatory(CSS)` minus mean `SCR_Anticipatory(CS-)`
- `SCR_ThreatMinusSafety`: mean `SCR_Anticipatory(CSR)` minus mean `SCR_Anticipatory(CSS)`

Models should be run separately within diagnostic group and restricted to placebo-session participants:

```text
SCR_metric ~ neural_metric + covariates
```

Do not collapse SAD and HC. Do not include oxytocin participants.

---

## Groups

Run and visualize models separately for:

- `SAD-placebo`
- `HC-placebo`

---

## Secondary SCR outcomes

Use only the secondary SCR outcomes listed below:

1. `SCR_SafetyMinusBackground`
2. `SCR_ThreatMinusSafety`

Do not include the primary SCR trajectory slopes in this secondary figure.

---

## Neural predictors

Use the six primary neural metrics from Aim 2 as predictors.

### Geometry

- `Neural_Dist_Safety_Background`
- `Neural_Dist_Threat_Safety`

### Certainty

- `Neural_SafetyEvidence`
- `Neural_ThreatEvidence`

### Trajectory

- `Neural_Safety_Trajectory_Slope`
- `Neural_Threat_Trajectory_Slope`

---

## Figure output

Generate one figure:

```text
Figure 4-S1. Secondary SCR evidence for neural-physiological convergence
```

Use a 1 x 3 layout.

---

## Panel A: Secondary neural-SCR association heatmap

Purpose: summarize all secondary SCR convergence models.

### Required content

- Rows = six primary neural metrics.
- Columns = two secondary SCR outcomes.
- Show SAD and HC separately, either as:
  - two side-by-side heatmaps inside Panel A, or
  - grouped columns with group labels.
- Cell color = standardized beta coefficient.
- Cell text = standardized beta value.
- Add a significance marker when `q_FDR < .05`.
- Use a centered diverging color scale around zero.
- Use the same color limits for SAD and HC.
- Group neural metrics visually by category:
  - Geometry
  - Certainty
  - Trajectory

### Recommended display order

Rows:

1. `Neural_Dist_Safety_Background`
2. `Neural_Dist_Threat_Safety`
3. `Neural_SafetyEvidence`
4. `Neural_ThreatEvidence`
5. `Neural_Safety_Trajectory_Slope`
6. `Neural_Threat_Trajectory_Slope`

Columns:

1. `SCR_SafetyMinusBackground`
2. `SCR_ThreatMinusSafety`

---

## Panel B: Safety contrast convergence scatter plot

Purpose: show the most interpretable safety-related secondary SCR association.

### Required content

- X-axis: `Neural_Dist_Safety_Background`
- Y-axis: `SCR_SafetyMinusBackground`
- Plot individual participants.
- Show SAD and HC separately using consistent markers or regression lines.
- Add linear regression fit lines.
- Add 95% confidence bands if already available or easy to compute.
- Annotate each group with standardized beta and `q_FDR`.

### Interpretation

This panel tests whether neural safety-background geometry aligns with peripheral safety-vs-background SCR differentiation.

---

## Panel C: Threat contrast convergence scatter plot

Purpose: show the most interpretable threat-related secondary SCR association.

### Required content

- X-axis: `Neural_Dist_Threat_Safety`
- Y-axis: `SCR_ThreatMinusSafety`
- Plot individual participants.
- Show SAD and HC separately using the same style as Panel B.
- Add linear regression fit lines.
- Add 95% confidence bands if already available or easy to compute.
- Annotate each group with standardized beta and `q_FDR`.

### Interpretation

This panel tests whether neural threat-safety geometry aligns with peripheral threat-vs-safety SCR differentiation.

---

## Companion statistics table

Create one table:

```text
Table S4. Secondary neural-SCR convergence models
```

The table should include all combinations of:

- 2 groups: SAD, HC
- 2 secondary SCR outcomes
- 6 primary neural predictors

Expected number of rows: 24 models.

### Required columns

- `Group`
- `SCR_outcome`
- `Neural_predictor`
- `beta_std`
- `CI_95`
- `t`
- `p`
- `q_FDR`
- `N`

### Formatting requirements

- Preserve original variable names in the table.
- Use rounded display values for presentation:
  - `beta_std`: 2 decimals
  - `t`: 2 decimals
  - `p`: 3 decimals, or `<.001`
  - `q_FDR`: 3 decimals, or `<.001`
- Include confidence intervals as a single string column, for example `[0.05, 0.48]`.
- Sort rows by:
  1. `Group`
  2. `SCR_outcome`
  3. neural metric category
  4. neural metric order listed above

---

## Output files

Save the figure as:

```text
Figure4S1_Aim4_secondary_neural_SCR_convergence.svg
Figure4S1_Aim4_secondary_neural_SCR_convergence.png
```

Save the statistics table as:

```text
TableS4_Aim4_secondary_neural_SCR_convergence.csv
```

Optionally also save a markdown version:

```text
TableS4_Aim4_secondary_neural_SCR_convergence.md
```

---

## Implementation requirements

Use:

- `pandas`
- `numpy`
- `matplotlib`
- `scipy` or `statsmodels` if needed for model fitting
- `seaborn` optional, but keep the style consistent with existing Aim 3 and Aim 4 primary figures

General requirements:

- Use clean labels in the figure.
- Preserve original variable names in the statistics table.
- Add clear panel labels: `A`, `B`, `C`.
- Use readable axis labels and compact annotations.
- Use group-specific models and group-specific regression lines.
- Keep the figure publication-quality and visually consistent with the Aim 4 primary figure.
- Make missing-data handling explicit in code comments.
- Keep all scaling, residualization, and covariate handling consistent with the primary Aim 4 analysis.

---

## Exclusions

Do not include:

- Primary SCR trajectory outcomes in this secondary figure.
- Oxytocin participants.
- Drug effects.
- Group x Drug models.
- SCR-defined learner sensitivity cohorts.
- Secondary neural metrics unless explicitly requested elsewhere.
- Combined SAD + HC regression lines.

---

## Acceptance criteria

The updated notebook section is complete when:

1. The figure has exactly three panels: A, B, and C.
2. Panel A shows all six primary neural predictors crossed with the two secondary SCR outcomes, separately for SAD and HC.
3. Panels B and C show selected safety and threat contrast scatter plots with group-specific regression lines.
4. The statistics table includes all 24 group-specific secondary SCR convergence models.
5. All outputs are saved with the requested filenames.
6. The section is clearly labeled as Aim 4 secondary/supportive evidence.
7. The primary Aim 4 figure and table are not overwritten.
