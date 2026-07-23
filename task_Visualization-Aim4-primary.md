# Task: Aim 4 Primary Visualization — Neural–SCR Convergence

Update the **Aim 4 primary-results visualization** section in `mvpa_l2.ipynb`.

## Purpose

Aim 4 tests whether neural profiles of vicarious safety/threat learning align with SCR indices of physiological learning.

Primary model:

```text
SCR_metric ~ neural_metric + covariates
```

Run models **separately within diagnostic group**:

- SAD-placebo
- HC-placebo

Use **placebo-session participants only**.

## Primary variables

### Primary SCR outcomes

- `SCR_Safety_Trajectory_Slope`
- `SCR_Threat_Trajectory_Slope`

### Primary neural predictors

Geometry:

- `Neural_Dist_Safety_Background`
- `Neural_Dist_Threat_Safety`

Certainty:

- `Neural_SafetyEvidence`
- `Neural_ThreatEvidence`

Trajectory:

- `Neural_Safety_Trajectory_Slope`
- `Neural_Threat_Trajectory_Slope`

## Required outputs

Generate:

1. **Figure 4. Neural–SCR convergence during vicarious safety and threat learning**
2. **Table 4. Primary neural–SCR convergence models**

Save outputs as:

- `Figure4_Aim4_primary_neural_SCR_convergence.svg`
- `Figure4_Aim4_primary_neural_SCR_convergence.png`
- `Table4_Aim4_primary_neural_SCR_convergence.csv`
- Optional: `Table4_Aim4_primary_neural_SCR_convergence.md`

## Figure 4 layout

Use a **1 × 3 layout**.

### Panel A. Primary neural–SCR association heatmap

Goal: summarize all confirmatory primary neural × SCR associations.

Rows:

1. `Neural_Dist_Safety_Background`
2. `Neural_Dist_Threat_Safety`
3. `Neural_SafetyEvidence`
4. `Neural_ThreatEvidence`
5. `Neural_Safety_Trajectory_Slope`
6. `Neural_Threat_Trajectory_Slope`

Columns:

1. `SCR_Safety_Trajectory_Slope`
2. `SCR_Threat_Trajectory_Slope`

Requirements:

- Show results separately for **SAD** and **HC**.
- Preferred format: two small side-by-side heatmaps within Panel A, one for SAD and one for HC.
- Cell color = standardized beta coefficient, `beta_std`.
- Cell text = beta value rounded to 2 decimals.
- Add a significance marker to cell text when `q_FDR < .05`.
- Use a centered diverging color scale with zero as the midpoint.
- Use the same color scale limits for SAD and HC.
- Visually group rows by metric family: Geometry, Certainty, Trajectory.
- Use clean display labels in the plot, but keep original variable names in the table.

### Panel B. Safety convergence scatter plot

Goal: show the key primary safety-convergence association.

- X-axis: `Neural_Safety_Trajectory_Slope`
- Y-axis: `SCR_Safety_Trajectory_Slope`
- Show SAD and HC in the same panel.
- Use distinct markers and/or line styles for SAD and HC.
- Add group-specific linear fit lines.
- Add 95% CI bands if supported by the plotting code; otherwise omit CI bands.
- Annotate each group with `beta_std` and `q_FDR` for this model.

### Panel C. Threat convergence scatter plot

Goal: show the key primary threat-convergence association.

- X-axis: `Neural_Threat_Trajectory_Slope`
- Y-axis: `SCR_Threat_Trajectory_Slope`
- Match the visual format of Panel B.
- Show SAD and HC in the same panel.
- Use group-specific linear fit lines.
- Add 95% CI bands if supported by the plotting code; otherwise omit CI bands.
- Annotate each group with `beta_std` and `q_FDR` for this model.

## Table 4 requirements

Create a compact statistics table named:

```text
Table 4. Primary neural–SCR convergence models
```

Required columns:

- `Group`
- `SCR_outcome`
- `Neural_predictor`
- `beta_std`
- `CI_95`
- `t`
- `p`
- `q_FDR`
- `N`

Content requirements:

- Include every primary neural predictor × primary SCR outcome model.
- Run/report models separately for SAD and HC.
- Expected number of rows: `6 neural predictors × 2 SCR outcomes × 2 groups = 24 rows`.
- Apply FDR correction across the full family of Aim 4 primary tests unless the notebook already defines a stricter primary-test family.
- Preserve original variable names in `SCR_outcome` and `Neural_predictor`.

## Implementation requirements

- Use Python.
- Preferred packages: `pandas`, `numpy`, `matplotlib`; `seaborn` is optional.
- Keep style consistent with the Aim 3 primary figure: heatmap summary plus selected regression scatter plots.
- Use clear panel labels: **A**, **B**, **C**.
- Use readable axis labels and a concise figure title.
- Standardize predictors and outcomes before estimating `beta_std`, or clearly use an existing standardized estimate if already computed.
- Use covariates already defined in the notebook for Aim 4 primary models.
- If a required column is missing, stop with a clear error listing the missing columns.
- Do not overwrite or remove unrelated Aim 1–3 or Aim 5 sections.

## Exclusions

Do **not** include:

- Secondary SCR metrics:
  - `SCR_SafetyMinusBackground`
  - `SCR_ThreatMinusSafety`
- SCR responder/learner sensitivity cohorts.
- Oxytocin participants.
- Pooled SAD + HC associations.
- Any group-collapsed primary result.

## Acceptance criteria

The revised notebook section should produce:

1. A saved 3-panel Figure 4 in both SVG and PNG.
2. A saved Table 4 CSV with 24 rows.
3. Group-specific SAD and HC results throughout.
4. No secondary SCR metrics or oxytocin data in the primary Aim 4 figure/table.
5. Plot labels that are publication-ready while preserving original variable names in the statistics table.
