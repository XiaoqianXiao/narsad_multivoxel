Update the Aim 3 sensitivity-results visualization in `mvpa_l2.ipynb` to generate:
Aim 3 sensitivity visualization and statistics table.

Goal:
Use the same figure/table structure as `task_Visualization-Aim3-primary.md`, but repeat the Aim 3 clinical-neural association models under prespecified sensitivity conditions.

Aim 3 model:
`z(clinical_score) ~ z(neural_metric) + covariates`

Run models separately within:
- SAD-placebo participants
- HC-placebo participants

Do not pool SAD and HC in the same model or the same plotted cell.

Primary clinical outcomes:
- dass_anxiety
- lsas_total

Primary neural metrics, ordered by domain:
1. Neural_Dist_Safety_Background
2. Neural_Dist_Threat_Safety
3. Neural_SafetyEvidence
4. Neural_ThreatEvidence
5. Neural_Safety_Trajectory_Slope
6. Neural_Threat_Trajectory_Slope

Sensitivity specifications:
Run the same 24 Aim 3 models for each available sensitivity specification:

1. Alternative feature-space sensitivity
   - MemoryFearNetwork
   - Schaefer/Tian parcellation or whole-brain/parcellation output, if available

2. SCR-defined cohort sensitivity
   - SCR_Physiological_Responder == 1
   - SCR_Simple_Acquisition_Differential_Learner == 1
   - SCR_Habituation_Adjusted_Learner == 1
   - SCR_Late_Phase_Sensitivity_Learner == 1

Keep the Aim 3 placebo restriction unless the notebook already defines SCR sensitivity as an all-drug robustness check. Do not split sensitivity models by Drug. If Drug is included or excluded differently from the primary Aim 3 model, record this clearly in the output table and figure caption.

Figure layout:
Generate one 3-panel figure per sensitivity specification, using the same format as the primary Aim 3 figure.

Panel A. Regression-annotated association heatmap
- Use a 6 x 4 matrix.
- Rows = six primary neural metrics.
- Columns:
  1) SAD: DASS anxiety
  2) SAD: LSAS total
  3) HC: DASS anxiety
  4) HC: LSAS total
- Each cell should show only the participants from that exact group, outcome, neural metric, and sensitivity specification.
- Do NOT overlay SAD and HC in the same cell.
- Background tile color = standardized beta for that exact model.
- Use a diverging colormap centered at zero.
- Overlay faint residualized scatter points for that cell only.
- Overlay one regression line for that cell only.
- Add cell text:
  - β = value
  - q = value
  - optional significance symbol: * q < .05; † p < .05 only.
- Residualize both x and y for covariates before plotting if covariates are used.
- Use shared x/y limits across cells within each sensitivity figure.
- Keep points faint and small; regression line should be visually clear.
- If n is too small for a stable regression line, show points and cell statistics if estimable, but annotate the cell with `low n`.

Panel B. SAD forest plot
- Include all six neural metrics.
- Show standardized beta ± 95% CI.
- Use separate markers for dass_anxiety and lsas_total.
- Add vertical reference line at β = 0.
- Use the same x-axis limits as Panel C.
- Add the sensitivity specification name in the panel title or subtitle.

Panel C. HC forest plot
- Same format as Panel B.
- Include all six neural metrics.
- Same x-axis scale as the SAD forest plot.
- Add the sensitivity specification name in the panel title or subtitle.

Statistics:
Create one combined Aim 3 sensitivity statistics table with one row per model and sensitivity specification.

Required columns:
- sensitivity_type
- sensitivity_spec
- feature_space
- cohort_flag
- group
- clinical_outcome
- neural_metric
- beta_standardized
- ci_low
- ci_high
- t
- p
- q_fdr_within_spec
- q_fdr_all_sensitivity
- n
- covariates
- drug_filter
- model_formula
- notes

Multiple-comparison correction:
- For each sensitivity specification, apply FDR correction across the same 24-model family:
  2 groups x 2 clinical outcomes x 6 neural metrics.
- Save this as `q_fdr_within_spec`.
- Also compute an optional global FDR across all Aim 3 sensitivity rows and save it as `q_fdr_all_sensitivity`.
- Make the FDR scope easy to change.

Outputs:
Save one figure per sensitivity specification:
- `FigureS3_Aim3_sensitivity_clinical_relevance_<sensitivity_spec>.svg`
- `FigureS3_Aim3_sensitivity_clinical_relevance_<sensitivity_spec>.png`

Save the combined sensitivity table as:
- `Table_Aim3_sensitivity_statistics.csv`
- `Table_Aim3_sensitivity_statistics.xlsx`

Implementation notes:
- Use pandas, numpy, scipy/statsmodels, matplotlib.
- Use real project data/results if available; do not invent values for final outputs.
- If required columns are missing, print a clear message listing missing columns by sensitivity specification.
- If a sensitivity specification is unavailable, skip it and record the skipped reason.
- Keep code modular:
  1) prepare_aim3_sensitivity_data()
  2) define_aim3_sensitivity_specs()
  3) run_ols_model()
  4) fdr_correct_results()
  5) make_panel_a_regression_heatmap()
  6) make_forest_plot()
  7) save_sensitivity_outputs()
- Reuse the primary Aim 3 plotting functions where possible instead of duplicating plotting logic.
- Use clean publication style, readable labels, consistent metric ordering, and identical visual grammar to the Aim 3 primary figure.
- Do not embed the full statistics table inside the figure.

Caption template:
“Figure S3. Sensitivity analysis for Aim 3 clinical-neural associations. Each figure repeats the primary Aim 3 visualization under one sensitivity specification. Panel A shows group-specific residualized clinical-neural associations; each cell includes only the corresponding group, clinical outcome, neural metric, and sensitivity cohort/feature space. Tile color encodes standardized beta. Panels B-C show standardized beta estimates with 95% confidence intervals for SAD and HC participants, respectively. Sensitivity analyses assess robustness across alternative feature spaces and SCR-defined responder/learner cohorts.”
