Update the Aim 3 secondary-results visualization in mvpa_l2.ipynb to generate:
secondary supportive clinical visualization and statistics table.

Goal:
Figure S3 should show whether the primary Aim 3 clinical-neural association pattern is supported by related secondary clinical measures:
z(clinical_score) ~ z(neural_metric) + covariates
run separately within SAD-placebo and HC-placebo participants.

Secondary clinical outcomes:
- lsas_fear
- lsas_avoid
- dass_stress
- dass_depression
- ecr_total

Primary neural metrics, ordered by domain:
1. Neural_Dist_Safety_Background
2. Neural_Dist_Threat_Safety
3. Neural_SafetyEvidence
4. Neural_ThreatEvidence
5. Neural_Safety_Trajectory_Slope
6. Neural_Threat_Trajectory_Slope

Figure layout:
Generate one 3-panel figure using the same visual logic as the Aim 3 primary-results figure.

Panel A. Regression-annotated association heatmap
- Use a 6 x 10 matrix.
- Rows = six primary neural metrics.
- Columns:
  1) SAD: LSAS fear
  2) SAD: LSAS avoid
  3) SAD: DASS stress
  4) SAD: DASS depression
  5) SAD: ECR total
  6) HC: LSAS fear
  7) HC: LSAS avoid
  8) HC: DASS stress
  9) HC: DASS depression
  10) HC: ECR total
- Each cell should show only the participants from that specific group/model.
- Do NOT overlay both SAD and HC in the same cell.
- Background tile color = standardized beta for that exact model.
- Use a diverging colormap centered at zero.
- Use the same beta color scale as the primary Aim 3 heatmap if possible.
- Overlay faint residualized scatter points for that group only.
- Overlay one regression line for that group only.
- Add cell text:
  β = value
  q = value
  optional significance symbol: * q < .05; † p < .05 only.
- Residualize both x and y for covariates before plotting if covariates are used.
- Use shared x/y limits across cells within Panel A.
- Keep points faint and small; regression line should be visually clear.
- Because this matrix is wider than the primary heatmap, use compact labels and rotate column labels if needed.
- Visually separate SAD and HC columns with a slightly thicker vertical divider or grouped column headers.

Panel B. SAD secondary forest plot
- Include all six neural metrics.
- Show standardized beta ± 95% CI.
- Use separate markers for:
  - lsas_fear
  - lsas_avoid
  - dass_stress
  - dass_depression
  - ecr_total
- Add vertical reference line at β = 0.
- Use the same x-axis limits as Panel C.
- Use slight vertical jitter/dodge so markers do not overlap.
- Keep labels readable; use concise display labels.

Panel C. HC secondary forest plot
- Same format as Panel B.
- Include all six neural metrics.
- Use the same clinical-outcome marker scheme as Panel B.
- Use the same x-axis scale as SAD forest plot.

Statistics:
Create a secondary Aim 3 statistics table with one row per model:
columns:
- group
- clinical_outcome
- neural_metric
- beta_standardized
- ci_low
- ci_high
- t
- p
- q_fdr
- n
- covariates

Multiple-comparison correction:
- Apply FDR correction across the secondary Aim 3 model family.
- Preferred: correct across all 60 secondary tests:
  2 groups x 5 secondary clinical outcomes x 6 neural metrics.
- Also make this easy to change if needed, for example:
  - across all secondary tests
  - within group
  - within clinical outcome family

Outputs:
- Save figure as:
  FigureS3_Aim3_secondary_clinical_support.svg
  FigureS3_Aim3_secondary_clinical_support.png
- Save table as:
  Table_Aim3_secondary_statistics.csv
  Table_Aim3_secondary_statistics.xlsx

Implementation notes:
- Use pandas, numpy, scipy/statsmodels, matplotlib.
- Use real project data/results if available; do not invent values for final outputs.
- If required columns are missing, print a clear message listing missing columns.
- Reuse the Aim 3 primary-results helper functions where possible:
  1) prepare_aim3_data()
  2) run_ols_model()
  3) fdr_correct_results()
  4) make_panel_a_regression_heatmap()
  5) make_forest_plot()
  6) save_outputs()
- Keep the same metric ordering, residualization approach, FDR logic, visual style, and output conventions as the primary Aim 3 figure.
- Make secondary analysis clearly labeled as supportive/exploratory, not confirmatory.
- Caption should state:
  “Panel A shows group-specific residualized clinical-neural associations for secondary clinical measures. Each cell includes only the corresponding group and clinical outcome; tile color encodes standardized beta. Panels B-C show standardized beta estimates with 95% confidence intervals for SAD-placebo and HC-placebo participants, respectively. Secondary analyses are intended as supportive evidence for the primary Aim 3 clinical relevance tests.”
