Update the Aim 3 primary-results visualization in mvpa_l2.ipynb to generate:  
primary results visualization and statistics table.

Goal:
Figure 3 should show clinical relevance of neural vicarious learning profiles:
z(clinical_score) ~ z(neural_metric) + covariates
run separately within SAD-placebo and HC-placebo participants.

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

Figure layout:
Generate one 3-panel figure.

Panel A. Regression-annotated association heatmap
- Use a 6 x 4 matrix.
- Rows = six primary neural metrics.
- Columns:
  1) SAD: DASS anxiety
  2) SAD: LSAS total
  3) HC: DASS anxiety
  4) HC: LSAS total
- Each cell should show only the participants from that specific group/model.
- Do NOT overlay both SAD and HC in the same cell.
- Background tile color = standardized beta for that exact model.
- Use a diverging colormap centered at zero.
- Overlay faint residualized scatter points for that group only.
- Overlay one regression line for that group only.
- Add cell text:
  β = value
  q = value
  optional significance symbol: * q < .05; † p < .05 only.
- Residualize both x and y for covariates before plotting if covariates are used.
- Use shared x/y limits across cells.
- Keep points faint and small; regression line should be visually clear.

Panel B. SAD forest plot
- Include all six neural metrics.
- Show standardized beta ± 95% CI.
- Use separate markers for dass_anxiety and lsas_total.
- Add vertical reference line at β = 0.
- Use the same x-axis limits as Panel C.

Panel C. HC forest plot
- Same format as Panel B.
- Include all six neural metrics.
- Same x-axis scale as SAD forest plot.

Statistics:
Create a primary Aim 3 statistics table with one row per model:
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
- Apply FDR correction across the primary Aim 3 model family.
- Preferred: correct across all 24 primary tests:
  2 groups x 2 clinical outcomes x 6 neural metrics.
- Also make this easy to change if needed.

Outputs:
- Save figure as:
  Figure3_Aim3_primary_clinical_relevance.svg
  Figure3_Aim3_primary_clinical_relevance.png
- Save table as:
  Table_Aim3_primary_statistics.csv
  Table_Aim3_primary_statistics.xlsx

Implementation notes:
- Use pandas, numpy, scipy/statsmodels, matplotlib.
- Use real project data/results if available; do not invent values for final outputs.
- If required columns are missing, print a clear message listing missing columns.
- Keep code modular:
  1) prepare_aim3_data()
  2) run_ols_model()
  3) fdr_correct_results()
  4) make_panel_a_regression_heatmap()
  5) make_forest_plot()
  6) save_outputs()
- Use clean publication style, readable labels, and consistent metric ordering.
- Caption should state:
  “Panel A shows group-specific residualized clinical-neural associations. Each cell includes only the corresponding group; tile color encodes standardized beta. Panels B-C show standardized beta estimates with 95% confidence intervals.”