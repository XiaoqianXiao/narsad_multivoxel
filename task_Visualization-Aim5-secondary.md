Update `mvpa_l2.ipynb` to generate the secondary Aim 5 results figure and statistics table.

This task should follow the same figure logic and visual style as `task_Visualization-Aim5-primary.md`, but use secondary/supportive neural metrics instead of the primary Aim 2 neural metrics.

Figure title:
Figure 5—Supplement. Secondary evidence for oxytocin modulation of threat-safety neural profiles

Scientific goal:
Aim 5 secondary analysis tests whether oxytocin modulates supporting neural features of vicarious safety/threat learning through Group × Drug effects. The key interpretation remains whether SAD-OXT shifts toward the HC-PBO reference, but these analyses should be framed as secondary/supportive evidence rather than primary claims.

Primary model for secondary metrics:
For each secondary neural metric, fit:

neural_metric ~ Group * Drug + covariates

where:
- Group = SAD vs HC
- Drug = Placebo vs Oxytocin
- Primary effect of interest = Group × Drug interaction

Use available covariates only if present in the input data, for example:
- age
- sex or gender
- run/order/session covariates if already available

Secondary/supportive neural metrics:

Preferred metrics, in this order:

1. Neural_Dist_Threat_Background
2. Neural_Decoder_Entropy_CSS
3. Neural_Decoder_Entropy_CSR
4. ShockAnchor_Safety_Trajectory_Slope
5. ShockAnchor_Threat_Trajectory_Slope
6. Residualized_ShockAnchor_Safety_Slope
7. Residualized_ShockAnchor_Threat_Slope

Metric domains:
- Geometry:
  - Neural_Dist_Threat_Background
- Entropy / uncertainty:
  - Neural_Decoder_Entropy_CSS
  - Neural_Decoder_Entropy_CSR
- Shock-anchor trajectory:
  - ShockAnchor_Safety_Trajectory_Slope
  - ShockAnchor_Threat_Trajectory_Slope
- Residualized shock-anchor trajectory:
  - Residualized_ShockAnchor_Safety_Slope
  - Residualized_ShockAnchor_Threat_Slope

Important metric-handling rule:
If exact shock-anchor metric names differ in the existing notebook/data, use the existing columns that correspond to:
- shock-anchor safety trajectory slope
- shock-anchor threat trajectory slope
- residualized shock-anchor safety trajectory slope
- residualized shock-anchor threat trajectory slope

Do not fail only because these column names differ. Search for close existing column names and map them explicitly in the code. If a metric is not available, skip it with a clear warning and record the skipped metric in the notebook output.

Outputs:

1. A 3-panel figure saved as:
   - Figure5_Aim5_secondary_oxytocin_modulation.svg
   - Figure5_Aim5_secondary_oxytocin_modulation.png

2. A statistics table saved as:
   - Table_Aim5_secondary_oxytocin_modulation.csv

Figure layout:
Use the same horizontal 1 × 3 layout as the Aim 5 primary figure.

Panel A: Group × Drug interaction forest plot
- Rows = secondary neural metrics, in the order listed above.
- Group rows visually by domain:
  - Geometry
  - Entropy / uncertainty
  - Shock-anchor trajectory
  - Residualized shock-anchor trajectory
- X-axis = standardized Group × Drug beta estimate.
- Show point estimate and 95% CI.
- Add a vertical dashed reference line at 0.
- Label x-axis:
  Standardized Group × Drug effect (β)
- Add small note:
  Points = β, whiskers = 95% CI
- Visually distinguish FDR-significant and non-significant effects if q-values are available.
  Example:
  - filled marker = q < .05
  - open marker = q ≥ .05

Panel B: Estimated marginal means heatmap
- Rows = secondary neural metrics.
- Columns:
  HC-PBO, HC-OXT, SAD-PBO, SAD-OXT
- Values should be estimated marginal means or adjusted group means from the model.
- Z-score each metric before plotting so all metrics share one comparable scale.
- Use a diverging color scale centered at 0.
- Print numeric values inside cells.
- Label colorbar:
  z-scored estimated mean
- HC-PBO should serve as the visual reference condition.
- Use the same row order as Panel A.

Panel C: SAD shift toward HC-placebo reference
- Rows = secondary neural metrics.
- X-axis = difference from HC-PBO reference, z-scored.
- Add vertical dashed line at 0 labeled:
  HC-PBO reference
- For each metric, plot:
  - SAD-PBO difference from HC-PBO
  - SAD-OXT difference from HC-PBO
- Connect SAD-PBO to SAD-OXT with an arrow.
- The arrow should show whether SAD-OXT moves toward or away from HC-PBO.
- Use the same row order as Panels A and B.
- Add legend:
  SAD-PBO
  SAD-OXT

Statistics table:
Create one row per available secondary metric with columns:

- metric
- domain
  - Geometry
  - Entropy / uncertainty
  - Shock-anchor trajectory
  - Residualized shock-anchor trajectory
- n_total
- n_HC_PBO
- n_HC_OXT
- n_SAD_PBO
- n_SAD_OXT
- beta_group
- beta_drug
- beta_group_x_drug
- ci95_low_group_x_drug
- ci95_high_group_x_drug
- p_group_x_drug
- q_fdr_group_x_drug
- emm_HC_PBO
- emm_HC_OXT
- emm_SAD_PBO
- emm_SAD_OXT
- SAD_PBO_minus_HC_PBO
- SAD_OXT_minus_HC_PBO
- shift_toward_HC_reference
- normalization_index
- interpretation

Normalization index:
Compute a secondary normalization index for each metric:

normalization_index = abs(SAD_PBO_minus_HC_PBO) - abs(SAD_OXT_minus_HC_PBO)

Interpretation:
- positive value = SAD-OXT moved closer to HC-PBO
- zero or near zero = little directional shift
- negative value = SAD-OXT moved farther from HC-PBO

Interpretation rules:
- HC-reference shift:
  SAD-PBO differs from HC-PBO, and SAD-OXT moves closer to HC-PBO.
- General drug effect:
  oxytocin shifts SAD and HC in the same direction.
- SAD-specific modulation:
  oxytocin changes SAD more than HC, but not necessarily toward HC-PBO.
- No modulation:
  Group × Drug effect is weak or uncertain.
- No normalization:
  SAD-OXT does not move closer to HC-PBO, even if the metric changes.

Important coding requirements:
- Use Python.
- Use pandas, numpy, matplotlib, scipy/statsmodels if needed.
- Avoid seaborn unless the project already depends on it.
- Match the primary Aim 5 figure structure and style as closely as possible.
- Make the script robust to missing covariates.
- Make the script robust to missing secondary metric columns.
- Do not invent real results.
- If real Aim 5 secondary results are unavailable, include a clearly separated mock-data mode for testing the figure only.
- Label all mock outputs as “Illustrative mock data only”.
- Keep the code clean and easy to edit.
- Save outputs into the project’s figure/results directory if such paths already exist; otherwise create:
  results/figures/
  results/tables/

Visual style:
- Publication-quality.
- White background.
- Clear panel labels A, B, C.
- Compact but readable metric labels.
- Use consistent row order across all panels and the table.
- Use domain grouping or subtle separators to make secondary metric families easy to read.
- Use symmetric x-limits where appropriate.
- Use symmetric heatmap color scale centered at 0.
- Do not over-interpret secondary results as primary evidence.
- Include a concise caption string in the script describing:
  Panel A = Group × Drug interaction effects for secondary metrics.
  Panel B = estimated marginal means by Group × Drug.
  Panel C = SAD shift relative to HC-placebo reference.
