Update the mvpa_l2.ipynb to generate the primary Aim 5 results figure and statistics table.

Figure title:
Figure 5. Oxytocin modulation of threat-safety neural profiles

Scientific goal:
Aim 5 tests whether oxytocin modulates neural profiles of vicarious safety/threat learning through Group × Drug effects, with special interest in whether SAD-OXT shifts toward the HC-PBO reference.

Primary model:
For each primary neural metric, fit:

neural_metric ~ Group * Drug + covariates

where:
- Group = SAD vs HC
- Drug = Placebo vs Oxytocin
- Primary effect of interest = Group × Drug interaction

Use available covariates only if present in the input data, for example:
- age
- sex or gender
- run/order/session covariates if already available

Primary neural metrics:
1. Neural_Dist_Safety_Background
2. Neural_Dist_Threat_Safety
3. Neural_SafetyEvidence
4. Neural_ThreatEvidence
5. Neural_Safety_Trajectory_Slope
6. Neural_Threat_Trajectory_Slope

Outputs:
1. A 3-panel figure saved as:
   - Figure5_Aim5_primary_oxytocin_modulation.svg
   - Figure5_Aim5_primary_oxytocin_modulation.png

2. A statistics table saved as:
   - Table_Aim5_primary_oxytocin_modulation.csv

Figure layout:
Use a horizontal 1 × 3 layout.

Panel A: Group × Drug interaction forest plot
- Rows = the six primary neural metrics, in the order listed above.
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
- Rows = the six primary neural metrics.
- Columns:
  HC-PBO, HC-OXT, SAD-PBO, SAD-OXT
- Values should be estimated marginal means or adjusted group means from the model.
- Z-score each metric before plotting so all metrics share one comparable scale.
- Use a diverging color scale centered at 0.
- Print numeric values inside cells.
- Label colorbar:
  z-scored estimated mean
- HC-PBO should serve as the visual reference condition.

Panel C: SAD shift toward HC-placebo reference
- Rows = the six primary neural metrics.
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
Create one row per primary metric with columns:

- metric
- domain
  - Geometry
  - Certainty
  - Trajectory
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
- interpretation

Interpretation rules:
- HC-reference shift:
  SAD-PBO differs from HC-PBO, and SAD-OXT moves closer to HC-PBO.
- General drug effect:
  oxytocin shifts SAD and HC in the same direction.
- SAD-specific modulation:
  oxytocin changes SAD more than HC, but not necessarily toward HC-PBO.
- No modulation:
  Group × Drug effect is weak or uncertain.

Important coding requirements:
- Use Python.
- Use pandas, numpy, matplotlib, scipy/statsmodels if needed.
- Avoid seaborn unless the project already depends on it.
- Make the script robust to missing covariates.
- Do not invent real results.
- If real Aim 5 results are unavailable, include a clearly separated mock-data mode for testing the figure only.
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
- Use consistent row order across panels.
- Use symmetric x-limits where appropriate.
- Use symmetric heatmap color scale centered at 0.
- Include a concise caption string in the script describing:
  Panel A = Group × Drug interaction effects.
  Panel B = estimated marginal means by Group × Drug.
  Panel C = SAD shift relative to HC-placebo reference.