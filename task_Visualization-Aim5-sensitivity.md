Update `mvpa_l2.ipynb` to generate the sensitivity-analysis results figure and statistics table for Aim 5.

Figure title:
Figure 5—Supplement. Sensitivity analysis for oxytocin modulation of threat-safety neural profiles

Scientific goal:
Aim 5 tests whether oxytocin modulates neural profiles of vicarious safety/threat learning through Group × Drug effects. The sensitivity analysis should test whether the primary Aim 5 findings are robust across reasonable analytic choices, especially feature spaces/masks and model specifications, while keeping the same core interpretation: whether SAD-OXT shifts toward the HC-PBO reference.

Primary model repeated in sensitivity checks:
For each primary neural metric and each sensitivity specification, fit:

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

Sensitivity checks:

1. Feature-space / mask robustness
Run the same Aim 5 model in each available feature space:
- FearNetwork
- MemoryFearNetwork
- whole-brain or parcellation feature space, if already available, such as Schaefer/Tian

Use only feature spaces that are already implemented or available in the project. Do not create unsupported inputs.

2. Model-specification robustness
For the primary FearNetwork results, rerun the Aim 5 model under reasonable alternative specifications if the required columns are available:
- Base model: Group * Drug only
- Demographic-adjusted model: Group * Drug + age + sex/gender
- Run/order-adjusted model: Group * Drug + run/order/session covariates
- Full covariate model: Group * Drug + all available prespecified covariates
- Robust regression or outlier-robust model, if already supported
- Complete-case model
- Outlier-excluded model, excluding extreme neural metric values, for example absolute z > 3

Do not use SCR-defined learner subgroups as the main Aim 5 sensitivity analysis because Group × Drug subgroup cells may be too small. If SCR subgroup sensitivity is already implemented, report it only as optional descriptive output, not as a primary Aim 5 sensitivity claim.

Outputs:
1. A 3-panel sensitivity figure saved as:
   - Figure5_Aim5_sensitivity_oxytocin_modulation.svg
   - Figure5_Aim5_sensitivity_oxytocin_modulation.png

2. A sensitivity statistics table saved as:
   - Table_Aim5_sensitivity_oxytocin_modulation.csv

Figure layout:
Use a horizontal 1 × 3 layout.

Panel A: Group × Drug robustness heatmap across feature spaces
- Rows = the six primary neural metrics, in the order listed above.
- Columns = available feature spaces/masks.
- Cell value = standardized Group × Drug beta estimate from `neural_metric ~ Group * Drug + covariates`.
- Use a diverging color scale centered at 0.
- Print beta values inside cells.
- Add a visual indicator for whether the direction matches the primary FearNetwork result.
  Example:
  - add a small check mark or dot for direction match
  - leave blank or add an x for direction mismatch
- Label colorbar:
  Standardized Group × Drug β
- If q-values are available, optionally add a subtle marker for q-FDR < .05.

Panel B: HC-reference shift robustness heatmap across feature spaces
- Rows = the same six primary neural metrics.
- Columns = available feature spaces/masks.
- Cell value = HC-reference normalization index:

  |SAD-PBO − HC-PBO| − |SAD-OXT − HC-PBO|

- Positive values mean SAD-OXT moved closer to the HC-PBO reference.
- Negative values mean SAD-OXT moved farther from the HC-PBO reference.
- Use a diverging color scale centered at 0.
- Print normalization-index values inside cells.
- Label colorbar:
  HC-reference normalization index
- Add note:
  Positive = SAD-OXT closer to HC-PBO.

Panel C: Model-specification robustness forest plot
- Use the primary FearNetwork feature space.
- Summarize robustness across model specifications.
- Preferred option:
  Plot one row per model specification for a prespecified summary effect, such as the mean standardized Group × Drug beta across primary metrics, with bootstrap or analytic 95% CI if available.
- Alternative option if a summary effect is not appropriate:
  Plot one row per metric × model-specification combination, grouped by metric, but keep the panel compact.
- X-axis = standardized Group × Drug beta estimate.
- Add vertical dashed reference line at 0.
- Show point estimate and 95% CI.
- Label x-axis:
  Standardized Group × Drug effect (β)
- Add note:
  Points = β, whiskers = 95% CI.

Statistics table:
Create one row per metric × sensitivity check with columns:

- metric
- domain
  - Geometry
  - Certainty
  - Trajectory
- sensitivity_family
  - feature_space
  - model_specification
  - outlier_handling
  - complete_case
  - optional_scr_descriptive, only if used
- sensitivity_name
- feature_space
- model_specification
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
- normalization_index
- direction_matches_primary
- significant_after_fdr
- robustness_interpretation

Robustness interpretation rules:
- Robust:
  Direction matches the primary FearNetwork result and the uncertainty interval is reasonably consistent with the primary estimate.
- Directionally consistent:
  Direction matches the primary result, but the effect is weaker, less precise, or not FDR-significant.
- Inconsistent:
  Direction reverses relative to the primary result or the normalization index changes sign.
- Not estimable:
  Model cannot be fit because the metric, feature space, covariates, or cell sample sizes are unavailable.

HC-reference interpretation rules:
- Positive normalization index:
  SAD-OXT moved closer to HC-PBO.
- Near-zero normalization index:
  little evidence of HC-reference shift.
- Negative normalization index:
  SAD-OXT moved farther from HC-PBO.

Important coding requirements:
- Use Python.
- Use pandas, numpy, matplotlib, scipy/statsmodels if needed.
- Avoid seaborn unless the project already depends on it.
- Make the code robust to missing covariates, missing feature spaces, and unavailable model specifications.
- Do not invent real results.
- If real Aim 5 sensitivity results are unavailable, include a clearly separated mock-data mode for figure testing only.
- Label all mock outputs as “Illustrative mock data only”.
- Keep the code clean and easy to edit.
- Save outputs into the project’s figure/results directory if such paths already exist; otherwise create:
  - results/figures/
  - results/tables/

Visual style:
- Publication-quality.
- White background.
- Clear panel labels A, B, C.
- Compact but readable metric labels.
- Use consistent metric order across Panels A and B.
- Use symmetric heatmap color scales centered at 0.
- Use symmetric x-limits for Panel C when appropriate.
- Keep the sensitivity figure visually aligned with the primary Aim 5 figure.
- Include a concise caption string in the script describing:
  Panel A = Group × Drug beta robustness across feature spaces.
  Panel B = HC-reference shift robustness across feature spaces.
  Panel C = model-specification robustness of Group × Drug effects.
