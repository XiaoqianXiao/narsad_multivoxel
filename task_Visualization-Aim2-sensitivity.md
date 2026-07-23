# Codex Task: Aim 2 Figure for `mvpa_l2.ipynb`

## Goal

Revise the Aim 2 visualization section of `mvpa_l2.ipynb`, or create a companion Python script, to generate a sensitivity-analysis figure for Aim 2.

Figure title:
Figure S2. Sensitivity analysis of SAD–HC differences in neural representations of vicarious learning

Scientific context:
Aim 2 tests whether SAD participants differ from HC participants in neural representations of vicarious safety/threat learning, including:
1. representational geometry
2. decision certainty
3. learning trajectories

The sensitivity analysis should test whether the Aim 2 SAD–HC effects are robust across:
1. different neural masks / feature spaces
2. different SCR-defined participant subgroups

The analysis plan defines sensitivity checks using alternative feature spaces such as MemoryFearNetwork and whole-brain/parcellation approaches, as well as SCR-defined responder or learner cohorts. :contentReference[oaicite:0]{index=0}

Goal:
Generate a single two-panel heatmap figure.

Panel A:
Robustness across masks / feature spaces

Panel B:
Robustness across participant subgroups

Use:
- Python
- pandas
- numpy
- matplotlib
- scipy if needed
- avoid seaborn unless absolutely necessary

Output files:
1. `FigureS2_Aim2_Sensitivity_RobustnessHeatmap.png`
2. `FigureS2_Aim2_Sensitivity_RobustnessHeatmap.svg`
3. `TableS2_Aim2_Sensitivity_Stats.csv`

Overall figure layout:
- 1 row x 2 columns
- Panel A on the left
- Panel B on the right
- Use the same row order and the same color scale in both panels
- Use a diverging color map centered at zero
- Color indicates SAD–HC standardized effect size
- Positive values mean SAD > HC
- Negative values mean SAD < HC
- White or neutral color means near-zero effect

Rows:
Use the Aim 2 neural metrics as rows, grouped by domain.

Primary metrics:
Geometry:
- Neural_Dist_Safety_Background
- Neural_Dist_Threat_Safety

Decision certainty:
- Neural_SafetyEvidence
- Neural_ThreatEvidence

Trajectory:
- Neural_Safety_Trajectory_Slope
- Neural_Threat_Trajectory_Slope

Optional secondary/supportive metrics, if available:
- Neural_Dist_Threat_Background
- Neural_Decoder_Entropy_CSS
- Neural_Decoder_Entropy_CSR
- Shock_Anchor_Trajectory_Slope
- Residualized_Shock_Anchor_Trajectory_Slope

Panel A columns:
Different masks / feature spaces:
- FearNetwork
- MemoryFearNetwork
- Schaefer
- Tian
- WholeBrain

Panel B columns:
Participant subgroups:
- AllPlacebo
- SCR_Physiological_Responder
- SCR_Simple_Acquisition_Differential_Learner
- SCR_Habituation_Adjusted_Learner
- SCR_Late_Phase_Sensitivity_Learner

Input data:
The script should expect a long-format CSV file named:

`aim2_sensitivity_results.csv`

Required columns:
- metric_family
- metric_name
- sensitivity_type
- specification
- n_sad
- n_hc
- effect_size
- ci_low
- ci_high
- p_value
- q_value

Definitions:
- `metric_family` should be one of: Geometry, Certainty, Trajectory, Secondary
- `sensitivity_type` should be either: Mask or Subgroup
- `specification` should identify the mask or subgroup
- `effect_size` should be standardized SAD–HC effect size, such as Cohen's d or standardized beta
- `q_value` should be FDR-corrected p-value if available

If `aim2_sensitivity_results.csv` is not found:
- create a small mock dataset inside the script
- clearly label it as mock data
- still generate the figure and table
- do not make the mock values look like real results

Statistical annotation:
Overlay small symbols on each heatmap cell:
- `†` if q_value < 0.05
- `*` if p_value < 0.05 but q_value >= 0.05
- no symbol otherwise

Also optionally display the rounded effect size in each cell, for example:
`+0.42†`
`-0.35*`
`+0.08`

Design requirements:
- Make the figure clean and manuscript-ready
- Use readable font sizes
- Rotate x-axis labels if needed
- Keep metric labels readable
- Add horizontal group dividers between:
  1. Geometry
  2. Certainty
  3. Trajectory
  4. Secondary, if included
- Label the two panels clearly:
  - A. Across masks / feature spaces
  - B. Across SCR-defined participant subgroups
- Add one shared colorbar labeled:
  `SAD–HC effect size`
- Add a small note below the figure:
  `Positive values indicate SAD > HC; negative values indicate SAD < HC. † FDR q < .05; * nominal p < .05.`

Table output:
Generate `TableS2_Aim2_Sensitivity_Stats.csv` containing:
- metric_family
- metric_name
- sensitivity_type
- specification
- n_sad
- n_hc
- effect_size
- ci_low
- ci_high
- p_value
- q_value
- direction
- robustness_label

Where:
- direction = `SAD > HC`, `SAD < HC`, or `Near zero`
- robustness_label =
  - `FDR significant` if q_value < 0.05
  - `Nominal` if p_value < 0.05 and q_value >= 0.05
  - `Direction only` if effect size is not near zero but p_value >= 0.05
  - `Weak / inconsistent` if effect size is near zero

Implementation details:
- Write modular functions:
  1. `load_or_create_data()`
  2. `prepare_heatmap_matrix()`
  3. `make_annotation_matrix()`
  4. `plot_heatmap_panel()`
  5. `make_figure()`
  6. `write_summary_table()`
- Include clear comments
- Use deterministic mock data with a fixed random seed
- Save outputs to the current working directory
- Ensure the script can run from command line:
