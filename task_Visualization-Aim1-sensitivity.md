Task:
Create a publication-quality Python script to generate the Aim 1 sensitivity analysis figure and companion statistics table.

Output files:
1. Figure_S1_Aim1_sensitivity.png
2. Figure_S1_Aim1_sensitivity.svg
3. Table_S1_Aim1_sensitivity_statistics.csv
4. Table_S1_Aim1_sensitivity_statistics.xlsx if openpyxl is available

Scientific goal:
Aim 1 tests whether CSR and CSS can be decoded within the placebo condition separately for SAD and HC, and whether the result shows functional and spatial specificity. This sensitivity figure should show whether the Aim 1 findings are robust across alternative feature spaces and SCR-defined responder/learner cohorts.

Important:
Do not invent numeric values.
Read values from existing Aim 1 sensitivity result files if available.
If required columns are missing, stop with a clear error message listing the expected columns.

Figure title:
Figure S1. Aim 1 sensitivity analysis: robustness of CSR–CSS decoding

Figure layout:
Use a clean 3-panel layout.

Panel A. Within-group decoding robustness heatmap
Purpose:
Show whether SAD and HC decoding accuracy remains above chance across sensitivity settings.

Rows:
- FearNetwork primary
- MemoryFearNetwork
- Schaefer/Tian parcellation
- Whole brain, if available
- SCR_Physiological_Responder
- SCR_Simple_Acquisition_Differential_Learner
- SCR_Habituation_Adjusted_Learner
- SCR_Late_Phase_Sensitivity_Learner

Columns:
- SAD
- HC

Cell value:
accuracy

Cell labels:
Show accuracy, for example 0.61.
Add significance stars based on FDR q if available:
q < .01 = **
q < .05 = *
Otherwise no star.
If q is unavailable, use permutation p and state this in the figure note.

Panel A visual style:
Use a sequential heatmap.
Include colorbar labeled “Accuracy”.
Include a short note:
“Cell = 2AFC accuracy; stars indicate FDR-corrected significance.”

Panel B. Functional specificity robustness
Purpose:
Show whether within-group/self-decoding is consistently stronger than cross-group decoding across robustness and sensitivity settings.

The key statistical readout is the paired functional-specificity effect:

`self_minus_cross = within_group_accuracy - cross_group_accuracy`

A positive value means that decoding is stronger when the model is trained and tested within the same diagnostic group than when a model trained in the other group is tested on that target group.

Important interpretation rule:
Do not infer significance from the visual separation or overlapping uncertainty of the two absolute accuracy estimates. Significance must come from the paired self-minus-cross comparison.

Sensitivity settings:
Include both feature-space/parcellation robustness and SCR-defined cohort sensitivity settings.

Feature-space/parcellation settings:
- FearNetwork primary
- MemoryFearNetwork
- Schaefer/Tian parcellation
- Whole brain, if available

SCR-defined cohort settings:
- SCR_Physiological_Responder
- SCR_Simple_Acquisition_Differential_Learner
- SCR_Habituation_Adjusted_Learner
- SCR_Late_Phase_Sensitivity_Learner

Plot type:
Use a paired dumbbell plot plus an adjacent paired-drop null-distribution plot.

Panel B layout:
Panel B should have two target-group blocks:
- SAD target group
- HC target group

Within each target-group block, use two aligned subcolumns:
1. Absolute accuracy dumbbell plot
2. Paired self-minus-cross null-distribution plot

Recommended layout:

```text
Panel B. Functional specificity robustness

SAD target group
[accuracy dumbbell: cross vs self]    [upper null histogram: observed self − cross]

HC target group
[accuracy dumbbell: cross vs self]    [upper null histogram: observed self − cross]
```

Each accuracy dumbbell and paired-drop null-distribution plot must use the same ordered sensitivity settings on the y-axis.

For each sensitivity setting in the accuracy dumbbell plot:
- Plot one point for `cross_group_accuracy`.
- Plot one point for `within_group_accuracy` / self-decoding accuracy.
- Connect the two points with a horizontal dumbbell line.
- Add a vertical dashed reference line at 2AFC accuracy = 0.50 to mark chance-level decoding.
- Use consistent colors or marker shapes for Cross-group and Within-group/Self-decoding.
- Do not plot subject-level dots.
- Do not plot fold/bootstrap/resample-level raw dots in the main figure unless specifically requested; the goal of Panel B is a compact robustness summary.
- Avoid long text labels on top of the dumbbell lines. Keep the dumbbell panel focused on absolute accuracy values.

For each sensitivity setting in the paired-drop null-distribution plot:
- Upper gray histogram = paired sign-flip null distribution of `self_minus_cross`, if available.
- Gray horizontal line = central 95% interval of the sign-flip null distribution, if available.
- Gray tick = sign-flip null mean, if available.
- Dot = observed `self_minus_cross`.
- Vertical reference line = 0.
- Add a compact significance marker next to the dot if a paired permutation p-value or FDR-corrected q-value is available.
- Use FDR q for stars when available; otherwise use permutation p and state this in the figure note.
- If the full sign-flip null distribution is unavailable but a null interval is available, show the gray null interval and null mean tick only.
- If only aggregate within-group and cross-group accuracies are available, show the observed dot only and state in the figure note that the sign-flip null distribution was unavailable for that setting.
- If space allows, label the drop column with compact values such as `Δ = 0.23*`; otherwise report exact values in Table S1.

Important interpretation rule:
The gray null interval is not a confidence interval around the observed self-minus-cross effect. It is the central interval of the paired sign-flip null distribution. Significance is read from the paired sign-flip p-value and FDR q-value, not from overlap with a bootstrap confidence interval.

Recommended ordering:
List feature-space/parcellation settings first, followed by SCR-defined cohort settings. Add a subtle horizontal separator or small group label between:
1. Feature-space robustness
2. SCR cohort sensitivity

Panel B labels:
- Accuracy subcolumn x-axis = “2AFC accuracy”
- Accuracy subcolumn vertical reference line = 0.50
- Drop/null subcolumn x-axis = “Self − cross 2AFC accuracy”
- Drop subcolumn vertical reference line = 0
- Y-axis = “Sensitivity setting”
- Subplot titles = “SAD target group” and “HC target group”
- Legend = “Cross-group” and “Within-group/self”

Panel B significance:
The significance marker in Panel B must reflect the paired self-minus-cross test, not the significance of within-group decoding above chance and not the significance of cross-group decoding above chance.

Use:
- `**` for FDR q < .01
- `*` for FDR q < .05
- If q is unavailable, use permutation p and clearly mark this in the figure note.

Suggested visual encoding:
- Dumbbell connecting line should remain thin and neutral.
- The observed self-minus-cross dot can be emphasized because it is the inferential readout.
- Use a filled or larger drop dot for significant paired drops if helpful.
- Keep exact p/q values in the companion table rather than crowding the figure.

Companion-table value:
Report the paired specificity effect in the companion statistics table:

`self_minus_cross = within_group_accuracy - cross_group_accuracy`

Keep the existing `functional_drop` column if already used in the project, but define it clearly as `within_group_accuracy - cross_group_accuracy`.

Figure note for Panel B:
“Panel B shows absolute cross-group and within-group/self-decoding accuracies as paired dumbbell plots and the paired functional-specificity effect as adjacent sign-flip null-distribution plots. The paired effect is defined as within-group/self-decoding accuracy minus cross-group decoding accuracy. Values greater than 0 indicate stronger within-group than cross-group decoding. Significance markers refer to the paired self-minus-cross sign-flip test, not to either absolute accuracy alone. Upper gray histograms show the paired sign-flip null distribution when available; gray lines show the null central 95% interval and gray ticks show the null mean.”

Panel C. Spatial specificity robustness
Purpose:
Show whether SAD-HC classifier/Haufe-map similarity remains low or nonsignificant across feature spaces and SCR-defined cohort sensitivity settings.

Rows:
- FearNetwork primary
- MemoryFearNetwork
- Schaefer/Tian parcellation
- Whole brain, if available
- SCR_Physiological_Responder
- SCR_Simple_Acquisition_Differential_Learner
- SCR_Habituation_Adjusted_Learner
- SCR_Late_Phase_Sensitivity_Learner

X-axis:
SAD–HC cosine similarity

Plot type:
Compact row-wise upper-half null histogram with observed similarity overlay.

For each row:
- Upper gray histogram = permutation null distribution, if available
- Gray horizontal line = permutation null 95% interval
- Gray tick = permutation null mean
- Orange dot = observed SAD–HC cosine similarity
- Vertical reference line at 0

If the full permutation null distribution is unavailable but null summary statistics are available:
- Show the gray null 95% interval and null mean tick only.
- State in the figure note that histograms are shown when the null distribution is available.

Important interpretation rule:
The null interval is not a confidence interval around the observed cosine similarity. It is the 95% interval of the permutation null distribution. The observed orange dot may be off-center relative to the null interval; this is expected.

Panel C labels:
X-axis = “SAD–HC cosine similarity”
Note:
“Upper gray histogram = permutation null; gray line = null 95% interval; gray tick = null mean; orange dot = observed similarity.”

Companion table:
Create Table S1. Aim 1 sensitivity statistics.

Rows should include all tests shown in the figure.

Required columns:
- analysis_family
  Values: within_group_decoding, functional_specificity, spatial_specificity
- sensitivity_set
- group_or_test
  Examples: SAD, HC, within_group, cross_group, SAD_vs_HC_cosine
- target_group
  Required for functional_specificity rows; values: SAD, HC
- feature_space
- cohort
  Examples: full_placebo, SCR_Physiological_Responder, SCR_Simple_Acquisition_Differential_Learner
- n_subjects
- n_trials
- accuracy
- accuracy_minus_chance
- cross_group_accuracy
- within_group_accuracy
- self_minus_cross
  Definition: within_group_accuracy - cross_group_accuracy
- functional_drop
  Same as `self_minus_cross`; keep this column only if already used in the project
- self_minus_cross_ci_low
- self_minus_cross_ci_high
- functional_drop_ci_low
  Same as `self_minus_cross_ci_low`; keep this column only if already used in the project
- functional_drop_ci_high
  Same as `self_minus_cross_ci_high`; keep this column only if already used in the project
- self_minus_cross_p
  Paired-test p-value for within/self minus cross
- permutation_p
  Use for paired self-minus-cross test in functional_specificity rows if no separate `self_minus_cross_p` column exists
- self_minus_cross_fdr_q
  FDR-corrected q-value for the paired self-minus-cross test
- fdr_q
  Use for paired self-minus-cross test in functional_specificity rows if no separate `self_minus_cross_fdr_q` column exists
- cosine_similarity
- null_mean
- null_ci_low
- null_ci_high
- robust
  Values: Yes, Directionally consistent, No, Not estimable
- interpretation

Table rules:
For within_group_decoding:
- Fill accuracy and accuracy_minus_chance.
- robust = Yes if accuracy > 0.50 and fdr_q < .05.
- If q is unavailable, use permutation_p < .05 but mark interpretation as uncorrected.

For functional_specificity:
- Fill target_group, within_group_accuracy, cross_group_accuracy, self_minus_cross, self_minus_cross_ci_low, self_minus_cross_ci_high, self_minus_cross_p, and self_minus_cross_fdr_q when available.
- `self_minus_cross` must equal `within_group_accuracy - cross_group_accuracy`.
- If using the existing project column `functional_drop`, it must be identical to `self_minus_cross`.
- robust = Yes if self_minus_cross > 0 and the paired self-minus-cross test is significant after FDR correction when q is available.
- robust = Yes if self_minus_cross > 0 and paired permutation p < .05 when q is unavailable, but mark interpretation as uncorrected.
- robust = Directionally consistent if self_minus_cross > 0 but the CI overlaps 0 or the paired test is not significant.
- robust = No if self_minus_cross <= 0.
- robust = Not estimable if the required within-group or cross-group accuracy is unavailable.

For spatial_specificity:
- Fill cosine_similarity, null_mean, null_ci_low, null_ci_high, permutation_p, and fdr_q if available.
- robust = Yes if the observed result supports the same spatial-specificity interpretation as the primary analysis.

Implementation requirements:
- Use Python, pandas, numpy, matplotlib.
- Avoid seaborn and unnecessary dependencies.
- Save high-resolution PNG and SVG.
- Make fonts readable for publication.
- Use consistent ordering across figure and table.
- Do not overplot raw points in Panel A.
- Keep Panel A as the visual anchor.
- Keep Panel B and Panel C compact.
- Add a clear caption string in the script.
- In Panel B, keep the dumbbell plot and the paired-drop forest plot visually aligned by sensitivity setting.
- In Panel B, the paired-drop null-distribution plot is required, not optional when the sign-flip null distribution is available.

Expected input schema:
The script should either adapt to existing result-file names or define a clear input section at the top.

Minimum expected columns for within-group decoding:
- analysis_family
- sensitivity_set
- group
- feature_space
- cohort
- n_subjects
- n_trials
- accuracy
- permutation_p
- fdr_q

Minimum expected columns for functional specificity:
- analysis_family
- sensitivity_set
- target_group
- feature_space
- cohort
- within_group_accuracy
- cross_group_accuracy
- self_minus_cross
- self_minus_cross_ci_low
- self_minus_cross_ci_high
- self_minus_cross_p
- self_minus_cross_fdr_q

Acceptable aliases for functional specificity:
- `functional_drop` may be used instead of `self_minus_cross`
- `functional_drop_ci_low` may be used instead of `self_minus_cross_ci_low`
- `functional_drop_ci_high` may be used instead of `self_minus_cross_ci_high`
- `permutation_p` may be used instead of `self_minus_cross_p`
- `fdr_q` may be used instead of `self_minus_cross_fdr_q`

Minimum expected columns for spatial specificity:
- analysis_family
- sensitivity_set
- feature_space
- cosine_similarity
- null_mean
- null_ci_low
- null_ci_high
- permutation_p
- fdr_q

If the current project uses different column names, map them explicitly near the top of the script using a COLUMN_MAP dictionary.

Caption:
Figure S1. Sensitivity analysis for Aim 1. Panel A shows robustness of within-group CSR–CSS decoding separately for SAD and HC across alternative feature spaces and SCR-defined responder/learner cohorts. Cell values represent 2AFC accuracy. Panel B shows functional specificity using paired dumbbell plots of cross-group versus within-group/self-decoding accuracy, separately for SAD and HC target groups, with adjacent upper-half histograms of the paired sign-flip null distribution for the self-minus-cross effect. Significance markers in Panel B refer to the paired self-minus-cross sign-flip test. Panel C shows spatial specificity using upper-half histograms of the permutation null distribution, null 95% intervals, null means, and observed SAD–HC classifier/Haufe-map cosine similarity. SCR-defined cohort analyses are interpreted as sensitivity checks rather than confirmatory primary tests.
