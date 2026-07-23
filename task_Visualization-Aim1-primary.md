Update the Aim 1 primary-results visualization in mvpa_l2.ipynb to generate:  
1) a publication-quality 2 × 2 figure matching the current preferred layout, and  
2) a companion statistics table for Aim 1 primary results.  
  
Figure title:  
Aim 1: FearNetwork group-specific neural representation identification  
  
Scientific goal:  
Aim 1 tests whether CSR versus CSS can be decoded within the FearNetwork separately in SAD-placebo and HC-placebo participants, and whether the resulting neural representations show functional and spatial group specificity.  
  
Important:  
Do not redesign the main figure concept. Keep the same structure as the provided example:  
- Top-left: SAD-placebo self-decoding permutation-null plot  
- Top-right: HC-placebo self-decoding permutation-null plot  
- Bottom-left: Functional specificity cross-group 2AFC generalization heatmap  
- Bottom-right: Spatial specificity discrimination-weight similarity heatmap  
  
--------------------------------  
PART 1 — MAIN FIGURE  
--------------------------------  
  
Panel A / top-left:  
SAD-placebo self-decoding  
- Plot permutation-null distribution of 2AFC accuracy.  
- Show gray histogram and smooth density curve if available.  
- Add vertical dotted black line at chance = 0.50.  
- Add vertical dashed blue line for the 95% null threshold.  
- Add vertical solid red line for observed SAD-placebo 2AFC accuracy.  
- Title format:  
SAD-Placebo self-decoding  
2AFC acc; n_perm=<n>; p=<p>  
- X-axis: Subject-level forced-choice accuracy  
- Y-axis: Density  
- Legend should include:  
95% null  
Chance  
Observed <value>  
Null Dist  
  
Panel B / top-right:  
HC-placebo self-decoding  
- Same format as SAD panel.  
- Use HC-placebo null distribution, observed accuracy, 95% null threshold, and p-value.  
- Keep visual style consistent with Panel A.  
  
Panel C / bottom-left:  
Functional Specificity  
Cross-Group 2AFC Generalization  
- Create a 2 × 2 heatmap.  
- Rows:  
Train SAD  
Train HC  
- Columns:  
Test SAD  
Test HC  
- Cell values should be subject-level 2AFC accuracy:  
Train SAD → Test SAD = SAD self-decoding  
Train SAD → Test HC = SAD-trained / HC-tested cross-decoding  
Train HC → Test SAD = HC-trained / SAD-tested cross-decoding  
Train HC → Test HC = HC self-decoding  
- Annotate each cell with the accuracy rounded to 3 decimals.  
- Add significance marker under the value when available, using the same compact style as the example:  
(*)  
()  
- Colorbar label:  
Subject-level 2AFC accuracy  
- Use a warm sequential/diverging palette similar to the example.  
- Use a color range approximately 0.30 to 0.90 unless the existing code already defines a better range.  
  
Panel D / bottom-right:  
Spatial Specificity  
Discrimination-Weight Similarity  
- Keep this as a 2 × 2 cosine-similarity heatmap, not a null-distribution plot.  
- Rows:  
SAD Map  
HC Map  
- Columns:  
SAD Map  
HC Map  
- Diagonal cells should be 1.000.  
- Off-diagonal cells should show SAD-HC cosine similarity.  
- Annotate each cell with the cosine similarity rounded to 3 decimals.  
- Add significance marker under each value if available, following the same compact style as the example.  
- Colorbar label:  
Cosine similarity  
- Use a diverging color scale from -1.00 to 1.00.  
  
Style requirements:  
- Overall layout: 2 rows × 2 columns.  
- Large figure size suitable for manuscript or slide export.  
- Add enough spacing between panels so titles and colorbars do not overlap.  
- Keep the clean visual style of the example.  
- Use consistent font sizes across panels.  
- Keep the top-panel legends inside the axes, upper-left.  
- Use red for observed accuracy lines, blue dashed for 95% null, black dotted for chance, and gray for null distribution.  
- Save outputs as both:  
figures/aim1_primary_results.png  
figures/aim1_primary_results.svg  
  
--------------------------------  
PART 2 — COMPANION STATISTICS TABLE  
--------------------------------  
  
Also create a clean companion statistics table for Aim 1 primary results.  
  
Goal of the table:  
Provide the exact numerical results that support the 4 figure panels, without overcrowding the figure.  
  
Create a pandas DataFrame named:  
aim1_primary_stats_table  
  
Include one row per primary result entry.  
  
Recommended rows:  
1. SAD self-decoding  
2. HC self-decoding  
3. Train SAD → Test SAD  
4. Train SAD → Test HC  
5. Train HC → Test SAD  
6. Train HC → Test HC  
7. SAD vs HC discrimination-weight similarity  
  
Required columns:  
- analysis_block  
- comparison  
- train_group  
- test_group  
- metric  
- observed_value  
- null_mean  
- null_95_threshold_low  
- null_95_threshold_high  
- chance_value  
- p_perm  
- q_fdr  
- n_subjects  
- n_perm  
- significance_label  
  
Column guidance:  
- analysis_block:  
one of ["self_decoding", "functional_specificity", "spatial_specificity"]  
- comparison:  
human-readable label, e.g.  
"SAD self-decoding"  
"HC self-decoding"  
"Train SAD → Test HC"  
"Train HC → Test SAD"  
"SAD vs HC weight similarity"  
- train_group / test_group:  
fill when applicable; otherwise use NA  
- metric:  
"2AFC accuracy" or "Cosine similarity"  
- observed_value:  
main statistic shown in the figure  
- null_mean:  
permutation-null mean if available  
- null_95_threshold_low / null_95_threshold_high:  
the 95% null interval or threshold bounds if available  
- chance_value:  
0.50 for decoding analyses; NA for cosine similarity if not applicable  
- p_perm:  
permutation p-value  
- q_fdr:  
FDR-adjusted p if available; otherwise compute it across the confirmatory Aim 1 tests if those tests are part of the current notebook workflow  
- n_subjects:  
sample size used in that analysis  
- n_perm:  
number of permutations  
- significance_label:  
compact display string such as:  
"*"  
"**"  
"ns"  
or the exact compact label already used in the heatmap  
  
Formatting requirements:  
- Round observed_value, null_mean, null thresholds to 3 decimals.  
- Format p-values clearly:  
e.g., "<0.0001" or "0.0010"  
- Use readable column names in the final exported table if needed, but keep the internal DataFrame name stable.  
  
Export the table to:  
- figures/aim1_primary_stats_table.csv  
- figures/aim1_primary_stats_table.xlsx  
  
Also display the table in the notebook in a readable format.  
  
Optional but preferred:  
- Render a publication-friendly table image as:  
figures/aim1_primary_stats_table.png  
This can be a simple matplotlib table or a clean dataframe-to-figure export.  
Do not over-style it.  
  
--------------------------------  
PART 3 — OPTIONAL SUMMARY TABLE FOR MANUSCRIPT USE  
--------------------------------  
  
If feasible, also create a second compact manuscript-ready version of the table with fewer columns.  
  
Suggested columns for the compact display table:  
- Result  
- Metric  
- Observed  
- Null / Chance reference  
- p_perm  
- q_fdr  
- N  
  
Suggested rows:  
- SAD self-decoding  
- HC self-decoding  
- Train SAD → Test SAD  
- Train SAD → Test HC  
- Train HC → Test SAD  
- Train HC → Test HC  
- SAD vs HC weight similarity  
  
Name this compact table:  
aim1_primary_stats_table_compact  
  
Export to:  
- figures/aim1_primary_stats_table_compact.csv  
- figures/aim1_primary_stats_table_compact.xlsx  
  
--------------------------------  
IMPORTANT CONSTRAINTS  
--------------------------------  
  
- Do not change the underlying analysis, model fitting, permutation testing, or decoding results.  
- Only update the visualization code and any helper functions needed for plotting and table generation.  
- Reuse existing result variables/tables whenever possible.  
- Do not hard-code values unless the notebook currently lacks structured result variables; if hard-coding is unavoidable, clearly mark the block as temporary/example values.  
- Make sure the code runs from a clean notebook restart.  
- Make sure the figure and table can be regenerated reproducibly.  
  
After updating the notebook:  
1. Run the Aim 1 visualization/statistics cell(s).  
2. Confirm the figure and table files are saved successfully.  
3. Print the output paths.  
4. Briefly summarize what cells/functions were changed.