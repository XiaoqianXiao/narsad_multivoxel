## **Task:**  
Create the **Aim 1 secondary supportive evidence figure and companion table** for the Haufe-transform analysis.

## Goal

Generate a publication-quality **3-row composite figure** plus a **companion summary table** that matches the analysis plan for **Aim 1 secondary evidence: distinct distribution of Haufe transform score**.

## Figure title

Use an Aim 1 title, **not Aim 2**.  
Suggested overall title:

**Aim 1 secondary evidence: distinct distribution of Haufe-transformed scores**

## Figure layout

Create a **3-row figure** with the following panels:

### Row A

**A. SAD FearNetwork Haufe pattern (ROI-FDR, n=53)**

- Show the SAD Haufe-transformed spatial pattern.
- Use the existing SAD brain-map visualization style.
- Keep the multi-view brain layout and a colorbar.

### Row B

**B. HC FearNetwork Haufe pattern (top 53 matched voxels)**

- Show the HC Haufe-transformed spatial pattern.
- Use the same multi-view brain layout and a colorbar.
- HC should display the **top 53 matched voxels** for comparability with the SAD ROI-FDR count.

### Row C

**C. ROI distribution of displayed voxels across FearNetwork ROIs**

- Show the ROI distribution summary comparing SAD vs HC.
- Use the y-axis:

**Displayed voxels (% of ROI)**

- Use the x-axis:

**FearNetwork ROI**

- Compare **SAD vs HC** clearly for each ROI.
- A grouped bar plot is acceptable, but keep it clean and publication quality.
- Include value labels if they remain readable.

## ROI ordering

Order the ROIs by **anatomical family**, in this order:

1. hippocampus
2. amygdala
3. insula
4. ACC
5. vmPFC

Within each family, order left then right:

- left_hippocampus
- right_hippocampus
- left_amygdala
- right_amygdala
- left_insula
- right_insula
- left_acc
- right_acc
- left_vmpfc
- right_vmpfc

## Color scale requirement

If possible, use the **same symmetric color scale** for SAD and HC Haufe maps.

- Prefer a shared symmetric range centered at zero.
- If this is feasible, use it.

If it is **not possible or not visually appropriate**, then do **not force it**.  
Instead, make this explicit in the **figure caption**:

- the maps are shown for **within-group visualization**
- comparative interpretation should focus **primarily on Row C**

## Caption requirement

Write a caption that matches the analysis plan and does **not** overstate inference.

The caption should communicate that:

- Rows A and B show the spatial distribution of Haufe-transformed scores in SAD and HC.
- SAD is shown using ROI-wise BH-FDR surviving voxels.
- HC is shown using the same number of top absolute Haufe-score voxels for matched comparison.
- Row C summarizes the distribution of displayed voxels across FearNetwork ROIs.
- These results provide **secondary supportive evidence** that the discriminative neural pattern is distributed differently across FearNetwork regions in SAD and HC.
- If color scales are not shared, explicitly note that map comparisons are primarily descriptive and that **Row C is the primary comparative panel**.

## Companion table

Also generate a companion table for the manuscript/supplement.

### Table title

**Table X. ROI distribution of displayed Haufe voxels across FearNetwork ROIs**

### Required columns

- ROI
- SAD % displayed voxels
- HC % displayed voxels
- Difference (SAD − HC)

### Required row order

Use the same anatomical-family order as the figure:

- left_hippocampus
- right_hippocampus
- left_amygdala
- right_amygdala
- left_insula
- right_insula
- left_acc
- right_acc
- left_vmpfc
- right_vmpfc

## Style requirements

- Publication quality
- Clean layout
- Consistent fonts
- Remove any old labels that say **Aim 2**
- Keep panel labels **A, B, C**
- Use clear axis labels and legend
- Avoid clutter
- Ensure the main message is **distinct distribution of Haufe-transformed scores**, not decoding accuracy, cross-decoding, or spatial similarity testing

## Output

Return:

1. the final figure
2. the companion table
3. the final caption text

If needed, refactor the current plotting code to implement this cleanly.