## Task

Create a publication-quality Python script to generate:

1. **One supplementary multi-panel figure** named:  
    **Figure S2. Secondary supportive evidence for Aim 2**
2. **One standalone compact statistics table** summarizing all secondary metrics, saved as:
    - `.csv`
    - `.xlsx`

Use:

- Python
- matplotlib
- pandas
- numpy
- scipy if needed
- statsmodels if needed for FDR correction
- avoid overly complex dependencies unless necessary

---

## Goal

Generate a supplementary-results output for:

**Aim 2: Characterization of SAD–HC differences in neural representations of vicarious learning**

This output should present **secondary supportive evidence only**, not the primary metrics.

The script should create:

### A. A multi-panel figure with 4 panels

- **Panel A** = forest plot of all secondary metrics
- **Panel B** = geometry triangle schematic + distance-profile plot
- **Panel C** = entropy dot plot for CSS and CSR
- **Panel D** = shock-anchor / residualized trajectory plot

### B. A standalone statistics table

A clean, publication-ready summary table including descriptive statistics and inferential results for all secondary metrics.

---

## Scientific context

Aim 2 tests whether SAD participants differ from HC participants in:

1. representational geometry
2. decision certainty
3. learning dynamics

This supplementary figure should show **secondary supportive evidence** for those domains.

---

## Secondary metrics to include

### Geometry

- `Neural_Dist_Threat_Background` = dist(CSR, CS−)

### Decision certainty

- `Neural_Decoder_Entropy_CSS`
- `Neural_Decoder_Entropy_CSR`

Interpretation:

- Higher entropy = lower decoder certainty

### Learning dynamics

- `Shock_Anchor_Trajectory_Slope`
- `Residualized_Shock_Anchor_Trajectory_Slope`

If exact trajectory variable names differ in the dataset, detect and use the closest available equivalents.

---

# PART 1. FIGURE REQUIREMENTS

## Figure title

**Figure S2. Secondary supportive evidence for Aim 2**

## Overall layout

- Use a **2 x 2 layout**
- White background
- Clean publication-quality styling
- Consistent fonts and axis formatting
- Clear panel labels:
    - **A. Secondary metric summary**
    - **B. Geometry support**
    - **C. Decoder entropy**
    - **D. Shock-anchor trajectories**

## Cohort / grouping

- Compare **SAD** vs **HC**
- Prefer the **placebo cohort** if that is the intended Aim 2 analysis cohort
- If needed, robustly map group labels to SAD and HC

---

## Panel A. Forest plot of all secondary metrics

Create a **horizontal dot-and-whisker forest plot** summarizing the **SAD vs HC group effect** for all secondary metrics.

Include:

- `Neural_Dist_Threat_Background`
- `Neural_Decoder_Entropy_CSS`
- `Neural_Decoder_Entropy_CSR`
- `Shock_Anchor_Trajectory_Slope`
- `Residualized_Shock_Anchor_Trajectory_Slope`

For each metric, show:

- point estimate
- 95% confidence interval

Preferred effect metric:

- **standardized mean difference / Cohen’s d**
- If better aligned with the existing analysis pipeline, regression beta may be used, but standardized effect size is preferred

Formatting:

- Add a vertical reference line at **0**
- Order rows by domain:
    1. Geometry
    2. Certainty
    3. Trajectory

Use reader-friendly labels:

- Threat–background distance
- Entropy (CSS)
- Entropy (CSR)
- Shock-anchor slope
- Residualized shock-anchor slope

X-axis label:

- **Standardized SAD − HC effect size**

Add a note:

- Positive values indicate higher metric values in SAD than HC

Purpose:

- This panel should provide a compact statistical overview of all secondary supportive metrics.

---

## Panel B. Geometry triangle schematic + distance-profile plot

Create a geometry support panel with **two subcomponents inside the same panel**:

### Left side: triangle schematic

Draw a simple representational triangle with three condition centroids:

- **CSR** = vicarious threat cue
- **CSS** = vicarious safety cue
- **CS−** = safe/background reference cue

Show the three pairwise distances:

- **CSS–CS−** = primary safety-background distance
- **CSR–CSS** = primary threat-safety distance
- **CSR–CS−** = secondary threat-background distance

Requirements:

- Visually highlight **CSR–CS−** as the **secondary geometry metric**
- Make clear that this is the **third side of the representational triangle**
- Keep styling subtle and schematic, not overly decorative

### Right side: distance-profile plot

Show a compact profile plot with the three pairwise distances:

- CSS–CS−
- CSR–CSS
- CSR–CS−

Display:

- X-axis = pairwise distance type
- Y-axis = neural distance
- Separate SAD and HC values for each distance
- Show group mean ± 95% CI
- Individual points may be added if readable

Interpretation:

- The two primary distances should be shown as context
- The secondary geometry metric `Neural_Dist_Threat_Background` should be emphasized
- The purpose is to show whether the secondary geometry result is directionally consistent with the broader representational structure

Suggested y-axis label:

- **Neural distance**

Purpose:

- This panel should explain and support the geometry interpretation, not just present a single isolated secondary metric.

---

## Panel C. Entropy dot plot for CSS and CSR

Create a compact grouped plot showing:

- `Neural_Decoder_Entropy_CSS`
- `Neural_Decoder_Entropy_CSR`

Display:

- X-axis = condition:
    - CSS
    - CSR
- Y-axis = entropy
- For each condition, show separate SAD and HC distributions

Preferred style:

- jittered individual points
- overlaid group mean ± 95% CI  
    (or mean ± SEM if needed)

Y-axis label:

- **Decoder entropy**

Optional annotation:

- Higher entropy indicates lower certainty

### Important format instruction

**Panel C should follow the same visual format as the corresponding primary Aim 2 figure panel for decision certainty**, as closely as possible.

That means:

- keep the same basic visual grammar
- use the same style of group comparison
- keep axis treatment, summary overlays, and overall level of simplicity aligned with the primary Aim 2 certainty panel
- only replace the primary metrics with the secondary entropy metrics

Purpose:

- This panel should serve as the secondary/supportive counterpart to the primary decision-certainty panel.

---

## Panel D. Shock-anchor / residualized trajectory plot

Create a plot focused on:

- `Shock_Anchor_Trajectory_Slope`
- `Residualized_Shock_Anchor_Trajectory_Slope`

Display:

- Compare SAD vs HC for both metrics
- Show individual subject points if readable
- Overlay group mean ± 95% CI
- Include a zero reference line if appropriate

Suggested axis labeling:

- X-axis = trajectory metric
- Y-axis = trajectory slope

### Important format instruction

**Panel D should follow the same visual format as the corresponding primary Aim 2 figure panel for learning trajectories**, as closely as possible.

That means:

- keep the same basic visual grammar
- use the same style of group comparison and summary display
- keep axis treatment and overall appearance aligned with the primary Aim 2 learning-trajectories panel
- only replace the primary trajectory metrics with the shock-anchor / residualized shock-anchor metrics

Purpose:

- This panel should act as the secondary/supportive counterpart to the primary learning-dynamics panel.

---

# PART 2. STANDALONE TABLE REQUIREMENTS

In addition to the figure, generate a **standalone compact statistics table** as a separate deliverable.

## Output file names

Save as:

- `Table_S2_secondary_support.csv`
- `Table_S2_secondary_support.xlsx`

## Table title

**Table S2. Secondary supportive evidence for Aim 2**

## Required columns

The standalone table should include at minimum:

- `Metric`
- `Domain`
- `N_SAD`
- `N_HC`
- `SAD_mean`
- `SAD_SD`
- `HC_mean`
- `HC_SD`
- `Mean_difference_SAD_minus_HC`
- `Effect_size`
- `CI_lower`
- `CI_upper`
- `Test_statistic`
- `p_value`
- `q_value`
- `Interpretation`

## Domain labels

Use:

- Geometry
- Certainty
- Trajectory

## Reader-friendly metric labels

Map raw metric names to readable labels:

- `Neural_Dist_Threat_Background` → Threat–background distance
- `Neural_Decoder_Entropy_CSS` → Entropy (CSS)
- `Neural_Decoder_Entropy_CSR` → Entropy (CSR)
- `Shock_Anchor_Trajectory_Slope` → Shock-anchor slope
- `Residualized_Shock_Anchor_Trajectory_Slope` → Residualized shock-anchor slope

## Interpretation column

Add a brief interpretation string such as:

- “Higher in SAD”
- “Lower in SAD”
- “Minimal group difference”
- “Directionally consistent with altered certainty”
- “Supportive trajectory difference”

Keep wording restrained and descriptive, not overstated.

---

# PART 3. STATISTICAL ANALYSIS REQUIREMENTS

For each secondary metric:

1. Compute descriptive statistics separately for SAD and HC
    - N
    - mean
    - SD
2. Run a group comparison
    - independent-samples t-test preferred
    - Welch’s t-test allowed if variance assumptions are problematic
3. Compute:
    - mean difference (SAD − HC)
    - effect size (preferably Cohen’s d)
    - 95% confidence interval
4. Apply **FDR correction across all secondary metrics**
    - Benjamini–Hochberg
    - report corrected values as `q`
5. Use the same computed results to populate:
    - Panel A
    - Panel B where relevant
    - standalone Table S2

---

# PART 4. INPUT DATA EXPECTATIONS

Assume a subject-level dataframe with one row per subject and columns including:

- `group`
- secondary metric columns

If possible:

- define the input file path near the top of the script
- allow easy editing of file paths

If some expected columns are missing:

- do not crash
- print a clear warning
- omit missing metrics from both figure and table
- continue with available metrics

If placebo-only filtering is needed:

- implement it clearly and document it

---

# PART 5. OUTPUT REQUIREMENTS

The script should save:

## Figure files

- `Figure_S2_secondary_support.png`
- `Figure_S2_secondary_support.svg`

## Table files

- `Table_S2_secondary_support.csv`
- `Table_S2_secondary_support.xlsx`

## Console summary

Print a short console summary including:

- metrics found
- metrics missing
- sample size in SAD and HC
- output file paths

---

# PART 6. VISUAL STYLE GUIDANCE

- Keep the figure compact and clean
- Prioritize readability
- Avoid unnecessary decorative elements
- Use consistent styling across panels
- Keep the supplementary figure visually aligned with the primary Aim 2 figure
- Especially ensure that **Panel C and Panel D visually mirror the corresponding primary Aim 2 panels**
- Ensure the figure is appropriate for a supplementary figure in a clinical neuroscience / neuroimaging paper

---

# PART 7. INTERPRETATION GUIDANCE

This is a **secondary supportive evidence** output.  
The goal is to show whether the secondary metrics are **directionally consistent with the primary Aim 2 findings**, and to provide supporting mechanistic context.

Do **not** overstate significance in plot titles, labels, annotations, or interpretation text.