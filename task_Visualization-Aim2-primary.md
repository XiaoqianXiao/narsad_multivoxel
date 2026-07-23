# Codex Task: Aim 2 Figure for `mvpa_l2.ipynb`

## Goal

Revise the Aim 2 visualization section of `mvpa_l2.ipynb`, or create a companion Python script, to generate a publication-quality multi-panel figure:

**Figure 2. Characterization of SAD–HC differences in neural representations of vicarious safety/threat learning**

The figure should summarize Aim 2 results under the **placebo condition**, comparing **SAD** participants with **HC** participants.

`PROJECT_CONTEXT.md` is the canonical project guide. If this task file conflicts with `PROJECT_CONTEXT.md`, follow `PROJECT_CONTEXT.md`.

---

## Scientific question

Aim 2 tests whether SAD participants differ from HC participants in neural representations of vicarious learning across three domains:

1. **Representational geometry**
2. **Decision certainty**
3. **Learning dynamics**

The primary placebo group model is:

```text
neural_metric ~ Group + covariates
```

Use **HC as the reference group**. The main contrast is **SAD minus HC**.

Expected statistical outputs include:

- estimate
- 95% CI
- p value
- q/FDR value
- n
- R²
- formula
- model status

Apply FDR correction within each planned Aim 2 question.

---

## Required outputs

Create a full runnable Python script, not pseudocode.

Use these exact output filenames:

```python
fig.savefig("figure2_aim2.svg", bbox_inches="tight")
fig.savefig("figure2_aim2.pdf", bbox_inches="tight")
fig.savefig("figure2_aim2.png", dpi=600, bbox_inches="tight")
```

The SVG is the primary editable figure output.

Requirements for SVG:

- Keep text editable with `svg.fonttype = "none"`.
- Do not rasterize the whole figure.
- Do not plot voxel-level or thousands of trial-level points directly into SVG.
- If data are dense, summarize at the subject level.

---

## Required Python packages

Use:

- `matplotlib`
- `pandas`
- `numpy`
- `scipy`, if needed
- `seaborn`, for one consistent global theme and distribution plots

Avoid unnecessary complex dependencies.

---

## Primary Aim 2 metrics

### Geometry

```text
Neural_Dist_Safety_Background = dist(CSS, CS−)
Neural_Dist_Threat_Safety     = dist(CSR, CSS)
```

### Decision certainty

```text
Neural_SafetyEvidence = P(safety | CSS)
Neural_ThreatEvidence = P(threat | CSR)
```

### Learning trajectories

```text
Neural_Safety_Trajectory_Slope
Neural_Threat_Trajectory_Slope
```

---

## Overall figure layout

Use a 2 × 2 layout:

```text
Panel A: top-left      Main summary
Panel B: top-right     Representational geometry
Panel C: bottom-left   Decision certainty
Panel D: bottom-right  Learning trajectories
```

Use bold panel labels: **A**, **B**, **C**, **D**.

General style:

- Clean white background
- Publication-ready spacing
- Consistent fonts, line widths, marker sizes, and legends
- Readable after reduction to manuscript column width
- Vector-friendly plotting choices
- Colorblind-friendly colors
- Interpretable in grayscale by using marker shapes and/or line styles
- Minimal grid lines
- `sns.despine()` where appropriate

---

## Global theme

Apply one seaborn theme once near the top of the script. Do not set multiple competing themes.

Recommended:

```python
sns.set_theme(
    context="poster",
    style="white",
    font="Arial",
    font_scale=0.75,
    rc={
        "axes.linewidth": 1.2,
        "axes.labelsize": 15,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.titlesize": 18,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    },
)
```

If the figure is too crowded, use:

```python
sns.set_theme(context="talk", style="white", font="Arial", font_scale=1.0)
```

---

## Recommended colors and markers

Use a calm, publication-ready palette.

Groups:

```text
HC  = blue or teal
SAD = orange or red
```

Conditions:

```text
CSR = purple circles
CSS = blue squares
CS- = green triangles
```

Do not overuse color.

---

# Panel A. Main summary

## Purpose

Panel A is the headline summary of SAD–HC effects across the six primary Aim 2 metrics.

## Plot type

Use a horizontal forest plot / dot-and-whisker plot.

## Content

Show standardized group differences for:

1. `Neural_Dist_Safety_Background`
2. `Neural_Dist_Threat_Safety`
3. `Neural_SafetyEvidence`
4. `Neural_ThreatEvidence`
5. `Neural_Safety_Trajectory_Slope`
6. `Neural_Threat_Trajectory_Slope`

## Required visual elements

- X-axis: Cohen’s d for SAD minus HC
- 95% bootstrap CI for each effect
- Vertical reference line at 0
- Metrics ordered by family:
  1. geometry
  2. certainty
  3. trajectories
- Color-code by metric family
- Print Cohen’s d numerically on the right side of each row
- Add note: `Positive values indicate higher values in SAD`

## Axis label

```text
Standardized SAD–HC difference (Cohen's d)
```

## Panel title

```text
Main summary of SAD–HC differences
```

## Effect size rules

- Cohen’s d must be computed as **SAD minus HC**.
- 95% CI must be estimated by bootstrap resampling.
- Bootstrap resampling must resample subjects within each group.
- Use a fixed random seed.

Return a tidy effect-size dataframe with:

```text
metric
metric_family
cohens_d
ci_low
ci_high
```

---

# Panel B. Representational geometry

## Purpose

Panel B shows condition representations in a biologically interpretable target-safety by target-threat space.

## Required design

Panel B must contain two side-by-side subplots:

1. SAD
2. HC

Do **not** use generic labels such as `Dimension 1` or `Dimension 2`.

## Axes

```text
X-axis: Alignment with target safety (early CS−)
Y-axis: Alignment with target threat (early reinstatement CSR)
```

## Content

For each group, plot condition-level representations for:

```text
CSR
CSS
CS-
```

## Required visual elements

- Individual subject points for each condition
- Group centroid for each condition
- Confidence ellipse or dispersion ellipse for each condition
- Distinct condition colors and marker shapes:
  - CSR = purple circles
  - CSS = blue squares
  - CS− = green triangles

## Panel title

```text
Representational geometry
```

## Methodological note

The x/y values must represent alignment or similarity to target reference patterns.

They must **not** be arbitrary MDS dimensions.

If real `safety_alignment` and `threat_alignment` scores are unavailable, use placeholder values only in demo mode and add TODO comments explaining where the real model-derived alignment scores should be inserted.

---

# Panel C. Decision certainty

## Purpose

Panel C should make the certainty interpretation visually obvious.

Decoder posterior probabilities near **0.50** indicate ambiguous or chance-level evidence. Values closer to **1.00** indicate stronger certainty for the target state.

## Required design

Panel C must contain two side-by-side distribution subplots:

1. Safety certainty
2. Threat certainty

## Metrics

```text
Safety certainty: P(safety | CSS)
Threat certainty: P(threat | CSR)
```

## Plot type

Use distribution-style plots, preferably:

- `sns.histplot`, or
- `sns.kdeplot`

Do not use only bar plots.

## Required visual elements

- Overlapping distributions by group
- Transparent fills
- X-axis fixed from 0 to 1 in both subplots
- Same binning or KDE bandwidth across groups
- Vertical dashed reference line at x = 0.50
- Label the line as `0.50 = ambiguous` or `chance / ambiguous`

If using KDE, clip/restrict densities to 0–1 because posterior probabilities cannot be outside this range.

If using histograms, use `stat="density"` so groups remain comparable even if sample sizes differ.

Optional additions:

- Cohen’s d and bootstrap 95% CI as small text inside each subplot
- Group means as subtle rug marks or vertical ticks
- Sample sizes in legend

Do not add p values unless they come from an existing analysis table or are computed consistently.

## Subplot labels

Safety subplot:

```text
Title: Safety certainty
X-axis: P(safety | CSS)
Y-axis: Density
```

Threat subplot:

```text
Title: Threat certainty
X-axis: P(threat | CSR)
Y-axis: Density
```

## Panel title

```text
Decision certainty relative to ambiguity
```

---

# Panel D. Learning trajectories

## Purpose

Panel D shows learning dynamics over trials.

## Required design

Panel D must contain two side-by-side trajectory subplots:

1. Safety trajectory
2. Threat trajectory

## Subplot 1: safety trajectory

```text
Title: CSS toward target safety
X-axis: Trial
Y-axis: Safety-reference alignment
```

Outcome:

```text
CSS alignment toward target safety / CS−
```

## Subplot 2: threat trajectory

```text
Title: CSR toward target threat
X-axis: Trial
Y-axis: Threat-reference alignment
```

Outcome:

```text
CSR threat evidence over trials
```

## Required visual elements

- Separate lines for HC and SAD
- Markers at each trial
- Shaded 95% CI bands
- Optional dashed horizontal reference line if scientifically meaningful
- Readable and comparable scales

## Data handling rule

Compute CIs across subjects, not across raw trials.

If the trial-level table contains multiple rows per subject/trial, first average within:

```text
subject_id × group × condition × trial
```

Then compute group mean and 95% CI across subjects.

---

## Required code structure

Write modular code with these functions:

```python
setup_publication_theme()
load_data()
generate_mock_data()
compute_cohens_d()
bootstrap_cohens_d_ci()
compute_effect_sizes()
confidence_ellipse()
plot_panel_a_summary()
plot_panel_b_geometry()
plot_panel_c_certainty()
plot_panel_d_trajectories()
make_figure2()
```

The script should be easy to adapt to real Aim 2 outputs later.

---

## Draft figure caption

Add this as a multiline Python string near the bottom of the script:

```python
FIGURE_CAPTION = """
Figure 2. Characterization of SAD–HC differences in neural representations of vicarious learning.
Panel A shows standardized SAD–HC group differences across the primary Aim 2 neural metrics. Positive values indicate higher values in SAD than HC.
Panel B shows representational geometry in a biologically interpretable target-safety by target-threat space, where rightward values indicate stronger alignment with the CS− safety reference and upward values indicate stronger alignment with the CSR threat reference.
Panel C shows distributions of decoder posterior probabilities for safety and threat certainty. The dashed vertical line at 0.50 indicates ambiguous or chance-level decoder evidence; values closer to 1.00 indicate stronger certainty for the corresponding target state.
Panel D shows early-phase learning trajectories for CSS safety updating during extinction and CSR threat evidence during reinstatement.
If demo data are used, the figure is illustrative only.
"""
```

---

## Acceptance checklist

Codex should finish with code that:

- Runs without real data by using mock data.
- Uses real CSV files automatically when available.
- Saves `figure2_aim2.svg`, `figure2_aim2.pdf`, and `figure2_aim2.png`.
- Uses one consistent seaborn theme.
- Keeps SVG text editable.
- Uses Cohen’s d as SAD minus HC.
- Computes bootstrap 95% CIs by resampling subjects within group.
- Uses biologically meaningful axes for Panel B, not generic dimensions.
- Shows the 0.50 ambiguity reference in Panel C.
- Computes trajectory CIs across subjects in Panel D.
- Keeps the full figure clean, readable, and publication-ready.
