Update `mvpa_l2.ipynb` so the Aim 1 section contains ONLY the items listed in these three task files:

1. `task_Visualization-Aim1-primary.md`
2. `task_Visualization-Aim1-secondary.md`
3. `task_Visualization-Aim1-sensitivity.md`

These three files are the whitelist for Aim 1. Do not rely on older Aim 1 content already present in `mvpa_l2.ipynb` unless it is explicitly listed in one of these three task files.

Task:

1. Open and read all three task files.

2. Extract a clear checklist of allowed Aim 1 items from them:

    - primary items from `task_Visualization-Aim1-primary.md`
    - secondary/supportive items from `task_Visualization-Aim1-secondary.md`
    - sensitivity items from `task_Visualization-Aim1-sensitivity.md`

3. Locate the Aim 1 section in `mvpa_l2.ipynb`.

    - Start at the Aim 1 markdown header.
    - End right before the Aim 2 markdown header.

4. Compare the current Aim 1 notebook cells against the checklist from the three task files.

5. Keep only cells that implement items explicitly listed in those three files.

6. Remove or rewrite any Aim 1 cells that are not listed in those three files.

7. Do not change Aim 2, Aim 3, Aim 4, Aim 5, reproducibility/QC sections, or final reporting sections unless a tiny compatibility fix is required.

Important cleanup rule:

If an Aim 1 block is not named or requested in one of the three task files, remove it from the Aim 1 section. This includes old exploratory or legacy blocks even if they are scientifically interesting.

Examples of blocks to remove unless they are explicitly listed in the three task files:

- old exploratory decoding summaries
- old or duplicated Haufe spatial-pattern sections
- extra Haufe maps, voxel tables, or ROI summaries not requested in the task files
- representational-geometry, decision-certainty, or trajectory sections that belong to Aim 2
- clinical-symptom association sections that belong to Aim 3
- SCR association sections that belong to Aim 4
- oxytocin-modulation sections that belong to Aim 5
- support/contextual metric registry sections
- alternative mask or feature-space sensitivity sections not requested in the sensitivity task file
- extra forest plots, heatmaps, bar plots, trajectory plots, or tables not requested in the task files
- any old Aim 1 figure/table that duplicates or conflicts with the three task files

Required notebook organization:

Use this Aim 1 order:

```text
## Aim 1. Group-specific neural representation identification under placebo

### Aim 1 primary visualization
# only items from task_Visualization-Aim1-primary.md

### Aim 1 secondary/supportive visualization
# only items from task_Visualization-Aim1-secondary.md

### Aim 1 sensitivity visualization
# only items from task_Visualization-Aim1-sensitivity.md
```

For each subsection:

- Include a short markdown cell stating which task file controls that subsection.
- Keep the code modular and readable.
- Keep missing-output behavior explicit.
- Do not treat missing files as null results.
- Preserve raw metric names in data processing.
- Use display-label mappings only for plots/tables.
- Keep primary, secondary, and sensitivity results visually and inferentially separate.
- Do not mix SAD and HC results unless the task file explicitly requests a direct comparison.
- Keep placebo-condition filtering explicit.
- Keep the CSR versus CSS classification target explicit.
- Keep FearNetwork-mask filtering explicit when required by the task file.

Primary section requirements:

Follow `task_Visualization-Aim1-primary.md` exactly. Keep only the primary Aim 1 visualizations/tables listed there.

Expected primary Figure 1 content, if listed in the task file:

- Panel A: SAD placebo CSR-vs-CSS decoding / separability result
- Panel B: HC placebo CSR-vs-CSS decoding / separability result
- Panel C: functional/spatial specificity or model-evidence summary, only if requested in the primary task file
- Statistics table for the primary Aim 1 test

Use only the primary metrics listed in the primary task file.

Secondary section requirements:

Follow `task_Visualization-Aim1-secondary.md` exactly. Keep only the secondary/supportive items listed there.

Expected secondary/supportive content, if listed in the task file:

- distinct distribution of Haufe transform scores
- SAD FearNetwork Haufe pattern
- HC FearNetwork Haufe pattern
- ROI distribution of displayed voxels across FearNetwork ROIs
- symmetric color scale for SAD and HC maps when possible
- explicit caption note when symmetric color scale is not possible
- ROI ordering by anatomical family:
    - hippocampus
    - amygdala
    - insula
    - ACC
    - vmPFC

Do not promote secondary/supportive items to primary evidence.

Sensitivity section requirements:

Follow `task_Visualization-Aim1-sensitivity.md` exactly. Keep only the sensitivity items listed there.

Expected sensitivity content, if listed in the task file:

- same general figure/table structure as the primary Aim 1 analysis
- robustness/sensitivity version of the SAD placebo CSR-vs-CSS separability result
- robustness/sensitivity version of the HC placebo CSR-vs-CSS separability result
- corresponding sensitivity statistics table

Do not mix sensitivity outputs into the primary or secondary sections.

After editing:

1. Run a structural check on the notebook.

2. Confirm that Aim 1 contains only items from the three task files.

3. Print or display a small checklist table with columns:

```text
task_file
allowed_item
implemented_in_mvpa_l2
cell_or_section
status
notes
```

4. Also print a removed-items table with columns:

```text
removed_item
reason
old_cell_or_section
```

Acceptance criteria:

- `mvpa_l2.ipynb` has a clean Aim 1 section.
- Aim 1 is organized into primary, secondary/supportive, and sensitivity subsections.
- Every kept Aim 1 item is traceable to one of:
    - `task_Visualization-Aim1-primary.md`
    - `task_Visualization-Aim1-secondary.md`
    - `task_Visualization-Aim1-sensitivity.md`
- No extra Aim 1 blocks remain.
- Aim 2 and Aim 3–5 are unchanged.
- Missing outputs are labeled as missing/pending, not interpreted as null.
- The notebook remains runnable from top to bottom.
