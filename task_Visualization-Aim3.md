Update `mvpa_l2.ipynb` so the Aim 3 section contains ONLY the items listed in these three task files:

1. `task_Visualization-Aim3-primary.md`
2. `task_Visualization-Aim3-secondary.md`
3. `task_Visualization-Aim3-sensitivity.md`

These three files are the whitelist for Aim 3. Do not rely on older Aim 3 content already present in `mvpa_l2.ipynb` unless it is explicitly listed in one of these three task files.

Task:

1. Open and read all three task files.

2. Extract a clear checklist of allowed Aim 3 items from them:

    - primary items from `task_Visualization-Aim3-primary.md`
    - secondary/supportive items from `task_Visualization-Aim3-secondary.md`
    - sensitivity items from `task_Visualization-Aim3-sensitivity.md`

3. Locate the Aim 3 section in `mvpa_l2.ipynb`.

    - Start at the Aim 3 markdown header.
    - End right before the Aim 4 markdown header.

4. Compare the current Aim 3 notebook cells against the checklist from the three task files.

5. Keep only cells that implement items explicitly listed in those three files.

6. Remove or rewrite any Aim 3 cells that are not listed in those three files.

7. Do not change Aim 1, Aim 2, Aim 4, Aim 5, reproducibility/QC sections, or final reporting sections unless a tiny compatibility fix is required.

Important cleanup rule:

If an Aim 3 block is not named or requested in one of the three task files, remove it from the Aim 3 section. This includes old exploratory or legacy blocks even if they are scientifically interesting.

Examples of blocks to remove unless they are explicitly listed in the three task files:

- old exploratory symptom-association summaries
- extra clinical metrics not requested in the task files
- extra symptom scales beyond those requested in the task files
- decoding/separability sections that belong to Aim 1
- representational-geometry, decision-certainty, or trajectory group-difference sections that belong to Aim 2
- SCR association sections that belong to Aim 4
- oxytocin-modulation sections that belong to Aim 5
- old Haufe spatial-pattern sections unless explicitly requested for Aim 3
- support/contextual metric registry sections
- alternative mask or feature-space sensitivity sections not requested in the sensitivity task file
- extra forest plots, heatmaps, scatter plots, regression plots, or tables not requested in the task files
- any old Aim 3 figure/table that duplicates or conflicts with the three task files

Required notebook organization:

Use this Aim 3 order:

```text
## Aim 3. Clinical relevance of neural profiles under placebo

### Aim 3 primary visualization
# only items from task_Visualization-Aim3-primary.md

### Aim 3 secondary/supportive visualization
# only items from task_Visualization-Aim3-secondary.md

### Aim 3 sensitivity visualization
# only items from task_Visualization-Aim3-sensitivity.md
```

For each subsection:

- Include a short markdown cell stating which task file controls that subsection.
- Keep the code modular and readable.
- Keep missing-output behavior explicit.
- Do not treat missing files as null results.
- Preserve raw metric names in data processing.
- Use display-label mappings only for plots/tables.
- Keep primary, secondary, and sensitivity results visually and inferentially separate.
- Keep placebo-condition filtering explicit when required by the task file.
- Keep group-specific analyses explicit when required by the task file.
- Keep symptom variables explicit, especially `dass_anxiety` and `lsas_total` when they are requested.

Primary section requirements:

Follow `task_Visualization-Aim3-primary.md` exactly. Keep only the primary Aim 3 visualizations/tables listed there.

Expected primary Figure content, if listed in the task file:

- Panel A: association heatmap with selected regression scatter-line overlays
- Panel B: SAD-specific forest plot for clinical associations
- Panel C: HC-specific forest plot for clinical associations
- Statistics table for the primary Aim 3 clinical-association tests

Use only the primary metrics and symptom variables listed in the primary task file.

Secondary section requirements:

Follow `task_Visualization-Aim3-secondary.md` exactly. Keep only the secondary/supportive items listed there.

Expected secondary/supportive content, if listed in the task file:

- same general figure type as the Aim 3 primary analysis
- association heatmap for secondary/supportive clinical evidence
- group-specific forest plots for secondary/supportive clinical evidence
- compact statistics table for secondary/supportive tests

Do not promote secondary/supportive items to primary evidence.

Sensitivity section requirements:

Follow `task_Visualization-Aim3-sensitivity.md` exactly. Keep only the sensitivity items listed there.

Expected sensitivity content, if listed in the task file:

- same general figure/table structure as the Aim 3 primary analysis
- robustness/sensitivity version of the association heatmap
- robustness/sensitivity version of SAD-specific forest plot
- robustness/sensitivity version of HC-specific forest plot
- corresponding sensitivity statistics table

Do not mix sensitivity outputs into the primary or secondary sections.

After editing:

1. Run a structural check on the notebook.

2. Confirm that Aim 3 contains only items from the three task files.

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

- `mvpa_l2.ipynb` has a clean Aim 3 section.
- Aim 3 is organized into primary, secondary/supportive, and sensitivity subsections.
- Every kept Aim 3 item is traceable to one of:
    - `task_Visualization-Aim3-primary.md`
    - `task_Visualization-Aim3-secondary.md`
    - `task_Visualization-Aim3-sensitivity.md`
- No extra Aim 3 blocks remain.
- Aim 1, Aim 2, Aim 4, and Aim 5 are unchanged.
- Missing outputs are labeled as missing/pending, not interpreted as null.
- The notebook remains runnable from top to bottom.
