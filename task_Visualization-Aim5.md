Update `mvpa_l2.ipynb` so the Aim 5 section contains ONLY the items listed in these three task files:

1. `task_Visualization-Aim5-primary.md`
2. `task_Visualization-Aim5-secondary.md`
3. `task_Visualization-Aim5-sensitivity.md`

These three files are the whitelist for Aim 5. Do not rely on older Aim 5 content already present in `mvpa_l2.ipynb` unless it is explicitly listed in one of these three task files.

Task:

1. Open and read all three task files.

2. Extract a clear checklist of allowed Aim 5 items from them:

    - primary items from `task_Visualization-Aim5-primary.md`
    - secondary/supportive items from `task_Visualization-Aim5-secondary.md`
    - sensitivity items from `task_Visualization-Aim5-sensitivity.md`

3. Locate the Aim 5 section in `mvpa_l2.ipynb`.

    - Start at the Aim 5 markdown header.
    - End at the next major markdown header after Aim 5, such as reproducibility, QC, final reporting, or notebook end.

4. Compare the current Aim 5 notebook cells against the checklist from the three task files.

5. Keep only cells that implement items explicitly listed in those three files.

6. Remove or rewrite any Aim 5 cells that are not listed in those three files.

7. Do not change Aim 1, Aim 2, Aim 3, Aim 4, reproducibility/QC sections, or final reporting sections unless a tiny compatibility fix is required.

Important cleanup rule:

If an Aim 5 block is not named or requested in one of the three task files, remove it from the Aim 5 section. This includes old exploratory or legacy blocks even if they are scientifically interesting.

Examples of blocks to remove unless they are explicitly listed in the three task files:

- old exploratory oxytocin summaries
- extra drug-condition contrasts not requested in the task files
- extra moderation models not requested in the task files
- extra subgroup analyses not requested in the task files
- decoding/separability sections that belong to Aim 1
- representational-geometry, decision-certainty, or trajectory group-difference sections that belong to Aim 2
- clinical-symptom association sections that belong to Aim 3
- SCR association sections that belong to Aim 4
- old Haufe spatial-pattern sections unless explicitly requested for Aim 5
- support/contextual metric registry sections
- alternative mask or feature-space sensitivity sections not requested in the sensitivity task file
- extra forest plots, heatmaps, interaction plots, slope plots, trajectory plots, or tables not requested in the task files
- any old Aim 5 figure/table that duplicates or conflicts with the three task files

Required notebook organization:

Use this Aim 5 order:

```text
## Aim 5. Oxytocin modulation of neural profiles

### Aim 5 primary visualization
# only items from task_Visualization-Aim5-primary.md

### Aim 5 secondary/supportive visualization
# only items from task_Visualization-Aim5-secondary.md

### Aim 5 sensitivity visualization
# only items from task_Visualization-Aim5-sensitivity.md
```

For each subsection:

- Include a short markdown cell stating which task file controls that subsection.
- Keep the code modular and readable.
- Keep missing-output behavior explicit.
- Do not treat missing files as null results.
- Preserve raw metric names in data processing.
- Use display-label mappings only for plots/tables.
- Keep primary, secondary, and sensitivity results visually and inferentially separate.
- Keep placebo versus oxytocin condition coding explicit when required by the task file.
- Keep group-specific analyses explicit when required by the task file.
- Keep drug-effect and drug-by-group interaction terms explicit when they are requested.

Primary section requirements:

Follow `task_Visualization-Aim5-primary.md` exactly. Keep only the primary Aim 5 visualizations/tables listed there.

Expected primary Figure content, if listed in the task file:

- Panel A: main oxytocin-effect summary or drug-condition contrast heatmap
- Panel B: SAD-specific oxytocin-effect forest plot or paired-condition plot
- Panel C: HC-specific oxytocin-effect forest plot or paired-condition plot
- Statistics table for the primary Aim 5 oxytocin-modulation tests

Use only the primary metrics and oxytocin-modulation terms listed in the primary task file.

Secondary section requirements:

Follow `task_Visualization-Aim5-secondary.md` exactly. Keep only the secondary/supportive items listed there.

Expected secondary/supportive content, if listed in the task file:

- same general figure type as the Aim 5 primary analysis
- secondary/supportive oxytocin-effect summary
- group-specific secondary/supportive forest plots or paired-condition plots
- compact statistics table for secondary/supportive tests

Do not promote secondary/supportive items to primary evidence.

Sensitivity section requirements:

Follow `task_Visualization-Aim5-sensitivity.md` exactly. Keep only the sensitivity items listed there.

Expected sensitivity content, if listed in the task file:

- same general figure/table structure as the Aim 5 primary analysis
- robustness/sensitivity version of the oxytocin-effect summary
- robustness/sensitivity version of SAD-specific effect plot
- robustness/sensitivity version of HC-specific effect plot
- corresponding sensitivity statistics table

Do not mix sensitivity outputs into the primary or secondary sections.

After editing:

1. Run a structural check on the notebook.

2. Confirm that Aim 5 contains only items from the three task files.

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

- `mvpa_l2.ipynb` has a clean Aim 5 section.
- Aim 5 is organized into primary, secondary/supportive, and sensitivity subsections.
- Every kept Aim 5 item is traceable to one of:
    - `task_Visualization-Aim5-primary.md`
    - `task_Visualization-Aim5-secondary.md`
    - `task_Visualization-Aim5-sensitivity.md`
- No extra Aim 5 blocks remain.
- Aim 1, Aim 2, Aim 3, and Aim 4 are unchanged.
- Missing outputs are labeled as missing/pending, not interpreted as null.
- The notebook remains runnable from top to bottom.
