Update `mvpa_l2.ipynb` so the Aim 2 section contains ONLY the items listed in these three task files:

1. `task_Visualization-Aim2-primary.md`
    
2. `task_Visualization-Aim2-secondary.md`
    
3. `task_Visualization-Aim2-sensitivity.md`
    

These three files are the whitelist for Aim 2. Do not rely on older Aim 2 content already present in `mvpa_l2.ipynb` unless it is explicitly listed in one of these three task files.

Task:

1. Open and read all three task files.
    
2. Extract a clear checklist of allowed Aim 2 items from them:
    
    - primary items from `task_Visualization-Aim2-primary.md`
        
    - secondary/supportive items from `task_Visualization-Aim2-secondary.md`
        
    - sensitivity items from `task_Visualization-Aim2-sensitivity.md`
        
3. Locate the Aim 2 section in `mvpa_l2.ipynb`.
    
    - Start at the Aim 2 markdown header.
        
    - End right before the Aim 3 markdown header.
        
4. Compare the current Aim 2 notebook cells against the checklist from the three task files.
    
5. Keep only cells that implement items explicitly listed in those three files.
    
6. Remove or rewrite any Aim 2 cells that are not listed in those three files.
    
7. Do not change Aim 1, Aim 3, Aim 4, Aim 5, reproducibility/QC sections, or final reporting sections unless a tiny compatibility fix is required.
    

Important cleanup rule:

If an Aim 2 block is not named or requested in one of the three task files, remove it from the Aim 2 section. This includes old exploratory or legacy blocks even if they are scientifically interesting.

Examples of blocks to remove unless they are explicitly listed in the three task files:

- old Haufe spatial-pattern sections
    
- SCR subgroup Haufe sensitivity sections
    
- residualized shock-anchor sections
    
- support/contextual metric registry sections
    
- alternative mask or feature-space sensitivity sections
    
- extra forest plots, heatmaps, trajectory plots, or tables not requested in the task files
    
- any old Aim 2 figure/table that duplicates or conflicts with the three task files
    

Required notebook organization:

Use this Aim 2 order:

```text
## Aim 2. SAD–HC neural-profile difference under placebo

### Aim 2 primary visualization
# only items from task_Visualization-Aim2-primary.md

### Aim 2 secondary/supportive visualization
# only items from task_Visualization-Aim2-secondary.md

### Aim 2 sensitivity visualization
# only items from task_Visualization-Aim2-sensitivity.md
```

For each subsection:

- Include a short markdown cell stating which task file controls that subsection.
    
- Keep the code modular and readable.
    
- Keep missing-output behavior explicit.
    
- Do not treat missing files as null results.
    
- Preserve raw metric names in data processing.
    
- Use display-label mappings only for plots/tables.
    
- Keep primary, secondary, and sensitivity results visually and inferentially separate.
    

Primary section requirements:

Follow `task_Visualization-Aim2-primary.md` exactly. Keep only the primary Aim 2 visualizations/tables listed there.

Expected primary Figure 2 content, if listed in the task file:

- Panel A: main summary of SAD–HC differences
    
- Panel B: representational geometry
    
- Panel C: decision certainty
    
- Panel D: learning trajectories
    

Use only the primary metrics listed in the primary task file.

Secondary section requirements:

Follow `task_Visualization-Aim2-secondary.md` exactly. Keep only the secondary/supportive items listed there.

Do not promote secondary/supportive items to primary evidence.

Sensitivity section requirements:

Follow `task_Visualization-Aim2-sensitivity.md` exactly. Keep only the sensitivity items listed there.

Do not mix sensitivity outputs into the primary or secondary sections.

After editing:

1. Run a structural check on the notebook.
    
2. Confirm that Aim 2 contains only items from the three task files.
    
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

- `mvpa_l2.ipynb` has a clean Aim 2 section.
    
- Aim 2 is organized into primary, secondary/supportive, and sensitivity subsections.
    
- Every kept Aim 2 item is traceable to one of:
    
    - `task_Visualization-Aim2-primary.md`
        
    - `task_Visualization-Aim2-secondary.md`
        
    - `task_Visualization-Aim2-sensitivity.md`
        
- No extra Aim 2 blocks remain.
    
- Aim 1 and Aim 3–5 are unchanged.
    
- Missing outputs are labeled as missing/pending, not interpreted as null.
    
- The notebook remains runnable from top to bottom.
