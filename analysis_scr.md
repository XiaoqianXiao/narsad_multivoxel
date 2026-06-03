# Role & Task
You are an expert NeuralScience, Anxiety disorders, and Machine Learning developer. Generate a production-ready Python analysis notebook named `analysis_scr.ipynb` to analyze Skin Conductance Response (SCR) data across acquisition, extinction, and reinstatement phases.

# Data Sources & Schema
Always use explicitly defined variables for all file paths, dataframes, and column names.

1. **SCR Data Paths:**
   - Acquisition Phase: `/Users/xiaoqianxiao/projects/NARSAD/EDR/peak_stats_table-phase1.csv`
   - Extinction & Reinstatement Phases: `/Users/xiaoqianxiao/projects/NARSAD/EDR/peak_stats_table-phase2.3.csv`
     *Note on Phase 2.3 trial splitting (per subject):*
     - Trials 1–24: Extinction Phase
     - Trials 25–48: Reinstatement Phase

2. **Metadata Path:**
   - Group & Drug Assignment: `/Users/xiaoqianxiao/projects/NARSAD/MRI/source_data/behav/drug_order.csv` 
     - Columns include diagnostic group (e.g., 'SAD' vs. 'HC') and drug condition (e.g., 'Placebo' vs. 'Oxytocin').

# SCR Data Preprocessing Pipeline
- **Thresholding:** Code all negative responses and responses below 0.02 µS as exactly `0`.
- **Transformation:** Apply a square-root transformation to the thresholded amplitudes: `sqrt_scr = sqrt(scr)`.

# Subject Inclusion & Sensitivity Cohort Definitions
Calculate the following boolean classification flags for each subject to allow for flexible sensitivity analyses. The primary analysis must retain all participants, while these four definitions serve as sub-cohort filters:

1. **Physiological SCR Responder:** At least two acquisition CS+ trials with raw SCR amplitude $\ge$ 0.05 µS.
2. **Simple Acquisition Differential Learner:** Must be a *Physiological Responder* AND have a mean `sqrt_scr` for CS+ greater than the mean `sqrt_scr` for CS- across the entire acquisition phase.
3. **Habituation-Adjusted Learner:** Must be a *Physiological Responder* AND exhibit a positive ($\beta > 0$) within-subject CS+ coefficient from the ordinary least squares (OLS) linear model: `sqrt_scr ~ CS_type + Trial_Z` (where `Trial_Z` is the z-scored trial number to control for habituation).
4. **Late-Phase Sensitivity Learner:** Must meet two conditions:
   a) Late-phase CS+ > Late-phase CS-
   b) (Late-phase CS+ - Late-phase CS-) > (Early-phase CS+ - Early-phase CS-)

# Statistical Analysis Plan

## Part 1: Main Analysis (Phase-Level Effects)
For each phase, compute subject-level mean SCR contrasts using the thresholded square-root SCR score. Primary analyses retain all participants; responder/learner cohorts are sensitivity analyses.

1. **Within-Group Phase Effects**
   - Acquisition fear learning: `CS+ - CS-`.
   - Extinction vicarious fear/safety differentiation: `CSR - CSS`, `CSS - CS-`, and `CSR - CS-`.
   - Reinstatement vicarious fear/safety differentiation: `CSR - CSS`, `CSS - CS-`, and `CSR - CS-`.
   - The primary CSS/CSR contrast is `CSR - CSS`, because it directly tests whether threat-like responding to CSR exceeds safety-like responding to CSS.

2. **Between-Group Differences**
   - Acquisition: compare **SAD vs HC** on `CS+ - CS-`.
   - Extinction and reinstatement: compare **SAD vs HC** on `CSR - CSS`, `CSS - CS-`, and `CSR - CS-`; also evaluate the **Group (SAD, HC) x Drug (Placebo, Oxytocin)** interaction for post-acquisition phases.

## Part 2: CSS/CSR Dynamic Modeling Plan
The key scientific question is whether SAD and HC differ not only in mean `CSR - CSS`, but in how that contrast evolves across extinction and reinstatement trials. Use trial-level mixed-effects models as the primary inferential backbone and sliding-window plots as descriptive localization of when effects appear.

### Phase-specific dynamic models
Fit extinction and reinstatement separately:

```text
sqrtSCR ~ Condition(CSR vs CSS) * Trial_Z * Group + Drug + (1 | Subject)
```

Interpretation of primary terms:

- `Condition[CSR]`: average `CSR - CSS` difference within a phase.
- `Condition[CSR] x Trial_Z`: whether the `CSR - CSS` difference changes across trials.
- `Condition[CSR] x Group[SAD]`: whether SAD differs from HC in average `CSR - CSS`.
- `Condition[CSR] x Trial_Z x Group[SAD]`: whether SAD differs from HC in the dynamic trajectory of `CSR - CSS`.

### Integrated extinction-vs-reinstatement model
To test whether the SAD-HC dynamic difference changes between extinction and reinstatement, fit a combined model:

```text
sqrtSCR ~ Phase * Condition(CSR vs CSS) * Trial_Z * Group + Drug + (1 | Subject)
```

The primary term is:

```text
Phase x Condition x Trial_Z x Group
```

This tests whether SAD vs HC differences in the `CSR - CSS` trajectory differ across extinction and reinstatement.

### Sliding-window descriptive analyses
Complement the mixed models with subject-level sliding-window contrasts:

- 3-trial windows: `T1-T3`, `T2-T4`, ..., `T6-T8`.
- 2-trial windows: `T1-T2`, `T2-T3`, ..., `T7-T8`.
- Trial-1-excluded summary: `All = T2-T8`, `Early = T2-T4`, `Late = T5-T8`.

For each window, plot the ordered contrasts:

1. Acquisition: `CS+ - CS-`
2. Extinction: `CSR - CSS`, `CSS - CS-`, `CSR - CS-`
3. Reinstatement: `CSR - CSS`, `CSS - CS-`, `CSR - CS-`

Generate versions for all subjects, each diagnostic group, and each Group x Drug cell. Mark uncorrected one-sample `contrast > 0` results with `*` for descriptive visualization only; corrected/model-based inference should be prioritized for confirmatory claims.

### Working hypothesis
SAD participants may show greater persistence of threat-related physiological differentiation, reflected by stronger or less declining `CSR - CSS` during extinction and/or reinstatement compared with HC. In reinstatement, SAD may show stronger re-emergence or maintenance of `CSR > CSS`, consistent with reduced safety generalization or impaired regulation of vicariously acquired threat representations.

# Output Expectations
- Clean, modular Python code using `pandas`, `numpy`, `statsmodels` (for LME/regression), and `seaborn`/`matplotlib` for data visualization.
- Clear markdown cell separations distinguishing Preprocessing, Cohort Definition, Mean Analysis, and Dynamic Modeling.
- High-quality plots showing the trial-by-trial time-course trajectories of CS+ vs. CS- for each Group and Drug combination across all phases.