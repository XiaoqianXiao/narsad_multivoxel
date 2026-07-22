# NARSAD Multivoxel Project Context

Characterize the neural representation of vicarious safety/threat learning in SAD.
_Last updated: 2026-07-16. Revised to align with decision made during the 1:1 meeting with Angela .

## Project Purpose

This repository supports multivoxel fMRI analyses of vicarious threat and safety learning in the NARSAD dataset. The central scientific question is whether vicarious threat and safety cues are represented by group-specific neural signatures in participants with Social Anxiety Disorder (`SAD`) and Healthy Controls (`HC`), and whether those neural profiles relate to clinical symptoms, skin conductance response (`SCR`), and oxytocin modulation.

Decoding is treated as the entry point, not the final scientific claim. Classifier performance establishes whether condition-relevant information is present in the feature space. Interpretation should rely on converging evidence from representational geometry, decision-certainty metrics and single-trial learning dynamics, and their clinical associations, SCR convergence, and drug-modulation effects.

## Core Scientific Aims

1. **Group-specific neural representation identification:** in the placebo condition, test whether `CSR` and `CSS` cues are separable within the prespecified `FearNetwork` during extinction, separately for `SAD` and `HC` participants. Evaluate both functional specificity and spatial specificity.
2. **Characterization of SAD-HC differences:** test whether `SAD` participants differ from `HC` participants in threat-safety representational geometry, decision certainty, or learning dynamics.
3. **Clinical relevance:** test whether neural profiles of vicarious learning are associated with anxiety symptom severity, using `dass_anxiety` and `lsas_total` as primary clinical measures and other clinical variables as secondary supporting evidence.
4. **Physiological convergence:** test whether neural profiles of vicarious learning align with SCR indices of threat and safety learning.
5. **Oxytocin modulation:** test whether oxytocin shifts threat-safety neural profiles, focusing on `Group x Drug` effects.

## Analysis Scope Across Aims
- **Aim 1** is restricted to the placebo session to establish baseline neural representations before any oxytocin-induced modulation.
- **Aims 2–4** use the **placebo cohort as the primary analysis sample**. As a sensitivity analysis, the same models are also evaluated in the full sample by including **Drug** as an additional  factor to assess the robustness of the observed effect sizes.
- **Aim 5** analyzes the **combined placebo and oxytocin data** to directly quantify drug-related changes in neural representations and test `Group × Drug` interaction effects.

## Dataset, Design, And Conditions

### Participant Groups

- `SAD`: Social Anxiety Disorder.
- `HC`: Healthy Control.

### Drug Conditions

- `Placebo`.
- `Oxytocin`.

### Task Conditions

- `CSR`: vicarious threat cue.
- `CSS`: vicarious safety cue.
- `CS-`: safe/background reference cue.

### Task Phases

- `acquisition`: fear-learning phase.
- `extinction`: vicarious fear-extinction phase.
- `reinstatement`: fear-reinstatement phase with four shocks at the very beginning of the phase.

### Analysis Cohorts

- **Aims 1-4:** placebo participants only.
- **Aim 5:** placebo and oxytocin.
- **Aims 2-4 sensitivity analysis for effect in the whole sample:** placebo and oxytocin participants.
- **SCR sensitivity cohorts:** used for sensitivity analyses. Because `Group x Drug` subgroups may be small, SCR-defined sensitivity analyses should pool placebo and oxytocin participants within each SCR-defined subgroup rather than restricting to placebo only or splitting by drug, unless an analysis is explicitly designed as an Aim 5 `Group x Drug` sensitivity check.

SCR sensitivity cohort defination:
- `SCR_Physiological_Responder`: at least two acquisition `CS+` trials with raw SCR amplitude `>= 0.05 uS`.
- `SCR_Simple_Acquisition_Differential_Learner`: physiological responder with acquisition `sqrt_scr(CS+) > sqrt_scr(CS-)`.
- `SCR_Habituation_Adjusted_Learner`: physiological responder with positive `CS+` coefficient from `sqrt_scr ~ CS_type + Trial_Z`.
- `SCR_Late_Phase_Sensitivity_Learner`: late acquisition `CS+ > CS-` and late differential greater than early differential.

### SCR Transformations And Recoding

SCR analyses use trial-level peak-amplitude scores from `phaBase2Peak`. Trials marked bad are excluded when the source table provides a `bad` indicator. The raw condition labels are recoded so `CS+S` becomes `CSS`, `CS+R` becomes `CSR`, and `CS-` remains `CS-`. `CSS` and `CSR` are also pooled as the `CS+` domain for acquisition fear-learning summaries.

The primary SCR analysis score is a thresholded square-root transform:

```text
SCR_thresholded = raw SCR if raw SCR >= 0.02 uS, otherwise 0
sqrt_scr = sqrt(SCR_thresholded)
```

The `0.02 uS` threshold is used to code very small responses as zero before the square-root transform. Physiological responder status uses the stricter raw-response threshold of `0.05 uS`, requiring at least two acquisition `CS+` trials above threshold.

Trial order is recoded within subject, phase, and condition as `ConditionTrial`. Acquisition, extinction, and reinstatement trials are summarized across `All`, `Early`, and `Late` windows, with early trials defined from the beginning of the condition-specific sequence and late trials from the end of that sequence. For dynamic models, `Trial_Z` is computed by z-scoring phase trial order within each subject and phase.

Subject-level SCR metrics are derived from thresholded square-root SCR. Acquisition learning is summarized as `CS+ - CS-`. Extinction and reinstatement use targeted contrasts including `CSR - CSS`, `CSS - CS-`, `CSR - CS-`, and `(CSR - CS-) - (CSS - CS-)`. For neural-SCR convergence models, SCR indices are z-scored with the same 3 SD outlier rule used for other continuous subject-level association variables.

## Feature Spaces And Neural Inputs

### Feature Spaces

- **Primary feature space:** `FearNetwork`.
- **Sensitivity feature spaces:** `MemoryFearNetwork`, whole-brain features, and parcellation-based features such as `Schaefer` and `Tian`.

### Neural Inputs

Analyses use single-trial LSS beta estimates converted into feature matrices, condition labels, and subject vectors, commonly including:

- `X_ext`, `y_ext`, `sub_ext`.
- `X_reinst`, `y_reinst`, `sub_reinst`.

### Single-Trial LSS Estimation

Single-trial neural estimates are generated with a first-level Least Squares Separate (`LSS`) GLM before MVPA. For each subject, task, and `trial_ID`, the event file is recoded into two first-level conditions:

- `trial`: the target event for the current `trial_ID`.
- `others`: all non-target events in the same run.

This produces a separate trial-level beta/contrast image while modeling the remaining task events together as the `others` condition. The LSS model includes the preprocessed BOLD image, brain mask, event timing, and fMRIPrep confounds. Nuisance regressors include the 6 motion parameters, their first derivatives, up to 2 aCompCor components, cosine drift terms, and spike regressors for high-motion or high-DVARS time points. LSS is run without spatial smoothing by default to preserve trial-specific multivoxel pattern information.

The trial-level LSS outputs are then merged into subject-task 4D images and used to construct downstream MVPA feature matrices and labels for conditions such as `CSR`, `CSS`, `CS-`, `SHOCK`, and `FIXATION`.

Subject-wise centering should be applied within the cross-validation loops to normalize baseline variance across participants before classifier training.

## Machine-Learning Pipeline

The primary model is L2-regularized logistic regression trained to distinguish `CSR` from `CSS` in the extinction phase.

### Primary Decoder

- `StandardScaler` for feature scaling.
- `LogisticRegression`.
- `penalty="l2"`.
- `solver="lbfgs"`.
- `class_weight="balanced"` to account for unequal trial counts between conditions.
- `max_iter=5000` to support numerical convergence.
- Hyperparameter grid: `C` selected from a 20-point logspace between `0.010` and `100`.

### Validation Scheme

- **Outer CV:** 5-fold `StratifiedGroupKFold` for performance evaluation.
- **Inner CV:** 5-fold cross-validation for `C` selection.
- **Repeated nested CV:** repeat the full nested CV procedure 10 times with different random seeds to evaluate stability.
- **Grouping variable:** subject ID. Training and test sets must remain subject-disjoint.
- **Primary performance metric:** subject-level forced-choice accuracy. For each subject, compute mean held-out decision scores across trials and assign the condition using the argmax of the mean decision scores.
- **Permutation inference:** subject-level label shuffling with 5,000 iterations.

### Leakage Prevention

All feature scaling, subject-wise centering, feature selection, mask generation used for prediction, hyperparameter tuning, probability calibration, and model fitting must occur strictly inside the training folds. No held-out subject data may contribute to any preprocessing or model-selection step.

## Primary Neural Metrics

Use a stable primary metric family when comparing group differences, symptom associations, SCR convergence, and drug effects. Keep the primary metrics organized around the three neural-profile questions below. For the current FearNetwork extinction-focused analysis, use one primary metric per question and keep legacy slope or dispersion metrics as supporting diagnostics.
## Primary Aim 2 Neural Metrics

| Domain                        | Primary metric                            | Description                                                                                                                                                                                                    | Operational definition                                                                                                                                                                                                                                                                                                         | Scale and direction                                                                                                                                                                                                                                                        | Primary interpretation                                                                                                                                                                                        | Key reporting and inferential considerations                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ----------------------------- | ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Representational geometry** | ``Neural_Threat_Safety_Distance``         | Relative differentiation of the reinstated threat representation (`CSR`) and safety representation (`CSS`) with respect to the background safety reference (`CS−`), normalized by total CS−-anchored distance. | For each participant, compute the normalized difference between the distance of `CSR` from `CS−` and the distance of `CSS` from `CS−`:`[d(CSR, CS-) - d(CSS, CS-)] / [d(CSR, CS-) + d(CSS, CS-)]`                                                                                                                              | Bounded continuous score from approximately **-1 to 1** when component distances are nonnegative.**Positive:** `CSR` is farther from `CS−` than `CSS` is.**Zero:** `CSR` and `CSS` are equally distant from `CS−`.**Negative:** `CSS` is farther from `CS−` than `CSR` is. | Larger positive values indicate a representational configuration in which the threat cue is more distinct from the background safety reference than the safety cue is, scaled by total CS−-anchored distance. | Report the distance metric used, feature normalization, reference-pattern construction, and cross-validation procedure. Because this is a normalized relative difference score, it should be interpreted alongside the component distances `d(CSR, CS−)` and `d(CSS, CS−)` to determine which component drives an observed effect.                                                                                                                                                        |
| **Decision certainty**        | `Prototype_Certainty`                     | Strength of neural evidence supporting the expected safety state for `CSS` and expected threat state for `CSR`, relative to maximal posterior ambiguity.                                                       | Estimate the target-class posterior probabilities:`p_safety_css = P(safety \| CSS)``p_threat_csr = P(threat \| CSR)`Convert each posterior to certainty:`CSS_Certainty = 2 × \|p_safety_css − 0.50\|``CSR_Certainty = 2 × \|p_threat_csr − 0.50\|`Primary aggregate:`Prototype_Certainty = mean(CSS_Certainty, CSR_Certainty)` | Bounded continuous score from **0 to 1**.**0:** posterior probability equals `0.50`, indicating maximal ambiguity.**1:** posterior probability equals `0` or `1`, indicating maximal distance from ambiguity.                                                              | Higher values indicate stronger, less ambiguous decoder evidence for the represented state. The aggregate score summarizes certainty across the safety and threat conditions.                                 | Absolute distance from `0.50` measures certainty but not whether the decoder favors the scientifically expected class. A confidently incorrect posterior can therefore yield high certainty. Report `p_safety_css` and `p_threat_csr`, or signed target-class margins, alongside the aggregate certainty score. Decoder probabilities should be generated from held-out data and assessed for calibration. Condition-specific certainty scores should be retained for secondary analyses. |
| **Learning dynamics**         | `Neural_DynamicDiscrimination_Volatility` | Temporal instability in neural discrimination between threat and safety representations during extinction learning.                                                                                            | First calculate trial-wise neural discrimination:`Δ_t = ThreatEvidence(CSR_t) − SafetyEvidence(CSS_t)`Then calculate the root mean square of successive changes across the `T` ordered trials:`sqrt[(1 / (T − 1)) × Σ_(t=2)^T (Δ_t − Δ_(t−1))²]`                                                                               | Nonnegative continuous score in the same evidence units as `Δ_t`.**0:** no change in discrimination across successive trials.**Higher values:** larger trial-to-trial changes in discrimination.                                                                           | Higher values indicate greater temporal instability or volatility in neural threat–safety discrimination. Lower values indicate a more stable discrimination profile over the extinction period.              | Volatility quantifies fluctuation magnitude, not the mean level, direction, or adaptive quality of learning. Report it alongside the mean discrimination level and trial-wise trajectory. Trial ordering, missing-trial handling, minimum trial requirements, and any smoothing or residualization must be prespecified. Because volatility can be sensitive to measurement noise, trial-level evidence should be obtained using cross-validated or otherwise independent estimates.      |

### Prespecified primary formulas

```text
Neural_Threat_Safety_Distance =
    [d(CSR, CS−) − d(CSS, CS−)] / [d(CSR, CS−) + d(CSS, CS−)]
```

```text
CSS_Certainty =
    2 × |P(safety | CSS) − 0.50|

CSR_Certainty =
    2 × |P(threat | CSR) − 0.50|

Prototype_Certainty =
    (CSS_Certainty + CSR_Certainty) / 2
```

```text
Δ_t =
    ThreatEvidence(CSR_t) − SafetyEvidence(CSS_t)

Neural_DynamicDiscrimination_Volatility =
    sqrt[
        (1 / (T − 1)) ×
        Σ_(t=2)^T (Δ_t − Δ_(t−1))²
    ]
```

### Recommended publication-level reporting

For each metric, report the group-specific mean and standard deviation or median and interquartile range, the estimated between-group difference, a 95% confidence interval, standardized effect size, exact two-sided `p` value, and the prespecified multiple-comparison correction. Primary analyses should also document the neural feature space, prototype or reference-pattern construction, cross-validation scheme, participant-level aggregation procedure, and handling of missing or low-quality trials.



## Secondary Or Support Neural Metrics

| Category                      | Metric                           | Definition                                                                                                                                       |
| ----------------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Representational geometry** | `Neural_Dist_Safety_Background`  | Representational distance between `CSS` and `CS-` vectors.                                                                                       |
| **Representational geometry** | `Neural_Dist_Threat_Background`  | Representational distance between `CSR` and `CS-` vectors.                                                                                       |
| **Representational geometry** | `Neural_Dist_Threat_Safe`        | Representational distance between `CSR` and `CSS` vectors.                                                                                       |
| **Decision certainty**        | `Neural_Certainty_CSS`           | Certainty of the decoder evidence for the expected safety state of `CSS`, relative to chance-level ambiguity: `2 × \|P(safety \| CSS) − 0.50\|`  |
| **Decision certainty**        | `Neural_Certainty_CSR`           | Certainty of the decoder evidence for the expected threat state of `CSR`, relative to chance-level ambiguity: `2 × \|P(threat \| CSR) − 0.50\|`. |
| **Learning dynamics**         | `Neural_Safety_Trajectory_Slope` | Trial-wise vector translation of CSS toward the target safety reference (CS-) localized to the early extinction phase to mitigate floor effects. |
| **Learning dynamics**         | `Neural_Threat_Trajectory_Slope` | Trial-wise vector translation of CSR toward the true threat reference (CSR) localized to the early reinstatement phase.                          |


## Clinical Measures

Primary Aim 3 clinical outcome measures:

- `dass_anxiety`.
- `lsas_total`.

These are the primary symptom scores for clinical relevance analyses.

Secondary symptom or covariate measures may be used as supporting evidence, including:

- `lsas_fear`.
- `lsas_avoid`.
- `dass_stress`.
- `dass_depression`.
- `ecr_total`.
- `age`, `gender/sex`, `group`, and `drug`, when available and scientifically justified.

## Physiological Measures

Primary SCR metrics:
- `SCR_Safety_Trajectory_Slope`: slope of `SCR_Anticipatory` across `CSS` trials.
- `SCR_Threat_Trajectory_Slope`: slope of `SCR_Anticipatory` across `CSR` trials.
Secondary SCR metrics:
- `SCR_SafetyMinusBackground`: mean `SCR_Anticipatory(CSS)` - mean `SCR_Anticipatory(CS-)`.
- `SCR_ThreatMinusSafety`: mean `SCR_Anticipatory(CSR)` - mean `SCR_Anticipatory(CSS)`.

For Aim 4, treat `SCR_Safety_Trajectory_Slope` and `SCR_Threat_Trajectory_Slope` as the primary SCR convergence indices. Treat `SCR_SafetyMinusBackground` and `SCR_ThreatMinusSafety` as secondary SCR indices unless the analysis plan is updated before final inference.

## Aim-Specific Analysis Plan

### Aim 1: Group-Specific Neural Representation Identification

**Objective:** Determine whether distinct neural representations exist for vicarious threat (`CSR`) versus vicarious safety (`CSS`) in `SAD` and `HC` participants separately.

**Cohort:** `SAD-Placebo` and `HC-Placebo`.

**Primary hypothesis:** Both groups will show above-chance `CSR` vs `CSS` decoding accuracy, indicating that fMRI multivoxel patterns contain information about vicarious learning.

**Within-group decoding method:**

- Train L2-regularized logistic regression classifiers for `CSS` vs `CSR` separately in `SAD-Placebo` and `HC-Placebo`.
- Use nested 5-fold subject-aware CV with inner-loop hyperparameter tuning.
- Evaluate subject-level forced-choice accuracy.
- Assess significance against a 5,000-iteration permutation null distribution.

**Functional specificity / cross-group decoding:**

- Train on the full `HC-Placebo` dataset and test on `SAD-Placebo`.
- Train on the full `SAD-Placebo` dataset and test on `HC-Placebo`.
- Compare cross-group accuracy with within-group decoding performance.
- Use a generalization index based on cross-group 2AFC accuracy.

**Spatial specificity / weight similarity:**

- Extract multivariate discrimination weight vectors from optimized group-specific classifiers.
- Compute cosine similarity between mean `SAD` and mean `HC` weight vectors.
- Compare observed similarity against a 5,000-iteration null distribution generated by randomly shuffling group labels.
- Use Haufe-transformed patterns as secondary evidence for interpretable feature-space contributions.

### Aim 2: Characterization Of SAD-HC Differences

**Objective:** Quantify divergence between `SAD` and `HC` participants' neural representations of vicarious learning using threat-safety geometry, decision certainty, and temporal learning dynamics.

**Cohort:** placebo-session participants.

**Primary tests:**

- Compare `SAD` and `HC` on `Neural_Threat_Safety_Distance`.
- Compare `SAD` and `HC` on `Prototype_Certainty`.
- Compare `SAD` and `HC` on `Neural_DynamicDiscrimination_Volatility`.

**Secondary tests:**

- Geometry
	- `Neural_Dist_Safety_Background`
	- `Neural_Dist_Threat_Background`
	- `Neural_Dist_Threat_Safe`.
- Certainty
	- `Neural_Certainty_CSS`.
	- `Neural_Certainty_CSR`.
- Dynamics
	- `Neural_Safety_Trajectory_Slope`
	- `Neural_Threat_Trajectory_Slope`


**Statistical comparison:**
- Use independent-sample group comparisons for subject-level metrics.
- Validate inference using permutation testing where appropriate.
- Correct confirmatory test families using FDR.

### Aim 3: Clinical Relevance

**Objective:** Test whether neural vicarious learning profiles are associated with anxiety symptom severity.

**Cohort:** placebo-session participants, analyzed separately within `SAD-Placebo` and `HC-Placebo`.

**Primary clinical scores:**

- `dass_anxiety`.
- `lsas_total`.

**Secondary clinical scores:**

- `lsas_fear`.
- `lsas_avoid`.
- `dass_stress`.
- `dass_depression`.
- `ecr_total`.

**Model family:**

```text
z(clinical_score) ~ z(neural_metric) + covariates
```

Use OLS regression with normalized clinical scores and normalized neural metrics. Covariates may include `age`, `gender/sex`, motion/QC variables, or other scientifically justified variables, depending on availability and final model specification.

### Aim 4: Neural-SCR Convergence

**Objective:** Test whether neural vicarious learning profiles align with SCR indices of threat and safety learning.

**Cohort:** placebo-session participants, analyzed separately within `SAD-Placebo` and `HC-Placebo`.

**Primary SCR convergence metrics:**

- `SCR_Safety_Trajectory_Slope`.
- `SCR_Threat_Trajectory_Slope`.

**Secondary SCR convergence metrics:**

- `SCR_SafetyMinusBackground`.
- `SCR_ThreatMinusSafety`.

**Model family:**

```text
scr_index ~ neural_metric + covariates
```

Interpret convergence as physiological validation or dissociation of the neural metric, not as proof that the neural metric has one fixed psychological meaning.

### Aim 5: Oxytocin Modulation And Directional Reference Shift

**Objective:** Test whether oxytocin shifts threat-safety neural profiles through `Group x Drug` effects.

**Cohort:** both placebo and oxytocin sessions.

**Model family:**

```text
neural_metric ~ Group * Drug + covariates
```

**Interpretation framework:**

- **HC-reference shift:** `SAD-Oxytocin` shifts toward `HC-Placebo` on metrics where `SAD-Placebo` differs from `HC-Placebo`. This should be described as a directional reference shift, not automatically as clinical improvement.
- **General drug effect:** oxytocin shifts `SAD` and `HC` in the same direction.
- **SAD-specific modulation:** oxytocin changes `SAD` but not `HC`, without necessarily moving `SAD` toward `HC-Placebo`.
- **No modulation:** oxytocin does not meaningfully change the neural profile after accounting for uncertainty and sample size.

## Analysis Hierarchy

### Confirmatory Analyses

Confirmatory analyses support the main paper claims and should be prioritized for interpretation:

- Placebo `CSR` vs `CSS` decoding.
- Placebo `SAD` vs `HC` tests of prespecified primary neural metrics.
- Primary SCR-neural convergence tests.
- Associations of `dass_anxiety` and `lsas_total` with prespecified primary neural metrics.
- `Group x Drug` tests of the same primary neural metrics.

### Secondary Analyses

Secondary analyses provide mechanistic or interpretive extension:

- Secondary neural metrics.
- Secondary clinical scores.
- Secondary SCR indices.


### Sensitivity Analyses

Sensitivity analyses evaluate robustness:

- Alternative feature spaces and parcellations, including `MemoryFearNetwork`and whole-brain (`Schaefer`, and `Tian`).
- SCR-defined responder or learner cohorts.
- For aim 2 to 4, using the whole dataset with including drug as a factor instead of in primary analysis only use placebo dataset.

## Evaluation And Inference

- Use subject-aware validation for all trial-level decoding.
- Use subject-level permutation tests with 5,000 iterations for decoding and weight-similarity inference where specified.
- Correct confirmatory test families using FDR Benjamini-Hochberg at `alpha = 0.05`.
- Report exact subject counts, trial counts, and feature counts for every primary and sensitivity analysis.
- Report both significant and null results.

## Repository Layout

The active project files are currently organized under `code/`.
- `code/`: analysis notebooks, first-level/LSS scripts, MVPA scripts, visualization notebooks, and Hyak job scripts.
- `code/scripts/`: post-Hyak harmonization, statistics, export, and summary scripts for manuscript-facing MVPA L2 results.
- `code/hyak/`: SLURM submission scripts, Hyak-specific MVPA jobs, merge jobs, and inference scripts.
- `results/`: local or synchronized analysis outputs.
- `PROJECT_CONTEXT.md`: compact, current guide for future work in this repository.

## Analysis Workflow

The intended analysis sequence is:

1. Estimate trial-wise LSS beta images.
2. Convert beta images to feature matrices with condition labels and subject IDs.
3. Build primary and sensitivity feature spaces.
4. Run group-specific L2-regularized logistic decoding for the primary `CSR` vs `CSS` contrast.
5. Generate subject-aware permutation null distributions.
6. Refit interpretable models and export Haufe-transformed maps where appropriate.
7. Generate empirical feature-importance masks only within valid training structures when they are used for predictive evaluation.
8. Compute representational geometry metrics.
9. Compute decision-certainty and uncertainty metrics.
10. Compute trial-wise trajectory and drift metrics.
11. Harmonize subject-level neural, clinical, drug, and SCR variables.
12. Run Aim 2-5 statistical models.
13. Run predefined secondary and sensitivity analyses.
14. Export manuscript-facing tables, figures, and summary files.

## Key Output Families

Expected output families include:
- Decoding results and permutation null distributions.
- Cross-decoding and functional-specificity outputs.
- Weight-similarity and spatial-specificity outputs.
- Haufe-transformed maps.
- Permutation-importance masks.
- Representational geometry outputs.
- Trial-wise trajectory and drift outputs.
- Decision-certainty and uncertainty outputs.
- Clinical association outputs.
- SCR convergence outputs.
- Oxytocin modulation outputs.
- Sensitivity outputs for alternative feature spaces, mask thresholds, and SCR responder/learner cohorts.

Post-Hyak outputs described by `code/scripts/README_mvpa_l2_pipeline.md` include:
- `outputs/mvpa_l2/harmonized/scr_sensitivity_groups.csv`.
- `outputs/mvpa_l2/harmonized/mvpa_l2_subject_metrics.csv`.
- `outputs/mvpa_l2/stats/aim2_group_difference.csv`.
- `outputs/mvpa_l2/stats/aim3_clinical_relevance.csv`.
- `outputs/mvpa_l2/stats/aim4_scr_convergence.csv`.
- `outputs/mvpa_l2/stats/aim5_oxytocin_modulation.csv`.
- `outputs/mvpa_l2/stats/sensitivity_models_all.csv`.
- `outputs/mvpa_l2/stats/mvpa_l2_results_summary.md`.

## Statistical Guardrails

- Predefine primary, secondary, and sensitivity analyses before interpreting results.
- Use subject-aware validation for trial-level decoding whenever possible.
- Use FDR correction within predefined test families.
- Report null findings explicitly.
- Report subject counts, trial counts, and feature counts for every primary and sensitivity analysis.
- Keep placebo diagnostic analyses separate from oxytocin-modulation analyses.
- Interpret clinical associations separately from diagnostic group effects.
- Treat SCR convergence as physiological validation or dissociation, not as proof that the neural metric has one fixed psychological meaning.
- Do not promote exploratory findings to primary conclusions after seeing results.


## Leakage Prevention Rules

No held-out subject data may contribute to:

- Scaling.
- Subject-wise centering estimates.
- Feature selection.
- Hyperparameter tuning.
- Probability calibration.
- Model fitting.
- Mask generation used for predictive evaluation.

Preferred validation strategies:

- `StratifiedGroupKFold`.
- `LeaveOneGroupOut`, when appropriate for robustness checks.

`StandardScaler` must be fit inside the training fold only. Feature masks used for predictive evaluation must be generated inside the appropriate training structure. Group-specific masks should be labeled clearly and should not be treated as interchangeable anatomical activation maps.

## Reproducibility Checklist

For any rerun or new result export, record:

- Git commit or working-tree status.
- Input data paths.
- Feature space and mask definition.
- Subject inclusion table.
- Trial inclusion rules.
- Early-phase trial definition for trajectory metrics.
- Hyperparameter grid.
- Cross-validation strategy.
- Permutation count and random seed.
- Covariate set used in each statistical model.
- Multiple-comparison correction family.
- Software environment or container.
- QC summaries.
- Output paths.

On Hyak, the current post-Hyak pipeline expects outputs from the expensive feature-space MVPA jobs and can run downstream harmonization/statistics through `code/scripts/run_mvpa_l2_posthyak.sh`. The Hyak submission wrappers in `code/hyak/` are the preferred entry points for full cluster reruns.

## Interpretation Principles

- Decoding accuracy answers whether information is present; geometry, trajectories, margins, symptoms, and SCR answer what the representation may mean.
- SAD effects should be described as altered organization, sensitivity, certainty, or dynamics unless the metric profile directly supports an impairment interpretation.
- Oxytocin effects should be interpreted as directional shifts in neural profiles, not automatic clinical normalization or clinical improvement.
- Haufe maps and importance masks help interpret feature-space contributions, but feature importance is not the same as univariate activation.
- Strong claims should be supported by converging evidence across more than one analysis domain.

## Known Assumptions And Open Decisions

Before final manuscript-facing inference, confirm and document:

- The exact trial window defining "early extinction" and "early reinstatement."
- The final covariate set for Aim 2-5 models.
- Whether any probability calibration step is used; if used, calibration must occur inside training folds.
- The exact FDR families for confirmatory, secondary, and sensitivity tests.
- Whether cross-decoding metrics are reported as standalone generalization indices or compared formally with within-group performance.
- The final feature-space hierarchy if sensitivity analyses produce divergent results.
