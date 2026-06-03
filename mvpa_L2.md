# MVPA L2 Analysis Plan

## Executive Summary

This plan defines a multivoxel fMRI analysis of vicarious fear and safety learning in the NARSAD dataset. The study uses L2-regularized linear logistic regression to identify trial-wise neural patterns that distinguish vicarious threat from safety, then uses representational geometry, single-trial trajectories, decision-boundary metrics, anxiety symptoms, and SCR to test clinical relevance and physiological convergence.

The analysis is organized around five research aims:

1. **Neural Pattern Identification:** determine whether vicarious threat and safety cues have separable multivoxel representations.
2. **Anxiety Group Difference:** test whether SAD and HC participants differ in the geometry and dynamics of these representations.
3. **Clinical Relevance:** test whether neural learning profiles are associated with social anxiety and broader anxiety symptoms.
4. **Physiological Convergence:** test whether neural learning profiles align with SCR indices of peripheral threat/safety learning.
5. **Oxytocin Modulation:** test whether oxytocin changes the neural profile of vicarious fear/safety learning, especially in SAD.

Throughout this document, "L2 regression" refers to L2-regularized logistic regression used as a linear multivoxel decoder. Decoding is treated as the entry point for identifying information-bearing patterns; the main scientific interpretation comes from geometry, dynamics, and clinical/physiological convergence.

## 1. Scientific Premise

Social anxiety disorder is characterized by heightened sensitivity to socially relevant threat and evaluative cues. A vicarious learning task is well suited to this question because participants learn threat and safety through observation, a social learning process that is clinically relevant to SAD.

The key neural states are:

- `CSR`: cue associated with vicarious threat/fear learning.
- `CSS`: cue associated with vicarious safety learning.
- `CS-`: safe/background reference cue.
- `SHOCK`: shock/US event, used only for secondary threat-anchor analyses.

The central premise is that anxiety is not expected to alter only whether threat and safety can be decoded. SAD may alter the *organization and sensitivity* of the learning space: participants with SAD may show stronger cue differentiation, faster or more persistent updating, sharper decision evidence, or different safety/threat geometry. These differences could reflect enhanced sensitivity to social cues rather than impaired learning. Oxytocin may change these profiles by altering social salience, safety updating, threat maintenance, or uncertainty.

## 2. Conceptual Model

The analysis separates five linked but distinct claims:

1. **Pattern identification:** `CSR` and `CSS` can be decoded from multivoxel activity, establishing that vicarious threat and safety information is present.
2. **SAD-related neural alteration:** SAD and HC differ in the geometry, decoder evidence, decision-boundary confidence, or trial-wise dynamics of the threat-safety neural profile.
3. **Clinical relevance:** individual differences in the neural profile relate to anxiety symptom severity, especially LSAS and DASS anxiety.
4. **Physiological convergence:** individual differences in the neural profile align with SCR indices of peripheral threat/safety learning.
5. **Oxytocin modulation:** oxytocin changes the neural profile, especially if SAD-oxytocin patterns move toward HC-placebo patterns on metrics that differ between SAD-placebo and HC-placebo.

This structure keeps the interpretation disciplined. A significant decoder only establishes that threat/safety information is present. Evidence for SAD-related neural differences, symptom relevance, physiological learning convergence, and oxytocin modulation must come from the later aim-specific analyses.

## 3. Specific Aims And Hypotheses

### Aim 1: Identify Multivoxel Patterns Of Vicarious Threat And Safety

**Hypothesis 1:** In placebo sessions, `CSR` and `CSS` trials will be discriminable above chance within SAD and HC groups.

Primary estimand:

- Cross-validated `CSR` versus `CSS` decoding accuracy within each diagnostic group under placebo.

Primary inference:

- Accuracy versus a subject-aware label-permutation null.

Interpretation:

- Above-chance performance indicates that the feature space contains vicarious threat/safety information.
- The magnitude of accuracy is descriptive; mechanistic conclusions require downstream topology and trajectory analyses.

### Aim 2: Test Whether SAD Alters Threat-Safety Neural Profiles

**Hypothesis 2:** Relative to HC, SAD participants will show a different neural learning profile for vicarious safety and threat. This difference may reflect heightened sensitivity to socially learned cues, expressed as stronger differentiation, sharper classifier evidence, faster updating, or more persistent threat/safety representations. The most convincing evidence would be a coherent profile across geometry, classifier evidence, decision-boundary confidence, and trial-wise dynamics.

The placebo-session SAD versus HC test is organized around four questions:

1. **Where are safety and threat located in neural space?**
   Distance metrics test whether `CSS` and `CSR` occupy different positions relative to the safe/background cue `CS-`.

2. **How does the decoder read safety and threat trials?**
   Decoder-evidence metrics test whether held-out `CSS` trials look more threat-like, and whether held-out `CSR` trials look more or less threat-like, according to the `CSR` versus `CSS` classifier.

3. **How confidently does the decoder separate safety from threat?**
   Decision-boundary metrics test whether safety trials sit close to the `CSS`/`CSR` boundary, which would indicate ambiguous or uncertain safety representation even when the predicted class is correct.

4. **How do safety and threat representations change over learning?**
   Trajectory metrics test whether `CSS` patterns move toward the `CS-` safe reference and whether `CSR` patterns persist as threat-like across extinction and reinstatement.

Primary operational metrics:

| Construct | Metric | Operational definition | Predicted SAD-HC difference |
| --- | --- | --- | --- |
| Safety integration | `Neural_Dist_Safety_Background` | `dist(CSS, CS-)`: distance between the subject-level `CSS` centroid and `CS-` centroid | Altered in SAD; lower values suggest stronger safety integration, higher values suggest more distinct learned-safety representation |
| Threat-background distance | `Neural_Dist_Threat_Background` | `dist(CSR, CS-)`: distance between the subject-level `CSR` centroid and `CS-` centroid | Altered in SAD; higher values suggest stronger threat-background differentiation |
| Threat-safety distance | `Neural_Dist_Threat_Safety` | `dist(CSR, CSS)`: distance between the subject-level `CSR` centroid and `CSS` centroid | Altered in SAD; higher values suggest stronger threat-safety differentiation |
| Threat-like safety evidence | `Neural_ThreatLike_Safety` | `P(CSR | CSS)`: average held-out probability/evidence for the threat class on safety trials | Altered in SAD; higher values suggest safety carries more threat-like evidence |
| Safety-like safety evidence | `Neural_SafetyLike_Safety` | `P(CSS | CSS)`: average held-out probability/evidence for the safety class on safety trials | Altered in SAD; higher values suggest stronger safety evidence |
| Threat-like threat evidence | `Neural_ThreatLike_Threat` | `P(CSR | CSR)`: average held-out probability/evidence for the threat class on threat trials | Altered in SAD; higher values suggest stronger threat evidence |
| Safety-like threat evidence | `Neural_SafetyLike_Threat` | `P(CSS | CSR)`: average held-out probability/evidence for the safety class on threat trials | Altered in SAD; higher values suggest threat trials carry more safety-like evidence |
| Threat-safety decision separation | `Neural_Boundary_Separation` | `Neural_ThreatLike_Threat - Neural_ThreatLike_Safety` | Altered in SAD; higher values suggest sharper threat-safety evidence separation |
| Safety decision confidence | `Neural_Decision_Margin_CSS` | Average distance of held-out `CSS` trials from the `CSS`/`CSR` decision boundary | Altered in SAD; higher values suggest more confident safety classification |
| Safety updating | `Neural_Safety_Trajectory_Slope` | Within-subject slope of trial-wise `CSS` movement toward the `CS-` reference | Altered in SAD; higher values suggest stronger/faster safety updating |
| Threat maintenance | `Neural_Threat_Trajectory_Slope` | Within-subject slope of trial-wise `CSR` movement toward the reinstatement `CSR` or `SHOCK` threat reference | Altered in SAD; higher persistence suggests stronger threat maintenance |

Primary inference:

- The main placebo-only SAD versus HC test focuses on `Neural_Dist_Safety_Background`, `Neural_ThreatLike_Safety`, `Neural_SafetyLike_Safety`, `Neural_Boundary_Separation`, `Neural_Decision_Margin_CSS`, `Neural_Safety_Trajectory_Slope`, and `Neural_Threat_Trajectory_Slope`.
- `Neural_Dist_Threat_Background`, `Neural_Dist_Threat_Safety`, `Neural_ThreatLike_Threat`, and `Neural_SafetyLike_Threat` are interpreted as matched companion metrics that clarify whether the abnormality is safety-specific, threat-specific, or a broader reorganization of the threat-safety space.

Secondary inference:

- Additional decision uncertainty for `CSS`, reflected by entropy or whole-distribution uncertainty measures. This is secondary because `Neural_Decision_Margin_CSS` is the primary boundary-confidence metric; entropy adds information about the full shape of the classifier evidence distribution rather than the distance from the decision boundary alone.
- Reinstatement-specific or shock-anchor threat maintenance. This is secondary because `Neural_Threat_Trajectory_Slope` is the primary threat-dynamics metric; reinstatement and `SHOCK` anchors clarify whether group differences reflect persistent threat representation, stronger safety learning, or both.

### Aim 3: Test Clinical Relevance Of Neural Profiles

**Hypothesis 3:** Higher anxiety symptoms will be associated with altered neural sensitivity to vicarious safety/threat cues. Depending on the metric, this may appear as stronger cue differentiation, sharper class evidence, greater threat-like safety evidence, or altered safety/threat updating.

Primary symptom measures:

- `lsas_total`.
- `lsas_fear`.
- `lsas_avoid`.
- `dass_anxiety`.

Primary neural measures:

- `Neural_Dist_Safety_Background`.
- `Neural_ThreatLike_Safety`.
- `Neural_SafetyLike_Safety`.
- `Neural_Boundary_Separation`.
- `Neural_Decision_Margin_CSS`.
- `Neural_Safety_Trajectory_Slope`.
- `Neural_Threat_Trajectory_Slope`.

Matched companion neural measures:

- `Neural_Dist_Threat_Background`.
- `Neural_Dist_Threat_Safety`.
- `Neural_ThreatLike_Threat`.
- `Neural_SafetyLike_Threat`.

Primary inference:

- Symptom-neural association models controlling for diagnosis, drug condition, and prespecified covariates.
- Within-group associations when sample size is sufficient, to distinguish dimensional symptom effects from diagnostic group separation.

### Aim 4: Test Physiological Convergence With SCR Learning

**Hypothesis 4:** Neural safety/threat profiles will align with peripheral SCR indices of threat and safety learning. Convergence between multivoxel neural metrics and SCR strengthens the interpretation that the neural profile reflects learning-relevant physiology rather than classifier behavior alone.

Primary physiology measures:

- `SCR_SafetyMinusBackground`.
- `SCR_ThreatMinusSafety`.
- `SCR_Safety_Trajectory_Slope`.
- `SCR_Threat_Trajectory_Slope`.

Primary neural measures:

- `Neural_Dist_Safety_Background`.
- `Neural_ThreatLike_Safety`.
- `Neural_SafetyLike_Safety`.
- `Neural_Boundary_Separation`.
- `Neural_Decision_Margin_CSS`.
- `Neural_Safety_Trajectory_Slope`.
- `Neural_Threat_Trajectory_Slope`.

Matched companion neural measures:

- `Neural_Dist_Threat_Background`.
- `Neural_Dist_Threat_Safety`.
- `Neural_ThreatLike_Threat`.
- `Neural_SafetyLike_Threat`.

Primary inference:

- SCR-neural association models controlling for diagnosis, drug condition, and prespecified covariates.
- Within-group associations when sample size is sufficient, to distinguish physiological learning effects from diagnostic group separation.
- SCR responder/learner cohorts are used as sensitivity analyses to test whether neural findings are robust among participants with measurable peripheral acquisition learning.

### Aim 5: Test Whether Oxytocin Modulates The Neural Learning Profile

**Hypothesis 5:** Oxytocin will alter safety/threat neural profiles, with the strongest mechanistic interest in whether SAD-oxytocin patterns move toward HC-placebo patterns.

Primary estimand:

- The `Group * Drug` interaction for core topology, trajectory, and decision-boundary metrics.

Primary inference:

```text
neural_metric ~ Group * Drug + covariates
```

Interpretation framework:

- **HC-reference shift:** SAD-oxytocin shifts toward HC-placebo on metrics where SAD-placebo differs from HC-placebo. This should be described as a directional reference shift, not automatically as clinical improvement.
- **General drug effect:** oxytocin shifts SAD and HC in the same direction.
- **SAD-specific modulation:** oxytocin changes SAD but not HC, without necessarily moving SAD toward HC-placebo.
- **No modulation:** oxytocin does not meaningfully change the neural profile after accounting for uncertainty and sample size.

## 4. Analysis Hierarchy

The analysis distinguishes primary, secondary, and sensitivity results before inference:

- **Primary:** placebo `CSR` versus `CSS` decoding; placebo SAD versus HC tests of the prespecified core neural metrics; primary LSAS/DASS anxiety associations; primary SCR-neural convergence tests; `Group * Drug` tests of the same core neural metrics.
- **Secondary:** cross-group decoding, shock-anchor analyses, reinstatement-specific threat-maintenance tests, ECR/stress/depression associations, secondary SCR indices, and spatial realignment tests.
- **Sensitivity:** alternative masks, alternative feature spaces, SCR responder/learner cohorts, stricter trial/motion filters, robust outlier handling, and placebo-only versus all-drug clinical models.

The manuscript should lead with the primary family. Secondary and sensitivity findings should be used to test specificity and robustness, not to rescue unsupported primary claims.

### Primary Test Matrix

This matrix defines the executable core of the plan. Companion metrics and secondary analyses can clarify interpretation, but the primary claims should be anchored to this table.

| Aim | Primary population | Primary metric family | Primary model or test | Multiple-comparison family |
| --- | --- | --- | --- | --- |
| Aim 1: pattern identification | SAD-placebo and HC-placebo, separately | `CSR` versus `CSS` decoding accuracy | Subject-aware cross-validated L2 logistic regression with permutation testing | Decoding tests |
| Aim 2: SAD-HC neural profile | SAD-placebo versus HC-placebo | `Neural_Dist_Safety_Background`, `Neural_ThreatLike_Safety`, `Neural_SafetyLike_Safety`, `Neural_Boundary_Separation`, `Neural_Decision_Margin_CSS`, `Neural_Safety_Trajectory_Slope`, `Neural_Threat_Trajectory_Slope` | `neural_metric ~ Group + covariates` | Primary neural profile metrics |
| Aim 3: clinical relevance | Primary: placebo sample; sensitivity: all-drug sample with drug covariate | Core neural metrics with `lsas_total`, `lsas_fear`, `lsas_avoid`, `dass_anxiety` | `clinical_score ~ neural_metric + Group + Drug + covariates`; within-group follow-ups when powered | Primary clinical-neural associations |
| Aim 4: physiological convergence | Primary: participants with SCR and fMRI; sensitivity: SCR responder/learner cohorts | Core neural metrics with `SCR_SafetyMinusBackground`, `SCR_ThreatMinusSafety`, `SCR_Safety_Trajectory_Slope`, `SCR_Threat_Trajectory_Slope` | `scr_index ~ neural_metric + Group + Drug + covariates`; trial-wise mixed models when using trial-level SCR | Primary SCR-neural associations |
| Aim 5: oxytocin modulation | Full factorial sample: SAD-placebo, SAD-oxytocin, HC-placebo, HC-oxytocin | Same core neural metrics as Aim 2 | `neural_metric ~ Group * Drug + covariates` | Drug-modulation tests |

### Candidate Additions From Prior Project Work

`project.md` lists analyses that have been explored in the broader project. These analyses should be treated as a source of candidate additions, not as automatic primary tests. The final plan should keep the primary claims focused, then use selected prior analyses when they clarify mechanism, robustness, or interpretation.

| Prior project analysis | Best use in final plan | Recommendation |
| --- | --- | --- |
| Analysis 1.1: within-group `CSS` vs `CSR` neural dissociation | Establishes decodable vicarious threat/safety information; provides Haufe maps and permutation nulls | Keep as Aim 1 primary |
| Stage 11: empirical permutation-importance masks | Defines interpretable feature spaces for topology, trajectory, and boundary analyses | Keep, but report feature counts and fallback rules transparently |
| Analysis 1.2: static representational topology | Tests safety integration and threat/safety geometry | Keep as a core Aim 2 analysis |
| Analysis 1.3: dynamic representational drift | Tests safety restoration and threat maintenance using target-directed movement metrics | Keep if the drift metrics are stable and clearly interpretable |
| Analysis 1.3 Part 2: single-trial trajectories | Tests trial-wise safety updating and threat maintenance | Keep as the main dynamic Aim 2 analysis; use clear target-specific labels |
| Analysis 1.4: decision-boundary and uncertainty | Decomposes decoding into threat-like safety evidence, boundary separation, margin, and uncertainty | Keep as a core Aim 2 analysis |
| Cross-group decoding / reverse cross-decoding | Tests whether neural codes generalize across SAD and HC | Secondary mechanistic analysis, not a primary claim |
| Shock/US target analyses | Tests whether `CSR` or `CSS` align with actual aversive/shock representation | Sensitivity or secondary threat-anchor analysis |
| Safety restoration / threat discrimination | Re-expresses topology in clinically intuitive terms | Useful secondary framing if metrics match primary topology definitions |
| Drift efficiency | Adds mechanism for learning dynamics | Secondary unless it is more stable than trial-wise trajectory slopes |
| Probabilistic opening / decision probability extraction | Extends boundary/probability interpretation | Keep only if it uses the same probability definitions as Aim 2 |
| Spatial realignment | Tests whether oxytocin shifts SAD patterns toward the HC-placebo reference profile | Secondary Aim 5 mechanism |
| Clinical-neural correlations and partial correlations | Tests symptom relevance | Keep as Aim 3, with LSAS/DASS anxiety primary and ECR/stress/depression secondary |
| Trial-wise and subject-level SCR-neural coupling | Tests physiological convergence | Keep as Aim 4, with SCR responder/learner cohorts as sensitivity analyses |

The final plan should prioritize analyses that directly answer the five aims and are interpretable with the current sample size. Prior project analyses that are redundant, weakly powered, or hard to interpret should be labeled secondary or omitted from the main manuscript narrative.

## 5. Data Inputs

### Neural Data

Primary data are LSS single-trial beta estimates converted into trial-by-feature matrices:

- `X_ext`: phase 2 extinction/vicarious learning features.
- `y_ext`: phase 2 trial labels.
- `sub_ext`: phase 2 subject IDs.
- `X_reinst`: phase 3 reinstatement features.
- `y_reinst`: phase 3 trial labels.
- `sub_reinst`: phase 3 subject IDs.

Primary feature space:

- Fear-network voxelwise features.

Sensitivity feature spaces:

- Memory-fear-network voxelwise features.
- Whole-brain parcellation / Schaefer-Tian features.
- Whole-brain voxelwise features only if inference and compute constraints are adequately controlled.

### Participant Metadata

Required fields:

- `subject_id`.
- `Group`: `SAD` or `HC`.
- `Drug`: `Placebo` or `Oxytocin`.

Recommended covariates:

- Age.
- Sex/gender.
- Session/order.
- Motion summary, preferably mean framewise displacement.
- Valid trial count by condition.

### Symptoms And Physiology

Clinical measures:

- LSAS: `lsas_total`, `lsas_fear`, `lsas_avoid`.
- DASS: `dass_anxiety`, `dass_stress`, `dass_depression`.
- ECR: `ecr_total`, secondary.

SCR measures:

- `SCR_Safety_Mean`.
- `SCR_Threat_Mean`.
- `SCR_Background_Mean`.
- `SCR_SafetyMinusBackground`.
- `SCR_ThreatMinusSafety`.
- `SCR_Safety_Trajectory_Slope`.
- `SCR_Threat_Trajectory_Slope`.

SCR sensitivity cohort flags:

- `SCR_Physiological_Responder`: at least two acquisition CS+ trials with raw SCR amplitude >= 0.05 uS.
- `SCR_Simple_Acquisition_Differential_Learner`: physiological responder with acquisition `sqrt_scr(CS+) > sqrt_scr(CS-)`.
- `SCR_Habituation_Adjusted_Learner`: physiological responder with positive CS+ coefficient from `sqrt_scr ~ CS_type + Trial_Z`.
- `SCR_Late_Phase_Sensitivity_Learner`: late acquisition `CS+ > CS-` and late differential greater than early differential.

Primary MVPA analyses retain all valid fMRI participants. SCR-defined cohorts are used only as sensitivity analyses because responder/learner filters change the target population and can differentially alter SAD and HC sample composition.

## 6. Analysis Populations

Primary population:

- All participants with valid fMRI data, valid group/drug metadata, and sufficient `CSS` and `CSR` trials for the relevant model.

Placebo mechanistic population:

- SAD-placebo and HC-placebo participants. This is the primary population for group-difference analyses because it avoids conflating anxiety diagnosis with acute drug effects.

Full factorial population:

- SAD-placebo, SAD-oxytocin, HC-placebo, and HC-oxytocin. This is the primary population for oxytocin analyses.

Sensitivity populations:

- SCR physiological responders.
- SCR simple acquisition differential learners.
- SCR habituation-adjusted learners.
- SCR late-phase sensitivity learners.
- Participants passing stricter trial-count or motion thresholds.

For every analysis population, report retained subject counts by `Group`, `Drug`, and feature space.

## 7. Quality Control And Leakage Prevention

### Trial Inclusion

Include valid trials labeled:

- `CS-`.
- `CSS`.
- `CSR`.
- `SHOCK`, only for predefined secondary shock-anchor analyses.

Exclude trials with missing LSS estimates, invalid labels, severe artifacts, or failed first-level estimation.

Minimum requirements:

- At least 2 valid trials per condition for subject-level topology or trajectory metrics.
- Enough trials per class to support cross-validation for decoding.
- Report trial counts rather than hiding exclusions inside model code.

### Feature Scaling

Fit `StandardScaler()` inside the cross-validation pipeline. Scaling parameters must be learned only from training folds and applied to held-out folds.

### Cross-Validation

Use subject-aware cross-validation whenever models include trials from multiple participants:

- `StratifiedGroupKFold` for group-level decoding when feasible.
- `LeaveOneGroupOut` for subject-held-out decoding and subject-level probability summaries.
- Stratified folds only when subject grouping is not meaningful, with justification.

No held-out subject's data may contribute to scaling, hyperparameter selection, feature selection, calibration, or final model fitting for that held-out prediction.

### Feature Selection

Feature masks derived from the data must be computed inside the proper training structure when used for predictive evaluation. Group-level masks used for descriptive representational characterization must be clearly labeled as post-decoding characterization masks and not presented as independent prediction tests.

## 8. Primary Decoder

The primary model is linear L2-regularized logistic regression:

```text
Pipeline:
  StandardScaler()
  LogisticRegression(
    penalty = "l2",
    solver = "lbfgs",
    class_weight = "balanced",
    max_iter = 5000
  )
```

Primary contrast:

- `CSR` versus `CSS`.

Secondary contrasts:

- `CSS` versus `CS-`.
- `CSR` versus `CS-`.
- Multiclass `CS-`/`CSS`/`CSR`, only if class balance and interpretability are adequate.

Hyperparameter:

- Tune `C` using a prespecified log-spaced grid, for example `10^-2` to `10^2`.
- Use identical grids across groups unless a deviation is necessary and documented.

Primary model outputs:

- Cross-validated accuracy or balanced accuracy.
- Subject-level accuracy.
- Decision scores.
- Decision margins.
- Calibrated probabilities, only when calibration can be done without leakage and with stable folds.
- Refit final model for Haufe transformation and descriptive maps.

## 9. Analysis 1: Decoding And Pattern Identification

### 9.1 Within-Group Decoding

Run separate placebo-session decoders:

- SAD-placebo.
- HC-placebo.

Procedure:

1. Select `CSR` and `CSS` trials.
2. Fit L2 logistic regression with subject-aware nested or repeated cross-validation.
3. Estimate held-out decoding accuracy.
4. Generate a subject-aware label-permutation null.
5. Refit a final group model for descriptive map extraction.

Primary statistical result:

- Observed decoding accuracy versus permutation null.

Report:

- Number of subjects.
- Trial counts by class.
- Accuracy, confidence interval, and empirical p-value.
- Best `C`.
- Fold structure.
- Any convergence or failed-fold issues.

### 9.2 Cross-Group Generalization

Run cross-decoding as a functional specificity analysis:

- Train HC-placebo, test SAD-placebo.
- Train SAD-placebo, test HC-placebo.

Interpretation:

- Strong within-group decoding and weak cross-group generalization suggests group-specific representational structure.
- Strong cross-group generalization suggests shared safety-threat coding.
- Cross-decoding is secondary because group differences in trial count, noise, and feature stability can affect generalization.

### 9.3 Spatial Interpretation

Use complementary maps:

- Haufe-transformed patterns for activation-like interpretability.
- Cross-validated permutation importance for predictive contribution.

Do not interpret raw classifier coefficients as activation maps. Report map thresholds, selected feature counts, and feature-space coverage.

## 10. Analysis 2: Feature Masks For Characterization

Downstream topology, trajectory, and boundary analyses use important-feature masks derived from the primary `CSR` versus `CSS` decoder.

Primary characterization mask:

- Positive cross-validated permutation-importance features with empirical `p < .05`.
- Constructed separately for SAD and HC.

Primary group-comparison mask:

- A common feature mask derived without using the SAD versus HC contrast, such as the union of SAD-placebo and HC-placebo positive importance masks or an all-placebo importance mask.
- The common mask is used to verify that SAD-HC group differences are not an artifact of comparing metrics computed in different group-specific feature spaces.
- Group-specific masks remain useful for describing the native representational profile within each diagnostic group.

Sensitivity masks:

- Empirical `p < .01`.
- FDR-corrected permutation importance.
- Haufe high-magnitude masks.
- All-positive importance masks if the empirical mask is too sparse for stable characterization.

For every mask, report:

- Feature space.
- Threshold rule.
- Number of selected features by group.
- ROI/parcel coverage.
- Whether a fallback rule was used.

Mask-based characterization is not independent evidence of prediction unless the mask was selected entirely within training folds for the corresponding predictive test.

## 11. Analysis 3: Representational Topology

Purpose:

- Quantify how `CS-`, `CSS`, and `CSR` are arranged in neural space.

Procedure:

1. Apply the prespecified important-feature mask.
2. Compute subject-level condition centroids.
3. Estimate distances using crossnobis or shrinkage-Mahalanobis methods when feasible.
4. Save subject-level RDMs and derived scalar metrics.

Primary metrics:

- `Neural_Dist_Safety_Background = dist(CSS, CS-)`.
- `Neural_Dist_Threat_Safety = dist(CSR, CSS)`.
- `Neural_Dist_Threat_Background = dist(CSR, CS-)`.
- `Neural_Topology_Safety_Integration = dist(CSR, CSS) - dist(CSS, CS-)`.

Secondary metrics:

- `Neural_Threat_Bias = dist(CSR, CS-) - dist(CSS, CS-)`.
- Shock-anchor distances: `dist(SHOCK, CS-)`, `dist(SHOCK, CSS)`, `dist(SHOCK, CSR)`.

Primary placebo model:

```text
topology_metric ~ Group + covariates
```

Full factorial model:

```text
topology_metric ~ Group * Drug + covariates
```

Interpretation:

- `Neural_Dist_Safety_Background` indexes how closely learned safety resembles the safe/background reference; lower values suggest stronger safety integration, whereas higher values suggest a more distinct learned-safety representation.
- Higher `Neural_Dist_Threat_Safety` suggests greater threat-safety separation. This may reflect enhanced cue differentiation, not necessarily impairment.
- A SAD-HC difference is interpreted by considering the full profile across safety integration, threat-safety separation, decoder evidence, decision confidence, trajectories, symptoms, and SCR, rather than assigning adaptive or maladaptive meaning to a single metric.

## 12. Analysis 4: Trial-Wise Neural Trajectories

Purpose:

- Test whether neural representations change across learning in ways consistent with safety acquisition or threat persistence.

Primary safety trajectory:

- Source: `CSS` extinction trials.
- Target: `CS-` extinction centroid.
- Metric: movement toward `CS-` across trial order.

Primary threat trajectory:

- Source: `CSR` extinction trials.
- Target: `CSR` reinstatement centroid or `SHOCK` pattern for secondary threat-anchor analysis.
- Metric: maintenance of or movement toward threat-like representation.

Subject-level summaries:

- Mean trajectory score.
- Trial-wise slope.
- Initial distance.
- Projection magnitude.
- Cosine fidelity.

Trial-level model:

```text
neural_score ~ Group * Drug * Domain * Trial_Z + covariates + (1 | Subject)
```

If supported by the data, include a subject-level random slope for `Trial_Z`. If the model is unstable, simplify the random-effects structure and report the simplification.

Key contrasts:

- SAD-placebo versus HC-placebo safety slope.
- SAD-placebo versus SAD-oxytocin safety slope.
- `Group * Drug` and `Group * Drug * Trial_Z` interactions.

## 13. Analysis 5: Decision-Boundary Geometry

Purpose:

- Test whether safety cues carry threat-like or uncertain decision evidence.

Use cross-validated decision scores or calibrated probabilities from the `CSR` versus `CSS` decoder.

Primary metrics:

- `Neural_ThreatLike_Safety = P(CSR | CSS)`.
- `Neural_SafetyLike_Safety = P(CSS | CSS)`.
- `Neural_ThreatLike_Threat = P(CSR | CSR)`.
- `Neural_SafetyLike_Threat = P(CSS | CSR)`.
- `Neural_Boundary_Separation = Neural_ThreatLike_Threat - Neural_ThreatLike_Safety`.
- `Neural_Decision_Margin_CSS`.

Secondary metrics:

- `Decision_Margin_All`.
- Entropy.
- Kurtosis/sharpness.

Primary placebo model:

```text
boundary_metric ~ Group + covariates
```

Full factorial model:

```text
boundary_metric ~ Group * Drug + covariates
```

Expected SAD profile:

- Altered `Neural_ThreatLike_Safety`.
- Altered `Neural_SafetyLike_Safety`.
- Altered `Neural_ThreatLike_Threat` and `Neural_SafetyLike_Threat`.
- Altered `Neural_Boundary_Separation`.
- Altered `Neural_Decision_Margin_CSS`.
- Altered entropy or distributional uncertainty.

## 14. Analysis 6: Oxytocin Modulation

Primary model:

```text
neural_metric ~ C(Group, reference="HC") * C(Drug, reference="Placebo") + covariates
```

Primary coefficient:

```text
C(Group)[SAD] : C(Drug)[Oxytocin]
```

Core oxytocin metrics:

- `Neural_Dist_Safety_Background`.
- `Neural_ThreatLike_Safety`.
- `Neural_SafetyLike_Safety`.
- `Neural_Boundary_Separation`.
- `Neural_Decision_Margin_CSS`.
- `Neural_Safety_Trajectory_Slope`.
- `Neural_Threat_Trajectory_Slope`.

Companion oxytocin metrics:

- `Neural_Dist_Threat_Background`.
- `Neural_Dist_Threat_Safety`.
- `Neural_ThreatLike_Threat`.
- `Neural_SafetyLike_Threat`.

Planned follow-up contrasts:

- SAD-placebo versus HC-placebo.
- SAD-oxytocin versus SAD-placebo.
- SAD-oxytocin versus HC-placebo.
- HC-oxytocin versus HC-placebo.

Additional mechanistic tests:

- Spatial realignment: train HC-placebo template and compare SAD-placebo versus SAD-oxytocin test performance.
- Reverse cross-decoding: train SAD-placebo template and compare HC-placebo versus HC-oxytocin.
- Drift efficiency: test whether oxytocin changes safety movement toward `CS-` or threat maintenance.

Oxytocin findings should be described with directionality and confidence intervals, not only p-values.

## 15. Analysis 7: Clinical Relevance And Physiological Convergence

Create a subject-level table containing:

- Topology metrics.
- Trajectory metrics.
- Decision-boundary metrics.
- SCR indices.
- Clinical measures.
- Group, drug, and covariates.

### 15.1 Clinical Symptom Associations

Primary clinical model:

```text
clinical_score ~ neural_metric + Group + Drug + covariates
```

Group moderation model:

```text
clinical_score ~ neural_metric * Group + Drug + covariates
```

Drug moderation model:

```text
clinical_score ~ neural_metric * Group * Drug + covariates
```

Primary clinical family:

- `lsas_total`, `lsas_fear`, `lsas_avoid`, `dass_anxiety`.

Primary neural family:

- `Neural_Dist_Safety_Background`.
- `Neural_ThreatLike_Safety`.
- `Neural_SafetyLike_Safety`.
- `Neural_Boundary_Separation`.
- `Neural_Decision_Margin_CSS`.
- `Neural_Safety_Trajectory_Slope`.
- `Neural_Threat_Trajectory_Slope`.

### 15.2 SCR Physiological Convergence

Primary SCR-neural model:

```text
scr_index ~ neural_metric + Group + Drug + covariates
```

SCR group moderation model:

```text
scr_index ~ neural_metric * Group + Drug + covariates
```

SCR drug moderation model:

```text
scr_index ~ neural_metric * Group * Drug + covariates
```

Primary physiology family:

- `SCR_SafetyMinusBackground`.
- `SCR_ThreatMinusSafety`.
- `SCR_Safety_Trajectory_Slope`.
- `SCR_Threat_Trajectory_Slope`.

Primary neural family for SCR convergence:

- `Neural_Dist_Safety_Background`.
- `Neural_ThreatLike_Safety`.
- `Neural_SafetyLike_Safety`.
- `Neural_Boundary_Separation`.
- `Neural_Decision_Margin_CSS`.
- `Neural_Safety_Trajectory_Slope`.
- `Neural_Threat_Trajectory_Slope`.

SCR responder/learner cohorts are sensitivity populations, not replacement primary analyses. For each cohort, report retained subject counts by `Group` and `Drug`.

Secondary families:

- Matched companion neural metrics:
  - `Neural_Dist_Threat_Background`.
  - `Neural_Dist_Threat_Safety`.
  - `Neural_ThreatLike_Threat`.
  - `Neural_SafetyLike_Threat`.
- `dass_stress`, `dass_depression`, `ecr_total`.
- Other SCR indices.
- Shock-anchor neural metrics.
- Broad exploratory neural metrics.

Within-group analyses:

- Run within SAD and HC when sample size permits.
- Use partial correlations or regression controlling for age and sex/gender.
- Add motion and trial count as sensitivity covariates.

Interpretation rule:

- A neural-symptom association is strongest when it appears within group and survives covariate adjustment.
- Neural-SCR convergence is strongest when the direction of association matches the behavioral learning interpretation and remains stable in SCR responder/learner sensitivity cohorts.

## 16. Statistical Inference

### Decoding Inference

Use subject-aware permutation testing:

```text
p = (count(null >= observed) + 1) / (n_permutations + 1)
```

Permutation structure should preserve the exchangeability unit. If labels are shuffled within subject, state that explicitly. If labels are shuffled across trials, justify why that is appropriate.

### Group, Drug, And Clinical Models

For one value per subject:

- Use linear models with covariates.
- Report effect estimates, confidence intervals, p-values, and FDR-adjusted q-values.

For repeated measures:

- Use mixed-effects models.
- Include subject random intercepts.
- Include random slopes when estimable and theoretically justified.

Use robust or permutation-based sensitivity tests if residual assumptions, outliers, or small samples threaten inference.

### Multiple-Comparison Strategy

Control FDR within prespecified families:

- Decoding tests.
- Primary neural profile metrics.
- Companion neural profile metrics, reported separately from primary metrics.
- Primary clinical-neural associations.
- SCR/physiology associations.
- Sensitivity analyses, reported separately.

Do not pool primary, secondary, and sensitivity tests into one large correction family unless the manuscript frames them as a single exploratory screen.

## 17. Sensitivity Analyses

Feature-space and mask sensitivity:

1. Repeat core topology, boundary, and trajectory analyses with `p < .01` importance masks.
2. Repeat with FDR-corrected importance masks.
3. Repeat with Haufe-derived masks.
4. Repeat with all-positive importance masks when primary masks are too sparse.
5. Repeat using memory-fear-network and whole-brain parcellation features.
6. Repeat direct SAD-HC comparisons using common masks, group-specific masks, and whole-feature-space summaries.

Participant sensitivity:

7. Exclude participants with low valid-trial counts.
8. Add motion and trial-count covariates.
9. Repeat primary analyses in SCR-defined cohorts:
   - physiological SCR responders.
   - simple acquisition differential learners.
   - habituation-adjusted learners.
   - late-phase sensitivity learners.

Clinical and model sensitivity:

10. Test placebo-only clinical associations separately from all-drug associations.
11. Winsorize or robustly model neural and clinical outliers.
12. Repeat analyses excluding shock/US trials from mask construction.
13. Use shock/US only as a secondary target pattern.

For each sensitivity cohort or filter, report retained `Group * Drug` cell counts.

## 18. Reporting Plan

### Main Tables

Table 1: participant and data quality characteristics.

- Subjects by `Group` and `Drug`.
- Age, sex/gender, clinical scores.
- Valid trials per condition.
- Motion and quality metrics.
- SCR responder/learner counts.

Table 2: decoding results.

- Feature space.
- Group.
- Contrast.
- Accuracy or balanced accuracy.
- Confidence interval.
- Permutation p-value.
- Best `C`.
- Number of selected features.

Table 3: primary neural profile metrics.

- Topology, trajectory, and boundary metrics.
- Group and drug means.
- SAD-placebo versus HC-placebo effect.
- `Group * Drug` effect.
- FDR-adjusted q-values.

Table 4: clinical relevance and physiological convergence.

- Neural metric.
- Clinical/SCR measure.
- Model specification.
- Effect estimate.
- Confidence interval.
- p-value and q-value.

### Main Figures

Figure 1: study and analysis schematic.

- Vicarious learning task.
- LSS trial estimates.
- L2 decoding.
- Important-feature masks.
- Topology, trajectory, boundary, clinical, and SCR convergence analyses.

Figure 2: decoding and spatial patterns.

- SAD and HC placebo decoding.
- Permutation null distributions.
- Haufe maps and selected feature summaries.

Figure 3: representational topology.

- `CS-`, `CSS`, `CSR` RDM geometry.
- SAD-placebo versus HC-placebo.
- Primary topology metrics with confidence intervals.

Figure 4: neural trajectories.

- Safety trajectory toward `CS-`.
- Threat maintenance trajectory.
- Trial-wise group and drug effects.

Figure 5: decision-boundary and oxytocin effects.

- `Neural_ThreatLike_Safety`.
- `Neural_SafetyLike_Safety`.
- `Neural_ThreatLike_Threat`.
- `Neural_SafetyLike_Threat`.
- Boundary separation.
- Safety decision margin.
- `Group * Drug` patterns.

Figure 6: clinical relevance.

- Primary LSAS/DASS anxiety associations.
- Within-group regression panels when powered.
- Symptom-neural association estimates with confidence intervals.

Figure 7: physiological convergence.

- Neural-SCR associations for safety and threat indices.
- SCR responder/learner sensitivity cohorts.
- Concordance between neural trajectory metrics and SCR trajectory metrics.

## 19. Interpretation Standards

1. Decoding is necessary but not sufficient for mechanistic claims.
2. SAD-related neural differences should be inferred from converging topology, boundary, trajectory, symptom, and SCR evidence.
3. Stronger threat-safety separation is not automatically pathological; it may reflect enhanced social-cue sensitivity and should be interpreted alongside safety-background integration, symptoms, and SCR.
4. Oxytocin should be interpreted by direction and uncertainty. Movement of SAD-oxytocin toward HC-placebo can be described as an HC-reference shift, not automatically as improvement.
5. Clinical associations should not be claimed as dimensional anxiety effects if they are driven only by SAD versus HC separation.
6. Feature importance is feature-space specific; do not overstate anatomical localization unless results replicate across masks or feature spaces.
7. Report null and imprecise findings explicitly, especially for oxytocin interactions and within-group clinical associations.

## 20. Minimal High-Impact Analysis Set

If the manuscript must focus on the most convincing core results, prioritize:

1. Placebo `CSR` versus `CSS` L2 decoding in SAD and HC.
2. Permutation-importance masks with feature counts and Haufe maps.
3. Placebo SAD versus HC tests of:
   - `Neural_Dist_Safety_Background`.
   - `Neural_ThreatLike_Safety`.
   - `Neural_SafetyLike_Safety`.
   - `Neural_Boundary_Separation`.
   - `Neural_Decision_Margin_CSS`.
   - `Neural_Safety_Trajectory_Slope`.
   - `Neural_Threat_Trajectory_Slope`.
4. Companion SAD-HC metrics for interpretation:
   - `Neural_Dist_Threat_Background`.
   - `Neural_Dist_Threat_Safety`.
   - `Neural_ThreatLike_Threat`.
   - `Neural_SafetyLike_Threat`.
5. Primary clinical associations with LSAS and DASS anxiety.
6. Primary SCR physiological convergence tests.
7. `Group * Drug` tests of the same core neural metrics.
8. SCR responder/learner cohort sensitivity analyses.

This minimal set directly answers whether vicarious threat/safety neural profiles exist, whether SAD alters them, whether they are clinically meaningful, whether they converge with peripheral physiology, and whether oxytocin modulates the profile.

## 21. Reproducibility Checklist

Archive before final reporting:

- Script path and git commit.
- Feature-space name and input file paths.
- Feature matrix source: FearNetwork, MemoryFearNetwork, or Schaefer/Tian whole-brain parcellation.
- Subject inclusion table.
- Trial counts per subject and condition.
- Motion and quality-control summaries.
- Cross-validation split strategy.
- Hyperparameter grid.
- Number of permutations.
- Random seed.
- Decoding outputs.
- Haufe maps.
- Permutation-importance masks and feature counts.
- Mask rule: empirical `p < .05`, empirical `p < .01`, FDR-corrected, Haufe-derived, all-positive fallback, or common-mask comparison.
- Topology/RDM tables.
- Drift metrics: projection magnitude, cosine fidelity, and initial distance.
- Trajectory tables.
- Decision-boundary tables.
- Clinical/SCR merged table.
- FDR families and correction rules.
- Sensitivity cohort counts and results.

Useful project output bundles to archive with the manuscript supplement:

- `cell_06.joblib`: Analysis 1.1 decoding and refit model outputs.
- `stage11_importance_masks.joblib` and split SAD/HC mask files: permutation-importance feature masks.
- `analysis_12_topology.joblib`: representational topology outputs.
- `cell_12_trajectories.joblib`: dynamic trajectory outputs.
- `cell_13_decision_stats_opt.joblib`: decision-boundary and uncertainty outputs.
- `cell_16_opening_test.joblib`: probabilistic opening / decision-probability outputs.
- `cell_17_realignment.joblib`: spatial realignment outputs.
- `cell_18_reverse_cross_decoding.joblib`: reverse cross-decoding outputs.
