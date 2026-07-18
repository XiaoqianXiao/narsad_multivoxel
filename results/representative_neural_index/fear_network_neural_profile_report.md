# FearNetwork Neural Profile Exploration

This report keeps the original analysis structure but expands the neural profile vocabulary within the FearNetwork mask.

## Profile Domains

- Q1 geometry/topology: where safety, threat, and background sit in representational space.
- Q2 decision/evidence: whether patterns express safety-like or threat-like evidence.
- Q3 learning dynamics: trialwise change in safety/threat representational evidence.
- Q4 precision/dispersion: within-cue stability of the neural representation.
- Q5 activation/magnitude: raw mean or norm contrasts, treated as secondary because they are less representationally specific.
- Q6 shock-anchor: secondary reinstatement metrics that quantify cue alignment with SHOCK/US while controlling for global-amplitude components where possible.

## Whole FearNetwork: Best Metric Per Profile

| phase | profile | metric | direction_summary | cohens_d_SAD_minus_HC | hedges_g_SAD_minus_HC | rank_biserial_SAD_vs_HC | p | q_within_phase_feature | scalar_auc_abs_direction | all_subjects_adjusted_group_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase2_extinction | Q1_geometry_topology | Neural_ThreatTriangleOpenness | SAD lower than HC | -0.5308 | -0.5227 | -0.295 | 0.06689 | 0.2984 | 0.6475 | 0.01392 |
| phase2_extinction | Q2_decision_evidence | Neural_BoundarySeparation | SAD lower than HC | -0.4456 | -0.4387 | -0.2453 | 0.1107 | 0.2984 | 0.6227 | 0.06699 |
| phase2_extinction | Q3_learning_dynamics | Neural_DynamicDiscrimination_Volatility | SAD lower than HC | -0.6734 | -0.6631 | -0.3634 | 0.0178 | 0.2984 | 0.6817 | 0.005085 |
| phase2_extinction | Q4_precision_dispersion | Neural_TrialwiseEvidenceCertainty | SAD lower than HC | -0.704 | -0.6932 | -0.3199 | 0.01142 | 0.2984 | 0.6599 | 0.003752 |
| phase2_extinction | Q5_activation_magnitude_secondary | Neural_RawMean_SafetyMinusBackground | SAD higher than HC | 0.5431 | 0.5348 | 0.2516 | 0.05805 | 0.2984 | 0.6258 | 0.06973 |
| phase3_reinstatement | Q1_geometry_topology | Neural_ThreatAxisSeparation | SAD higher than HC | 0.549 | 0.5403 | 0.2305 | 0.06118 | 0.8478 | 0.6153 | 0.09136 |
| phase3_reinstatement | Q2_decision_evidence | Neural_SafetyEvidence | SAD higher than HC | 0.3399 | 0.3345 | 0.1071 | 0.1907 | 0.8842 | 0.5536 | 0.2125 |
| phase3_reinstatement | Q3_learning_dynamics | Neural_Safety_LatePhaseEvidence | SAD higher than HC | 0.317 | 0.312 | 0.2078 | 0.2239 | 0.8842 | 0.6039 | 0.2537 |
| phase3_reinstatement | Q4_precision_dispersion | Neural_TrialwiseEvidenceCertaintySNR | SAD higher than HC | 0.5145 | 0.5064 | 0.09091 | 0.1037 | 0.8842 | 0.5455 | 0.07352 |
| phase3_reinstatement | Q5_activation_magnitude_secondary | Neural_RawMean_SafetyMinusBackground | SAD higher than HC | 0.3335 | 0.3282 | 0.1299 | 0.2103 | 0.8842 | 0.5649 | 0.2093 |
| phase3_reinstatement | Q6_shock_anchor_secondary | Neural_ResidualizedShockAxis_CSR_Projection | SAD lower than HC | -0.7434 | -0.7318 | -0.3961 | 0.01536 | 0.8405 | 0.6981 | 0.007103 |

## ROI Localization: Strongest Interpretable Rows

| phase | profile | roi_name | metric | direction_summary | cohens_d_SAD_minus_HC | hedges_g_SAD_minus_HC | rank_biserial_SAD_vs_HC | p | q_within_phase_feature | all_subjects_adjusted_group_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase2_extinction | Q1_geometry_topology | right_insula | Neural_ThreatTriangleOpenness | SAD lower than HC | -0.571 | -0.5622 | -0.3727 | 0.05335 | 0.3012 | 0.01412 |
| phase2_extinction | Q1_geometry_topology | right_insula | Neural_Safety_Differentiation | SAD lower than HC | -0.571 | -0.5622 | -0.3727 | 0.05335 | 0.3012 | 0.01412 |
| phase2_extinction | Q1_geometry_topology | right_insula | Neural_Threat_Safety_Distance | SAD lower than HC | -0.571 | -0.5622 | -0.3727 | 0.05335 | 0.3012 | 0.01412 |
| phase2_extinction | Q1_geometry_topology | right_insula | Neural_Dist_Threat_Background | SAD lower than HC | -0.6131 | -0.6036 | -0.3043 | 0.04416 | 0.3012 | 0.01306 |
| phase2_extinction | Q1_geometry_topology | hippocampus | Neural_Dist_Threat_Background | SAD lower than HC | -0.5863 | -0.5773 | -0.3385 | 0.05687 | 0.3843 | 0.009337 |
| phase2_extinction | Q2_decision_evidence | right_insula | Neural_ThreatEvidence | SAD lower than HC | -0.6077 | -0.5984 | -0.3043 | 0.04858 | 0.3012 | 0.0131 |
| phase2_extinction | Q2_decision_evidence | hippocampus | Neural_ThreatEvidence | SAD lower than HC | -0.5566 | -0.5481 | -0.3385 | 0.07456 | 0.3843 | 0.01052 |
| phase2_extinction | Q2_decision_evidence | left_hippocampus | Neural_ThreatEvidence | SAD lower than HC | -0.548 | -0.5395 | -0.3012 | 0.07622 | 0.4183 | 0.01316 |
| phase2_extinction | Q2_decision_evidence | insula | Neural_ThreatEvidence | SAD lower than HC | -0.5812 | -0.5722 | -0.2484 | 0.05974 | 0.4295 | 0.0151 |
| phase2_extinction | Q2_decision_evidence | right_hippocampus | Neural_ThreatEvidence | SAD lower than HC | -0.5179 | -0.5099 | -0.2795 | 0.0946 | 0.4077 | 0.01589 |
| phase2_extinction | Q3_learning_dynamics | left_hippocampus | Neural_DynamicDiscrimination_Volatility | SAD lower than HC | -0.7689 | -0.7571 | -0.4596 | 0.009768 | 0.4183 | 0.003075 |
| phase2_extinction | Q3_learning_dynamics | hippocampus | Neural_DynamicDiscrimination_Volatility | SAD lower than HC | -0.7441 | -0.7326 | -0.472 | 0.01066 | 0.3843 | 0.002892 |
| phase2_extinction | Q3_learning_dynamics | left_acc | Neural_DynamicDiscrimination_Volatility | SAD lower than HC | -0.7401 | -0.7287 | -0.4037 | 0.009027 | 0.1866 | 0.005262 |
| phase2_extinction | Q3_learning_dynamics | right_hippocampus | Neural_DynamicDiscrimination_Volatility | SAD lower than HC | -0.6621 | -0.6519 | -0.3913 | 0.0204 | 0.4077 | 0.005928 |
| phase2_extinction | Q3_learning_dynamics | left_acc | Neural_Safety_Volatility | SAD lower than HC | -0.6985 | -0.6877 | -0.3634 | 0.01748 | 0.2167 | 0.009122 |
| phase2_extinction | Q4_precision_dispersion | left_acc | Neural_SafetyEvidenceCertainty | SAD lower than HC | -0.792 | -0.7798 | -0.5714 | 0.006282 | 0.1866 | 0.00161 |
| phase2_extinction | Q4_precision_dispersion | right_acc | Neural_TrialwiseEvidenceCertainty | SAD lower than HC | -0.7569 | -0.7453 | -0.4503 | 0.007792 | 0.3542 | 0.001681 |
| phase2_extinction | Q4_precision_dispersion | acc | Neural_TrialwiseEvidenceCertainty | SAD lower than HC | -0.79 | -0.7779 | -0.4006 | 0.005245 | 0.2514 | 0.00123 |
| phase2_extinction | Q4_precision_dispersion | acc | Neural_SafetyEvidenceCertainty | SAD lower than HC | -0.722 | -0.7109 | -0.5435 | 0.01214 | 0.2514 | 0.003626 |
| phase2_extinction | Q4_precision_dispersion | left_acc | Neural_TrialwiseEvidenceCertainty | SAD lower than HC | -0.8136 | -0.8011 | -0.3789 | 0.003962 | 0.1866 | 0.001184 |
| phase3_reinstatement | Q1_geometry_topology | left_acc | Neural_ThreatAxisSeparation | SAD higher than HC | 0.9119 | 0.8976 | 0.4448 | 0.002279 | 0.1105 | 0.00167 |
| phase3_reinstatement | Q1_geometry_topology | left_acc | Neural_Safety_ThreatAxisProjection | SAD lower than HC | -0.9119 | -0.8976 | -0.4448 | 0.002279 | 0.1105 | 0.00167 |
| phase3_reinstatement | Q1_geometry_topology | left_acc | Neural_ThreatToSafetyDistanceRatio | SAD higher than HC | 0.8033 | 0.7907 | 0.3701 | 0.01097 | 0.3546 | 0.04692 |
| phase3_reinstatement | Q1_geometry_topology | right_amygdala | Neural_SafetySpecificity | SAD higher than HC | 0.6431 | 0.633 | 0.3149 | 0.0231 | 0.7803 | 0.04064 |
| phase3_reinstatement | Q1_geometry_topology | hippocampus | Neural_ThreatToBackgroundDistanceRatio | SAD higher than HC | 0.5833 | 0.5741 | 0.3701 | 0.05883 | 0.662 | 0.07583 |
| phase3_reinstatement | Q2_decision_evidence | right_hippocampus | Neural_BoundarySeparation | SAD higher than HC | 0.6845 | 0.6738 | 0.3182 | 0.02457 | 0.5148 | 0.07861 |
| phase3_reinstatement | Q2_decision_evidence | right_amygdala | Neural_SafetyEvidence | SAD higher than HC | 0.6436 | 0.6335 | 0.3149 | 0.02302 | 0.7803 | 0.0404 |
| phase3_reinstatement | Q2_decision_evidence | left_acc | Neural_SafetyEvidence | SAD higher than HC | 0.4983 | 0.4905 | 0.3084 | 0.06042 | 0.7493 | 0.07413 |
| phase3_reinstatement | Q2_decision_evidence | amygdala | Neural_SafetyEvidence | SAD higher than HC | 0.474 | 0.4665 | 0.3052 | 0.07781 | 0.8981 | 0.08086 |
| phase3_reinstatement | Q2_decision_evidence | hippocampus | Neural_BoundarySeparation | SAD higher than HC | 0.5379 | 0.5294 | 0.224 | 0.07847 | 0.662 | 0.167 |

## Interpretation Notes

- The most manuscript-ready whole-network profile is geometry/topology, especially threat-vs-background openness during extinction.
- Decision/evidence metrics tell the same story in a classifier-like language: SAD tends to show weaker threat evidence and boundary separation in phase-2 extinction.
- Learning-dynamics slopes are weaker in the whole-network profile, so they are better framed as descriptive unless replicated or tied to behavior.
- The most informative shock-focused whole-network metric is residualized CSR projection on the subject-specific SHOCK-minus-CS- axis during reinstatement; SAD shows lower shock-axis alignment than HC, but this secondary family does not survive the broad whole-network FDR screen.
- ROI shock-anchor exploration highlights right vmPFC residualized CSR-minus-CSS shock-axis projection/cosine as the strongest localized follow-up signal. Treat it as supportive and hypothesis-generating unless promoted in a preregistered follow-up.
- ROI exploration suggests left ACC threat-axis geometry in phase-3 reinstatement is unusually strong; treat this as a targeted follow-up because the ROI search is larger than the whole-network test family.
- Raw activation/magnitude metrics can be included as secondary checks, but they should not replace representational geometry as the central neural index.
