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
| phase2_extinction | Q1_geometry_topology | Neural_ThreatTriangleOpenness_Normalized | SAD lower than HC | -0.5296 | -0.5215 | -0.3012 | 0.06753 | 0.4781 | 0.6506 | 0.01222 |
| phase2_extinction | Q2_decision_evidence | Neural_PrototypeBoundarySeparation | SAD lower than HC | -0.4456 | -0.4387 | -0.2453 | 0.1107 | 0.4781 | 0.6227 | 0.06699 |
| phase2_extinction | Q3_learning_dynamics | Neural_PrototypeDynamicDiscrimination_Volatility | SAD lower than HC | -0.6734 | -0.6631 | -0.3634 | 0.0178 | 0.4781 | 0.6817 | 0.005085 |
| phase2_extinction | Q4_precision_dispersion | Neural_TrialwiseEvidenceCertainty | SAD lower than HC | -0.704 | -0.6932 | -0.3199 | 0.01142 | 0.4781 | 0.6599 | 0.003752 |
| phase2_extinction | Q5_activation_magnitude_secondary | Neural_RawMean_SafetyMinusBackground | SAD higher than HC | 0.5431 | 0.5348 | 0.2516 | 0.05805 | 0.4781 | 0.6258 | 0.06973 |
| phase3_reinstatement | Q1_geometry_topology | Neural_Euclid_TriangleAngle_Threat | SAD lower than HC | -0.5126 | -0.5046 | -0.3247 | 0.0775 | 0.6297 | 0.6623 | 0.1031 |
| phase3_reinstatement | Q2_decision_evidence | Neural_Certainty_CSR | SAD higher than HC | 0.8405 | 0.8273 | 0.487 | 0.005864 | 0.3441 | 0.7435 | 0.004928 |
| phase3_reinstatement | Q3_learning_dynamics | Neural_DynamicDiscrimination_Volatility | SAD lower than HC | -0.4296 | -0.4229 | -0.2273 | 0.1406 | 0.8129 | 0.6136 | 0.1288 |
| phase3_reinstatement | Q4_precision_dispersion | Neural_ThreatEvidenceCertainty | SAD higher than HC | 0.8405 | 0.8273 | 0.487 | 0.005864 | 0.3441 | 0.7435 | 0.004928 |
| phase3_reinstatement | Q5_activation_magnitude_secondary | Neural_RawMean_SafetyMinusBackground | SAD higher than HC | 0.3335 | 0.3282 | 0.1299 | 0.2103 | 0.8239 | 0.5649 | 0.2093 |
| phase3_reinstatement | Q6_shock_anchor_secondary | Neural_ResidualizedShockAxis_CSR_Projection | SAD lower than HC | -0.7434 | -0.7318 | -0.3961 | 0.01536 | 0.4423 | 0.6981 | 0.007103 |

## ROI Localization: Strongest Interpretable Rows

| phase | profile | roi_name | metric | direction_summary | cohens_d_SAD_minus_HC | hedges_g_SAD_minus_HC | rank_biserial_SAD_vs_HC | p | q_within_phase_feature | all_subjects_adjusted_group_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase2_extinction | Q1_geometry_topology | right_insula | Neural_ThreatTriangleOpenness | SAD lower than HC | -0.571 | -0.5622 | -0.3727 | 0.05335 | 0.4171 | 0.01412 |
| phase2_extinction | Q1_geometry_topology | right_insula | Neural_Safety_Differentiation | SAD lower than HC | -0.571 | -0.5622 | -0.3727 | 0.05335 | 0.4171 | 0.01412 |
| phase2_extinction | Q1_geometry_topology | hippocampus | Neural_CorrTriangleAngle_Safety | SAD lower than HC | -0.6127 | -0.6033 | -0.3292 | 0.04161 | 0.4911 | 0.009701 |
| phase2_extinction | Q1_geometry_topology | left_hippocampus | Neural_CorrTriangleAngle_Safety | SAD lower than HC | -0.6068 | -0.5975 | -0.3168 | 0.04218 | 0.4522 | 0.01289 |
| phase2_extinction | Q1_geometry_topology | right_insula | Neural_CorrTriangleAngle_Safety | SAD lower than HC | -0.6072 | -0.5978 | -0.3106 | 0.04159 | 0.4171 | 0.01555 |
| phase2_extinction | Q2_decision_evidence | right_insula | Neural_Certainty_CSR | SAD lower than HC | -0.5429 | -0.5346 | -0.264 | 0.05111 | 0.4171 | 0.09121 |
| phase2_extinction | Q2_decision_evidence | left_acc | Neural_Certainty_CSR | SAD lower than HC | -0.5591 | -0.5505 | -0.2453 | 0.04452 | 0.5458 | 0.02742 |
| phase2_extinction | Q2_decision_evidence | right_acc | Neural_PrototypeBoundarySeparation | SAD lower than HC | -0.4313 | -0.4247 | -0.2516 | 0.1257 | 0.4117 | 0.1057 |
| phase2_extinction | Q2_decision_evidence | vmpfc | Neural_PrototypeBoundarySeparation | SAD lower than HC | -0.422 | -0.4155 | -0.2484 | 0.1255 | 0.6461 | 0.08393 |
| phase2_extinction | Q2_decision_evidence | acc | Neural_Certainty_CSR | SAD lower than HC | -0.411 | -0.4047 | -0.2453 | 0.1451 | 0.4446 | 0.09094 |
| phase2_extinction | Q3_learning_dynamics | left_hippocampus | Neural_DynamicDiscrimination_InitialFinal_Change | SAD higher than HC | 0.7741 | 0.7622 | 0.4814 | 0.006746 | 0.4522 | 0.01039 |
| phase2_extinction | Q3_learning_dynamics | hippocampus | Neural_DynamicDiscrimination_InitialFinal_Change | SAD higher than HC | 0.7616 | 0.7499 | 0.4876 | 0.0082 | 0.4911 | 0.01454 |
| phase2_extinction | Q3_learning_dynamics | left_hippocampus | Neural_PrototypeDynamicDiscrimination_Volatility | SAD lower than HC | -0.7689 | -0.7571 | -0.4596 | 0.009768 | 0.4522 | 0.003075 |
| phase2_extinction | Q3_learning_dynamics | hippocampus | Neural_PrototypeDynamicDiscrimination_Volatility | SAD lower than HC | -0.7441 | -0.7326 | -0.472 | 0.01066 | 0.4911 | 0.002892 |
| phase2_extinction | Q3_learning_dynamics | left_acc | Neural_PrototypeDynamicDiscrimination_Volatility | SAD lower than HC | -0.7401 | -0.7287 | -0.4037 | 0.009027 | 0.4288 | 0.005262 |
| phase2_extinction | Q4_precision_dispersion | right_acc | Neural_TrialwiseEvidenceCertainty | SAD lower than HC | -0.7569 | -0.7453 | -0.4503 | 0.007792 | 0.3912 | 0.001681 |
| phase2_extinction | Q4_precision_dispersion | acc | Neural_TrialwiseEvidenceCertainty | SAD lower than HC | -0.79 | -0.7779 | -0.4006 | 0.005245 | 0.4446 | 0.00123 |
| phase2_extinction | Q4_precision_dispersion | left_acc | Neural_TrialwiseEvidenceCertainty | SAD lower than HC | -0.8136 | -0.8011 | -0.3789 | 0.003962 | 0.3764 | 0.001184 |
| phase2_extinction | Q4_precision_dispersion | left_hippocampus | Neural_TrialwiseEvidenceCertainty | SAD lower than HC | -0.6859 | -0.6753 | -0.3696 | 0.0182 | 0.4522 | 0.007016 |
| phase2_extinction | Q4_precision_dispersion | right_vmpfc | Neural_SafetyVsBackgroundDispersion | SAD higher than HC | 0.6229 | 0.6133 | 0.3634 | 0.03623 | 0.7081 | 0.02059 |
| phase3_reinstatement | Q1_geometry_topology | left_acc | Neural_Euclid_ThreatToSafetyDistanceRatio | SAD higher than HC | 0.9807 | 0.9653 | 0.4968 | 0.00141 | 0.09876 | 0.008206 |
| phase3_reinstatement | Q1_geometry_topology | left_acc | Neural_ThreatAxisSeparation | SAD higher than HC | 0.9119 | 0.8976 | 0.4448 | 0.002279 | 0.09876 | 0.00167 |
| phase3_reinstatement | Q1_geometry_topology | left_acc | Neural_Safety_ThreatAxisProjection | SAD lower than HC | -0.9119 | -0.8976 | -0.4448 | 0.002279 | 0.09876 | 0.00167 |
| phase3_reinstatement | Q1_geometry_topology | left_acc | Neural_Euclid_TriangleAngle_Background | SAD higher than HC | 0.8699 | 0.8562 | 0.4026 | 0.003698 | 0.1202 | 0.006018 |
| phase3_reinstatement | Q1_geometry_topology | left_acc | Neural_Euclid_VicariousDiscrimination_Normalized | SAD higher than HC | 0.7973 | 0.7848 | 0.4188 | 0.007356 | 0.1913 | 0.01056 |
| phase3_reinstatement | Q2_decision_evidence | right_insula | Neural_Certainty_CSS | SAD higher than HC | 0.8353 | 0.8222 | 0.4968 | 0.006726 | 0.4372 | 0.002613 |
| phase3_reinstatement | Q2_decision_evidence | insula | Neural_Certainty_CSR | SAD higher than HC | 0.8939 | 0.8799 | 0.4383 | 0.005153 | 0.3349 | 0.0006491 |
| phase3_reinstatement | Q2_decision_evidence | left_insula | Neural_Certainty_CSR | SAD higher than HC | 0.8299 | 0.8169 | 0.4513 | 0.00744 | 0.4836 | 0.003636 |
| phase3_reinstatement | Q2_decision_evidence | amygdala | Neural_Certainty_CSR | SAD higher than HC | 0.802 | 0.7894 | 0.4253 | 0.01044 | 0.6785 | 0.004795 |
| phase3_reinstatement | Q2_decision_evidence | right_hippocampus | Neural_Certainty_CSR | SAD higher than HC | 0.6606 | 0.6502 | 0.3571 | 0.02551 | 0.345 | 0.01055 |

## Interpretation Notes

- The most manuscript-ready whole-network profile is geometry/topology, especially threat-vs-background openness during extinction.
- Decision/evidence metrics tell the same story in a classifier-like language: SAD tends to show weaker threat evidence and boundary separation in phase-2 extinction.
- Learning-dynamics slopes are weaker in the whole-network profile, so they are better framed as descriptive unless replicated or tied to behavior.
- The most informative shock-focused whole-network metric is residualized CSR projection on the subject-specific SHOCK-minus-CS- axis during reinstatement; SAD shows lower shock-axis alignment than HC, but this secondary family does not survive the broad whole-network FDR screen.
- ROI shock-anchor exploration highlights right vmPFC residualized CSR-minus-CSS shock-axis projection/cosine as the strongest localized follow-up signal. Treat it as supportive and hypothesis-generating unless promoted in a preregistered follow-up.
- ROI exploration suggests left ACC threat-axis geometry in phase-3 reinstatement is unusually strong; treat this as a targeted follow-up because the ROI search is larger than the whole-network test family.
- Raw activation/magnitude metrics can be included as secondary checks, but they should not replace representational geometry as the central neural index.
