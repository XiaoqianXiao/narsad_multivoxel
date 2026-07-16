# FearNetwork Neural Profile Exploration

This report keeps the original analysis structure but expands the neural profile vocabulary within the FearNetwork mask.

## Profile Domains

- Q1 geometry/topology: where safety, threat, and background sit in representational space.
- Q2 decision/evidence: whether patterns express safety-like or threat-like evidence.
- Q3 learning dynamics: trialwise change in safety/threat representational evidence.
- Q4 precision/dispersion: within-cue stability of the neural representation.
- Q5 activation/magnitude: raw mean or norm contrasts, treated as secondary because they are less representationally specific.

## Whole FearNetwork: Best Metric Per Profile

| phase | profile | metric | direction_summary | cohens_d_SAD_minus_HC | p | q_within_phase_feature | scalar_auc_abs_direction | all_subjects_adjusted_group_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase2_extinction | Q1_geometry_topology | Neural_ThreatTriangleOpenness | SAD lower than HC | -0.5308 | 0.06689 | 0.3838 | 0.6475 | 0.01392 |
| phase2_extinction | Q2_decision_evidence | Neural_ThreatEvidence | SAD lower than HC | -0.4932 | 0.108 | 0.3838 | 0.6056 | 0.01771 |
| phase2_extinction | Q3_learning_dynamics | Neural_DynamicDiscrimination_Volatility | SAD lower than HC | -0.6734 | 0.0178 | 0.3838 | 0.6817 | 0.005085 |
| phase2_extinction | Q4_precision_dispersion | Neural_SafetyVsBackgroundDispersion | SAD higher than HC | 0.4916 | 0.1016 | 0.3838 | 0.5885 | 0.07283 |
| phase2_extinction | Q5_activation_magnitude_secondary | Neural_RawMean_SafetyMinusBackground | SAD higher than HC | 0.5431 | 0.05805 | 0.3838 | 0.6258 | 0.06973 |
| phase3_reinstatement | Q1_geometry_topology | Neural_ThreatAxisSeparation | SAD higher than HC | 0.549 | 0.06118 | 0.8945 | 0.6153 | 0.09136 |
| phase3_reinstatement | Q2_decision_evidence | Neural_SafetyEvidence | SAD higher than HC | 0.3399 | 0.1907 | 0.8945 | 0.5536 | 0.2125 |
| phase3_reinstatement | Q3_learning_dynamics | Neural_Safety_LatePhaseEvidence | SAD higher than HC | 0.317 | 0.2239 | 0.8945 | 0.6039 | 0.2537 |
| phase3_reinstatement | Q4_precision_dispersion | Neural_SafetyVsBackgroundDispersion | SAD higher than HC | 0.1564 | 0.5723 | 0.8945 | 0.5731 | 0.6088 |
| phase3_reinstatement | Q5_activation_magnitude_secondary | Neural_RawMean_SafetyMinusBackground | SAD higher than HC | 0.3335 | 0.2103 | 0.8945 | 0.5649 | 0.2093 |

## ROI Localization: Strongest Interpretable Rows

| phase | profile | roi_name | metric | direction_summary | cohens_d_SAD_minus_HC | p | q_within_phase_feature | all_subjects_adjusted_group_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase2_extinction | Q1_geometry_topology | right_insula | Neural_ThreatTriangleOpenness | SAD lower than HC | -0.571 | 0.05335 | 0.4632 | 0.01412 |
| phase2_extinction | Q1_geometry_topology | right_insula | Neural_Safety_Differentiation | SAD lower than HC | -0.571 | 0.05335 | 0.4632 | 0.01412 |
| phase2_extinction | Q1_geometry_topology | hippocampus | Neural_Dist_Threat_Background | SAD lower than HC | -0.5863 | 0.05687 | 0.4463 | 0.009337 |
| phase2_extinction | Q1_geometry_topology | right_insula | Neural_Dist_Threat_Background | SAD lower than HC | -0.6131 | 0.04416 | 0.4632 | 0.01306 |
| phase2_extinction | Q1_geometry_topology | hippocampus | Neural_ThreatTriangleOpenness | SAD lower than HC | -0.5677 | 0.05225 | 0.4463 | 0.00852 |
| phase2_extinction | Q2_decision_evidence | right_insula | Neural_ThreatEvidence | SAD lower than HC | -0.6077 | 0.04858 | 0.4632 | 0.0131 |
| phase2_extinction | Q2_decision_evidence | hippocampus | Neural_ThreatEvidence | SAD lower than HC | -0.5566 | 0.07456 | 0.4463 | 0.01052 |
| phase2_extinction | Q2_decision_evidence | left_hippocampus | Neural_ThreatEvidence | SAD lower than HC | -0.548 | 0.07622 | 0.495 | 0.01316 |
| phase2_extinction | Q2_decision_evidence | insula | Neural_ThreatEvidence | SAD lower than HC | -0.5812 | 0.05974 | 0.6304 | 0.0151 |
| phase2_extinction | Q2_decision_evidence | right_hippocampus | Neural_ThreatEvidence | SAD lower than HC | -0.5179 | 0.0946 | 0.4844 | 0.01589 |
| phase2_extinction | Q3_learning_dynamics | left_hippocampus | Neural_DynamicDiscrimination_Volatility | SAD lower than HC | -0.7689 | 0.009768 | 0.495 | 0.003075 |
| phase2_extinction | Q3_learning_dynamics | hippocampus | Neural_DynamicDiscrimination_Volatility | SAD lower than HC | -0.7441 | 0.01066 | 0.4463 | 0.002892 |
| phase2_extinction | Q3_learning_dynamics | left_acc | Neural_DynamicDiscrimination_Volatility | SAD lower than HC | -0.7401 | 0.009027 | 0.4544 | 0.005262 |
| phase2_extinction | Q3_learning_dynamics | right_hippocampus | Neural_DynamicDiscrimination_Volatility | SAD lower than HC | -0.6621 | 0.0204 | 0.4844 | 0.005928 |
| phase2_extinction | Q3_learning_dynamics | left_acc | Neural_Safety_Volatility | SAD lower than HC | -0.6985 | 0.01748 | 0.4544 | 0.009122 |
| phase2_extinction | Q4_precision_dispersion | right_vmpfc | Neural_SafetyVsBackgroundDispersion | SAD higher than HC | 0.6229 | 0.03623 | 0.5253 | 0.02059 |
| phase2_extinction | Q4_precision_dispersion | vmpfc | Neural_SafetyVsBackgroundDispersion | SAD higher than HC | 0.5869 | 0.04692 | 0.4379 | 0.06558 |
| phase2_extinction | Q4_precision_dispersion | left_acc | Neural_SafetyDispersion | SAD higher than HC | 0.4957 | 0.09595 | 0.6 | 0.04811 |
| phase2_extinction | Q4_precision_dispersion | left_acc | Neural_SafetyPrecision | SAD lower than HC | -0.4957 | 0.09595 | 0.6 | 0.04811 |
| phase2_extinction | Q4_precision_dispersion | vmpfc | Neural_SafetyDispersion | SAD higher than HC | 0.4997 | 0.09081 | 0.4379 | 0.05865 |
| phase3_reinstatement | Q1_geometry_topology | left_acc | Neural_ThreatAxisSeparation | SAD higher than HC | 0.9119 | 0.002279 | 0.05926 | 0.00167 |
| phase3_reinstatement | Q1_geometry_topology | left_acc | Neural_Safety_ThreatAxisProjection | SAD lower than HC | -0.9119 | 0.002279 | 0.05926 | 0.00167 |
| phase3_reinstatement | Q1_geometry_topology | left_acc | Neural_ThreatToSafetyDistanceRatio | SAD higher than HC | 0.8033 | 0.01097 | 0.1901 | 0.04692 |
| phase3_reinstatement | Q1_geometry_topology | right_amygdala | Neural_SafetySpecificity | SAD higher than HC | 0.6431 | 0.0231 | 0.4183 | 0.04064 |
| phase3_reinstatement | Q1_geometry_topology | hippocampus | Neural_ThreatToBackgroundDistanceRatio | SAD higher than HC | 0.5833 | 0.05883 | 0.5323 | 0.07583 |
| phase3_reinstatement | Q2_decision_evidence | right_hippocampus | Neural_BoundarySeparation | SAD higher than HC | 0.6845 | 0.02457 | 0.4692 | 0.07861 |
| phase3_reinstatement | Q2_decision_evidence | right_amygdala | Neural_SafetyEvidence | SAD higher than HC | 0.6436 | 0.02302 | 0.4183 | 0.0404 |
| phase3_reinstatement | Q2_decision_evidence | left_acc | Neural_SafetyEvidence | SAD higher than HC | 0.4983 | 0.06042 | 0.5273 | 0.07413 |
| phase3_reinstatement | Q2_decision_evidence | amygdala | Neural_SafetyEvidence | SAD higher than HC | 0.474 | 0.07781 | 0.8119 | 0.08086 |
| phase3_reinstatement | Q2_decision_evidence | hippocampus | Neural_BoundarySeparation | SAD higher than HC | 0.5379 | 0.07847 | 0.5323 | 0.167 |

## Interpretation Notes

- The most manuscript-ready whole-network profile is geometry/topology, especially threat-vs-background openness during extinction.
- Decision/evidence metrics tell the same story in a classifier-like language: SAD tends to show weaker threat evidence and boundary separation in phase-2 extinction.
- Learning-dynamics slopes are weaker in the whole-network profile, so they are better framed as descriptive unless replicated or tied to behavior.
- ROI exploration suggests left ACC threat-axis geometry in phase-3 reinstatement is unusually strong; treat this as a targeted follow-up because the ROI search is larger than the whole-network test family.
- Raw activation/magnitude metrics can be included as secondary checks, but they should not replace representational geometry as the central neural index.
