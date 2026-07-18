# Primary Neural Metric Clinical Associations

Model table rows: 42. Merged subject rows before placebo/group filtering: 102.

## Sources

- Neural metrics: recomputed current primary metrics from /Users/xiaoqianxiao/projects/NARSAD/MRI/derivatives/fMRI_analysis/LSS/firstLevel/all_subjects/group_level/phase2_X_ext_y_ext_roi_voxels.npz because results/representative_neural_index/derived_subject_neural_indices.csv was missing Prototype_Certainty; reused metadata from results/representative_neural_index/derived_subject_neural_indices.csv.
- DASS: `/Users/xiaoqianxiao/projects/NARSAD/MRI/source_data/behav/SocialSafetyLearning-DASS_DATA_2026-04-25_2306.csv`.
- LSAS: `/Users/xiaoqianxiao/projects/NARSAD/MRI/source_data/behav/SocialSafetyLearning-LSASSubtotals_DATA_2026-04-25_2306.csv`.
- ECR: `/Users/xiaoqianxiao/projects/NARSAD/MRI/source_data/behav/SocialSafetyLearning-ECR_DATA_2026-04-25_2306.csv`.

## Primary Clinical Outcomes

| Group   | clinical_score   | metric                                  |   n |   estimate |   ci_low |   ci_high |       t |       p |   q_within_group_clinical_score |      r2 | covariates_used   |
|:--------|:-----------------|:----------------------------------------|----:|-----------:|---------:|----------:|--------:|--------:|--------------------------------:|--------:|:------------------|
| HC      | dass_anxiety     | Neural_Threat_Safety_Distance           |  27 |   -0.114   | -0.5502  |   0.3222  | -0.5407 | 0.5939  |                         0.8646  | 0.02502 | z_demo_age,Gender |
| HC      | dass_anxiety     | Prototype_Certainty                     |  27 |    0.379   | -0.04068 |   0.7988  |  1.868  | 0.07453 |                         0.2236  | 0.1427  | z_demo_age,Gender |
| HC      | dass_anxiety     | Neural_DynamicDiscrimination_Volatility |  27 |   -0.03613 | -0.4697  |   0.3974  | -0.1724 | 0.8646  |                         0.8646  | 0.0139  | z_demo_age,Gender |
| HC      | lsas_total       | Neural_Threat_Safety_Distance           |  28 |   -0.02899 | -0.4573  |   0.3993  | -0.1397 | 0.8901  |                         0.8901  | 0.04979 | z_demo_age,Gender |
| HC      | lsas_total       | Prototype_Certainty                     |  28 |    0.1336  | -0.3024  |   0.5697  |  0.6326 | 0.533   |                         0.7995  | 0.06462 | z_demo_age,Gender |
| HC      | lsas_total       | Neural_DynamicDiscrimination_Volatility |  28 |   -0.2369  | -0.6438  |   0.1701  | -1.201  | 0.2414  |                         0.7243  | 0.1029  | z_demo_age,Gender |
| SAD     | dass_anxiety     | Neural_Threat_Safety_Distance           |  22 |   -0.2547  | -0.7154  |   0.206   | -1.162  | 0.2605  |                         0.3908  | 0.1769  | z_demo_age,Gender |
| SAD     | dass_anxiety     | Prototype_Certainty                     |  23 |   -0.398   | -0.8357  |   0.03968 | -1.903  | 0.07227 |                         0.2168  | 0.245   | z_demo_age,Gender |
| SAD     | dass_anxiety     | Neural_DynamicDiscrimination_Volatility |  23 |   -0.1048  | -0.5883  |   0.3787  | -0.4538 | 0.6551  |                         0.6551  | 0.1107  | z_demo_age,Gender |
| SAD     | lsas_total       | Neural_Threat_Safety_Distance           |  22 |   -0.5554  | -0.9704  |  -0.1403  | -2.811  | 0.01155 |                         0.03466 | 0.3448  | z_demo_age,Gender |
| SAD     | lsas_total       | Prototype_Certainty                     |  23 |    0.03945 | -0.4488  |   0.5277  |  0.1691 | 0.8675  |                         0.8675  | 0.06038 | z_demo_age,Gender |
| SAD     | lsas_total       | Neural_DynamicDiscrimination_Volatility |  23 |    0.07459 | -0.4215  |   0.5707  |  0.3147 | 0.7564  |                         0.8675  | 0.06385 | z_demo_age,Gender |