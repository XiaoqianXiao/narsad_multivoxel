# Searchlight Workflow (Chunked Subject Maps + Post-Merge TFCE)

This workflow runs whole-brain searchlight analyses in three stages:
1) **Chunked subject-map generation** (fast, parallel)
2) **Merge chunk outputs** (per mode)
3) **Post-merge TFCE** (single run per analysis/condition)

All scripts are in `hyak/` and run on Hyak via Apptainer.

---

## 0) Prepare NPZ inputs (Schaefer + Tian)
Generate the voxelwise NPZ inputs used by all searchlight stages:

```bash
cd /gscratch/scrubbed/fanglab/xiaoqian/repo/narsad_multivoxel
python prepare_X_y_voxel_WholeBrain.py
```

This creates:
```
phase2_X_ext_y_ext_voxels_schaefer_tian.npz
phase3_X_reinst_y_reinst_voxels_schaefer_tian.npz
```

## 0.1) Required paths (Hyak)
These are already configured in:
- `submit_searchlight_stageA.sh`
- `submit_searchlight_merge_stageB.sh`
- `submit_searchlight_tfce_stageC.sh`

Paths:
- `PROJECT_ROOT=/gscratch/fang/NARSAD`
- `APP_PATH=/gscratch/scrubbed/fanglab/xiaoqian/repo/narsad_multivoxel/hyak`
- `OUT_BASE=/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/LSS/searchlight`
- `REFERENCE_LSS=/gscratch/fang/NARSAD/MRI/derivatives/fMRI_analysis/LSS/firstLevel/all_subjects/subjects/sub-N101_task-phase2_contrast1.nii.gz`
- `SCHAEFER_ATLAS=/gscratch/fang/NARSAD/ROI/schaefer_2018/Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.nii.gz`
- `TIAN_ATLAS=/gscratch/fang/NARSAD/ROI/Tian/3T/Subcortex-Only/Tian_Subcortex_S4_3T_2009cAsym.nii.gz`

---

## 1) Stage A: Chunked subject-map generation
These jobs **only save subject maps** when `--post_merge_tfce` is used.

From a login node:
```bash
cd /gscratch/scrubbed/fanglab/xiaoqian/repo/narsad_multivoxel/hyak
bash submit_searchlight_stageA.sh
```

Outputs written to:
```
${OUT_BASE}/ext
${OUT_BASE}/rst
${OUT_BASE}/dyn_ext
${OUT_BASE}/dyn_rst
${OUT_BASE}/crossphase
```

Note: NPZ inputs are now:
```
phase2_X_ext_y_ext_voxels_schaefer_tian.npz
phase3_X_reinst_y_reinst_voxels_schaefer_tian.npz
```

---

## 2) Stage B: Merge chunk outputs (parallel)
Submit **one merge job per analysis** in parallel:

```bash
bash submit_searchlight_merge_stageB.sh
```

This creates:
```
${OUT_BASE}/ext/merged
${OUT_BASE}/rst/merged
${OUT_BASE}/dyn_ext/merged
${OUT_BASE}/dyn_rst/merged
${OUT_BASE}/crossphase/merged
```

---

## 3) Stage C: Post-merge TFCE (parallel jobs)
Submit **one job per condition/analysis**:

```bash
bash submit_searchlight_tfce_stageC.sh
```

That script submits **parallel** jobs for:
- Extinction: CS-, CSS, CSR
- Reinstatement: CS-, CSS, CSR
- Dynamic: dyn_ext, dyn_rst
- Crossphase

If you need to **serialize** these jobs (to avoid job limits), let me know and I’ll add a sequential mode.

---

## 4) Outputs (post-merge TFCE stage)
Each merged folder will contain:
- `*_mean.nii.gz`, `*_p.nii.gz`, `*_q.nii.gz`
- `*_summary_contrasts.csv`
- `*_sig_merged.csv`
- `dynamic_summary.csv` / `crossphase_summary.csv` (as applicable)

---

## Cross-Half Pipeline (A2 → B2 → C2)
This pipeline computes **cross‑condition similarity + half‑split effects** using the same 3‑stage flow, but writes to separate folders.

### A2) Chunked cross-half subject maps
```bash
bash submit_searchlight_crosshalf_stageA2.sh
```
Outputs (per mode):
```
${OUT_BASE}/ext/crosshalf_subj_maps
${OUT_BASE}/rst/crosshalf_subj_maps
${OUT_BASE}/dyn_ext/crosshalf_subj_maps
${OUT_BASE}/dyn_rst/crosshalf_subj_maps
${OUT_BASE}/crossphase/crosshalf_subj_maps
```

### B2) Merge chunked cross-half maps
```bash
bash submit_merge_crosshalf_stageB2.sh
```
This merges chunked files inside each `crosshalf_subj_maps` folder.

### C2) Post-merge TFCE on cross-half maps
```bash
bash submit_searchlight_crosshalf_stageC2.sh
```
Outputs (per mode):
```
${OUT_BASE}/<mode>/crosshalf_permutation
```

---

## Notes
- **Chunked stage** only saves subject maps (no p/q, no CSV summaries).
- **Post-merge TFCE** is the only stage that produces significance maps and summary CSVs.
- Use `CHUNKS=384` for chunked subject maps under 4h walltime.
- Atlas mask is now **Schaefer‑400 (17-network, 2mm MNI) + Tian subcortex**.

## MVPA L2 Whole-Brain (Schaefer+Tian) Stage/Resume Workflow

This pipeline now uses **logical stages 1–9** for the main whole-brain analyses, plus **stage 17** for the parcel/searchlight RSM block. Cells 1–5 still always run.

### Output Roots
- Final outputs:
  `/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/LSS/results/wholebrain_parcellation_schaefer`
- Checkpoints:
  `/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/LSS/results/wholebrain_parcellation_schaefer/checkpoints`
- Intermediates:
  `/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/LSS/results/wholebrain_parcellation_schaefer/intermediate`

### Logical Stage Mapping
- `1`: Neural Dissociation
  old cells `6 + 7 + 8`
- `2`: Static Representational Topology
  old cell `9`
- `3`: Dynamic Representational Drift
  old cell `10` plus the single-trial trajectory block
- `4`: Decision Boundary Characteristics
  old cell `11`
- `5`: Safety Restoration
  old cell `12`
- `6`: Drift Efficiency
  old cell `13`
- `7`: Probabilistic Opening
  old cell `14`
- `8`: Spatial Re-Alignment
  old cell `15`
- `9`: Reverse Cross-Decoding
  old cell `16`
- `17`: Parcel/Searchlight RSM

### Exact Resume Behavior
Each logical stage now saves a full stage bundle in `intermediate/`:
- `stage01_NeuralDissociation.joblib`
- `stage02_StaticRepresentationalTopology.joblib`
- `stage03_DynamicRepresentationalDrift.joblib`
- `stage04_DecisionBoundaryCharacteristics.joblib`
- `stage05_SafetyRestoration.joblib`
- `stage06_DriftEfficiency.joblib`
- `stage07_ProbabilisticOpening.joblib`
- `stage08_SpatialReAlignment.joblib`
- `stage09_ReverseCrossDecoding.joblib`

If you rerun the same logical stage with `--resume`, the script will load the existing stage bundle and skip recomputation. That is the mechanism that avoids rerunning permutation tests or other expensive calculations for the exact same analysis.

### Run a Single Logical Stage
```bash
bash hyak/submit_mvpa_L2_schaefer_stage.sh 3 --resume
```

### Run All Logical Stages
```bash
bash hyak/submit_mvpa_L2_schaefer_driver.sh
```

### Importance Loading
Stage 1 includes the old importance block. Downstream stages can load importance in three ways:
- `--importance_source combined`
  requires `stage08_importance.joblib`
- `--importance_source group`
  requires `stage08_importance_SAD.joblib` and `stage08_importance_HC.joblib`
- `--importance_source auto`
  default; tries combined, then per-group

**Examples**
- Logical stage 1, SAD-only importance refresh:
  `bash hyak/submit_mvpa_L2_schaefer_stage.sh 1 --resume --stage8_group SAD`
- Logical stage 1, HC-only importance refresh:
  `bash hyak/submit_mvpa_L2_schaefer_stage.sh 1 --resume --stage8_group HC`
- Logical stage 2 using per-group importance only:
  `bash hyak/submit_mvpa_L2_schaefer_stage.sh 2 --importance_source group --resume`

### Stage Dependencies
```text
Stage 1
├─ Stage 2
├─ Stage 3
├─ Stage 4
├─ Stage 5
├─ Stage 6
├─ Stage 7
├─ Stage 8
└─ Stage 9

Stage 17
```

Interpretation:
- `Stage 1` is the root analysis stage for the main Schaefer+Tian whole-brain workflow.
- `Stages 2–9` depend on `Stage 1`.
- `Stage 17` is separate from the 1–9 chain.
