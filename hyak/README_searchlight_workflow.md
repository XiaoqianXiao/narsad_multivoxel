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

This pipeline supports per-cell stages (Cells 6–17) with checkpointing. Cells 1–5 always run.

### Checkpoints
Saved to:
`/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/LSS/results/wholebrain_parcellation_schaefer/checkpoints`

Each stage writes `cell_XX.joblib` and later stages can load them using `--resume`.

### Run a Single Stage
```bash
bash hyak/submit_mvpa_L2_schaefer_stage.sh 10 --resume
```

### Run All Stages (6–17)
```bash
bash hyak/submit_mvpa_L2_schaefer_driver.sh
```

### Notes
- Stage 10 covers **Analysis 1.3** (Top 5% Features) **and** the **Single-Trial Trajectories** block.
- All outputs are written to:
  `/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/LSS/results/wholebrain_parcellation_schaefer`

### Stage Dependencies (Cells 6–17)
Cells 1–5 always run. Stages 6–17 are analysis stages with checkpoints/intermediates.

```
Stage 6
├─ Stage 7
└─ Stage 8
   ├─ Stage 9
   ├─ Stage 10
   ├─ Stage 11
   ├─ Stage 12
   ├─ Stage 13
   ├─ Stage 14
   ├─ Stage 15
   ├─ Stage 16
   └─ Stage 17
```

Interpretation:
- **Stage 6** is the root analysis step.
- **Stage 7** depends on Stage 6.
- **Stage 8** depends on Stage 6.
- **Stages 9–17** depend on Stage 8 (they need `importance_scores`/`importance_masks`).
