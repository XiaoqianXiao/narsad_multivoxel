# Searchlight Workflow (Chunked Subject Maps + Post-Merge TFCE)

This workflow runs whole-brain searchlight analyses in two stages:
1) **Chunked subject-map generation** (fast, parallel)
2) **Post-merge TFCE** (single run per analysis/condition)

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

## Notes
- **Chunked stage** only saves subject maps (no p/q, no CSV summaries).
- **Post-merge TFCE** is the only stage that produces significance maps and summary CSVs.
- Use `CHUNKS=384` for chunked subject maps under 4h walltime.
- Atlas mask is now **Schaefer‑400 (17-network, 2mm MNI) + Tian subcortex**.
