#!/usr/bin/env python3
"""
FDR (Benjamini–Hochberg) correction for voxelwise p-maps (dynamic extinction).
Uses strict valid mask: brain_mask & all_finite & var>0 from subject maps.
"""
from __future__ import annotations

import argparse
import glob

import os
from typing import Dict, Tuple

import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img
from statsmodels.stats.multitest import multipletests
from scipy import stats
SCHAEFER_ATLAS_PATH = "/gscratch/fang/NARSAD/ROI/schaefer_2018/Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.nii.gz"


PAIR_LIST = [
    ("CS-", "CSS"),
    ("CS-", "CSR"),
    ("CSS", "CSR"),
]


def find_reference_lss(project_root: str, task: str) -> str:
    patterns = [
        os.path.join(
            project_root,
            "MRI/derivatives/fMRI_analysis/LSS/firstLevel/all_subjects/subjects",
            f"sub-*_task-{task}_contrast1.nii*",
        ),
        os.path.join(
            project_root,
            "MRI/derivatives/fMRI_analysis/LSS/firstLevel/all_subjects/wholeBrain_S4",
            f"sub-*_task-{task}_contrast1.nii*",
        ),
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No LSS files found with patterns: {patterns}")


def build_master_mask_from_reference(
    glasser_path: str,
    tian_path: str,
    reference_img_path: str,
) -> Tuple[np.ndarray, nib.Nifti1Image]:
    ref_img = nib.load(reference_img_path)
    img_s = nib.load(SCHAEFER_ATLAS_PATH)
    img_t = nib.load(tian_path)
    schaefer_res = resample_to_img(img_s, ref_img, interpolation="nearest")
    tian_res = resample_to_img(img_t, ref_img, interpolation="nearest")
    data_g = schaefer_res.get_fdata()
    data_t = tian_res.get_fdata()
    mask = (data_g > 0) | (data_t > 0)
    return mask, nib.Nifti1Image(mask.astype(np.uint8), ref_img.affine)


def build_label_maps(
    glasser_path: str,
    tian_path: str,
    reference_img_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    ref_img = nib.load(reference_img_path)
    img_s = nib.load(SCHAEFER_ATLAS_PATH)
    img_t = nib.load(tian_path)
    schaefer_res = resample_to_img(img_s, ref_img, interpolation="nearest")
    tian_res = resample_to_img(img_t, ref_img, interpolation="nearest")
    data_g = schaefer_res.get_fdata()
    data_t = tian_res.get_fdata()
    return data_g, data_t


def build_valid_mask_for_pair(merged_dir: str, pair_name: str, mask: np.ndarray) -> np.ndarray:
    maps = sorted(glob.glob(os.path.join(merged_dir, f"subjmap_{pair_name}_delta_*.nii.gz")))
    if not maps:
        raise FileNotFoundError(f"No subject maps found for {pair_name} in {merged_dir}")
    X = np.stack([nib.load(m).get_fdata()[mask] for m in maps], axis=0)
    finite_mask = np.all(np.isfinite(X), axis=0)
    var_mask = np.nanvar(X, axis=0) > 0
    return finite_mask & var_mask


def save_map(values: np.ndarray, mask: np.ndarray, ref_img: nib.Nifti1Image, out_path: str) -> None:
    vol = np.zeros(mask.shape, dtype=float)
    vol[mask] = values
    nib.save(nib.Nifti1Image(vol, ref_img.affine), out_path)


def load_effect_map(p_path: str, mask: np.ndarray) -> Tuple[np.ndarray, str | None]:
    candidates = [
        p_path.replace("_p.nii.gz", "_diff.nii.gz"),
        p_path.replace("_p.nii.gz", "_mean.nii.gz"),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return nib.load(cand).get_fdata()[mask], cand
    return np.full(mask.sum(), np.nan), None


def main() -> None:
    parser = argparse.ArgumentParser(description="FDR correction for dynamic extinction p-maps")
    parser.add_argument("--project_root", default="/gscratch/fang/NARSAD")
    parser.add_argument("--merged_dir", default="/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/LSS/searchlight/dyn_ext/merged")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--task", default="phase2")
    parser.add_argument("--glasser_atlas", default="/gscratch/fang/NARSAD/ROI/Glasser/HCP-MMP1_2mm.nii")
    parser.add_argument("--tian_atlas", default="/gscratch/fang/NARSAD/ROI/Tian/3T/Subcortex-Only/Tian_Subcortex_S4_3T_2009cAsym.nii.gz")
    parser.add_argument("--reference_lss", default=None)
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = args.merged_dir
    if args.reference_lss is None:
        args.reference_lss = find_reference_lss(args.project_root, args.task)

    mask, mask_img = build_master_mask_from_reference(
        args.glasser_atlas, args.tian_atlas, args.reference_lss
    )
    label_g, label_t = build_label_maps(args.glasser_atlas, args.tian_atlas, args.reference_lss)
    ijk = np.column_stack(np.where(mask))
    coords = nib.affines.apply_affine(mask_img.affine, ijk)

    valid_masks: Dict[str, np.ndarray] = {}
    for pair in PAIR_LIST:
        pair_name = f"{pair[0]}_vs_{pair[1]}"
        valid_masks[pair_name] = build_valid_mask_for_pair(args.merged_dir, pair_name, mask)

    p_dir = os.path.join(args.merged_dir, "permutation")
    search_dir = p_dir if os.path.isdir(p_dir) else args.merged_dir
    p_files = sorted(glob.glob(os.path.join(search_dir, "*_p.nii.gz")))
    if not p_files:
        raise FileNotFoundError(f"No p-maps found in {search_dir}")

    for p_path in p_files:
        base = os.path.basename(p_path)
        pair_name = None
        for pair in PAIR_LIST:
            pname = f"{pair[0]}_vs_{pair[1]}"
            if base.startswith(pname):
                pair_name = pname
                break
        if pair_name is None:
            continue
        p_img = nib.load(p_path)
        p = p_img.get_fdata()[mask]
        valid = valid_masks[pair_name] & np.isfinite(p)
        q = np.full(p.shape, np.nan, dtype=float)
        if np.any(valid):
            _, qv, _, _ = multipletests(p[valid], alpha=0.05, method="fdr_bh")
            q[valid] = qv
        out_path = p_path.replace("_p.nii.gz", "_q_fdr.nii.gz")
        save_map(q, mask, mask_img, out_path)
        sig = np.isfinite(q) & (q <= 0.05)
        print(f"{os.path.basename(out_path)}: sig voxels {int(sig.sum())}")

        if np.any(sig):
            effect, eff_path = load_effect_map(p_path, mask)
            sign = np.sign(effect)
            z = stats.norm.isf(np.clip(p, 1e-300, 1.0) / 2.0) * sign
            g_lab = label_g[mask]
            t_lab = label_t[mask]
            atlas = np.where(g_lab > 0, "Glasser", np.where(t_lab > 0, "Tian", "None"))
            label_id = np.where(g_lab > 0, g_lab, t_lab).astype(int)

            import pandas as pd
            df = pd.DataFrame({
                "x": coords[sig, 0],
                "y": coords[sig, 1],
                "z": coords[sig, 2],
                "p": p[sig],
                "q_fdr": q[sig],
                "z_stat": z[sig],
                "effect": effect[sig],
                "Atlas": atlas[sig],
                "LabelID": label_id[sig],
            })
            csv_path = p_path.replace("_p.nii.gz", "_sig_fdr.csv")
            df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
