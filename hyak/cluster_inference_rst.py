#!/usr/bin/env python3
"""
Post-merge cluster-level inference (FWE) for reinstatement (phase3) searchlight.

Uses subject-level maps saved in merged directory and runs permutation-based
cluster-mass inference with a cluster-forming threshold at z > 3.2.
Outputs z-maps and cluster-corrected p-maps (p<0.01).
"""
from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img
from nilearn.maskers import NiftiMasker
from nilearn.mass_univariate import permuted_ols
from scipy import stats


CS_LABELS = ["CS-", "CSS", "CSR"]


@dataclass
class SubjectInfo:
    group: str
    drug: str


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
        os.path.join(
            project_root,
            "MRI/derivatives/fMRI_analysis/LSS/firstLevel/all_subjects/wholeBrain_S4",
            "**",
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
    img_g = nib.load(glasser_path)
    img_t = nib.load(tian_path)
    glasser_res = resample_to_img(img_g, ref_img, interpolation="nearest")
    tian_res = resample_to_img(img_t, ref_img, interpolation="nearest")
    data_g = glasser_res.get_fdata()
    data_t = tian_res.get_fdata()
    mask = (data_g > 0) | (data_t > 0)
    return mask, nib.Nifti1Image(mask.astype(np.uint8), ref_img.affine)


def t_to_z(tvals: np.ndarray, df: int) -> np.ndarray:
    tvals = np.asarray(tvals, dtype=float)
    p = stats.t.sf(np.abs(tvals), df) * 2.0
    z = stats.norm.isf(p / 2.0) * np.sign(tvals)
    z[np.isnan(tvals)] = np.nan
    return z


def z_to_t_threshold(z_thresh: float, df: int) -> float:
    p = stats.norm.sf(z_thresh) * 2.0
    return stats.t.ppf(1.0 - p / 2.0, df)


def cluster_pvals(
    values: np.ndarray,
    tested_vars: np.ndarray,
    mask_img: nib.Nifti1Image,
    n_perm: int,
    two_sided: bool,
    seed: int,
    n_jobs: int,
    model_intercept: bool,
    z_thresh: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cluster-corrected p-values (mass) and z-map; invalid voxels are NaN."""
    finite_mask = np.all(np.isfinite(values), axis=0)
    var_mask = np.nanvar(values, axis=0) > 0
    valid = finite_mask & var_mask
    p_full = np.full(values.shape[1], np.nan, dtype=float)
    z_full = np.full(values.shape[1], np.nan, dtype=float)
    if not np.any(valid):
        return p_full, z_full, valid

    mask_data = mask_img.get_fdata().astype(bool)
    if int(mask_data.sum()) != values.shape[1]:
        raise ValueError("Mask voxel count does not match value columns.")
    mask_data_valid = mask_data.copy()
    mask_data_valid[mask_data] = valid
    valid_mask_img = nib.Nifti1Image(mask_data_valid.astype(np.uint8), mask_img.affine)

    vals = values[:, valid]
    n_samples = vals.shape[0]
    if model_intercept:
        design = np.column_stack([tested_vars, np.ones(n_samples)])
    else:
        design = tested_vars
    df = n_samples - np.linalg.matrix_rank(design)
    if df <= 0:
        raise ValueError("Non-positive degrees of freedom for z conversion.")
    t_thresh = z_to_t_threshold(z_thresh, df)

    masker = NiftiMasker(mask_img=valid_mask_img)
    masker.fit()
    out = permuted_ols(
        tested_vars,
        vals,
        model_intercept=model_intercept,
        n_perm=n_perm,
        two_sided_test=two_sided,
        random_state=seed,
        n_jobs=n_jobs,
        threshold=t_thresh,
        tfce=False,
        masker=masker,
        output_type="dict",
    )

    tmap = out.get("t", out.get("t_stat", out.get("tstat")))
    if tmap is None:
        raise KeyError("Could not find t-stat map in permuted_ols output.")
    tmap = np.asarray(tmap)[0]
    zmap = t_to_z(tmap, df)
    z_full[valid] = zmap

    logp = out.get("logp_max_cluster_mass", out.get("logp_max_cluster_size"))
    if logp is None:
        raise KeyError("Could not find cluster logp in permuted_ols output.")
    pvals = 10 ** (-np.asarray(logp)[0])
    p_full[valid] = pvals
    return p_full, z_full, valid


def save_map(values: np.ndarray, mask: np.ndarray, ref_img: nib.Nifti1Image, out_path: str) -> None:
    vol = np.zeros(mask.shape, dtype=float)
    vol[mask] = values
    img_out = nib.Nifti1Image(vol, ref_img.affine)
    nib.save(img_out, out_path)


def load_subject_maps(
    merged_dir: str,
    cond_list: List[str],
    mask: np.ndarray,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, SubjectInfo]]:
    meta_path = os.path.join(merged_dir, "subj_meta.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing subj_meta.csv in {merged_dir}")
    meta_df = pd.read_csv(meta_path)
    subj_scores: Dict[str, Dict[str, np.ndarray]] = {}
    subj_info: Dict[str, SubjectInfo] = {}
    for row in meta_df.itertuples(index=False):
        s_id = str(row.Subject).strip()
        subj_scores[s_id] = {}
        for cond in cond_list:
            map_path = os.path.join(merged_dir, f"subjmap_{cond}_{s_id}.nii.gz")
            if not os.path.exists(map_path):
                raise FileNotFoundError(f"Missing subject map: {map_path}")
            data = nib.load(map_path).get_fdata()
            subj_scores[s_id][cond] = data[mask]
        subj_info[s_id] = SubjectInfo(group=row.Group, drug=row.Drug)
    return subj_scores, subj_info


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster-level inference for reinstatement (phase3)")
    parser.add_argument("--project_root", default="/gscratch/fang/NARSAD")
    parser.add_argument("--merged_dir", default="/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/LSS/searchlight/rst/merged")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--task", default="phase3")
    parser.add_argument("--glasser_atlas", default="/gscratch/fang/NARSAD/ROI/Glasser/HCP-MMP1_2mm.nii")
    parser.add_argument("--tian_atlas", default="/gscratch/fang/NARSAD/ROI/Tian/3T/Subcortex-Only/Tian_Subcortex_S4_3T_2009cAsym.nii.gz")
    parser.add_argument("--reference_lss", default=None)
    parser.add_argument("--n_perm", type=int, default=5000)
    parser.add_argument("--z_thresh", type=float, default=3.1)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=8)
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = args.merged_dir

    if args.reference_lss is None:
        args.reference_lss = find_reference_lss(args.project_root, args.task)

    mask, mask_img = build_master_mask_from_reference(
        args.glasser_atlas,
        args.tian_atlas,
        args.reference_lss,
    )

    subj_scores, subj_info = load_subject_maps(args.merged_dir, CS_LABELS, mask)
    all_subs = sorted(subj_scores.keys())

    def select_subjects(group: str | None = None, drug: str | None = None) -> List[str]:
        selected = []
        for s_id in all_subs:
            info = subj_info[s_id]
            if group is not None and info.group != group:
                continue
            if drug is not None and info.drug != drug:
                continue
            selected.append(s_id)
        return selected

    # 1) Placebo within-condition (SAD and HC)
    for cond in CS_LABELS:
        for group in ["SAD", "HC"]:
            subs = select_subjects(group=group, drug="Placebo")
            if len(subs) < 3:
                continue
            values = np.stack([subj_scores[s][cond] for s in subs], axis=0)
            tested = np.ones((values.shape[0], 1))
            pvals, zvals, valid = cluster_pvals(
                values, tested, mask_img, args.n_perm, True, args.seed, args.n_jobs, False, args.z_thresh
            )
            base = os.path.join(args.out_dir, f"cluster_within_{cond}_{group}_PLC")
            save_map(zvals, mask, mask_img, base + "_z.nii.gz")
            save_map(pvals, mask, mask_img, base + "_clustp.nii.gz")
            sig = np.isfinite(pvals) & (pvals <= args.alpha)
            print(f"{base}: sig voxels {sig.sum()}")

    # 2) Placebo group differences (SAD vs HC)
    for cond in CS_LABELS:
        sad = select_subjects(group="SAD", drug="Placebo")
        hc = select_subjects(group="HC", drug="Placebo")
        if len(sad) < 3 or len(hc) < 3:
            continue
        all_subs_diff = sad + hc
        values = np.stack([subj_scores[s][cond] for s in all_subs_diff], axis=0)
        tested = np.array([1.0] * len(sad) + [-1.0] * len(hc))[:, None]
        pvals, zvals, valid = cluster_pvals(
            values, tested, mask_img, args.n_perm, True, args.seed, args.n_jobs, True, args.z_thresh
        )
        base = os.path.join(args.out_dir, f"cluster_diff_{cond}_SAD-HC_PLC")
        save_map(zvals, mask, mask_img, base + "_z.nii.gz")
        save_map(pvals, mask, mask_img, base + "_clustp.nii.gz")
        sig = np.isfinite(pvals) & (pvals <= args.alpha)
        print(f"{base}: sig voxels {sig.sum()}")

    # 3) OXT-PLC modulation within group
    for cond in CS_LABELS:
        for group in ["SAD", "HC"]:
            oxt = select_subjects(group=group, drug="Oxytocin")
            plc = select_subjects(group=group, drug="Placebo")
            if len(oxt) < 3 or len(plc) < 3:
                continue
            all_subs_mod = oxt + plc
            values = np.stack([subj_scores[s][cond] for s in all_subs_mod], axis=0)
            tested = np.array([1.0] * len(oxt) + [-1.0] * len(plc))[:, None]
            pvals, zvals, valid = cluster_pvals(
                values, tested, mask_img, args.n_perm, True, args.seed, args.n_jobs, True, args.z_thresh
            )
            base = os.path.join(args.out_dir, f"cluster_mod_{cond}_{group}_OXT-PLC")
            save_map(zvals, mask, mask_img, base + "_z.nii.gz")
            save_map(pvals, mask, mask_img, base + "_clustp.nii.gz")
            sig = np.isfinite(pvals) & (pvals <= args.alpha)
            print(f"{base}: sig voxels {sig.sum()}")


if __name__ == "__main__":
    main()
