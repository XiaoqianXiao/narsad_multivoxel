#!/usr/bin/env python3
"""Cross-phase searchlight: extinction vs reinstatement representational change.

Computes per-condition cross-phase similarity (phase2 vs phase3) within each searchlight sphere.
Analyses:
1) Placebo within-group effects (SAD, HC)
2) Placebo group differences (SAD vs HC)
3) Oxytocin modulation (OXT - PLC) within group
"""

from __future__ import annotations

import argparse
import os

import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import nibabel as nib
from numpy.linalg import norm
from scipy.spatial import cKDTree
from nilearn.maskers import NiftiMasker
from nilearn.mass_univariate import permuted_ols
from joblib import Parallel, delayed
SCHAEFER_ATLAS_PATH = "/gscratch/fang/NARSAD/ROI/schaefer_2018/Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.nii.gz"

CS_LABELS = ["CS-", "CSS", "CSR"]
PAIR_LIST = [("CS-", "CSS"), ("CS-", "CSR"), ("CSS", "CSR")]
MIN_VALID_FRAC = 0.80
SCHAEFER_N_ROIS = 400
SCHAEFER_YEO = 17
SCHAEFER_RES_MM = 2


@dataclass
class SubjectData:
    X_ext: np.ndarray
    y_ext: np.ndarray
    X_rst: np.ndarray
    y_rst: np.ndarray
    group: str
    drug: str


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.size == 0 or (mask.mean() < MIN_VALID_FRAC):
        return np.nan
    av = a[mask]
    bv = b[mask]
    na = norm(av)
    nb = norm(bv)
    if na == 0 or nb == 0:
        return np.nan
    return float(np.dot(av, bv) / (na * nb))


def mean_pattern(X: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return np.array([], dtype=float)
    valid_frac = np.mean(np.isfinite(X), axis=1)
    Xv = X[valid_frac >= MIN_VALID_FRAC]
    if Xv.size == 0:
        return np.full((X.shape[1],), np.nan)
    return np.nanmean(Xv, axis=0)


def get_default_n_jobs() -> int:
    env_val = os.environ.get("SLURM_CPUS_PER_TASK")
    if env_val and env_val.isdigit():
        return max(1, int(env_val))
    return max(1, os.cpu_count() or 1)


def build_batches(indices: np.ndarray, batch_size: int) -> List[np.ndarray]:
    if batch_size <= 0:
        raise ValueError("batch_size must be >= 1")
    return [indices[i:i + batch_size] for i in range(0, indices.size, batch_size)]


def split_half_indices(y_sub: np.ndarray, cond_list: List[str]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    halves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for cond in cond_list:
        idx = np.where(y_sub == cond)[0]
        if idx.size == 0:
            halves[cond] = (np.array([], dtype=int), np.array([], dtype=int))
            continue
        split = idx.size // 2
        halves[cond] = (idx[:split], idx[split:])
    return halves


def subset_subject_by_half(subj: SubjectData, cond_list: List[str], half: str) -> SubjectData:
    ext_halves = split_half_indices(subj.y_ext, cond_list)
    rst_halves = split_half_indices(subj.y_rst, cond_list)
    keep_ext = []
    keep_rst = []
    for cond in cond_list:
        e_first, e_second = ext_halves[cond]
        r_first, r_second = rst_halves[cond]
        if half == "H1":
            keep_ext.append(e_first)
            keep_rst.append(r_first)
        else:
            keep_ext.append(e_second)
            keep_rst.append(r_second)
    idx_e = np.sort(np.concatenate(keep_ext)) if keep_ext else np.array([], dtype=int)
    idx_r = np.sort(np.concatenate(keep_rst)) if keep_rst else np.array([], dtype=int)
    return SubjectData(
        X_ext=subj.X_ext[idx_e],
        y_ext=subj.y_ext[idx_e],
        X_rst=subj.X_rst[idx_r],
        y_rst=subj.y_rst[idx_r],
        group=subj.group,
        drug=subj.drug,
    )




def tfce_pvals(values: np.ndarray, tested_vars: np.ndarray, mask_img: nib.Nifti1Image, n_perm: int, two_sided: bool, seed: int, n_jobs: int, model_intercept: bool) -> Tuple[np.ndarray, np.ndarray]:
    finite_mask = np.all(np.isfinite(values), axis=0)
    var_mask = np.nanvar(values, axis=0) > 0
    valid = finite_mask & var_mask
    # Initialize with NaN so invalid voxels are excluded from inference
    p_full = np.full(values.shape[1], np.nan, dtype=float)
    if not np.any(valid):
        return p_full, valid
    mask_data = mask_img.get_fdata().astype(bool)
    if int(mask_data.sum()) != values.shape[1]:
        raise ValueError("Mask voxel count does not match value columns.")
    mask_data_valid = mask_data.copy()
    mask_data_valid[mask_data] = valid
    valid_mask_img = nib.Nifti1Image(mask_data_valid.astype(np.uint8), mask_img.affine)
    vals = values[:, valid]
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
        tfce=True,
        masker=masker,
        output_type="dict",
    )
    neglog = out.get("logp_max_tfce", out.get("logp_max_t"))
    pvals = 10 ** (-neglog[0])
    p_full[valid] = pvals
    return p_full, valid

def find_reference_lss(project_root: str, task: str) -> str:
    patterns = [
        os.path.join(
            project_root,
            "MRI/derivatives/fMRI_analysis/LSS/firstLevel/all_subjects",
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
    img_s = nib.load(SCHAEFER_ATLAS_PATH)
    img_t = nib.load(tian_path)

    from nilearn.image import resample_to_img

    schaefer_res = resample_to_img(img_s, ref_img, interpolation="nearest")
    tian_res = resample_to_img(img_t, ref_img, interpolation="nearest")

    data_g = schaefer_res.get_fdata()
    data_t = tian_res.get_fdata()
    mask = (data_g > 0) | (data_t > 0)
    out_img = nib.Nifti1Image(mask.astype(np.uint8), ref_img.affine)
    return mask, out_img


def save_map(values: np.ndarray, mask: np.ndarray, ref_img: nib.Nifti1Image, out_path: str) -> None:
    vol = np.zeros(mask.shape, dtype=float)
    vol[mask] = values
    img_out = nib.Nifti1Image(vol, ref_img.affine)
    nib.save(img_out, out_path)


def load_subject_maps_from_disk(
    out_dir: str,
    mask: np.ndarray,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, SubjectData]]:
    meta_path = os.path.join(out_dir, "subj_meta.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing subj_meta.csv in {out_dir}")
    meta_df = pd.read_csv(meta_path)
    subj_maps: Dict[str, Dict[str, np.ndarray]] = {}
    subj_data: Dict[str, SubjectData] = {}
    for row in meta_df.itertuples(index=False):
        s_id = str(row.Subject).strip()
        subj_maps[s_id] = {}
        for cond in CS_LABELS:
            map_path = os.path.join(out_dir, f"subjmap_{cond}_{s_id}.nii.gz")
            if not os.path.exists(map_path):
                raise FileNotFoundError(f"Missing subject map: {map_path}")
            data = nib.load(map_path).get_fdata()
            subj_maps[s_id][cond] = data[mask]
        subj_data[s_id] = SubjectData(
            X_ext=np.empty((0, 0)),
            y_ext=np.empty(0),
            X_rst=np.empty((0, 0)),
            y_rst=np.empty(0),
            group=row.Group,
            drug=row.Drug,
        )
    return subj_maps, subj_data


def load_crosshalf_maps_from_disk(
    maps_dir: str,
    cond_list: List[str],
    pair_list: List[Tuple[str, str]],
    mask: np.ndarray,
) -> Tuple[
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[str, Dict[str, np.ndarray]]],
    Dict[str, Dict[str, Dict[str, np.ndarray]]],
    Dict[str, SubjectData],
    bool,
]:
    meta_path = os.path.join(maps_dir, "subj_meta.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing subj_meta.csv in {maps_dir}")
    meta_df = pd.read_csv(meta_path)
    subj_scores: Dict[str, Dict[str, np.ndarray]] = {}
    pair_scores: Dict[str, Dict[str, np.ndarray]] = {}
    subj_scores_half: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {"H1": {}, "H2": {}}
    pair_scores_half: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {"H1": {}, "H2": {}}
    subj_data: Dict[str, SubjectData] = {}
    has_half = False
    for row in meta_df.itertuples(index=False):
        s_id = str(row.Subject).strip()
        subj_scores[s_id] = {}
        pair_scores[s_id] = {}
        for cond in cond_list:
            map_path = os.path.join(maps_dir, f"subjmap_{cond}_{s_id}.nii.gz")
            if not os.path.exists(map_path):
                raise FileNotFoundError(f"Missing subject map: {map_path}")
            subj_scores[s_id][cond] = nib.load(map_path).get_fdata()[mask]
        for cond_a, cond_b in pair_list:
            pair_name = f"{cond_a}_vs_{cond_b}"
            map_path = os.path.join(maps_dir, f"subjmap_cross_{pair_name}_{s_id}.nii.gz")
            if not os.path.exists(map_path):
                raise FileNotFoundError(f"Missing cross map: {map_path}")
            pair_scores[s_id][pair_name] = nib.load(map_path).get_fdata()[mask]

        h1_probe = os.path.join(maps_dir, f"subjmap_{cond_list[0]}_{s_id}_H1.nii.gz")
        h2_probe = os.path.join(maps_dir, f"subjmap_{cond_list[0]}_{s_id}_H2.nii.gz")
        if os.path.exists(h1_probe) and os.path.exists(h2_probe):
            has_half = True
            subj_scores_half["H1"][s_id] = {}
            subj_scores_half["H2"][s_id] = {}
            pair_scores_half["H1"][s_id] = {}
            pair_scores_half["H2"][s_id] = {}
            for cond in cond_list:
                h1_path = os.path.join(maps_dir, f"subjmap_{cond}_{s_id}_H1.nii.gz")
                h2_path = os.path.join(maps_dir, f"subjmap_{cond}_{s_id}_H2.nii.gz")
                if not os.path.exists(h1_path) or not os.path.exists(h2_path):
                    raise FileNotFoundError(f"Missing half maps for {s_id} {cond}")
                subj_scores_half["H1"][s_id][cond] = nib.load(h1_path).get_fdata()[mask]
                subj_scores_half["H2"][s_id][cond] = nib.load(h2_path).get_fdata()[mask]
            for cond_a, cond_b in pair_list:
                pair_name = f"{cond_a}_vs_{cond_b}"
                h1_path = os.path.join(maps_dir, f"subjmap_cross_{pair_name}_{s_id}_H1.nii.gz")
                h2_path = os.path.join(maps_dir, f"subjmap_cross_{pair_name}_{s_id}_H2.nii.gz")
                if not os.path.exists(h1_path) or not os.path.exists(h2_path):
                    raise FileNotFoundError(f"Missing half cross maps for {s_id} {pair_name}")
                pair_scores_half["H1"][s_id][pair_name] = nib.load(h1_path).get_fdata()[mask]
                pair_scores_half["H2"][s_id][pair_name] = nib.load(h2_path).get_fdata()[mask]

        subj_data[s_id] = SubjectData(
            X=np.empty((0, 0)),
            y=np.empty(0),
            group=row.Group,
            drug=row.Drug,
        )
    return subj_scores, pair_scores, subj_scores_half, pair_scores_half, subj_data, has_half


def compute_subject_crossphase_trial_scores(
    subj: SubjectData,
    neighbors: List[List[int]],
    voxel_indices: np.ndarray,
    min_voxels: int,
    batches: List[np.ndarray],
    n_jobs: int,
    cond_list: List[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    trial_scores: Dict[str, Dict[str, np.ndarray]] = {}
    voxel_indices = np.asarray(voxel_indices, dtype=int)
    for cond in cond_list:
        idx_e = np.where(subj.y_ext == cond)[0]
        idx_r = np.where(subj.y_rst == cond)[0]
        trial_scores[cond] = {
            "ext_idx": idx_e,
            "rst_idx": idx_r,
            "ext_labels": subj.y_ext[idx_e],
            "rst_labels": subj.y_rst[idx_r],
            "ext_scores": np.full((idx_e.size, voxel_indices.size), np.nan, dtype=float),
            "rst_scores": np.full((idx_r.size, voxel_indices.size), np.nan, dtype=float),
        }
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_crossphase_trial_batch)(
                subj.X_ext,
                subj.y_ext,
                subj.X_rst,
                subj.y_rst,
                neighbors,
                cond,
                min_voxels,
                batch,
                idx_e.size,
                idx_r.size,
            )
            for batch in batches
        )
        for voxels, ext_vals, rst_vals in results:
            pos = np.searchsorted(voxel_indices, voxels)
            trial_scores[cond]["ext_scores"][:, pos] = ext_vals
            trial_scores[cond]["rst_scores"][:, pos] = rst_vals
    return trial_scores


def save_subject_crossphase_trial_npz(
    trial_dir: str,
    s_id: str,
    trial_scores: Dict[str, Dict[str, np.ndarray]],
    cond_list: List[str],
    subj_maps: Dict[str, np.ndarray],
    voxel_indices: np.ndarray,
    chunk_idx: int | None,
) -> None:
    os.makedirs(trial_dir, exist_ok=True)
    for cond in cond_list:
        if cond not in trial_scores:
            continue
        ext_scores = trial_scores[cond]["ext_scores"].astype(np.float32, copy=False)
        rst_scores = trial_scores[cond]["rst_scores"].astype(np.float32, copy=False)
        suffix = f"_chunk{chunk_idx:03d}" if chunk_idx is not None else ""
        out_path = os.path.join(trial_dir, f"trialmaps_{cond}_{s_id}{suffix}.npz")
        mean_map = subj_maps.get(cond)
        mean_slice = None
        if mean_map is not None:
            mean_slice = mean_map[np.asarray(voxel_indices, dtype=int)]
        np.savez_compressed(
            out_path,
            ext_scores=ext_scores,
            rst_scores=rst_scores,
            ext_idx=trial_scores[cond]["ext_idx"],
            rst_idx=trial_scores[cond]["rst_idx"],
            ext_labels=trial_scores[cond]["ext_labels"],
            rst_labels=trial_scores[cond]["rst_labels"],
            mean_map=mean_slice.astype(np.float32, copy=False) if mean_slice is not None else None,
            voxels=voxel_indices,
            subject=s_id,
            condition=cond,
        )


def compute_crossphase_similarity(
    X_ext: np.ndarray,
    y_ext: np.ndarray,
    X_rst: np.ndarray,
    y_rst: np.ndarray,
    neigh_idx: np.ndarray,
    cond: str,
) -> float:
    idx_e = np.where(y_ext == cond)[0]
    idx_r = np.where(y_rst == cond)[0]
    if len(idx_e) == 0 or len(idx_r) == 0:
        return np.nan
    Xe = X_ext[idx_e][:, neigh_idx]
    Xr = X_rst[idx_r][:, neigh_idx]
    return cosine_sim(mean_pattern(Xe), mean_pattern(Xr))


def compute_crossphase_between_conditions(
    X_ext: np.ndarray,
    y_ext: np.ndarray,
    X_rst: np.ndarray,
    y_rst: np.ndarray,
    cond_a: str,
    cond_b: str,
) -> float:
    idx_ea = np.where(y_ext == cond_a)[0]
    idx_eb = np.where(y_ext == cond_b)[0]
    idx_ra = np.where(y_rst == cond_a)[0]
    idx_rb = np.where(y_rst == cond_b)[0]
    if idx_ea.size == 0 or idx_eb.size == 0 or idx_ra.size == 0 or idx_rb.size == 0:
        return np.nan
    Xe_a = X_ext[idx_ea]
    Xe_b = X_ext[idx_eb]
    Xr_a = X_rst[idx_ra]
    Xr_b = X_rst[idx_rb]
    sim_ab = cosine_sim(mean_pattern(Xe_a), mean_pattern(Xr_b))
    sim_ba = cosine_sim(mean_pattern(Xe_b), mean_pattern(Xr_a))
    return float(np.nanmean([sim_ab, sim_ba]))


def compute_crossphase_trial_scores(
    X_ext: np.ndarray,
    y_ext: np.ndarray,
    X_rst: np.ndarray,
    y_rst: np.ndarray,
    neigh_idx: np.ndarray,
    cond: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx_e = np.where(y_ext == cond)[0]
    idx_r = np.where(y_rst == cond)[0]
    if idx_e.size == 0 or idx_r.size == 0:
        return np.full(idx_e.size, np.nan), np.full(idx_r.size, np.nan), idx_e, idx_r
    Xe = X_ext[idx_e][:, neigh_idx]
    Xr = X_rst[idx_r][:, neigh_idx]
    mean_r = mean_pattern(Xr)
    mean_e = mean_pattern(Xe)
    scores_e = np.array([cosine_sim(x, mean_r) for x in Xe], dtype=float)
    scores_r = np.array([cosine_sim(x, mean_e) for x in Xr], dtype=float)
    return scores_e, scores_r, idx_e, idx_r


def compute_subject_crossphase_maps(
    subj: SubjectData,
    neighbors: List[List[int]],
    n_voxels: int,
    min_voxels: int,
    batches: List[np.ndarray],
    n_jobs: int,
    cond_list: List[str],
) -> Dict[str, np.ndarray]:
    scores: Dict[str, np.ndarray] = {}
    for cond in cond_list:
        scores[cond] = np.full(n_voxels, np.nan)
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_crossphase_batch)(
                subj.X_ext,
                subj.y_ext,
                subj.X_rst,
                subj.y_rst,
                neighbors,
                cond,
                min_voxels,
                batch,
            )
            for batch in batches
        )
        for voxels, vals in results:
            scores[cond][voxels] = vals
    return scores


def compute_subject_crossphase_pair_maps(
    subj: SubjectData,
    neighbors: List[List[int]],
    n_voxels: int,
    min_voxels: int,
    batches: List[np.ndarray],
    n_jobs: int,
    pairs: List[Tuple[str, str]],
) -> Dict[str, np.ndarray]:
    scores: Dict[str, np.ndarray] = {}
    for cond_a, cond_b in pairs:
        pair_name = f"{cond_a}_vs_{cond_b}"
        scores[pair_name] = np.full(n_voxels, np.nan)
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_crossphase_pair_batch)(
                subj.X_ext,
                subj.y_ext,
                subj.X_rst,
                subj.y_rst,
                neighbors,
                cond_a,
                cond_b,
                min_voxels,
                batch,
            )
            for batch in batches
        )
        for voxels, vals in results:
            scores[pair_name][voxels] = vals
    return scores


def _crossphase_batch(
    X_ext: np.ndarray,
    y_ext: np.ndarray,
    X_rst: np.ndarray,
    y_rst: np.ndarray,
    neighbors: List[List[int]],
    cond: str,
    min_voxels: int,
    voxels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    vals = np.full(voxels.size, np.nan, dtype=float)
    for i, v in enumerate(voxels):
        neigh = neighbors[v]
        if len(neigh) < min_voxels:
            continue
        vals[i] = compute_crossphase_similarity(
            X_ext,
            y_ext,
            X_rst,
            y_rst,
            np.asarray(neigh),
            cond,
        )
    return voxels, vals


def _crossphase_pair_batch(
    X_ext: np.ndarray,
    y_ext: np.ndarray,
    X_rst: np.ndarray,
    y_rst: np.ndarray,
    neighbors: List[List[int]],
    cond_a: str,
    cond_b: str,
    min_voxels: int,
    voxels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    vals = np.full(voxels.size, np.nan, dtype=float)
    for i, v in enumerate(voxels):
        neigh = neighbors[v]
        if len(neigh) < min_voxels:
            continue
        vals[i] = compute_crossphase_between_conditions(
            X_ext,
            y_ext,
            X_rst,
            y_rst,
            cond_a,
            cond_b,
        )
    return voxels, vals


def _crossphase_trial_batch(
    X_ext: np.ndarray,
    y_ext: np.ndarray,
    X_rst: np.ndarray,
    y_rst: np.ndarray,
    neighbors: List[List[int]],
    cond: str,
    min_voxels: int,
    voxels: np.ndarray,
    n_ext: int,
    n_rst: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ext_vals = np.full((n_ext, voxels.size), np.nan, dtype=float)
    rst_vals = np.full((n_rst, voxels.size), np.nan, dtype=float)
    for i, v in enumerate(voxels):
        neigh = neighbors[v]
        if len(neigh) < min_voxels:
            continue
        scores_e, scores_r, _, _ = compute_crossphase_trial_scores(
            X_ext,
            y_ext,
            X_rst,
            y_rst,
            np.asarray(neigh),
            cond,
        )
        if scores_e.size == n_ext:
            ext_vals[:, i] = scores_e
        if scores_r.size == n_rst:
            rst_vals[:, i] = scores_r
    return voxels, ext_vals, rst_vals


def subset_voxels_by_chunk(
    valid_voxels: np.ndarray,
    chunk_idx: int | None,
    chunk_count: int | None,
) -> np.ndarray:
    if chunk_idx is None and chunk_count is None:
        return valid_voxels
    if chunk_idx is None or chunk_count is None:
        raise ValueError("Both chunk_idx and chunk_count must be provided.")
    if chunk_count <= 0:
        raise ValueError("chunk_count must be >= 1")
    if chunk_idx < 0 or chunk_idx >= chunk_count:
        raise ValueError("chunk_idx must be in [0, chunk_count).")
    edges = np.linspace(0, valid_voxels.size, chunk_count + 1, dtype=int)
    return valid_voxels[edges[chunk_idx]:edges[chunk_idx + 1]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-phase searchlight RSA (whole brain)")
    parser.add_argument("--project_root", default=os.environ.get("PROJECT_ROOT", "/Users/xiaoqianxiao/projects/NARSAD"))
    parser.add_argument("--phase2_npz", default=None)
    parser.add_argument("--phase3_npz", default=None)
    parser.add_argument("--meta_csv", default=None)
    parser.add_argument("--glasser_atlas", default="/Users/xiaoqianxiao/tool/parcellation/Glasser/HCP-MMP1_2mm.nii")
    parser.add_argument("--tian_atlas", default="/Users/xiaoqianxiao/tool/parcellation/Tian/3T/Subcortex-Only/Tian_Subcortex_S4_3T_2009cAsym.nii.gz")
    parser.add_argument("--reference_lss", default=None)
    parser.add_argument("--task_ref", default="phase2")
    parser.add_argument("--radius", type=float, default=6.0)
    parser.add_argument("--min_voxels", type=int, default=10)
    parser.add_argument("--n_perm", type=int, default=5000)
    parser.add_argument("--one_tailed", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=get_default_n_jobs(), help="Parallel workers")
    parser.add_argument("--batch_size", type=int, default=256, help="Voxels per batch")
    parser.add_argument("--chunk_idx", type=int, default=None, help="Voxel chunk index (0-based)")
    parser.add_argument("--chunk_count", type=int, default=None, help="Total voxel chunks")
    parser.add_argument("--no_tfce", action="store_true", help="Disable TFCE (use voxelwise FDR)")
    parser.add_argument("--post_merge_tfce", action="store_true", help="Save subject maps and skip TFCE until post-merge")
    parser.add_argument("--save_trial_npz", action="store_true", help="Save trial-level searchlight scores (NPZ)")
    parser.add_argument("--cross_half_only", action="store_true", help="(Deprecated) Alias for --cross_half_stage")
    parser.add_argument("--cross_half_stage", action="store_true", help="Compute and save cross-condition + half-split subject maps (chunkable)")
    parser.add_argument("--cross_half_tfce", action="store_true", help="Run TFCE on merged cross-condition + half-split subject maps")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()
    if args.cross_half_only:
        args.cross_half_stage = True
    if args.cross_half_tfce and args.chunk_idx is not None:
        raise RuntimeError("--cross_half_tfce cannot be used with chunking.")
    if args.cross_half_tfce and args.no_tfce:
        raise RuntimeError("--cross_half_tfce requires TFCE (do not use --no_tfce).")

    project_root = args.project_root
    if args.phase2_npz is None:
        args.phase2_npz = os.path.join(
            project_root,
            "MRI/derivatives/fMRI_analysis/LSS",
            "firstLevel",
            "all_subjects/group_level",
            "phase2_X_ext_y_ext_voxels_schaefer_tian.npz",
        )
    if args.phase3_npz is None:
        args.phase3_npz = os.path.join(
            project_root,
            "MRI/derivatives/fMRI_analysis/LSS",
            "firstLevel",
            "all_subjects/group_level",
            "phase3_X_reinst_y_reinst_voxels_schaefer_tian.npz",
        )
    if args.meta_csv is None:
        args.meta_csv = os.path.join(project_root, "MRI/source_data/behav/drug_order.csv")
    if args.out_dir is None:
        args.out_dir = os.path.join(
            project_root,
            "MRI/derivatives/fMRI_analysis/LSS",
            "results",
            "searchlight_crossphase_wholebrain",
        )
    os.makedirs(args.out_dir, exist_ok=True)

    print("[Load] Phase2:", args.phase2_npz)
    p2 = np.load(args.phase2_npz, allow_pickle=True)
    X_ext = p2["X_ext"]
    y_ext = p2["y_ext"]
    sub_ext = p2["subjects"]
    parcel_names = p2["parcel_names"] if "parcel_names" in p2.files else None
    parcel_indices = p2["parcel_indices"] if "parcel_indices" in p2.files else None
    parcel_atlas = p2["parcel_atlas"] if "parcel_atlas" in p2.files else None

    print("[Load] Phase3:", args.phase3_npz)
    p3 = np.load(args.phase3_npz, allow_pickle=True)
    X_rst = p3["X_reinst"]
    y_rst = p3["y_reinst"]
    sub_rst = p3["subjects"]

    mask_cs_ext = np.isin(y_ext, CS_LABELS)
    X_ext = X_ext[mask_cs_ext]
    y_ext = y_ext[mask_cs_ext]
    sub_ext = sub_ext[mask_cs_ext]

    mask_cs_rst = np.isin(y_rst, CS_LABELS)
    X_rst = X_rst[mask_cs_rst]
    y_rst = y_rst[mask_cs_rst]
    sub_rst = sub_rst[mask_cs_rst]

    print("[Meta]", args.meta_csv)
    meta = pd.read_csv(args.meta_csv)
    sub_to_meta = meta.set_index("subject_id")[['Group', 'Drug']].to_dict("index")

    if args.reference_lss is None:
        args.reference_lss = find_reference_lss(project_root, args.task_ref)

    mask, mask_img = build_master_mask_from_reference(
        args.glasser_atlas,
        args.tian_atlas,
        args.reference_lss,
    )

    ijk = np.column_stack(np.where(mask))
    coords = nib.affines.apply_affine(mask_img.affine, ijk)
    n_vox = int(mask.sum())

    if X_ext.shape[1] != n_vox or X_rst.shape[1] != n_vox:
        raise ValueError("Mask voxel count does not match NPZ columns.")

    tree = cKDTree(coords)
    neighbors = tree.query_ball_point(coords, r=args.radius)
    neighbors = [n for n in neighbors]
    valid_voxels = np.array([i for i, n in enumerate(neighbors) if len(n) >= args.min_voxels], dtype=int)
    valid_voxels = subset_voxels_by_chunk(valid_voxels, args.chunk_idx, args.chunk_count)
    batches = build_batches(valid_voxels, args.batch_size)
    chunk_suffix = f"_chunk{args.chunk_idx:03d}" if args.chunk_idx is not None else ""
    cond_list = CS_LABELS

    # intersect subjects across phases
    sub_common = np.intersect1d(np.unique(sub_ext), np.unique(sub_rst))

    subj_data: Dict[str, SubjectData] = {}
    for s in sub_common:
        s_str = str(s).strip()
        meta_info = sub_to_meta.get(s_str) or sub_to_meta.get(f"sub-{s_str}")
        if meta_info is None:
            continue
        m_ext = sub_ext == s
        m_rst = sub_rst == s
        subj_data[s_str] = SubjectData(
            X_ext=X_ext[m_ext],
            y_ext=y_ext[m_ext],
            X_rst=X_rst[m_rst],
            y_rst=y_rst[m_rst],
            group=meta_info['Group'],
            drug=meta_info['Drug'],
        )

    print(f"[Info] Subjects used: {len(subj_data)}")
    use_tfce = not args.no_tfce

    # ------------------------------------------------------------------
    # Permutation testing helpers (use_tfce, mask_img, args in scope)
    # ------------------------------------------------------------------
    def permute_group_diff(values_a, values_b, n_perm, rng):
        obs = np.nanmean(values_a, axis=0) - np.nanmean(values_b, axis=0)
        if use_tfce:
            tested = np.array([1.0] * values_a.shape[0] + [-1.0] * values_b.shape[0])[:, None]
            values = np.concatenate([values_a, values_b], axis=0)
            p, valid_mask = tfce_pvals(values, tested, mask_img, n_perm, True, args.seed, args.n_jobs, model_intercept=True)
            return obs, p, valid_mask
        valid = np.isfinite(obs)
        pooled = np.concatenate([values_a, values_b], axis=0)
        n_a = values_a.shape[0]
        count = np.zeros(obs.shape[0], dtype=int)
        for _ in range(n_perm):
            idx = rng.permutation(pooled.shape[0])
            a_idx = idx[:n_a]
            b_idx = idx[n_a:]
            perm = np.nanmean(pooled[a_idx], axis=0) - np.nanmean(pooled[b_idx], axis=0)
            perm_valid = valid & np.isfinite(perm)
            count[perm_valid] += (np.abs(perm[perm_valid]) >= np.abs(obs[perm_valid])).astype(int)
        p = np.full(obs.shape[0], np.nan, dtype=float)
        p[valid] = (count[valid] + 1) / (n_perm + 1)
        return obs, p

    def permute_sign_flip(values, n_perm, rng, two_tailed=True):
        obs = np.nanmean(values, axis=0)
        if use_tfce:
            tested = np.ones((values.shape[0], 1), dtype=float)
            p, valid_mask = tfce_pvals(values, tested, mask_img, n_perm, True, args.seed, args.n_jobs, model_intercept=False)
            return obs, p, valid_mask
        valid = np.isfinite(obs)
        count = np.zeros(obs.shape[0], dtype=int)
        for _ in range(n_perm):
            signs = rng.choice([-1, 1], size=values.shape[0])[:, None]
            perm = np.nanmean(values * signs, axis=0)
            perm_valid = valid & np.isfinite(perm)
            if two_tailed:
                count[perm_valid] += (np.abs(perm[perm_valid]) >= np.abs(obs[perm_valid])).astype(int)
            else:
                count[perm_valid] += (perm[perm_valid] >= obs[perm_valid]).astype(int)
        p = np.full(obs.shape[0], np.nan, dtype=float)
        p[valid] = (count[valid] + 1) / (n_perm + 1)
        return obs, p

    def fdr_q(pvals):
        if use_tfce:
            return pvals
        q = np.full_like(pvals, np.nan, dtype=float)
        mask = np.isfinite(pvals)
        if mask.any():
            from statsmodels.stats.multitest import multipletests
            _, qv, _, _ = multipletests(pvals[mask], alpha=0.05, method="fdr_bh")
            q[mask] = qv
        return q

    # subject-level cross-phase maps
    subj_maps: Dict[str, Dict[str, np.ndarray]] = {}
    pair_maps: Dict[str, Dict[str, np.ndarray]] = {}
    post_merge_stage = args.post_merge_tfce and args.chunk_idx is None and os.path.exists(os.path.join(args.out_dir, "subj_meta.csv"))
    if post_merge_stage:
        subj_maps, subj_data = load_subject_maps_from_disk(args.out_dir, mask)
        print("[Loaded] Subject maps for post-merge TFCE.")
        if args.save_trial_npz:
            print("[Skip] Trial-level NPZ requires raw trial data; not available in post-merge TFCE stage.")
    else:
        for s_id, s_data in subj_data.items():
            subj_maps[s_id] = compute_subject_crossphase_maps(
                s_data,
                neighbors,
                n_vox,
                args.min_voxels,
                batches,
                args.n_jobs,
                CS_LABELS,
            )
            if args.cross_half_stage:
                pair_maps[s_id] = compute_subject_crossphase_pair_maps(
                    s_data,
                    neighbors,
                    n_vox,
                    args.min_voxels,
                    batches,
                    args.n_jobs,
                    PAIR_LIST,
                )

        if args.save_trial_npz:
            trial_dir = os.path.join(args.out_dir, "trial_npz")
            if args.chunk_idx is not None:
                trial_dir = os.path.join(trial_dir, "chunks")
            for s_id, s_data in subj_data.items():
                trial_scores = compute_subject_crossphase_trial_scores(
                    s_data,
                    neighbors,
                    np.asarray(valid_voxels),
                    args.min_voxels,
                    batches,
                    args.n_jobs,
                    CS_LABELS,
                )
                save_subject_crossphase_trial_npz(
                    trial_dir,
                    s_id,
                    trial_scores,
                    CS_LABELS,
                    subj_maps[s_id],
                    np.asarray(valid_voxels),
                    args.chunk_idx,
                )
            print("[Saved] Trial-level NPZ maps.")

        if args.post_merge_tfce:
            subj_dir = os.path.join(args.out_dir, "subj_maps")
            os.makedirs(subj_dir, exist_ok=True)
            meta_rows = []
            for s_id, s_data in subj_data.items():
                for cond in CS_LABELS:
                    vals = subj_maps[s_id][cond]
                    save_map(
                        vals,
                        mask,
                        mask_img,
                        os.path.join(subj_dir, f"subjmap_{cond}_{s_id}{chunk_suffix}.nii.gz"),
                    )
                meta_rows.append({
                    "Subject": s_id,
                    "Group": s_data.group,
                    "Drug": s_data.drug,
                })
            pd.DataFrame(meta_rows).to_csv(os.path.join(args.out_dir, "subj_meta.csv"), index=False)
            print("[Saved] Subject maps for post-merge TFCE.")
            return

    # compute first/second half subject maps
    subj_data_half: Dict[str, Dict[str, SubjectData]] = {"H1": {}, "H2": {}}
    subj_maps_half: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {"H1": {}, "H2": {}}
    pair_maps_half: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {"H1": {}, "H2": {}}
    if post_merge_stage:
        print("[Skip] Half-split effects require raw trial data.")
    else:
        if args.cross_half_stage:
            for s_id, s_data in subj_data.items():
                for half in ["H1", "H2"]:
                    half_data = subset_subject_by_half(s_data, CS_LABELS, half)
                    subj_data_half[half][s_id] = half_data
                    subj_maps_half[half][s_id] = compute_subject_crossphase_maps(
                        half_data,
                        neighbors,
                        n_vox,
                        args.min_voxels,
                        batches,
                        args.n_jobs,
                        CS_LABELS,
                    )
                    pair_maps_half[half][s_id] = compute_subject_crossphase_pair_maps(
                        half_data,
                        neighbors,
                        n_vox,
                        args.min_voxels,
                        batches,
                        args.n_jobs,
                        PAIR_LIST,
                    )

    def run_crossphase_group_level(
        subj_maps_cur: Dict[str, Dict[str, np.ndarray]],
        pair_maps_cur: Dict[str, Dict[str, np.ndarray]],
        subj_data_cur: Dict[str, SubjectData],
        perm_dir: str,
        suffix: str,
        half_label: str,
    ) -> None:
        # within-condition crossphase
        for cond in CS_LABELS:
            for group in ["SAD", "HC"]:
                for drug in ["Placebo", "Oxytocin"]:
                    subs = [s for s, d in subj_data_cur.items() if d.group == group and d.drug == drug]
                    if not subs:
                        continue
                    vals = np.stack([subj_maps_cur[s][cond] for s in subs], axis=0)
                    if use_tfce:
                        obs, p, valid_mask = permute_sign_flip(vals, args.n_perm, rng, two_tailed=not args.one_tailed)
                    else:
                        obs, p = permute_sign_flip(vals, args.n_perm, rng, two_tailed=not args.one_tailed)
                        valid_mask = None
                    q = fdr_q(p)
                    base = os.path.join(perm_dir, f"crossphase_{cond}_{group}_{drug}{suffix}")
                    save_map(obs, mask, mask_img, base + "_mean.nii.gz")
                    save_map(p, mask, mask_img, base + "_p.nii.gz")
                    save_map(q, mask, mask_img, base + "_q.nii.gz")
                    if use_tfce and valid_mask is not None:
                        save_map(valid_mask.astype(float), mask, mask_img, base + "_validmask.nii.gz")

            # placebo group diff
            sad_subs = [s for s, d in subj_data_cur.items() if d.group == "SAD" and d.drug == "Placebo"]
            hc_subs = [s for s, d in subj_data_cur.items() if d.group == "HC" and d.drug == "Placebo"]
            if len(sad_subs) >= 2 and len(hc_subs) >= 2:
                vals_sad = np.stack([subj_maps_cur[s][cond] for s in sad_subs], axis=0)
                vals_hc = np.stack([subj_maps_cur[s][cond] for s in hc_subs], axis=0)
                if use_tfce:
                    obs, p, valid_mask = permute_group_diff(vals_sad, vals_hc, args.n_perm, rng)
                else:
                    obs, p = permute_group_diff(vals_sad, vals_hc, args.n_perm, rng)
                    valid_mask = None
                q = fdr_q(p)
                base = os.path.join(perm_dir, f"crossphase_{cond}_SAD-HC_PLC{suffix}")
                save_map(obs, mask, mask_img, base + "_diff.nii.gz")
                save_map(p, mask, mask_img, base + "_p.nii.gz")
                save_map(q, mask, mask_img, base + "_q.nii.gz")
                if use_tfce and valid_mask is not None:
                    save_map(valid_mask.astype(float), mask, mask_img, base + "_validmask.nii.gz")

            # OXT-PLC modulation within group
            for group in ["SAD", "HC"]:
                oxt_subs = [s for s, d in subj_data_cur.items() if d.group == group and d.drug == "Oxytocin"]
                plc_subs = [s for s, d in subj_data_cur.items() if d.group == group and d.drug == "Placebo"]
                if len(oxt_subs) >= 2 and len(plc_subs) >= 2:
                    vals_oxt = np.stack([subj_maps_cur[s][cond] for s in oxt_subs], axis=0)
                    vals_plc = np.stack([subj_maps_cur[s][cond] for s in plc_subs], axis=0)
                    if use_tfce:
                        obs, p, valid_mask = permute_group_diff(vals_oxt, vals_plc, args.n_perm, rng)
                    else:
                        obs, p = permute_group_diff(vals_oxt, vals_plc, args.n_perm, rng)
                        valid_mask = None
                    q = fdr_q(p)
                    base = os.path.join(perm_dir, f"crossphase_{cond}_{group}_OXT-PLC{suffix}")
                    save_map(obs, mask, mask_img, base + "_diff.nii.gz")
                    save_map(p, mask, mask_img, base + "_p.nii.gz")
                    save_map(q, mask, mask_img, base + "_q.nii.gz")
                    if use_tfce and valid_mask is not None:
                        save_map(valid_mask.astype(float), mask, mask_img, base + "_validmask.nii.gz")

            # OXT-PLC modulation group difference (interaction)
            sad_oxt = [s for s, d in subj_data_cur.items() if d.group == "SAD" and d.drug == "Oxytocin"]
            sad_plc = [s for s, d in subj_data_cur.items() if d.group == "SAD" and d.drug == "Placebo"]
            hc_oxt = [s for s, d in subj_data_cur.items() if d.group == "HC" and d.drug == "Oxytocin"]
            hc_plc = [s for s, d in subj_data_cur.items() if d.group == "HC" and d.drug == "Placebo"]
            if len(sad_oxt) >= 2 and len(sad_plc) >= 2 and len(hc_oxt) >= 2 and len(hc_plc) >= 2:
                vals_sad_oxt = np.stack([subj_maps_cur[s][cond] for s in sad_oxt], axis=0)
                vals_sad_plc = np.stack([subj_maps_cur[s][cond] for s in sad_plc], axis=0)
                vals_hc_oxt = np.stack([subj_maps_cur[s][cond] for s in hc_oxt], axis=0)
                vals_hc_plc = np.stack([subj_maps_cur[s][cond] for s in hc_plc], axis=0)
                obs = (np.nanmean(vals_sad_oxt, axis=0) - np.nanmean(vals_sad_plc, axis=0)) - (
                    np.nanmean(vals_hc_oxt, axis=0) - np.nanmean(vals_hc_plc, axis=0)
                )
                if use_tfce:
                    all_subs = sad_oxt + sad_plc + hc_oxt + hc_plc
                    group_vec = np.array([1.0] * (len(sad_oxt) + len(sad_plc)) + [-1.0] * (len(hc_oxt) + len(hc_plc)))
                    drug_vec = np.array([1.0] * len(sad_oxt) + [-1.0] * len(sad_plc) + [1.0] * len(hc_oxt) + [-1.0] * len(hc_plc))
                    tested = (group_vec * drug_vec)[:, None]
                    values = np.stack([subj_maps_cur[s][cond] for s in all_subs], axis=0)
                    p, valid_mask = tfce_pvals(values, tested, mask_img, args.n_perm, True, args.seed, args.n_jobs, model_intercept=True)
                else:
                    valid = np.isfinite(obs)
                    count = np.zeros(n_vox, dtype=int)
                    oxt_all = sad_oxt + hc_oxt
                    plc_all = sad_plc + hc_plc
                    labels_oxt = np.array(["SAD"] * len(sad_oxt) + ["HC"] * len(hc_oxt))
                    labels_plc = np.array(["SAD"] * len(sad_plc) + ["HC"] * len(hc_plc))
                    for _ in range(args.n_perm):
                        perm_oxt = rng.permutation(labels_oxt)
                        perm_plc = rng.permutation(labels_plc)
                        sad_oxt_idx = perm_oxt == "SAD"
                        hc_oxt_idx = perm_oxt == "HC"
                        sad_plc_idx = perm_plc == "SAD"
                        hc_plc_idx = perm_plc == "HC"
                        if sad_oxt_idx.sum() < 2 or hc_oxt_idx.sum() < 2 or sad_plc_idx.sum() < 2 or hc_plc_idx.sum() < 2:
                            continue
                        perm_sad_oxt = np.stack([subj_maps_cur[s][cond] for s, m in zip(oxt_all, sad_oxt_idx) if m], axis=0)
                        perm_hc_oxt = np.stack([subj_maps_cur[s][cond] for s, m in zip(oxt_all, hc_oxt_idx) if m], axis=0)
                        perm_sad_plc = np.stack([subj_maps_cur[s][cond] for s, m in zip(plc_all, sad_plc_idx) if m], axis=0)
                        perm_hc_plc = np.stack([subj_maps_cur[s][cond] for s, m in zip(plc_all, hc_plc_idx) if m], axis=0)
                        perm_diff = (np.nanmean(perm_sad_oxt, axis=0) - np.nanmean(perm_sad_plc, axis=0)) - (
                            np.nanmean(perm_hc_oxt, axis=0) - np.nanmean(perm_hc_plc, axis=0)
                        )
                        perm_valid = valid & np.isfinite(perm_diff)
                        count[perm_valid] += (np.abs(perm_diff[perm_valid]) >= np.abs(obs[perm_valid])).astype(int)
                    p = np.full(n_vox, np.nan, dtype=float)
                    p[valid] = (count[valid] + 1) / (args.n_perm + 1)
                    valid_mask = None
                q = fdr_q(p)
                base = os.path.join(perm_dir, f"crossphase_{cond}_SAD-HC_OXT-PLC{suffix}")
                save_map(obs, mask, mask_img, base + "_diff.nii.gz")
                save_map(p, mask, mask_img, base + "_p.nii.gz")
                save_map(q, mask, mask_img, base + "_q.nii.gz")
                if use_tfce and valid_mask is not None:
                    save_map(valid_mask.astype(float), mask, mask_img, base + "_validmask.nii.gz")

        # between-condition crossphase similarity
        for pair in PAIR_LIST:
            pair_name = f"{pair[0]}_vs_{pair[1]}"
            for group in ["SAD", "HC"]:
                for drug in ["Placebo", "Oxytocin"]:
                    subs = [s for s, d in subj_data_cur.items() if d.group == group and d.drug == drug]
                    if not subs:
                        continue
                    vals = np.stack([pair_maps_cur[s][pair_name] for s in subs], axis=0)
                    if use_tfce:
                        obs, p, valid_mask = permute_sign_flip(vals, args.n_perm, rng, two_tailed=not args.one_tailed)
                    else:
                        obs, p = permute_sign_flip(vals, args.n_perm, rng, two_tailed=not args.one_tailed)
                        valid_mask = None
                    q = fdr_q(p)
                    base = os.path.join(perm_dir, f"cross_{pair_name}_{group}_{drug}{suffix}")
                    save_map(obs, mask, mask_img, base + "_mean.nii.gz")
                    save_map(p, mask, mask_img, base + "_p.nii.gz")
                    save_map(q, mask, mask_img, base + "_q.nii.gz")
                    if use_tfce and valid_mask is not None:
                        save_map(valid_mask.astype(float), mask, mask_img, base + "_validmask.nii.gz")

            sad_subs = [s for s, d in subj_data_cur.items() if d.group == "SAD" and d.drug == "Placebo"]
            hc_subs = [s for s, d in subj_data_cur.items() if d.group == "HC" and d.drug == "Placebo"]
            if len(sad_subs) >= 2 and len(hc_subs) >= 2:
                vals_sad = np.stack([pair_maps_cur[s][pair_name] for s in sad_subs], axis=0)
                vals_hc = np.stack([pair_maps_cur[s][pair_name] for s in hc_subs], axis=0)
                if use_tfce:
                    obs, p, valid_mask = permute_group_diff(vals_sad, vals_hc, args.n_perm, rng)
                else:
                    obs, p = permute_group_diff(vals_sad, vals_hc, args.n_perm, rng)
                    valid_mask = None
                q = fdr_q(p)
                base = os.path.join(perm_dir, f"cross_{pair_name}_SAD-HC_PLC{suffix}")
                save_map(obs, mask, mask_img, base + "_diff.nii.gz")
                save_map(p, mask, mask_img, base + "_p.nii.gz")
                save_map(q, mask, mask_img, base + "_q.nii.gz")
                if use_tfce and valid_mask is not None:
                    save_map(valid_mask.astype(float), mask, mask_img, base + "_validmask.nii.gz")

            for group in ["SAD", "HC"]:
                oxt_subs = [s for s, d in subj_data_cur.items() if d.group == group and d.drug == "Oxytocin"]
                plc_subs = [s for s, d in subj_data_cur.items() if d.group == group and d.drug == "Placebo"]
                if len(oxt_subs) >= 2 and len(plc_subs) >= 2:
                    vals_oxt = np.stack([pair_maps_cur[s][pair_name] for s in oxt_subs], axis=0)
                    vals_plc = np.stack([pair_maps_cur[s][pair_name] for s in plc_subs], axis=0)
                    if use_tfce:
                        obs, p, valid_mask = permute_group_diff(vals_oxt, vals_plc, args.n_perm, rng)
                    else:
                        obs, p = permute_group_diff(vals_oxt, vals_plc, args.n_perm, rng)
                        valid_mask = None
                    q = fdr_q(p)
                    base = os.path.join(perm_dir, f"cross_{pair_name}_{group}_OXT-PLC{suffix}")
                    save_map(obs, mask, mask_img, base + "_diff.nii.gz")
                    save_map(p, mask, mask_img, base + "_p.nii.gz")
                    save_map(q, mask, mask_img, base + "_q.nii.gz")
                    if use_tfce and valid_mask is not None:
                        save_map(valid_mask.astype(float), mask, mask_img, base + "_validmask.nii.gz")

            sad_oxt = [s for s, d in subj_data_cur.items() if d.group == "SAD" and d.drug == "Oxytocin"]
            sad_plc = [s for s, d in subj_data_cur.items() if d.group == "SAD" and d.drug == "Placebo"]
            hc_oxt = [s for s, d in subj_data_cur.items() if d.group == "HC" and d.drug == "Oxytocin"]
            hc_plc = [s for s, d in subj_data_cur.items() if d.group == "HC" and d.drug == "Placebo"]
            if len(sad_oxt) >= 2 and len(sad_plc) >= 2 and len(hc_oxt) >= 2 and len(hc_plc) >= 2:
                vals_sad_oxt = np.stack([pair_maps_cur[s][pair_name] for s in sad_oxt], axis=0)
                vals_sad_plc = np.stack([pair_maps_cur[s][pair_name] for s in sad_plc], axis=0)
                vals_hc_oxt = np.stack([pair_maps_cur[s][pair_name] for s in hc_oxt], axis=0)
                vals_hc_plc = np.stack([pair_maps_cur[s][pair_name] for s in hc_plc], axis=0)
                obs = (np.nanmean(vals_sad_oxt, axis=0) - np.nanmean(vals_sad_plc, axis=0)) - (
                    np.nanmean(vals_hc_oxt, axis=0) - np.nanmean(vals_hc_plc, axis=0)
                )
                if use_tfce:
                    all_subs = sad_oxt + sad_plc + hc_oxt + hc_plc
                    group_vec = np.array([1.0] * (len(sad_oxt) + len(sad_plc)) + [-1.0] * (len(hc_oxt) + len(hc_plc)))
                    drug_vec = np.array([1.0] * len(sad_oxt) + [-1.0] * len(sad_plc) + [1.0] * len(hc_oxt) + [-1.0] * len(hc_plc))
                    tested = (group_vec * drug_vec)[:, None]
                    values = np.stack([pair_maps_cur[s][pair_name] for s in all_subs], axis=0)
                    p, valid_mask = tfce_pvals(values, tested, mask_img, args.n_perm, True, args.seed, args.n_jobs, model_intercept=True)
                else:
                    valid = np.isfinite(obs)
                    count = np.zeros(n_vox, dtype=int)
                    oxt_all = sad_oxt + hc_oxt
                    plc_all = sad_plc + hc_plc
                    labels_oxt = np.array(["SAD"] * len(sad_oxt) + ["HC"] * len(hc_oxt))
                    labels_plc = np.array(["SAD"] * len(sad_plc) + ["HC"] * len(hc_plc))
                    for _ in range(args.n_perm):
                        perm_oxt = rng.permutation(labels_oxt)
                        perm_plc = rng.permutation(labels_plc)
                        sad_oxt_idx = perm_oxt == "SAD"
                        hc_oxt_idx = perm_oxt == "HC"
                        sad_plc_idx = perm_plc == "SAD"
                        hc_plc_idx = perm_plc == "HC"
                        if sad_oxt_idx.sum() < 2 or hc_oxt_idx.sum() < 2 or sad_plc_idx.sum() < 2 or hc_plc_idx.sum() < 2:
                            continue
                        perm_sad_oxt = np.stack([pair_maps_cur[s][pair_name] for s, m in zip(oxt_all, sad_oxt_idx) if m], axis=0)
                        perm_hc_oxt = np.stack([pair_maps_cur[s][pair_name] for s, m in zip(oxt_all, hc_oxt_idx) if m], axis=0)
                        perm_sad_plc = np.stack([pair_maps_cur[s][pair_name] for s, m in zip(plc_all, sad_plc_idx) if m], axis=0)
                        perm_hc_plc = np.stack([pair_maps_cur[s][pair_name] for s, m in zip(plc_all, hc_plc_idx) if m], axis=0)
                        perm_diff = (np.nanmean(perm_sad_oxt, axis=0) - np.nanmean(perm_sad_plc, axis=0)) - (
                            np.nanmean(perm_hc_oxt, axis=0) - np.nanmean(perm_hc_plc, axis=0)
                        )
                        perm_valid = valid & np.isfinite(perm_diff)
                        count[perm_valid] += (np.abs(perm_diff[perm_valid]) >= np.abs(obs[perm_valid])).astype(int)
                    p = np.full(n_vox, np.nan, dtype=float)
                    p[valid] = (count[valid] + 1) / (args.n_perm + 1)
                    valid_mask = None
                q = fdr_q(p)
                base = os.path.join(perm_dir, f"cross_{pair_name}_SAD-HC_OXT-PLC{suffix}")
                save_map(obs, mask, mask_img, base + "_diff.nii.gz")
                save_map(p, mask, mask_img, base + "_p.nii.gz")
                save_map(q, mask, mask_img, base + "_q.nii.gz")
                if use_tfce and valid_mask is not None:
                    save_map(valid_mask.astype(float), mask, mask_img, base + "_validmask.nii.gz")

    if args.cross_half_stage:
        if post_merge_stage:
            raise RuntimeError("cross_half_stage requires raw trial data (not post-merge TFCE).")
        cross_dir = os.path.join(args.out_dir, "crosshalf_subj_maps")
        os.makedirs(cross_dir, exist_ok=True)
        meta_rows = []
        for s_id, s_data in subj_data.items():
            for cond in cond_list:
                values = subj_maps[s_id][cond]
                out_name = f"subjmap_{cond}_{s_id}{chunk_suffix}.nii.gz"
                save_map(values, mask, mask_img, os.path.join(cross_dir, out_name))
            for pair_name, values in pair_maps[s_id].items():
                out_name = f"subjmap_cross_{pair_name}_{s_id}{chunk_suffix}.nii.gz"
                save_map(values, mask, mask_img, os.path.join(cross_dir, out_name))
            if subj_maps_half["H1"] and subj_maps_half["H2"]:
                for cond in cond_list:
                    save_map(
                        subj_maps_half["H1"][s_id][cond],
                        mask,
                        mask_img,
                        os.path.join(cross_dir, f"subjmap_{cond}_{s_id}_H1{chunk_suffix}.nii.gz"),
                    )
                    save_map(
                        subj_maps_half["H2"][s_id][cond],
                        mask,
                        mask_img,
                        os.path.join(cross_dir, f"subjmap_{cond}_{s_id}_H2{chunk_suffix}.nii.gz"),
                    )
                for pair_name, values in pair_maps_half["H1"][s_id].items():
                    save_map(
                        values,
                        mask,
                        mask_img,
                        os.path.join(cross_dir, f"subjmap_cross_{pair_name}_{s_id}_H1{chunk_suffix}.nii.gz"),
                    )
                for pair_name, values in pair_maps_half["H2"][s_id].items():
                    save_map(
                        values,
                        mask,
                        mask_img,
                        os.path.join(cross_dir, f"subjmap_cross_{pair_name}_{s_id}_H2{chunk_suffix}.nii.gz"),
                    )
            meta_rows.append({
                "Subject": s_id,
                "Group": s_data.group,
                "Drug": s_data.drug,
            })
        pd.DataFrame(meta_rows).to_csv(os.path.join(cross_dir, "subj_meta.csv"), index=False)
        print("[Saved] Cross-half subject maps.")
        return

    if args.cross_half_tfce:
        cross_dir = os.path.join(args.out_dir, "crosshalf_subj_maps")
        subj_scores, pair_scores, subj_scores_half, pair_scores_half, subj_data, has_half = load_crosshalf_maps_from_disk(
            cross_dir, cond_list, PAIR_LIST, mask
        )
        perm_dir = os.path.join(args.out_dir, "crosshalf_permutation")
        os.makedirs(perm_dir, exist_ok=True)
        run_crossphase_group_level(subj_scores, pair_scores, subj_data, perm_dir, "", "FULL")
        if has_half and subj_scores_half["H1"] and subj_scores_half["H2"]:
            run_crossphase_group_level(subj_scores_half["H1"], pair_scores_half["H1"], subj_data, perm_dir, "_H1", "H1")
            run_crossphase_group_level(subj_scores_half["H2"], pair_scores_half["H2"], subj_data, perm_dir, "_H2", "H2")
        print("[Done] Cross-half TFCE.")
        return

    # group-level mean maps
    print("[Step] Saving group-level mean maps...")
    rows = []
    for cond in CS_LABELS:
        for group in ["SAD", "HC"]:
            for drug in ["Placebo", "Oxytocin"]:
                subs = [s for s, d in subj_data.items() if d.group == group and d.drug == drug]
                if not subs:
                    continue
                stack = np.stack([subj_maps[s][cond] for s in subs], axis=0)
                mean_map = np.nanmean(stack, axis=0)
                base = os.path.join(args.out_dir, f"crossphase_{cond}_{group}_{drug}{chunk_suffix}")
                save_map(mean_map, mask, mask_img, base + "_mean.nii.gz")

                for s in subs:
                    rows.append({
                        "Subject": s,
                        "Group": group,
                        "Drug": drug,
                        "Condition": cond,
                        "MeanSim": float(np.nanmean(subj_maps[s][cond])),
                    })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(os.path.join(args.out_dir, f"crossphase_summary{chunk_suffix}.csv"), index=False)

    # permutation tests
    rng = np.random.default_rng(args.seed)
    perm_dir = os.path.join(args.out_dir, f"permutation{chunk_suffix}")
    os.makedirs(perm_dir, exist_ok=True)

    for cond in CS_LABELS:
        for group in ["SAD", "HC"]:
            # placebo within-group
            subs = [s for s, d in subj_data.items() if d.group == group and d.drug == "Placebo"]
            if len(subs) >= 2:
                vals = np.stack([subj_maps[s][cond] for s in subs], axis=0)
                if use_tfce:
                    obs, p, valid_mask = permute_sign_flip(vals, args.n_perm, rng, two_tailed=not args.one_tailed)
                else:
                    obs, p = permute_sign_flip(vals, args.n_perm, rng, two_tailed=not args.one_tailed)
                    valid_mask = None
                q = fdr_q(p)
                base = os.path.join(perm_dir, f"crossphase_{cond}_{group}_PLC")
                save_map(obs, mask, mask_img, base + "_mean.nii.gz")
                save_map(p, mask, mask_img, base + "_p.nii.gz")
                save_map(q, mask, mask_img, base + "_q.nii.gz")
                if use_tfce and valid_mask is not None:
                    save_map(valid_mask.astype(float), mask, mask_img, base + "_validmask.nii.gz")

        # placebo group diff
        sad_subs = [s for s, d in subj_data.items() if d.group == "SAD" and d.drug == "Placebo"]
        hc_subs = [s for s, d in subj_data.items() if d.group == "HC" and d.drug == "Placebo"]
        if len(sad_subs) >= 2 and len(hc_subs) >= 2:
            vals_sad = np.stack([subj_maps[s][cond] for s in sad_subs], axis=0)
            vals_hc = np.stack([subj_maps[s][cond] for s in hc_subs], axis=0)
            if use_tfce:
                obs, p, valid_mask = permute_group_diff(vals_sad, vals_hc, args.n_perm, rng)
            else:
                obs, p = permute_group_diff(vals_sad, vals_hc, args.n_perm, rng)
                valid_mask = None
            q = fdr_q(p)
            base = os.path.join(perm_dir, f"crossphase_{cond}_SAD-HC_PLC")
            save_map(obs, mask, mask_img, base + "_diff.nii.gz")
            save_map(p, mask, mask_img, base + "_p.nii.gz")
            save_map(q, mask, mask_img, base + "_q.nii.gz")
            if use_tfce and valid_mask is not None:
                save_map(valid_mask.astype(float), mask, mask_img, base + "_validmask.nii.gz")

        # OXT-PLC modulation
        for group in ["SAD", "HC"]:
            oxt_subs = [s for s, d in subj_data.items() if d.group == group and d.drug == "Oxytocin"]
            plc_subs = [s for s, d in subj_data.items() if d.group == group and d.drug == "Placebo"]
            if len(oxt_subs) >= 2 and len(plc_subs) >= 2:
                vals_oxt = np.stack([subj_maps[s][cond] for s in oxt_subs], axis=0)
                vals_plc = np.stack([subj_maps[s][cond] for s in plc_subs], axis=0)
                if use_tfce:
                    obs, p, valid_mask = permute_group_diff(vals_oxt, vals_plc, args.n_perm, rng)
                else:
                    obs, p = permute_group_diff(vals_oxt, vals_plc, args.n_perm, rng)
                    valid_mask = None
                q = fdr_q(p)
                base = os.path.join(perm_dir, f"crossphase_{cond}_{group}_OXT-PLC")
                save_map(obs, mask, mask_img, base + "_diff.nii.gz")
                save_map(p, mask, mask_img, base + "_p.nii.gz")
                save_map(q, mask, mask_img, base + "_q.nii.gz")
                if use_tfce and valid_mask is not None:
                    save_map(valid_mask.astype(float), mask, mask_img, base + "_validmask.nii.gz")

        # OXT-PLC modulation group difference (interaction)
        sad_oxt = [s for s, d in subj_data.items() if d.group == "SAD" and d.drug == "Oxytocin"]
        sad_plc = [s for s, d in subj_data.items() if d.group == "SAD" and d.drug == "Placebo"]
        hc_oxt = [s for s, d in subj_data.items() if d.group == "HC" and d.drug == "Oxytocin"]
        hc_plc = [s for s, d in subj_data.items() if d.group == "HC" and d.drug == "Placebo"]
        if len(sad_oxt) >= 2 and len(sad_plc) >= 2 and len(hc_oxt) >= 2 and len(hc_plc) >= 2:
            vals_sad_oxt = np.stack([subj_maps[s][cond] for s in sad_oxt], axis=0)
            vals_sad_plc = np.stack([subj_maps[s][cond] for s in sad_plc], axis=0)
            vals_hc_oxt = np.stack([subj_maps[s][cond] for s in hc_oxt], axis=0)
            vals_hc_plc = np.stack([subj_maps[s][cond] for s in hc_plc], axis=0)
            obs = (np.nanmean(vals_sad_oxt, axis=0) - np.nanmean(vals_sad_plc, axis=0)) - (
                np.nanmean(vals_hc_oxt, axis=0) - np.nanmean(vals_hc_plc, axis=0)
            )
            if use_tfce:
                all_subs = sad_oxt + sad_plc + hc_oxt + hc_plc
                group_vec = np.array([1.0] * (len(sad_oxt) + len(sad_plc)) + [-1.0] * (len(hc_oxt) + len(hc_plc)))
                drug_vec = np.array([1.0] * len(sad_oxt) + [-1.0] * len(sad_plc) + [1.0] * len(hc_oxt) + [-1.0] * len(hc_plc))
                tested = (group_vec * drug_vec)[:, None]
                values = np.stack([subj_maps[s][cond] for s in all_subs], axis=0)
                p, valid_mask = tfce_pvals(values, tested, mask_img, args.n_perm, True, args.seed, args.n_jobs, model_intercept=True)
            else:
                valid = np.isfinite(obs)
                count = np.zeros(n_vox, dtype=int)
                oxt_all = sad_oxt + hc_oxt
                plc_all = sad_plc + hc_plc
                labels_oxt = np.array(["SAD"] * len(sad_oxt) + ["HC"] * len(hc_oxt))
                labels_plc = np.array(["SAD"] * len(sad_plc) + ["HC"] * len(hc_plc))
                for _ in range(args.n_perm):
                    perm_oxt = rng.permutation(labels_oxt)
                    perm_plc = rng.permutation(labels_plc)
                    sad_oxt_idx = perm_oxt == "SAD"
                    hc_oxt_idx = perm_oxt == "HC"
                    sad_plc_idx = perm_plc == "SAD"
                    hc_plc_idx = perm_plc == "HC"
                    if sad_oxt_idx.sum() < 2 or hc_oxt_idx.sum() < 2 or sad_plc_idx.sum() < 2 or hc_plc_idx.sum() < 2:
                        continue
                    perm_sad_oxt = np.stack([subj_maps[s][cond] for s, m in zip(oxt_all, sad_oxt_idx) if m], axis=0)
                    perm_hc_oxt = np.stack([subj_maps[s][cond] for s, m in zip(oxt_all, hc_oxt_idx) if m], axis=0)
                    perm_sad_plc = np.stack([subj_maps[s][cond] for s, m in zip(plc_all, sad_plc_idx) if m], axis=0)
                    perm_hc_plc = np.stack([subj_maps[s][cond] for s, m in zip(plc_all, hc_plc_idx) if m], axis=0)
                    perm_diff = (np.nanmean(perm_sad_oxt, axis=0) - np.nanmean(perm_sad_plc, axis=0)) - (
                        np.nanmean(perm_hc_oxt, axis=0) - np.nanmean(perm_hc_plc, axis=0)
                    )
                    perm_valid = valid & np.isfinite(perm_diff)
                    count[perm_valid] += (np.abs(perm_diff[perm_valid]) >= np.abs(obs[perm_valid])).astype(int)
                p = np.full(n_vox, np.nan, dtype=float)
                p[valid] = (count[valid] + 1) / (args.n_perm + 1)
                valid_mask = None
            q = fdr_q(p)
            base = os.path.join(perm_dir, f"crossphase_{cond}_SAD-HC_OXT-PLC")
            save_map(obs, mask, mask_img, base + "_diff.nii.gz")
            save_map(p, mask, mask_img, base + "_p.nii.gz")
            save_map(q, mask, mask_img, base + "_q.nii.gz")
            if use_tfce and valid_mask is not None:
                save_map(valid_mask.astype(float), mask, mask_img, base + "_validmask.nii.gz")

    # summary tables + merged table
    summary_rows = []
    merged_rows = []
    if os.path.isdir(perm_dir):
        for cond in CS_LABELS:
            for group in ["SAD", "HC"]:
                q_path = os.path.join(perm_dir, f"crossphase_{cond}_{group}_PLC_q.nii.gz")
                if os.path.exists(q_path):
                    q_img = nib.load(q_path).get_fdata()
                    valid_mask_path = q_path.replace("_q.nii.gz", "_validmask.nii.gz")
                    if os.path.exists(valid_mask_path):
                        valid_mask = nib.load(valid_mask_path).get_fdata() > 0
                    else:
                        valid_mask = mask
                    n_sig = int(np.sum((q_img <= 0.05) & np.isfinite(q_img) & valid_mask))
                    summary_rows.append({
                        "Condition": cond,
                        "Contrast": f"{group} PLC (mean>0)",
                        "N_sig_vox": n_sig,
                    })
                    sig_mask = (q_img <= 0.05) & np.isfinite(q_img) & valid_mask
                    if np.any(sig_mask):
                        sig_coords = coords[sig_mask[mask]]
                        q_vals = q_img[mask][sig_mask[mask]]
                        p_path = os.path.join(perm_dir, f"crossphase_{cond}_{group}_PLC_p.nii.gz")
                        p_vals = None
                        if os.path.exists(p_path):
                            p_img = nib.load(p_path).get_fdata()
                            p_vals = p_img[mask][sig_mask[mask]]
                        if parcel_names is not None:
                            names = parcel_names[sig_mask[mask]]
                        else:
                            names = np.array([None] * len(q_vals))
                        if parcel_indices is not None:
                            ids = parcel_indices[sig_mask[mask]]
                        else:
                            ids = np.array([None] * len(q_vals))
                        if parcel_atlas is not None:
                            atl = parcel_atlas[sig_mask[mask]]
                        else:
                            atl = np.array([None] * len(q_vals))
                        for i in range(len(q_vals)):
                            merged_rows.append({
                                "Contrast": f"crossphase_{cond}_{group}_PLC",
                                "Condition": cond,
                                "x": sig_coords[i, 0],
                                "y": sig_coords[i, 1],
                                "z": sig_coords[i, 2],
                                "p": float(p_vals[i]) if p_vals is not None else np.nan,
                                "q": float(q_vals[i]),
                                "Name": names[i],
                                "LabelID": ids[i],
                                "Atlas": atl[i],
                            })

            q_path = os.path.join(perm_dir, f"crossphase_{cond}_SAD-HC_PLC_q.nii.gz")
            if os.path.exists(q_path):
                q_img = nib.load(q_path).get_fdata()
                valid_mask_path = q_path.replace("_q.nii.gz", "_validmask.nii.gz")
                if os.path.exists(valid_mask_path):
                    valid_mask = nib.load(valid_mask_path).get_fdata() > 0
                else:
                    valid_mask = mask
                n_sig = int(np.sum((q_img <= 0.05) & np.isfinite(q_img) & valid_mask))
                summary_rows.append({
                    "Condition": cond,
                    "Contrast": "SAD-HC PLC",
                    "N_sig_vox": n_sig,
                })
                sig_mask = (q_img <= 0.05) & np.isfinite(q_img) & valid_mask
                if np.any(sig_mask):
                    sig_coords = coords[sig_mask[mask]]
                    q_vals = q_img[mask][sig_mask[mask]]
                    p_path = os.path.join(perm_dir, f"crossphase_{cond}_SAD-HC_PLC_p.nii.gz")
                    p_vals = None
                    if os.path.exists(p_path):
                        p_img = nib.load(p_path).get_fdata()
                        p_vals = p_img[mask][sig_mask[mask]]
                    if parcel_names is not None:
                        names = parcel_names[sig_mask[mask]]
                    else:
                        names = np.array([None] * len(q_vals))
                    if parcel_indices is not None:
                        ids = parcel_indices[sig_mask[mask]]
                    else:
                        ids = np.array([None] * len(q_vals))
                    if parcel_atlas is not None:
                        atl = parcel_atlas[sig_mask[mask]]
                    else:
                        atl = np.array([None] * len(q_vals))
                    for i in range(len(q_vals)):
                        merged_rows.append({
                            "Contrast": f"crossphase_{cond}_SAD-HC_PLC",
                            "Condition": cond,
                            "x": sig_coords[i, 0],
                            "y": sig_coords[i, 1],
                            "z": sig_coords[i, 2],
                            "p": float(p_vals[i]) if p_vals is not None else np.nan,
                            "q": float(q_vals[i]),
                            "Name": names[i],
                            "LabelID": ids[i],
                            "Atlas": atl[i],
                        })

            for group in ["SAD", "HC"]:
                q_path = os.path.join(perm_dir, f"crossphase_{cond}_{group}_OXT-PLC_q.nii.gz")
                if os.path.exists(q_path):
                    q_img = nib.load(q_path).get_fdata()
                    valid_mask_path = q_path.replace("_q.nii.gz", "_validmask.nii.gz")
                    if os.path.exists(valid_mask_path):
                        valid_mask = nib.load(valid_mask_path).get_fdata() > 0
                    else:
                        valid_mask = mask
                    n_sig = int(np.sum((q_img <= 0.05) & np.isfinite(q_img) & valid_mask))
                    summary_rows.append({
                        "Condition": cond,
                        "Contrast": f"{group} OXT-PLC",
                        "N_sig_vox": n_sig,
                    })
                    sig_mask = (q_img <= 0.05) & np.isfinite(q_img) & valid_mask
                    if np.any(sig_mask):
                        sig_coords = coords[sig_mask[mask]]
                        q_vals = q_img[mask][sig_mask[mask]]
                        p_path = os.path.join(perm_dir, f"crossphase_{cond}_{group}_OXT-PLC_p.nii.gz")
                        p_vals = None
                        if os.path.exists(p_path):
                            p_img = nib.load(p_path).get_fdata()
                            p_vals = p_img[mask][sig_mask[mask]]
                        if parcel_names is not None:
                            names = parcel_names[sig_mask[mask]]
                        else:
                            names = np.array([None] * len(q_vals))
                        if parcel_indices is not None:
                            ids = parcel_indices[sig_mask[mask]]
                        else:
                            ids = np.array([None] * len(q_vals))
                        if parcel_atlas is not None:
                            atl = parcel_atlas[sig_mask[mask]]
                        else:
                            atl = np.array([None] * len(q_vals))
                        for i in range(len(q_vals)):
                                merged_rows.append({
                                    "Contrast": f"crossphase_{cond}_{group}_OXT-PLC",
                                    "Condition": cond,
                                "x": sig_coords[i, 0],
                                "y": sig_coords[i, 1],
                                "z": sig_coords[i, 2],
                                "p": float(p_vals[i]) if p_vals is not None else np.nan,
                                "q": float(q_vals[i]),
                                "Name": names[i],
                                "LabelID": ids[i],
                                    "Atlas": atl[i],
                                })
            q_path = os.path.join(perm_dir, f"crossphase_{cond}_SAD-HC_OXT-PLC_q.nii.gz")
            if os.path.exists(q_path):
                q_img = nib.load(q_path).get_fdata()
                valid_mask_path = q_path.replace("_q.nii.gz", "_validmask.nii.gz")
                if os.path.exists(valid_mask_path):
                    valid_mask = nib.load(valid_mask_path).get_fdata() > 0
                else:
                    valid_mask = mask
                n_sig = int(np.sum((q_img <= 0.05) & np.isfinite(q_img) & valid_mask))
                summary_rows.append({
                    "Condition": cond,
                    "Contrast": "SAD-HC OXT-PLC",
                    "N_sig_vox": n_sig,
                })
                sig_mask = (q_img <= 0.05) & np.isfinite(q_img) & valid_mask
                if np.any(sig_mask):
                    sig_coords = coords[sig_mask[mask]]
                    q_vals = q_img[mask][sig_mask[mask]]
                    p_path = os.path.join(perm_dir, f"crossphase_{cond}_SAD-HC_OXT-PLC_p.nii.gz")
                    p_vals = None
                    if os.path.exists(p_path):
                        p_img = nib.load(p_path).get_fdata()
                        p_vals = p_img[mask][sig_mask[mask]]
                    if parcel_names is not None:
                        names = parcel_names[sig_mask[mask]]
                    else:
                        names = np.array([None] * len(q_vals))
                    if parcel_indices is not None:
                        ids = parcel_indices[sig_mask[mask]]
                    else:
                        ids = np.array([None] * len(q_vals))
                    if parcel_atlas is not None:
                        atl = parcel_atlas[sig_mask[mask]]
                    else:
                        atl = np.array([None] * len(q_vals))
                    for i in range(len(q_vals)):
                        merged_rows.append({
                            "Contrast": f"crossphase_{cond}_SAD-HC_OXT-PLC",
                            "Condition": cond,
                            "x": sig_coords[i, 0],
                            "y": sig_coords[i, 1],
                            "z": sig_coords[i, 2],
                            "p": float(p_vals[i]) if p_vals is not None else np.nan,
                            "q": float(q_vals[i]),
                            "Name": names[i],
                            "LabelID": ids[i],
                            "Atlas": atl[i],
                        })

    summary_cols = ["Condition", "Contrast", "N_sig_vox"]
    summary_df = pd.DataFrame(summary_rows, columns=summary_cols)
    summary_df.to_csv(
        os.path.join(perm_dir, f"crossphase_summary_contrasts{chunk_suffix}.csv"),
        index=False,
    )
    merged_cols = ["Contrast", "Condition", "x", "y", "z", "p", "q", "Name", "LabelID", "Atlas"]
    merged_df = pd.DataFrame(merged_rows, columns=merged_cols)
    merged_df.to_csv(
        os.path.join(perm_dir, f"crossphase_sig_merged{chunk_suffix}.csv"),
        index=False,
    )

    print("Done.")


if __name__ == "__main__":
    main()
