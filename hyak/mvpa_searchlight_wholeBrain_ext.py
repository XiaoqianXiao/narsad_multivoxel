#!/usr/bin/env python3
"""Voxel-wise whole-brain searchlight for condition-specific within>between similarity.

Implements Analysis 2.7-style contrasts:
1) Placebo within-condition effects (SAD and HC separately)
2) Placebo group differences (SAD vs HC)
3) Oxytocin modulation within group (OXT - PLC)

Notes:
- Uses condition-label permutation within subject for placebo within-condition null.
- Uses group-label permutation for placebo SAD-HC diff.
- Uses drug-label permutation within group for OXT-PLC.
"""

from __future__ import annotations

import argparse

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple
import glob

import numpy as np
import pandas as pd
import nibabel as nib
from numpy.linalg import norm
from scipy.spatial import cKDTree
from statsmodels.stats.multitest import multipletests
from nilearn.maskers import NiftiMasker
from nilearn.mass_univariate import permuted_ols
from joblib import Parallel, delayed
SCHAEFER_ATLAS_PATH = "/gscratch/fang/NARSAD/ROI/schaefer_2018/Schaefer2018_400Parcels_17Networks_order_FSLMNI152_2mm.nii.gz"

CS_LABELS = ["CS-", "CSS", "CSR"]
MIN_VALID_FRAC = 0.80
SCHAEFER_N_ROIS = 400
SCHAEFER_YEO = 17
SCHAEFER_RES_MM = 2


@dataclass
class SubjectData:
    X: np.ndarray
    y: np.ndarray
    group: str
    drug: str


def _filter_trials_by_valid_frac(X: np.ndarray, min_valid_frac: float) -> np.ndarray:
    if X.size == 0:
        return X
    valid_frac = np.mean(np.isfinite(X), axis=1)
    return X[valid_frac >= min_valid_frac]


def _center_normalize(X: np.ndarray) -> np.ndarray:
    means = np.nanmean(X, axis=1, keepdims=True)
    Xc = X - means
    Xc = np.where(np.isfinite(Xc), Xc, 0.0)
    denom = norm(Xc, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return Xc / denom


def mean_pairwise_cosine(X: np.ndarray) -> float:
    X = _filter_trials_by_valid_frac(X, MIN_VALID_FRAC)
    if X.shape[0] < 2:
        return np.nan
    Xn = _center_normalize(X)
    sim = Xn @ Xn.T
    iu = np.triu_indices(sim.shape[0], k=1)
    return float(np.nanmean(sim[iu]))


def mean_between_cosine(Xa: np.ndarray, Xb: np.ndarray) -> float:
    Xa = _filter_trials_by_valid_frac(Xa, MIN_VALID_FRAC)
    Xb = _filter_trials_by_valid_frac(Xb, MIN_VALID_FRAC)
    if Xa.shape[0] == 0 or Xb.shape[0] == 0:
        return np.nan
    Xa_n = _center_normalize(Xa)
    Xb_n = _center_normalize(Xb)
    sim = Xa_n @ Xb_n.T
    return float(np.nanmean(sim))


def trial_within_between_scores(
    X_sub: np.ndarray,
    y_sub: np.ndarray,
    neigh_idx: np.ndarray,
    cond: str,
) -> Tuple[np.ndarray, np.ndarray]:
    idx_c = np.where(y_sub == cond)[0]
    idx_o = np.where(y_sub != cond)[0]
    if idx_c.size == 0 or idx_o.size == 0:
        return np.full(idx_c.size, np.nan), idx_c

    Xc = X_sub[idx_c][:, neigh_idx]
    Xo = X_sub[idx_o][:, neigh_idx]
    valid_c = np.mean(np.isfinite(Xc), axis=1) >= MIN_VALID_FRAC
    valid_o = np.mean(np.isfinite(Xo), axis=1) >= MIN_VALID_FRAC

    scores = np.full(idx_c.size, np.nan, dtype=float)
    if valid_c.sum() < 2 or valid_o.sum() == 0:
        return scores, idx_c

    Xc_v = Xc[valid_c]
    Xo_v = Xo[valid_o]
    Xc_n = _center_normalize(Xc_v)
    Xo_n = _center_normalize(Xo_v)

    sim_cc = Xc_n @ Xc_n.T
    np.fill_diagonal(sim_cc, np.nan)
    within = np.nanmean(sim_cc, axis=1)
    sim_cb = Xc_n @ Xo_n.T
    between = np.nanmean(sim_cb, axis=1)
    scores[valid_c] = within - between
    return scores, idx_c


def score_within_between_condition(X_sub: np.ndarray, y_sub: np.ndarray, neigh_idx: np.ndarray, cond: str) -> float:
    idx_c = np.where(y_sub == cond)[0]
    idx_o = np.where(y_sub != cond)[0]
    Xc = X_sub[idx_c][:, neigh_idx]
    Xo = X_sub[idx_o][:, neigh_idx]
    within = mean_pairwise_cosine(Xc)
    between = mean_between_cosine(Xc, Xo)
    return within - between


def get_default_n_jobs() -> int:
    env_val = os.environ.get("SLURM_CPUS_PER_TASK")
    if env_val and env_val.isdigit():
        return max(1, int(env_val))
    return max(1, os.cpu_count() or 1)


def build_batches(indices: np.ndarray, batch_size: int) -> List[np.ndarray]:
    if batch_size <= 0:
        raise ValueError("batch_size must be >= 1")
    return [indices[i:i + batch_size] for i in range(0, indices.size, batch_size)]


def _score_cond_batch(
    X_sub: np.ndarray,
    y_sub: np.ndarray,
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
        vals[i] = score_within_between_condition(X_sub, y_sub, np.asarray(neigh), cond)
    return voxels, vals


def _score_cond_trial_batch(
    X_sub: np.ndarray,
    y_sub: np.ndarray,
    neighbors: List[List[int]],
    cond: str,
    min_voxels: int,
    voxels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    n_trials = int(np.sum(y_sub == cond))
    vals = np.full((n_trials, voxels.size), np.nan, dtype=float)
    for i, v in enumerate(voxels):
        neigh = neighbors[v]
        if len(neigh) < min_voxels:
            continue
        scores, _ = trial_within_between_scores(X_sub, y_sub, np.asarray(neigh), cond)
        if scores.size == n_trials:
            vals[:, i] = scores
    return voxels, vals


def filter_conditions(cond: str | None) -> List[str]:
    if cond is None:
        return CS_LABELS
    if cond not in CS_LABELS:
        raise ValueError(f"Unknown condition: {cond}. Must be one of {CS_LABELS}.")
    return [cond]


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




def tfce_pvals(values: np.ndarray, tested_vars: np.ndarray, mask_img: nib.Nifti1Image, n_perm: int, two_sided: bool, seed: int, n_jobs: int, model_intercept: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Compute TFCE-corrected p-values using nilearn.permuted_ols."""
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
def load_mask_and_coords(mask_img_path: str) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Image]:
    img = nib.load(mask_img_path)
    data = img.get_fdata()
    mask = data > 0
    ijk = np.column_stack(np.where(mask))
    coords = nib.affines.apply_affine(img.affine, ijk)
    return mask, coords, img


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
    out_path: str | None = None,
) -> Tuple[str, np.ndarray, nib.Nifti1Image]:
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
    if out_path:
        nib.save(out_img, out_path)
    return out_path or "<master_mask>", mask, out_img


def compute_subject_scores(
    subj: SubjectData,
    neighbors: List[List[int]],
    n_voxels: int,
    min_voxels: int,
    batches: List[np.ndarray],
    n_jobs: int,
) -> Dict[str, np.ndarray]:
    scores = {cond: np.full(n_voxels, np.nan) for cond in CS_LABELS}
    for cond in CS_LABELS:
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_score_cond_batch)(subj.X, subj.y, neighbors, cond, min_voxels, batch)
            for batch in batches
        )
        for voxels, vals in results:
            scores[cond][voxels] = vals
    return scores


def compute_subject_trial_scores(
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
        idx_c = np.where(subj.y == cond)[0]
        trial_scores[cond] = {
            "trial_idx": idx_c,
            "trial_labels": subj.y[idx_c],
            "scores": np.full((idx_c.size, voxel_indices.size), np.nan, dtype=float),
        }
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_score_cond_trial_batch)(subj.X, subj.y, neighbors, cond, min_voxels, batch)
            for batch in batches
        )
        for voxels, vals in results:
            pos = np.searchsorted(voxel_indices, voxels)
            trial_scores[cond]["scores"][:, pos] = vals
    return trial_scores


def save_subject_trial_npz(
    trial_dir: str,
    s_id: str,
    trial_scores: Dict[str, Dict[str, np.ndarray]],
    cond_list: List[str],
    subj_scores: Dict[str, np.ndarray],
    voxel_indices: np.ndarray,
    chunk_idx: int | None,
) -> None:
    os.makedirs(trial_dir, exist_ok=True)
    for cond in cond_list:
        if cond not in trial_scores:
            continue
        scores = trial_scores[cond]["scores"].astype(np.float32, copy=False)
        trial_idx = trial_scores[cond]["trial_idx"]
        trial_labels = trial_scores[cond]["trial_labels"]
        mean_map = subj_scores.get(cond)
        suffix = f"_chunk{chunk_idx:03d}" if chunk_idx is not None else ""
        out_path = os.path.join(trial_dir, f"trialmaps_{cond}_{s_id}{suffix}.npz")
        mean_slice = None
        if mean_map is not None:
            mean_slice = mean_map[np.asarray(voxel_indices, dtype=int)]
        np.savez_compressed(
            out_path,
            scores=scores,
            trial_idx=trial_idx,
            trial_labels=trial_labels,
            mean_map=mean_slice.astype(np.float32, copy=False) if mean_slice is not None else None,
            voxels=voxel_indices,
            subject=s_id,
            condition=cond,
        )


def save_map(values: np.ndarray, mask: np.ndarray, ref_img: nib.Nifti1Image, out_path: str) -> None:
    vol = np.zeros(mask.shape, dtype=float)
    vol[mask] = values
    img_out = nib.Nifti1Image(vol, ref_img.affine)
    nib.save(img_out, out_path)


def load_subject_maps_from_disk(
    out_dir: str,
    cond_list: List[str],
    mask: np.ndarray,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, SubjectData]]:
    meta_path = os.path.join(out_dir, "subj_meta.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing subj_meta.csv in {out_dir}")
    meta_df = pd.read_csv(meta_path)
    subj_scores: Dict[str, Dict[str, np.ndarray]] = {}
    subj_data: Dict[str, SubjectData] = {}
    for row in meta_df.itertuples(index=False):
        s_id = str(row.Subject).strip()
        subj_scores[s_id] = {}
        for cond in cond_list:
            map_path = os.path.join(out_dir, f"subjmap_{cond}_{s_id}.nii.gz")
            if not os.path.exists(map_path):
                raise FileNotFoundError(f"Missing subject map: {map_path}")
            data = nib.load(map_path).get_fdata()
            subj_scores[s_id][cond] = data[mask]
        subj_data[s_id] = SubjectData(
            X=np.empty((0, 0)),
            y=np.empty(0),
            group=row.Group,
            drug=row.Drug,
        )
    return subj_scores, subj_data


def save_sig_csv(
    out_path: str,
    coords: np.ndarray,
    effect: np.ndarray,
    qvals: np.ndarray,
    label_cols: Dict[str, str],
    parcel_names: np.ndarray | None = None,
    parcel_indices: np.ndarray | None = None,
    parcel_atlas: np.ndarray | None = None,
) -> None:
    mask_sig = np.isfinite(qvals) & (qvals <= 0.05)
    if np.any(mask_sig):
        df = pd.DataFrame({
            "x": coords[mask_sig, 0],
            "y": coords[mask_sig, 1],
            "z": coords[mask_sig, 2],
            "effect": effect[mask_sig],
            "q": qvals[mask_sig],
        })
        if parcel_names is not None:
            df["Name"] = parcel_names[mask_sig]
        if parcel_indices is not None:
            df["LabelID"] = parcel_indices[mask_sig]
        if parcel_atlas is not None:
            df["Atlas"] = parcel_atlas[mask_sig]
    else:
        cols = ["x", "y", "z", "effect", "q"]
        if parcel_names is not None:
            cols.append("Name")
        if parcel_indices is not None:
            cols.append("LabelID")
        if parcel_atlas is not None:
            cols.append("Atlas")
        df = pd.DataFrame({c: [] for c in cols})
    for k, v in label_cols.items():
        df[k] = v
    df.to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Voxel-wise searchlight for Analysis 2.7 (whole brain)")
    parser.add_argument("--project_root", default=os.environ.get("PROJECT_ROOT", "/Users/xiaoqianxiao/projects/NARSAD"))
    parser.add_argument("--phase2_npz", default=None, help="Path to phase2 voxel NPZ")
    parser.add_argument("--meta_csv", default=None, help="Subject metadata CSV")
    parser.add_argument("--glasser_atlas", default="/Users/xiaoqianxiao/tool/parcellation/Glasser/HCP-MMP1_2mm.nii")
    parser.add_argument("--tian_atlas", default="/Users/xiaoqianxiao/tool/parcellation/Tian/3T/Subcortex-Only/Tian_Subcortex_S4_3T_2009cAsym.nii.gz")
    parser.add_argument("--mask_img", default=None, help="Explicit common mask image (overrides atlas-based master mask)")
    parser.add_argument("--task", default="phase2", help="Task name (default: phase2)")
    parser.add_argument("--reference_lss", default=None, help="Reference LSS NIfTI to define atlas resampling grid")
    parser.add_argument("--radius", type=float, default=6.0, help="Searchlight radius (mm)")
    parser.add_argument("--min_voxels", type=int, default=10, help="Minimum voxels in sphere")
    parser.add_argument("--n_perm", type=int, default=5000, help="Number of permutations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=get_default_n_jobs(), help="Parallel workers")
    parser.add_argument("--batch_size", type=int, default=256, help="Voxels per batch")
    parser.add_argument("--cond", default=None, help="Run a single condition (CS-, CSS, CSR)")
    parser.add_argument("--chunk_idx", type=int, default=None, help="Voxel chunk index (0-based)")
    parser.add_argument("--chunk_count", type=int, default=None, help="Total voxel chunks")
    parser.add_argument("--no_tfce", action="store_true", help="Disable TFCE (use voxelwise FDR)")
    parser.add_argument("--post_merge_tfce", action="store_true", help="Save subject maps and skip TFCE until post-merge")
    parser.add_argument("--save_trial_npz", action="store_true", help="Save trial-level searchlight scores (NPZ)")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    project_root = args.project_root
    if args.phase2_npz is None:
        args.phase2_npz = os.path.join(
            project_root,
            "MRI/derivatives/fMRI_analysis/LSS",
            "firstLevel",
            "all_subjects/group_level",
            "phase2_X_ext_y_ext_voxels_schaefer_tian.npz",
        )
    if args.meta_csv is None:
        args.meta_csv = os.path.join(project_root, "MRI/source_data/behav/drug_order.csv")
    if args.out_dir is None:
        args.out_dir = os.path.join(project_root, "MRI/derivatives/fMRI_analysis/LSS", "results", "searchlight_rsm_wholebrain_ext")
    os.makedirs(args.out_dir, exist_ok=True)

    print("[Load] Phase2 voxel data:", args.phase2_npz)
    npz = np.load(args.phase2_npz, allow_pickle=True)
    X_ext = npz["X_ext"]
    y_ext = npz["y_ext"]
    sub_ext = npz["subjects"]
    parcel_names = npz["parcel_names"] if "parcel_names" in npz.files else None
    parcel_indices = npz["parcel_indices"] if "parcel_indices" in npz.files else None
    parcel_atlas = npz["parcel_atlas"] if "parcel_atlas" in npz.files else None

    # filter CS labels only
    mask_cs = np.isin(y_ext, CS_LABELS)
    X_ext = X_ext[mask_cs]
    y_ext = y_ext[mask_cs]
    sub_ext = sub_ext[mask_cs]

    print("[Meta]", args.meta_csv)
    meta = pd.read_csv(args.meta_csv)
    sub_to_meta = meta.set_index("subject_id")[["Group", "Drug"]].to_dict("index")

    print("[Step 1] Building master mask and searchlight neighborhoods...")

    if args.mask_img:
        mask, coords, mask_img = load_mask_and_coords(args.mask_img)
    else:
        if args.reference_lss is None:
            args.reference_lss = find_reference_lss(project_root, args.task)
        _, mask, mask_img = build_master_mask_from_reference(
            args.glasser_atlas,
            args.tian_atlas,
            args.reference_lss,
        )
        ijk = np.column_stack(np.where(mask))
        coords = nib.affines.apply_affine(mask_img.affine, ijk)

    n_vox = int(mask.sum())
    if X_ext.shape[1] != n_vox:
        raise ValueError(
            f"Mask voxel count ({n_vox}) does not match X_ext columns ({X_ext.shape[1]}). "
            "This script expects the master mask used to create the NPZ."
        )

    print(f"[Info] Trials: {X_ext.shape[0]} | Voxels: {n_vox}")

    tree = cKDTree(coords)
    neighbors = tree.query_ball_point(coords, r=args.radius)
    neighbors = [n for n in neighbors]
    valid_voxels = np.array([i for i, n in enumerate(neighbors) if len(n) >= args.min_voxels], dtype=int)
    valid_voxels = subset_voxels_by_chunk(valid_voxels, args.chunk_idx, args.chunk_count)
    batches = build_batches(valid_voxels, args.batch_size)
    cond_list = filter_conditions(args.cond)
    chunk_suffix = f"_chunk{args.chunk_idx:03d}" if args.chunk_idx is not None else ""

    # build subject data
    print("[Step 2] Building subject datasets...")
    subjects = np.unique(sub_ext)
    subj_data: Dict[str, SubjectData] = {}
    for sub in subjects:
        s_str = str(sub).strip()
        meta_info = sub_to_meta.get(s_str) or sub_to_meta.get(f"sub-{s_str}")
        if meta_info is None:
            continue
        m_sub = sub_ext == sub
        subj_data[s_str] = SubjectData(
            X=X_ext[m_sub],
            y=y_ext[m_sub],
            group=meta_info["Group"],
            drug=meta_info["Drug"],
        )
    print(f"[Info] Subjects used: {len(subj_data)}")

    # compute or load subject-level maps (observed)
    print("[Step 3] Computing subject-level maps...")
    subj_scores: Dict[str, Dict[str, np.ndarray]] = {}
    post_merge_stage = args.post_merge_tfce and args.chunk_idx is None and os.path.exists(os.path.join(args.out_dir, "subj_meta.csv"))
    if post_merge_stage:
        subj_scores, subj_data = load_subject_maps_from_disk(args.out_dir, cond_list, mask)
        print("[Loaded] Subject maps for post-merge TFCE.")
        if args.save_trial_npz:
            print("[Skip] Trial-level NPZ requires raw trial data; not available in post-merge TFCE stage.")
    else:
        for s_id, s_data in subj_data.items():
            subj_scores[s_id] = compute_subject_scores(
                s_data,
                neighbors,
                n_vox,
                args.min_voxels,
                batches,
                args.n_jobs,
            )

        if args.save_trial_npz:
            trial_dir = os.path.join(args.out_dir, "trial_npz")
            if args.chunk_idx is not None:
                trial_dir = os.path.join(trial_dir, "chunks")
            for s_id, s_data in subj_data.items():
                trial_scores = compute_subject_trial_scores(
                    s_data,
                    neighbors,
                    np.asarray(valid_voxels),
                    args.min_voxels,
                    batches,
                    args.n_jobs,
                    cond_list,
                )
                save_subject_trial_npz(
                    trial_dir,
                    s_id,
                    trial_scores,
                    cond_list,
                    subj_scores[s_id],
                    np.asarray(valid_voxels),
                    args.chunk_idx,
                )
            print("[Saved] Trial-level NPZ maps.")

        if args.post_merge_tfce:
            subj_dir = os.path.join(args.out_dir, "subj_maps")
            os.makedirs(subj_dir, exist_ok=True)
            meta_rows = []
            for s_id, s_data in subj_data.items():
                for cond in cond_list:
                    values = subj_scores[s_id][cond]
                    out_name = f"subjmap_{cond}_{s_id}{chunk_suffix}.nii.gz"
                    save_map(values, mask, mask_img, os.path.join(subj_dir, out_name))
                meta_rows.append({
                    "Subject": s_id,
                    "Group": s_data.group,
                    "Drug": s_data.drug,
                })
            pd.DataFrame(meta_rows).to_csv(os.path.join(args.out_dir, "subj_meta.csv"), index=False)
            print("[Saved] Subject maps for post-merge TFCE.")
            return

    rng = np.random.default_rng(args.seed)
    use_tfce = not args.no_tfce

    # helper to get subject ids by filters
    def select_subjects(group: str | None = None, drug: str | None = None) -> List[str]:
        out = []
        for s_id, s_data in subj_data.items():
            if group is not None and s_data.group != group:
                continue
            if drug is not None and s_data.drug != drug:
                continue
            out.append(s_id)
        return out

    # 4.1 Placebo within-condition (permute condition labels within subject)
    print("[Step 4] Placebo within-condition (label permutation) ...")
    within_results = []
    for cond in cond_list:
        for group in ["SAD", "HC"]:
            subs = select_subjects(group=group, drug="Placebo")
            if not subs:
                continue
            obs_mat = np.stack([subj_scores[s][cond] for s in subs], axis=0)
            obs_mean = np.nanmean(obs_mat, axis=0)
            if use_tfce:
                tested = np.ones((len(subs), 1), dtype=float)
                p_perm, valid_mask = tfce_pvals(obs_mat, tested, mask_img, args.n_perm, True, args.seed, args.n_jobs, model_intercept=False)
            else:
                valid = np.isfinite(obs_mean)
                count = np.zeros(n_vox, dtype=int)
                for _ in range(args.n_perm):
                    perm_means = []
                    for s in subs:
                        s_data = subj_data[s]
                        y_perm = rng.permutation(s_data.y)
                        perm_scores = np.full(n_vox, np.nan)
                        results = Parallel(n_jobs=args.n_jobs, prefer="threads")(
                            delayed(_score_cond_batch)(s_data.X, y_perm, neighbors, cond, args.min_voxels, batch)
                            for batch in batches
                        )
                        for voxels, vals in results:
                            perm_scores[voxels] = vals
                        perm_means.append(perm_scores)
                    perm_mean = np.nanmean(np.stack(perm_means, axis=0), axis=0)
                    perm_valid = valid & np.isfinite(perm_mean)
                    count[perm_valid] += (perm_mean[perm_valid] >= obs_mean[perm_valid]).astype(int)
                p_perm = np.full(n_vox, np.nan, dtype=float)
                p_perm[valid] = (count[valid] + 1) / (args.n_perm + 1)

            if use_tfce:
                q_perm = p_perm.copy()
            else:
                q_perm = np.full(n_vox, np.nan)
                mask = np.isfinite(p_perm)
                if np.any(mask):
                    _, qvals, _, _ = multipletests(p_perm[mask], alpha=0.05, method='fdr_bh')
                    q_perm[mask] = qvals

            within_results.append((cond, group, obs_mean, p_perm, q_perm))

    # 4.2 Placebo SAD-HC group differences
    print("[Step 5] Placebo group differences (SAD-HC) ...")
    groupdiff_results = []
    for cond in cond_list:
        sad_subs = select_subjects(group="SAD", drug="Placebo")
        hc_subs = select_subjects(group="HC", drug="Placebo")
        if not sad_subs or not hc_subs:
            continue
        obs_sad = np.stack([subj_scores[s][cond] for s in sad_subs], axis=0)
        obs_hc = np.stack([subj_scores[s][cond] for s in hc_subs], axis=0)
        obs_diff = np.nanmean(obs_sad, axis=0) - np.nanmean(obs_hc, axis=0)
        if use_tfce:
            tested = np.array([1.0] * len(sad_subs) + [-1.0] * len(hc_subs))[:, None]
            values = np.stack([subj_scores[s][cond] for s in sad_subs + hc_subs], axis=0)
            p_perm, valid_mask = tfce_pvals(values, tested, mask_img, args.n_perm, True, args.seed, args.n_jobs, model_intercept=True)
        else:
            valid = np.isfinite(obs_diff)
            all_subs = sad_subs + hc_subs
            labels = np.array(["SAD"] * len(sad_subs) + ["HC"] * len(hc_subs))
            count = np.zeros(n_vox, dtype=int)
            for _ in range(args.n_perm):
                perm_labels = rng.permutation(labels)
                sad_idx = perm_labels == "SAD"
                hc_idx = perm_labels == "HC"
                if sad_idx.sum() < 2 or hc_idx.sum() < 2:
                    continue
                perm_sad = np.stack([subj_scores[s][cond] for s, m in zip(all_subs, sad_idx) if m], axis=0)
                perm_hc = np.stack([subj_scores[s][cond] for s, m in zip(all_subs, hc_idx) if m], axis=0)
                perm_diff = np.nanmean(perm_sad, axis=0) - np.nanmean(perm_hc, axis=0)
                perm_valid = valid & np.isfinite(perm_diff)
                count[perm_valid] += (np.abs(perm_diff[perm_valid]) >= np.abs(obs_diff[perm_valid])).astype(int)
            p_perm = np.full(n_vox, np.nan, dtype=float)
            p_perm[valid] = (count[valid] + 1) / (args.n_perm + 1)

        if use_tfce:
            q_perm = p_perm.copy()
        else:
            q_perm = np.full(n_vox, np.nan)
            mask = np.isfinite(p_perm)
            if np.any(mask):
                _, qvals, _, _ = multipletests(p_perm[mask], alpha=0.05, method='fdr_bh')
                q_perm[mask] = qvals

        groupdiff_results.append((cond, obs_diff, p_perm, q_perm))

    # 4.3 OXT-PLC modulation within group
    print("[Step 6] OXT-PLC modulation within group ...")
    mod_results = []
    for cond in cond_list:
        for group in ["SAD", "HC"]:
            oxt_subs = select_subjects(group=group, drug="Oxytocin")
            plc_subs = select_subjects(group=group, drug="Placebo")
            if not oxt_subs or not plc_subs:
                continue
            obs_oxt = np.stack([subj_scores[s][cond] for s in oxt_subs], axis=0)
            obs_plc = np.stack([subj_scores[s][cond] for s in plc_subs], axis=0)
            obs_diff = np.nanmean(obs_oxt, axis=0) - np.nanmean(obs_plc, axis=0)
            if use_tfce:
                tested = np.array([1.0] * len(oxt_subs) + [-1.0] * len(plc_subs))[:, None]
                values = np.stack([subj_scores[s][cond] for s in oxt_subs + plc_subs], axis=0)
                p_perm, valid_mask = tfce_pvals(values, tested, mask_img, args.n_perm, True, args.seed, args.n_jobs, model_intercept=True)
            else:
                valid = np.isfinite(obs_diff)
                all_subs = oxt_subs + plc_subs
                labels = np.array(["OXT"] * len(oxt_subs) + ["PLC"] * len(plc_subs))
                count = np.zeros(n_vox, dtype=int)
                for _ in range(args.n_perm):
                    perm_labels = rng.permutation(labels)
                    oxt_idx = perm_labels == "OXT"
                    plc_idx = perm_labels == "PLC"
                    if oxt_idx.sum() < 2 or plc_idx.sum() < 2:
                        continue
                    perm_oxt = np.stack([subj_scores[s][cond] for s, m in zip(all_subs, oxt_idx) if m], axis=0)
                    perm_plc = np.stack([subj_scores[s][cond] for s, m in zip(all_subs, plc_idx) if m], axis=0)
                    perm_diff = np.nanmean(perm_oxt, axis=0) - np.nanmean(perm_plc, axis=0)
                    perm_valid = valid & np.isfinite(perm_diff)
                    count[perm_valid] += (np.abs(perm_diff[perm_valid]) >= np.abs(obs_diff[perm_valid])).astype(int)
                p_perm = np.full(n_vox, np.nan, dtype=float)
                p_perm[valid] = (count[valid] + 1) / (args.n_perm + 1)

            if use_tfce:
                q_perm = p_perm.copy()
            else:
                q_perm = np.full(n_vox, np.nan)
                mask = np.isfinite(p_perm)
                if np.any(mask):
                    _, qvals, _, _ = multipletests(p_perm[mask], alpha=0.05, method='fdr_bh')
                    q_perm[mask] = qvals

            mod_results.append((cond, group, obs_diff, p_perm, q_perm))

    # Save outputs
    print("[Step 7] Saving outputs...")
    perm_dir = os.path.join(args.out_dir, f"permutation{chunk_suffix}")
    os.makedirs(perm_dir, exist_ok=True)
    results = {
        "within": [],
        "groupdiff": [],
        "modulation": [],
        "n_perm": args.n_perm,
        "radius": args.radius,
        "min_voxels": args.min_voxels,
    }
    sig_csvs = []
    # Placebo within-condition
    for cond, group, obs_mean, p_perm, q_perm in within_results:
        base = os.path.join(perm_dir, f"within_{cond}_{group}_PLC")
        save_map(obs_mean, mask, mask_img, base + "_mean.nii.gz")
        save_map(q_perm, mask, mask_img, base + "_q.nii.gz")
        if use_tfce:
            save_map(valid_mask.astype(float), mask, mask_img, base + "_validmask.nii.gz")
        results["within"].append({
            "Condition": cond,
            "Group": group,
            "Drug": "Placebo",
            "Mean": obs_mean,
            "p_perm": p_perm,
            "q_perm": q_perm,
        })
        sig_csv = base + "_sig.csv"
        save_sig_csv(sig_csv, coords, obs_mean, q_perm, {
            "contrast": "within_placebo",
            "Condition": cond,
            "Group": group,
        }, parcel_names, parcel_indices, parcel_atlas)
        if os.path.exists(sig_csv):
            sig_csvs.append(sig_csv)

    # Placebo group diff
    for cond, obs_diff, p_perm, q_perm in groupdiff_results:
        base = os.path.join(perm_dir, f"diff_{cond}_SAD-HC_PLC")
        save_map(obs_diff, mask, mask_img, base + "_diff.nii.gz")
        save_map(q_perm, mask, mask_img, base + "_q.nii.gz")
        if use_tfce:
            save_map(valid_mask.astype(float), mask, mask_img, base + "_validmask.nii.gz")
        results["groupdiff"].append({
            "Condition": cond,
            "Diff": "SAD-HC",
            "Drug": "Placebo",
            "Diff_mean": obs_diff,
            "p_perm": p_perm,
            "q_perm": q_perm,
        })
        sig_csv = base + "_sig.csv"
        save_sig_csv(sig_csv, coords, obs_diff, q_perm, {
            "contrast": "groupdiff_placebo",
            "Condition": cond,
            "GroupA": "SAD",
            "GroupB": "HC",
        }, parcel_names, parcel_indices, parcel_atlas)
        if os.path.exists(sig_csv):
            sig_csvs.append(sig_csv)

    # OXT-PLC modulation
    for cond, group, obs_diff, p_perm, q_perm in mod_results:
        base = os.path.join(perm_dir, f"mod_{cond}_{group}_OXT-PLC")
        save_map(obs_diff, mask, mask_img, base + "_diff.nii.gz")
        save_map(q_perm, mask, mask_img, base + "_q.nii.gz")
        if use_tfce:
            save_map(valid_mask.astype(float), mask, mask_img, base + "_validmask.nii.gz")
        results["modulation"].append({
            "Condition": cond,
            "Group": group,
            "Diff": "OXT-PLC",
            "Diff_mean": obs_diff,
            "p_perm": p_perm,
            "q_perm": q_perm,
        })
        sig_csv = base + "_sig.csv"
        save_sig_csv(sig_csv, coords, obs_diff, q_perm, {
            "contrast": "modulation",
            "Condition": cond,
            "Group": group,
        }, parcel_names, parcel_indices, parcel_atlas)
        if os.path.exists(sig_csv):
            sig_csvs.append(sig_csv)

    results_path = os.path.join(
        perm_dir,
        "analysis_27_results_wholebrain_ext.npz",
    )
    np.savez_compressed(results_path, **results)
    print(f"[Saved] Results dict: {results_path}")
    if sig_csvs:
        print(f"[Saved] Significant voxel CSVs: {len(sig_csvs)} files")

    print("Done.")


if __name__ == "__main__":
    main()
