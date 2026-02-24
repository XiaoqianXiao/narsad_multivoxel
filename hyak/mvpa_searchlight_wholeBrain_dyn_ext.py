#!/usr/bin/env python3
"""Voxel-wise whole-brain searchlight for dynamic representational change.

Computes two dynamic metrics per condition pair:
1) Early vs Late similarity change (mean of trials 1-4 vs 5-8)
2) Trial-index slope of similarity over trials (1..8)

Outputs group-level mean maps for each Group x Drug, and CSV summaries.
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

CS_LABELS = ["CS-", "CSS", "CSR"]
PAIR_LIST = [("CS-", "CSS"), ("CS-", "CSR"), ("CSS", "CSR")]
MIN_VALID_FRAC = 0.80


@dataclass
class SubjectData:
    X: np.ndarray
    y: np.ndarray
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


def slope_vs_trial(sims: np.ndarray) -> float:
    if sims.size < 2 or np.all(np.isnan(sims)):
        return np.nan
    t = np.arange(1, sims.size + 1, dtype=float)
    mask = np.isfinite(sims)
    if mask.sum() < 2:
        return np.nan
    t = t[mask]
    y = sims[mask]
    t_mean = t.mean()
    y_mean = y.mean()
    denom = np.sum((t - t_mean) ** 2)
    if denom == 0:
        return np.nan
    return float(np.sum((t - t_mean) * (y - y_mean)) / denom)


def get_default_n_jobs() -> int:
    env_val = os.environ.get("SLURM_CPUS_PER_TASK")
    if env_val and env_val.isdigit():
        return max(1, int(env_val))
    return max(1, os.cpu_count() or 1)


def build_batches(indices: np.ndarray, batch_size: int) -> List[np.ndarray]:
    if batch_size <= 0:
        raise ValueError("batch_size must be >= 1")
    return [indices[i:i + batch_size] for i in range(0, indices.size, batch_size)]




def tfce_pvals(values: np.ndarray, tested_vars: np.ndarray, mask_img: nib.Nifti1Image, n_perm: int, two_sided: bool, seed: int, n_jobs: int, model_intercept: bool) -> np.ndarray:
    valid = np.all(np.isfinite(values), axis=0)
    p_full = np.full(values.shape[1], np.nan, dtype=float)
    if not np.any(valid):
        return p_full
    vals = values[:, valid]
    masker = NiftiMasker(mask_img=mask_img)
    neglog_pvals, _, _ = permuted_ols(
        tested_vars,
        vals,
        model_intercept=model_intercept,
        n_perm=n_perm,
        two_sided_test=two_sided,
        random_state=seed,
        n_jobs=n_jobs,
        tfce=True,
        masker=masker,
    )
    pvals = 10 ** (-neglog_pvals[0])
    p_full[valid] = pvals
    return p_full

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
    img_g = nib.load(glasser_path)
    img_t = nib.load(tian_path)

    from nilearn.image import resample_to_img

    glasser_res = resample_to_img(img_g, ref_img, interpolation="nearest")
    tian_res = resample_to_img(img_t, ref_img, interpolation="nearest")

    data_g = glasser_res.get_fdata()
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
) -> Tuple[Dict[str, Dict[Tuple[str, str], Dict[str, np.ndarray]]], Dict[str, SubjectData]]:
    meta_path = os.path.join(out_dir, "subj_meta.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing subj_meta.csv in {out_dir}")
    meta_df = pd.read_csv(meta_path)
    subj_maps: Dict[str, Dict[Tuple[str, str], Dict[str, np.ndarray]]] = {}
    subj_data: Dict[str, SubjectData] = {}
    for row in meta_df.itertuples(index=False):
        s_id = str(row.Subject).strip()
        subj_maps[s_id] = {}
        for pair in PAIR_LIST:
            pair_name = f"{pair[0]}_vs_{pair[1]}"
            delta_path = os.path.join(out_dir, f"subjmap_{pair_name}_delta_{s_id}.nii.gz")
            slope_path = os.path.join(out_dir, f"subjmap_{pair_name}_slope_{s_id}.nii.gz")
            if not os.path.exists(delta_path) or not os.path.exists(slope_path):
                raise FileNotFoundError(f"Missing subject maps for {s_id} {pair_name}")
            delta_vals = nib.load(delta_path).get_fdata()[mask]
            slope_vals = nib.load(slope_path).get_fdata()[mask]
            subj_maps[s_id][pair] = {
                "delta": delta_vals,
                "slope": slope_vals,
            }
        subj_data[s_id] = SubjectData(
            X=np.empty((0, 0)),
            y=np.empty(0),
            group=row.Group,
            drug=row.Drug,
        )
    return subj_maps, subj_data


def compute_dynamic_metrics(
    X_sub: np.ndarray,
    y_sub: np.ndarray,
    neigh_idx: np.ndarray,
    early_n: int,
    late_n: int,
    pair: Tuple[str, str],
) -> Tuple[float, float]:
    a, b = pair
    idx_a = np.where(y_sub == a)[0]
    idx_b = np.where(y_sub == b)[0]
    if len(idx_a) < (early_n + late_n) or len(idx_b) < (early_n + late_n):
        return np.nan, np.nan

    Xa = X_sub[idx_a][:, neigh_idx]
    Xb = X_sub[idx_b][:, neigh_idx]

    Xa_early = Xa[:early_n]
    Xa_late = Xa[-late_n:]
    Xb_early = Xb[:early_n]
    Xb_late = Xb[-late_n:]

    sim_early = cosine_sim(mean_pattern(Xa_early), mean_pattern(Xb_early))
    sim_late = cosine_sim(mean_pattern(Xa_late), mean_pattern(Xb_late))
    delta_ab = sim_late - sim_early

    centroid_a = mean_pattern(Xa)
    centroid_b = mean_pattern(Xb)

    sims_a = np.array([cosine_sim(x, centroid_b) for x in Xa], dtype=float)
    sims_b = np.array([cosine_sim(x, centroid_a) for x in Xb], dtype=float)

    slope_a = slope_vs_trial(sims_a)
    slope_b = slope_vs_trial(sims_b)

    slope = np.nanmean([slope_a, slope_b])
    delta = np.nanmean([delta_ab])
    return delta, slope


def _dynamic_batch(
    X_sub: np.ndarray,
    y_sub: np.ndarray,
    neighbors: List[List[int]],
    early_n: int,
    late_n: int,
    pair: Tuple[str, str],
    min_voxels: int,
    voxels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta_vals = np.full(voxels.size, np.nan, dtype=float)
    slope_vals = np.full(voxels.size, np.nan, dtype=float)
    for i, v in enumerate(voxels):
        neigh = neighbors[v]
        if len(neigh) < min_voxels:
            continue
        delta, slope = compute_dynamic_metrics(
            X_sub,
            y_sub,
            np.asarray(neigh),
            early_n,
            late_n,
            pair,
        )
        delta_vals[i] = delta
        slope_vals[i] = slope
    return voxels, delta_vals, slope_vals


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
    parser = argparse.ArgumentParser(description="Dynamic searchlight RSA (whole brain)")
    parser.add_argument("--project_root", default=os.environ.get("PROJECT_ROOT", "/Users/xiaoqianxiao/projects/NARSAD"))
    parser.add_argument("--npz", default=None, help="Path to phase NPZ")
    parser.add_argument("--meta_csv", default=None, help="Subject metadata CSV")
    parser.add_argument("--glasser_atlas", default="/Users/xiaoqianxiao/tool/parcellation/Glasser/HCP-MMP1_2mm.nii")
    parser.add_argument("--tian_atlas", default="/Users/xiaoqianxiao/tool/parcellation/Tian/3T/Subcortex-Only/Tian_Subcortex_S4_3T_2009cAsym.nii.gz")
    parser.add_argument("--task", default="phase2", help="Task name")
    parser.add_argument("--reference_lss", default=None, help="Reference LSS NIfTI to define atlas resampling grid")
    parser.add_argument("--radius", type=float, default=6.0, help="Searchlight radius (mm)")
    parser.add_argument("--min_voxels", type=int, default=10, help="Minimum voxels in sphere")
    parser.add_argument("--early_n", type=int, default=4)
    parser.add_argument("--late_n", type=int, default=4)
    parser.add_argument("--n_perm", type=int, default=5000)
    parser.add_argument("--one_tailed", action="store_true", help="Use one-tailed sign-flip for within-group tests")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=get_default_n_jobs(), help="Parallel workers")
    parser.add_argument("--batch_size", type=int, default=256, help="Voxels per batch")
    parser.add_argument("--chunk_idx", type=int, default=None, help="Voxel chunk index (0-based)")
    parser.add_argument("--chunk_count", type=int, default=None, help="Total voxel chunks")
    parser.add_argument("--no_tfce", action="store_true", help="Disable TFCE (use voxelwise FDR)")
    parser.add_argument("--post_merge_tfce", action="store_true", help="Save subject maps and skip TFCE until post-merge")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    project_root = args.project_root
    if args.npz is None:
        args.npz = os.path.join(
            project_root,
            "MRI/derivatives/fMRI_analysis/LSS",
            "firstLevel",
            "all_subjects/wholeBrain_S4/cope_voxels",
            "phase2_X_ext_y_ext_voxels_glasser_tian.npz",
        )
    if args.meta_csv is None:
        args.meta_csv = os.path.join(project_root, "MRI/source_data/behav/drug_order.csv")
    if args.out_dir is None:
        args.out_dir = os.path.join(
            project_root,
            "MRI/derivatives/fMRI_analysis/LSS",
            "results",
            "searchlight_dyn_wholebrain_ext",
        )
    os.makedirs(args.out_dir, exist_ok=True)

    print("[Load] NPZ:", args.npz)
    npz = np.load(args.npz, allow_pickle=True)
    X = npz["X_ext"]
    y = npz["y_ext"]
    sub = npz["subjects"]
    parcel_names = npz["parcel_names"] if "parcel_names" in npz.files else None
    parcel_indices = npz["parcel_indices"] if "parcel_indices" in npz.files else None
    parcel_atlas = npz["parcel_atlas"] if "parcel_atlas" in npz.files else None

    mask_cs = np.isin(y, CS_LABELS)
    X = X[mask_cs]
    y = y[mask_cs]
    sub = sub[mask_cs]

    print("[Meta]", args.meta_csv)
    meta = pd.read_csv(args.meta_csv)
    sub_to_meta = meta.set_index("subject_id")[["Group", "Drug"]].to_dict("index")

    if args.reference_lss is None:
        args.reference_lss = find_reference_lss(project_root, args.task)

    mask, mask_img = build_master_mask_from_reference(
        args.glasser_atlas,
        args.tian_atlas,
        args.reference_lss,
    )

    ijk = np.column_stack(np.where(mask))
    coords = nib.affines.apply_affine(mask_img.affine, ijk)
    n_vox = int(mask.sum())

    if X.shape[1] != n_vox:
        raise ValueError(
            f"Mask voxel count ({n_vox}) does not match X columns ({X.shape[1]})."
        )

    print(f"[Info] Trials: {X.shape[0]} | Voxels: {n_vox}")

    tree = cKDTree(coords)
    neighbors = tree.query_ball_point(coords, r=args.radius)
    neighbors = [n for n in neighbors]
    valid_voxels = np.array([i for i, n in enumerate(neighbors) if len(n) >= args.min_voxels], dtype=int)
    valid_voxels = subset_voxels_by_chunk(valid_voxels, args.chunk_idx, args.chunk_count)
    batches = build_batches(valid_voxels, args.batch_size)
    chunk_suffix = f"_chunk{args.chunk_idx:03d}" if args.chunk_idx is not None else ""

    subjects = np.unique(sub)
    subj_data: Dict[str, SubjectData] = {}
    for s in subjects:
        s_str = str(s).strip()
        meta_info = sub_to_meta.get(s_str) or sub_to_meta.get(f"sub-{s_str}")
        if meta_info is None:
            continue
        m = sub == s
        subj_data[s_str] = SubjectData(
            X=X[m],
            y=y[m],
            group=meta_info["Group"],
            drug=meta_info["Drug"],
        )
    print(f"[Info] Subjects used: {len(subj_data)}")
    use_tfce = not args.no_tfce

    print("[Step] Computing dynamic maps...")
    subj_maps: Dict[str, Dict[Tuple[str, str], Dict[str, np.ndarray]]] = {}
    post_merge_stage = args.post_merge_tfce and args.chunk_idx is None and os.path.exists(os.path.join(args.out_dir, "subj_meta.csv"))
    if post_merge_stage:
        subj_maps, subj_data = load_subject_maps_from_disk(args.out_dir, mask)
        print("[Loaded] Subject maps for post-merge TFCE.")
    else:
        for s_id, s_data in subj_data.items():
            subj_maps[s_id] = {}
            for pair in PAIR_LIST:
                delta_map = np.full(n_vox, np.nan)
                slope_map = np.full(n_vox, np.nan)
                results = Parallel(n_jobs=args.n_jobs, prefer="threads")(
                    delayed(_dynamic_batch)(
                        s_data.X,
                        s_data.y,
                        neighbors,
                        args.early_n,
                        args.late_n,
                        pair,
                        args.min_voxels,
                        batch,
                    )
                    for batch in batches
                )
                for voxels, delta_vals, slope_vals in results:
                    delta_map[voxels] = delta_vals
                    slope_map[voxels] = slope_vals
                subj_maps[s_id][pair] = {
                    "delta": delta_map,
                    "slope": slope_map,
                }

    if args.post_merge_tfce and not post_merge_stage:
        subj_dir = os.path.join(args.out_dir, "subj_maps")
        os.makedirs(subj_dir, exist_ok=True)
        meta_rows = []
        for s_id, s_data in subj_data.items():
            for pair in PAIR_LIST:
                pair_name = f"{pair[0]}_vs_{pair[1]}"
                delta_vals = subj_maps[s_id][pair]["delta"]
                slope_vals = subj_maps[s_id][pair]["slope"]
                save_map(
                    delta_vals,
                    mask,
                    mask_img,
                    os.path.join(subj_dir, f"subjmap_{pair_name}_delta_{s_id}{chunk_suffix}.nii.gz"),
                )
                save_map(
                    slope_vals,
                    mask,
                    mask_img,
                    os.path.join(subj_dir, f"subjmap_{pair_name}_slope_{s_id}{chunk_suffix}.nii.gz"),
                )
            meta_rows.append({
                "Subject": s_id,
                "Group": s_data.group,
                "Drug": s_data.drug,
            })
        pd.DataFrame(meta_rows).to_csv(os.path.join(args.out_dir, "subj_meta.csv"), index=False)
        print("[Saved] Subject maps for post-merge TFCE.")
        return

    
    # ------------------------------------------------------------------
    # Permutation testing + FDR (group-level)
    # ------------------------------------------------------------------
    def permute_group_diff(values_a, values_b, n_perm, rng):
        """Two-tailed permutation of group labels on voxel-wise means."""
        obs = np.nanmean(values_a, axis=0) - np.nanmean(values_b, axis=0)
        if use_tfce:
            tested = np.array([1.0] * values_a.shape[0] + [-1.0] * values_b.shape[0])[:, None]
            values = np.concatenate([values_a, values_b], axis=0)
            p = tfce_pvals(values, tested, mask_img, n_perm, True, args.seed, args.n_jobs, model_intercept=True)
            return obs, p
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

    def permute_sign_flip(values, n_perm, rng, two_tailed=False):
        """Sign-flip test for mean (optionally two-tailed)."""
        obs = np.nanmean(values, axis=0)
        if use_tfce:
            tested = np.ones((values.shape[0], 1), dtype=float)
            p = tfce_pvals(values, tested, mask_img, n_perm, True, args.seed, args.n_jobs, model_intercept=False)
            return obs, p
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

    print("[Step] Saving group-level mean maps + CSV summaries...")
    rows = []
    for pair in PAIR_LIST:
        pair_name = f"{pair[0]}_vs_{pair[1]}"
        for group in ["SAD", "HC"]:
            for drug in ["Placebo", "Oxytocin"]:
                subs = [s for s, d in subj_data.items() if d.group == group and d.drug == drug]
                if not subs:
                    continue
                delta_stack = np.stack([subj_maps[s][pair]["delta"] for s in subs], axis=0)
                slope_stack = np.stack([subj_maps[s][pair]["slope"] for s in subs], axis=0)
                delta_mean = np.nanmean(delta_stack, axis=0)
                slope_mean = np.nanmean(slope_stack, axis=0)

                base = os.path.join(args.out_dir, f"{pair_name}_{group}_{drug}{chunk_suffix}")
                save_map(delta_mean, mask, mask_img, base + "_delta_mean.nii.gz")
                save_map(slope_mean, mask, mask_img, base + "_slope_mean.nii.gz")

                for s in subs:
                    rows.append({
                        "Subject": s,
                        "Group": group,
                        "Drug": drug,
                        "Pair": pair_name,
                        "DeltaMean": float(np.nanmean(subj_maps[s][pair]["delta"])),
                        "SlopeMean": float(np.nanmean(subj_maps[s][pair]["slope"])),
                    })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(os.path.join(args.out_dir, f"dynamic_summary{chunk_suffix}.csv"), index=False)

    
    # ------------------------------------------------------------------
    # Permutation testing at group level (delta and slope)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    perm_dir = os.path.join(args.out_dir, f"permutation{chunk_suffix}")
    os.makedirs(perm_dir, exist_ok=True)

    for pair in PAIR_LIST:
        pair_name = f"{pair[0]}_vs_{pair[1]}"
        for group in ["SAD", "HC"]:
            # Placebo-only within-group tests (sign-flip, mean > 0)
            for metric in ["delta", "slope"]:
                subs = [s for s, d in subj_data.items() if d.group == group and d.drug == "Placebo"]
                if len(subs) < 2:
                    continue
                vals = np.stack([subj_maps[s][pair][metric] for s in subs], axis=0)
                obs, p = permute_sign_flip(vals, args.n_perm, rng, two_tailed=not args.one_tailed)
                q = fdr_q(p)
                base = os.path.join(perm_dir, f"{pair_name}_{group}_PLC_{metric}")
                save_map(obs, mask, mask_img, base + "_mean.nii.gz")
                save_map(p, mask, mask_img, base + "_p.nii.gz")
                save_map(q, mask, mask_img, base + "_q.nii.gz")

        # Placebo group difference SAD vs HC (two-tailed)
        for metric in ["delta", "slope"]:
            sad_subs = [s for s, d in subj_data.items() if d.group == "SAD" and d.drug == "Placebo"]
            hc_subs = [s for s, d in subj_data.items() if d.group == "HC" and d.drug == "Placebo"]
            if len(sad_subs) < 2 or len(hc_subs) < 2:
                continue
            vals_sad = np.stack([subj_maps[s][pair][metric] for s in sad_subs], axis=0)
            vals_hc = np.stack([subj_maps[s][pair][metric] for s in hc_subs], axis=0)
            obs, p = permute_group_diff(vals_sad, vals_hc, args.n_perm, rng)
            q = fdr_q(p)
            base = os.path.join(perm_dir, f"{pair_name}_SAD-HC_PLC_{metric}")
            save_map(obs, mask, mask_img, base + "_diff.nii.gz")
            save_map(p, mask, mask_img, base + "_p.nii.gz")
            save_map(q, mask, mask_img, base + "_q.nii.gz")

        # Oxytocin modulation within group (OXT-PLC, two-tailed)
        for group in ["SAD", "HC"]:
            for metric in ["delta", "slope"]:
                oxt_subs = [s for s, d in subj_data.items() if d.group == group and d.drug == "Oxytocin"]
                plc_subs = [s for s, d in subj_data.items() if d.group == group and d.drug == "Placebo"]
                if len(oxt_subs) < 2 or len(plc_subs) < 2:
                    continue
                vals_oxt = np.stack([subj_maps[s][pair][metric] for s in oxt_subs], axis=0)
                vals_plc = np.stack([subj_maps[s][pair][metric] for s in plc_subs], axis=0)
                obs, p = permute_group_diff(vals_oxt, vals_plc, args.n_perm, rng)
                q = fdr_q(p)
                base = os.path.join(perm_dir, f"{pair_name}_{group}_OXT-PLC_{metric}")
                save_map(obs, mask, mask_img, base + "_diff.nii.gz")
                save_map(p, mask, mask_img, base + "_p.nii.gz")
                save_map(q, mask, mask_img, base + "_q.nii.gz")

    # ------------------------------------------------------------------
    # Summary tables (significant voxel counts per contrast)
    # + merged significant voxel table
    # ------------------------------------------------------------------
    summary_rows = []
    merged_rows = []
    perm_dir = os.path.join(args.out_dir, f"permutation{chunk_suffix}")
    if os.path.isdir(perm_dir):
        # Precompute voxel coordinates
        ijk = np.column_stack(np.where(mask))
        coords = nib.affines.apply_affine(mask_img.affine, ijk)

        for pair in PAIR_LIST:
            pair_name = f"{pair[0]}_vs_{pair[1]}"
            for metric in ["delta", "slope"]:
                # Placebo within-group
                for group in ["SAD", "HC"]:
                    q_path = os.path.join(perm_dir, f"{pair_name}_{group}_PLC_{metric}_q.nii.gz")
                    if os.path.exists(q_path):
                        q_img = nib.load(q_path).get_fdata()
                        n_sig = int(np.sum((q_img <= 0.05) & np.isfinite(q_img)))
                        summary_rows.append({
                            "Pair": pair_name,
                            "Metric": metric,
                            "Contrast": f"{group} PLC (mean>0)",
                            "N_sig_vox": n_sig,
                        })
                        # Save significant voxel coords CSV
                        sig_mask = (q_img <= 0.05) & np.isfinite(q_img) & mask
                        if np.any(sig_mask):
                            sig_coords = coords[sig_mask[mask]]
                            q_vals = q_img[mask][sig_mask[mask]]
                            # p-values from corresponding perm map in memory not saved, so reconstruct from q? not possible
                            # Instead, load p-map if it exists; if not, skip p.
                            p_vals = None
                            p_path = os.path.join(perm_dir, f"{pair_name}_{group}_PLC_{metric}_p.nii.gz")
                            if os.path.exists(p_path):
                                p_img = nib.load(p_path).get_fdata()
                                p_vals = p_img[mask][sig_mask[mask]]
                        # merged rows
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
                                    "Contrast": f"{pair_name}_{group}_PLC_{metric}",
                                    "Pair": pair_name,
                                    "Metric": metric,
                                    "x": sig_coords[i, 0],
                                    "y": sig_coords[i, 1],
                                    "z": sig_coords[i, 2],
                                    "p": float(p_vals[i]) if p_vals is not None else np.nan,
                                    "q": float(q_vals[i]),
                                    "Name": names[i],
                                    "LabelID": ids[i],
                                    "Atlas": atl[i],
                                })
                # Placebo group diff
                q_path = os.path.join(perm_dir, f"{pair_name}_SAD-HC_PLC_{metric}_q.nii.gz")
                if os.path.exists(q_path):
                    q_img = nib.load(q_path).get_fdata()
                    n_sig = int(np.sum((q_img <= 0.05) & np.isfinite(q_img)))
                    summary_rows.append({
                        "Pair": pair_name,
                        "Metric": metric,
                        "Contrast": "SAD-HC PLC",
                        "N_sig_vox": n_sig,
                    })
                    sig_mask = (q_img <= 0.05) & np.isfinite(q_img) & mask
                    if np.any(sig_mask):
                        sig_coords = coords[sig_mask[mask]]
                        q_vals = q_img[mask][sig_mask[mask]]
                        p_vals = None
                        p_path = os.path.join(perm_dir, f"{pair_name}_SAD-HC_PLC_{metric}_p.nii.gz")
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
                                "Contrast": f"{pair_name}_SAD-HC_PLC_{metric}",
                                "Pair": pair_name,
                                "Metric": metric,
                                "x": sig_coords[i, 0],
                                "y": sig_coords[i, 1],
                                "z": sig_coords[i, 2],
                                "p": float(p_vals[i]) if p_vals is not None else np.nan,
                                "q": float(q_vals[i]),
                                "Name": names[i],
                                "LabelID": ids[i],
                                "Atlas": atl[i],
                            })
                # OXT-PLC modulation
                for group in ["SAD", "HC"]:
                    q_path = os.path.join(perm_dir, f"{pair_name}_{group}_OXT-PLC_{metric}_q.nii.gz")
                    if os.path.exists(q_path):
                        q_img = nib.load(q_path).get_fdata()
                        n_sig = int(np.sum((q_img <= 0.05) & np.isfinite(q_img)))
                        summary_rows.append({
                            "Pair": pair_name,
                            "Metric": metric,
                            "Contrast": f"{group} OXT-PLC",
                            "N_sig_vox": n_sig,
                        })
                        sig_mask = (q_img <= 0.05) & np.isfinite(q_img) & mask
                        if np.any(sig_mask):
                            sig_coords = coords[sig_mask[mask]]
                            q_vals = q_img[mask][sig_mask[mask]]
                            p_vals = None
                            p_path = os.path.join(perm_dir, f"{pair_name}_{group}_OXT-PLC_{metric}_p.nii.gz")
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
                                    "Contrast": f"{pair_name}_{group}_OXT-PLC_{metric}",
                                    "Pair": pair_name,
                                    "Metric": metric,
                                    "x": sig_coords[i, 0],
                                    "y": sig_coords[i, 1],
                                    "z": sig_coords[i, 2],
                                    "p": float(p_vals[i]) if p_vals is not None else np.nan,
                                    "q": float(q_vals[i]),
                                    "Name": names[i],
                                    "LabelID": ids[i],
                                    "Atlas": atl[i],
                                })
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(
            os.path.join(args.out_dir, f"dynamic_summary_contrasts{chunk_suffix}.csv"),
            index=False,
        )
    if merged_rows:
        merged_df = pd.DataFrame(merged_rows)
        merged_df.to_csv(
            os.path.join(args.out_dir, f"dynamic_sig_merged{chunk_suffix}.csv"),
            index=False,
        )

    print("Done.")


if __name__ == "__main__":
    main()
