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

CS_LABELS = ["CS-", "CSS", "CSR"]
MIN_VALID_FRAC = 0.80


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
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    project_root = args.project_root
    if args.phase2_npz is None:
        args.phase2_npz = os.path.join(
            project_root,
            "MRI/derivatives/fMRI_analysis/LSS",
            "firstLevel",
            "all_subjects/wholeBrain_S4/cope_voxels",
            "phase2_X_ext_y_ext_voxels_glasser_tian.npz",
        )
    if args.phase3_npz is None:
        args.phase3_npz = os.path.join(
            project_root,
            "MRI/derivatives/fMRI_analysis/LSS",
            "firstLevel",
            "all_subjects/wholeBrain_S4/cope_voxels",
            "phase3_X_reinst_y_reinst_voxels_glasser_tian.npz",
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
    post_merge_stage = args.post_merge_tfce and args.chunk_idx is None and os.path.exists(os.path.join(args.out_dir, "subj_meta.csv"))
    if post_merge_stage:
        subj_maps, subj_data = load_subject_maps_from_disk(args.out_dir, mask)
        print("[Loaded] Subject maps for post-merge TFCE.")
    else:
        for s_id, s_data in subj_data.items():
            subj_maps[s_id] = {}
            for cond in CS_LABELS:
                sim_map = np.full(n_vox, np.nan)
                results = Parallel(n_jobs=args.n_jobs, prefer="threads")(
                    delayed(_crossphase_batch)(
                        s_data.X_ext,
                        s_data.y_ext,
                        s_data.X_rst,
                        s_data.y_rst,
                        neighbors,
                        cond,
                        args.min_voxels,
                        batch,
                    )
                    for batch in batches
                )
                for voxels, vals in results:
                    sim_map[voxels] = vals
                subj_maps[s_id][cond] = sim_map

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

    # summary tables + merged table
    summary_rows = []
    merged_rows = []
    if os.path.isdir(perm_dir):
        for cond in CS_LABELS:
            for group in ["SAD", "HC"]:
                q_path = os.path.join(perm_dir, f"crossphase_{cond}_{group}_PLC_q.nii.gz")
                if os.path.exists(q_path):
                    q_img = nib.load(q_path).get_fdata()
                    n_sig = int(np.sum((q_img <= 0.05) & np.isfinite(q_img)))
                    summary_rows.append({
                        "Condition": cond,
                        "Contrast": f"{group} PLC (mean>0)",
                        "N_sig_vox": n_sig,
                    })
                    sig_mask = (q_img <= 0.05) & np.isfinite(q_img) & mask
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
                n_sig = int(np.sum((q_img <= 0.05) & np.isfinite(q_img)))
                summary_rows.append({
                    "Condition": cond,
                    "Contrast": "SAD-HC PLC",
                    "N_sig_vox": n_sig,
                })
                sig_mask = (q_img <= 0.05) & np.isfinite(q_img) & mask
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
                    n_sig = int(np.sum((q_img <= 0.05) & np.isfinite(q_img)))
                    summary_rows.append({
                        "Condition": cond,
                        "Contrast": f"{group} OXT-PLC",
                        "N_sig_vox": n_sig,
                    })
                    sig_mask = (q_img <= 0.05) & np.isfinite(q_img) & mask
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

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(args.out_dir, f"crossphase_summary_contrasts{chunk_suffix}.csv"),
            index=False,
        )
    if merged_rows:
        pd.DataFrame(merged_rows).to_csv(
            os.path.join(args.out_dir, f"crossphase_sig_merged{chunk_suffix}.csv"),
            index=False,
        )

    print("Done.")


if __name__ == "__main__":
    main()
