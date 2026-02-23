#!/usr/bin/env python3
"""Merge chunked searchlight outputs into whole-brain outputs.

Merges:
- NIfTI maps (nanmean across chunks)
- *_sig.csv (concatenate)
- *_summary_contrasts.csv (sum N_sig_vox)
- dynamic_summary.csv / crossphase_summary.csv (weighted average by voxel count)
- dynamic_sig_merged.csv / crossphase_sig_merged.csv (concatenate)
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import nibabel as nib


CHUNK_RE = re.compile(r"_chunk\d{3}")


def strip_chunk(name: str) -> str:
    return CHUNK_RE.sub("", name)


def find_chunk_files(in_dir: str) -> List[str]:
    matches: List[str] = []
    for root, _, files in os.walk(in_dir):
        for f in files:
            if "_chunk" in f:
                matches.append(os.path.join(root, f))
    return matches


def group_by_base(paths: List[str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for p in paths:
        base = strip_chunk(os.path.basename(p))
        grouped.setdefault(base, []).append(p)
    return grouped


def merge_nifti_group(paths: List[str], out_path: str) -> None:
    imgs = [nib.load(p) for p in paths]
    data = np.stack([img.get_fdata() for img in imgs], axis=0)
    merged = np.nanmean(data, axis=0)
    nib.save(nib.Nifti1Image(merged, imgs[0].affine), out_path)


def build_master_mask_from_reference(
    glasser_path: str,
    tian_path: str,
    reference_img_path: str,
) -> Tuple[np.ndarray, nib.Nifti1Image]:
    from nilearn.image import resample_to_img

    ref_img = nib.load(reference_img_path)
    img_g = nib.load(glasser_path)
    img_t = nib.load(tian_path)

    glasser_res = resample_to_img(img_g, ref_img, interpolation="nearest")
    tian_res = resample_to_img(img_t, ref_img, interpolation="nearest")

    data_g = glasser_res.get_fdata()
    data_t = tian_res.get_fdata()
    mask = (data_g > 0) | (data_t > 0)
    out_img = nib.Nifti1Image(mask.astype(np.uint8), ref_img.affine)
    return mask, out_img


def is_vector_image(img: nib.Nifti1Image) -> bool:
    return len(img.shape) == 1


def vector_to_3d(
    vector: np.ndarray,
    mask: np.ndarray,
    ref_img: nib.Nifti1Image,
) -> nib.Nifti1Image:
    if vector.ndim != 1:
        raise ValueError("Expected 1D vector for unmasking.")
    if int(mask.sum()) != vector.shape[0]:
        raise ValueError("Vector length does not match master mask voxels.")
    vol = np.zeros(mask.shape, dtype=float)
    vol[mask] = vector
    return nib.Nifti1Image(vol, ref_img.affine)


def merge_sig_csv(paths: List[str], out_path: str) -> None:
    frames = [pd.read_csv(p) for p in paths]
    pd.concat(frames, ignore_index=True).to_csv(out_path, index=False)


def merge_summary_contrasts(paths: List[str], out_path: str) -> None:
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    if "N_sig_vox" not in df.columns:
        df.to_csv(out_path, index=False)
        return
    group_cols = [c for c in df.columns if c != "N_sig_vox"]
    merged = df.groupby(group_cols, dropna=False, as_index=False)["N_sig_vox"].sum()
    merged.to_csv(out_path, index=False)


def chunk_weights_from_maps(in_dir: str, pattern: str) -> Dict[str, int]:
    weights: Dict[str, int] = {}
    for root, _, files in os.walk(in_dir):
        for f in files:
            if pattern in f and "_chunk" in f and f.endswith(".nii.gz"):
                chunk = CHUNK_RE.search(f)
                if chunk is None:
                    continue
                chunk_id = chunk.group(0)
                if chunk_id in weights:
                    continue
                img = nib.load(os.path.join(root, f))
                data = img.get_fdata()
                weights[chunk_id] = int(np.sum(np.isfinite(data)))
    return weights


def merge_weighted_summary(
    paths: List[str],
    out_path: str,
    weights: Dict[str, int],
    value_cols: List[str],
    group_cols: List[str],
) -> None:
    frames = []
    for p in paths:
        chunk = CHUNK_RE.search(os.path.basename(p))
        if chunk is None:
            continue
        w = weights.get(chunk.group(0))
        if w is None or w == 0:
            continue
        df = pd.read_csv(p)
        df["__weight__"] = w
        frames.append(df)
    if not frames:
        return
    df_all = pd.concat(frames, ignore_index=True)
    agg = {}
    for col in value_cols:
        if col in df_all.columns:
            agg[col] = lambda x, c=col: np.average(x, weights=df_all.loc[x.index, "__weight__"])
    merged = df_all.groupby(group_cols, dropna=False, as_index=False).agg(agg)
    merged.to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge chunked searchlight outputs")
    parser.add_argument("--in_dir", required=True, help="Directory containing chunk outputs")
    parser.add_argument("--out_dir", required=True, help="Directory to write merged outputs")
    parser.add_argument("--reference_lss", default=None, help="Reference LSS NIfTI for mask geometry")
    parser.add_argument("--glasser_atlas", default=None, help="Glasser atlas NIfTI")
    parser.add_argument("--tian_atlas", default=None, help="Tian atlas NIfTI")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    chunk_files = find_chunk_files(args.in_dir)
    grouped = group_by_base(chunk_files)

    # If vector maps exist, we need a master mask to unmask to 3D.
    mask = None
    mask_img = None
    if args.reference_lss and args.glasser_atlas and args.tian_atlas:
        mask, mask_img = build_master_mask_from_reference(
            args.glasser_atlas,
            args.tian_atlas,
            args.reference_lss,
        )

    # NIfTI maps
    for base, paths in grouped.items():
        if not base.endswith(".nii.gz"):
            continue
        imgs = [nib.load(p) for p in paths]
        if any(is_vector_image(img) for img in imgs):
            if mask is None or mask_img is None:
                raise ValueError(
                    "Vector NIfTI detected but reference_lss/glasser_atlas/tian_atlas not provided."
                )
            data = np.stack([img.get_fdata() for img in imgs], axis=0)
            merged_vec = np.nanmean(data, axis=0)
            out_img = vector_to_3d(merged_vec, mask, mask_img)
            nib.save(out_img, os.path.join(args.out_dir, base))
        else:
            merge_nifti_group(paths, os.path.join(args.out_dir, base))

    # CSVs: sig, summary_contrasts, merged sig
    for base, paths in grouped.items():
        if base.endswith("_sig.csv") or base.endswith("sig_merged.csv"):
            merge_sig_csv(paths, os.path.join(args.out_dir, base))
        elif base.endswith("summary_contrasts.csv"):
            merge_summary_contrasts(paths, os.path.join(args.out_dir, base))

    # Copy subject metadata if present
    for root, _, files in os.walk(args.in_dir):
        if "subj_meta.csv" in files:
            src = os.path.join(root, "subj_meta.csv")
            dst = os.path.join(args.out_dir, "subj_meta.csv")
            if not os.path.exists(dst):
                pd.read_csv(src).to_csv(dst, index=False)
            break

    # Weighted summaries
    weights = chunk_weights_from_maps(args.in_dir, "_mean.nii.gz")
    dynamic_paths = [p for p in chunk_files if os.path.basename(p).startswith("dynamic_summary") and p.endswith(".csv")]
    cross_paths = [p for p in chunk_files if os.path.basename(p).startswith("crossphase_summary") and p.endswith(".csv")]

    if dynamic_paths:
        merge_weighted_summary(
            dynamic_paths,
            os.path.join(args.out_dir, "dynamic_summary.csv"),
            weights,
            value_cols=["DeltaMean", "SlopeMean"],
            group_cols=["Subject", "Group", "Drug", "Pair"],
        )
    if cross_paths:
        merge_weighted_summary(
            cross_paths,
            os.path.join(args.out_dir, "crossphase_summary.csv"),
            weights,
            value_cols=["MeanSim"],
            group_cols=["Subject", "Group", "Drug", "Condition"],
        )


if __name__ == "__main__":
    main()
