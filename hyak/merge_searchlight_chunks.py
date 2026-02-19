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
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    chunk_files = find_chunk_files(args.in_dir)
    grouped = group_by_base(chunk_files)

    # NIfTI maps
    for base, paths in grouped.items():
        if base.endswith(".nii.gz"):
            merge_nifti_group(paths, os.path.join(args.out_dir, base))

    # CSVs: sig, summary_contrasts, merged sig
    for base, paths in grouped.items():
        if base.endswith("_sig.csv") or base.endswith("sig_merged.csv"):
            merge_sig_csv(paths, os.path.join(args.out_dir, base))
        elif base.endswith("summary_contrasts.csv"):
            merge_summary_contrasts(paths, os.path.join(args.out_dir, base))

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
