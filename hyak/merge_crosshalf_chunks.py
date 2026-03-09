#!/usr/bin/env python3
"""Merge chunked cross-half subject maps for ext/rst/dyn/crossphase.

Merges any *_chunkNNN_* maps (nii.gz) by summing into the full-volume
output (non-chunked) path. Uses the first chunk as reference header.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np

CHUNK_RE = re.compile(r"_chunk\d{3}")


def merge_maps(folder: Path) -> int:
    chunk_files = sorted(folder.glob("*chunk*.nii.gz"))
    groups: Dict[str, List[Path]] = {}
    for f in chunk_files:
        base = CHUNK_RE.sub("", f.name)
        groups.setdefault(base, []).append(f)

    merged = 0
    for base, files in groups.items():
        ref = nib.load(str(files[0]))
        sum_arr = np.zeros(ref.shape, dtype=np.float64)
        count = np.zeros(ref.shape, dtype=np.uint16)
        for f in files:
            d = nib.load(str(f)).get_fdata()
            finite = np.isfinite(d)
            if not np.any(finite):
                continue
            sum_arr[finite] += d[finite]
            count[finite] += 1
        data = np.full(ref.shape, np.nan, dtype=np.float32)
        valid = count > 0
        if np.any(valid):
            data[valid] = (sum_arr[valid] / count[valid]).astype(np.float32, copy=False)
        out_path = folder / base
        nib.save(nib.Nifti1Image(data, ref.affine), str(out_path))
        merged += 1
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder containing mode dirs (ext/rst/dyn_ext/dyn_rst/crossphase)")
    ap.add_argument("--mode", default="all", help="Mode to merge: ext | rst | dyn_ext | dyn_rst | crossphase | all")
    args = ap.parse_args()
    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Missing root: {root}")

    if args.mode == "all":
        modes = ["ext", "rst", "dyn_ext", "dyn_rst", "crossphase"]
    else:
        modes = [args.mode]

    total = 0
    for mode in modes:
        folder = root / mode / "crosshalf_subj_maps"
        if not folder.exists():
            continue
        n = merge_maps(folder)
        print(f"{mode}: merged {n} files")
        total += n
    print(f"Total merged: {total}")


if __name__ == "__main__":
    main()
