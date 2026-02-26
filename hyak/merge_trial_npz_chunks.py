#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Tuple

import numpy as np


CHUNK_RE = re.compile(r"^(.*)_chunk\d{3}$")


def group_chunks(files: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        m = CHUNK_RE.match(base)
        if not m:
            continue
        key = m.group(1)
        groups.setdefault(key, []).append(f)
    return groups


def _load_npz(path: str) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as npz:
        return {k: npz[k] for k in npz.files}


def _merge_matrix(chunks: List[Dict[str, np.ndarray]], key: str) -> Tuple[np.ndarray, np.ndarray]:
    vox_all = np.unique(np.concatenate([c["voxels"] for c in chunks]))
    vox_all.sort()
    n_trials = chunks[0][key].shape[0]
    out = np.full((n_trials, vox_all.size), np.nan, dtype=np.float32)
    for c in chunks:
        vox = c["voxels"]
        pos = np.searchsorted(vox_all, vox)
        out[:, pos] = c[key].astype(np.float32, copy=False)
    return out, vox_all


def _merge_vector(chunks: List[Dict[str, np.ndarray]], key: str, vox_all: np.ndarray) -> np.ndarray:
    out = np.full((vox_all.size,), np.nan, dtype=np.float32)
    for c in chunks:
        if key not in c:
            continue
        vox = c["voxels"]
        pos = np.searchsorted(vox_all, vox)
        out[pos] = c[key].astype(np.float32, copy=False)
    return out


def merge_group(key: str, files: List[str], out_dir: str) -> None:
    chunks = [_load_npz(f) for f in files]
    sample = chunks[0]

    if "scores" in sample:
        scores, vox_all = _merge_matrix(chunks, "scores")
        mean_map = _merge_vector(chunks, "mean_map", vox_all)
        out = {
            "scores": scores,
            "trial_idx": sample["trial_idx"],
            "trial_labels": sample["trial_labels"],
            "mean_map": mean_map,
            "voxels": vox_all,
            "subject": sample.get("subject", ""),
            "condition": sample.get("condition", ""),
        }
    elif "ext_scores" in sample:
        ext_scores, vox_all = _merge_matrix(chunks, "ext_scores")
        rst_scores, _ = _merge_matrix(chunks, "rst_scores")
        mean_map = _merge_vector(chunks, "mean_map", vox_all)
        out = {
            "ext_scores": ext_scores,
            "rst_scores": rst_scores,
            "ext_idx": sample["ext_idx"],
            "rst_idx": sample["rst_idx"],
            "ext_labels": sample["ext_labels"],
            "rst_labels": sample["rst_labels"],
            "mean_map": mean_map,
            "voxels": vox_all,
            "subject": sample.get("subject", ""),
            "condition": sample.get("condition", ""),
        }
    elif "sims_a" in sample:
        sims_a, vox_all = _merge_matrix(chunks, "sims_a")
        sims_b, _ = _merge_matrix(chunks, "sims_b")
        delta_map = _merge_vector(chunks, "delta_map", vox_all)
        slope_map = _merge_vector(chunks, "slope_map", vox_all)
        out = {
            "sims_a": sims_a,
            "sims_b": sims_b,
            "trial_idx_a": sample["trial_idx_a"],
            "trial_idx_b": sample["trial_idx_b"],
            "trial_labels_a": sample["trial_labels_a"],
            "trial_labels_b": sample["trial_labels_b"],
            "delta_map": delta_map,
            "slope_map": slope_map,
            "voxels": vox_all,
            "subject": sample.get("subject", ""),
            "pair": sample.get("pair", ""),
        }
    else:
        raise ValueError(f"Unknown NPZ format for {key}")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{key}.npz")
    np.savez_compressed(out_path, **out)
    print(f"[Merged] {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge chunked trial NPZs into full NPZs.")
    parser.add_argument("--chunk_dir", required=True, help="Directory containing chunked NPZs")
    parser.add_argument("--out_dir", required=True, help="Output directory for merged NPZs")
    args = parser.parse_args()

    files = [os.path.join(args.chunk_dir, f) for f in os.listdir(args.chunk_dir) if f.endswith(".npz")]
    groups = group_chunks(files)
    if not groups:
        raise FileNotFoundError(f"No chunked NPZs found in {args.chunk_dir}")

    for key, fset in groups.items():
        merge_group(key, sorted(fset), args.out_dir)


if __name__ == "__main__":
    main()
