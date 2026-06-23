#!/usr/bin/env python3
"""Export Aim 1 primary decoding results from the primary Stage 6 checkpoint."""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd


def scalar(value: object) -> object:
    """Return numeric scalars where possible and NaN for absent values."""
    if value is None:
        return np.nan
    try:
        return float(value)
    except Exception:
        return value


def read_result(path: Path) -> Dict:
    """Read a checkpoint and return its results_11 dictionary."""
    payload = joblib.load(path)
    result = payload.get("results_11", payload) if isinstance(payload, dict) else payload
    if not isinstance(result, dict):
        raise ValueError(f"Checkpoint does not contain a results_11 dictionary: {path}")
    return result


def first_existing(paths: List[Path]) -> Optional[Path]:
    """Return the first existing path from a candidate list."""
    return next((path for path in paths if path.exists()), None)


def permutation_summary(values: object) -> Dict[str, object]:
    """Summarize a permutation null distribution for compact CSV export."""
    if values is None:
        return {
            "null_mean": np.nan,
            "null_sd": np.nan,
            "null_ci_low": np.nan,
            "null_ci_high": np.nan,
            "n_permutations": 0,
        }
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "null_mean": np.nan,
            "null_sd": np.nan,
            "null_ci_low": np.nan,
            "null_ci_high": np.nan,
            "n_permutations": 0,
        }
    return {
        "null_mean": float(np.mean(arr)),
        "null_sd": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "null_ci_low": float(np.quantile(arr, 0.025)),
        "null_ci_high": float(np.quantile(arr, 0.975)),
        "n_permutations": int(arr.size),
    }


def add_row(
    rows: List[Dict],
    result: Dict,
    group: str,
    test: str,
    accuracy: object,
    p_value: object,
    permutation_values: object,
    feature_space: str,
) -> None:
    """Append one primary decoding row."""
    row = {
        "analysis": "Aim1_Primary_Decoding",
        "feature_space": feature_space,
        "session": "Placebo",
        "Group": group,
        "test": test,
        "accuracy": scalar(accuracy),
        "accuracy_minus_chance": scalar(accuracy) - 0.5 if pd.notna(scalar(accuracy)) else np.nan,
        "chance": 0.5,
        "p_value": scalar(p_value),
        "n_sad_subjects": result.get("n_sad_subjects"),
        "n_hc_subjects": result.get("n_hc_subjects"),
        "n_sad_trials": result.get("n_sad_trials"),
        "n_hc_trials": result.get("n_hc_trials"),
        "best_c_sad": result.get("best_c_sad"),
        "best_c_hc": result.get("best_c_hc"),
        "status": "ok" if pd.notna(scalar(accuracy)) else "missing_accuracy",
    }
    row.update(permutation_summary(permutation_values))
    rows.append(row)


def export(feature_dir: Path, out: Path, feature_space: str) -> pd.DataFrame:
    """Write Aim 1 primary decoding rows."""
    checkpoint = first_existing(
        [
            feature_dir / "checkpoints" / "cell_06.joblib",
            feature_dir / "cell_06.joblib",
            feature_dir / "intermediate" / "cell_06.joblib",
        ]
    )
    if checkpoint is None:
        table = pd.DataFrame(
            [
                {
                    "analysis": "Aim1_Primary_Decoding",
                    "feature_space": feature_space,
                    "session": "Placebo",
                    "status": "missing_checkpoint",
                    "checkpoint": str(feature_dir / "checkpoints" / "cell_06.joblib"),
                }
            ]
        )
    else:
        result = read_result(checkpoint)
        rows: List[Dict] = []
        add_row(
            rows,
            result,
            "SAD",
            "SAD self-decoding CV accuracy",
            result.get("acc_sad_cv"),
            result.get("p_sad"),
            result.get("perm_dist_sad"),
            feature_space,
        )
        add_row(
            rows,
            result,
            "HC",
            "HC self-decoding CV accuracy",
            result.get("acc_hc_cv"),
            result.get("p_hc"),
            result.get("perm_dist_hc"),
            feature_space,
        )
        func_matrix = result.get("func_matrix")
        func_pvals = result.get("p_func_pvals")
        if func_matrix is not None:
            matrix = np.asarray(func_matrix)
            pvals = np.asarray(func_pvals) if func_pvals is not None else np.full_like(matrix, np.nan, dtype=float)
            if matrix.shape == (2, 2):
                add_row(rows, result, "SAD_to_HC", "SAD model tested on HC", matrix[0, 1], pvals[0, 1], None, feature_space)
                add_row(rows, result, "HC_to_SAD", "HC model tested on SAD", matrix[1, 0], pvals[1, 0], None, feature_space)
        if "sim_spatial" in result:
            rows.append(
                {
                    "analysis": "Aim1_Spatial_Specificity",
                    "feature_space": feature_space,
                    "session": "Placebo",
                    "Group": "SAD_HC",
                    "test": "SAD-HC Haufe map cosine similarity",
                    "accuracy": np.nan,
                    "accuracy_minus_chance": np.nan,
                    "chance": np.nan,
                    "cosine_similarity": scalar(result.get("sim_spatial")),
                    "p_value": scalar(result.get("p_sim")),
                    "status": "ok" if pd.notna(scalar(result.get("sim_spatial"))) else "missing_similarity",
                }
            )
        table = pd.DataFrame(rows)
        table["checkpoint"] = str(checkpoint)

    if "p_value" in table.columns:
        valid = pd.to_numeric(table["p_value"], errors="coerce").notna()
        table["q"] = np.nan
        if valid.any():
            from statsmodels.stats.multitest import multipletests

            table.loc[valid, "q"] = multipletests(pd.to_numeric(table.loc[valid, "p_value"], errors="coerce"), method="fdr_bh")[1]
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"Wrote {len(table)} Aim 1 primary rows -> {out}")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=Path("outputs/mvpa_l2/FearNetwork"))
    parser.add_argument("--out", type=Path, default=Path("outputs/mvpa_l2/stats/aim1_decoding_primary.csv"))
    parser.add_argument("--feature-space", default="FearNetwork")
    args = parser.parse_args()
    export(args.feature_dir, args.out, args.feature_space)


if __name__ == "__main__":
    main()
