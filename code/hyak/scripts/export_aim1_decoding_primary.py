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


def cosine_from_vectors(a: object, b: object) -> float:
    """Return cosine similarity between two finite vectors."""
    a_arr = np.asarray(a, dtype=float).ravel()
    b_arr = np.asarray(b, dtype=float).ravel()
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0 or not np.isfinite(denom):
        return np.nan
    return float(np.dot(a_arr, b_arr) / denom)


def spatial_null_from_weight_checkpoints(feature_dir: Path) -> object:
    """Derive a spatial null from saved SAD/HC Haufe-weight permutation files."""
    checkpoint_dir = feature_dir / "checkpoints"
    sad_path = checkpoint_dir / "perm_results_SAD_fear_network_2way.joblib"
    hc_path = checkpoint_dir / "perm_results_HC_fear_network_2way.joblib"
    if not sad_path.exists() or not hc_path.exists():
        return None
    sad_payload = joblib.load(sad_path)
    hc_payload = joblib.load(hc_path)
    if not isinstance(sad_payload, dict) or not isinstance(hc_payload, dict):
        return None
    sad_null = sad_payload.get("null_weights")
    hc_null = hc_payload.get("null_weights")
    if sad_null is None or hc_null is None:
        return None
    sad_null = np.asarray(sad_null, dtype=float)
    hc_null = np.asarray(hc_null, dtype=float)
    n_draws = min(len(sad_null), len(hc_null))
    if n_draws == 0:
        return None
    sims = np.array([cosine_from_vectors(sad_null[i], hc_null[i]) for i in range(n_draws)])
    sims = sims[np.isfinite(sims)]
    return sims if sims.size else None


def spatial_permutation_values(result: Dict, feature_dir: Path) -> object:
    """Return the saved spatial permutation null distribution, if present."""
    for key in ["spatial_perm_dist", "perm_sim", "perm_dist_sim", "perm_dist_spatial"]:
        values = result.get(key)
        if values is not None:
            return values
    return spatial_null_from_weight_checkpoints(feature_dir)


def spatial_permutation_source(result: Dict, values: object) -> str:
    for key in ["spatial_perm_dist", "perm_sim", "perm_dist_sim", "perm_dist_spatial"]:
        if result.get(key) is not None:
            return key
    if values is not None:
        return "perm_results_null_weights"
    return "unavailable"


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


def feature_label(feature_space: str) -> str:
    """Return the Figure S1 sensitivity-set label for a feature space."""
    labels = {
        "FearNetwork": "FearNetwork primary",
        "MemoryFearNetwork": "MemoryFearNetwork",
        "Schaefer": "Schaefer/Tian parcellation",
        "WholeBrain": "Whole brain",
    }
    return labels.get(str(feature_space), str(feature_space))


def finite_vector(values: object) -> np.ndarray:
    """Return a finite one-dimensional float vector."""
    if values is None:
        return np.array([], dtype=float)
    arr = np.asarray(values, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def paired_raincloud_rows(
    result: Dict,
    feature_space: str,
    sensitivity_label: Optional[str] = None,
) -> pd.DataFrame:
    """Build panel-B raincloud rows from saved 2AFC fold/subject distributions."""
    label = sensitivity_label or feature_label(feature_space)
    func_matrix = result.get("func_matrix")
    if func_matrix is None:
        return pd.DataFrame()
    matrix = np.asarray(func_matrix, dtype=float)
    if matrix.shape != (2, 2):
        return pd.DataFrame()

    specs = [
        ("SAD", result.get("cv_fold_scores_sad"), result.get("accs_hc2sad"), matrix[0, 0], matrix[1, 0]),
        ("HC", result.get("cv_fold_scores_hc"), result.get("accs_sad2hc"), matrix[1, 1], matrix[0, 1]),
    ]
    rows: List[Dict] = []
    for target_group, within_values, cross_values, within_aggregate, cross_aggregate in specs:
        within = finite_vector(within_values)
        cross = finite_vector(cross_values)
        n = int(max(len(within), len(cross), 1))
        for i in range(n):
            rows.append(
                {
                    "sensitivity_set": label,
                    "feature_space": feature_space,
                    "cohort": "full_placebo",
                    "target_group": target_group,
                    "resample_id": i,
                    "within_accuracy": float(within[i % len(within)]) if len(within) else scalar(within_aggregate),
                    "cross_accuracy": float(cross[i % len(cross)]) if len(cross) else scalar(cross_aggregate),
                    "within_aggregate": scalar(within_aggregate),
                    "cross_aggregate": scalar(cross_aggregate),
                    "within_metric_source": "cv_fold_scores" if len(within) else "aggregate_only",
                    "cross_metric_source": "cross_subject_scores" if len(cross) else "aggregate_only",
                }
            )
    return pd.DataFrame(rows)


def write_raincloud_csv(table: pd.DataFrame, out: Optional[Path]) -> None:
    if out is None:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"Wrote {len(table)} Aim 1 raincloud rows -> {out}")


def export(
    feature_dir: Path,
    out: Path,
    feature_space: str,
    raincloud_out: Optional[Path] = None,
    sensitivity_label: Optional[str] = None,
) -> pd.DataFrame:
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
            spatial_row = {
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
            spatial_null = spatial_permutation_values(result, feature_dir)
            spatial_row.update(permutation_summary(spatial_null))
            spatial_row["spatial_null_source"] = spatial_permutation_source(result, spatial_null)
            rows.append(spatial_row)
        table = pd.DataFrame(rows)
        table["checkpoint"] = str(checkpoint)
        write_raincloud_csv(paired_raincloud_rows(result, feature_space, sensitivity_label), raincloud_out)

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
    parser.add_argument("--raincloud-out", type=Path, default=None)
    parser.add_argument("--sensitivity-label", default=None)
    args = parser.parse_args()
    raincloud_out = args.raincloud_out
    if raincloud_out is None:
        raincloud_out = args.out.with_name(args.out.stem + "_raincloud.csv")
    export(args.feature_dir, args.out, args.feature_space, raincloud_out=raincloud_out, sensitivity_label=args.sensitivity_label)


if __name__ == "__main__":
    main()
