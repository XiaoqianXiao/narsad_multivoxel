#!/usr/bin/env python3
"""Export Aim 1 primary decoding results from the primary Stage 6 checkpoint."""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from statsmodels.stats.multitest import multipletests


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


def read_checkpoint(path: Path) -> tuple[Dict, Dict]:
    """Read a checkpoint and return its results_11 dictionary plus full payload."""
    payload = joblib.load(path)
    result = payload.get("results_11", payload) if isinstance(payload, dict) else payload
    if not isinstance(result, dict):
        raise ValueError(f"Checkpoint does not contain a results_11 dictionary: {path}")
    return result, payload if isinstance(payload, dict) else {}


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


def checkpoint_group_data(payload: Dict, group: str) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return placebo extinction data for one group from a Stage 6 checkpoint payload."""
    subsets = payload.get("data_subsets") if isinstance(payload, dict) else None
    if not isinstance(subsets, dict):
        return None
    group_data = subsets.get(f"{group}_Placebo")
    phase_data = group_data.get("ext") if isinstance(group_data, dict) else None
    if not isinstance(phase_data, dict) or any(key not in phase_data for key in ("X", "y", "sub")):
        return None
    return np.asarray(phase_data["X"]), np.asarray(phase_data["y"]), np.asarray(phase_data["sub"]).astype(str)


def forced_choice_predict(scores: object, classes: object) -> np.ndarray:
    """Predict labels from binary or multiclass decision scores."""
    scores_arr = np.asarray(scores)
    if scores_arr.ndim == 1:
        scores_arr = np.column_stack((-scores_arr, scores_arr))
    class_arr = np.asarray(classes)
    return class_arr[np.argmax(scores_arr, axis=1)]


def forced_choice_accuracy(model: object, X: np.ndarray, y: np.ndarray) -> float:
    """Return trial-wise forced-choice accuracy for a fitted model."""
    return float(np.mean(np.asarray(y) == forced_choice_predict(model.decision_function(X), model.classes_)))


def reconstruct_functional_drop_pairs(result: Dict, payload: Dict, feature_space: str, label: str) -> pd.DataFrame:
    """Reconstruct matched subject-level self/cross rows from saved models and data."""
    models = {"SAD": result.get("model_sad"), "HC": result.get("model_hc")}
    best_c = {"SAD": result.get("best_c_sad", 1.0), "HC": result.get("best_c_hc", 1.0)}
    if any(model is None for model in models.values()):
        return pd.DataFrame()
    data = {group: checkpoint_group_data(payload, group) for group in ["SAD", "HC"]}
    if any(value is None for value in data.values()):
        return pd.DataFrame()
    rows = []
    for target_group, source_group in [("SAD", "HC"), ("HC", "SAD")]:
        X_target, y_target, sub_target = data[target_group]
        source_model = models[source_group]
        keep = np.isin(y_target, source_model.classes_)
        X_target = X_target[keep]
        y_target = y_target[keep]
        sub_target = sub_target[keep]
        for subject_id in pd.Series(sub_target).dropna().astype(str).unique():
            test_mask = sub_target == subject_id
            train_mask = ~test_mask
            if not np.any(test_mask) or not np.any(train_mask) or len(np.unique(y_target[train_mask])) < 2:
                continue
            self_model = clone(models[target_group])
            if hasattr(self_model, "set_params"):
                self_model.set_params(classification__C=float(best_c[target_group]))
                self_model.set_params(classification__n_jobs=1)
            self_model.fit(X_target[train_mask], y_target[train_mask])
            rows.append({
                "sensitivity_set": label,
                "feature_space": feature_space,
                "cohort": "full_placebo",
                "target_group": target_group,
                "subject_id": subject_id,
                "resample_id": subject_id,
                "within_accuracy": forced_choice_accuracy(self_model, X_target[test_mask], y_target[test_mask]),
                "cross_accuracy": forced_choice_accuracy(source_model, X_target[test_mask], y_target[test_mask]),
                "within_aggregate": scalar(result.get("acc_sad_cv" if target_group == "SAD" else "acc_hc_cv")),
                "cross_aggregate": scalar(np.asarray(result.get("func_matrix"), dtype=float)[1, 0] if target_group == "SAD" else np.asarray(result.get("func_matrix"), dtype=float)[0, 1]),
                "n_trials": int(np.sum(test_mask)),
                "within_metric_source": "leave_one_subject_out_refit",
                "cross_metric_source": "opposite_group_refit_model",
                "distribution_source": "checkpoint_reconstruction",
            })
    return pd.DataFrame(rows)


def functional_drop_pairs(result: Dict, payload: Dict, feature_space: str, label: str) -> pd.DataFrame:
    """Return saved or reconstructed matched rows for self-vs-cross visualization."""
    saved = result.get("functional_drop_pairs")
    if saved is not None:
        frame = pd.DataFrame(saved).copy()
        if not frame.empty:
            frame["sensitivity_set"] = label
            frame["feature_space"] = feature_space
            frame["cohort"] = frame.get("cohort", "full_placebo")
            frame["resample_id"] = frame.get("resample_id", frame.get("subject_id", pd.Series(np.arange(len(frame)), index=frame.index))).astype(str)
            frame["distribution_source"] = "checkpoint_functional_drop_pairs"
            return frame
    return reconstruct_functional_drop_pairs(result, payload, feature_space, label)


def paired_sign_flip_drop_test(pairs: pd.DataFrame, n_perm: int = 10000, seed: int = 20260624) -> tuple[Dict, np.ndarray]:
    """Run a paired sign-flip test on within-minus-cross accuracy."""
    if pairs is None or pairs.empty:
        return {}, np.array([], dtype=float)
    drops = pd.to_numeric(pairs["within_accuracy"], errors="coerce") - pd.to_numeric(pairs["cross_accuracy"], errors="coerce")
    drops = drops.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if drops.size == 0:
        return {}, np.array([], dtype=float)
    observed = float(np.mean(drops))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(int(n_perm), drops.size))
    null = np.mean(signs * drops, axis=1)
    p_value = float((1 + np.sum(null >= observed)) / (len(null) + 1))
    boot_idx = rng.integers(0, drops.size, size=(int(n_perm), drops.size))
    boot_means = np.mean(drops[boot_idx], axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return {
        "n_pairs": int(drops.size),
        "within_group_accuracy": float(pd.to_numeric(pairs["within_accuracy"], errors="coerce").mean()),
        "cross_group_accuracy": float(pd.to_numeric(pairs["cross_accuracy"], errors="coerce").mean()),
        "functional_drop": observed,
        "drop_ci_low": float(ci_low),
        "drop_ci_high": float(ci_high),
        "functional_drop_p": p_value,
        "n_permutations": int(len(null)),
        "test": "paired_sign_flip_mean_within_minus_cross",
    }, null


def functional_drop_test_rows(result: Dict, pairs: pd.DataFrame, feature_space: str, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return self-vs-cross drop-test rows and optional null-distribution rows."""
    saved_tests = result.get("functional_drop_tests")
    tests = []
    null_rows = []
    for group in ["SAD", "HC"]:
        group_pairs = pairs[pairs["target_group"].astype(str).eq(group)].copy() if not pairs.empty else pd.DataFrame()
        if isinstance(saved_tests, dict) and group in saved_tests:
            test = dict(saved_tests[group])
        else:
            test, null = paired_sign_flip_drop_test(group_pairs, seed=20260624 + (0 if group == "SAD" else 1000))
            for i, value in enumerate(null):
                null_rows.append({"sensitivity_set": label, "feature_space": feature_space, "cohort": "full_placebo", "target_group": group, "permutation_id": i, "null_functional_drop": value})
        if test:
            test.update({"sensitivity_set": label, "feature_space": feature_space, "cohort": "full_placebo", "target_group": group})
            tests.append(test)
    return pd.DataFrame(tests), pd.DataFrame(null_rows)


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
    drop_tests_out: Optional[Path] = None,
    drop_nulls_out: Optional[Path] = None,
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
        result, payload = read_checkpoint(checkpoint)
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
        label = sensitivity_label or feature_label(feature_space)
        pairs = functional_drop_pairs(result, payload, feature_space, label)
        if pairs.empty:
            pairs = paired_raincloud_rows(result, feature_space, sensitivity_label)
        write_raincloud_csv(pairs, raincloud_out)
        drop_tests, drop_nulls = functional_drop_test_rows(result, pairs, feature_space, label)
        if not drop_tests.empty:
            valid = pd.to_numeric(drop_tests["functional_drop_p"], errors="coerce").notna()
            drop_tests["functional_drop_q"] = np.nan
            if valid.any():
                drop_tests.loc[valid, "functional_drop_q"] = multipletests(
                    pd.to_numeric(drop_tests.loc[valid, "functional_drop_p"], errors="coerce"),
                    method="fdr_bh",
                )[1]
        if drop_tests_out is not None:
            drop_tests_out.parent.mkdir(parents=True, exist_ok=True)
            drop_tests.to_csv(drop_tests_out, index=False)
            print(f"Wrote {len(drop_tests)} Aim 1 functional-drop test rows -> {drop_tests_out}")
        if drop_nulls_out is not None:
            drop_nulls_out.parent.mkdir(parents=True, exist_ok=True)
            drop_nulls.to_csv(drop_nulls_out, index=False)
            print(f"Wrote {len(drop_nulls)} Aim 1 functional-drop null rows -> {drop_nulls_out}")

    if "p_value" in table.columns:
        valid = pd.to_numeric(table["p_value"], errors="coerce").notna()
        table["q"] = np.nan
        if valid.any():
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
    parser.add_argument("--drop-tests-out", type=Path, default=None)
    parser.add_argument("--drop-nulls-out", type=Path, default=None)
    parser.add_argument("--sensitivity-label", default=None)
    args = parser.parse_args()
    raincloud_out = args.raincloud_out
    if raincloud_out is None:
        raincloud_out = args.out.with_name(args.out.stem + "_raincloud.csv")
    drop_tests_out = args.drop_tests_out or args.out.with_name(args.out.stem + "_functional_drop_tests.csv")
    drop_nulls_out = args.drop_nulls_out or args.out.with_name(args.out.stem + "_functional_drop_nulls.csv")
    export(
        args.feature_dir,
        args.out,
        args.feature_space,
        raincloud_out=raincloud_out,
        drop_tests_out=drop_tests_out,
        drop_nulls_out=drop_nulls_out,
        sensitivity_label=args.sensitivity_label,
    )


if __name__ == "__main__":
    main()
