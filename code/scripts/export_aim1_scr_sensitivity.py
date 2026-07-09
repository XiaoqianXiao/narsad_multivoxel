#!/usr/bin/env python3
"""Export labeled Analysis 1 SCR-subgroup sensitivity checkpoints to CSV.

The exported table contains two complementary sensitivity readouts:

1. SCR-subgroup trained models from labeled Stage 6 checkpoints.
2. Full placebo-session models from the primary Stage 6 checkpoint applied to
   pooled-drug rows within each SCR-defined subgroup. These full-model subgroup rows are descriptive
   hold-in checks, not subgroup cross-validation tests.
"""

import argparse
import glob
import os

import joblib
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


SCR_FLAGS = [
    "SCR_Physiological_Responder",
    "SCR_Simple_Acquisition_Differential_Learner",
    "SCR_Habituation_Adjusted_Learner",
    "SCR_Late_Phase_Sensitivity_Learner",
]


def scalar(value):
    if value is None:
        return np.nan
    try:
        return float(value)
    except Exception:
        return value


def permutation_summary(values):
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


def spatial_permutation_values(result):
    for key in ["spatial_perm_dist", "perm_sim", "perm_dist_sim", "perm_dist_spatial"]:
        values = result.get(key)
        if values is not None:
            return values
    return None


def as_bool_series(series):
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def add_row(rows, result, label, feature_space, test, estimate, p_value, extra=None):
    session = result.get("session") or ("pooled_drug" if result.get("include_subjects_flag") else "Placebo")
    row = {
        "analysis_label": label,
        "feature_space": feature_space,
        "session": session,
        "test": test,
        "estimate": scalar(estimate),
        "p_value": scalar(p_value),
        "include_subjects_flag": result.get("include_subjects_flag"),
        "include_subjects_csv": result.get("include_subjects_csv"),
        "n_sad_subjects": result.get("n_sad_subjects"),
        "n_hc_subjects": result.get("n_hc_subjects"),
        "n_sad_trials": result.get("n_sad_trials"),
        "n_hc_trials": result.get("n_hc_trials"),
        "best_c_sad": result.get("best_c_sad"),
        "best_c_hc": result.get("best_c_hc"),
    }
    if extra:
        row.update(extra)
    rows.append(row)


def finite_vector(values):
    if values is None:
        return np.array([], dtype=float)
    arr = np.asarray(values, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def paired_raincloud_rows(result, label, feature_space):
    """Build panel-B raincloud rows from saved 2AFC fold/subject distributions."""
    func_matrix = result.get("func_matrix")
    if func_matrix is None:
        return pd.DataFrame()
    matrix = np.asarray(func_matrix, dtype=float)
    if matrix.shape != (2, 2):
        return pd.DataFrame()
    cohort = result.get("include_subjects_flag") or label
    specs = [
        ("SAD", result.get("cv_fold_scores_sad"), result.get("accs_hc2sad"), matrix[0, 0], matrix[1, 0]),
        ("HC", result.get("cv_fold_scores_hc"), result.get("accs_sad2hc"), matrix[1, 1], matrix[0, 1]),
    ]
    rows = []
    for target_group, within_values, cross_values, within_aggregate, cross_aggregate in specs:
        within = finite_vector(within_values)
        cross = finite_vector(cross_values)
        n = int(max(len(within), len(cross), 1))
        for i in range(n):
            rows.append(
                {
                    "sensitivity_set": cohort,
                    "feature_space": feature_space,
                    "cohort": cohort,
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


def functional_drop_pairs(result, label, feature_space):
    """Return saved self-vs-cross paired rows when available."""
    saved = result.get("functional_drop_pairs")
    if saved is None:
        return pd.DataFrame()
    frame = pd.DataFrame(saved).copy()
    if frame.empty:
        return frame
    cohort = result.get("include_subjects_flag") or label
    frame["sensitivity_set"] = cohort
    frame["feature_space"] = feature_space
    frame["cohort"] = cohort
    frame["resample_id"] = frame.get("resample_id", frame.get("subject_id", pd.Series(np.arange(len(frame)), index=frame.index))).astype(str)
    frame["distribution_source"] = "checkpoint_functional_drop_pairs"
    return frame


def paired_sign_flip_drop_test(pairs, n_perm=10000, seed=20260624):
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


def aggregate_drop_test(result, group):
    """Use the same aggregate self/cross values shown in Panel B dumbbells."""
    matrix = result.get("func_matrix")
    if matrix is None:
        return {}, np.array([], dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (2, 2):
        return {}, np.array([], dtype=float)

    if group == "SAD":
        within = scalar(result.get("acc_sad_cv", matrix[0, 0]))
        cross = scalar(matrix[1, 0])
        within_null = finite_vector(result.get("perm_dist_sad"))
        cross_null = finite_vector(result.get("perm_dist_hc2sad"))
    elif group == "HC":
        within = scalar(result.get("acc_hc_cv", matrix[1, 1]))
        cross = scalar(matrix[0, 1])
        within_null = finite_vector(result.get("perm_dist_hc"))
        cross_null = finite_vector(result.get("perm_dist_sad2hc"))
    else:
        return {}, np.array([], dtype=float)

    if pd.isna(within) or pd.isna(cross):
        return {}, np.array([], dtype=float)

    observed = float(within - cross)
    n_null = min(len(within_null), len(cross_null))
    null = within_null[:n_null] - cross_null[:n_null] if n_null else np.array([], dtype=float)
    null = null[np.isfinite(null)]
    if len(null):
        p_value = float((1 + np.sum(null >= observed)) / (len(null) + 1))
        ci_low, ci_high = np.percentile(null, [2.5, 97.5])
    else:
        p_value = np.nan
        ci_low, ci_high = np.nan, np.nan

    return {
        "n_pairs": 1,
        "within_group_accuracy": float(within),
        "cross_group_accuracy": float(cross),
        "functional_drop": observed,
        "drop_ci_low": float(ci_low) if pd.notna(ci_low) else np.nan,
        "drop_ci_high": float(ci_high) if pd.notna(ci_high) else np.nan,
        "functional_drop_p": p_value,
        "n_permutations": int(len(null)),
        "test": "aggregate_self_minus_cross",
    }, null


def functional_drop_test_rows(result, pairs, label, feature_space):
    """Return aggregate self-vs-cross drop-test rows and optional null rows."""
    cohort = result.get("include_subjects_flag") or label
    tests = []
    null_rows = []
    for group in ["SAD", "HC"]:
        test, null = aggregate_drop_test(result, group)
        for i, value in enumerate(null):
            if np.isfinite(value):
                null_rows.append({
                    "sensitivity_set": cohort,
                    "feature_space": feature_space,
                    "cohort": cohort,
                    "target_group": group,
                    "permutation_id": i,
                    "null_functional_drop": float(value),
                    "null_source": "aggregate_self_minus_cross_null",
                })
        if test:
            test.update({"sensitivity_set": cohort, "feature_space": feature_space, "cohort": cohort, "target_group": group})
            tests.append(test)
    return pd.DataFrame(tests), pd.DataFrame(null_rows)


def write_raincloud_csv(rows, out_csv):
    if out_csv is None:
        return
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=[
            "sensitivity_set",
            "feature_space",
            "cohort",
            "target_group",
            "resample_id",
            "within_accuracy",
            "cross_accuracy",
            "within_aggregate",
            "cross_aggregate",
            "within_metric_source",
            "cross_metric_source",
        ]
    )
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out.to_csv(out_csv, index=False)
    print("Wrote %d Aim 1 SCR raincloud rows -> %s" % (len(out), out_csv))


def force_choice_scores_to_2d(scores):
    scores_arr = np.asarray(scores)
    if scores_arr.ndim == 1:
        scores_arr = np.column_stack((-scores_arr, scores_arr))
    return scores_arr


def forced_choice_predict(scores, classes):
    scores_2d = force_choice_scores_to_2d(scores)
    class_arr = np.asarray(classes)
    return class_arr[np.argmax(scores_2d, axis=1)]


def subject_mean_accuracy(model, X, y, subjects):
    if X is None or y is None or subjects is None or len(y) == 0:
        return np.nan
    scores = model.decision_function(X)
    y_pred = forced_choice_predict(scores, model.classes_)
    subjects = np.asarray(subjects)
    y = np.asarray(y)
    accs = []
    for sub in np.unique(subjects):
        mask = subjects == sub
        if np.any(mask):
            accs.append(float(np.mean(y[mask] == y_pred[mask])))
    return float(np.mean(accs)) if accs else np.nan


def read_result(path):
    payload = joblib.load(path)
    result = payload.get("results_11", payload) if isinstance(payload, dict) else payload
    if not isinstance(result, dict):
        raise ValueError("Checkpoint does not contain a results_11 dictionary: %s" % path)
    return result


def infer_scr_groups_csv(out_csv, scr_groups_csv=None):
    if scr_groups_csv:
        return scr_groups_csv
    out_dir = os.path.dirname(os.path.abspath(out_csv))
    root = os.path.dirname(out_dir) if os.path.basename(out_dir) == "stats" else out_dir
    candidate = os.path.join(root, "harmonized", "scr_sensitivity_groups.csv")
    return candidate


def load_extinction_subset(cell5, group_key):
    if not isinstance(cell5, dict):
        return None
    ext_subsets = cell5.get("ext_subsets", {})
    data_subsets = cell5.get("data_subsets", {})
    subset = ext_subsets.get(group_key)
    if subset is None and group_key in data_subsets:
        subset = data_subsets[group_key].get("ext")
    if not isinstance(subset, dict):
        return None
    return subset


def combine_extinction_subsets(cell5, group_keys):
    subsets = [load_extinction_subset(cell5, group_key) for group_key in group_keys]
    subsets = [subset for subset in subsets if isinstance(subset, dict)]
    if not subsets:
        return None
    return {
        "X": np.concatenate([np.asarray(subset.get("X")) for subset in subsets], axis=0),
        "y": np.concatenate([np.asarray(subset.get("y")) for subset in subsets], axis=0),
        "sub": np.concatenate([np.asarray(subset.get("sub")) for subset in subsets], axis=0),
    }


def add_full_model_subgroup_rows(rows, feature_dir, out_csv, feature_space, scr_groups_csv=None):
    scr_path = infer_scr_groups_csv(out_csv, scr_groups_csv)
    if not os.path.exists(scr_path):
        print("SCR groups file not found; skipping full-model subgroup application: %s" % scr_path)
        return

    full_result_path = os.path.join(feature_dir, "checkpoints", "cell_06.joblib")
    cell5_path = os.path.join(feature_dir, "checkpoints", "cell_05.joblib")
    if not os.path.exists(full_result_path) or not os.path.exists(cell5_path):
        print("Primary Cell 5/6 checkpoints missing; skipping full-model subgroup application.")
        return

    full_result = read_result(full_result_path)
    cell5 = joblib.load(cell5_path)
    model_sad = full_result.get("model_sad")
    model_hc = full_result.get("model_hc")
    if model_sad is None or model_hc is None:
        print("Primary Cell 6 models missing; skipping full-model subgroup application.")
        return

    scr = pd.read_csv(scr_path)
    if "sub_ID" not in scr.columns:
        print("SCR groups file lacks sub_ID; skipping full-model subgroup application.")
        return

    flags = [flag for flag in SCR_FLAGS if flag in scr.columns]
    if not flags:
        print("No SCR cohort flags found; skipping full-model subgroup application.")
        return

    datasets = {
        "SAD": combine_extinction_subsets(cell5, ["SAD_Placebo", "SAD_Oxytocin"]),
        "HC": combine_extinction_subsets(cell5, ["HC_Placebo", "HC_Oxytocin"]),
    }
    model_by_group = {"SAD": model_sad, "HC": model_hc}

    for flag in flags:
        include_subjects = set(scr.loc[as_bool_series(scr[flag]), "sub_ID"].astype(str))
        sad_subject_mask = np.isin(np.asarray(datasets["SAD"]["sub"]).astype(str), list(include_subjects)) if datasets.get("SAD") is not None else np.array([], dtype=bool)
        hc_subject_mask = np.isin(np.asarray(datasets["HC"]["sub"]).astype(str), list(include_subjects)) if datasets.get("HC") is not None else np.array([], dtype=bool)
        n_sad_subjects = int(len(np.unique(np.asarray(datasets["SAD"]["sub"]).astype(str)[sad_subject_mask]))) if datasets.get("SAD") is not None else np.nan
        n_hc_subjects = int(len(np.unique(np.asarray(datasets["HC"]["sub"]).astype(str)[hc_subject_mask]))) if datasets.get("HC") is not None else np.nan
        n_sad_trials = int(np.sum(sad_subject_mask)) if datasets.get("SAD") is not None else np.nan
        n_hc_trials = int(np.sum(hc_subject_mask)) if datasets.get("HC") is not None else np.nan
        for train_group, model in model_by_group.items():
            for test_group, data in datasets.items():
                if data is None:
                    continue
                subjects = np.asarray(data.get("sub")).astype(str)
                mask = np.isin(subjects, list(include_subjects))
                X = np.asarray(data.get("X"))[mask]
                y = np.asarray(data.get("y"))[mask]
                sub = subjects[mask]
                estimate = subject_mean_accuracy(model, X, y, sub)
                test = "Full %s-placebo model tested on pooled-drug %s SCR subgroup" % (train_group, test_group)
                add_row(
                    rows,
                    {
                        "include_subjects_flag": flag,
                        "include_subjects_csv": scr_path,
                        "session": "pooled_drug",
                        "n_sad_subjects": n_sad_subjects,
                        "n_hc_subjects": n_hc_subjects,
                        "n_sad_trials": n_sad_trials,
                        "n_hc_trials": n_hc_trials,
                        "best_c_sad": full_result.get("best_c_sad"),
                        "best_c_hc": full_result.get("best_c_hc"),
                    },
                    flag,
                    feature_space,
                    test,
                    estimate,
                    np.nan,
                    extra={
                        "model_source": "full_placebo_model",
                        "evaluation_scope": "pooled-drug SCR subgroup hold-in application",
                        "train_group": train_group,
                        "test_group": test_group,
                        "n_test_subjects": int(len(np.unique(sub))),
                        "n_test_trials": int(len(sub)),
                    },
                )


def export(feature_dir, out_csv, feature_space, scr_groups_csv=None, raincloud_out=None, drop_tests_out=None, drop_nulls_out=None):
    pattern = os.path.join(feature_dir, "checkpoints", "cell_06_aim1_*.joblib")
    paths = sorted(p for p in glob.glob(pattern) if not p.endswith("_plot.joblib"))
    rows = []
    raincloud_rows = []
    drop_test_rows = []
    drop_null_rows = []
    existing = pd.DataFrame()
    if os.path.exists(out_csv):
        try:
            existing = pd.read_csv(out_csv)
        except Exception:
            existing = pd.DataFrame()

    for path in paths:
        result = read_result(path)
        label = result.get("analysis_label") or os.path.basename(path).replace("cell_06_", "").replace(".joblib", "")
        func_matrix = result.get("func_matrix")
        func_pvals = result.get("p_func_pvals")
        drop_pairs = functional_drop_pairs(result, label, feature_space)
        raincloud = drop_pairs if not drop_pairs.empty else paired_raincloud_rows(result, label, feature_space)
        if not raincloud.empty:
            raincloud_rows.append(raincloud)
        drop_tests, drop_nulls = functional_drop_test_rows(result, drop_pairs, label, feature_space)
        if not drop_tests.empty:
            drop_test_rows.append(drop_tests)
        if not drop_nulls.empty:
            drop_null_rows.append(drop_nulls)

        add_row(rows, result, label, feature_space, "SAD self-decoding CV accuracy", result.get("acc_sad_cv"), result.get("p_sad"))
        add_row(rows, result, label, feature_space, "HC self-decoding CV accuracy", result.get("acc_hc_cv"), result.get("p_hc"))

        if func_matrix is not None and func_pvals is not None:
            func_matrix = np.asarray(func_matrix)
            func_pvals = np.asarray(func_pvals)
            if func_matrix.shape == (2, 2) and func_pvals.shape == (2, 2):
                add_row(
                    rows,
                    result,
                    label,
                    feature_space,
                    "SAD model tested on HC",
                    func_matrix[0, 1],
                    func_pvals[0, 1],
                    extra=permutation_summary(result.get("perm_dist_sad2hc")),
                )
                add_row(
                    rows,
                    result,
                    label,
                    feature_space,
                    "HC model tested on SAD",
                    func_matrix[1, 0],
                    func_pvals[1, 0],
                    extra=permutation_summary(result.get("perm_dist_hc2sad")),
                )

        add_row(
            rows,
            result,
            label,
            feature_space,
            "SAD-HC Haufe map cosine similarity",
            result.get("sim_spatial"),
            result.get("p_sim"),
            extra=permutation_summary(spatial_permutation_values(result)),
        )

    if not paths and not existing.empty:
        keep = existing[
            ~existing.get("model_source", pd.Series(index=existing.index, dtype=object))
            .astype(str)
            .isin(["full_placebo_dataset", "full_placebo_model"])
        ].copy()
        rows.extend(keep.to_dict("records"))

    add_full_model_subgroup_rows(rows, feature_dir, out_csv, feature_space, scr_groups_csv=scr_groups_csv)
    write_raincloud_csv(raincloud_rows, raincloud_out)

    if drop_tests_out is not None:
        drop_tests_table = pd.concat(drop_test_rows, ignore_index=True) if drop_test_rows else pd.DataFrame()
        if drop_tests_table.empty:
            drop_tests_table = pd.DataFrame(columns=[
                "sensitivity_set", "feature_space", "cohort", "target_group", "n_pairs",
                "within_group_accuracy", "cross_group_accuracy", "functional_drop",
                "drop_ci_low", "drop_ci_high", "functional_drop_p", "functional_drop_q",
                "n_permutations", "test",
            ])
        if not drop_tests_table.empty:
            valid = pd.to_numeric(drop_tests_table["functional_drop_p"], errors="coerce").notna()
            drop_tests_table["functional_drop_q"] = np.nan
            if valid.any():
                drop_tests_table.loc[valid, "functional_drop_q"] = multipletests(
                    pd.to_numeric(drop_tests_table.loc[valid, "functional_drop_p"], errors="coerce"),
                    method="fdr_bh",
                )[1]
        os.makedirs(os.path.dirname(drop_tests_out), exist_ok=True)
        drop_tests_table.to_csv(drop_tests_out, index=False)
        print("Wrote %d Aim 1 SCR functional-drop test rows -> %s" % (len(drop_tests_table), drop_tests_out))

    if drop_nulls_out is not None:
        drop_nulls_table = pd.concat(drop_null_rows, ignore_index=True) if drop_null_rows else pd.DataFrame()
        if drop_nulls_table.empty:
            drop_nulls_table = pd.DataFrame(columns=[
                "sensitivity_set", "feature_space", "cohort", "target_group",
                "permutation_id", "null_functional_drop",
            ])
        os.makedirs(os.path.dirname(drop_nulls_out), exist_ok=True)
        drop_nulls_table.to_csv(drop_nulls_out, index=False)
        print("Wrote %d Aim 1 SCR functional-drop null rows -> %s" % (len(drop_nulls_table), drop_nulls_out))

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if out.empty and os.path.exists(out_csv):
        old = pd.read_csv(out_csv)
        if not old.empty:
            print(
                "No labeled Analysis 1 SCR checkpoints matched; preserving existing non-empty CSV -> %s"
                % out_csv
            )
            return
    out.to_csv(out_csv, index=False)
    print("Wrote %d rows from %d labeled checkpoints plus full-model subgroup rows -> %s" % (len(out), len(paths), out_csv))
    if not paths:
        print("No labeled Analysis 1 SCR checkpoints matched: %s" % pattern)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", default="outputs/mvpa_l2/FearNetwork")
    parser.add_argument("--out", default="outputs/mvpa_l2/stats/aim1_scr_sensitivity.csv")
    parser.add_argument("--feature-space", default="FearNetwork")
    parser.add_argument("--scr-groups-csv", default=None)
    parser.add_argument("--raincloud-out", default=None)
    parser.add_argument("--drop-tests-out", default=None)
    parser.add_argument("--drop-nulls-out", default=None)
    args = parser.parse_args()
    raincloud_out = args.raincloud_out
    if raincloud_out is None:
        root, ext = os.path.splitext(args.out)
        raincloud_out = root + "_raincloud" + (ext or ".csv")
    root, ext = os.path.splitext(args.out)
    drop_tests_out = args.drop_tests_out or root + "_functional_drop_tests" + (ext or ".csv")
    drop_nulls_out = args.drop_nulls_out or root + "_functional_drop_nulls" + (ext or ".csv")
    export(
        args.feature_dir,
        args.out,
        args.feature_space,
        scr_groups_csv=args.scr_groups_csv,
        raincloud_out=raincloud_out,
        drop_tests_out=drop_tests_out,
        drop_nulls_out=drop_nulls_out,
    )


if __name__ == "__main__":
    main()
