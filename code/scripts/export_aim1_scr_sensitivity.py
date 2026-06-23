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


def export(feature_dir, out_csv, feature_space, scr_groups_csv=None):
    pattern = os.path.join(feature_dir, "checkpoints", "cell_06_aim1_*.joblib")
    paths = sorted(p for p in glob.glob(pattern) if not p.endswith("_plot.joblib"))
    rows = []
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

        add_row(rows, result, label, feature_space, "SAD self-decoding CV accuracy", result.get("acc_sad_cv"), result.get("p_sad"))
        add_row(rows, result, label, feature_space, "HC self-decoding CV accuracy", result.get("acc_hc_cv"), result.get("p_hc"))

        if func_matrix is not None and func_pvals is not None:
            func_matrix = np.asarray(func_matrix)
            func_pvals = np.asarray(func_pvals)
            if func_matrix.shape == (2, 2) and func_pvals.shape == (2, 2):
                add_row(rows, result, label, feature_space, "SAD model tested on HC", func_matrix[0, 1], func_pvals[0, 1])
                add_row(rows, result, label, feature_space, "HC model tested on SAD", func_matrix[1, 0], func_pvals[1, 0])

        add_row(rows, result, label, feature_space, "SAD-HC Haufe map cosine similarity", result.get("sim_spatial"), result.get("p_sim"))

    if not paths and not existing.empty:
        keep = existing[
            ~existing.get("model_source", pd.Series(index=existing.index, dtype=object))
            .astype(str)
            .isin(["full_placebo_dataset", "full_placebo_model"])
        ].copy()
        rows.extend(keep.to_dict("records"))

    add_full_model_subgroup_rows(rows, feature_dir, out_csv, feature_space, scr_groups_csv=scr_groups_csv)

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
    args = parser.parse_args()
    export(args.feature_dir, args.out, args.feature_space, scr_groups_csv=args.scr_groups_csv)


if __name__ == "__main__":
    main()
