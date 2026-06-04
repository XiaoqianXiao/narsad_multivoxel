#!/usr/bin/env python3
"""Export labeled Analysis 1 SCR-subgroup sensitivity checkpoints to CSV."""

import argparse
import glob
import os

import joblib
import numpy as np
import pandas as pd


def scalar(value):
    if value is None:
        return np.nan
    try:
        return float(value)
    except Exception:
        return value


def add_row(rows, result, label, feature_space, test, estimate, p_value, extra=None):
    row = {
        "analysis_label": label,
        "feature_space": feature_space,
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


def read_result(path):
    payload = joblib.load(path)
    result = payload.get("results_11", payload) if isinstance(payload, dict) else payload
    if not isinstance(result, dict):
        raise ValueError("Checkpoint does not contain a results_11 dictionary: %s" % path)
    return result


def export(feature_dir, out_csv, feature_space):
    pattern = os.path.join(feature_dir, "checkpoints", "cell_06_aim1_*.joblib")
    paths = sorted(p for p in glob.glob(pattern) if not p.endswith("_plot.joblib"))
    rows = []

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

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out.to_csv(out_csv, index=False)
    print("Wrote %d rows from %d checkpoints -> %s" % (len(out), len(paths), out_csv))
    if not paths:
        print("No labeled Analysis 1 SCR checkpoints matched: %s" % pattern)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", default="outputs/mvpa_l2/FearNetwork")
    parser.add_argument("--out", default="outputs/mvpa_l2/stats/aim1_scr_sensitivity.csv")
    parser.add_argument("--feature-space", default="FearNetwork")
    args = parser.parse_args()
    export(args.feature_dir, args.out, args.feature_space)


if __name__ == "__main__":
    main()
