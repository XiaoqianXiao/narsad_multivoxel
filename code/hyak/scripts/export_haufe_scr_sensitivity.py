#!/usr/bin/env python3
"""Export SCR-subgroup Haufe spatial-pattern stability checks.

This is a lightweight sensitivity analysis for the Haufe-transformed maps used
in the MVPA L2 report. It retrains the placebo CSR-vs-CSS linear decoder within
SCR-defined subject subsets, computes Haufe activation patterns using the same
scaled-covariance transform as the primary script, and compares the selected
voxel ROI distribution with the full placebo-sample Haufe pattern.
"""

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCR_FLAGS = [
    "SCR_Physiological_Responder",
    "SCR_Simple_Acquisition_Differential_Learner",
    "SCR_Habituation_Adjusted_Learner",
    "SCR_Late_Phase_Sensitivity_Learner",
]


def as_bool_series(series):
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def normalize_subject_id(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if text.lower().startswith("sub-"):
        text = text[4:]
    return text


def load_joblib(path):
    return joblib.load(path)


def roi_labels_from_parcels(parcel_names):
    labels = []
    for name in parcel_names:
        parts = str(name).split("_")
        if len(parts) >= 3 and parts[-1].isdigit():
            labels.append("_".join(parts[:-1]))
        elif len(parts) >= 2 and parts[-1].isdigit():
            labels.append(parts[0])
        else:
            labels.append(str(name))
    return np.asarray(labels)


def compute_haufe_pattern(X, y, c_value):
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classification",
                LogisticRegression(
                    penalty="l2",
                    C=float(c_value),
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X, y)
    x_scaled = model.named_steps["scaler"].transform(X)
    weights = model.named_steps["classification"].coef_
    pattern = np.cov(x_scaled, rowvar=False).dot(weights.T).ravel()
    return pattern, model


def top_abs_mask(values, top_n):
    values = np.asarray(values, dtype=float).ravel()
    finite = np.isfinite(values)
    mask = np.zeros(values.shape[0], dtype=bool)
    if top_n <= 0 or not finite.any():
        return mask
    finite_idx = np.flatnonzero(finite)
    top_n = min(int(top_n), finite_idx.size)
    selected = finite_idx[np.argsort(np.abs(values[finite_idx]))[-top_n:]]
    mask[selected] = True
    return mask


def roi_wise_fdr_mask(p_values, roi_labels, alpha=0.05):
    p_values = np.asarray(p_values, dtype=float).ravel()
    roi_labels = np.asarray(roi_labels)
    mask = np.zeros(p_values.shape[0], dtype=bool)
    for roi in pd.unique(roi_labels):
        idx = np.flatnonzero(roi_labels == roi)
        valid = idx[np.isfinite(p_values[idx])]
        if valid.size == 0:
            continue
        rejected = multipletests(p_values[valid], alpha=alpha, method="fdr_bh")[0]
        mask[valid[rejected]] = True
    return mask


def roi_distribution(mask, roi_labels, roi_order):
    total = max(int(np.sum(mask)), 1)
    rows = []
    vector = []
    for roi in roi_order:
        roi_mask = roi_labels == roi
        selected = int(np.sum(mask & roi_mask))
        available = int(np.sum(roi_mask))
        pct_selected = selected / total
        pct_roi = selected / available if available else np.nan
        rows.append(
            {
                "roi": roi,
                "roi_voxels": available,
                "selected_voxels": selected,
                "selected_pct_of_selected": pct_selected,
                "selected_pct_of_roi": pct_roi,
            }
        )
        vector.append(pct_selected)
    return pd.DataFrame(rows), np.asarray(vector, dtype=float)


def safe_corr(fun, x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.nanstd(x[ok]) == 0 or np.nanstd(y[ok]) == 0:
        return np.nan
    return float(fun(x[ok], y[ok])[0])


def cosine(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() == 0:
        return np.nan
    denom = np.linalg.norm(x[ok]) * np.linalg.norm(y[ok])
    return float(np.dot(x[ok], y[ok]) / denom) if denom else np.nan


def load_roi_fdr_masks(feature_dir, roi_labels, alpha=0.05):
    masks = {}
    counts = {}
    for group in ["SAD", "HC"]:
        candidates = [
            Path(feature_dir) / "checkpoints" / f"perm_results_{group}_fear_network_2way.joblib",
            Path(feature_dir) / f"perm_results_{group}_fear_network_2way.joblib",
            Path(feature_dir) / "intermediate" / f"stage11_importance_masks_{group}.joblib",
            Path(feature_dir) / "checkpoints" / f"cell_11_{group}.joblib",
        ]
        payload = None
        for path in candidates:
            if path.exists():
                payload = load_joblib(path)
                break
        p_values = None
        if isinstance(payload, dict):
            p_values = payload.get("p_values_raw")
            if p_values is None:
                p_values = payload.get("p_values_permutated", {}).get(group)
        if p_values is None:
            masks[group] = np.zeros(len(roi_labels), dtype=bool)
        else:
            masks[group] = roi_wise_fdr_mask(p_values, roi_labels, alpha=alpha)
        counts[group] = int(np.sum(masks[group]))
    return masks, counts


def export(feature_dir, scr_groups_csv, out_summary, out_roi, feature_space="FearNetwork", min_subjects=6):
    feature_dir = Path(feature_dir)
    cell6 = load_joblib(feature_dir / "checkpoints" / "cell_06.joblib")
    cell3 = load_joblib(feature_dir / "checkpoints" / "cell_03.joblib")
    result = cell6.get("results_11", cell6)
    roi_labels = roi_labels_from_parcels(cell3["parcel_names_ext"])
    roi_order = sorted(pd.unique(roi_labels))
    roi_fdr_masks, roi_fdr_counts = load_roi_fdr_masks(feature_dir, roi_labels)
    nonzero_counts = [count for count in roi_fdr_counts.values() if count > 0]
    top_n = int(max(nonzero_counts)) if nonzero_counts else 53

    full_patterns = {
        "SAD": np.asarray(result["map_sad"], dtype=float).ravel(),
        "HC": np.asarray(result["map_hc"], dtype=float).ravel(),
    }
    c_values = {"SAD": result.get("best_c_sad", 0.01), "HC": result.get("best_c_hc", 0.01)}
    datasets = {
        "SAD": cell6["data_subsets"]["SAD_Placebo"]["ext"],
        "HC": cell6["data_subsets"]["HC_Placebo"]["ext"],
    }

    full_dist = {}
    full_masks = {}
    for group, pattern in full_patterns.items():
        mask = roi_fdr_masks[group] if roi_fdr_counts[group] > 0 else top_abs_mask(pattern, top_n)
        full_masks[group] = mask
        _, full_dist[group] = roi_distribution(mask, roi_labels, roi_order)

    scr = pd.read_csv(scr_groups_csv)
    scr["sub_ID"] = scr["sub_ID"].map(normalize_subject_id)
    flags = [flag for flag in SCR_FLAGS if flag in scr.columns]

    summary_rows = []
    roi_rows = []
    for flag in flags:
        include_subjects = set(scr.loc[as_bool_series(scr[flag]), "sub_ID"].astype(str))
        for group, data in datasets.items():
            subjects = np.asarray(data["sub"]).astype(str)
            keep = np.isin(subjects, list(include_subjects))
            X = np.asarray(data["X"])[keep]
            y = np.asarray(data["y"])[keep]
            sub = subjects[keep]
            n_subjects = int(len(np.unique(sub)))
            n_trials = int(len(sub))
            n_classes = int(len(np.unique(y))) if len(y) else 0
            base = {
                "feature_space": feature_space,
                "session": "Placebo",
                "scr_cohort": flag,
                "group": group,
                "top_n_selected_voxels": int(top_n),
                "full_group_roi_fdr_voxels": int(roi_fdr_counts.get(group, 0)),
                "full_group_display_mode": "ROI-FDR" if roi_fdr_counts.get(group, 0) > 0 else f"Top {int(top_n)} matched to ROI-FDR count",
                "n_subjects": n_subjects,
                "n_trials": n_trials,
            }
            if n_subjects < min_subjects or n_classes < 2:
                row = dict(base)
                row.update({"status": "too_few_subjects_or_classes"})
                summary_rows.append(row)
                continue

            pattern, _ = compute_haufe_pattern(X, y, c_values[group])
            mask = top_abs_mask(pattern, top_n)
            dist_df, dist_vec = roi_distribution(mask, roi_labels, roi_order)
            full_vec = full_dist[group]
            full_mask = full_masks[group]
            row = dict(base)
            row.update(
                {
                    "status": "ok",
                    "haufe_cosine_to_full_group_map": cosine(pattern, full_patterns[group]),
                    "roi_distribution_pearson_to_full": safe_corr(pearsonr, dist_vec, full_vec),
                    "roi_distribution_spearman_to_full": safe_corr(spearmanr, dist_vec, full_vec),
                    "roi_distribution_js_distance_to_full": float(jensenshannon(dist_vec + 1e-12, full_vec + 1e-12)),
                    "selected_voxel_overlap_jaccard_to_full": float(np.sum(mask & full_mask) / max(np.sum(mask | full_mask), 1)),
                    "dominant_roi": roi_order[int(np.nanargmax(dist_vec))] if np.isfinite(dist_vec).any() else np.nan,
                    "full_dominant_roi": roi_order[int(np.nanargmax(full_vec))] if np.isfinite(full_vec).any() else np.nan,
                    "dominant_roi_matches_full": bool(np.nanargmax(dist_vec) == np.nanargmax(full_vec)) if np.isfinite(dist_vec).any() and np.isfinite(full_vec).any() else np.nan,
                }
            )
            summary_rows.append(row)
            dist_df.insert(0, "group", group)
            dist_df.insert(0, "scr_cohort", flag)
            dist_df.insert(0, "session", "Placebo")
            dist_df.insert(0, "feature_space", feature_space)
            dist_df["top_n_selected_voxels"] = int(top_n)
            dist_df["full_group_roi_fdr_voxels"] = int(roi_fdr_counts.get(group, 0))
            dist_df["full_group_display_mode"] = "ROI-FDR" if roi_fdr_counts.get(group, 0) > 0 else f"Top {int(top_n)} matched to ROI-FDR count"
            dist_df["n_subjects"] = n_subjects
            dist_df["n_trials"] = n_trials
            roi_rows.extend(dist_df.to_dict("records"))

    out_summary = Path(out_summary)
    out_roi = Path(out_roi)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_roi.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(out_summary, index=False)
    pd.DataFrame(roi_rows).to_csv(out_roi, index=False)
    print(f"Wrote Haufe SCR sensitivity summary -> {out_summary}")
    print(f"Wrote Haufe SCR sensitivity ROI distribution -> {out_roi}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", default="outputs/mvpa_l2/FearNetwork")
    parser.add_argument("--scr-groups-csv", default="outputs/mvpa_l2/harmonized/scr_sensitivity_groups.csv")
    parser.add_argument("--out-summary", default="outputs/mvpa_l2/stats/aim2_haufe_scr_sensitivity.csv")
    parser.add_argument("--out-roi", default="outputs/mvpa_l2/stats/aim2_haufe_scr_sensitivity_roi_distribution.csv")
    parser.add_argument("--feature-space", default="FearNetwork")
    parser.add_argument("--min-subjects", type=int, default=6)
    args = parser.parse_args()
    export(
        feature_dir=args.feature_dir,
        scr_groups_csv=args.scr_groups_csv,
        out_summary=args.out_summary,
        out_roi=args.out_roi,
        feature_space=args.feature_space,
        min_subjects=args.min_subjects,
    )


if __name__ == "__main__":
    main()
