#!/usr/bin/env python3
"""Run feature-space and SCR-cohort sensitivity models for MVPA L2."""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from mvpa_l2_common import (
    ALL_CLINICAL_SCORES,
    ALL_SCR_INDICES,
    CLINICAL_SCORE_HIERARCHY,
    PRIMARY_SCR_INDICES,
    PRESPECIFIED_NEURAL_METRICS,
    SCR_SENSITIVITY_FLAGS,
    add_fdr,
    available_covariates,
    derive_final_metrics,
    fit_lm,
    harmonize_group_drug,
    write_csv,
)
from run_mvpa_l2_primary_models import (
    AIM2_PRIMARY_METRICS,
    AIM2_SECONDARY_METRICS,
    metric_hierarchy_fields,
    neural_question_fields,
    apply_stage29_zscore,
    zscore_numeric_covariates,
)


GROUP_TERM = "C(Group, Treatment(reference='HC'))[T.SAD]"
DRUG_TERM = "C(Drug, Treatment(reference='Placebo'))"
DRUG_INTERACTION_TERM = "C(Group, Treatment(reference='HC'))[T.SAD]:C(Drug, Treatment(reference='Placebo'))[T.Oxytocin]"
AIM2_GROUP_METRICS = AIM2_PRIMARY_METRICS + AIM2_SECONDARY_METRICS
FEATURE_SPACE_ALIASES = {
    "Schaefer": "Schaefer_Tian",
    "Schaefer+Tian": "Schaefer_Tian",
    "Tian": "Schaefer_Tian",
    "WholeBrain": "Schaefer_Tian",
    "WholeBrain_Schaefer": "Schaefer_Tian",
    "WholeBrain_Parcellation": "Schaefer_Tian",
}


def normalize_feature_space(value: object) -> str:
    text = str(value).strip()
    return FEATURE_SPACE_ALIASES.get(text, text)


def sad_minus_hc_cohens_d(df: pd.DataFrame, outcome: str) -> Dict[str, object]:
    """Compute unadjusted standardized SAD-HC effect size for display."""
    if outcome not in df.columns or "Group" not in df.columns:
        return {
            "effect_size": np.nan,
            "effect_size_scale": "cohens_d_sad_minus_hc",
            "n_sad": 0,
            "n_hc": 0,
        }
    sub = df[["Group", outcome]].copy()
    sub[outcome] = pd.to_numeric(sub[outcome], errors="coerce")
    sub = sub.dropna(subset=["Group", outcome])
    sad = sub.loc[sub["Group"].astype(str).eq("SAD"), outcome].to_numpy(dtype=float)
    hc = sub.loc[sub["Group"].astype(str).eq("HC"), outcome].to_numpy(dtype=float)
    n_sad = int(sad.size)
    n_hc = int(hc.size)
    if n_sad < 2 or n_hc < 2:
        effect = np.nan
    else:
        sad_sd = float(np.nanstd(sad, ddof=1))
        hc_sd = float(np.nanstd(hc, ddof=1))
        pooled_var = ((n_sad - 1) * sad_sd**2 + (n_hc - 1) * hc_sd**2) / (n_sad + n_hc - 2)
        pooled_sd = float(np.sqrt(pooled_var)) if np.isfinite(pooled_var) and pooled_var > 0 else np.nan
        effect = float((np.nanmean(sad) - np.nanmean(hc)) / pooled_sd) if np.isfinite(pooled_sd) else np.nan
    return {
        "effect_size": effect,
        "effect_size_scale": "cohens_d_sad_minus_hc",
        "n_sad": n_sad,
        "n_hc": n_hc,
        "sad_mean": float(np.nanmean(sad)) if n_sad else np.nan,
        "hc_mean": float(np.nanmean(hc)) if n_hc else np.nan,
    }


def run_group_model(
    df: pd.DataFrame,
    label: str,
    feature_space: str,
    covariates: List[str],
    drug_scope: str = "placebo",
) -> List[Dict]:
    rows = []
    sub = df[df["FeatureSpace"] == feature_space].copy()
    session = "pooled_drug"
    predictor_terms = ["C(Group, Treatment(reference='HC'))"]
    if drug_scope == "placebo":
        sub = sub[sub["Drug"] == "Placebo"].copy()
        session = "Placebo"
    elif sub["Drug"].dropna().nunique() > 1:
        predictor_terms.append(DRUG_TERM)

    for metric in AIM2_GROUP_METRICS:
        effect_fields = sad_minus_hc_cohens_d(sub, metric)
        row = fit_lm(
            sub,
            outcome=metric,
            predictor_terms=predictor_terms,
            covariates=covariates,
            term_of_interest=GROUP_TERM,
            min_n=10,
        )
        row.update(
            {
                "analysis": "Sensitivity_Aim2_Group",
                "sensitivity": label,
                "metric": metric,
                "feature_space": feature_space,
                "session": session,
            }
        )
        row.update(effect_fields)
        row.update(neural_question_fields(metric))
        row.update(metric_hierarchy_fields(metric))
        rows.append(row)
    return rows


def clinical_hierarchy_fields(clinical_score: str) -> Dict[str, object]:
    fields = CLINICAL_SCORE_HIERARCHY.get(clinical_score, {})
    return {
        "clinical_score_order": fields.get("order", 999),
        "clinical_score_role": fields.get("role", "exploratory"),
        "clinical_score_family": fields.get("family", "exploratory"),
        "clinical_score_label": fields.get("label", clinical_score),
    }


def run_clinical_model(
    df: pd.DataFrame,
    label: str,
    feature_space: str,
    covariates: List[str],
    outlier_z: float,
    drug_scope: str = "pooled",
) -> List[Dict]:
    rows = []
    sub = df[df["FeatureSpace"] == feature_space].copy()
    session = "pooled_drug"
    if drug_scope == "placebo":
        sub = sub[sub["Drug"] == "Placebo"].copy()
        session = "Placebo"

    groups = [g for g in ["SAD", "HC"] if g in set(sub["Group"].dropna())]
    for group in groups:
        group_df = sub[sub["Group"] == group].copy()
        model_covariates = list(covariates)
        if drug_scope == "pooled" and group_df["Drug"].dropna().nunique() > 1:
            model_covariates = ["Drug"] + model_covariates
        group_df, model_covariates, covariate_outliers = zscore_numeric_covariates(group_df, model_covariates, outlier_z)
        for clinical in ALL_CLINICAL_SCORES:
            clinical_df, clinical_z, n_clinical_outliers, clinical_method = apply_stage29_zscore(group_df, clinical, outlier_z)
            if clinical_z is None:
                row = {
                    "analysis": "Sensitivity_Aim3_Clinical",
                    "sensitivity": label,
                    "feature_space": feature_space,
                    "session": session,
                    "Group": group,
                    "clinical_score": clinical,
                    "clinical_score_z": None,
                    "clinical_outlier_method": clinical_method,
                    "outlier_threshold": outlier_z,
                    "n_clinical_outliers_removed": n_clinical_outliers,
                    "status": "missing_or_constant_clinical_score",
                    "n": 0,
                    "outcome": clinical,
                }
                row.update(clinical_hierarchy_fields(clinical))
                rows.append(row)
                continue
            for metric in PRESPECIFIED_NEURAL_METRICS:
                if metric not in clinical_df.columns:
                    continue
                model_df, metric_z, n_metric_outliers, metric_method = apply_stage29_zscore(clinical_df, metric, outlier_z)
                if metric_z is None:
                    row = {
                        "analysis": "Sensitivity_Aim3_Clinical",
                        "sensitivity": label,
                        "feature_space": feature_space,
                        "session": session,
                        "Group": group,
                        "metric": metric,
                        "metric_z": None,
                        "clinical_score": clinical,
                        "clinical_score_z": clinical_z,
                        "clinical_outlier_method": clinical_method,
                        "metric_outlier_method": metric_method,
                        "outlier_threshold": outlier_z,
                        "n_clinical_outliers_removed": n_clinical_outliers,
                        "n_metric_outliers_removed": n_metric_outliers,
                        "status": "missing_or_constant_neural_metric",
                        "n": 0,
                        "outcome": clinical_z,
                    }
                    row.update(clinical_hierarchy_fields(clinical))
                    rows.append(row)
                    continue
                row = fit_lm(
                    model_df,
                    outcome=clinical_z,
                    predictor_terms=[f"Q('{metric_z}')"],
                    covariates=model_covariates,
                    term_of_interest=f"Q('{metric_z}')",
                    min_n=10,
                )
                row.update(
                    {
                        "analysis": "Sensitivity_Aim3_Clinical",
                        "sensitivity": label,
                        "feature_space": feature_space,
                        "session": session,
                        "Group": group,
                        "metric": metric,
                        "metric_z": metric_z,
                        "clinical_score": clinical,
                        "clinical_score_z": clinical_z,
                        "clinical_outlier_method": clinical_method,
                        "metric_outlier_method": metric_method,
                        "outlier_threshold": outlier_z,
                        "n_clinical_outliers_removed": n_clinical_outliers,
                        "n_metric_outliers_removed": n_metric_outliers,
                        "covariates_used": ",".join(model_covariates),
                        "covariate_outliers_removed": ";".join(f"{k}:{v}" for k, v in covariate_outliers.items()),
                    }
                )
                row.update(clinical_hierarchy_fields(clinical))
                rows.append(row)
    return rows


def run_scr_model(df: pd.DataFrame, label: str, feature_space: str, covariates: List[str], outlier_z: float) -> List[Dict]:
    rows = []
    sub = df[df["FeatureSpace"] == feature_space].copy()
    for scr in ALL_SCR_INDICES:
        scr_df, scr_z, n_scr_outliers, scr_method = apply_stage29_zscore(sub, scr, outlier_z)
        if scr_z is None:
            for metric in PRESPECIFIED_NEURAL_METRICS:
                rows.append(
                    {
                        "analysis": "Sensitivity_Aim4_SCR",
                        "sensitivity": label,
                        "metric": metric,
                        "scr_index": scr,
                        "scr_index_z": None,
                        "scr_index_role": "primary" if scr in PRIMARY_SCR_INDICES else "secondary",
                        "scr_outlier_method": scr_method,
                        "outlier_threshold": outlier_z,
                        "n_scr_outliers_removed": n_scr_outliers,
                        "n_metric_outliers_removed": 0,
                        "feature_space": feature_space,
                        "status": "missing_or_constant_scr_index",
                        "n": 0,
                        "outcome": scr,
                    }
                )
            continue
        for metric in PRESPECIFIED_NEURAL_METRICS:
            if metric not in scr_df.columns:
                continue
            model_df, metric_z, n_metric_outliers, metric_method = apply_stage29_zscore(scr_df, metric, outlier_z)
            if metric_z is None:
                row = {
                    "analysis": "Sensitivity_Aim4_SCR",
                    "sensitivity": label,
                    "metric": metric,
                    "metric_z": None,
                    "scr_index": scr,
                    "scr_index_z": scr_z,
                    "scr_index_role": "primary" if scr in PRIMARY_SCR_INDICES else "secondary",
                    "scr_outlier_method": scr_method,
                    "metric_outlier_method": metric_method,
                    "outlier_threshold": outlier_z,
                    "n_scr_outliers_removed": n_scr_outliers,
                    "n_metric_outliers_removed": n_metric_outliers,
                    "feature_space": feature_space,
                    "status": "missing_or_constant_neural_metric",
                    "n": 0,
                    "outcome": scr_z,
                }
                rows.append(row)
                continue
            row = fit_lm(
                model_df,
                outcome=scr_z,
                predictor_terms=[f"Q('{metric_z}')", "C(Group, Treatment(reference='HC'))", "C(Drug, Treatment(reference='Placebo'))"],
                covariates=covariates,
                term_of_interest=f"Q('{metric_z}')",
                min_n=10,
            )
            row.update({
                "analysis": "Sensitivity_Aim4_SCR",
                "sensitivity": label,
                "metric": metric,
                "metric_z": metric_z,
                "scr_index": scr,
                "scr_index_z": scr_z,
                "scr_index_role": "primary" if scr in PRIMARY_SCR_INDICES else "secondary",
                "scr_outlier_method": scr_method,
                "metric_outlier_method": metric_method,
                "outlier_threshold": outlier_z,
                "n_scr_outliers_removed": n_scr_outliers,
                "n_metric_outliers_removed": n_metric_outliers,
                "feature_space": feature_space,
            })
            rows.append(row)
    return rows


def run_drug_model(df: pd.DataFrame, label: str, feature_space: str, covariates: List[str]) -> List[Dict]:
    rows = []
    sub = df[df["FeatureSpace"] == feature_space].copy()
    for metric in PRESPECIFIED_NEURAL_METRICS:
        row = fit_lm(
            sub,
            outcome=metric,
            predictor_terms=["C(Group, Treatment(reference='HC')) * C(Drug, Treatment(reference='Placebo'))"],
            covariates=covariates,
            term_of_interest=DRUG_INTERACTION_TERM,
            min_n=12,
        )
        row.update({"analysis": "Sensitivity_Aim5_Group_x_Drug", "sensitivity": label, "metric": metric, "feature_space": feature_space})
        rows.append(row)
    return rows


def cell_count_ok(df: pd.DataFrame, min_cell_n: int) -> bool:
    if not {"Group", "Drug"}.issubset(df.columns):
        return False
    counts = df.groupby(["Group", "Drug"], dropna=False).size()
    return len(counts) >= 2 and counts.min() >= min_cell_n


def truthy_flag(series: pd.Series) -> pd.Series:
    """Parse boolean-like SCR cohort flags without pandas downcast warnings."""
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype("string").str.strip().str.lower().isin({"true", "1", "yes", "y"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/mvpa_l2/harmonized/mvpa_l2_subject_metrics.csv"),
    )
    parser.add_argument("--primary-feature-space", default="FearNetwork")
    parser.add_argument("--covariates", nargs="*", default=None)
    parser.add_argument("--min-cell-n", type=int, default=6)
    parser.add_argument("--outlier-z", type=float, default=3.0)
    parser.add_argument("--out", type=Path, default=Path("outputs/mvpa_l2/stats/sensitivity_models_all.csv"))
    args = parser.parse_args()

    df = derive_final_metrics(harmonize_group_drug(pd.read_csv(args.input)))
    if "FeatureSpace" in df.columns:
        df["FeatureSpace"] = df["FeatureSpace"].map(normalize_feature_space)
    covariates = available_covariates(df, args.covariates)
    rows = []

    primary_feature_label = f"FeatureSpace:{args.primary_feature_space}"
    rows.extend(run_group_model(df, primary_feature_label, args.primary_feature_space, covariates, drug_scope="placebo"))
    rows.extend(run_group_model(df, "AllPlacebo", args.primary_feature_space, covariates, drug_scope="placebo"))

    full_sample_label = "FullSample:DrugAdjusted"
    rows.extend(run_group_model(df, full_sample_label, args.primary_feature_space, covariates, drug_scope="pooled"))
    rows.extend(run_clinical_model(df, full_sample_label, args.primary_feature_space, covariates, args.outlier_z, drug_scope="pooled"))
    rows.extend(run_scr_model(df, full_sample_label, args.primary_feature_space, covariates, args.outlier_z))

    feature_spaces = sorted([fs for fs in df["FeatureSpace"].dropna().unique() if fs != args.primary_feature_space])
    for feature_space in feature_spaces:
        label = f"FeatureSpace:{feature_space}"
        rows.extend(run_group_model(df, label, feature_space, covariates))
        rows.extend(run_clinical_model(df, label, feature_space, covariates, args.outlier_z, drug_scope="placebo"))
        rows.extend(run_scr_model(df, label, feature_space, covariates, args.outlier_z))
        rows.extend(run_drug_model(df, label, feature_space, covariates))

    primary_df = df[df["FeatureSpace"] == args.primary_feature_space].copy()
    for flag in SCR_SENSITIVITY_FLAGS:
        if flag not in primary_df.columns:
            print(f"[WARN] SCR sensitivity flag missing: {flag}")
            continue
        sub = primary_df[truthy_flag(primary_df[flag])].copy()
        label = f"SCRCohort:{flag}"
        if len(sub) == 0:
            print(f"[WARN] Empty cohort: {flag}")
            continue
        rows.extend(run_group_model(sub, label, args.primary_feature_space, covariates, drug_scope="pooled"))
        rows.extend(run_clinical_model(sub, label, args.primary_feature_space, covariates, args.outlier_z, drug_scope="pooled"))
        rows.extend(run_scr_model(sub, label, args.primary_feature_space, covariates, args.outlier_z))
        if cell_count_ok(sub, args.min_cell_n):
            rows.extend(run_drug_model(sub, label, args.primary_feature_space, covariates))
        else:
            print(f"[INFO] Skipping Group*Drug sensitivity for {flag}; cell counts below {args.min_cell_n}.")

    out = pd.DataFrame(rows)
    if not out.empty:
        if "metric" in out.columns:
            metric = out["metric"].astype("string")
            out = out[metric.isna() | metric.isin(AIM2_GROUP_METRICS)].copy()
            metric_fields = out["metric"].map(metric_hierarchy_fields).apply(pd.Series)
            for col in metric_fields.columns:
                if col not in out.columns:
                    out[col] = metric_fields[col]
                else:
                    out[col] = out[col].combine_first(metric_fields[col])
        out = add_fdr(out, family_cols=["analysis", "sensitivity", "metric_role"])
        aim2 = out["analysis"].astype(str).eq("Sensitivity_Aim2_Group")
        out.loc[aim2, "correction_family"] = (
            out.loc[aim2, "analysis"].astype(str)
            + " | "
            + out.loc[aim2, "sensitivity"].astype(str)
            + " | Aim2 planned metric family"
        )
        other_aims = ~aim2
        out.loc[other_aims, "correction_family"] = (
            out.loc[other_aims, "analysis"].astype(str)
            + " | "
            + out.loc[other_aims, "sensitivity"].astype(str)
            + " | "
            + out.loc[other_aims, "metric_role"].astype(str)
        )
    write_csv(out, args.out)
    print(f"Wrote {len(out)} sensitivity rows -> {args.out}")


if __name__ == "__main__":
    main()
