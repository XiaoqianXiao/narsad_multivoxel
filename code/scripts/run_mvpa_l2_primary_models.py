#!/usr/bin/env python3
"""Run primary MVPA L2 models from the harmonized subject table."""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from mvpa_l2_common import (
    ALL_CLINICAL_SCORES,
    ALL_SCR_INDICES,
    CLINICAL_SCORE_HIERARCHY,
    COMPANION_NEURAL_METRICS,
    CORE_NEURAL_METRICS,
    NEURAL_METRIC_HIERARCHY,
    add_fdr,
    available_covariates,
    derive_final_metrics,
    fit_lm,
    harmonize_group_drug,
    write_csv,
)


GROUP_TERM = "C(Group, Treatment(reference='HC'))[T.SAD]"
DRUG_INTERACTION_TERM = "C(Group, Treatment(reference='HC'))[T.SAD]:C(Drug, Treatment(reference='Placebo'))[T.Oxytocin]"

AIM2_QUESTION_METRICS = {
    "Q1_geometry": [
        "Neural_Dist_Safety_Background",
        "Neural_Dist_Threat_Safety",
        "Neural_Dist_Threat_Background",
    ],
    "Q2_decision_certainty": [
        "Neural_SafetyEvidence",
        "Neural_ThreatEvidence",
        "Neural_Decoder_Entropy_CSS",
        "Neural_Decoder_Entropy_CSR",
    ],
    "Q3_learning_dynamics": [
        "Neural_Safety_Trajectory_Slope",
        "Neural_Threat_Trajectory_Slope",
        "Shock_Anchor_Trajectory_Slope",
        "Residualized_Shock_Anchor_Trajectory_Slope",
    ],
}

AIM2_QUESTION_LABELS = {
    "Q1_geometry": "Where are safety and threat in neural space?",
    "Q2_decision_certainty": "Are safety and threat cues represented decisively or ambiguously?",
    "Q3_learning_dynamics": "How do safety and threat representations change over learning?",
}

AIM2_SECONDARY_METRICS = {
    "Neural_Dist_Threat_Background",
    "Neural_Decoder_Entropy_CSS",
    "Neural_Decoder_Entropy_CSR",
    "Shock_Anchor_Trajectory_Slope",
    "Residualized_Shock_Anchor_Trajectory_Slope",
}

AIM2_PRIMARY_METRICS = set(CORE_NEURAL_METRICS)
AIM2_METRIC_TO_QUESTION = {
    metric: question
    for question, metrics in AIM2_QUESTION_METRICS.items()
    for metric in metrics
}


def neural_question_fields(metric: str) -> Dict[str, object]:
    question = AIM2_METRIC_TO_QUESTION.get(metric)
    if question is None:
        return {
            "aim2_question": None,
            "aim2_question_order": 999,
            "aim2_question_label": None,
        }
    return {
        "aim2_question": question,
        "aim2_question_order": list(AIM2_QUESTION_METRICS).index(question) + 1,
        "aim2_question_label": AIM2_QUESTION_LABELS[question],
    }


def clinical_hierarchy_fields(clinical_score: str) -> Dict[str, object]:
    fields = CLINICAL_SCORE_HIERARCHY.get(clinical_score, {})
    return {
        "clinical_score_order": fields.get("order", 999),
        "clinical_score_role": fields.get("role", "exploratory"),
        "clinical_score_family": fields.get("family", "exploratory"),
        "clinical_score_label": fields.get("label", clinical_score),
    }


def metric_hierarchy_fields(metric: str) -> Dict[str, object]:
    fields = NEURAL_METRIC_HIERARCHY.get(metric, {})
    return {
        "metric_order": fields.get("order", 999),
        "metric_role": fields.get("role", "exploratory"),
    }


def apply_stage29_zscore(df: pd.DataFrame, column: str, threshold: float) -> Tuple[pd.DataFrame, Optional[str], int, Optional[str]]:
    """Apply the FearNetwork stage-29 outlier rule and add a final z column."""
    if column not in df.columns:
        return df, None, 0, None
    out = df.copy()
    values = pd.to_numeric(out[column], errors="coerce")
    non_missing = values.notna()
    if non_missing.sum() < 4:
        return out, None, 0, "too_few_nonmissing"

    sd = values.std(skipna=True, ddof=0)
    if pd.isna(sd) or sd == 0:
        return out, None, 0, "constant"

    scores = (values - values.mean(skipna=True)) / sd
    outlier_mask = non_missing & (scores.abs() > threshold)
    out.loc[outlier_mask, column] = pd.NA
    values_clean = pd.to_numeric(out[column], errors="coerce")
    sd_clean = values_clean.std(skipna=True, ddof=0)
    if pd.isna(sd_clean) or sd_clean == 0:
        return out, None, int(outlier_mask.sum()), "constant_after_outlier_removal"
    z_col = f"z_{column}"
    out[z_col] = (values_clean - values_clean.mean(skipna=True)) / sd_clean
    return out, z_col, int(outlier_mask.sum()), "fearnetwork_stage29_zscore"


def zscore_numeric_covariates(df: pd.DataFrame, covariates: List[str], threshold: float) -> Tuple[pd.DataFrame, List[str], Dict[str, int]]:
    """Z-score numeric covariates with the same outlier rule; keep categorical covariates unchanged."""
    out = df.copy()
    model_covariates = []
    removed = {}
    for cov in covariates:
        if cov not in out.columns:
            continue
        numeric = pd.to_numeric(out[cov], errors="coerce")
        if numeric.notna().sum() >= max(4, len(out[cov].dropna()) // 2):
            out, z_cov, n_out, _ = apply_stage29_zscore(out, cov, threshold)
            removed[cov] = n_out
            if z_cov is not None:
                model_covariates.append(z_cov)
        else:
            model_covariates.append(cov)
    return out, model_covariates, removed


def run_aim2(df: pd.DataFrame, feature_space: str, covariates: List[str]) -> pd.DataFrame:
    rows = []
    sub = df[(df["FeatureSpace"] == feature_space) & (df["Drug"] == "Placebo")].copy()
    for question_order, (question_key, metrics) in enumerate(AIM2_QUESTION_METRICS.items(), start=1):
        for metric in metrics:
            if metric not in sub.columns:
                continue
            row = fit_lm(
                sub,
                outcome=metric,
                predictor_terms=["C(Group, Treatment(reference='HC'))"],
                covariates=covariates,
                term_of_interest=GROUP_TERM,
            )
            row.update(
                {
                    "analysis": "Aim2_SAD_HC_Placebo",
                    "aim2_question": question_key,
                    "aim2_question_order": question_order,
                    "aim2_question_label": AIM2_QUESTION_LABELS[question_key],
                    "metric": metric,
                    "metric_role": "secondary" if metric in AIM2_SECONDARY_METRICS else "primary",
                    "feature_space": feature_space,
                }
            )
            row.update(metric_hierarchy_fields(metric))
            rows.append(row)
    return pd.DataFrame(rows)


def run_aim3(df: pd.DataFrame, feature_space: str, covariates: List[str], clinical_outlier_z: float) -> pd.DataFrame:
    rows = []
    sub = df[(df["FeatureSpace"] == feature_space) & (df["Drug"] == "Placebo")].copy()
    groups = [g for g in ["SAD", "HC"] if g in set(sub["Group"].dropna())]
    for group in groups:
        group_df = sub[sub["Group"] == group].copy()
        group_df, model_covariates, covariate_outliers = zscore_numeric_covariates(group_df, covariates, clinical_outlier_z)
        for clinical in ALL_CLINICAL_SCORES:
            clinical_df, clinical_z, n_clinical_outliers, clinical_method = apply_stage29_zscore(group_df, clinical, clinical_outlier_z)
            if clinical_z is None:
                row = {"status": "missing_or_constant_clinical_score", "n": 0, "outcome": clinical}
                row.update(
                    {
                        "analysis": "Aim3_Clinical_Relevance_Groupwise_Placebo_ZOLS",
                        "session": "Placebo",
                        "Group": group,
                        "clinical_score": clinical,
                        "clinical_score_z": None,
                        "clinical_outlier_method": clinical_method,
                        "clinical_outlier_threshold": clinical_outlier_z,
                        "n_clinical_outliers_removed": n_clinical_outliers,
                        "feature_space": feature_space,
                    }
                )
                row.update(clinical_hierarchy_fields(clinical))
                rows.append(row)
                continue
            for metric in CORE_NEURAL_METRICS:
                if metric not in clinical_df.columns:
                    continue
                model_df, metric_z, n_metric_outliers, metric_method = apply_stage29_zscore(clinical_df, metric, clinical_outlier_z)
                if metric_z is None:
                    row = {"status": "missing_or_constant_neural_metric", "n": 0, "outcome": clinical_z}
                    row.update(
                        {
                            "analysis": "Aim3_Clinical_Relevance_Groupwise_Placebo_ZOLS",
                            "session": "Placebo",
                            "Group": group,
                            "metric": metric,
                            "metric_z": None,
                            "clinical_score": clinical,
                            "clinical_score_z": clinical_z,
                            "clinical_outlier_method": clinical_method,
                            "metric_outlier_method": metric_method,
                            "clinical_outlier_threshold": clinical_outlier_z,
                            "n_clinical_outliers_removed": n_clinical_outliers,
                            "n_metric_outliers_removed": n_metric_outliers,
                            "feature_space": feature_space,
                        }
                    )
                    row.update(clinical_hierarchy_fields(clinical))
                    row.update(neural_question_fields(metric))
                    row.update(metric_hierarchy_fields(metric))
                    rows.append(row)
                    continue
                row = fit_lm(
                    model_df,
                    outcome=clinical_z,
                    predictor_terms=[f"Q('{metric_z}')"],
                    covariates=model_covariates,
                    term_of_interest=f"Q('{metric_z}')",
                )
                row.update(
                    {
                        "analysis": "Aim3_Clinical_Relevance_Groupwise_Placebo_ZOLS",
                        "session": "Placebo",
                        "Group": group,
                        "metric": metric,
                        "metric_z": metric_z,
                        "clinical_score": clinical,
                        "clinical_score_z": clinical_z,
                        "clinical_outlier_method": clinical_method,
                        "metric_outlier_method": metric_method,
                        "clinical_outlier_threshold": clinical_outlier_z,
                        "n_clinical_outliers_removed": n_clinical_outliers,
                        "n_metric_outliers_removed": n_metric_outliers,
                        "covariates_used": ",".join(model_covariates),
                        "covariate_outliers_removed": ";".join(f"{k}:{v}" for k, v in covariate_outliers.items()),
                        "feature_space": feature_space,
                    }
                )
                row.update(clinical_hierarchy_fields(clinical))
                row.update(neural_question_fields(metric))
                row.update(metric_hierarchy_fields(metric))
                rows.append(row)
    return pd.DataFrame(rows)


def run_aim4(df: pd.DataFrame, feature_space: str, covariates: List[str], outlier_z: float) -> pd.DataFrame:
    rows = []
    sub = df[(df["FeatureSpace"] == feature_space) & (df["Drug"] == "Placebo")].copy()
    groups = [g for g in ["SAD", "HC"] if g in set(sub["Group"].dropna())]
    for group in groups:
        group_df = sub[sub["Group"] == group].copy()
        for scr in ALL_SCR_INDICES:
            scr_df, scr_z, n_scr_outliers, scr_method = apply_stage29_zscore(group_df, scr, outlier_z)
            if scr_z is None:
                row = {"status": "missing_or_constant_scr_index", "n": 0, "outcome": scr}
                row.update(
                    {
                        "analysis": "Aim4_SCR_Convergence_Groupwise_Placebo",
                        "session": "Placebo",
                        "Group": group,
                        "scr_index": scr,
                        "scr_index_z": None,
                        "scr_index_role": "primary" if scr in ALL_SCR_INDICES[:2] else "secondary",
                        "scr_outlier_method": scr_method,
                        "outlier_threshold": outlier_z,
                        "n_scr_outliers_removed": n_scr_outliers,
                        "feature_space": feature_space,
                    }
                )
                rows.append(row)
                continue
            for metric in CORE_NEURAL_METRICS:
                if metric not in scr_df.columns:
                    continue
                model_df, metric_z, n_metric_outliers, metric_method = apply_stage29_zscore(scr_df, metric, outlier_z)
                if metric_z is None:
                    row = {"status": "missing_or_constant_neural_metric", "n": 0, "outcome": scr_z}
                    row.update(
                        {
                            "analysis": "Aim4_SCR_Convergence_Groupwise_Placebo",
                            "session": "Placebo",
                            "Group": group,
                            "metric": metric,
                            "metric_z": None,
                            "scr_index": scr,
                            "scr_index_z": scr_z,
                            "scr_index_role": "primary" if scr in ALL_SCR_INDICES[:2] else "secondary",
                            "scr_outlier_method": scr_method,
                            "metric_outlier_method": metric_method,
                            "outlier_threshold": outlier_z,
                            "n_scr_outliers_removed": n_scr_outliers,
                            "n_metric_outliers_removed": n_metric_outliers,
                            "feature_space": feature_space,
                        }
                    )
                    row.update(neural_question_fields(metric))
                    row.update(metric_hierarchy_fields(metric))
                    rows.append(row)
                    continue
                row = fit_lm(
                    model_df,
                    outcome=scr_z,
                    predictor_terms=[f"Q('{metric_z}')"],
                    covariates=covariates,
                    term_of_interest=f"Q('{metric_z}')",
                )
                row.update(
                    {
                        "analysis": "Aim4_SCR_Convergence_Groupwise_Placebo",
                        "session": "Placebo",
                        "Group": group,
                        "metric": metric,
                        "metric_z": metric_z,
                        "scr_index": scr,
                        "scr_index_z": scr_z,
                        "scr_index_role": "primary" if scr in ALL_SCR_INDICES[:2] else "secondary",
                        "scr_outlier_method": scr_method,
                        "metric_outlier_method": metric_method,
                        "outlier_threshold": outlier_z,
                        "n_scr_outliers_removed": n_scr_outliers,
                        "n_metric_outliers_removed": n_metric_outliers,
                        "feature_space": feature_space,
                    }
                )
                row.update(neural_question_fields(metric))
                row.update(metric_hierarchy_fields(metric))
                rows.append(row)
    return pd.DataFrame(rows)


def run_aim5(df: pd.DataFrame, feature_space: str, covariates: List[str]) -> pd.DataFrame:
    rows = []
    sub = df[df["FeatureSpace"] == feature_space].copy()
    expected_cells = [("HC", "Placebo"), ("HC", "Oxytocin"), ("SAD", "Placebo"), ("SAD", "Oxytocin")]

    def _cell_counts(frame: pd.DataFrame) -> Dict[str, int]:
        if not {"Group", "Drug"}.issubset(frame.columns):
            return {}
        counts = frame.groupby(["Group", "Drug"], dropna=False).size().to_dict()
        return {f"{group}_{drug}": int(counts.get((group, drug), 0)) for group, drug in expected_cells}

    for metric in CORE_NEURAL_METRICS + COMPANION_NEURAL_METRICS:
        needed = [metric, "Group", "Drug"] + [cov for cov in covariates if cov in sub.columns]
        if metric not in sub.columns:
            row = {"status": "missing_outcome", "n": 0, "outcome": metric}
            row.update({"analysis": "Aim5_Group_x_Drug", "metric": metric, "feature_space": feature_space})
            row.update(neural_question_fields(metric))
            row.update(metric_hierarchy_fields(metric))
            rows.append(row)
            continue

        metric_df = sub.dropna(subset=[metric, "Group", "Drug"]).copy()
        complete_df = sub.dropna(subset=needed).copy()
        metric_counts = _cell_counts(metric_df)
        complete_counts = _cell_counts(complete_df)
        missing_complete_cells = [cell for cell, n in complete_counts.items() if n == 0]

        if missing_complete_cells:
            row = {
                "status": "missing_group_drug_complete_cases",
                "outcome": metric,
                "term": DRUG_INTERACTION_TERM,
                "n": int(len(complete_df)),
                "metric_cell_counts": ";".join(f"{cell}:{metric_counts.get(cell, 0)}" for cell in complete_counts),
                "complete_case_cell_counts": ";".join(f"{cell}:{complete_counts.get(cell, 0)}" for cell in complete_counts),
                "missing_complete_cells": ";".join(missing_complete_cells),
                "covariates_used": ",".join([cov for cov in covariates if cov in sub.columns]),
                "formula": (
                    f"Q('{metric}') ~ "
                    "C(Group, Treatment(reference='HC')) * C(Drug, Treatment(reference='Placebo'))"
                    + (
                        " + "
                        + " + ".join(
                            f"Q('{cov}')" if pd.to_numeric(sub[cov], errors="coerce").notna().sum() >= max(3, len(sub[cov].dropna()) // 2)
                            else f"C(Q('{cov}'))"
                            for cov in covariates
                            if cov in sub.columns
                        )
                        if covariates
                        else ""
                    )
                ),
            }
            row.update({"analysis": "Aim5_Group_x_Drug", "metric": metric, "feature_space": feature_space})
            row.update(neural_question_fields(metric))
            row.update(metric_hierarchy_fields(metric))
            rows.append(row)
            continue

        row = fit_lm(
            complete_df,
            outcome=metric,
            predictor_terms=["C(Group, Treatment(reference='HC')) * C(Drug, Treatment(reference='Placebo'))"],
            covariates=covariates,
            term_of_interest=DRUG_INTERACTION_TERM,
        )
        row.update(
            {
                "analysis": "Aim5_Group_x_Drug",
                "metric": metric,
                "feature_space": feature_space,
                "metric_cell_counts": ";".join(f"{cell}:{metric_counts.get(cell, 0)}" for cell in complete_counts),
                "complete_case_cell_counts": ";".join(f"{cell}:{complete_counts.get(cell, 0)}" for cell in complete_counts),
                "covariates_used": ",".join([cov for cov in covariates if cov in sub.columns]),
            }
        )
        row.update(neural_question_fields(metric))
        row.update(metric_hierarchy_fields(metric))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/mvpa_l2/harmonized/mvpa_l2_subject_metrics.csv"),
    )
    parser.add_argument("--primary-feature-space", default="FearNetwork")
    parser.add_argument("--covariates", nargs="*", default=None)
    parser.add_argument("--clinical-outlier-z", type=float, default=3.0)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/mvpa_l2/stats"))
    args = parser.parse_args()

    df = derive_final_metrics(harmonize_group_drug(pd.read_csv(args.input)))
    covariates = available_covariates(df, args.covariates)
    print(f"Using covariates: {covariates}")

    results = {
        "aim2_group_difference": run_aim2(df, args.primary_feature_space, covariates),
        "aim3_clinical_relevance": run_aim3(df, args.primary_feature_space, covariates, args.clinical_outlier_z),
        "aim4_scr_convergence": run_aim4(df, args.primary_feature_space, covariates, args.clinical_outlier_z),
        "aim5_oxytocin_modulation": run_aim5(df, args.primary_feature_space, covariates),
    }

    all_rows = []
    for name, table in results.items():
        if name == "aim2_group_difference" and "aim2_question" in table.columns:
            table = add_fdr(table, family_cols=["analysis", "aim2_question"])
            table = table.rename(columns={"q": "q_within_question"})
            table["q"] = table["q_within_question"]
            table["correction_family"] = table["aim2_question"].astype(str)
        elif name == "aim3_clinical_relevance" and {"Group", "aim2_question"}.issubset(table.columns):
            table = add_fdr(table, family_cols=["analysis", "Group", "aim2_question"])
            table = table.rename(columns={"q": "q_within_question"})
            table["q"] = table["q_within_question"]
            table["correction_family"] = table["Group"].astype(str) + " | " + table["aim2_question"].astype(str)
            if "clinical_score" in table.columns:
                clinical_table = add_fdr(table.copy(), family_cols=["analysis", "Group", "clinical_score"])
                table["q_within_group_clinical_score"] = clinical_table["q"]
                table["aim3_clinical_score_family"] = (
                    table["Group"].astype(str) + " | " + table["clinical_score"].astype(str)
                )
        elif name == "aim4_scr_convergence" and {"Group", "aim2_question"}.issubset(table.columns):
            table = add_fdr(table, family_cols=["analysis", "Group", "aim2_question"])
            table = table.rename(columns={"q": "q_within_question"})
            table["q"] = table["q_within_question"]
            table["correction_family"] = table["Group"].astype(str) + " | " + table["aim2_question"].astype(str)
        elif name == "aim5_oxytocin_modulation" and {"metric_role", "aim2_question"}.issubset(table.columns):
            table = add_fdr(table, family_cols=["analysis", "metric_role", "aim2_question"])
            table = table.rename(columns={"q": "q_within_question"})
            table["q"] = table["q_within_question"]
            table["correction_family"] = table["metric_role"].astype(str) + " | " + table["aim2_question"].astype(str)
        else:
            table = add_fdr(table, family_cols=["analysis"])
        write_csv(table, args.out_dir / f"{name}.csv")
        print(f"Wrote {name}: {len(table)} rows")
        all_rows.append(table)

    combined = pd.concat(all_rows, ignore_index=True, sort=False)
    write_csv(combined, args.out_dir / "primary_models_all.csv")
    print(f"Wrote combined primary model table -> {args.out_dir / 'primary_models_all.csv'}")


if __name__ == "__main__":
    main()
