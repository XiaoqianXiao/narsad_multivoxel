#!/usr/bin/env python3
"""Run primary MVPA L2 models from the harmonized subject table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mvpa_l2_common import (
    CORE_NEURAL_METRICS,
    PRIMARY_CLINICAL_SCORES,
    PRIMARY_SCR_INDICES,
    add_fdr,
    available_covariates,
    fit_lm,
    harmonize_group_drug,
    write_csv,
)


GROUP_TERM = "C(Group, Treatment(reference='HC'))[T.SAD]"
DRUG_INTERACTION_TERM = "C(Group, Treatment(reference='HC'))[T.SAD]:C(Drug, Treatment(reference='Placebo'))[T.Oxytocin]"


def apply_stage29_zscore(df: pd.DataFrame, column: str, threshold: float) -> tuple[pd.DataFrame, str | None, int, str | None]:
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


def zscore_numeric_covariates(df: pd.DataFrame, covariates: list[str], threshold: float) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
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


def run_aim2(df: pd.DataFrame, feature_space: str, covariates: list[str]) -> pd.DataFrame:
    rows = []
    sub = df[(df["FeatureSpace"] == feature_space) & (df["Drug"] == "Placebo")].copy()
    for metric in CORE_NEURAL_METRICS:
        row = fit_lm(
            sub,
            outcome=metric,
            predictor_terms=["C(Group, Treatment(reference='HC'))"],
            covariates=covariates,
            term_of_interest=GROUP_TERM,
        )
        row.update({"analysis": "Aim2_SAD_HC_Placebo", "metric": metric, "feature_space": feature_space})
        rows.append(row)
    return pd.DataFrame(rows)


def run_aim3(df: pd.DataFrame, feature_space: str, covariates: list[str], clinical_outlier_z: float) -> pd.DataFrame:
    rows = []
    sub = df[df["FeatureSpace"] == feature_space].copy()
    groups = [g for g in ["SAD", "HC"] if g in set(sub["Group"].dropna())]
    for group in groups:
        group_df = sub[sub["Group"] == group].copy()
        group_df, model_covariates, covariate_outliers = zscore_numeric_covariates(group_df, covariates, clinical_outlier_z)
        for clinical in PRIMARY_CLINICAL_SCORES:
            clinical_df, clinical_z, n_clinical_outliers, clinical_method = apply_stage29_zscore(group_df, clinical, clinical_outlier_z)
            if clinical_z is None:
                row = {"status": "missing_or_constant_clinical_score", "n": 0, "outcome": clinical}
                row.update(
                    {
                        "analysis": "Aim3_Clinical_Relevance_Groupwise_ZOLS",
                        "Group": group,
                        "clinical_score": clinical,
                        "clinical_score_z": None,
                        "clinical_outlier_method": clinical_method,
                        "clinical_outlier_threshold": clinical_outlier_z,
                        "n_clinical_outliers_removed": n_clinical_outliers,
                        "feature_space": feature_space,
                    }
                )
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
                            "analysis": "Aim3_Clinical_Relevance_Groupwise_ZOLS",
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
                        "analysis": "Aim3_Clinical_Relevance_Groupwise_ZOLS",
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
                rows.append(row)
    return pd.DataFrame(rows)


def run_aim4(df: pd.DataFrame, feature_space: str, covariates: list[str]) -> pd.DataFrame:
    rows = []
    sub = df[df["FeatureSpace"] == feature_space].copy()
    for scr in PRIMARY_SCR_INDICES:
        for metric in CORE_NEURAL_METRICS:
            if metric not in sub.columns:
                continue
            row = fit_lm(
                sub,
                outcome=scr,
                predictor_terms=[f"Q('{metric}')", "C(Group, Treatment(reference='HC'))", "C(Drug, Treatment(reference='Placebo'))"],
                covariates=covariates,
                term_of_interest=f"Q('{metric}')",
            )
            row.update({"analysis": "Aim4_SCR_Convergence", "metric": metric, "scr_index": scr, "feature_space": feature_space})
            rows.append(row)
    return pd.DataFrame(rows)


def run_aim5(df: pd.DataFrame, feature_space: str, covariates: list[str]) -> pd.DataFrame:
    rows = []
    sub = df[df["FeatureSpace"] == feature_space].copy()
    for metric in CORE_NEURAL_METRICS:
        row = fit_lm(
            sub,
            outcome=metric,
            predictor_terms=["C(Group, Treatment(reference='HC')) * C(Drug, Treatment(reference='Placebo'))"],
            covariates=covariates,
            term_of_interest=DRUG_INTERACTION_TERM,
        )
        row.update({"analysis": "Aim5_Group_x_Drug", "metric": metric, "feature_space": feature_space})
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

    df = harmonize_group_drug(pd.read_csv(args.input))
    covariates = available_covariates(df, args.covariates)
    print(f"Using covariates: {covariates}")

    results = {
        "aim2_group_difference": run_aim2(df, args.primary_feature_space, covariates),
        "aim3_clinical_relevance": run_aim3(df, args.primary_feature_space, covariates, args.clinical_outlier_z),
        "aim4_scr_convergence": run_aim4(df, args.primary_feature_space, covariates),
        "aim5_oxytocin_modulation": run_aim5(df, args.primary_feature_space, covariates),
    }

    all_rows = []
    for name, table in results.items():
        table = add_fdr(table, family_cols=["analysis"])
        write_csv(table, args.out_dir / f"{name}.csv")
        print(f"Wrote {name}: {len(table)} rows")
        all_rows.append(table)

    combined = pd.concat(all_rows, ignore_index=True, sort=False)
    combined = add_fdr(combined, family_cols=["analysis"])
    write_csv(combined, args.out_dir / "primary_models_all.csv")
    print(f"Wrote combined primary model table -> {args.out_dir / 'primary_models_all.csv'}")


if __name__ == "__main__":
    main()
