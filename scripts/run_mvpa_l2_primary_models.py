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


def run_aim3(df: pd.DataFrame, feature_space: str, covariates: list[str]) -> pd.DataFrame:
    rows = []
    sub = df[df["FeatureSpace"] == feature_space].copy()
    for clinical in PRIMARY_CLINICAL_SCORES:
        for metric in CORE_NEURAL_METRICS:
            if metric not in sub.columns:
                continue
            row = fit_lm(
                sub,
                outcome=clinical,
                predictor_terms=[f"Q('{metric}')", "C(Group, Treatment(reference='HC'))", "C(Drug, Treatment(reference='Placebo'))"],
                covariates=covariates,
                term_of_interest=f"Q('{metric}')",
            )
            row.update({"analysis": "Aim3_Clinical_Relevance", "metric": metric, "clinical_score": clinical, "feature_space": feature_space})
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
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/mvpa_l2/stats"))
    args = parser.parse_args()

    df = harmonize_group_drug(pd.read_csv(args.input))
    covariates = available_covariates(df, args.covariates)
    print(f"Using covariates: {covariates}")

    results = {
        "aim2_group_difference": run_aim2(df, args.primary_feature_space, covariates),
        "aim3_clinical_relevance": run_aim3(df, args.primary_feature_space, covariates),
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
