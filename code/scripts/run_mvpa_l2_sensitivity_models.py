#!/usr/bin/env python3
"""Run feature-space and SCR-cohort sensitivity models for MVPA L2."""

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from mvpa_l2_common import (
    ALL_SCR_INDICES,
    CORE_NEURAL_METRICS,
    SCR_SENSITIVITY_FLAGS,
    add_fdr,
    available_covariates,
    derive_final_metrics,
    fit_lm,
    harmonize_group_drug,
    write_csv,
)


GROUP_TERM = "C(Group, Treatment(reference='HC'))[T.SAD]"
DRUG_INTERACTION_TERM = "C(Group, Treatment(reference='HC'))[T.SAD]:C(Drug, Treatment(reference='Placebo'))[T.Oxytocin]"


def run_group_model(df: pd.DataFrame, label: str, feature_space: str, covariates: List[str]) -> List[Dict]:
    rows = []
    sub = df[(df["FeatureSpace"] == feature_space) & (df["Drug"] == "Placebo")].copy()
    for metric in CORE_NEURAL_METRICS:
        row = fit_lm(
            sub,
            outcome=metric,
            predictor_terms=["C(Group, Treatment(reference='HC'))"],
            covariates=covariates,
            term_of_interest=GROUP_TERM,
            min_n=10,
        )
        row.update({"analysis": "Sensitivity_Aim2_Group", "sensitivity": label, "metric": metric, "feature_space": feature_space})
        rows.append(row)
    return rows


def run_scr_model(df: pd.DataFrame, label: str, feature_space: str, covariates: List[str]) -> List[Dict]:
    rows = []
    sub = df[df["FeatureSpace"] == feature_space].copy()
    for scr in ALL_SCR_INDICES:
        for metric in CORE_NEURAL_METRICS:
            row = fit_lm(
                sub,
                outcome=scr,
                predictor_terms=[f"Q('{metric}')", "C(Group, Treatment(reference='HC'))", "C(Drug, Treatment(reference='Placebo'))"],
                covariates=covariates,
                term_of_interest=f"Q('{metric}')",
                min_n=10,
            )
            row.update({
                "analysis": "Sensitivity_Aim4_SCR",
                "sensitivity": label,
                "metric": metric,
                "scr_index": scr,
                "scr_index_role": "primary" if scr in ALL_SCR_INDICES[:2] else "secondary",
                "feature_space": feature_space,
            })
            rows.append(row)
    return rows


def run_drug_model(df: pd.DataFrame, label: str, feature_space: str, covariates: List[str]) -> List[Dict]:
    rows = []
    sub = df[df["FeatureSpace"] == feature_space].copy()
    for metric in CORE_NEURAL_METRICS:
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
    parser.add_argument("--out", type=Path, default=Path("outputs/mvpa_l2/stats/sensitivity_models_all.csv"))
    args = parser.parse_args()

    df = derive_final_metrics(harmonize_group_drug(pd.read_csv(args.input)))
    covariates = available_covariates(df, args.covariates)
    rows = []

    feature_spaces = sorted([fs for fs in df["FeatureSpace"].dropna().unique() if fs != args.primary_feature_space])
    for feature_space in feature_spaces:
        label = f"FeatureSpace:{feature_space}"
        rows.extend(run_group_model(df, label, feature_space, covariates))
        rows.extend(run_scr_model(df, label, feature_space, covariates))
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
        rows.extend(run_group_model(sub, label, args.primary_feature_space, covariates))
        rows.extend(run_scr_model(sub, label, args.primary_feature_space, covariates))
        if cell_count_ok(sub, args.min_cell_n):
            rows.extend(run_drug_model(sub, label, args.primary_feature_space, covariates))
        else:
            print(f"[INFO] Skipping Group*Drug sensitivity for {flag}; cell counts below {args.min_cell_n}.")

    out = pd.DataFrame(rows)
    if not out.empty:
        out = add_fdr(out, family_cols=["analysis", "sensitivity"])
    write_csv(out, args.out)
    print(f"Wrote {len(out)} sensitivity rows -> {args.out}")


if __name__ == "__main__":
    main()
