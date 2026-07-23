#!/usr/bin/env python3
"""Run secondary MVPA L2 models from the harmonized subject table."""

import argparse
from pathlib import Path

import pandas as pd

from mvpa_l2_common import (
    SECONDARY_CLINICAL_SCORES,
    SECONDARY_SCR_INDICES,
    add_fdr,
    available_covariates,
    derive_final_metrics,
    harmonize_group_drug,
    write_csv,
)
from run_mvpa_l2_primary_models import (
    AIM2_SECONDARY_METRICS,
    AIM2_SECONDARY_QUESTION_METRICS,
    apply_aim_metric_role_fdr,
    run_aim2,
    run_aim3,
    run_aim4,
    run_aim5,
)


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
        "aim2_secondary_group_difference": run_aim2(
            df,
            args.primary_feature_space,
            covariates,
            AIM2_SECONDARY_QUESTION_METRICS,
            "Aim2_Secondary_SAD_HC_Placebo",
        ),
        "aim3_secondary_clinical_relevance": run_aim3(
            df,
            args.primary_feature_space,
            covariates,
            args.clinical_outlier_z,
            SECONDARY_CLINICAL_SCORES,
            "Aim3_Secondary_Clinical_Relevance_Groupwise_Placebo_ZOLS",
        ),
        "aim4_secondary_scr_convergence": run_aim4(
            df,
            args.primary_feature_space,
            covariates,
            args.clinical_outlier_z,
            SECONDARY_SCR_INDICES,
            "Aim4_Secondary_SCR_Convergence_Groupwise_Placebo",
        ),
        "aim5_secondary_oxytocin_modulation": run_aim5(
            df,
            args.primary_feature_space,
            covariates,
            AIM2_SECONDARY_METRICS,
            "Aim5_Secondary_Group_x_Drug",
        ),
    }

    all_rows = []
    for name, table in results.items():
        if {"metric", "metric_role"}.issubset(table.columns):
            aim_label = name.split("_", 1)[0].capitalize()
            table = apply_aim_metric_role_fdr(table, aim_label)
            if name == "aim3_secondary_clinical_relevance" and "clinical_score" in table.columns:
                clinical_table = add_fdr(table.copy(), family_cols=["analysis", "Group", "clinical_score"])
                table["q_within_group_clinical_score"] = clinical_table["q"]
                table["aim3_clinical_score_family"] = (
                    table["Group"].astype(str) + " | " + table["clinical_score"].astype(str)
                )
        write_csv(table, args.out_dir / f"{name}.csv")
        print(f"Wrote {name}: {len(table)} rows")
        all_rows.append(table)

    combined = pd.concat(all_rows, ignore_index=True, sort=False) if all_rows else pd.DataFrame()
    write_csv(combined, args.out_dir / "aims_secondary_models_all.csv")
    print(f"Wrote combined secondary model table -> {args.out_dir / 'aims_secondary_models_all.csv'}")


if __name__ == "__main__":
    main()
