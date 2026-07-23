#!/usr/bin/env python3
"""Prepare notebook-facing MVPA L2 presentation tables from post-Hyak stats.

This script does not fit models or calculate inferential statistics. It reads
the CSVs produced by the post-Hyak model/export scripts and writes stable table
names used by ``mvpa_l2.ipynb`` for figure and manuscript rendering.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from mvpa_l2_common import (
    CORE_NEURAL_METRICS,
    PRIMARY_CLINICAL_SCORES,
    PRIMARY_SCR_INDICES,
    write_csv,
)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def first_col(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    return next((name for name in names if name in df.columns), None)


def select_rows(df: pd.DataFrame, column: str, values: Iterable[str]) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return df.copy()
    allowed = {str(value) for value in values}
    return df[df[column].astype("string").isin(allowed)].copy()


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = pd.NA
    return out


def aim2_primary(stats_dir: Path) -> None:
    df = read_csv(stats_dir / "aim2_primary_group_difference.csv")
    if df.empty:
        return
    df = select_rows(df, "metric", CORE_NEURAL_METRICS)
    out = pd.DataFrame(
        {
            "scientific_question": df.get("aim2_question_label"),
            "metric": df.get("metric"),
            "effect_label": "SAD minus HC",
            "n": df.get("n"),
            "estimate": df.get("estimate"),
            "std_error": df.get("std_error"),
            "ci_low": df.get("ci_low"),
            "ci_high": df.get("ci_high"),
            "t": df.get("t"),
            "model_family_p": df.get("p"),
            "model_family_q": df.get("q"),
            "correction_family": df.get("correction_family"),
            "formula": df.get("formula"),
            "status": df.get("status"),
        }
    )
    write_csv(out, stats_dir / "Table2_Aim2_primary_statistics.csv")


def aim2_secondary(stats_dir: Path) -> None:
    df = read_csv(stats_dir / "aim2_secondary_group_difference.csv")
    if df.empty:
        return
    out = pd.DataFrame(
        {
            "Metric": df.get("metric"),
            "Raw_metric": df.get("metric"),
            "Domain": df.get("aim2_question_label"),
            "N": df.get("n"),
            "Mean_difference_SAD_minus_HC": df.get("estimate"),
            "Effect_size": df.get("estimate"),
            "CI_lower": df.get("ci_low"),
            "CI_upper": df.get("ci_high"),
            "Test_statistic": df.get("t"),
            "p_value": df.get("p"),
            "q_value": df.get("q"),
            "Interpretation": df.get("status"),
        }
    )
    write_csv(out, stats_dir / "Table_S2_secondary_support.csv")


def aim3_tables(stats_dir: Path) -> None:
    primary = read_csv(stats_dir / "aim3_primary_clinical_relevance.csv")
    if not primary.empty:
        primary = select_rows(primary, "clinical_score", PRIMARY_CLINICAL_SCORES)
        out = pd.DataFrame(
            {
                "group": primary.get("Group"),
                "clinical_outcome": primary.get("clinical_score"),
                "neural_metric": primary.get("metric"),
                "beta_standardized": primary.get("estimate"),
                "ci_low": primary.get("ci_low"),
                "ci_high": primary.get("ci_high"),
                "t": primary.get("t"),
                "model_family_p": primary.get("p"),
                "model_family_q": primary.get("q"),
                "n": primary.get("n"),
                "covariates": primary.get("covariates_used"),
            }
        )
        write_csv(out, stats_dir / "Table_Aim3_primary_statistics.csv")

    secondary = read_csv(stats_dir / "aim3_secondary_clinical_relevance.csv")
    if not secondary.empty:
        out = pd.DataFrame(
            {
                "group": secondary.get("Group"),
                "clinical_outcome": secondary.get("clinical_score"),
                "neural_metric": secondary.get("metric"),
                "beta_standardized": secondary.get("estimate"),
                "ci_low": secondary.get("ci_low"),
                "ci_high": secondary.get("ci_high"),
                "t": secondary.get("t"),
                "p": secondary.get("p"),
                "q_fdr": secondary.get("q"),
                "n": secondary.get("n"),
                "covariates": secondary.get("covariates_used"),
            }
        )
        write_csv(out, stats_dir / "Table_Aim3_secondary_statistics.csv")

    sensitivity = read_csv(stats_dir / "aim3_sensitivity_clinical_relevance.csv")
    if not sensitivity.empty:
        out = pd.DataFrame(
            {
                "sensitivity_type": sensitivity.get("sensitivity"),
                "sensitivity_spec": sensitivity.get("sensitivity"),
                "feature_space": sensitivity.get("feature_space"),
                "group": sensitivity.get("Group"),
                "clinical_outcome": sensitivity.get("clinical_score"),
                "neural_metric": sensitivity.get("metric"),
                "beta_standardized": sensitivity.get("estimate"),
                "ci_low": sensitivity.get("ci_low"),
                "ci_high": sensitivity.get("ci_high"),
                "t": sensitivity.get("t"),
                "p": sensitivity.get("p"),
                "q_fdr_within_spec": sensitivity.get("q"),
                "n": sensitivity.get("n"),
                "covariates": sensitivity.get("covariates_used"),
                "model_formula": sensitivity.get("formula"),
                "notes": sensitivity.get("status"),
            }
        )
        write_csv(out, stats_dir / "Table_Aim3_sensitivity_statistics.csv")


def aim4_tables(stats_dir: Path) -> None:
    primary = read_csv(stats_dir / "aim4_primary_scr_convergence.csv")
    if not primary.empty:
        primary = select_rows(primary, "scr_index", PRIMARY_SCR_INDICES)
        out = pd.DataFrame(
            {
                "Group": primary.get("Group"),
                "SCR_outcome": primary.get("scr_index"),
                "Neural_predictor": primary.get("metric"),
                "beta_std": primary.get("estimate"),
                "ci_low": primary.get("ci_low"),
                "ci_high": primary.get("ci_high"),
                "t": primary.get("t"),
                "model_family_p": primary.get("p"),
                "model_family_q": primary.get("q"),
                "N": primary.get("n"),
                "n_scr_outliers_removed": primary.get("n_scr_outliers_removed"),
                "n_neural_outliers_removed": primary.get("n_metric_outliers_removed"),
                "outlier_rule": primary.get("outlier_threshold"),
            }
        )
        write_csv(out, stats_dir / "Table4_Aim4_primary_neural_SCR_convergence.csv")

    secondary = read_csv(stats_dir / "aim4_secondary_scr_convergence.csv")
    if not secondary.empty:
        out = pd.DataFrame(
            {
                "Group": secondary.get("Group"),
                "SCR_outcome": secondary.get("scr_index"),
                "Neural_predictor": secondary.get("metric"),
                "beta_std": secondary.get("estimate"),
                "ci_low": secondary.get("ci_low"),
                "ci_high": secondary.get("ci_high"),
                "t": secondary.get("t"),
                "p": secondary.get("p"),
                "q_FDR": secondary.get("q"),
                "N": secondary.get("n"),
            }
        )
        write_csv(out, stats_dir / "TableS4_Aim4_secondary_neural_SCR_convergence.csv")

    sensitivity = read_csv(stats_dir / "aim4_sensitivity_scr_convergence.csv")
    if not sensitivity.empty:
        out = pd.DataFrame(
            {
                "Sensitivity_type": sensitivity.get("sensitivity"),
                "Group": sensitivity.get("Group"),
                "Feature_space": sensitivity.get("feature_space"),
                "SCR_outcome": sensitivity.get("scr_index"),
                "Neural_predictor": sensitivity.get("metric"),
                "Model_label": sensitivity.get("analysis"),
                "beta_std": sensitivity.get("estimate"),
                "ci_low": sensitivity.get("ci_low"),
                "ci_high": sensitivity.get("ci_high"),
                "t": sensitivity.get("t"),
                "p": sensitivity.get("p"),
                "q_FDR": sensitivity.get("q"),
                "N": sensitivity.get("n"),
                "n_scr_outliers_removed": sensitivity.get("n_scr_outliers_removed"),
                "n_neural_outliers_removed": sensitivity.get("n_metric_outliers_removed"),
                "outlier_rule": sensitivity.get("outlier_threshold"),
                "Status": sensitivity.get("status"),
            }
        )
        write_csv(out, stats_dir / "TableS5_Aim4_sensitivity_neural_SCR_convergence.csv")


def aim5_tables(stats_dir: Path) -> None:
    for source_name, target_name in [
        ("aim5_primary_oxytocin_modulation.csv", "Table_Aim5_primary_oxytocin_modulation.csv"),
        ("aim5_secondary_oxytocin_modulation.csv", "Table_Aim5_secondary_oxytocin_modulation.csv"),
        ("aim5_sensitivity_oxytocin_modulation.csv", "Table_Aim5_sensitivity_oxytocin_modulation.csv"),
    ]:
        df = read_csv(stats_dir / source_name)
        if df.empty:
            continue
        write_csv(df, stats_dir / target_name)


def write_manifest(stats_dir: Path) -> None:
    rows = []
    for path in sorted(stats_dir.glob("*.csv")):
        rows.append(
            {
                "artifact": path.name,
                "path": str(path),
                "bytes": path.stat().st_size,
                "role": "notebook_input",
            }
        )
    write_csv(pd.DataFrame(rows), stats_dir / "notebook_artifact_manifest.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-dir", type=Path, default=Path("outputs/mvpa_l2/stats"))
    args = parser.parse_args()

    args.stats_dir.mkdir(parents=True, exist_ok=True)
    aim2_primary(args.stats_dir)
    aim2_secondary(args.stats_dir)
    aim3_tables(args.stats_dir)
    aim4_tables(args.stats_dir)
    aim5_tables(args.stats_dir)
    write_manifest(args.stats_dir)
    print(f"Wrote notebook-facing presentation artifacts -> {args.stats_dir}")


if __name__ == "__main__":
    main()
