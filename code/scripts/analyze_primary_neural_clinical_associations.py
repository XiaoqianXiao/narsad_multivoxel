#!/usr/bin/env python3
"""Test clinical associations for current primary representative neural metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mvpa_l2_common import (  # noqa: E402
    ALL_CLINICAL_SCORES,
    CLINICAL_SCORE_HIERARCHY,
    CORE_NEURAL_METRICS,
    NEURAL_METRIC_HIERARCHY,
    add_fdr,
    fit_lm,
    harmonize_group_drug,
    normalize_subject_id,
)
from run_mvpa_l2_primary_models import (  # noqa: E402
    AIM2_METRIC_TO_QUESTION,
    AIM2_QUESTION_LABELS,
    apply_stage29_zscore,
    zscore_numeric_covariates,
)
from explore_representative_neural_index import load_npz_table  # noqa: E402


DASS_DEPRESSION_ITEMS = [
    "dass_q3_positive",
    "dass_q5_initiative",
    "dass_q10_forward",
    "dass_q13_blue",
    "das_q16_enthusiastic",
    "dass_q17_worth",
    "dass_q21_life",
]
DASS_ANXIETY_ITEMS = [
    "dass_q2_drymouth",
    "dass_q4_breathing",
    "dass_q7_trembling",
    "dass_q9_panic",
    "dass_q15_panic",
    "dass_q19_heart",
    "dass_q20_scared",
]
DASS_STRESS_ITEMS = [
    "dass_q1_winddown",
    "dass_q6_overreact",
    "dass_q8_nervousenergy",
    "dass_q11_agitated",
    "dass_q12_relax",
    "dass_q14_intolerant",
    "dass_q18_touch",
]


def clinical_hierarchy_fields(clinical_score: str) -> Dict[str, object]:
    fields = CLINICAL_SCORE_HIERARCHY.get(clinical_score, {})
    return {
        "clinical_score_order": fields.get("order", 999),
        "clinical_score_role": fields.get("role", "exploratory"),
        "clinical_score_family": fields.get("family", "exploratory"),
        "clinical_score_label": fields.get("label", clinical_score),
    }


def neural_fields(metric: str) -> Dict[str, object]:
    question = AIM2_METRIC_TO_QUESTION.get(metric)
    fields = NEURAL_METRIC_HIERARCHY.get(metric, {})
    return {
        "metric_order": fields.get("order", 999),
        "metric_role": fields.get("role", "exploratory"),
        "aim2_question": question,
        "aim2_question_label": AIM2_QUESTION_LABELS.get(question),
    }


def read_clinical_scores(behav_dir: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    paths = {
        "LSAS": behav_dir / "SocialSafetyLearning-LSASSubtotals_DATA_2026-04-25_2306.csv",
        "ECR": behav_dir / "SocialSafetyLearning-ECR_DATA_2026-04-25_2306.csv",
        "DASS": behav_dir / "SocialSafetyLearning-DASS_DATA_2026-04-25_2306.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing clinical source CSV(s): " + ", ".join(missing))

    lsas_raw = pd.read_csv(paths["LSAS"])
    lsas = pd.DataFrame(
        {
            "sub_ID": lsas_raw["login_participantid"].map(normalize_subject_id),
            "lsas_fear": lsas_raw["lsas_fear_total"],
            "lsas_avoid": lsas_raw["lsas_avoid_total"],
            "lsas_total": lsas_raw["lsas_total"],
        }
    )

    ecr_raw = pd.read_csv(paths["ECR"])
    ecr = pd.DataFrame(
        {
            "sub_ID": ecr_raw["login_participantid"].map(normalize_subject_id),
            "ecr_total": ecr_raw["ecr_total"],
        }
    )

    dass_raw = pd.read_csv(paths["DASS"])
    dass = pd.DataFrame({"sub_ID": dass_raw["login_participantid"].map(normalize_subject_id)})
    dass["dass_depression"] = dass_raw[DASS_DEPRESSION_ITEMS].sum(axis=1) * 2
    dass["dass_anxiety"] = dass_raw[DASS_ANXIETY_ITEMS].sum(axis=1) * 2
    dass["dass_stress"] = dass_raw[DASS_STRESS_ITEMS].sum(axis=1) * 2

    clinical = dass.merge(lsas, on="sub_ID", how="inner").merge(ecr, on="sub_ID", how="inner")
    return clinical, {key: str(path) for key, path in paths.items()}


def load_analysis_table(
    derived_path: Path,
    behav_dir: Path,
    phase: str,
    feature_space: str,
    npz_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    derived = harmonize_group_drug(pd.read_csv(derived_path))
    source_note = f"started from {derived_path}"
    derived["sub_ID"] = derived["sub_ID"].map(normalize_subject_id)
    derived = derived[(derived["phase"].eq(phase)) & (derived["feature_space"].eq(feature_space))].copy()
    missing_core = [metric for metric in CORE_NEURAL_METRICS if metric not in derived.columns]
    if missing_core:
        npz_name = {
            "phase2_ext_roi": "phase2_X_ext_y_ext_roi_voxels.npz",
            "phase2_ext_memory_fear_network": "phase2_X_ext_y_ext_roi_voxels_MemoryFearNetwork.npz",
            "phase2_ext_schaefer_tian": "phase2_X_ext_y_ext_voxels_schaefer_tian.npz",
            "phase3_reinst_roi": "phase3_X_reinst_y_reinst_roi_voxels.npz",
            "phase3_reinst_memory_fear_network": "phase3_X_reinst_y_reinst_roi_voxels_MemoryFearNetwork.npz",
            "phase3_reinst_schaefer_tian": "phase3_X_reinst_y_reinst_voxels_schaefer_tian.npz",
        }.get(feature_space)
        if npz_name is None:
            raise ValueError(f"Cannot infer NPZ filename for feature space: {feature_space}")
        npz_path = npz_dir / npz_name
        current = load_npz_table(npz_path, feature_space)
        current = current[current["phase"].eq(phase)].copy()
        current["sub_ID"] = current["sub_ID"].map(normalize_subject_id)
        meta_cols = [
            col
            for col in ["sub_ID", "phase", "feature_space", "Group", "Drug", "drug_condition", "Gender", "gender_code", "demo_age", "guess"]
            if col in derived.columns
        ]
        derived = current.merge(
            derived[meta_cols].drop_duplicates(["sub_ID", "phase", "feature_space"]),
            on=["sub_ID", "phase", "feature_space"],
            how="left",
        )
        source_note = (
            f"recomputed current primary metrics from {npz_path} because "
            f"{derived_path} was missing {', '.join(missing_core)}; reused metadata from {derived_path}"
        )
    derived["FeatureSpace"] = "FearNetwork" if feature_space == "phase2_ext_roi" else feature_space

    clinical, clinical_paths = read_clinical_scores(behav_dir)
    merged = derived.merge(clinical, on="sub_ID", how="inner")
    clinical_paths["NeuralMetrics"] = source_note
    return harmonize_group_drug(merged), clinical_paths


def run_groupwise_clinical_models(
    df: pd.DataFrame,
    metrics: Iterable[str],
    clinical_scores: Iterable[str],
    covariates: List[str],
    outlier_z: float,
) -> pd.DataFrame:
    rows = []
    sub = df[df["Drug"].eq("Placebo")].copy()
    for group in [g for g in ["SAD", "HC"] if g in set(sub["Group"].dropna())]:
        group_df = sub[sub["Group"].eq(group)].copy()
        group_df, model_covariates, covariate_outliers = zscore_numeric_covariates(group_df, covariates, outlier_z)
        for clinical in clinical_scores:
            clinical_df, clinical_z, n_clinical_outliers, clinical_method = apply_stage29_zscore(
                group_df,
                clinical,
                outlier_z,
            )
            if clinical_z is None:
                row = {
                    "status": "missing_or_constant_clinical_score",
                    "n": 0,
                    "outcome": clinical,
                    "Group": group,
                    "clinical_score": clinical,
                    "clinical_score_z": None,
                    "clinical_outlier_method": clinical_method,
                    "clinical_outlier_threshold": outlier_z,
                    "n_clinical_outliers_removed": n_clinical_outliers,
                }
                row.update(clinical_hierarchy_fields(clinical))
                rows.append(row)
                continue

            for metric in metrics:
                model_df, metric_z, n_metric_outliers, metric_method = apply_stage29_zscore(
                    clinical_df,
                    metric,
                    outlier_z,
                )
                if metric_z is None:
                    row = {
                        "status": "missing_or_constant_neural_metric",
                        "n": 0,
                        "outcome": clinical_z,
                        "Group": group,
                        "metric": metric,
                        "metric_z": None,
                        "clinical_score": clinical,
                        "clinical_score_z": clinical_z,
                        "clinical_outlier_method": clinical_method,
                        "metric_outlier_method": metric_method,
                        "clinical_outlier_threshold": outlier_z,
                        "n_clinical_outliers_removed": n_clinical_outliers,
                        "n_metric_outliers_removed": n_metric_outliers,
                    }
                    row.update(clinical_hierarchy_fields(clinical))
                    row.update(neural_fields(metric))
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
                        "analysis": "PrimaryRepresentativeNeural_Aim3_Groupwise_Placebo_ZOLS",
                        "session": "Placebo",
                        "Group": group,
                        "metric": metric,
                        "metric_z": metric_z,
                        "clinical_score": clinical,
                        "clinical_score_z": clinical_z,
                        "clinical_outlier_method": clinical_method,
                        "metric_outlier_method": metric_method,
                        "clinical_outlier_threshold": outlier_z,
                        "n_clinical_outliers_removed": n_clinical_outliers,
                        "n_metric_outliers_removed": n_metric_outliers,
                        "covariates_used": ",".join(model_covariates),
                        "covariate_outliers_removed": ";".join(f"{k}:{v}" for k, v in covariate_outliers.items()),
                    }
                )
                row.update(clinical_hierarchy_fields(clinical))
                row.update(neural_fields(metric))
                rows.append(row)
    return pd.DataFrame(rows)


def add_primary_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["is_primary_clinical_score"] = out["clinical_score_role"].eq("primary")
    out["is_primary_neural_metric"] = out["metric_role"].eq("primary")
    return out


def write_summary_markdown(path: Path, results: pd.DataFrame, sources: Dict[str, str], model_n: int) -> None:
    primary = results[
        results["status"].eq("ok")
        & results["clinical_score_role"].eq("primary")
        & results["metric_role"].eq("primary")
    ].copy()
    primary = primary.sort_values(["Group", "clinical_score_order", "metric_order"])

    lines = [
        "# Primary Neural Metric Clinical Associations",
        "",
        f"Model table rows: {len(results)}. Merged subject rows before placebo/group filtering: {model_n}.",
        "",
        "## Sources",
        "",
        f"- Neural metrics: {sources['NeuralMetrics']}.",
        f"- DASS: `{sources['DASS']}`.",
        f"- LSAS: `{sources['LSAS']}`.",
        f"- ECR: `{sources['ECR']}`.",
        "",
        "## Primary Clinical Outcomes",
        "",
    ]
    if primary.empty:
        lines.append("_No primary rows were estimable._")
    else:
        display = primary[
            [
                "Group",
                "clinical_score",
                "metric",
                "n",
                "estimate",
                "ci_low",
                "ci_high",
                "t",
                "p",
                "q_within_group_clinical_score",
                "r2",
                "covariates_used",
            ]
        ].copy()
        for col in ["estimate", "ci_low", "ci_high", "t", "p", "q_within_group_clinical_score", "r2"]:
            display[col] = pd.to_numeric(display[col], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
        lines.append(display.to_markdown(index=False))
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--derived",
        type=Path,
        default=Path("results/representative_neural_index/derived_subject_neural_indices.csv"),
    )
    parser.add_argument(
        "--behav-dir",
        type=Path,
        default=Path("/Users/xiaoqianxiao/projects/NARSAD/MRI/source_data/behav"),
    )
    parser.add_argument(
        "--npz-dir",
        type=Path,
        default=Path(
            "/Users/xiaoqianxiao/projects/NARSAD/MRI/derivatives/fMRI_analysis/"
            "LSS/firstLevel/all_subjects/group_level"
        ),
    )
    parser.add_argument("--phase", default="phase2_extinction")
    parser.add_argument("--feature-space", default="phase2_ext_roi")
    parser.add_argument("--covariates", nargs="*", default=["demo_age", "Gender"])
    parser.add_argument("--outlier-z", type=float, default=3.0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/representative_neural_index/clinical_associations"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df, clinical_paths = load_analysis_table(args.derived, args.behav_dir, args.phase, args.feature_space, args.npz_dir)
    missing_metrics = [metric for metric in CORE_NEURAL_METRICS if metric not in df.columns]
    if missing_metrics:
        raise ValueError(f"Missing primary neural metrics in derived table: {missing_metrics}")
    covariates = [cov for cov in args.covariates if cov in df.columns]

    results = run_groupwise_clinical_models(
        df,
        metrics=CORE_NEURAL_METRICS,
        clinical_scores=ALL_CLINICAL_SCORES,
        covariates=covariates,
        outlier_z=args.outlier_z,
    )
    results = add_primary_flags(results)
    results = add_fdr(results, family_cols=["analysis", "Group", "aim2_question"]).rename(
        columns={"q": "q_within_group_question"}
    )
    by_clinical = add_fdr(results.copy(), family_cols=["analysis", "Group", "clinical_score"]).rename(
        columns={"q": "q_within_group_clinical_score"}
    )
    if "q_within_group_clinical_score" in by_clinical.columns:
        results["q_within_group_clinical_score"] = by_clinical["q_within_group_clinical_score"]

    model_path = args.out_dir / "primary_neural_clinical_associations.csv"
    master_path = args.out_dir / "primary_neural_clinical_master.csv"
    summary_path = args.out_dir / "primary_neural_clinical_associations.md"
    results.to_csv(model_path, index=False)
    df.to_csv(master_path, index=False)
    write_summary_markdown(summary_path, results, clinical_paths, model_n=len(df))

    print(f"Merged rows: {len(df)}")
    print(f"Covariates used: {covariates}")
    print(f"Wrote {model_path}")
    print(f"Wrote {master_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
