#!/usr/bin/env python3
"""Export manuscript-facing MVPA L2 tables and reproducibility summaries."""

import argparse
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from mvpa_l2_common import (
    CLINICAL_SCORE_HIERARCHY,
    CORE_NEURAL_METRICS,
    NEURAL_METRIC_HIERARCHY,
    PRIMARY_CLINICAL_SCORES,
    PRIMARY_SCR_INDICES,
    derive_final_metrics,
    write_csv,
)


PRIMARY_TABLE_COLUMNS = [
    "aim",
    "scientific_question",
    "analysis",
    "feature_space",
    "session",
    "Group",
    "clinical_score_label",
    "clinical_score_role",
    "scr_index",
    "metric",
    "metric_role",
    "effect_label",
    "effect_scale",
    "estimate",
    "ci_low",
    "ci_high",
    "p",
    "q",
    "n",
    "status",
]


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    text = df.fillna("").astype(str)
    lines = [
        "| " + " | ".join(text.columns) + " |",
        "| " + " | ".join(["---"] * len(text.columns)) + " |",
    ]
    for row in text.values.tolist():
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def format_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.3g}")
    return out


def clinical_fields(score: object) -> Dict[str, object]:
    info = CLINICAL_SCORE_HIERARCHY.get(str(score), {})
    return {
        "clinical_score_order": info.get("order", 999),
        "clinical_score_role": info.get("role", "exploratory"),
        "clinical_score_family": info.get("family", "exploratory"),
        "clinical_score_label": info.get("label", score),
    }


def metric_fields(metric: object) -> Dict[str, object]:
    info = NEURAL_METRIC_HIERARCHY.get(str(metric), {})
    return {
        "metric_order": info.get("order", 999),
        "metric_role": info.get("role", "exploratory"),
    }


def ensure_hierarchy_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "clinical_score" in out.columns:
        clinical = out["clinical_score"].map(clinical_fields).apply(pd.Series)
        for col in clinical.columns:
            if col not in out.columns:
                out[col] = clinical[col]
    if "metric" in out.columns:
        metric = out["metric"].map(metric_fields).apply(pd.Series)
        for col in metric.columns:
            if col not in out.columns:
                out[col] = metric[col]
    return out


def add_missing_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def primary_rows(stats_dir: Path) -> pd.DataFrame:
    aim2 = ensure_hierarchy_columns(read_csv_if_exists(stats_dir / "aim2_group_difference.csv"))
    aim3 = ensure_hierarchy_columns(read_csv_if_exists(stats_dir / "aim3_clinical_relevance.csv"))
    aim4 = ensure_hierarchy_columns(read_csv_if_exists(stats_dir / "aim4_scr_convergence.csv"))
    aim5 = ensure_hierarchy_columns(read_csv_if_exists(stats_dir / "aim5_oxytocin_modulation.csv"))

    rows = []
    if not aim2.empty:
        sub = aim2[aim2.get("metric_role", pd.Series(index=aim2.index, dtype=object)).eq("primary")].copy()
        sub["aim"] = "Aim 2"
        sub["scientific_question"] = sub.get("aim2_question_label", "SAD-HC neural representation difference")
        sub["session"] = "Placebo"
        sub["effect_label"] = "SAD minus HC"
        sub["effect_scale"] = "raw neural metric units"
        rows.append(sub)

    if not aim3.empty:
        sub = aim3[
            aim3.get("clinical_score", pd.Series(index=aim3.index, dtype=object)).isin(PRIMARY_CLINICAL_SCORES)
            & aim3.get("metric_role", pd.Series(index=aim3.index, dtype=object)).eq("primary")
        ].copy()
        sub["aim"] = "Aim 3"
        sub["scientific_question"] = "Clinical relevance of neural learning profiles"
        sub["effect_label"] = "standardized neural metric slope"
        sub["effect_scale"] = "standardized beta; z clinical outcome, z neural predictor"
        rows.append(sub)

    if not aim4.empty:
        sub = aim4[
            aim4.get("scr_index", pd.Series(index=aim4.index, dtype=object)).isin(PRIMARY_SCR_INDICES)
            & aim4.get("metric_role", pd.Series(index=aim4.index, dtype=object)).eq("primary")
        ].copy()
        sub["aim"] = "Aim 4"
        sub["scientific_question"] = "Neural-SCR convergence"
        sub["effect_label"] = "neural metric slope"
        sub["effect_scale"] = "SCR index units per neural metric unit"
        rows.append(sub)

    if not aim5.empty:
        sub = aim5[aim5.get("metric_role", pd.Series(index=aim5.index, dtype=object)).eq("primary")].copy()
        if sub.empty and "metric" in aim5.columns:
            sub = aim5[aim5["metric"].isin(CORE_NEURAL_METRICS)].copy()
        sub["aim"] = "Aim 5"
        sub["scientific_question"] = "Oxytocin modulation of diagnostic effects"
        sub["effect_label"] = "SAD x oxytocin interaction"
        sub["effect_scale"] = "raw neural metric units"
        rows.append(sub)

    if not rows:
        return pd.DataFrame(columns=PRIMARY_TABLE_COLUMNS)

    out = pd.concat(rows, ignore_index=True, sort=False)
    out = add_missing_columns(out, PRIMARY_TABLE_COLUMNS)
    order_cols = [c for c in ["aim", "clinical_score_order", "Group", "metric_order", "scr_index"] if c in out.columns]
    if order_cols:
        out = out.sort_values(order_cols, na_position="last")
    return out[PRIMARY_TABLE_COLUMNS]


def write_primary_table(stats_dir: Path) -> None:
    table = primary_rows(stats_dir)
    write_csv(table, stats_dir / "manuscript_primary_results.csv")

    preview_cols = [
        "aim",
        "scientific_question",
        "Group",
        "clinical_score_label",
        "scr_index",
        "metric",
        "effect_label",
        "estimate",
        "ci_low",
        "ci_high",
        "q",
        "n",
        "status",
    ]
    preview = table[[col for col in preview_cols if col in table.columns]].copy()
    preview = format_numeric_columns(preview, ["estimate", "ci_low", "ci_high", "q"])
    lines = [
        "# Manuscript Primary Results",
        "",
        "Primary rows are ordered by aim, clinical hierarchy, group, and neural metric hierarchy. Effect estimates are foregrounded with confidence intervals; corrected p-values are retained for reference.",
        "",
        _markdown_table(preview),
        "",
    ]
    (stats_dir / "manuscript_primary_results.md").write_text("\n".join(lines))


def convergence_matrix(stats_dir: Path) -> None:
    aim4 = ensure_hierarchy_columns(read_csv_if_exists(stats_dir / "aim4_scr_convergence.csv"))
    if aim4.empty:
        write_csv(pd.DataFrame(), stats_dir / "aim4_convergence_matrix.csv")
        (stats_dir / "aim4_convergence_matrix.md").write_text("# Aim 4 Convergence Matrix\n\nNo Aim 4 table found.\n")
        return

    sub = aim4[
        aim4.get("metric", pd.Series(index=aim4.index, dtype=object)).isin(CORE_NEURAL_METRICS)
        & aim4.get("scr_index", pd.Series(index=aim4.index, dtype=object)).isin(PRIMARY_SCR_INDICES)
    ].copy()
    sub = add_missing_columns(sub, ["Group", "scr_index", "metric", "estimate", "ci_low", "ci_high", "q", "n", "status"])
    sub = sub.sort_values(["Group", "scr_index", "metric_order"], na_position="last")
    write_csv(sub, stats_dir / "aim4_convergence_matrix.csv")

    matrix = sub.copy()
    matrix["effect_ci_q"] = matrix.apply(format_convergence_cell, axis=1)
    wide = matrix.pivot_table(
        index=["Group", "scr_index"],
        columns="metric",
        values="effect_ci_q",
        aggfunc="first",
    ).reset_index()
    write_csv(wide, stats_dir / "aim4_convergence_matrix_wide.csv")

    lines = [
        "# Aim 4 Neural-SCR Convergence Matrix",
        "",
        "Cells show effect estimate with 95% CI and q-value. This matrix is intended as a convergence/dissociation readout, not a significance-first screen.",
        "",
        _markdown_table(wide),
        "",
    ]
    (stats_dir / "aim4_convergence_matrix.md").write_text("\n".join(lines))


def format_convergence_cell(row: pd.Series) -> str:
    estimate = pd.to_numeric(row.get("estimate"), errors="coerce")
    ci_low = pd.to_numeric(row.get("ci_low"), errors="coerce")
    ci_high = pd.to_numeric(row.get("ci_high"), errors="coerce")
    q = pd.to_numeric(row.get("q"), errors="coerce")
    status = row.get("status", "")
    if pd.isna(estimate):
        return str(status) if status else ""
    ci = "" if pd.isna(ci_low) or pd.isna(ci_high) else f" [{ci_low:.2g}, {ci_high:.2g}]"
    q_text = "" if pd.isna(q) else f"; q={q:.2g}"
    return f"{estimate:.2g}{ci}{q_text}"


def git_value(args: List[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(["git"] + args, cwd=str(cwd), text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def model_status_counts(stats_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(stats_dir.glob("*.csv")):
        if path.name.startswith(("qc_", "manuscript_", "aim4_convergence_matrix")):
            continue
        table = read_csv_if_exists(path)
        if table.empty or "status" not in table.columns:
            continue
        for status, count in table["status"].fillna("missing_status").value_counts().items():
            rows.append({"table": path.name, "status": status, "rows": int(count)})
    return pd.DataFrame(rows)


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    variables = [c for c in CORE_NEURAL_METRICS + PRIMARY_CLINICAL_SCORES + PRIMARY_SCR_INDICES if c in df.columns]
    rows = []
    for feature_space, sub in df.groupby("FeatureSpace", dropna=False):
        for variable in variables:
            values = sub[variable]
            rows.append(
                {
                    "FeatureSpace": feature_space,
                    "variable": variable,
                    "nonmissing_n": int(values.notna().sum()),
                    "missing_n": int(values.isna().sum()),
                    "missing_pct": float(values.isna().mean() * 100) if len(values) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def write_aim2_panel_inputs(input_path: Path, stats_dir: Path) -> None:
    """Write notebook-facing Figure 2 geometry and trajectory input CSVs."""
    df = derive_final_metrics(read_csv_if_exists(input_path))
    if df.empty:
        write_csv(pd.DataFrame(), stats_dir / "aim2_geometry_panel.csv")
        write_csv(pd.DataFrame(), stats_dir / "aim2_trajectory_panel.csv")
        return

    geometry = build_aim2_geometry_panel(stats_dir)
    write_csv(geometry, stats_dir / "aim2_geometry_panel.csv")

    trajectory = build_aim2_trajectory_panel(stats_dir)
    write_csv(trajectory, stats_dir / "aim2_trajectory_panel.csv")


GEOMETRY_PANEL_COLUMNS = ["subject_id", "group", "condition", "safety_alignment", "threat_alignment", "source"]


def build_aim2_geometry_panel(stats_dir: Path) -> pd.DataFrame:
    """Load true centroid geometry if an upstream export exists; otherwise empty."""
    candidate_names = [
        "aim2_geometry_panel.csv",
        "aim2_geometry_centroids.csv",
        "aim2_condition_centroids.csv",
        "aim2_subject_condition_centroids.csv",
        "aim2_true_centroid_geometry.csv",
    ]
    for name in candidate_names:
        path = stats_dir / name
        if not path.exists():
            continue
        panel_data = read_csv_if_exists(path)
        if panel_data.empty:
            continue
        panel = standardize_aim2_geometry_panel(panel_data, path.name)
        if not panel.empty:
            return panel
    return pd.DataFrame(columns=GEOMETRY_PANEL_COLUMNS)


def standardize_aim2_geometry_panel(data: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Standardize a true subject-level centroid/alignment export for Figure 2."""
    subject_col = next((col for col in ["subject_id", "Subject", "sub_ID", "sub"] if col in data.columns), None)
    group_col = next((col for col in ["group", "Group"] if col in data.columns), None)
    condition_col = next((col for col in ["condition", "Condition", "Cue", "cue"] if col in data.columns), None)
    safety_col = next(
        (col for col in ["safety_alignment", "SafetyAlignment", "target_safety_alignment", "TargetSafetyAlignment"] if col in data.columns),
        None,
    )
    threat_col = next(
        (col for col in ["threat_alignment", "ThreatAlignment", "target_threat_alignment", "TargetThreatAlignment"] if col in data.columns),
        None,
    )
    if None in {subject_col, group_col, condition_col, safety_col, threat_col}:
        return pd.DataFrame(columns=GEOMETRY_PANEL_COLUMNS)
    out = data[[subject_col, group_col, condition_col, safety_col, threat_col]].copy()
    out = out.rename(
        columns={
            subject_col: "subject_id",
            group_col: "group",
            condition_col: "condition",
            safety_col: "safety_alignment",
            threat_col: "threat_alignment",
        }
    )
    out["safety_alignment"] = pd.to_numeric(out["safety_alignment"], errors="coerce")
    out["threat_alignment"] = pd.to_numeric(out["threat_alignment"], errors="coerce")
    out["group"] = out["group"].astype(str)
    out["condition"] = out["condition"].astype(str)
    out["source"] = source_name
    out = out[out["group"].isin(["SAD", "HC"])]
    out = out[out["condition"].isin(["CSR", "CSS", "CS-"])]
    return out.dropna(subset=["safety_alignment", "threat_alignment"])[GEOMETRY_PANEL_COLUMNS]


def build_aim2_trajectory_panel(stats_dir: Path) -> pd.DataFrame:
    """Load Figure 2 panel-D input only from the true upstream trajectory export."""
    path = stats_dir / "aim2_trajectory_panel.csv"
    columns = ["subject_id", "group", "trial", "trajectory", "trajectory_metric", "value"]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    data = read_csv_if_exists(path)
    if data.empty:
        return pd.DataFrame(columns=columns)
    if not set(columns).issubset(data.columns):
        return pd.DataFrame(columns=columns)
    if not data["trajectory_metric"].astype(str).eq("target_centroid_cosine").all():
        return pd.DataFrame(columns=columns)
    if "drug" in data.columns:
        data = data[data["drug"].astype(str).eq("Placebo")].copy()
    out = data[columns].copy()
    out["subject_id"] = out["subject_id"].astype(str)
    out["group"] = out["group"].astype(str)
    out["trajectory"] = out["trajectory"].astype(str)
    out["trial"] = pd.to_numeric(out["trial"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out[out["group"].isin(["SAD", "HC"])]
    out = out[out["trajectory"].isin(["safety", "threat"])]
    return out.dropna(subset=["trial", "value"])[columns]


def write_qc_dashboard(input_path: Path, stats_dir: Path, repo_root: Path) -> None:
    df = read_csv_if_exists(input_path)
    if df.empty:
        (stats_dir / "mvpa_l2_qc_dashboard.md").write_text("# MVPA L2 QC Dashboard\n\nNo harmonized subject table found.\n")
        return

    count_cols = [c for c in ["FeatureSpace", "Group", "Drug"] if c in df.columns]
    subject_counts = df.groupby(count_cols, dropna=False).size().reset_index(name="rows") if count_cols else pd.DataFrame()
    missingness = missingness_table(df) if "FeatureSpace" in df.columns else pd.DataFrame()
    status_counts = model_status_counts(stats_dir)

    write_csv(subject_counts, stats_dir / "qc_subject_counts.csv")
    write_csv(missingness, stats_dir / "qc_missingness.csv")
    write_csv(status_counts, stats_dir / "qc_model_status_counts.csv")

    commit = git_value(["rev-parse", "--short", "HEAD"], repo_root)
    status = git_value(["status", "--short"], repo_root)
    dirty_rows = 0 if status == "unavailable" or not status else len(status.splitlines())
    tracked_note = "clean" if dirty_rows == 0 else f"{dirty_rows} changed paths"

    top_missing = missingness.sort_values("missing_pct", ascending=False).head(20) if not missingness.empty else pd.DataFrame()
    lines = [
        "# MVPA L2 Reproducibility/QC Dashboard",
        "",
        "## Run State",
        "",
        f"- Harmonized input: `{input_path}`",
        f"- Git commit: `{commit}`",
        f"- Working tree: {tracked_note}",
        "",
        "## Subject Counts",
        "",
        _markdown_table(subject_counts),
        "",
        "## Highest Missingness",
        "",
        _markdown_table(format_numeric_columns(top_missing, ["missing_pct"])),
        "",
        "## Model Status Counts",
        "",
        _markdown_table(status_counts),
        "",
        "## Leakage Audit",
        "",
        "- Downstream manuscript scripts operate on harmonized subject-level metrics and do not refit decoders, scalers, feature masks, or probability calibration models.",
        "- Predictive leakage risk must be audited in the upstream Hyak feature-space scripts, where subject-aware scaling, mask generation, feature selection, and cross-validation are implemented.",
        "- Required upstream checks: `StandardScaler` fit inside train folds only; subject-aware validation (`StratifiedGroupKFold` or `LeaveOneGroupOut`); no held-out subject data in mask generation, hyperparameter tuning, or calibration.",
        "",
        "## Exported Manuscript Artifacts",
        "",
        "- `manuscript_primary_results.csv` and `.md`",
        "- `aim4_convergence_matrix.csv`, `_wide.csv`, and `.md`",
        "- `qc_subject_counts.csv`, `qc_missingness.csv`, and `qc_model_status_counts.csv`",
        "",
    ]
    (stats_dir / "mvpa_l2_qc_dashboard.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/mvpa_l2/harmonized/mvpa_l2_subject_metrics.csv"),
    )
    parser.add_argument("--stats-dir", type=Path, default=Path("outputs/mvpa_l2/stats"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()

    args.stats_dir.mkdir(parents=True, exist_ok=True)
    write_aim2_panel_inputs(args.input, args.stats_dir)
    write_primary_table(args.stats_dir)
    convergence_matrix(args.stats_dir)
    write_qc_dashboard(args.input, args.stats_dir, args.repo_root)
    print(f"Wrote manuscript artifacts and QC dashboard -> {args.stats_dir}")


if __name__ == "__main__":
    main()
