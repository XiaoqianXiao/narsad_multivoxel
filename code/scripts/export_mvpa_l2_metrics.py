#!/usr/bin/env python3
"""Export harmonized subject-level MVPA L2 metrics from Hyak caches.

The expensive analyses live in the feature-space-specific Hyak scripts. This
script reads their cached joblib outputs and writes a stable CSV with the metric
names used in mvpa_L2.md.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from mvpa_l2_common import (
    coalesce_duplicate_columns,
    derive_final_metrics,
    ensure_subject_column,
    find_existing,
    harmonize_group_drug,
    maybe_read_joblib,
    merge_on_subject,
    normalize_subject_id,
    payload_value,
    write_csv,
)


DEFAULT_FEATURE_DIRS = {
    "FearNetwork": Path("outputs/mvpa_l2/FearNetwork"),
    "MemoryFearNetwork": Path("outputs/mvpa_l2/MemoryFearNetwork"),
    "WholeBrain_Schaefer": Path("outputs/mvpa_l2/WholeBrain_Schaefer"),
}

AIM2_TRAJECTORY_METRIC = "early_to_target_normalized_projection"


def parse_feature_dir(values: Optional[List[str]]) -> Dict[str, Path]:
    if not values:
        return DEFAULT_FEATURE_DIRS.copy()
    out = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--feature-dir must be FeatureSpace=/path")
        name, path = value.split("=", 1)
        out[name] = Path(path)
    return out


def load_payload(base_dir: Path, names: List[str]):
    path = find_existing(base_dir, names)
    if path is None:
        return None
    return maybe_read_joblib(path)


def load_master(base_dir: Path) -> Optional[pd.DataFrame]:
    payload = load_payload(
        base_dir,
        [
            "df_master_analysis.joblib",
            "stage26_MasterClinicalNeural.joblib",
            "stage26_master_clinical_neural.joblib",
            "cell_26.joblib",
            "checkpoints/cell_26.joblib",
        ],
    )
    df = payload_value(payload, "df_master_analysis")
    if isinstance(df, pd.DataFrame):
        return ensure_subject_column(df)
    if isinstance(payload, pd.DataFrame):
        return ensure_subject_column(payload)
    return None


def load_stage24_core_metrics(base_dir: Path) -> Optional[pd.DataFrame]:
    payload = load_payload(
        base_dir,
        [
            "stage24_NeuralClinicalIndices.joblib",
            "stage24_neural_clinical_indices.joblib",
            "cell_24.joblib",
            "checkpoints/cell_24.joblib",
        ],
    )
    df = payload_value(payload, "df_core_representative")
    if isinstance(df, pd.DataFrame) and not df.empty:
        return derive_final_metrics(ensure_subject_column(df))
    return None


def topology_from_results12(payload) -> Optional[pd.DataFrame]:
    results = payload_value(payload, "results_12") or payload
    if not isinstance(results, dict):
        return None
    rdms_sad = results.get("rdms_sad_raw_pv")
    if rdms_sad is None:
        rdms_sad = results.get("rdms_sad_raw")
    rdms_hc = results.get("rdms_hc_raw_pv")
    if rdms_hc is None:
        rdms_hc = results.get("rdms_hc_raw")
    subs_sad = results.get("subs_sad_rdm")
    if subs_sad is None:
        subs_sad = results.get("subs_sad_rdm_z")
    subs_hc = results.get("subs_hc_rdm")
    if subs_hc is None:
        subs_hc = results.get("subs_hc_rdm_z")
    if subs_sad is None or subs_hc is None:
        subgroups = payload_value(payload, "subgroups_21")
        if isinstance(subgroups, dict):
            subs_sad = subs_sad if subs_sad is not None else subgroups.get("SAD_Placebo")
            subs_hc = subs_hc if subs_hc is not None else subgroups.get("HC_Placebo")
    if rdms_sad is None or rdms_hc is None or subs_sad is None or subs_hc is None:
        return None

    def rows(rdms, subs, group):
        arr = np.asarray(rdms)
        records = []
        for i, sub in enumerate(subs):
            if i >= arr.shape[0] or arr.shape[1] < 3 or arr.shape[2] < 3:
                continue
            records.append(
                {
                    "sub_ID": normalize_subject_id(sub),
                    "Group": group,
                    "Drug": "Placebo",
                    "Neural_Dist_Safety_Background": arr[i, 1, 0],
                    "Neural_Dist_Threat_Safety": arr[i, 2, 1],
                    "Neural_Dist_Threat_Background": arr[i, 2, 0],
                    "Neural_ThreatTriangleOpenness": arr[i, 2, 0] - arr[i, 1, 0],
                    "Neural_Threat_Safety_Distance": arr[i, 2, 0] - arr[i, 1, 0],
                    "Neural_Topology_Safety_Integration": arr[i, 2, 1] - arr[i, 1, 0],
                    "Neural_Threat_Bias": arr[i, 2, 0] - arr[i, 1, 0],
                }
            )
        return records

    return pd.DataFrame(rows(rdms_sad, subs_sad, "SAD") + rows(rdms_hc, subs_hc, "HC"))


def load_topology(base_dir: Path) -> Optional[pd.DataFrame]:
    payload17 = load_payload(base_dir, ["cell_17.joblib", "checkpoints/cell_17.joblib"])
    df = payload_value(payload17, "df_topo")
    if isinstance(df, pd.DataFrame) and not df.empty:
        out = ensure_subject_column(df)
        out = derive_final_metrics(out)
        from_rdm = topology_from_results12(payload17)
        if isinstance(from_rdm, pd.DataFrame) and not from_rdm.empty:
            out = merge_on_subject(out, from_rdm)
            out = coalesce_duplicate_columns(out)
            out = derive_final_metrics(out)
        return out

    payload12 = load_payload(
        base_dir,
        [
            "analysis_12_topology.joblib",
            "stage12_topology_stats.joblib",
            "stage12_StaticRepresentationalTopology.joblib",
            "cell_12.joblib",
            "checkpoints/cell_12.joblib",
            "results_12.joblib",
        ],
    )
    return topology_from_results12(payload12)


def load_decision(base_dir: Path) -> Optional[pd.DataFrame]:
    frames = []

    def add_placebo_self_tables(payload) -> None:
        results = payload_value(payload, "results_14_self") or payload
        if not isinstance(results, dict):
            return
        for group, key in [("SAD", "df_sad"), ("HC", "df_hc")]:
            df_group = results.get(key)
            if isinstance(df_group, pd.DataFrame) and not df_group.empty:
                frame = ensure_subject_column(df_group)
                frame["Group"] = group
                frame["Drug"] = "Placebo"
                frames.append(derive_final_metrics(frame))

    def add_all_drug_opening_table(payload) -> None:
        results = payload_value(payload, "results_23")
        if isinstance(results, dict):
            df_opening = results.get("df")
        else:
            df_opening = None
        if isinstance(df_opening, pd.DataFrame) and not df_opening.empty:
            frames.append(derive_final_metrics(ensure_subject_column(df_opening)))

    payload19 = load_payload(
        base_dir,
        [
            "cell_19.joblib",
            "cell_16_opening_test.joblib",
            "stage15_results_23.joblib",
            "checkpoints/cell_19.joblib",
        ],
    )
    add_all_drug_opening_table(payload19)
    add_placebo_self_tables(payload19)
    df = payload_value(payload19, "df_metrics")
    if isinstance(df, pd.DataFrame) and not df.empty:
        out = ensure_subject_column(df)
        out = derive_final_metrics(out)
        frames.append(out)

    payload15 = load_payload(
        base_dir,
        [
            "cell_15.joblib",
            "cell_13_decision_stats_opt.joblib",
            "stage15_decision_stats.joblib",
            "stage15_DecisionBoundaryCharacteristics.joblib",
            "checkpoints/cell_15.joblib",
        ],
    )
    add_placebo_self_tables(payload15)
    if not frames:
        return None
    out = None
    for frame in frames:
        out = merge_on_subject(out, frame)
    return derive_final_metrics(coalesce_duplicate_columns(out))


def _trajectory_slopes(df: pd.DataFrame, condition: str, metric_name: str) -> pd.DataFrame:
    rows = []
    if df is None or df.empty:
        return pd.DataFrame()
    group_cols = ["sub", "Group"]
    if "Drug" in df.columns:
        group_cols.append("Drug")
    for keys, sub_df in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_cols, keys))
        sub_df = sub_df.sort_values("trial")
        if len(sub_df) < 3:
            continue
        slope, _ = np.polyfit(pd.to_numeric(sub_df["trial"], errors="coerce"), pd.to_numeric(sub_df["score"], errors="coerce"), 1)
        rows.append(
            {
                "sub_ID": normalize_subject_id(key_map.get("sub")),
                "Group": key_map.get("Group"),
                "Drug": key_map.get("Drug", pd.NA),
                metric_name: slope,
            }
        )
    return pd.DataFrame(rows)


def _standardize_trial_alignment(data: pd.DataFrame, trajectory: str, value_col: Optional[str] = None) -> pd.DataFrame:
    """Return notebook-ready trial alignment rows for Figure 2 panel D."""
    if data is None or data.empty:
        return pd.DataFrame()
    subject_col = next((col for col in ["subject_id", "Subject", "sub_ID", "sub"] if col in data.columns), None)
    group_col = next((col for col in ["Group", "group"] if col in data.columns), None)
    drug_col = next((col for col in ["Drug", "drug"] if col in data.columns), None)
    trial_col = next((col for col in ["trial", "Trial", "block", "Block"] if col in data.columns), None)
    score_col = value_col or next(
        (col for col in ["safety_alignment", "threat_evidence", "score", "Score", "similarity", "Similarity", "value", "evidence"] if col in data.columns),
        None,
    )
    if subject_col is None or group_col is None or trial_col is None or score_col is None:
        return pd.DataFrame()
    out = data[[subject_col, group_col, trial_col, score_col]].copy()
    out = out.rename(columns={subject_col: "subject_id", group_col: "group", trial_col: "trial", score_col: "value"})
    if drug_col is not None:
        out["drug"] = data[drug_col].astype(str).values
        out = out[out["drug"].eq("Placebo")].copy()
    out["trajectory"] = trajectory
    out["trajectory_metric"] = AIM2_TRAJECTORY_METRIC
    out["trial"] = pd.to_numeric(out["trial"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out["group"] = out["group"].astype(str)
    columns = ["subject_id", "group", "drug", "trial", "trajectory", "trajectory_metric", "value"] if "drug" in out.columns else ["subject_id", "group", "trial", "trajectory", "trajectory_metric", "value"]
    return out.dropna(subset=["trial", "value"])[columns]


def trajectory_panel_from_feature_dir(base_dir: Path) -> pd.DataFrame:
    """Extract trial-level safety/threat trajectories for Figure 2 panel D."""
    payload14 = load_payload(
        base_dir,
        [
            "cell_14.joblib",
            "cell_12_trajectories.joblib",
            "stage14_trajectories.joblib",
            "stage14_DynamicTrajectories.joblib",
            "checkpoints/cell_14.joblib",
        ],
    )
    results = payload_value(payload14, "results_13_2") or payload14
    frames = []
    if isinstance(results, dict):
        if results.get("trajectory_metric") != AIM2_TRAJECTORY_METRIC:
            return pd.DataFrame()
        data_safe = results.get("data_safe")
        data_threat = results.get("data_threat")
        if isinstance(data_safe, pd.DataFrame):
            frames.append(_standardize_trial_alignment(data_safe, "safety"))
        if isinstance(data_threat, pd.DataFrame):
            frames.append(_standardize_trial_alignment(data_threat, "threat"))
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_trajectories(base_dir: Path) -> Optional[pd.DataFrame]:
    payload17 = load_payload(base_dir, ["cell_17.joblib", "checkpoints/cell_17.joblib"])
    payload14 = load_payload(
        base_dir,
        [
            "cell_14.joblib",
            "cell_12_trajectories.joblib",
            "stage14_trajectories.joblib",
            "stage14_DynamicTrajectories.joblib",
            "checkpoints/cell_14.joblib",
        ],
    )
    results = payload_value(payload14, "results_13_2") or payload14
    frames = []
    shock_anchor_all_drug = payload_value(payload17, "shock_anchor_all_drug", "df_shock_anchor_all_drug")
    if (
        isinstance(shock_anchor_all_drug, pd.DataFrame)
        and not shock_anchor_all_drug.empty
        and "Cosine_CSR_minus_CSS" in shock_anchor_all_drug.columns
    ):
        residualized_all_drug = ensure_subject_column(shock_anchor_all_drug.rename(columns={"Subject": "sub_ID"}))
        residualized_all_drug["Residualized_Shock_Anchor_Trajectory_Slope"] = pd.to_numeric(
            residualized_all_drug["Cosine_CSR_minus_CSS"],
            errors="coerce",
        )
        frames.append(residualized_all_drug[["sub_ID", "Group", "Drug", "Residualized_Shock_Anchor_Trajectory_Slope"]])
    if isinstance(results, dict):
        data_safe = results.get("data_safe")
        data_threat = results.get("data_threat")
        data_threat_shock = results.get("data_threat_shock")
        slopes = results.get("trajectory_slopes")
        if isinstance(slopes, pd.DataFrame) and not slopes.empty:
            for condition, metric in [
                ("Safety Learning", "Neural_Safety_Trajectory_Slope"),
                ("Threat Maintenance", "Neural_Threat_Trajectory_Slope"),
                ("Threat Shock Target", "Shock_Anchor_Trajectory_Slope"),
            ]:
                sub = slopes[slopes["Condition"] == condition].copy()
                if not sub.empty:
                    sub = ensure_subject_column(sub.rename(columns={"slope": metric}))
                    if "Drug" not in sub.columns:
                        sub["Drug"] = "Placebo"
                    frames.append(sub[["sub_ID", "Group", "Drug", metric]])
        if isinstance(data_safe, pd.DataFrame):
            frames.append(_trajectory_slopes(data_safe, "Safety Learning", "Neural_Safety_Trajectory_Slope"))
        if isinstance(data_threat, pd.DataFrame):
            frames.append(_trajectory_slopes(data_threat, "Threat Maintenance", "Neural_Threat_Trajectory_Slope"))
        if isinstance(data_threat_shock, pd.DataFrame):
            frames.append(_trajectory_slopes(data_threat_shock, "Threat Shock Target", "Shock_Anchor_Trajectory_Slope"))

    payload12 = load_payload(
        base_dir,
        [
            "cell_12.joblib",
            "stage12_topology_stats.joblib",
            "stage12_StaticRepresentationalTopology.joblib",
            "checkpoints/cell_12.joblib",
        ],
    )
    shock_anchor_df = payload_value(payload12, "shock_anchor_df")
    if isinstance(shock_anchor_df, pd.DataFrame) and not shock_anchor_df.empty and "Cosine_CSR_minus_CSS" in shock_anchor_df.columns:
        residualized = ensure_subject_column(shock_anchor_df.rename(columns={"Subject": "sub_ID"}))
        residualized["Drug"] = "Placebo"
        if "Group" not in residualized.columns:
            residualized["Group"] = pd.NA
        residualized["Residualized_Shock_Anchor_Trajectory_Slope"] = pd.to_numeric(
            residualized["Cosine_CSR_minus_CSS"],
            errors="coerce",
        )
        frames.append(residualized[["sub_ID", "Group", "Drug", "Residualized_Shock_Anchor_Trajectory_Slope"]])

    payload18 = load_payload(base_dir, ["cell_18.joblib", "checkpoints/cell_18.joblib"])
    drift = payload_value(payload18, "df_drift", "df")
    if isinstance(drift, pd.DataFrame) and not drift.empty:
        drift = ensure_subject_column(drift)
        wide = []
        for domain, prefix in [
            ("Safety", "Neural_Safety_Drift"),
            ("Threat", "Neural_Threat_Drift"),
            ("Threat Shock Target", "Neural_ThreatShock_Drift"),
        ]:
            sub = drift[drift["Domain"] == domain].copy()
            if sub.empty:
                continue
            cols = ["sub_ID", "Group", "Drug"]
            for old, new in [("Projection", f"{prefix}_Projection"), ("Cosine", f"{prefix}_Cosine")]:
                if old in sub.columns:
                    sub[new] = sub[old]
                    cols.append(new)
            wide.append(sub[cols])
        frames.extend(wide)

    if not frames:
        return None
    out = None
    for frame in frames:
        out = merge_on_subject(out, frame)
    out = coalesce_duplicate_columns(out)
    if "Shock_Anchor_Trajectory_Slope" in out.columns:
        out["ShockAnchor_Threat_Trajectory_Slope"] = pd.to_numeric(
            out["Shock_Anchor_Trajectory_Slope"],
            errors="coerce",
        )
    if "Residualized_Shock_Anchor_Trajectory_Slope" in out.columns:
        out["Residualized_ShockAnchor_Threat_Slope"] = pd.to_numeric(
            out["Residualized_Shock_Anchor_Trajectory_Slope"],
            errors="coerce",
        )
    return out


GEOMETRY_PANEL_COLUMNS = ["subject_id", "group", "condition", "safety_alignment", "threat_alignment", "source"]


def empty_geometry_panel() -> pd.DataFrame:
    """Return an empty Figure 2 panel-B schema."""
    return pd.DataFrame(columns=GEOMETRY_PANEL_COLUMNS)


def standardize_geometry_panel(data: pd.DataFrame, source_name: str) -> pd.DataFrame:
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
        return empty_geometry_panel()
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


def geometry_panel_from_feature_dir(base_dir: Path) -> pd.DataFrame:
    """Load true centroid geometry if an upstream export exists; otherwise empty."""
    candidate_names = [
        "aim2_geometry_centroids.csv",
        "aim2_condition_centroids.csv",
        "aim2_subject_condition_centroids.csv",
        "aim2_true_centroid_geometry.csv",
    ]
    search_dirs = [base_dir, base_dir / "intermediate", base_dir / "stats", base_dir / "exports"]
    for directory in search_dirs:
        for name in candidate_names:
            path = directory / name
            if not path.exists():
                continue
            panel = standardize_geometry_panel(pd.read_csv(path), path.name)
            if not panel.empty:
                return panel
    return empty_geometry_panel()


def export_aim2_panel_inputs(subject_df: pd.DataFrame, feature_dirs: Dict[str, Path], stats_dir: Path, feature_space: str = "FearNetwork") -> None:
    """Write Figure 2 geometry and trajectory input CSVs for the notebook."""
    stats_dir.mkdir(parents=True, exist_ok=True)
    geometry = geometry_panel_from_feature_dir(feature_dirs[feature_space]) if feature_space in feature_dirs else empty_geometry_panel()
    write_csv(geometry, stats_dir / "aim2_geometry_panel.csv")
    write_csv(geometry, stats_dir / "aim2_subject_condition_centroids.csv")

    trajectory = pd.DataFrame()
    if feature_space in feature_dirs:
        trajectory = trajectory_panel_from_feature_dir(feature_dirs[feature_space])
    if trajectory.empty:
        trajectory = pd.DataFrame(columns=["subject_id", "group", "trial", "trajectory", "trajectory_metric", "value"])
    write_csv(trajectory, stats_dir / "aim2_trajectory_panel.csv")
    print(f"Wrote Aim 2 panel inputs -> {stats_dir / 'aim2_geometry_panel.csv'} and {stats_dir / 'aim2_trajectory_panel.csv'}")


def load_clinical(base_dir: Path) -> Optional[pd.DataFrame]:
    # This table can include clinical/SCR subjects outside the analyzed neural
    # sample and usually does not carry Group/Drug. The later outer merge keeps
    # those rows visible; if metadata lacks them, they appear as Group/Drug NaN.
    payload23 = load_payload(
        base_dir,
        [
            "cell_23.joblib",
            "stage23_ClinicalScores.joblib",
            "stage23_clinical_scores.joblib",
            "checkpoints/cell_23.joblib",
        ],
    )
    df = payload_value(payload23, "df_scored_clinical")
    if isinstance(df, pd.DataFrame):
        return ensure_subject_column(df)
    return None


def attach_metadata(df: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    # cell_04 is the source for treatment-arm metadata. It may cover fewer
    # subjects than df_scored_clinical, so combine_first fills only overlapping
    # subjects and intentionally leaves clinical-only subjects as missing.
    payload4 = load_payload(base_dir, ["cell_04.joblib", "cell_4.joblib", "checkpoints/cell_04.joblib"])
    meta = payload_value(payload4, "meta")
    if isinstance(meta, pd.DataFrame):
        meta = ensure_subject_column(meta)
        cols = [c for c in ["sub_ID", "Group", "Drug", "demo_age", "Age", "age", "Sex", "sex", "Gender", "gender"] if c in meta.columns]
        if cols:
            df = df.merge(meta[cols].drop_duplicates("sub_ID"), on="sub_ID", how="left", suffixes=("", "_meta"))
            for col in ["Group", "Drug", "demo_age", "Age", "age", "Sex", "sex", "Gender", "gender"]:
                meta_col = f"{col}_meta"
                if meta_col in df.columns:
                    df[col] = df[col].combine_first(df[meta_col])
                    df = df.drop(columns=[meta_col])
    return df


def export_feature_space(feature_space: str, base_dir: Path) -> pd.DataFrame:
    # Keep the merge inclusive across neural, clinical, SCR, and master exports.
    # A final Group/Drug NaN count usually means retained clinical-only rows,
    # not failed neural estimates. Check nonmissing Neural_* columns before
    # interpreting those rows as analysis failures.
    pieces = [
        load_stage24_core_metrics(base_dir),
        load_topology(base_dir),
        load_decision(base_dir),
        load_trajectories(base_dir),
        load_clinical(base_dir),
        load_master(base_dir),
    ]
    merged = None
    for piece in pieces:
        if piece is not None and not piece.empty:
            merged = merge_on_subject(merged, piece)
    if merged is None:
        print(f"[WARN] No usable outputs found for {feature_space}: {base_dir}")
        return pd.DataFrame()
    merged = coalesce_duplicate_columns(merged)
    merged = attach_metadata(merged, base_dir)
    merged = harmonize_group_drug(merged)
    merged["FeatureSpace"] = feature_space
    merged = derive_final_metrics(merged)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", action="append", help="FeatureSpace=/path/to/output_dir")
    parser.add_argument(
        "--scr-flags",
        type=Path,
        default=Path("outputs/mvpa_l2/harmonized/scr_sensitivity_groups.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/mvpa_l2/harmonized/mvpa_l2_subject_metrics.csv"),
    )
    parser.add_argument(
        "--stats-out-dir",
        type=Path,
        default=None,
        help="Directory for notebook-facing auxiliary inputs such as Aim 2 geometry/trajectory panels.",
    )
    args = parser.parse_args()

    feature_dirs = parse_feature_dir(args.feature_dir)
    frames = []
    for feature_space, base_dir in feature_dirs.items():
        frame = export_feature_space(feature_space, base_dir)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise SystemExit("No feature-space outputs found. Pass --feature-dir FeatureSpace=/path after Hyak finishes.")

    out = pd.concat(frames, ignore_index=True, sort=False)
    out = ensure_subject_column(out)
    out = harmonize_group_drug(out)
    if args.scr_flags.exists():
        flags = ensure_subject_column(pd.read_csv(args.scr_flags))
        out = out.merge(flags, on="sub_ID", how="left")
    out = coalesce_duplicate_columns(out)
    out = out.sort_values(["FeatureSpace", "Group", "Drug", "sub_ID"], na_position="last")
    write_csv(out, args.out)
    stats_out_dir = args.stats_out_dir or args.out.parent.parent / "stats"
    export_aim2_panel_inputs(out, feature_dirs, stats_out_dir)
    print(f"Wrote {len(out)} rows x {len(out.columns)} columns -> {args.out}")
    print(out.groupby(["FeatureSpace", "Group", "Drug"], dropna=False).size())


if __name__ == "__main__":
    main()
