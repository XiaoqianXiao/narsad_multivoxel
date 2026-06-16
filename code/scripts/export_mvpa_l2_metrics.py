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
}


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
    for (sub, group), sub_df in df.groupby(["sub", "Group"], dropna=False):
        sub_df = sub_df.sort_values("trial")
        if len(sub_df) < 3:
            continue
        slope, _ = np.polyfit(pd.to_numeric(sub_df["trial"], errors="coerce"), pd.to_numeric(sub_df["score"], errors="coerce"), 1)
        rows.append(
            {
                "sub_ID": normalize_subject_id(sub),
                "Group": group,
                "Drug": "Placebo",
                metric_name: slope,
            }
        )
    return pd.DataFrame(rows)


def load_trajectories(base_dir: Path) -> Optional[pd.DataFrame]:
    payload14 = load_payload(
        base_dir,
        [
            "cell_14.joblib",
            "cell_12_trajectories.joblib",
            "checkpoints/cell_14.joblib",
        ],
    )
    results = payload_value(payload14, "results_13_2") or payload14
    frames = []
    if isinstance(results, dict):
        data_safe = results.get("data_safe")
        data_threat = results.get("data_threat")
        slopes = results.get("trajectory_slopes")
        if isinstance(slopes, pd.DataFrame) and not slopes.empty:
            for condition, metric in [
                ("Safety Learning", "Neural_Safety_Trajectory_Slope"),
                ("Threat Maintenance", "Neural_Threat_Trajectory_Slope"),
            ]:
                sub = slopes[slopes["Condition"] == condition].copy()
                if not sub.empty:
                    sub = ensure_subject_column(sub.rename(columns={"slope": metric}))
                    sub["Drug"] = "Placebo"
                    frames.append(sub[["sub_ID", "Group", "Drug", metric]])
        if isinstance(data_safe, pd.DataFrame):
            frames.append(_trajectory_slopes(data_safe, "Safety Learning", "Neural_Safety_Trajectory_Slope"))
        if isinstance(data_threat, pd.DataFrame):
            frames.append(_trajectory_slopes(data_threat, "Threat Maintenance", "Neural_Threat_Trajectory_Slope"))

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
    return coalesce_duplicate_columns(out)


def load_clinical(base_dir: Path) -> Optional[pd.DataFrame]:
    payload23 = load_payload(base_dir, ["cell_23.joblib", "checkpoints/cell_23.joblib"])
    df = payload_value(payload23, "df_scored_clinical")
    if isinstance(df, pd.DataFrame):
        return ensure_subject_column(df)
    return None


def attach_metadata(df: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
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
    pieces = [
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
    merged = derive_final_metrics(merged)
    merged["FeatureSpace"] = feature_space
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
    print(f"Wrote {len(out)} rows x {len(out.columns)} columns -> {args.out}")
    print(out.groupby(["FeatureSpace", "Group", "Drug"], dropna=False).size())


if __name__ == "__main__":
    main()
