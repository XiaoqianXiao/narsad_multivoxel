#!/usr/bin/env python3
"""Export Aim 1 feature-space sensitivity rows and paired-drop nulls."""

import argparse
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from export_aim1_decoding_primary import export, feature_label


def parse_feature_dir(value: str) -> Tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        return label.strip(), Path(path)
    path = Path(value)
    return path.name, path


def first_value(frame: pd.DataFrame, test: str, value_col: str) -> float:
    if frame.empty or "test" not in frame.columns or value_col not in frame.columns:
        return np.nan
    match = frame[frame["test"].astype(str).eq(test)]
    if match.empty:
        return np.nan
    return pd.to_numeric(match.iloc[0].get(value_col), errors="coerce")


def compatibility_rows(table: pd.DataFrame, feature_space: str, checkpoint: object) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if table.empty or "test" not in table.columns:
        return rows
    label = feature_label(feature_space)
    mappings = [
        ("SAD self-decoding CV accuracy", "SAD-Placebo self-decoding", "Sensitivity: self-decoding", "group_specific_identification", "accuracy"),
        ("HC self-decoding CV accuracy", "HC-Placebo self-decoding", "Sensitivity: self-decoding", "group_specific_identification", "accuracy"),
        ("SAD model tested on HC", "SAD-Placebo model tested on HC-Placebo", "Sensitivity: specificity", "functional_specificity", "accuracy"),
        ("HC model tested on SAD", "HC-Placebo model tested on SAD-Placebo", "Sensitivity: specificity", "functional_specificity", "accuracy"),
        ("SAD-HC Haufe map cosine similarity", "SAD-Placebo vs HC-Placebo discrimination-weight cosine similarity", "Sensitivity: specificity", "spatial_specificity", "cosine_similarity"),
    ]
    for source_test, target_test, family, domain, value_col in mappings:
        match = table[table["test"].astype(str).eq(source_test)].copy()
        if match.empty:
            continue
        source = match.iloc[0]
        rows.append(
            {
                "feature_space": feature_space,
                "session": source.get("session", "Placebo"),
                "test": target_test,
                "estimate": pd.to_numeric(pd.Series([source.get(value_col)]), errors="coerce").iloc[0],
                "p_value": pd.to_numeric(pd.Series([source.get("p_value")]), errors="coerce").iloc[0],
                "family": family,
                "observed_permutations": source.get("n_permutations", np.nan),
                "analysis_role": "sensitivity",
                "specificity_domain": domain,
                "planned_permutations": 5000,
                "mask_or_feature_space": label,
                "checkpoint": checkpoint,
                "status": source.get("status", "ok"),
            }
        )
    return rows


def wide_row(table: pd.DataFrame, feature_space: str) -> Dict[str, object]:
    status = "missing"
    if not table.empty:
        if "test" in table.columns:
            status = "ok"
        elif "status" in table.columns and table["status"].notna().any():
            status = str(table["status"].dropna().iloc[0])
    sad_to_hc = first_value(table, "SAD model tested on HC", "accuracy")
    hc_to_sad = first_value(table, "HC model tested on SAD", "accuracy")
    cross_values = [value for value in [sad_to_hc, hc_to_sad] if pd.notna(value)]
    return {
        "mask_or_feature_space": feature_label(feature_space),
        "feature_space": feature_space,
        "session": "Placebo",
        "status": status,
        "SAD self": first_value(table, "SAD self-decoding CV accuracy", "accuracy"),
        "SAD self p": first_value(table, "SAD self-decoding CV accuracy", "p_value"),
        "HC self": first_value(table, "HC self-decoding CV accuracy", "accuracy"),
        "HC self p": first_value(table, "HC self-decoding CV accuracy", "p_value"),
        "Cross-group generalization index": float(np.mean(cross_values)) if cross_values else np.nan,
        "SAD -> HC": sad_to_hc,
        "SAD -> HC p": first_value(table, "SAD model tested on HC", "p_value"),
        "HC -> SAD": hc_to_sad,
        "HC -> SAD p": first_value(table, "HC model tested on SAD", "p_value"),
        "Full SAD model on SAD subgroup": np.nan,
        "Full HC model on HC subgroup": np.nan,
        "Full SAD model on HC subgroup": np.nan,
        "Full HC model on SAD subgroup": np.nan,
        "Weight similarity": first_value(table, "SAD-HC Haufe map cosine similarity", "cosine_similarity"),
        "Weight similarity p": first_value(table, "SAD-HC Haufe map cosine similarity", "p_value"),
    }


def concat_csv(paths: List[Path], out: Path) -> None:
    frames = []
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            try:
                frames.append(pd.read_csv(path))
            except pd.errors.EmptyDataError:
                pass
    table = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"Wrote {len(table)} rows -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", action="append", required=True, help="Feature label and directory as LABEL=PATH. May be repeated.")
    parser.add_argument("--out", type=Path, default=Path("outputs/mvpa_l2/stats/aim1_sensitivity_feature_space.csv"))
    parser.add_argument("--wide-out", type=Path, default=None)
    parser.add_argument("--raincloud-out", type=Path, default=None)
    parser.add_argument("--drop-tests-out", type=Path, default=None)
    parser.add_argument("--drop-nulls-out", type=Path, default=None)
    args = parser.parse_args()

    wide_out = args.wide_out or args.out.with_name(args.out.stem + "_wide.csv")
    raincloud_out = args.raincloud_out or args.out.with_name(args.out.stem + "_raincloud.csv")
    drop_tests_out = args.drop_tests_out or args.out.with_name(args.out.stem + "_functional_drop_tests.csv")
    drop_nulls_out = args.drop_nulls_out or args.out.with_name(args.out.stem + "_functional_drop_nulls.csv")

    rows: List[Dict[str, object]] = []
    wide_rows: List[Dict[str, object]] = []
    raincloud_parts: List[Path] = []
    drop_test_parts: List[Path] = []
    drop_null_parts: List[Path] = []

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aim1_feature_sensitivity_") as tmp:
        tmp_dir = Path(tmp)
        for feature_space, feature_dir in [parse_feature_dir(item) for item in args.feature_dir]:
            stem = feature_space.lower().replace("/", "_").replace(" ", "_")
            primary_out = tmp_dir / f"{stem}_aim1.csv"
            raincloud_part = tmp_dir / f"{stem}_raincloud.csv"
            drop_tests_part = tmp_dir / f"{stem}_functional_drop_tests.csv"
            drop_nulls_part = tmp_dir / f"{stem}_functional_drop_nulls.csv"
            table = export(
                feature_dir=feature_dir,
                out=primary_out,
                feature_space=feature_space,
                raincloud_out=raincloud_part,
                drop_tests_out=drop_tests_part,
                drop_nulls_out=drop_nulls_part,
                sensitivity_label=feature_label(feature_space),
            )
            checkpoint = table["checkpoint"].dropna().iloc[0] if "checkpoint" in table.columns and table["checkpoint"].notna().any() else feature_dir
            rows.extend(compatibility_rows(table, feature_space, checkpoint))
            wide_rows.append(wide_row(table, feature_space))
            raincloud_parts.append(raincloud_part)
            drop_test_parts.append(drop_tests_part)
            drop_null_parts.append(drop_nulls_part)

        pd.DataFrame(rows).to_csv(args.out, index=False)
        pd.DataFrame(wide_rows).to_csv(wide_out, index=False)
        print(f"Wrote {len(rows)} rows -> {args.out}")
        print(f"Wrote {len(wide_rows)} rows -> {wide_out}")
        concat_csv(raincloud_parts, raincloud_out)
        concat_csv(drop_test_parts, drop_tests_out)
        concat_csv(drop_null_parts, drop_nulls_out)


if __name__ == "__main__":
    main()
