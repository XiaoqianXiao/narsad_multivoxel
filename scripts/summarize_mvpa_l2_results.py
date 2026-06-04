#!/usr/bin/env python3
"""Create a compact Markdown summary of MVPA L2 model outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _markdown_table(df: pd.DataFrame) -> str:
    """Render a small markdown table without optional pandas dependencies."""
    if df.empty:
        return "_No rows._"
    rows = df.astype(str).values.tolist()
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def read_tables(stats_dir: Path) -> dict[str, pd.DataFrame]:
    tables = {}
    for path in sorted(stats_dir.glob("*.csv")):
        try:
            tables[path.stem] = pd.read_csv(path)
        except Exception as exc:
            print(f"[WARN] Could not read {path}: {exc}")
    return tables


def format_table(df: pd.DataFrame, n: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    cols = [
        c
        for c in [
            "analysis",
            "sensitivity",
            "feature_space",
            "Group",
            "metric",
            "metric_z",
            "clinical_score",
            "clinical_score_z",
            "scr_index",
            "term",
            "estimate",
            "ci_low",
            "ci_high",
            "p",
            "q",
            "n",
            "n_clinical_outliers_removed",
            "n_metric_outliers_removed",
            "status",
        ]
        if c in df.columns
    ]
    view = df[cols].copy()
    if "p" in view.columns:
        view = view.sort_values(["p"], na_position="last")
    for col in ["estimate", "ci_low", "ci_high", "p", "q"]:
        if col in view.columns:
            view[col] = pd.to_numeric(view[col], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
    return _markdown_table(view.head(n))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-dir", type=Path, default=Path("outputs/mvpa_l2/stats"))
    parser.add_argument("--out", type=Path, default=Path("outputs/mvpa_l2/stats/mvpa_l2_results_summary.md"))
    args = parser.parse_args()

    tables = read_tables(args.stats_dir)
    lines = ["# MVPA L2 Results Summary", ""]
    if not tables:
        lines.append("No CSV result tables found.")
    for name, table in tables.items():
        lines.extend([f"## {name}", "", format_table(table), ""])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"Wrote summary -> {args.out}")


if __name__ == "__main__":
    main()
