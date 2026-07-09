#!/usr/bin/env python3
from __future__ import annotations

"""Generate Figure S2 Aim 2 sensitivity heatmaps."""

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "metric_family",
    "metric_name",
    "sensitivity_type",
    "specification",
    "n_sad",
    "n_hc",
    "effect_size",
    "ci_low",
    "ci_high",
    "p_value",
    "q_value",
]

METRIC_ORDER = [
    ("Geometry", "Neural_Dist_Safety_Background"),
    ("Geometry", "Neural_Dist_Threat_Safety"),
    ("Certainty", "Neural_SafetyEvidence"),
    ("Certainty", "Neural_ThreatEvidence"),
    ("Trajectory", "Neural_Safety_Trajectory_Slope"),
    ("Trajectory", "Neural_Threat_Trajectory_Slope"),
    ("Secondary", "Neural_Dist_Threat_Background"),
    ("Secondary", "Neural_Decoder_Entropy_CSS"),
    ("Secondary", "Neural_Decoder_Entropy_CSR"),
    ("Secondary", "Shock_Anchor_Trajectory_Slope"),
    ("Secondary", "Residualized_Shock_Anchor_Trajectory_Slope"),
]

MASK_ORDER = ["FearNetwork", "MemoryFearNetwork", "Schaefer", "Tian", "WholeBrain"]
SUBGROUP_ORDER = [
    "AllPlacebo",
    "SCR_Physiological_Responder",
    "SCR_Simple_Acquisition_Differential_Learner",
    "SCR_Habituation_Adjusted_Learner",
    "SCR_Late_Phase_Sensitivity_Learner",
]

DISPLAY_LABELS = {
    "AllPlacebo": "All placebo",
    "SCR_Physiological_Responder": "Physiological\nresponder",
    "SCR_Simple_Acquisition_Differential_Learner": "Simple acquisition\nlearner",
    "SCR_Habituation_Adjusted_Learner": "Habituation adjusted\nlearner",
    "SCR_Late_Phase_Sensitivity_Learner": "Late-phase sensitivity\nlearner",
}


METRIC_FAMILY = {metric: family for family, metric in METRIC_ORDER}


def is_summary_format(data: pd.DataFrame) -> bool:
    return all(col in data.columns for col in REQUIRED_COLUMNS)


def is_pipeline_sensitivity_format(data: pd.DataFrame) -> bool:
    required = {"analysis", "sensitivity", "metric", "estimate", "ci_low", "ci_high", "p", "q"}
    return required.issubset(data.columns)


def convert_pipeline_sensitivity(data: pd.DataFrame, source: Path) -> pd.DataFrame:
    """Convert stats/sensitivity_models_all.csv rows into Figure S2 input rows."""
    if not is_pipeline_sensitivity_format(data):
        missing = sorted({"analysis", "sensitivity", "metric", "estimate", "ci_low", "ci_high", "p", "q"} - set(data.columns))
        raise ValueError(f"{source} is missing pipeline sensitivity columns: {', '.join(missing)}")

    aim2 = data[data["analysis"].astype(str).eq("Sensitivity_Aim2_Group")].copy()
    if aim2.empty:
        raise ValueError(f"{source} contains no Sensitivity_Aim2_Group rows")

    sensitivity = aim2["sensitivity"].astype(str)
    aim2["sensitivity_type"] = np.select(
        [sensitivity.str.startswith("FeatureSpace:"), sensitivity.str.startswith("SCRCohort:")],
        ["Mask", "Subgroup"],
        default=None,
    )
    aim2["specification"] = sensitivity.str.replace(r"^(FeatureSpace|SCRCohort):", "", regex=True)
    aim2 = aim2[aim2["sensitivity_type"].notna()].copy()
    if aim2.empty:
        raise ValueError(f"{source} has Aim 2 rows, but none use FeatureSpace: or SCRCohort: sensitivity labels")

    out = pd.DataFrame(
        {
            "metric_family": aim2["metric"].map(METRIC_FAMILY).fillna("Secondary"),
            "metric_name": aim2["metric"],
            "sensitivity_type": aim2["sensitivity_type"],
            "specification": aim2["specification"],
            "n_sad": np.nan,
            "n_hc": np.nan,
            "effect_size": aim2["estimate"],
            "ci_low": aim2["ci_low"],
            "ci_high": aim2["ci_high"],
            "p_value": aim2["p"],
            "q_value": aim2["q"],
        }
    )
    return out[REQUIRED_COLUMNS]


def normalize_input_data(data: pd.DataFrame, source: Path) -> pd.DataFrame:
    if is_summary_format(data):
        return data[REQUIRED_COLUMNS].copy()
    if is_pipeline_sensitivity_format(data):
        return convert_pipeline_sensitivity(data, source)
    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def load_or_create_data(input_path: Path, fallback_paths: Sequence[Path]) -> Tuple[pd.DataFrame, Path | None]:
    """Load Aim 2 sensitivity results from the current pipeline or legacy table."""
    for path in [input_path, *fallback_paths]:
        if path.exists() and path.stat().st_size > 0:
            data = pd.read_csv(path)
            if not data.empty:
                return normalize_input_data(data, path), path
    return create_mock_data(), None


def create_mock_data() -> pd.DataFrame:
    """Create visibly mock data only when no real input table is available."""
    rng = np.random.default_rng(20240622)
    rows = []
    for family, metric in METRIC_ORDER[:6]:
        for sensitivity_type, specifications in [("Mask", MASK_ORDER), ("Subgroup", SUBGROUP_ORDER)]:
            for specification in specifications:
                effect = float(rng.normal(0.0, 0.08))
                rows.append(
                    {
                        "metric_family": family,
                        "metric_name": metric,
                        "sensitivity_type": sensitivity_type,
                        "specification": specification,
                        "n_sad": 10,
                        "n_hc": 10,
                        "effect_size": effect,
                        "ci_low": effect - 0.4,
                        "ci_high": effect + 0.4,
                        "p_value": 0.99,
                        "q_value": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def complete_summary_columns(data: pd.DataFrame, near_zero: float = 0.10) -> pd.DataFrame:
    out = data.copy()
    for col in ["effect_size", "ci_low", "ci_high", "p_value", "q_value"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["direction"] = np.select(
        [out["effect_size"] > near_zero, out["effect_size"] < -near_zero],
        ["SAD > HC", "SAD < HC"],
        default="Near zero",
    )
    out["robustness_label"] = np.select(
        [
            out["q_value"] < 0.05,
            (out["p_value"] < 0.05) & ~(out["q_value"] < 0.05),
            out["effect_size"].abs() >= near_zero,
        ],
        ["FDR significant", "Nominal", "Direction only"],
        default="Weak / inconsistent",
    )
    return out


def ordered_metrics(data: pd.DataFrame) -> List[Tuple[str, str]]:
    present = set(zip(data["metric_family"].astype(str), data["metric_name"].astype(str)))
    ordered = [item for item in METRIC_ORDER if item in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def prepare_heatmap_matrix(
    data: pd.DataFrame,
    sensitivity_type: str,
    specifications: Sequence[str],
    metrics: Sequence[Tuple[str, str]],
) -> pd.DataFrame:
    sub = data[data["sensitivity_type"].astype(str).eq(sensitivity_type)].copy()
    sub["metric_key"] = list(zip(sub["metric_family"].astype(str), sub["metric_name"].astype(str)))
    matrix = sub.pivot_table(
        index="metric_key",
        columns="specification",
        values="effect_size",
        aggfunc="first",
    )
    return matrix.reindex(index=list(metrics), columns=list(specifications))


def make_annotation_matrix(
    data: pd.DataFrame,
    sensitivity_type: str,
    specifications: Sequence[str],
    metrics: Sequence[Tuple[str, str]],
) -> pd.DataFrame:
    sub = data[data["sensitivity_type"].astype(str).eq(sensitivity_type)].copy()
    sub["metric_key"] = list(zip(sub["metric_family"].astype(str), sub["metric_name"].astype(str)))
    sub["symbol"] = np.select(
        [sub["q_value"] < 0.05, (sub["p_value"] < 0.05) & ~(sub["q_value"] < 0.05)],
        ["\u2020", "*"],
        default="",
    )
    sub["annotation"] = sub.apply(
        lambda row: "" if pd.isna(row["effect_size"]) else f"{row['effect_size']:+.2f}{row['symbol']}",
        axis=1,
    )
    matrix = sub.pivot_table(
        index="metric_key",
        columns="specification",
        values="annotation",
        aggfunc="first",
    )
    return matrix.reindex(index=list(metrics), columns=list(specifications)).fillna("")


def metric_labels(metrics: Iterable[Tuple[str, str]]) -> List[str]:
    return [metric for _, metric in metrics]


def display_labels(values: Iterable[object]) -> List[str]:
    return [DISPLAY_LABELS.get(str(value), str(value)) for value in values]


def divider_positions(metrics: Sequence[Tuple[str, str]]) -> List[int]:
    positions = []
    for i in range(1, len(metrics)):
        if metrics[i][0] != metrics[i - 1][0]:
            positions.append(i)
    return positions


def plot_heatmap_panel(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    annotations: pd.DataFrame,
    title: str,
    metrics: Sequence[Tuple[str, str]],
    vlim: float,
):
    masked = np.ma.masked_invalid(matrix.to_numpy(dtype=float))
    image = ax.imshow(masked, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(display_labels(matrix.columns), rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(metric_labels(metrics), fontsize=8)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    for pos in divider_positions(metrics):
        ax.axhline(pos - 0.5, color="black", linewidth=0.8)
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            text = annotations.iat[y, x]
            if text:
                ax.text(x, y, text, ha="center", va="center", fontsize=7, color="black")
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def make_figure(data: pd.DataFrame, output_png: Path, output_svg: Path, mock: bool = False) -> None:
    metrics = ordered_metrics(data)
    mask_matrix = prepare_heatmap_matrix(data, "Mask", MASK_ORDER, metrics)
    subgroup_matrix = prepare_heatmap_matrix(data, "Subgroup", SUBGROUP_ORDER, metrics)
    mask_annotations = make_annotation_matrix(data, "Mask", MASK_ORDER, metrics)
    subgroup_annotations = make_annotation_matrix(data, "Subgroup", SUBGROUP_ORDER, metrics)

    max_abs = np.nanmax(np.abs(pd.concat([mask_matrix, subgroup_matrix], axis=1).to_numpy(dtype=float)))
    vlim = max(0.5, float(np.ceil(max_abs * 10) / 10)) if np.isfinite(max_abs) else 0.5

    fig, axes = plt.subplots(1, 2, figsize=(14, 8), constrained_layout=False)
    image = plot_heatmap_panel(
        axes[0],
        mask_matrix,
        mask_annotations,
        "A. Across masks / feature spaces",
        metrics,
        vlim,
    )
    plot_heatmap_panel(
        axes[1],
        subgroup_matrix,
        subgroup_annotations,
        "B. Across SCR-defined participant subgroups (pooled drug)",
        metrics,
        vlim,
    )
    axes[1].set_yticklabels([])
    fig.subplots_adjust(left=0.20, right=0.90, bottom=0.26, top=0.88, wspace=0.08)
    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("SAD-HC effect size", fontsize=9)
    title = "Figure S2. Sensitivity analysis of SAD-HC differences in neural representations of vicarious learning"
    if mock:
        title += " (mock data)"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.96)
    fig.text(
        0.20,
        0.04,
        "Positive values indicate SAD > HC; negative values indicate SAD < HC. "
        "\u2020 FDR q < .05; * nominal p < .05.",
        fontsize=9,
    )
    fig.text(
        0.01,
        0.01,
        "SCR-defined subgroup sensitivity checks pool placebo and oxytocin participants within each subgroup because subgroup sample sizes are small.",
        fontsize=8,
    )
    fig.savefig(output_png, dpi=300)
    fig.savefig(output_svg)
    plt.close(fig)


def write_summary_table(data: pd.DataFrame, output_csv: Path) -> None:
    summary = complete_summary_columns(data)
    summary = summary[[*REQUIRED_COLUMNS, "direction", "robustness_label"]]
    summary.to_csv(output_csv, index=False)


def default_fallback_paths(input_path: Path) -> List[Path]:
    return [
        input_path.with_name("sensitivity_models_all.csv"),
        Path("outputs/mvpa_l2/stats/sensitivity_models_all.csv"),
        Path("results/outputs/mvpa_l2/stats/sensitivity_models_all.csv"),
        Path("aim2_sensitivity_results.csv"),
        input_path.with_name("TableS2_Aim2_Sensitivity_Stats.csv"),
        Path("TableS2_Aim2_Sensitivity_Stats.csv"),
        Path("code/TableS2_Aim2_Sensitivity_Stats.csv"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/mvpa_l2/stats/sensitivity_models_all.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument("--table-dir", type=Path, default=None)
    args = parser.parse_args()

    figure_dir = args.figure_dir or args.out_dir
    table_dir = args.table_dir or args.out_dir
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    data, source = load_or_create_data(args.input, default_fallback_paths(args.input))
    data = complete_summary_columns(data)
    output_csv = table_dir / "TableS2_Aim2_Sensitivity_Stats.csv"
    output_png = figure_dir / "FigureS2_Aim2_Sensitivity_RobustnessHeatmap.png"
    output_svg = figure_dir / "FigureS2_Aim2_Sensitivity_RobustnessHeatmap.svg"
    write_summary_table(data, output_csv)
    make_figure(data, output_png, output_svg, mock=source is None)
    source_label = "deterministic mock data" if source is None else str(source)
    print(f"Loaded {len(data)} rows from {source_label}")
    print(f"Wrote {output_png}")
    print(f"Wrote {output_svg}")
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
