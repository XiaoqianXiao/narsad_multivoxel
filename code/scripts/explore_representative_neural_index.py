#!/usr/bin/env python3
"""Explore representative neural indices for vicarious fear/safety learning.

The group-level LSS inputs are trial-level voxel matrices. This script derives
compact, subject-level indices from cue pattern geometry and ranks them by how
well they reflect SAD-vs-HC differences, especially in the placebo session.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


CS_LABELS = ("CS-", "CSS", "CSR")
SHOCK_LABEL = "SHOCK"
EPS = 1e-8


INPUT_FILES = {
    "phase2_ext_roi": "phase2_X_ext_y_ext_roi_voxels.npz",
    "phase2_ext_memory_fear_network": "phase2_X_ext_y_ext_roi_voxels_MemoryFearNetwork.npz",
    "phase2_ext_schaefer_tian": "phase2_X_ext_y_ext_voxels_schaefer_tian.npz",
    "phase3_reinst_roi": "phase3_X_reinst_y_reinst_roi_voxels.npz",
    "phase3_reinst_memory_fear_network": "phase3_X_reinst_y_reinst_roi_voxels_MemoryFearNetwork.npz",
    "phase3_reinst_schaefer_tian": "phase3_X_reinst_y_reinst_voxels_schaefer_tian.npz",
}


PROFILE_ORDER = {
    "Q1_geometry_topology": 1,
    "Q2_decision_evidence": 2,
    "Q3_learning_dynamics": 3,
    "Q4_precision_dispersion": 4,
    "Q5_activation_magnitude_secondary": 5,
    "Q6_shock_anchor_secondary": 6,
}


def metric_profile(metric: str) -> str:
    """Map derived metrics back to the original analysis logic."""
    if "Shock" in metric:
        return "Q6_shock_anchor_secondary"
    if "RawMean" in metric or "RawNorm" in metric:
        return "Q5_activation_magnitude_secondary"
    if metric in {"Prototype_Certainty", "Neural_Certainty_CSS", "Neural_Certainty_CSR"}:
        return "Q2_decision_evidence"
    if any(token in metric for token in ["Trajectory", "EarlyLate", "InitialFinal", "Learning", "DynamicDiscrimination", "Volatility", "LatePhase"]):
        return "Q3_learning_dynamics"
    if "Dispersion" in metric or "Precision" in metric or "Certainty" in metric or "Entropy" in metric or "Margin" in metric:
        return "Q4_precision_dispersion"
    if "Evidence" in metric or "Boundary" in metric:
        return "Q2_decision_evidence"
    return "Q1_geometry_topology"


def metric_family(metric: str) -> str:
    if "ResidualizedShockAxis" in metric:
        return "residualized_shock_axis"
    if "ShockEvidence" in metric and "Discrimination" in metric:
        return "shock_evidence_discrimination"
    if "ShockEvidence" in metric:
        return "shock_prototype_evidence"
    if "Shock_Anchor" in metric:
        return "shock_anchor_geometry"
    if "Shock_Dist" in metric:
        return "shock_pairwise_geometry"
    if "EarlyLate" in metric:
        return "early_late_change"
    if "InitialFinal" in metric:
        return "initial_final_change"
    if "LatePhase" in metric:
        return "late_phase_evidence"
    if "LearningAUC" in metric or "DynamicDiscriminationAUC" in metric:
        return "trajectory_auc"
    if "Volatility" in metric:
        return "trajectory_volatility"
    if "Certainty" in metric or "Entropy" in metric or "Margin" in metric:
        return "representational_certainty"
    if "DynamicDiscrimination" in metric:
        return "dynamic_cue_discrimination"
    if "CV_" in metric:
        return "split_half_geometry"
    if "Ratio" in metric:
        return "normalized_geometry_ratio"
    if "Triangle" in metric:
        return "geometry_triangle"
    if "ThreatAxis" in metric:
        return "axis_projection_geometry"
    if "VicariousDiscrimination" in metric:
        return "composite_geometry"
    if "Dist_" in metric or "Similarity" in metric or "Specificity" in metric:
        return "pairwise_geometry"
    if "Evidence" in metric:
        return "prototype_evidence"
    if "Boundary" in metric:
        return "decision_boundary"
    if "Trajectory" in metric:
        return "trialwise_slope"
    if "Dispersion" in metric:
        return "within_cue_dispersion"
    if "Precision" in metric:
        return "within_cue_precision"
    if "RawMean" in metric:
        return "raw_activation_mean"
    if "RawNorm" in metric:
        return "raw_pattern_norm"
    return "other"


def metric_interpretation(metric: str) -> str:
    descriptions = {
        "Neural_Safety_Differentiation": "Threat-background minus safety-background distance; positive values indicate threat is more differentiated from CS- than safety is.",
        "Neural_Threat_Safety_Distance": "Normalized threat-background minus safety-background distance; positive values indicate threat is more differentiated from CS- than safety is, scaled by the total CS-anchored distance.",
        "Neural_ThreatTriangleOpenness": "Whether threat is farther from background than safety is; positive values indicate an open threat-safety-background geometry.",
        "Neural_VicariousDiscrimination": "Composite separation of threat from safety/background after accounting for safety-background distance.",
        "Neural_ThreatToBackgroundDistanceRatio": "Ratio of threat-background distance to safety-background distance.",
        "Neural_ThreatToSafetyDistanceRatio": "Ratio of threat-safety distance to safety-background distance.",
        "Neural_ThreatAxisSeparation": "How far safety sits away from the threat end of the background-to-threat axis.",
        "Neural_Safety_ThreatAxisProjection": "Projection of safety onto the background-to-threat axis; higher values mean safety is more threat-like.",
        "Neural_Dist_Threat_Background": "Pattern distance between reinforced/threat cue and background cue.",
        "Neural_Dist_Safety_Background": "Pattern distance between safety cue and background cue.",
        "Neural_Dist_Threat_Safety": "Pattern distance between reinforced/threat cue and safety cue.",
        "Neural_ThreatEvidence": "Prototype evidence that threat trials look threat-like rather than background-like.",
        "Neural_SafetyEvidence": "Prototype evidence that safety trials look background-like rather than threat-like.",
        "Neural_Certainty_CSS": "Certainty of expected safety-state evidence for CSS relative to maximal ambiguity.",
        "Neural_Certainty_CSR": "Certainty of expected threat-state evidence for CSR relative to maximal ambiguity.",
        "Prototype_Certainty": "Mean certainty of expected safety evidence for CSS and expected threat evidence for CSR.",
        "Neural_BoundarySeparation": "Difference between threat evidence for CSR and threat evidence for CSS.",
        "Neural_PrototypeEvidenceMargin": "Mean signed distance from ambiguous prototype evidence for safety and threat cues.",
        "Neural_PrototypeEvidenceCertainty": "Mean absolute distance from ambiguous prototype evidence for safety and threat cues.",
        "Neural_PrototypeEvidenceEntropy": "Mean binary entropy of safety and threat prototype evidence; lower values indicate more certain evidence.",
        "Neural_TrialwiseEvidenceMargin": "Mean signed trialwise distance from the safety/threat decision boundary.",
        "Neural_TrialwiseEvidenceCertainty": "Mean absolute trialwise distance from the safety/threat decision boundary.",
        "Neural_TrialwiseEvidenceCertaintySNR": "Trialwise evidence margin divided by trialwise evidence variability.",
        "Neural_BoundaryCertainty": "Absolute separation between prototype threat evidence for CSR and CSS.",
        "Neural_Safety_Trajectory_Slope": "Trialwise change in safety evidence across extinction/reinstatement trials.",
        "Neural_Threat_Trajectory_Slope": "Trialwise change in threat evidence across extinction/reinstatement trials.",
        "Neural_Safety_EarlyLate_Change": "Late-minus-early change in safety evidence; positive values indicate safety becomes more background-like over trials.",
        "Neural_Threat_EarlyLate_Change": "Late-minus-early change in threat evidence; negative values indicate declining threat evidence over trials.",
        "Neural_DynamicDiscrimination_EarlyLate_Change": "Late-minus-early change in threat-vs-safety evidence separation.",
        "Neural_Safety_LatePhaseEvidence": "Mean safety evidence during the last half of extinction trials.",
        "Neural_Threat_LatePhaseEvidence": "Mean threat evidence during the last half of extinction trials.",
        "Neural_DynamicDiscrimination_LatePhase": "Late-phase separation between threat evidence and safety evidence.",
        "Neural_Safety_LearningAUC": "Mean safety evidence across extinction trials.",
        "Neural_Threat_LearningAUC": "Mean threat evidence across extinction trials.",
        "Neural_DynamicDiscriminationAUC": "Mean trialwise threat-vs-safety evidence separation across extinction trials.",
        "Neural_Safety_Volatility": "Trial-to-trial instability in safety evidence.",
        "Neural_Threat_Volatility": "Trial-to-trial instability in threat evidence.",
        "Neural_SafetyVsBackgroundDispersion": "Whether safety patterns are more variable around their prototype than background patterns.",
        "Neural_ThreatVsBackgroundDispersion": "Whether threat patterns are more variable around their prototype than background patterns.",
        "Neural_ShockEvidence_Discrimination": "Difference in shock-prototype evidence for threat versus safety cues; positive values mean CSR is more shock-like than CSS.",
        "Neural_ShockEvidence_CSR": "Prototype evidence that CSR patterns look shock-like rather than background-like.",
        "Neural_ShockEvidence_CSS": "Prototype evidence that CSS patterns look shock-like rather than background-like.",
        "Neural_ShockEvidence_DiscriminationAUC": "Mean trialwise CSR-minus-CSS shock evidence across reinstatement.",
        "Neural_ShockEvidence_Discrimination_Slope": "Trialwise change in CSR-minus-CSS shock evidence across reinstatement.",
        "Neural_ShockEvidence_Discrimination_Volatility": "Trial-to-trial instability in CSR-minus-CSS shock evidence.",
        "Neural_Shock_Anchor_ThreatMinusSafety_Proximity": "Shock-anchor proximity contrast; positive values mean CSR is closer to SHOCK than CSS is.",
        "Neural_Shock_Anchor_ThreatMinusBackground_Proximity": "Threat-cue proximity to SHOCK relative to CS-; positive values mean CSR is closer to SHOCK than to CS-.",
        "Neural_Shock_Anchor_SafetySpecificity": "Safety-cue specificity away from SHOCK; positive values mean CSS is closer to CS- than to SHOCK.",
        "Neural_ResidualizedShockAxis_CSRMinusCSS_Cosine": "Global-amplitude-residualized cosine contrast along each subject's SHOCK-minus-CS- axis.",
        "Neural_ResidualizedShockAxis_CSRMinusCSS_Projection": "Global-amplitude-residualized projection contrast along each subject's SHOCK-minus-CS- axis.",
        "Neural_ResidualizedShockAxis_DiscriminationAUC": "Mean trialwise residualized shock-axis contrast for CSR versus CSS.",
        "Neural_ResidualizedShockAxis_Discrimination_Slope": "Trialwise change in residualized shock-axis contrast for CSR versus CSS.",
    }
    return descriptions.get(metric, "Derived FearNetwork profile metric.")


def annotate_rankings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["profile"] = out["metric"].map(metric_profile)
    out["profile_order"] = out["profile"].map(PROFILE_ORDER)
    out["metric_family"] = out["metric"].map(metric_family)
    out["interpretation"] = out["metric"].map(metric_interpretation)
    out["direction_summary"] = np.where(
        out["diff_SAD_minus_HC"] > 0,
        "SAD higher than HC",
        "SAD lower than HC",
    )
    out["passes_nominal_p05"] = pd.to_numeric(out["p"], errors="coerce") < 0.05
    out["passes_family_q05"] = pd.to_numeric(out["q_within_phase_feature"], errors="coerce") < 0.05
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "/Users/xiaoqianxiao/projects/NARSAD/MRI/derivatives/fMRI_analysis/"
            "LSS/firstLevel/all_subjects/group_level"
        ),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("/Users/xiaoqianxiao/projects/NARSAD/MRI/source_data/behav/drug_order.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/representative_neural_index"),
    )
    return parser.parse_args()


def normalize_subject(value: object) -> str:
    text = str(value).strip()
    if text.startswith("sub-"):
        text = text[4:]
    return text


def zscore_subject_trials(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd[~np.isfinite(sd) | (sd < EPS)] = 1.0
    out = (x - mu) / sd
    return np.nan_to_num(out, copy=False)


def corr_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < EPS:
        return np.nan
    corr = float(np.dot(a, b) / denom)
    return 1.0 - float(np.clip(corr, -1.0, 1.0))


def corr_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dist = corr_distance(a, b)
    if not np.isfinite(dist):
        return np.nan
    return 1.0 - dist


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Dimension-normalized Euclidean distance between two centroid vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return np.nan
    return float(np.linalg.norm(a - b) / math.sqrt(a.size))


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) < EPS:
        return np.nan
    return float(numerator / denominator)


def triangle_area_from_sides(a: float, b: float, c: float) -> float:
    """Heron area for side lengths; returns NaN for invalid/degenerated triples."""
    if not all(np.isfinite(value) and value > 0 for value in [a, b, c]):
        return np.nan
    s = 0.5 * (a + b + c)
    area_sq = s * (s - a) * (s - b) * (s - c)
    if area_sq < 0:
        return np.nan
    return float(math.sqrt(max(area_sq, 0.0)))


def triangle_angle(opposite: float, side_a: float, side_b: float) -> float:
    """Law-of-cosines angle in degrees."""
    if not all(np.isfinite(value) and value > 0 for value in [opposite, side_a, side_b]):
        return np.nan
    denom = 2.0 * side_a * side_b
    if denom < EPS:
        return np.nan
    cos_value = (side_a**2 + side_b**2 - opposite**2) / denom
    return float(np.degrees(np.arccos(np.clip(cos_value, -1.0, 1.0))))


def softmax_threat_evidence(d_background: float, d_threat: float) -> float:
    if not np.isfinite(d_background) or not np.isfinite(d_threat):
        return np.nan
    scores = np.array([-d_background, -d_threat], dtype=float)
    scores -= scores.max()
    probs = np.exp(scores) / np.exp(scores).sum()
    return float(probs[1])


def heldout_decoder_evidence(x: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    """Compute held-out P(CSR) values for CSS/CSR trials in their observed order."""
    mask = np.isin(y, ["CSS", "CSR"])
    x_bin = np.asarray(x[mask], dtype=float)
    y_bin = np.asarray(y[mask])
    if len(y_bin) < 10 or len(np.unique(y_bin)) < 2:
        return {}
    if min(np.sum(y_bin == "CSS"), np.sum(y_bin == "CSR")) < 2:
        return {}

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classification",
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )
    probs_threat = np.full(len(y_bin), np.nan, dtype=float)
    try:
        for train_idx, test_idx in LeaveOneGroupOut().split(x_bin, y_bin, groups=np.arange(len(y_bin))):
            if len(np.unique(y_bin[train_idx])) < 2:
                continue
            model.fit(x_bin[train_idx], y_bin[train_idx])
            if "CSR" not in model.classes_:
                continue
            threat_idx = np.where(model.classes_ == "CSR")[0][0]
            probs_threat[test_idx] = model.predict_proba(x_bin[test_idx])[:, threat_idx]
    except Exception:
        return {}

    probs_css = probs_threat[y_bin == "CSS"]
    probs_csr = probs_threat[y_bin == "CSR"]
    probs_css = probs_css[np.isfinite(probs_css)]
    probs_csr = probs_csr[np.isfinite(probs_csr)]
    if len(probs_css) == 0 or len(probs_csr) == 0:
        return {}

    p_threat_css = float(np.mean(probs_css))
    p_safety_css = 1.0 - p_threat_css
    p_threat_csr = float(np.mean(probs_csr))
    safety_evidence = 1.0 - probs_threat[y_bin == "CSS"]
    threat_evidence = probs_threat[y_bin == "CSR"]
    n_dynamic = min(len(safety_evidence), len(threat_evidence))
    dynamic_discrimination = (
        np.asarray(threat_evidence[:n_dynamic], dtype=float) - np.asarray(safety_evidence[:n_dynamic], dtype=float)
        if n_dynamic
        else np.asarray([], dtype=float)
    )
    return {
        "p_threat_css": p_threat_css,
        "p_safety_css": p_safety_css,
        "p_threat_csr": p_threat_csr,
        "probs_css": probs_css,
        "probs_csr": probs_csr,
        "dynamic_discrimination": dynamic_discrimination,
    }


def binary_entropy(p: float) -> float:
    if not np.isfinite(p):
        return np.nan
    p = float(np.clip(p, EPS, 1.0 - EPS))
    return float(-(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)))


def standardize_margin(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return np.nan
    sd = float(arr.std(ddof=1))
    if sd < EPS:
        return np.nan
    return float(arr.mean() / sd)


def slope(values: Iterable[float]) -> float:
    y = np.asarray(list(values), dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    x = np.arange(1, len(y) + 1, dtype=float)
    return float(stats.linregress(x[ok], y[ok]).slope)


def mean_valid(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(arr.mean())


def early_late_change(values: Iterable[float], half: int = 4) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size < half * 2:
        return np.nan
    return mean_valid(arr[-half:]) - mean_valid(arr[:half])


def initial_final_change(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return np.nan
    return float(arr[-1] - arr[0])


def late_phase_mean(values: Iterable[float], half: int = 4) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size < half:
        return np.nan
    return mean_valid(arr[-half:])


def rmssd(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return np.nan
    return float(np.sqrt(np.mean(np.diff(arr) ** 2)))


def split_half_centroids(xz: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    first = {}
    second = {}
    for label in CS_LABELS:
        rows = xz[y == label]
        first[label] = rows[::2].mean(axis=0)
        second[label] = rows[1::2].mean(axis=0)
    return first, second


def cv_corr_distance(
    first: Dict[str, np.ndarray],
    second: Dict[str, np.ndarray],
    label_a: str,
    label_b: str,
) -> float:
    return float(
        np.nanmean(
            [
                corr_distance(first[label_a], second[label_b]),
                corr_distance(second[label_a], first[label_b]),
            ]
        )
    )


def mean_distance_to_centroid(rows: np.ndarray, centroid: np.ndarray) -> float:
    values = [corr_distance(row, centroid) for row in rows]
    return float(np.nanmean(values))


def normalized_axis_projection(point: np.ndarray, origin: np.ndarray, target: np.ndarray) -> float:
    axis = target - origin
    denom = float(np.dot(axis, axis))
    if denom < EPS:
        return np.nan
    return float(np.dot(point - origin, axis) / denom)


def unit_vector(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm < EPS:
        return np.full_like(vec, np.nan, dtype=float)
    return vec / norm


def axis_projection_metrics(vec: np.ndarray, origin: np.ndarray, axis_unit: np.ndarray) -> Tuple[float, float]:
    if not np.all(np.isfinite(axis_unit)):
        return np.nan, np.nan
    delta = vec - origin
    delta_norm = float(np.linalg.norm(delta))
    projection = float(np.dot(delta, axis_unit))
    cosine = np.nan if delta_norm < EPS else float(projection / delta_norm)
    return projection, cosine


def subject_indices(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    xz = zscore_subject_trials(x)
    centroids = {label: xz[y == label].mean(axis=0) for label in CS_LABELS}
    first, second = split_half_centroids(xz, y)

    d_safety_background = corr_distance(centroids["CSS"], centroids["CS-"])
    d_threat_safety = corr_distance(centroids["CSR"], centroids["CSS"])
    d_threat_background = corr_distance(centroids["CSR"], centroids["CS-"])
    e_safety_background = euclidean_distance(centroids["CSS"], centroids["CS-"])
    e_threat_safety = euclidean_distance(centroids["CSR"], centroids["CSS"])
    e_threat_background = euclidean_distance(centroids["CSR"], centroids["CS-"])
    cv_d_safety_background = cv_corr_distance(first, second, "CSS", "CS-")
    cv_d_threat_safety = cv_corr_distance(first, second, "CSR", "CSS")
    cv_d_threat_background = cv_corr_distance(first, second, "CSR", "CS-")
    corr_triangle_perimeter = d_safety_background + d_threat_safety + d_threat_background
    corr_triangle_mean_edge = corr_triangle_perimeter / 3.0
    corr_triangle_area = triangle_area_from_sides(
        d_safety_background,
        d_threat_safety,
        d_threat_background,
    )
    euclid_triangle_perimeter = e_safety_background + e_threat_safety + e_threat_background
    euclid_triangle_mean_edge = euclid_triangle_perimeter / 3.0
    euclid_triangle_area = triangle_area_from_sides(
        e_safety_background,
        e_threat_safety,
        e_threat_background,
    )

    p_proto_threat_css = softmax_threat_evidence(
        corr_distance(centroids["CSS"], centroids["CS-"]),
        corr_distance(centroids["CSS"], centroids["CSR"]),
    )
    p_proto_threat_csr = softmax_threat_evidence(
        corr_distance(centroids["CSR"], centroids["CS-"]),
        corr_distance(centroids["CSR"], centroids["CSR"]),
    )
    decoder_evidence = heldout_decoder_evidence(xz, y)
    p_threat_css = decoder_evidence.get("p_threat_css", np.nan)
    p_safety_css = decoder_evidence.get("p_safety_css", np.nan)
    p_threat_csr = decoder_evidence.get("p_threat_csr", np.nan)
    prototype_margins = [p_safety_css - 0.5, p_threat_csr - 0.5]
    prototype_certainty = [abs(value) * 2.0 for value in prototype_margins if np.isfinite(value)]

    safety_contrast = []
    threat_contrast = []
    for label, holder in (("CSS", safety_contrast), ("CSR", threat_contrast)):
        for trial in xz[y == label]:
            sim_bg = corr_similarity(trial, centroids["CS-"])
            sim_threat = corr_similarity(trial, centroids["CSR"])
            holder.append(sim_bg - sim_threat if label == "CSS" else sim_threat - sim_bg)
    prototype_dynamic_discrimination = np.asarray(threat_contrast, dtype=float) - np.asarray(safety_contrast, dtype=float)
    dynamic_discrimination = decoder_evidence.get("dynamic_discrimination", np.asarray([], dtype=float))
    trialwise_margins = np.r_[
        np.asarray(safety_contrast, dtype=float),
        np.asarray(threat_contrast, dtype=float),
    ]

    css_dispersion = mean_distance_to_centroid(xz[y == "CSS"], centroids["CSS"])
    csr_dispersion = mean_distance_to_centroid(xz[y == "CSR"], centroids["CSR"])
    background_dispersion = mean_distance_to_centroid(xz[y == "CS-"], centroids["CS-"])
    mean_csplus_dispersion = float(np.nanmean([css_dispersion, csr_dispersion]))

    css_threat_axis_projection = normalized_axis_projection(centroids["CSS"], centroids["CS-"], centroids["CSR"])
    threat_axis_separation = 1.0 - css_threat_axis_projection

    raw_condition_means = {label: float(np.nanmean(x[y == label])) for label in CS_LABELS}
    raw_condition_norms = {label: float(np.linalg.norm(np.nanmean(x[y == label], axis=0))) for label in CS_LABELS}
    metrics = {
        "Neural_Dist_Safety_Background": d_safety_background,
        "Neural_Dist_Threat_Safety": d_threat_safety,
        "Neural_Dist_Threat_Background": d_threat_background,
        "Neural_CV_Dist_Safety_Background": cv_d_safety_background,
        "Neural_CV_Dist_Threat_Safety": cv_d_threat_safety,
        "Neural_CV_Dist_Threat_Background": cv_d_threat_background,
        "Neural_SafetySimilarity_Background": 1.0 - d_safety_background,
        "Neural_SafetySpecificity": d_threat_safety - d_safety_background,
        "Neural_ThreatVsBackgroundSpecificity": d_threat_background - d_threat_safety,
        "Neural_VicariousDiscrimination": 0.5 * (d_threat_safety + d_threat_background) - d_safety_background,
        "Neural_VicariousDiscrimination_Normalized": safe_ratio(
            0.5 * (d_threat_safety + d_threat_background) - d_safety_background,
            corr_triangle_mean_edge,
        ),
        "Neural_CV_VicariousDiscrimination": 0.5 * (cv_d_threat_safety + cv_d_threat_background) - cv_d_safety_background,
        "Neural_ThreatToSafetyDistanceRatio": d_threat_safety / (d_safety_background + EPS),
        "Neural_ThreatToBackgroundDistanceRatio": d_threat_background / (d_safety_background + EPS),
        "Neural_ThreatTriangleOpenness": d_threat_background - d_safety_background,
        "Neural_ThreatTriangleOpenness_Normalized": safe_ratio(
            d_threat_background - d_safety_background,
            d_threat_background + d_safety_background,
        ),
        "Neural_ThreatSafetyVsBackgroundMean": safe_ratio(
            d_threat_safety,
            0.5 * (d_threat_background + d_safety_background),
        ),
        "Neural_CorrTrianglePerimeter": corr_triangle_perimeter,
        "Neural_CorrTriangleArea": corr_triangle_area,
        "Neural_CorrTriangleArea_Normalized": safe_ratio(corr_triangle_area, corr_triangle_mean_edge**2),
        "Neural_CorrTriangleAngle_Background": triangle_angle(
            d_threat_safety,
            d_safety_background,
            d_threat_background,
        ),
        "Neural_CorrTriangleAngle_Safety": triangle_angle(
            d_threat_background,
            d_safety_background,
            d_threat_safety,
        ),
        "Neural_CorrTriangleAngle_Threat": triangle_angle(
            d_safety_background,
            d_threat_background,
            d_threat_safety,
        ),
        "Neural_Safety_Differentiation": d_threat_background - d_safety_background,
        "Neural_Threat_Safety_Distance": safe_ratio(
            d_threat_background - d_safety_background,
            d_threat_background + d_safety_background,
        ),
        "Neural_Euclid_Dist_Safety_Background": e_safety_background,
        "Neural_Euclid_Dist_Threat_Safety": e_threat_safety,
        "Neural_Euclid_Dist_Threat_Background": e_threat_background,
        "Neural_Euclid_ThreatTriangleOpenness": e_threat_background - e_safety_background,
        "Neural_Euclid_ThreatTriangleOpenness_Normalized": safe_ratio(
            e_threat_background - e_safety_background,
            e_threat_background + e_safety_background,
        ),
        "Neural_Euclid_VicariousDiscrimination": 0.5 * (e_threat_safety + e_threat_background) - e_safety_background,
        "Neural_Euclid_VicariousDiscrimination_Normalized": safe_ratio(
            0.5 * (e_threat_safety + e_threat_background) - e_safety_background,
            euclid_triangle_mean_edge,
        ),
        "Neural_Euclid_ThreatToBackgroundDistanceRatio": safe_ratio(e_threat_background, e_safety_background),
        "Neural_Euclid_ThreatToSafetyDistanceRatio": safe_ratio(e_threat_safety, e_safety_background),
        "Neural_Euclid_TrianglePerimeter": euclid_triangle_perimeter,
        "Neural_Euclid_TriangleArea": euclid_triangle_area,
        "Neural_Euclid_TriangleArea_Normalized": safe_ratio(euclid_triangle_area, euclid_triangle_mean_edge**2),
        "Neural_Euclid_TriangleAngle_Background": triangle_angle(
            e_threat_safety,
            e_safety_background,
            e_threat_background,
        ),
        "Neural_Euclid_TriangleAngle_Safety": triangle_angle(
            e_threat_background,
            e_safety_background,
            e_threat_safety,
        ),
        "Neural_Euclid_TriangleAngle_Threat": triangle_angle(
            e_safety_background,
            e_threat_background,
            e_threat_safety,
        ),
        "Neural_SafetyEvidence": 1.0 - p_threat_css,
        "Neural_ThreatEvidence": p_threat_csr,
        "Neural_BoundarySeparation": p_threat_csr - p_threat_css,
        "Neural_PrototypeThreatLike_Safety": p_proto_threat_css,
        "Neural_PrototypeThreatLike_Threat": p_proto_threat_csr,
        "Neural_PrototypeBoundarySeparation": p_proto_threat_csr - p_proto_threat_css,
        "Neural_BoundaryCertainty": abs(p_threat_csr - p_threat_css),
        "Neural_PrototypeEvidenceMargin": mean_valid(prototype_margins),
        "Neural_PrototypeEvidenceCertainty": mean_valid(prototype_certainty),
        "Neural_PrototypeEvidenceEntropy": mean_valid([binary_entropy(p_safety_css), binary_entropy(p_threat_csr)]),
        "Neural_Certainty_CSS": abs(p_safety_css - 0.5) * 2.0,
        "Neural_Certainty_CSR": abs(p_threat_csr - 0.5) * 2.0,
        "Prototype_Certainty": mean_valid([abs(p_safety_css - 0.5) * 2.0, abs(p_threat_csr - 0.5) * 2.0]),
        "Neural_SafetyEvidenceCertainty": abs(p_safety_css - 0.5) * 2.0,
        "Neural_ThreatEvidenceCertainty": abs(p_threat_csr - 0.5) * 2.0,
        "Neural_TrialwiseEvidenceMargin": mean_valid(trialwise_margins),
        "Neural_TrialwiseEvidenceCertainty": mean_valid(np.abs(trialwise_margins)),
        "Neural_TrialwiseEvidenceCertaintySNR": standardize_margin(trialwise_margins),
        "Neural_Safety_Trajectory_Slope": slope(safety_contrast),
        "Neural_Threat_Trajectory_Slope": slope(threat_contrast),
        "Neural_Safety_EarlyLate_Change": early_late_change(safety_contrast),
        "Neural_Threat_EarlyLate_Change": early_late_change(threat_contrast),
        "Neural_DynamicDiscrimination_EarlyLate_Change": early_late_change(dynamic_discrimination),
        "Neural_PrototypeDynamicDiscrimination_EarlyLate_Change": early_late_change(prototype_dynamic_discrimination),
        "Neural_Safety_InitialFinal_Change": initial_final_change(safety_contrast),
        "Neural_Threat_InitialFinal_Change": initial_final_change(threat_contrast),
        "Neural_DynamicDiscrimination_InitialFinal_Change": initial_final_change(dynamic_discrimination),
        "Neural_PrototypeDynamicDiscrimination_InitialFinal_Change": initial_final_change(prototype_dynamic_discrimination),
        "Neural_Safety_LatePhaseEvidence": late_phase_mean(safety_contrast),
        "Neural_Threat_LatePhaseEvidence": late_phase_mean(threat_contrast),
        "Neural_DynamicDiscrimination_LatePhase": late_phase_mean(dynamic_discrimination),
        "Neural_Safety_LearningAUC": mean_valid(safety_contrast),
        "Neural_Threat_LearningAUC": mean_valid(threat_contrast),
        "Neural_DynamicDiscriminationAUC": mean_valid(dynamic_discrimination),
        "Neural_PrototypeDynamicDiscriminationAUC": mean_valid(prototype_dynamic_discrimination),
        "Neural_Safety_Volatility": rmssd(safety_contrast),
        "Neural_Threat_Volatility": rmssd(threat_contrast),
        "Neural_DynamicDiscrimination_Volatility": rmssd(dynamic_discrimination),
        "Neural_PrototypeDynamicDiscrimination_Volatility": rmssd(prototype_dynamic_discrimination),
        "Neural_Safety_ThreatAxisProjection": css_threat_axis_projection,
        "Neural_ThreatAxisSeparation": threat_axis_separation,
        "Neural_SafetyDispersion": css_dispersion,
        "Neural_ThreatDispersion": csr_dispersion,
        "Neural_BackgroundDispersion": background_dispersion,
        "Neural_CSPlusDispersion": mean_csplus_dispersion,
        "Neural_SafetyPrecision": -css_dispersion,
        "Neural_ThreatPrecision": -csr_dispersion,
        "Neural_CSPlusPrecision": -mean_csplus_dispersion,
        "Neural_SafetyVsBackgroundDispersion": css_dispersion - background_dispersion,
        "Neural_ThreatVsBackgroundDispersion": csr_dispersion - background_dispersion,
        "Neural_RawMean_SafetyMinusBackground": raw_condition_means["CSS"] - raw_condition_means["CS-"],
        "Neural_RawMean_ThreatMinusSafety": raw_condition_means["CSR"] - raw_condition_means["CSS"],
        "Neural_RawMean_ThreatMinusBackground": raw_condition_means["CSR"] - raw_condition_means["CS-"],
        "Neural_RawNorm_SafetyMinusBackground": raw_condition_norms["CSS"] - raw_condition_norms["CS-"],
        "Neural_RawNorm_ThreatMinusSafety": raw_condition_norms["CSR"] - raw_condition_norms["CSS"],
        "Neural_RawNorm_ThreatMinusBackground": raw_condition_norms["CSR"] - raw_condition_norms["CS-"],
    }
    if np.sum(y == SHOCK_LABEL) >= 2:
        shock_centroid = xz[y == SHOCK_LABEL].mean(axis=0)
        d_shock_background = corr_distance(shock_centroid, centroids["CS-"])
        d_shock_safety = corr_distance(shock_centroid, centroids["CSS"])
        d_shock_threat = corr_distance(shock_centroid, centroids["CSR"])

        p_shock_css = softmax_threat_evidence(
            corr_distance(centroids["CSS"], centroids["CS-"]),
            corr_distance(centroids["CSS"], shock_centroid),
        )
        p_shock_csr = softmax_threat_evidence(
            corr_distance(centroids["CSR"], centroids["CS-"]),
            corr_distance(centroids["CSR"], shock_centroid),
        )
        p_shock_shock = softmax_threat_evidence(
            corr_distance(shock_centroid, centroids["CS-"]),
            corr_distance(shock_centroid, shock_centroid),
        )

        css_shock_evidence = []
        csr_shock_evidence = []
        for label, holder in (("CSS", css_shock_evidence), ("CSR", csr_shock_evidence)):
            for trial in xz[y == label]:
                holder.append(
                    softmax_threat_evidence(
                        corr_distance(trial, centroids["CS-"]),
                        corr_distance(trial, shock_centroid),
                    )
                )
        shock_discrimination = np.asarray(csr_shock_evidence, dtype=float) - np.asarray(css_shock_evidence, dtype=float)

        x_resid = xz - np.nanmean(xz, axis=1, keepdims=True)
        resid_centroids = {label: x_resid[y == label].mean(axis=0) for label in CS_LABELS}
        resid_shock = x_resid[y == SHOCK_LABEL].mean(axis=0)
        shock_axis = unit_vector(resid_shock - resid_centroids["CS-"])
        proj_css, cos_css = axis_projection_metrics(resid_centroids["CSS"], resid_centroids["CS-"], shock_axis)
        proj_csr, cos_csr = axis_projection_metrics(resid_centroids["CSR"], resid_centroids["CS-"], shock_axis)
        resid_css = []
        resid_csr = []
        for label, holder in (("CSS", resid_css), ("CSR", resid_csr)):
            for trial in x_resid[y == label]:
                projection, _ = axis_projection_metrics(trial, resid_centroids["CS-"], shock_axis)
                holder.append(projection)
        resid_discrimination = np.asarray(resid_csr, dtype=float) - np.asarray(resid_css, dtype=float)

        metrics.update(
            {
                "Neural_Shock_Dist_Background": d_shock_background,
                "Neural_Shock_Dist_Safety": d_shock_safety,
                "Neural_Shock_Dist_Threat": d_shock_threat,
                "Neural_ShockEvidence_CSS": p_shock_css,
                "Neural_ShockEvidence_CSR": p_shock_csr,
                "Neural_ShockEvidence_SHOCK": p_shock_shock,
                "Neural_ShockEvidence_Discrimination": p_shock_csr - p_shock_css,
                "Neural_ShockEvidence_CSS_AUC": mean_valid(css_shock_evidence),
                "Neural_ShockEvidence_CSR_AUC": mean_valid(csr_shock_evidence),
                "Neural_ShockEvidence_DiscriminationAUC": mean_valid(shock_discrimination),
                "Neural_ShockEvidence_CSS_Slope": slope(css_shock_evidence),
                "Neural_ShockEvidence_CSR_Slope": slope(csr_shock_evidence),
                "Neural_ShockEvidence_Discrimination_Slope": slope(shock_discrimination),
                "Neural_ShockEvidence_CSS_EarlyLate_Change": early_late_change(css_shock_evidence),
                "Neural_ShockEvidence_CSR_EarlyLate_Change": early_late_change(csr_shock_evidence),
                "Neural_ShockEvidence_Discrimination_EarlyLate_Change": early_late_change(shock_discrimination),
                "Neural_ShockEvidence_Discrimination_LatePhase": late_phase_mean(shock_discrimination),
                "Neural_ShockEvidence_Discrimination_Volatility": rmssd(shock_discrimination),
                "Neural_Shock_Anchor_ThreatMinusSafety_Proximity": d_shock_safety - d_shock_threat,
                "Neural_Shock_Anchor_ThreatMinusBackground_Proximity": d_threat_background - d_shock_threat,
                "Neural_Shock_Anchor_SafetyMinusBackground_Proximity": d_safety_background - d_shock_safety,
                "Neural_Shock_Anchor_SafetySpecificity": d_shock_safety - d_safety_background,
                "Neural_Shock_Anchor_Axis_Norm": float(np.linalg.norm(shock_centroid - centroids["CS-"])),
                "Neural_ResidualizedShockAxis_CSS_Projection": proj_css,
                "Neural_ResidualizedShockAxis_CSR_Projection": proj_csr,
                "Neural_ResidualizedShockAxis_CSRMinusCSS_Projection": proj_csr - proj_css,
                "Neural_ResidualizedShockAxis_CSS_Cosine": cos_css,
                "Neural_ResidualizedShockAxis_CSR_Cosine": cos_csr,
                "Neural_ResidualizedShockAxis_CSRMinusCSS_Cosine": cos_csr - cos_css if np.isfinite(cos_css) and np.isfinite(cos_csr) else np.nan,
                "Neural_ResidualizedShockAxis_Axis_Norm": float(np.linalg.norm(resid_shock - resid_centroids["CS-"])),
                "Neural_ResidualizedShockAxis_DiscriminationAUC": mean_valid(resid_discrimination),
                "Neural_ResidualizedShockAxis_Discrimination_Slope": slope(resid_discrimination),
                "Neural_ResidualizedShockAxis_Discrimination_EarlyLate_Change": early_late_change(resid_discrimination),
                "Neural_ResidualizedShockAxis_Discrimination_LatePhase": late_phase_mean(resid_discrimination),
                "Neural_ResidualizedShockAxis_Discrimination_Volatility": rmssd(resid_discrimination),
            }
        )
    return metrics


def load_npz_table(path: Path, feature_space: str) -> pd.DataFrame:
    payload = np.load(path, allow_pickle=True)
    if "X_ext" in payload.files:
        x = payload["X_ext"]
        y = payload["y_ext"].astype(str)
        phase = "phase2_extinction"
    else:
        x = payload["X_reinst"]
        y = payload["y_reinst"].astype(str)
        phase = "phase3_reinstatement"
    subjects = np.asarray([normalize_subject(s) for s in payload["subjects"]])

    rows: List[Dict[str, object]] = []
    for sub_id in sorted(set(subjects)):
        mask = subjects == sub_id
        y_sub = y[mask]
        if not all(np.sum(y_sub == label) >= 3 for label in CS_LABELS):
            continue
        row: Dict[str, object] = {
            "sub_ID": sub_id,
            "phase": phase,
            "feature_space": feature_space,
            "n_trials": int(mask.sum()),
            "n_features": int(x.shape[1]),
        }
        row.update(subject_indices(x[mask], y_sub))
        rows.append(row)
    return pd.DataFrame(rows)


def load_roi_metric_table(path: Path, feature_space: str) -> pd.DataFrame:
    payload = np.load(path, allow_pickle=True)
    if "roi_names" not in payload.files or "roi_voxel_counts" not in payload.files:
        return pd.DataFrame()
    if "X_ext" in payload.files:
        x = payload["X_ext"]
        y = payload["y_ext"].astype(str)
        phase = "phase2_extinction"
    else:
        x = payload["X_reinst"]
        y = payload["y_reinst"].astype(str)
        phase = "phase3_reinstatement"
    subjects = np.asarray([normalize_subject(s) for s in payload["subjects"]])
    roi_names = payload["roi_names"].astype(str)
    roi_counts = payload["roi_voxel_counts"].astype(int)
    starts = np.r_[0, np.cumsum(roi_counts)[:-1]]
    stops = np.cumsum(roi_counts)

    rows: List[Dict[str, object]] = []
    for roi_name, start, stop in zip(roi_names, starts, stops):
        x_roi = x[:, start:stop]
        for sub_id in sorted(set(subjects)):
            mask = subjects == sub_id
            y_sub = y[mask]
            if not all(np.sum(y_sub == label) >= 3 for label in CS_LABELS):
                continue
            row: Dict[str, object] = {
                "sub_ID": sub_id,
                "phase": phase,
                "feature_space": f"{feature_space}_by_roi",
                "roi_name": roi_name,
                "n_trials": int(mask.sum()),
                "n_features": int(stop - start),
            }
            row.update(subject_indices(x_roi[mask], y_sub))
            rows.append(row)
    return pd.DataFrame(rows)


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    if pooled < EPS:
        return np.nan
    return float((a.mean() - b.mean()) / pooled)


def hedges_g(a: pd.Series, b: pd.Series) -> float:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()
    d = cohens_d(a, b)
    df = len(a) + len(b) - 2
    if not np.isfinite(d) or df <= 1:
        return np.nan
    correction = 1.0 - (3.0 / (4.0 * df - 1.0))
    return float(d * correction)


def rank_biserial_sad_vs_hc(values: pd.Series, group: pd.Series) -> float:
    data = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce"), "group": group}).dropna()
    if data["group"].nunique() != 2:
        return np.nan
    y = (data["group"] == "SAD").astype(int)
    auc = roc_auc_score(y, data["value"])
    return float((2.0 * auc) - 1.0)


def scalar_auc(values: pd.Series, group: pd.Series) -> float:
    data = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce"), "group": group}).dropna()
    if data["group"].nunique() != 2:
        return np.nan
    y = (data["group"] == "SAD").astype(int)
    auc = roc_auc_score(y, data["value"])
    return float(max(auc, 1.0 - auc))


def summarize_group_tests(
    df: pd.DataFrame,
    metrics: List[str],
    group_cols: Iterable[str] = ("phase", "feature_space"),
) -> pd.DataFrame:
    rows = []
    group_cols = list(group_cols)
    for group_values, sub in df.groupby(group_cols):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group_context = dict(zip(group_cols, group_values))
        placebo = sub[sub["Drug"] == "Placebo"].copy()
        for metric in metrics:
            sad = placebo.loc[placebo["Group"] == "SAD", metric]
            hc = placebo.loc[placebo["Group"] == "HC", metric]
            if sad.notna().sum() < 5 or hc.notna().sum() < 5:
                continue
            t_res = stats.ttest_ind(sad, hc, equal_var=False, nan_policy="omit")
            row = {
                "metric": metric,
                "analysis": "placebo_SAD_vs_HC",
                "n_SAD": int(sad.notna().sum()),
                "n_HC": int(hc.notna().sum()),
                "mean_SAD": float(sad.mean()),
                "mean_HC": float(hc.mean()),
                "diff_SAD_minus_HC": float(sad.mean() - hc.mean()),
                "cohens_d_SAD_minus_HC": cohens_d(sad, hc),
                "hedges_g_SAD_minus_HC": hedges_g(sad, hc),
                "rank_biserial_SAD_vs_HC": rank_biserial_sad_vs_hc(placebo[metric], placebo["Group"]),
                "t": float(t_res.statistic),
                "p": float(t_res.pvalue),
                "scalar_auc_abs_direction": scalar_auc(placebo[metric], placebo["Group"]),
            }
            row.update(group_context)
            model_df = sub.dropna(subset=[metric, "Group", "Drug", "demo_age", "Gender"]).copy()
            if len(model_df) >= 30 and model_df["Group"].nunique() == 2 and model_df["Drug"].nunique() == 2:
                try:
                    fit = smf.ols(
                        f"Q('{metric}') ~ C(Group, Treatment(reference='HC'))"
                        " * C(Drug, Treatment(reference='Placebo')) + Q('demo_age') + C(Gender)",
                        data=model_df,
                    ).fit()
                    term = "C(Group, Treatment(reference='HC'))[T.SAD]"
                    row["all_subjects_adjusted_group_beta"] = float(fit.params.get(term, np.nan))
                    row["all_subjects_adjusted_group_p"] = float(fit.pvalues.get(term, np.nan))
                    row["all_subjects_adjusted_model_n"] = int(fit.nobs)
                except Exception as exc:
                    row["all_subjects_adjusted_error"] = str(exc)
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_within_phase_feature"] = np.nan
    for _, idx in out.groupby(group_cols, dropna=False).groups.items():
        pvals = out.loc[idx, "p"]
        ok = pvals.notna()
        if ok.any():
            out.loc[pvals[ok].index, "q_within_phase_feature"] = multipletests(pvals[ok], method="fdr_bh")[1]
    log_p_strength = -np.log10(out["p"].clip(lower=1e-300))
    out["rank_score"] = (
        out["cohens_d_SAD_minus_HC"].abs().rank(ascending=True, pct=True)
        + out["scalar_auc_abs_direction"].rank(ascending=True, pct=True)
        + log_p_strength.rank(ascending=True, pct=True)
    )
    return out.sort_values(["rank_score", "p"], ascending=[False, True])


def top_rows_by_profile(rankings: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    if rankings.empty:
        return rankings.copy()
    out = annotate_rankings(rankings)
    out = out.sort_values(
        list(group_cols) + ["profile_order", "rank_score", "p"],
        ascending=[True] * (len(list(group_cols)) + 1) + [False, True],
    )
    return out.groupby(list(group_cols) + ["profile"], as_index=False).head(1)


def top_roi_localization(roi_rankings: pd.DataFrame) -> pd.DataFrame:
    if roi_rankings.empty:
        return roi_rankings.copy()
    out = annotate_rankings(roi_rankings)
    interpretable = ~out["profile"].eq("Q5_activation_magnitude_secondary")
    out = out[interpretable].copy()
    out = out.sort_values(
        ["phase", "profile_order", "rank_score", "p"],
        ascending=[True, True, False, True],
    )
    return out.groupby(["phase", "profile"], as_index=False).head(5)


def write_profile_report(
    path: Path,
    fear_whole: pd.DataFrame,
    profile_top: pd.DataFrame,
    roi_top: pd.DataFrame,
) -> None:
    def markdown_table(df: pd.DataFrame, cols: List[str]) -> str:
        if df.empty:
            return "_No rows available._"
        sub = df.loc[:, cols].copy()
        for col in sub.columns:
            if pd.api.types.is_float_dtype(sub[col]):
                sub[col] = sub[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
            else:
                sub[col] = sub[col].map(lambda x: "" if pd.isna(x) else str(x))
        header = "| " + " | ".join(sub.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(sub.columns)) + " |"
        rows = ["| " + " | ".join(row) + " |" for row in sub.to_numpy(dtype=str)]
        return "\n".join([header, sep] + rows)

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# FearNetwork Neural Profile Exploration",
        "",
        "This report keeps the original analysis structure but expands the neural profile vocabulary within the FearNetwork mask.",
        "",
        "## Profile Domains",
        "",
        "- Q1 geometry/topology: where safety, threat, and background sit in representational space.",
        "- Q2 decision/evidence: whether patterns express safety-like or threat-like evidence.",
        "- Q3 learning dynamics: trialwise change in safety/threat representational evidence.",
        "- Q4 precision/dispersion: within-cue stability of the neural representation.",
        "- Q5 activation/magnitude: raw mean or norm contrasts, treated as secondary because they are less representationally specific.",
        "- Q6 shock-anchor: secondary reinstatement metrics that quantify cue alignment with SHOCK/US while controlling for global-amplitude components where possible.",
        "",
        "## Whole FearNetwork: Best Metric Per Profile",
        "",
    ]

    display_cols = [
        "phase",
        "profile",
        "metric",
        "direction_summary",
        "cohens_d_SAD_minus_HC",
        "hedges_g_SAD_minus_HC",
        "rank_biserial_SAD_vs_HC",
        "p",
        "q_within_phase_feature",
        "scalar_auc_abs_direction",
        "all_subjects_adjusted_group_p",
    ]
    if not profile_top.empty:
        lines.append(markdown_table(profile_top, display_cols))
    else:
        lines.append("_No profile rows available._")

    lines.extend(["", "## ROI Localization: Strongest Interpretable Rows", ""])
    roi_cols = [
        "phase",
        "profile",
        "roi_name",
        "metric",
        "direction_summary",
        "cohens_d_SAD_minus_HC",
        "hedges_g_SAD_minus_HC",
        "rank_biserial_SAD_vs_HC",
        "p",
        "q_within_phase_feature",
        "all_subjects_adjusted_group_p",
    ]
    if not roi_top.empty:
        lines.append(markdown_table(roi_top.head(30), roi_cols))
    else:
        lines.append("_No ROI rows available._")

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- The most manuscript-ready whole-network profile is geometry/topology, especially threat-vs-background openness during extinction.",
            "- Decision/evidence metrics tell the same story in a classifier-like language: SAD tends to show weaker threat evidence and boundary separation in phase-2 extinction.",
            "- Learning-dynamics slopes are weaker in the whole-network profile, so they are better framed as descriptive unless replicated or tied to behavior.",
            "- The most informative shock-focused whole-network metric is residualized CSR projection on the subject-specific SHOCK-minus-CS- axis during reinstatement; SAD shows lower shock-axis alignment than HC, but this secondary family does not survive the broad whole-network FDR screen.",
            "- ROI shock-anchor exploration highlights right vmPFC residualized CSR-minus-CSS shock-axis projection/cosine as the strongest localized follow-up signal. Treat it as supportive and hypothesis-generating unless promoted in a preregistered follow-up.",
            "- ROI exploration suggests left ACC threat-axis geometry in phase-3 reinstatement is unusually strong; treat this as a targeted follow-up because the ROI search is larger than the whole-network test family.",
            "- Raw activation/magnitude metrics can be included as secondary checks, but they should not replace representational geometry as the central neural index.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.metadata)
    meta = meta.rename(columns={"subject_id": "sub_ID"})
    meta["sub_ID"] = meta["sub_ID"].map(normalize_subject)

    tables = []
    roi_tables = []
    for feature_space, filename in INPUT_FILES.items():
        path = args.input_dir / filename
        if path.exists():
            print(f"Deriving indices: {filename}")
            tables.append(load_npz_table(path, feature_space))
            if feature_space in {"phase2_ext_roi", "phase3_reinst_roi"}:
                print(f"Deriving ROI-wise FearNetwork indices: {filename}")
                roi_tables.append(load_roi_metric_table(path, feature_space))
        else:
            print(f"Missing input, skipping: {path}")

    subject_indices_df = pd.concat(tables, ignore_index=True)
    subject_indices_df = subject_indices_df.merge(meta, on="sub_ID", how="left")
    metric_cols = [c for c in subject_indices_df.columns if c.startswith("Neural_")]

    rankings = summarize_group_tests(subject_indices_df, metric_cols)
    consensus = (
        rankings.groupby("metric", as_index=False)
        .agg(
            n_tests=("metric", "size"),
            best_p=("p", "min"),
            best_q=("q_within_phase_feature", "min"),
            median_abs_d=("cohens_d_SAD_minus_HC", lambda x: float(np.nanmedian(np.abs(x)))),
            max_abs_d=("cohens_d_SAD_minus_HC", lambda x: float(np.nanmax(np.abs(x)))),
            median_abs_g=("hedges_g_SAD_minus_HC", lambda x: float(np.nanmedian(np.abs(x)))),
            max_abs_g=("hedges_g_SAD_minus_HC", lambda x: float(np.nanmax(np.abs(x)))),
            max_abs_rank_biserial=("rank_biserial_SAD_vs_HC", lambda x: float(np.nanmax(np.abs(x)))),
            median_auc=("scalar_auc_abs_direction", "median"),
            best_rank_score=("rank_score", "max"),
        )
        .sort_values(["best_rank_score", "best_p"], ascending=[False, True])
    )

    subject_path = args.output_dir / "derived_subject_neural_indices.csv"
    ranking_path = args.output_dir / "representative_neural_index_rankings.csv"
    consensus_path = args.output_dir / "representative_neural_index_consensus.csv"
    subject_indices_df.to_csv(subject_path, index=False)
    rankings.to_csv(ranking_path, index=False)
    consensus.to_csv(consensus_path, index=False)

    fear_network_rankings = rankings[
        rankings["feature_space"].isin(["phase2_ext_roi", "phase3_reinst_roi"])
    ].copy()
    fear_network_rankings = annotate_rankings(fear_network_rankings)
    profile_top = top_rows_by_profile(fear_network_rankings, group_cols=("phase",))
    fear_network_ranking_path = args.output_dir / "fear_network_representative_neural_index_rankings.csv"
    profile_top_path = args.output_dir / "fear_network_profile_top_metrics.csv"
    fear_network_rankings.to_csv(fear_network_ranking_path, index=False)
    profile_top.to_csv(profile_top_path, index=False)

    roi_metric_path = None
    roi_ranking_path = None
    roi_profile_top_path = None
    roi_profile_top = pd.DataFrame()
    if roi_tables:
        roi_indices_df = pd.concat(roi_tables, ignore_index=True)
        roi_indices_df = roi_indices_df.merge(meta, on="sub_ID", how="left")
        roi_metric_cols = [c for c in roi_indices_df.columns if c.startswith("Neural_")]
        roi_rankings = summarize_group_tests(
            roi_indices_df,
            roi_metric_cols,
            group_cols=("phase", "feature_space", "roi_name"),
        )
        roi_rankings = annotate_rankings(roi_rankings)
        roi_profile_top = top_roi_localization(roi_rankings)
        roi_metric_path = args.output_dir / "fear_network_roi_neural_indices.csv"
        roi_ranking_path = args.output_dir / "fear_network_roi_metric_rankings.csv"
        roi_profile_top_path = args.output_dir / "fear_network_roi_profile_top_metrics.csv"
        roi_indices_df.to_csv(roi_metric_path, index=False)
        roi_rankings.to_csv(roi_ranking_path, index=False)
        roi_profile_top.to_csv(roi_profile_top_path, index=False)

    report_path = args.output_dir / "fear_network_neural_profile_report.md"
    write_profile_report(report_path, fear_network_rankings, profile_top, roi_profile_top)

    print(f"Wrote {subject_path}")
    print(f"Wrote {ranking_path}")
    print(f"Wrote {consensus_path}")
    print(f"Wrote {fear_network_ranking_path}")
    print(f"Wrote {profile_top_path}")
    if roi_metric_path and roi_ranking_path:
        print(f"Wrote {roi_metric_path}")
        print(f"Wrote {roi_ranking_path}")
    if roi_profile_top_path:
        print(f"Wrote {roi_profile_top_path}")
    print(f"Wrote {report_path}")
    print("\nTop placebo SAD-vs-HC tests:")
    print(
        rankings[
            [
                "phase",
                "feature_space",
                "metric",
                "n_SAD",
                "n_HC",
                "diff_SAD_minus_HC",
                "cohens_d_SAD_minus_HC",
                "hedges_g_SAD_minus_HC",
                "rank_biserial_SAD_vs_HC",
                "p",
                "q_within_phase_feature",
                "scalar_auc_abs_direction",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )
    print("\nConsensus metric ranking:")
    print(consensus.head(12).to_string(index=False))
    if roi_ranking_path:
        print("\nTop FearNetwork ROI-level tests:")
        print(
            pd.read_csv(roi_ranking_path)[
                [
                    "phase",
                    "roi_name",
                    "metric",
                    "n_SAD",
                    "n_HC",
                    "diff_SAD_minus_HC",
                    "cohens_d_SAD_minus_HC",
                    "hedges_g_SAD_minus_HC",
                    "rank_biserial_SAD_vs_HC",
                    "p",
                    "q_within_phase_feature",
                    "scalar_auc_abs_direction",
                ]
            ]
            .head(20)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
