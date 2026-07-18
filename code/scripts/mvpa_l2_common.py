#!/usr/bin/env python3
"""Shared utilities for the MVPA L2 analysis plan.

The Hyak scripts produce rich cached joblib objects. The manuscript-facing
analysis should work from a stable subject-level table, so this module keeps
the names, model families, and light statistical helpers in one place.
"""

import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


CORE_NEURAL_METRICS = [
    "Neural_Threat_Safety_Distance",
    "Prototype_Certainty",
    "Neural_DynamicDiscrimination_Volatility",
]

COMPANION_NEURAL_METRICS = [
    "Neural_Dist_Safety_Background",
    "Neural_Dist_Threat_Safety",
    "Neural_Dist_Threat_Background",
    "Neural_Certainty_CSS",
    "Neural_Certainty_CSR",
    "Neural_Decoder_Entropy_CSS",
    "Neural_Decoder_Entropy_CSR",
    "Neural_Safety_Trajectory_Slope",
    "Neural_Threat_Trajectory_Slope",
    "Shock_Anchor_Trajectory_Slope",
    "Residualized_Shock_Anchor_Trajectory_Slope",
]

PRESPECIFIED_NEURAL_METRICS = CORE_NEURAL_METRICS + COMPANION_NEURAL_METRICS

NEURAL_METRIC_FAMILIES = {
    "Neural_Threat_Safety_Distance": "Geometry",
    "Prototype_Certainty": "Certainty",
    "Neural_DynamicDiscrimination_Volatility": "Trajectory",
    "Neural_Dist_Safety_Background": "Geometry",
    "Neural_Dist_Threat_Background": "Geometry",
    "Neural_Dist_Threat_Safety": "Geometry",
    "Neural_Certainty_CSS": "Certainty",
    "Neural_Certainty_CSR": "Certainty",
    "Neural_Decoder_Entropy_CSS": "Certainty",
    "Neural_Decoder_Entropy_CSR": "Certainty",
    "Neural_Safety_Trajectory_Slope": "Trajectory",
    "Neural_Threat_Trajectory_Slope": "Trajectory",
    "Shock_Anchor_Trajectory_Slope": "Trajectory",
    "Residualized_Shock_Anchor_Trajectory_Slope": "Trajectory",
}

NEURAL_METRIC_LABELS = {
    "Neural_Threat_Safety_Distance": "Threat-safety CS- distance diff",
    "Prototype_Certainty": "Prototype certainty",
    "Neural_DynamicDiscrimination_Volatility": "Dynamic discrimination volatility",
}

DERIVED_NEURAL_FEATURE_SPACE_MAP = {
    "FearNetwork": "phase2_ext_roi",
    "MemoryFearNetwork": "phase2_ext_memory_fear_network",
    "Schaefer_Tian": "phase2_ext_schaefer_tian",
    "Schaefer": "phase2_ext_schaefer_tian",
    "Tian": "phase2_ext_schaefer_tian",
}

PRIMARY_CLINICAL_SCORES = [
    "dass_anxiety",
    "lsas_total",
]

SECONDARY_CLINICAL_SCORES = [
    "lsas_fear",
    "lsas_avoid",
    "dass_stress",
    "dass_depression",
    "ecr_total",
]

ALL_CLINICAL_SCORES = PRIMARY_CLINICAL_SCORES + SECONDARY_CLINICAL_SCORES

CLINICAL_SCORE_HIERARCHY = {
    "dass_anxiety": {
        "order": 1,
        "role": "primary",
        "family": "general_anxiety",
        "label": "DASS anxiety",
    },
    "lsas_total": {
        "order": 2,
        "role": "primary",
        "family": "social_anxiety_total",
        "label": "LSAS total",
    },
    "lsas_fear": {
        "order": 3,
        "role": "secondary",
        "family": "social_anxiety_subscale",
        "label": "LSAS fear",
    },
    "lsas_avoid": {
        "order": 4,
        "role": "secondary",
        "family": "social_anxiety_subscale",
        "label": "LSAS avoidance",
    },
    "dass_stress": {
        "order": 5,
        "role": "secondary",
        "family": "general_distress",
        "label": "DASS stress",
    },
    "dass_depression": {
        "order": 6,
        "role": "secondary",
        "family": "general_distress",
        "label": "DASS depression",
    },
    "ecr_total": {
        "order": 7,
        "role": "secondary",
        "family": "attachment",
        "label": "ECR total",
    },
}

NEURAL_METRIC_HIERARCHY = {
    metric: {"order": order, "role": "primary"}
    for order, metric in enumerate(CORE_NEURAL_METRICS, start=1)
}
NEURAL_METRIC_HIERARCHY.update(
    {
        metric: {"order": order, "role": "secondary"}
        for order, metric in enumerate(COMPANION_NEURAL_METRICS, start=len(CORE_NEURAL_METRICS) + 1)
    }
)

PRIMARY_SCR_INDICES = [
    "SCR_Safety_Trajectory_Slope",
    "SCR_Threat_Trajectory_Slope",
]

SECONDARY_SCR_INDICES = [
    "SCR_SafetyMinusBackground",
    "SCR_ThreatMinusSafety",
]

ALL_SCR_INDICES = PRIMARY_SCR_INDICES + SECONDARY_SCR_INDICES

SCR_SENSITIVITY_FLAGS = [
    "SCR_Physiological_Responder",
    "SCR_Simple_Acquisition_Differential_Learner",
    "SCR_Habituation_Adjusted_Learner",
    "SCR_Late_Phase_Sensitivity_Learner",
]


def normalize_subject_id(value: object) -> str:
    """Return a stable subject key that tolerates sub- prefixes and floats."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = re.sub(r"\.0$", "", text)
    text = re.sub(r"^sub-", "", text, flags=re.IGNORECASE)
    return text


def coalesced_string_series(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    """Coalesce the first non-empty string value across candidate columns."""
    values = pd.Series(pd.NA, index=df.index, dtype="string")
    for col in candidates:
        if col not in df.columns:
            continue
        candidate = df[col].astype("string").str.strip()
        candidate = candidate.mask(candidate.eq(""))
        values = values.fillna(candidate)
    return values


def ensure_subject_column(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize subject identifiers to sub_ID."""
    out = df.copy()
    candidates = ["sub_ID", "Subject", "subject_id", "sub", "participant_id"]
    for col in candidates:
        if col in out.columns:
            out["sub_ID"] = out[col].map(normalize_subject_id)
            return out
    raise ValueError(f"No subject ID column found. Available columns: {list(out.columns)}")


def harmonize_group_drug(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common group/drug aliases used across source tables."""
    out = df.copy()
    if "Group" in out.columns:
        group_map = {
            "healthy": "HC",
            "Healthy": "HC",
            "control": "HC",
            "Control": "HC",
            "sad": "SAD",
        }
        out["Group"] = out["Group"].map(lambda x: group_map.get(str(x).strip(), str(x).strip()) if pd.notna(x) else x)
    if "Drug" in out.columns:
        drug_map = {
            "PLC": "Placebo",
            "PBO": "Placebo",
            "placebo": "Placebo",
            "OT": "Oxytocin",
            "OXT": "Oxytocin",
            "oxytocin": "Oxytocin",
        }
        out["Drug"] = out["Drug"].map(lambda x: drug_map.get(str(x).strip(), str(x).strip()) if pd.notna(x) else x)
    return out


def read_joblib(path: Path):
    return joblib.load(path)


def maybe_read_joblib(path: Path):
    if path.exists():
        return read_joblib(path)
    return None


def payload_value(payload, *keys):
    """Fetch the first available key from a joblib payload."""
    if not isinstance(payload, dict):
        return None
    for key in keys:
        if key in payload:
            return payload[key]
    for value in payload.values():
        if isinstance(value, dict):
            found = payload_value(value, *keys)
            if found is not None:
                return found
    return None


def find_existing(base_dir: Path, names: Iterable[str]) -> Optional[Path]:
    for name in names:
        for path in (base_dir / name, base_dir / "intermediate" / name):
            if path.exists():
                return path
    return None


def merge_on_subject(left: Optional[pd.DataFrame], right: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if left is None or left.empty:
        return right.copy() if right is not None else None
    if right is None or right.empty:
        return left.copy()
    left = ensure_subject_column(left)
    right = ensure_subject_column(right)
    join_cols = ["sub_ID"]
    for col in ["Group", "Drug", "FeatureSpace"]:
        if col in left.columns and col in right.columns:
            join_cols.append(col)
    return left.merge(right, on=join_cols, how="outer", suffixes=("", "_dup"))


def coalesce_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.columns.duplicated().any():
        collapsed = pd.DataFrame(index=out.index)
        for col in pd.unique(out.columns):
            same_name = out.loc[:, out.columns == col]
            if isinstance(same_name, pd.Series):
                collapsed[col] = same_name
            elif same_name.shape[1] == 1:
                collapsed[col] = same_name.iloc[:, 0]
            else:
                collapsed[col] = same_name.bfill(axis=1).iloc[:, 0]
        out = collapsed

    dup_cols = [c for c in out.columns if c.endswith("_dup")]
    for dup in dup_cols:
        base = dup[:-4]
        if base in out.columns:
            base_values = out.loc[:, base]
            dup_values = out.loc[:, dup]
            if isinstance(base_values, pd.DataFrame):
                base_values = base_values.bfill(axis=1).iloc[:, 0]
            if isinstance(dup_values, pd.DataFrame):
                dup_values = dup_values.bfill(axis=1).iloc[:, 0]
            out[base] = base_values.combine_first(dup_values)
        else:
            dup_values = out.loc[:, dup]
            out[base] = dup_values.bfill(axis=1).iloc[:, 0] if isinstance(dup_values, pd.DataFrame) else dup_values
        out = out.drop(columns=[dup])
    return out


def derived_neural_index_candidates() -> List[Path]:
    """Candidate locations for trialwise-derived representative neural indices."""
    script_root = Path(__file__).resolve().parents[2]
    candidates = []
    env_path = os.environ.get("DERIVED_NEURAL_INDEX_PATH")
    if env_path:
        candidates.append(Path(env_path))
    out_root = os.environ.get("OUT_ROOT")
    if out_root:
        candidates.append(Path(out_root) / "representative_neural_index" / "derived_subject_neural_indices.csv")
    out_base = os.environ.get("OUT_BASE")
    if out_base:
        candidates.extend(
            [
                Path(out_base) / "representative_neural_index" / "derived_subject_neural_indices.csv",
                Path(out_base) / "mvpa_l2" / "representative_neural_index" / "derived_subject_neural_indices.csv",
            ]
        )
    candidates.extend(
        [
        Path.cwd() / "results" / "representative_neural_index" / "derived_subject_neural_indices.csv",
        Path.cwd().parent / "results" / "representative_neural_index" / "derived_subject_neural_indices.csv",
        script_root / "results" / "representative_neural_index" / "derived_subject_neural_indices.csv",
        ]
    )
    return candidates


def find_derived_neural_index_path() -> Optional[Path]:
    for path in derived_neural_index_candidates():
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def merge_derived_primary_neural_metrics(df: pd.DataFrame, phase: str = "phase2_extinction") -> pd.DataFrame:
    """Merge primary metrics from the trialwise representative-index export."""
    path = find_derived_neural_index_path()
    if path is None or df.empty:
        return df
    derived = pd.read_csv(path)
    if derived.empty or "sub_ID" not in derived.columns:
        return df
    if phase and "phase" in derived.columns:
        derived = derived[derived["phase"].astype(str).eq(phase)].copy()
    if "FeatureSpace" not in df.columns and "feature_space" in derived.columns and derived["feature_space"].nunique(dropna=True) > 1:
        return df
    value_cols = [
        col
        for col in [
            "Neural_Threat_Safety_Distance",
            "Prototype_Certainty",
            "Neural_Certainty_CSS",
            "Neural_Certainty_CSR",
            "Neural_SafetyEvidence",
            "Neural_ThreatEvidence",
            "Neural_ThreatTriangleOpenness",
            "Neural_DynamicDiscrimination_Volatility",
        ]
        if col in derived.columns
    ]
    if not value_cols:
        return df

    out = df.copy()
    out["_subject_key"] = coalesced_string_series(out, ["sub_ID", "Subject", "subject_id", "sub", "participant_id"]).map(normalize_subject_id)
    if out["_subject_key"].eq("").all():
        return out.drop(columns=["_subject_key"])

    derived = derived[["sub_ID"] + value_cols + [col for col in ["feature_space", "Drug", "drug", "drug_condition"] if col in derived.columns]].copy()
    derived["_subject_key"] = derived["sub_ID"].map(normalize_subject_id)
    merge_keys = ["_subject_key"]

    if "FeatureSpace" in out.columns and "feature_space" in derived.columns:
        out["_derived_feature_space_key"] = out["FeatureSpace"].astype("string").map(
            lambda value: DERIVED_NEURAL_FEATURE_SPACE_MAP.get(str(value), str(value))
        )
        derived["_derived_feature_space_key"] = derived["feature_space"].astype("string")
        merge_keys.append("_derived_feature_space_key")

    out_drug = coalesced_string_series(out, ["Drug", "drug", "drug_condition"]).str.lower()
    derived_drug = coalesced_string_series(derived, ["Drug", "drug", "drug_condition"]).str.lower()
    if out_drug.notna().any() and derived_drug.notna().any():
        out["_drug_key"] = out_drug
        derived["_drug_key"] = derived_drug
        merge_keys.append("_drug_key")

    drop_cols = ["sub_ID", "feature_space", "Drug", "drug", "drug_condition"]
    derived = derived.drop_duplicates(merge_keys)
    merged = out.merge(derived.drop(columns=drop_cols, errors="ignore"), on=merge_keys, how="left", suffixes=("", "_derived"))
    for col in value_cols:
        derived_col = f"{col}_derived" if f"{col}_derived" in merged.columns else col
        if derived_col not in merged.columns:
            continue
        values = pd.to_numeric(merged[derived_col], errors="coerce")
        if col in CORE_NEURAL_METRICS or col == "Neural_ThreatTriangleOpenness":
            merged[col] = values.combine_first(pd.to_numeric(merged[col], errors="coerce")) if col in merged.columns else values
        elif col in out.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(values)
        else:
            merged[col] = values
        if derived_col != col:
            merged = merged.drop(columns=[derived_col])
    return merged.drop(columns=["_subject_key", "_derived_feature_space_key", "_drug_key"], errors="ignore")


def derive_final_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add final mvpa_L2.md metric names from legacy/script-specific names."""
    out = df.copy()
    rename_map = {
        "Dist_Safety": "Neural_Dist_Safety_Background",
        "Dist_Safety_PV": "Neural_Dist_Safety_Background",
        "Neural_Dist_Safe_Background": "Neural_Dist_Safety_Background",
        "Dist_Threat": "Neural_Dist_Threat_Safety",
        "Dist_Threat_PV": "Neural_Dist_Threat_Safety",
        "Dist_Threat_Background": "Neural_Dist_Threat_Background",
        "Dist_Threat_Background_PV": "Neural_Dist_Threat_Background",
        "Neural_Safety_Differentiation": "Neural_Threat_Safety_Distance",
        "P_CSR_CSS": "Neural_ThreatLike_Safety",
        "P_CSR_CSR": "Neural_ThreatLike_Threat",
        "Boundary_Separation": "Neural_Boundary_Separation",
        "Decision_Margin_CSS": "Neural_Decision_Margin_CSS",
        "Decision_Margin_CSR": "Neural_Decision_Margin_CSR",
        "decision_margin_css": "Neural_Decision_Margin_CSS",
        "decision_margin_csr": "Neural_Decision_Margin_CSR",
        "Entropy_CSS": "Neural_Decoder_Entropy_CSS",
        "Entropy_CSR": "Neural_Decoder_Entropy_CSR",
        "entropy_css": "Neural_Decoder_Entropy_CSS",
        "entropy_csr": "Neural_Decoder_Entropy_CSR",
        "p_csr_css": "Neural_ThreatLike_Safety",
        "p_csr_csr": "Neural_ThreatLike_Threat",
        "Neural_Threat_Evidence_CSR": "Neural_ThreatLike_Threat",
        "Prototype_P_CSR_CSS": "Neural_PrototypeThreatLike_Safety",
        "Prototype_P_CSR_CSR": "Neural_PrototypeThreatLike_Threat",
        "Prototype_Boundary_Separation": "Neural_PrototypeBoundary_Separation",
    }
    for old, new in rename_map.items():
        if old in out.columns and new not in out.columns:
            out[new] = out[old]

    if "Neural_ThreatLike_Safety" in out.columns and "Neural_SafetyLike_Safety" not in out.columns:
        out["Neural_SafetyLike_Safety"] = 1 - pd.to_numeric(out["Neural_ThreatLike_Safety"], errors="coerce")
    if "Neural_ThreatLike_Threat" in out.columns and "Neural_SafetyLike_Threat" not in out.columns:
        out["Neural_SafetyLike_Threat"] = 1 - pd.to_numeric(out["Neural_ThreatLike_Threat"], errors="coerce")
    if "Neural_ThreatLike_Safety" in out.columns:
        out["Neural_SafetyEvidence"] = 1 - pd.to_numeric(out["Neural_ThreatLike_Safety"], errors="coerce")
    if "Neural_ThreatLike_Threat" in out.columns:
        out["Neural_ThreatEvidence"] = pd.to_numeric(out["Neural_ThreatLike_Threat"], errors="coerce")
    if "Neural_SafetyEvidence" in out.columns:
        out["Neural_Certainty_CSS"] = 2 * (
            pd.to_numeric(out["Neural_SafetyEvidence"], errors="coerce") - 0.5
        ).abs()
    elif "Neural_SafetyEvidenceCertainty" in out.columns:
        out["Neural_Certainty_CSS"] = pd.to_numeric(out["Neural_SafetyEvidenceCertainty"], errors="coerce")
    if "Neural_ThreatEvidence" in out.columns:
        out["Neural_Certainty_CSR"] = 2 * (
            pd.to_numeric(out["Neural_ThreatEvidence"], errors="coerce") - 0.5
        ).abs()
    elif "Neural_ThreatEvidenceCertainty" in out.columns:
        out["Neural_Certainty_CSR"] = pd.to_numeric(out["Neural_ThreatEvidenceCertainty"], errors="coerce")
    if {"Neural_Certainty_CSS", "Neural_Certainty_CSR"}.issubset(out.columns):
        prototype_certainty = out[["Neural_Certainty_CSS", "Neural_Certainty_CSR"]].mean(axis=1)
        if "Prototype_Certainty" in out.columns:
            out["Prototype_Certainty"] = pd.to_numeric(out["Prototype_Certainty"], errors="coerce").combine_first(
                prototype_certainty
            )
        else:
            out["Prototype_Certainty"] = prototype_certainty
    if "Neural_Threat_Safety_Distance" not in out.columns:
        if "Neural_ThreatTriangleOpenness" in out.columns:
            out["Neural_Threat_Safety_Distance"] = pd.to_numeric(out["Neural_ThreatTriangleOpenness"], errors="coerce")
        elif {"Neural_Dist_Threat_Background", "Neural_Dist_Safety_Background"}.issubset(out.columns):
            out["Neural_Threat_Safety_Distance"] = (
                pd.to_numeric(out["Neural_Dist_Threat_Background"], errors="coerce")
                - pd.to_numeric(out["Neural_Dist_Safety_Background"], errors="coerce")
            )
    if "Neural_ThreatLike_Safety" in out.columns:
        out["Neural_Decision_Margin_CSS"] = 0.5 - pd.to_numeric(out["Neural_ThreatLike_Safety"], errors="coerce")
    if "Neural_ThreatLike_Threat" in out.columns:
        out["Neural_Decision_Margin_CSR"] = pd.to_numeric(out["Neural_ThreatLike_Threat"], errors="coerce") - 0.5
    if "probabilities" in out.columns and "Neural_Decoder_Entropy_CSS" not in out.columns:
        out["Neural_Decoder_Entropy_CSS"] = out["probabilities"].map(mean_binary_entropy)
    elif "Neural_ThreatLike_Safety" in out.columns and "Neural_Decoder_Entropy_CSS" not in out.columns:
        out["Neural_Decoder_Entropy_CSS"] = binary_entropy(pd.to_numeric(out["Neural_ThreatLike_Safety"], errors="coerce"))
    if "probabilities_csr" in out.columns and "Neural_Decoder_Entropy_CSR" not in out.columns:
        out["Neural_Decoder_Entropy_CSR"] = out["probabilities_csr"].map(mean_binary_entropy)
    elif "Neural_ThreatLike_Threat" in out.columns and "Neural_Decoder_Entropy_CSR" not in out.columns:
        out["Neural_Decoder_Entropy_CSR"] = binary_entropy(pd.to_numeric(out["Neural_ThreatLike_Threat"], errors="coerce"))
    out = merge_derived_primary_neural_metrics(out)
    if "Neural_SafetyEvidence" in out.columns:
        out["Neural_Certainty_CSS"] = 2 * (
            pd.to_numeric(out["Neural_SafetyEvidence"], errors="coerce") - 0.5
        ).abs()
    if "Neural_ThreatEvidence" in out.columns:
        out["Neural_Certainty_CSR"] = 2 * (
            pd.to_numeric(out["Neural_ThreatEvidence"], errors="coerce") - 0.5
        ).abs()
    if {"Neural_Certainty_CSS", "Neural_Certainty_CSR"}.issubset(out.columns):
        prototype_certainty = out[["Neural_Certainty_CSS", "Neural_Certainty_CSR"]].mean(axis=1)
        if "Prototype_Certainty" in out.columns:
            out["Prototype_Certainty"] = pd.to_numeric(out["Prototype_Certainty"], errors="coerce").combine_first(
                prototype_certainty
            )
        else:
            out["Prototype_Certainty"] = prototype_certainty
    if "Neural_ThreatTriangleOpenness" in out.columns:
        triangle = pd.to_numeric(out["Neural_ThreatTriangleOpenness"], errors="coerce")
        if "Neural_Threat_Safety_Distance" in out.columns:
            out["Neural_Threat_Safety_Distance"] = pd.to_numeric(
                out["Neural_Threat_Safety_Distance"],
                errors="coerce",
            ).combine_first(triangle)
        else:
            out["Neural_Threat_Safety_Distance"] = triangle
    return out


def binary_entropy(probabilities) -> pd.Series:
    """Binary entropy in bits for scalar probabilities."""
    p = pd.to_numeric(probabilities, errors="coerce")
    if not isinstance(p, pd.Series):
        p = pd.Series(p)
    p = p.clip(1e-9, 1 - 1e-9)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def mean_binary_entropy(values) -> float:
    """Mean binary entropy in bits for a stored list/array of probabilities."""
    if values is None or (isinstance(values, float) and np.isnan(values)):
        return np.nan
    if isinstance(values, str):
        text = values.strip()
        if not text:
            return np.nan
        try:
            values = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text)]
        except Exception:
            return np.nan
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    p = np.clip(arr, 1e-9, 1 - 1e-9)
    ent = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return float(np.mean(ent))


def available_covariates(df: pd.DataFrame, requested: Optional[Iterable[str]] = None) -> List[str]:
    if requested is not None:
        return [c for c in requested if c in df.columns]
    candidates = [
        "demo_age",
        "Age",
        "age",
        "Sex",
        "sex",
        "Gender",
        "gender",
        "mean_fd",
        "MeanFD",
        "ValidTrialCount",
        "valid_trial_count",
    ]
    return [c for c in candidates if c in df.columns]


def formula_term(column: str) -> str:
    if pd.api.types.is_numeric_dtype(column):
        return column
    return column


def is_numeric_series(series: pd.Series) -> bool:
    return pd.to_numeric(series, errors="coerce").notna().sum() >= max(3, len(series.dropna()) // 2)


def covariate_terms(df: pd.DataFrame, covariates: Iterable[str]) -> List[str]:
    terms = []
    for cov in covariates:
        if cov not in df.columns:
            continue
        if is_numeric_series(df[cov]):
            terms.append(f"Q('{cov}')")
        else:
            terms.append(f"C(Q('{cov}'))")
    return terms


def fit_lm(
    df: pd.DataFrame,
    outcome: str,
    predictor_terms: List[str],
    covariates: Optional[Iterable[str]] = None,
    term_of_interest: Optional[str] = None,
    min_n: int = 12,
) -> Dict:
    """Fit an OLS model and return a tidy row for the primary term."""
    covariates = list(covariates or [])
    needed = [outcome]
    for cov in covariates:
        if cov in df.columns:
            needed.append(cov)
    model_df = df.copy()
    if outcome not in model_df.columns:
        return {"status": "missing_outcome", "n": 0, "outcome": outcome}
    model_df[outcome] = pd.to_numeric(model_df[outcome], errors="coerce")
    model_df = model_df.dropna(subset=needed)
    if len(model_df) < min_n:
        return {"status": "too_few_rows", "n": len(model_df), "outcome": outcome}

    terms = predictor_terms + covariate_terms(model_df, covariates)
    formula = f"Q('{outcome}') ~ " + " + ".join(terms)
    try:
        fit = smf.ols(formula, data=model_df).fit()
    except Exception as exc:
        return {"status": f"fit_failed: {exc}", "n": len(model_df), "outcome": outcome, "formula": formula}

    term = term_of_interest
    if term is None:
        term = predictor_terms[0]
    if term not in fit.params.index:
        matching = [idx for idx in fit.params.index if term in idx]
        term = matching[0] if matching else None
    if term is None:
        return {"status": "term_missing", "n": int(fit.nobs), "outcome": outcome, "formula": formula}

    conf = fit.conf_int().loc[term]
    return {
        "status": "ok",
        "outcome": outcome,
        "term": term,
        "estimate": float(fit.params[term]),
        "std_error": float(fit.bse[term]),
        "t": float(fit.tvalues[term]),
        "p": float(fit.pvalues[term]),
        "ci_low": float(conf.iloc[0]),
        "ci_high": float(conf.iloc[1]),
        "n": int(fit.nobs),
        "r2": float(fit.rsquared) if math.isfinite(fit.rsquared) else np.nan,
        "formula": formula,
    }


def add_fdr(df: pd.DataFrame, family_cols: Iterable[str] = ("analysis",)) -> pd.DataFrame:
    out = df.copy()
    out["q"] = np.nan
    if "p" not in out.columns:
        return out
    for _, idx in out.groupby(list(family_cols), dropna=False).groups.items():
        pvals = pd.to_numeric(out.loc[idx, "p"], errors="coerce")
        valid = pvals.notna()
        if valid.any():
            out.loc[pvals[valid].index, "q"] = multipletests(pvals[valid], method="fdr_bh")[1]
    return out


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
