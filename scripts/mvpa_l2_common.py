#!/usr/bin/env python3
"""Shared utilities for the MVPA L2 analysis plan.

The Hyak scripts produce rich cached joblib objects. The manuscript-facing
analysis should work from a stable subject-level table, so this module keeps
the names, model families, and light statistical helpers in one place.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


CORE_NEURAL_METRICS = [
    "Neural_Dist_Safety_Background",
    "Neural_ThreatLike_Safety",
    "Neural_SafetyLike_Safety",
    "Neural_Boundary_Separation",
    "Neural_Decision_Margin_CSS",
    "Neural_Safety_Trajectory_Slope",
    "Neural_Threat_Trajectory_Slope",
]

COMPANION_NEURAL_METRICS = [
    "Neural_Dist_Threat_Background",
    "Neural_Dist_Threat_Safety",
    "Neural_ThreatLike_Threat",
    "Neural_SafetyLike_Threat",
]

PRIMARY_CLINICAL_SCORES = [
    "lsas_total",
    "lsas_fear",
    "lsas_avoid",
    "dass_anxiety",
]

PRIMARY_SCR_INDICES = [
    "SCR_SafetyMinusBackground",
    "SCR_ThreatMinusSafety",
    "SCR_Safety_Trajectory_Slope",
    "SCR_Threat_Trajectory_Slope",
]

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


def find_existing(base_dir: Path, names: Iterable[str]) -> Path | None:
    for name in names:
        path = base_dir / name
        if path.exists():
            return path
    return None


def merge_on_subject(left: pd.DataFrame | None, right: pd.DataFrame | None) -> pd.DataFrame | None:
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
    dup_cols = [c for c in out.columns if c.endswith("_dup")]
    for dup in dup_cols:
        base = dup[:-4]
        if base in out.columns:
            out[base] = out[base].combine_first(out[dup])
        else:
            out[base] = out[dup]
        out = out.drop(columns=[dup])
    return out


def derive_final_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add final mvpa_L2.md metric names from legacy/script-specific names."""
    out = df.copy()
    rename_map = {
        "Dist_Safety": "Neural_Dist_Safety_Background",
        "Dist_Safety_PV": "Neural_Dist_Safety_Background",
        "Dist_Threat": "Neural_Dist_Threat_Safety",
        "Dist_Threat_PV": "Neural_Dist_Threat_Safety",
        "P_CSR_CSS": "Neural_ThreatLike_Safety",
        "P_CSR_CSR": "Neural_ThreatLike_Threat",
        "Boundary_Separation": "Neural_Boundary_Separation",
        "Decision_Margin_CSS": "Neural_Decision_Margin_CSS",
        "decision_margin_css": "Neural_Decision_Margin_CSS",
        "p_csr_css": "Neural_ThreatLike_Safety",
        "p_csr_csr": "Neural_ThreatLike_Threat",
        "Neural_Threat_Evidence_CSR": "Neural_ThreatLike_Threat",
    }
    for old, new in rename_map.items():
        if old in out.columns and new not in out.columns:
            out[new] = out[old]

    if "Neural_ThreatLike_Safety" in out.columns and "Neural_SafetyLike_Safety" not in out.columns:
        out["Neural_SafetyLike_Safety"] = 1 - pd.to_numeric(out["Neural_ThreatLike_Safety"], errors="coerce")
    if "Neural_ThreatLike_Threat" in out.columns and "Neural_SafetyLike_Threat" not in out.columns:
        out["Neural_SafetyLike_Threat"] = 1 - pd.to_numeric(out["Neural_ThreatLike_Threat"], errors="coerce")
    return out


def available_covariates(df: pd.DataFrame, requested: Iterable[str] | None = None) -> list[str]:
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


def covariate_terms(df: pd.DataFrame, covariates: Iterable[str]) -> list[str]:
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
    predictor_terms: list[str],
    covariates: Iterable[str] | None = None,
    term_of_interest: str | None = None,
    min_n: int = 12,
) -> dict:
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
    model_df = model_df.dropna(subset=[outcome])
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
