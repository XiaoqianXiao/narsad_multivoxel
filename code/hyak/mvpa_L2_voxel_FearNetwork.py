#!/usr/bin/env python
# Generated from mvpa_L2_voxel_FearNetwork.py for Hyak execution.
# %% [cell 1]
# Cell 1: Imports & basic config

import os
import numpy as np
import pandas as pd
import glob

from numpy.linalg import norm

from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold, GroupKFold, permutation_test_score, LeaveOneGroupOut, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.covariance import LedoitWolf
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_selection import RFE
from sklearn.utils import resample, shuffle, resample
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.inspection import permutation_importance



import nibabel as nib


from joblib import Parallel, delayed, dump, load



import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


import itertools
from itertools import combinations

from nilearn import plotting, image, masking
from nilearn.maskers import NiftiLabelsMasker


from scipy import stats
from scipy.stats import pearsonr, ttest_1samp, ttest_ind, entropy, kurtosis
from scipy.spatial.distance import pdist, squareform, cdist

from itertools import combinations
from joblib import Parallel, delayed
import time
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt
import seaborn as sns

from typing import List, Sequence, Tuple, Union
import plotly.graph_objects as go
# Nice plotting defaults
sns.set_context("poster")

RANDOM_STATE = 42
N_SPLITS = 5   # GroupKFold folds
INNER_CV_SPLITS = 5     
CS_LABELS = ["CS-", "CSS", "CSR", "SHOCK"]  # the three CS types of interest
MAX_ITER = 5000
thresh_hold_p = 1 - 0.05
N_PERMUTATION = 5000
#N_PERMUTATION = 1
#N_REPEATS = 10
N_REPEATS = 10
N_JOBS = 1
N_BOOT = 50
CROSSNOBIS_REPEATS = 50
SUBJECT_CV_SPLITS = 5
SUBJECT_INNER_SPLITS = 3
CALIB_BINS = 5
TOP_PCT = 95
LOW_PCT = 5
TWO_TAIL_LOW = 2.5
TWO_TAIL_HIGH = 97.5
MIN_TRIALS_PER_SUBJECT = 10
C_MIN_EXP = -2
C_MAX_EXP = 2
C_POINTS = 20
MIN_FEATURES_FOR_PRIMARY_MASK = 20

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
PRIMARY_CLINICAL_SCORES = ["dass_anxiety", "lsas_total"]
SECONDARY_CLINICAL_SCORES = ["lsas_fear", "lsas_avoid", "dass_stress", "dass_depression", "ecr_total"]
PRIMARY_SCR_INDICES = ["SCR_Safety_Trajectory_Slope", "SCR_Threat_Trajectory_Slope"]
SECONDARY_SCR_INDICES = ["SCR_SafetyMinusBackground", "SCR_ThreatMinusSafety"]
NOTEBOOK_REQUIRED_SUBJECT_COLUMNS = (
    ["sub_ID", "FeatureSpace", "Group", "Drug"]
    + CORE_NEURAL_METRICS
    + COMPANION_NEURAL_METRICS
    + PRIMARY_CLINICAL_SCORES
    + SECONDARY_CLINICAL_SCORES
    + PRIMARY_SCR_INDICES
    + SECONDARY_SCR_INDICES
)

# Hyak/runtime configuration adapted from hyak/mvpa_L2_voxel_WholeBrain_Schaefer.py.
import argparse
import pickle
import types
import joblib


def parse_runtime_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--project_root", default=os.environ.get("PROJECT_ROOT", "/gscratch/fang/NARSAD"))
    parser.add_argument("--output_dir", default=os.environ.get("OUTPUT_DIR"))
    parser.add_argument(
        "--roi_dir",
        default=os.environ.get(
            "FEAR_ROI_DIR",
            os.path.join(os.environ.get("PROJECT_ROOT", "/gscratch/fang/NARSAD"), "ROI/Gillian_anatomically_constrained"),
        ),
    )
    parser.add_argument("--n_jobs", type=int, default=int(os.environ.get("N_JOBS", "1")))
    parser.add_argument("--n_jobs_cv", type=int, default=int(os.environ.get("N_JOBS_CV", os.environ.get("N_JOBS", "1"))))
    parser.add_argument("--n_permutation", type=int, default=int(os.environ.get("N_PERMUTATION", str(N_PERMUTATION))))
    parser.add_argument("--n_null_perms", type=int, default=int(os.environ.get("N_NULL_PERMS", "5000")))
    parser.add_argument(
        "--stage11_actual_repeats",
        type=int,
        default=int(os.environ.get("STAGE11_ACTUAL_REPEATS", os.environ.get("N_NULL_PERMS", "5000"))),
        help="Total actual permutation-importance repeats for stage 11.",
    )
    parser.add_argument(
        "--stage11_chunk_idx",
        type=int,
        default=(
            None
            if os.environ.get("STAGE11_CHUNK_IDX", os.environ.get("SLURM_ARRAY_TASK_ID")) is None
            else int(os.environ.get("STAGE11_CHUNK_IDX", os.environ.get("SLURM_ARRAY_TASK_ID")))
        ),
        help="Zero-based stage 11 chunk index. Defaults to SLURM_ARRAY_TASK_ID when set.",
    )
    parser.add_argument(
        "--stage11_chunk_count",
        type=int,
        default=int(os.environ.get("STAGE11_CHUNK_COUNT", "1")),
        help="Total number of stage 11 chunks.",
    )
    parser.add_argument(
        "--stage11_merge",
        action="store_true",
        default=os.environ.get("STAGE11_MERGE", "0") == "1",
        help="Merge stage 11 chunk outputs for the selected group instead of computing a chunk.",
    )
    parser.add_argument(
        "--stage11_group",
        default=os.environ.get("STAGE11_GROUP", "ALL"),
        choices=["ALL", "SAD", "HC"],
        help="For stage 11 only, run empirical permutation-importance masks for ALL, SAD, or HC.",
    )
    parser.add_argument(
        "--stage11_mask_mode",
        default=os.environ.get("STAGE11_MASK_MODE", "current"),
        choices=["current", "original_notebook"],
        help=(
            "Stage 11 feature-mask policy. 'current' uses the current .py policy "
            "including the all-positive fallback downstream; 'original_notebook' "
            "uses the empirical p<.05 positive mask directly and disables fallback."
        ),
    )
    parser.add_argument(
        "--stage11_scoring",
        default=os.environ.get("STAGE11_SCORING", "auto"),
        choices=["auto", "decision_margin", "forced_choice"],
        help=(
            "Permutation-importance scorer for Stage 11. 'auto' maps current -> "
            "decision_margin and original_notebook -> forced_choice."
        ),
    )
    parser.add_argument(
        "--stage",
        default=os.environ.get("STAGE"),
        help="Analysis stage to run, for example 6, 12, or a comma/range list like 6,10-12. Omit to run all.",
    )
    parser.add_argument(
        "--include_subjects_csv",
        default=os.environ.get("INCLUDE_SUBJECTS_CSV"),
        help="Optional subject-level CSV used to restrict Stage 6 Analysis 1.1 to a sensitivity subgroup.",
    )
    parser.add_argument(
        "--include_subjects_flag",
        default=os.environ.get("INCLUDE_SUBJECTS_FLAG"),
        help="Optional boolean column in --include_subjects_csv. Subjects with truthy values are retained.",
    )
    parser.add_argument(
        "--include_subjects_column",
        default=os.environ.get("INCLUDE_SUBJECTS_COLUMN", "sub_ID"),
        help="Subject ID column in --include_subjects_csv.",
    )
    parser.add_argument(
        "--analysis_label",
        default=os.environ.get("ANALYSIS_LABEL", ""),
        help="Optional label for subgroup/sensitivity runs. Used to avoid overwriting the primary Stage 6 checkpoint.",
    )
    return parser.parse_known_args()


def configure_blas_threads():
    if N_JOBS != 1 and N_JOBS_CV != 1:
        return
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    try:
        cpu_count = int(slurm_cpus) if slurm_cpus else (os.cpu_count() or 1)
    except ValueError:
        cpu_count = os.cpu_count() or 1
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, str(cpu_count))


SCR_CONDITION_MAP = {"CS-": "CS-", "CS+S": "CSS", "CS+R": "CSR", "CSS": "CSS", "CSR": "CSR"}
SCR_BEHAVIORAL_INDICES = [
    "SCR_Safety_Mean",
    "SCR_Threat_Mean",
    "SCR_Background_Mean",
    "SCR_SafetyMinusBackground",
    "SCR_ThreatMinusSafety",
    "SCR_Safety_Trajectory_Slope",
    "SCR_Threat_Trajectory_Slope",
]


def extract_metrics_pv(rdms_pv, idx_cs_minus=0, idx_css=1, idx_csr=2, include_threat_background=False):
    """Extract per-voxel topology metrics from RDMs ordered as CS-, CSS, CSR."""
    threat_safety = rdms_pv[:, idx_csr, idx_css]
    safety_background = rdms_pv[:, idx_css, idx_cs_minus]
    threat_background = rdms_pv[:, idx_csr, idx_cs_minus]
    if include_threat_background:
        return threat_safety, safety_background, threat_background
    return threat_safety, safety_background


def resolve_topology_subject_ids(results_12, group_name, distance_len=None):
    """Resolve Analysis 1.2 subject IDs from saved results or reconstruct them cheaply."""
    group = group_name.upper()
    key = f"subs_{group.lower()}_rdm"
    if isinstance(results_12, dict) and key in results_12:
        sub_ids = np.asarray(results_12[key]).astype(str)
    else:
        required = ("X_ext", "y_ext", "sub_ext", "data_subsets")
        missing = [name for name in required if name not in globals()]
        if missing:
            raise ValueError(
                f"Cannot recover Analysis 1.2 {group} subject IDs; missing {missing}. "
                "Re-run stage 12 once with the updated script so subject IDs are saved."
            )
        placebo_key = f"{group}_Placebo"
        if placebo_key not in data_subsets:
            raise KeyError(f"Missing {placebo_key} in data_subsets; cannot recover topology subject IDs.")
        known_subs = np.unique(data_subsets[placebo_key]["ext"]["sub"])
        mask = np.isin(sub_ext, known_subs) & np.isin(y_ext, ["CS-", "CSS", "CSR"])
        sub_ids = []
        for sub in np.unique(sub_ext[mask]):
            y_sub = y_ext[mask & (sub_ext == sub)]
            if all(np.sum(y_sub == cond) >= 2 for cond in ["CS-", "CSS", "CSR"]):
                sub_ids.append(sub)
        sub_ids = np.asarray(sub_ids).astype(str)
        print(f"[INFO] Recovered Analysis 1.2 {group} subject IDs from extinction data.")
    if distance_len is not None and len(sub_ids) != int(distance_len):
        raise ValueError(
            f"Analysis 1.2 {group} subject ID count ({len(sub_ids)}) does not match RDM metric length ({distance_len}). "
            "Re-run stage 12 once with the updated script."
        )
    return sub_ids


def partial_corr_residualized(df, x_col, y_col, covariates):
    """Partial correlation via residualization; avoids requiring pingouin on the cluster."""
    covars = [c for c in covariates if c in df.columns]
    cols = [x_col, y_col] + covars
    valid = df[cols].dropna().apply(pd.to_numeric, errors="coerce").dropna()
    if len(valid) <= len(covars) + 2:
        return np.nan, np.nan, len(valid)
    if not covars:
        r_val, p_val = pearsonr(valid[x_col], valid[y_col])
        return r_val, p_val, len(valid)

    design = sm.add_constant(valid[covars], has_constant="add")
    x_resid = sm.OLS(valid[x_col], design).fit().resid
    y_resid = sm.OLS(valid[y_col], design).fit().resid
    r_val, p_val = pearsonr(x_resid, y_resid)
    return r_val, p_val, len(valid)


def resolve_trial_scr_path(project_root):
    candidates = [
        os.environ.get("TRIAL_SCR_PATH"),
        os.environ.get("SCR_TRIAL_PATH"),
        "/gscratch/fang/NARSAD/EDR/peak_stats_table-phase2.3.csv",
        os.path.join(project_root, "EDR", "peak_stats_table-phase2.3.csv"),
        "/Users/xiaoqianxiao/projects/NARSAD/EDR/peak_stats_table-phase2.3.csv",
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def resolve_clinical_csv_path(clinical_dir, export_label, expected_filename):
    """Resolve dated REDCap clinical exports without hard-coding one timestamp."""
    env_key = f"CLINICAL_{export_label.upper().replace('-', '_')}_PATH"
    search_dirs = [
        os.environ.get("CLINICAL_DIR"),
        clinical_dir,
        os.path.join(PROJECT_ROOT, "source_data", "behav"),
        os.path.join(PROJECT_ROOT, "behav"),
    ]
    search_dirs = [path for path in dict.fromkeys(search_dirs) if path]
    token = "LSAS" if export_label.upper().startswith("LSAS") else export_label
    candidates = [os.environ.get(env_key)]
    for search_dir in search_dirs:
        candidates.append(os.path.join(search_dir, expected_filename))
        candidates.extend(glob.glob(os.path.join(search_dir, f"SocialSafetyLearning-{export_label}_DATA_*.csv")))
        candidates.extend(glob.glob(os.path.join(search_dir, f"SocialSafetyLearning-*{token}*_DATA_*.csv")))
        candidates.extend(glob.glob(os.path.join(search_dir, f"*{token}*.csv")))
    existing = [path for path in candidates if path and os.path.exists(path)]
    if existing:
        resolved = max(existing, key=os.path.getmtime)
        if resolved != os.path.join(clinical_dir, expected_filename):
            print(f"[Clinical] Using {export_label} export: {resolved}")
        return resolved
    raise FileNotFoundError(
        f"No {export_label} clinical CSV found. Checked directories {search_dirs} "
        f"with patterns SocialSafetyLearning-{export_label}_DATA_*.csv, SocialSafetyLearning-*{token}*_DATA_*.csv, and *{token}*.csv. "
        f"You can also set {env_key}."
    )


def load_trialwise_scr(project_root):
    scr_path = resolve_trial_scr_path(project_root)
    if scr_path is None:
        print("[SCR] Trial-wise SCR file not found; SCR behavioral indices will be skipped.")
        return pd.DataFrame(), None
    df_scr = pd.read_csv(scr_path)
    required = {"sid", "stTy", "stNum", "phaBase2Peak"}
    missing = required.difference(df_scr.columns)
    if missing:
        raise ValueError(f"SCR file {scr_path} is missing required columns: {sorted(missing)}")
    if "bad" in df_scr.columns:
        df_scr = df_scr[df_scr["bad"] == 0].copy()
    df_scr["sub_ID"] = df_scr["sid"].astype(str).str.strip()
    df_scr["SCR_Condition"] = df_scr["stTy"].map(SCR_CONDITION_MAP)
    df_scr = df_scr[df_scr["SCR_Condition"].isin(["CS-", "CSS", "CSR"])].copy()
    df_scr["SCR_Trial"] = pd.to_numeric(df_scr["stNum"], errors="coerce")
    df_scr["SCR_Anticipatory"] = pd.to_numeric(df_scr["phaBase2Peak"], errors="coerce")
    df_scr["SCR_US"] = pd.to_numeric(df_scr.get("USbase2peak", np.nan), errors="coerce")
    df_scr = df_scr.sort_values(["sub_ID", "SCR_Condition", "SCR_Trial"])
    df_scr["condition_trial"] = df_scr.groupby(["sub_ID", "SCR_Condition"]).cumcount() + 1
    keep_cols = ["sub_ID", "SCR_Condition", "SCR_Trial", "condition_trial", "SCR_Anticipatory", "SCR_US"]
    print(f"[SCR] Loaded {len(df_scr)} valid trial-wise SCR rows from {scr_path}")
    return df_scr[keep_cols].copy(), scr_path


def _safe_slope(x, y):
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 3 or valid["x"].nunique() < 2:
        return np.nan
    return float(np.polyfit(valid["x"], valid["y"], 1)[0])


def summarize_scr_indices(df_scr):
    rows = []
    if df_scr is None or df_scr.empty:
        return pd.DataFrame(columns=["sub_ID"] + SCR_BEHAVIORAL_INDICES)
    for sub_id, sub_df in df_scr.groupby("sub_ID"):
        by_cond = sub_df.groupby("SCR_Condition")["SCR_Anticipatory"].mean()
        safety_mean = by_cond.get("CSS", np.nan)
        threat_mean = by_cond.get("CSR", np.nan)
        background_mean = by_cond.get("CS-", np.nan)
        css = sub_df[sub_df["SCR_Condition"] == "CSS"]
        csr = sub_df[sub_df["SCR_Condition"] == "CSR"]
        rows.append({
            "sub_ID": str(sub_id),
            "SCR_Safety_Mean": safety_mean,
            "SCR_Threat_Mean": threat_mean,
            "SCR_Background_Mean": background_mean,
            "SCR_SafetyMinusBackground": safety_mean - background_mean,
            "SCR_ThreatMinusSafety": threat_mean - safety_mean,
            "SCR_Safety_Trajectory_Slope": _safe_slope(css["condition_trial"], css["SCR_Anticipatory"]),
            "SCR_Threat_Trajectory_Slope": _safe_slope(csr["condition_trial"], csr["SCR_Anticipatory"]),
        })
    return pd.DataFrame(rows)


def calculate_neural_scr_safety_coupling(df_safe, df_scr):
    if df_safe is None or df_scr is None or df_safe.empty or df_scr.empty:
        return pd.DataFrame(columns=["sub_ID", "Neural_SCR_Safety_Coupling", "Neural_SCR_Safety_Coupling_N"])
    neural = df_safe.rename(columns={"sub": "sub_ID", "trial": "condition_trial", "score": "Neural_Safety_Score"}).copy()
    neural["sub_ID"] = neural["sub_ID"].astype(str)
    neural["condition_trial"] = pd.to_numeric(neural["condition_trial"], errors="coerce")
    scr_css = df_scr[df_scr["SCR_Condition"] == "CSS"][["sub_ID", "condition_trial", "SCR_Anticipatory"]].copy()
    merged = neural.merge(scr_css, on=["sub_ID", "condition_trial"], how="inner")
    rows = []
    for sub_id, sub_df in merged.groupby("sub_ID"):
        valid = sub_df[["Neural_Safety_Score", "SCR_Anticipatory"]].dropna()
        if len(valid) >= 3 and valid["Neural_Safety_Score"].nunique() > 1 and valid["SCR_Anticipatory"].nunique() > 1:
            r_val, _ = pearsonr(valid["Neural_Safety_Score"], valid["SCR_Anticipatory"])
        else:
            r_val = np.nan
        rows.append({
            "sub_ID": str(sub_id),
            "Neural_SCR_Safety_Coupling": r_val,
            "Neural_SCR_Safety_Coupling_N": len(valid),
        })
    return pd.DataFrame(rows)


def make_all_positive_importance_mask(scores):
    scores = np.asarray(scores)
    mask = np.zeros(scores.shape, dtype=bool)
    mask[np.isfinite(scores) & (scores > 0)] = True
    return mask


def resolve_stage11_scoring_label(mask_mode, scoring):
    if scoring == "auto":
        return "forced_choice" if mask_mode == "original_notebook" else "decision_margin"
    return scoring


def stage11_scorer_name_from_label(scoring_label):
    if scoring_label == "forced_choice":
        return "forced_choice_scorer"
    if scoring_label == "decision_margin":
        return "decision_margin_scorer"
    raise ValueError(f"Unknown Stage 11 scoring label: {scoring_label}")


def get_stage11_scorer_func():
    if STAGE11_SCORING == "forced_choice":
        return forced_choice_scorer
    if STAGE11_SCORING == "decision_margin":
        return decision_margin_scorer
    raise ValueError(f"Unknown Stage 11 scoring label: {STAGE11_SCORING}")


def _stage11_loaded_mode_for_group(group_name):
    diagnostics = globals().get("importance_diagnostics_permutated", {})
    diag = diagnostics.get(group_name, {}) if isinstance(diagnostics, dict) else {}
    if isinstance(diag, dict):
        mode = diag.get("stage11_mask_mode")
        if mode:
            return str(mode)
        scoring = str(diag.get("importance_scoring", ""))
        if scoring == "forced_choice_scorer":
            return "original_notebook"
    return "current"


def _stage11_loaded_scoring_for_group(group_name):
    diagnostics = globals().get("importance_diagnostics_permutated", {})
    diag = diagnostics.get(group_name, {}) if isinstance(diagnostics, dict) else {}
    if isinstance(diag, dict):
        scoring = diag.get("stage11_scoring") or diag.get("importance_scoring")
        if scoring:
            return str(scoring)
    return STAGE11_SCORER_NAME


def _validate_stage11_mode_for_downstream(group_name, analysis_label):
    loaded_mode = _stage11_loaded_mode_for_group(group_name)
    if STAGE11_MASK_MODE != loaded_mode:
        raise ValueError(
            f"{analysis_label}: requested STAGE11_MASK_MODE={STAGE11_MASK_MODE}, but loaded "
            f"Stage 11 mask for {group_name} was generated with mode={loaded_mode}. "
            "Use a separate output directory or rerun Stage 11 and all downstream stages "
            "with the same mask mode."
        )


def get_analysis_feature_masks(analysis_label, min_primary_features=MIN_FEATURES_FOR_PRIMARY_MASK):
    if "importance_mask_permutated" not in globals() or not importance_mask_permutated:
        load_stage11_split_results()
    if "importance_scores_permutated" not in globals() or not importance_scores_permutated:
        load_stage11_split_results()
    if "importance_mask_permutated" not in globals() or not importance_mask_permutated:
        raise ValueError(
            f"{analysis_label}: importance_mask_permutated missing. Run Stage 11 first "
            "(submit_mvpa_L2_fearnetwork_stage.sh all, or run 11:SAD and 11:HC plus their merge jobs)."
        )
    if "importance_scores_permutated" not in globals() or not importance_scores_permutated:
        raise ValueError(
            f"{analysis_label}: importance_scores_permutated missing. Run Stage 11 first "
            "(submit_mvpa_L2_fearnetwork_stage.sh all, or run 11:SAD and 11:HC plus their merge jobs)."
        )

    selected_masks = {}
    feature_space = {}
    for grp in ("SAD", "HC"):
        if grp not in importance_mask_permutated or grp not in importance_scores_permutated:
            raise ValueError(f"{analysis_label}: missing {grp} permutation-importance outputs.")
        _validate_stage11_mode_for_downstream(grp, analysis_label)
        primary_mask = np.asarray(importance_mask_permutated[grp], dtype=bool)
        scores = np.asarray(importance_scores_permutated[grp])
        primary_n = int(np.sum(primary_mask))
        selected_mask = primary_mask
        source = "empirical_p_lt_0.05_positive_permutation_importance"

        if primary_n < min_primary_features:
            if STAGE11_MASK_MODE == "original_notebook":
                raise ValueError(
                    f"{analysis_label} {grp}: original_notebook mask has {primary_n} features "
                    f"(< {min_primary_features}). The original-notebook policy disables the "
                    "all-positive fallback; inspect Stage 11 outputs or choose STAGE11_MASK_MODE=current."
                )
            fallback_mask = make_all_positive_importance_mask(scores)
            fallback_n = int(np.sum(fallback_mask))
            print(
                f"  [FEATURE SPACE] {analysis_label} {grp}: primary empirical mask has "
                f"{primary_n} features; using all {fallback_n} positive permutation-importance voxels."
            )
            if fallback_n == 0:
                raise ValueError(
                    f"{analysis_label} {grp}: empirical mask has {primary_n} features and no positive "
                    "permutation-importance scores are available for the all-positive fallback."
                )
            selected_mask = fallback_mask
            source = "all_positive_permutation_importance_sensitivity"
        else:
            print(f"  [FEATURE SPACE] {analysis_label} {grp}: using empirical mask with {primary_n} features.")

        selected_masks[grp] = selected_mask
        feature_space[grp] = {
            "source": source,
            "stage11_mask_mode": STAGE11_MASK_MODE,
            "stage11_scoring": _stage11_loaded_scoring_for_group(grp),
            "primary_empirical_features": primary_n,
            "selected_features": int(np.sum(selected_mask)),
            "fallback_rule": "all_positive_permutation_importance" if source == "all_positive_permutation_importance_sensitivity" else None,
        }

    return selected_masks, feature_space


_args, _unknown = parse_runtime_args()
PROJECT_ROOT = _args.project_root
OUTPUT_DIR = _args.output_dir
ROI_DIR = _args.roi_dir
N_JOBS = _args.n_jobs
N_JOBS_CV = _args.n_jobs_cv
N_PERMUTATION = _args.n_permutation
N_NULL_PERMS = _args.n_null_perms
STAGE11_ACTUAL_REPEATS = _args.stage11_actual_repeats
STAGE11_CHUNK_IDX = _args.stage11_chunk_idx
STAGE11_CHUNK_COUNT = _args.stage11_chunk_count
STAGE11_MERGE = _args.stage11_merge
STAGE11_GROUP = _args.stage11_group
STAGE11_MASK_MODE = _args.stage11_mask_mode
STAGE11_SCORING = resolve_stage11_scoring_label(STAGE11_MASK_MODE, _args.stage11_scoring)
STAGE11_SCORER_NAME = stage11_scorer_name_from_label(STAGE11_SCORING)
INCLUDE_SUBJECTS_CSV = _args.include_subjects_csv
INCLUDE_SUBJECTS_FLAG = _args.include_subjects_flag
INCLUDE_SUBJECTS_COLUMN = _args.include_subjects_column
ANALYSIS_LABEL = _args.analysis_label
configure_blas_threads()


def _parse_stage_selection(stage_arg):
    if stage_arg in (None, "", "all", "ALL"):
        return None
    selected = set()
    for part in str(stage_arg).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            selected.update(range(int(start), int(end) + 1))
        else:
            selected.add(int(part))
    return selected


STAGE_SELECTION = _parse_stage_selection(_args.stage)
MAX_SELECTED_STAGE = max(STAGE_SELECTION) if STAGE_SELECTION else None


def cell_active(cell_id: int) -> bool:
    return STAGE_SELECTION is None or cell_id in STAGE_SELECTION


def load_stage11_split_results() -> bool:
    """Merge split SAD/HC stage-11 outputs for downstream stages."""
    merged_masks = {}
    merged_scores = {}
    merged_p_values = {}
    merged_diagnostics = {}
    loaded_any = False

    def _merge_stage11_payload(payload):
        if isinstance(payload, dict):
            masks = payload.get("importance_mask_permutated") or payload.get("importance_masks_permutated") or {}
            scores = payload.get("importance_scores_permutated") or {}
            merged_masks.update(masks)
            merged_scores.update(scores)
            p_vals = payload.get("p_values_permutated", {})
            if isinstance(p_vals, dict):
                merged_p_values.update(p_vals)
            diagnostics = payload.get("importance_diagnostics_permutated", {})
            if isinstance(diagnostics, dict):
                merged_diagnostics.update(diagnostics)

    main_path = _script_ckpt_path(11)
    combined_paths = [
        main_path,
        _script_intermediate_path("stage11_importance_masks"),
        os.path.join(CHECKPOINT_DIR, "stage11_importance_masks.joblib"),
    ]
    for combined_path in combined_paths:
        if not os.path.exists(combined_path):
            continue
        payload = joblib.load(combined_path)
        if combined_path == main_path:
            globals().update(payload if isinstance(payload, dict) else {})
            print(f"[Cell checkpoint] Loaded cell 11 <- {combined_path}")
        else:
            print(f"[Cell checkpoint] Loaded combined stage 11 <- {combined_path}")
        _merge_stage11_payload(payload)
        loaded_any = True
        if {"SAD", "HC"}.issubset(set(merged_masks)):
            break

    for group_name in ("SAD", "HC"):
        group_paths = [
            os.path.join(CHECKPOINT_DIR, f"cell_11_{group_name}.joblib"),
            os.path.join(INTERMEDIATE_DIR, f"stage11_importance_masks_{group_name}.joblib"),
        ]
        for group_path in group_paths:
            if not os.path.exists(group_path):
                continue
            payload = joblib.load(group_path)
            if not isinstance(payload, dict):
                continue
            _merge_stage11_payload(payload)
            print(f"[Cell checkpoint] Loaded split stage 11 {group_name} <- {group_path}")
            loaded_any = True
            break

    if merged_masks:
        missing_groups = {"SAD", "HC"} - set(merged_masks)
        if missing_groups:
            raise FileNotFoundError(
                "Stage 11 downstream inputs are incomplete. Missing final split output for: "
                f"{sorted(missing_groups)}. Expected cell_11_SAD.joblib and cell_11_HC.joblib "
                f"in {CHECKPOINT_DIR} or matching stage11_importance_masks_* files in {INTERMEDIATE_DIR}."
            )
        globals()["importance_mask_permutated"] = merged_masks
        globals()["importance_scores_permutated"] = merged_scores
        globals()["p_values_permutated"] = merged_p_values
        globals()["importance_diagnostics_permutated"] = merged_diagnostics
        combined_payload = {
            "importance_mask_permutated": merged_masks,
            "importance_scores_permutated": merged_scores,
            "p_values_permutated": merged_p_values,
            "importance_diagnostics_permutated": merged_diagnostics,
            "stage11_mask_mode": STAGE11_MASK_MODE,
            "stage11_scoring": STAGE11_SCORING,
            "importance_scoring": STAGE11_SCORER_NAME,
        }
        joblib.dump(combined_payload, main_path)
        joblib.dump(combined_payload, _script_intermediate_path("stage11_importance_masks"))
        print(f"[Cell checkpoint] Merged split stage 11 outputs -> {main_path}")
        return True

    return loaded_any


def reconstruct_cell6_state() -> None:
    """Recreate notebook convenience variables from the saved Analysis 1.1 bundle."""
    if "results_11" not in globals() or not isinstance(results_11, dict):
        return
    globals()["res_sad"] = {
        "accuracy": results_11["acc_sad_cv"],
        "haufe_pattern": results_11["map_sad"],
        "model_sad": results_11["model_sad"],
        "model": results_11["model_sad"],
        "best_C": results_11.get("best_c_sad", 1.0),
    }
    globals()["res_hc"] = {
        "accuracy": results_11["acc_hc_cv"],
        "haufe_pattern": results_11["map_hc"],
        "model_hc": results_11["model_hc"],
        "model": results_11["model_hc"],
        "best_C": results_11.get("best_c_hc", 1.0),
    }
    globals()["perm_acc_sad"] = results_11.get("perm_dist_sad")
    globals()["perm_acc_hc"] = results_11.get("perm_dist_hc")
    globals()["func_matrix"] = results_11.get("func_matrix")
    globals()["func_pvals"] = results_11.get("p_func_pvals")
    globals()["obs_sim"] = results_11.get("sim_spatial")
    globals()["p_sim_spatial"] = results_11.get("p_sim")
    globals()["p_sad"] = results_11.get("p_sad")
    globals()["p_hc"] = results_11.get("p_hc")
    if globals().get("func_pvals") is not None:
        globals()["p_sad2hc"] = func_pvals[0, 1]
        globals()["p_hc2sad"] = func_pvals[1, 0]
    if globals().get("func_matrix") is not None:
        globals()["mean_sad2hc"] = func_matrix[0, 1]
        globals()["mean_hc2sad"] = func_matrix[1, 0]


def get_analysis11_model(group_name: str):
    """Load the refit Analysis 1.1 model from current globals or saved checkpoints."""
    group = group_name.upper()
    lower = group.lower()
    model_key = f"model_{lower}"

    res_obj = globals().get(f"res_{lower}")
    if isinstance(res_obj, dict):
        for key in (model_key, "model"):
            if res_obj.get(key) is not None:
                return res_obj[key]

    results_obj = globals().get("results_11")
    if isinstance(results_obj, dict) and results_obj.get(model_key) is not None:
        return results_obj[model_key]

    candidate_paths = [
        _script_ckpt_path(6),
        os.path.join(CHECKPOINT_DIR, "results_11.joblib"),
        os.path.join(CHECKPOINT_DIR, "results_analysis_11.joblib"),
    ]
    for path in candidate_paths:
        if not os.path.exists(path):
            continue
        payload = joblib.load(path)
        if isinstance(payload, dict) and "results_11" in payload:
            payload = payload["results_11"]
        if isinstance(payload, dict):
            if payload.get(model_key) is not None:
                print(f"  [LOAD] Retrieved {model_key} from {path}")
                return payload[model_key]
            nested_model = payload.get(group, {}).get("model") if isinstance(payload.get(group), dict) else None
            if nested_model is not None:
                print(f"  [LOAD] Retrieved nested {group} model from {path}")
                return nested_model

    raise FileNotFoundError(
        f"Analysis 1.1 {group} model not found. Run stage 6 first, or check cell_06.joblib in {CHECKPOINT_DIR}."
    )


def maybe_load_cell_results(cell_id: int) -> None:
    if cell_active(cell_id):
        return
    if MAX_SELECTED_STAGE is not None and cell_id > MAX_SELECTED_STAGE:
        return
    if cell_id == 11:
        if load_stage11_split_results():
            return
        print("[Cell checkpoint] No combined or split saved output found for skipped cell 11.")
        return
    path = _script_ckpt_path(cell_id)
    if not os.path.exists(path):
        print(f"[Cell checkpoint] No saved output for skipped cell {cell_id}: {path}")
        return
    load_cell_results(cell_id)
    if cell_id == 6:
        reconstruct_cell6_state()


def resolve_output_dirs():
    out_dir = OUTPUT_DIR or os.path.join(
        PROJECT_ROOT,
        "MRI/derivatives/fMRI_analysis/LSS",
        "results",
        "FearNetwork",
    )
    checkpoint_dir = os.path.join(out_dir, "checkpoints")
    intermediate_dir = os.path.join(out_dir, "intermediate")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(intermediate_dir, exist_ok=True)
    return out_dir, checkpoint_dir, intermediate_dir


OUT_DIR_MAIN, CHECKPOINT_DIR, INTERMEDIATE_DIR = resolve_output_dirs()
print(f"[Runtime] PROJECT_ROOT={PROJECT_ROOT}")
print(f"[Runtime] OUT_DIR_MAIN={OUT_DIR_MAIN}")
print(f"[Runtime] CHECKPOINT_DIR={CHECKPOINT_DIR}")
print(f"[Runtime] ROI_DIR={ROI_DIR}")
if INCLUDE_SUBJECTS_CSV:
    print(
        "[Runtime] Stage 6 subject filter: "
        f"csv={INCLUDE_SUBJECTS_CSV}, flag={INCLUDE_SUBJECTS_FLAG}, "
        f"id_col={INCLUDE_SUBJECTS_COLUMN}, label={ANALYSIS_LABEL or 'unlabeled'}"
    )


def make_safe_label(label):
    """Return a filesystem-safe label for sensitivity checkpoints."""
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(label).strip())
    return cleaned.strip("_")


def normalize_subject_id_for_filter(value):
    """Normalize subject identifiers across SCR CSVs and MVPA trial labels."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(".0"):
        text = text[:-2]
    if text.startswith("sub-"):
        text = text[4:]
    return text


def _truthy_subject_flag(series):
    """Convert common CSV boolean encodings into a boolean mask."""
    if series.dtype == bool:
        return series.fillna(False)
    numeric = pd.to_numeric(series, errors="coerce")
    string_mask = series.astype(str).str.strip().str.lower().isin({"true", "t", "yes", "y", "1"})
    return numeric.fillna(0).ne(0) | string_mask


def load_stage6_include_subjects():
    """Load optional subject IDs for Analysis 1.1 subgroup sensitivity runs."""
    if not INCLUDE_SUBJECTS_CSV:
        return None
    if not os.path.exists(INCLUDE_SUBJECTS_CSV):
        raise FileNotFoundError(f"Subject-filter CSV not found: {INCLUDE_SUBJECTS_CSV}")
    df = pd.read_csv(INCLUDE_SUBJECTS_CSV)
    if INCLUDE_SUBJECTS_COLUMN not in df.columns:
        raise ValueError(
            f"Subject-filter CSV {INCLUDE_SUBJECTS_CSV} does not contain "
            f"'{INCLUDE_SUBJECTS_COLUMN}'. Available columns: {list(df.columns)}"
        )
    if INCLUDE_SUBJECTS_FLAG:
        if INCLUDE_SUBJECTS_FLAG not in df.columns:
            raise ValueError(
                f"Subject-filter CSV {INCLUDE_SUBJECTS_CSV} does not contain "
                f"flag '{INCLUDE_SUBJECTS_FLAG}'. Available columns: {list(df.columns)}"
            )
        df = df[_truthy_subject_flag(df[INCLUDE_SUBJECTS_FLAG])]
    subjects = {
        sub for sub in df[INCLUDE_SUBJECTS_COLUMN].map(normalize_subject_id_for_filter).dropna().tolist()
    }
    if not subjects:
        raise ValueError(
            f"Subject-filter CSV produced zero subjects after applying flag "
            f"{INCLUDE_SUBJECTS_FLAG or '<none>'}."
        )
    return subjects


def apply_stage6_subject_filter(X, y, sub, include_subjects, label):
    if include_subjects is None:
        return X, y, sub
    subject_ids = np.array([normalize_subject_id_for_filter(s) for s in sub])
    keep = np.isin(subject_ids, list(include_subjects))
    X_f, y_f, sub_f = X[keep], y[keep], sub[keep]
    print(
        f"  [FILTER:{label}] retained {len(np.unique(sub_f))}/{len(np.unique(sub))} "
        f"subjects and {len(sub_f)}/{len(sub)} trials."
    )
    if len(np.unique(sub_f)) < 4:
        raise ValueError(
            f"Subject filter {label} retained fewer than four subjects. "
            "Stage 6 decoding would be unstable."
        )
    if len(np.unique(y_f)) < 2:
        raise ValueError(f"Subject filter {label} retained fewer than two classes.")
    return X_f, y_f, sub_f


def _script_ckpt_path(cell_id: int) -> str:
    return os.path.join(CHECKPOINT_DIR, f"cell_{cell_id:02d}.joblib")


def _script_intermediate_path(name: str) -> str:
    return os.path.join(INTERMEDIATE_DIR, f"{name}.joblib")


def save_cell_results(cell_id: int, names):
    payload = {}
    for name in names:
        if name not in globals():
            continue
        value = globals()[name]
        if isinstance(value, types.ModuleType) or callable(value):
            continue
        if value.__class__.__module__.startswith("matplotlib."):
            continue
        try:
            pickle.dumps(value)
        except Exception:
            continue
        payload[name] = value
    if not payload:
        return
    path = _script_ckpt_path(cell_id)
    try:
        joblib.dump(payload, path)
        print(f"[Cell checkpoint] Saved cell {cell_id} -> {path}")
    except Exception as exc:
        print(f"[Cell checkpoint] Failed to save cell {cell_id}: {exc}")


def load_cell_results(cell_id: int):
    path = _script_ckpt_path(cell_id)
    payload = joblib.load(path)
    if isinstance(payload, dict) and len(payload) == 1 and f"results_" in next(iter(payload.keys())):
        pass
    globals().update(payload if isinstance(payload, dict) else {})
    print(f"[Cell checkpoint] Loaded cell {cell_id} <- {path}")
    return payload


def _cosine_alignment_01(vector, target) -> Tuple[float, float]:
    """Return raw cosine and a 0-1 display alignment from true centroid vectors."""
    vector = np.asarray(vector, dtype=float)
    target = np.asarray(target, dtype=float)
    denom = float(np.linalg.norm(vector) * np.linalg.norm(target))
    if not np.isfinite(denom) or denom <= 0:
        return np.nan, np.nan
    cosine = float(np.dot(vector, target) / denom)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return cosine, (cosine + 1.0) / 2.0


def build_true_condition_centroid_geometry(X, y, subjects, group_label, conditions, feature_space=None):
    """Build notebook-facing geometry from true subject condition centroids."""
    rows = []
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y).astype(str)
    sub_arr = np.asarray(subjects).astype(str)
    if X_arr.ndim != 2 or len(X_arr) != len(y_arr) or len(y_arr) != len(sub_arr):
        return pd.DataFrame()

    safety_condition = "CS-"
    threat_condition = "CSR"
    n_features = int(X_arr.shape[1])
    feature_source = feature_space.get(group_label, feature_space) if isinstance(feature_space, dict) else feature_space
    feature_source = str(feature_source) if feature_source is not None else ""
    for subject_id in np.unique(sub_arr):
        subject_mask = sub_arr == subject_id
        centroids = {}
        counts = {}
        for condition in conditions:
            condition_mask = subject_mask & (y_arr == condition)
            if not np.any(condition_mask):
                continue
            centroids[condition] = X_arr[condition_mask].mean(axis=0)
            counts[condition] = int(np.sum(condition_mask))
        if safety_condition not in centroids or threat_condition not in centroids:
            continue
        safety_target = centroids[safety_condition]
        threat_target = centroids[threat_condition]
        for condition in conditions:
            if condition not in centroids:
                continue
            safety_cos, safety_alignment = _cosine_alignment_01(centroids[condition], safety_target)
            threat_cos, threat_alignment = _cosine_alignment_01(centroids[condition], threat_target)
            rows.append({
                "subject_id": subject_id,
                "group": group_label,
                "condition": condition,
                "safety_alignment": safety_alignment,
                "threat_alignment": threat_alignment,
                "safety_cosine": safety_cos,
                "threat_cosine": threat_cos,
                "n_condition_trials": counts[condition],
                "n_features": n_features,
                "feature_space": feature_source,
                "alignment_scale": "cosine_mapped_0_1",
                "source": "true_subject_condition_centroids",
            })
    return pd.DataFrame(rows)


def export_aim2_geometry_panel(geometry_panel):
    """Write true centroid geometry where post-Hyak exporters and the notebook can find it."""
    columns = [
        "subject_id", "group", "condition", "safety_alignment", "threat_alignment",
        "safety_cosine", "threat_cosine", "n_condition_trials", "n_features",
        "feature_space", "alignment_scale", "source",
    ]
    if geometry_panel is None or geometry_panel.empty:
        geometry_panel = pd.DataFrame(columns=columns)
    else:
        geometry_panel = geometry_panel.copy()
        for column in columns:
            if column not in geometry_panel.columns:
                geometry_panel[column] = np.nan
        geometry_panel = geometry_panel[columns]
    for output_dir in (OUT_DIR_MAIN, INTERMEDIATE_DIR):
        os.makedirs(output_dir, exist_ok=True)
        for filename in ("aim2_subject_condition_centroids.csv", "aim2_geometry_panel.csv"):
            path = os.path.join(output_dir, filename)
            geometry_panel.to_csv(path, index=False)
            print(f"[Aim2 export] Wrote true centroid geometry -> {path} ({len(geometry_panel)} rows)")


def _first_existing_column(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def _standardize_aim2_trajectory_frame(df, trajectory):
    if df is None or df.empty:
        return pd.DataFrame()
    subject_col = _first_existing_column(df, ("subject_id", "Subject", "sub_ID", "sub"))
    group_col = _first_existing_column(df, ("group", "Group"))
    drug_col = _first_existing_column(df, ("drug", "Drug"))
    trial_col = _first_existing_column(df, ("trial", "Trial", "block"))
    value_col = _first_existing_column(df, ("value", "score", "similarity", "evidence"))
    if not all((subject_col, group_col, trial_col, value_col)):
        return pd.DataFrame()
    out = pd.DataFrame({
        "subject_id": df[subject_col].astype(str),
        "group": df[group_col].astype(str),
        "drug": df[drug_col].astype(str) if drug_col else pd.NA,
        "trial": pd.to_numeric(df[trial_col], errors="coerce"),
        "trajectory": trajectory,
        "trajectory_metric": "early_to_target_normalized_projection",
        "value": pd.to_numeric(df[value_col], errors="coerce"),
        "source": "true_early_to_target_normalized_projection",
    })
    return out.dropna(subset=["trial", "value"])


def export_aim2_trajectory_panel(trajectory_payload):
    """Write true trial-level trajectories needed by mvpa_l2.ipynb Figure 2."""
    frames = []
    if isinstance(trajectory_payload, dict):
        frames.append(_standardize_aim2_trajectory_frame(
            trajectory_payload.get("data_safe"),
            "CSS projection to CS- safety",
        ))
        frames.append(_standardize_aim2_trajectory_frame(
            trajectory_payload.get("data_threat"),
            "CSR projection to CSR threat",
        ))
        frames.append(_standardize_aim2_trajectory_frame(
            trajectory_payload.get("data_threat_shock"),
            "CSR toward shock target",
        ))
    panel = pd.concat([frame for frame in frames if frame is not None and not frame.empty], ignore_index=True) if frames else pd.DataFrame()
    columns = ["subject_id", "group", "drug", "trial", "trajectory", "trajectory_metric", "value", "source"]
    if panel.empty:
        panel = pd.DataFrame(columns=columns)
    else:
        if "drug" in panel.columns:
            panel = panel[panel["drug"].astype(str).eq("Placebo")].copy()
        panel = panel[columns]
    for output_dir in (OUT_DIR_MAIN, INTERMEDIATE_DIR):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "aim2_trajectory_panel.csv")
        panel.to_csv(path, index=False)
        print(f"[Aim2 export] Wrote true trial trajectories -> {path} ({len(panel)} rows)")


# %% [cell 2]
# cell 2 helper functions
# =============================================================================
# 1. Pipeline & Preprocessing
# =============================================================================
param_grid = {
    'classification__C': np.logspace(C_MIN_EXP, C_MAX_EXP, C_POINTS)
}

def build_binary_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ('classification', LogisticRegression(
            penalty='l2', 
            solver='lbfgs', 
            class_weight='balanced', 
            max_iter=MAX_ITER, 
            random_state=RANDOM_STATE, 
            n_jobs=N_JOBS
        ))
    ])


def get_cv(y, groups=None, n_splits=SUBJECT_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE):
    """Return StratifiedGroupKFold if multiple groups exist; otherwise StratifiedKFold."""
    if groups is None or len(np.unique(groups)) < 2:
        return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=(random_state if shuffle else None))
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=(random_state if shuffle else None))


def effective_cv_splits(y, groups=None, requested_splits=SUBJECT_CV_SPLITS):
    """Choose a CV split count that is valid for small subject-filtered cohorts."""
    y_arr = np.asarray(y)
    if len(np.unique(y_arr)) < 2:
        raise ValueError("CV requires at least two classes.")
    class_counts = [int(np.sum(y_arr == cls)) for cls in np.unique(y_arr)]
    max_by_class = min(class_counts)
    if groups is None or len(np.unique(groups)) < 2:
        max_splits = min(int(requested_splits), max_by_class)
    else:
        groups_arr = np.asarray(groups)
        class_group_counts = [
            len(np.unique(groups_arr[y_arr == cls]))
            for cls in np.unique(y_arr)
        ]
        max_splits = min(int(requested_splits), len(np.unique(groups_arr)), min(class_group_counts), max_by_class)
    if max_splits < 2:
        raise ValueError(
            "CV requires at least two valid splits after filtering. "
            f"Class counts={dict(zip(np.unique(y_arr), class_counts))}."
        )
    return max_splits


def make_valid_cv_splits(X, y, groups=None, requested_splits=SUBJECT_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE, label="CV"):
    """Return checked split indices, dropping empty folds from sparse subgroup CV."""
    y_arr = np.asarray(y)
    groups_arr = None if groups is None else np.asarray(groups)
    n_splits = effective_cv_splits(y_arr, groups_arr, requested_splits)
    cv = get_cv(y_arr, groups_arr, n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    split_iter = cv.split(X, y_arr, groups=groups_arr) if groups_arr is not None else cv.split(X, y_arr)
    valid = []
    for train_idx, test_idx in split_iter:
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        if len(np.unique(y_arr[train_idx])) < 2:
            continue
        valid.append((train_idx, test_idx))
    if len(valid) < 2:
        raise ValueError(
            f"{label} produced fewer than two usable folds after filtering "
            f"(requested={requested_splits}, effective={n_splits}, usable={len(valid)})."
        )
    return valid


def _force_choice_scores_to_2d(scores: np.ndarray) -> np.ndarray:
    """Ensure decision scores are 2D (n_samples, n_classes)."""
    scores_arr = np.asarray(scores)
    if scores_arr.ndim == 1:
        scores_arr = np.column_stack((-scores_arr, scores_arr))
    return scores_arr


def forced_choice_predict(scores: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    """Predict class labels via forced-choice (argmax of decision scores)."""
    scores_2d = _force_choice_scores_to_2d(scores)
    class_arr = np.asarray(classes)
    if scores_2d.shape[1] != class_arr.shape[0]:
        raise ValueError("Decision scores do not match class labels.")
    return class_arr[np.argmax(scores_2d, axis=1)]


def compute_forced_choice_accuracy(
    y_true: np.ndarray,
    scores: np.ndarray,
    classes: Sequence[str],
) -> float:
    """Compute trial-wise forced-choice accuracy from decision scores."""
    y_pred = forced_choice_predict(scores, classes)
    return float(np.mean(y_true == y_pred))


def compute_subject_forced_choice_accs(
    y_true: np.ndarray,
    scores: np.ndarray,
    subjects: np.ndarray,
    classes: Sequence[str],
) -> np.ndarray:
    """Compute per-subject forced-choice accuracies from decision scores."""
    y_pred = forced_choice_predict(scores, classes)
    accs = []
    for sub in np.unique(subjects):
        mask = subjects == sub
        if np.sum(mask) == 0:
            continue
        accs.append(float(np.mean(y_true[mask] == y_pred[mask])))
    return np.array(accs)


def forced_choice_scorer(estimator, X, y) -> float:
    """Scorer wrapper for GridSearchCV/cross_val_score."""
    if len(X) == 0:
        return np.nan
    scores = estimator.decision_function(X)
    return compute_forced_choice_accuracy(y, scores, estimator.classes_)


def decision_margin_scorer(estimator, X, y) -> float:
    """Continuous scorer for permutation importance: true-class score minus strongest alternative."""
    scores_2d = _force_choice_scores_to_2d(estimator.decision_function(X))
    classes = np.asarray(estimator.classes_)
    y_arr = np.asarray(y)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    true_idx = np.array([class_to_idx[label] for label in y_arr])
    true_scores = scores_2d[np.arange(len(y_arr)), true_idx]
    other_scores = scores_2d.copy()
    other_scores[np.arange(len(y_arr)), true_idx] = -np.inf
    margins = true_scores - np.max(other_scores, axis=1)
    return float(np.mean(margins))


def compute_pairwise_forced_choice(y_true, scores, class_labels):
    """Forced-choice accuracy given decision-score columns per class."""
    scores_arr = np.asarray(scores)
    return compute_forced_choice_accuracy(y_true, scores_arr, class_labels)


def run_cross_decoding(model, X, y, groups, classes=None):
    """
    Applies a pre-trained model to a new dataset and computes subject-level
    forced-choice accuracy.
    """
    scores = model.decision_function(X)
    return compute_subject_forced_choice_accs(y, scores, groups, model.classes_)


def run_perm_simple(X, y, groups, n_iters):
    """
    Runs permutation testing iterations for a single job using
    trial-wise forced-choice accuracy.
    """
    scores = []
    y_shuffled = y.copy()

    pipe = build_binary_pipeline()
    cv_splits = make_valid_cv_splits(
        X, y, groups,
        requested_splits=N_SPLITS,
        shuffle=False,
        random_state=RANDOM_STATE,
        label="Permutation CV",
    )

    for _ in range(n_iters):
        np.random.shuffle(y_shuffled)
        cv_scores = cross_val_score(
            pipe,
            X,
            y_shuffled,
            groups=groups,
            cv=cv_splits,
            scoring=forced_choice_scorer,
            n_jobs=N_JOBS,
            error_score=np.nan,
        )
        mean_score = float(np.nanmean(cv_scores))
        if np.isfinite(mean_score):
            scores.append(mean_score)

    return scores

    
def run_cross_perm(model, X, y, subs, n_iter):
    """Cross-decoding permutation using trial-wise forced-choice accuracy."""
    null_scores = []
    mask_c = np.isin(y, model.classes_)
    X_f = X[mask_c]
    y_f = y[mask_c]

    scores = model.decision_function(X_f)

    for _ in range(n_iter):
        y_shuff = np.random.permutation(y_f)
        null_scores.append(compute_forced_choice_accuracy(y_shuff, scores, model.classes_))
    return np.array(null_scores)


def run_loso_self_cross_pairs(target_group, X_target, y_target, sub_target, cross_model, self_c):
    """Return matched subject-level self and cross decoding accuracies."""
    rows = []
    classes = np.asarray(cross_model.classes_)
    keep = np.isin(y_target, classes)
    X_target = np.asarray(X_target)[keep]
    y_target = np.asarray(y_target)[keep]
    sub_target = np.asarray(sub_target)[keep].astype(str)
    for subject_id in pd.Series(sub_target).dropna().astype(str).unique():
        test_mask = sub_target == subject_id
        train_mask = ~test_mask
        if not np.any(test_mask) or not np.any(train_mask):
            continue
        if len(np.unique(y_target[train_mask])) < 2:
            continue
        self_model = build_binary_pipeline()
        self_model.set_params(classification__C=float(self_c))
        self_model.fit(X_target[train_mask], y_target[train_mask])
        self_scores = self_model.decision_function(X_target[test_mask])
        cross_scores = cross_model.decision_function(X_target[test_mask])
        rows.append({
            "subject_id": subject_id,
            "target_group": target_group,
            "within_accuracy": compute_forced_choice_accuracy(y_target[test_mask], self_scores, self_model.classes_),
            "cross_accuracy": compute_forced_choice_accuracy(y_target[test_mask], cross_scores, cross_model.classes_),
            "n_trials": int(np.sum(test_mask)),
            "within_metric_source": "leave_one_subject_out_refit",
            "cross_metric_source": "opposite_group_refit_model",
        })
    return pd.DataFrame(rows)


def paired_sign_flip_drop_test(pairs, n_perm=N_PERMUTATION, seed=RANDOM_STATE):
    """Test mean within-minus-cross decoding drop with paired sign flips."""
    if pairs is None or len(pairs) == 0:
        return {}, np.array([], dtype=float)
    drops = pd.to_numeric(pairs["within_accuracy"], errors="coerce") - pd.to_numeric(pairs["cross_accuracy"], errors="coerce")
    drops = drops.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if drops.size == 0:
        return {}, np.array([], dtype=float)
    observed = float(np.mean(drops))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(int(n_perm), drops.size))
    null = np.mean(signs * drops, axis=1)
    p_value = float((1 + np.sum(null >= observed)) / (len(null) + 1))
    boot_idx = rng.integers(0, drops.size, size=(int(n_perm), drops.size))
    boot_means = np.mean(drops[boot_idx], axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    return {
        "n_pairs": int(drops.size),
        "within_group_accuracy": float(pd.to_numeric(pairs["within_accuracy"], errors="coerce").mean()),
        "cross_group_accuracy": float(pd.to_numeric(pairs["cross_accuracy"], errors="coerce").mean()),
        "functional_drop": observed,
        "drop_ci_low": float(ci_low),
        "drop_ci_high": float(ci_high),
        "functional_drop_p": p_value,
        "n_permutations": int(len(null)),
        "test": "paired_sign_flip_mean_within_minus_cross",
    }, null


def build_functional_drop_outputs(
    X_sad, y_sad, sub_sad, X_hc, y_hc, sub_hc, model_sad, model_hc, best_c_sad, best_c_hc,
    acc_sad_cv=None, acc_hc_cv=None, mean_sad2hc=None, mean_hc2sad=None,
    perm_acc_sad=None, perm_acc_hc=None, perm_sad2hc=None, perm_hc2sad=None
):
    """Build paired rows and aggregate self-vs-cross drop tests."""
    sad_pairs = run_loso_self_cross_pairs("SAD", X_sad, y_sad, sub_sad, model_hc, best_c_sad)
    hc_pairs = run_loso_self_cross_pairs("HC", X_hc, y_hc, sub_hc, model_sad, best_c_hc)
    pairs = pd.concat([sad_pairs, hc_pairs], ignore_index=True) if not sad_pairs.empty or not hc_pairs.empty else pd.DataFrame()
    tests = {}
    nulls = {}
    def finite_vector_local(values):
        if values is None:
            return np.array([], dtype=float)
        arr = np.asarray(values, dtype=float).ravel()
        return arr[np.isfinite(arr)]
    aggregate_specs = {
        "SAD": (acc_sad_cv, mean_hc2sad, perm_acc_sad, perm_hc2sad),
        "HC": (acc_hc_cv, mean_sad2hc, perm_acc_hc, perm_sad2hc),
    }
    for group, (within, cross, within_null, cross_null) in aggregate_specs.items():
        if within is None or cross is None:
            continue
        within = float(within)
        cross = float(cross)
        observed = within - cross
        within_null = finite_vector_local(within_null)
        cross_null = finite_vector_local(cross_null)
        n_null = min(len(within_null), len(cross_null))
        null = within_null[:n_null] - cross_null[:n_null] if n_null else np.array([], dtype=float)
        null = null[np.isfinite(null)]
        if len(null):
            p_value = float((1 + np.sum(null >= observed)) / (len(null) + 1))
            ci_low, ci_high = np.percentile(null, [2.5, 97.5])
        else:
            p_value = np.nan
            ci_low, ci_high = np.nan, np.nan
        tests[group] = {
            "n_pairs": 1,
            "within_group_accuracy": within,
            "cross_group_accuracy": cross,
            "functional_drop": observed,
            "drop_ci_low": float(ci_low) if np.isfinite(ci_low) else np.nan,
            "drop_ci_high": float(ci_high) if np.isfinite(ci_high) else np.nan,
            "functional_drop_p": p_value,
            "n_permutations": int(len(null)),
            "test": "aggregate_self_minus_cross",
        }
        nulls[group] = null
    if not tests:
        for group, group_pairs in pairs.groupby("target_group", dropna=True):
            test, null = paired_sign_flip_drop_test(group_pairs, n_perm=N_PERMUTATION, seed=RANDOM_STATE + (0 if group == "SAD" else 1000))
            if test:
                tests[group] = test
                nulls[group] = null
    return pairs, tests, nulls

    
def run_spatial_perm(seed, maps, groups):
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(groups)
    w_sad_p = np.mean(maps[shuffled == "SAD"], axis=0)
    w_hc_p = np.mean(maps[shuffled == "HC"], axis=0)
    return cosine_similarity(w_sad_p.reshape(1, -1), w_hc_p.reshape(1, -1))[0][0]


def run_pairwise_decoding_analysis(X, y, subjects, n_repeats=10):
    X = np.array(X); y = np.array(y); subjects = np.array(subjects)
    
    classes = np.unique(y); pairs = list(combinations(classes, 2)); results = {}
    
    print(f"\n=== Starting Repeated Pairwise Decoding ({len(pairs)} pairs, {n_repeats} repeats) ===")
    
    for c1, c2 in pairs:
        pair_name = f"{c1} vs {c2}"; print(f"\n--- Analysis: {pair_name} ---")
        mask = np.isin(y, [c1, c2]); X_pair = X[mask]; y_pair = y[mask]; sub_pair = subjects[mask]
        
        # ---------------------------------------------------------------------
        # PHASE 1: EVALUATION (Repeated Nested CV with Forced-Choice)
        # ---------------------------------------------------------------------
        all_repeat_scores = []
        
        for r in range(n_repeats):
            # Use a different random_state for each repeat to get different splits
            # Important: shuffle=True is required for the seed to change the split
            outer_splits = make_valid_cv_splits(
                X_pair, y_pair, sub_pair,
                requested_splits=N_SPLITS,
                shuffle=True,
                random_state=RANDOM_STATE + r,
                label=f"{pair_name} outer CV repeat {r + 1}",
            )
            
            repeat_scores = []
            print(f"  > Repeat {r+1}/{n_repeats}...")
            
            for i, (train_idx, test_idx) in enumerate(outer_splits, 1):
                cv_inner = make_valid_cv_splits(
                    X_pair[train_idx], y_pair[train_idx], sub_pair[train_idx],
                    requested_splits=INNER_CV_SPLITS,
                    shuffle=True,
                    random_state=RANDOM_STATE + r,
                    label=f"{pair_name} inner CV repeat {r + 1} fold {i}",
                )
                # Inner loop for hyperparameter tuning
                gs = GridSearchCV(build_binary_pipeline(), param_grid, cv=cv_inner, scoring=forced_choice_scorer, n_jobs=N_JOBS, error_score="raise")
                gs.fit(X_pair[train_idx], y_pair[train_idx], groups=sub_pair[train_idx])
                
                best_model = gs.best_estimator_
                
                # Forced-Choice logic on the Outer Test Fold
                raw_val = best_model.decision_function(X_pair[test_idx])
                scores_2d = np.column_stack((-raw_val, raw_val)) if raw_val.ndim == 1 else raw_val
                
                val_df = pd.DataFrame(scores_2d, columns=best_model.classes_)
                val_df['sub'] = sub_pair[test_idx]
                val_df['y'] = y_pair[test_idx]
                mean_val = val_df.groupby(['sub', 'y']).mean().reset_index()
                
                fold_fc_acc = compute_pairwise_forced_choice(
                    mean_val['y'].values, 
                    mean_val[best_model.classes_].values, 
                    best_model.classes_
                )
                repeat_scores.append(fold_fc_acc)
            
            all_repeat_scores.extend(repeat_scores)
            
        avg_cv_acc = np.mean(all_repeat_scores)
        std_cv_acc = np.std(all_repeat_scores) # Total variance across all repeats/folds
        if not np.isfinite(avg_cv_acc):
            raise ValueError(f"{pair_name} produced no finite cross-validation accuracy values.")
        print(f"  > Final Mean Forced-Choice Accuracy ({n_repeats} repeats): {avg_cv_acc:.4f} (+/- {std_cv_acc:.4f})")

        # ---------------------------------------------------------------------
        # PHASE 2: MODEL GENERATION (Refit on Full Data)
        # ---------------------------------------------------------------------
        # For the final model, we still refit once using a stable inner CV
        print("  > Generating final model (Refit on full data for Haufe patterns)...")
        cv_inner_final = make_valid_cv_splits(
            X_pair, y_pair, sub_pair,
            requested_splits=INNER_CV_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
            label=f"{pair_name} final-model CV",
        )
        gs_final = GridSearchCV(build_binary_pipeline(), param_grid, cv=cv_inner_final, scoring=forced_choice_scorer, n_jobs=N_JOBS, error_score="raise")
        gs_final.fit(X_pair, y_pair, groups=sub_pair)
        
        final_model = gs_final.best_estimator_
        
        # Haufe Pattern calculation (using variables for stability)
        W = final_model.named_steps['classification'].coef_
        X_scaled = X_pair
        A = np.cov(X_scaled, rowvar=False) @ W.T 
        
        results[pair_name] = {
            'model': final_model,
            'accuracy': avg_cv_acc, 
            'std': std_cv_acc,
            'cv_fold_scores': np.asarray(all_repeat_scores, dtype=float),
            'best_C': gs_final.best_params_['classification__C'], 
            'haufe_pattern': A.flatten(), 
            'classes': final_model.classes_
        }
    return results


def plot_dist_with_thresh(null_dist, obs_val, ax, title, tail='upper', color='gray'):
    # Fallback thresholds if globals not defined
    one_tail_high = globals().get('ONE_TAIL_HIGH', 95)
    one_tail_low = globals().get('ONE_TAIL_LOW', 5)
    two_tail_low = globals().get('TWO_TAIL_LOW', 2.5)
    two_tail_high = globals().get('TWO_TAIL_HIGH', 97.5)
    sns.histplot(null_dist, color='gray', stat='density', kde=True, alpha=0.4, ax=ax, label='Null Dist')
    ax.axvline(obs_val, color='red', lw=2.5, label=f'Obs: {obs_val:.2f}')
    if tail == 'upper':
        thresh = np.percentile(null_dist, one_tail_high)
        ax.axvline(thresh, color='blue', ls='--', lw=2)
        p_val = np.mean(null_dist >= obs_val)
    elif tail == 'lower':
        thresh = np.percentile(null_dist, one_tail_low)
        ax.axvline(thresh, color='blue', ls='--', lw=2)
        p_val = np.mean(null_dist <= obs_val)
    elif tail == 'two-tailed':
        t_low = np.percentile(null_dist, two_tail_low)
        t_high = np.percentile(null_dist, two_tail_high)
        ax.axvline(t_low, color='blue', ls='--', lw=2)
        ax.axvline(t_high, color='blue', ls='--', lw=2)
        p_val = 2 * min(np.mean(null_dist <= obs_val), np.mean(null_dist >= obs_val))
    else:
        raise ValueError(f'Unknown tail: {tail}')
    ax.set_title(f"{title}\n(p = {p_val:.4f})")
    ax.legend(loc='best', fontsize='small')
    return p_val


def make_permutation_chunks(n_permutations, n_jobs):
    """Split permutations across jobs without dropping the remainder."""
    n_permutations = int(n_permutations)
    n_jobs = max(1, int(n_jobs))
    active_jobs = min(n_jobs, max(1, n_permutations))
    base, remainder = divmod(n_permutations, active_jobs)
    return [base + (idx < remainder) for idx in range(active_jobs)]


def make_river_plot_importance(importance_dict, feature_names, top_k=20, title="Neural Signatures"):
    # (Same as before)
    pass 


def get_group_key(sub_id):
    """Returns 'Group_Drug' key (e.g., 'SAD_Placebo') for a subject ID."""
    s_str = str(sub_id).strip()
    
    # Try different ID formats
    conds = None
    if s_str in sub_to_meta: conds = sub_to_meta[s_str]
    elif f"sub-{s_str}" in sub_to_meta: conds = sub_to_meta[f"sub-{s_str}"]
    elif s_str.replace("sub-", "") in sub_to_meta: conds = sub_to_meta[s_str.replace("sub-", "")]
    
    if conds:
        return f"{conds['Group']}_{conds['Drug']}"
    return None


def process_phase_data(X_all, y_all, sub_all, phase_name, keep_conditions=("CSS", "CSR")):
    print(f"\nProcessing {phase_name} Phase...")
    if X_all is None: return {k: None for k in group_keys}
    
    # Storage for results
    grouped_data = {k: {'X': [], 'y': [], 'sub': []} for k in group_keys}
    
    # 1. Identify Unique Subjects
    unique_subs = np.unique(sub_all)
    print(f"  > Found {len(unique_subs)} unique subjects.")
    
    count_missing_meta = 0
    
    for sub in unique_subs:
        # 2. Get Group Key
        g_key = get_group_key(sub)
        if not g_key:
            count_missing_meta += 1
            continue
            
        # 3. Extract Subject's FULL Data
        mask_sub = (sub_all == sub)
        X_sub_full = X_all[mask_sub]
        y_sub_full = y_all[mask_sub]
        
        # 4. CENTER DATA (Full Subject Mean)
        #    We subtract the mean of ALL trials (CS+, CS-, etc.) to preserve true baseline
        sub_mean = np.mean(X_sub_full, axis=0)
        X_sub_centered = X_sub_full - sub_mean
        
        # 5. FILTER CONDITIONS
        mask_cond = np.isin(y_sub_full, list(keep_conditions))
        
        if np.sum(mask_cond) > 0:
            grouped_data[g_key]['X'].append(X_sub_centered[mask_cond])
            grouped_data[g_key]['y'].append(y_sub_full[mask_cond])
            # Create subject ID array matching the filtered length
            grouped_data[g_key]['sub'].append(np.full(np.sum(mask_cond), sub))
            
    if count_missing_meta > 0:
        print(f"  ! Warning: {count_missing_meta} subjects missing metadata skipped.")

    # 6. Final Assembly
    final_output = {}
    for key in group_keys:
        if len(grouped_data[key]['X']) > 0:
            final_output[key] = {
                "X": np.vstack(grouped_data[key]['X']),
                "y": np.concatenate(grouped_data[key]['y']),
                "sub": np.concatenate(grouped_data[key]['sub'])
            }
            n_sub = len(np.unique(final_output[key]['sub']))
            print(f"  [{key}] {phase_name}: {n_sub} subjects | Matrix: {final_output[key]['X'].shape}")
        else:
            final_output[key] = None
            print(f"  [{key}] {phase_name}: No data.")
            
    return final_output


def get_extinction_data(group_key):
    if group_key not in data_subsets:
        raise ValueError(f"Group {group_key} missing from data_subsets.")
    
    phase_data = data_subsets[group_key]['ext']
    if phase_data is None:
        raise ValueError(f"Extinction data missing for {group_key}.")
        
    # X is already centered from Cell 5
    return phase_data["X"], phase_data["y"], phase_data["sub"]


def reconstruct_roi_map(flat_data, roi_names, roi_dir):
    """
    Paints a 1D array of values back into a 3D brain volume by iterating 
    through the specific list of ROI masks.
    """
    # 1. Determine Reference Space (Load first mask)
    first_mask_path = glob.glob(os.path.join(roi_dir, f"*{roi_names[0]}*.nii*"))[0]
    ref_img = nib.load(first_mask_path)
    affine = ref_img.affine
    final_vol = np.zeros(ref_img.shape)
    
    current_idx = 0
    
    # 2. Iterate and Paint
    for name in roi_names:
        # Find file (handle potential suffixes like .nii or .nii.gz)
        fpaths = glob.glob(os.path.join(roi_dir, f"*{name}*.nii*"))
        if not fpaths:
            print(f"  ! Error: Mask for '{name}' not found in {roi_dir}")
            return None
        
        mask_img = nib.load(fpaths[0])
        mask_data = mask_img.get_fdata() > 0 # Boolean mask
        n_voxels = np.sum(mask_data)
        
        # Check if we have enough data left
        if current_idx + n_voxels > len(flat_data):
            print(f"  ! Error: Data mismatch. Feature vector too short for ROI {name}.")
            return None
            
        # Extract chunk and paint
        roi_values = flat_data[current_idx : current_idx + n_voxels]
        final_vol[mask_data] = roi_values # Place values in 3D space
        
        current_idx += n_voxels
        
    # Check if data was fully consumed
    if current_idx != len(flat_data):
         print(f"  ! Warning: {len(flat_data) - current_idx} features were unused (Feature vector longer than ROIs).")

    return nib.Nifti1Image(final_vol, affine)


def compute_haufe_binary_robust(model, X):
    scores = model.decision_function(X)
    return np.dot((X - np.mean(X, axis=0)).T, scores - np.mean(scores)) / (X.shape[0] - 1)


def get_robust_weights(X, y, subjects, pipeline, n_boot=N_BOOT):
    unique_subs = np.unique(subjects)
    accumulated_weights = np.zeros(X.shape[1])
    for i in range(n_boot):
        boot_subs = resample(unique_subs, replace=True, random_state=i)
        X_boot_list, y_boot_list = [], []
        for sub in boot_subs:
            mask = (subjects == sub)
            X_sub = X[mask]
            X_boot_list.append(X_sub - np.mean(X_sub, axis=0))
            y_boot_list.append(y[mask])
        X_boot = np.vstack(X_boot_list)
        y_boot = np.hstack(y_boot_list)
        
        clf = clone(pipeline)
        clf.fit(X_boot, y_boot)
        accumulated_weights += compute_haufe_binary_robust(clf, X_boot)
    return accumulated_weights / n_boot


def run_wen_paper_analysis_voxelwise(X, y, subjects, pipeline_template, best_C, n_permutations):
    print(f"  Estimating Weights ({n_permutations} perms)...")
    pipe = clone(pipeline_template); pipe.set_params(classification__C=best_C)
    obs_weights = get_robust_weights(X, y, subjects, pipe, n_boot=N_BOOT)
    
    def run_null(i):
        y_shuff = shuffle(y, random_state=i)
        return get_robust_weights(X, y_shuff, subjects, pipe, n_boot=N_BOOT)

    null_weights_list = Parallel(n_jobs=N_JOBS, verbose=1)(delayed(run_null)(i) for i in range(n_permutations))
    null_weights = np.array(null_weights_list)
    
    null_mean = np.mean(null_weights, axis=0)
    null_std = np.std(null_weights, axis=0)
    z_scores = (obs_weights - null_mean) / (null_std + 1e-12)
    
    n_extreme = np.sum(np.abs(null_weights) >= np.abs(obs_weights), axis=0)
    p_values = (n_extreme + 1) / (n_permutations + 1)
    reject, _, _, _ = multipletests(p_values, alpha=fdr_alpha, method='fdr_bh')
    
    return z_scores, reject


def compute_perm_importance_cv(
    model_template,
    X,
    y,
    groups,
    n_repeats=10,
    n_splits=5
):
    """Cross-validated permutation importance.

    Fits a cloned model on each training fold and computes permutation
    importance on the corresponding test fold to estimate generalization.
    """
    from sklearn.inspection import permutation_importance

    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)

    cv = get_cv(y, groups, n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_importances = []

    for train_idx, test_idx in cv.split(X, y, groups=groups):
        model = clone(model_template)
        model.fit(X[train_idx], y[train_idx])
        result = permutation_importance(
            model,
            X[test_idx],
            y[test_idx],
            n_repeats=n_repeats,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            scoring=forced_choice_scorer
        )
        fold_importances.append(result.importances_mean)

    return np.mean(fold_importances, axis=0)


def extract_metrics(rdms):
    # Metric A: Threat (CSR) vs Safety (CSS)
    m_a = rdms[:, idx_csr, idx_css]
    # Metric B: Safety (CSS) vs Baseline (CS-)
    m_b = rdms[:, idx_css, idx_cs_minus]
    return m_a, m_b


def one_sample_test(data, name):
    # Test if distance is greater than 0
    t_val, p_val = ttest_1samp(data, 0, alternative='greater')
    sig = "*" if p_val < 0.05 else "ns"
    print(f"  > {name}: Mean={np.mean(data):.3f}, t={t_val:.3f}, p={p_val:.4f} ({sig})")
    return p_val


def perm_ttest_ind(data1, data2, n_perm=N_PERMUTATION):
    """
    Performs a permutation t-test for two independent samples.
    Returns: t-stat, p-value, mean1, mean2
    """
    from scipy.stats import ttest_ind
    
    # 1. Calculate observed t-statistic
    t_obs, _ = ttest_ind(data1, data2)
    
    # 2. Permutation loop
    pooled = np.concatenate([data1, data2])
    n1 = len(data1)
    null_dist = []
    
    rng = np.random.default_rng(42) # Fixed seed
    
    for _ in range(n_perm):
        shuffled = rng.permutation(pooled)
        # Split into two groups of same size as originals
        g1 = shuffled[:n1]
        g2 = shuffled[n1:]
        
        # Calculate t-stat for shuffled data
        t_shuff, _ = ttest_ind(g1, g2)
        null_dist.append(t_shuff)
        
    null_dist = np.array(null_dist)
    
    # 3. Calculate P-value (Two-tailed)
    # Proportion of null t-stats more extreme than observed t
    p_val = np.mean(np.abs(null_dist) >= np.abs(t_obs))
    
    return t_obs, p_val, np.mean(data1), np.mean(data2)


def get_sig_star(p): return "*" if p < 0.05 else "ns"


def get_phase_data(group, phase):
    try:
        d = data_subsets[group][phase]
        if d is None: return None, None, None
        return d["X"], d["y"], d["sub"]
    except KeyError:
        return None, None, None


# DEFAULT_SHOCK_TARGET_LABELS = [
#     "Shock", "SHOCK", "shock", "US", "UCS", "unconditioned_stimulus",
#     "Shock1", "Shock2", "Shock3", "US1", "US2", "US3",
# ]
DEFAULT_SHOCK_TARGET_LABELS = [
     "Shock", "SHOCK", "shock"
 ]
SHOCK_TARGET_LABELS = [
    item.strip()
    for item in os.environ.get("SHOCK_TARGET_LABELS", ",".join(DEFAULT_SHOCK_TARGET_LABELS)).split(",")
    if item.strip()
]


def _condition_mask(labels, condition):
    if isinstance(condition, (list, tuple, set, np.ndarray)):
        return np.isin(labels, list(condition))
    return labels == condition


def _subject_matches_group_key(sub, group_key):
    if "sub_to_meta" not in globals() or not isinstance(sub_to_meta, dict):
        return False
    try:
        group, drug = group_key.split("_", 1)
    except ValueError:
        return False
    s = str(sub).strip()
    info = sub_to_meta.get(s) or sub_to_meta.get(f"sub-{s}") or sub_to_meta.get(s.replace("sub-", ""))
    if not isinstance(info, dict):
        return False
    return str(info.get("Group")) == group and str(info.get("Drug")) == drug


def get_shock_target_data(group_key, labels=None):
    labels = SHOCK_TARGET_LABELS if labels is None else labels
    parts = []
    for X_name, y_name, sub_name in [
        ("X_ext_all", "y_ext_all", "sub_ext_all"),
        ("X_reinst_all", "y_reinst_all", "sub_reinst_all"),
    ]:
        if not all(name in globals() for name in [X_name, y_name, sub_name]):
            continue
        X_all = globals()[X_name]
        y_all = globals()[y_name]
        sub_all = globals()[sub_name]
        mask = _condition_mask(y_all, labels) & np.array([_subject_matches_group_key(s, group_key) for s in sub_all], dtype=bool)
        if np.any(mask):
            parts.append((X_all[mask], y_all[mask], sub_all[mask]))
    if not parts:
        return None, None, None
    return np.vstack([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts])


def calculate_residualized_shock_anchor_projection(X_cs, y_cs, sub_cs, X_shock, y_shock, sub_shock):
    """Project CS patterns onto a subject-specific residualized Shock-minus-CS- axis."""
    rows = []
    if X_cs is None or X_shock is None or X_cs.shape[1] == 0:
        return pd.DataFrame(rows)

    X_cs_resid = X_cs - np.mean(X_cs, axis=1, keepdims=True)
    X_shock_resid = X_shock - np.mean(X_shock, axis=1, keepdims=True)
    shared_subjects = np.intersect1d(np.unique(sub_cs), np.unique(sub_shock))

    for sub in shared_subjects:
        m_base = (sub_cs == sub) & _condition_mask(y_cs, "CS-")
        m_css = (sub_cs == sub) & _condition_mask(y_cs, "CSS")
        m_csr = (sub_cs == sub) & _condition_mask(y_cs, "CSR")
        m_shock = (sub_shock == sub)
        if not (np.any(m_base) and np.any(m_css) and np.any(m_csr) and np.any(m_shock)):
            continue

        base = np.mean(X_cs_resid[m_base], axis=0)
        css = np.mean(X_cs_resid[m_css], axis=0)
        csr = np.mean(X_cs_resid[m_csr], axis=0)
        shock = np.mean(X_shock_resid[m_shock], axis=0)
        shock_axis = shock - base
        axis_norm = np.linalg.norm(shock_axis)
        if not np.isfinite(axis_norm) or axis_norm <= 0:
            continue
        shock_axis = shock_axis / axis_norm

        def _projection_metrics(vec):
            delta = vec - base
            delta_norm = np.linalg.norm(delta)
            projection = float(np.dot(delta, shock_axis))
            cosine = np.nan if delta_norm <= 0 else float(projection / delta_norm)
            return projection, cosine

        proj_css, cos_css = _projection_metrics(css)
        proj_csr, cos_csr = _projection_metrics(csr)
        rows.append({
            "Subject": sub,
            "Projection_CSS": proj_css,
            "Projection_CSR": proj_csr,
            "Projection_CSR_minus_CSS": proj_csr - proj_css,
            "Cosine_CSS": cos_css,
            "Cosine_CSR": cos_csr,
            "Cosine_CSR_minus_CSS": cos_csr - cos_css if np.isfinite(cos_css) and np.isfinite(cos_csr) else np.nan,
            "Shock_Axis_Norm": float(axis_norm),
            "n_shock_trials": int(np.sum(m_shock)),
        })
    return pd.DataFrame(rows)


def summarize_shock_anchor_projection(df_sad, df_hc, n_perm=N_PERMUTATION):
    metrics = [
        "Projection_CSS",
        "Projection_CSR",
        "Projection_CSR_minus_CSS",
        "Cosine_CSS",
        "Cosine_CSR",
        "Cosine_CSR_minus_CSS",
        "Shock_Axis_Norm",
    ]
    stats = {}
    for metric in metrics:
        if metric not in df_sad.columns or metric not in df_hc.columns:
            continue
        sad_vals = pd.to_numeric(df_sad[metric], errors="coerce").dropna().to_numpy()
        hc_vals = pd.to_numeric(df_hc[metric], errors="coerce").dropna().to_numpy()
        if len(sad_vals) < 2 or len(hc_vals) < 2:
            stats[metric] = {
                "SAD": sad_vals,
                "HC": hc_vals,
                "means": {"SAD": np.nan, "HC": np.nan},
                "group_stats": (np.nan, np.nan),
            }
            continue
        t_val, p_val, m_sad, m_hc = perm_ttest_ind(sad_vals, hc_vals, n_perm=n_perm)
        stats[metric] = {
            "SAD": sad_vals,
            "HC": hc_vals,
            "means": {"SAD": m_sad, "HC": m_hc},
            "group_stats": (t_val, p_val),
        }
    return stats


def print_shock_anchor_summary(shock_anchor_stats):
    if not isinstance(shock_anchor_stats, dict) or not shock_anchor_stats:
        print("\n[Shock-anchor sensitivity] Unavailable.")
        return
    print("\n[Shock-anchor sensitivity] Residualized projection onto subject-specific Shock-minus-CS- axis")
    for metric in ["Projection_CSS", "Projection_CSR", "Projection_CSR_minus_CSS", "Cosine_CSS", "Cosine_CSR", "Cosine_CSR_minus_CSS"]:
        if metric not in shock_anchor_stats:
            continue
        info = shock_anchor_stats[metric]
        means = info.get("means", {})
        t_val, p_val = info.get("group_stats", (np.nan, np.nan))
        print(
            f"  {metric:<24}: SAD={means.get('SAD', np.nan):.6f}, "
            f"HC={means.get('HC', np.nan):.6f} | t(GroupDiff)={t_val:.3f}, p={p_val:.4f}"
        )


def calculate_plasticity_vectors(X_learn, y_learn, sub_learn, X_targ, y_targ, sub_targ, mask, cond_l, cond_t):
    """
    Calculates representational drift between a learning state and a target state.
    """
    unique_subs = np.unique(sub_learn)
    res = {'sub': [], 'projection': [], 'cosine': [], 'init_dist': []}

    for sub in unique_subs:
        # 1. Slice Learning Data (e.g., Extinction CSS)
        mask_l = (sub_learn == sub) & _condition_mask(y_learn, cond_l)
        # 2. Slice Target Data (e.g., Extinction CS- or Reinstatement CSR)
        mask_t = (sub_targ == sub) & _condition_mask(y_targ, cond_t)
        
        # Check if subject has data for both conditions
        if np.sum(mask_l) == 0 or np.sum(mask_t) == 0:
            continue
            
        # Apply the Voxel Mask (Significant features only)
        vec_l = np.mean(X_learn[mask_l][:, mask], axis=0)
        vec_t = np.mean(X_targ[mask_t][:, mask], axis=0)
        
        # Vector Math: Learning is a vector from origin to vec_l
        # But specifically, we want to see how vec_l moves toward vec_t
        
        # Magnitude (Scalar Projection of vec_l onto vec_t)
        proj = np.dot(vec_l, vec_t) / np.linalg.norm(vec_t)
        
        # Fidelity (Cosine Similarity)
        cos = np.dot(vec_l, vec_t) / (np.linalg.norm(vec_l) * np.linalg.norm(vec_t))
        
        # Initial Distance (How far they were apart)
        dist = np.linalg.norm(vec_l - vec_t)
        
        res['sub'].append(sub)
        res['projection'].append(proj)
        res['cosine'].append(cos)
        res['init_dist'].append(dist)
        
    return pd.DataFrame(res)


def tag_df(df, grp, cond):
    if df.empty: return df
    d = df.copy(); d['Group'] = grp; d['Condition'] = cond
    return d


def calc_trajectory(X_learn, y_learn, sub_learn, X_targ, y_targ, sub_targ, mask, cond_l, cond_t):
    """
    Scores individual learning trials by normalized projection from early learning to target.
    """
    unique_subs = np.unique(sub_learn)
    res = {'sub': [], 'trial': [], 'score': []}

    for sub in unique_subs:
        # 1. Get Subject Data
        mask_sub_l = (sub_learn == sub) & _condition_mask(y_learn, cond_l)
        mask_sub_t = (sub_targ == sub) & _condition_mask(y_targ, cond_t)
        
        if np.sum(mask_sub_l) < 2 or np.sum(mask_sub_t) == 0:
            continue
            
        xl = X_learn[mask_sub_l][:, mask]
        xt = X_targ[mask_sub_t][:, mask]

        cutoff = max(1, len(xl) // 2)
        p_start = np.mean(xl[:cutoff], axis=0)
        p_target = np.mean(xt, axis=0)
        axis = p_target - p_start
        sq_norm = np.dot(axis, axis)
        if sq_norm == 0:
            continue

        scores = np.dot(xl - p_start, axis) / sq_norm
        for trial, score in enumerate(scores, 1):
            res['sub'].append(sub)
            res['trial'].append(trial)
            res['score'].append(score)
            
    return pd.DataFrame(res)



def run_detailed_stats(df_sad, df_hc, label):
    if df_sad.empty or df_hc.empty: return pd.DataFrame()
    
    trials = sorted(list(set(df_sad['trial'].unique()) & set(df_hc['trial'].unique())))
    results = []
    
    for t in trials:
        s_vals = df_sad[df_sad['trial'] == t]['score'].values
        h_vals = df_hc[df_hc['trial'] == t]['score'].values
        
        # A. SAD > 0
        t_s, p_s = ttest_1samp(s_vals, 0, alternative='greater')
        df_s = len(s_vals) - 1
        
        # B. HC > 0
        t_h, p_h = ttest_1samp(h_vals, 0, alternative='greater')
        df_h = len(h_vals) - 1
        
        # C. SAD != HC
        t_d, p_d = ttest_ind(s_vals, h_vals)
        df_d = len(s_vals) + len(h_vals) - 2
        
        results.append({
            'Trial': t,
            'SAD_t': t_s, 'SAD_df': df_s, 'SAD_p': p_s,
            'HC_t': t_h, 'HC_df': df_h, 'HC_p': p_h,
            'Diff_t': t_d, 'Diff_df': df_d, 'Diff_p': p_d
        })
        
    stats_df = pd.DataFrame(results)
    
    # FDR Correction
    if not stats_df.empty:
        _, stats_df['SAD_p_fdr'], _, _ = multipletests(stats_df['SAD_p'], alpha=0.05, method='fdr_bh')
        _, stats_df['HC_p_fdr'], _, _ = multipletests(stats_df['HC_p'], alpha=0.05, method='fdr_bh')
        _, stats_df['Diff_p_fdr'], _, _ = multipletests(stats_df['Diff_p'], alpha=0.05, method='fdr_bh')
        
    print(f"\n--- Statistics: {label} ---")
    # Print significant trials (Diff)
    sig_diff = stats_df[stats_df['Diff_p_fdr'] < 0.05]
    if not sig_diff.empty:
        print("Significant Group Differences (FDR < 0.05):")
        print(sig_diff[['Trial', 'Diff_t', 'Diff_df', 'Diff_p', 'Diff_p_fdr']].to_string(index=False))
    else:
        print("No significant group differences found (FDR corrected).")
        
    return stats_df


def prepare_plot(df_sad, df_hc, name):
    if df_sad.empty and df_hc.empty: return pd.DataFrame()
    d_list = []
    if not df_sad.empty:
        d1 = df_sad.copy(); d1['Group'] = 'SAD'; d_list.append(d1)
    if not df_hc.empty:
        d2 = df_hc.copy();  d2['Group'] = 'HC'; d_list.append(d2)
    
    if not d_list: return pd.DataFrame()
    
    df = pd.concat(d_list)
    df['Condition'] = name
    # Bin trials if needed
    if BLOCK_SIZE > 1:
        df['trial'] = ((df['trial'] - 1) // BLOCK_SIZE) + 1
    return df


def prepare_group_drug_plot(frames, name):
    d_list = []
    for group, drug, frame in frames:
        if frame is None or frame.empty:
            continue
        d = frame.copy()
        d["Group"] = group
        d["Drug"] = drug
        d_list.append(d)
    if not d_list:
        return pd.DataFrame()
    df = pd.concat(d_list, ignore_index=True)
    df["Condition"] = name
    if BLOCK_SIZE > 1:
        df["trial"] = ((df["trial"] - 1) // BLOCK_SIZE) + 1
    return df


def get_significant_mask(scores): return scores > 0


EPS = 1e-8
PROTOTYPE_EPS = 1e-8


def zscore_subject_trials(x):
    x = x.astype(np.float64, copy=False)
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd[~np.isfinite(sd) | (sd < PROTOTYPE_EPS)] = 1.0
    out = (x - mu) / sd
    return np.nan_to_num(out, copy=False)


def corr_distance_vector(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < PROTOTYPE_EPS:
        return np.nan
    corr = float(np.dot(a, b) / denom)
    return 1.0 - float(np.clip(corr, -1.0, 1.0))


def softmax_threat_evidence(d_background, d_threat):
    if not np.isfinite(d_background) or not np.isfinite(d_threat):
        return np.nan
    scores = np.array([-d_background, -d_threat], dtype=float)
    scores -= scores.max()
    probs = np.exp(scores) / np.exp(scores).sum()
    return float(probs[1])


def prototype_decision_metrics(X_sub, y_sub, cond_threat, cond_safe, cond_background="CS-"):
    """Match the representative-index prototype evidence metric definitions."""
    needed = (cond_background, cond_safe, cond_threat)
    if any(np.sum(y_sub == label) == 0 for label in needed):
        return np.nan, np.nan, np.nan
    xz = zscore_subject_trials(X_sub)
    centroids = {label: xz[y_sub == label].mean(axis=0) for label in needed}
    p_threat_css = softmax_threat_evidence(
        corr_distance_vector(centroids[cond_safe], centroids[cond_background]),
        corr_distance_vector(centroids[cond_safe], centroids[cond_threat]),
    )
    p_threat_csr = softmax_threat_evidence(
        corr_distance_vector(centroids[cond_threat], centroids[cond_background]),
        corr_distance_vector(centroids[cond_threat], centroids[cond_threat]),
    )
    boundary = p_threat_csr - p_threat_css if np.isfinite(p_threat_csr) and np.isfinite(p_threat_css) else np.nan
    return p_threat_css, p_threat_csr, boundary


def corr_similarity_vector(a, b):
    dist = corr_distance_vector(a, b)
    if not np.isfinite(dist):
        return np.nan
    return 1.0 - dist


def rmssd(values):
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return np.nan
    return float(np.sqrt(np.mean(np.diff(arr) ** 2)))


def heldout_decoder_evidence_metrics(X_sub, y_sub, cond_threat="CSR", cond_safe="CSS", c_val=1.0):
    """Return held-out decoder evidence matching the PROJECT_CONTEXT metric definitions."""
    mask_binary = np.isin(y_sub, [cond_safe, cond_threat])
    X_binary = np.asarray(X_sub[mask_binary], dtype=float)
    y_binary = np.asarray(y_sub[mask_binary])
    if len(y_binary) < MIN_TRIALS_PER_SUBJECT or len(np.unique(y_binary)) < 2:
        return {}
    if min(np.sum(y_binary == cond_safe), np.sum(y_binary == cond_threat)) < 2:
        return {}

    model = build_binary_pipeline()
    model.set_params(classification__C=c_val)
    trial_groups = np.arange(len(y_binary))
    probs_threat = np.full(len(y_binary), np.nan, dtype=float)
    try:
        for train_idx, test_idx in LeaveOneGroupOut().split(X_binary, y_binary, groups=trial_groups):
            if len(np.unique(y_binary[train_idx])) < 2:
                continue
            fold_model = clone(model)
            fold_model.fit(X_binary[train_idx], y_binary[train_idx])
            if cond_threat not in fold_model.classes_:
                continue
            threat_idx = np.where(fold_model.classes_ == cond_threat)[0][0]
            probs_threat[test_idx] = fold_model.predict_proba(X_binary[test_idx])[:, threat_idx]
    except Exception:
        return {}

    labels = y_binary
    probs_css = probs_threat[labels == cond_safe]
    probs_csr = probs_threat[labels == cond_threat]
    probs_css = probs_css[np.isfinite(probs_css)]
    probs_csr = probs_csr[np.isfinite(probs_csr)]
    if len(probs_css) == 0 or len(probs_csr) == 0:
        return {}

    p_threat_css = float(np.mean(probs_css))
    p_safety_css = 1.0 - p_threat_css
    p_threat_csr = float(np.mean(probs_csr))
    safety_evidence = 1.0 - probs_threat[labels == cond_safe]
    threat_evidence = probs_threat[labels == cond_threat]
    n_dynamic = min(len(safety_evidence), len(threat_evidence))
    if n_dynamic:
        dynamic_discrimination = np.asarray(threat_evidence[:n_dynamic], dtype=float) - np.asarray(safety_evidence[:n_dynamic], dtype=float)
    else:
        dynamic_discrimination = np.asarray([], dtype=float)

    css_certainty = 2.0 * abs(p_safety_css - 0.5)
    csr_certainty = 2.0 * abs(p_threat_csr - 0.5)
    return {
        "Neural_SafetyEvidence": p_safety_css,
        "Neural_ThreatEvidence": p_threat_csr,
        "Neural_ThreatLike_Safety": p_threat_css,
        "Neural_Threat_Evidence_CSR": p_threat_csr,
        "Neural_Certainty_CSS": css_certainty,
        "Neural_Certainty_CSR": csr_certainty,
        "Prototype_Certainty": float(np.nanmean([css_certainty, csr_certainty])),
        "Neural_Boundary_Separation": p_threat_csr - p_threat_css,
        "Neural_DynamicDiscrimination_Volatility": rmssd(dynamic_discrimination),
    }


def representative_core_neural_metrics(X_sub, y_sub):
    """Return the four primary neural metrics using representative-index definitions."""
    needed = ("CS-", "CSS", "CSR")
    if any(np.sum(_condition_mask(y_sub, label)) == 0 for label in needed):
        return {}
    xz = zscore_subject_trials(X_sub)
    centroids = {label: xz[_condition_mask(y_sub, label)].mean(axis=0) for label in needed}

    d_safety_background = corr_distance_vector(centroids["CSS"], centroids["CS-"])
    d_threat_safety = corr_distance_vector(centroids["CSR"], centroids["CSS"])
    d_threat_background = corr_distance_vector(centroids["CSR"], centroids["CS-"])
    p_proto_threat_css = softmax_threat_evidence(
        corr_distance_vector(centroids["CSS"], centroids["CS-"]),
        corr_distance_vector(centroids["CSS"], centroids["CSR"]),
    )
    p_proto_threat_csr = softmax_threat_evidence(
        corr_distance_vector(centroids["CSR"], centroids["CS-"]),
        corr_distance_vector(centroids["CSR"], centroids["CSR"]),
    )

    safety_contrast = []
    threat_contrast = []
    for label, holder in (("CSS", safety_contrast), ("CSR", threat_contrast)):
        for trial in xz[_condition_mask(y_sub, label)]:
            sim_bg = corr_similarity_vector(trial, centroids["CS-"])
            sim_threat = corr_similarity_vector(trial, centroids["CSR"])
            holder.append(sim_bg - sim_threat if label == "CSS" else sim_threat - sim_bg)
    n_dynamic = min(len(safety_contrast), len(threat_contrast))
    if n_dynamic:
        dynamic_discrimination = np.asarray(threat_contrast[:n_dynamic], dtype=float) - np.asarray(safety_contrast[:n_dynamic], dtype=float)
    else:
        dynamic_discrimination = np.asarray([], dtype=float)
    threat_safety_denominator = d_threat_background + d_safety_background
    neural_threat_safety_distance = (
        (d_threat_background - d_safety_background) / threat_safety_denominator
        if np.isfinite(threat_safety_denominator) and abs(threat_safety_denominator) > EPS
        else np.nan
    )

    metrics = {
        "Neural_Dist_Safety_Background": d_safety_background,
        "Neural_Dist_Threat_Safety": d_threat_safety,
        "Neural_Dist_Threat_Background": d_threat_background,
        "Neural_Threat_Safety_Distance": neural_threat_safety_distance,
        "Neural_ThreatTriangleOpenness": d_threat_background - d_safety_background,
        "Neural_PrototypeThreatLike_Safety": p_proto_threat_css,
        "Neural_PrototypeThreatLike_Threat": p_proto_threat_csr,
        "Neural_PrototypeBoundary_Separation": p_proto_threat_csr - p_proto_threat_css,
        "Neural_SafetyEvidence": np.nan,
        "Neural_ThreatEvidence": np.nan,
        "Neural_ThreatLike_Safety": np.nan,
        "Neural_Threat_Evidence_CSR": np.nan,
        "Neural_Boundary_Separation": np.nan,
        "Neural_Certainty_CSS": np.nan,
        "Neural_Certainty_CSR": np.nan,
        "Prototype_Certainty": np.nan,
    }
    metrics.update(heldout_decoder_evidence_metrics(X_sub, y_sub))
    if "Neural_DynamicDiscrimination_Volatility" not in metrics:
        metrics["Neural_DynamicDiscrimination_Volatility"] = np.nan
    return metrics


def representative_core_metrics_from_data_subsets(data_subsets, group_masks):
    rows = []
    for key, payload in data_subsets.items():
        if not isinstance(payload, dict) or "ext" not in payload or payload["ext"] is None:
            continue
        try:
            group, drug = key.split("_", 1)
        except ValueError:
            continue
        if group not in group_masks:
            continue
        ext = payload["ext"]
        X, y, subs = ext["X"], ext["y"], ext["sub"]
        curr_mask = group_masks[group]
        for sub in np.unique(subs):
            m = subs == sub
            metrics = representative_core_neural_metrics(X[m][:, curr_mask], y[m])
            if not metrics:
                continue
            metrics.update({"sub_ID": str(sub), "Group": group, "Drug": drug})
            rows.append(metrics)
    return pd.DataFrame(rows)


def calculate_distribution_stats(X, y, subjects, feature_mask, best_params_dict):
    # Slice Features & Center
    X_masked = X[:, feature_mask]
    
    unique_subs = np.unique(subjects)
    res = {
        'sub': [], 'entropy': [], 'kurtosis': [], 'variance': [], 'probabilities': [],
        'probabilities_csr': [], 'p_csr_css': [], 'p_csr_csr': [], 'boundary_separation': [],
        'decision_margin_css': [], 'decision_margin_csr': [], 'decision_margin_all': [],
        'entropy_css': [], 'entropy_csr': []
    }
    
    for sub in unique_subs:
        c_val = best_params_dict.get(sub, 1.0)
        mask_sub = (subjects == sub)
        X_sub = X_masked[mask_sub]; y_sub = y[mask_sub]
        
        # Filter Boundary Classes
        mask_binary = np.isin(y_sub, [COND_CLASS_THREAT, COND_CLASS_SAFE])
        X_binary = X_sub[mask_binary]; y_binary = y_sub[mask_binary]
        
        if len(y_binary) < MIN_TRIALS_PER_SUBJECT: continue
        
        try:
            # Configure Model
            fixed_model = build_binary_pipeline()
            fixed_model.set_params(classification__C=c_val)
            
            # Cross-Validation
            cv = get_cv(y_binary, np.full(len(y_binary), sub), n_splits=SUBJECT_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
            calib_model = CalibratedClassifierCV(
                fixed_model,
                method="sigmoid",
                cv=3
            )
            probs_all = cross_val_predict(calib_model, X_binary, y_binary, groups=np.full(len(y_binary), sub), cv=cv, method='predict_proba', n_jobs=N_JOBS)
            
            # Extract Safety Cue Probabilities (P(Threat | Safety Cue))
            classes = sorted(np.unique(y_binary))
            if COND_CLASS_THREAT not in classes: continue
            idx_threat = classes.index(COND_CLASS_THREAT)
            
            mask_css = (y_binary == COND_CLASS_SAFE)
            if np.sum(mask_css) == 0: continue
            probs_css = probs_all[mask_css, idx_threat]
            probs_csr = probs_all[y_binary == COND_CLASS_THREAT, idx_threat]
            p_proto_csr_css, p_proto_csr_csr, proto_boundary_separation = prototype_decision_metrics(
                X_sub, y_sub, COND_CLASS_THREAT, COND_CLASS_SAFE
            )
            p_csr_css = float(np.mean(probs_css))
            p_csr_csr = float(np.mean(probs_csr))
            boundary_separation = p_csr_csr - p_csr_css
            
            # Metrics
            # 1. Entropy
            p_clean = np.clip(probs_css, 1e-9, 1-1e-9)
            trial_entropies = [entropy([p, 1-p], base=2) for p in p_clean]
            p_csr_clean = np.clip(probs_csr, 1e-9, 1-1e-9)
            trial_entropies_csr = [entropy([p, 1-p], base=2) for p in p_csr_clean]
            
            # 2. Kurtosis (Fisher's definition, Normal = 0.0)
            k_val = kurtosis(probs_css, fisher=True)
            
            # 3. Variance
            v_val = np.var(probs_css)
            
            res['sub'].append(sub)
            res['entropy'].append(np.mean(trial_entropies))
            res['kurtosis'].append(k_val)
            res['variance'].append(v_val)
            res['probabilities'].append(probs_css)
            res['probabilities_csr'].append(probs_csr)
            res['p_csr_css'].append(p_csr_css)
            res['p_csr_csr'].append(p_csr_csr)
            res['boundary_separation'].append(boundary_separation)
            res.setdefault('prototype_p_csr_css', []).append(p_proto_csr_css)
            res.setdefault('prototype_p_csr_csr', []).append(p_proto_csr_csr)
            res.setdefault('prototype_boundary_separation', []).append(proto_boundary_separation)
            res['decision_margin_css'].append(float(0.5 - p_csr_css) if np.isfinite(p_csr_css) else np.nan)
            res['decision_margin_csr'].append(float(p_csr_csr - 0.5) if np.isfinite(p_csr_csr) else np.nan)
            res['decision_margin_all'].append(float(np.mean(np.abs(probs_all[:, idx_threat] - 0.5))))
            res['entropy_css'].append(float(np.mean(trial_entropies)))
            res['entropy_csr'].append(float(np.mean(trial_entropies_csr)) if len(trial_entropies_csr) else np.nan)
            
        except Exception as e:
            # print(f"  ! Subject {sub} failed: {e}")
            pass
            
    return pd.DataFrame(res)


def get_ext_data(group_key):
    if group_key not in data_subsets: raise ValueError(f"{group_key} missing.")
    d = data_subsets[group_key]['ext']
    return d["X"], d["y"], d["sub"]


def compare_metric(vec1, vec2, metric_name):
    print(f"\n--- Metric: {metric_name} ---")
    if len(vec1) == 0 or len(vec2) == 0:
        print("  ! Insufficient data.")
        return 1.0
        
    print(f"  > SAD Mean: {np.mean(vec1):.3f}")
    print(f"  > HC Mean:  {np.mean(vec2):.3f}")
    
    t, p, _, _ = perm_ttest_ind(vec1, vec2, n_perm=N_PERMUTATION)
    sig = "*" if p < 0.05 else "ns"
    print(f"  > Comparison: t={t:.3f}, p={p:.4f} ({sig})")
    return p


def run_lme(formula, data, title):
    print(f"\n--- {title} ---")
    # Groups='Subject' handles random intercepts per subject
    # If design is between-subject, this converges to GLM/ANOVA but handles missingness better
    md = smf.mixedlm(formula, data, groups=data["Subject"]) 
    try:
        mdf = md.fit()
        print(mdf.summary())
        
        # Extract Interaction P-Value safely
        term = "C(Group, Treatment(reference='HC'))[T.SAD]:C(Drug, Treatment(reference='Placebo'))[T.Oxytocin]"
        if term in mdf.pvalues:
            p_val = mdf.pvalues[term]
            print(f"  >>> Interaction P-Value: {p_val:.5f} {'*' if p_val < 0.05 else ''}")
            return p_val
        else:
            print("  ! Interaction term not found in model results.")
            return 1.0
            
    except Exception as e:
        print(f"  ! Model Convergence Failed: {e}")
        return 1.0


def calc_drift_metrics(X_start_phase, y_start_phase, X_tgt_phase, y_tgt_phase, 
                       cond_start, cond_target, mask, sub_id):
    # Mask & Center (Phase-wise centering)
    X_s = X_start_phase[:, mask]
    
    X_t = X_tgt_phase[:, mask]
    
    # Target Centroid
    mask_tgt = _condition_mask(y_tgt_phase, cond_target)
    if np.sum(mask_tgt) < 2: return None
    P_target = np.mean(X_t[mask_tgt], axis=0)
    
    # Trajectory
    mask_lrn = _condition_mask(y_start_phase, cond_start)
    idx_lrn = np.where(mask_lrn)[0]
    if len(idx_lrn) < 4: return None
    
    cutoff = len(idx_lrn) // 2
    P_start = np.mean(X_s[idx_lrn[:cutoff]], axis=0)
    P_end = np.mean(X_s[idx_lrn[cutoff:]], axis=0)
    
    # Vectors
    V_axis = P_target - P_start
    V_drift = P_end - P_start
    
    nA, nD = norm(V_axis), norm(V_drift)
    if nA == 0 or nD == 0: return None
    
    dot = np.dot(V_drift, V_axis)
    return {'Cosine': dot / (nA * nD), 'Projection': dot / nA}


def plot_interaction(ax, df, domain, metric, p_val):
    data = df[df["Domain"] == domain]
    if data.empty: return
    
    # Error bars = Standard Error (se)
    # This approximates within-subject error visualization for group means
    sns.pointplot(data=data, x='Drug', y=metric, hue='Group', 
                  palette=pal_group, order=['Placebo', 'Oxytocin'], hue_order=['SAD', 'HC'],
                  dodge=0.15, markers=['o', 's'], linestyles=['-', '--'], 
                  capsize=0.1, err_kws={'linewidth': 2.5}, scale=1.2, 
                  errorbar='se', ax=ax)
    
    ax.set_title(f"{domain} - {metric}")
    ax.axhline(0, color='gray', ls='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=12)
    
    if p_val < 0.05:
        ax.text(0.5, 0.9, f"Interaction p={p_val:.3f}", transform=ax.transAxes, 
                ha='center', fontweight='bold', color='black')


def calc_metrics_for_subject(X, y, sub_id, feature_mask, C_param=1.0):
    # 1. Mask & Center
    X_m = X[:, feature_mask]
    
    # 2. Filter Binary Classes
    mask_bin = np.isin(y, [COND_CLASS_THREAT, COND_CLASS_SAFE])
    X_bin, y_bin = X_m[mask_bin], y[mask_bin]
    
    if len(y_bin) < MIN_TRIALS_PER_SUBJECT: return None
    
    try:
        # 3. CV Probabilities
        model = build_binary_pipeline()
        model.set_params(classification__C=C_param)
        cv = get_cv(y_bin, np.full(len(y_bin), sub_id), n_splits=SUBJECT_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        calib_model = CalibratedClassifierCV(
            model,
            method="sigmoid",
            cv=3
        )
        probs_all = cross_val_predict(calib_model, X_bin, y_bin, groups=np.full(len(y_bin), sub_id), cv=cv, method='predict_proba', n_jobs=N_JOBS)
        
        # 4. Extract Safety Cue Probabilities
        classes = sorted(np.unique(y_bin))
        if COND_CLASS_THREAT not in classes: return None
        idx_threat = classes.index(COND_CLASS_THREAT)
        
        mask_css = (y_bin == COND_CLASS_SAFE)
        mask_csr = (y_bin == COND_CLASS_THREAT)
        if np.sum(mask_css) == 0: return None
        
        # Prob(Threat | Safety Cue)
        probs_css = probs_all[mask_css, idx_threat]
        probs_csr = probs_all[mask_csr, idx_threat]
        p_proto_csr_css, p_proto_csr_csr, proto_boundary_separation = prototype_decision_metrics(
            X_m, y, COND_CLASS_THREAT, COND_CLASS_SAFE
        )
        p_csr_css = float(np.mean(probs_css))
        p_csr_csr = float(np.mean(probs_csr))
        boundary_separation = p_csr_csr - p_csr_css
        
        # --- Metrics ---
        # A. Entropy (Uncertainty)
        p_clean = np.clip(probs_css, 1e-9, 1-1e-9)
        ents = [entropy([p, 1-p], base=2) for p in p_clean]
        val_ent = np.mean(ents)
        p_csr_clean = np.clip(probs_csr, 1e-9, 1-1e-9)
        ents_csr = [entropy([p, 1-p], base=2) for p in p_csr_clean]
        val_ent_csr = np.mean(ents_csr) if len(ents_csr) else np.nan
        
        # B. Kurtosis (Sharpness) - Fisher's (Normal=0)
        val_kurt = kurtosis(probs_css, fisher=True)
        
        # C. Variance (Spread)
        val_var = np.var(probs_css)

        # D. Prototype evidence measures used by representative-index workflow exports
        decision_margin_css = float(0.5 - p_csr_css) if np.isfinite(p_csr_css) else np.nan
        decision_margin_csr = float(p_csr_csr - 0.5) if np.isfinite(p_csr_csr) else np.nan
        decision_margin_all = float(np.mean(np.abs(probs_all[:, idx_threat] - 0.5)))
        
        return {
            'Entropy': val_ent,
            'Kurtosis': val_kurt,
            'Variance': val_var,
            'P_CSR_CSS': p_csr_css,
            'P_CSR_CSR': p_csr_csr,
            'Boundary_Separation': boundary_separation,
            'Prototype_P_CSR_CSS': p_proto_csr_css,
            'Prototype_P_CSR_CSR': p_proto_csr_csr,
            'Prototype_Boundary_Separation': proto_boundary_separation,
            'Decision_Margin_CSS': decision_margin_css,
            'Decision_Margin_CSR': decision_margin_csr,
            'Decision_Margin_All': decision_margin_all,
            'Entropy_CSS': float(val_ent),
            'Entropy_CSR': float(val_ent_csr) if np.isfinite(val_ent_csr) else np.nan,
        }
        
    except Exception:
        return None


def plot_metric(ax, metric, p_val):
    sns.pointplot(data=df_metrics, x='Drug', y=metric, hue='Group', 
                  palette=pal_group, order=['Placebo', 'Oxytocin'], hue_order=['SAD', 'HC'],
                  dodge=0.2, markers=['o', 's'], linestyles=['-', '--'], 
                  capsize=0.1, errorbar='se', scale=1.1, ax=ax)
    
    ax.set_title(f"{metric}")
    ax.set_ylabel(metric)
    if metric == "Entropy": ax.set_ylabel("Entropy (Uncertainty)")
    if metric == "Kurtosis": ax.set_ylabel("Kurtosis (Sharpness)")
    
    # Annotate Significance
    if p_val < 0.05:
        ax.text(0.5, 0.9, f"Interaction\np={p_val:.3f}", transform=ax.transAxes, 
                ha='center', fontweight='bold', color='black')

def plot_topology(rdms_sad, rdms_hc, vec_a_sad, vec_a_hc, vec_b_sad, vec_b_hc,
                  p_a, p_b, p_a_sad_0, p_a_hc_0, p_b_sad_0, p_b_hc_0,
                  title_suffix):
    sns.set_context("poster")
    fig = plt.figure(figsize=(24, 8))
    gs = fig.add_gridspec(1, 3)

    # Heatmaps
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(
        np.mean(rdms_sad, axis=0),
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=0,
        vmax=1.2,
        xticklabels=RDM_CONDITIONS,
        yticklabels=RDM_CONDITIONS,
        ax=ax1,
        cbar=False,
    )
    ax1.set_title(f"SAD Topology ({title_suffix})\n(n={len(rdms_sad)})")

    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(
        np.mean(rdms_hc, axis=0),
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=0,
        vmax=1.2,
        xticklabels=RDM_CONDITIONS,
        yticklabels=RDM_CONDITIONS,
        ax=ax2,
    )
    ax2.set_title(f"HC Topology ({title_suffix})\n(n={len(rdms_hc)})")

    # Violins
    ax3 = fig.add_subplot(gs[0, 2])
    df_res = pd.DataFrame({
        'Group': ['SAD'] * len(vec_a_sad) + ['HC'] * len(vec_a_hc) + ['SAD'] * len(vec_b_sad) + ['HC'] * len(vec_b_hc),
        'Distance': np.concatenate([vec_a_sad, vec_a_hc, vec_b_sad, vec_b_hc]),
        'Metric': ['A: Threat Dist'] * len(vec_a_sad) + ['A: Threat Dist'] * len(vec_a_hc) +
                  ['B: Safety Dist'] * len(vec_b_sad) + ['B: Safety Dist'] * len(vec_b_hc),
    })
    sns.violinplot(
        data=df_res,
        x='Metric',
        y='Distance',
        hue='Group',
        split=True,
        inner='quartile',
        palette={'SAD': '#c44e52', 'HC': '#4c72b0'},
        ax=ax3,
    )
    ax3.set_title(f"Topological Metrics (Centroid | {title_suffix})")
    ax3.set_ylabel("Crossnobis Distance")

    # Annotate Group Differences
    y_max = df_res['Distance'].max()
    if p_a < 0.05:
        ax3.text(0, y_max + 0.05, f'* (p={p_a:.3f})', ha='center', fontsize=18)
    if p_b < 0.05:
        ax3.text(1, y_max + 0.05, f'* (p={p_b:.3f})', ha='center', fontsize=18)


    # For Metric A
    ax3.text(-0.2, -0.15, f"SAD: {get_sig_star(p_a_sad_0)}", transform=ax3.get_xaxis_transform(),
             ha='center', fontsize=14, color='#c44e52')
    ax3.text(0.2, -0.15, f"HC: {get_sig_star(p_a_hc_0)}", transform=ax3.get_xaxis_transform(),
             ha='center', fontsize=14, color='#4c72b0')

    # For Metric B
    ax3.text(0.8, -0.15, f"SAD: {get_sig_star(p_b_sad_0)}", transform=ax3.get_xaxis_transform(),
             ha='center', fontsize=14, color='#c44e52')
    ax3.text(1.2, -0.15, f"HC: {get_sig_star(p_b_hc_0)}", transform=ax3.get_xaxis_transform(),
             ha='center', fontsize=14, color='#4c72b0')

    plt.tight_layout()
    plt.show()


def compute_perm_importance_cv_with_p(
    model_template,
    X,
    y,
    groups,
    n_repeats=10,
    n_splits=5,
):
    """Return (mean_importance, p_one_sided_le0) from CV permutation importance."""
    from sklearn.inspection import permutation_importance

    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)

    cv = get_cv(y, groups, n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    all_imps = []  # list of (n_features, n_repeats)

    for train_idx, test_idx in cv.split(X, y, groups=groups):
        model = clone(model_template)
        model.fit(X[train_idx], y[train_idx])
        result = permutation_importance(
            model,
            X[test_idx],
            y[test_idx],
            n_repeats=n_repeats,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            scoring=forced_choice_scorer,
        )
        all_imps.append(result.importances)

    imps = np.concatenate(all_imps, axis=1)  # (n_features, n_folds*n_repeats)
    mean_imp = imps.mean(axis=1)
    p_le0 = (imps <= 0).mean(axis=1)
    return mean_imp, p_le0


def run_roi_specific_analysis(X, y, subjects, pipeline_template, best_C, n_permutations, roi_names, roi_dir):
    print(f"  Estimating Weights ({n_permutations} perms, ROI-Specific FDR)...")
    pipe = clone(pipeline_template); pipe.set_params(classification__C=best_C)
    
    # 1. Stable Observed Weights
    obs_weights = get_robust_weights(X, y, subjects, pipe, n_boot=100)
    
    # 2. Permutation Loop
    def run_null(i):
        y_shuff = shuffle(y, random_state=i)
        return get_robust_weights(X, y_shuff, subjects, pipe, n_boot=N_BOOT)

    null_weights = np.array(Parallel(n_jobs=N_JOBS)(delayed(run_null)(i) for i in range(n_permutations)))
    
    # 3. Raw P-values
    p_values_all = (np.sum(np.abs(null_weights) >= np.abs(obs_weights), axis=0) + 1) / (n_permutations + 1)
    
    # 4. ROI-Specific FDR & Mapping
    sig_mask = np.zeros_like(p_values_all, dtype=bool)
    roi_stats = {}
    curr = 0
    
    for name in roi_names:
        fpaths = glob.glob(os.path.join(roi_dir, f"*{name}*.nii*"))
        n_vox = int(np.sum(nib.load(fpaths[0]).get_fdata() > 0))
        
        roi_p = p_values_all[curr : curr + n_vox]
        if len(roi_p) > 0:
            reject, _, _, _ = multipletests(roi_p, alpha=fdr_alpha, method='fdr_bh')
            sig_mask[curr : curr + n_vox] = reject
            
            n_sig = np.sum(reject)
            if n_sig > 0:
                roi_stats[name] = {'count': n_sig, 'perc': (n_sig / n_vox) * 100}
        curr += n_vox

    # 5. Z-Score Calculation
    z_scores = (obs_weights - np.mean(null_weights, axis=0)) / (np.std(null_weights, axis=0) + 1e-12)
    return z_scores, sig_mask, roi_stats

import joblib
import os
from sklearn.utils import shuffle

def save_permutation_results(obs_weights, null_weights, p_values, filename):
    """Saves raw permutation results to disk using compression."""
    data = {
        'obs_weights': obs_weights,
        'null_weights': null_weights,
        'p_values_raw': p_values
    }
    joblib.dump(data, filename, compress=3)
    print(f"  [CACHE] Saved raw results to: {filename}")

def run_raw_permutations(X, y, subjects, pipeline_template, best_C, n_permutations):
    """Executes the heavy parallel math for Haufe weights and permutations."""
    pipe = clone(pipeline_template); pipe.set_params(classification__C=best_C)
    
    # 1. Stable Observed Weights
    print(f"  > Calculating robust observed weights (n_boot=100)...")
    obs_weights = get_robust_weights(X, y, subjects, pipe, n_boot=100)
    
    # 2. Parallel Permutations
    print(f"  > Running {n_permutations} permutations on {N_JOBS} cores...")
    def run_null(i):
        y_shuff = shuffle(y, random_state=i)
        return get_robust_weights(X, y_shuff, subjects, pipe, n_boot=1)

    null_weights_list = Parallel(n_jobs=N_JOBS)(
        delayed(run_null)(i) for i in range(n_permutations)
    )
    null_weights = np.array(null_weights_list)
    
    # 3. Calculate Raw P-values (Before FDR)
    p_values_raw = (np.sum(np.abs(null_weights) >= np.abs(obs_weights), axis=0) + 1) / (n_permutations + 1)
    
    return obs_weights, null_weights, p_values_raw


def apply_voxel_correction(obs_weights, null_weights, p_values_raw, 
                           method='roi', roi_names=None, roi_dir=None, top_n=50):
    """
    Applies correction or rank-based selection of top N voxels.
    method: 'roi' (FDR) or 'top_n' (Rank-based)
    top_n: Integer count of voxels to keep in fallback mode.
    """
    z_scores = (obs_weights - np.mean(null_weights, axis=0)) / (np.std(null_weights, axis=0) + 1e-12)
    sig_mask = np.zeros_like(p_values_raw, dtype=bool)
    roi_stats = {}
    
    if method == 'roi':
        # Standard FDR logic
        curr = 0
        for name in roi_names:
            fpaths = glob.glob(os.path.join(roi_dir, f"*{name}*.nii*"))
            n_vox = int(np.sum(nib.load(fpaths[0]).get_fdata() > 0))
            roi_p = p_values_raw[curr : curr + n_vox]
            if len(roi_p) > 0:
                reject, _, _, _ = multipletests(roi_p, alpha=fdr_alpha, method='fdr_bh')
                sig_mask[curr : curr + n_vox] = reject
            curr += n_vox
    else:
        # NEW Top-N Logic: Find indices of N largest absolute Z-scores
        abs_z = np.abs(z_scores)
        # Handle case where top_n might be larger than total voxels
        n_to_select = min(top_n, len(abs_z))
        # Get indices of top N values
        top_indices = np.argsort(abs_z)[-n_to_select:]
        sig_mask[top_indices] = True

    # Calculate stats for the chosen mask
    curr = 0
    for name in roi_names:
        fpaths = glob.glob(os.path.join(roi_dir, f"*{name}*.nii*"))
        n_vox = int(np.sum(nib.load(fpaths[0]).get_fdata() > 0))
        roi_mask = sig_mask[curr : curr + n_vox]
        
        n_sig = np.sum(roi_mask)
        if n_sig > 0:
            roi_stats[name] = {'count': n_sig, 'perc': (n_sig / n_vox) * 100}
        curr += n_vox

    return z_scores, sig_mask, roi_stats


def apply_flexible_voxel_correction(obs_weights, null_weights, p_values_raw, 
                                   method='roi', roi_names=None, roi_dir=None, 
                                   match_count=None, fallback_percentile=98):
    """
    Applies correction with fallback logic.
    match_count: If provided, ignores p-values and picks exactly this many top voxels.
    """
    z_scores = (obs_weights - np.mean(null_weights, axis=0)) / (np.std(null_weights, axis=0) + 1e-12)
    sig_mask = np.zeros_like(p_values_raw, dtype=bool)
    roi_stats = {}
    
    if match_count is not None:
        # Pick exactly match_count voxels based on absolute Z magnitude
        abs_z = np.abs(z_scores)
        top_idx = np.argsort(abs_z)[-int(match_count):]
        sig_mask[top_idx] = True
        mode = f"Top {match_count} (Matched)"
    elif method == 'roi' and roi_names:
        # Standard ROI-specific FDR
        curr = 0
        for name in roi_names:
            fpaths = glob.glob(os.path.join(roi_dir, f"*{name}*.nii*"))
            n_vox = int(np.sum(nib.load(fpaths[0]).get_fdata() > 0))
            roi_p = p_values_raw[curr : curr + n_vox]
            if len(roi_p) > 0:
                reject, _, _, _ = multipletests(roi_p, alpha=fdr_alpha, method='fdr_bh')
                sig_mask[curr : curr + n_vox] = reject
            curr += n_vox
        mode = "ROI-FDR"
    else:
        # Standard Percentile Fallback
        abs_z = np.abs(z_scores)
        thresh = np.percentile(abs_z, fallback_percentile)
        sig_mask = abs_z >= thresh
        mode = f"Top {100-fallback_percentile}%"

    # Calculate ROI Stats for the final mask
    curr = 0
    for name in roi_names:
        fpaths = glob.glob(os.path.join(roi_dir, f"*{name}*.nii*"))
        n_vox = int(np.sum(nib.load(fpaths[0]).get_fdata() > 0))
        n_sig = np.sum(sig_mask[curr : curr + n_vox])
        if n_sig > 0:
            roi_stats[name] = {'count': n_sig, 'perc': (n_sig / n_vox) * 100}
        curr += n_vox

    return z_scores, sig_mask, roi_stats, mode


def _ckpt_path(cell_id: int) -> str:
    """Constructs the full path for checkpoint files."""
    ckpt_dir = CHECKPOINT_DIR
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
    return os.path.join(ckpt_dir, f"cell_{cell_id:02d}.joblib")
    

def save_checkpoint(cell_id: int, data: dict) -> None:
    """Save checkpoint data for a given analysis cell."""
    path = _ckpt_path(cell_id)
    try:
        joblib.dump(data, path)
        print(f"[Checkpoint] Saved cell {cell_id} -> {path}")
    except Exception as exc:
        print(f"[Checkpoint] Failed to save cell {cell_id}: {exc}")

def save_intermediate(name: str, obj) -> None:
    """Save intermediate objects for downstream stages."""
    path = _intermediate_path(name)
    try:
        dump(obj, path)
        print(f"[Intermediate] Saved {name} -> {path}")
    except Exception as exc:
        print(f"[Intermediate] Failed to save {name}: {exc}")


def _intermediate_path(name: str) -> str:
    """Constructs the full path for intermediate files."""
    # Ensure the intermediate directory exists
    inter_dir = os.path.join(OUT_DIR_MAIN, "intermediate")
    if not os.path.exists(inter_dir):
        os.makedirs(inter_dir, exist_ok=True)
    return os.path.join(inter_dir, f"{name}.joblib")


def load_intermediate(name: str):
    """Load intermediate objects saved in previous stages."""
    path = _intermediate_path(name)
    if os.path.exists(path):
        return load(path)
    else:
        raise FileNotFoundError(f"No intermediate file found at {path}")


def save_stage_bundle(stage_id: int, name: str, payload: dict) -> None:
    """Save a named stage bundle for the post-Hyak notebook exporter."""
    bundle = dict(payload)
    bundle.setdefault("stage_id", stage_id)
    bundle.setdefault("analysis_feature_space", globals().get("ANALYSIS_LABEL", "FearNetwork"))
    bundle.setdefault("notebook_required_subject_columns", NOTEBOOK_REQUIRED_SUBJECT_COLUMNS)
    save_intermediate(name, bundle)


def keep_existing_columns(df, columns, label):
    """Keep requested columns that exist in df and report optional omissions."""
    existing = [col for col in columns if col in df.columns]
    missing = [col for col in columns if col not in df.columns]
    if missing:
        print(f"[Column filter] Skipping missing {label}: {missing}")
    return existing


def calculate_crossnobis_rdm(
    X,
    y,
    subjects,
    conditions,
    n_repeats=CROSSNOBIS_REPEATS,
    random_state=RANDOM_STATE,
    standardize=False,
):
    # Crossnobis RDM per subject with Ledoit-Wolf shrinkage, averaged over repeats.
    unique_subs = np.unique(subjects)
    rdms = []
    sub_ids = []
    rng = np.random.default_rng(random_state)

    for sub in unique_subs:
        mask_sub = (subjects == sub)
        X_sub = X[mask_sub]
        y_sub = y[mask_sub]

        if standardize:
            mean = np.mean(X_sub, axis=0)
            std = np.std(X_sub, axis=0)
            std = np.where(std == 0, 1.0, std)
            X_sub = (X_sub - mean) / std

        rdm_accum = None
        valid_reps = 0

        for rep in range(n_repeats):
            # Build split-half means per condition
            means_a = {}
            means_b = {}
            ok = True
            for cond in conditions:
                idx = np.where(y_sub == cond)[0]
                if len(idx) < 2:
                    ok = False
                    break
                idx = idx.copy()
                rng.shuffle(idx)
                half = len(idx) // 2
                idx_a = idx[:half]
                idx_b = idx[half:]
                if len(idx_a) == 0 or len(idx_b) == 0:
                    ok = False
                    break
                means_a[cond] = np.mean(X_sub[idx_a], axis=0)
                means_b[cond] = np.mean(X_sub[idx_b], axis=0)
            if not ok:
                continue

            # Estimate noise covariance from residuals (all trials, condition-demeaned)
            resid = []
            for cond in conditions:
                idx = np.where(y_sub == cond)[0]
                cond_mean = np.mean(X_sub[idx], axis=0)
                resid.append(X_sub[idx] - cond_mean)
            resid = np.vstack(resid)
            cov = LedoitWolf().fit(resid).covariance_
            prec = np.linalg.pinv(cov)

            # Crossnobis distance matrix
            n = len(conditions)
            rdm = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    c_i = conditions[i]
                    c_j = conditions[j]
                    d_a = means_a[c_i] - means_a[c_j]
                    d_b = means_b[c_i] - means_b[c_j]
                    dist = float(d_a.T @ prec @ d_b)
                    rdm[i, j] = dist
                    rdm[j, i] = dist

            if rdm_accum is None:
                rdm_accum = rdm
            else:
                rdm_accum += rdm
            valid_reps += 1

        if valid_reps == 0:
            continue
        rdm_mean = rdm_accum / valid_reps
        rdms.append(rdm_mean)
        sub_ids.append(sub)

    return np.array(rdms), np.array(sub_ids)
save_cell_results(2, ['data_subsets', 'importance_mask_permutated', 'importance_scores_permutated', 'meta', 'param_grid', 'results_11', 'results_12', 'results_13', 'results_13_2', 'results_14_self', 'results_21', 'results_21_pv', 'results_22', 'results_23', 'results_24', 'results_25', 'strict_cross_phase_results', 'sub_to_meta'])


# %% [cell 3]
# Cell 3: Load phase2 (extinction) and phase3 (reinstatement) data
# Update: Filters FEATURES to include only specific ROIs (Amygdala, Hippocampus, Insula, vmPFC, ACC).

import numpy as np
import os

print("--- Cell 3: Data Loading & ROI Filtering ---")

project_root = PROJECT_ROOT
data_root = os.path.join(project_root, "MRI/derivatives/fMRI_analysis/LSS", "firstLevel/all_subjects/group_level")
scripts_dir = os.path.dirname(os.path.abspath(__file__))
# OUT_DIR_MAIN, CHECKPOINT_DIR, and INTERMEDIATE_DIR are resolved by the Hyak runtime block.
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

print(f"Paths re-defined successfully!")
print(f"Checkpoints will be saved to: {CHECKPOINT_DIR}")

print("--- Data Loading (Whole-Brain Parcellation) ---")


phase2_npz_path = os.path.join(data_root, "phase2_X_ext_y_ext_roi_voxels.npz")
phase3_npz_path = os.path.join(data_root, "phase3_X_reinst_y_reinst_roi_voxels.npz") # Note: using 'reinst' variable name

# Define the specific ROIs to keep
TARGET_ROIS = [
    'left_acc', 'left_amygdala', 'left_hippocampus', 'left_insula', 'left_vmpfc',
    'right_acc', 'right_amygdala', 'right_hippocampus', 'right_insula', 'right_vmpfc'
]

# Load Files
phase2_npz = np.load(phase2_npz_path, allow_pickle=True)
phase3_npz = np.load(phase3_npz_path, allow_pickle=True)

# ---- Helper: ROI Feature Selection ----
def filter_features_by_roi(X, roi_names, roi_counts, target_list):
    """
    Creates a boolean mask for voxels belonging to target ROIs and filters X.
    Returns: Filtered X, Filtered Parcel Names
    """
    feature_mask = []
    new_parcel_names = []
    
    # Iterate through each ROI metadata entry
    for name, count in zip(roi_names, roi_counts):
        # Create labels for this ROI (e.g., "left_amygdala_0")
        current_labels = [f"{name}_{i}" for i in range(count)]
        
        if name in target_list:
            # Keep these voxels
            feature_mask.extend([True] * count)
            new_parcel_names.extend(current_labels)
        else:
            # Drop these voxels
            feature_mask.extend([False] * count)
            
    feature_mask = np.array(feature_mask)
    
    # Apply mask to columns (features)
    if X.shape[1] != len(feature_mask):
        raise ValueError(f"Shape mismatch: X has {X.shape[1]} features, but ROI counts imply {len(feature_mask)}.")
        
    X_filtered = X[:, feature_mask]
    
    return X_filtered, new_parcel_names

# ---- Process Phase 2 (Extinction) ----
X_ext_raw = phase2_npz["X_ext"]
y_ext = phase2_npz["y_ext"]
sub_ext = phase2_npz["subjects"]
roi_names_ext = phase2_npz["roi_names"]
roi_counts_ext = phase2_npz["roi_voxel_counts"]

print(f"Original Extinction Shape: {X_ext_raw.shape}")

# Apply ROI Filter
X_ext, parcel_names_ext = filter_features_by_roi(X_ext_raw, roi_names_ext, roi_counts_ext, TARGET_ROIS)
print(f"Filtered Extinction Shape: {X_ext.shape} (kept {len(TARGET_ROIS)} ROIs)")
X_ext_all = X_ext.copy()
y_ext_all = y_ext.copy()
sub_ext_all = sub_ext.copy()


# ---- Process Phase 3 (Reinstatement) ----
X_reinst_raw = phase3_npz["X_reinst"]
y_reinst = phase3_npz["y_reinst"]
sub_reinst = phase3_npz["subjects"]
roi_names_reinst = phase3_npz["roi_names"]
roi_counts_reinst = phase3_npz["roi_voxel_counts"]

# Apply ROI Filter
X_reinst, parcel_names_reinst = filter_features_by_roi(X_reinst_raw, roi_names_reinst, roi_counts_reinst, TARGET_ROIS)
print(f"Filtered Reinstatement Shape: {X_reinst.shape}")
X_reinst_all = X_reinst.copy()
y_reinst_all = y_reinst.copy()
sub_reinst_all = sub_reinst.copy()


# ---- Filter for CS Trials Only ----
# Constants (Define if not present)
if 'CS_LABELS' not in locals(): CS_LABELS = ["CS-", "CSS", "CSR"]

# Keep only CS-, CSS, CSR trials
mask_ext = np.isin(y_ext, CS_LABELS)
mask_reinst = np.isin(y_reinst, CS_LABELS)

X_ext = X_ext[mask_ext]
y_ext = y_ext[mask_ext]
sub_ext = sub_ext[mask_ext]

X_reinst = X_reinst[mask_reinst]
y_reinst = y_reinst[mask_reinst]
sub_reinst = sub_reinst[mask_reinst]

print("\nAfter CS filtering:")
print("Phase2 (Ext):", X_ext.shape, np.unique(y_ext, return_counts=True))
print("Phase3 (Reinst):", X_reinst.shape, np.unique(y_reinst, return_counts=True))
print(f"Target ROIs included: {TARGET_ROIS}")
save_cell_results(3, ['CHECKPOINT_DIR', 'INTERMEDIATE_DIR', 'OUT_DIR_MAIN', 'TARGET_ROIS', 'X_ext', 'X_ext_all', 'X_ext_raw', 'X_reinst', 'X_reinst_all', 'X_reinst_raw', 'data_root', 'data_subsets', 'importance_mask_permutated', 'importance_scores_permutated', 'mask_ext', 'mask_reinst', 'meta', 'parcel_names_ext', 'parcel_names_reinst', 'phase2_npz', 'phase2_npz_path', 'phase3_npz', 'phase3_npz_path', 'project_root', 'results_11', 'results_12', 'results_13', 'results_13_2', 'results_14_self', 'results_21', 'results_21_pv', 'results_22', 'results_23', 'results_24', 'results_25', 'roi_counts_ext', 'roi_counts_reinst', 'roi_names_ext', 'roi_names_reinst', 'scripts_dir', 'strict_cross_phase_results', 'sub_ext', 'sub_ext_all', 'sub_reinst', 'sub_reinst_all', 'sub_to_meta', 'y_ext', 'y_ext_all', 'y_reinst', 'y_reinst_all'])


# %% [cell 4]
# Cell 4: Load subject-level metadata (Group, Drug, etc.)

# Example: a CSV with one row per subject, columns like:
#   subject_id, Group, Drug, Age, Sex, ...
# where Group ∈ {SAD, HC}, Drug ∈ {OT, PLC} or similar
meta_path = os.path.join(project_root, "MRI/source_data/behav/drug_order.csv")

meta = pd.read_csv(meta_path)

print(meta.head())
print(meta.columns)

# Basic sanity check: make sure subjects in X_ext/X_reinst exist in metadata
unique_subs_ext = np.unique(sub_ext)
unique_subs_reinst = np.unique(sub_reinst)

print("Phase2 unique subjects:", len(unique_subs_ext))
print("Phase3 unique subjects:", len(unique_subs_reinst))

missing_in_meta_ext = [s for s in unique_subs_ext if s not in set(meta["subject_id"])]
missing_in_meta_reinst = [s for s in unique_subs_reinst if s not in set(meta["subject_id"])]

print("Missing in meta (phase2):", missing_in_meta_ext)
print("Missing in meta (phase3):", missing_in_meta_reinst)
save_cell_results(4, ['data_subsets', 'importance_mask_permutated', 'importance_scores_permutated', 'meta', 'meta_path', 'missing_in_meta_ext', 'missing_in_meta_reinst', 'results_11', 'results_12', 'results_13', 'results_13_2', 'results_14_self', 'results_21', 'results_21_pv', 'results_22', 'results_23', 'results_24', 'results_25', 'strict_cross_phase_results', 'sub_to_meta', 'unique_subs_ext', 'unique_subs_reinst'])


# %% [cell 5]
# Cell 5: Data Preparation & Subsetting (Optimized: Center -> Filter)
# Task: 1. Split data by subject.
#       2. Center FULL subject data (to preserve true baseline).
#       3. Filter for CSS/CSR conditions.
#       4. Organize into Groups (SAD/HC).

print("--- Cell 5: Data Preparation & Subsetting (Center -> Filter) ---")

import numpy as np
import pandas as pd

# =============================================================================
# 0. Helper: Group Assignment Logic
# =============================================================================
if 'meta' in locals():
    # Standardize IDs
    meta['subject_id'] = meta['subject_id'].astype(str).str.strip()
    sub_to_meta = meta.set_index("subject_id")[["Group", "Drug"]].to_dict('index')
    print(f"Metadata loaded for {len(sub_to_meta)} subjects.")
else:
    raise ValueError("Metadata 'meta' not found. Please run Cell 3.")

group_keys = ["SAD_Placebo", "SAD_Oxytocin", "HC_Placebo", "HC_Oxytocin"]

if 'X_ext' not in locals(): raise ValueError("X_ext missing. Run Cell 2.")
    
# Handle Reinstatement variable naming
if 'X_rst' in locals():
    X_rein, y_rein, sub_rein = X_rst, y_rst, sub_rst
elif 'X_reinst' in locals():
    X_rein, y_rein, sub_rein = X_reinst, y_reinst, sub_reinst
else:
    print("  ! Reinstatement data missing.")
    X_rein, y_rein, sub_rein = None, None, None

# =============================================================================
# 3. Execute
# =============================================================================
ext_subsets = process_phase_data(X_ext, y_ext, sub_ext, "Extinction")
rst_subsets = process_phase_data(X_rein, y_rein, sub_rein, "Reinstatement")

# Structure Results
data_subsets = {}
for key in group_keys:
    data_subsets[key] = {
        "ext": ext_subsets.get(key),
        "rst": rst_subsets.get(key)
    }

print("\nCell 5 Complete. Data is Centered (Full-Session) and Filtered.")
save_cell_results(5, ['data_subsets', 'ext_subsets', 'group_keys', 'importance_mask_permutated', 'importance_scores_permutated', 'key', 'meta', 'results_11', 'results_12', 'results_13', 'results_13_2', 'results_14_self', 'results_21', 'results_21_pv', 'results_22', 'results_23', 'results_24', 'results_25', 'rst_subsets', 'strict_cross_phase_results', 'sub_to_meta'])


# %% [cell 6]
if cell_active(6):
    # Keep the runtime/SLURM-configured N_JOBS value from the Hyak argument parser.
    # Cell 6: Analysis 1.1 - Neural Dissociation Execution
    # Protocol: SAD -> HC
    # Updates:
    #   - Uses 'run_pairwise_decoding_analysis' (Forced-Choice Accuracy).
    #   - Functional Specificity Heatmap uses Mean CV Accuracy for diagonals (Evaluation).
    #   - Cross-Decoding uses the 'final_model' (Refitted on full data) on the other group.
    #   - Hyperparameter selected from 20 values (0.01-100) using training data only.
    #   - 5-fold CV, repeated 10 times with different splits; mean performance reported.
    #   - Forced-choice accuracy used to mitigate inter-site activation differences.

    print("--- Running Analysis 1.1: Neural Dissociation ---")

    target_param = 'classification__C'  # Variable for the hyperparameter key
    stage6_filter_label = make_safe_label(ANALYSIS_LABEL)
    if INCLUDE_SUBJECTS_CSV and not stage6_filter_label:
        stage6_filter_label = make_safe_label(INCLUDE_SUBJECTS_FLAG or "subject_filter")
    cache_cell6 = _ckpt_path(6)
    if stage6_filter_label:
        cache_cell6 = os.path.join(CHECKPOINT_DIR, f"cell_06_{stage6_filter_label}.joblib")
        print(f"  [SENSITIVITY] Stage 6 will use labeled checkpoint: {cache_cell6}")

    # =============================================================================
    # CHECK CACHE
    # =============================================================================
    if os.path.exists(cache_cell6):
        print(f"  [LOAD] Found cached results in {cache_cell6}. Skipping computation.")
        _cell6_payload = joblib.load(cache_cell6)
        results_11 = _cell6_payload.get("results_11", _cell6_payload) if isinstance(_cell6_payload, dict) else _cell6_payload
    
        # RECONSTRUCT res_sad and res_hc so the plotting code can find them
        res_sad = {
            'accuracy': results_11["acc_sad_cv"],
            'haufe_pattern': results_11["map_sad"],
            'model_sad': results_11['model_sad'],
            'model': results_11['model_sad'],
            'best_C': results_11.get('best_c_sad', 1.0)
        }
        res_hc = {
            'accuracy': results_11["acc_hc_cv"],
            'haufe_pattern': results_11["map_hc"],
            'model_hc': results_11['model_hc'],
            'model': results_11['model_hc'],
            'best_C': results_11.get('best_c_hc', 1.0)
        }
    
        # Unpack remaining variables for the visualization block
        perm_acc_sad = results_11["perm_dist_sad"]
        perm_acc_hc = results_11["perm_dist_hc"]
        func_matrix = results_11["func_matrix"]
        func_pvals = results_11["p_func_pvals"]
        obs_sim = results_11["sim_spatial"]
        p_sim_spatial = results_11["p_sim"]
    
        # These were assigned in your error slip but need to match the plot calls
        p_sad = results_11["p_sad"]
        p_hc = results_11["p_hc"]
        p_sad2hc = func_pvals[0, 1] 
        p_hc2sad = func_pvals[1, 0]
        mean_sad2hc = func_matrix[0, 1]
        mean_hc2sad = func_matrix[1, 0]
    else:
        # =============================================================================
        # 0. Data Slicing
        # =============================================================================
        # Load Data
        try:
            def load_stage6_group_data(group):
                group_keys = [f"{group}_Placebo"]
                if INCLUDE_SUBJECTS_FLAG:
                    group_keys.append(f"{group}_Oxytocin")
                arrays = [get_extinction_data(group_key) for group_key in group_keys]
                X = np.concatenate([item[0] for item in arrays], axis=0)
                y = np.concatenate([item[1] for item in arrays], axis=0)
                sub = np.concatenate([item[2] for item in arrays], axis=0)
                return X, y, sub, group_keys

            X_hc, y_hc, sub_hc, hc_keys = load_stage6_group_data("HC")
            X_sad, y_sad, sub_sad, sad_keys = load_stage6_group_data("SAD")
            session_scope = "pooled_drug" if INCLUDE_SUBJECTS_FLAG else "placebo"
            print(f"Data Loaded ({session_scope}): SAD {sad_keys} (n={len(np.unique(sub_sad))}), HC {hc_keys} (n={len(np.unique(sub_hc))})")
            include_subjects = load_stage6_include_subjects()
            if include_subjects is not None:
                X_hc, y_hc, sub_hc = apply_stage6_subject_filter(
                    X_hc, y_hc, sub_hc, include_subjects, f"{stage6_filter_label}:HC"
                )
                X_sad, y_sad, sub_sad = apply_stage6_subject_filter(
                    X_sad, y_sad, sub_sad, include_subjects, f"{stage6_filter_label}:SAD"
                )
                print(
                    f"Data After Filter: SAD (n={len(np.unique(sub_sad))}), "
                    f"HC (n={len(np.unique(sub_hc))})"
                )
        except ValueError as e:
            print(f"CRITICAL ERROR: {e}")
            raise

        # =============================================================================
        # TEST 1: Baseline Neural Discriminability (Self-Decoding)
        # =============================================================================
        print("\n--- TEST 1: Baseline Neural Discriminability ---")

        print("Processing SAD...")
        res_sad_dict = run_pairwise_decoding_analysis(X_sad, y_sad, sub_sad, n_repeats=N_REPEATS)
        best_c_sad = res_sad_dict[list(res_sad_dict.keys())[0]]['model'].get_params()[target_param]
        print(f"  > Best {target_param} for SAD: {best_c_sad}")

        print("Processing HC...")
        res_hc_dict = run_pairwise_decoding_analysis(X_hc, y_hc, sub_hc, n_repeats=N_REPEATS)
        best_c_hc = res_hc_dict[list(res_hc_dict.keys())[0]]['model'].get_params()[target_param]
        print(f"  > Best {target_param} for HC: {best_c_hc}")

        # Select the target contrast (CSS vs CSR)
        pair_key = "CSR vs CSS" if "CSR vs CSS" in res_sad_dict else "CSS vs CSR"
        if pair_key not in res_sad_dict or pair_key not in res_hc_dict:
            raise ValueError(f"Contrast {pair_key} not found. Check if CSS/CSR labels exist.")

        res_sad = res_sad_dict[pair_key]
        res_hc = res_hc_dict[pair_key]

        # Permutation Test (Comparing Observed CV Score against Null CV Scores)
        print(f"Running Permutation Test (Self-Decoding, {N_PERMUTATION} iter)...")
        permutation_chunks = make_permutation_chunks(N_PERMUTATION, N_JOBS)
        perm_acc_sad = np.concatenate(Parallel(n_jobs=N_JOBS)(delayed(run_perm_simple)(X_sad, y_sad, sub_sad, n_iter) for n_iter in permutation_chunks))
        perm_acc_hc = np.concatenate(Parallel(n_jobs=N_JOBS)(delayed(run_perm_simple)(X_hc, y_hc, sub_hc, n_iter) for n_iter in permutation_chunks))

        # =============================================================================
        # TEST 2: Functional Specificity (Cross-Decoding)
        # =============================================================================
        print("\n--- TEST 2: Functional Specificity ---")
        # Logic: Use Final Refit Model (Trained on All A) -> Predict All B -> Avg Subject Accuracy

        # A. SAD Model -> HC Data
        model_sad = res_sad['model'] # This is the Refit model
        # 'run_cross_decoding' calculates raw accuracy per subject
        accs_sad2hc = run_cross_decoding(model_sad, X_hc, y_hc, sub_hc, model_sad.classes_)
        mean_sad2hc = np.mean(accs_sad2hc)
        print(f"  > SAD Model -> HC Data: {mean_sad2hc:.4f}")

        # B. HC Model -> SAD Data
        model_hc = res_hc['model']
        accs_hc2sad = run_cross_decoding(model_hc, X_sad, y_sad, sub_sad, model_hc.classes_)
        mean_hc2sad = np.mean(accs_hc2sad)
        print(f"  > HC Model -> SAD Data: {mean_hc2sad:.4f}")

        # Permutation Test (Cross-Decoding)
        print(f"Running Permutation Test (Cross-Decoding, {N_PERMUTATION} iter)...")
        perm_sad2hc = np.concatenate(Parallel(n_jobs=N_JOBS)(
            delayed(run_cross_perm)(model_sad, X_hc, y_hc, sub_hc, n_iter) for n_iter in permutation_chunks))
        p_sad2hc = np.mean(perm_sad2hc >= mean_sad2hc)

        perm_hc2sad = np.concatenate(Parallel(n_jobs=N_JOBS)(
            delayed(run_cross_perm)(model_hc, X_sad, y_sad, sub_sad, n_iter) for n_iter in permutation_chunks))
        p_hc2sad = np.mean(perm_hc2sad >= mean_hc2sad)

        functional_drop_pairs, functional_drop_tests, functional_drop_nulls = build_functional_drop_outputs(
            X_sad, y_sad, sub_sad,
            X_hc, y_hc, sub_hc,
            model_sad, model_hc,
            best_c_sad, best_c_hc,
            acc_sad_cv=res_sad['accuracy'],
            acc_hc_cv=res_hc['accuracy'],
            mean_sad2hc=mean_sad2hc,
            mean_hc2sad=mean_hc2sad,
            perm_acc_sad=perm_acc_sad,
            perm_acc_hc=perm_acc_hc,
            perm_sad2hc=perm_sad2hc,
            perm_hc2sad=perm_hc2sad,
        )
        for group_name, drop_test in functional_drop_tests.items():
            print(
                f"  > {group_name} self-vs-cross drop: "
                f"drop={drop_test['functional_drop']:.4f}, "
                f"p={drop_test['functional_drop_p']:.4f}, "
                f"n={drop_test['n_pairs']}"
            )

        # =============================================================================
        # TEST 3: Spatial Specificity
        # =============================================================================
        print("\n--- TEST 3: Spatial Specificity ---")
        map_sad, map_hc = res_sad['haufe_pattern'], res_hc['haufe_pattern']
        obs_sim = cosine_similarity(map_sad.reshape(1, -1), map_hc.reshape(1, -1))[0][0]

        # Prepare Data for Permutation (Combine groups)
        X_comb = np.concatenate([X_sad, X_hc])
        y_comb = np.concatenate([y_sad, y_hc])
        sub_comb = np.concatenate([sub_sad, sub_hc])

        all_sub_maps, all_sub_groups = [], []
        perm_pipe = build_binary_pipeline(); perm_pipe.set_params(classification__C=1.0)

        # Pre-compute subject maps
        print(f"Pre-computing {len(np.unique(sub_comb))} individual subject maps...")
        for sub in np.unique(sub_comb):
            mask = sub_comb == sub
            perm_pipe.fit(X_comb[mask], y_comb[mask])
            W = perm_pipe.named_steps['classification'].coef_
            # Calculate Covariance (Scaler handles centered input)
            cov = np.cov(X_comb[mask], rowvar=False)
            A = cov @ W.T
        
            if perm_pipe.classes_[1] == 'CSS': A = -A 
            all_sub_maps.append(A.flatten())
            all_sub_groups.append("SAD" if sub in sub_sad else "HC")

        # Run Spatial Permutation
        print(f"Running Spatial Permutation ({N_PERMUTATION} iter)...")
        perm_sims = np.array(Parallel(n_jobs=N_JOBS)(delayed(run_spatial_perm)(i, np.array(all_sub_maps), np.array(all_sub_groups)) for i in range(N_PERMUTATION)))

        p_sim_spatial = 2 * min(np.mean(perm_sims <= obs_sim), np.mean(perm_sims >= obs_sim))

        # Assemble Result Dictionary
        p_sad = float(np.mean(perm_acc_sad >= res_sad['accuracy']))
        p_hc = float(np.mean(perm_acc_hc >= res_hc['accuracy']))
        func_matrix = np.array([[res_sad['accuracy'], mean_sad2hc], [mean_hc2sad, res_hc['accuracy']]])
        func_pvals = np.array([[p_sad, p_sad2hc], [p_hc2sad, p_hc]])
    
        results_11 = {
            "acc_sad_cv": res_sad['accuracy'], 
            "cv_fold_scores_sad": res_sad.get('cv_fold_scores'),
            "p_sad": p_sad, 
            "acc_hc_cv": res_hc['accuracy'], 
            "cv_fold_scores_hc": res_hc.get('cv_fold_scores'),
            "p_hc": p_hc, 
            "func_matrix": func_matrix, 
            "p_func_pvals": func_pvals,
            "accs_sad2hc": accs_sad2hc,
            "accs_hc2sad": accs_hc2sad,
            "functional_drop_pairs": functional_drop_pairs,
            "functional_drop_tests": functional_drop_tests,
            "functional_drop_nulls": functional_drop_nulls,
            "sim_spatial": obs_sim, 
            "p_sim": p_sim_spatial,
            "spatial_perm_dist": perm_sims,
            "map_sad": map_sad, 
            "map_hc": map_hc,
            "perm_dist_sad": perm_acc_sad, 
            "perm_dist_hc": perm_acc_hc,
            "perm_dist_sad2hc": perm_sad2hc,
            "perm_dist_hc2sad": perm_hc2sad,
            "model_sad": res_sad['model'], # The refitted SAD pipeline
            "model_hc": res_hc['model'],    # The refitted HC pipeline
            "best_c_sad": best_c_sad,
            "best_c_hc": best_c_hc,
            "analysis_label": stage6_filter_label or "primary",
            "include_subjects_csv": INCLUDE_SUBJECTS_CSV,
            "include_subjects_flag": INCLUDE_SUBJECTS_FLAG,
            "include_subjects_column": INCLUDE_SUBJECTS_COLUMN,
            "session": session_scope,
            "n_sad_subjects": int(len(np.unique(sub_sad))),
            "n_hc_subjects": int(len(np.unique(sub_hc))),
            "n_sad_trials": int(len(sub_sad)),
            "n_hc_trials": int(len(sub_hc)),
        }
    
        # Save to Cache
        if stage6_filter_label:
            joblib.dump({"results_11": results_11}, cache_cell6)
            print(f"[Checkpoint] Saved labeled cell 6 -> {cache_cell6}")
        else:
            save_checkpoint(6, {
                "results_11": results_11
            })
        print(f"  [SAVE] Analysis complete. Results cached to {cache_cell6}.")

    # =============================================================================
    # VISUALIZATION
    # =============================================================================
    print("\n--- Generating Plots ---")
    sns.set_context("poster")

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])

    # Row 1: Self-Decoding (Permutation Distribution)
    p_sad = plot_dist_with_thresh(perm_acc_sad, res_sad['accuracy'], fig.add_subplot(gs[0, 0]), 
                                  f"SAD Self-Decoding (CV Acc: {res_sad['accuracy']:.2f})")
    p_hc = plot_dist_with_thresh(perm_acc_hc, res_hc['accuracy'], fig.add_subplot(gs[0, 1]), 
                                 f"HC Self-Decoding (CV Acc: {res_hc['accuracy']:.2f})")

    # Row 2: Matrices
    # Functional Specificity
    ax3 = fig.add_subplot(gs[1, 0])

    # Matrix: [CV Accuracy] vs [Mean Cross Accuracy]
    # Diagonals: Generalization within group (CV)
    # Off-Diagonals: Generalization across groups (Cross-Decoding)
    func_matrix = np.array([
        [res_sad['accuracy'], mean_sad2hc], 
        [mean_hc2sad, res_hc['accuracy']]
    ])
    func_pvals = np.array([[p_sad, p_sad2hc], [p_hc2sad, p_hc]])

    annot_func = np.empty_like(func_matrix, dtype=object)
    for i in range(2):
        for j in range(2):
            val_str = f"{func_matrix[i, j]:.3f}"
            sig_str = "*" if func_pvals[i, j] < 0.05 else ""
            annot_func[i, j] = f"{val_str}\n({sig_str})"

    sns.heatmap(func_matrix, annot=annot_func, fmt="", cmap="RdBu_r", center=0.5, vmin=0.3, vmax=0.9, cbar=True,
                xticklabels=['Test SAD', 'Test HC'], yticklabels=['Train SAD', 'Train HC'], ax=ax3)
    ax3.set_title("Functional Specificity\n(Forced-Choice Accuracy)")

    # Spatial Specificity
    ax4 = fig.add_subplot(gs[1, 1])
    spatial_matrix = np.array([[1.0, obs_sim], [obs_sim, 1.0]])
    spatial_pvals = np.array([[0.0, p_sim_spatial], [p_sim_spatial, 0.0]])
    annot_spatial = np.empty_like(spatial_matrix, dtype=object)
    for i in range(2):
        for j in range(2):
            star = "*" if (spatial_pvals[i, j] < 0.05 and i != j) else ""
            annot_spatial[i, j] = f"{spatial_matrix[i, j]:.3f}\n{star}"

    sns.heatmap(spatial_matrix, annot=annot_spatial, fmt="", cmap="RdBu_r", center=0, vmin=-1, vmax=1, cbar=True,
                xticklabels=['SAD Map', 'HC Map'], yticklabels=['SAD Map', 'HC Map'], ax=ax4)
    ax4.set_title("Spatial Specificity")

    plt.tight_layout()
    plt.show()
    if stage6_filter_label:
        labeled_plot_payload = {
            "results_11": results_11,
            "func_matrix": func_matrix,
            "func_pvals": func_pvals,
            "spatial_matrix": spatial_matrix,
            "spatial_pvals": spatial_pvals,
            "analysis_label": stage6_filter_label,
        }
        labeled_plot_path = os.path.join(CHECKPOINT_DIR, f"cell_06_{stage6_filter_label}_plot.joblib")
        joblib.dump(labeled_plot_payload, labeled_plot_path)
        print(f"[Cell checkpoint] Saved labeled cell 6 plot payload -> {labeled_plot_path}")
    else:
        save_cell_results(6, ['N_JOBS', 'annot_func', 'annot_spatial', 'ax3', 'ax4', 'cache_cell6', 'data_subsets', 'fig', 'func_matrix', 'func_pvals', 'gs', 'i', 'importance_mask_permutated', 'importance_scores_permutated', 'meta', 'p_hc', 'p_sad', 'results_11', 'results_12', 'results_13', 'results_13_2', 'results_14_self', 'results_21', 'results_21_pv', 'results_22', 'results_23', 'results_24', 'results_25', 'spatial_matrix', 'spatial_pvals', 'strict_cross_phase_results', 'sub_to_meta', 'target_param'])


else:
    maybe_load_cell_results(6)

# %% [cell 7]
if cell_active(7):
    # Cell 7: Corrected Cross-Phase Permutation (Strict Cell 6 Mirror)
    # This creates saved cross-phase null distributions without relying on notebook-only variables.

    def run_strict_forced_choice_perm(model, X, y, sub, iters):
        """Strict cross-phase permutation using subject-level forced-choice accuracy."""
        perm_accs = []
        classes = model.classes_
        for _ in range(iters):
            y_shuffled = np.random.permutation(y)
            sub_scores = []
            for s in np.unique(sub):
                mask = sub == s
                X_s, y_s = X[mask], y_shuffled[mask]
                idx0 = np.where(y_s == classes[0])[0]
                idx1 = np.where(y_s == classes[1])[0]
                if len(idx0) > 0 and len(idx1) > 0:
                    d_scores = model.decision_function(X_s)
                    mean_ev_0 = np.mean(d_scores[idx0])
                    mean_ev_1 = np.mean(d_scores[idx1])
                    sub_scores.append(float(mean_ev_1 > mean_ev_0))
            if sub_scores:
                perm_accs.append(np.mean(sub_scores))
        return np.array(perm_accs)

    print("--- Regenerating Valid Cross-Phase Null Distributions (Centered at 0.5) ---")
    strict_cross_phase_results = {}
    for grp in ["SAD", "HC"]:
        X_rst, y_rst, sub_rst = get_phase_data(f"{grp}_Placebo", "rst")
        model_ext = results_11["model_sad"] if grp == "SAD" else results_11["model_hc"]
        obs_accs = run_cross_decoding(model_ext, X_rst, y_rst, sub_rst, model_ext.classes_)
        obs_mean = float(np.mean(obs_accs))
        permutation_chunks = make_permutation_chunks(N_PERMUTATION, N_JOBS)
        perm_dist_corrected = np.concatenate(Parallel(n_jobs=N_JOBS)(
            delayed(run_strict_forced_choice_perm)(model_ext, X_rst, y_rst, sub_rst, n_iter)
            for n_iter in permutation_chunks
        ))
        p_val = float(np.mean(perm_dist_corrected >= obs_mean)) if len(perm_dist_corrected) else np.nan
        strict_cross_phase_results[grp] = {
            "observed_accuracy": obs_mean,
            "subject_accuracies": obs_accs,
            "perm_dist": perm_dist_corrected,
            "p_val": p_val,
        }
        joblib.dump(strict_cross_phase_results[grp], os.path.join(CHECKPOINT_DIR, f"strict_cross_phase_{grp}_fear_network.joblib"))
        print(f"  > {grp}: observed={obs_mean:.4f}, p={p_val:.4f}")
    save_checkpoint(7, {"strict_cross_phase_results": strict_cross_phase_results})
    save_intermediate("stage07_strict_cross_phase", strict_cross_phase_results)
    print("Cross-phase permutations complete.")
    save_cell_results(7, ['data_subsets', 'grp', 'importance_mask_permutated', 'importance_scores_permutated', 'meta', 'results_11', 'results_12', 'results_13', 'results_13_2', 'results_14_self', 'results_21', 'results_21_pv', 'results_22', 'results_23', 'results_24', 'results_25', 'strict_cross_phase_results', 'sub_to_meta'])


else:
    maybe_load_cell_results(7)

# %% [cell 8]
if cell_active(8):
    for name in ['SAD', 'HC']:
            # Appended fear_network to differentiate from future whole-brain analysis
            cache_file = os.path.join(CHECKPOINT_DIR, f'perm_results_{name}_fear_network_2way.joblib')
    save_cell_results(8, ['data_subsets', 'importance_mask_permutated', 'importance_scores_permutated', 'meta', 'name', 'results_11', 'results_12', 'results_13', 'results_13_2', 'results_14_self', 'results_21', 'results_21_pv', 'results_22', 'results_23', 'results_24', 'results_25', 'strict_cross_phase_results', 'sub_to_meta'])


else:
    maybe_load_cell_results(8)

# %% [cell 9]
if cell_active(9):
    # notebook display only: cache_file
    pass

else:
    maybe_load_cell_results(9)

# %% [cell 10]
if cell_active(10):
    #stage 7 Haufe/Z-scoring in fear network
    STAGE = 7
    if STAGE is None or STAGE == 7:
        print("--- Cell 7: Voxel-wise Spatial Analysis (Fear Network with Symmetry Fallback) ---")
        alpha_val = thresh_hold_p if 'thresh_hold_p' in locals() else 0.05
        fdr_alpha = 1 - alpha_val if alpha_val > 0.5 else alpha_val
        print(f"  [CONFIG] Using FDR Alpha: {fdr_alpha}")
    
        ROI_ORDER = [
                'left_acc', 'left_amygdala', 'left_hippocampus', 'left_insula', 'left_vmpfc',
                'right_acc', 'right_amygdala', 'right_hippocampus', 'right_insula', 'right_vmpfc'
            ]
        target_pair = ['CSR', 'CSS']
    
        # Storage for raw data and initial FDR counts
        raw_results = {}
        fdr_counts = {}
    
        for name in ['SAD', 'HC']:
            # Appended fear_network to differentiate from future whole-brain analysis
            cache_file = os.path.join(CHECKPOINT_DIR, f'perm_results_{name}_fear_network_2way.joblib')
        
            # 1. LOAD CACHE OR CALCULATE
            if os.path.exists(cache_file):
                print(f"  [LOAD] {name} fear_network permutations found.")
                cache = joblib.load(cache_file)
                obs_w, null_w, p_raw = cache['obs_weights'], cache['null_weights'], cache['p_values_raw']
            else:
                print(f"  [CALC] No cache for {name}. Running {N_PERMUTATION} permutations...")
                X_p, y_p, sub_p = get_extinction_data(f"{name}_Placebo")
                mask_cls = np.isin(y_p, target_pair)
            
                # Retrieve best C from Analysis 1.1 (Cell 6) results
                best_c = results_11[f"acc_{name.lower()}_cv"] # Or however you stored the dict in Cell 6
                # If Cell 6 used the res_sad/res_hc naming convention:
                best_c = res_sad['best_C'] if name == 'SAD' else res_hc['best_C']
            
                obs_w, null_w, p_raw = run_raw_permutations(
                    X_p[mask_cls], y_p[mask_cls], sub_p[mask_cls], 
                    build_binary_pipeline(), best_c, N_PERMUTATION
                )
                save_permutation_results(obs_w, null_w, p_raw, cache_file)
            
            # Initial FDR Check to determine the logic for Phase 2
            _, sig_mask, _, _ = apply_flexible_voxel_correction(
                obs_w, null_w, p_raw, method='roi', roi_names=ROI_ORDER, roi_dir=ROI_DIR
            )
        
            raw_results[name] = {'obs': obs_w, 'null': null_w, 'p_raw': p_raw}
            fdr_counts[name] = np.sum(sig_mask)
            print(f"  > {name} Initial FDR-significant voxels: {fdr_counts[name]}")

        # 2. MATCHING LOGIC
        # CASE 1: Both have sig voxels -> Keep both (ROI-FDR)
        if fdr_counts['SAD'] > 0 and fdr_counts['HC'] > 0:
            match_cfg = {'SAD': None, 'HC': None}
        # CASE 2: SAD has sig, HC does not -> HC matches SAD count
        elif fdr_counts['SAD'] > 0 and fdr_counts['HC'] == 0:
            match_cfg = {'SAD': None, 'HC': fdr_counts['SAD']}
        # CASE 3: HC has sig, SAD does not -> SAD matches HC count
        elif fdr_counts['HC'] > 0 and fdr_counts['SAD'] == 0:
            match_cfg = {'SAD': fdr_counts['HC'], 'HC': None}
        # CASE 4: Both have 0 -> Both use 2% fallback
        else:
            match_cfg = {'SAD': 'fallback', 'HC': 'fallback'}

        # 3. FINAL CORRECTION, RECONSTRUCTION & PLOTTING
        for name in ['SAD', 'HC']:
            cfg = match_cfg[name]
            data = raw_results[name]
        
            if cfg == 'fallback':
                z_scores, sig_mask, roi_stats, mode = apply_flexible_voxel_correction(
                    data['obs'], data['null'], data['p_raw'], method='top_n', 
                    roi_names=ROI_ORDER, roi_dir=ROI_DIR, fallback_percentile=98)
            else:
                # If cfg is None, it uses FDR. If cfg is a number, it uses match_count.
                z_scores, sig_mask, roi_stats, mode = apply_flexible_voxel_correction(
                    data['obs'], data['null'], data['p_raw'], method='roi', 
                    roi_names=ROI_ORDER, roi_dir=ROI_DIR, match_count=cfg)

            # Direction Correction for plotting (CSR+ = Red)
            X_p, y_p, _ = get_extinction_data(f"{name}_Placebo")
            mask_cls = np.isin(y_p, target_pair)
            dummy_pipe = build_binary_pipeline(); dummy_pipe.fit(X_p[mask_cls], y_p[mask_cls])
            if dummy_pipe.classes_[0] == 'CSR': z_scores = -z_scores
        
            print(f"\nResults for {name} ({mode}):")
            print(f"{'ROI Name':<25} | {'Voxels':<8} | {'% ROI'}")
            print("-" * 50)
            for roi, stats in sorted(roi_stats.items(), key=lambda x: x[1]['count'], reverse=True):
                print(f"{roi:<25} | {stats['count']:<8} | {stats['perc']:.2f}%")

            # Visualization
            z_img = reconstruct_roi_map(z_scores * sig_mask, ROI_ORDER, ROI_DIR)
            if z_img:
                # Dynamically set threshold to the minimum significant voxel magnitude
                v_thresh = np.min(np.abs(z_scores[sig_mask])) if np.sum(sig_mask) > 0 else 1.96
                plotting.plot_glass_brain(z_img, threshold=v_thresh, colorbar=True, cmap='RdBu_r', 
                                          title=f"{name} Fear Network: {mode}")
                plt.show()

        print("--- Cell 7 Fear Network Analysis Complete ---")
    save_cell_results(10, ['STAGE', 'data_subsets', 'importance_mask_permutated', 'importance_scores_permutated', 'meta', 'results_11', 'results_12', 'results_13', 'results_13_2', 'results_14_self', 'results_21', 'results_21_pv', 'results_22', 'results_23', 'results_24', 'results_25', 'strict_cross_phase_results', 'sub_to_meta'])


else:
    maybe_load_cell_results(10)

# %% [cell 11]
if cell_active(11):
    # Stage 11: empirical permutation-importance masks, chunkable for Slurm arrays.
    from sklearn.base import clone
    import os
    import glob
    import joblib
    import numpy as np

    ALPHA_LEVEL = 0.05
    importance_mask_permutated = {}
    importance_scores_permutated = {}
    p_values_permutated = {}
    importance_diagnostics_permutated = {}
    stage11_groups = ['SAD', 'HC'] if STAGE11_GROUP == "ALL" else [STAGE11_GROUP]
    stage11_chunk_suffix = "" if (STAGE11_MASK_MODE == "current" and STAGE11_SCORING == "decision_margin") else f"_{STAGE11_MASK_MODE}_{STAGE11_SCORING}"
    stage11_chunk_dir = os.path.join(CHECKPOINT_DIR, f"stage11_chunks{stage11_chunk_suffix}")
    os.makedirs(stage11_chunk_dir, exist_ok=True)
    stage11_scorer = get_stage11_scorer_func()

    def stage11_bounds(total, chunk_idx, chunk_count):
        total = int(total)
        chunk_count = max(1, int(chunk_count))
        chunk_idx = 0 if chunk_idx is None else int(chunk_idx)
        if chunk_idx < 0 or chunk_idx >= chunk_count:
            raise ValueError(f"Invalid chunk index {chunk_idx}; expected 0..{chunk_count - 1}")
        base = total // chunk_count
        rem = total % chunk_count
        start = chunk_idx * base + min(chunk_idx, rem)
        end = start + base + (1 if chunk_idx < rem else 0)
        return start, end

    def stage11_prepare_group(group_name):
        data_ptr = data_subsets[f"{group_name}_Placebo"]["ext"]
        mask_cls = np.isin(data_ptr["y"], ['CSR', 'CSS'])
        X_target = data_ptr["X"][mask_cls]
        y_target = data_ptr["y"][mask_cls]
        model_template = res_sad['model'] if group_name == 'SAD' else res_hc['model']
        return X_target, y_target, model_template

    def stage11_chunk_path(group_name, chunk_idx):
        return os.path.join(stage11_chunk_dir, f"stage11_{group_name}_chunk_{int(chunk_idx):04d}.joblib")

    def stage11_save_group(group_name, actual_imp, p_values, null_n):
        sig_mask = (p_values < ALPHA_LEVEL) & (actual_imp > 0)
        positive_n = int(np.sum(actual_imp > 0))
        diag = {
            "n_features": int(actual_imp.size),
            "n_positive_importance": positive_n,
            "n_significant_p_lt_0_05_positive": int(np.sum(sig_mask)),
            "importance_min": float(np.nanmin(actual_imp)),
            "importance_max": float(np.nanmax(actual_imp)),
            "importance_mean": float(np.nanmean(actual_imp)),
            "importance_p95": float(np.nanpercentile(actual_imp, 95)),
            "p_min": float(np.nanmin(p_values)),
            "p_p05": float(np.nanpercentile(p_values, 5)),
            "p_median": float(np.nanmedian(p_values)),
            "null_permutations": int(null_n),
            "importance_scoring": STAGE11_SCORER_NAME,
            "stage11_scoring": STAGE11_SCORING,
            "stage11_mask_mode": STAGE11_MASK_MODE,
            "stage11_fallback_policy": (
                "all_positive_if_empirical_mask_too_small"
                if STAGE11_MASK_MODE == "current"
                else "disabled_original_notebook"
            ),
        }
        payload = {
            "importance_mask_permutated": {group_name: sig_mask},
            "importance_scores_permutated": {group_name: actual_imp},
            "p_values_permutated": {group_name: p_values},
            "importance_diagnostics_permutated": {group_name: diag},
            "null_permutations": {group_name: int(null_n)},
            "actual_repeats": {group_name: int(STAGE11_ACTUAL_REPEATS)},
        }
        group_ckpt = os.path.join(CHECKPOINT_DIR, f"cell_11_{group_name}.joblib")
        group_intermediate = os.path.join(INTERMEDIATE_DIR, f"stage11_importance_masks_{group_name}.joblib")
        joblib.dump(payload, group_ckpt)
        joblib.dump(payload, group_intermediate)
        importance_mask_permutated[group_name] = sig_mask
        importance_scores_permutated[group_name] = actual_imp
        p_values_permutated[group_name] = p_values
        importance_diagnostics_permutated[group_name] = diag
        print(f"   > Result: {np.sum(sig_mask)} voxels significant at p < {ALPHA_LEVEL}.")
        print(
            f"   > Importance diagnostics: positive={positive_n}/{actual_imp.size}, "
            f"max={diag['importance_max']:.6f}, p_min={diag['p_min']:.6f}."
        )
        print(f"   > Saved final stage 11 output for {group_name} -> {group_ckpt}")

    def stage11_compute_chunk(group_name):
        chunk_idx = 0 if STAGE11_CHUNK_IDX is None else int(STAGE11_CHUNK_IDX)
        chunk_count = max(1, int(STAGE11_CHUNK_COUNT))
        actual_start, actual_end = stage11_bounds(STAGE11_ACTUAL_REPEATS, chunk_idx, chunk_count)
        null_start, null_end = stage11_bounds(N_NULL_PERMS, chunk_idx, chunk_count)
        actual_repeats = actual_end - actual_start
        null_repeats = null_end - null_start
        if actual_repeats <= 0 and null_repeats <= 0:
            print(f"  [SKIP] {group_name} chunk {chunk_idx}/{chunk_count} has no work.")
            return

        X_target, y_target, model_template = stage11_prepare_group(group_name)
        print(
            f"--- Stage 11 chunk {chunk_idx + 1}/{chunk_count} for {group_name}: "
            f"actual repeats {actual_start}:{actual_end}, null perms {null_start}:{null_end} ---"
        )

        actual_sum = np.zeros(X_target.shape[1], dtype=np.float64)
        if actual_repeats > 0:
            actual_res = permutation_importance(
                model_template,
                X_target,
                y_target,
                n_repeats=actual_repeats,
                scoring=stage11_scorer,
                n_jobs=N_JOBS,
                random_state=RANDOM_STATE + actual_start,
            )
            actual_mean = actual_res.importances_mean if hasattr(actual_res, 'importances_mean') else actual_res
            actual_sum = np.asarray(actual_mean, dtype=np.float64) * actual_repeats

        null_dist = np.zeros((null_repeats, X_target.shape[1]), dtype=np.float32)
        for row, perm_idx in enumerate(range(null_start, null_end)):
            rng = np.random.default_rng(RANDOM_STATE + perm_idx)
            y_shuffled = rng.permutation(y_target)
            null_model = clone(model_template).fit(X_target, y_shuffled)
            null_res = permutation_importance(
                null_model,
                X_target,
                y_shuffled,
                n_repeats=1,
                scoring=stage11_scorer,
                n_jobs=N_JOBS,
                random_state=RANDOM_STATE + 100000 + perm_idx,
            )
            null_imp = null_res.importances_mean if hasattr(null_res, 'importances_mean') else null_res
            null_dist[row, :] = np.asarray(null_imp, dtype=np.float32)
            if (row + 1) % 10 == 0 or (row + 1) == null_repeats:
                print(f"   > {group_name} chunk {chunk_idx + 1}/{chunk_count}: {row + 1}/{null_repeats} null permutations complete.")

        chunk_payload = {
            "group": group_name,
            "chunk_idx": chunk_idx,
            "chunk_count": chunk_count,
            "actual_start": actual_start,
            "actual_end": actual_end,
            "actual_repeats": actual_repeats,
            "actual_importance_sum": actual_sum,
            "null_start": null_start,
            "null_end": null_end,
            "null_dist": null_dist,
            "stage11_mask_mode": STAGE11_MASK_MODE,
            "stage11_scoring": STAGE11_SCORING,
            "importance_scoring": STAGE11_SCORER_NAME,
        }
        out_path = stage11_chunk_path(group_name, chunk_idx)
        joblib.dump(chunk_payload, out_path, compress=3)
        print(f"  [SAVE] Stage 11 chunk saved -> {out_path}")

        if chunk_count == 1:
            actual_imp = actual_sum / max(1, actual_repeats)
            p_values = ((np.sum(null_dist >= actual_imp, axis=0) + 1) / (max(1, null_repeats) + 1)).astype(np.float64)
            stage11_save_group(group_name, actual_imp, p_values, null_repeats)

    def stage11_merge_group(group_name):
        paths = sorted(glob.glob(os.path.join(stage11_chunk_dir, f"stage11_{group_name}_chunk_*.joblib")))
        if not paths:
            raise FileNotFoundError(f"No Stage 11 chunk files found for {group_name} in {stage11_chunk_dir}")
        print(f"--- Stage 11 merge for {group_name}: {len(paths)} chunk files ---")
        actual_sum = None
        actual_n = 0
        null_n = 0
        count_ge = None
        chunks_seen = set()

        for path in paths:
            payload = joblib.load(path)
            if payload.get("stage11_mask_mode", STAGE11_MASK_MODE) != STAGE11_MASK_MODE:
                raise ValueError(
                    f"Stage 11 merge for {group_name}: chunk {path} was generated with "
                    f"mode={payload.get('stage11_mask_mode')}, requested {STAGE11_MASK_MODE}."
                )
            if payload.get("stage11_scoring", STAGE11_SCORING) != STAGE11_SCORING:
                raise ValueError(
                    f"Stage 11 merge for {group_name}: chunk {path} was generated with "
                    f"scoring={payload.get('stage11_scoring')}, requested {STAGE11_SCORING}."
                )
            chunks_seen.add(int(payload["chunk_idx"]))
            chunk_actual = np.asarray(payload["actual_importance_sum"], dtype=np.float64)
            actual_sum = chunk_actual if actual_sum is None else actual_sum + chunk_actual
            actual_n += int(payload["actual_repeats"])

        if actual_sum is None or actual_n == 0:
            raise ValueError(f"No actual importance repeats found for {group_name}.")
        actual_imp = actual_sum / actual_n

        for path in paths:
            payload = joblib.load(path)
            if payload.get("stage11_mask_mode", STAGE11_MASK_MODE) != STAGE11_MASK_MODE:
                raise ValueError(
                    f"Stage 11 merge for {group_name}: chunk {path} was generated with "
                    f"mode={payload.get('stage11_mask_mode')}, requested {STAGE11_MASK_MODE}."
                )
            if payload.get("stage11_scoring", STAGE11_SCORING) != STAGE11_SCORING:
                raise ValueError(
                    f"Stage 11 merge for {group_name}: chunk {path} was generated with "
                    f"scoring={payload.get('stage11_scoring')}, requested {STAGE11_SCORING}."
                )
            null_dist = np.asarray(payload["null_dist"])
            if null_dist.size == 0:
                continue
            ge = np.sum(null_dist >= actual_imp, axis=0, dtype=np.int64)
            count_ge = ge if count_ge is None else count_ge + ge
            null_n += null_dist.shape[0]

        if count_ge is None or null_n == 0:
            raise ValueError(f"No null permutation rows found for {group_name}.")
        expected_chunks = max(int(joblib.load(paths[0]).get("chunk_count", len(paths))), len(paths))
        if len(chunks_seen) < expected_chunks:
            raise FileNotFoundError(
                f"Only found {len(chunks_seen)}/{expected_chunks} Stage 11 chunks for {group_name}. "
                "Wait for all array tasks to finish before merging."
            )
        p_values = (count_ge + 1) / (null_n + 1)
        stage11_save_group(group_name, actual_imp, p_values, null_n)

    print(
        f"--- Stage 11: Empirical permutation-importance masks "
        f"(group={STAGE11_GROUP}, null={N_NULL_PERMS}, actual_repeats={STAGE11_ACTUAL_REPEATS}, "
        f"chunks={STAGE11_CHUNK_COUNT}, merge={STAGE11_MERGE}, "
        f"mask_mode={STAGE11_MASK_MODE}, scoring={STAGE11_SCORING}) ---"
    )

    for group_name in stage11_groups:
        if STAGE11_MERGE:
            stage11_merge_group(group_name)
        else:
            stage11_compute_chunk(group_name)

    if importance_mask_permutated:
        combined_stage11_payload = {
            "importance_mask_permutated": importance_mask_permutated,
            "importance_scores_permutated": importance_scores_permutated,
            "p_values_permutated": p_values_permutated,
            "importance_diagnostics_permutated": importance_diagnostics_permutated,
            "stage11_mask_mode": STAGE11_MASK_MODE,
            "stage11_scoring": STAGE11_SCORING,
            "importance_scoring": STAGE11_SCORER_NAME,
            "null_permutations": globals().get("null_permutations", {}),
            "actual_repeats": {
                group_name: int(STAGE11_ACTUAL_REPEATS)
                for group_name in importance_mask_permutated.keys()
            },
        }
        joblib.dump(combined_stage11_payload, _script_ckpt_path(11))
        joblib.dump(combined_stage11_payload, _script_intermediate_path("stage11_importance_masks"))
        save_checkpoint(11, combined_stage11_payload)
        save_intermediate("stage11_importance_masks", combined_stage11_payload)
        save_cell_results(11, ['ALPHA_LEVEL', 'N_NULL_PERMS', 'STAGE11_GROUP', 'data_subsets', 'importance_mask_permutated', 'importance_scores_permutated', 'meta', 'p_values_permutated', 'results_11', 'results_12', 'results_13', 'results_13_2', 'results_14_self', 'results_21', 'results_21_pv', 'results_22', 'results_23', 'results_24', 'results_25', 'stage11_groups', 'strict_cross_phase_results', 'sub_to_meta'])
    elif not STAGE11_MERGE and STAGE11_CHUNK_COUNT > 1:
        print(
            "--- Stage 11 chunk complete. No final masks were produced in this run; "
            "merge chunks with --stage11_merge before downstream analyses. ---"
        )
        raise SystemExit(0)

    print("--- Stage 11 Complete ---")


else:
    maybe_load_cell_results(11)

# %% [cell 12]
if cell_active(12):
    # Cell 10: Analysis 1.2 - Static Representational Topology (features with positive importance + p at .05)
    # Objective: Characterize the stable organization of the social learning space.
    # Method: Cross-validated Mahalanobis (crossnobis) distance with shrinkage covariance, averaged over split-half repeats.
    # Tests: Group Comparison (SAD vs HC) AND One-Sample Test (Dist > 0).
    # Reporting: Raw, z-scored, and per-voxel normalized crossnobis distances.

    print("--- Running Analysis 1.2: Static Representational Topology (I > 0, p < .05| Centroid) ---")

    from scipy.stats import ttest_1samp

    # Global Constants
    RDM_CONDITIONS = ["CS-", "CSS", "CSR"] 
    idx_cs_minus, idx_css, idx_csr = 0, 1, 2

    cache_cell10 = os.path.join(CHECKPOINT_DIR, "analysis_12_topology.joblib")
    legacy_cache_cell10 = os.path.join(CHECKPOINT_DIR, "cell_10.joblib")
    cache_cell10_load = None
    for candidate in (cache_cell10, legacy_cache_cell10):
        if not os.path.exists(candidate):
            continue
        candidate_payload = joblib.load(candidate)
        if isinstance(candidate_payload, dict) and "results_12" in candidate_payload:
            candidate_results = candidate_payload.get("results_12", {})
            if "shock_anchor_results" not in candidate_results:
                print(f"  [STALE] Found existing RDM results in {candidate}, but Shock-anchor sensitivity is missing. Recomputing...")
                continue
            if "aim2_geometry_panel" not in candidate_results:
                print(f"  [STALE] Found existing RDM results in {candidate}, but true Aim 2 centroid geometry is missing. Recomputing...")
                continue
            cache_cell10_load = candidate_payload
            print(f"  [LOAD] Found existing RDM results in {candidate}. Skipping Crossnobis calculation...")
            break
    if cache_cell10_load is not None:
        results_12 = cache_cell10_load['results_12']
        # Extract calculated RDMs for plotting/stats blocks
        rdms_sad_raw = results_12["rdms_sad_raw"]
        rdms_hc_raw = results_12["rdms_hc_raw"]
        rdms_sad_z = results_12["rdms_sad_z"]
        rdms_hc_z = results_12["rdms_hc_z"]
        rdms_sad_raw_pv = results_12["rdms_sad_raw_pv"]
        rdms_hc_raw_pv = results_12["rdms_hc_raw_pv"]
        mask_sad_analysis = cache_cell10_load.get("mask_sad_analysis", cache_cell10_load.get("mask_sad_top5"))
        mask_hc_analysis = cache_cell10_load.get("mask_hc_analysis", cache_cell10_load.get("mask_hc_top5"))
        if mask_sad_analysis is not None and mask_hc_analysis is not None:
            mask_sad_top5 = mask_sad_analysis
            mask_hc_top5 = mask_hc_analysis
        export_aim2_geometry_panel(results_12.get("aim2_geometry_panel"))
    else:
        # =============================================================================
        # 0. Feature Selection
        print("\n[Step 0] Using empirical permutation-importance feature masks.")
        importance_masks, feature_space_12 = get_analysis_feature_masks("Analysis 1.2")

        mask_sad_analysis = importance_masks['SAD']
        mask_hc_analysis = importance_masks['HC']
        # Legacy aliases kept so older result bundles remain readable.
        mask_sad_top5 = mask_sad_analysis
        mask_hc_top5 = mask_hc_analysis
        # =============================================================================
        # 1. Data Preparation (Recovering CS-)
        # =============================================================================
        print("\n[Step 1] Preparing Centroid Data...")

        # Validate Source Data
        if 'X_ext' not in locals() or 'y_ext' not in locals():
            raise ValueError("Global 'X_ext' variables missing. Cannot retrieve CS- trials (Cell 5 filtered them out).")

        # Retrieve Subject Lists from the Nested Dictionary (created in Cell 5)
        # structure: data_subsets['Group']['ext']['sub']
        try:
            known_hc = np.unique(data_subsets["HC_Placebo"]["ext"]["sub"])
            known_sad = np.unique(data_subsets["SAD_Placebo"]["ext"]["sub"])
        except (KeyError, TypeError):
            raise ValueError("Data structure mismatch. Ensure Cell 5 generated 'data_subsets' with ['ext'] keys.")

        # Create a temporary group mapping array matching the global X_ext
        group_ext = np.array(["Unknown"] * len(sub_ext), dtype=object)
        group_ext[np.isin(sub_ext, known_hc)] = "HC"
        group_ext[np.isin(sub_ext, known_sad)] = "SAD"

        # Filter Global Data for RDM Conditions
        mask_conds = np.isin(y_ext, RDM_CONDITIONS)
        X_raw = X_ext[mask_conds]
        y_raw = y_ext[mask_conds]
        sub_raw = sub_ext[mask_conds]
        grp_raw = group_ext[mask_conds]

        # Split by Group
        mask_sad_grp = (grp_raw == "SAD")
        mask_hc_grp = (grp_raw == "HC")

        # Slice Features (apply selected empirical masks)
        X_sad_12 = X_raw[mask_sad_grp][:, mask_sad_analysis]
        y_sad_12 = y_raw[mask_sad_grp]
        sub_sad_12 = sub_raw[mask_sad_grp]

        X_hc_12 = X_raw[mask_hc_grp][:, mask_hc_analysis]
        y_hc_12 = y_raw[mask_hc_grp]
        sub_hc_12 = sub_raw[mask_hc_grp]

        print(f"  > SAD Matrix (selected features): {X_sad_12.shape} | HC Matrix (selected features): {X_hc_12.shape}")

        # =============================================================================
        # 2. Centroid RDM Calculation
        # =============================================================================


        # Compute RDMs (raw + z-scored)
        print(f"  Calculating Centroid RDMs (Conditions: {RDM_CONDITIONS}) with {CROSSNOBIS_REPEATS} split-half repeats...")
        rdms_sad_raw, subs_sad_rdm = calculate_crossnobis_rdm(X_sad_12, y_sad_12, sub_sad_12, RDM_CONDITIONS, standardize=False)
        rdms_hc_raw, subs_hc_rdm = calculate_crossnobis_rdm(X_hc_12, y_hc_12, sub_hc_12, RDM_CONDITIONS, standardize=False)

        rdms_sad_z, subs_sad_rdm_z = calculate_crossnobis_rdm(X_sad_12, y_sad_12, sub_sad_12, RDM_CONDITIONS, standardize=True)
        rdms_hc_z, subs_hc_rdm_z = calculate_crossnobis_rdm(X_hc_12, y_hc_12, sub_hc_12, RDM_CONDITIONS, standardize=True)

        print(f"  > Computed RDMs (raw): SAD (n={len(subs_sad_rdm)}), HC (n={len(subs_hc_rdm)})")
        print(f"  > Computed RDMs (z-scored): SAD (n={len(subs_sad_rdm_z)}), HC (n={len(subs_hc_rdm_z)})")

        aim2_geometry_panel = pd.concat([
            build_true_condition_centroid_geometry(
                X_sad_12, y_sad_12, sub_sad_12, "SAD", RDM_CONDITIONS, feature_space_12
            ),
            build_true_condition_centroid_geometry(
                X_hc_12, y_hc_12, sub_hc_12, "HC", RDM_CONDITIONS, feature_space_12
            ),
        ], ignore_index=True)
        export_aim2_geometry_panel(aim2_geometry_panel)

        # Per-voxel normalization (scale by number of features)
        n_feat_sad = X_sad_12.shape[1]
        n_feat_hc = X_hc_12.shape[1]
        rdms_sad_raw_pv = rdms_sad_raw / n_feat_sad
        rdms_hc_raw_pv = rdms_hc_raw / n_feat_hc
        rdms_sad_z_pv = rdms_sad_z / n_feat_sad
        rdms_hc_z_pv = rdms_hc_z / n_feat_hc

        # Secondary Shock-anchor sensitivity: residualize each trial across features,
        # then project CS patterns onto each subject's Shock-minus-CS- axis. Shock is
        # intentionally not added as a fourth RDM condition because its global evoked
        # amplitude can dominate crossnobis distances.
        shock_anchor_df_sad = pd.DataFrame()
        shock_anchor_df_hc = pd.DataFrame()
        shock_anchor_df = pd.DataFrame()
        shock_anchor_stats = {}
        X_shock_sad, y_shock_sad, sub_shock_sad = get_shock_target_data("SAD_Placebo")
        X_shock_hc, y_shock_hc, sub_shock_hc = get_shock_target_data("HC_Placebo")
        if X_shock_sad is not None and X_shock_hc is not None:
            print("  Calculating residualized Shock-anchor projection sensitivity...")
            shock_anchor_df_sad = calculate_residualized_shock_anchor_projection(
                X_sad_12, y_sad_12, sub_sad_12, X_shock_sad[:, mask_sad_analysis], y_shock_sad, sub_shock_sad
            )
            shock_anchor_df_hc = calculate_residualized_shock_anchor_projection(
                X_hc_12, y_hc_12, sub_hc_12, X_shock_hc[:, mask_hc_analysis], y_shock_hc, sub_shock_hc
            )
            if not shock_anchor_df_sad.empty:
                shock_anchor_df_sad["Group"] = "SAD"
            if not shock_anchor_df_hc.empty:
                shock_anchor_df_hc["Group"] = "HC"
            shock_anchor_df = pd.concat([shock_anchor_df_sad, shock_anchor_df_hc], ignore_index=True)
            shock_anchor_stats = summarize_shock_anchor_projection(shock_anchor_df_sad, shock_anchor_df_hc, n_perm=N_PERMUTATION)
        else:
            print(f"  ! Shock-anchor sensitivity unavailable: no shock/US target trials found for labels {SHOCK_TARGET_LABELS}.")

        # =============================================================================
        # 3. Metrics & Statistical Tests
        # =============================================================================
        # Conditions: 0=CS-, 1=CSS, 2=CSR
        idx_cs_minus, idx_css, idx_csr = 0, 1, 2

        print("\n[Step 3] Statistical Testing...")


        # ---- RAW ----
        print("\n[RAW] Metric A: Threat (CSR) vs Safety (CSS) Distance")
        vec_a_sad_raw, vec_b_sad_raw = extract_metrics(rdms_sad_raw)
        vec_a_hc_raw, vec_b_hc_raw = extract_metrics(rdms_hc_raw)

        p_a_sad_0_raw = one_sample_test(vec_a_sad_raw, "SAD (Dist > 0)")
        p_a_hc_0_raw = one_sample_test(vec_a_hc_raw, "HC  (Dist > 0)")

        print("  > Group Comparison (SAD vs HC):")
        t_a_raw, p_a_raw, m_a_sad_raw, m_a_hc_raw = perm_ttest_ind(vec_a_sad_raw, vec_a_hc_raw, n_perm=N_PERMUTATION)
        print(f"    Diff: SAD={m_a_sad_raw:.3f}, HC={m_a_hc_raw:.3f} | t={t_a_raw:.3f}, p={p_a_raw:.4f}")

        print("\n[RAW] Metric B: Safety (CSS) vs Background (CS-) Distance")
        p_b_sad_0_raw = one_sample_test(vec_b_sad_raw, "SAD (Dist > 0)")
        p_b_hc_0_raw = one_sample_test(vec_b_hc_raw, "HC  (Dist > 0)")

        print("  > Group Comparison (SAD vs HC):")
        t_b_raw, p_b_raw, m_b_sad_raw, m_b_hc_raw = perm_ttest_ind(vec_b_sad_raw, vec_b_hc_raw, n_perm=N_PERMUTATION)
        print(f"    Diff: SAD={m_b_sad_raw:.3f}, HC={m_b_hc_raw:.3f} | t={t_b_raw:.3f}, p={p_b_raw:.4f}")

        # ---- Z-SCORED ----
        print("\n[Z-SCORED] Metric A: Threat (CSR) vs Safety (CSS) Distance")
        vec_a_sad_z, vec_b_sad_z = extract_metrics(rdms_sad_z)
        vec_a_hc_z, vec_b_hc_z = extract_metrics(rdms_hc_z)

        p_a_sad_0_z = one_sample_test(vec_a_sad_z, "SAD (Dist > 0)")
        p_a_hc_0_z = one_sample_test(vec_a_hc_z, "HC  (Dist > 0)")

        print("  > Group Comparison (SAD vs HC):")
        t_a_z, p_a_z, m_a_sad_z, m_a_hc_z = perm_ttest_ind(vec_a_sad_z, vec_a_hc_z, n_perm=N_PERMUTATION)
        print(f"    Diff: SAD={m_a_sad_z:.3f}, HC={m_a_hc_z:.3f} | t={t_a_z:.3f}, p={p_a_z:.4f}")

        print("\n[Z-SCORED] Metric B: Safety (CSS) vs Background (CS-) Distance")
        p_b_sad_0_z = one_sample_test(vec_b_sad_z, "SAD (Dist > 0)")
        p_b_hc_0_z = one_sample_test(vec_b_hc_z, "HC  (Dist > 0)")

        print("  > Group Comparison (SAD vs HC):")
        t_b_z, p_b_z, m_b_sad_z, m_b_hc_z = perm_ttest_ind(vec_b_sad_z, vec_b_hc_z, n_perm=N_PERMUTATION)
        print(f"    Diff: SAD={m_b_sad_z:.3f}, HC={m_b_hc_z:.3f} | t={t_b_z:.3f}, p={p_b_z:.4f}")

        # ---- PER-VOXEL (RAW) ----
        print("\n[PER-VOXEL RAW] Metric A: Threat (CSR) vs Safety (CSS) Distance")
        vec_a_sad_raw_pv, vec_b_sad_raw_pv = extract_metrics(rdms_sad_raw_pv)
        vec_a_hc_raw_pv, vec_b_hc_raw_pv = extract_metrics(rdms_hc_raw_pv)

        p_a_sad_0_raw_pv = one_sample_test(vec_a_sad_raw_pv, "SAD (Dist > 0)")
        p_a_hc_0_raw_pv = one_sample_test(vec_a_hc_raw_pv, "HC  (Dist > 0)")

        print("  > Group Comparison (SAD vs HC):")
        t_a_raw_pv, p_a_raw_pv, m_a_sad_raw_pv, m_a_hc_raw_pv = perm_ttest_ind(vec_a_sad_raw_pv, vec_a_hc_raw_pv, n_perm=N_PERMUTATION)
        print(f"    Diff: SAD={m_a_sad_raw_pv:.6f}, HC={m_a_hc_raw_pv:.6f} | t={t_a_raw_pv:.3f}, p={p_a_raw_pv:.4f}")

        print("\n[PER-VOXEL RAW] Metric B: Safety (CSS) vs Background (CS-) Distance")
        p_b_sad_0_raw_pv = one_sample_test(vec_b_sad_raw_pv, "SAD (Dist > 0)")
        p_b_hc_0_raw_pv = one_sample_test(vec_b_hc_raw_pv, "HC  (Dist > 0)")

        print("  > Group Comparison (SAD vs HC):")
        t_b_raw_pv, p_b_raw_pv, m_b_sad_raw_pv, m_b_hc_raw_pv = perm_ttest_ind(vec_b_sad_raw_pv, vec_b_hc_raw_pv, n_perm=N_PERMUTATION)
        print(f"    Diff: SAD={m_b_sad_raw_pv:.6f}, HC={m_b_hc_raw_pv:.6f} | t={t_b_raw_pv:.3f}, p={p_b_raw_pv:.4f}")

        def topology_indices(rdms_sad_curr, rdms_hc_curr):
            vec_a_sad_curr, vec_b_sad_curr = extract_metrics(rdms_sad_curr)
            vec_a_hc_curr, vec_b_hc_curr = extract_metrics(rdms_hc_curr)
            vec_c_sad_curr = rdms_sad_curr[:, idx_csr, idx_cs_minus]
            vec_c_hc_curr = rdms_hc_curr[:, idx_csr, idx_cs_minus]
            sii_sad = vec_a_sad_curr - vec_b_sad_curr
            sii_hc = vec_a_hc_curr - vec_b_hc_curr
            tbi_sad = vec_c_sad_curr - vec_b_sad_curr
            tbi_hc = vec_c_hc_curr - vec_b_hc_curr
            t_sii, p_sii, m_sii_sad, m_sii_hc = perm_ttest_ind(sii_sad, sii_hc, n_perm=N_PERMUTATION)
            t_tbi, p_tbi, m_tbi_sad, m_tbi_hc = perm_ttest_ind(tbi_sad, tbi_hc, n_perm=N_PERMUTATION)
            return {
                "safety_integration_index": {"SAD": sii_sad, "HC": sii_hc},
                "threat_bias_index": {"SAD": tbi_sad, "HC": tbi_hc},
                "safety_integration_stats": (t_sii, p_sii),
                "threat_bias_stats": (t_tbi, p_tbi),
                "safety_integration_means": {"SAD": m_sii_sad, "HC": m_sii_hc},
                "threat_bias_means": {"SAD": m_tbi_sad, "HC": m_tbi_hc},
            }

        topology_indices_raw = topology_indices(rdms_sad_raw, rdms_hc_raw)
        topology_indices_z = topology_indices(rdms_sad_z, rdms_hc_z)
        topology_indices_raw_pv = topology_indices(rdms_sad_raw_pv, rdms_hc_raw_pv)
        print("\nPrimary Topology Index: Safety Integration = dist(CSR,CSS) - dist(CSS,CS-)")
        print(
            f"  [PV] SAD={topology_indices_raw_pv['safety_integration_means']['SAD']:.6f}, "
            f"HC={topology_indices_raw_pv['safety_integration_means']['HC']:.6f}, "
            f"p={topology_indices_raw_pv['safety_integration_stats'][1]:.4f}"
        )
        print("Secondary Topology Index: Threat Bias = dist(CSR,CS-) - dist(CSS,CS-)")
        print(
            f"  [PV] SAD={topology_indices_raw_pv['threat_bias_means']['SAD']:.6f}, "
            f"HC={topology_indices_raw_pv['threat_bias_means']['HC']:.6f}, "
            f"p={topology_indices_raw_pv['threat_bias_stats'][1]:.4f}"
        )

        # Store Results
        results_12 = {
            "rdms_sad_raw": rdms_sad_raw,
            "rdms_hc_raw": rdms_hc_raw,
            "rdms_sad_z": rdms_sad_z,
            "rdms_hc_z": rdms_hc_z,
            "rdms_sad_raw_pv": rdms_sad_raw_pv,
            "rdms_hc_raw_pv": rdms_hc_raw_pv,
            "rdms_sad_z_pv": rdms_sad_z_pv,
            "rdms_hc_z_pv": rdms_hc_z_pv,
            "shock_target_labels": SHOCK_TARGET_LABELS,
            "shock_anchor_description": "Trial-wise feature-mean residualized projection of CSS/CSR onto each subject's Shock-minus-CS- axis; Shock is not included in the primary RDM.",
            "shock_anchor_df": shock_anchor_df,
            "shock_anchor_stats": shock_anchor_stats,
            "shock_anchor_results": {
                "df": shock_anchor_df,
                "SAD": shock_anchor_df_sad,
                "HC": shock_anchor_df_hc,
                "stats": shock_anchor_stats,
            },
            "subs_sad_rdm": subs_sad_rdm,
            "subs_hc_rdm": subs_hc_rdm,
            "subs_sad_rdm_z": subs_sad_rdm_z,
            "subs_hc_rdm_z": subs_hc_rdm_z,
            "mask_sad_analysis": mask_sad_analysis,
            "mask_hc_analysis": mask_hc_analysis,
            "feature_space": feature_space_12,
            "aim2_geometry_panel": aim2_geometry_panel,
            "metric_a_stats_raw": (t_a_raw, p_a_raw),
            "metric_b_stats_raw": (t_b_raw, p_b_raw),
            "metric_a_stats_z": (t_a_z, p_a_z),
            "metric_b_stats_z": (t_b_z, p_b_z),
            "metric_a_stats_raw_pv": (t_a_raw_pv, p_a_raw_pv),
            "metric_b_stats_raw_pv": (t_b_raw_pv, p_b_raw_pv),
            "topology_indices_raw": topology_indices_raw,
            "topology_indices_z": topology_indices_z,
            "topology_indices_raw_pv": topology_indices_raw_pv,
            "safety_integration_index_raw_pv": topology_indices_raw_pv["safety_integration_index"],
            "threat_bias_index_raw_pv": topology_indices_raw_pv["threat_bias_index"],
            "safety_integration_stats_raw_pv": topology_indices_raw_pv["safety_integration_stats"],
            "threat_bias_stats_raw_pv": topology_indices_raw_pv["threat_bias_stats"],
            "one_sample_stats_raw": {
                "p_a_sad": p_a_sad_0_raw,
                "p_a_hc": p_a_hc_0_raw,
                "p_b_sad": p_b_sad_0_raw,
                "p_b_hc": p_b_hc_0_raw,
            },
            "one_sample_stats_z": {
                "p_a_sad": p_a_sad_0_z,
                "p_a_hc": p_a_hc_0_z,
                "p_b_sad": p_b_sad_0_z,
                "p_b_hc": p_b_hc_0_z,
            },
            "one_sample_stats_raw_pv": {
                "p_a_sad": p_a_sad_0_raw_pv,
                "p_a_hc": p_a_hc_0_raw_pv,
                "p_b_sad": p_b_sad_0_raw_pv,
                "p_b_hc": p_b_hc_0_raw_pv,
            },
        }
        cache_payload_12 = {
            "results_12": results_12,
            "rdms_sad_raw": rdms_sad_raw,
            "rdms_hc_raw": rdms_hc_raw,
            "rdms_sad_z": rdms_sad_z,
            "rdms_hc_z": rdms_hc_z,
            "rdms_sad_raw_pv": rdms_sad_raw_pv,
            "rdms_hc_raw_pv": rdms_hc_raw_pv,
            "rdms_sad_z_pv": rdms_sad_z_pv,
            "rdms_hc_z_pv": rdms_hc_z_pv,
            "shock_target_labels": SHOCK_TARGET_LABELS,
            "shock_anchor_description": "Trial-wise feature-mean residualized projection of CSS/CSR onto each subject's Shock-minus-CS- axis; Shock is not included in the primary RDM.",
            "shock_anchor_df": shock_anchor_df,
            "shock_anchor_stats": shock_anchor_stats,
            "shock_anchor_results": {
                "df": shock_anchor_df,
                "SAD": shock_anchor_df_sad,
                "HC": shock_anchor_df_hc,
                "stats": shock_anchor_stats,
            },
            "subs_sad_rdm": subs_sad_rdm,
            "subs_hc_rdm": subs_hc_rdm,
            "subs_sad_rdm_z": subs_sad_rdm_z,
            "subs_hc_rdm_z": subs_hc_rdm_z,
            "mask_sad_analysis": mask_sad_analysis,
            "mask_hc_analysis": mask_hc_analysis,
            "feature_space": feature_space_12,
            "aim2_geometry_panel": aim2_geometry_panel,
            "mask_sad_top5": mask_sad_top5,
            "mask_hc_top5": mask_hc_top5,
        }
        joblib.dump(cache_payload_12, cache_cell10)
        save_checkpoint(12, cache_payload_12)
        save_intermediate("stage12_topology_stats", cache_payload_12)

    if 'vec_a_sad_raw' not in locals():
        # Rehydrate plotting/reporting variables when stage 12 is loaded from cache.
        rdms_sad_z_pv = results_12.get("rdms_sad_z_pv")
        rdms_hc_z_pv = results_12.get("rdms_hc_z_pv")
        vec_a_sad_raw, vec_b_sad_raw = extract_metrics(rdms_sad_raw)
        vec_a_hc_raw, vec_b_hc_raw = extract_metrics(rdms_hc_raw)
        vec_a_sad_z, vec_b_sad_z = extract_metrics(rdms_sad_z)
        vec_a_hc_z, vec_b_hc_z = extract_metrics(rdms_hc_z)
        vec_a_sad_raw_pv, vec_b_sad_raw_pv = extract_metrics(rdms_sad_raw_pv)
        vec_a_hc_raw_pv, vec_b_hc_raw_pv = extract_metrics(rdms_hc_raw_pv)
        def _cached_group_stat(key, sad_vec, hc_vec):
            if key in results_12:
                return results_12[key]
            return perm_ttest_ind(sad_vec, hc_vec, n_perm=N_PERMUTATION)[:2]

        def _cached_one_sample(one_dict, key, vec, label):
            if key in one_dict:
                return one_dict[key]
            return one_sample_test(vec, label)

        t_a_raw, p_a_raw = _cached_group_stat("metric_a_stats_raw", vec_a_sad_raw, vec_a_hc_raw)
        t_b_raw, p_b_raw = _cached_group_stat("metric_b_stats_raw", vec_b_sad_raw, vec_b_hc_raw)
        t_a_z, p_a_z = _cached_group_stat("metric_a_stats_z", vec_a_sad_z, vec_a_hc_z)
        t_b_z, p_b_z = _cached_group_stat("metric_b_stats_z", vec_b_sad_z, vec_b_hc_z)
        t_a_raw_pv, p_a_raw_pv = _cached_group_stat("metric_a_stats_raw_pv", vec_a_sad_raw_pv, vec_a_hc_raw_pv)
        t_b_raw_pv, p_b_raw_pv = _cached_group_stat("metric_b_stats_raw_pv", vec_b_sad_raw_pv, vec_b_hc_raw_pv)
        one_raw = results_12.get("one_sample_stats_raw", {})
        one_z = results_12.get("one_sample_stats_z", {})
        one_pv = results_12.get("one_sample_stats_raw_pv", {})
        p_a_sad_0_raw = _cached_one_sample(one_raw, "p_a_sad", vec_a_sad_raw, "SAD (Dist > 0)")
        p_a_hc_0_raw = _cached_one_sample(one_raw, "p_a_hc", vec_a_hc_raw, "HC  (Dist > 0)")
        p_b_sad_0_raw = _cached_one_sample(one_raw, "p_b_sad", vec_b_sad_raw, "SAD (Dist > 0)")
        p_b_hc_0_raw = _cached_one_sample(one_raw, "p_b_hc", vec_b_hc_raw, "HC  (Dist > 0)")
        p_a_sad_0_z = _cached_one_sample(one_z, "p_a_sad", vec_a_sad_z, "SAD (Dist > 0)")
        p_a_hc_0_z = _cached_one_sample(one_z, "p_a_hc", vec_a_hc_z, "HC  (Dist > 0)")
        p_b_sad_0_z = _cached_one_sample(one_z, "p_b_sad", vec_b_sad_z, "SAD (Dist > 0)")
        p_b_hc_0_z = _cached_one_sample(one_z, "p_b_hc", vec_b_hc_z, "HC  (Dist > 0)")
        p_a_sad_0_raw_pv = _cached_one_sample(one_pv, "p_a_sad", vec_a_sad_raw_pv, "SAD (Dist > 0)")
        p_a_hc_0_raw_pv = _cached_one_sample(one_pv, "p_a_hc", vec_a_hc_raw_pv, "HC  (Dist > 0)")
        p_b_sad_0_raw_pv = _cached_one_sample(one_pv, "p_b_sad", vec_b_sad_raw_pv, "SAD (Dist > 0)")
        p_b_hc_0_raw_pv = _cached_one_sample(one_pv, "p_b_hc", vec_b_hc_raw_pv, "HC  (Dist > 0)")

    print_shock_anchor_summary(results_12.get("shock_anchor_stats", {}))

    # =============================================================================
    # 4. Visualization
    # =============================================================================
    # Plot RAW
    plot_topology(
        rdms_sad_raw,
        rdms_hc_raw,
        vec_a_sad_raw,
        vec_a_hc_raw,
        vec_b_sad_raw,
        vec_b_hc_raw,
        p_a_raw,
        p_b_raw,
        p_a_sad_0_raw,
        p_a_hc_0_raw,
        p_b_sad_0_raw,
        p_b_hc_0_raw,
        title_suffix="Important Features\n (p < 0.05, Raw)",
    )

    # Plot Z-SCORED
    plot_topology(
        rdms_sad_z,
        rdms_hc_z,
        vec_a_sad_z,
        vec_a_hc_z,
        vec_b_sad_z,
        vec_b_hc_z,
        p_a_z,
        p_b_z,
        p_a_sad_0_z,
        p_a_hc_0_z,
        p_b_sad_0_z,
        p_b_hc_0_z,
        title_suffix="Important Features\n (p < 0.05, Z-Scored)",
    )

    # Plot PER-VOXEL RAW
    plot_topology(
        rdms_sad_raw_pv,
        rdms_hc_raw_pv,
        vec_a_sad_raw_pv,
        vec_a_hc_raw_pv,
        vec_b_sad_raw_pv,
        vec_b_hc_raw_pv,
        p_a_raw_pv,
        p_b_raw_pv,
        p_a_sad_0_raw_pv,
        p_a_hc_0_raw_pv,
        p_b_sad_0_raw_pv,
        p_b_hc_0_raw_pv,
        title_suffix="Important Features\n (p < 0.05, Raw, Per-Voxel)",
    )
    # =============================================================================
    # 5. PER-VOXEL (PV) STATISTICAL REPORTING
    # =============================================================================
    from scipy.stats import ttest_1samp

    # Constants for Condition Indices: ["CS-", "CSS", "CSR"]
    I_CS_MINUS, I_CSS, I_CSR = 0, 1, 2 

    def extract_metrics_pv_report(rdms_pv):
        """Slices the 3D Per-Voxel RDM array."""
        m_a = rdms_pv[:, I_CSR, I_CSS]        # Threat vs Safety
        m_b = rdms_pv[:, I_CSS, I_CS_MINUS]   # Safety vs Background
        return m_a, m_b

    # 1. Extract PV Vectors
    vA_sad_pv, vB_sad_pv = extract_metrics_pv_report(results_12["rdms_sad_raw_pv"])
    vA_hc_pv, vB_hc_pv   = extract_metrics_pv_report(results_12["rdms_hc_raw_pv"])

    print("\n" + "="*110)
    print(f"{'INDEX (PER-VOXEL)':<25} | {'GROUP':<5} | {'MEAN':<10} | {'t(vs 0)':<8} | {'p(vs 0)':<8} || {'t(GroupDiff)':<12} | {'p(GroupDiff)':<12}")
    print("-" * 110)

    def report_pv_row(label, sad_vec, hc_vec, group_t_p):
        # Within-Group Existence (One-sample vs 0)
        t_sad_0, p_sad_0 = ttest_1samp(sad_vec, 0)
        t_hc_0, p_hc_0   = ttest_1samp(hc_vec, 0)
    
        # Between-Group Difference (Permutation results)
        t_diff, p_diff = group_t_p
    
        # Print Rows
        print(f"{label:<25} | {'SAD':<5} | {np.mean(sad_vec):<10.6f} | {t_sad_0:<8.3f} | {p_sad_0:<8.4f} || {t_diff:<12.3f} | {p_diff:<12.4f}")
        print(f"{' ':<25} | {'HC':<5} | {np.mean(hc_vec):<10.6f} | {t_hc_0:<8.3f} | {p_hc_0:<8.4f} || {'':<12} | {'' :<12}")
        print("-" * 110)

    # --- Execute PV Reporting ---
    report_pv_row("Threat vs Safety (PV)", vA_sad_pv, vA_hc_pv, results_12["metric_a_stats_raw_pv"])
    report_pv_row("Safety vs Backgr (PV)", vB_sad_pv, vB_hc_pv, results_12["metric_b_stats_raw_pv"])

    print("="*110)
    save_cell_results(12, ['I_CSR', 'I_CSS', 'I_CS_MINUS', 'RDM_CONDITIONS', 'cache_cell10', 'data_subsets', 'importance_mask_permutated', 'importance_scores_permutated', 'meta', 'results_11', 'results_12', 'results_13', 'results_13_2', 'results_14_self', 'results_21', 'results_21_pv', 'results_22', 'results_23', 'results_24', 'results_25', 'strict_cross_phase_results', 'sub_to_meta', 'vA_hc_pv', 'vA_sad_pv', 'vB_hc_pv', 'vB_sad_pv'])


else:
    maybe_load_cell_results(12)

# %% [cell 13]
if cell_active(13):
    # Cell 11: Analysis 1.3 - Dynamic Representational Drift
    print("--- Running Analysis 1.3: Dynamic Representational Drift (I > 0, p < .05) ---")

    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os
    import joblib
    from scipy.stats import ttest_1samp, ttest_ind

    # Constants
    COND_SAFETY_TARGET = "CS-"
    COND_SAFETY_LEARN = "CSS"
    COND_THREAT_LEARN = "CSR"

    # =============================================================================
    # 0. CACHE GATE & DATA SETUP
    # =============================================================================
    # Using your requested naming convention from Cell 11
    cache_cell11 = os.path.join(CHECKPOINT_DIR, "analysis_13_drift.joblib")
    legacy_cache_cell11 = os.path.join(CHECKPOINT_DIR, "cell_11.joblib")
    cache_cell11_load = None
    for candidate in (cache_cell11, legacy_cache_cell11):
        if not os.path.exists(candidate):
            continue
        candidate_payload = joblib.load(candidate)
        if isinstance(candidate_payload, dict) and "df_plot" in candidate_payload:
            candidate_results = candidate_payload.get("results_13", candidate_payload)
            if "shock_target_labels" in candidate_results:
                cache_cell11_load = candidate_payload
                print(f"  [LOAD] Found existing Drift results in {candidate}. Skipping calculation...")
                break
            print(f"  [RECALC] Existing Drift cache in {candidate} lacks shock-target metric; recomputing.")

    if cache_cell11_load is not None:
        results_13 = cache_cell11_load.get("results_13", cache_cell11_load)
        df_plot = cache_cell11_load.get("df_plot", results_13["df_plot"])
    else:
        print(f"\n[Step 0] Significance Mask Setup...")
        # [Internal Note: Ensure you run the Stage 9 Significance Block before this cell]
    
        # 1. Feature Selection: empirical mask with all-positive fallback
        importance_masks, feature_space_13 = get_analysis_feature_masks("Analysis 1.3")
        mask_sad = importance_masks['SAD']
        mask_hc = importance_masks['HC']
        print(f"  > Using selected voxels: SAD={np.sum(mask_sad)}, HC={np.sum(mask_hc)}")

        # 2. Load Phase Data (Beta Maps)
        X_ext_sad, y_ext_sad, sub_ext_sad = get_phase_data("SAD_Placebo", "ext")
        X_ext_hc, y_ext_hc, sub_ext_hc = get_phase_data("HC_Placebo", "ext")
        X_rst_sad, y_rst_sad, sub_rst_sad = get_phase_data("SAD_Placebo", "rst")
        X_rst_hc, y_rst_hc, sub_rst_hc = get_phase_data("HC_Placebo", "rst")

        # Safety Baseline (CS-) recovery
        if 'X_ext' in locals():
            X_global, y_global, sub_global = X_ext, y_ext, sub_ext
        else:
            raise ValueError("Global extinction data missing (X_ext/y_ext/sub_ext). Run Cell 5 before Analysis 1.3.")

        # =============================================================================
        # 2. Execution (Vector Plasticity Calculation)
        # =============================================================================
        # Note: Requires updated Cell 22 function definition (NameError fix)
        print("[Step 2] Computing Projection and Cosine Fidelity Vectors...")
    
        # Safety: CSS(Ext) -> CS-(Ext)
        df_safe_sad = calculate_plasticity_vectors(X_ext_sad, y_ext_sad, sub_ext_sad, X_global, y_global, sub_global, mask_sad, COND_SAFETY_LEARN, COND_SAFETY_TARGET)
        df_safe_hc = calculate_plasticity_vectors(X_ext_hc, y_ext_hc, sub_ext_hc, X_global, y_global, sub_global, mask_hc, COND_SAFETY_LEARN, COND_SAFETY_TARGET)
        # Threat: CSR(Ext) -> CSR(Rst)
        df_threat_sad = calculate_plasticity_vectors(X_ext_sad, y_ext_sad, sub_ext_sad, X_rst_sad, y_rst_sad, sub_rst_sad, mask_sad, COND_THREAT_LEARN, COND_THREAT_LEARN)
        df_threat_hc = calculate_plasticity_vectors(X_ext_hc, y_ext_hc, sub_ext_hc, X_rst_hc, y_rst_hc, sub_rst_hc, mask_hc, COND_THREAT_LEARN, COND_THREAT_LEARN)
        X_shock_sad, y_shock_sad, sub_shock_sad = get_shock_target_data("SAD_Placebo")
        X_shock_hc, y_shock_hc, sub_shock_hc = get_shock_target_data("HC_Placebo")
        if X_shock_sad is not None and X_shock_hc is not None:
            print(f"  > Threat shock-target: using labels {SHOCK_TARGET_LABELS}")
            df_threat_shock_sad = calculate_plasticity_vectors(X_ext_sad, y_ext_sad, sub_ext_sad, X_shock_sad, y_shock_sad, sub_shock_sad, mask_sad, COND_THREAT_LEARN, SHOCK_TARGET_LABELS)
            df_threat_shock_hc = calculate_plasticity_vectors(X_ext_hc, y_ext_hc, sub_ext_hc, X_shock_hc, y_shock_hc, sub_shock_hc, mask_hc, COND_THREAT_LEARN, SHOCK_TARGET_LABELS)
        else:
            print(f"  ! Threat shock-target unavailable: no shock/US target trials found for labels {SHOCK_TARGET_LABELS}.")
            df_threat_shock_sad = pd.DataFrame()
            df_threat_shock_hc = pd.DataFrame()

        # Combine into unified dataset
        plot_frames = [
            tag_df(df_safe_sad, 'SAD', 'Safety'), tag_df(df_safe_hc, 'HC', 'Safety'),
            tag_df(df_threat_sad, 'SAD', 'Threat'), tag_df(df_threat_hc, 'HC', 'Threat')
        ]
        if not df_threat_shock_sad.empty or not df_threat_shock_hc.empty:
            plot_frames.extend([
                tag_df(df_threat_shock_sad, 'SAD', 'Threat Shock Target'),
                tag_df(df_threat_shock_hc, 'HC', 'Threat Shock Target'),
            ])
        df_plot = pd.concat(plot_frames)

        # Store for next time
        drift_summary = (
            df_plot.groupby(["Group", "Condition"])[["projection", "cosine", "init_dist"]]
            .agg(["mean", "sem", "count"])
            .reset_index()
            if not df_plot.empty else pd.DataFrame()
        )
        results_13 = {
            'df_plot': df_plot,
            'drift_summary': drift_summary,
            'primary_metric': 'projection',
            'feature_space': feature_space_13,
            'shock_target_labels': SHOCK_TARGET_LABELS,
            'threat_shock_sad': df_threat_shock_sad,
            'threat_shock_hc': df_threat_shock_hc,
        }
        joblib.dump(results_13, cache_cell11)
        save_checkpoint(13, {"results_13": results_13, "df_plot": df_plot, "drift_summary": drift_summary, "feature_space": feature_space_13})
        save_intermediate("stage13_drift", {"results_13": results_13, "df_plot": df_plot, "drift_summary": drift_summary, "feature_space": feature_space_13})

    # =============================================================================
    # 3. STATISTICS & VISUALIZATION (Updated Layout: 1 Row, 3 Columns)
    # =============================================================================
    if df_plot.empty:
        print("! Error: Dataframe is empty. Vector calculation likely failed.")
    else:
        print(f"\n[Step 3] Analyzing {len(df_plot)} subject vectors...")
    
        sns.set_context("poster")
        # UPDATED: Changed height_ratios for 1 row, adjusted figsize
        fig, axes = plt.subplots(1, 3, figsize=(26, 8), gridspec_kw={'height_ratios': [1]})
    
        # Plot A: Magnitude (Scalar Projection)
        sns.barplot(data=df_plot, x='Condition', y='projection', hue='Group', 
                    palette={'SAD': '#c44e52', 'HC': '#4c72b0'}, ax=axes[0], capsize=.1)
        axes[0].axhline(0, color='k', ls='--')
        axes[0].set_title("Neural Plasticity Magnitude\n(Projection)")
        axes[0].set_ylabel("Plasticity (au)")
    
        # Plot B: Fidelity (Cosine Similarity)
        sns.barplot(data=df_plot, x='Condition', y='cosine', hue='Group', 
                    palette={'SAD': '#c44e52', 'HC': '#4c72b0'}, ax=axes[1], capsize=.1)
        axes[1].axhline(0, color='k', ls='--')
        axes[1].set_title("Representational Fidelity\n(Cosine Similarity)")
        axes[1].set_ylabel("Fidelity (cos)")
    
        # Plot C: Learning vs Initial Distance
        # This scatter plot stays in the new Row 1, Column 3 slot
        sns.scatterplot(data=df_plot, x='init_dist', y='projection', hue='Group', style='Condition', 
                        palette={'SAD': '#c44e52', 'HC': '#4c72b0'}, ax=axes[2], s=150, alpha=0.8, edgecolor='black')
        axes[2].axhline(0, color='k', ls='--')
        axes[2].set_title("Plasticity vs. Initial Distance\n(Voxel Space)")
        axes[2].set_ylabel("Plasticity (au)")
        axes[2].grid(True, linestyle=':', alpha=0.5)

        # FINAL STEP: Print Statistical Summaries (Independent of Cache)
        print("\n--- Statistical Summary (p-values) ---")
        for met in ['projection', 'cosine']:
            for cond in df_plot['Condition'].dropna().unique():
                d_s = df_plot[(df_plot['Condition']==cond) & (df_plot['Group']=='SAD')][met]
                d_h = df_plot[(df_plot['Condition']==cond) & (df_plot['Group']=='HC')][met]
                if len(d_s)>1 and len(d_h)>1:
                    t, p = ttest_ind(d_s, d_h)
                    sig = "*" if p < 0.05 else "ns"
                    print(f"  > Group Diff: {cond} {met} t={t:.3f}, p={p:.4f} {sig}")

        plt.tight_layout()
        # Save the finalized figure to results dir
        plt.savefig(os.path.join(CHECKPOINT_DIR, "stage11_drift_plots_row.png"), dpi=300)
        plt.show()

    print("--- Cell 11 Complete: Visualization reorganized ---")
    save_cell_results(13, ['COND_SAFETY_LEARN', 'COND_SAFETY_TARGET', 'COND_THREAT_LEARN', 'SHOCK_TARGET_LABELS', 'cache_cell11', 'data_subsets', 'df_threat_shock_hc', 'df_threat_shock_sad', 'importance_mask_permutated', 'importance_scores_permutated', 'meta', 'results_11', 'results_12', 'results_13', 'results_13_2', 'results_14_self', 'results_21', 'results_21_pv', 'results_22', 'results_23', 'results_24', 'results_25', 'strict_cross_phase_results', 'sub_to_meta'])


else:
    maybe_load_cell_results(13)

# %% [cell 14]
if cell_active(14):
    # Cell 12: Analysis 1.3 part 2 - Single-Trial Trajectories
    print("--- Running Analysis 1.3 part 2: Single-Trial Trajectories (I > 0, p < .05) ---")

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import joblib
    from numpy.linalg import norm

    # Constants
    COND_SAFETY_TARGET = "CS-"
    COND_SAFETY_LEARN = "CSS"
    COND_THREAT_LEARN = "CSR"
    BLOCK_SIZE = 1  

    # =============================================================================
    # 0. CACHE GATE & DATA SETUP
    # =============================================================================
    cache_cell12_part2 = os.path.join(CHECKPOINT_DIR, "cell_12_trajectories.joblib")

    cache_cell12_payload = None
    if os.path.exists(cache_cell12_part2):
        loaded_trajectory = joblib.load(cache_cell12_part2)
        has_drug_cells = (
            isinstance(loaded_trajectory, dict)
            and all(
                isinstance(loaded_trajectory.get(key), pd.DataFrame)
                and "Drug" in loaded_trajectory.get(key).columns
                and not loaded_trajectory.get(key).empty
                for key in ["data_safe", "data_threat"]
            )
            and isinstance(loaded_trajectory.get("trajectory_slopes"), pd.DataFrame)
            and "Drug" in loaded_trajectory.get("trajectory_slopes").columns
        )
        has_early_to_target_projection = (
            isinstance(loaded_trajectory, dict)
            and loaded_trajectory.get("trajectory_metric") == "early_to_target_normalized_projection"
        )
        if (
            isinstance(loaded_trajectory, dict)
            and "data_threat_shock" in loaded_trajectory
            and has_drug_cells
            and has_early_to_target_projection
        ):
            print(f"  [LOAD] Found existing Trajectory results in {cache_cell12_part2}. Skipping calculation...")
            cache_cell12_payload = loaded_trajectory
        else:
            print(f"  [RECALC] Existing Trajectory cache lacks shock-target, Drug-resolved metrics, or early-to-target projection scores; recomputing.")

    if cache_cell12_payload is not None:
        results_13_2 = cache_cell12_payload
        stats_safe = results_13_2['stats_safe']
        stats_threat = results_13_2['stats_threat']
        stats_threat_shock = results_13_2.get('stats_threat_shock', pd.DataFrame())
        df_safe = results_13_2['data_safe']
        df_threat = results_13_2['data_threat']
        df_threat_shock = results_13_2.get('data_threat_shock', pd.DataFrame())
    else:
        print(f"\n[Step 0] Significance Mask Setup & Calculation...")

        # Logic: empirical mask with all-positive fallback
        importance_masks, feature_space_13b = get_analysis_feature_masks("Analysis 1.3 part 2")
        mask_sad = importance_masks['SAD']
        mask_hc = importance_masks['HC']

        # 1. Load phase data and execute trajectory calculations for every Aim 5 cell.
        print("  Calculating Trajectories for Safety and Threat...")
        if not all(name in globals() for name in ("X_ext", "y_ext", "sub_ext")):
            raise ValueError("Global extinction data missing (X_ext/y_ext/sub_ext). Run or load Cell 3 before Stage 14.")
        ext_full_subsets = process_phase_data(
            X_ext, y_ext, sub_ext, "Extinction full CS",
            keep_conditions=(COND_SAFETY_TARGET, COND_SAFETY_LEARN, COND_THREAT_LEARN),
        )
        trajectory_frames = {"safe": {}, "threat": {}, "threat_shock": {}}
        for group, drug, group_key, group_mask in [
            ("SAD", "Placebo", "SAD_Placebo", mask_sad),
            ("SAD", "Oxytocin", "SAD_Oxytocin", mask_sad),
            ("HC", "Placebo", "HC_Placebo", mask_hc),
            ("HC", "Oxytocin", "HC_Oxytocin", mask_hc),
        ]:
            try:
                X_ext_gd, y_ext_gd, sub_ext_gd = get_phase_data(group_key, "ext")
            except Exception as exc:
                print(f"  ! {group_key}: extinction data unavailable for trajectories ({exc})")
                continue

            try:
                X_rst_gd, y_rst_gd, sub_rst_gd = get_phase_data(group_key, "rst")
            except Exception as exc:
                print(f"  ! {group_key}: reinstatement data unavailable for threat trajectory ({exc})")
                X_rst_gd = y_rst_gd = sub_rst_gd = None

            ext_full_gd = ext_full_subsets.get(group_key)
            if ext_full_gd is None:
                print(f"  ! {group_key}: full extinction data unavailable for safety trajectory")
                df_safe_gd = pd.DataFrame()
            else:
                df_safe_gd = calc_trajectory(
                    ext_full_gd["X"], ext_full_gd["y"], ext_full_gd["sub"],
                    ext_full_gd["X"], ext_full_gd["y"], ext_full_gd["sub"],
                    group_mask, COND_SAFETY_LEARN, COND_SAFETY_TARGET,
                )
            trajectory_frames["safe"][group_key] = (group, drug, df_safe_gd)

            if X_rst_gd is not None:
                df_threat_gd = calc_trajectory(
                    X_ext_gd, y_ext_gd, sub_ext_gd,
                    X_rst_gd, y_rst_gd, sub_rst_gd,
                    group_mask, COND_THREAT_LEARN, COND_THREAT_LEARN,
                )
            else:
                df_threat_gd = pd.DataFrame()
            trajectory_frames["threat"][group_key] = (group, drug, df_threat_gd)

            X_shock_gd, y_shock_gd, sub_shock_gd = get_shock_target_data(group_key)
            if X_shock_gd is not None:
                df_threat_shock_gd = calc_trajectory(
                    X_ext_gd, y_ext_gd, sub_ext_gd,
                    X_shock_gd, y_shock_gd, sub_shock_gd,
                    group_mask, COND_THREAT_LEARN, SHOCK_TARGET_LABELS,
                )
            else:
                df_threat_shock_gd = pd.DataFrame()
            trajectory_frames["threat_shock"][group_key] = (group, drug, df_threat_shock_gd)

        df_safe_sad = trajectory_frames["safe"].get("SAD_Placebo", ("SAD", "Placebo", pd.DataFrame()))[2]
        df_safe_hc = trajectory_frames["safe"].get("HC_Placebo", ("HC", "Placebo", pd.DataFrame()))[2]
        df_threat_sad = trajectory_frames["threat"].get("SAD_Placebo", ("SAD", "Placebo", pd.DataFrame()))[2]
        df_threat_hc = trajectory_frames["threat"].get("HC_Placebo", ("HC", "Placebo", pd.DataFrame()))[2]
        df_threat_shock_sad = trajectory_frames["threat_shock"].get("SAD_Placebo", ("SAD", "Placebo", pd.DataFrame()))[2]
        df_threat_shock_hc = trajectory_frames["threat_shock"].get("HC_Placebo", ("HC", "Placebo", pd.DataFrame()))[2]

        # 3. Calculate Statistics
        print("  Calculating Statistics...")
        stats_safe = run_detailed_stats(df_safe_sad, df_safe_hc, "Safety Learning")
        stats_threat = run_detailed_stats(df_threat_sad, df_threat_hc, "Threat Maintenance")
        stats_threat_shock = run_detailed_stats(df_threat_shock_sad, df_threat_shock_hc, "Threat Shock Target")

        # 4. Prepare Plotting Data
        df_safe = prepare_group_drug_plot(trajectory_frames["safe"].values(), "Safety Learning")
        df_threat = prepare_group_drug_plot(trajectory_frames["threat"].values(), "Threat Maintenance")
        df_threat_shock = prepare_group_drug_plot(trajectory_frames["threat_shock"].values(), "Threat Shock Target")

        def subject_trajectory_slopes(df, domain):
            rows = []
            if df is None or df.empty:
                return pd.DataFrame(columns=["sub", "Group", "Drug", "Condition", "slope", "mean_score"])
            group_cols = ["sub", "Group"]
            if "Drug" in df.columns:
                group_cols.append("Drug")
            for keys, sub_df in df.groupby(group_cols):
                if len(group_cols) == 3:
                    sub, group, drug = keys
                else:
                    sub, group = keys
                    drug = pd.NA
                sub_df = sub_df.sort_values("trial")
                if len(sub_df) < 3:
                    continue
                slope, _ = np.polyfit(sub_df["trial"], sub_df["score"], 1)
                rows.append({
                    "sub": sub,
                    "Group": group,
                    "Drug": drug,
                    "Condition": domain,
                    "slope": slope,
                    "mean_score": sub_df["score"].mean(),
                })
            return pd.DataFrame(rows)

        trajectory_slopes = pd.concat([
            subject_trajectory_slopes(df_safe, "Safety Learning"),
            subject_trajectory_slopes(df_threat, "Threat Maintenance"),
            subject_trajectory_slopes(df_threat_shock, "Threat Shock Target"),
        ], ignore_index=True)

        # Save to Cache
        results_13_2 = {
            'stats_safe': stats_safe, 
            'stats_threat': stats_threat,
            'stats_threat_shock': stats_threat_shock,
            'data_safe': df_safe,
            'data_threat': df_threat,
            'data_threat_shock': df_threat_shock,
            'trajectory_slopes': trajectory_slopes,
            'primary_metric': 'safety_trajectory_slope',
            'trajectory_metric': 'early_to_target_normalized_projection',
            'feature_space': feature_space_13b,
            'shock_target_labels': SHOCK_TARGET_LABELS,
        }
        joblib.dump(results_13_2, cache_cell12_part2)
        save_checkpoint(14, {"results_13_2": results_13_2, "feature_space": feature_space_13b})
        save_intermediate("stage14_trajectories", {"results_13_2": results_13_2, "feature_space": feature_space_13b})

    export_aim2_trajectory_panel(results_13_2)

    # =============================================================================
    # 4. VISUALIZATION (Always Runs)
    # =============================================================================
    if df_safe.empty and df_threat.empty and df_threat_shock.empty:
        print("! No data to plot.")
    else:
        sns.set_context("poster")
        fig, axes = plt.subplots(1, 3, figsize=(30, 9), sharey=True)
    
        # 1. Safety Plot
        if not df_safe.empty:
            sns.lineplot(data=df_safe, x='trial', y='score', hue='Group', 
                         palette={'SAD': '#c44e52', 'HC': '#4c72b0'}, 
                         lw=3, marker="o", err_style="band", ax=axes[0])
            axes[0].set_title("A. Safety Trajectory\nCSS projection toward CS- centroid")
            axes[0].set_ylabel("Similarity score (0=start, 1=target)")
            axes[0].axhline(0, color='gray', ls='--', label='Start')
            axes[0].axhline(1, color='#2ca02c', ls='-', lw=2, label='Target')
            axes[0].legend(loc='upper left')

        # 2. Threat Plot
        if not df_threat.empty:
            sns.lineplot(data=df_threat, x='trial', y='score', hue='Group', 
                         palette={'SAD': '#c44e52', 'HC': '#4c72b0'}, 
                         lw=3, marker="s", err_style="band", ax=axes[1])
            axes[1].set_title("B. Threat Maintenance\nCSR projection toward CSR centroid")
            axes[1].set_xlabel(f"Trial (Block Size: {BLOCK_SIZE})")
            axes[1].axhline(0, color='gray', ls='--', label='Start')
            axes[1].axhline(1, color='#d62728', ls='-', lw=2, label='Target')
            axes[1].legend(loc='upper left')
        else:
            axes[1].axis('off')

        # 3. Threat Shock-Target Plot
        if not df_threat_shock.empty:
            sns.lineplot(data=df_threat_shock, x='trial', y='score', hue='Group',
                         palette={'SAD': '#c44e52', 'HC': '#4c72b0'},
                         lw=3, marker="^", err_style="band", ax=axes[2])
            axes[2].set_title("C. Threat Acquisition\n(Target = Shock/US)")
            axes[2].set_xlabel(f"Trial (Block Size: {BLOCK_SIZE})")
            axes[2].axhline(0, color='gray', ls='--', label='Start')
            axes[2].axhline(1, color='#8c1d40', ls='-', lw=2, label='Target')
            axes[2].legend(loc='upper left')
        else:
            axes[2].axis('off')
    
        plt.tight_layout()
        # Save high-res figure to global results
        plt.savefig(os.path.join(CHECKPOINT_DIR, "stage12_trajectories_plot.png"), dpi=300)
        plt.show()

    print("--- Cell 12 Complete: Persistent Trajectories stored ---")
    save_cell_results(14, ['BLOCK_SIZE', 'COND_SAFETY_LEARN', 'COND_SAFETY_TARGET', 'COND_THREAT_LEARN', 'SHOCK_TARGET_LABELS', 'cache_cell12_part2', 'data_subsets', 'df_threat_shock', 'importance_mask_permutated', 'importance_scores_permutated', 'meta', 'results_11', 'results_12', 'results_13', 'results_13_2', 'results_14_self', 'results_21', 'results_21_pv', 'results_22', 'results_23', 'results_24', 'results_25', 'stats_threat_shock', 'strict_cross_phase_results', 'sub_to_meta'])


else:
    maybe_load_cell_results(14)

# %% [cell 15]
if cell_active(15):
    # Cell 13: Analysis 1.4 - Decision Boundary Characteristics
    print("--- Running Analysis 1.4: Self-Network Statistics (Optimized C) ---")

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    import joblib
    from scipy.stats import entropy, kurtosis, ttest_ind, ks_2samp

    # =============================================================================
    # 0. CACHE GATE
    # =============================================================================
    cache_cell13 = os.path.join(CHECKPOINT_DIR, "cell_13_decision_stats_opt.joblib")

    if os.path.exists(cache_cell13): 
        print(f"  [LOAD] Found existing optimized results in {cache_cell13}.")
        results_14_self = joblib.load(cache_cell13)
        df_sad_stats = results_14_self['df_sad']
        df_hc_stats = results_14_self['df_hc']
        required_decision_cols = {'decision_margin_css', 'decision_margin_csr', 'entropy_css', 'entropy_csr', 'p_csr_css', 'p_csr_csr', 'boundary_separation'}
        decision_definition = results_14_self.get('decision_metric_definition')
        if (
            decision_definition != 'heldout_decoder_probability_v1'
            or not required_decision_cols.issubset(df_sad_stats.columns)
            or not required_decision_cols.issubset(df_hc_stats.columns)
        ):
            print("  [RECALC] Cached decision stats are stale or incomplete; recomputing.")
            del results_14_self, df_sad_stats, df_hc_stats
    if 'df_sad_stats' not in locals():
        print(f"  [CALC] Calculating Decision Stats using Optimized C...")
        # Retrieve optimized C from your Cell 6 results
        c_sad = res_sad['model_sad'].named_steps['classification'].C
        c_hc = res_hc['model_hc'].named_steps['classification'].C

        def get_stats_calibrated(data_ptr, mask, opt_c):
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import LeaveOneGroupOut
        
            X, y, subs = data_ptr["X"], data_ptr["y"], data_ptr["sub"]
            u_subs = np.unique(subs)
            rows = []
        
            for sub in u_subs:
                m = (subs == sub) & np.isin(y, ["CSR", "CSS"])
                m_proto = (subs == sub) & np.isin(y, ["CS-", "CSS", "CSR"])
                if np.sum(m) < 8: continue
            
                xs, ys = X[m][:, mask], y[m]
                xs_proto, ys_proto = X[m_proto][:, mask], y[m_proto]
                # Leave-One-Trial-Out ensures we don't overfit to the test trial
                logo = LeaveOneGroupOut()
                probs = []
                prob_labels = []
            
                for train, test in logo.split(xs, ys, groups=np.arange(len(ys))):
                    clf = LogisticRegression(C=opt_c, solver='liblinear', random_state=42)
                    clf.fit(xs[train], ys[train])
                
                    csr_idx = np.where(clf.classes_ == "CSR")[0][0]
                    probs.append(clf.predict_proba(xs[test])[0, csr_idx])
                    prob_labels.append(ys[test][0])
            
                probs = np.array(probs)
                prob_labels = np.array(prob_labels)
                probs_css = probs[prob_labels == "CSS"]
                probs_csr = probs[prob_labels == "CSR"]
                if len(probs_css) == 0 or len(probs_csr) == 0:
                    continue
                p_proto_csr_css, p_proto_csr_csr, proto_boundary_separation = prototype_decision_metrics(
                    xs_proto, ys_proto, "CSR", "CSS"
                )
                p_csr_css = float(np.mean(probs_css))
                p_csr_csr = float(np.mean(probs_csr))
                boundary_separation = p_csr_csr - p_csr_css
                # 20-bin histogram for Shannon Entropy
                hist, _ = np.histogram(probs_css, bins=20, range=(0, 1), density=True)
                ent_val = entropy(hist + 1e-9)
                kurt_val = kurtosis(probs_css, fisher=True)
            
                rows.append({
                    'sub': sub,
                    'entropy': ent_val,
                    'kurtosis': kurt_val,
                    'variance': np.var(probs_css),
                    'probabilities': probs_css,
                    'probabilities_csr': probs_csr,
                    'p_csr_css': p_csr_css,
                    'p_csr_csr': p_csr_csr,
                    'boundary_separation': boundary_separation,
                    'prototype_p_csr_css': p_proto_csr_css,
                    'prototype_p_csr_csr': p_proto_csr_csr,
                    'prototype_boundary_separation': proto_boundary_separation,
                    'decision_margin_css': float(0.5 - p_csr_css) if np.isfinite(p_csr_css) else np.nan,
                    'decision_margin_csr': float(p_csr_csr - 0.5) if np.isfinite(p_csr_csr) else np.nan,
                    'decision_margin_all': float(np.mean(np.abs(probs - 0.5))),
                    'entropy_css': float(np.mean([entropy([p, 1 - p], base=2) for p in np.clip(probs_css, 1e-9, 1 - 1e-9)])),
                    'entropy_csr': float(np.mean([entropy([p, 1 - p], base=2) for p in np.clip(probs_csr, 1e-9, 1 - 1e-9)])),
                })
            return pd.DataFrame(rows)

        decision_masks, feature_space_14 = get_analysis_feature_masks("Analysis 1.4")
        df_sad_stats = get_stats_calibrated(data_subsets["SAD_Placebo"]["ext"], decision_masks['SAD'], c_sad)
        df_hc_stats = get_stats_calibrated(data_subsets["HC_Placebo"]["ext"], decision_masks['HC'], c_hc)

        results_14_self = {
            'df_sad': df_sad_stats,
            'df_hc': df_hc_stats,
            'feature_space': feature_space_14,
            'decision_metric_definition': 'heldout_decoder_probability_v1',
        }
        joblib.dump(results_14_self, cache_cell13)
        save_checkpoint(15, {"results_14_self": results_14_self, "feature_space": feature_space_14})
        save_intermediate("stage15_decision_stats", {"results_14_self": results_14_self, "feature_space": feature_space_14})

    # =============================================================================
    # 4. STATISTICS & VISUALIZATION
    # =============================================================================
    # A. Metric Tests (Subject-wise)
    t_ent, p_ent = ttest_ind(df_sad_stats['entropy'], df_hc_stats['entropy'])
    t_kurt, p_kurt = ttest_ind(df_sad_stats['kurtosis'], df_hc_stats['kurtosis'])
    t_margin, p_margin = ttest_ind(df_sad_stats['decision_margin_css'], df_hc_stats['decision_margin_css'])
    t_pcsr_css, p_pcsr_css = ttest_ind(df_sad_stats['p_csr_css'], df_hc_stats['p_csr_css'])
    t_boundary, p_boundary = ttest_ind(df_sad_stats['boundary_separation'], df_hc_stats['boundary_separation'])

    # B. Distribution Test (Trial-wise KS Test)
    all_p_sad = np.concatenate([p for p in df_sad_stats['probabilities'].values if len(p) > 0])
    all_p_hc = np.concatenate([p for p in df_hc_stats['probabilities'].values if len(p) > 0])
    ks_stat, p_ks = ks_2samp(all_p_sad, all_p_hc)

    print("\n" + "="*45)
    print("  STATISTICAL SUMMARY: SAD vs HC")
    print("="*45)
    print(f"  Entropy (Uncertainty): t = {t_ent:.4f}, p = {p_ent:.4f}")
    print(f"  Kurtosis (Sharpness):  t = {t_kurt:.4f}, p = {p_kurt:.4f}")
    print(f"  Decision Margin CSS:   t = {t_margin:.4f}, p = {p_margin:.4f}")
    print(f"  P(CSR|CSS):            t = {t_pcsr_css:.4f}, p = {p_pcsr_css:.4f}")
    print(f"  Boundary Separation:   t = {t_boundary:.4f}, p = {p_boundary:.4f}")
    print(f"  KS Test (Distribution): D = {ks_stat:.4f}, p = {p_ks:.4f}")
    print("="*45 + "\n")

    sns.set_context("poster")
    fig, axes = plt.subplots(2, 3, figsize=(30, 14))
    axes = axes.ravel()
    df_plot = pd.concat([df_sad_stats.assign(G='SAD'), df_hc_stats.assign(G='HC')])

    # Panel 1: Entropy
    sns.violinplot(data=df_plot, x='G', y='entropy', hue='G', legend=False, ax=axes[0])
    axes[0].set_title(f"Uncertainty (Entropy)\np={p_ent:.3f}")

    # Panel 2: Kurtosis
    sns.boxplot(data=df_plot, x='G', y='kurtosis', hue='G', legend=False, ax=axes[1])
    axes[1].set_title(f"Sharpness (Kurtosis)\np={p_kurt:.3f}")

    # Panel 3: Probability Density
    sns.kdeplot(all_p_sad, fill=True, label='SAD', ax=axes[2], bw_adjust=1.0)
    sns.kdeplot(all_p_hc, fill=True, label='HC', ax=axes[2], bw_adjust=1.0)
    axes[2].set_title(f"Neural Decision Density\nKS p={p_ks:.4f}")
    axes[2].set_xlabel("P(Threat)")
    axes[2].set_xlim(0, 1)
    axes[2].legend()

    sns.violinplot(data=df_plot, x='G', y='decision_margin_css', hue='G', legend=False, ax=axes[3])
    axes[3].set_title(f"CSS Decision Margin\np={p_margin:.3f}")
    axes[3].set_ylabel("|P(CSR|CSS) - 0.5|")

    sns.violinplot(data=df_plot, x='G', y='p_csr_css', hue='G', legend=False, ax=axes[4])
    axes[4].axhline(0.5, color='gray', linestyle='--', alpha=0.6)
    axes[4].set_title(f"Threat-Like Safety\np={p_pcsr_css:.3f}")
    axes[4].set_ylabel("Mean P(CSR | CSS)")

    sns.violinplot(data=df_plot, x='G', y='boundary_separation', hue='G', legend=False, ax=axes[5])
    axes[5].axhline(0, color='gray', linestyle='--', alpha=0.6)
    axes[5].set_title(f"Boundary Separation\np={p_boundary:.3f}")
    axes[5].set_ylabel("P(CSR|CSR) - P(CSR|CSS)")

    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_DIR, "stage13_decision_final_stats.png"), dpi=300)
    plt.show()

    print("--- Cell 13 Complete ---")
    save_cell_results(15, ['all_p_hc', 'all_p_sad', 'axes', 'cache_cell13', 'data_subsets', 'df_plot', 'fig', 'importance_mask_permutated', 'importance_scores_permutated', 'ks_stat', 'meta', 'p_ent', 'p_ks', 'p_kurt', 'results_11', 'results_12', 'results_13', 'results_13_2', 'results_14_self', 'results_21', 'results_21_pv', 'results_22', 'results_23', 'results_24', 'results_25', 'strict_cross_phase_results', 'sub_to_meta', 't_ent', 't_kurt'])


else:
    maybe_load_cell_results(15)

# %% [cell 16]
if cell_active(16):
    # Cell 14: Analysis 2.1 - Safety Restoration & Threat Discrimination (Mixed Effects - PV Normalized)
    # Objective: Test if Oxytocin rescues network topology in SAD using Per-Voxel normalization.

    print("--- Running Analysis 2.1: Safety Restoration & Threat Discrimination (LME | Per-Voxel Normalized) ---")

    # Constants
    COND_SAFE_LEARN = "CSS"
    COND_SAFE_BASE  = "CS-"
    COND_THREAT     = "CSR"

    # =============================================================================
    # 0. Validate Masks & Constants from Cell 9/10
    # =============================================================================
    importance_masks, feature_space_21_pv = get_analysis_feature_masks("Analysis 2.1 per-voxel")
    mask_sad = importance_masks['SAD']
    mask_hc = importance_masks['HC']

    # Get voxel counts for Per-Voxel (PV) normalization
    n_vox_sad = np.sum(mask_sad)
    n_vox_hc = np.sum(mask_hc)

    print(f"  > Normalizing by Voxel Counts: SAD={n_vox_sad}, HC={n_vox_hc}")

    # =============================================================================
    # 1. Calculate PV-Normalized Distances
    # =============================================================================
    subgroups_21 = {"SAD_Placebo": [], "SAD_Oxytocin": [], "HC_Placebo": [], "HC_Oxytocin": []}

    if 'sub_to_meta' not in locals():
        if 'meta' in locals():
            sub_to_meta = meta.set_index("subject_id")[["Group", "Drug"]].to_dict('index')
        else:
            raise ValueError("Metadata not found.")

    for sub in np.unique(sub_ext):
        s_str = str(sub).strip()
        # Handle various subject string formats
        info = sub_to_meta.get(s_str) or sub_to_meta.get(f"sub-{s_str}")
        if not info: continue

        key = f"{info['Group']}_{info['Drug']}"
        if key in subgroups_21: subgroups_21[key].append(sub)

    data_rows = []
    print("  > Calculating Per-Voxel (PV) Metrics...")

    for key, subject_list in subgroups_21.items():
        group, drug = key.split('_')
    
        # Select Native Mask and PV Scale Factor
        current_mask = mask_sad if group == "SAD" else mask_hc
        n_feat = n_vox_sad if group == "SAD" else n_vox_hc
        
        for sub in subject_list:
            mask_sub = (sub_ext == sub)
            if not np.any(mask_sub): continue
        
            X_sub = X_ext[mask_sub][:, current_mask]
            y_sub = y_ext[mask_sub]
        
            # Extract Prototypes (Centroids)
            idx_css = (y_sub == COND_SAFE_LEARN)
            idx_cs_ = (y_sub == COND_SAFE_BASE)
            idx_csr = (y_sub == COND_THREAT)
        
            # Requirement: At least 1 trial per condition for centroid calculation
            if not (np.any(idx_css) and np.any(idx_cs_) and np.any(idx_csr)): continue
        
            p_css = np.mean(X_sub[idx_css], axis=0).reshape(1, -1)
            p_cs_ = np.mean(X_sub[idx_cs_], axis=0).reshape(1, -1)
            p_csr = np.mean(X_sub[idx_csr], axis=0).reshape(1, -1)
        
            # 1. Calculate Raw Correlation Distance
            raw_dist_safety = cdist(p_css, p_cs_, metric='correlation')[0][0]
            raw_dist_threat = cdist(p_csr, p_css, metric='correlation')[0][0]
            raw_dist_threat_background = cdist(p_csr, p_cs_, metric='correlation')[0][0]
        
            # 2. PV Normalization (Metric / Number of Voxels)
            # This matches the logic used in Cell 10 for static topology
            pv_dist_safety = raw_dist_safety / n_feat
            pv_dist_threat = raw_dist_threat / n_feat
            pv_dist_threat_background = raw_dist_threat_background / n_feat
            
            data_rows.append({
                "Subject": sub, "Group": group, "Drug": drug, "Condition": key,
                "Dist_Safety_PV": pv_dist_safety,
                "Dist_Threat_PV": pv_dist_threat,
                "Dist_Threat_Background_PV": pv_dist_threat_background
            })

    df_topo_pv = pd.DataFrame(data_rows)
    print(f"  > Computed PV metrics for {len(df_topo_pv)} subjects.")

    # =============================================================================
    # 2. Statistical Tests (Linear Mixed Effects)
    # =============================================================================
    print("\n[Step 2] Testing for Interaction (Mixed Effects on PV Metrics)...")
    form_base = "~ C(Group, Treatment(reference='HC')) * C(Drug, Treatment(reference='Placebo'))"

    # Test 1: Safety Restoration (PV)
    p_int_safe = run_lme("Dist_Safety_PV " + form_base, df_topo_pv, "Metric 1: Safety Restoration (PV)")

    # Test 2: Threat Discrimination (PV)
    p_int_threat = run_lme("Dist_Threat_PV " + form_base, df_topo_pv, "Metric 2: Threat Discrimination (PV)")

    # =============================================================================
    # 3. Visualization
    # =============================================================================
    sns.set_context("poster")
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    pal_group = {'SAD': '#c44e52', 'HC': '#4c72b0'}

    # Plotting Function to keep code clean
    def plot_pv_metric(ax, y_col, title, ylabel, p_val):
        sns.pointplot(data=df_topo_pv, x='Drug', y=y_col, hue='Group', 
                      palette=pal_group, order=['Placebo', 'Oxytocin'], hue_order=['SAD', 'HC'],
                      dodge=0.2, markers=['o', 's'], capsize=0.1, ax=ax)
        sns.stripplot(data=df_topo_pv, x='Drug', y=y_col, hue='Group', 
                      palette=pal_group, order=['Placebo', 'Oxytocin'], hue_order=['SAD', 'HC'],
                      dodge=True, alpha=0.3, jitter=True, legend=False, ax=ax)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if p_val < 0.05:
            ax.text(0.5, 0.95, f"Interaction: p={p_val:.3f}*", transform=ax.transAxes, ha='center', color='red', fontweight='bold')
        else:
            ax.text(0.5, 0.95, f"Interaction: p={p_val:.3f}", transform=ax.transAxes, ha='center')

    plot_pv_metric(axes[0], 'Dist_Safety_PV', "A. Safety Restoration (PV)\n(CSS vs CS-)", "PV Correlation Dist (Lower = Better)", p_int_safe)
    plot_pv_metric(axes[1], 'Dist_Threat_PV', "B. Threat Discrimination (PV)\n(CSR vs CSS)", "PV Correlation Dist (Higher = Better)", p_int_threat)

    plt.tight_layout()
    plt.show()

    results_21_pv = {'df': df_topo_pv, 'p_safe': p_int_safe, 'p_threat': p_int_threat}
    save_cell_results(16, ['COND_SAFE_BASE', 'COND_SAFE_LEARN', 'COND_THREAT', 'axes', 'data_rows', 'data_subsets', 'df_topo_pv', 'fig', 'form_base', 'importance_mask_permutated', 'importance_scores_permutated', 'key', 'mask_hc', 'mask_sad', 'meta', 'n_vox_hc', 'n_vox_sad', 'p_int_safe', 'p_int_threat', 'pal_group', 'results_11', 'results_12', 'results_13', 'results_13_2', 'results_14_self', 'results_21', 'results_21_pv', 'results_22', 'results_23', 'results_24', 'results_25', 'strict_cross_phase_results', 'sub', 'sub_to_meta', 'subgroups_21', 'subject_list'])


else:
    maybe_load_cell_results(16)

# %% [cell 17]
if cell_active(17):
    # Cell 14: Analysis 2.1 - Safety Restoration & Threat Discrimination (Mixed Effects)
    # Objective: Test if Oxytocin rescues network topology in SAD.
    # Metrics:
    #   1. Safety Restoration: Dist(CSS, CS-) -> Should DECREASE (Return to baseline).
    #   2. Threat Discrimination: Dist(CSR, CSS) -> Should INCREASE (Better separation).
    # Statistical Model: Linear Mixed Effects (LME)
    #   Formula: Metric ~ Group * Drug
    #   Random Effect: 1 | Subject (Implicitly handles variance if repeated measures exist)

    print("--- Running Analysis 2.1: Safety Restoration & Threat Discrimination (LME) ---")

    # Constants
    COND_SAFE_LEARN = "CSS"
    COND_SAFE_BASE  = "CS-"
    COND_THREAT     = "CSR"

    # =============================================================================
    # 0. Validate Masks from Cell 9
    # =============================================================================
    if 'mask_sad_analysis' not in locals() or 'mask_hc_analysis' not in locals():
        importance_masks, feature_space_21 = get_analysis_feature_masks("Analysis 2.1")
        mask_sad_analysis = importance_masks['SAD']
        mask_hc_analysis = importance_masks['HC']
        mask_sad_top5 = mask_sad_analysis
        mask_hc_top5 = mask_hc_analysis

    # =============================================================================
    # 1. Calculate Distances (Both Metrics)
    # =============================================================================
    subgroups_21 = {"SAD_Placebo": [], "SAD_Oxytocin": [], "HC_Placebo": [], "HC_Oxytocin": []}

    # Link subjects to groups
    if 'sub_to_meta' not in locals():
        if 'meta' in locals():
            sub_to_meta = meta.set_index("subject_id")[["Group", "Drug"]].to_dict('index')
        else:
            raise ValueError("Metadata not found.")

    for sub in np.unique(sub_ext):
        s_str = str(sub).strip()
        if s_str in sub_to_meta: info = sub_to_meta[s_str]
        elif f"sub-{s_str}" in sub_to_meta: info = sub_to_meta[f"sub-{s_str}"]
        else: continue

        key = f"{info['Group']}_{info['Drug']}"
        if key in subgroups_21: subgroups_21[key].append(sub)

    data_rows = []
    print("  > Calculating distances (Metric A & Metric B)...")

    for key, subject_list in subgroups_21.items():
        group, drug = key.split('_')
    
        # Select Native Mask
        current_mask = mask_sad_analysis if group == "SAD" else mask_hc_analysis
        
        for sub in subject_list:
            mask_sub = (sub_ext == sub)
            X_sub = X_ext[mask_sub]
            y_sub = y_ext[mask_sub]
        
            # Apply Mask & Center
            X_masked = X_sub[:, current_mask]
        
            # Extract Prototypes
            idx_css = (y_sub == COND_SAFE_LEARN)
            idx_cs_ = (y_sub == COND_SAFE_BASE)
            idx_csr = (y_sub == COND_THREAT)
        
            if np.sum(idx_css) < 3 or np.sum(idx_cs_) < 3 or np.sum(idx_csr) < 3: continue
        
            p_css = np.mean(X_masked[idx_css], axis=0).reshape(1, -1)
            p_cs_ = np.mean(X_masked[idx_cs_], axis=0).reshape(1, -1)
            p_csr = np.mean(X_masked[idx_csr], axis=0).reshape(1, -1)
        
            # Metric 1: Safety Restoration (CSS vs CS-)
            dist_safety = cdist(p_css, p_cs_, metric='correlation')[0][0]
        
            # Metric 2: Threat Discrimination (CSR vs CSS)
            dist_threat = cdist(p_csr, p_css, metric='correlation')[0][0]
            dist_threat_background = cdist(p_csr, p_cs_, metric='correlation')[0][0]
            
            data_rows.append({
                "Subject": sub, "Group": group, "Drug": drug, "Condition": key,
                "Dist_Safety": dist_safety,
                "Dist_Threat": dist_threat,
                "Dist_Threat_Background": dist_threat_background
            })

    shock_anchor_rows = []
    for key, subject_list in subgroups_21.items():
        group, drug = key.split('_')
        current_mask = mask_sad_analysis if group == "SAD" else mask_hc_analysis
        X_cs, y_cs, sub_cs = get_ext_data(key)
        X_shock, y_shock, sub_shock = get_shock_target_data(key)
        if X_shock is None:
            continue
        shock_anchor_group = calculate_residualized_shock_anchor_projection(
            X_cs[:, current_mask],
            y_cs,
            sub_cs,
            X_shock[:, current_mask],
            y_shock,
            sub_shock,
        )
        if shock_anchor_group.empty:
            continue
        shock_anchor_group["Group"] = group
        shock_anchor_group["Drug"] = drug
        shock_anchor_rows.append(shock_anchor_group)

    df_topo = pd.DataFrame(data_rows)
    df_shock_anchor_all_drug = pd.concat(shock_anchor_rows, ignore_index=True) if shock_anchor_rows else pd.DataFrame()
    print(f"  > Computed metrics for {len(df_topo)} subjects.")
    print(f"  > Computed all-drug residualized shock-anchor metrics for {len(df_shock_anchor_all_drug)} subjects.")

    # =============================================================================
    # 2. Statistical Tests (Linear Mixed Effects)
    # =============================================================================
    print("\n[Step 2] Testing for Interaction (Mixed Effects)...")
    # Formula: Metric ~ Group * Drug
    # We set references explicitly: Group=HC, Drug=Placebo
    form_base = "~ C(Group, Treatment(reference='HC')) * C(Drug, Treatment(reference='Placebo'))"

    # Test 1: Safety Restoration
    p_int_safe = run_lme("Dist_Safety " + form_base, df_topo, "Metric 1: Safety Restoration (CSS - CS-)")

    # Test 2: Threat Discrimination
    p_int_threat = run_lme("Dist_Threat " + form_base, df_topo, "Metric 2: Threat Discrimination (CSR - CSS)")

    # =============================================================================
    # 3. Visualization
    # =============================================================================
    sns.set_context("poster")
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    pal_group = {'SAD': '#c44e52', 'HC': '#4c72b0'}

    # Plot A: Safety Restoration
    sns.pointplot(data=df_topo, x='Drug', y='Dist_Safety', hue='Group', 
                  palette=pal_group, order=['Placebo', 'Oxytocin'], hue_order=['SAD', 'HC'],
                  dodge=0.2, markers=['o', 's'], capsize=0.1, ax=axes[0])
    sns.stripplot(data=df_topo, x='Drug', y='Dist_Safety', hue='Group', 
                  palette=pal_group, order=['Placebo', 'Oxytocin'], hue_order=['SAD', 'HC'],
                  dodge=True, alpha=0.4, jitter=True, legend=False, ax=axes[0])

    axes[0].set_title("A. Safety Restoration\n(CSS vs CS-)")
    axes[0].set_ylabel("Correlation Distance (Lower = Better)")
    if p_int_safe < 0.05:
        axes[0].text(0.5, 0.95, f"Interaction: p={p_int_safe:.3f}", transform=axes[0].transAxes, ha='center', fontweight='bold')

    # Plot B: Threat Discrimination
    sns.pointplot(data=df_topo, x='Drug', y='Dist_Threat', hue='Group', 
                  palette=pal_group, order=['Placebo', 'Oxytocin'], hue_order=['SAD', 'HC'],
                  dodge=0.2, markers=['o', 's'], capsize=0.1, ax=axes[1])
    sns.stripplot(data=df_topo, x='Drug', y='Dist_Threat', hue='Group', 
                  palette=pal_group, order=['Placebo', 'Oxytocin'], hue_order=['SAD', 'HC'],
                  dodge=True, alpha=0.4, jitter=True, legend=False, ax=axes[1])

    axes[1].set_title("B. Threat Discrimination\n(CSR vs CSS)")
    axes[1].set_ylabel("Correlation Distance (Higher = Better)")
    if p_int_threat < 0.05:
        axes[1].text(0.5, 0.95, f"Interaction: p={p_int_threat:.3f}", transform=axes[1].transAxes, ha='center', fontweight='bold')

    plt.tight_layout()
    plt.show()

    results_21 = {'df': df_topo, 'shock_anchor_all_drug': df_shock_anchor_all_drug, 'p_safe': p_int_safe, 'p_threat': p_int_threat}
    save_cell_results(17, ['COND_SAFE_BASE', 'COND_SAFE_LEARN', 'COND_THREAT', 'axes', 'data_rows', 'data_subsets', 'df_shock_anchor_all_drug', 'df_topo', 'fig', 'form_base', 'importance_mask_permutated', 'importance_scores_permutated', 'key', 'meta', 'p_int_safe', 'p_int_threat', 'pal_group', 'results_11', 'results_12', 'results_13', 'results_13_2', 'results_14_self', 'results_21', 'results_21_pv', 'results_22', 'results_23', 'results_24', 'results_25', 'shock_anchor_rows', 'strict_cross_phase_results', 'sub', 'sub_to_meta', 'subgroups_21', 'subject_list'])


else:
    maybe_load_cell_results(17)

# %% [cell 18]
if cell_active(18):
    # Cell 15: Analysis 2.2 - Drift Efficiency (Safety & Threat Maintenance)
    # Objective: Test OXT effect on neural drift efficiency in the empirical feature network.
    # Domains:
    #   1. Safety Learning:    CSS(Ext) -> CS-(Ext)
    #   2. Threat Maintenance: CSR(Ext) -> CSR(Reinst)
    # Stats: Linear Mixed Effects (LME)
    # Visualization: Line plots (Means ± SEM)

    print("--- Running Analysis 2.2: Drift Efficiency (Means ± SEM) ---")

    # Constants
    COND_SAFE_TGT = "CS-"
    COND_SAFE_LRN = "CSS"
    COND_THREAT_LRN = "CSR"
    PERCENTILE_THRESH = None  # Uses empirical permutation masks.
    SHOCK_TARGET_DOMAIN = "Threat Shock Target"

    # =============================================================================
    # 0. Setup: Masks & Data Loading
    # =============================================================================
    print(f"\n[Step 0] Setup & Data Loading...")

    # Reinstatement data source from Cell 5 (X_rein/y_rein/sub_rein)
    if 'X_rein' not in locals() or X_rein is None:
        raise ValueError("Reinstatement data missing (X_rein). Run Cell 5.")
    X_rst_all = X_rein
    y_rst_all = y_rein
    sub_rst_all = sub_rein

    # Ensure core masks exist
    if 'mask_sad_core' not in locals() or 'mask_hc_core' not in locals():
        importance_masks, feature_space_22 = get_analysis_feature_masks("Analysis 2.2")
        mask_sad_core = importance_masks['SAD']
        mask_hc_core = importance_masks['HC']

    # Build subject lists if missing
    if 'subgroups_22' not in locals():
        subgroups_22 = {}
        for key, d in data_subsets.items():
            if d is None or 'ext' not in d or d['ext'] is None:
                continue
            subgroups_22[key] = np.unique(d['ext']['sub'])

    # Container for outputs

    data_rows = []

    if 'importance_masks' not in locals():
        importance_masks, feature_space_22 = get_analysis_feature_masks("Analysis 2.2")


    for key, subject_list in subgroups_22.items():
        group, drug = key.split('_')
        curr_mask = mask_sad_core if group == "SAD" else mask_hc_core
    
        for sub in subject_list:
            m_ext = (sub_ext == sub)
            X_e, y_e = X_ext[m_ext], y_ext[m_ext]
        
            # 1. Safety
            res_safe = calc_drift_metrics(X_e, y_e, X_e, y_e, COND_SAFE_LRN, COND_SAFE_TGT, curr_mask, sub)
            if res_safe:
                data_rows.append({"Subject": sub, "Group": group, "Drug": drug, "Domain": "Safety", **res_safe})
            
            # 2. Threat Maintenance: CSR(Ext) -> CSR(Reinstatement)
            if X_rst_all is not None:
                m_rst = (sub_rst_all == sub)
                if np.sum(m_rst) > 0:
                    X_r, y_r = X_rst_all[m_rst], y_rst_all[m_rst]
                    res_threat = calc_drift_metrics(X_e, y_e, X_r, y_r, COND_THREAT_LRN, COND_THREAT_LRN, curr_mask, sub)
                    if res_threat:
                        data_rows.append({"Subject": sub, "Group": group, "Drug": drug, "Domain": "Threat", **res_threat})

            # 3. Threat Shock Target: CSR(Ext) -> Shock/US target pattern
            X_shock, y_shock, sub_shock = get_shock_target_data(key)
            if X_shock is not None:
                m_shock = (sub_shock == sub)
                if np.sum(m_shock) > 0:
                    res_threat_shock = calc_drift_metrics(
                        X_e, y_e, X_shock[m_shock], y_shock[m_shock],
                        COND_THREAT_LRN, SHOCK_TARGET_LABELS, curr_mask, sub
                    )
                    if res_threat_shock:
                        data_rows.append({"Subject": sub, "Group": group, "Drug": drug, "Domain": SHOCK_TARGET_DOMAIN, **res_threat_shock})

    df_drift = pd.DataFrame(data_rows)
    print(f"  > Computed vectors for {len(df_drift['Subject'].unique())} subjects.")

    # =============================================================================
    # 2. Statistics (LME)
    # =============================================================================
    print("\n[Step 2] Statistical Testing (LME)...")
    lme_results = {}

    for domain in ["Safety", "Threat", SHOCK_TARGET_DOMAIN]:
        if domain not in df_drift['Domain'].values: continue
        df_sub = df_drift[df_drift["Domain"] == domain].copy()
        form_base = "~ C(Group, Treatment(reference='HC')) * C(Drug, Treatment(reference='Placebo'))"
    
        print(f"\n--- Domain: {domain} ---")
        for metric in ["Cosine", "Projection"]:
            try:
                md = smf.mixedlm(f"{metric} {form_base}", df_sub, groups=df_sub["Subject"])
                mdf = md.fit()
                term = "C(Group, Treatment(reference='HC'))[T.SAD]:C(Drug, Treatment(reference='Placebo'))[T.Oxytocin]"
                p_val = mdf.pvalues.get(term, 1.0)
                print(f"  > {metric}: Interaction p={p_val:.4f} {'*' if p_val<0.05 else ''}")
                lme_results[f"{domain}_{metric}"] = p_val
            except:
                lme_results[f"{domain}_{metric}"] = 1.0

    # =============================================================================
    # 3. Visualization (Lines Only, Error=SE)
    # =============================================================================
    sns.set_context("poster", font_scale=0.8)
    plot_domains = [d for d in ["Safety", "Threat", SHOCK_TARGET_DOMAIN] if d in df_drift.get("Domain", pd.Series(dtype=str)).values]
    fig, axes = plt.subplots(max(len(plot_domains), 1), 2, figsize=(18, 6 * max(len(plot_domains), 1)), squeeze=False)
    pal_group = {'SAD': '#c44e52', 'HC': '#4c72b0'}
    for row, domain in enumerate(plot_domains):
        plot_interaction(axes[row, 0], df_drift, domain, "Cosine", lme_results.get(f"{domain}_Cosine", 1.0))
        plot_interaction(axes[row, 1], df_drift, domain, "Projection", lme_results.get(f"{domain}_Projection", 1.0))

    plt.tight_layout()
    plt.show()

    print("Note: Error bars represent Standard Error of the Mean (SEM).")
    results_22 = {'df': df_drift, 'stats': lme_results, 'shock_target_labels': SHOCK_TARGET_LABELS}
    save_cell_results(18, ['COND_SAFE_LRN', 'COND_SAFE_TGT', 'COND_THREAT_LRN', 'SHOCK_TARGET_LABELS', 'SHOCK_TARGET_DOMAIN', 'PERCENTILE_THRESH', 'X_rst_all', 'axes', 'data_rows', 'data_subsets', 'df_drift', 'domain', 'fig', 'importance_mask_permutated', 'importance_scores_permutated', 'key', 'lme_results', 'meta', 'pal_group', 'results_11', 'results_12', 'results_13', 'results_13_2', 'results_14_self', 'results_21', 'results_21_pv', 'results_22', 'results_23', 'results_24', 'results_25', 'strict_cross_phase_results', 'sub_rst_all', 'sub_to_meta', 'subject_list', 'y_rst_all'])


else:
    maybe_load_cell_results(18)

# %% [cell 19]
if cell_active(19):
    # =============================================================================
    # 0. CACHE GATE & SETUP
    # =============================================================================
    cache_cell16 = os.path.join(CHECKPOINT_DIR, "cell_16_opening_test.joblib")

    opening_metrics = [
        "Entropy", "Entropy_CSS", "Entropy_CSR", "Kurtosis", "Variance",
        "P_CSR_CSS", "P_CSR_CSR", "Boundary_Separation",
        "Decision_Margin_CSS", "Decision_Margin_CSR", "Decision_Margin_All"
    ]

    if os.path.exists(cache_cell16):
        print(f"  [LOAD] Found existing metrics in {cache_cell16}. Skipping calculation...")
        results_23 = joblib.load(cache_cell16)
        df_metrics = results_23['df']
        stats_results = results_23['stats']
        missing_opening_cols = [m for m in opening_metrics if m not in df_metrics.columns]
        if missing_opening_cols:
            print(f"  [RECALC] Cached opening metrics are missing {missing_opening_cols}; recomputing Analysis 2.3.")
            results_23 = None
            df_metrics = None
            stats_results = None
    else:
        results_23 = None

    if results_23 is None:
        print(f"  [CALC] Executing Probability Extraction across Groups/Drugs...")
    
        # 1. Verification of Required Variables
        importance_masks, feature_space_23 = get_analysis_feature_masks("Analysis 2.3")
        mask_sad_native = importance_masks['SAD']
        mask_hc_native = importance_masks['HC']

        # 2. Extract Optimized C from Cell 6
        try:
            c_sad = res_sad['model_sad'].named_steps['classification'].C
            c_hc = res_hc['model_hc'].named_steps['classification'].C
        except:
            c_sad, c_hc = 0.01, 1.0 # Defaulting to your known optimized values
            print("    [WARN] Optimized C not found in memory. Using SAD=0.01, HC=1.0.")

        # 3. Execution Loop
        subgroups_23 = {"SAD_Placebo": [], "SAD_Oxytocin": [], "HC_Placebo": [], "HC_Oxytocin": []}
        for sub in np.unique(sub_ext):
            s_str = str(sub).strip()
            info = sub_to_meta.get(s_str) or sub_to_meta.get(f"sub-{s_str}")
            if info:
                key = f"{info['Group']}_{info['Drug']}"
                if key in subgroups_23: subgroups_23[key].append(sub)

        data_rows = []
        for key, sub_list in subgroups_23.items():
            group, drug = key.split('_')
            curr_mask = mask_sad_native if group == "SAD" else mask_hc_native
            curr_c = c_sad if group == "SAD" else c_hc
        
            for sub in sub_list:
                mask_s = (sub_ext == sub) & np.isin(y_ext, ["CSR", "CSS"])
                mask_proto = (sub_ext == sub) & np.isin(y_ext, ["CS-", "CSS", "CSR"])
                if np.sum(mask_s) < 8: continue
            
                # Slicing data for current subject and mask
                X_sub = X_ext[mask_s][:, curr_mask]
                y_sub = y_ext[mask_s]
                X_proto = X_ext[mask_proto][:, curr_mask]
                y_proto = y_ext[mask_proto]
            
                # LOTO (Leave-One-Trial-Out) Probability Extraction
                logo = LeaveOneGroupOut()
                probs = []
            
                # Treat each trial as its own group for LOTO
                trial_indices = np.arange(len(y_sub))
            
                for train, test in logo.split(X_sub, y_sub, groups=trial_indices):
                    clf = LogisticRegression(C=curr_c, solver='liblinear', random_state=42)
                    clf.fit(X_sub[train], y_sub[train])
                
                    # Get P(Threat)
                    csr_idx = np.where(clf.classes_ == "CSR")[0][0]
                    probs.append(clf.predict_proba(X_sub[test])[0, csr_idx])
            
                probs = np.array(probs)
            
                # Metric Calculation (matching Analysis 1.4 logic)
                hist, _ = np.histogram(probs, bins=20, range=(0, 1), density=True)
                ent_val = entropy(hist + 1e-9)
                kurt_val = kurtosis(probs, fisher=True)
                var_val = np.var(probs)
                probs_css = probs[y_sub == "CSS"]
                probs_csr = probs[y_sub == "CSR"]
                p_proto_csr_css, p_proto_csr_csr, proto_boundary_separation = prototype_decision_metrics(
                    X_proto, y_proto, "CSR", "CSS"
                )
                p_csr_css = float(np.mean(probs_css)) if len(probs_css) else np.nan
                p_csr_csr = float(np.mean(probs_csr)) if len(probs_csr) else np.nan
                boundary_separation = p_csr_csr - p_csr_css
                decision_margin_css = float(0.5 - p_csr_css) if np.isfinite(p_csr_css) else np.nan
                decision_margin_csr = float(p_csr_csr - 0.5) if np.isfinite(p_csr_csr) else np.nan
                decision_margin_all = float(np.mean(np.abs(probs - 0.5)))
                entropy_css = float(np.mean([entropy([p, 1 - p], base=2) for p in np.clip(probs_css, 1e-9, 1 - 1e-9)])) if len(probs_css) else np.nan
                entropy_csr = float(np.mean([entropy([p, 1 - p], base=2) for p in np.clip(probs_csr, 1e-9, 1 - 1e-9)])) if len(probs_csr) else np.nan
            
                data_rows.append({
                    "Subject": sub, "Group": group, "Drug": drug, 
                    "Entropy": ent_val, 
                    "Kurtosis": kurt_val, 
                    "Variance": var_val,
                    "P_CSR_CSS": p_csr_css,
                    "P_CSR_CSR": p_csr_csr,
                    "Boundary_Separation": boundary_separation,
                    "Prototype_P_CSR_CSS": p_proto_csr_css,
                    "Prototype_P_CSR_CSR": p_proto_csr_csr,
                    "Prototype_Boundary_Separation": proto_boundary_separation,
                    "Decision_Margin_CSS": decision_margin_css,
                    "Decision_Margin_CSR": decision_margin_csr,
                    "Decision_Margin_All": decision_margin_all,
                    "Entropy_CSS": entropy_css,
                    "Entropy_CSR": entropy_csr,
                })

        df_metrics = pd.DataFrame(data_rows)
    
        # 4. Statistical Testing (LME)
        print("\n[Step 2] Statistical Testing (Metric ~ Group * Drug)...")
        stats_results = {}
        for met in opening_metrics:
            try:
                formula = f"{met} ~ C(Group, Treatment(reference='HC')) * C(Drug, Treatment(reference='Placebo'))"
                md = smf.mixedlm(formula, df_metrics, groups=df_metrics["Subject"])
                mdf = md.fit()
            
                term_int = "C(Group, Treatment(reference='HC'))[T.SAD]:C(Drug, Treatment(reference='Placebo'))[T.Oxytocin]"
                stats_results[met] = mdf.pvalues.get(term_int, 1.0)
                print(f"  > {met} Interaction p={stats_results[met]:.4f}")
            except Exception as e:
                print(f"  ! {met} failed: {e}")
                stats_results[met] = 1.0

        results_23 = {'df': df_metrics, 'stats': stats_results}
        joblib.dump(results_23, cache_cell16)

    # =============================================================================
    # 5. VISUALIZATION
    # =============================================================================
    sns.set_context("poster", font_scale=0.8)
    n_cols = 3
    n_rows = int(np.ceil(len(opening_metrics) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 6 * n_rows))
    axes = np.ravel(axes)

    for i, met in enumerate(opening_metrics):
        sns.boxplot(data=df_metrics, x='Group', y=met, hue='Drug', ax=axes[i])
        axes[i].set_title(f"{met}\nInteraction p={stats_results.get(met, np.nan):.3f}")
        if i > 0 and axes[i].get_legend() is not None:
            axes[i].get_legend().remove()
    for ax in axes[len(opening_metrics):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    print("--- Cell 16 Complete ---")
    save_cell_results(19, ['axes', 'cache_cell16', 'data_subsets', 'fig', 'i', 'importance_mask_permutated', 'importance_scores_permutated', 'met', 'meta', 'results_11', 'results_12', 'results_13', 'results_13_2', 'results_14_self', 'results_21', 'results_21_pv', 'results_22', 'results_23', 'results_24', 'results_25', 'strict_cross_phase_results', 'sub_to_meta'])


else:
    maybe_load_cell_results(19)

# %% [cell 20]
if cell_active(20):
    # Cell 17: Analysis 2.4 - Spatial Re-Alignment (The "Normalizing" Effect)
    print("--- Running Analysis 2.4: Spatial Re-Alignment (Using Analysis 1.1 Output) ---")
    # Constants
    COND_SAFE = "CSS"
    COND_THREAT = "CSR"
    LABELS = [COND_SAFE, COND_THREAT]

    # =============================================================================
    # 0. Setup: Retrieve Correct Model from Cell 6 Results
    # =============================================================================
    gold_model = get_analysis11_model("HC")

    # Verify Classes
    print(f"  > Model Classes: {gold_model.classes_}")
    if not set(LABELS).issubset(set(gold_model.classes_)):
        raise ValueError(f"Model trained on {gold_model.classes_}, but {LABELS} required.")

    # =============================================================================
    # 1. Data Retrieval (SAD Placebo vs SAD Oxytocin)
    # =============================================================================
    # Pulling the 'X', 'y', and 'sub' arrays for the respective drug groups
    X_sad_plc, y_sad_plc, sub_sad_plc = get_ext_data("SAD_Placebo")
    X_sad_oxt, y_sad_oxt, sub_sad_oxt = get_ext_data("SAD_Oxytocin")

    # =============================================================================
    # 2. Cross-Decoding (Forced-Choice Accuracy)
    # =============================================================================
    print("\n[Step 1] Applying Healthy Model to SAD Subgroups...")

    def get_forced_choice_metrics(X, y, subs, model):
        mask = np.isin(y, LABELS)
        X_f, y_f, sub_f = X[mask], y[mask], subs[mask]
    
        # Get decision scores (distance from hyperplane)
        scores = model.decision_function(X_f)
    
        # Convert 1D scores to 2D for the forced-choice helper
        scores_2d = np.column_stack((-scores, scores)) if scores.ndim == 1 else scores
    
        # Calculate subject-level forced-choice accuracy
        # Requires compute_subject_forced_choice_accs to be defined in Utils
        accs = compute_subject_forced_choice_accs(
            y_f, scores_2d, sub_f, list(model.classes_)
        )
        return accs

    acc_sad_plc = get_forced_choice_metrics(X_sad_plc, y_sad_plc, sub_sad_plc, gold_model)
    acc_sad_oxt = get_forced_choice_metrics(X_sad_oxt, y_sad_oxt, sub_sad_oxt, gold_model)

    m_plc = np.mean(acc_sad_plc)
    m_oxt = np.mean(acc_sad_oxt)

    print(f"  > Accuracy (Train HC -> Test SAD-PLC): {m_plc:.1%}")
    print(f"  > Accuracy (Train HC -> Test SAD-OXT): {m_oxt:.1%}")

    # =============================================================================
    # 3. Statistical Test & Visualization
    # =============================================================================
    t_stat, p_val = ttest_ind(acc_sad_oxt, acc_sad_plc, alternative='greater')
    sig_label = "*" if p_val < 0.05 else "ns"

    sns.set_context("poster", font_scale=0.8)
    fig, ax = plt.subplots(figsize=(10, 5))

    # Prepare Matrix: 1 Row (Train HC) x 2 Cols (Test PLC, Test OXT)
    matrix_data = np.array([[m_plc, m_oxt]])

    # Annotation String
    annot_data = np.array([
        [f"{m_plc:.3f}", f"{m_oxt:.3f}\n({sig_label})"]
    ])

    # Draw Heatmap
    sns.heatmap(matrix_data, annot=annot_data, fmt="", cmap="RdBu_r", 
                vmin=0.3, vmax=0.7, center=0.5, cbar=True,
                xticklabels=['Test: SAD-Placebo', 'Test: SAD-Oxytocin'], 
                yticklabels=['Train: HC-Placebo (Anal 1.1)'], ax=ax)

    ax.set_title(f"Analysis 2.4: Spatial Re-Alignment\n(OXT vs PLC Improvement: p={p_val:.3f})")
    plt.yticks(rotation=0) 

    plt.tight_layout()
    plt.show()

    # Save results for manuscript
    results_24 = {'acc_plc': acc_sad_plc, 'acc_oxt': acc_sad_oxt, 'p_val': p_val}
    joblib.dump(results_24, os.path.join(CHECKPOINT_DIR, "cell_17_realignment.joblib"))
    save_cell_results(20, ['COND_SAFE', 'COND_THREAT', 'LABELS', 'X_sad_oxt', 'X_sad_plc', 'acc_sad_oxt', 'acc_sad_plc', 'annot_data', 'ax', 'data_subsets', 'fig', 'importance_mask_permutated', 'importance_scores_permutated', 'm_oxt', 'm_plc', 'matrix_data', 'meta', 'p_val', 'results_11', 'results_12', 'results_13', 'results_13_2', 'results_14_self', 'results_21', 'results_21_pv', 'results_22', 'results_23', 'results_24', 'results_25', 'sig_label', 'strict_cross_phase_results', 'sub_sad_oxt', 'sub_sad_plc', 'sub_to_meta', 't_stat', 'y_sad_oxt', 'y_sad_plc'])


else:
    maybe_load_cell_results(20)

# %% [cell 21]
if cell_active(21):
    # Cell 18: Analysis 2.5 - Reverse Cross-Decoding (SAD Template -> HC)
    # Objective: Test if the "Disordered" SAD representation generalizes to Healthy brains.
    # Protocol:
    #   1. Train Model on SAD-Placebo (CSS vs CSR).
    #   2. Feature Selection: Full feature set.
    #   3. Test on HC-Placebo and HC-Oxytocin.
    #   4. Metric: Subject-level forced-choice accuracy from decision scores.
    # Hypothesis: Accuracy should be LOW (near chance), confirming "Functional Specificity".

    print("--- Running Analysis 2.5: Reverse Cross-Decoding (SAD -> HC) ---")

    # Constants
    COND_SAFE = "CSS"
    COND_THREAT = "CSR"
    LABELS = [COND_SAFE, COND_THREAT]

    # =============================================================================
    # 0. Setup: Retrieve Correct Model from Cell 6 Results
    # =============================================================================
    gold_model = get_analysis11_model("SAD")

    # Verify Classes
    print(f"  > Model Classes: {gold_model.classes_}")
    if not set(LABELS).issubset(set(gold_model.classes_)):
        raise ValueError(f"Model trained on {gold_model.classes_}, but {LABELS} required.")

    # =============================================================================
    # 1. Data Retrieval (HC Placebo vs HC Oxytocin)
    # =============================================================================
    # Pulling the 'X', 'y', and 'sub' arrays for the respective drug groups
    X_hc_plc, y_hc_plc, sub_hc_plc = get_ext_data("HC_Placebo")
    X_hc_oxt, y_hc_oxt, sub_hc_oxt = get_ext_data("HC_Oxytocin")

    # =============================================================================
    # 2. Cross-Decoding (Forced-Choice Accuracy)
    # =============================================================================
    print("\n[Step 1] Applying SAD Model to Healthy Subgroups...")

    def get_forced_choice_metrics(X, y, subs, model):
        mask = np.isin(y, LABELS)
        X_f, y_f, sub_f = X[mask], y[mask], subs[mask]
    
        # Get decision scores (distance from hyperplane)
        scores = model.decision_function(X_f)
    
        # Convert 1D scores to 2D for the forced-choice helper
        scores_2d = np.column_stack((-scores, scores)) if scores.ndim == 1 else scores
    
        # Calculate subject-level forced-choice accuracy
        # Requires compute_subject_forced_choice_accs to be defined in Utils
        accs = compute_subject_forced_choice_accs(
            y_f, scores_2d, sub_f, list(model.classes_)
        )
        return accs

    acc_hc_plc = get_forced_choice_metrics(X_hc_plc, y_hc_plc, sub_hc_plc, gold_model)
    acc_hc_oxt = get_forced_choice_metrics(X_hc_oxt, y_hc_oxt, sub_hc_oxt, gold_model)

    m_plc = np.mean(acc_hc_plc)
    m_oxt = np.mean(acc_hc_oxt)

    print(f"  > Accuracy (Train SAD -> Test HC-PLC): {m_plc:.1%}")
    print(f"  > Accuracy (Train SAD -> Test HC-OXT): {m_oxt:.1%}")

    # =============================================================================
    # 3. Statistical Test & Visualization
    # =============================================================================
    t_stat, p_val = ttest_ind(acc_hc_oxt, acc_hc_plc, alternative='greater')
    sig_label = "*" if p_val < 0.05 else "ns"

    sns.set_context("poster", font_scale=0.8)
    fig, ax = plt.subplots(figsize=(10, 5))

    # Prepare Matrix: 1 Row (Train SAD) x 2 Cols (Test HC PLC, Test HC OXT)
    matrix_data = np.array([[m_plc, m_oxt]])

    # Annotation String
    annot_data = np.array([
        [f"{m_plc:.3f}", f"{m_oxt:.3f}\n({sig_label})"]
    ])

    # Draw Heatmap
    sns.heatmap(matrix_data, annot=annot_data, fmt="", cmap="RdBu_r", 
                vmin=0.3, vmax=0.7, center=0.5, cbar=True,
                xticklabels=['Test: HC-Placebo', 'Test: HC-Oxytocin'], 
                yticklabels=['Train: SAD-Placebo (Anal 1.1)'], ax=ax)

    ax.set_title(f"Analysis 2.5: Reverse Cross-Decoding\n(OXT vs PLC Difference: p={p_val:.3f})")
    plt.yticks(rotation=0) 

    plt.tight_layout()
    plt.show()

    # Save results for manuscript
    results_25 = {'acc_plc': acc_hc_plc, 'acc_oxt': acc_hc_oxt, 'p_val': p_val}
    joblib.dump(results_25, os.path.join(CHECKPOINT_DIR, "cell_18_reverse_cross_decoding.joblib"))
    joblib.dump(results_25, os.path.join(CHECKPOINT_DIR, "cell_18_realignment.joblib"))  # Legacy filename.
    save_cell_results(21, ['COND_SAFE', 'COND_THREAT', 'LABELS', 'X_hc_oxt', 'X_hc_plc', 'acc_hc_oxt', 'acc_hc_plc', 'annot_data', 'ax', 'data_subsets', 'fig', 'importance_mask_permutated', 'importance_scores_permutated', 'm_oxt', 'm_plc', 'matrix_data', 'meta', 'p_val', 'results_11', 'results_12', 'results_13', 'results_13_2', 'results_14_self', 'results_21', 'results_21_pv', 'results_22', 'results_23', 'results_24', 'results_25', 'sig_label', 'strict_cross_phase_results', 'sub_hc_oxt', 'sub_hc_plc', 'sub_to_meta', 't_stat', 'y_hc_oxt', 'y_hc_plc'])


else:
    maybe_load_cell_results(21)

# %% [cell 23]
if cell_active(23):
    clinical_dir = os.path.join(PROJECT_ROOT, "MRI/source_data/behav")
    LSAS_path = resolve_clinical_csv_path(clinical_dir, "LSASSubtotals", "SocialSafetyLearning-LSASSubtotals_DATA_2026-04-25_2306.csv")
    ECR_path = resolve_clinical_csv_path(clinical_dir, "ECR", "SocialSafetyLearning-ECR_DATA_2026-04-25_2306.csv")
    DASS_path = resolve_clinical_csv_path(clinical_dir, "DASS", "SocialSafetyLearning-DASS_DATA_2026-04-25_2306.csv")
    #login_participantid; lsas_fear_total; lsas_avoid_total; lsas_total;
    df_lsas_raw = pd.read_csv(LSAS_path)
    df_lsas = pd.DataFrame()
    df_lsas['sub_ID'] = df_lsas_raw['login_participantid']
    df_lsas['lsas_fear'] = df_lsas_raw['lsas_fear_total']
    df_lsas['lsas_avoid'] = df_lsas_raw['lsas_avoid_total']
    df_lsas['lsas_total'] = df_lsas_raw['lsas_total']
    #login_participantid; ecr_total;
    df_ecr_raw = pd.read_csv(ECR_path)
    df_ecr = pd.DataFrame()
    df_ecr['sub_ID'] = df_ecr_raw['login_participantid']
    df_ecr['ecr_total'] = df_ecr_raw['ecr_total']
    #login_participantid; ecr_total;
    df_dass_raw = pd.read_csv(DASS_path)
    # Define item groupings (Note: indices match your dass_q# names)
    depression_items = ['dass_q3_positive', 'dass_q5_initiative', 'dass_q10_forward', 
                        'dass_q13_blue', 'das_q16_enthusiastic', 'dass_q17_worth', 'dass_q21_life']

    anxiety_items = ['dass_q2_drymouth', 'dass_q4_breathing', 'dass_q7_trembling', 
                     'dass_q9_panic', 'dass_q15_panic', 'dass_q19_heart', 'dass_q20_scared']

    stress_items = ['dass_q1_winddown', 'dass_q6_overreact', 'dass_q8_nervousenergy', 
                    'dass_q11_agitated', 'dass_q12_relax', 'dass_q14_intolerant', 'dass_q18_touch']

    # Calculate Subscale Totals (Summing and multiplying by 2 per DASS-21 protocol)
    df_dass = pd.DataFrame()
    df_dass['sub_ID'] = df_dass_raw['login_participantid']
    df_dass['dass_depression'] = df_dass_raw[depression_items].sum(axis=1) * 2
    df_dass['dass_anxiety'] = df_dass_raw[anxiety_items].sum(axis=1) * 2
    df_dass['dass_stress'] = df_dass_raw[stress_items].sum(axis=1) * 2
    df_scored_clinical = df_dass \
        .merge(df_lsas, on='sub_ID', how='inner') \
        .merge(df_ecr, on='sub_ID', how='inner')
    df_scr_trials, SCR_path = load_trialwise_scr(PROJECT_ROOT)
    df_scr_indices = summarize_scr_indices(df_scr_trials)
    if not df_scr_indices.empty:
        df_scored_clinical = df_scored_clinical.merge(df_scr_indices, on='sub_ID', how='left')
        print(f"SCR behavioral indices merged for {df_scr_indices['sub_ID'].nunique()} subjects.")
    stage23_payload = {
        "df_scored_clinical": df_scored_clinical,
        "df_scr_trials": df_scr_trials,
        "df_scr_indices": df_scr_indices,
        "clinical_paths": {"LSAS": LSAS_path, "ECR": ECR_path, "DASS": DASS_path, "SCR": SCR_path},
    }
    save_stage_bundle(23, "stage23_ClinicalScores", stage23_payload)
    save_cell_results(23, ['DASS_path', 'ECR_path', 'LSAS_path', 'SCR_path', 'anxiety_items', 'clinical_dir', 'data_subsets', 'depression_items', 'df_dass', 'df_dass_raw', 'df_ecr', 'df_ecr_raw', 'df_lsas', 'df_lsas_raw', 'df_scr_indices', 'df_scr_trials', 'df_scored_clinical', 'importance_mask_permutated', 'importance_scores_permutated', 'meta', 'results_11', 'results_12', 'results_13', 'results_13_2', 'results_14_self', 'results_21', 'results_21_pv', 'results_22', 'results_23', 'results_24', 'results_25', 'stress_items', 'strict_cross_phase_results', 'sub_to_meta'])


else:
    maybe_load_cell_results(23)

# %% [cell 24]
if cell_active(24):
    # 1. Extract PV Distance Vectors
    vA_sad_pv, vB_sad_pv, vC_sad_pv = extract_metrics_pv(
        results_12["rdms_sad_raw_pv"],
        include_threat_background=True,
    )
    vA_hc_pv, vB_hc_pv, vC_hc_pv = extract_metrics_pv(
        results_12["rdms_hc_raw_pv"],
        include_threat_background=True,
    )

    # 2. Extract subject IDs from saved Analysis 1.2 results, or reconstruct from raw trial labels.
    s_id_sad = resolve_topology_subject_ids(results_12, "SAD", len(vA_sad_pv))
    s_id_hc = resolve_topology_subject_ids(results_12, "HC", len(vA_hc_pv))

    # 3. Validation Check
    print(f"Lengths: SAD_IDs({len(s_id_sad)}) vs SAD_Dist({len(vA_sad_pv)})")
    print(f"Lengths: HC_IDs({len(s_id_hc)}) vs HC_Dist({len(vA_hc_pv)})")

    # 4. Final Dataframe Construction
    # Using explicit slicing to force match lengths in case of trailing index errors
    df_neural_topology = pd.DataFrame({
        'sub_ID': np.concatenate([s_id_sad, s_id_hc]).astype(str),
        'Neural_Dist_Threat_Safety': np.concatenate([vA_sad_pv, vA_hc_pv]),
        'Neural_Dist_Safety_Background': np.concatenate([vB_sad_pv, vB_hc_pv]),
        'Neural_Dist_Safety_Backgr': np.concatenate([vB_sad_pv, vB_hc_pv]),
        'Neural_Dist_Threat_Background': np.concatenate([vC_sad_pv, vC_hc_pv]),
        'Group': ['SAD']*len(s_id_sad) + ['HC']*len(s_id_hc)
    })
    threat_background = pd.to_numeric(df_neural_topology['Neural_Dist_Threat_Background'], errors='coerce')
    safety_background = pd.to_numeric(df_neural_topology['Neural_Dist_Safety_Background'], errors='coerce')
    threat_safety_denominator = threat_background + safety_background
    df_neural_topology['Neural_ThreatTriangleOpenness'] = threat_background - safety_background
    df_neural_topology['Neural_Threat_Safety_Distance'] = (
        df_neural_topology['Neural_ThreatTriangleOpenness']
        / threat_safety_denominator.where(threat_safety_denominator.abs() > EPS)
    )

    # =============================================================================
    # NEURAL INDEX GENERATION (Analysis 1.3: Trajectory Slope)
    # =============================================================================

    # 1. Prepare Data Containers
    # We extract the slopes from df_safe (Safety Learning) 
    # because it is the most clinical relevant learning phase.
    if 'df_safe' not in locals():
        # If the cell above just ran, it might be nested in the results dictionary
        df_safe = results_13_2['data_safe']

    # 2. Calculate Linear Slope per Subject
    # This represents the 'Rate of Safety Learning'
    def calculate_subject_slopes(df, metric_name, mean_name=None):
        slopes = []
        columns = ['sub_ID', metric_name]
        if mean_name:
            columns.append(mean_name)
        if df is None or df.empty:
            return pd.DataFrame(columns=columns)
        if 'Drug' in df.columns:
            df = df[df['Drug'].astype(str).eq('Placebo')].copy()
        for sub in df['sub'].unique():
            sub_data = df[df['sub'] == sub].sort_values('trial')
            if len(sub_data) < 3: continue  # Need enough points for a trend
        
            # Linear fit: y = mx + b (we want 'm')
            m, _ = np.polyfit(sub_data['trial'], sub_data['score'], 1)
        
            row = {
                'sub_ID': str(sub),
                metric_name: m,
            }
            if metric_name == 'Neural_Safety_Trajectory_Slope':
                row['Neural_Rigidity_Slope'] = m  # Legacy alias for older saved analyses.
            if mean_name:
                row[mean_name] = sub_data['score'].mean()
            slopes.append(row)
        return pd.DataFrame(slopes)

    def calculate_dynamic_discrimination_metrics(data_safe, data_threat):
        columns = [
            'sub_ID',
            'Neural_DynamicDiscrimination_Volatility',
        ]
        if data_safe is None or data_threat is None or data_safe.empty or data_threat.empty:
            return pd.DataFrame(columns=columns)
        safe = data_safe.copy()
        threat = data_threat.copy()
        if 'Drug' in safe.columns:
            safe = safe[safe['Drug'].astype(str).eq('Placebo')].copy()
        if 'Drug' in threat.columns:
            threat = threat[threat['Drug'].astype(str).eq('Placebo')].copy()
        needed = {'sub', 'trial', 'score'}
        if not needed.issubset(safe.columns) or not needed.issubset(threat.columns):
            return pd.DataFrame(columns=columns)
        merged = safe[['sub', 'trial', 'score']].rename(columns={'score': 'safety_score'}).merge(
            threat[['sub', 'trial', 'score']].rename(columns={'score': 'threat_score'}),
            on=['sub', 'trial'],
            how='inner',
        )
        rows = []
        for sub, sub_data in merged.groupby('sub'):
            sub_data = sub_data.sort_values('trial').copy()
            if len(sub_data) < 3:
                continue
            delta = pd.to_numeric(sub_data['threat_score'], errors='coerce') - pd.to_numeric(sub_data['safety_score'], errors='coerce')
            valid = delta.notna() & pd.to_numeric(sub_data['trial'], errors='coerce').notna()
            if valid.sum() < 3:
                continue
            trial = pd.to_numeric(sub_data.loc[valid, 'trial'], errors='coerce').to_numpy(dtype=float)
            values = delta.loc[valid].to_numpy(dtype=float)
            diffs = np.diff(values)
            volatility = float(np.sqrt(np.nanmean(diffs ** 2))) if diffs.size else np.nan
            rows.append({
                'sub_ID': str(sub),
                'Neural_DynamicDiscrimination_Volatility': volatility,
            })
        return pd.DataFrame(rows, columns=columns)

    trajectory_payload = results_13_2 if isinstance(results_13_2, dict) else {}
    df_neural_trajectories = calculate_subject_slopes(
        trajectory_payload.get('data_safe', df_safe),
        'Neural_Safety_Trajectory_Slope',
        'Neural_Safety_Mean',
    )
    for data_key, metric_name in [
        ('data_threat', 'Neural_Threat_Trajectory_Slope'),
        ('data_threat_shock', 'Shock_Anchor_Trajectory_Slope'),
    ]:
        metric_slopes = calculate_subject_slopes(trajectory_payload.get(data_key, pd.DataFrame()), metric_name)
        if not metric_slopes.empty:
            df_neural_trajectories = df_neural_trajectories.merge(metric_slopes, on='sub_ID', how='outer')
    dynamic_discrimination = calculate_dynamic_discrimination_metrics(
        trajectory_payload.get('data_safe', df_safe),
        trajectory_payload.get('data_threat', pd.DataFrame()),
    )
    if not dynamic_discrimination.empty:
        df_neural_trajectories = df_neural_trajectories.merge(dynamic_discrimination, on='sub_ID', how='outer')
    shock_anchor_df = pd.DataFrame()
    if isinstance(results_12, dict):
        shock_anchor_results = results_12.get('shock_anchor_results', {})
        if isinstance(shock_anchor_results, dict):
            shock_anchor_df = shock_anchor_results.get('df', pd.DataFrame())
        if shock_anchor_df.empty:
            shock_anchor_df = results_12.get('shock_anchor_df', pd.DataFrame())
    if isinstance(shock_anchor_df, pd.DataFrame) and not shock_anchor_df.empty and 'Cosine_CSR_minus_CSS' in shock_anchor_df.columns:
        residualized_shock_anchor = shock_anchor_df[['Subject', 'Cosine_CSR_minus_CSS']].rename(columns={
            'Subject': 'sub_ID',
            'Cosine_CSR_minus_CSS': 'Residualized_Shock_Anchor_Trajectory_Slope',
        })
        residualized_shock_anchor['sub_ID'] = residualized_shock_anchor['sub_ID'].astype(str)
        df_neural_trajectories = df_neural_trajectories.merge(residualized_shock_anchor, on='sub_ID', how='outer')
    if 'df_scr_trials' not in locals():
        df_scr_trials, SCR_path = load_trialwise_scr(PROJECT_ROOT)
    df_scr_neural_coupling = calculate_neural_scr_safety_coupling(df_safe, df_scr_trials)
    if not df_scr_neural_coupling.empty:
        df_neural_trajectories = df_neural_trajectories.merge(df_scr_neural_coupling, on='sub_ID', how='left')

    print(f"Trajectory Indices generated for {len(df_neural_trajectories)} subjects.")
    print(df_neural_trajectories.head())

    # =============================================================================
    # NEURAL INDEX GENERATION (Analysis 1.4: Decision evidence and certainty)
    # =============================================================================

    # 1. Access the Dataframes from memory (or cache if just loaded)
    if 'df_sad_stats' not in locals():
        df_sad_stats = results_14_self['df_sad']
        df_hc_stats = results_14_self['df_hc']

    # 2. Tag Groups and Combine
    uncertainty_cols = [
        'sub', 'entropy', 'entropy_css', 'entropy_csr', 'kurtosis', 'decision_margin_css', 'decision_margin_csr',
        'p_csr_css', 'p_csr_csr', 'boundary_separation'
    ]
    available_uncertainty_cols = [col for col in uncertainty_cols if col in df_sad_stats.columns and col in df_hc_stats.columns]
    df_sad_idx = df_sad_stats[available_uncertainty_cols].copy()
    df_sad_idx['Group'] = 'SAD'

    df_hc_idx = df_hc_stats[available_uncertainty_cols].copy()
    df_hc_idx['Group'] = 'HC'

    # 3. Final Neural Profile Construction
    df_neural_uncertainty = pd.concat([df_sad_idx, df_hc_idx])
    df_neural_uncertainty = df_neural_uncertainty.rename(columns={
        'sub': 'sub_ID',
        'entropy': 'Neural_Uncertainty_Entropy',
        'entropy_css': 'Neural_Decoder_Entropy_CSS',
        'entropy_csr': 'Neural_Decoder_Entropy_CSR',
        'kurtosis': 'Neural_Sharpness_Kurtosis',
        'decision_margin_css': 'Neural_Decision_Margin_CSS',
        'decision_margin_csr': 'Neural_Decision_Margin_CSR',
        'p_csr_css': 'Neural_ThreatLike_Safety',
        'p_csr_csr': 'Neural_Threat_Evidence_CSR',
        'boundary_separation': 'Neural_Boundary_Separation'
    })
    if 'Neural_ThreatLike_Safety' in df_neural_uncertainty.columns:
        df_neural_uncertainty['Neural_SafetyEvidence'] = 1 - pd.to_numeric(
            df_neural_uncertainty['Neural_ThreatLike_Safety'],
            errors='coerce',
        )
        # CSS certainty = Neural_Certainty_CSS = 2 * abs(P(safety | CSS) - 0.50).
        df_neural_uncertainty['Neural_Certainty_CSS'] = 2 * (
            df_neural_uncertainty['Neural_SafetyEvidence'] - 0.5
        ).abs()
    if 'Neural_Threat_Evidence_CSR' in df_neural_uncertainty.columns:
        df_neural_uncertainty['Neural_ThreatEvidence'] = pd.to_numeric(
            df_neural_uncertainty['Neural_Threat_Evidence_CSR'],
            errors='coerce',
        )
        # CSR certainty = Neural_Certainty_CSR = 2 * abs(P(threat | CSR) - 0.50).
        df_neural_uncertainty['Neural_Certainty_CSR'] = 2 * (
            df_neural_uncertainty['Neural_ThreatEvidence'] - 0.5
        ).abs()
    if {'Neural_Certainty_CSS', 'Neural_Certainty_CSR'}.issubset(df_neural_uncertainty.columns):
        df_neural_uncertainty['Prototype_Certainty'] = df_neural_uncertainty[
            ['Neural_Certainty_CSS', 'Neural_Certainty_CSR']
        ].mean(axis=1)

    # Ensure ID is a string for merging
    df_neural_uncertainty['sub_ID'] = df_neural_uncertainty['sub_ID'].astype(str)

    representative_masks, representative_feature_space = get_analysis_feature_masks("Analysis 1.4")
    df_core_representative = representative_core_metrics_from_data_subsets(data_subsets, representative_masks)
    if not df_core_representative.empty:
        topology_cols = [
            'sub_ID', 'Group', 'Drug',
            'Neural_Dist_Threat_Safety',
            'Neural_Dist_Safety_Background',
            'Neural_Dist_Threat_Background',
            'Neural_Threat_Safety_Distance',
            'Neural_ThreatTriangleOpenness',
        ]
        df_neural_topology = df_core_representative[topology_cols].copy()
        df_neural_topology['Neural_Dist_Safety_Backgr'] = df_neural_topology['Neural_Dist_Safety_Background']

        dynamic_cols = ['sub_ID', 'Group', 'Drug', 'Neural_DynamicDiscrimination_Volatility']
        trajectory_diagnostics = df_neural_trajectories.drop(
            columns=['Group', 'Drug', 'Neural_DynamicDiscrimination_Volatility'],
            errors='ignore',
        )
        df_neural_trajectories = df_core_representative[dynamic_cols].merge(
            trajectory_diagnostics,
            on='sub_ID',
            how='left',
        )

        uncertainty_core_cols = [
            'sub_ID', 'Group', 'Drug',
            'Neural_ThreatLike_Safety',
            'Neural_Threat_Evidence_CSR',
            'Neural_Boundary_Separation',
            'Neural_SafetyEvidence',
            'Neural_ThreatEvidence',
            'Neural_Certainty_CSS',
            'Neural_Certainty_CSR',
            'Prototype_Certainty',
        ]
        uncertainty_diagnostics = df_neural_uncertainty.drop(
            columns=[
                'Group', 'Drug',
                'Neural_ThreatLike_Safety',
                'Neural_Threat_Evidence_CSR',
                'Neural_Boundary_Separation',
                'Neural_SafetyEvidence',
                'Neural_ThreatEvidence',
                'Neural_Certainty_CSS',
                'Neural_Certainty_CSR',
                'Prototype_Certainty',
            ],
            errors='ignore',
        )
        df_neural_uncertainty = df_core_representative[uncertainty_core_cols].merge(
            uncertainty_diagnostics,
            on='sub_ID',
            how='left',
        )
        print(f"Representative core metrics generated for {len(df_core_representative)} subjects.")

    print(f"Decision Profile generated for {len(df_neural_uncertainty)} subjects.")
    print(df_neural_uncertainty.head())
    stage24_payload = {
        "df_core_representative": df_core_representative,
        "df_neural_topology": df_neural_topology,
        "df_neural_trajectories": df_neural_trajectories,
        "df_neural_uncertainty": df_neural_uncertainty,
        "df_scr_neural_coupling": df_scr_neural_coupling,
    }
    save_stage_bundle(24, "stage24_NeuralClinicalIndices", stage24_payload)
    save_cell_results(24, ['data_subsets', 'df_hc_idx', 'df_neural_topology', 'df_neural_trajectories', 'df_neural_uncertainty', 'df_sad_idx', 'df_scr_neural_coupling', 'df_scr_trials', 'importance_mask_permutated', 'importance_scores_permutated', 'meta', 'results_11', 'results_12', 'results_13', 'results_13_2', 'results_14_self', 'results_21', 'results_21_pv', 'results_22', 'results_23', 'results_24', 'results_25', 'strict_cross_phase_results', 'sub_to_meta', 'vA_hc_pv', 'vA_sad_pv', 'vB_hc_pv', 'vB_sad_pv'])


else:
    maybe_load_cell_results(24)

# %% [cell 25]
if cell_active(25):
    # notebook display only: merged clinical-neural columns
    pass

else:
    maybe_load_cell_results(25)

# %% [cell 26]
if cell_active(26):
    required_stage26_vars = [
        "df_scored_clinical",
        "df_neural_topology",
        "df_neural_trajectories",
        "df_neural_uncertainty",
    ]
    missing_stage26_vars = [name for name in required_stage26_vars if name not in globals()]
    if missing_stage26_vars:
        raise RuntimeError(
            "Stage 26 requires Stage 23 and 24 outputs in memory or checkpoint. "
            f"Missing: {missing_stage26_vars}. Re-run with --stage 23,24,26 "
            "or use need_to_run.sh so late stages execute in order."
        )

    #df_neural_topology
    df_final_clinical_neural = df_scored_clinical \
        .merge(df_neural_topology, on='sub_ID', how='inner') \
        .merge(df_neural_trajectories, on='sub_ID', how='inner') \
        .merge(df_neural_uncertainty, on='sub_ID', how='inner')
    df_final_indecision = df_final_clinical_neural  # Legacy alias for older saved analyses.

    df_final_clinical_neural['sub_ID'] = df_final_clinical_neural['sub_ID'].astype(str)
    meta['sub_ID'] = meta['subject_id'].astype(str)

    df_master_analysis = df_final_clinical_neural.merge(
        meta, 
        on='sub_ID', 
        how='inner'
    )
    save_stage_bundle(26, "stage26_MasterClinicalNeural", {
        "df_final_clinical_neural": df_final_clinical_neural,
        "df_final_indecision": df_final_indecision,
        "df_master_analysis": df_master_analysis,
    })
    save_cell_results(26, ['data_subsets', 'df_final_clinical_neural', 'df_final_indecision', 'df_master_analysis', 'importance_mask_permutated', 'importance_scores_permutated', 'meta', 'results_11', 'results_12', 'results_13', 'results_13_2', 'results_14_self', 'results_21', 'results_21_pv', 'results_22', 'results_23', 'results_24', 'results_25', 'strict_cross_phase_results', 'sub_to_meta'])


else:
    maybe_load_cell_results(26)

# %% [cell 27]
if cell_active(27):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import pearsonr

    # 1. Config
    groups = df_master_analysis['Group'].unique()
    neural_metrics = CORE_NEURAL_METRICS.copy()
    clinical_indices = PRIMARY_CLINICAL_SCORES + SECONDARY_CLINICAL_SCORES + PRIMARY_SCR_INDICES + SECONDARY_SCR_INDICES
    neural_metrics = keep_existing_columns(df_master_analysis, neural_metrics, "stage 27 neural metrics")
    clinical_indices = keep_existing_columns(df_master_analysis, clinical_indices, "stage 27 clinical/SCR indices")

    def get_sig_star(p):
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return "ns"

    # 2. Execution & Statistics Printout
    group_results = []
    print(f"{'Group':<6} | {'Neural Metric':<28} | {'Clinical':<12} | {'r':<6} | {'p':<8} | {'Sig'}")
    print("-" * 80)

    for grp in groups:
        df_grp = df_master_analysis[df_master_analysis['Group'] == grp]
        for n_m in neural_metrics:
            for c_i in clinical_indices:
                if n_m not in df_grp.columns or c_i not in df_grp.columns:
                    continue
                valid = df_grp[[n_m, c_i]].dropna()
                if len(valid) > 5:
                    r, p = pearsonr(valid[n_m], valid[c_i])
                    sig = get_sig_star(p)
                
                    # Print stats for the table
                    print(f"{grp:<6} | {n_m:<28} | {c_i:<12} | {r:<6.2f} | {p:<8.4f} | {sig}")
                
                    group_results.append({
                        'Group': grp, 'Neural': n_m, 'Clinical': c_i, 
                        'r': r, 'p': p, 'sig': sig
                    })

    df_res_grp = pd.DataFrame(group_results)
    save_stage_bundle(27, "stage27_NeuralClinicalPearson", {"df_res_grp": df_res_grp})

    # 3. Visualization: Group-Wise Comparative Grids
    sns.set_context("talk")

    for n_m in neural_metrics:
        fig, axes = plt.subplots(1, len(clinical_indices), figsize=(32, 7))
    
        for j, c_i in enumerate(clinical_indices):
            # We use a combined legend to show significance per group
            for grp in groups:
                grp_data = df_master_analysis[df_master_analysis['Group'] == grp]
                stats = df_res_grp[(df_res_grp['Group'] == grp) & 
                                   (df_res_grp['Neural'] == n_m) & 
                                   (df_res_grp['Clinical'] == c_i)]
            
                label_str = grp
                if not stats.empty:
                    sig = stats['sig'].values[0]
                    r = stats['r'].values[0]
                    label_str = f"{grp} (r={r:.2f}{'' if sig=='ns' else sig})"

                sns.regplot(data=grp_data, x=n_m, y=c_i, ax=axes[j], 
                            label=label_str, scatter_kws={'alpha':0.4})
            
            axes[j].set_title(f"{c_i.upper()}")
            handles, labels = axes[j].get_legend_handles_labels()
            if handles:
                axes[j].legend(handles, labels, fontsize=12, loc='best')
            sns.despine(ax=axes[j])
        
        plt.suptitle(f"Group-Wise Associations: {n_m.replace('_', ' ')}\n(* p<0.05, ** p<0.01, *** p<0.001)", 
                     fontsize=22, y=1.08)
        plt.tight_layout()
        plt.show()
        plt.close(fig)
    save_cell_results(27, ['clinical_indices', 'data_subsets', 'df_res_grp', 'group_results', 'groups', 'grp', 'importance_mask_permutated', 'importance_scores_permutated', 'meta', 'n_m', 'neural_metrics', 'results_11', 'results_12', 'results_13', 'results_13_2', 'results_14_self', 'results_21', 'results_21_pv', 'results_22', 'results_23', 'results_24', 'results_25', 'strict_cross_phase_results', 'sub_to_meta'])


else:
    maybe_load_cell_results(27)

# %% [cell 28]
if cell_active(28):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import pearsonr

    # 1. Config
    # Using the merged df_master_analysis
    groups = df_master_analysis['Group'].unique()
    neural_metrics = CORE_NEURAL_METRICS.copy()
    clinical_indices = PRIMARY_CLINICAL_SCORES + SECONDARY_CLINICAL_SCORES + PRIMARY_SCR_INDICES + SECONDARY_SCR_INDICES
    neural_metrics = keep_existing_columns(df_master_analysis, neural_metrics, "stage 28 neural metrics")
    clinical_indices = keep_existing_columns(df_master_analysis, clinical_indices, "stage 28 clinical/SCR indices")
    covariates = ['demo_age'] # Standard demographic controls

    def get_sig_star(p):
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return "ns"

    # 2. Execution & Statistics Printout
    group_results = []
    print(f"{'Group':<6} | {'Neural Metric':<28} | {'Clinical':<12} | {'r_adj':<6} | {'p':<8} | {'Sig'}")
    print("-" * 85)

    for grp in groups:
        # Filter group
        df_grp = df_master_analysis[df_master_analysis['Group'] == grp]
    
        for n_m in neural_metrics:
            for c_i in clinical_indices:
                if n_m not in df_grp.columns or c_i not in df_grp.columns:
                    continue
                r_adj, p_val, n_valid = partial_corr_residualized(df_grp, n_m, c_i, covariates)
                if np.isfinite(r_adj) and np.isfinite(p_val):
                    sig = get_sig_star(p_val)
                
                    print(f"{grp:<6} | {n_m:<28} | {c_i:<12} | {r_adj:<6.2f} | {p_val:<8.4f} | {sig}")
                
                    group_results.append({
                        'Group': grp, 'Neural': n_m, 'Clinical': c_i, 
                        'r': r_adj, 'p': p_val, 'sig': sig, 'n': n_valid,
                        'covariates': ",".join([c for c in covariates if c in df_grp.columns]),
                    })

    df_res_grp = pd.DataFrame(group_results)
    save_stage_bundle(28, "stage28_NeuralClinicalPartial", {"df_res_partial": df_res_grp})

    # 3. Visualization: Group-Wise Comparative Grids
    sns.set_context("talk")

    for n_m in neural_metrics:
        fig, axes = plt.subplots(1, len(clinical_indices), figsize=(32, 7))
    
        for j, c_i in enumerate(clinical_indices):
            for grp in groups:
                grp_data = df_master_analysis[df_master_analysis['Group'] == grp]
                stats = df_res_grp[(df_res_grp['Group'] == grp) & 
                                   (df_res_grp['Neural'] == n_m) & 
                                   (df_res_grp['Clinical'] == c_i)]
            
                label_str = grp
                if not stats.empty:
                    sig = stats['sig'].values[0]
                    r = stats['r'].values[0]
                    # Label shows the adjusted r-value (partial correlation)
                    label_str = f"{grp} (adj_r={r:.2f}{'' if sig=='ns' else sig})"

                sns.regplot(data=grp_data, x=n_m, y=c_i, ax=axes[j], 
                            label=label_str, scatter_kws={'alpha':0.4})
            
            axes[j].set_title(f"{c_i.upper()}")
            handles, labels = axes[j].get_legend_handles_labels()
            if handles:
                axes[j].legend(handles, labels, fontsize=12, loc='best')
            sns.despine(ax=axes[j])
        
        plt.suptitle(f"Group Associations (Controlled for Age/Gender): {n_m.replace('_', ' ')}\n(* p<0.05, ** p<0.01, *** p<0.001)", 
                     fontsize=22, y=1.08)
        plt.tight_layout()
        plt.show()
        plt.close(fig)
    save_cell_results(28, ['clinical_indices', 'covariates', 'data_subsets', 'df_res_grp', 'group_results', 'groups', 'grp', 'importance_mask_permutated', 'importance_scores_permutated', 'meta', 'n_m', 'neural_metrics', 'results_11', 'results_12', 'results_13', 'results_13_2', 'results_14_self', 'results_21', 'results_21_pv', 'results_22', 'results_23', 'results_24', 'results_25', 'strict_cross_phase_results', 'sub_to_meta'])


else:
    maybe_load_cell_results(28)

# %% [cell 29]
if cell_active(29):
    import pandas as pd
    import numpy as np
    from scipy.stats import zscore

    # 1. Configuration
    neural_metrics = CORE_NEURAL_METRICS.copy()
    clinical_indices = PRIMARY_CLINICAL_SCORES + SECONDARY_CLINICAL_SCORES + PRIMARY_SCR_INDICES + SECONDARY_SCR_INDICES
    covariates = ['demo_age']
    neural_metrics = keep_existing_columns(df_master_analysis, neural_metrics, "stage 29 neural metrics")
    clinical_indices = keep_existing_columns(df_master_analysis, clinical_indices, "stage 29 clinical/SCR indices")
    covariates = keep_existing_columns(df_master_analysis, covariates, "stage 29 covariates")

    all_cols = neural_metrics + clinical_indices + covariates
    z_limit = 3.0 

    # 2. Execution: Identify Outliers -> Drop -> Final Z-scoring
    for col in all_cols:
        if col in df_master_analysis.columns:
            # Pass 1: Identify outliers using initial Z-scores
            initial_z = zscore(df_master_analysis[col], nan_policy='omit')
            outlier_mask = np.abs(initial_z) > z_limit
        
            num_removed = outlier_mask.sum()
        
            # Pass 2: Set outliers to NaN and calculate final Z-scores
            # We work on a copy to avoid SettingWithCopy warnings
            df_master_analysis.loc[outlier_mask, col] = np.nan
        
            # Calculate the final Z column based on the cleaned data
            z_col_name = f'{col}_z'
            df_master_analysis[z_col_name] = zscore(df_master_analysis[col], nan_policy='omit')
        
            if num_removed > 0:
                print(f"Column {col:<28}: Removed {num_removed} outliers (> {z_limit} SD)")
            
        else:
            print(f"Warning: {col} not found in dataframe.")

    print("\nOutlier removal and final Z-scoring complete.")
    df_master_analysis_z = df_master_analysis.copy()
    save_stage_bundle(29, "stage29_NeuralClinicalZScore", {"df_master_analysis": df_master_analysis, "df_master_analysis_z": df_master_analysis_z})
    save_cell_results(29, ['all_cols', 'clinical_indices', 'col', 'covariates', 'data_subsets', 'df_master_analysis', 'df_master_analysis_z', 'importance_mask_permutated', 'importance_scores_permutated', 'meta', 'neural_metrics', 'results_11', 'results_12', 'results_13', 'results_13_2', 'results_14_self', 'results_21', 'results_21_pv', 'results_22', 'results_23', 'results_24', 'results_25', 'strict_cross_phase_results', 'sub_to_meta', 'z_limit'])


else:
    maybe_load_cell_results(29)

# %% [cell 30]
if cell_active(30):
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    import matplotlib.pyplot as plt
    import seaborn as sns

    required_stage30_vars = ["df_master_analysis", "neural_metrics", "clinical_indices"]
    missing_stage30_vars = [name for name in required_stage30_vars if name not in globals()]
    if missing_stage30_vars:
        raise RuntimeError(
            "Stage 30 requires Stage 29 outputs in memory or checkpoint. "
            f"Missing: {missing_stage30_vars}. Re-run with --stage 29,30 "
            "or use need_to_run.sh so late stages execute in order."
        )

    # 1. Config
    missing_neural_z = [c for c in neural_metrics if f'{c}_z' not in df_master_analysis.columns]
    missing_clinical_z = [c for c in clinical_indices if f'{c}_z' not in df_master_analysis.columns]
    if missing_neural_z:
        print(f"[Column filter] Skipping missing stage 30 neural z metrics: {missing_neural_z}")
    if missing_clinical_z:
        print(f"[Column filter] Skipping missing stage 30 clinical/SCR z indices: {missing_clinical_z}")
    neural_metrics = [c for c in neural_metrics if f'{c}_z' in df_master_analysis.columns]
    clinical_indices = [c for c in clinical_indices if f'{c}_z' in df_master_analysis.columns]
    neural_z = [f'{c}_z' for c in neural_metrics]
    clinical_z = [f'{c}_z' for c in clinical_indices]
    # Ensure groups are sorted for consistent coloring (e.g., SAD vs HC)
    groups = sorted(df_master_analysis['Group'].unique()) if 'Group' in df_master_analysis.columns else [None]

    sns.set_context("talk")
    sns.set_style("white")
    ols_results = []

    # 2. Analysis & Plotting Loop
    for n_m in neural_z:
        # Print header for the specific neural metric
        print(f"\n{'='*80}\nNEURAL PREDICTOR: {n_m.upper()}\n{'='*80}")
    
        # Create subplots: 1 column for each clinical index
        fig, axes = plt.subplots(1, len(clinical_z), figsize=(36, 7), sharey=True)
        if len(clinical_z) == 1: axes = [axes] # Ensure axes is iterable for single plots
    
        for i, c_i in enumerate(clinical_z):
            ax = axes[i]
            print(f"\n--- Clinical Outcome: {c_i.upper()} ---")
        
            for grp in groups:
                # Subset data for the group
                df_grp = df_master_analysis[df_master_analysis['Group'] == grp]
                if n_m not in df_grp.columns or c_i not in df_grp.columns:
                    continue
            
                # Drop NaNs (crucial after outlier removal) to ensure model runs
                analysis_df = df_grp[[n_m, c_i]].dropna()
            
                if len(analysis_df) > 5: # Minimum N check
                    # A. STATISTICAL ANALYSIS
                    X = sm.add_constant(analysis_df[n_m])
                    y = analysis_df[c_i]
                    model = sm.OLS(y, X).fit()
                
                    # Print concise stats for each group
                    p_val = model.pvalues[n_m]
                    beta = model.params[n_m]
                    sig_note = " *SIGNIFICANT*" if p_val < 0.05 else ""
                    print(f"[{grp:<3}] N={len(analysis_df):<3} | Beta={beta:>6.3f} | t={model.tvalues[n_m]:>6.2f} | p={p_val:>6.4f}{sig_note}")
                    ols_results.append({
                        "Group": grp,
                        "neural_metric_z": n_m,
                        "clinical_score_z": c_i,
                        "n": int(len(analysis_df)),
                        "estimate": float(beta),
                        "t": float(model.tvalues[n_m]),
                        "p": float(p_val),
                        "r2": float(model.rsquared),
                    })
                
                    # B. VISUALIZATION
                    sns.regplot(
                        data=analysis_df, x=n_m, y=c_i, ax=ax, 
                        label=f"{grp} (p={p_val:.3f})", 
                        scatter_kws={'alpha': 0.4},
                        line_kws={'lw': 3}
                    )
                else:
                    print(f"[{grp:<3}] Insufficient data (N < 5)")

            # Formatting each subplot
            ax.set_title(f"{c_i.replace('_z', '').upper()}")
            ax.set_xlabel(f"{n_m}")
            ax.set_ylabel(f"{c_i}")
            ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
            ax.axvline(0, color='gray', linestyle='--', alpha=0.3)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, fontsize='small', loc='best')

        plt.suptitle(f"Brain-Behavior Associations: {n_m.replace('_z', '')} (Outliers Removed)", 
                     fontsize=24, y=1.08)
        sns.despine()
        plt.tight_layout()
        plt.show()
        plt.close(fig)
    df_ols_results = pd.DataFrame(ols_results)
    save_stage_bundle(30, "stage30_NeuralClinicalOLS", {"df_ols_results": df_ols_results})
    save_cell_results(30, ['clinical_z', 'data_subsets', 'df_ols_results', 'groups', 'importance_mask_permutated', 'importance_scores_permutated', 'meta', 'n_m', 'neural_z', 'results_11', 'results_12', 'results_13', 'results_13_2', 'results_14_self', 'results_21', 'results_21_pv', 'results_22', 'results_23', 'results_24', 'results_25', 'strict_cross_phase_results', 'sub_to_meta'])


else:
    maybe_load_cell_results(30)

# %% [cell 31]
if cell_active(31):
    # notebook display only: res_sad
    pass

else:
    maybe_load_cell_results(31)

# %% [cell 32]
if cell_active(32):
    pass
else:
    maybe_load_cell_results(32)
