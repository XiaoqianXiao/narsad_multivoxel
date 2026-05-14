# %% [cell 1]
# 1. Imports
# Standard library
import argparse
import glob
import itertools
import os
import time
from typing import Dict, List, Sequence, Union
# Core scientific/data stack
import joblib
from joblib import Parallel, delayed, dump
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from numpy.linalg import norm
from scipy import stats
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import entropy, kurtosis, pearsonr, ttest_1samp, ttest_ind
# Neuroimaging
from nilearn import image, masking, plotting
from nilearn.maskers import NiftiLabelsMasker
# Scikit-learn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.covariance import LedoitWolf
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    LeaveOneGroupOut,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    permutation_test_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils import resample, shuffle
# Statistics
from statsmodels.stats.multitest import multipletests


# 2. Configuration
RANDOM_STATE = 42
N_SPLITS = 5
INNER_CV_SPLITS = 5
CS_LABELS = ["CS-", "CSS", "CSR"]
N_JOBS = 1
N_JOBS_CV = 1
MAX_ITER = 5000
thresh_hold_p = 1 - 0.05
N_PERMUTATION = 5000
N_REPEATS = 10
CROSSNOBIS_REPEATS = 50
SUBJECT_CV_SPLITS = 5
SUBJECT_INNER_SPLITS = 3
CALIB_BINS = 5
TOP_PCT = 95
MIN_FDR_FEATURES_FOR_PRIMARY = 100
SENSITIVITY_TOP_K = 100
LOW_PCT = 5
TWO_TAIL_LOW = 2.5
TWO_TAIL_HIGH = 97.5
MIN_TRIALS_PER_SUBJECT = 10
C_MIN_EXP = -2
C_MAX_EXP = 2
C_POINTS = 20

STAGE_INTERMEDIATE_MAP = {
    1: ["stage01_NeuralDissociation"],
    2: ["stage02_StaticRepresentationalTopology"],
    3: ["stage03_DynamicRepresentationalDrift"],
    4: ["stage04_DecisionBoundaryCharacteristics"],
    5: ["stage05_SafetyRestoration"],
    6: ["stage06_DriftEfficiency"],
    7: ["stage07_ProbabilisticOpening"],
    8: ["stage08_SpatialReAlignment"],
    9: ["stage09_ReverseCrossDecoding"],
    10: ["stage10_ClinicalScores", "stage23_clinical_scores"],
    11: ["stage11_NeuralClinicalIndices", "stage24_neural_clinical_indices"],
    12: ["stage12_MasterClinicalNeural", "stage26_master_clinical_neural"],
    13: ["stage13_NeuralClinicalPearson", "stage27_neural_clinical_pearson"],
    14: ["stage14_NeuralClinicalPartial", "stage28_neural_clinical_partial"],
    15: ["stage15_NeuralClinicalZScore", "stage29_neural_clinical_zscore"],
    16: ["stage16_NeuralClinicalOLS", "stage30_neural_clinical_ols"],
}

sns.set_context("poster")

param_grid = {
    'classification__C': np.logspace(C_MIN_EXP, C_MAX_EXP, C_POINTS)
}

def parse_runtime_args():
    """Parse runtime options without interfering with notebook-style execution."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--project_root", default=os.environ.get("PROJECT_ROOT", "/gscratch/fang/NARSAD"))
    parser.add_argument("--output_dir", default=os.environ.get("OUTPUT_DIR"))
    parser.add_argument("--stage", type=int, default=None, help="Run a single logical stage (1-16).")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load checkpoints for prior cells when running a single stage.",
    )
    parser.add_argument(
        "--stage1_group",
        default=os.environ.get("STAGE1_GROUP", "ALL"),
        choices=["SAD", "HC", "ALL"],
        help="For stage 1, compute importance for SAD, HC, or ALL.",
    )
    parser.add_argument(
        "--importance_source",
        default=os.environ.get("IMPORTANCE_SOURCE", "auto"),
        choices=["auto", "combined", "group"],
        help=(
            "How to load stage 1 importance for downstream stages: "
            "'combined' uses only a combined importance joblib; "
            "'group' requires per-group SAD/HC importance files; "
            "'auto' tries combined then per-group."
        ),
    )
    parser.add_argument("--n_jobs", type=int, default=int(os.environ.get("N_JOBS", "1")))
    parser.add_argument("--n_jobs_cv", type=int, default=int(os.environ.get("N_JOBS_CV", "1")))
    parser.add_argument("--n_permutation", type=int, default=int(os.environ.get("N_PERMUTATION", "5000")))
    parser.add_argument("--n_null_perms", type=int, default=int(os.environ.get("N_NULL_PERMS", "5000")))
    parser.add_argument(
        "--stage1_actual_repeats",
        type=int,
        default=int(os.environ.get("STAGE1_ACTUAL_REPEATS", os.environ.get("N_NULL_PERMS", "5000"))),
        help="Total actual permutation-importance repeats for stage 1.",
    )
    parser.add_argument(
        "--stage1_chunk_idx",
        type=int,
        default=(
            None
            if os.environ.get("STAGE1_CHUNK_IDX", os.environ.get("SLURM_ARRAY_TASK_ID")) is None
            else int(os.environ.get("STAGE1_CHUNK_IDX", os.environ.get("SLURM_ARRAY_TASK_ID")))
        ),
        help="Zero-based stage 1 permutation-importance chunk index.",
    )
    parser.add_argument(
        "--stage1_chunk_count",
        type=int,
        default=int(os.environ.get("STAGE1_CHUNK_COUNT", "1")),
        help="Total number of stage 1 permutation-importance chunks.",
    )
    parser.add_argument(
        "--stage1_merge",
        action="store_true",
        default=os.environ.get("STAGE1_MERGE", "0") == "1",
        help="Merge stage 1 permutation-importance chunks instead of computing a chunk.",
    )
    return parser.parse_known_args()


_args, _ = parse_runtime_args()
PROJECT_ROOT = _args.project_root
OUTPUT_DIR = _args.output_dir
STAGE = _args.stage
RESUME = _args.resume
IMPORTANCE_SOURCE = _args.importance_source.lower()
N_JOBS = _args.n_jobs
N_JOBS_CV = _args.n_jobs_cv
N_PERMUTATION = _args.n_permutation
N_NULL_PERMS = _args.n_null_perms
STAGE1_ACTUAL_REPEATS = _args.stage1_actual_repeats
STAGE1_CHUNK_IDX = _args.stage1_chunk_idx
STAGE1_CHUNK_COUNT = _args.stage1_chunk_count
STAGE1_MERGE = _args.stage1_merge


def configure_blas_threads():
    """Allow BLAS to use the full allocation when Python-level parallelism is disabled."""
    if N_JOBS != 1 or N_JOBS_CV != 1:
        return
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    try:
        cpu_count = int(slurm_cpus) if slurm_cpus else (os.cpu_count() or 1)
    except ValueError:
        cpu_count = os.cpu_count() or 1
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, str(cpu_count))


def resolve_output_dirs():
    """Create and return the main output/checkpoint/intermediate directories."""
    out_dir = OUTPUT_DIR or os.path.join(
        PROJECT_ROOT,
        "MRI/derivatives/fMRI_analysis/LSS",
        "results",
        "wholebrain_parcellation",
    )
    checkpoint_dir = os.path.join(out_dir, "checkpoints")
    intermediate_dir = os.path.join(out_dir, "intermediate")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(intermediate_dir, exist_ok=True)
    return out_dir, checkpoint_dir, intermediate_dir


configure_blas_threads()
OUT_DIR_MAIN, CHECKPOINT_DIR, INTERMEDIATE_DIR = resolve_output_dirs()


# 3. Helper functions
# Runtime, persistence, and shared analysis helpers are grouped by purpose.

# Runtime persistence helpers
def _ckpt_path(cell_id: int) -> str:
    return os.path.join(CHECKPOINT_DIR, f"cell_{cell_id:02d}.joblib")


def _intermediate_path(name: str) -> str:
    return os.path.join(INTERMEDIATE_DIR, f"{name}.joblib")


def _save_result(name: str, obj) -> None:
    """Persist result objects for each analysis."""
    path = os.path.join(OUT_DIR_MAIN, f"{name}.joblib")
    try:
        dump(obj, path)
    except Exception as exc:
        print(f"  ! Failed to save {name}: {exc}")


def _save_fig(name: str) -> None:
    """Save current matplotlib figure."""
    try:
        plt.savefig(os.path.join(OUT_DIR_MAIN, f"{name}.png"), dpi=300, bbox_inches="tight")
    except Exception as exc:
        print(f"  ! Failed to save figure {name}: {exc}")


def save_checkpoint(cell_id: int, data: dict) -> None:
    """Save checkpoint data for a given analysis cell."""
    path = _ckpt_path(cell_id)
    try:
        dump(data, path)
        print(f"[Checkpoint] Saved cell {cell_id} -> {path}")
    except Exception as exc:
        print(f"[Checkpoint] Failed to save cell {cell_id}: {exc}")


def load_checkpoint(cell_id: int) -> dict:
    """Load checkpoint data for a given analysis cell."""
    path = _ckpt_path(cell_id)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing checkpoint for cell {cell_id}: {path}")
    data = joblib.load(path)
    if not isinstance(data, dict):
        raise ValueError(f"Checkpoint {path} is not a dict.")
    print(f"[Checkpoint] Loaded cell {cell_id} <- {path}")
    return data


def save_intermediate(name: str, obj) -> None:
    """Save intermediate objects for downstream stages."""
    path = _intermediate_path(name)
    try:
        dump(obj, path)
        print(f"[Intermediate] Saved {name} -> {path}")
    except Exception as exc:
        print(f"[Intermediate] Failed to save {name}: {exc}")


def load_intermediate(name: str):
    path = _intermediate_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    obj = joblib.load(path)
    print(f"[Intermediate] Loaded {name} <- {path}")
    return obj


def ensure_importance_loaded():
    """Ensure importance scores/masks are available using user-selected source."""
    global importance_scores, importance_masks, importance_mask_permutated, importance_scores_permutated
    global importance_masks_top100_positive
    if (
        "importance_scores" in globals()
        and importance_scores
        and "importance_masks" in globals()
        and importance_masks
    ):
        return importance_scores, importance_masks

    merged_scores = {}
    merged_masks = {}
    merged_p_values = {}
    merged_top100_masks = {}

    def load_combined():
        for name in (
            "stage01_importance_permutated",
            "stage08_importance",
            "stage09_permutation_masks",
        ):
            try:
                prev = load_intermediate(name)
            except FileNotFoundError:
                continue
            merged_scores.update(prev.get("importance_scores_permutated", prev.get("importance_scores", {})))
            merged_masks.update(prev.get("importance_mask_permutated", prev.get("importance_masks_permutated", prev.get("importance_masks", {}))))
            merged_p_values.update(prev.get("p_values_permutated", {}))
            merged_top100_masks.update(prev.get("importance_masks_top100_positive", prev.get("importance_mask_top100_positive", {})))
            return
        raise FileNotFoundError("No combined importance intermediate found.")

    def load_groups():
        for grp in ("SAD", "HC"):
            loaded = False
            for name in (
                f"stage01_importance_permutated_{grp}",
                f"stage08_importance_{grp}",
                f"stage09_permutation_masks_{grp}",
            ):
                try:
                    prev = load_intermediate(name)
                except FileNotFoundError:
                    continue
                merged_scores.update(prev.get("importance_scores_permutated", prev.get("importance_scores", {})))
                merged_masks.update(prev.get("importance_mask_permutated", prev.get("importance_masks_permutated", prev.get("importance_masks", {}))))
                merged_p_values.update(prev.get("p_values_permutated", {}))
                merged_top100_masks.update(prev.get("importance_masks_top100_positive", prev.get("importance_mask_top100_positive", {})))
                loaded = True
                break
            if not loaded:
                raise FileNotFoundError(f"Missing per-group importance intermediate for {grp}.")

    if IMPORTANCE_SOURCE == "combined":
        try_order = [load_combined]
    elif IMPORTANCE_SOURCE == "group":
        try_order = [load_groups]
    else:
        try_order = [load_combined, load_groups]

    for fn in try_order:
        try:
            fn()
        except FileNotFoundError:
            continue

    if not merged_scores:
        if IMPORTANCE_SOURCE == "group":
            raise FileNotFoundError(
                "Missing per-group importance intermediates. Expected "
                "stage08_importance_SAD.joblib and stage08_importance_HC.joblib."
            )
        if IMPORTANCE_SOURCE == "combined":
            raise FileNotFoundError(
                "Missing combined importance intermediate stage08_importance.joblib."
            )
        raise FileNotFoundError(
            "Missing importance intermediates. Expected stage08_importance.joblib "
            "or stage08_importance_{SAD,HC}.joblib in /intermediate."
        )

    importance_scores = merged_scores
    importance_masks = merged_masks
    importance_scores_permutated = merged_scores
    importance_mask_permutated = merged_masks
    importance_masks_top100_positive = merged_top100_masks
    if merged_p_values:
        globals()["p_values_permutated"] = merged_p_values
    return importance_scores, importance_masks


def make_top_positive_importance_mask(scores, top_k=SENSITIVITY_TOP_K):
    """Return a pre-specified top-k mask among features with positive importance."""
    scores = np.asarray(scores)
    mask = np.zeros(scores.shape, dtype=bool)
    positive_idx = np.where(scores > 0)[0]
    if positive_idx.size == 0:
        return mask
    ranked_positive = positive_idx[np.argsort(scores[positive_idx])[::-1]]
    selected = ranked_positive[:min(int(top_k), ranked_positive.size)]
    mask[selected] = True
    return mask


def get_analysis_feature_masks(label):
    """Use FDR masks when sufficiently populated, otherwise top-100 positive sensitivity masks."""
    global importance_masks_top100_positive
    if 'importance_masks' not in globals() or not importance_masks:
        ensure_importance_loaded()
    if 'importance_masks_top100_positive' not in globals() or not importance_masks_top100_positive:
        importance_masks_top100_positive = {
            grp: make_top_positive_importance_mask(scores)
            for grp, scores in importance_scores.items()
        }

    selected_masks = {}
    feature_space = {}
    for grp in ("SAD", "HC"):
        fdr_mask = np.asarray(importance_masks[grp], dtype=bool)
        fdr_n = int(np.sum(fdr_mask))
        top100_mask = np.asarray(importance_masks_top100_positive.get(grp), dtype=bool)
        top100_n = int(np.sum(top100_mask)) if top100_mask.size else 0

        if fdr_n < MIN_FDR_FEATURES_FOR_PRIMARY and top100_n > 0:
            selected_masks[grp] = top100_mask
            feature_space[grp] = {
                "source": "top100_positive_permutation_importance_sensitivity",
                "n_features": top100_n,
                "primary_fdr_n_features": fdr_n,
                "threshold": f"FDR feature count < {MIN_FDR_FEATURES_FOR_PRIMARY}",
            }
            print(
                f"  ! {label} {grp}: whole-brain FDR selected {fdr_n} features "
                f"(< {MIN_FDR_FEATURES_FOR_PRIMARY}); using pre-specified top-{SENSITIVITY_TOP_K} "
                f"positive permutation-importance sensitivity mask ({top100_n} features)."
            )
        else:
            selected_masks[grp] = fdr_mask
            feature_space[grp] = {
                "source": "whole_brain_fdr_permutation_importance",
                "n_features": fdr_n,
                "primary_fdr_n_features": fdr_n,
                "threshold": "q < 0.05 and positive importance",
            }
            print(f"  > {label} {grp}: using whole-brain FDR mask ({fdr_n} features).")

    return selected_masks, feature_space


def stage_active(stage_id: int) -> bool:
    """Return True when a logical stage should execute."""
    return STAGE is None or STAGE == stage_id


def save_stage_bundle(stage_id: int, name: str, payload: dict) -> None:
    """Save a stage-level checkpoint + intermediate bundle for exact resume."""
    save_checkpoint(stage_id, payload)
    save_intermediate(name, payload)


def calculate_crossnobis_rdm(X, y, subjects, conditions, n_repeats=CROSSNOBIS_REPEATS, random_state=RANDOM_STATE):
    unique_subs = np.unique(subjects)
    rdms, sub_ids = [], []
    rng = np.random.default_rng(random_state)
    for sub in unique_subs:
        mask_sub = subjects == sub
        X_sub = X[mask_sub]
        y_sub = y[mask_sub]
        rdm_accum = None
        valid_reps = 0
        for _ in range(n_repeats):
            means_a, means_b = {}, {}
            ok = True
            for cond in conditions:
                idx = np.where(y_sub == cond)[0]
                if len(idx) < 2:
                    ok = False
                    break
                idx = idx.copy()
                rng.shuffle(idx)
                half = len(idx) // 2
                idx_a, idx_b = idx[:half], idx[half:]
                if len(idx_a) == 0 or len(idx_b) == 0:
                    ok = False
                    break
                means_a[cond] = np.mean(X_sub[idx_a], axis=0)
                means_b[cond] = np.mean(X_sub[idx_b], axis=0)
            if not ok:
                continue
            resid = []
            for cond in conditions:
                idx = np.where(y_sub == cond)[0]
                cond_mean = np.mean(X_sub[idx], axis=0)
                resid.append(X_sub[idx] - cond_mean)
            resid = np.vstack(resid)
            cov = LedoitWolf().fit(resid).covariance_
            prec = np.linalg.pinv(cov)
            n = len(conditions)
            rdm = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    d_a = means_a[conditions[i]] - means_a[conditions[j]]
                    d_b = means_b[conditions[i]] - means_b[conditions[j]]
                    dist = float(d_a.T @ prec @ d_b)
                    rdm[i, j] = dist
                    rdm[j, i] = dist
            rdm_accum = rdm if rdm_accum is None else (rdm_accum + rdm)
            valid_reps += 1
        if valid_reps == 0:
            continue
        rdms.append(rdm_accum / valid_reps)
        sub_ids.append(sub)
    return np.array(rdms), np.array(sub_ids)


def extract_topology_metrics(rdms, idx_cs_minus=0, idx_css=1, idx_csr=2):
    return rdms[:, idx_csr, idx_css], rdms[:, idx_css, idx_cs_minus]


def one_sample_distance_test(data, name):
    t_val, p_val = ttest_1samp(data, 0, alternative='greater')
    sig = "*" if p_val < 0.05 else "ns"
    print(f"  > {name}: Mean={np.mean(data):.3f}, t={t_val:.3f}, p={p_val:.4f} ({sig})")
    return p_val


def get_sig_star(p):
    return "*" if p < 0.05 else "ns"


def clinical_group_column(df):
    """Return the most reliable group column after clinical/neural/meta merges."""
    for col in ("Analysis_Group", "Group", "Group_x", "Group_y"):
        if col in df.columns:
            return col
    return None


def partial_corr_residualized(df, x_col, y_col, covariates):
    """Partial correlation via residualization; avoids requiring pingouin on the cluster."""
    cols = [x_col, y_col] + [c for c in covariates if c in df.columns]
    valid = df[cols].dropna().apply(pd.to_numeric, errors="coerce").dropna()
    if len(valid) <= len(covariates) + 2:
        return np.nan, np.nan, len(valid)
    if not covariates:
        r_val, p_val = pearsonr(valid[x_col], valid[y_col])
        return r_val, p_val, len(valid)

    covars = [c for c in covariates if c in valid.columns]
    if not covars:
        r_val, p_val = pearsonr(valid[x_col], valid[y_col])
        return r_val, p_val, len(valid)

    design = sm.add_constant(valid[covars], has_constant="add")
    x_resid = sm.OLS(valid[x_col], design).fit().resid
    y_resid = sm.OLS(valid[y_col], design).fit().resid
    r_val, p_val = pearsonr(x_resid, y_resid)
    return r_val, p_val, len(valid)


NEURAL_CLINICAL_METRICS = [
    "Neural_Dist_Threat_Safety",
    "Neural_Dist_Safety_Backgr",
    "Neural_Rigidity_Slope",
    "Neural_Safety_Mean",
    "Neural_Uncertainty_Entropy",
    "Neural_Sharpness_Kurtosis",
]
CLINICAL_INDICES = ["lsas_total", "lsas_fear", "lsas_avoid", "dass_anxiety", "dass_stress", "ecr_total"]
CLINICAL_COVARIATES = ["demo_age"]


def calculate_plasticity_vectors(X_learn, y_learn, sub_learn, X_targ, y_targ, sub_targ, mask, cond_l, cond_t):
    """Calculate masked representational alignment between learning and target states."""
    unique_subs = np.intersect1d(np.unique(sub_learn), np.unique(sub_targ))
    res = {'sub': [], 'projection': [], 'cosine': [], 'init_dist': []}
    for sub in unique_subs:
        mask_l = (sub_learn == sub) & (y_learn == cond_l)
        mask_t = (sub_targ == sub) & (y_targ == cond_t)
        if np.sum(mask_l) == 0 or np.sum(mask_t) == 0:
            continue

        vec_l = np.mean(X_learn[mask_l][:, mask], axis=0)
        vec_t = np.mean(X_targ[mask_t][:, mask], axis=0)
        norm_l = norm(vec_l)
        norm_t = norm(vec_t)
        if norm_l == 0 or norm_t == 0:
            continue

        dot_prod = np.dot(vec_l, vec_t)
        res['sub'].append(sub)
        res['projection'].append(dot_prod / norm_t)
        res['cosine'].append(dot_prod / (norm_l * norm_t))
        res['init_dist'].append(norm(vec_l - vec_t))
    return pd.DataFrame(res)


def get_group_for_sub(sub_id, meta_map, best_c_sad=None, best_c_hc=None):
    s_str = str(sub_id).strip()
    conds = meta_map.get(s_str) or meta_map.get(f"sub-{s_str}") or meta_map.get(s_str.replace("sub-", ""))
    if conds:
        return conds.get("Group")
    return None


def get_default_c_for_sub(sub_id, meta_map, best_c_sad=None, best_c_hc=None):
    group = get_group_for_sub(sub_id, meta_map, best_c_sad, best_c_hc)
    if group == "SAD" and best_c_sad is not None:
        return float(best_c_sad)
    if group == "HC" and best_c_hc is not None:
        return float(best_c_hc)
    return 1.0


def calculate_distribution_stats(X, y, subjects, feature_mask, best_params_dict, cond_threat, cond_safe, meta_map, best_c_sad=None, best_c_hc=None):
    X_masked = X[:, feature_mask]
    unique_subs = np.unique(subjects)
    res = {'sub': [], 'entropy': [], 'kurtosis': [], 'variance': [], 'probabilities': [], 'brier': [], 'calib': []}
    for sub in unique_subs:
        c_val = best_params_dict.get(sub, get_default_c_for_sub(sub, meta_map, best_c_sad, best_c_hc))
        mask_sub = subjects == sub
        X_sub = X_masked[mask_sub]
        y_sub = y[mask_sub]
        mask_binary = np.isin(y_sub, [cond_threat, cond_safe])
        X_binary = X_sub[mask_binary]
        y_binary = y_sub[mask_binary]
        if len(y_binary) < 10:
            continue
        try:
            fixed_model = build_binary_pipeline()
            fixed_model.set_params(classification__C=c_val)
            calib_model = CalibratedClassifierCV(fixed_model, method="sigmoid", cv=3)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            probs_all = cross_val_predict(calib_model, X_binary, y_binary, cv=cv, method='predict_proba', n_jobs=1)
            classes = sorted(np.unique(y_binary))
            if cond_threat not in classes:
                continue
            idx_threat = classes.index(cond_threat)
            probs_css = probs_all[y_binary == cond_safe, idx_threat]
            if len(probs_css) == 0:
                continue
            y_bin_threat = (y_binary == cond_threat).astype(int)
            brier = brier_score_loss(y_bin_threat, probs_all[:, idx_threat])
            frac_pos, mean_pred = calibration_curve(y_bin_threat, probs_all[:, idx_threat], n_bins=CALIB_BINS, strategy='uniform')
            p_clean = np.clip(probs_css, 1e-9, 1 - 1e-9)
            trial_entropies = [entropy([p, 1 - p], base=2) for p in p_clean]
            res['sub'].append(sub)
            res['entropy'].append(np.mean(trial_entropies))
            res['kurtosis'].append(kurtosis(probs_css, fisher=True))
            res['variance'].append(np.var(probs_css))
            res['probabilities'].append(probs_css)
            res['brier'].append(brier)
            res['calib'].append({'frac_pos': frac_pos, 'mean_pred': mean_pred})
        except Exception:
            pass
    return pd.DataFrame(res)


def get_ext_data(group_key):
    if group_key not in data_subsets or data_subsets[group_key]['ext'] is None:
        raise ValueError(f"Extinction data for {group_key} missing.")
    d = data_subsets[group_key]['ext']
    return d["X"], d["y"], d["sub"]


def calc_drift_metrics(X_start_phase, y_start_phase, X_tgt_phase, y_tgt_phase, cond_start, cond_target, mask):
    X_s = X_start_phase[:, mask]
    X_t = X_tgt_phase[:, mask]
    mask_tgt = y_tgt_phase == cond_target
    if np.sum(mask_tgt) < 2:
        return None
    P_target = np.mean(X_t[mask_tgt], axis=0)
    idx_lrn = np.where(y_start_phase == cond_start)[0]
    if len(idx_lrn) < 4:
        return None
    cutoff = len(idx_lrn) // 2
    P_start = np.mean(X_s[idx_lrn[:cutoff]], axis=0)
    P_end = np.mean(X_s[idx_lrn[cutoff:]], axis=0)
    V_axis = P_target - P_start
    V_drift = P_end - P_start
    nA, nD = norm(V_axis), norm(V_drift)
    if nA == 0 or nD == 0:
        return None
    dot = np.dot(V_drift, V_axis)
    return {'Cosine': dot / (nA * nD), 'Projection': dot / nA}


def calc_metrics_for_subject(X, y, sub_id, feature_mask, cond_threat, cond_safe):
    X_m = X[:, feature_mask]
    mask_bin = np.isin(y, [cond_threat, cond_safe])
    X_bin, y_bin = X_m[mask_bin], y[mask_bin]
    if len(y_bin) < 10:
        return None
    try:
        outer_cv = get_cv(y_bin, np.full(len(y_bin), sub_id), n_splits=SUBJECT_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        inner_cv = get_cv(y_bin, np.full(len(y_bin), sub_id), n_splits=SUBJECT_INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        probs_all = np.zeros((len(y_bin), 2))
        for train_idx, test_idx in outer_cv.split(X_bin, y_bin, groups=np.full(len(y_bin), sub_id)):
            gs = GridSearchCV(build_binary_pipeline(), param_grid, cv=inner_cv, scoring=forced_choice_scorer, n_jobs=N_JOBS_CV)
            gs.fit(X_bin[train_idx], y_bin[train_idx], groups=np.full(len(train_idx), sub_id))
            calib_model = CalibratedClassifierCV(gs.best_estimator_, method="sigmoid", cv=3)
            calib_model.fit(X_bin[train_idx], y_bin[train_idx])
            probs_all[test_idx] = calib_model.predict_proba(X_bin[test_idx])
        classes = sorted(np.unique(y_bin))
        if cond_threat not in classes:
            return None
        idx_threat = classes.index(cond_threat)
        probs_css = probs_all[y_bin == cond_safe, idx_threat]
        if len(probs_css) == 0:
            return None
        p_clean = np.clip(probs_css, 1e-9, 1 - 1e-9)
        ents = [entropy([p, 1 - p], base=2) for p in p_clean]
        return {'Entropy': np.mean(ents), 'Kurtosis': kurtosis(probs_css, fisher=True), 'Variance': np.var(probs_css)}
    except Exception:
        return None

def build_binary_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ('classification', LogisticRegression(
            penalty='l2', 
            solver='lbfgs', 
            class_weight='balanced', 
            max_iter=MAX_ITER, 
            random_state=RANDOM_STATE, 
            n_jobs=N_JOBS_CV
        ))
    ])


def get_cv(y, groups=None, n_splits=SUBJECT_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE):
    """Return StratifiedGroupKFold if multiple groups exist; otherwise StratifiedKFold."""
    if groups is None or len(np.unique(groups)) < 2:
        return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=(random_state if shuffle else None))
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=(random_state if shuffle else None))


def get_top_percentile_mask(scores, percentile):
    thresh = np.percentile(scores, percentile)
    return (scores >= thresh) & (scores > 0)


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
    forced-choice accuracy.
    
    Parameters:
    -----------
    X : array-like, feature matrix
    y : array-like, labels
    groups : array-like, group labels (subjects)
    n_iters : int, number of permutations to run in this batch
    
    Returns:
    --------
    scores : list of mean accuracy scores
    """
    scores = []
    y_shuffled = y.copy()
    
    # Use the same pipeline builder and CV splitter as your main analysis
    pipe = build_binary_pipeline()
    cv = get_cv(y, groups, n_splits=N_SPLITS, shuffle=False)
    
    for i in range(n_iters):
        if i == 0 or (i + 1) % 100 == 0:
            print(f"    perm {i + 1}/{n_iters}")
        # 1. Shuffle labels randomly (breaking the relationship between X and y)
        np.random.shuffle(y_shuffled)
        
        # 2. Cross-validated accuracy on shuffled labels
        cv_scores = cross_val_score(
            pipe,
            X,
            y_shuffled,
            groups=groups,
            cv=cv,
            scoring=forced_choice_scorer,
            n_jobs=N_JOBS_CV
        )
        scores.append(float(np.mean(cv_scores)))
        
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
        # PHASE 1: EVALUATION (Repeated Nested CV with Forced-Choice Accuracy)
        # ---------------------------------------------------------------------
        all_repeat_scores = []
        
        for r in range(n_repeats):
            # Use a different random_state for each repeat to get different splits
            # Important: shuffle=True is required for the seed to change the split
            gkf_outer = get_cv(y_pair, sub_pair, n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + r)
            
            repeat_scores = []
            print(f"  > Repeat {r+1}/{n_repeats}...")
            
            for i, (train_idx, test_idx) in enumerate(gkf_outer.split(X_pair, y_pair, groups=sub_pair), 1):
                cv_inner = get_cv(y_pair[train_idx], sub_pair[train_idx], n_splits=INNER_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE + r)
                # Inner loop for hyperparameter tuning
                gs = GridSearchCV(build_binary_pipeline(), param_grid, cv=cv_inner, scoring=forced_choice_scorer, n_jobs=N_JOBS)
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
        print(f"  > Final Mean Forced-Choice Accuracy ({n_repeats} repeats): {avg_cv_acc:.4f} (+/- {std_cv_acc:.4f})")

        # ---------------------------------------------------------------------
        # PHASE 2: MODEL GENERATION (Refit on Full Data)
        # ---------------------------------------------------------------------
        # For the final model, we still refit once using a stable inner CV
        print("  > Generating final model (Refit on full data for Haufe patterns)...")
        cv_inner_final = get_cv(y_pair, sub_pair, n_splits=INNER_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        gs_final = GridSearchCV(build_binary_pipeline(), param_grid, cv=cv_inner_final, scoring=forced_choice_scorer, n_jobs=N_JOBS)
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
            'best_C': gs_final.best_params_['classification__C'], 
            'haufe_pattern': A.flatten(), 
            'classes': final_model.classes_
        }
    return results


def plot_dist_with_thresh(null_dist, obs_val, ax, title, tail='upper', color='gray'):
    sns.histplot(null_dist, color='gray', stat='density', kde=True, alpha=0.4, ax=ax, label='Null Dist')
    ax.axvline(obs_val, color='red', lw=2.5, label=f'Obs: {obs_val:.2f}')
    if tail == 'upper':
        thresh = np.percentile(null_dist, TOP_PCT); ax.axvline(thresh, color='blue', ls='--', lw=2, label=f'95%: {thresh:.2f}'); p_val = np.mean(null_dist >= obs_val)
    elif tail == 'lower':
        thresh = np.percentile(null_dist, LOW_PCT); ax.axvline(thresh, color='blue', ls='--', lw=2, label=f'5%: {thresh:.2f}'); p_val = np.mean(null_dist <= obs_val)
    elif tail == 'two-tailed':
        t_low = np.percentile(null_dist, TWO_TAIL_LOW); t_high = np.percentile(null_dist, TWO_TAIL_HIGH)
        ax.axvline(t_low, color='blue', ls='--', lw=2); ax.axvline(t_high, color='blue', ls='--', lw=2)
        p_val = 2 * min(np.mean(null_dist <= obs_val), np.mean(null_dist >= obs_val))
    ax.set_title(f"{title}\n(p = {p_val:.4f})"); ax.legend(loc='best', fontsize='small')
    return p_val


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


def process_phase_data(X_all, y_all, sub_all, phase_name):
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
        
        # 5. FILTER CONDITIONS (Keep only CSS / CSR)
        mask_cond = np.isin(y_sub_full, ["CSS", "CSR"])
        
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


def compute_haufe_binary_robust(model, X):
    scores = model.decision_function(X)
    return np.dot((X - np.mean(X, axis=0)).T, scores - np.mean(scores)) / (X.shape[0] - 1)


def get_robust_weights(X, y, subjects, pipeline, n_boot=10):
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
    scores = estimator.decision_function(X)
    return compute_forced_choice_accuracy(y, scores, estimator.classes_)


def compute_pairwise_forced_choice(y_true, scores, class_labels):
    """Forced-choice accuracy given decision-score columns per class."""
    scores_arr = np.asarray(scores)
    return compute_forced_choice_accuracy(y_true, scores_arr, class_labels)


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
    
    for i in range(n_perm):
        if i == 0 or (i + 1) % 500 == 0:
            print(f"    perm {i + 1}/{n_perm}")
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


#--- Function: get_phase_data ---
def get_phase_data(group, phase):
    try:
        d = data_subsets[group][phase]
        if d is None: return None, None, None
        return d["X"], d["y"], d["sub"]
    except KeyError:
        return None, None, None



def tag_df(df, grp, cond):
    if df.empty: return df
    d = df.copy(); d['Group'] = grp; d['Condition'] = cond
    return d


def calc_trajectory(X_learn, y_learn, sub_learn, X_targ, y_targ, sub_targ, mask, cond_l, cond_t):
    """Project individual trials onto the axis from early learning to target centroid."""
    unique_subs = np.intersect1d(np.unique(sub_learn), np.unique(sub_targ))
    res = {'sub': [], 'trial': [], 'score': []}
    
    for sub in unique_subs:
        mask_sub_l = (sub_learn == sub) & (y_learn == cond_l)
        mask_sub_t = (sub_targ == sub) & (y_targ == cond_t)
        if np.sum(mask_sub_l) < 2 or np.sum(mask_sub_t) == 0:
            continue
        
        xl = X_learn[mask_sub_l][:, mask]
        xt = X_targ[mask_sub_t][:, mask]
        
        half_idx = len(xl) // 2
        vec_start = np.mean(xl[:half_idx], axis=0)
        vec_target = np.mean(xt, axis=0)
        axis_vec = vec_target - vec_start
        axis_norm = norm(axis_vec)
        if axis_norm == 0:
            continue

        for i, trial_vec in enumerate(xl):
            score = np.dot(trial_vec - vec_start, axis_vec) / (axis_norm**2)
            res['sub'].append(sub)
            res['trial'].append(i + 1)
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


def get_significant_mask(scores): return scores > 0


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

# 4. Stage-wise execution
# Load phase2 (extinction) and phase3 (reinstatement) data
# Whole-brain parcellation (Schaefer + Tian) feature space.

print("--- Data Loading (Whole-Brain Parcellation) ---")

project_root = PROJECT_ROOT
data_root = os.path.join(
    project_root,
    "MRI/derivatives/fMRI_analysis/LSS",
    "firstLevel",
    "all_subjects/group_level"
)
phase2_npz_path = os.path.join(data_root, "phase2_X_ext_y_ext_voxels_schaefer_tian.npz")
phase3_npz_path = os.path.join(data_root, "phase3_X_reinst_y_reinst_voxels_schaefer_tian.npz")

phase2_npz = np.load(phase2_npz_path, allow_pickle=True)
phase3_npz = np.load(phase3_npz_path, allow_pickle=True)

print("Phase2 npz keys:", phase2_npz.files)
print("Phase3 npz keys:", phase3_npz.files)

# ---- Adjust these key names if your .npz uses different ones ----
X_ext = phase2_npz["X_ext"]         # (n_trials_phase2, n_parcels)
y_ext = phase2_npz["y_ext"]         # (n_trials_phase2,)
sub_ext = phase2_npz["subjects"]    # (n_trials_phase2,)

X_reinst = phase3_npz["X_reinst"]       # (n_trials_phase3, n_parcels)
y_reinst = phase3_npz["y_reinst"]       # (n_trials_phase3,)
sub_reinst = phase3_npz["subjects"]     # (n_trials_phase3,)

parcel_names_ext = list(phase2_npz["parcel_names"])
parcel_names_reinst = list(phase3_npz["parcel_names"])

print("Phase2 shapes:", X_ext.shape, y_ext.shape, sub_ext.shape)
print("Phase3 shapes:", X_reinst.shape, y_reinst.shape, sub_reinst.shape)

# ---- Filter for CS Trials Only ----
if "CS_LABELS" not in locals():
    CS_LABELS = ["CS-", "CSS", "CSR"]

mask_ext = np.isin(y_ext, CS_LABELS)
mask_reinst = np.isin(y_reinst, CS_LABELS)

X_ext = X_ext[mask_ext]
y_ext = y_ext[mask_ext]
sub_ext = sub_ext[mask_ext]

X_reinst = X_reinst[mask_reinst]
y_reinst = y_reinst[mask_reinst]
sub_reinst = sub_reinst[mask_reinst]

print("\nAfter CS filtering:")
print("Phase2:", X_ext.shape, np.unique(y_ext, return_counts=True))
print("Phase3:", X_reinst.shape, np.unique(y_reinst, return_counts=True))

# Load subject-level metadata (Group, Drug, etc.)

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

# Data Preparation & Subsetting (Optimized: Center -> Filter)
# Task: 1. Split data by subject.
#       2. Center FULL subject data (to preserve true baseline).
#       3. Filter for CSS/CSR conditions.
#       4. Organize into Groups (SAD/HC).

print("--- Data Preparation & Subsetting (Center -> Filter) ---")
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
ext_subsets = process_phase_data(X_ext, y_ext, sub_ext, "Extinction")
rst_subsets = process_phase_data(X_rein, y_rein, sub_rein, "Reinstatement")

# Structure Results
data_subsets = {}
for key in group_keys:
    data_subsets[key] = {
        "ext": ext_subsets.get(key),
        "rst": rst_subsets.get(key)
    }

print("\nProcessing Complete. Data is Centered (Full-Session) and Filtered.")

# Resume support: load prior logical stages if running a single stage
if RESUME and STAGE is not None:
    for _cid in range(1, STAGE):
        loaded = False
        for _name in STAGE_INTERMEDIATE_MAP.get(_cid, []):
            try:
                _obj = load_intermediate(_name)
                if isinstance(_obj, dict):
                    globals().update(_obj)
                else:
                    globals()[_name] = _obj
                loaded = True
            except FileNotFoundError:
                continue
        if not loaded:
            try:
                _ckpt = load_checkpoint(_cid)
                globals().update(_ckpt)
            except FileNotFoundError:
                print(f"[Resume] Skipping missing checkpoint for cell {_cid}.")

    # If the exact same logical stage already exists, reuse it and skip recomputation.
    _current_stage_loaded = False
    for _name in STAGE_INTERMEDIATE_MAP.get(STAGE, []):
        try:
            _obj = load_intermediate(_name)
            if isinstance(_obj, dict):
                globals().update(_obj)
            else:
                globals()[_name] = _obj
            _current_stage_loaded = True
        except FileNotFoundError:
            continue
    if _current_stage_loaded:
        print(f"[Resume] Reused saved outputs for logical stage {STAGE}; skipping recomputation.")
        raise SystemExit(0)


if stage_active(1):
    # Analysis 1.1 - Neural Dissociation Execution
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
    
    # =============================================================================
    # 0. Data Slicing
    # =============================================================================
    # Load Data
    try:
        X_hc, y_hc, sub_hc = get_extinction_data("HC_Placebo")
        X_sad, y_sad, sub_sad = get_extinction_data("SAD_Placebo")
        print(f"Data Loaded: SAD (n={len(np.unique(sub_sad))}), HC (n={len(np.unique(sub_hc))})")
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
    iters_per_job = N_PERMUTATION // N_JOBS
    perm_acc_sad = np.concatenate(Parallel(n_jobs=N_JOBS)(delayed(run_perm_simple)(X_sad, y_sad, sub_sad, iters_per_job) for _ in range(N_JOBS)))
    perm_acc_hc = np.concatenate(Parallel(n_jobs=N_JOBS)(delayed(run_perm_simple)(X_hc, y_hc, sub_hc, iters_per_job) for _ in range(N_JOBS)))
    
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
        delayed(run_cross_perm)(model_sad, X_hc, y_hc, sub_hc, iters_per_job) for _ in range(N_JOBS)))
    p_sad2hc = np.mean(perm_sad2hc >= mean_sad2hc)
    
    perm_hc2sad = np.concatenate(Parallel(n_jobs=N_JOBS)(
        delayed(run_cross_perm)(model_hc, X_sad, y_sad, sub_sad, iters_per_job) for _ in range(N_JOBS)))
    p_hc2sad = np.mean(perm_hc2sad >= mean_hc2sad)
    
    # =============================================================================
    # TEST 3: Spatial Specificity
    # =============================================================================
    print("\n--- TEST 3: Spatial Specificity ---")
    map_sad, map_hc = res_sad['haufe_pattern'], res_hc['haufe_pattern']
    
    # Prepare Data for Subject-Level Haufe Maps
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
    
    all_sub_maps = np.array(all_sub_maps)
    all_sub_groups = np.array(all_sub_groups)
    
    # Observed Similarity: Mean SAD vs Mean HC (Subject-Level Maps)
    w_sad = np.mean(all_sub_maps[all_sub_groups == "SAD"], axis=0)
    w_hc = np.mean(all_sub_maps[all_sub_groups == "HC"], axis=0)
    obs_sim = cosine_similarity(w_sad.reshape(1, -1), w_hc.reshape(1, -1))[0][0]
    
    # Run Spatial Permutation (Group-Label Shuffle on Subject Maps)
    print(f"Running Spatial Permutation ({N_PERMUTATION} iter)...")
    perm_sims = np.array(Parallel(n_jobs=N_JOBS)(
        delayed(run_spatial_perm)(i, all_sub_maps, all_sub_groups) for i in range(N_PERMUTATION)
    ))
    
    p_sim_spatial = 2 * min(np.mean(perm_sims <= obs_sim), np.mean(perm_sims >= obs_sim))
    
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
    _save_fig("analysis_11_neural_dissociation")
    _save_fig("results_11_neural_dissociation")
    plt.show()
    
    # Save Results
    results_11 = {
        "acc_sad_cv": res_sad['accuracy'], 
        "p_sad": p_sad, 
        "acc_hc_cv": res_hc['accuracy'], 
        "p_hc": p_hc, 
        "func_matrix": func_matrix, 
        "sim_spatial": obs_sim, 
        "p_sim": p_sim_spatial,
        "map_sad": map_sad, 
        "map_hc": map_hc
    }
    _save_result("results_11", results_11)
    _save_result("results_11", results_11)
    
    save_checkpoint(1, {
        "res_sad_dict": res_sad_dict,
        "res_hc_dict": res_hc_dict,
        "res_sad": res_sad,
        "res_hc": res_hc,
        "best_c_sad": best_c_sad,
        "best_c_hc": best_c_hc,
        "X_sad": X_sad, "y_sad": y_sad, "sub_sad": sub_sad,
        "X_hc": X_hc, "y_hc": y_hc, "sub_hc": sub_hc,
        "func_matrix": func_matrix,
        "p_sad": p_sad, "p_hc": p_hc,
        "p_sad2hc": p_sad2hc, "p_hc2sad": p_hc2sad,
        "accs_sad2hc": accs_sad2hc, "accs_hc2sad": accs_hc2sad,
        "mean_sad2hc": mean_sad2hc, "mean_hc2sad": mean_hc2sad,
        "perm_acc_sad": perm_acc_sad, "perm_acc_hc": perm_acc_hc,
        "perm_sad2hc": perm_sad2hc, "perm_hc2sad": perm_hc2sad,
        "map_sad": map_sad, "map_hc": map_hc,
        "obs_sim": obs_sim, "p_sim_spatial": p_sim_spatial,
    })
    save_intermediate("stage06_models", {
        "res_sad_dict": res_sad_dict,
        "res_hc_dict": res_hc_dict,
        "res_sad": res_sad,
        "res_hc": res_hc,
        "best_c_sad": best_c_sad,
        "best_c_hc": best_c_hc,
    })
    save_intermediate("stage06_permutation", {
        "perm_acc_sad": perm_acc_sad, "perm_acc_hc": perm_acc_hc,
        "perm_sad2hc": perm_sad2hc, "perm_hc2sad": perm_hc2sad,
        "p_sad": p_sad, "p_hc": p_hc, "p_sad2hc": p_sad2hc, "p_hc2sad": p_hc2sad,
    })
    save_intermediate("stage01_maps", {"map_sad": map_sad, "map_hc": map_hc, "obs_sim": obs_sim, "p_sim_spatial": p_sim_spatial})
    
# %% importance
##Hauf score
    
## permuted importance
if stage_active(1):
    # Cell 8: Feature Importance (Permutation) & Mask Generation
    # Objective: Identify task-relevant voxels/regions using Permutation Importance.
    # Context: Used as the primary feature selector for downstream analysis (Cell 9 & 10).
    
    print("--- Generating Permutation Importance Masks ---")
    # =============================================================================
    # 0. Setup & Dependency Checks
    # =============================================================================
    # 1. Check for Results (Models) from Cell 6
    if 'res_sad' not in locals() or 'res_hc' not in locals():
        raise ValueError("Models ('res_sad', 'res_hc') not found. Please run Cell 6 first.")
    
    # 2. Check for Data (Global or Nested)
    # We ensure X_sad/X_hc are available, reloading from data_subsets if necessary.
    if 'X_sad' not in locals():
        print("  > Reloading extinction data from 'data_subsets'...")
        try:
            X_sad = data_subsets["SAD_Placebo"]["ext"]["X"]
            y_sad = data_subsets["SAD_Placebo"]["ext"]["y"]
            sub_sad = data_subsets["SAD_Placebo"]["ext"]["sub"]
            
            X_hc = data_subsets["HC_Placebo"]["ext"]["X"]
            y_hc = data_subsets["HC_Placebo"]["ext"]["y"]
            sub_hc = data_subsets["HC_Placebo"]["ext"]["sub"]
        except (KeyError, TypeError):
            raise ValueError("Data missing. Please run Cell 5 (Data Prep).")
    
    # 3. Check for ROI Labels
    if 'parcel_names_ext' not in locals():
        print("  ! WARNING: 'parcel_names_ext' not found. Using generic feature indices for plotting.")
        parcel_names_ext = [f"Feat_{i}" for i in range(X_sad.shape[1])]
    
    # Settings: mirror MemoryFearNetwork's empirical permutation-importance
    # procedure, but use whole-brain BH-FDR for Schaefer feature selection.
    target_pair = ['CSR', 'CSS']
    ALPHA_LEVEL = 0.05
    PERCENTILE_THRESH = None
    importance_masks = {}
    importance_scores = {}
    importance_masks_top100_positive = {}
    feature_space_reports = {}
    p_values_permutated = {}
    q_values_permutated = {}
    stage1_group = _args.stage1_group.upper()
    stage1_groups = ['SAD', 'HC'] if stage1_group == "ALL" else [stage1_group]
    stage1_chunk_dir = os.path.join(CHECKPOINT_DIR, "stage01_importance_chunks")
    os.makedirs(stage1_chunk_dir, exist_ok=True)

    # If resuming, merge any existing stage 1 intermediates (SAD/HC)
    if RESUME:
        try:
            prev = load_intermediate("stage01_importance")
            if isinstance(prev, dict):
                importance_masks.update(prev.get("importance_masks", {}))
                importance_scores.update(prev.get("importance_scores", {}))
                print(f"  > Loaded existing stage01_importance for merge: {list(importance_scores.keys())}")
        except FileNotFoundError:
            pass

    def stage1_bounds(total, chunk_idx, chunk_count):
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

    def stage1_prepare_group(group_name):
        if group_name == "SAD":
            mask_cls = np.isin(y_sad, target_pair)
            return X_sad[mask_cls], y_sad[mask_cls], res_sad['model']
        mask_cls = np.isin(y_hc, target_pair)
        return X_hc[mask_cls], y_hc[mask_cls], res_hc['model']

    def stage1_chunk_path(group_name, chunk_idx):
        return os.path.join(stage1_chunk_dir, f"stage01_{group_name}_chunk_{int(chunk_idx):04d}.joblib")

    def stage1_save_group(group_name, actual_imp, p_values, null_n):
        _, q_values, _, _ = multipletests(p_values, alpha=ALPHA_LEVEL, method='fdr_bh')
        sig_mask = (q_values < ALPHA_LEVEL) & (actual_imp > 0)
        top100_mask = make_top_positive_importance_mask(actual_imp, SENSITIVITY_TOP_K)
        fdr_n = int(np.sum(sig_mask))
        top100_n = int(np.sum(top100_mask))
        fallback_recommended = fdr_n < MIN_FDR_FEATURES_FOR_PRIMARY
        payload = {
            "importance_mask_permutated": {group_name: sig_mask},
            "importance_masks_permutated": {group_name: sig_mask},
            "importance_masks_top100_positive": {group_name: top100_mask},
            "importance_scores_permutated": {group_name: actual_imp},
            "p_values_permutated": {group_name: p_values},
            "q_values_permutated": {group_name: q_values},
            "null_permutations": {group_name: int(null_n)},
            "actual_repeats": {group_name: int(STAGE1_ACTUAL_REPEATS)},
            "fdr_feature_counts": {group_name: fdr_n},
            "top100_positive_feature_counts": {group_name: top100_n},
            "fallback_sensitivity_recommended": {group_name: fallback_recommended},
            "fallback_sensitivity_rule": (
                f"Use top-{SENSITIVITY_TOP_K} positive permutation-importance mask "
                f"when whole-brain FDR selects fewer than {MIN_FDR_FEATURES_FOR_PRIMARY} features."
            ),
            "fdr_method": "fdr_bh_whole_brain",
        }
        group_ckpt = os.path.join(CHECKPOINT_DIR, f"stage01_importance_{group_name}.joblib")
        group_intermediate = _intermediate_path(f"stage01_importance_permutated_{group_name}")
        joblib.dump(payload, group_ckpt)
        joblib.dump(payload, group_intermediate)
        importance_masks[group_name] = sig_mask
        importance_masks_top100_positive[group_name] = top100_mask
        importance_scores[group_name] = actual_imp
        p_values_permutated[group_name] = p_values
        q_values_permutated[group_name] = q_values
        feature_space_reports[group_name] = {
            "fdr_n_features": fdr_n,
            "top100_positive_n_features": top100_n,
            "fallback_sensitivity_recommended": fallback_recommended,
        }
        print(
            f"   > {group_name}: {fdr_n} whole-brain FDR-significant "
            f"features (q < {ALPHA_LEVEL}, positive importance)."
        )
        if fallback_recommended:
            print(
                f"   ! {group_name}: FDR selected fewer than {MIN_FDR_FEATURES_FOR_PRIMARY} features; "
                f"pre-specified top-{SENSITIVITY_TOP_K} positive-importance sensitivity mask has "
                f"{top100_n} features."
            )

    def stage1_compute_chunk(group_name):
        chunk_idx = 0 if STAGE1_CHUNK_IDX is None else int(STAGE1_CHUNK_IDX)
        chunk_count = max(1, int(STAGE1_CHUNK_COUNT))
        actual_start, actual_end = stage1_bounds(STAGE1_ACTUAL_REPEATS, chunk_idx, chunk_count)
        null_start, null_end = stage1_bounds(N_NULL_PERMS, chunk_idx, chunk_count)
        actual_repeats = actual_end - actual_start
        null_repeats = null_end - null_start
        if actual_repeats <= 0 and null_repeats <= 0:
            print(f"  [SKIP] {group_name} chunk {chunk_idx + 1}/{chunk_count} has no work.")
            return

        X_target, y_target, model_template = stage1_prepare_group(group_name)
        print(
            f"--- Stage 1 importance chunk {chunk_idx + 1}/{chunk_count} for {group_name}: "
            f"actual repeats {actual_start}:{actual_end}, null perms {null_start}:{null_end} ---"
        )

        actual_sum = np.zeros(X_target.shape[1], dtype=np.float64)
        if actual_repeats > 0:
            actual_res = permutation_importance(
                model_template,
                X_target,
                y_target,
                n_repeats=actual_repeats,
                scoring=forced_choice_scorer,
                n_jobs=N_JOBS,
                random_state=RANDOM_STATE + actual_start,
            )
            actual_sum = np.asarray(actual_res.importances_mean, dtype=np.float64) * actual_repeats

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
                scoring=forced_choice_scorer,
                n_jobs=N_JOBS,
                random_state=RANDOM_STATE + 100000 + perm_idx,
            )
            null_dist[row, :] = np.asarray(null_res.importances_mean, dtype=np.float32)
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
        }
        out_path = stage1_chunk_path(group_name, chunk_idx)
        joblib.dump(chunk_payload, out_path, compress=3)
        print(f"  [SAVE] Stage 1 importance chunk saved -> {out_path}")

        if chunk_count == 1:
            actual_imp = actual_sum / max(1, actual_repeats)
            p_values = (np.sum(null_dist >= actual_imp, axis=0) / max(1, null_repeats)).astype(np.float64)
            stage1_save_group(group_name, actual_imp, p_values, null_repeats)

    def stage1_merge_group(group_name):
        paths = sorted(glob.glob(os.path.join(stage1_chunk_dir, f"stage01_{group_name}_chunk_*.joblib")))
        if not paths:
            raise FileNotFoundError(f"No stage 1 importance chunk files found for {group_name} in {stage1_chunk_dir}")
        print(f"--- Stage 1 importance merge for {group_name}: {len(paths)} chunk files ---")
        actual_sum = None
        actual_n = 0
        null_n = 0
        count_ge = None
        chunks_seen = set()

        for path in paths:
            payload = joblib.load(path)
            chunks_seen.add(int(payload["chunk_idx"]))
            chunk_actual = np.asarray(payload["actual_importance_sum"], dtype=np.float64)
            actual_sum = chunk_actual if actual_sum is None else actual_sum + chunk_actual
            actual_n += int(payload["actual_repeats"])

        if actual_sum is None or actual_n == 0:
            raise ValueError(f"No actual importance repeats found for {group_name}.")
        actual_imp = actual_sum / actual_n

        for path in paths:
            payload = joblib.load(path)
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
                f"Only found {len(chunks_seen)}/{expected_chunks} stage 1 chunks for {group_name}. "
                "Wait for all array tasks to finish before merging."
            )
        p_values = count_ge / null_n
        stage1_save_group(group_name, actual_imp, p_values, null_n)

    print(
        f"--- Stage 1 empirical permutation-importance masks "
        f"(group={stage1_group}, null={N_NULL_PERMS}, actual_repeats={STAGE1_ACTUAL_REPEATS}, "
        f"chunks={STAGE1_CHUNK_COUNT}, merge={STAGE1_MERGE}, correction=whole-brain FDR) ---"
    )

    for group_name in stage1_groups:
        if STAGE1_MERGE:
            stage1_merge_group(group_name)
        else:
            stage1_compute_chunk(group_name)

    if not importance_scores:
        print(
            "--- Stage 1 importance chunk complete. No final masks were produced in this run; "
            "merge chunks with --stage1_merge before downstream analyses. ---"
        )
        raise SystemExit(0)
    
    # =============================================================================
    # 3. Visualization (River Plot)
    # =============================================================================
    print("3. Generating River Plot...")
    
    # Prepare dictionary for plotting function
    plot_data = {}
    if 'SAD' in importance_scores:
        plot_data['SAD Placebo'] = importance_scores['SAD']
    if 'HC' in importance_scores:
        plot_data['HC Placebo'] = importance_scores['HC']

    # Use the helper function from Cell 4
    # Assumes make_river_plot_importance handles the figure creation
    try:
        if plot_data:
            make_river_plot_importance(
                plot_data,
                parcel_names_ext,
                top_k=20,  # Show top 20 most important features per group
                title="Neural Signatures (Permutation Importance)"
            )
            _save_fig("results_1_importance_river")
    except Exception as e:
        print(f"  ! Visualization skipped due to error: {e}")
    
    importance_mask_permutated = importance_masks
    importance_scores_permutated = importance_scores
    print("Permutated Importance masks generated and stored in 'importance_mask_permutated'.")
    _save_result("results_1_importance_mask_permutated", importance_masks)
    _save_result("results_1_importance_mask_top100_positive", importance_masks_top100_positive)
    _save_result("results_1_importance_scores_permutated", importance_scores)
    _save_result("results_1_importance_p_values_permutated", p_values_permutated)
    _save_result("results_1_importance_q_values_permutated", q_values_permutated)
    for grp in importance_scores.keys():
        _save_result(f"results_1_importance_masks_permutated_{grp}", {grp: importance_masks.get(grp)})
        _save_result(f"results_1_importance_masks_top100_positive_{grp}", {grp: importance_masks_top100_positive.get(grp)})
        _save_result(f"results_1_importance_scores_permutated_{grp}", {grp: importance_scores.get(grp)})
    save_checkpoint(1, {
        "importance_mask_permutated": importance_masks,
        "importance_masks_permutated": importance_masks,
        "importance_masks_top100_positive": importance_masks_top100_positive,
        "importance_scores_permutated": importance_scores,
        "p_values_permutated": p_values_permutated,
        "q_values_permutated": q_values_permutated,
        "feature_space_reports": feature_space_reports,
        "fdr_method": "fdr_bh_whole_brain",
        "fallback_sensitivity_rule": (
            f"Use top-{SENSITIVITY_TOP_K} positive permutation-importance mask when whole-brain FDR "
            f"selects fewer than {MIN_FDR_FEATURES_FOR_PRIMARY} features."
        ),
        "PERCENTILE_THRESH_permutated": PERCENTILE_THRESH,
        "thr_sad_permutated": locals().get("thr_sad"),
        "thr_hc_permutated": locals().get("thr_hc"),
        "parcel_names_ext_permutated": parcel_names_ext,
    })
    save_intermediate("stage01_importance_permutated", {
        "importance_mask_permutated": importance_masks,
        "importance_masks_permutated": importance_masks,
        "importance_masks_top100_positive": importance_masks_top100_positive,
        "importance_scores_permutated": importance_scores,
        "p_values_permutated": p_values_permutated,
        "q_values_permutated": q_values_permutated,
        "feature_space_reports": feature_space_reports,
        "fdr_method": "fdr_bh_whole_brain",
        "fallback_sensitivity_rule": (
            f"Use top-{SENSITIVITY_TOP_K} positive permutation-importance mask when whole-brain FDR "
            f"selects fewer than {MIN_FDR_FEATURES_FOR_PRIMARY} features."
        ),
        "PERCENTILE_THRESH_permutated": PERCENTILE_THRESH,
        "thr_sad_permutated": locals().get("thr_sad"),
        "thr_hc_permutated": locals().get("thr_hc"),
        "parcel_names_ext_permutated": parcel_names_ext,
    })
    save_intermediate("stage09_permutation_masks", {
        "importance_mask_permutated": importance_masks,
        "importance_masks_permutated": importance_masks,
        "importance_masks_top100_positive": importance_masks_top100_positive,
        "importance_scores_permutated": importance_scores,
        "p_values_permutated": p_values_permutated,
        "q_values_permutated": q_values_permutated,
        "feature_space_reports": feature_space_reports,
        "fdr_method": "fdr_bh_whole_brain",
        "fallback_sensitivity_rule": (
            f"Use top-{SENSITIVITY_TOP_K} positive permutation-importance mask when whole-brain FDR "
            f"selects fewer than {MIN_FDR_FEATURES_FOR_PRIMARY} features."
        ),
        "PERCENTILE_THRESH_permutated": PERCENTILE_THRESH,
        "thr_sad_permutated": locals().get("thr_sad"),
        "thr_hc_permutated": locals().get("thr_hc"),
        "parcel_names_ext_permutated": parcel_names_ext,
    })
    for grp in importance_scores.keys():
        save_intermediate(f"stage01_importance_permutated_{grp}", {
            "importance_mask_permutated": {grp: importance_masks.get(grp)},
            "importance_masks_permutated": {grp: importance_masks.get(grp)},
            "importance_masks_top100_positive": {grp: importance_masks_top100_positive.get(grp)},
            "importance_scores_permutated": {grp: importance_scores.get(grp)},
            "p_values_permutated": {grp: p_values_permutated.get(grp)},
            "q_values_permutated": {grp: q_values_permutated.get(grp)},
            "feature_space_reports": {grp: feature_space_reports.get(grp)},
            "fdr_method": "fdr_bh_whole_brain",
            "fallback_sensitivity_rule": (
                f"Use top-{SENSITIVITY_TOP_K} positive permutation-importance mask when whole-brain FDR "
                f"selects fewer than {MIN_FDR_FEATURES_FOR_PRIMARY} features."
            ),
            "PERCENTILE_THRESH_permutated": PERCENTILE_THRESH,
            "thr_sad_permutated": locals().get("thr_sad"),
            "thr_hc_permutated": locals().get("thr_hc"),
            "parcel_names_ext_permutated": parcel_names_ext,
        })
        save_intermediate(f"stage09_permutation_masks_{grp}", {
            "importance_mask_permutated": {grp: importance_masks.get(grp)},
            "importance_masks_permutated": {grp: importance_masks.get(grp)},
            "importance_masks_top100_positive": {grp: importance_masks_top100_positive.get(grp)},
            "importance_scores_permutated": {grp: importance_scores.get(grp)},
            "p_values_permutated": {grp: p_values_permutated.get(grp)},
            "q_values_permutated": {grp: q_values_permutated.get(grp)},
            "feature_space_reports": {grp: feature_space_reports.get(grp)},
            "fdr_method": "fdr_bh_whole_brain",
            "fallback_sensitivity_rule": (
                f"Use top-{SENSITIVITY_TOP_K} positive permutation-importance mask when whole-brain FDR "
                f"selects fewer than {MIN_FDR_FEATURES_FOR_PRIMARY} features."
            ),
            "PERCENTILE_THRESH_permutated": PERCENTILE_THRESH,
            "thr_sad_permutated": locals().get("thr_sad"),
            "thr_hc_permutated": locals().get("thr_hc"),
            "parcel_names_ext_permutated": parcel_names_ext,
        })
    save_stage_bundle(
        1,
        "stage01_NeuralDissociation",
        {
            "results_11": locals().get("results_11"),
            "results_1_importance_masks_permutated": importance_masks,
            "results_1_importance_scores_permutated": importance_scores,
            "res_sad_dict": res_sad_dict,
            "res_hc_dict": res_hc_dict,
            "res_sad": res_sad,
            "res_hc": res_hc,
            "best_c_sad": best_c_sad,
            "best_c_hc": best_c_hc,
            "X_sad": X_sad, "y_sad": y_sad, "sub_sad": sub_sad,
            "X_hc": X_hc, "y_hc": y_hc, "sub_hc": sub_hc,
            "func_matrix": func_matrix,
            "p_sad": p_sad, "p_hc": p_hc,
            "p_sad2hc": locals().get("p_sad2hc"),
            "p_hc2sad": locals().get("p_hc2sad"),
            "accs_sad2hc": locals().get("accs_sad2hc"),
            "accs_hc2sad": locals().get("accs_hc2sad"),
            "mean_sad2hc": locals().get("mean_sad2hc"),
            "mean_hc2sad": locals().get("mean_hc2sad"),
            "perm_acc_sad": locals().get("perm_acc_sad"),
            "perm_acc_hc": locals().get("perm_acc_hc"),
            "perm_sad2hc": locals().get("perm_sad2hc"),
            "perm_hc2sad": locals().get("perm_hc2sad"),
            "map_sad": map_sad,
            "map_hc": map_hc,
            "obs_sim": obs_sim,
            "p_sim_spatial": p_sim_spatial,
            "spatial_results": locals().get("spatial_results"),
            "importance_masks": importance_masks,
            "importance_scores": importance_scores,
            "importance_mask_permutated": importance_mask_permutated,
            "importance_masks_top100_positive": importance_masks_top100_positive,
            "importance_scores_permutated": importance_scores_permutated,
            "p_values_permutated": p_values_permutated,
            "q_values_permutated": q_values_permutated,
            "feature_space_reports": feature_space_reports,
            "PERCENTILE_THRESH": PERCENTILE_THRESH,
            "thr_sad": locals().get("thr_sad"),
            "thr_hc": locals().get("thr_hc"),
            "parcel_names_ext": parcel_names_ext,
        },
    )
    

if stage_active(2):
    # Cell 9: Analysis 1.2 - Static Representational Topology (FDR or top-100 sensitivity | Centroid)
    # Objective: Characterize the stable organization of the social learning space.
    # Constraint: whole-brain FDR permutation-importance masks, with top-100 positive sensitivity fallback.
    # Method: Cross-validated Mahalanobis (crossnobis) distance with shrinkage covariance, averaged over split-half repeats.
    # Tests: Group Comparison (SAD vs HC) AND One-Sample Test (Dist > 0).
    
    print("--- Running Analysis 1.2: Static Representational Topology (FDR/top-100 sensitivity | Centroid) ---")
    
    from scipy.stats import ttest_1samp
    
    # Global Constants
    RDM_CONDITIONS = ["CS-", "CSS", "CSR"] 
    PERCENTILE_THRESH = None
    
    # =============================================================================
    # 0. Feature Selection (Empirical whole-brain FDR masks)
    # Rationale: use the same permutation-derived feature set as downstream analyses.
    # =============================================================================
    print("\n[Step 0] Selecting empirical whole-brain FDR neural features...")
    
    analysis_masks, feature_space_12 = get_analysis_feature_masks("Analysis 1.2")
    mask_sad_top5 = analysis_masks['SAD']
    mask_hc_top5 = analysis_masks['HC']
    
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
    
    # Slice Features (apply selected FDR or top-100 sensitivity masks)
    X_sad_12 = X_raw[mask_sad_grp][:, mask_sad_top5]
    y_sad_12 = y_raw[mask_sad_grp]
    sub_sad_12 = sub_raw[mask_sad_grp]
    
    X_hc_12 = X_raw[mask_hc_grp][:, mask_hc_top5]
    y_hc_12 = y_raw[mask_hc_grp]
    sub_hc_12 = sub_raw[mask_hc_grp]
    
    print(f"  > SAD Matrix: {X_sad_12.shape} | HC Matrix: {X_hc_12.shape}")
    
    # =============================================================================
    # 2. Centroid RDM Calculation
    # =============================================================================
    # Compute RDMs
    print(f"  Calculating Centroid RDMs (Conditions: {RDM_CONDITIONS}) with {CROSSNOBIS_REPEATS} split-half repeats...")
    rdms_sad, subs_sad_rdm = calculate_crossnobis_rdm(X_sad_12, y_sad_12, sub_sad_12, RDM_CONDITIONS)
    rdms_hc, subs_hc_rdm = calculate_crossnobis_rdm(X_hc_12, y_hc_12, sub_hc_12, RDM_CONDITIONS)
    
    print(f"  > Computed RDMs: SAD (n={len(subs_sad_rdm)}), HC (n={len(subs_hc_rdm)})")
    
    # =============================================================================
    # 3. Metrics & Statistical Tests
    # =============================================================================
    # Conditions: 0=CS-, 1=CSS, 2=CSR
    idx_cs_minus, idx_css, idx_csr = 0, 1, 2
    
    vec_a_sad, vec_b_sad = extract_topology_metrics(rdms_sad, idx_cs_minus, idx_css, idx_csr)
    vec_a_hc, vec_b_hc = extract_topology_metrics(rdms_hc, idx_cs_minus, idx_css, idx_csr)
    
    print("\n[Step 3] Statistical Testing...")
    
    # --- Helper for One-Sample Test (Significantly > 0?) ---
    # Metric A: Threat Distance (The Canyon)
    print("\nMetric A: Threat (CSR) vs Safety (CSS) Distance")
    p_a_sad_0 = one_sample_distance_test(vec_a_sad, "SAD (Dist > 0)")
    p_a_hc_0 = one_sample_distance_test(vec_a_hc, "HC  (Dist > 0)")
    
    print("  > Group Comparison (SAD vs HC):")
    t_a, p_a, m_a_sad, m_a_hc = perm_ttest_ind(vec_a_sad, vec_a_hc, n_perm=N_PERMUTATION)
    print(f"    Diff: SAD={m_a_sad:.3f}, HC={m_a_hc:.3f} | t={t_a:.3f}, p={p_a:.4f}")
    
    # Metric B: Safety Distance (The Collapse)
    print("\nMetric B: Safety (CSS) vs Background (CS-) Distance")
    p_b_sad_0 = one_sample_distance_test(vec_b_sad, "SAD (Dist > 0)")
    p_b_hc_0 = one_sample_distance_test(vec_b_hc, "HC  (Dist > 0)")
    
    print("  > Group Comparison (SAD vs HC):")
    t_b, p_b, m_b_sad, m_b_hc = perm_ttest_ind(vec_b_sad, vec_b_hc, n_perm=N_PERMUTATION)
    print(f"    Diff: SAD={m_b_sad:.3f}, HC={m_b_hc:.3f} | t={t_b:.3f}, p={p_b:.4f}")
    
    # =============================================================================
    # 4. Visualization
    # =============================================================================
    sns.set_context("poster")
    fig = plt.figure(figsize=(24, 8))
    gs = fig.add_gridspec(1, 3)
    
    # Heatmaps
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(np.mean(rdms_sad, axis=0), annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1.2, 
                xticklabels=RDM_CONDITIONS, yticklabels=RDM_CONDITIONS, ax=ax1, cbar=False)
    ax1.set_title(f"SAD Topology\n(n={len(subs_sad_rdm)})")
    
    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(np.mean(rdms_hc, axis=0), annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1.2,
                xticklabels=RDM_CONDITIONS, yticklabels=RDM_CONDITIONS, ax=ax2)
    ax2.set_title(f"HC Topology\n(n={len(subs_hc_rdm)})")
    
    # Violins
    ax3 = fig.add_subplot(gs[0, 2])
    df_res = pd.DataFrame({
        'Group': ['SAD']*len(vec_a_sad) + ['HC']*len(vec_a_hc) + ['SAD']*len(vec_b_sad) + ['HC']*len(vec_b_hc),
        'Distance': np.concatenate([vec_a_sad, vec_a_hc, vec_b_sad, vec_b_hc]),
        'Metric': ['A: Threat Dist']*len(vec_a_sad) + ['A: Threat Dist']*len(vec_a_hc) + 
                  ['B: Safety Dist']*len(vec_b_sad) + ['B: Safety Dist']*len(vec_b_hc)
    })
    sns.violinplot(data=df_res, x='Metric', y='Distance', hue='Group', 
                   split=True, inner='quartile', palette={'SAD': '#c44e52', 'HC': '#4c72b0'}, ax=ax3)
    ax3.set_title("Topological Metrics (Centroid)")
    ax3.set_ylabel("Crossnobis Distance")
    
    # Annotate Group Differences
    y_max = df_res['Distance'].max()
    if p_a < 0.05: ax3.text(0, y_max + 0.05, f'* (p={p_a:.3f})', ha='center', fontsize=18)
    if p_b < 0.05: ax3.text(1, y_max + 0.05, f'* (p={p_b:.3f})', ha='center', fontsize=18)
    
    # Annotate Sig > 0 (Below X Axis)
    # For Metric A
    ax3.text(-0.2, -0.15, f"SAD: {get_sig_star(p_a_sad_0)}", transform=ax3.get_xaxis_transform(), ha='center', fontsize=14, color='#c44e52')
    ax3.text(0.2, -0.15, f"HC: {get_sig_star(p_a_hc_0)}", transform=ax3.get_xaxis_transform(), ha='center', fontsize=14, color='#4c72b0')
    
    # For Metric B
    ax3.text(0.8, -0.15, f"SAD: {get_sig_star(p_b_sad_0)}", transform=ax3.get_xaxis_transform(), ha='center', fontsize=14, color='#c44e52')
    ax3.text(1.2, -0.15, f"HC: {get_sig_star(p_b_hc_0)}", transform=ax3.get_xaxis_transform(), ha='center', fontsize=14, color='#4c72b0')
    
    plt.tight_layout()
    _save_fig("analysis_12_topology")
    _save_fig("results_12_topology")
    plt.show()
    
    # Store Results
    n_feat_sad = max(int(np.sum(mask_sad_top5)), 1)
    n_feat_hc = max(int(np.sum(mask_hc_top5)), 1)
    rdms_sad_pv = rdms_sad / n_feat_sad
    rdms_hc_pv = rdms_hc / n_feat_hc
    vec_a_sad_pv, vec_b_sad_pv = extract_topology_metrics(rdms_sad_pv, idx_cs_minus, idx_css, idx_csr)
    vec_a_hc_pv, vec_b_hc_pv = extract_topology_metrics(rdms_hc_pv, idx_cs_minus, idx_css, idx_csr)
    t_a_pv, p_a_pv, m_a_sad_pv, m_a_hc_pv = perm_ttest_ind(vec_a_sad_pv, vec_a_hc_pv, n_perm=N_PERMUTATION)
    t_b_pv, p_b_pv, m_b_sad_pv, m_b_hc_pv = perm_ttest_ind(vec_b_sad_pv, vec_b_hc_pv, n_perm=N_PERMUTATION)

    results_12 = {
        "rdms_sad": rdms_sad, "rdms_hc": rdms_hc,
        "rdms_sad_pv": rdms_sad_pv, "rdms_hc_pv": rdms_hc_pv,
        "rdms_sad_raw_pv": rdms_sad_pv, "rdms_hc_raw_pv": rdms_hc_pv,
        "subs_sad_rdm": subs_sad_rdm, "subs_hc_rdm": subs_hc_rdm,
        "metric_a_stats": (t_a, p_a), "metric_b_stats": (t_b, p_b),
        "metric_a_stats_pv": (t_a_pv, p_a_pv), "metric_b_stats_pv": (t_b_pv, p_b_pv),
        "metric_a_means_pv": {"SAD": m_a_sad_pv, "HC": m_a_hc_pv},
        "metric_b_means_pv": {"SAD": m_b_sad_pv, "HC": m_b_hc_pv},
        "features_sad": n_feat_sad, "features_hc": n_feat_hc,
        "feature_space": feature_space_12,
        "one_sample_stats": {"p_a_sad": p_a_sad_0, "p_a_hc": p_a_hc_0, "p_b_sad": p_b_sad_0, "p_b_hc": p_b_hc_0}
    }
    _save_result("results_12", results_12)
    _save_result("results_12", results_12)
    save_checkpoint(9, {
        "results_12": results_12
    })
    save_intermediate("stage09_results_12", results_12)
    save_intermediate("stage12_topology_stats", {"results_12": results_12})
    save_intermediate("stage10_topology_stats", results_12)
    save_stage_bundle(
        2,
        "stage02_StaticRepresentationalTopology",
        {
            "results_12": results_12,
            "rdms_sad": locals().get("rdms_sad"),
            "rdms_hc": locals().get("rdms_hc"),
            "rdms_sad_pv": locals().get("rdms_sad_pv"),
            "rdms_hc_pv": locals().get("rdms_hc_pv"),
            "subs_sad_rdm": locals().get("subs_sad_rdm"),
            "subs_hc_rdm": locals().get("subs_hc_rdm"),
            "sub_sad_12": locals().get("sub_sad_12"),
            "sub_hc_12": locals().get("sub_hc_12"),
            "mask_sad_top5": locals().get("mask_sad_top5"),
            "mask_hc_top5": locals().get("mask_hc_top5"),
            "feature_space": locals().get("feature_space_12"),
            "p_a": locals().get("p_a"),
            "p_b": locals().get("p_b"),
        },
    )
    
# %% [cell 12]
if stage_active(3):
    # Cell 10: Analysis 1.3 - Dynamic Representational Drift (FDR or top-100 sensitivity features)
    # Objective: Quantify plasticity magnitude (Projection) and fidelity (Cosine).
    # Target Definitions:
    #   - Safety:  Extinction CSS -> Extinction CS-
    #   - Threat:  Extinction CSR -> Reinstatement CSR
    # Feature Selection: whole-brain FDR permutation importance, with top-100 positive sensitivity fallback.
    
    print("--- Running Analysis 1.3: Dynamic Representational Drift (FDR/top-100 sensitivity features) ---")
    
    import pandas as pd
    import statsmodels.api as sm
    from numpy.linalg import norm
    from scipy.stats import ttest_1samp, ttest_ind, levene, shapiro, mannwhitneyu
    
    # Constants
    COND_SAFETY_TARGET = "CS-"
    COND_SAFETY_LEARN = "CSS"
    COND_THREAT_LEARN = "CSR"
    
    # =============================================================================
    # 0. Feature Selection & Data Loading
    # Rationale: use the same empirical whole-brain FDR masks as MemoryFearNetwork's downstream workflow.
    # =============================================================================
    print(f"\n[Step 0] Setup & Data Loading...")
    
    analysis_masks, feature_space_13 = get_analysis_feature_masks("Analysis 1.3")
    mask_sad = analysis_masks['SAD']
    mask_hc = analysis_masks['HC']
    
    # 2. Load Data Helpers (Nested Dictionary Access)
    # Load Extinction (Start/Learning Phase)
    X_ext_sad, y_ext_sad, sub_ext_sad = get_phase_data("SAD_Placebo", "ext")
    X_ext_hc, y_ext_hc, sub_ext_hc = get_phase_data("HC_Placebo", "ext")
    
    # Load Reinstatement (Target Phase for Threat)
    X_rst_sad, y_rst_sad, sub_rst_sad = get_phase_data("SAD_Placebo", "rst")
    X_rst_hc, y_rst_hc, sub_rst_hc = get_phase_data("HC_Placebo", "rst")
    
    # Validate Reinstatement Data
    if X_rst_sad is None or X_rst_hc is None:
        print("  ! WARNING: Reinstatement data missing. Threat analysis will fallback to Extinction (Trivial).")
        X_rst_sad, y_rst_sad, sub_rst_sad = X_ext_sad, y_ext_sad, sub_ext_sad
        X_rst_hc, y_rst_hc, sub_rst_hc = X_ext_hc, y_ext_hc, sub_ext_hc
    
    # Handle CS- (Safety Target) - likely missing from subsets, need global X_ext
    if 'X_ext' in locals():
        X_global, y_global, sub_global = X_ext, y_ext, sub_ext
    else:
        print("  ! WARNING: Global X_ext missing. Safety Target (CS-) might be unavailable.")
        X_global, y_global, sub_global = X_ext_sad, y_ext_sad, sub_ext_sad
    
    
    # =============================================================================
    # 2. Execution
    # =============================================================================
    print("\n[Step 2] Calculating Vectors...")
    
    # A. Safety Learning (CSS -> CS-)
    # Both Start and Target are in Extinction (or Global)
    print("  > Safety Analysis: Start=CSS(Ext) -> Target=CS-(Ext)")
    df_safe_sad = calculate_plasticity_vectors(
        X_ext_sad, y_ext_sad, sub_ext_sad,     # Learn: Extinction
        X_global, y_global, sub_global,        # Target: Global (contains CS-)
        mask_sad, COND_SAFETY_LEARN, COND_SAFETY_TARGET
    )
    df_safe_hc = calculate_plasticity_vectors(
        X_ext_hc, y_ext_hc, sub_ext_hc, 
        X_global, y_global, sub_global, 
        mask_hc, COND_SAFETY_LEARN, COND_SAFETY_TARGET
    )
    
    # B. Threat Maintenance (CSR -> Reinstatement CSR)
    # Start is Extinction, Target is REINSTATEMENT
    print("  > Threat Analysis: Start=CSR(Ext) -> Target=CSR(Reinstatement)")
    df_threat_sad = calculate_plasticity_vectors(
        X_ext_sad, y_ext_sad, sub_ext_sad,     # Learn: Extinction
        X_rst_sad, y_rst_sad, sub_rst_sad,     # Target: Reinstatement
        mask_sad, COND_THREAT_LEARN, COND_THREAT_LEARN 
    )
    df_threat_hc = calculate_plasticity_vectors(
        X_ext_hc, y_ext_hc, sub_ext_hc, 
        X_rst_hc, y_rst_hc, sub_rst_hc, 
        mask_hc, COND_THREAT_LEARN, COND_THREAT_LEARN
    )
    
    # =============================================================================
    # 3. Statistics & Visualization
    # =============================================================================
    df_plot = pd.concat([
        tag_df(df_safe_sad, 'SAD', 'Safety'), tag_df(df_safe_hc, 'HC', 'Safety'),
        tag_df(df_threat_sad, 'SAD', 'Threat'), tag_df(df_threat_hc, 'HC', 'Threat')
    ])
    
    if df_plot.empty:
        print("! No data generated. Check inputs.")
    else:
        print(f"\n[Step 3] Generated {len(df_plot)} subject vectors.")
        
        sns.set_context("poster")
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        # 1. Projection (Magnitude)
        sns.barplot(data=df_plot, x='Condition', y='projection', hue='Group', 
                    palette={'SAD': '#c44e52', 'HC': '#4c72b0'}, ax=axes[0,0], 
                    capsize=.1, errorbar='se')
        axes[0,0].axhline(0, color='k', ls='--')
        axes[0,0].set_title("Magnitude (Scalar Projection)")
        
        # 2. Cosine (Fidelity)
        sns.barplot(data=df_plot, x='Condition', y='cosine', hue='Group', 
                    palette={'SAD': '#c44e52', 'HC': '#4c72b0'}, ax=axes[0,1], 
                    capsize=.1, errorbar='se')
        axes[0,1].axhline(0, color='k', ls='--')
        axes[0,1].set_title("Directional Fidelity (Cosine)")
        
        # 3. Stats (Printout)
        print("\n--- Statistical Summary (SAD vs HC) ---")
        for cond in ['Safety', 'Threat']:
            print(f"\nCondition: {cond}")
            for met in ['projection', 'cosine']:
                d_s = df_plot[(df_plot['Condition']==cond) & (df_plot['Group']=='SAD')][met]
                d_h = df_plot[(df_plot['Condition']==cond) & (df_plot['Group']=='HC')][met]
                
                # One-sample t-test (vs 0)
                if len(d_s)>1: 
                    t0_s, p0_s = ttest_1samp(d_s, 0, alternative='greater')
                    print(f"  > SAD > 0 ({met}): t={t0_s:.3f}, p={p0_s:.4f}")
                if len(d_h)>1:
                    t0_h, p0_h = ttest_1samp(d_h, 0, alternative='greater')
                    print(f"  > HC  > 0 ({met}): t={t0_h:.3f}, p={p0_h:.4f}")
    
                # Group Diff
                if len(d_s)>1 and len(d_h)>1:
                    t, p = ttest_ind(d_s, d_h)
                    sig = "*" if p < 0.05 else "ns"
                    print(f"  > Group Diff ({met}): t={t:.3f}, p={p:.4f} {sig}")
    
        # 4. Scatter (Init Dist vs Projection)
        sns.scatterplot(data=df_plot, x='init_dist', y='projection', hue='Group', style='Condition', 
                        palette={'SAD': '#c44e52', 'HC': '#4c72b0'}, alpha=0.7, ax=axes[1,0], s=100)
        axes[1,0].axhline(0, color='k', ls='--')
        axes[1,0].set_title("Learning vs Initial Distance")
        
        axes[1,1].axis('off') # Empty slot
        plt.tight_layout()
    _save_fig("analysis_13_drift")
    _save_fig("results_13_drift")
    plt.show()
    
    results_13 = {'safe_sad': df_safe_sad, 'threat_sad': df_threat_sad, 'feature_space': feature_space_13}
    results_13_main = results_13
    _save_result("results_13", results_13)
    _save_result("results_13", results_13)
    save_checkpoint(10, {
        "results_13": results_13,
        "df_safe_sad": locals().get("df_safe_sad"),
        "df_safe_hc": locals().get("df_safe_hc"),
        "df_threat_sad": locals().get("df_threat_sad"),
        "df_threat_hc": locals().get("df_threat_hc"),
        "feature_space": locals().get("feature_space_13"),
    })
    save_intermediate("stage13_drift", {
        "results_13": results_13,
        "df_plot": locals().get("df_plot"),
        "df_safe_sad": locals().get("df_safe_sad"),
        "df_safe_hc": locals().get("df_safe_hc"),
        "df_threat_sad": locals().get("df_threat_sad"),
        "df_threat_hc": locals().get("df_threat_hc"),
        "feature_space": locals().get("feature_space_13"),
    })
    
# %% [cell 13]
if stage_active(3):
    # Cell 10: Analysis 1.3 - Dynamic Representational Drift (Single-Trial Trajectories)
    # Objective: Visualize plasticity trial-by-trial using FDR or top-100 sensitivity features.
    # Method: Project every trial onto the Ideal Axis (Start -> Target).
    #   - Score 0 = Resembles Early Extinction (Start)
    #   - Score 1 = Resembles Target (CS- or Reinstated CSR)
    
    print("--- Running Analysis 1.3: Single-Trial Trajectories (FDR/top-100 sensitivity) ---")
    
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    from numpy.linalg import norm
    
    # Constants
    COND_SAFETY_TARGET = "CS-"
    COND_SAFETY_LEARN = "CSS"
    COND_THREAT_LEARN = "CSR"
    BLOCK_SIZE = 1  # Group trials for smoother plotting (1 = Raw Single Trial)
    
    # =============================================================================
    # 0. Feature Selection (Empirical whole-brain FDR masks)
    # =============================================================================
    print("\n[Step 0] Selecting empirical whole-brain FDR features...")
    
    analysis_masks, feature_space_13b = get_analysis_feature_masks("Analysis 1.3 part 2")
    mask_sad = analysis_masks['SAD']
    mask_hc = analysis_masks['HC']
    
    
    # --- FIXED FUNCTION ---
    # Load Start Data (Extinction)
    X_ext_sad, y_ext_sad, sub_ext_sad = get_phase_data("SAD_Placebo", "ext")
    X_ext_hc, y_ext_hc, sub_ext_hc = get_phase_data("HC_Placebo", "ext")
    
    # Load Target Data (Reinstatement)
    X_rst_sad, y_rst_sad, sub_rst_sad = get_phase_data("SAD_Placebo", "rst")
    X_rst_hc, y_rst_hc, sub_rst_hc = get_phase_data("HC_Placebo", "rst")
    
    # Check Reinstatement Availability
    if X_rst_sad is None or X_rst_hc is None:
        raise ValueError("Reinstatement data missing in data_subsets. Run Cell 5 and ensure phase3 is loaded.")
    # Check Global Availability (for CS-)
    if 'X_ext' in locals():
        X_glob, y_glob, sub_glob = X_ext, y_ext, sub_ext
    else:
        # Fallback to group data if global is missing
        X_glob, y_glob, sub_glob = X_ext_sad, y_ext_sad, sub_ext_sad
    
    # =============================================================================
    # 3. Execute Analysis
    # =============================================================================
    print("\n[Step 2] Calculating Single-Trial Trajectories...")
    
    # A. Safety Learning
    # Axis: Early CSS (Ext) --> CS- Centroid (Ext/Global)
    print("  > Safety: CSS Trials projecting onto [Early CSS -> CS-]")
    df_safe_sad = calc_trajectory(X_ext_sad, y_ext_sad, sub_ext_sad, X_glob, y_glob, sub_glob, mask_sad, COND_SAFETY_LEARN, COND_SAFETY_TARGET)
    df_safe_hc = calc_trajectory(X_ext_hc, y_ext_hc, sub_ext_hc, X_glob, y_glob, sub_glob, mask_hc, COND_SAFETY_LEARN, COND_SAFETY_TARGET)
    
    # B. Threat Maintenance
    # Axis: Early CSR (Ext) --> Reinstated CSR Centroid (Rst)
    print("  > Threat: CSR Trials projecting onto [Early CSR -> Reinstated CSR]")
    df_threat_sad = calc_trajectory(X_ext_sad, y_ext_sad, sub_ext_sad, X_rst_sad, y_rst_sad, sub_rst_sad, mask_sad, COND_THREAT_LEARN, COND_THREAT_LEARN)
    df_threat_hc = calc_trajectory(X_ext_hc, y_ext_hc, sub_ext_hc, X_rst_hc, y_rst_hc, sub_rst_hc, mask_hc, COND_THREAT_LEARN, COND_THREAT_LEARN)
    
    # =============================================================================
    # Detailed Statistics
    # =============================================================================
    print("\n[Step 3] Calculating Statistics...")
    stats_safe = run_detailed_stats(df_safe_sad, df_safe_hc, "Safety Learning")
    stats_threat = run_detailed_stats(df_threat_sad, df_threat_hc, "Threat Maintenance")
    # =============================================================================
    # 4. Visualization
    # =============================================================================    
    df_safe = prepare_plot(df_safe_sad, df_safe_hc, "Safety Learning")
    df_threat = prepare_plot(df_threat_sad, df_threat_hc, "Threat Maintenance")
    
    if df_safe.empty and df_threat.empty:
        print("! No data to plot.")
    else:
        sns.set_context("poster")
        fig, axes = plt.subplots(1, 2, figsize=(22, 9), sharey=True)
        
        # 1. Safety Plot
        if not df_safe.empty:
            sns.lineplot(data=df_safe, x='trial', y='score', hue='Group', 
                         palette={'SAD': '#c44e52', 'HC': '#4c72b0'}, 
                         lw=3, marker="o", err_style="band", ax=axes[0])
            axes[0].set_title("A. Safety Trajectory\n(Target = CS-)")
            axes[0].set_ylabel("Similarity Score (0=Start, 1=Target)")
            axes[0].axhline(0, color='gray', ls='--', label='Start (Fear)')
            axes[0].axhline(1, color='#2ca02c', ls='-', lw=2, label='Target (CS-)')
            axes[0].legend(loc='upper left')
    
        # 2. Threat Plot
        if not df_threat.empty:
            sns.lineplot(data=df_threat, x='trial', y='score', hue='Group', 
                         palette={'SAD': '#c44e52', 'HC': '#4c72b0'}, 
                         lw=3, marker="s", err_style="band", ax=axes[1])
            axes[1].set_title("B. Threat Maintenance\n(Target = Early Half Reinstated CSR)")
            axes[1].set_xlabel(f"Trial (Block Size: {BLOCK_SIZE})")
            axes[1].axhline(0, color='gray', ls='--', label='Start (Ext Early)')
            axes[1].axhline(1, color='#d62728', ls='-', lw=2, label='Target (Early Half Reinstated CSR)')
            axes[1].legend(loc='upper left')
        
        plt.tight_layout()
    _save_fig("analysis_13_trajectories")
    _save_fig("results_13_trajectories")
    plt.show()
    
    results_13 = {
        'stats_safe': stats_safe, 
        'stats_threat': stats_threat,
        'data_safe': df_safe,
        'data_threat': df_threat,
        'feature_space': feature_space_13b,
    }
    results_13_2 = results_13
    _save_result("results_13b", results_13)
    _save_result("results_13b", results_13)
    _save_result("results_13_2", results_13_2)
    save_checkpoint(11, {
        "results_13b": results_13,
        "results_13_2": results_13_2,
    })
    save_intermediate("stage14_trajectories", {"results_13_2": results_13_2})
    save_stage_bundle(
        3,
        "stage03_DynamicRepresentationalDrift",
        {
            "results_13": globals().get("results_13_main"),
            "results_13b": results_13,
            "results_13_2": results_13_2,
            "df_safe_sad": locals().get("df_safe_sad"),
            "df_safe_hc": locals().get("df_safe_hc"),
            "df_threat_sad": locals().get("df_threat_sad"),
            "df_threat_hc": locals().get("df_threat_hc"),
            "stats_safe": locals().get("stats_safe"),
            "stats_threat": locals().get("stats_threat"),
            "df_safe": locals().get("df_safe"),
            "df_threat": locals().get("df_threat"),
            "feature_space": locals().get("feature_space_13b"),
        },
    )
    
# %% [cell 14]
if stage_active(4):
    # Cell 11: Analysis 1.4 - Decision Boundary Characteristics (Self-Network with Stats)
    # Objective: Quantify "Cognitive Certainty" (Entropy) and "Decision Sharpness" (Kurtosis) 
    #            using each group's NATIVE feature network.
    # Method: Cross-Validated Probability Extraction (Fixed Optimal C).
    
    print("--- Running Analysis 1.4: Self-Network Statistics (Entropy, Kurtosis, Variance) ---")
    
    # Constants
    COND_CLASS_THREAT = "CSR"
    COND_CLASS_SAFE = "CSS"
    
    # =============================================================================
    # 0. Setup Feature Masks (Native) & Best Params
    # Rationale: entropy/kurtosis should reflect the full positive-importance decision space.
    # =============================================================================
    analysis_masks, feature_space_14 = get_analysis_feature_masks("Analysis 1.4")
    mask_sad_native = analysis_masks['SAD']
    mask_hc_native = analysis_masks['HC']
    
    if 'subject_best_params' not in locals():
        print("  > 'subject_best_params' not found. Using default C=1.0.")
        # Fallback default
        subject_best_params = {}
    
    # Use best C from Analysis 1.1 when phase/labels/group match
    best_c_sad = locals().get("best_c_sad", None)
    best_c_hc = locals().get("best_c_hc", None)
    
    
    # =============================================================================
    # 2. Execution (Self-Network)
    # =============================================================================
    print("\n[Step 2] Calculating Statistics (Native Networks)...")
    
    
    # Load SAD Data
    X_sad, y_sad, sub_sad = get_ext_data("SAD_Placebo")
    # Load HC Data
    X_hc, y_hc, sub_hc = get_ext_data("HC_Placebo")
    
    # SAD Analysis (Native)
    print("  > Analyzing SAD Placebo...")
    df_sad_stats = calculate_distribution_stats(
        X_sad, y_sad, sub_sad, 
        mask_sad_native, subject_best_params
    )
    
    # HC Analysis (Native)
    print("  > Analyzing HC Placebo...")
    df_hc_stats = calculate_distribution_stats(
        X_hc, y_hc, sub_hc, 
        mask_hc_native, subject_best_params
    )
    
    # =============================================================================
    # 3. Statistical Comparison
    # =============================================================================
    print("\n--- RESULTS: Self-Network Decision Statistics ---")
    p_ent = compare_metric(df_sad_stats['entropy'], df_hc_stats['entropy'], "Entropy (Uncertainty)")
    p_kurt = compare_metric(df_sad_stats['kurtosis'], df_hc_stats['kurtosis'], "Kurtosis (Sharpness)")
    p_var = compare_metric(df_sad_stats['variance'], df_hc_stats['variance'], "Variance (Spread)")
    
    # =============================================================================
    # 4. Visualization
    # =============================================================================
    sns.set_context("poster")
    fig = plt.figure(figsize=(24, 8))
    gs = fig.add_gridspec(1, 3)
    
    # A. Entropy (Violin)
    ax1 = fig.add_subplot(gs[0, 0])
    if not df_sad_stats.empty and not df_hc_stats.empty:
        df_ent_plot = pd.concat([
            pd.DataFrame({'Val': df_sad_stats['entropy'], 'Group': 'SAD'}),
            pd.DataFrame({'Val': df_hc_stats['entropy'], 'Group': 'HC'})
        ])
        sns.violinplot(data=df_ent_plot, x='Group', y='Val', palette={'SAD': '#c44e52', 'HC': '#4c72b0'}, ax=ax1)
        ax1.set_title("Uncertainty (Entropy)")
        ax1.set_ylabel("Shannon Entropy (bits)")
        if p_ent < 0.05: ax1.text(0.5, df_ent_plot['Val'].max(), f'* p={p_ent:.3f}', ha='center', fontsize=16)
    
    # B. Kurtosis (Box)
    ax2 = fig.add_subplot(gs[0, 1])
    if not df_sad_stats.empty and not df_hc_stats.empty:
        df_kurt_plot = pd.concat([
            pd.DataFrame({'Val': df_sad_stats['kurtosis'], 'Group': 'SAD'}),
            pd.DataFrame({'Val': df_hc_stats['kurtosis'], 'Group': 'HC'})
        ])
        sns.boxplot(data=df_kurt_plot, x='Group', y='Val', palette={'SAD': '#c44e52', 'HC': '#4c72b0'}, ax=ax2)
        ax2.set_title("Sharpness (Kurtosis)")
        ax2.set_ylabel("Fisher Kurtosis")
        if p_kurt < 0.05: ax2.text(0.5, df_kurt_plot['Val'].max(), f'* p={p_kurt:.3f}', ha='center', fontsize=16)
    
    # C. Density (Distribution)
    ax3 = fig.add_subplot(gs[0, 2])
    if not df_sad_stats.empty and not df_hc_stats.empty:
        probs_sad = np.concatenate(df_sad_stats['probabilities'].values)
        probs_hc = np.concatenate(df_hc_stats['probabilities'].values)
        sns.kdeplot(probs_sad, color='#c44e52', fill=True, label='SAD', bw_adjust=0.6, ax=ax3)
        sns.kdeplot(probs_hc, color='#4c72b0', fill=True, label='HC', bw_adjust=0.6, ax=ax3)
        ax3.set_title("Probability Distribution")
        ax3.set_xlabel("P(Threat) for Safety Cues")
        ax3.set_xlim(0, 1)
        ax3.legend()
    
    plt.tight_layout()
    _save_fig("analysis_14_decision_stats")
    _save_fig("results_14_self")
    plt.show()
    
    results_14_self = {'df_sad': df_sad_stats, 'df_hc': df_hc_stats, 'feature_space': feature_space_14}
    _save_result("results_14_self", results_14_self)
    _save_result("results_14_self", results_14_self)
    save_checkpoint(12, {
        "results_14_self": results_14_self,
        "df_sad_stats": locals().get("df_sad_stats"),
        "df_hc_stats": locals().get("df_hc_stats"),
        "feature_space": locals().get("feature_space_14"),
    })
    save_intermediate("stage15_decision_stats", {
        "results_14_self": results_14_self,
        "df_sad_stats": locals().get("df_sad_stats"),
        "df_hc_stats": locals().get("df_hc_stats"),
        "feature_space": locals().get("feature_space_14"),
    })
    save_stage_bundle(
        4,
        "stage04_DecisionBoundaryCharacteristics",
        {
            "results_14_self": results_14_self,
            "df_sad_stats": locals().get("df_sad_stats"),
            "df_hc_stats": locals().get("df_hc_stats"),
            "p_ent": locals().get("p_ent"),
            "p_kurt": locals().get("p_kurt"),
            "feature_space": locals().get("feature_space_14"),
        },
    )
    
# %% [cell 15]
if stage_active(5):
    # Cell 12: Analysis 2.1 - Safety Restoration & Threat Discrimination (Mixed Effects)
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
    # 0. Validate / Recompute Masks from Cell 9
    # =============================================================================
    if 'mask_sad_top5' not in locals() or 'mask_hc_top5' not in locals():
        # Reload empirical FDR masks so this stage can run standalone
        importance_scores, importance_masks = ensure_importance_loaded()
        mask_sad_top5 = importance_masks['SAD']
        mask_hc_top5 = importance_masks['HC']
    
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
        current_mask = mask_sad_top5 if group == "SAD" else mask_hc_top5
            
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
                
            data_rows.append({
                "Subject": sub, "Group": group, "Drug": drug, "Condition": key,
                "Dist_Safety": dist_safety,
                "Dist_Threat": dist_threat
            })
    
    df_topo = pd.DataFrame(data_rows)
    print(f"  > Computed metrics for {len(df_topo)} subjects.")
    
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
    _save_fig("analysis_14")
    _save_fig("results_14_self")
    plt.show()
    
    results_21 = {'df': df_topo, 'p_safe': p_int_safe, 'p_threat': p_int_threat}
    _save_result("results_21", results_21)
    _save_result("results_21", results_21)
    save_checkpoint(13, {
        "results_21": results_21,
        "df_topo": locals().get("df_topo"),
        "lme_results": locals().get("lme_results"),
    })
    save_intermediate("stage13_results_21", {
        "results_21": results_21,
        "df_topo": locals().get("df_topo"),
        "lme_results": locals().get("lme_results"),
    })
    save_stage_bundle(
        5,
        "stage05_SafetyRestoration",
        {
            "results_21": results_21,
            "df_topo": locals().get("df_topo"),
            "p_int_safe": locals().get("p_int_safe"),
            "p_int_threat": locals().get("p_int_threat"),
        },
    )
    
# %% [cell 16]
if stage_active(6):
    # Cell 13: Analysis 2.2 - Drift Efficiency (Safety & Threat Maintenance)
    # Objective: Test OXT effect on neural drift efficiency in the Core Top 5% Network.
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
    PERCENTILE_THRESH = 95  # Top 5%
    
    # =============================================================================
    # 0. Setup: Masks & Data Loading
    # =============================================================================
    print(f"\n[Step 0] Setup & Data Loading...")
    
    if 'importance_scores' not in locals():
        importance_scores, importance_masks = ensure_importance_loaded()
    if "SAD" not in importance_scores or "HC" not in importance_scores:
        missing = [g for g in ("SAD", "HC") if g not in importance_scores]
        raise ValueError(
            f"Stage 9 requires importance_scores for both SAD and HC. Missing: {missing}. "
            "Run Stage 8 with --stage1_group SAD and --stage1_group HC (or ALL), then resume."
        )
    
    if 'importance_masks' not in locals() or not importance_masks:
        importance_scores, importance_masks = ensure_importance_loaded()
    mask_sad_core = importance_masks['SAD']
    mask_hc_core = importance_masks['HC']
    print(f"  > Core Masks: SAD={np.sum(mask_sad_core)}, HC={np.sum(mask_hc_core)}")
    
    # Load Reinstatement Data
    X_rst_all, y_rst_all, sub_rst_all = None, None, None
    if 'X_reinst' in locals():
        X_rst_all, y_rst_all, sub_rst_all = X_reinst, y_reinst, sub_reinst
    else:
        try:
            xs, ys, ss = [], [], []
            for grp in ["SAD_Placebo", "SAD_Oxytocin", "HC_Placebo", "HC_Oxytocin"]:
                if grp in data_subsets and data_subsets[grp]['rst'] is not None:
                    d = data_subsets[grp]['rst']
                    xs.append(d['X']); ys.append(d['y']); ss.append(d['sub'])
            if xs:
                X_rst_all = np.vstack(xs)
                y_rst_all = np.concatenate(ys)
                sub_rst_all = np.concatenate(ss)
        except:
            print("  ! Threat analysis skipped (Reinstatement data missing).")
    
    # =============================================================================
    # 1. Vector Calculation
    # =============================================================================
    subgroups_22 = {"SAD_Placebo": [], "SAD_Oxytocin": [], "HC_Placebo": [], "HC_Oxytocin": []}
    
    if 'sub_to_meta' not in locals():
        if 'meta' in locals():
            sub_to_meta = meta.set_index("subject_id")[["Group", "Drug"]].to_dict('index')
        else: raise ValueError("Metadata not found.")
    
    for sub in np.unique(sub_ext):
        s_str = str(sub).strip()
        if s_str in sub_to_meta: info = sub_to_meta[s_str]
        elif f"sub-{s_str}" in sub_to_meta: info = sub_to_meta[f"sub-{s_str}"]
        else: continue
        key = f"{info['Group']}_{info['Drug']}"
        if key in subgroups_22: subgroups_22[key].append(sub)
    
    data_rows = []
    print("\n[Step 1] Calculating Drift Vectors...")
    
    
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
                
            # 2. Threat
            if X_rst_all is not None:
                m_rst = (sub_rst_all == sub)
                if np.sum(m_rst) > 0:
                    X_r, y_r = X_rst_all[m_rst], y_rst_all[m_rst]
                    res_threat = calc_drift_metrics(X_e, y_e, X_r, y_r, COND_THREAT_LRN, COND_THREAT_LRN, curr_mask, sub)
                    if res_threat:
                        data_rows.append({"Subject": sub, "Group": group, "Drug": drug, "Domain": "Threat", **res_threat})
    
    df_drift = pd.DataFrame(data_rows)
    print(f"  > Computed vectors for {len(df_drift['Subject'].unique())} subjects.")
    
    # =============================================================================
    # 2. Statistics (LME)
    # =============================================================================
    print("\n[Step 2] Statistical Testing (LME)...")
    lme_results = {}
    
    for domain in ["Safety", "Threat"]:
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
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    pal_group = {'SAD': '#c44e52', 'HC': '#4c72b0'}
    
    # Plot Grid
    plot_interaction(axes[0,0], df_drift, "Safety", "Cosine", lme_results.get("Safety_Cosine", 1.0))
    plot_interaction(axes[0,1], df_drift, "Safety", "Projection", lme_results.get("Safety_Projection", 1.0))
    plot_interaction(axes[1,0], df_drift, "Threat", "Cosine", lme_results.get("Threat_Cosine", 1.0))
    plot_interaction(axes[1,1], df_drift, "Threat", "Projection", lme_results.get("Threat_Projection", 1.0))
    
    plt.tight_layout()
    _save_fig("analysis_21")
    _save_fig("results_21")
    plt.show()
    
    print("Note: Error bars represent Standard Error of the Mean (SEM).")
    results_22 = {'df': df_drift, 'stats': lme_results}
    _save_result("results_22", results_22)
    _save_result("results_22", results_22)
    save_checkpoint(14, {
        "results_22": results_22,
        "df_drift": locals().get("df_drift"),
    })
    save_intermediate("stage14_results_22", {
        "results_22": results_22,
        "df_drift": locals().get("df_drift"),
    })
    save_stage_bundle(
        6,
        "stage06_DriftEfficiency",
        {
            "results_22": results_22,
            "df_drift": locals().get("df_drift"),
            "lme_results": locals().get("lme_results"),
        },
    )
    
# %% [cell 17]
if stage_active(7):
    # Cell 14: Analysis 2.3 - The "Probabilistic Opening" Test (Entropy, Kurtosis, Variance)
    # Objective: Test if Oxytocin increases "Cognitive Uncertainty" in SAD.
    # Hypothesis: SAD-OXT will show HIGHER Entropy, LOWER Kurtosis, HIGHER Variance than SAD-PLC.
    # Method: Cross-Validated Probability Extraction -> Metrics.
    # Stats: Linear Mixed Effects (Metric ~ Group * Drug).
    
    print("--- Running Analysis 2.3: Probabilistic Opening (Entropy, Kurtosis, Variance) ---")
    
    # Constants
    COND_CLASS_THREAT = "CSR"
    COND_CLASS_SAFE = "CSS"
    RANDOM_STATE = 42
    
    # =============================================================================
    # 0. Setup: Masks & Data
    # =============================================================================
    if 'importance_masks' not in locals():
        importance_scores, importance_masks = ensure_importance_loaded()
    
    # Define Native Networks
    mask_sad_native = importance_masks['SAD']
    mask_hc_native = importance_masks['HC']
    print(f"  > SAD Native Network: {np.sum(mask_sad_native)} voxels")
    print(f"  > HC Native Network:  {np.sum(mask_hc_native)} voxels")
    
    # Load Subject-Group-Drug Mapping
    if 'sub_to_meta' not in locals():
        if 'meta' in locals():
            sub_to_meta = meta.set_index("subject_id")[["Group", "Drug"]].to_dict('index')
        else: raise ValueError("Metadata not found.")
    
    subgroups_23 = {"SAD_Placebo": [], "SAD_Oxytocin": [], "HC_Placebo": [], "HC_Oxytocin": []}
    for sub in np.unique(sub_ext):
        s_str = str(sub).strip()
        if s_str in sub_to_meta: info = sub_to_meta[s_str]
        elif f"sub-{s_str}" in sub_to_meta: info = sub_to_meta[f"sub-{s_str}"]
        else: continue
        
        key = f"{info['Group']}_{info['Drug']}"
        if key in subgroups_23: subgroups_23[key].append(sub)
    
    
    # =============================================================================
    # 2. Execution Loop
    # =============================================================================
    data_rows = []
    print("\n[Step 1] Calculating Decision Metrics...")
    
    if 'subject_best_params' not in locals(): subject_best_params = {}
    
    for key, sub_list in subgroups_23.items():
        group, drug = key.split('_')
        curr_mask = mask_sad_native if group == "SAD" else mask_hc_native
        
        for sub in sub_list:
            mask_s = (sub_ext == sub)
            X_s, y_s = X_ext[mask_s], y_ext[mask_s]
            
            c_val = subject_best_params.get(sub, 1.0)
            
            res = calc_metrics_for_subject(X_s, y_s, sub, curr_mask, c_val)
            
            if res is not None:
                data_rows.append({
                    "Subject": sub, "Group": group, "Drug": drug, 
                    "Entropy": res['Entropy'], 
                    "Kurtosis": res['Kurtosis'], 
                    "Variance": res['Variance']
                })
    
    df_metrics = pd.DataFrame(data_rows)
    print(f"  > Computed metrics for {len(df_metrics)} subjects.")
    
    # =============================================================================
    # 3. Statistical Testing (LME Loop)
    # =============================================================================
    print("\n[Step 2] Statistical Testing (LME for each metric)...")
    
    stats_results = {}
    metrics_list = ["Entropy", "Kurtosis", "Variance"]
    
    for met in metrics_list:
        print(f"\n--- Metric: {met} ---")
        try:
            # LME: Metric ~ Group * Drug + (1|Subject)
            md = smf.mixedlm(f"{met} ~ C(Group, Treatment(reference='HC')) * C(Drug, Treatment(reference='Placebo'))", 
                             df_metrics, groups=df_metrics["Subject"])
            mdf = md.fit()
            print(mdf.summary())
            
            # Interaction P-Value
            term_int = "C(Group, Treatment(reference='HC'))[T.SAD]:C(Drug, Treatment(reference='Placebo'))[T.Oxytocin]"
            p_val = mdf.pvalues.get(term_int, 1.0)
            stats_results[met] = p_val
            print(f"  >>> Interaction p={p_val:.4f} {'*' if p_val < 0.05 else ''}")
            
        except Exception as e:
            print(f"  ! Model Failed: {e}")
            stats_results[met] = 1.0
    
    # =============================================================================
    # 4. Visualization (Side-by-Side)
    # =============================================================================
    sns.set_context("poster", font_scale=0.8)
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    pal_group = {'SAD': '#c44e52', 'HC': '#4c72b0'}
    
    # Plot all 3
    plot_metric(axes[0], "Entropy", stats_results["Entropy"])
    plot_metric(axes[1], "Kurtosis", stats_results["Kurtosis"])
    plot_metric(axes[2], "Variance", stats_results["Variance"])
    
    axes[1].get_legend().remove()
    axes[2].get_legend().remove()
    axes[0].legend(loc='lower left', fontsize=12)
    
    plt.tight_layout()
    _save_fig("analysis_22")
    _save_fig("results_22")
    plt.show()
    
    results_23 = {'df': df_metrics, 'stats': stats_results}
    _save_result("results_23", results_23)
    _save_result("results_23", results_23)
    save_checkpoint(15, {
        "results_23": results_23,
        "df_ent": locals().get("df_ent"),
    })
    save_intermediate("stage15_results_23", {
        "results_23": results_23,
        "df_ent": locals().get("df_ent"),
    })
    save_stage_bundle(
        7,
        "stage07_ProbabilisticOpening",
        {
            "results_23": results_23,
            "df_metrics": locals().get("df_metrics"),
            "stats_results": locals().get("stats_results"),
        },
    )
    
# %% [cell 18]
if stage_active(8):
    # Cell 15: Analysis 2.4 - Spatial Re-Alignment (The "Normalizing" Effect)
    # Objective: Test if OXT shifts SAD representations to align with the "Healthy" template.
    # Protocol: 
    #   1. Retrieve the 'CSS vs CSR' model specifically from Analysis 1.1 (Cell 6).
    #   2. Cross-Decode on SAD-Placebo vs. SAD-Oxytocin (using full feature set).
    #   3. Metric: Subject-level forced-choice accuracy from decision scores.
    # Visualization: Accuracy Heatmap (Train HC -> Test SAD groups).
    
    print("--- Running Analysis 2.4: Spatial Re-Alignment (Using Analysis 1.1 Output) ---")
    
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import ttest_ind
    
    # Constants
    COND_SAFE = "CSS"
    COND_THREAT = "CSR"
    LABELS = [COND_SAFE, COND_THREAT]
    
    # =============================================================================
    # 0. Setup: Retrieve Correct Model from Cell 6 Results
    # =============================================================================
    # We need the specific model trained on Safety (CSS) vs Threat (CSR).
    target_contrast = "CSS vs CSR"
    alt_contrast = "CSR vs CSS"
    
    if 'res_hc_dict' in locals():
        # Check which key exists in the dictionary
        if target_contrast in res_hc_dict:
            gold_model = res_hc_dict[target_contrast]['model']
            print(f"  > Retrieved Analysis 1.1 Model for: {target_contrast}")
        elif alt_contrast in res_hc_dict:
            gold_model = res_hc_dict[alt_contrast]['model']
            print(f"  > Retrieved Analysis 1.1 Model for: {alt_contrast}")
        else:
            raise ValueError(f"Analysis 1.1 results found, but '{target_contrast}' is missing.\n"
                             f"    Available keys: {list(res_hc_dict.keys())}")
    else:
        raise ValueError("Analysis 1.1 results ('res_hc_dict') not found. Please run Cell 6 first.")
    
    # Verify Classes (Must be Safety/Threat)
    print(f"  > Model Classes: {gold_model.classes_}")
    if COND_THREAT not in gold_model.classes_ or COND_SAFE not in gold_model.classes_:
        raise ValueError(f"CRITICAL: The retrieved model was trained on {gold_model.classes_}, "
                         f"but this analysis requires {LABELS}.")
    
    
    X_sad_plc, y_sad_plc, sub_sad_plc = get_ext_data("SAD_Placebo")
    X_sad_oxt, y_sad_oxt, sub_sad_oxt = get_ext_data("SAD_Oxytocin")
    
    # =============================================================================
    # 1. Cross-Decoding (Subject-Level Forced-Choice Accuracy)
    # =============================================================================
    print("\n[Step 1] Cross-Decoding on SAD Subgroups (Subject-Level Accuracy)...")
    
    # Filter to only the two classes of interest (CSS and CSR)
    mask_sad_plc = np.isin(y_sad_plc, LABELS)
    mask_sad_oxt = np.isin(y_sad_oxt, LABELS)
    
    X_sad_plc_filtered = X_sad_plc[mask_sad_plc]
    y_sad_plc_filtered = y_sad_plc[mask_sad_plc]
    sub_sad_plc_filtered = sub_sad_plc[mask_sad_plc]
    
    X_sad_oxt_filtered = X_sad_oxt[mask_sad_oxt]
    y_sad_oxt_filtered = y_sad_oxt[mask_sad_oxt]
    sub_sad_oxt_filtered = sub_sad_oxt[mask_sad_oxt]
    
    # Decision scores -> subject-level forced-choice accuracy
    scores_plc = gold_model.decision_function(X_sad_plc_filtered)
    scores_oxt = gold_model.decision_function(X_sad_oxt_filtered)
    
    acc_sad_plc = compute_subject_forced_choice_accs(
        y_sad_plc_filtered,
        scores_plc,
        sub_sad_plc_filtered,
        gold_model.classes_
    )
    acc_sad_oxt = compute_subject_forced_choice_accs(
        y_sad_oxt_filtered,
        scores_oxt,
        sub_sad_oxt_filtered,
        gold_model.classes_
    )
    
    m_plc = np.mean(acc_sad_plc) if len(acc_sad_plc) > 0 else 0
    m_oxt = np.mean(acc_sad_oxt) if len(acc_sad_oxt) > 0 else 0
    
    print(f"  > SAD-Placebo Acc (decoded by HC Model):  {m_plc:.1%} (n={len(acc_sad_plc)})")
    print(f"  > SAD-Oxytocin Acc (decoded by HC Model): {m_oxt:.1%} (n={len(acc_sad_oxt)})")
    
    # =============================================================================
    # 2. Statistical Comparison
    # =============================================================================
    print("\n[Step 2] Statistical Test...")
    if len(acc_sad_oxt) > 1 and len(acc_sad_plc) > 1:
        # One-tailed t-test: OXT > Placebo
        t_stat, p_val = ttest_ind(acc_sad_oxt, acc_sad_plc, alternative='greater')
        sig_label = "*" if p_val < 0.05 else "ns"
        print(f"  > Hypothesis (OXT > PLC): t={t_stat:.3f}, p={p_val:.4f} ({sig_label})")
    else:
        print("  ! Insufficient data for statistics.")
        p_val = 1.0; sig_label="nA"
    
    # =============================================================================
    # 3. Visualization (Heatmap)
    # =============================================================================
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
    _save_fig("analysis_23")
    _save_fig("results_23")
    plt.show()
    
    print("\nInterpretation:")
    print(f" - SAD-Placebo Accuracy ({m_plc:.1%}): How well the SAD brain fits the Healthy template naturally.")
    print(f" - SAD-Oxytocin Accuracy ({m_oxt:.1%}): How well it fits AFTER treatment.")
    print(" - A significant increase indicates OXT 'normalizes' the neural code for Threat vs Safety.")
    
    results_24 = {'acc_plc': acc_sad_plc, 'acc_oxt': acc_sad_oxt, 'p_val': p_val, 'model': gold_model}
    _save_result("results_24", results_24)
    _save_result("results_24", results_24)
    save_checkpoint(16, {
        "results_24": results_24
    })
    save_intermediate("stage16_results_24", {"results_24": results_24})
    save_stage_bundle(
        8,
        "stage08_SpatialReAlignment",
        {
            "results_24": results_24,
            "gold_model": locals().get("gold_model"),
            "acc_sad_plc": locals().get("acc_sad_plc"),
            "acc_sad_oxt": locals().get("acc_sad_oxt"),
            "p_val": locals().get("p_val"),
        },
    )
    
# %% [cell 19]
if stage_active(9):
    # Cell 16: Analysis 2.5 - Reverse Cross-Decoding (SAD Template -> HC)
    # Objective: Test if the "Disordered" SAD representation generalizes to Healthy brains.
    # Protocol:
    #   1. Train Model on SAD-Placebo (CSS vs CSR).
    #   2. Feature Selection: Full feature set.
    #   3. Test on HC-Placebo and HC-Oxytocin.
    #   4. Metric: Subject-level forced-choice accuracy from decision scores.
    # Hypothesis: Accuracy should be LOW (near chance), confirming "Functional Specificity".
    
    print("--- Running Analysis 2.5: Reverse Cross-Decoding (SAD -> HC) ---")
    
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import ttest_1samp, ttest_ind
    
    # Constants
    COND_SAFE = "CSS"
    COND_THREAT = "CSR"
    LABELS = [COND_SAFE, COND_THREAT]
    
    # =============================================================================
    # 0. Setup: Full feature set (no mask)
    # =============================================================================
    print("  > Feature Space: Full feature set (no mask)")
    
    
    # Load Groups
    X_sad_plc, y_sad_plc, sub_sad_plc = get_ext_data("SAD_Placebo")
    X_hc_plc, y_hc_plc, sub_hc_plc = get_ext_data("HC_Placebo")
    X_hc_oxt, y_hc_oxt, sub_hc_oxt = get_ext_data("HC_Oxytocin")
    
    # =============================================================================
    # 1. Train SAD-Placebo Model (The "Disordered" Classifier)
    # =============================================================================
    print("\n[Step 1] Training SAD-Placebo Model...")
    
    # Filter for CSS vs CSR
    mask_train = np.isin(y_sad_plc, LABELS)
    X_train = X_sad_plc[mask_train]
    y_train = y_sad_plc[mask_train]
    s_train = sub_sad_plc[mask_train]
    
    # Center (Subject-wise)
    
    # Train Classifier
    sad_model = build_binary_pipeline()
    sad_model.fit(X_train, y_train)
    
    print(f"  > Model Trained on {len(np.unique(s_train))} SAD subjects.")
    print(f"  > Classes: {sad_model.classes_}")
    
    # =============================================================================
    # 2. Cross-Decode on HC Groups (Subject-Level Forced-Choice Accuracy)
    # =============================================================================
    print("\n[Step 2] Testing on HC Subgroups (Subject-Level Accuracy)...")
    
    # Filter to labels of interest (full feature set)
    mask_hc_plc_labels = np.isin(y_hc_plc, LABELS)
    mask_hc_oxt_labels = np.isin(y_hc_oxt, LABELS)
    
    X_hc_plc_filtered = X_hc_plc[mask_hc_plc_labels]
    y_hc_plc_filtered = y_hc_plc[mask_hc_plc_labels]
    sub_hc_plc_filtered = sub_hc_plc[mask_hc_plc_labels]
    
    X_hc_oxt_filtered = X_hc_oxt[mask_hc_oxt_labels]
    y_hc_oxt_filtered = y_hc_oxt[mask_hc_oxt_labels]
    sub_hc_oxt_filtered = sub_hc_oxt[mask_hc_oxt_labels]
    
    scores_hc_plc = sad_model.decision_function(X_hc_plc_filtered)
    scores_hc_oxt = sad_model.decision_function(X_hc_oxt_filtered)
    
    acc_hc_plc = compute_subject_forced_choice_accs(
        y_hc_plc_filtered,
        scores_hc_plc,
        sub_hc_plc_filtered,
        sad_model.classes_
    )
    acc_hc_oxt = compute_subject_forced_choice_accs(
        y_hc_oxt_filtered,
        scores_hc_oxt,
        sub_hc_oxt_filtered,
        sad_model.classes_
    )
    
    m_hc_plc = np.mean(acc_hc_plc) if len(acc_hc_plc) > 0 else 0
    m_hc_oxt = np.mean(acc_hc_oxt) if len(acc_hc_oxt) > 0 else 0
    
    print(f"  > HC-Placebo Acc (decoded by SAD):  {m_hc_plc:.1%} (n={len(acc_hc_plc)})")
    print(f"  > HC-Oxytocin Acc (decoded by SAD): {m_hc_oxt:.1%} (n={len(acc_hc_oxt)})")
    
    # =============================================================================
    # 3. Statistical Comparison
    # =============================================================================
    print("\n[Step 3] Statistical Test (Vs Chance 50%)...")
    
    # Test if HC-Placebo decoding is significantly above chance
    # If p > 0.05, it confirms SAD representations do NOT generalize to HC (High Specificity)
    t_chance, p_chance = ttest_1samp(acc_hc_plc, 0.5)
    sig_chance = "*" if p_chance < 0.05 else "ns"
    
    print(f"  > SAD->HC Generalization (vs 50%): t={t_chance:.3f}, p={p_chance:.4f} ({sig_chance})")
    print("    (Note: 'ns' is GOOD here -> implies disordered code is specific to SAD)")
    
    # Compare HC-PLC vs HC-OXT (Exploratory)
    t_drug, p_drug = ttest_ind(acc_hc_oxt, acc_hc_plc)
    print(f"  > Drug Effect in HC (OXT vs PLC): p={p_drug:.4f}")
    
    # =============================================================================
    # 4. Visualization (Heatmap)
    # =============================================================================
    sns.set_context("poster", font_scale=0.8)
    fig, ax = plt.subplots(figsize=(10, 5))
    
    matrix_data = np.array([[m_hc_plc, m_hc_oxt]])
    annot_data = np.array([
        [f"{m_hc_plc:.3f}\n({sig_chance} vs 0.5)", f"{m_hc_oxt:.3f}"]
    ])
    
    sns.heatmap(matrix_data, annot=annot_data, fmt="", cmap="RdBu_r", 
                vmin=0.3, vmax=0.7, center=0.5, cbar=True,
                xticklabels=['Test: HC-Placebo', 'Test: HC-Oxytocin'], 
                yticklabels=['Train: SAD-Placebo'], ax=ax)
    
    ax.set_title("Analysis 2.5: Reverse Cross-Decoding\n(Does SAD 'Disorder' generalize to Healthy?)")
    plt.yticks(rotation=0) 
    
    plt.tight_layout()
    _save_fig("analysis_24")
    _save_fig("results_24")
    plt.show()
    
    results_25 = {
        'acc_hc_plc': acc_hc_plc,
        'acc_hc_oxt': acc_hc_oxt,
        'model': sad_model,
        'p_chance': p_chance,
        'p_drug': p_drug,
    }
    _save_result("results_25", results_25)
    _save_result("results_25", results_25)
    save_stage_bundle(
        9,
        "stage09_ReverseCrossDecoding",
        {
            "results_25": results_25,
            "sad_model": sad_model,
            "acc_hc_plc": acc_hc_plc,
            "acc_hc_oxt": acc_hc_oxt,
            "p_chance": p_chance,
            "p_drug": p_drug,
        },
    )

# %% [cell 20]
if stage_active(10):
    print("--- Running Stage 10: Clinical Score Loading ---")

    clinical_dir = os.path.join(PROJECT_ROOT, "MRI/source_data/behav")
    LSAS_path = os.path.join(clinical_dir, "SocialSafetyLearning-LSASSubtotals_DATA_2026-04-25_2306.csv")
    ECR_path = os.path.join(clinical_dir, "SocialSafetyLearning-ECR_DATA_2026-04-25_2306.csv")
    DASS_path = os.path.join(clinical_dir, "SocialSafetyLearning-DASS_DATA_2026-04-25_2306.csv")

    df_lsas_raw = pd.read_csv(LSAS_path)
    df_lsas = pd.DataFrame({
        "sub_ID": df_lsas_raw["login_participantid"].astype(str),
        "lsas_fear": df_lsas_raw["lsas_fear_total"],
        "lsas_avoid": df_lsas_raw["lsas_avoid_total"],
        "lsas_total": df_lsas_raw["lsas_total"],
    })

    df_ecr_raw = pd.read_csv(ECR_path)
    df_ecr = pd.DataFrame({
        "sub_ID": df_ecr_raw["login_participantid"].astype(str),
        "ecr_total": df_ecr_raw["ecr_total"],
    })

    df_dass_raw = pd.read_csv(DASS_path)
    depression_items = [
        "dass_q3_positive", "dass_q5_initiative", "dass_q10_forward",
        "dass_q13_blue", "das_q16_enthusiastic", "dass_q17_worth", "dass_q21_life",
    ]
    anxiety_items = [
        "dass_q2_drymouth", "dass_q4_breathing", "dass_q7_trembling",
        "dass_q9_panic", "dass_q15_panic", "dass_q19_heart", "dass_q20_scared",
    ]
    stress_items = [
        "dass_q1_winddown", "dass_q6_overreact", "dass_q8_nervousenergy",
        "dass_q11_agitated", "dass_q12_relax", "dass_q14_intolerant", "dass_q18_touch",
    ]

    df_dass = pd.DataFrame()
    df_dass["sub_ID"] = df_dass_raw["login_participantid"].astype(str)
    df_dass["dass_depression"] = df_dass_raw[depression_items].sum(axis=1) * 2
    df_dass["dass_anxiety"] = df_dass_raw[anxiety_items].sum(axis=1) * 2
    df_dass["dass_stress"] = df_dass_raw[stress_items].sum(axis=1) * 2

    df_scored_clinical = df_dass.merge(df_lsas, on="sub_ID", how="inner").merge(df_ecr, on="sub_ID", how="inner")
    print(f"Clinical scores generated for {len(df_scored_clinical)} subjects.")

    stage10_payload = {
        "df_scored_clinical": df_scored_clinical,
        "clinical_paths": {"LSAS": LSAS_path, "ECR": ECR_path, "DASS": DASS_path},
    }
    _save_result("df_scored_clinical", df_scored_clinical)
    save_stage_bundle(10, "stage10_ClinicalScores", stage10_payload)
    save_intermediate("stage23_clinical_scores", stage10_payload)

# %% [cell 21]
if stage_active(11):
    print("--- Running Stage 11: Neural Clinical Index Generation ---")

    if "results_12" not in globals():
        raise ValueError("results_12 missing. Run/resume Stage 2 before Stage 11.")
    if "results_14_self" not in globals():
        raise ValueError("results_14_self missing. Run/resume Stage 4 before Stage 11.")

    idx_cs_minus, idx_css, idx_csr = 0, 1, 2
    sad_feature_count = max(float(results_12.get("features_sad", 1)), 1.0)
    hc_feature_count = max(float(results_12.get("features_hc", 1)), 1.0)
    rdms_sad_pv = results_12.get("rdms_sad_pv", results_12["rdms_sad"] / sad_feature_count)
    rdms_hc_pv = results_12.get("rdms_hc_pv", results_12["rdms_hc"] / hc_feature_count)
    vA_sad_pv, vB_sad_pv = extract_topology_metrics(rdms_sad_pv, idx_cs_minus, idx_css, idx_csr)
    vA_hc_pv, vB_hc_pv = extract_topology_metrics(rdms_hc_pv, idx_cs_minus, idx_css, idx_csr)

    s_id_sad = np.asarray(results_12.get("subs_sad_rdm", globals().get("subs_sad_rdm", []))).astype(str)
    s_id_hc = np.asarray(results_12.get("subs_hc_rdm", globals().get("subs_hc_rdm", []))).astype(str)
    if len(s_id_sad) != len(vA_sad_pv) or len(s_id_hc) != len(vA_hc_pv):
        raise ValueError("Topology subject IDs do not match RDM metric lengths. Re-run Stage 2 with the updated script.")

    df_neural_topology = pd.DataFrame({
        "sub_ID": np.concatenate([s_id_sad, s_id_hc]).astype(str),
        "Neural_Dist_Threat_Safety": np.concatenate([vA_sad_pv, vA_hc_pv]),
        "Neural_Dist_Safety_Backgr": np.concatenate([vB_sad_pv, vB_hc_pv]),
        "Group": ["SAD"] * len(s_id_sad) + ["HC"] * len(s_id_hc),
    })

    trajectory_payload = globals().get("results_13b") or globals().get("results_13_2") or globals().get("results_13")
    if not isinstance(trajectory_payload, dict) or "data_safe" not in trajectory_payload:
        raise ValueError("Single-trial trajectory data missing. Run/resume Stage 3 before Stage 11.")

    def calculate_subject_slopes(df):
        slopes = []
        if df is None or df.empty:
            return pd.DataFrame(columns=["sub_ID", "Neural_Rigidity_Slope", "Neural_Safety_Mean"])
        for sub in df["sub"].unique():
            sub_data = df[df["sub"] == sub].sort_values("trial")
            if len(sub_data) < 3:
                continue
            slope, _ = np.polyfit(sub_data["trial"], sub_data["score"], 1)
            slopes.append({
                "sub_ID": str(sub),
                "Neural_Rigidity_Slope": slope,
                "Neural_Safety_Mean": sub_data["score"].mean(),
            })
        return pd.DataFrame(slopes)

    df_neural_trajectories = calculate_subject_slopes(trajectory_payload["data_safe"])

    df_sad_stats = results_14_self["df_sad"]
    df_hc_stats = results_14_self["df_hc"]
    df_sad_idx = df_sad_stats[["sub", "entropy", "kurtosis"]].copy()
    df_sad_idx["Group"] = "SAD"
    df_hc_idx = df_hc_stats[["sub", "entropy", "kurtosis"]].copy()
    df_hc_idx["Group"] = "HC"
    df_neural_uncertainty = pd.concat([df_sad_idx, df_hc_idx], ignore_index=True).rename(columns={
        "sub": "sub_ID",
        "entropy": "Neural_Uncertainty_Entropy",
        "kurtosis": "Neural_Sharpness_Kurtosis",
    })
    df_neural_uncertainty["sub_ID"] = df_neural_uncertainty["sub_ID"].astype(str)

    print(f"Topology indices: {len(df_neural_topology)} subjects.")
    print(f"Trajectory indices: {len(df_neural_trajectories)} subjects.")
    print(f"Uncertainty indices: {len(df_neural_uncertainty)} subjects.")

    stage11_payload = {
        "df_neural_topology": df_neural_topology,
        "df_neural_trajectories": df_neural_trajectories,
        "df_neural_uncertainty": df_neural_uncertainty,
        "topology_feature_counts": {"SAD": sad_feature_count, "HC": hc_feature_count},
    }
    _save_result("df_neural_topology", df_neural_topology)
    _save_result("df_neural_trajectories", df_neural_trajectories)
    _save_result("df_neural_uncertainty", df_neural_uncertainty)
    save_stage_bundle(11, "stage11_NeuralClinicalIndices", stage11_payload)
    save_intermediate("stage24_neural_clinical_indices", stage11_payload)

# %% [cell 22]
if stage_active(12):
    print("--- Running Stage 12: Clinical-Neural Master Merge ---")

    required = ["df_scored_clinical", "df_neural_topology", "df_neural_trajectories", "df_neural_uncertainty"]
    missing = [name for name in required if name not in globals()]
    if missing:
        raise ValueError(f"Missing inputs for Stage 12: {missing}. Run/resume Stages 10-11 first.")

    df_final_indecision = (
        df_scored_clinical
        .merge(df_neural_topology, on="sub_ID", how="inner")
        .merge(df_neural_trajectories, on="sub_ID", how="inner")
        .merge(df_neural_uncertainty, on="sub_ID", how="inner")
    )
    group_col_pre_meta = clinical_group_column(df_final_indecision)
    if group_col_pre_meta is not None:
        df_final_indecision["Analysis_Group"] = df_final_indecision[group_col_pre_meta]

    meta_merge = meta.copy()
    meta_merge["sub_ID"] = meta_merge["subject_id"].astype(str)
    df_master_analysis = df_final_indecision.merge(meta_merge, on="sub_ID", how="inner", suffixes=("", "_meta"))
    if "Analysis_Group" not in df_master_analysis.columns:
        group_col = clinical_group_column(df_master_analysis)
        if group_col is not None:
            df_master_analysis["Analysis_Group"] = df_master_analysis[group_col]

    print(f"Merged clinical-neural sample: {len(df_master_analysis)} subjects.")
    stage12_payload = {
        "df_final_indecision": df_final_indecision,
        "df_master_analysis": df_master_analysis,
    }
    _save_result("df_final_indecision", df_final_indecision)
    _save_result("df_master_analysis", df_master_analysis)
    save_stage_bundle(12, "stage12_MasterClinicalNeural", stage12_payload)
    save_intermediate("stage26_master_clinical_neural", stage12_payload)

# %% [cell 23]
if stage_active(13):
    print("--- Running Stage 13: Group-Wise Neural-Clinical Pearson Correlations ---")

    if "df_master_analysis" not in globals():
        raise ValueError("df_master_analysis missing. Run/resume Stage 12 before Stage 13.")

    group_col = clinical_group_column(df_master_analysis)
    groups = sorted(df_master_analysis[group_col].dropna().unique()) if group_col else [None]
    group_results = []
    print(f"{'Group':<6} | {'Neural Metric':<28} | {'Clinical':<12} | {'r':<6} | {'p':<8} | {'Sig'}")
    print("-" * 80)

    for grp in groups:
        df_grp = df_master_analysis if grp is None else df_master_analysis[df_master_analysis[group_col] == grp]
        for n_m in NEURAL_CLINICAL_METRICS:
            for c_i in CLINICAL_INDICES:
                if n_m not in df_grp.columns or c_i not in df_grp.columns:
                    continue
                valid = df_grp[[n_m, c_i]].dropna().apply(pd.to_numeric, errors="coerce").dropna()
                if len(valid) > 5:
                    r_val, p_val = pearsonr(valid[n_m], valid[c_i])
                    sig = get_sig_star(p_val)
                    print(f"{str(grp):<6} | {n_m:<28} | {c_i:<12} | {r_val:<6.2f} | {p_val:<8.4f} | {sig}")
                    group_results.append({
                        "Group": grp, "Neural": n_m, "Clinical": c_i,
                        "r": r_val, "p": p_val, "sig": sig, "n": len(valid),
                    })

    df_res_grp = pd.DataFrame(group_results)
    _save_result("df_neural_clinical_pearson", df_res_grp)
    save_stage_bundle(13, "stage13_NeuralClinicalPearson", {"df_res_grp": df_res_grp})
    save_intermediate("stage27_neural_clinical_pearson", {"df_res_grp": df_res_grp})

# %% [cell 24]
if stage_active(14):
    print("--- Running Stage 14: Partial Neural-Clinical Correlations ---")

    if "df_master_analysis" not in globals():
        raise ValueError("df_master_analysis missing. Run/resume Stage 12 before Stage 14.")

    group_col = clinical_group_column(df_master_analysis)
    groups = sorted(df_master_analysis[group_col].dropna().unique()) if group_col else [None]
    group_results = []
    print(f"{'Group':<6} | {'Neural Metric':<28} | {'Clinical':<12} | {'r_adj':<6} | {'p':<8} | {'Sig'}")
    print("-" * 85)

    for grp in groups:
        df_grp = df_master_analysis if grp is None else df_master_analysis[df_master_analysis[group_col] == grp]
        for n_m in NEURAL_CLINICAL_METRICS:
            for c_i in CLINICAL_INDICES:
                if n_m not in df_grp.columns or c_i not in df_grp.columns:
                    continue
                r_adj, p_val, n_valid = partial_corr_residualized(df_grp, n_m, c_i, CLINICAL_COVARIATES)
                if np.isfinite(r_adj) and np.isfinite(p_val):
                    sig = get_sig_star(p_val)
                    print(f"{str(grp):<6} | {n_m:<28} | {c_i:<12} | {r_adj:<6.2f} | {p_val:<8.4f} | {sig}")
                    group_results.append({
                        "Group": grp, "Neural": n_m, "Clinical": c_i,
                        "r": r_adj, "p": p_val, "sig": sig, "n": n_valid,
                        "covariates": ",".join([c for c in CLINICAL_COVARIATES if c in df_grp.columns]),
                    })

    df_res_partial = pd.DataFrame(group_results)
    _save_result("df_neural_clinical_partial", df_res_partial)
    save_stage_bundle(14, "stage14_NeuralClinicalPartial", {"df_res_partial": df_res_partial})
    save_intermediate("stage28_neural_clinical_partial", {"df_res_partial": df_res_partial})

# %% [cell 25]
if stage_active(15):
    print("--- Running Stage 15: Outlier Removal and Z-Scoring ---")

    if "df_master_analysis" not in globals():
        raise ValueError("df_master_analysis missing. Run/resume Stage 12 before Stage 15.")

    df_master_analysis_z = df_master_analysis.copy()
    all_cols = NEURAL_CLINICAL_METRICS + CLINICAL_INDICES + CLINICAL_COVARIATES
    z_limit = 3.0
    outlier_summary = []

    for col in all_cols:
        if col not in df_master_analysis_z.columns:
            print(f"Warning: {col} not found in dataframe.")
            continue
        numeric_col = pd.to_numeric(df_master_analysis_z[col], errors="coerce")
        initial_z = stats.zscore(numeric_col, nan_policy="omit")
        outlier_mask = np.abs(initial_z) > z_limit
        num_removed = int(np.nansum(outlier_mask))
        df_master_analysis_z.loc[outlier_mask, col] = np.nan
        df_master_analysis_z[f"{col}_z"] = stats.zscore(pd.to_numeric(df_master_analysis_z[col], errors="coerce"), nan_policy="omit")
        outlier_summary.append({"column": col, "outliers_removed": num_removed})
        if num_removed > 0:
            print(f"Column {col:<28}: Removed {num_removed} outliers (> {z_limit} SD)")

    df_master_analysis = df_master_analysis_z
    df_outlier_summary = pd.DataFrame(outlier_summary)
    _save_result("df_master_analysis_z", df_master_analysis_z)
    _save_result("df_outlier_summary", df_outlier_summary)
    save_stage_bundle(
        15,
        "stage15_NeuralClinicalZScore",
        {"df_master_analysis": df_master_analysis, "df_master_analysis_z": df_master_analysis_z, "df_outlier_summary": df_outlier_summary},
    )
    save_intermediate(
        "stage29_neural_clinical_zscore",
        {"df_master_analysis": df_master_analysis, "df_master_analysis_z": df_master_analysis_z, "df_outlier_summary": df_outlier_summary},
    )

# %% [cell 26]
if stage_active(16):
    print("--- Running Stage 16: Z-Scored OLS Neural-Clinical Associations ---")

    if "df_master_analysis" not in globals():
        raise ValueError("df_master_analysis missing. Run/resume Stage 12 or Stage 15 before Stage 16.")

    group_col = clinical_group_column(df_master_analysis)
    groups = sorted(df_master_analysis[group_col].dropna().unique()) if group_col else [None]
    neural_z = [f"{c}_z" for c in NEURAL_CLINICAL_METRICS]
    clinical_z = [f"{c}_z" for c in CLINICAL_INDICES]
    ols_rows = []

    for n_m in neural_z:
        print(f"\n{'=' * 80}\nNEURAL PREDICTOR: {n_m.upper()}\n{'=' * 80}")
        for c_i in clinical_z:
            print(f"\n--- Clinical Outcome: {c_i.upper()} ---")
            for grp in groups:
                df_grp = df_master_analysis if grp is None else df_master_analysis[df_master_analysis[group_col] == grp]
                if n_m not in df_grp.columns or c_i not in df_grp.columns:
                    continue
                analysis_df = df_grp[[n_m, c_i]].dropna().apply(pd.to_numeric, errors="coerce").dropna()
                if len(analysis_df) > 5:
                    X = sm.add_constant(analysis_df[n_m], has_constant="add")
                    y = analysis_df[c_i]
                    model = sm.OLS(y, X).fit()
                    p_val = model.pvalues[n_m]
                    beta = model.params[n_m]
                    print(f"[{str(grp):<3}] N={len(analysis_df):<3} | Beta={beta:>6.3f} | t={model.tvalues[n_m]:>6.2f} | p={p_val:>6.4f}")
                    ols_rows.append({
                        "Group": grp, "Neural_z": n_m, "Clinical_z": c_i,
                        "beta": beta, "t": model.tvalues[n_m], "p": p_val,
                        "n": len(analysis_df), "r_squared": model.rsquared,
                    })
                else:
                    print(f"[{str(grp):<3}] Insufficient data (N < 5)")

    df_ols_results = pd.DataFrame(ols_rows)
    _save_result("df_neural_clinical_ols", df_ols_results)
    save_stage_bundle(16, "stage16_NeuralClinicalOLS", {"df_ols_results": df_ols_results})
    save_intermediate("stage30_neural_clinical_ols", {"df_ols_results": df_ols_results})
