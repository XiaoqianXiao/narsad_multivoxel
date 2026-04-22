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
    9: ["stage09_ReverseCrossDecoding"]
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
    parser.add_argument("--stage", type=int, default=None, help="Run a single logical stage (1-9 or 17).")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load checkpoints for prior cells when running a single stage.",
    )
    parser.add_argument(
        "--stage1_group",
        default=os.environ.get("STAGE1_GROUP", "ALL"),
        choices=["SAD", "HC", "ALL"],
        help="For stage 8, compute importance for SAD, HC, or ALL.",
    )
    parser.add_argument(
        "--importance_source",
        default=os.environ.get("IMPORTANCE_SOURCE", "auto"),
        choices=["auto", "combined", "group"],
        help=(
            "How to load stage08 importance for downstream stages: "
            "'combined' uses only stage08_importance.joblib; "
            "'group' requires stage08_importance_SAD/HC; "
            "'auto' tries combined then per-group."
        ),
    )
    parser.add_argument("--n_jobs", type=int, default=int(os.environ.get("N_JOBS", "1")))
    parser.add_argument("--n_jobs_cv", type=int, default=int(os.environ.get("N_JOBS_CV", "1")))
    return parser.parse_known_args()


_args, _ = parse_runtime_args()
PROJECT_ROOT = _args.project_root
OUTPUT_DIR = _args.output_dir
STAGE = _args.stage
RESUME = _args.resume
IMPORTANCE_SOURCE = _args.importance_source.lower()
N_JOBS = _args.n_jobs
N_JOBS_CV = _args.n_jobs_cv


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
    """Ensure importance_scores/masks are available using user-selected source."""
    global importance_scores, importance_masks
    if "importance_scores" in globals() and importance_scores:
        return importance_scores, importance_masks

    merged_scores = {}
    merged_masks = {}

    def load_combined():
        prev = load_intermediate("stage08_importance")
        merged_scores.update(prev.get("importance_scores", {}))
        merged_masks.update(prev.get("importance_masks", {}))

    def load_groups():
        for grp in ("SAD", "HC"):
            prev = load_intermediate(f"stage08_importance_{grp}")
            merged_scores.update(prev.get("importance_scores", {}))
            merged_masks.update(prev.get("importance_masks", {}))

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
    return importance_scores, importance_masks


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


def calculate_plasticity_vectors(X_learn, y_learn, sub_learn, X_targ, y_targ, sub_targ, cond_learn, cond_target_label):
    unique_subs = np.intersect1d(np.unique(sub_learn), np.unique(sub_targ))
    res = {'sub': [], 'projection': [], 'cosine': [], 'init_dist': []}
    for sub in unique_subs:
        xl = X_learn[sub_learn == sub]
        yl = y_learn[sub_learn == sub]
        xt = X_targ[sub_targ == sub]
        yt = y_targ[sub_targ == sub]
        mask_tgt_cond = yt == cond_target_label
        if np.sum(mask_tgt_cond) == 0:
            continue
        P_target = np.mean(xt[mask_tgt_cond], axis=0)
        idx_lrn = np.where(yl == cond_learn)[0]
        if len(idx_lrn) < 2:
            continue
        cutoff = len(idx_lrn) // 2
        P_start = np.mean(xl[idx_lrn[:cutoff]], axis=0)
        P_end = np.mean(xl[idx_lrn[cutoff:]], axis=0)
        V_axis = P_target - P_start
        V_drift = P_end - P_start
        norm_axis = norm(V_axis)
        norm_drift = norm(V_drift)
        if norm_axis == 0 or norm_drift == 0:
            continue
        dot_prod = np.dot(V_drift, V_axis)
        res['sub'].append(sub)
        res['projection'].append(dot_prod / norm_axis)
        res['cosine'].append(dot_prod / (norm_drift * norm_axis))
        res['init_dist'].append(norm_axis)
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


def calc_trajectory(
    X_learn, y_learn, sub_learn,    # The trials we want to project (the "Movie")
    X_targ, y_targ, sub_targ,       # The dataset containing the Goal State
    mask, 
    cond_learn,                     # Condition to track (e.g., CSS)
    cond_target_label               # Label of Goal State (e.g., CS- or CSR)
):
    # Center Data separately to remove session effects
    
    unique_subs = np.intersect1d(np.unique(sub_learn), np.unique(sub_targ))
    res = {'sub': [], 'trial': [], 'score': []}
    
    for sub in unique_subs:
        # 1. Get Subject Data
        xl = X_L[sub_learn == sub]; yl = y_learn[sub_learn == sub]
        xt = X_T[sub_targ == sub]; yt = y_targ[sub_targ == sub]
        
        # 2. Define Start Point (Early Learning)
        # We define "Start" as the centroid of the FIRST HALF of the learning trials
        mask_l = (yl == cond_learn)
        trials_l = xl[mask_l]
        if len(trials_l) < 2: continue
        
        cutoff = max(1, len(trials_l) // 2)
        P_start = np.mean(trials_l[:cutoff], axis=0)
        
        # 3. Define Target Point
        mask_t = (yt == cond_target_label)
        if np.sum(mask_t) == 0: continue
        P_target = np.mean(xt[mask_t], axis=0)
        
        # 4. Define Axis
        V_axis = P_target - P_start
        sq_norm = np.dot(V_axis, V_axis)
        if sq_norm == 0: continue
        
        # 5. Project Each Trial
        # Logic: Score = ((Trial - Start) . Axis) / ||Axis||^2
        # This normalizes the progress: 0.0 = Start, 1.0 = Target
        
        # We center the trials relative to the Start Point of this specific axis
        trials_centered = trials_l - P_start
        
        scores = np.dot(trials_centered, V_axis) / sq_norm
        
        for i, s in enumerate(scores):
            res['sub'].append(sub)
            res['trial'].append(i + 1)
            res['score'].append(s)
            
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
    
    # Settings
    target_pair = ['CSR', 'CSS']
    n_repeats = 100 # Number of permutation iterations for importance
    PERCENTILE_THRESH = 95  # Top 5% most important voxels
    importance_masks = {}
    importance_scores = {}
    stage1_group = _args.stage1_group.upper()

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

    # =============================================================================
    # 1. Compute Importance for SAD
    # =============================================================================
    if stage1_group in ("SAD", "ALL"):
        print("1. Computing Importance for SAD Placebo...")

        # Slice Data (CSR vs CSS only)
        mask_sad = np.isin(y_sad, target_pair)
        X_sad_p = X_sad[mask_sad]
        y_sad_p = y_sad[mask_sad]
        sub_sad_p = sub_sad[mask_sad]

        # Compute Importance (CV-based)
        imp_sad_mean = compute_perm_importance_cv(
            res_sad['model'], X_sad_p, y_sad_p, sub_sad_p,
            n_repeats=n_repeats, n_splits=SUBJECT_CV_SPLITS
        )

        # Define Mask: Top 5% most important voxels
        thr_sad = np.percentile(imp_sad_mean, PERCENTILE_THRESH)
        mask_sad_sig = imp_sad_mean >= thr_sad
        importance_masks['SAD'] = mask_sad_sig
        importance_scores['SAD'] = imp_sad_mean

        print(f"   > SAD: Found {np.sum(mask_sad_sig)} predictive voxels (Top 5%, thr={thr_sad:.6f}).")

    # =============================================================================
    # 2. Compute Importance for HC
    # =============================================================================
    if stage1_group in ("HC", "ALL"):
        print("2. Computing Importance for HC Placebo...")

        # Slice Data
        mask_hc = np.isin(y_hc, target_pair)
        X_hc_p = X_hc[mask_hc]
        y_hc_p = y_hc[mask_hc]
        sub_hc_p = sub_hc[mask_hc]

        # Compute Importance (CV-based)
        imp_hc_mean = compute_perm_importance_cv(
            res_hc['model'], X_hc_p, y_hc_p, sub_hc_p,
            n_repeats=n_repeats, n_splits=SUBJECT_CV_SPLITS
        )

        # Define Mask: Top 5% most important voxels
        thr_hc = np.percentile(imp_hc_mean, PERCENTILE_THRESH)
        mask_hc_sig = imp_hc_mean >= thr_hc
        importance_masks['HC'] = mask_hc_sig
        importance_scores['HC'] = imp_hc_mean

        print(f"   > HC:  Found {np.sum(mask_hc_sig)} predictive voxels (Top 5%, thr={thr_hc:.6f}).")
    
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
    
    print("Permutated Importance masks generated and stored in 'permutated_importance_masks'.")
    _save_result("results_1_importance_mask_permutated", importance_masks)
    _save_result("results_1_importance_scores_permutated", importance_scores)
    for grp in importance_scores.keys():
        _save_result(f"results_1_importance_masks_permutated_{grp}", {grp: importance_masks.get(grp)})
        _save_result(f"results_1_importance_scores_permutated_{grp}", {grp: importance_scores.get(grp)})
    save_checkpoint(1, {
        "importance_masks_permutated": importance_masks,
        "importance_scores_permutated": importance_scores,
        "PERCENTILE_THRESH_permutated": PERCENTILE_THRESH,
        "thr_sad_permutated": locals().get("thr_sad"),
        "thr_hc_permutated": locals().get("thr_hc"),
        "parcel_names_ext_permutated": parcel_names_ext,
    })
    save_intermediate("stage01_importance_permutated", {
        "importance_masks_permutated": importance_masks,
        "importance_scores_permutated": importance_scores,
        "PERCENTILE_THRESH_permutated": PERCENTILE_THRESH,
        "thr_sad_permutated": locals().get("thr_sad"),
        "thr_hc_permutated": locals().get("thr_hc"),
        "parcel_names_ext_permutated": parcel_names_ext,
    })
    for grp in importance_scores.keys():
        save_intermediate(f"stage01_importance_permutated_{grp}", {
            "importance_masks_permutated": {grp: importance_masks.get(grp)},
            "importance_scores_permutated": {grp: importance_scores.get(grp)},
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
            "PERCENTILE_THRESH": PERCENTILE_THRESH,
            "thr_sad": locals().get("thr_sad"),
            "thr_hc": locals().get("thr_hc"),
            "parcel_names_ext": parcel_names_ext,
        },
    )
    

if stage_active(2):
    # Cell 9: Analysis 1.2 - Static Representational Topology (Top 5% | Centroid)
    # Objective: Characterize the stable organization of the social learning space.
    # Constraint: Top 5% most predictive features per group.
    # Method: Cross-validated Mahalanobis (crossnobis) distance with shrinkage covariance, averaged over split-half repeats.
    # Tests: Group Comparison (SAD vs HC) AND One-Sample Test (Dist > 0).
    
    print("--- Running Analysis 1.2: Static Representational Topology (Top 5% | Centroid) ---")
    
    from scipy.stats import ttest_1samp
    
    # Global Constants
    RDM_CONDITIONS = ["CS-", "CSS", "CSR"] 
    PERCENTILE_THRESH = TOP_PCT  # Top 5%
    
    # =============================================================================
    # 0. Feature Selection (Top 5%)
    # Rationale: focus on the strongest predictive voxels to stabilize crossnobis geometry.
    # =============================================================================
    print(f"\n[Step 0] Selecting Top {100-PERCENTILE_THRESH}% Neural Features...")
    
    if 'importance_scores' not in locals() or not importance_scores:
        importance_scores, importance_masks = ensure_importance_loaded()
    
    scores_sad = importance_scores['SAD']
    mask_sad_top5, thresh_sad = get_top_percentile_mask(scores_sad, PERCENTILE_THRESH)
    
    scores_hc = importance_scores['HC']
    mask_hc_top5, thresh_hc = get_top_percentile_mask(scores_hc, PERCENTILE_THRESH)
    
    print(f"  > SAD Top 5% Network: {np.sum(mask_sad_top5)} voxels (Threshold: {thresh_sad:.5f})")
    print(f"  > HC Top 5% Network:  {np.sum(mask_hc_top5)} voxels (Threshold: {thresh_hc:.5f})")
    
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
    
    # Slice Features (Apply the Top 5% Masks)
    X_sad_12 = X_raw[mask_sad_grp][:, mask_sad_top5]
    y_sad_12 = y_raw[mask_sad_grp]
    sub_sad_12 = sub_raw[mask_sad_grp]
    
    X_hc_12 = X_raw[mask_hc_grp][:, mask_hc_top5]
    y_hc_12 = y_raw[mask_hc_grp]
    sub_hc_12 = sub_raw[mask_hc_grp]
    
    print(f"  > SAD Matrix (Top 5%): {X_sad_12.shape} | HC Matrix (Top 5%): {X_hc_12.shape}")
    
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
    ax1.set_title(f"SAD Topology (Top 5%)\n(n={len(subs_sad_rdm)})")
    
    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(np.mean(rdms_hc, axis=0), annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1.2,
                xticklabels=RDM_CONDITIONS, yticklabels=RDM_CONDITIONS, ax=ax2)
    ax2.set_title(f"HC Topology (Top 5%)\n(n={len(subs_hc_rdm)})")
    
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
    results_12 = {
        "rdms_sad": rdms_sad, "rdms_hc": rdms_hc, 
        "metric_a_stats": (t_a, p_a), "metric_b_stats": (t_b, p_b),
        "features_sad": np.sum(mask_sad_top5), "features_hc": np.sum(mask_hc_top5),
        "one_sample_stats": {"p_a_sad": p_a_sad_0, "p_a_hc": p_a_hc_0, "p_b_sad": p_b_sad_0, "p_b_hc": p_b_hc_0}
    }
    _save_result("results_12", results_12)
    _save_result("results_12", results_12)
    save_checkpoint(9, {
        "results_12": results_12
    })
    save_intermediate("stage09_results_12", results_12)
    save_stage_bundle(
        2,
        "stage02_StaticRepresentationalTopology",
        {
            "results_12": results_12,
            "rdms_sad": locals().get("rdms_sad"),
            "rdms_hc": locals().get("rdms_hc"),
            "mask_sad_top5": locals().get("mask_sad_top5"),
            "mask_hc_top5": locals().get("mask_hc_top5"),
            "p_a": locals().get("p_a"),
            "p_b": locals().get("p_b"),
        },
    )
    
# %% [cell 12]
if stage_active(3):
    # Cell 10: Analysis 1.3 - Dynamic Representational Drift (Top 5% Features)
    # Objective: Quantify plasticity magnitude (Projection) and fidelity (Cosine).
    # Target Definitions:
    #   - Safety:  Extinction CSS -> Extinction CS-
    #   - Threat:  Extinction CSR -> Reinstatement CSR
    # Feature Selection: Top 5% Importance (Permutation Scores)
    
    print("--- Running Analysis 1.3: Dynamic Representational Drift (Top 5% Features) ---")
    
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
    # Rationale: drift vectors are noise-sensitive; use top 5% predictive features for stability.
    # =============================================================================
    print(f"\n[Step 0] Setup & Data Loading...")
    
    if 'importance_scores' not in locals():
        importance_scores, importance_masks = ensure_importance_loaded()
    
    # 1. Select Top 5% Features
    mask_sad, t_sad = get_top_percentile_mask(importance_scores['SAD'], PERCENTILE_THRESH)
    mask_hc, t_hc = get_top_percentile_mask(importance_scores['HC'], PERCENTILE_THRESH)
    
    print(f"  > SAD Top 5% Network: {np.sum(mask_sad)} voxels (Thresh={t_sad:.4f})")
    print(f"  > HC Top 5% Network:  {np.sum(mask_hc)} voxels (Thresh={t_hc:.4f})")
    
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
    _save_fig("analysis_12")
    _save_fig("results_12")
    plt.show()
    
    results_13 = {'safe_sad': df_safe_sad, 'threat_sad': df_threat_sad}
    results_13_main = results_13
    _save_result("results_13", results_13)
    _save_result("results_13", results_13)
    save_checkpoint(10, {
        "results_13": results_13,
        "df_safe_sad": locals().get("df_safe_sad"),
        "df_safe_hc": locals().get("df_safe_hc"),
        "df_threat_sad": locals().get("df_threat_sad"),
        "df_threat_hc": locals().get("df_threat_hc"),
    })
    save_intermediate("stage10_results_13", {
        "results_13": results_13,
        "df_safe_sad": locals().get("df_safe_sad"),
        "df_safe_hc": locals().get("df_safe_hc"),
        "df_threat_sad": locals().get("df_threat_sad"),
        "df_threat_hc": locals().get("df_threat_hc"),
    })
    
# %% [cell 13]
if stage_active(3):
    # Cell 10: Analysis 1.3 - Dynamic Representational Drift (Single-Trial Trajectories)
    # Objective: Visualize plasticity trial-by-trial using Top 5% Features.
    # Method: Project every trial onto the Ideal Axis (Start -> Target).
    #   - Score 0 = Resembles Early Extinction (Start)
    #   - Score 1 = Resembles Target (CS- or Reinstated CSR)
    
    print("--- Running Analysis 1.3: Single-Trial Trajectories (Top 5%) ---")
    
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
    # 0. Feature Selection (Top 5% Positive)
    # =============================================================================
    print(f"\n[Step 0] Selecting Top {100-PERCENTILE_THRESH}% Features...")
    
    # Use Top 5% voxel masks from Analysis 1.1 (Cell 9)
    if 'importance_masks' not in locals():
        raise ValueError("Top 5% masks not found. Run Analysis 1.1 / Cell 9 first.")
    mask_sad = importance_masks['SAD']
    mask_hc = importance_masks['HC']
    print(f"  > Using Top 5% voxels: SAD={int(np.sum(mask_sad))}, HC={int(np.sum(mask_hc))}")
    
    
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
    # 2. Trajectory Calculation Helper
    # =============================================================================
        # Center Data separately to remove session effects
        X_learn = X_learn[:, mask]
        X_targ = X_targ[:, mask]
        
        unique_subs = np.intersect1d(np.unique(sub_learn), np.unique(sub_targ))
        res = {'sub': [], 'trial': [], 'score': []}
        
        for sub in unique_subs:
            # 1. Get Subject Data
            xl = X_learn[sub_learn == sub]; yl = y_learn[sub_learn == sub]
            xt = X_targ[sub_targ == sub]; yt = y_targ[sub_targ == sub]
            
            # 2. Define Start Point (Early Learning)
            # We define "Start" as the centroid of the FIRST HALF of the learning trials
            mask_l = (yl == cond_learn)
            trials_l = xl[mask_l]
            if len(trials_l) < 2: continue
            
            cutoff = max(1, len(trials_l) // 2)
            P_start = np.mean(trials_l[:cutoff], axis=0)
            
            # 3. Define Target Point
            mask_t = (yt == cond_target_label)
            if np.sum(mask_t) == 0: continue
            P_target = np.mean(xt[mask_t], axis=0)
            
            # 4. Define Axis
            V_axis = P_target - P_start
            sq_norm = np.dot(V_axis, V_axis)
            if sq_norm == 0: continue
            
            # 5. Project Each Trial
            # Logic: Score = ((Trial - Start) . Axis) / ||Axis||^2
            # This normalizes the progress: 0.0 = Start, 1.0 = Target
            
            # We center the trials relative to the Start Point of this specific axis
            trials_centered = trials_l - P_start
            
            scores = np.dot(trials_centered, V_axis) / sq_norm
            
            for i, s in enumerate(scores):
                res['sub'].append(sub)
                res['trial'].append(i + 1)
                res['score'].append(s)
                
        return pd.DataFrame(res)
    
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
    _save_fig("analysis_13")
    _save_fig("results_13")
    plt.show()
    
    results_13 = {
        'stats_safe': stats_safe, 
        'stats_threat': stats_threat,
        'data_safe': df_safe,
        'data_threat': df_threat
    }
    _save_result("results_13b", results_13)
    _save_result("results_13b", results_13)
    save_checkpoint(11, {
        "results_13b": results_13
    })
    save_intermediate("stage11_results_13b", {"results_13b": results_13})
    save_stage_bundle(
        3,
        "stage03_DynamicRepresentationalDrift",
        {
            "results_13": globals().get("results_13_main"),
            "results_13b": results_13,
            "df_safe_sad": locals().get("df_safe_sad"),
            "df_safe_hc": locals().get("df_safe_hc"),
            "df_threat_sad": locals().get("df_threat_sad"),
            "df_threat_hc": locals().get("df_threat_hc"),
            "stats_safe": locals().get("stats_safe"),
            "stats_threat": locals().get("stats_threat"),
            "df_safe": locals().get("df_safe"),
            "df_threat": locals().get("df_threat"),
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
    if 'importance_scores' not in locals():
        importance_scores, importance_masks = ensure_importance_loaded()
    
    mask_sad_native = get_significant_mask(importance_scores['SAD'])
    mask_hc_native = get_significant_mask(importance_scores['HC'])
    
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
    _save_fig("analysis_13")
    _save_fig("results_13")
    plt.show()
    
    results_14_self = {'df_sad': df_sad_stats, 'df_hc': df_hc_stats}
    _save_result("results_14_self", results_14_self)
    _save_result("results_14_self", results_14_self)
    save_checkpoint(12, {
        "results_14_self": results_14_self,
        "df_sad_stats": locals().get("df_sad_stats"),
        "df_hc_stats": locals().get("df_hc_stats"),
    })
    save_intermediate("stage12_results_14_self", {
        "results_14_self": results_14_self,
        "df_sad_stats": locals().get("df_sad_stats"),
        "df_hc_stats": locals().get("df_hc_stats"),
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
        # Recompute from importance scores so this stage can run standalone
        importance_scores, importance_masks = ensure_importance_loaded()
        PERCENTILE_THRESH = TOP_PCT
        mask_sad_top5, _ = get_top_percentile_mask(importance_scores['SAD'], PERCENTILE_THRESH)
        mask_hc_top5, _ = get_top_percentile_mask(importance_scores['HC'], PERCENTILE_THRESH)
    
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
    
    mask_sad_core, _ = get_top_percentile_mask(importance_scores['SAD'], PERCENTILE_THRESH)
    mask_hc_core, _ = get_top_percentile_mask(importance_scores['HC'], PERCENTILE_THRESH)
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
    if 'importance_scores' not in locals():
        importance_scores, importance_masks = ensure_importance_loaded()
    
    # Define Native Networks
    mask_sad_native = get_significant_mask(importance_scores['SAD'])
    mask_hc_native = get_significant_mask(importance_scores['HC'])
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
    
def run_stage_17_searchlight_rsm():
    """Run Stage 17 searchlight/parcel RSM analyses with explicit local scope."""
    print("--- Running Cell 17: Searchlight RSM (CSR/CSS/CS-) ---")

    cond_list = ["CSR", "CSS", "CS-"]
    groups_to_run = ["ALL", "SAD_Placebo", "SAD_Oxytocin", "HC_Placebo", "HC_Oxytocin"]
    out_dir = OUTPUT_DIR or os.path.join(
        PROJECT_ROOT, "MRI/derivatives/fMRI_analysis/LSS", "results", "searchlight_rsm"
    )
    os.makedirs(out_dir, exist_ok=True)

    data_root = os.path.join(
        PROJECT_ROOT,
        "MRI/derivatives/fMRI_analysis/LSS",
        "firstLevel",
        "all_subjects/fear_network",
    )
    phase2 = np.load(os.path.join(data_root, "phase2_X_ext_y_ext_roi_voxels.npz"), allow_pickle=True)
    phase3 = np.load(os.path.join(data_root, "phase3_X_reinst_y_reinst_roi_voxels.npz"), allow_pickle=True)
    x_ext = phase2["X_ext"]
    y_ext_local = phase2["y_ext"]
    sub_ext_local = phase2["subjects"]
    x_reinst = phase3["X_reinst"]
    y_reinst_local = phase3["y_reinst"]
    sub_reinst_local = phase3["subjects"]

    meta_path = os.path.join(PROJECT_ROOT, "MRI/source_data/behav/drug_order.csv")
    meta_local = pd.read_csv(meta_path)
    meta_local["subject_id"] = meta_local["subject_id"].astype(str).str.strip()

    def normalize_subject_id_local(subject_id):
        s_str = str(subject_id).strip()
        if s_str.endswith(".0") and s_str.replace(".", "").isdigit():
            s_str = s_str[:-2]
        if s_str.startswith("sub-"):
            s_str = s_str[4:]
        return s_str

    sub_to_meta_local = meta_local.set_index("subject_id")[["Group", "Drug"]].to_dict("index")
    sub_to_meta_norm_local = {
        normalize_subject_id_local(sub_id): values for sub_id, values in sub_to_meta_local.items()
    }

    def group_key_for_subject(subject_id):
        return_key = None
        s_str = normalize_subject_id_local(subject_id)
        if s_str in sub_to_meta_norm_local:
            conds = sub_to_meta_norm_local[s_str]
            return_key = f"{conds['Group']}_{conds['Drug']}"
        return return_key

    def collect_phase_data_local(phase_key, group_key="ALL"):
        if phase_key == "ext":
            x_all, y_all, sub_all = x_ext, y_ext_local, sub_ext_local
        else:
            x_all, y_all, sub_all = x_reinst, y_reinst_local, sub_reinst_local
        if group_key == "ALL":
            return x_all, y_all, sub_all
        selected_subjects = [sub for sub in np.unique(sub_all) if group_key_for_subject(sub) == group_key]
        if not selected_subjects:
            return None, None, None
        mask = np.isin(sub_all, selected_subjects)
        return x_all[mask], y_all[mask], sub_all[mask]

    def build_stage_vectors_local(x_vals, y_vals, sub_vals, stage):
        subjects = np.unique(sub_vals)
        subj_mats = []
        for subject in subjects:
            rows = []
            for cond in cond_list:
                idx = np.where((sub_vals == subject) & (y_vals == cond))[0]
                if len(idx) < 2:
                    rows = []
                    break
                split = len(idx) // 2
                use_idx = idx[:split] if stage == "early" else idx[split:]
                if len(use_idx) == 0:
                    rows = []
                    break
                rows.append(np.mean(x_vals[use_idx], axis=0))
            if rows:
                subj_mats.append(np.vstack(rows))
        return subj_mats

    parcel_rsm_results = {}
    for phase_key, phase_name in [("ext", "Extinction"), ("rst", "Reinstatement")]:
        x_phase, y_phase, sub_phase = collect_phase_data_local(phase_key, group_key="ALL")
        if x_phase is None:
            print(f"  ! {phase_name} data missing. Skipping parcel RSM.")
            continue
        early_mats = build_stage_vectors_local(x_phase, y_phase, sub_phase, "early")
        late_mats = build_stage_vectors_local(x_phase, y_phase, sub_phase, "late")
        if not early_mats or not late_mats:
            print(f"  ! Not enough data for parcel RSM in {phase_name}.")
            continue
        rdm_early = np.mean([squareform(pdist(m, metric="correlation")) for m in early_mats], axis=0)
        rdm_late = np.mean([squareform(pdist(m, metric="correlation")) for m in late_mats], axis=0)
        parcel_rsm_results[f"{phase_key}_early"] = rdm_early
        parcel_rsm_results[f"{phase_key}_late"] = rdm_late
        parcel_rsm_results[f"{phase_key}_delta"] = rdm_late - rdm_early

    voxel_results = {"results_maps": None, "results_pvals": None, "results_fdr": None, "contrast_maps": None, "roi_df": None}
    enable_searchlight = False
    if not enable_searchlight:
        print("  ! Searchlight RSM skipped for parcellation space (requires voxel-wise masks).")
    else:
        roi_dir = os.environ.get(
            "SEARCHLIGHT_ROI_DIR",
            "/Users/xiaoqianxiao/tool/parcellation/Gillian_anatomically_constrained",
        )
        roi_order = [
            "left_acc", "left_amygdala", "left_hippocampus", "left_insula", "left_vmpfc",
            "right_acc", "right_amygdala", "right_hippocampus", "right_insula", "right_vmpfc",
        ]
        search_radius = 2.5
        min_voxels = 20
        n_permutation_searchlight = 200
        alpha_fdr = 0.05

        print("[Stage 17] Building feature-to-voxel mapping...")
        roi_paths = []
        for roi_name in roi_order:
            matches = glob.glob(os.path.join(roi_dir, f"*{roi_name}*.nii*"))
            if not matches:
                raise FileNotFoundError(f"ROI mask not found for: {roi_name}")
            roi_paths.append(matches[0])

        ref_img = nib.load(roi_paths[0])
        ref_shape = ref_img.shape
        coords = []
        roi_feature_idx = {}
        feature_idx = 0
        for roi_name, roi_path in zip(roi_order, roi_paths):
            mask_data = nib.load(roi_path).get_fdata() > 0
            inds = np.column_stack(np.where(mask_data))
            roi_inds = []
            for xyz in inds:
                coords.append(xyz)
                roi_inds.append(feature_idx)
                feature_idx += 1
            roi_feature_idx[roi_name] = np.array(roi_inds, dtype=int)
        coords = np.array(coords)
        neighbors = cKDTree(coords).query_ball_point(coords, r=search_radius)

        def rsm_score_for_sphere(cond_mat, feat_idx):
            if len(feat_idx) < min_voxels:
                return np.nan
            arr = cond_mat[:, feat_idx]
            return np.mean([
                1 - pearsonr(arr[0], arr[1])[0],
                1 - pearsonr(arr[0], arr[2])[0],
                1 - pearsonr(arr[1], arr[2])[0],
            ])

        def compute_searchlight_map(subj_mats):
            if not subj_mats:
                return None
            vals = np.full(coords.shape[0], np.nan)
            for center in range(coords.shape[0]):
                feat_idx = neighbors[center]
                subj_scores = []
                for subj_mat in subj_mats:
                    score = rsm_score_for_sphere(subj_mat, feat_idx)
                    if not np.isnan(score):
                        subj_scores.append(score)
                if subj_scores:
                    vals[center] = float(np.mean(subj_scores))
            return vals

        def permute_labels_within_subject(y_vals, sub_vals, rng):
            y_perm = y_vals.copy()
            for subject in np.unique(sub_vals):
                idx = np.where(sub_vals == subject)[0]
                y_perm[idx] = rng.permutation(y_perm[idx])
            return y_perm

        def permutation_null_maps(x_vals, y_vals, sub_vals, stage, n_perm):
            rng = np.random.default_rng(42)
            null_maps = []
            for perm_idx in range(n_perm):
                if perm_idx == 0 or (perm_idx + 1) % 50 == 0:
                    print(f"    {stage} perm {perm_idx + 1}/{n_perm}")
                y_perm = permute_labels_within_subject(y_vals, sub_vals, rng)
                mats = build_stage_vectors_local(x_vals, y_perm, sub_vals, stage)
                curr_map = compute_searchlight_map(mats)
                if curr_map is not None:
                    null_maps.append(curr_map)
            if not null_maps:
                return None
            return np.array(null_maps)

        def pvals_and_fdr(null_maps, obs_map):
            pvals = np.mean(null_maps >= obs_map, axis=0)
            flat = pvals[~np.isnan(pvals)]
            _, p_fdr, _, _ = multipletests(flat, alpha=alpha_fdr, method="fdr_bh")
            p_fdr_full = np.full_like(pvals, np.nan, dtype=float)
            p_fdr_full[~np.isnan(pvals)] = p_fdr
            return pvals, p_fdr_full

        def to_nifti(vals):
            data = np.full(ref_shape, np.nan, dtype=float)
            for idx, val in enumerate(vals):
                x_val, y_val, z_val = coords[idx]
                data[x_val, y_val, z_val] = val
            return nib.Nifti1Image(data, ref_img.affine)

        print("[Stage 17] Computing voxelwise maps...")
        results_maps = {}
        results_pvals = {}
        results_fdr = {}
        for group_key in groups_to_run:
            for phase_key, phase_name in [("ext", "Extinction"), ("rst", "Reinstatement")]:
                x_phase, y_phase, sub_phase = collect_phase_data_local(phase_key, group_key=group_key)
                if x_phase is None:
                    print(f"  ! {phase_name} data missing for {group_key}. Skipping.")
                    continue
                early_mats = build_stage_vectors_local(x_phase, y_phase, sub_phase, "early")
                late_mats = build_stage_vectors_local(x_phase, y_phase, sub_phase, "late")
                map_early = compute_searchlight_map(early_mats)
                map_late = compute_searchlight_map(late_mats)
                if map_early is None or map_late is None:
                    print(f"  ! Not enough data for {phase_name}, {group_key}.")
                    continue
                results_maps[(group_key, phase_key, "early")] = map_early
                results_maps[(group_key, phase_key, "late")] = map_late
                results_maps[(group_key, phase_key, "delta")] = map_late - map_early

                null_early = permutation_null_maps(x_phase, y_phase, sub_phase, "early", n_permutation_searchlight)
                null_late = permutation_null_maps(x_phase, y_phase, sub_phase, "late", n_permutation_searchlight)
                if null_early is not None:
                    results_pvals[(group_key, phase_key, "early")], results_fdr[(group_key, phase_key, "early")] = (
                        pvals_and_fdr(null_early, map_early)
                    )
                if null_late is not None:
                    results_pvals[(group_key, phase_key, "late")], results_fdr[(group_key, phase_key, "late")] = (
                        pvals_and_fdr(null_late, map_late)
                    )

        contrast_maps = {}
        for phase_key in ["ext", "rst"]:
            for stage_name in ["early", "late", "delta"]:
                sad_map = results_maps.get(("SAD_Placebo", phase_key, stage_name))
                hc_map = results_maps.get(("HC_Placebo", phase_key, stage_name))
                oxt_map = results_maps.get(("SAD_Oxytocin", phase_key, stage_name))
                plc_map = results_maps.get(("SAD_Placebo", phase_key, stage_name))
                if sad_map is not None and hc_map is not None:
                    contrast_maps[("SADminusHC", phase_key, stage_name)] = sad_map - hc_map
                if oxt_map is not None and plc_map is not None:
                    contrast_maps[("OXTminusPLC", phase_key, stage_name)] = oxt_map - plc_map

        roi_rows = []
        for key, vals in results_maps.items():
            group_key, phase_key, stage_name = key
            img = to_nifti(vals)
            fname = f"rsm_{group_key}_{phase_key}_{stage_name}.nii.gz"
            nib.save(img, os.path.join(out_dir, fname))
            for roi_name, idxs in roi_feature_idx.items():
                roi_rows.append(
                    {
                        "group": group_key,
                        "phase": phase_key,
                        "stage": stage_name,
                        "roi": roi_name,
                        "mean_rsm": float(np.nanmean(vals[idxs])),
                    }
                )
        for key, vals in results_pvals.items():
            nib.save(to_nifti(vals), os.path.join(out_dir, f"rsm_pvals_{key[0]}_{key[1]}_{key[2]}.nii.gz"))
        for key, vals in results_fdr.items():
            nib.save(to_nifti(vals), os.path.join(out_dir, f"rsm_fdr_{key[0]}_{key[1]}_{key[2]}.nii.gz"))
        for key, vals in contrast_maps.items():
            nib.save(to_nifti(vals), os.path.join(out_dir, f"rsm_{key[0]}_{key[1]}_{key[2]}.nii.gz"))

        voxel_results = {
            "results_maps": results_maps,
            "results_pvals": results_pvals,
            "results_fdr": results_fdr,
            "contrast_maps": contrast_maps,
            "roi_df": pd.DataFrame(roi_rows),
        }
        voxel_results["roi_df"].to_csv(os.path.join(out_dir, "rsm_roi_summary.csv"), index=False)
        _save_result("results_maps", results_maps)
        _save_result("results_pvals", results_pvals)
        _save_result("results_fdr", results_fdr)

    _save_result("parcel_rsm_results", parcel_rsm_results)
    print("Parcel-level RSM complete.")
    save_checkpoint(
        17,
        {
            "results_maps": voxel_results["results_maps"],
            "results_pvals": voxel_results["results_pvals"],
            "results_fdr": voxel_results["results_fdr"],
            "contrast_maps": voxel_results["contrast_maps"],
            "roi_df": voxel_results["roi_df"],
            "parcel_rsm_results": parcel_rsm_results,
        },
    )
    save_intermediate(
        "stage17_rsm",
        {
            "results_maps": voxel_results["results_maps"],
            "results_pvals": voxel_results["results_pvals"],
            "results_fdr": voxel_results["results_fdr"],
            "contrast_maps": voxel_results["contrast_maps"],
            "roi_df": voxel_results["roi_df"],
            "parcel_rsm_results": parcel_rsm_results,
        },
    )


# %% [cell 20]
if stage_active(17):
    run_stage_17_searchlight_rsm()
