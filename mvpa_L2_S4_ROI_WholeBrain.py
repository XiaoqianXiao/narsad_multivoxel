#!/usr/bin/env python3
"""
MVPA L2 ROI Whole Brain Analysis - S4 Glasser+Tian
Converted from Jupyter notebook to Python script.

This script performs comprehensive MVPA analyses on whole brain ROI data (Glasser+Tian parcellation)
following the exact logic and execution order of mvpa_L2_voxel_FearNetworkAll.py, adapted for ROI data.

ANALYSIS 1: BASELINE NEURAL REPRESENTATIONS
- Analysis 1.1: Neural Dissociation (Pairwise Forced-Choice)
- Analysis 1.2: Spatial Topology & Visualization (Haufe Transform)
- Analysis 1.2: Feature Importance (Permutation)
- Analysis 1.2: Static Representational Topology
- Analysis 1.3: Dynamic Representational Drift
- Analysis 1.3: Single-Trial Trajectories
- Analysis 1.4: Decision Boundary Characteristics

ANALYSIS 2: OXYTOCIN EFFECTS
- Analysis 2.1: Safety Restoration & Threat Discrimination
- Analysis 2.2: Drift Efficiency
- Analysis 2.3: Probabilistic Opening
- Analysis 2.4: Spatial Re-Alignment
- Analysis 2.5: Reverse Cross-Decoding

Usage:
    python mvpa_L2_S4_ROI_WholeBrain.py --output_dir /path/to/output [--project_root /path/to/project] [--skip_analyses 1.1 1.2 1.3 1.4 2.1 2.2 2.3 2.4 2.5]
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import glob
import pickle
from pathlib import Path

from numpy.linalg import norm

from sklearn.model_selection import (
    GridSearchCV, StratifiedGroupKFold, GroupKFold, 
    permutation_test_score, LeaveOneGroupOut, cross_val_score,
    StratifiedKFold, cross_val_predict
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_selection import RFE
from sklearn.utils import resample, shuffle
from sklearn.base import clone
from sklearn.inspection import permutation_importance

import nibabel as nib

from joblib import Parallel, delayed

import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

import itertools
from itertools import combinations

from nilearn import plotting, image, masking
from nilearn.maskers import NiftiLabelsMasker

from scipy import stats
from scipy.stats import (
    pearsonr, ttest_1samp, ttest_ind, entropy, kurtosis,
    levene, shapiro, mannwhitneyu
)
from scipy.spatial.distance import pdist, squareform, cdist

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

import argparse

# Argument parsing (allow run_mvpa.sh to override paths)
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--project_root", default=os.environ.get("PROJECT_ROOT", "/gscratch/fang/NARSAD"))
_parser.add_argument("--output_dir", default=os.environ.get("OUTPUT_DIR"))
_args, _ = _parser.parse_known_args()
PROJECT_ROOT = _args.project_root
OUTPUT_DIR = _args.output_dir

# Output root for all analyses
if OUTPUT_DIR:
    OUT_DIR_MAIN = OUTPUT_DIR
else:
    OUT_DIR_MAIN = os.path.join(PROJECT_ROOT, "MRI/derivatives/fMRI_analysis/LSS", "results", "mvpa_outputs")
os.makedirs(OUT_DIR_MAIN, exist_ok=True)

def _save_result(name: str, obj) -> None:
    from joblib import dump
    path = os.path.join(OUT_DIR_MAIN, f"{name}.joblib")
    try:
        dump(obj, path)
    except Exception as exc:
        print(f"  ! Failed to save {name}: {exc}")

def _save_fig(name: str) -> None:
    try:
        plt.savefig(os.path.join(OUT_DIR_MAIN, f"{name}.png"), dpi=300, bbox_inches="tight")
    except Exception as exc:
        print(f"  ! Failed to save figure {name}: {exc}")


from typing import List, Union

# =============================================================================
# CONFIGURATION
# =============================================================================

# Constants
RANDOM_STATE = 42
N_SPLITS = 5   # GroupKFold folds
INNER_CV_SPLITS = 5     
CS_LABELS = ["CS-", "CSS", "CSR"]  # the three CS types of interest
N_JOBS = 4
MAX_ITER = 5000
thresh_hold_p = 0.05
N_PERMUTATION = 5000
N_REPEATS = 10

# Parcellation Configuration
PARCEL_DIR = "/Users/xiaoqianxiao/tool/parcellation"
GLASSER_PATH = os.path.join(PARCEL_DIR, 'Glasser', 'HCP-MMP1_2mm.nii')
TIAN_PATH = os.path.join(PARCEL_DIR, 'Tian/3T/Subcortex-Only', 'Tian_Subcortex_S4_3T_2009cAsym.nii.gz')

# Nice plotting defaults
sns.set_context("poster")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def build_binary_pipeline():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classification', LogisticRegression(
            penalty='l2', 
            solver='lbfgs', 
            class_weight='balanced', 
            max_iter=MAX_ITER, 
            random_state=RANDOM_STATE, 
            n_jobs=1
        ))
    ])

def subject_wise_centering(X, subjects):
    """Centers data per subject."""
    X_centered = np.zeros_like(X)
    unique_subs = np.unique(subjects)
    for sub in unique_subs:
        mask = (subjects == sub)
        if np.sum(mask) > 0:
            X_centered[mask] = X[mask] - np.mean(X[mask], axis=0)
    return X_centered

def compute_pairwise_forced_choice(y_true, scores, class_labels):
    """Computes accuracy where for each trial, the class with the higher 
    aggregated decision score is chosen."""
    classes = sorted(list(set(y_true)))
    accs = []
    pairs = list(combinations(classes, 2))
    
    for c1, c2 in pairs:
        idx_c1 = np.where(y_true == c1)[0]
        idx_c2 = np.where(y_true == c2)[0]
        if len(idx_c1) == 0 or len(idx_c2) == 0: 
            continue
            
        col_c1 = list(class_labels).index(c1)
        col_c2 = list(class_labels).index(c2)
        
        subset_idx = np.concatenate([idx_c1, idx_c2])
        subset_y = y_true[subset_idx]
        subset_scores = scores[subset_idx]
        
        # Choice logic: is score for C1 > score for C2?
        diff = subset_scores[:, col_c1] - subset_scores[:, col_c2]
        subset_pred = np.where(diff > 0, c1, c2)
        
        accs.append(accuracy_score(subset_y, subset_pred))
        
    return np.mean(accs) if accs else 0.0

def run_pairwise_decoding_analysis(X, y, subjects, n_repeats=10):
    """Pairwise forced-choice decoding analysis."""
    X = np.array(X)
    y = np.array(y)
    subjects = np.array(subjects)
    
    classes = np.unique(y)
    pairs = list(combinations(classes, 2))
    results = {}
    
    print(f"\n=== Starting Repeated Pairwise Decoding ({len(pairs)} pairs, {n_repeats} repeats) ===")
    
    # Define parameter grid
    param_grid = {
        'classification__C': np.logspace(-4, 2, 10)
    }
    
    for c1, c2 in pairs:
        pair_name = f"{c1} vs {c2}"
        print(f"\n--- Analysis: {pair_name} ---")
        mask = np.isin(y, [c1, c2])
        X_pair = X[mask]
        y_pair = y[mask]
        sub_pair = subjects[mask]
        
        # PHASE 1: EVALUATION (Repeated Nested CV with Forced-Choice)
        all_repeat_scores = []
        
        for r in range(n_repeats):
            gkf_outer = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + r)
            cv_inner = StratifiedGroupKFold(n_splits=INNER_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE + r)
            
            repeat_scores = []
            print(f"  > Repeat {r+1}/{n_repeats}...")
            
            for i, (train_idx, test_idx) in enumerate(gkf_outer.split(X_pair, y_pair, groups=sub_pair), 1):
                # Inner loop for hyperparameter tuning
                gs = GridSearchCV(build_binary_pipeline(), param_grid, cv=cv_inner, scoring='accuracy', n_jobs=N_JOBS)
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
        std_cv_acc = np.std(all_repeat_scores)
        print(f"  > Final Mean Forced-Choice Accuracy ({n_repeats} repeats): {avg_cv_acc:.4f} (+/- {std_cv_acc:.4f})")

        # PHASE 2: MODEL GENERATION (Refit on Full Data)
        print("  > Generating final model (Refit on full data for Haufe patterns)...")
        cv_inner_final = StratifiedGroupKFold(n_splits=INNER_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        gs_final = GridSearchCV(build_binary_pipeline(), param_grid, cv=cv_inner_final, scoring='accuracy', n_jobs=N_JOBS)
        gs_final.fit(X_pair, y_pair, groups=sub_pair)
        
        final_model = gs_final.best_estimator_
        
        # Haufe Pattern calculation
        W = final_model.named_steps['classification'].coef_
        X_scaled = final_model.named_steps['scaler'].transform(X_pair)
        A = np.cov(X_scaled, rowvar=False) @ W.T 
        
        results[pair_name] = {
            'model': final_model,
            'accuracy': avg_cv_acc, 
            'std': std_cv_acc,
            'best_C': gs_final.best_params_['classification__C'], 
            'haufe_pattern': A.flatten(), 
            'classes': final_model.classes_,
            'X_pair': X_pair,
            'y_pair': y_pair
        }
    return results

def run_perm_simple(X, y, groups, n_iters):
    """Runs permutation testing iterations for a single job."""
    scores = []
    y_shuffled = y.copy()
    
    pipe = build_binary_pipeline()
    cv = StratifiedGroupKFold(n_splits=N_SPLITS)
    
    for _ in range(n_iters):
        # 1. Shuffle labels randomly
        np.random.shuffle(y_shuffled)
        
        # 2. Run Cross-Validation on shuffled data
        cv_scores = cross_val_score(
            pipe, 
            X, 
            y_shuffled, 
            groups=groups, 
            cv=cv, 
            scoring='accuracy', 
            n_jobs=1
        )
        
        # 3. Store the mean accuracy for this permutation
        scores.append(np.mean(cv_scores))
        
    return scores

def run_cross_decoding(model, X, y, groups, classes=None):
    """Applies a pre-trained model to a new dataset and calculates accuracy per subject."""
    unique_subjects = np.unique(groups)
    accuracies = []

    for sub in unique_subjects:
        mask_sub = (groups == sub)
        X_sub = X[mask_sub]
        y_sub = y[mask_sub]
        
        y_pred = model.predict(X_sub)
        acc = accuracy_score(y_sub, y_pred)
        accuracies.append(acc)

    return np.array(accuracies)

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

def permutation_weight_worker(i, X, y, subjects, pipeline):
    y_perm = shuffle(y, random_state=i)
    return get_robust_weights(X, y_perm, subjects, pipeline, n_boot=1)

def run_wen_paper_analysis_voxelwise(X, y, subjects, pipeline_template, best_C, n_permutations, fdr_alpha=0.05):
    print(f"  Estimating Weights ({n_permutations} perms)...")
    pipe = clone(pipeline_template)
    pipe.set_params(classification__C=best_C)
    obs_weights = get_robust_weights(X, y, subjects, pipe, n_boot=10)
    
    def run_null(i):
        y_shuff = shuffle(y, random_state=i)
        return get_robust_weights(X, y_shuff, subjects, pipe, n_boot=1)

    null_weights_list = Parallel(n_jobs=N_JOBS, verbose=1)(delayed(run_null)(i) for i in range(n_permutations))
    null_weights = np.array(null_weights_list)
    
    null_mean = np.mean(null_weights, axis=0)
    null_std = np.std(null_weights, axis=0)
    z_scores = (obs_weights - null_mean) / (null_std + 1e-12)
    
    n_extreme = np.sum(np.abs(null_weights) >= np.abs(obs_weights), axis=0)
    p_values = (n_extreme + 1) / (n_permutations + 1)
    reject, _, _, _ = multipletests(p_values, alpha=fdr_alpha, method='fdr_bh')
    
    return z_scores, reject

def create_stat_map_from_combined_atlas(z_scores_1d, glasser_data, tian_data, affine):
    """Creates a 3D statistical map from ROI z-scores."""
    stat_map_data = np.zeros_like(glasser_data, dtype=float)
    # Glasser (360 parcels)
    unique_glasser = np.unique(glasser_data)
    for label in range(1, 361): 
        if label in unique_glasser:
            idx = label - 1
            if idx < len(z_scores_1d):
                stat_map_data[glasser_data == label] = z_scores_1d[idx]
    # Tian (16 parcels, offset 360)
    offset = 360 
    unique_tian = np.unique(tian_data)
    for label in range(1, 17):
        if label in unique_tian:
            idx = offset + (label - 1)
            if idx < len(z_scores_1d):
                mask = (tian_data == label)
                stat_map_data[mask] = z_scores_1d[idx]
    return nib.Nifti1Image(stat_map_data, affine)

def get_top_percentile_mask(scores, percentile):
    thresh = np.percentile(scores, percentile)
    mask = (scores >= thresh) & (scores > 0)
    return mask, thresh

def calculate_centroid_rdm(X, y, subjects, conditions):
    unique_subs = np.unique(subjects)
    rdms = []
    sub_ids = []
    
    for sub in unique_subs:
        mask_sub = (subjects == sub)
        X_sub = X[mask_sub]
        y_sub = y[mask_sub]
        
        centroids = []
        valid = True
        for cond in conditions:
            mask_cond = (y_sub == cond)
            if np.sum(mask_cond) == 0: 
                valid = False
                break
            centroids.append(np.mean(X_sub[mask_cond], axis=0))
            
        if valid:
            rdm = squareform(pdist(np.array(centroids), metric='correlation'))
            rdms.append(rdm)
            sub_ids.append(sub)
            
    return np.array(rdms), np.array(sub_ids)

def calculate_plasticity_vectors(
    X_learn, y_learn, sub_learn,
    X_targ, y_targ, sub_targ,
    feature_mask, 
    cond_learn,
    cond_target_label
):
    """Calculates projection of learning onto axis towards Target."""
    X_L = subject_wise_centering(X_learn[:, feature_mask], sub_learn)
    X_T = subject_wise_centering(X_targ[:, feature_mask], sub_targ)
    
    unique_subs = np.intersect1d(np.unique(sub_learn), np.unique(sub_targ))
    res = {'sub': [], 'projection': [], 'cosine': [], 'init_dist': []}
    
    for sub in unique_subs:
        m_l = (sub_learn == sub)
        xl = X_L[m_l]
        yl = y_learn[m_l]
        
        m_t = (sub_targ == sub)
        xt = X_T[m_t]
        yt = y_targ[m_t]
        
        mask_tgt_cond = (yt == cond_target_label)
        if np.sum(mask_tgt_cond) == 0: 
            continue
        P_target = np.mean(xt[mask_tgt_cond], axis=0)
        
        mask_lrn_cond = (yl == cond_learn)
        idx_lrn = np.where(mask_lrn_cond)[0]
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
        
        projection = dot_prod / norm_axis
        cosine = dot_prod / (norm_drift * norm_axis)
        
        res['sub'].append(sub)
        res['projection'].append(projection)
        res['cosine'].append(cosine)
        res['init_dist'].append(norm_axis)
        
    return pd.DataFrame(res)

def calc_trajectory(
    X_learn, y_learn, sub_learn,
    X_targ, y_targ, sub_targ,
    mask, 
    cond_learn,
    cond_target_label
):
    X_L = subject_wise_centering(X_learn[:, mask], sub_learn)
    X_T = subject_wise_centering(X_targ[:, mask], sub_targ)
    
    unique_subs = np.intersect1d(np.unique(sub_learn), np.unique(sub_targ))
    res = {'sub': [], 'trial': [], 'score': []}
    
    for sub in unique_subs:
        xl = X_L[sub_learn == sub]
        yl = y_learn[sub_learn == sub]
        xt = X_T[sub_targ == sub]
        yt = y_targ[sub_targ == sub]
        
        mask_l = (yl == cond_learn)
        trials_l = xl[mask_l]
        if len(trials_l) < 2: 
            continue
        
        cutoff = max(1, len(trials_l) // 2)
        P_start = np.mean(trials_l[:cutoff], axis=0)
        
        mask_t = (yt == cond_target_label)
        if np.sum(mask_t) == 0: 
            continue
        P_target = np.mean(xt[mask_t], axis=0)
        
        V_axis = P_target - P_start
        sq_norm = np.dot(V_axis, V_axis)
        if sq_norm == 0: 
            continue
        
        trials_centered = trials_l - P_start
        scores = np.dot(trials_centered, V_axis) / sq_norm
        
        for i, s in enumerate(scores):
            res['sub'].append(sub)
            res['trial'].append(i + 1)
            res['score'].append(s)
            
    return pd.DataFrame(res)

def calculate_distribution_stats(X, y, subjects, feature_mask, best_params_dict, COND_CLASS_THREAT="CSR", COND_CLASS_SAFE="CSS"):
    X_masked = X[:, feature_mask]
    X_masked = subject_wise_centering(X_masked, subjects)
    
    unique_subs = np.unique(subjects)
    res = {'sub': [], 'entropy': [], 'kurtosis': [], 'variance': [], 'probabilities': []}
    
    for sub in unique_subs:
        c_val = best_params_dict.get(sub, 1.0)
        mask_sub = (subjects == sub)
        X_sub = X_masked[mask_sub]
        y_sub = y[mask_sub]
        
        mask_binary = np.isin(y_sub, [COND_CLASS_THREAT, COND_CLASS_SAFE])
        X_binary = X_sub[mask_binary]
        y_binary = y_sub[mask_binary]
        
        if len(y_binary) < 10: 
            continue
        
        try:
            fixed_model = build_binary_pipeline()
            fixed_model.set_params(classification__C=c_val)
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            probs_all = cross_val_predict(fixed_model, X_binary, y_binary, cv=cv, method='predict_proba', n_jobs=1)
            
            classes = sorted(np.unique(y_binary))
            if COND_CLASS_THREAT not in classes: 
                continue
            idx_threat = classes.index(COND_CLASS_THREAT)
            
            mask_css = (y_binary == COND_CLASS_SAFE)
            if np.sum(mask_css) == 0: 
                continue
            probs_css = probs_all[mask_css, idx_threat]
            
            p_clean = np.clip(probs_css, 1e-9, 1-1e-9)
            trial_entropies = [entropy([p, 1-p], base=2) for p in p_clean]
            
            k_val = kurtosis(probs_css, fisher=True)
            v_val = np.var(probs_css)
            
            res['sub'].append(sub)
            res['entropy'].append(np.mean(trial_entropies))
            res['kurtosis'].append(k_val)
            res['variance'].append(v_val)
            res['probabilities'].append(probs_css)
            
        except Exception as e:
            pass
            
    return pd.DataFrame(res)

def calc_drift_metrics(X_start_phase, y_start_phase, X_tgt_phase, y_tgt_phase, 
                       cond_start, cond_target, mask, sub_id):
    X_s = X_start_phase[:, mask]
    X_s = X_s - np.mean(X_s, axis=0)
    
    X_t = X_tgt_phase[:, mask]
    X_t = X_t - np.mean(X_t, axis=0)
    
    mask_tgt = (y_tgt_phase == cond_target)
    if np.sum(mask_tgt) < 2: 
        return None
    P_target = np.mean(X_t[mask_tgt], axis=0)
    
    mask_lrn = (y_start_phase == cond_start)
    idx_lrn = np.where(mask_lrn)[0]
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

def calc_forced_choice_acc(model, X, y, subs, feature_mask, COND_THREAT="CSR", COND_SAFE="CSS", LABELS=None):
    if LABELS is None:
        LABELS = [COND_SAFE, COND_THREAT]
    
    mask_c = np.isin(y, LABELS)
    X_f = X[mask_c][:, feature_mask]
    y_f = y[mask_c]
    s_f = subs[mask_c]
    
    if len(y_f) == 0: 
        return []

    X_f = subject_wise_centering(X_f, s_f)
    
    scores = model.decision_function(X_f)
    
    df_scores = pd.DataFrame({'sub': s_f, 'cond': y_f, 'score': scores})
    means = df_scores.groupby(['sub', 'cond'])['score'].mean().unstack()
    
    valid_subs = means.dropna().index
    means = means.loc[valid_subs]
    
    pos_idx = np.where(model.classes_ == COND_THREAT)[0][0]
    accs = []
    
    for sub in means.index:
        s_threat = means.loc[sub, COND_THREAT]
        s_safe = means.loc[sub, COND_SAFE]
        
        if pos_idx == 1: 
            correct = s_threat > s_safe
        else: 
            correct = s_threat < s_safe
        accs.append(1.0 if correct else 0.0)
        
    return accs

def perm_ttest_ind(data1, data2, n_perm=N_PERMUTATION):
    """Performs a permutation t-test for two independent samples."""
    from scipy.stats import ttest_ind
    
    # 1. Calculate observed t-statistic
    t_obs, _ = ttest_ind(data1, data2)
    
    # 2. Permutation loop
    pooled = np.concatenate([data1, data2])
    n1 = len(data1)
    null_dist = []
    
    rng = np.random.default_rng(42)
    
    for _ in range(n_perm):
        shuffled = rng.permutation(pooled)
        g1 = shuffled[:n1]
        g2 = shuffled[n1:]
        
        t_shuff, _ = ttest_ind(g1, g2)
        null_dist.append(t_shuff)
        
    null_dist = np.array(null_dist)
    
    # 3. Calculate P-value (Two-tailed)
    p_val = np.mean(np.abs(null_dist) >= np.abs(t_obs))
    
    return t_obs, p_val, np.mean(data1), np.mean(data2)

# =============================================================================
# MAIN ANALYSIS FUNCTIONS
# =============================================================================

def load_data(project_root, output_dir):
    """Load phase2 and phase3 data (Cell 2)."""
    print("--- Cell 2: Data Loading ---")
    
    data_root = os.path.join(project_root, "MRI/derivatives/fMRI_analysis/LSS", "firstLevel", "all_subjects/wholeBrain_S4/cope_ROI")
    phase2_npz_path = os.path.join(data_root, "phase2_X_ext_y_ext_glasser_tian.npz")
    phase3_npz_path = os.path.join(data_root, "phase3_X_reinst_y_reinst_glasser_tian.npz")
    
    # Load Files
    phase2_npz = np.load(phase2_npz_path, allow_pickle=True)
    phase3_npz = np.load(phase3_npz_path, allow_pickle=True)
    
    # Process Phase 2 (Extinction)
    X_ext_raw = phase2_npz["X_ext"]
    y_ext = phase2_npz["y_ext"]
    sub_ext = phase2_npz["subjects"]
    parcel_names_ext = list(phase2_npz["parcel_names"])
    
    print(f"Original Extinction Shape: {X_ext_raw.shape}")
    
    # Process Phase 3 (Reinstatement)
    X_reinst_raw = phase3_npz["X_reinst"]
    y_reinst = phase3_npz["y_reinst"]
    sub_reinst = phase3_npz["subjects"]
    parcel_names_reinst = list(phase3_npz["parcel_names"])
    
    print(f"Original Reinstatement Shape: {X_reinst_raw.shape}")
    
    # Filter for CS Trials Only
    mask_ext = np.isin(y_ext, CS_LABELS)
    mask_reinst = np.isin(y_reinst, CS_LABELS)
    
    X_ext = X_ext_raw[mask_ext]
    y_ext = y_ext[mask_ext]
    sub_ext = sub_ext[mask_ext]
    
    X_reinst = X_reinst_raw[mask_reinst]
    y_reinst = y_reinst[mask_reinst]
    sub_reinst = sub_reinst[mask_reinst]
    
    print("\nAfter CS filtering:")
    print("Phase2 (Ext):", X_ext.shape, np.unique(y_ext, return_counts=True))
    print("Phase3 (Reinst):", X_reinst.shape, np.unique(y_reinst, return_counts=True))
    print(f"Number of parcels: {len(parcel_names_ext)}")
    
    return X_ext, y_ext, sub_ext, X_reinst, y_reinst, sub_reinst, parcel_names_ext

def load_metadata(project_root):
    """Load subject-level metadata (Cell 3)."""
    print("--- Cell 3: Load subject-level metadata ---")
    
    meta_path = os.path.join(project_root, "MRI/source_data/behav/drug_order.csv")
    meta = pd.read_csv(meta_path)
    
    print(meta.head())
    print(meta.columns)
    
    # Basic sanity check
    print(f"\nMetadata loaded: {len(meta)} subjects")
    
    return meta

def prepare_data_subsets(X_ext, y_ext, sub_ext, X_reinst, y_reinst, sub_reinst, meta, output_dir):
    """Prepare data subsets by group and drug condition (Cell 5)."""
    print("--- Cell 5: Data Preparation & Subsetting (Center -> Filter) ---")
    
    # Standardize IDs
    meta['subject_id'] = meta['subject_id'].astype(str).str.strip()
    sub_to_meta = meta.set_index("subject_id")[["Group", "Drug"]].to_dict('index')
    print(f"Metadata loaded for {len(sub_to_meta)} subjects.")
    
    def get_group_key(sub_id):
        s_str = str(sub_id).strip()
        conds = None
        if s_str in sub_to_meta: 
            conds = sub_to_meta[s_str]
        elif f"sub-{s_str}" in sub_to_meta: 
            conds = sub_to_meta[f"sub-{s_str}"]
        elif s_str.replace("sub-", "") in sub_to_meta: 
            conds = sub_to_meta[s_str.replace("sub-", "")]
        
        if conds:
            return f"{conds['Group']}_{conds['Drug']}"
        return None
    
    group_keys = ["SAD_Placebo", "SAD_Oxytocin", "HC_Placebo", "HC_Oxytocin"]
    
    def process_phase_data(X_all, y_all, sub_all, phase_name):
        print(f"\nProcessing {phase_name} Phase...")
        if X_all is None: 
            return {k: None for k in group_keys}
        
        grouped_data = {k: {'X': [], 'y': [], 'sub': []} for k in group_keys}
        
        unique_subs = np.unique(sub_all)
        print(f"  > Found {len(unique_subs)} unique subjects.")
        
        count_missing_meta = 0
        
        for sub in unique_subs:
            g_key = get_group_key(sub)
            if not g_key:
                count_missing_meta += 1
                continue
                
            mask_sub = (sub_all == sub)
            X_sub_full = X_all[mask_sub]
            y_sub_full = y_all[mask_sub]
            
            sub_mean = np.mean(X_sub_full, axis=0)
            X_sub_centered = X_sub_full - sub_mean
            
            mask_cond = np.isin(y_sub_full, ["CSS", "CSR"])
            
            if np.sum(mask_cond) > 0:
                grouped_data[g_key]['X'].append(X_sub_centered[mask_cond])
                grouped_data[g_key]['y'].append(y_sub_full[mask_cond])
                grouped_data[g_key]['sub'].append(np.full(np.sum(mask_cond), sub))
                
        if count_missing_meta > 0:
            print(f"  ! Warning: {count_missing_meta} subjects missing metadata skipped.")

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
    
    ext_subsets = process_phase_data(X_ext, y_ext, sub_ext, "Extinction")
    rst_subsets = process_phase_data(X_reinst, y_reinst, sub_reinst, "Reinstatement")
    
    data_subsets = {}
    for key in group_keys:
        data_subsets[key] = {
            "ext": ext_subsets.get(key),
            "rst": rst_subsets.get(key)
        }
    
    print("\nCell 5 Complete. Data is Centered (Full-Session) and Filtered.")
    
    return data_subsets, sub_to_meta

def analysis_1_1_neural_dissociation(data_subsets, output_dir):
    """Analysis 1.1: Neural Dissociation Execution"""
    print("--- Running Analysis 1.1: Neural Dissociation ---")
    
    def get_extinction_data(group_key):
        if group_key not in data_subsets:
            raise ValueError(f"Group {group_key} missing from data_subsets.")
        phase_data = data_subsets[group_key]['ext']
        if phase_data is None:
            raise ValueError(f"Extinction data missing for {group_key}.")
        return phase_data["X"], phase_data["y"], phase_data["sub"]
    
    X_hc, y_hc, sub_hc = get_extinction_data("HC_Placebo")
    X_sad, y_sad, sub_sad = get_extinction_data("SAD_Placebo")
    print(f"Data Loaded: SAD (n={len(np.unique(sub_sad))}), HC (n={len(np.unique(sub_hc))})")
    
    # TEST 1: Baseline Neural Discriminability
    print("\n--- TEST 1: Baseline Neural Discriminability ---")
    
    print("Processing SAD...")
    res_sad_dict = run_pairwise_decoding_analysis(X_sad, y_sad, sub_sad)
    target_param = 'classification__C'
    best_c_sad = res_sad_dict[list(res_sad_dict.keys())[0]]['model'].get_params()[target_param]
    print(f"  > Best {target_param} for SAD: {best_c_sad}")
    
    print("Processing HC...")
    res_hc_dict = run_pairwise_decoding_analysis(X_hc, y_hc, sub_hc)
    best_c_hc = res_hc_dict[list(res_hc_dict.keys())[0]]['model'].get_params()[target_param]
    print(f"  > Best {target_param} for HC: {best_c_hc}")
    
    pair_key = "CSR vs CSS" if "CSR vs CSS" in res_sad_dict else "CSS vs CSR"
    if pair_key not in res_sad_dict or pair_key not in res_hc_dict:
        raise ValueError(f"Contrast {pair_key} not found.")
    
    res_sad = res_sad_dict[pair_key]
    res_hc = res_hc_dict[pair_key]
    
    # Permutation Test
    print(f"Running Permutation Test (Self-Decoding, {N_PERMUTATION} iter)...")
    iters_per_job = N_PERMUTATION // N_JOBS
    perm_acc_sad = np.concatenate(Parallel(n_jobs=N_JOBS)(delayed(run_perm_simple)(X_sad, y_sad, sub_sad, iters_per_job) for _ in range(N_JOBS)))
    perm_acc_hc = np.concatenate(Parallel(n_jobs=N_JOBS)(delayed(run_perm_simple)(X_hc, y_hc, sub_hc, iters_per_job) for _ in range(N_JOBS)))
    
    p_sad = np.mean(perm_acc_sad >= res_sad['accuracy'])
    p_hc = np.mean(perm_acc_hc >= res_hc['accuracy'])
    
    # TEST 2: Functional Specificity
    print("\n--- TEST 2: Functional Specificity ---")
    
    model_sad = res_sad['model']
    accs_sad2hc = run_cross_decoding(model_sad, X_hc, y_hc, sub_hc, model_sad.classes_)
    mean_sad2hc = np.mean(accs_sad2hc)
    print(f"  > SAD Model -> HC Data: {mean_sad2hc:.4f}")
    
    model_hc = res_hc['model']
    accs_hc2sad = run_cross_decoding(model_hc, X_sad, y_sad, sub_sad, model_hc.classes_)
    mean_hc2sad = np.mean(accs_hc2sad)
    print(f"  > HC Model -> SAD Data: {mean_hc2sad:.4f}")
    
    # TEST 3: Spatial Specificity
    print("\n--- TEST 3: Spatial Specificity ---")
    map_sad, map_hc = res_sad['haufe_pattern'], res_hc['haufe_pattern']
    obs_sim = cosine_similarity(map_sad.reshape(1, -1), map_hc.reshape(1, -1))[0][0]
    
    # Permutation test for spatial similarity
    X_comb = np.concatenate([X_sad, X_hc])
    y_comb = np.concatenate([y_sad, y_hc])
    sub_comb = np.concatenate([sub_sad, sub_hc])
    
    all_sub_maps, all_sub_groups = [], []
    perm_pipe = build_binary_pipeline()
    perm_pipe.set_params(classification__C=1.0)
    
    print(f"Pre-computing {len(np.unique(sub_comb))} individual subject maps...")
    for sub in np.unique(sub_comb):
        mask = sub_comb == sub
        perm_pipe.fit(X_comb[mask], y_comb[mask])
        W = perm_pipe.named_steps['classification'].coef_
        cov = np.cov(perm_pipe.named_steps['scaler'].transform(X_comb[mask]), rowvar=False)
        A = cov @ W.T
        
        if perm_pipe.classes_[1] == 'CSS': 
            A = -A 
        all_sub_maps.append(A.flatten())
        all_sub_groups.append("SAD" if sub in sub_sad else "HC")
    
    print(f"Running Spatial Permutation ({N_PERMUTATION} iter)...")
    def run_spatial_perm(seed, maps, groups):
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(groups)
        w_sad_p = np.mean(maps[shuffled == "SAD"], axis=0)
        w_hc_p = np.mean(maps[shuffled == "HC"], axis=0)
        return cosine_similarity(w_sad_p.reshape(1, -1), w_hc_p.reshape(1, -1))[0][0]
    
    perm_sims = np.array(Parallel(n_jobs=N_JOBS)(delayed(run_spatial_perm)(i, np.array(all_sub_maps), np.array(all_sub_groups)) for i in range(N_PERMUTATION)))
    
    p_sim_spatial = 2 * min(np.mean(perm_sims <= obs_sim), np.mean(perm_sims >= obs_sim))
    
    # VISUALIZATION
    print("\n--- Generating Plots ---")
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
    
    def plot_dist_with_thresh(null_dist, obs_val, ax, title, tail='upper', color='gray'):
        sns.histplot(null_dist, color='gray', stat='density', kde=True, alpha=0.4, ax=ax, label='Null Dist')
        ax.axvline(obs_val, color='red', lw=2.5, label=f'Obs: {obs_val:.2f}')
        if tail == 'upper':
            thresh = np.percentile(null_dist, 95)
            ax.axvline(thresh, color='blue', ls='--', lw=2, label=f'95%: {thresh:.2f}')
            p_val = np.mean(null_dist >= obs_val)
        elif tail == 'lower':
            thresh = np.percentile(null_dist, 5)
            ax.axvline(thresh, color='blue', ls='--', lw=2, label=f'5%: {thresh:.2f}')
            p_val = np.mean(null_dist <= obs_val)
        elif tail == 'two-tailed':
            t_low = np.percentile(null_dist, 2.5)
            t_high = np.percentile(null_dist, 97.5)
            ax.axvline(t_low, color='blue', ls='--', lw=2)
            ax.axvline(t_high, color='blue', ls='--', lw=2)
            p_val = 2 * min(np.mean(null_dist <= obs_val), np.mean(null_dist >= obs_val))
        ax.set_title(f"{title}\n(p = {p_val:.4f})")
        ax.legend(loc='best', fontsize='small')
        return p_val
    
    p_sad = plot_dist_with_thresh(perm_acc_sad, res_sad['accuracy'], fig.add_subplot(gs[0, 0]), 
                                  f"SAD Self-Decoding (CV Acc: {res_sad['accuracy']:.2f})")
    p_hc = plot_dist_with_thresh(perm_acc_hc, res_hc['accuracy'], fig.add_subplot(gs[0, 1]), 
                                 f"HC Self-Decoding (CV Acc: {res_hc['accuracy']:.2f})")
    
    ax3 = fig.add_subplot(gs[1, 0])
    func_matrix = np.array([
        [res_sad['accuracy'], mean_sad2hc], 
        [mean_hc2sad, res_hc['accuracy']]
    ])
    func_pvals = np.array([[p_sad, 0.5], [0.5, p_hc]])  # Placeholder for cross-decoding pvals
    
    annot_func = np.empty_like(func_matrix, dtype=object)
    for i in range(2):
        for j in range(2):
            val_str = f"{func_matrix[i, j]:.3f}"
            sig_str = "*" if func_pvals[i, j] < 0.05 else ""
            annot_func[i, j] = f"{val_str}\n({sig_str})"
    
    sns.heatmap(func_matrix, annot=annot_func, fmt="", cmap="RdBu_r", center=0.5, vmin=0.3, vmax=0.9, cbar=True,
                xticklabels=['Test SAD', 'Test HC'], yticklabels=['Train SAD', 'Train HC'], ax=ax3)
    ax3.set_title("Functional Specificity\n(Standard Accuracy)")
    
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
    fig_path = os.path.join(output_dir, "analysis_1_1_neural_dissociation.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {fig_path}")
    
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
    
    # Save results
    results_path = os.path.join(output_dir, "analysis_1_1_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results_11, f)
    print(f"Saved results to {results_path}")
    
    return res_sad_dict, res_hc_dict, results_11

def analysis_1_2_spatial_topology(data_subsets, res_sad_dict, res_hc_dict, parcel_names, output_dir):
    """Analysis 1.2: Spatial Topology & Visualization (Cell 7/8)"""
    print("--- Cell 7: Running ROI-wise Spatial Analysis ---")
    
    alpha_val = thresh_hold_p
    fdr_alpha = alpha_val
    print(f"FDR Alpha Level: {fdr_alpha}")
    
    # Load atlases
    print("Loading Atlases...")
    glasser_img = nib.load(GLASSER_PATH)
    tian_img = nib.load(TIAN_PATH)
    tian_resampled = image.resample_to_img(tian_img, glasser_img, interpolation='nearest')
    glasser_data = glasser_img.get_fdata().astype(int)
    tian_data = tian_resampled.get_fdata().astype(int)
    
    # Get data
    d_s = data_subsets["SAD_Placebo"]["ext"]
    d_h = data_subsets["HC_Placebo"]["ext"]
    X_sad, y_sad, sub_sad = d_s["X"], d_s["y"], d_s["sub"]
    X_hc, y_hc, sub_hc = d_h["X"], d_h["y"], d_h["sub"]
    
    # Use results from analysis 1.1 (don't recompute)
    pair_key = "CSR vs CSS" if "CSR vs CSS" in res_sad_dict else "CSS vs CSR"
    res_sad = res_sad_dict[pair_key]
    res_hc = res_hc_dict[pair_key]
    
    groups = {
        'SAD': {'X': X_sad, 'y': y_sad, 'sub': sub_sad, 'res': res_sad}, 
        'HC':  {'X': X_hc,  'y': y_hc,  'sub': sub_hc,  'res': res_hc}
    }
    target_pair = ['CSR', 'CSS']
    
    spatial_results = {}
    
    for name, data in groups.items():
        print(f"\nAnalyzing {name}...")
        mask_cls = np.isin(data['y'], target_pair)
        X_curr = data['X'][mask_cls]
        y_curr = data['y'][mask_cls]
        sub_curr = data['sub'][mask_cls]
        X_p = subject_wise_centering(X_curr, sub_curr)
        
        z_scores, sig_mask = run_wen_paper_analysis_voxelwise(
            X_p, y_curr, sub_curr, build_binary_pipeline(), data['res']['best_C'], 1000, fdr_alpha
        )
        
        dummy_pipe = build_binary_pipeline()
        dummy_pipe.fit(X_p, y_curr)
        if dummy_pipe.classes_[0] == 'CSR': 
            z_scores = -z_scores
        
        spatial_results[f"{name} Placebo"] = {'z_scores': z_scores, 'sig_mask': sig_mask}
        
        n_sig = np.sum(sig_mask)
        print(f"Significant Parcels: {n_sig} ({(n_sig/len(z_scores))*100:.2f}%)")
        
        if n_sig > 0:
            try:
                z_masked = z_scores * sig_mask
                print("  > Reconstructing 3D map from atlas...")
                z_img = create_stat_map_from_combined_atlas(z_masked, glasser_data, tian_data, glasser_img.affine)
                
                if z_img is not None:
                    fig = plt.figure(figsize=(16, 6))
                    plotting.plot_glass_brain(
                        z_img, 
                        threshold=1.96, 
                        plot_abs=False, 
                        display_mode='lyrz', 
                        colorbar=True, 
                        vmin=-5, vmax=5, 
                        cmap='RdBu_r', 
                        title=f"{name}: FDR < {fdr_alpha}", 
                        figure=fig
                    )
                    fig_path = os.path.join(output_dir, f"analysis_1_2_spatial_{name.lower()}.png")
                    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved figure to {fig_path}")
                    
            except Exception as e:
                print(f"  ! Visualization failed: {e}")
                import traceback
                traceback.print_exc()
    
    return spatial_results

def analysis_1_3_feature_importance(data_subsets, res_sad_dict, res_hc_dict, output_dir):
    """Analysis 1.3: Feature Importance (Permutation) & Mask Generation"""
    print("--- Cell 8: Generating Permutation Importance Masks ---")
    
    X_sad = data_subsets["SAD_Placebo"]["ext"]["X"]
    y_sad = data_subsets["SAD_Placebo"]["ext"]["y"]
    sub_sad = data_subsets["SAD_Placebo"]["ext"]["sub"]
    
    X_hc = data_subsets["HC_Placebo"]["ext"]["X"]
    y_hc = data_subsets["HC_Placebo"]["ext"]["y"]
    sub_hc = data_subsets["HC_Placebo"]["ext"]["sub"]
    
    target_pair = ['CSR', 'CSS']
    n_repeats = 100
    importance_masks = {}
    importance_scores = {}
    
    print("1. Computing Importance for SAD Placebo...")
    mask_sad = np.isin(y_sad, target_pair)
    X_sad_p = subject_wise_centering(X_sad[mask_sad], sub_sad[mask_sad])
    y_sad_p = y_sad[mask_sad]
    
    pair_key = "CSR vs CSS" if "CSR vs CSS" in res_sad_dict else "CSS vs CSR"
    res_sad = res_sad_dict[pair_key]
    
    imp_sad_mean = permutation_importance(
        res_sad['model'], X_sad_p, y_sad_p, n_repeats=n_repeats, random_state=42, n_jobs=N_JOBS, scoring='accuracy'
    ).importances_mean
    mask_sad_sig = imp_sad_mean > 0
    importance_masks['SAD'] = mask_sad_sig
    importance_scores['SAD'] = imp_sad_mean
    print(f"   > SAD: Found {np.sum(mask_sad_sig)} predictive parcels (Imp > 0).")
    
    print("2. Computing Importance for HC Placebo...")
    mask_hc = np.isin(y_hc, target_pair)
    X_hc_p = subject_wise_centering(X_hc[mask_hc], sub_hc[mask_hc])
    y_hc_p = y_hc[mask_hc]
    
    res_hc = res_hc_dict[pair_key]
    imp_hc_mean = permutation_importance(
        res_hc['model'], X_hc_p, y_hc_p, n_repeats=n_repeats, random_state=42, n_jobs=N_JOBS, scoring='accuracy'
    ).importances_mean
    mask_hc_sig = imp_hc_mean > 0
    importance_masks['HC'] = mask_hc_sig
    importance_scores['HC'] = imp_hc_mean
    print(f"   > HC:  Found {np.sum(mask_hc_sig)} predictive parcels (Imp > 0).")
    
    return importance_masks, importance_scores

def calc_metrics_for_subject(X, y, sub_id, feature_mask, C_param=1.0, COND_CLASS_THREAT="CSR", COND_CLASS_SAFE="CSS"):
    X_m = X[:, feature_mask]
    X_m = X_m - np.mean(X_m, axis=0)
    
    mask_bin = np.isin(y, [COND_CLASS_THREAT, COND_CLASS_SAFE])
    X_bin, y_bin = X_m[mask_bin], y[mask_bin]
    
    if len(y_bin) < 10: 
        return None
    
    try:
        model = build_binary_pipeline()
        model.set_params(classification__C=C_param)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        probs_all = cross_val_predict(model, X_bin, y_bin, cv=cv, method='predict_proba', n_jobs=1)
        
        classes = sorted(np.unique(y_bin))
        if COND_CLASS_THREAT not in classes: 
            return None
        idx_threat = classes.index(COND_CLASS_THREAT)
        
        mask_css = (y_bin == COND_CLASS_SAFE)
        if np.sum(mask_css) == 0: 
            return None
        
        probs_css = probs_all[mask_css, idx_threat]
        
        p_clean = np.clip(probs_css, 1e-9, 1-1e-9)
        ents = [entropy([p, 1-p], base=2) for p in p_clean]
        val_ent = np.mean(ents)
        
        val_kurt = kurtosis(probs_css, fisher=True)
        val_var = np.var(probs_css)
        
        return {'Entropy': val_ent, 'Kurtosis': val_kurt, 'Variance': val_var}
        
    except Exception:
        return None

def analysis_1_2_static_topology(data_subsets, X_ext, y_ext, sub_ext, importance_scores, output_dir):
    """Analysis 1.2: Static Representational Topology (Top 5% | Centroid)"""
    print("--- Running Analysis 1.2: Static Representational Topology (Top 5% | Centroid) ---")
    
    RDM_CONDITIONS = ["CS-", "CSS", "CSR"] 
    PERCENTILE_THRESH = 95
    
    print(f"\n[Step 0] Selecting Top {100-PERCENTILE_THRESH}% Neural Features...")
    
    scores_sad = importance_scores['SAD']
    mask_sad_top5, thresh_sad = get_top_percentile_mask(scores_sad, PERCENTILE_THRESH)
    
    scores_hc = importance_scores['HC']
    mask_hc_top5, thresh_hc = get_top_percentile_mask(scores_hc, PERCENTILE_THRESH)
    
    print(f"  > SAD Top 5% Network: {np.sum(mask_sad_top5)} parcels (Threshold: {thresh_sad:.5f})")
    print(f"  > HC Top 5% Network:  {np.sum(mask_hc_top5)} parcels (Threshold: {thresh_hc:.5f})")
    
    # Data Preparation
    print("\n[Step 1] Preparing Centroid Data...")
    
    known_hc = np.unique(data_subsets["HC_Placebo"]["ext"]["sub"])
    known_sad = np.unique(data_subsets["SAD_Placebo"]["ext"]["sub"])
    
    group_ext = np.array(["Unknown"] * len(sub_ext), dtype=object)
    group_ext[np.isin(sub_ext, known_hc)] = "HC"
    group_ext[np.isin(sub_ext, known_sad)] = "SAD"
    
    mask_conds = np.isin(y_ext, RDM_CONDITIONS)
    X_raw = X_ext[mask_conds]
    y_raw = y_ext[mask_conds]
    sub_raw = sub_ext[mask_conds]
    grp_raw = group_ext[mask_conds]
    
    mask_sad_grp = (grp_raw == "SAD")
    mask_hc_grp = (grp_raw == "HC")
    
    X_sad_12 = X_raw[mask_sad_grp][:, mask_sad_top5]
    y_sad_12 = y_raw[mask_sad_grp]
    sub_sad_12 = sub_raw[mask_sad_grp]
    
    X_hc_12 = X_raw[mask_hc_grp][:, mask_hc_top5]
    y_hc_12 = y_raw[mask_hc_grp]
    sub_hc_12 = sub_raw[mask_hc_grp]
    
    print(f"  > SAD Matrix (Top 5%): {X_sad_12.shape} | HC Matrix (Top 5%): {X_hc_12.shape}")
    
    # Centroid RDM Calculation
    X_sad_12 = subject_wise_centering(X_sad_12, sub_sad_12)
    X_hc_12 = subject_wise_centering(X_hc_12, sub_hc_12)
    
    print(f"  Calculating Centroid RDMs (Conditions: {RDM_CONDITIONS})...")
    rdms_sad, subs_sad_rdm = calculate_centroid_rdm(X_sad_12, y_sad_12, sub_sad_12, RDM_CONDITIONS)
    rdms_hc, subs_hc_rdm = calculate_centroid_rdm(X_hc_12, y_hc_12, sub_hc_12, RDM_CONDITIONS)
    
    print(f"  > Computed RDMs: SAD (n={len(subs_sad_rdm)}), HC (n={len(subs_hc_rdm)})")
    
    # Metrics & Statistical Tests
    idx_cs_minus, idx_css, idx_csr = 0, 1, 2
    
    def extract_metrics(rdms):
        m_a = rdms[:, idx_csr, idx_css] 
        m_b = rdms[:, idx_css, idx_cs_minus] 
        return m_a, m_b
    
    vec_a_sad, vec_b_sad = extract_metrics(rdms_sad)
    vec_a_hc, vec_b_hc = extract_metrics(rdms_hc)
    
    print("\n[Step 3] Statistical Testing...")
    
    def one_sample_test(data, name):
        t_val, p_val = ttest_1samp(data, 0, alternative='greater')
        sig = "*" if p_val < 0.05 else "ns"
        print(f"  > {name}: Mean={np.mean(data):.3f}, t={t_val:.3f}, p={p_val:.4f} ({sig})")
        return p_val
    
    print("\nMetric A: Threat (CSR) vs Safety (CSS) Distance")
    p_a_sad_0 = one_sample_test(vec_a_sad, "SAD (Dist > 0)")
    p_a_hc_0 = one_sample_test(vec_a_hc, "HC  (Dist > 0)")
    
    print("  > Group Comparison (SAD vs HC):")
    t_a, p_a, m_a_sad, m_a_hc = perm_ttest_ind(vec_a_sad, vec_a_hc, n_perm=N_PERMUTATION)
    print(f"    Diff: SAD={m_a_sad:.3f}, HC={m_a_hc:.3f} | t={t_a:.3f}, p={p_a:.4f}")
    
    print("\nMetric B: Safety (CSS) vs Background (CS-) Distance")
    p_b_sad_0 = one_sample_test(vec_b_sad, "SAD (Dist > 0)")
    p_b_hc_0 = one_sample_test(vec_b_hc, "HC  (Dist > 0)")
    
    print("  > Group Comparison (SAD vs HC):")
    t_b, p_b, m_b_sad, m_b_hc = perm_ttest_ind(vec_b_sad, vec_b_hc, n_perm=N_PERMUTATION)
    print(f"    Diff: SAD={m_b_sad:.3f}, HC={m_b_hc:.3f} | t={t_b:.3f}, p={p_b:.4f}")
    
    # Visualization
    fig = plt.figure(figsize=(24, 8))
    gs = fig.add_gridspec(1, 3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(np.mean(rdms_sad, axis=0), annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1.2, 
                xticklabels=RDM_CONDITIONS, yticklabels=RDM_CONDITIONS, ax=ax1, cbar=False)
    ax1.set_title(f"SAD Topology (Top 5%)\n(n={len(subs_sad_rdm)})")
    
    ax2 = fig.add_subplot(gs[0, 1])
    sns.heatmap(np.mean(rdms_hc, axis=0), annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1.2,
                xticklabels=RDM_CONDITIONS, yticklabels=RDM_CONDITIONS, ax=ax2)
    ax2.set_title(f"HC Topology (Top 5%)\n(n={len(subs_hc_rdm)})")
    
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
    ax3.set_ylabel("Correlation Distance (1-r)")
    
    y_max = df_res['Distance'].max()
    if p_a < 0.05: ax3.text(0, y_max + 0.05, f'* (p={p_a:.3f})', ha='center', fontsize=18)
    if p_b < 0.05: ax3.text(1, y_max + 0.05, f'* (p={p_b:.3f})', ha='center', fontsize=18)
    
    def get_sig_star(p): return "*" if p < 0.05 else "ns"
    
    ax3.text(-0.2, -0.15, f"SAD: {get_sig_star(p_a_sad_0)}", transform=ax3.get_xaxis_transform(), ha='center', fontsize=14, color='#c44e52')
    ax3.text(0.2, -0.15, f"HC: {get_sig_star(p_a_hc_0)}", transform=ax3.get_xaxis_transform(), ha='center', fontsize=14, color='#4c72b0')
    ax3.text(0.8, -0.15, f"SAD: {get_sig_star(p_b_sad_0)}", transform=ax3.get_xaxis_transform(), ha='center', fontsize=14, color='#c44e52')
    ax3.text(1.2, -0.15, f"HC: {get_sig_star(p_b_hc_0)}", transform=ax3.get_xaxis_transform(), ha='center', fontsize=14, color='#4c72b0')
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "analysis_1_2_static_topology.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {fig_path}")
    
    results_12 = {
        "rdms_sad": rdms_sad, "rdms_hc": rdms_hc, 
        "metric_a_stats": (t_a, p_a), "metric_b_stats": (t_b, p_b),
        "features_sad": np.sum(mask_sad_top5), "features_hc": np.sum(mask_hc_top5),
        "one_sample_stats": {"p_a_sad": p_a_sad_0, "p_a_hc": p_a_hc_0, "p_b_sad": p_b_sad_0, "p_b_hc": p_b_hc_0}
    }
    _save_result("results_12", results_12)
    
    results_path = os.path.join(output_dir, "analysis_1_2_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results_12, f)
    print(f"Saved results to {results_path}")
    
    return results_12, mask_sad_top5, mask_hc_top5

def analysis_1_3_dynamic_drift(data_subsets, X_ext, y_ext, sub_ext, X_reinst, y_reinst, sub_reinst, importance_scores, output_dir, X_ext_global=None, y_ext_global=None, sub_ext_global=None):
    """Analysis 1.3: Dynamic Representational Drift (Top 5% Features)"""
    print("--- Running Analysis 1.3: Dynamic Representational Drift (Top 5% Features) ---")
    
    COND_SAFETY_TARGET = "CS-"
    COND_SAFETY_LEARN = "CSS"
    COND_THREAT_LEARN = "CSR"
    PERCENTILE_THRESH = 95
    
    print(f"\n[Step 0] Setup & Data Loading...")
    
    mask_sad, t_sad = get_top_percentile_mask(importance_scores['SAD'], PERCENTILE_THRESH)
    mask_hc, t_hc = get_top_percentile_mask(importance_scores['HC'], PERCENTILE_THRESH)
    
    print(f"  > SAD Top 5% Network: {np.sum(mask_sad)} parcels (Thresh={t_sad:.4f})")
    print(f"  > HC Top 5% Network:  {np.sum(mask_hc)} parcels (Thresh={t_hc:.4f})")
    
    def get_phase_data(group, phase):
        try:
            d = data_subsets[group][phase]
            if d is None: return None, None, None
            return d["X"], d["y"], d["sub"]
        except KeyError:
            return None, None, None
    
    X_ext_sad, y_ext_sad, sub_ext_sad = get_phase_data("SAD_Placebo", "ext")
    X_ext_hc, y_ext_hc, sub_ext_hc = get_phase_data("HC_Placebo", "ext")
    
    X_rst_sad, y_rst_sad, sub_rst_sad = get_phase_data("SAD_Placebo", "rst")
    X_rst_hc, y_rst_hc, sub_rst_hc = get_phase_data("HC_Placebo", "rst")
    
    if X_rst_sad is None or X_rst_hc is None:
        print("  ! WARNING: Reinstatement data missing. Threat analysis will fallback to Extinction (Trivial).")
        X_rst_sad, y_rst_sad, sub_rst_sad = X_ext_sad, y_ext_sad, sub_ext_sad
        X_rst_hc, y_rst_hc, sub_rst_hc = X_ext_hc, y_ext_hc, sub_ext_hc
    
    X_global, y_global, sub_global = X_ext, y_ext, sub_ext
    
    print("\n[Step 2] Calculating Vectors...")
    
    print("  > Safety Analysis: Start=CSS(Ext) -> Target=CS-(Ext)")
    df_safe_sad = calculate_plasticity_vectors(
        X_ext_sad, y_ext_sad, sub_ext_sad,
        X_global, y_global, sub_global,
        mask_sad, COND_SAFETY_LEARN, COND_SAFETY_TARGET
    )
    df_safe_hc = calculate_plasticity_vectors(
        X_ext_hc, y_ext_hc, sub_ext_hc, 
        X_global, y_global, sub_global, 
        mask_hc, COND_SAFETY_LEARN, COND_SAFETY_TARGET
    )
    
    print("  > Threat Analysis: Start=CSR(Ext) -> Target=CSR(Reinstatement)")
    df_threat_sad = calculate_plasticity_vectors(
        X_ext_sad, y_ext_sad, sub_ext_sad,
        X_rst_sad, y_rst_sad, sub_rst_sad,
        mask_sad, COND_THREAT_LEARN, COND_THREAT_LEARN 
    )
    df_threat_hc = calculate_plasticity_vectors(
        X_ext_hc, y_ext_hc, sub_ext_hc, 
        X_rst_hc, y_rst_hc, sub_rst_hc, 
        mask_hc, COND_THREAT_LEARN, COND_THREAT_LEARN
    )
    
    def tag_df(df, grp, cond):
        if df.empty: return df
        d = df.copy()
        d['Group'] = grp
        d['Condition'] = cond
        return d
    
    df_plot = pd.concat([
        tag_df(df_safe_sad, 'SAD', 'Safety'), tag_df(df_safe_hc, 'HC', 'Safety'),
        tag_df(df_threat_sad, 'SAD', 'Threat'), tag_df(df_threat_hc, 'HC', 'Threat')
    ])
    
    if df_plot.empty:
        print("! No data generated. Check inputs.")
        return None
    else:
        print(f"\n[Step 3] Generated {len(df_plot)} subject vectors.")
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        sns.barplot(data=df_plot, x='Condition', y='projection', hue='Group', 
                    palette={'SAD': '#c44e52', 'HC': '#4c72b0'}, ax=axes[0,0], 
                    capsize=.1, errorbar='se')
        axes[0,0].axhline(0, color='k', ls='--')
        axes[0,0].set_title("Magnitude (Scalar Projection)")
        
        sns.barplot(data=df_plot, x='Condition', y='cosine', hue='Group', 
                    palette={'SAD': '#c44e52', 'HC': '#4c72b0'}, ax=axes[0,1], 
                    capsize=.1, errorbar='se')
        axes[0,1].axhline(0, color='k', ls='--')
        axes[0,1].set_title("Directional Fidelity (Cosine)")
        
        print("\n--- Statistical Summary (SAD vs HC) ---")
        for cond in ['Safety', 'Threat']:
            print(f"\nCondition: {cond}")
            for met in ['projection', 'cosine']:
                d_s = df_plot[(df_plot['Condition']==cond) & (df_plot['Group']=='SAD')][met]
                d_h = df_plot[(df_plot['Condition']==cond) & (df_plot['Group']=='HC')][met]
                
                if len(d_s)>1: 
                    t0_s, p0_s = ttest_1samp(d_s, 0, alternative='greater')
                    print(f"  > SAD > 0 ({met}): t={t0_s:.3f}, p={p0_s:.4f}")
                if len(d_h)>1:
                    t0_h, p0_h = ttest_1samp(d_h, 0, alternative='greater')
                    print(f"  > HC  > 0 ({met}): t={t0_h:.3f}, p={p0_h:.4f}")

                if len(d_s)>1 and len(d_h)>1:
                    t, p = ttest_ind(d_s, d_h)
                    sig = "*" if p < 0.05 else "ns"
                    print(f"  > Group Diff ({met}): t={t:.3f}, p={p:.4f} {sig}")

        sns.scatterplot(data=df_plot, x='init_dist', y='projection', hue='Group', style='Condition', 
                        palette={'SAD': '#c44e52', 'HC': '#4c72b0'}, alpha=0.7, ax=axes[1,0], s=100)
        axes[1,0].axhline(0, color='k', ls='--')
        axes[1,0].set_title("Learning vs Initial Distance")
        
        axes[1,1].axis('off')
        plt.tight_layout()
        fig_path = os.path.join(output_dir, "analysis_1_3_dynamic_drift.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved figure to {fig_path}")
        
        results_13 = {'safe_sad': df_safe_sad, 'threat_sad': df_threat_sad}
        _save_result("results_13", results_13)
        results_path = os.path.join(output_dir, "analysis_1_3_drift_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(results_13, f)
        print(f"Saved results to {results_path}")
        
        return results_13

def analysis_1_3_trajectories(data_subsets, X_ext, y_ext, sub_ext, X_reinst, y_reinst, sub_reinst, importance_scores, output_dir, X_ext_global=None, y_ext_global=None, sub_ext_global=None):
    """Analysis 1.3: Single-Trial Trajectories (Top 5%)"""
    print("--- Running Analysis 1.3: Single-Trial Trajectories (Top 5%) ---")
    
    COND_SAFETY_TARGET = "CS-"
    COND_SAFETY_LEARN = "CSS"
    COND_THREAT_LEARN = "CSR"
    PERCENTILE_THRESH = 95
    BLOCK_SIZE = 1
    
    print(f"\n[Step 0] Selecting Top {100-PERCENTILE_THRESH}% Features...")
    
    mask_sad, t_sad = get_top_percentile_mask(importance_scores['SAD'], PERCENTILE_THRESH)
    mask_hc, t_hc = get_top_percentile_mask(importance_scores['HC'], PERCENTILE_THRESH)
    
    print(f"  > SAD Top 5%: {np.sum(mask_sad)} parcels (Thresh: {t_sad:.4f})")
    print(f"  > HC Top 5%:  {np.sum(mask_hc)} parcels (Thresh: {t_hc:.4f})")
    
    def get_phase_data(group, phase):
        try:
            d = data_subsets[group][phase]
            if d is None: return None, None, None
            return d["X"], d["y"], d["sub"]
        except KeyError:
            return None, None, None
    
    X_ext_sad, y_ext_sad, sub_ext_sad = get_phase_data("SAD_Placebo", "ext")
    X_ext_hc, y_ext_hc, sub_ext_hc = get_phase_data("HC_Placebo", "ext")
    
    X_rst_sad, y_rst_sad, sub_rst_sad = get_phase_data("SAD_Placebo", "rst")
    X_rst_hc, y_rst_hc, sub_rst_hc = get_phase_data("HC_Placebo", "rst")
    
    if X_rst_sad is None:
        print("  ! WARNING: Reinstatement data missing. Using Extinction as placeholder.")
        X_rst_sad, y_rst_sad, sub_rst_sad = X_ext_sad, y_ext_sad, sub_ext_sad
        X_rst_hc, y_rst_hc, sub_rst_hc = X_ext_hc, y_ext_hc, sub_ext_hc
    
    if X_ext_global is not None:
        X_glob, y_glob, sub_glob = X_ext_global, y_ext_global, sub_ext_global
    else:
        X_glob, y_glob, sub_glob = X_ext_sad, y_ext_sad, sub_ext_sad
    
    print("\n[Step 2] Calculating Single-Trial Trajectories...")
    
    print("  > Safety: CSS Trials projecting onto [Early CSS -> CS-]")
    df_safe_sad = calc_trajectory(X_ext_sad, y_ext_sad, sub_ext_sad, X_glob, y_glob, sub_glob, mask_sad, COND_SAFETY_LEARN, COND_SAFETY_TARGET)
    df_safe_hc = calc_trajectory(X_ext_hc, y_ext_hc, sub_ext_hc, X_glob, y_glob, sub_glob, mask_hc, COND_SAFETY_LEARN, COND_SAFETY_TARGET)
    
    print("  > Threat: CSR Trials projecting onto [Early CSR -> Reinstated CSR]")
    df_threat_sad = calc_trajectory(X_ext_sad, y_ext_sad, sub_ext_sad, X_rst_sad, y_rst_sad, sub_rst_sad, mask_sad, COND_THREAT_LEARN, COND_THREAT_LEARN)
    df_threat_hc = calc_trajectory(X_ext_hc, y_ext_hc, sub_ext_hc, X_rst_hc, y_rst_hc, sub_rst_hc, mask_hc, COND_THREAT_LEARN, COND_THREAT_LEARN)
    
    def run_detailed_stats(df_sad, df_hc, label):
        if df_sad.empty or df_hc.empty: return pd.DataFrame()
        
        trials = sorted(list(set(df_sad['trial'].unique()) & set(df_hc['trial'].unique())))
        results = []
        
        for t in trials:
            s_vals = df_sad[df_sad['trial'] == t]['score'].values
            h_vals = df_hc[df_hc['trial'] == t]['score'].values
            
            t_s, p_s = ttest_1samp(s_vals, 0, alternative='greater')
            df_s = len(s_vals) - 1
            
            t_h, p_h = ttest_1samp(h_vals, 0, alternative='greater')
            df_h = len(h_vals) - 1
            
            t_d, p_d = ttest_ind(s_vals, h_vals)
            df_d = len(s_vals) + len(h_vals) - 2
            
            results.append({
                'Trial': t,
                'SAD_t': t_s, 'SAD_df': df_s, 'SAD_p': p_s,
                'HC_t': t_h, 'HC_df': df_h, 'HC_p': p_h,
                'Diff_t': t_d, 'Diff_df': df_d, 'Diff_p': p_d
            })
            
        stats_df = pd.DataFrame(results)
        
        if not stats_df.empty:
            _, stats_df['SAD_p_fdr'], _, _ = multipletests(stats_df['SAD_p'], alpha=0.05, method='fdr_bh')
            _, stats_df['HC_p_fdr'], _, _ = multipletests(stats_df['HC_p'], alpha=0.05, method='fdr_bh')
            _, stats_df['Diff_p_fdr'], _, _ = multipletests(stats_df['Diff_p'], alpha=0.05, method='fdr_bh')
            
        print(f"\n--- Statistics: {label} ---")
        sig_diff = stats_df[stats_df['Diff_p_fdr'] < 0.05]
        if not sig_diff.empty:
            print("Significant Group Differences (FDR < 0.05):")
            print(sig_diff[['Trial', 'Diff_t', 'Diff_df', 'Diff_p', 'Diff_p_fdr']].to_string(index=False))
        else:
            print("No significant group differences found (FDR corrected).")
            
        return stats_df
    
    print("\n[Step 3] Calculating Statistics...")
    stats_safe = run_detailed_stats(df_safe_sad, df_safe_hc, "Safety Learning")
    stats_threat = run_detailed_stats(df_threat_sad, df_threat_hc, "Threat Maintenance")
    
    def prepare_plot(df_sad, df_hc, name):
        if df_sad.empty and df_hc.empty: return pd.DataFrame()
        d_list = []
        if not df_sad.empty:
            d1 = df_sad.copy()
            d1['Group'] = 'SAD'
            d_list.append(d1)
        if not df_hc.empty:
            d2 = df_hc.copy()
            d2['Group'] = 'HC'
            d_list.append(d2)
        
        if not d_list: return pd.DataFrame()
        
        df = pd.concat(d_list)
        df['Condition'] = name
        if BLOCK_SIZE > 1:
            df['trial'] = ((df['trial'] - 1) // BLOCK_SIZE) + 1
        return df
    
    df_safe = prepare_plot(df_safe_sad, df_safe_hc, "Safety Learning")
    df_threat = prepare_plot(df_threat_sad, df_threat_hc, "Threat Maintenance")
    
    if df_safe.empty and df_threat.empty:
        print("! No data to plot.")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(22, 9), sharey=True)
        
        if not df_safe.empty:
            sns.lineplot(data=df_safe, x='trial', y='score', hue='Group', 
                         palette={'SAD': '#c44e52', 'HC': '#4c72b0'}, 
                         lw=3, marker="o", err_style="band", ax=axes[0])
            axes[0].set_title("A. Safety Trajectory\n(Target = CS-)")
            axes[0].set_ylabel("Similarity Score (0=Start, 1=Target)")
            axes[0].axhline(0, color='gray', ls='--', label='Start (Fear)')
            axes[0].axhline(1, color='#2ca02c', ls='-', lw=2, label='Target (CS-)')
            axes[0].legend(loc='upper left')

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
        fig_path = os.path.join(output_dir, "analysis_1_3_trajectories.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved figure to {fig_path}")
    
    results_13 = {
        'stats_safe': stats_safe, 
        'stats_threat': stats_threat,
        'data_safe': df_safe,
        'data_threat': df_threat
    }
    _save_result("results_13b", results_13)
    results_path = os.path.join(output_dir, "analysis_1_3_trajectories_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results_13, f)
    print(f"Saved results to {results_path}")
    
    return results_13

def analysis_1_4_decision_boundary(data_subsets, importance_scores, subject_best_params, output_dir):
    """Analysis 1.4: Decision Boundary Characteristics"""
    print("--- Running Analysis 1.4: Self-Network Statistics (Entropy, Kurtosis, Variance) ---")
    
    COND_CLASS_THREAT = "CSR"
    COND_CLASS_SAFE = "CSS"
    
    def get_significant_mask(scores): 
        return scores > 0
    
    mask_sad_native = get_significant_mask(importance_scores['SAD'])
    mask_hc_native = get_significant_mask(importance_scores['HC'])
    
    if subject_best_params is None:
        subject_best_params = {}
    
    def get_ext_data(group_key):
        if group_key not in data_subsets or data_subsets[group_key]['ext'] is None:
            raise ValueError(f"Extinction data for {group_key} missing. Check Cell 5.")
        d = data_subsets[group_key]['ext']
        return d["X"], d["y"], d["sub"]
    
    X_sad, y_sad, sub_sad = get_ext_data("SAD_Placebo")
    X_hc, y_hc, sub_hc = get_ext_data("HC_Placebo")
    
    print("  > Analyzing SAD Placebo...")
    df_sad_stats = calculate_distribution_stats(
        X_sad, y_sad, sub_sad, 
        mask_sad_native, subject_best_params, COND_CLASS_THREAT, COND_CLASS_SAFE
    )
    
    print("  > Analyzing HC Placebo...")
    df_hc_stats = calculate_distribution_stats(
        X_hc, y_hc, sub_hc, 
        mask_hc_native, subject_best_params, COND_CLASS_THREAT, COND_CLASS_SAFE
    )
    
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
    
    print("\n--- RESULTS: Self-Network Decision Statistics ---")
    p_ent = compare_metric(df_sad_stats['entropy'], df_hc_stats['entropy'], "Entropy (Uncertainty)")
    p_kurt = compare_metric(df_sad_stats['kurtosis'], df_hc_stats['kurtosis'], "Kurtosis (Sharpness)")
    p_var = compare_metric(df_sad_stats['variance'], df_hc_stats['variance'], "Variance (Spread)")
    
    fig = plt.figure(figsize=(24, 8))
    gs = fig.add_gridspec(1, 3)
    
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
    fig_path = os.path.join(output_dir, "analysis_1_4_decision_boundary.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {fig_path}")
    
    results_14_self = {'df_sad': df_sad_stats, 'df_hc': df_hc_stats}
    _save_result("results_14_self", results_14_self)
    results_path = os.path.join(output_dir, "analysis_1_4_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results_14_self, f)
    print(f"Saved results to {results_path}")
    
    return results_14_self

def analysis_2_1_safety_restoration(data_subsets, X_ext, y_ext, sub_ext, mask_sad_top5, mask_hc_top5, sub_to_meta, output_dir):
    """Analysis 2.1: Safety Restoration & Threat Discrimination - Cell 13"""
    print("--- Running Analysis 2.1: Safety Restoration & Threat Discrimination (LME) ---")
    
    COND_SAFE_LEARN = "CSS"
    COND_SAFE_BASE  = "CS-"
    COND_THREAT     = "CSR"
    
    subgroups_21 = {"SAD_Placebo": [], "SAD_Oxytocin": [], "HC_Placebo": [], "HC_Oxytocin": []}
    
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
        current_mask = mask_sad_top5 if group == "SAD" else mask_hc_top5
            
        for sub in subject_list:
            mask_sub = (sub_ext == sub)
            X_sub = X_ext[mask_sub]
            y_sub = y_ext[mask_sub]
            
            X_masked = X_sub[:, current_mask]
            X_masked = X_masked - np.mean(X_masked, axis=0)
            
            idx_css = (y_sub == COND_SAFE_LEARN)
            idx_cs_ = (y_sub == COND_SAFE_BASE)
            idx_csr = (y_sub == COND_THREAT)
            
            if np.sum(idx_css) < 3 or np.sum(idx_cs_) < 3 or np.sum(idx_csr) < 3: continue
            
            p_css = np.mean(X_masked[idx_css], axis=0).reshape(1, -1)
            p_cs_ = np.mean(X_masked[idx_cs_], axis=0).reshape(1, -1)
            p_csr = np.mean(X_masked[idx_csr], axis=0).reshape(1, -1)
            
            dist_safety = cdist(p_css, p_cs_, metric='correlation')[0][0]
            dist_threat = cdist(p_csr, p_css, metric='correlation')[0][0]
                
            data_rows.append({
                "Subject": sub, "Group": group, "Drug": drug, "Condition": key,
                "Dist_Safety": dist_safety,
                "Dist_Threat": dist_threat
            })
    
    df_topo = pd.DataFrame(data_rows)
    print(f"  > Computed metrics for {len(df_topo)} subjects.")
    
    print("\n[Step 2] Testing for Interaction (Mixed Effects)...")
    
    def run_lme(formula, data, title):
        print(f"\n--- {title} ---")
        md = smf.mixedlm(formula, data, groups=data["Subject"]) 
        try:
            mdf = md.fit()
            print(mdf.summary())
            
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
    
    form_base = "~ C(Group, Treatment(reference='HC')) * C(Drug, Treatment(reference='Placebo'))"
    
    p_int_safe = run_lme("Dist_Safety " + form_base, df_topo, "Metric 1: Safety Restoration (CSS - CS-)")
    p_int_threat = run_lme("Dist_Threat " + form_base, df_topo, "Metric 2: Threat Discrimination (CSR - CSS)")
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    pal_group = {'SAD': '#c44e52', 'HC': '#4c72b0'}
    
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
    fig_path = os.path.join(output_dir, "analysis_2_1_safety_restoration.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {fig_path}")
    
    results_21 = {'df': df_topo, 'p_safe': p_int_safe, 'p_threat': p_int_threat}
    _save_result("results_21", results_21)
    results_path = os.path.join(output_dir, "analysis_2_1_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results_21, f)
    print(f"Saved results to {results_path}")
    
    return results_21

def analysis_2_2_drift_efficiency(data_subsets, X_ext, y_ext, sub_ext, X_reinst, y_reinst, sub_reinst, importance_scores, sub_to_meta, output_dir):
    """Analysis 2.2: Drift Efficiency - Cell 14"""
    print("--- Running Analysis 2.2: Drift Efficiency (Means ± SEM) ---")
    
    COND_SAFE_TGT = "CS-"
    COND_SAFE_LRN = "CSS"
    COND_THREAT_LRN = "CSR"
    PERCENTILE_THRESH = 95
    
    print(f"\n[Step 0] Setup & Data Loading...")
    
    mask_sad_core = get_top_percentile_mask(importance_scores['SAD'], PERCENTILE_THRESH)[0]
    mask_hc_core = get_top_percentile_mask(importance_scores['HC'], PERCENTILE_THRESH)[0]
    print(f"  > Core Masks: SAD={np.sum(mask_sad_core)}, HC={np.sum(mask_hc_core)}")
    
    # Get reinstatement data from data_subsets or use passed parameters
    X_rst_all, y_rst_all, sub_rst_all = None, None, None
    if X_reinst is not None:
        X_rst_all, y_rst_all, sub_rst_all = X_reinst, y_reinst, sub_reinst
    else:
        try:
            xs, ys, ss = [], [], []
            for grp in ["SAD_Placebo", "SAD_Oxytocin", "HC_Placebo", "HC_Oxytocin"]:
                if grp in data_subsets and data_subsets[grp]['rst'] is not None:
                    d = data_subsets[grp]['rst']
                    xs.append(d['X'])
                    ys.append(d['y'])
                    ss.append(d['sub'])
            if xs:
                X_rst_all = np.vstack(xs)
                y_rst_all = np.concatenate(ys)
                sub_rst_all = np.concatenate(ss)
        except:
            print("  ! Threat analysis skipped (Reinstatement data missing).")
    
    subgroups_22 = {"SAD_Placebo": [], "SAD_Oxytocin": [], "HC_Placebo": [], "HC_Oxytocin": []}
    
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
            
            res_safe = calc_drift_metrics(X_e, y_e, X_e, y_e, COND_SAFE_LRN, COND_SAFE_TGT, curr_mask, sub)
            if res_safe:
                data_rows.append({"Subject": sub, "Group": group, "Drug": drug, "Domain": "Safety", **res_safe})
                
            if X_rst_all is not None:
                m_rst = (sub_rst_all == sub)
                if np.sum(m_rst) > 0:
                    X_r, y_r = X_rst_all[m_rst], y_rst_all[m_rst]
                    res_threat = calc_drift_metrics(X_e, y_e, X_r, y_r, COND_THREAT_LRN, COND_THREAT_LRN, curr_mask, sub)
                    if res_threat:
                        data_rows.append({"Subject": sub, "Group": group, "Drug": drug, "Domain": "Threat", **res_threat})
    
    df_drift = pd.DataFrame(data_rows)
    print(f"  > Computed vectors for {len(df_drift['Subject'].unique())} subjects.")
    
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
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    pal_group = {'SAD': '#c44e52', 'HC': '#4c72b0'}
    
    def plot_interaction(ax, df, domain, metric, p_val):
        data = df[df["Domain"] == domain]
        if data.empty: return
        
        sns.pointplot(data=data, x='Drug', y=metric, hue='Group', 
                      palette=pal_group, order=['Placebo', 'Oxytocin'], hue_order=['SAD', 'HC'],
                      dodge=0.15, markers=['o', 's'], linestyles=['-', '--'], 
                      capsize=0.1, err_kws={'linewidth': 2.5}, 
                      errorbar='se', ax=ax)
        
        ax.set_title(f"{domain} - {metric}")
        ax.axhline(0, color='gray', ls='--', alpha=0.5)
        ax.legend(loc='upper right', fontsize=12)
        
        if p_val < 0.05:
            ax.text(0.5, 0.9, f"Interaction p={p_val:.3f}", transform=ax.transAxes, 
                    ha='center', fontweight='bold', color='black')
    
    plot_interaction(axes[0,0], df_drift, "Safety", "Cosine", lme_results.get("Safety_Cosine", 1.0))
    plot_interaction(axes[0,1], df_drift, "Safety", "Projection", lme_results.get("Safety_Projection", 1.0))
    plot_interaction(axes[1,0], df_drift, "Threat", "Cosine", lme_results.get("Threat_Cosine", 1.0))
    plot_interaction(axes[1,1], df_drift, "Threat", "Projection", lme_results.get("Threat_Projection", 1.0))
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "analysis_2_2_drift_efficiency.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {fig_path}")
    
    results_22 = {'df': df_drift, 'stats': lme_results}
    _save_result("results_22", results_22)
    results_path = os.path.join(output_dir, "analysis_2_2_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results_22, f)
    print(f"Saved results to {results_path}")
    
    return results_22

def analysis_2_3_probabilistic_opening(data_subsets, X_ext, y_ext, sub_ext, importance_scores, sub_to_meta, subject_best_params, output_dir):
    """Analysis 2.3: Probabilistic Opening - Cell 15"""
    print("--- Running Analysis 2.3: Probabilistic Opening (Entropy, Kurtosis, Variance) ---")
    
    COND_CLASS_THREAT = "CSR"
    COND_CLASS_SAFE = "CSS"
    
    def get_significant_mask(scores): return scores > 0
    
    mask_sad_native = get_significant_mask(importance_scores['SAD'])
    mask_hc_native = get_significant_mask(importance_scores['HC'])
    print(f"  > SAD Native Network: {np.sum(mask_sad_native)} parcels")
    print(f"  > HC Native Network:  {np.sum(mask_hc_native)} parcels")
    
    subgroups_23 = {"SAD_Placebo": [], "SAD_Oxytocin": [], "HC_Placebo": [], "HC_Oxytocin": []}
    for sub in np.unique(sub_ext):
        s_str = str(sub).strip()
        if s_str in sub_to_meta: info = sub_to_meta[s_str]
        elif f"sub-{s_str}" in sub_to_meta: info = sub_to_meta[f"sub-{s_str}"]
        else: continue
        
        key = f"{info['Group']}_{info['Drug']}"
        if key in subgroups_23: subgroups_23[key].append(sub)
    
    data_rows = []
    print("\n[Step 1] Calculating Decision Metrics...")
    
    if subject_best_params is None:
        subject_best_params = {}
    
    for key, sub_list in subgroups_23.items():
        group, drug = key.split('_')
        curr_mask = mask_sad_native if group == "SAD" else mask_hc_native
        
        for sub in sub_list:
            mask_s = (sub_ext == sub)
            X_s, y_s = X_ext[mask_s], y_ext[mask_s]
            
            c_val = subject_best_params.get(sub, 1.0)
            
            res = calc_metrics_for_subject(X_s, y_s, sub, curr_mask, c_val, COND_CLASS_THREAT, COND_CLASS_SAFE)
            
            if res is not None:
                data_rows.append({
                    "Subject": sub, "Group": group, "Drug": drug, 
                    "Entropy": res['Entropy'], 
                    "Kurtosis": res['Kurtosis'], 
                    "Variance": res['Variance']
                })
    
    df_metrics = pd.DataFrame(data_rows)
    print(f"  > Computed metrics for {len(df_metrics)} subjects.")
    
    print("\n[Step 2] Statistical Testing (LME for each metric)...")
    
    stats_results = {}
    metrics_list = ["Entropy", "Kurtosis", "Variance"]
    
    for met in metrics_list:
        print(f"\n--- Metric: {met} ---")
        try:
            md = smf.mixedlm(f"{met} ~ C(Group, Treatment(reference='HC')) * C(Drug, Treatment(reference='Placebo'))", 
                             df_metrics, groups=df_metrics["Subject"])
            mdf = md.fit()
            print(mdf.summary())
            
            term_int = "C(Group, Treatment(reference='HC'))[T.SAD]:C(Drug, Treatment(reference='Placebo'))[T.Oxytocin]"
            p_val = mdf.pvalues.get(term_int, 1.0)
            stats_results[met] = p_val
            print(f"  >>> Interaction p={p_val:.4f} {'*' if p_val < 0.05 else ''}")
            
        except Exception as e:
            print(f"  ! Model Failed: {e}")
            stats_results[met] = 1.0
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    pal_group = {'SAD': '#c44e52', 'HC': '#4c72b0'}
    
    def plot_metric(ax, metric, p_val):
        sns.pointplot(data=df_metrics, x='Drug', y=metric, hue='Group', 
                      palette=pal_group, order=['Placebo', 'Oxytocin'], hue_order=['SAD', 'HC'],
                      dodge=0.2, markers=['o', 's'], linestyles=['-', '--'], 
                      capsize=0.1, errorbar='se', ax=ax)
        
        ax.set_title(f"{metric}")
        ax.set_ylabel(metric)
        if metric == "Entropy": ax.set_ylabel("Entropy (Uncertainty)")
        if metric == "Kurtosis": ax.set_ylabel("Kurtosis (Sharpness)")
        
        if p_val < 0.05:
            ax.text(0.5, 0.9, f"Interaction\np={p_val:.3f}", transform=ax.transAxes, 
                    ha='center', fontweight='bold', color='black')
    
    plot_metric(axes[0], "Entropy", stats_results["Entropy"])
    plot_metric(axes[1], "Kurtosis", stats_results["Kurtosis"])
    plot_metric(axes[2], "Variance", stats_results["Variance"])
    
    axes[1].get_legend().remove()
    axes[2].get_legend().remove()
    axes[0].legend(loc='lower left', fontsize=12)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "analysis_2_3_probabilistic_opening.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {fig_path}")
    
    results_23 = {'df': df_metrics, 'stats': stats_results}
    _save_result("results_23", results_23)
    results_path = os.path.join(output_dir, "analysis_2_3_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results_23, f)
    print(f"Saved results to {results_path}")
    
    return results_23

def analysis_2_4_spatial_realignment(data_subsets, res_hc_dict, output_dir):
    """Analysis 2.4: Spatial Re-Alignment - Cell 16"""
    print("--- Running Analysis 2.4: Spatial Re-Alignment (Using Analysis 1.1 Output) ---")
    
    COND_SAFE = "CSS"
    COND_THREAT = "CSR"
    LABELS = [COND_SAFE, COND_THREAT]
    
    target_contrast = "CSS vs CSR"
    alt_contrast = "CSR vs CSS"
    
    if target_contrast in res_hc_dict:
        gold_model = res_hc_dict[target_contrast]['model']
        print(f"  > Retrieved Analysis 1.1 Model for: {target_contrast}")
    elif alt_contrast in res_hc_dict:
        gold_model = res_hc_dict[alt_contrast]['model']
        print(f"  > Retrieved Analysis 1.1 Model for: {alt_contrast}")
    else:
        raise ValueError(f"Analysis 1.1 results found, but '{target_contrast}' is missing.")
    
    print(f"  > Model Classes: {gold_model.classes_}")
    if COND_THREAT not in gold_model.classes_ or COND_SAFE not in gold_model.classes_:
        raise ValueError(f"CRITICAL: The retrieved model was trained on {gold_model.classes_}, but this analysis requires {LABELS}.")
    
    def get_ext_data(group_key):
        if group_key not in data_subsets: raise ValueError(f"{group_key} missing.")
        d = data_subsets[group_key]['ext']
        return d["X"], d["y"], d["sub"]
    
    X_sad_plc, y_sad_plc, sub_sad_plc = get_ext_data("SAD_Placebo")
    X_sad_oxt, y_sad_oxt, sub_sad_oxt = get_ext_data("SAD_Oxytocin")
    
    print("\n[Step 1] Cross-Decoding on SAD Subgroups (Forced Choice)...")
    
    def calc_forced_choice_acc(model, X, y, subs):
        mask_c = np.isin(y, LABELS)
        X_f = X[mask_c] 
        y_f = y[mask_c]
        s_f = subs[mask_c]
        
        if len(y_f) == 0: return []

        X_f = subject_wise_centering(X_f, s_f)
        
        try:
            scores = model.decision_function(X_f)
        except ValueError as e:
            print(f"    ! Prediction Error (Shape Mismatch?): {e}")
            return []
        
        df_scores = pd.DataFrame({'sub': s_f, 'cond': y_f, 'score': scores})
        means = df_scores.groupby(['sub', 'cond'])['score'].mean().unstack()
        
        valid_subs = means.dropna().index
        means = means.loc[valid_subs]
        
        if COND_THREAT not in means.columns or COND_SAFE not in means.columns:
            return []

        pos_idx = np.where(gold_model.classes_ == COND_THREAT)[0][0]
        
        accs = []
        for sub in means.index:
            s_threat = means.loc[sub, COND_THREAT]
            s_safe = means.loc[sub, COND_SAFE]
            
            if pos_idx == 1:
                correct = s_threat > s_safe
            else:
                correct = s_threat < s_safe
                
            accs.append(1.0 if correct else 0.0)
            
        return accs
    
    acc_sad_plc = calc_forced_choice_acc(gold_model, X_sad_plc, y_sad_plc, sub_sad_plc)
    acc_sad_oxt = calc_forced_choice_acc(gold_model, X_sad_oxt, y_sad_oxt, sub_sad_oxt)
    
    m_plc = np.mean(acc_sad_plc) if len(acc_sad_plc) > 0 else 0
    m_oxt = np.mean(acc_sad_oxt) if len(acc_sad_oxt) > 0 else 0
    
    print(f"  > SAD-Placebo Acc (decoded by HC Model):  {m_plc:.1%} (n={len(acc_sad_plc)})")
    print(f"  > SAD-Oxytocin Acc (decoded by HC Model): {m_oxt:.1%} (n={len(acc_sad_oxt)})")
    
    print("\n[Step 2] Statistical Test...")
    if len(acc_sad_oxt) > 1 and len(acc_sad_plc) > 1:
        t_stat, p_val = ttest_ind(acc_sad_oxt, acc_sad_plc, alternative='greater')
        sig_label = "*" if p_val < 0.05 else "ns"
        print(f"  > Hypothesis (OXT > PLC): t={t_stat:.3f}, p={p_val:.4f} ({sig_label})")
    else:
        print("  ! Insufficient data for statistics.")
        p_val = 1.0
        sig_label = "nA"
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    matrix_data = np.array([[m_plc, m_oxt]])
    annot_data = np.array([
        [f"{m_plc:.3f}", f"{m_oxt:.3f}\n({sig_label})"]
    ])
    
    sns.heatmap(matrix_data, annot=annot_data, fmt="", cmap="RdBu_r", 
                vmin=0.3, vmax=0.7, center=0.5, cbar=True,
                xticklabels=['Test: SAD-Placebo', 'Test: SAD-Oxytocin'], 
                yticklabels=['Train: HC-Placebo (Anal 1.1)'], ax=ax)
    
    ax.set_title(f"Analysis 2.4: Spatial Re-Alignment\n(OXT vs PLC Improvement: p={p_val:.3f})")
    plt.yticks(rotation=0) 
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "analysis_2_4_spatial_realignment.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {fig_path}")
    
    results_24 = {'acc_plc': acc_sad_plc, 'acc_oxt': acc_sad_oxt, 'p_val': p_val, 'model': gold_model}
    _save_result("results_24", results_24)
    results_path = os.path.join(output_dir, "analysis_2_4_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results_24, f)
    print(f"Saved results to {results_path}")
    
    return results_24

def analysis_2_5_reverse_crossdecoding(data_subsets, importance_scores, output_dir):
    """Analysis 2.5: Reverse Cross-Decoding - Cell 17"""
    print("--- Running Analysis 2.5: Reverse Cross-Decoding (SAD -> HC) ---")
    
    COND_SAFE = "CSS"
    COND_THREAT = "CSR"
    LABELS = [COND_SAFE, COND_THREAT]
    
    scores_sad = importance_scores['SAD']
    thresh_sad = np.percentile(scores_sad, 95)
    mask_sad_native = (scores_sad >= thresh_sad) & (scores_sad > 0)
    print(f"  > Feature Space: SAD Top 5% ({np.sum(mask_sad_native)} parcels)")
    
    def get_ext_data(group_key):
        if group_key not in data_subsets: raise ValueError(f"{group_key} missing.")
        d = data_subsets[group_key]['ext']
        return d["X"], d["y"], d["sub"]
    
    X_sad_plc, y_sad_plc, sub_sad_plc = get_ext_data("SAD_Placebo")
    X_hc_plc, y_hc_plc, sub_hc_plc = get_ext_data("HC_Placebo")
    X_hc_oxt, y_hc_oxt, sub_hc_oxt = get_ext_data("HC_Oxytocin")
    
    print("\n[Step 1] Training SAD-Placebo Model...")
    
    mask_train = np.isin(y_sad_plc, LABELS)
    X_train = X_sad_plc[mask_train][:, mask_sad_native]
    y_train = y_sad_plc[mask_train]
    s_train = sub_sad_plc[mask_train]
    
    X_train = subject_wise_centering(X_train, s_train)
    
    sad_model = build_binary_pipeline()
    sad_model.fit(X_train, y_train)
    
    print(f"  > Model Trained on {len(np.unique(s_train))} SAD subjects.")
    print(f"  > Classes: {sad_model.classes_}")
    
    print("\n[Step 2] Testing on HC Subgroups...")
    
    # The model was trained on masked data, so we need to apply the same mask to test data
    # But calc_forced_choice_acc expects full X and applies mask internally
    # So we need to create a full-size mask that matches the original feature space
    # Since we trained on masked data, we need to pass the full X and a mask that selects the same features
    
    # Create a full-size boolean mask (all False, then set True for selected features)
    full_mask = np.zeros(X_hc_plc.shape[1], dtype=bool)
    # Find which indices in the original space correspond to the top 5%
    # mask_sad_native is already a boolean array of length matching importance_scores
    # We need to map it to the full feature space
    full_mask[mask_sad_native] = True
    
    acc_hc_plc = calc_forced_choice_acc(sad_model, X_hc_plc, y_hc_plc, sub_hc_plc, full_mask, COND_THREAT, COND_SAFE, LABELS)
    acc_hc_oxt = calc_forced_choice_acc(sad_model, X_hc_oxt, y_hc_oxt, sub_hc_oxt, full_mask, COND_THREAT, COND_SAFE, LABELS)
    
    m_hc_plc = np.mean(acc_hc_plc) if len(acc_hc_plc) > 0 else 0
    m_hc_oxt = np.mean(acc_hc_oxt) if len(acc_hc_oxt) > 0 else 0
    
    print(f"  > HC-Placebo Acc (decoded by SAD):  {m_hc_plc:.1%} (n={len(acc_hc_plc)})")
    print(f"  > HC-Oxytocin Acc (decoded by SAD): {m_hc_oxt:.1%} (n={len(acc_hc_oxt)})")
    
    print("\n[Step 3] Statistical Test (Vs Chance 50%)...")
    
    t_chance, p_chance = ttest_1samp(acc_hc_plc, 0.5)
    sig_chance = "*" if p_chance < 0.05 else "ns"
    
    print(f"  > SAD->HC Generalization (vs 50%): t={t_chance:.3f}, p={p_chance:.4f} ({sig_chance})")
    print("    (Note: 'ns' is GOOD here -> implies disordered code is specific to SAD)")
    
    t_drug, p_drug = ttest_ind(acc_hc_oxt, acc_hc_plc)
    print(f"  > Drug Effect in HC (OXT vs PLC): p={p_drug:.4f}")
    
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
    fig_path = os.path.join(output_dir, "analysis_2_5_reverse_crossdecoding.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved figure to {fig_path}")
    
    results_25 = {'acc_hc_plc': acc_hc_plc, 'acc_hc_oxt': acc_hc_oxt, 'model': sad_model}
    _save_result("results_25", results_25)
    results_path = os.path.join(output_dir, "analysis_2_5_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results_25, f)
    print(f"Saved results to {results_path}")
    
    return results_25

def main():
    parser = argparse.ArgumentParser(description='MVPA L2 ROI Whole Brain Analysis - S4 Glasser+Tian')
    parser.add_argument('--project_root', type=str, 
                       default='/Users/xiaoqianxiao/projects/NARSAD',
                       help='Root directory of the project')
    parser.add_argument('--output_dir', type=str,
                      default='/Users/xiaoqianxiao/projects/NARSAD/MRI/derivatives/fMRI_analysis/LSS/outputs/L2_ROI',
                       help='Output directory for saving results and figures')
    parser.add_argument('--skip_analyses', nargs='+', 
                       choices=['1.1', '1.2', '1.3', '1.4', '2.1', '2.2', '2.3', '2.4', '2.5'],
                       help='Analyses to skip (space-separated list)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Load data
    X_ext, y_ext, sub_ext, X_reinst, y_reinst, sub_reinst, parcel_names = load_data(args.project_root, args.output_dir)
    
    # Load metadata
    meta = load_metadata(args.project_root)
    
    # Prepare data subsets
    data_subsets, sub_to_meta = prepare_data_subsets(
        X_ext, y_ext, sub_ext, X_reinst, y_reinst, sub_reinst, meta, args.output_dir
    )
    
    # Run analyses
    skip_list = args.skip_analyses if args.skip_analyses else []
    
    # Initialize variables
    res_sad_dict = res_hc_dict = None
    importance_scores = None
    importance_masks = None
    mask_sad_top5 = mask_hc_top5 = None
    subject_best_params = {}
    
    # Analysis 1.1: Neural Dissociation
    if '1.1' not in skip_list:
        print("\n" + "="*80)
        print("ANALYSIS 1.1 - Neural Dissociation")
        print("="*80)
        res_sad_dict, res_hc_dict, results_11 = analysis_1_1_neural_dissociation(data_subsets, args.output_dir)
    
    # Analysis 1.2: Feature Importance (Cell 8) - Must run before spatial topology
    if '1.2' not in skip_list and res_sad_dict is not None:
        print("\n" + "="*80)
        print("CELL 8: ANALYSIS 1.2 - Feature Importance (Permutation)")
        print("="*80)
        importance_masks, importance_scores = analysis_1_3_feature_importance(
            data_subsets, res_sad_dict, res_hc_dict, args.output_dir
        )
        
        # Analysis 1.2: Spatial Topology (Cell 7/8) - Uses res from 1.1
        print("\n" + "="*80)
        print("CELL 7/8: ANALYSIS 1.2 - Spatial Topology (ROI-wise)")
        print("="*80)
        spatial_results = analysis_1_2_spatial_topology(
            data_subsets, res_sad_dict, res_hc_dict, parcel_names, args.output_dir
        )
        
        # Analysis 1.2: Static Representational Topology (Cell 9)
        print("\n" + "="*80)
        print("CELL 9: ANALYSIS 1.2 - Static Representational Topology (Top 5%)")
        print("="*80)
        results_12, mask_sad_top5, mask_hc_top5 = analysis_1_2_static_topology(
            data_subsets, X_ext, y_ext, sub_ext, importance_scores, args.output_dir
        )
    
    # Analysis 1.3: Dynamic Representational Drift (Cell 10)
    if '1.3' not in skip_list and importance_scores is not None:
        print("\n" + "="*80)
        print("CELL 10: ANALYSIS 1.3 - Dynamic Representational Drift (Top 5%)")
        print("="*80)
        results_13_drift = analysis_1_3_dynamic_drift(
            data_subsets, X_ext, y_ext, sub_ext, X_reinst, y_reinst, sub_reinst, 
            importance_scores, args.output_dir, X_ext, y_ext, sub_ext
        )
        
        # Analysis 1.3: Single-Trial Trajectories (Cell 11)
        print("\n" + "="*80)
        print("CELL 11: ANALYSIS 1.3 - Single-Trial Trajectories")
        print("="*80)
        results_13_traj = analysis_1_3_trajectories(
            data_subsets, X_ext, y_ext, sub_ext, X_reinst, y_reinst, sub_reinst,
            importance_scores, args.output_dir, X_ext, y_ext, sub_ext
        )
    
    # Analysis 1.4: Decision Boundary Characteristics (Cell 12)
    if '1.4' not in skip_list and importance_scores is not None:
        print("\n" + "="*80)
        print("CELL 12: ANALYSIS 1.4 - Decision Boundary Characteristics")
        print("="*80)
        results_14 = analysis_1_4_decision_boundary(
            data_subsets, importance_scores, subject_best_params, args.output_dir
        )
    
    # Analysis 2.1: Safety Restoration & Threat Discrimination (Cell 13)
    if '2.1' not in skip_list and mask_sad_top5 is not None:
        print("\n" + "="*80)
        print("CELL 13: ANALYSIS 2.1 - Safety Restoration & Threat Discrimination")
        print("="*80)
        results_21 = analysis_2_1_safety_restoration(
            data_subsets, X_ext, y_ext, sub_ext, mask_sad_top5, mask_hc_top5, 
            sub_to_meta, args.output_dir
        )
    
    # Analysis 2.2: Drift Efficiency (Cell 14)
    if '2.2' not in skip_list and importance_scores is not None:
        print("\n" + "="*80)
        print("CELL 14: ANALYSIS 2.2 - Drift Efficiency")
        print("="*80)
        results_22 = analysis_2_2_drift_efficiency(
            data_subsets, X_ext, y_ext, sub_ext, X_reinst, y_reinst, sub_reinst,
            importance_scores, sub_to_meta, args.output_dir
        )
    
    # Analysis 2.3: Probabilistic Opening (Cell 15)
    if '2.3' not in skip_list and importance_scores is not None:
        print("\n" + "="*80)
        print("CELL 15: ANALYSIS 2.3 - Probabilistic Opening")
        print("="*80)
        results_23 = analysis_2_3_probabilistic_opening(
            data_subsets, X_ext, y_ext, sub_ext, importance_scores, 
            sub_to_meta, subject_best_params, args.output_dir
        )
    
    # Analysis 2.4: Spatial Re-Alignment (Cell 16)
    if '2.4' not in skip_list and res_hc_dict is not None:
        print("\n" + "="*80)
        print("CELL 16: ANALYSIS 2.4 - Spatial Re-Alignment")
        print("="*80)
        results_24 = analysis_2_4_spatial_realignment(
            data_subsets, res_hc_dict, args.output_dir
        )
    
    # Analysis 2.5: Reverse Cross-Decoding (Cell 17)
    if '2.5' not in skip_list and importance_scores is not None:
        print("\n" + "="*80)
        print("CELL 17: ANALYSIS 2.5 - Reverse Cross-Decoding")
        print("="*80)
        results_25 = analysis_2_5_reverse_crossdecoding(
            data_subsets, importance_scores, args.output_dir
        )
    
    # Save all intermediate results for downstream analyses
    if importance_scores is not None:
        importance_path = os.path.join(args.output_dir, "importance_scores.pkl")
        with open(importance_path, 'wb') as f:
            pickle.dump(importance_scores, f)
        print(f"\nSaved importance scores to {importance_path}")
    
    if res_sad_dict is not None and res_hc_dict is not None:
        models_path = os.path.join(args.output_dir, "models.pkl")
        with open(models_path, 'wb') as f:
            pickle.dump({'res_sad_dict': res_sad_dict, 'res_hc_dict': res_hc_dict}, f)
        print(f"Saved models to {models_path}")
    
    if mask_sad_top5 is not None:
        masks_path = os.path.join(args.output_dir, "top5_masks.pkl")
        with open(masks_path, 'wb') as f:
            pickle.dump({'mask_sad_top5': mask_sad_top5, 'mask_hc_top5': mask_hc_top5}, f)
        print(f"Saved top 5% masks to {masks_path}")
    
    print("\n" + "="*80)
    print("=== Analysis Complete ===")
    print("="*80)
    print(f"All outputs saved to: {args.output_dir}")
    print(f"\nOutput files:")
    for file in sorted(os.listdir(args.output_dir)):
        if file.endswith(('.png', '.pkl')):
            print(f"  - {file}")

if __name__ == "__main__":
    main()
