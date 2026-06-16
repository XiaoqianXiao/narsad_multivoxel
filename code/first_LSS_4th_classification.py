#!/usr/bin/env python3
"""
Subject-level ROI MVPA using single-trial LSS estimates

Configured to:
  - Use a small, hypothesis-driven ROI list (Tier 1)
  - Use Glasser (HCP-MMP) for cortical ROIs
  - Use Tian atlas for subcortical ROIs
  - Aggregate parcels into named ROIs (e.g., vmPFC, Amygdala, AG_IPL, IPS, FFA, pSTS)

Core ROIs (Tier 1):
  Fear / safety circuit
    - Amygdala (BLA + CMA; Tian)
    - Hippocampus (anterior; Tian)
    - vmPFC (Glasser)
    - dACC (Glasser)
    - dAI (dorsal anterior insula; Glasser)

  Vicarious / semantic circuit
    - Angular gyrus / IPL (AG_IPL; Glasser)
    - IPS (Glasser)
    - dmPFC (Glasser)

  Visual social input
    - FFA (fusiform face area; Glasser)
    - pSTS (posterior STS; Glasser)
    - EarlyVisual (V1–V3; Glasser)

  Optional
    - NAcc (ventral striatum; Tian)

Aims captured:
  Aim 1 – Extinction representations (3-way decoding, early/late)
  Aim 2 – Reinstatement representations (3-way, CS+ vs CS−, early/late)
  Aim 3 – Cross-phase generalization (PSI, PSI_Early, PSI_Late, ΔPSI)

CRITICAL LABEL MAPPING
----------------------
Raw trial_type values in events:
    'CSR', 'CSS', 'CS-', 'FIXATION', 'US_CSR', 'US_CSS', ...

For MVPA we map:
    CSR -> 'CS_reinforcement'
    CSS -> 'CS_safe'
    CS- -> 'CS_minus'
All others (FIXATION, US_*) -> NaN and are dropped in decoding.

Inputs:
  - LSS 4D betas:
      /data/NARSAD/MRI/derivatives/fMRI_analysis/LSS/firstLevel/all_subjects/
        sub-{subject}_task-{task}_contrast1.nii[.gz]
  - Behavior events:
      /data/NARSAD/MRI/source_data/behav/task-Narsad_{task}_events.csv
        (and special case task-NARSAD_phase-3_sub-202_events.csv for N202 phase3)
  - Mask image: user-specified
  - Glasser atlas: /data/NARSAD/ROI/Glasser/...
  - Tian atlas:    /data/NARSAD/ROI/Tian/...

Output:
  /data/NARSAD/MRI/derivatives/fMRI_analysis/LSS/firstLevel/all_subjects/mvpa/roi_glasser_tian/
    sub-{subject}_mvpa_roi_metrics_glasser_tian.csv
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import argparse
import logging
import numpy as np
import pandas as pd

import nibabel as nib
from nilearn.image import load_img, resample_to_img
from nilearn.masking import apply_mask

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# =============================================================================
# ROI DEFINITIONS
# =============================================================================

ROI_DEFINITIONS = {
    # Fear / safety circuit
    'Amygdala': {
        'glasser_labels': [],
        # placeholder Tian labels – update with your LUT if needed
        'tian_labels': [2, 10]
    },
    'Hippocampus_a': {
        'glasser_labels': [],
        'tian_labels': [1, 9]
    },
    'vmPFC': {
        'glasser_labels': [164, 364, 165, 365, 166, 366],
        'tian_labels': []
    },
    'dACC': {
        'glasser_labels': [40, 41, 46, 240, 241, 262],
        'tian_labels': []
    },
    'dAI': {
        'glasser_labels': [112, 312],
        'tian_labels': []
    },

    # Vicarious / semantic circuit
    'AG_IPL': {
        'glasser_labels': [143, 150, 151, 343, 350, 351],
        'tian_labels': []
    },
    'IPS': {
        'glasser_labels': [17, 144, 145, 146, 217, 344, 345, 346],
        'tian_labels': []
    },
    'dmPFC': {
        'glasser_labels': [63, 69, 87, 263, 269, 287],
        'tian_labels': []
    },

    # Visual social input
    'FFA': {
        'glasser_labels': [18, 218],
        'tian_labels': []
    },
    'pSTS': {
        'glasser_labels': [129, 130, 329, 330],
        'tian_labels': []
    },
    'EarlyVisual': {
        'glasser_labels': [1, 4, 5, 6, 201, 204, 205, 206],
        'tian_labels': []
    },

    # Optional learning / valuation
    'NAcc': {
        'glasser_labels': [],
        'tian_labels': [5, 13]
    }
}

# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(verbose=False):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger

# =============================================================================
# ARG PARSING
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="ROI-based MVPA using single-trial LSS estimates with Glasser (cortex) + Tian (subcortex)"
    )
    parser.add_argument('--subject', required=True, help='Subject ID, e.g., N101')
    parser.add_argument('--phase2_task', default='phase2',
                        help='Task name for extinction (Aim 1; default: phase2)')
    parser.add_argument('--phase3_task', default='phase3',
                        help='Task name for reinstatement (Aim 2; default: phase3)')
    parser.add_argument('--mask_img_path', required=True,
                        help='Path to brain mask image in subject space (matches LSS images)')
    parser.add_argument('--n_splits', type=int, default=4,
                        help='Max number of CV folds for decoding (default: 4)')
    parser.add_argument('--output_dir', default=None,
                        help='Optional override for subject-level MVPA output directory')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose logging')
    args = parser.parse_args()
    return args

# =============================================================================
# HELPERS: classifier + CV
# =============================================================================

def build_classifier():
    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(
            penalty='l2',
            solver='lbfgs',
            max_iter=1000,
            class_weight='balanced',
            multi_class='auto'
        )
    )
    return clf

def safe_stratified_kfold(y, max_splits, logger):
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    if min_count < 2:
        logger.warning(f"Not enough trials per class to do CV (min_count={min_count}).")
        return None

    n_splits = min(max_splits, min_count)
    if n_splits < 2:
        logger.warning(f"Adjusted n_splits is < 2 (n_splits={n_splits}).")
        return None

    logger.debug(f"Using StratifiedKFold with n_splits={n_splits}")
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

def run_decoding(X, y, max_splits, logger):
    valid = ~pd.isna(y)
    X = X[valid]
    y = np.array(y)[valid]

    if X.shape[0] < 4 or X.shape[1] < 2:
        logger.debug("Not enough trials or features for decoding.")
        return np.nan

    cv = safe_stratified_kfold(y, max_splits, logger)
    if cv is None:
        return np.nan

    clf = build_classifier()
    try:
        scores = cross_val_score(clf, X, y, cv=cv)
        return float(scores.mean())
    except Exception as e:
        logger.warning(f"Decoding failed: {e}")
        return np.nan

# =============================================================================
# HELPERS: events loading + label mapping
# =============================================================================

def map_trial_labels(events, logger):
    """
    Map raw event trial_type to MVPA classes.

    Raw codes (from task):
        CSR, CSS, CS-, FIXATION, US_CSR, US_CSS, ...

    Mapped:
        CSR -> CS_reinforcement
        CSS -> CS_safe
        CS- -> CS_minus
        others -> NaN (dropped later)
    """
    events = events.copy()
    mapping = {
        'CSR': 'CS_reinforcement',
        'CSS': 'CS_safe',
        'CS-': 'CS_minus'
    }
    events['trial_type_mapped'] = events['trial_type'].map(mapping)
    logger.info(f"Raw trial_type values: {events['trial_type'].unique()}")
    logger.info(f"Mapped trial_type_mapped values: {events['trial_type_mapped'].unique()}")
    return events

def load_events(subject, task, behav_dir, logger):
    if subject == 'N202' and task == 'phase3':
        events_file = os.path.join(behav_dir, 'task-NARSAD_phase-3_sub-202_events.csv')
    else:
        events_file = os.path.join(behav_dir, f'task-Narsad_{task}_events.csv')

    logger.info(f"Events file for task-{task}: {events_file}, exists: {os.path.exists(events_file)}")

    if not os.path.exists(events_file):
        raise FileNotFoundError(f"Events file not found: {events_file}")

    from utils import read_csv_with_detection
    events = read_csv_with_detection(events_file)
    logger.info(f"Events loaded for {task}: shape={events.shape}, columns={list(events.columns)}")

    if 'trial_type' not in events.columns:
        raise ValueError(f"Events for {task} do not contain 'trial_type' column.")

    if 'usable' not in events.columns:
        logger.warning(f"No 'usable' column in {events_file}; assuming all trials are usable.")
        events['usable'] = 1

    # map labels to CS_safe / CS_reinforcement / CS_minus
    events = map_trial_labels(events, logger)

    return events

def define_stage_labels(events, logger):
    """
    Mark Early/Late for each raw trial_type, then we later
    combine with mapped labels and drop NaNs in decoding.
    """
    events = events.copy()
    events['stage'] = 'Other'

    trial_types = events['trial_type'].unique()
    logger.info(f"Defining Early/Late stages. Trial types: {trial_types}")

    for ttype in trial_types:
        mask_type = (events['trial_type'] == ttype) & (events['usable'] == 1)
        idx = np.where(mask_type)[0]

        if len(idx) == 0:
            continue

        idx_sorted = np.sort(idx)
        early_idx = idx_sorted[:4]
        late_idx = idx_sorted[-4:]

        events.loc[early_idx, 'stage'] = 'Early'
        events.loc[late_idx, 'stage'] = 'Late'

    return events

# =============================================================================
# HELPERS: images + ROIs
# =============================================================================

def load_LSS_4d(subject, task, firstlevel_dir, logger):
    """
    Load 4D LSS beta image for subject and task.
    Uses naming: sub-{subject}_task-{task}_contrast1.nii[.gz]
    """
    nii_path = os.path.join(
        firstlevel_dir,
        f"sub-{subject}_task-{task}_contrast1.nii.gz"
    )
    if not os.path.exists(nii_path):
        nii_path = os.path.join(
            firstlevel_dir,
            f"sub-{subject}_task-{task}_contrast1.nii"
        )

    logger.info(f"Loading 4D LSS image for {task}: {nii_path}, exists={os.path.exists(nii_path)}")
    if not os.path.exists(nii_path):
        raise FileNotFoundError(
            f"LSS 4D file not found for sub-{subject}, task-{task}: {nii_path}"
        )

    img = load_img(nii_path)
    logger.info(f"{task} LSS image shape: {img.shape}")
    return img

def prepare_roi_masks(mask_img_path, glasser_img_path, tian_img_path, logger):
    """
    Build ROI masks from Glasser + Tian atlases according to ROI_DEFINITIONS.
    All images should be in the same space as the LSS images.
    """
    mask_img = load_img(mask_img_path)
    mask_data = mask_img.get_fdata() > 0

    glasser_img = load_img(glasser_img_path)
    tian_img = load_img(tian_img_path)

    # If Glasser/Tian paths are directories, pick first NIfTI inside
    if os.path.isdir(glasser_img_path):
        files = [f for f in os.listdir(glasser_img_path) if f.endswith(('.nii', '.nii.gz'))]
        if not files:
            raise FileNotFoundError(f"No NIfTI found in Glasser dir: {glasser_img_path}")
        glasser_img = load_img(os.path.join(glasser_img_path, files[0]))

    if os.path.isdir(tian_img_path):
        files = [f for f in os.listdir(tian_img_path) if f.endswith(('.nii', '.nii.gz'))]
        if not files:
            raise FileNotFoundError(f"No NIfTI found in Tian dir: {tian_img_path}")
        tian_img = load_img(os.path.join(tian_img_path, files[0]))

    # Resample atlases to mask space if needed
    if (glasser_img.shape[:3] != mask_img.shape[:3]) or \
       (not np.allclose(glasser_img.affine, mask_img.affine)):
        logger.info("Resampling Glasser atlas to match mask image space.")
        glasser_img = resample_to_img(glasser_img, mask_img, interpolation='nearest')

    if (tian_img.shape[:3] != mask_img.shape[:3]) or \
       (not np.allclose(tian_img.affine, mask_img.affine)):
        logger.info("Resampling Tian atlas to match mask image space.")
        tian_img = resample_to_img(tian_img, mask_img, interpolation='nearest')

    glasser_data = glasser_img.get_fdata()
    tian_data = tian_img.get_fdata()

    roi_masks = {}
    for roi_name, defs in ROI_DEFINITIONS.items():
        g_labels = defs['glasser_labels']
        t_labels = defs['tian_labels']

        roi_mask = np.zeros(mask_data.shape, dtype=bool)

        if g_labels:
            for lab in g_labels:
                roi_mask |= (glasser_data == lab)
        if t_labels:
            for lab in t_labels:
                roi_mask |= (tian_data == lab)

        roi_mask &= mask_data

        if roi_mask.sum() == 0:
            logger.warning(f"ROI '{roi_name}' has 0 voxels in mask (check label IDs).")
        roi_masks[roi_name] = roi_mask

    return mask_img, roi_masks

def extract_roi_patterns(LSS_img, mask_img, roi_masks, logger):
    roi_patterns = {}
    n_trials = LSS_img.shape[3]
    logger.info(f"Extracting ROI patterns for {n_trials} trials.")

    for roi_name, roi_mask in roi_masks.items():
        if roi_mask.sum() == 0:
            roi_patterns[roi_name] = None
            continue

        roi_mask_img = nib.Nifti1Image(roi_mask.astype(np.int8), mask_img.affine)
        X = apply_mask(LSS_img, roi_mask_img)  # (n_trials, n_voxels)
        roi_patterns[roi_name] = X

    return roi_patterns

# =============================================================================
# PSI (Aim 3)
# =============================================================================

def compute_PSI(X_phase2, y_phase2, X_phase3, labels_phase3, logger):
    """
    PSI: classifier trained on phase2 CS_safe vs CS_reinforcement,
    applied to phase3, measure p_reinf - p_safe.
    """
    mask_train = np.isin(y_phase2, ['CS_safe', 'CS_reinforcement'])
    if mask_train.sum() < 4:
        logger.debug("Not enough CS_safe/CS_reinforcement trials in phase2 for PSI.")
        return np.nan, np.nan, np.nan, np.nan

    X_train = X_phase2[mask_train]
    y_train = np.array(y_phase2)[mask_train]

    classes = np.unique(y_train)
    if set(classes) != {'CS_safe', 'CS_reinforcement'}:
        logger.debug(f"Phase2 classes for PSI are not exactly CS_safe & CS_reinforcement: {classes}")
        return np.nan, np.nan, np.nan, np.nan

    clf = build_classifier()
    try:
        clf.fit(X_train, y_train)
    except Exception as e:
        logger.debug(f"PSI training failed: {e}")
        return np.nan, np.nan, np.nan, np.nan

    try:
        proba = clf.predict_proba(X_phase3)
        classes = clf.named_steps['logisticregression'].classes_
    except Exception as e:
        logger.debug(f"PSI predict_proba failed: {e}")
        return np.nan, np.nan, np.nan, np.nan

    try:
        idx_safe = np.where(classes == 'CS_safe')[0][0]
        idx_reinf = np.where(classes == 'CS_reinforcement')[0][0]
    except IndexError:
        logger.debug(f"Classes in PSI classifier are unexpected: {classes}")
        return np.nan, np.nan, np.nan, np.nan

    p_safe = proba[:, idx_safe]
    p_reinf = proba[:, idx_reinf]

    stage = labels_phase3['stage'].values

    PSI_all = float(np.nanmean(p_reinf) - np.nanmean(p_safe))

    mask_early = (stage == 'Early')
    mask_late = (stage == 'Late')

    if mask_early.sum() >= 2:
        PSI_early = float(np.nanmean(p_reinf[mask_early]) - np.nanmean(p_safe[mask_early]))
    else:
        PSI_early = np.nan

    if mask_late.sum() >= 2:
        PSI_late = float(np.nanmean(p_reinf[mask_late]) - np.nanmean(p_safe[mask_late]))
    else:
        PSI_late = np.nan

    if np.isnan(PSI_early) or np.isnan(PSI_late):
        Delta_PSI = np.nan
    else:
        Delta_PSI = float(PSI_late - PSI_early)

    return PSI_all, PSI_early, PSI_late, Delta_PSI

# =============================================================================
# MAIN
# =============================================================================

def main(args):
    logger = setup_logging(args.verbose)

    logger.info("=" * 70)
    logger.info("Subject-level ROI MVPA (Glasser cortex + Tian subcortex)")
    logger.info("Aims 1–3 subject-level metrics; ready for Aims 4–5 group-level models.")
    logger.info("=" * 70)
    logger.info(f"Subject:      {args.subject}")
    logger.info(f"Phase 2 task: {args.phase2_task} (Extinction; Aim 1)")
    logger.info(f"Phase 3 task: {args.phase3_task} (Reinstatement; Aim 2)")
    logger.info(f"Mask:         {args.mask_img_path}")

    # ROI atlas directory
    ROI_DIR = "/data/NARSAD/ROI"
    glasser_atlas_path = os.path.join(ROI_DIR, "Glasser", "HCP-MMP1_2mm.nii")
    tian_atlas_path    = os.path.join(
        ROI_DIR,
        "Tian",
        "3T",
        "Subcortex-Only",
        "Tian_Subcortex_S1_3T_2009cAsym.nii.gz"
    )

    logger.info(f"Glasser atlas path: {glasser_atlas_path}")
    logger.info(f"Tian atlas path:    {tian_atlas_path}")
    logger.info("=" * 70)

    # Paths
    root_dir = os.getenv('DATA_DIR', '/data')
    project = 'NARSAD'
    derivatives_dir = os.path.join(root_dir, project, 'MRI', 'derivatives')
    behav_dir = os.path.join(root_dir, project, 'MRI', 'source_data', 'behav')
    firstlevel_dir = os.path.join(derivatives_dir, 'fMRI_analysis', 'LSS',
                                  'firstLevel', 'all_subjects')

    if args.output_dir is None:
        output_dir = os.path.join(firstlevel_dir, 'mvpa', 'roi_glasser_tian')
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory (subject-level ROI metrics): {output_dir}")

    # Prepare ROI masks from Glasser + Tian
    mask_img, roi_masks = prepare_roi_masks(
        args.mask_img_path,
        glasser_atlas_path,
        tian_atlas_path,
        logger
    )

    # Load LSS images (Aim 1: phase2; Aim 2: phase3)
    img_phase2 = load_LSS_4d(args.subject, args.phase2_task, firstlevel_dir, logger)
    img_phase3 = load_LSS_4d(args.subject, args.phase3_task, firstlevel_dir, logger)

    # Load events and define stage labels
    events_phase2 = load_events(args.subject, args.phase2_task, behav_dir, logger)
    events_phase3 = load_events(args.subject, args.phase3_task, behav_dir, logger)

    # Truncate events to number of volumes
    n_trial_p2 = img_phase2.shape[3]
    n_trial_p3 = img_phase3.shape[3]
    logger.info(f"Phase2: {n_trial_p2} LSS trials, Events rows: {len(events_phase2)}")
    logger.info(f"Phase3: {n_trial_p3} LSS trials, Events rows: {len(events_phase3)}")

    events_phase2 = events_phase2.iloc[:n_trial_p2].reset_index(drop=True)
    events_phase3 = events_phase3.iloc[:n_trial_p3].reset_index(drop=True)

    events_phase2 = define_stage_labels(events_phase2, logger)
    events_phase3 = define_stage_labels(events_phase3, logger)

    # Extract ROI patterns
    roi_patterns_p2 = extract_roi_patterns(img_phase2, mask_img, roi_masks, logger)
    roi_patterns_p3 = extract_roi_patterns(img_phase3, mask_img, roi_masks, logger)

    # Labels: USE MAPPED LABELS
    y_p2 = events_phase2['trial_type_mapped'].values
    y_p3 = events_phase3['trial_type_mapped'].values

    # Usable masks
    usable_p2 = events_phase2['usable'].values.astype(bool)
    usable_p3 = events_phase3['usable'].values.astype(bool)

    # Stage labels
    stage_p2 = events_phase2['stage'].values
    stage_p3 = events_phase3['stage'].values

    logger.info("Starting ROI-wise MVPA computations (Aims 1–3)...")

    rows = []
    for roi_name in ROI_DEFINITIONS.keys():
        X_p2 = roi_patterns_p2.get(roi_name, None)
        X_p3 = roi_patterns_p3.get(roi_name, None)

        if X_p2 is None or X_p3 is None:
            logger.debug(f"Skipping ROI '{roi_name}' due to missing patterns.")
            continue

        X_p2_use = X_p2[usable_p2]
        y_p2_use = y_p2[usable_p2]
        stage_p2_use = stage_p2[usable_p2]

        X_p3_use = X_p3[usable_p3]
        y_p3_use = y_p3[usable_p3]
        stage_p3_use = stage_p3[usable_p3]

        n_trials_p2 = len(y_p2)
        n_trials_p3 = len(y_p3)
        n_usable_p2 = len(y_p2_use)
        n_usable_p3 = len(y_p3_use)

        # ------------------------------------------------------------------
        # Aim 1: Extinction 3-way decoding (CS_safe, CS_reinforcement, CS_minus)
        # ------------------------------------------------------------------
        mask_3way_p2 = np.isin(y_p2_use, ['CS_safe', 'CS_reinforcement', 'CS_minus'])
        acc_ext_3way = run_decoding(
            X_p2_use[mask_3way_p2],
            y_p2_use[mask_3way_p2],
            args.n_splits,
            logger
        )

        # ------------------------------------------------------------------
        # Aim 2: Reinstatement 3-way decoding
        # ------------------------------------------------------------------
        mask_3way_p3 = np.isin(y_p3_use, ['CS_safe', 'CS_reinforcement', 'CS_minus'])
        acc_reinst_3way = run_decoding(
            X_p3_use[mask_3way_p3],
            y_p3_use[mask_3way_p3],
            args.n_splits,
            logger
        )

        # Aim 2: Reinstatement CS+ vs CS− (CS_safe + CS_reinforcement vs CS_minus)
        mask_cp = np.isin(y_p3_use, ['CS_safe', 'CS_reinforcement', 'CS_minus'])
        y_binary = y_p3_use[mask_cp].copy()
        y_binary[np.isin(y_binary, ['CS_safe', 'CS_reinforcement'])] = 'CS_plus'
        y_binary[y_binary == 'CS_minus'] = 'CS_minus'

        acc_reinst_CSplus_vsCSminus = run_decoding(
            X_p3_use[mask_cp],
            y_binary,
            args.n_splits,
            logger
        )

        # ------------------------------------------------------------------
        # Aim 1: Early vs Late decoding – Extinction
        # ------------------------------------------------------------------
        mask_stage_p2 = np.isin(stage_p2_use, ['Early', 'Late'])
        X_p2_stage = X_p2_use[mask_stage_p2]
        y_p2_stage = y_p2_use[mask_stage_p2]
        stage_p2_stage = stage_p2_use[mask_stage_p2]

        mask_early_p2 = (stage_p2_stage == 'Early')
        acc_ext_early = run_decoding(
            X_p2_stage[mask_early_p2],
            y_p2_stage[mask_early_p2],
            args.n_splits,
            logger
        )

        mask_late_p2 = (stage_p2_stage == 'Late')
        acc_ext_late = run_decoding(
            X_p2_stage[mask_late_p2],
            y_p2_stage[mask_late_p2],
            args.n_splits,
            logger
        )

        if np.isnan(acc_ext_early) or np.isnan(acc_ext_late):
            delta_acc_ext = np.nan
        else:
            delta_acc_ext = float(acc_ext_late - acc_ext_early)

        # ------------------------------------------------------------------
        # Aim 2: Early vs Late decoding – Reinstatement
        # ------------------------------------------------------------------
        mask_stage_p3 = np.isin(stage_p3_use, ['Early', 'Late'])
        X_p3_stage = X_p3_use[mask_stage_p3]
        y_p3_stage = y_p3_use[mask_stage_p3]
        stage_p3_stage = stage_p3_use[mask_stage_p3]

        mask_early_p3 = (stage_p3_stage == 'Early')
        acc_reinst_early = run_decoding(
            X_p3_stage[mask_early_p3],
            y_p3_stage[mask_early_p3],
            args.n_splits,
            logger
        )

        mask_late_p3 = (stage_p3_stage == 'Late')
        acc_reinst_late = run_decoding(
            X_p3_stage[mask_late_p3],
            y_p3_stage[mask_late_p3],
            args.n_splits,
            logger
        )

        if np.isnan(acc_reinst_early) or np.isnan(acc_reinst_late):
            delta_acc_reinst = np.nan
        else:
            delta_acc_reinst = float(acc_reinst_late - acc_reinst_early)

        # ------------------------------------------------------------------
        # Aim 3: PSI (cross-phase generalization)
        # ------------------------------------------------------------------
        labels_p3_df = pd.DataFrame({
            'trial_type': y_p3_use,
            'stage': stage_p3_use
        })
        PSI_all, PSI_early, PSI_late, Delta_PSI = compute_PSI(
            X_p2_use, y_p2_use, X_p3_use, labels_p3_df, logger
        )

        row = {
            'subject': args.subject,
            'ROI_name': roi_name,
            'n_trials_phase2': n_trials_p2,
            'n_trials_phase3': n_trials_p3,
            'n_usable_phase2': n_usable_p2,
            'n_usable_phase3': n_usable_p3,
            # Aim 1
            'Acc_extinction_3way': acc_ext_3way,
            'Acc_ext_Early': acc_ext_early,
            'Acc_ext_Late': acc_ext_late,
            'DeltaAcc_ext': delta_acc_ext,
            # Aim 2
            'Acc_reinstatement_3way': acc_reinst_3way,
            'Acc_reinstatement_CSplus_vsCSminus': acc_reinst_CSplus_vsCSminus,
            'Acc_reinst_Early': acc_reinst_early,
            'Acc_reinst_Late': acc_reinst_late,
            'DeltaAcc_reinst': delta_acc_reinst,
            # Aim 3
            'PSI': PSI_all,
            'PSI_Early': PSI_early,
            'PSI_Late': PSI_late,
            'Delta_PSI': Delta_PSI
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = os.path.join(
        output_dir,
        f"sub-{args.subject}_mvpa_roi_metrics_glasser_tian.csv"
    )
    df.to_csv(out_path, index=False)
    logger.info(f"Saved ROI MVPA metrics for {args.subject} to {out_path}")

    logger.info("=" * 70)
    logger.info("MVPA processing complete (Aims 1–3, Glasser + Tian).")
    logger.info("Ready for Aim 4–5 Drug × Group analyses at the group level.")
    logger.info("=" * 70)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
