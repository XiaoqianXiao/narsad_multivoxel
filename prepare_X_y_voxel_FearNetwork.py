#!/usr/bin/env python3
"""
Prepare group-level feature matrices for extinction (phase2) and reinstatement (phase3)
using VOXEL-WISE features from anatomical ROI masks.

Source ROIs:
  /gscratch/fang/NARSAD/ROI/Gillian_anatomically_constrained

Each trial (CS-, CSS, CSR) becomes one row. The features are the concatenated 
voxel values from ALL masks found in the ROI directory.

Outputs (NPZ, compressed):
  [OUTPUT_DIR]/phase2_X_ext_y_ext_roi_voxels.npz
  [OUTPUT_DIR]/phase3_X_reinst_y_reinst_roi_voxels.npz

Structure of NPZ:
  - X_* : (n_trials, n_total_voxels)
  - y_* : (n_trials,) trial_type
  - subjects : (n_trials,)
  - roi_names : (n_rois,) names of the masks used
  - roi_voxel_counts : (n_rois,) number of voxels per mask (for reconstruction)
"""

import os
import glob
import logging
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import load_img, resample_to_img
from nilearn.masking import apply_mask
from bids import BIDSLayout
from utils import read_csv_with_detection  # Preserving your utils dependency

# =============================================================================
# PATHS / CONFIGURATION (Updated for Hyak)
# =============================================================================

# Updated Root for Hyak environment
ROOT_DIR = "/data"
PROJECT = "NARSAD"

# Derived paths
DERIV_DIR = os.path.join(ROOT_DIR, PROJECT, "MRI", "derivatives")
FMRIPREP_DIR = os.path.join(DERIV_DIR, "fmriprep")
FIRSTLEVEL_DIR = os.path.join(DERIV_DIR, "fMRI_analysis", "LSS", "firstLevel", "all_subjects")
BEHAV_DIR = os.path.join(ROOT_DIR, PROJECT, "MRI", "source_data", "behav")

# Specific ROI Directory requested
ROI_DIR = os.path.join(ROOT_DIR, PROJECT, "ROI", "Gillian_anatomically_constrained")

# Output Directory
OUTPUT_DIR = os.path.join(FIRSTLEVEL_DIR, "group_level")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PHASE2_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "phase2_X_ext_y_ext_roi_voxels.npz")
PHASE3_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "phase3_X_reinst_y_reinst_roi_voxels.npz")

CS_LABELS = ["CS-", "CSS", "CSR"]

# =============================================================================
# LOGGING
# =============================================================================

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger

# =============================================================================
# ROI HELPERS
# =============================================================================

def load_roi_masks(roi_dir, logger):
    """
    Find and load all ROI masks (nii/nii.gz) from the specified directory.
    Returns a list of Nifti images and their names.
    """
    # Grab both .nii and .nii.gz
    patterns = [os.path.join(roi_dir, "*.nii"), os.path.join(roi_dir, "*.nii.gz")]
    mask_files = []
    for p in patterns:
        mask_files.extend(glob.glob(p))
    
    mask_files = sorted(list(set(mask_files))) # Remove duplicates and sort
    
    if not mask_files:
        raise ValueError(f"No NIfTI mask files found in {roi_dir}")
        
    logger.info(f"Found {len(mask_files)} ROI masks in {roi_dir}")
    
    roi_imgs = []
    roi_names = []
    
    for f in mask_files:
        # Name is filename without extension(s)
        base = os.path.basename(f)
        if base.endswith(".nii.gz"):
            name = base[:-7]
        elif base.endswith(".nii"):
            name = base[:-4]
        else:
            name = base
            
        roi_names.append(name)
        roi_imgs.append(load_img(f))
        
    return roi_imgs, roi_names

# =============================================================================
# FEATURE EXTRACTION (Voxel-Wise)
# =============================================================================

def extract_voxel_features(LSS_img, roi_imgs, roi_names, logger):
    """
    Extracts ALL voxels from each ROI mask and concatenates them.
    
    Args:
        LSS_img: 4D Nifti image (trials in 4th dim)
        roi_imgs: List of 3D ROI mask images
        
    Returns:
        X_sub: (n_trials, n_total_voxels)
        roi_counts: List of integers (voxel count per ROI)
    """
    features_list = []
    roi_counts = []
    
    for i, mask in enumerate(roi_imgs):
        # apply_mask handles resampling if affine matches roughly, 
        # but ideally inputs should be in same space (MNI 2mm).
        # Returns shape: (n_trials, n_voxels_in_mask)
        try:
            roi_data = apply_mask(LSS_img, mask)
            features_list.append(roi_data)
            roi_counts.append(roi_data.shape[1])
        except Exception as e:
            logger.error(f"Error extracting features for ROI {roi_names[i]}: {e}")
            raise e
            
    # Concatenate all ROI voxels along feature dimension (axis 1)
    if not features_list:
        return None, []
        
    X_sub = np.hstack(features_list)
    return X_sub, roi_counts

# =============================================================================
# EVENTS & BIDS
# =============================================================================

def init_layout():
    return BIDSLayout(FMRIPREP_DIR, validate=False, derivatives=True)

def load_phase2_events(logger):
    events_path = os.path.join(BEHAV_DIR, "task-Narsad_phase2_events.csv")
    logger.info(f"  Loading phase2 events: {events_path}")
    events = read_csv_with_detection(events_path)
    if "usable" not in events.columns:
        events["usable"] = 1
    return events

def load_phase3_events_for_subject(subject, logger):
    # Handle N202 special naming
    if subject == "N202":
        fname = "task-NARSAD_phase-3_sub-202_events.csv"
    else:
        fname = "task-Narsad_phase3_events.csv"
        
    events_path = os.path.join(BEHAV_DIR, fname)
    logger.info(f"  Loading phase3 events: {events_path}")
    events = read_csv_with_detection(events_path)
    if "usable" not in events.columns:
        events["usable"] = 1
    return events

# =============================================================================
# PROCESSING PIPELINE
# =============================================================================

def process_phase(phase_name, lss_pattern, events_loader_func, roi_imgs, roi_names, logger):
    """
    Generic processor for Phase 2 or Phase 3.
    """
    lss_files = sorted(glob.glob(lss_pattern))
    subjects = []
    for f in lss_files:
        base = os.path.basename(f)
        # Assumes: sub-N101_task-phaseX_contrast1.nii.gz
        sub_id = base.split("_")[0].replace("sub-", "")
        subjects.append(sub_id)
    
    subjects = sorted(list(set(subjects)))
    logger.info(f"[{phase_name}] Found {len(subjects)} subjects with LSS files.")

    if not subjects:
        return None, None, None, None

    # Load template events if Phase 2 (shared), otherwise None
    events_template = None
    if phase_name == "PHASE2":
        events_template = events_loader_func(logger)

    X_list = []
    y_list = []
    subj_list = []
    
    # Store voxel counts from first valid subject to verify consistency
    reference_roi_counts = None

    for sub in subjects:
        logger.info(f"[{phase_name}] Processing {sub}...")
        
        # Find LSS file
        task_str = "phase2" if phase_name == "PHASE2" else "phase3"
        nii_path = os.path.join(FIRSTLEVEL_DIR, f"sub-{sub}_task-{task_str}_contrast1.nii.gz")
        if not os.path.exists(nii_path):
            nii_path = nii_path[:-3] # try .nii
        
        if not os.path.exists(nii_path):
            logger.warning(f"  Missing LSS file for {sub}, skipping.")
            continue

        # Load Data
        LSS_img = load_img(nii_path)
        n_trials = LSS_img.shape[3]
        
        # Load Events
        if phase_name == "PHASE2":
            events = events_template.copy()
        else:
            events = events_loader_func(sub, logger)

        # Sync lengths
        if len(events) < n_trials:
            LSS_img = LSS_img.slicer[..., :len(events)]
            n_trials = len(events)
        elif len(events) > n_trials:
            events = events.iloc[:n_trials].reset_index(drop=True)

        # Filter Usable CS Trials
        mask_cs = events["trial_type"].isin(CS_LABELS) & (events["usable"] == 1)
        if mask_cs.sum() == 0:
            logger.warning(f"  No usable trials for {sub}, skipping.")
            continue

        events_cs = events[mask_cs].reset_index(drop=True)

        # Extract Voxel-Wise Features
        try:
            X_sub_all, roi_counts = extract_voxel_features(LSS_img, roi_imgs, roi_names, logger)
        except Exception:
            logger.warning(f"  Feature extraction failed for {sub}, skipping.")
            continue

        # Consistency Check
        if reference_roi_counts is None:
            reference_roi_counts = roi_counts
            logger.info(f"  Voxel counts per ROI: {list(zip(roi_names, roi_counts))}")
            logger.info(f"  Total feature dimension: {sum(roi_counts)}")
        elif roi_counts != reference_roi_counts:
            logger.error(f"  Voxel count mismatch for {sub}! Skipping.")
            continue

        # Keep only CS trials
        X_sub = X_sub_all[mask_cs.values, :]
        y_sub = events_cs["trial_type"].astype(str).values
        subj_vec = np.array([sub] * len(y_sub), dtype=object)

        X_list.append(X_sub)
        y_list.append(y_sub)
        subj_list.append(subj_vec)

    if not X_list:
        logger.error(f"[{phase_name}] No valid data extracted.")
        return None, None, None, None

    # Stack all subjects
    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    subjects_all = np.concatenate(subj_list)

    logger.info(f"[{phase_name}] Finished. Shape: {X_all.shape}")
    return X_all, y_all, subjects_all, reference_roi_counts

# =============================================================================
# MAIN
# =============================================================================

def main():
    logger = setup_logging()
    logger.info("Starting Voxel-Wise ROI Feature Extraction")
    logger.info(f"ROI Directory: {ROI_DIR}")

    # 1. Load ROIs
    roi_imgs, roi_names = load_roi_masks(ROI_DIR, logger)

    # 2. Phase 2 (Extinction)
    phase2_pattern = os.path.join(FIRSTLEVEL_DIR, "sub-*_task-phase2_contrast1.nii*")
    X_ext, y_ext, subjects_ext, counts_ext = process_phase(
        "PHASE2", phase2_pattern, load_phase2_events, roi_imgs, roi_names, logger
    )

    if X_ext is not None:
        logger.info(f"Saving {PHASE2_OUTPUT_FILE}")
        np.savez_compressed(
            PHASE2_OUTPUT_FILE,
            X_ext=X_ext,
            y_ext=y_ext,
            subjects=subjects_ext,
            roi_names=roi_names,
            roi_voxel_counts=counts_ext
        )

    # 3. Phase 3 (Reinstatement)
    phase3_pattern = os.path.join(FIRSTLEVEL_DIR, "sub-*_task-phase3_contrast1.nii*")
    X_reinst, y_reinst, subjects_reinst, counts_reinst = process_phase(
        "PHASE3", phase3_pattern, load_phase3_events_for_subject, roi_imgs, roi_names, logger
    )

    if X_reinst is not None:
        logger.info(f"Saving {PHASE3_OUTPUT_FILE}")
        np.savez_compressed(
            PHASE3_OUTPUT_FILE,
            X_reinst=X_reinst,
            y_reinst=y_reinst,
            subjects=subjects_reinst,
            roi_names=roi_names,
            roi_voxel_counts=counts_reinst
        )

    logger.info("Done.")

if __name__ == "__main__":
    main()