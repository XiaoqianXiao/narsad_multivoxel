#!/usr/bin/env python3
"""
Prepare group-level feature matrices for extinction (phase2) and reinstatement (phase3)
using whole-brain VOXEL-WISE extraction.

STRATEGY:
  1. Detect the voxel grid (affine/shape) from the FIRST subject found.
  2. Resample the Glasser & Tian atlases to match this subject grid (nearest-neighbor).
  3. Create the Master Mask and Metadata from these resampled atlases.
  4. Extract data from all subjects using this mask.
     (This avoids resampling subject beta maps).

Outputs (NPZ, compressed):
  /data/NARSAD/MRI/derivatives/fMRI_analysis/LSS/firstLevel/all_subjects/group_level/
    phase2_X_ext_y_ext_voxels_glasser_tian.npz
    phase3_X_reinst_y_reinst_voxels_glasser_tian.npz

Each NPZ contains:
  - X_* : (n_trials_total, n_voxels)
  - y_* : (n_trials_total,) trial_type
  - subjects       : (n_trials_total,)
  - parcel_indices : (n_voxels,) integer parcel IDs
  - parcel_names   : (n_voxels,) parcel names
  - parcel_atlas   : (n_voxels,) 'Glasser' or 'Tian'
"""

import os
import glob
import logging
import json

import numpy as np
import pandas as pd
import nibabel as nib

from nilearn.image import load_img, resample_to_img
from nilearn.masking import apply_mask
from bids import BIDSLayout
from utils import read_csv_with_detection


# =============================================================================
# PATHS / CONSTANTS
# =============================================================================

ROOT_DIR = "/data"
PROJECT = "NARSAD"

DERIV_DIR = os.path.join(ROOT_DIR, PROJECT, "MRI", "derivatives")
FMRIPREP_DIR = os.path.join(DERIV_DIR, "fmriprep")
FIRSTLEVEL_DIR = os.path.join(
    DERIV_DIR, "fMRI_analysis", "LSS", "firstLevel", "all_subjects"
)
BEHAV_DIR = os.path.join(ROOT_DIR, PROJECT, "MRI", "source_data", "behav")

ROI_DIR = os.path.join(ROOT_DIR, PROJECT, "ROI")
GLASSER_ATLAS_PATH = os.path.join(ROI_DIR, "Glasser", "HCP-MMP1_2mm.nii")
GLASSER_LABELS_PATH = os.path.join(ROI_DIR, "Glasser", "HCP-MMP1_all.txt")

TIAN_ATLAS_PATH = os.path.join(
    ROI_DIR, "Tian", "3T", "Subcortex-Only",
    "Tian_Subcortex_S4_3T_2009cAsym.nii.gz" 
)
TIAN_LABELS_PATH = os.path.join(
    ROI_DIR, "Tian", "3T", "Subcortex-Only", 
    "Tian_Subcortex_S4_3T_label.txt"
)

OUTPUT_DIR = os.path.join(
    FIRSTLEVEL_DIR, "group_level"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

PHASE2_OUTPUT_FILE = os.path.join(
    OUTPUT_DIR, "phase2_X_ext_y_ext_voxels_glasser_tian.npz"
)
PHASE3_OUTPUT_FILE = os.path.join(
    OUTPUT_DIR, "phase3_X_reinst_y_reinst_voxels_glasser_tian.npz"
)

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
# LABEL / ATLAS HELPERS
# =============================================================================

def load_glasser_labels(glasser_labels_path):
    df = pd.read_csv(glasser_labels_path, sep=None, engine="python")
    label_dict = dict(zip(df['regionID'].astype(int), df['regionName'].astype(str)))
    return label_dict

def load_tian_labels(tian_labels_path):
    df = pd.read_csv(tian_labels_path, sep=None, engine="python", header=None, comment="#")
    first_col = df.columns[0]
    names = df[first_col].astype(str).values
    indices = np.arange(1, len(names) + 1, dtype=int)
    label_dict = dict(zip(indices, names))
    return label_dict

def get_reference_subject_geometry(logger):
    """
    Finds the FIRST available LSS file to use as the geometric reference 
    (affine, shape) for resampling the atlases.
    """
    pattern = os.path.join(FIRSTLEVEL_DIR, "sub-*_task-*_contrast1.nii*")
    files = sorted(glob.glob(pattern))
    
    if not files:
        raise RuntimeError("No LSS files found to determine reference geometry!")
    
    ref_path = files[0]
    logger.info(f"Using reference geometry from: {os.path.basename(ref_path)}")
    return load_img(ref_path)

def build_master_mask_and_metadata(glasser_img, tian_img, reference_img, logger):
    """
    1. Resamples Glasser & Tian to match 'reference_img' (Subject Space).
    2. Creates a combined 'Master Mask'.
    3. Generates voxel-wise metadata.
    """
    
    logger.info("  Resampling Atlases to match Reference Subject geometry (Nearest Neighbor)...")
    
    # Resample Glasser to Subject
    glasser_res = resample_to_img(glasser_img, reference_img, interpolation='nearest')
    # Resample Tian to Subject
    tian_res = resample_to_img(tian_img, reference_img, interpolation='nearest')

    glasser_data = glasser_res.get_fdata().astype(int)
    tian_data = tian_res.get_fdata().astype(int)

    # Load Names
    glasser_labels_dict = load_glasser_labels(GLASSER_LABELS_PATH)
    tian_labels_dict = load_tian_labels(TIAN_LABELS_PATH)

    # Offset Tian IDs
    max_glasser_id = int(glasser_data.max())
    tian_offset = max_glasser_id + 1
    logger.info(f"  Max Glasser ID: {max_glasser_id}. Tian offset: {tian_offset}")

    # Create map array
    roi_map_data = np.zeros_like(glasser_data)
    
    # Fill Glasser
    mask_glasser = glasser_data > 0
    roi_map_data[mask_glasser] = glasser_data[mask_glasser]
    
    # Fill Tian (where Glasser is 0)
    mask_tian = (tian_data > 0)
    mask_fill_tian = mask_tian & (roi_map_data == 0)
    roi_map_data[mask_fill_tian] = tian_data[mask_fill_tian] + tian_offset
    
    # Master Binary Mask
    master_mask_data = roi_map_data > 0
    master_mask_img = nib.Nifti1Image(master_mask_data.astype(np.int8), reference_img.affine)
    
    # Extract voxel-wise metadata
    unique_ids = roi_map_data[master_mask_data] # Shape (n_voxels,)
    n_voxels = len(unique_ids)
    logger.info(f"  Total voxels in combined mask: {n_voxels}")

    parcel_indices = np.zeros(n_voxels, dtype=int)
    parcel_names = np.empty(n_voxels, dtype=object)
    parcel_atlas = np.empty(n_voxels, dtype=object)

    unique_present_ids = np.unique(unique_ids)

    for uid in unique_present_ids:
        mask_uid = (unique_ids == uid)
        if uid <= max_glasser_id:
            original_id = uid
            atlas_name = "Glasser"
            p_name = glasser_labels_dict.get(original_id, f"Unknown_Glasser_{original_id}")
        else:
            original_id = uid - tian_offset
            atlas_name = "Tian"
            p_name = tian_labels_dict.get(original_id, f"Unknown_Tian_{original_id}")
            p_name = f"Tian_{p_name}"

        parcel_indices[mask_uid] = original_id
        parcel_names[mask_uid] = p_name
        parcel_atlas[mask_uid] = atlas_name

    return master_mask_img, parcel_indices, parcel_names, parcel_atlas


# =============================================================================
# BIDS HELPERS
# =============================================================================

def init_layout():
    layout = BIDSLayout(FMRIPREP_DIR, validate=False, derivatives=True)
    return layout

# =============================================================================
# EVENTS
# =============================================================================

def load_phase2_events(logger):
    events_path = os.path.join(BEHAV_DIR, "task-Narsad_phase2_events.csv")
    events = read_csv_with_detection(events_path)
    if "trial_type" not in events.columns:
        raise ValueError(f"'trial_type' not found in {events_path}")
    if "usable" not in events.columns:
        events["usable"] = 1
    return events

def load_phase3_events_for_subject(subject, logger):
    if subject == "N202":
        events_path = os.path.join(BEHAV_DIR, "task-NARSAD_phase-3_sub-202_events.csv")
    else:
        events_path = os.path.join(BEHAV_DIR, "task-Narsad_phase3_events.csv")
    events = read_csv_with_detection(events_path)
    if "trial_type" not in events.columns:
        raise ValueError(f"'trial_type' not found in {events_path}")
    if "usable" not in events.columns:
        events["usable"] = 1
    return events

# =============================================================================
# PHASE PROCESSING
# =============================================================================

def process_phase_generic(layout, phase_name, master_mask_img, logger):
    """
    Generic processor.
    We verify that every subject matches the master_mask_img affine.
    """
    contrast_suffix = "contrast1"
    
    pattern = os.path.join(FIRSTLEVEL_DIR, f"sub-*_task-{phase_name}_{contrast_suffix}.nii*")
    lss_files = sorted(glob.glob(pattern))
    subjects = []
    for f in lss_files:
        base = os.path.basename(f)
        parts = base.split("_")
        sub_id = parts[0].replace("sub-", "")
        subjects.append(sub_id)
    subjects = sorted(list(set(subjects)))
    
    logger.info(f"[{phase_name.upper()}] Found {len(subjects)} subjects.")
    if not subjects:
        return None, None, None

    if phase_name == "phase2":
        events_template = load_phase2_events(logger)
    
    X_list = []
    y_list = []
    subj_list = []
    
    # Cache the affine of the master mask for quick comparison
    reference_affine = master_mask_img.affine

    for sub in subjects:
        # Locate LSS
        nii_path_gz = os.path.join(FIRSTLEVEL_DIR, f"sub-{sub}_task-{phase_name}_{contrast_suffix}.nii.gz")
        nii_path = os.path.join(FIRSTLEVEL_DIR, f"sub-{sub}_task-{phase_name}_{contrast_suffix}.nii")
        use_path = nii_path_gz if os.path.exists(nii_path_gz) else nii_path
        
        if not os.path.exists(use_path):
            continue
            
        LSS_img = load_img(use_path)
        
        # --- SAFETY CHECK ---
        # Ensure subject matches the reference geometry used to create the mask
        if not np.allclose(LSS_img.affine, reference_affine, atol=1e-5):
            logger.warning(
                f"  SKIP {sub}: Voxel grid mismatch! \n"
                f"  Ref: {reference_affine[0]}\n"
                f"  Sub: {LSS_img.affine[0]}"
            )
            continue
        # ---------------------

        n_trials = LSS_img.shape[3]

        if phase_name == "phase2":
            events = events_template.copy()
        else:
            events = load_phase3_events_for_subject(sub, logger)

        if len(events) < n_trials:
            LSS_img = LSS_img.slicer[..., :len(events)]
            n_trials = len(events)
        elif len(events) > n_trials:
            events = events.iloc[:n_trials].reset_index(drop=True)

        mask_cs = events["trial_type"].isin(CS_LABELS) & (events["usable"] == 1)
        if mask_cs.sum() == 0:
            continue

        events_cs = events[mask_cs].reset_index(drop=True)
        
        # Extract Voxels (No resampling, strictly apply_mask)
        logger.info(f"  Processing {sub}...")
        X_sub_all = apply_mask(imgs=LSS_img, mask_img=master_mask_img)
        
        X_sub = X_sub_all[mask_cs.values, :]
        y_sub = events_cs["trial_type"].astype(str).values
        subj_vec = np.array([sub] * len(y_sub), dtype=object)

        X_list.append(X_sub)
        y_list.append(y_sub)
        subj_list.append(subj_vec)

    if not X_list:
        return None, None, None

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    subjects_all = np.concatenate(subj_list)
    
    return X_all, y_all, subjects_all


# =============================================================================
# MAIN
# =============================================================================

def main():
    logger = setup_logging()
    logger.info("Starting VOXEL-WISE feature extraction (Resample Atlas -> Subject)...")

    # 1. Determine Reference Geometry
    logger.info("Identifying reference subject geometry...")
    reference_img = get_reference_subject_geometry(logger)

    # 2. Load Original Atlases
    glasser_img = load_img(GLASSER_ATLAS_PATH)
    tian_img = load_img(TIAN_ATLAS_PATH)

    # 3. Build Master Mask & Metadata (Resampled to Reference)
    logger.info("Building Master Mask and Metadata arrays...")
    master_mask_img, parcel_indices, parcel_names, parcel_atlas = build_master_mask_and_metadata(
        glasser_img, tian_img, reference_img, logger
    )
    
    # 4. Initialize Layout
    layout = init_layout()

    # 5. Process Phase 2
    logger.info("--- PHASE 2 (Extinction) ---")
    X_ext, y_ext, subjects_ext = process_phase_generic(
        layout, "phase2", master_mask_img, logger
    )
    
    if X_ext is not None:
        logger.info(f"Saving Phase 2 to {PHASE2_OUTPUT_FILE}")
        logger.info(f"Shape: {X_ext.shape}")
        np.savez_compressed(
            PHASE2_OUTPUT_FILE,
            X_ext=X_ext,
            y_ext=y_ext,
            subjects=subjects_ext,
            parcel_indices=parcel_indices,
            parcel_names=parcel_names,
            parcel_atlas=parcel_atlas
        )

    # 6. Process Phase 3
    logger.info("--- PHASE 3 (Reinstatement) ---")
    X_reinst, y_reinst, subjects_reinst = process_phase_generic(
        layout, "phase3", master_mask_img, logger
    )

    if X_reinst is not None:
        logger.info(f"Saving Phase 3 to {PHASE3_OUTPUT_FILE}")
        logger.info(f"Shape: {X_reinst.shape}")
        np.savez_compressed(
            PHASE3_OUTPUT_FILE,
            X_reinst=X_reinst,
            y_reinst=y_reinst,
            subjects=subjects_reinst,
            parcel_indices=parcel_indices,
            parcel_names=parcel_names,
            parcel_atlas=parcel_atlas
        )

    logger.info("Done.")

if __name__ == "__main__":
    main()