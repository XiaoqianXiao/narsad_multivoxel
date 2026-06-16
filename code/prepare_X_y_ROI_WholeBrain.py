#!/usr/bin/env python3
"""
Prepare group-level feature matrices for extinction (phase2) and reinstatement (phase3)
using whole-brain parcellation:

  - Glasser HCP-MMP1 cortical atlas
  - Tian Subcortex S1 3T atlas

Each trial (CS-, CSS, CSR) becomes one row with features = mean LSS beta
within each parcel.

Outputs (NPZ, compressed):
  /data/NARSAD/MRI/derivatives/fMRI_analysis/LSS/firstLevel/all_subjects/group_level/
    phase2_X_ext_y_ext_glasser_tian.npz
    phase3_X_reinst_y_reinst_glasser_tian.npz

Each NPZ contains:
  - X_*        : (n_trials_total, n_parcels)
  - y_*        : (n_trials_total,) trial_type ('CS-', 'CSS', 'CSR')
  - subjects   : (n_trials_total,) subject ID string
  - parcel_indices : (n_parcels,) integer parcel IDs (Glasser first, then Tian)
  - parcel_names   : (n_parcels,) parcel names (HCP regionName / Tian label)
  - parcel_atlas   : (n_parcels,) 'Glasser' or 'Tian'
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
    #"Tian_Subcortex_S1_3T_2009cAsym.nii.gz"
)
TIAN_LABELS_PATH = os.path.join(
    ROI_DIR, "Tian", "3T", "Subcortex-Only", 
    "Tian_Subcortex_S4_3T_label.txt"
    #"Tian_Subcortex_S1_3T_label.txt"
)

OUTPUT_DIR = os.path.join(
    FIRSTLEVEL_DIR, "group_level"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

PHASE2_OUTPUT_FILE = os.path.join(
    OUTPUT_DIR, "phase2_X_ext_y_ext_glasser_tian.npz"
)
PHASE3_OUTPUT_FILE = os.path.join(
    OUTPUT_DIR, "phase3_X_reinst_y_reinst_glasser_tian.npz"
)

# Trial types we keep for modeling
CS_LABELS = ["CS-", "CSS", "CSR"]


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # clear existing handlers
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
    """
    Load Glasser region names and assume indices 1..N map to atlas labels.
    Uses column 'regionName' from HCP-MMP1_all.txt.
    """
    df = pd.read_csv(glasser_labels_path, sep=None, engine="python")
    if "regionName" not in df.columns:
        raise ValueError(
            f"'regionName' column not found in {glasser_labels_path}. "
            f"Columns are: {df.columns.tolist()}"
        )
    names = df["regionName"].astype(str).values
    #indices = np.arange(1, len(names) + 1, dtype=int)
    indices = df['regionID'].astype(int).values
    return indices, names


def load_tian_labels(tian_labels_path):
    """
    Load Tian parcel labels, using the *first column* as ROI name.

    We do NOT trust any integer column in the file for indices, because the
    format varies (e.g., strings like 'HIP-rh'). Instead, we:
      - Take the first column as the label name
      - Use indices 1..N to map to atlas labels

    This avoids 'HIP-rh' → int conversion issues.
    """
    df = pd.read_csv(
        tian_labels_path,
        sep=None,
        engine="python",
        header=None,
        comment="#"
    )
    first_col = df.columns[0]
    names = df[first_col].astype(str).values
    indices = np.arange(1, len(names) + 1, dtype=int)
    return indices, names


def build_parcel_metadata(glasser_img, tian_img, logger):
    """
    Build a combined list of parcel indices, names, and atlas source for
    Glasser + Tian.

    We simply use 1..N for each atlas; any parcel with 0 voxels after masking
    will effectively contribute no signal.
    """
    g_indices, g_names = load_glasser_labels(GLASSER_LABELS_PATH)
    t_indices, t_names = load_tian_labels(TIAN_LABELS_PATH)

    glasser_data = glasser_img.get_fdata()
    tian_data = tian_img.get_fdata()

    max_g = int(glasser_data.max())
    max_t = int(tian_data.max())

    if max_g < g_indices.max():
        logger.warning(
            f"Glasser atlas max label={max_g} < labels in table max={g_indices.max()}. "
            f"Truncating labels to 1..{max_g}"
        )
        keep = g_indices <= max_g
        g_indices = g_indices[keep]
        g_names = g_names[keep]

    if max_t < t_indices.max():
        logger.warning(
            f"Tian atlas max label={max_t} < labels in table max={t_indices.max()}. "
            f"Truncating labels to 1..{max_t}"
        )
        keep = t_indices <= max_t
        t_indices = t_indices[keep]
        t_names = t_names[keep]

    parcel_indices = np.concatenate([g_indices, t_indices])
    parcel_names = np.concatenate([g_names, t_names])
    parcel_atlas = np.array(
        ["Glasser"] * len(g_indices) + ["Tian"] * len(t_indices),
        dtype=object
    )

    logger.info(f"Total parcels: {len(parcel_indices)} "
                f"(Glasser={len(g_indices)}, Tian={len(t_indices)})")

    return parcel_indices, parcel_names, parcel_atlas


# =============================================================================
# BIDS / MASK HELPERS
# =============================================================================

def init_layout():
    layout = BIDSLayout(
        FMRIPREP_DIR,
        validate=False,
        derivatives=True
    )
    return layout


# def find_brain_mask(layout, subject, logger):
#     """
#     Find a brain mask for the subject in MNI152NLin2009cAsym space.
#     We do NOT fix session or task — we just take any functional brain mask
#     in that space and use the first match (prefer phase2/phase3 if present).
#     """
#     files = layout.get(
#         subject=subject,
#         suffix="mask",
#         desc="brain",
#         space="MNI152NLin2009cAsym",
#         datatype="func",
#         extension=[".nii", ".nii.gz"]
#     )

#     if not files:
#         logger.warning(f"  No brain mask found via BIDSLayout for {subject}.")
#         return None

#     chosen = None
#     for f in files:
#         if "task-phase2" in f.path or "task-phase3" in f.path:
#             chosen = f
#             break
#     if chosen is None:
#         chosen = files[0]

#     logger.info(f"  Using brain mask: {chosen.path}")
#     return chosen.path
def find_brain_mask(layout, subject, task, logger):
    """
    Find a brain mask for the subject in MNI152NLin2009cAsym space, specific to the given task (phase).
    """
    files = layout.get(
        subject=subject,
        task=task,
        suffix="mask",
        desc="brain",
        space="MNI152NLin2009cAsym",
        datatype="func",
        extension=[".nii", ".nii.gz"]
    )

    if not files:
        logger.warning(f"  No brain mask found via BIDSLayout for {subject} and task-{task}.")
        return None

    chosen = files[0]
    logger.info(f"  Using brain mask for task-{task}: {chosen.path}")
    return chosen.path


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_parcel_features(LSS_img, mask_img, glasser_img, tian_img,
                            parcel_indices, parcel_atlas, logger):
    """
    For a given subject and LSS 4D image, compute:
      - Mean beta per parcel per trial.

    Returns:
      X_sub : (n_trials, n_parcels)
    """
    # Ensure atlases are in the same space as the mask
    if glasser_img.shape[:3] != mask_img.shape[:3] or \
       not np.allclose(glasser_img.affine, mask_img.affine):
        logger.info("  Resampling Glasser atlas to mask space.")
        glasser_img = resample_to_img(glasser_img, mask_img, interpolation="nearest")

    if tian_img.shape[:3] != mask_img.shape[:3] or \
       not np.allclose(tian_img.affine, mask_img.affine):
        logger.info("  Resampling Tian atlas to mask space.")
        tian_img = resample_to_img(tian_img, mask_img, interpolation="nearest")

    glasser_data = glasser_img.get_fdata()
    tian_data = tian_img.get_fdata()
    mask_data = mask_img.get_fdata() > 0

    n_trials = LSS_img.shape[3]
    n_parcels = len(parcel_indices)
    X_sub = np.zeros((n_trials, n_parcels), dtype=np.float32)

    for j, (p_idx, atlas_name) in enumerate(zip(parcel_indices, parcel_atlas)):
        if atlas_name == "Glasser":
            roi_mask = (glasser_data == p_idx) & mask_data
        else:  # Tian
            roi_mask = (tian_data == p_idx) & mask_data

        if not roi_mask.any():
            # No voxels in this parcel under the mask; leave zeros
            continue

        roi_mask_img = nib.Nifti1Image(roi_mask.astype(np.int8), mask_img.affine)
        # Apply mask: returns (n_trials, n_voxels)
        X_vox = apply_mask(LSS_img, roi_mask_img)
        # Take mean across voxels
        X_sub[:, j] = X_vox.mean(axis=1)

    return X_sub


# =============================================================================
# EVENTS
# =============================================================================

def load_phase2_events(logger):
    """
    Phase2 uses a single events file for all subjects:
      /data/NARSAD/MRI/source_data/behav/task-Narsad_phase2_events.csv
    """
    events_path = os.path.join(BEHAV_DIR, "task-Narsad_phase2_events.csv")
    logger.info(f"  Loading phase2 events from: {events_path}")
    events = read_csv_with_detection(events_path)
    logger.info(
        f"  Phase2 events shape={events.shape}, "
        f"columns={list(events.columns)}"
    )

    if "trial_type" not in events.columns:
        raise ValueError(f"'trial_type' not found in {events_path}")

    if "usable" not in events.columns:
        logger.info("  No 'usable' column; assuming all trials are usable.")
        events["usable"] = 1

    return events


def load_phase3_events_for_subject(subject, logger):
    """
    Load phase3 events for a subject:
      - Standard case: task-Narsad_phase3_events.csv
      - N202 special case: task-NARSAD_phase-3_sub-202_events.csv
    """
    if subject == "N202":
        events_path = os.path.join(
            BEHAV_DIR, "task-NARSAD_phase-3_sub-202_events.csv"
        )
    else:
        events_path = os.path.join(
            BEHAV_DIR, "task-Narsad_phase3_events.csv"
        )

    logger.info(f"  Loading phase3 events from: {events_path}")
    events = read_csv_with_detection(events_path)
    logger.info(
        f"  Phase3 events shape={events.shape}, "
        f"columns={list(events.columns)}"
    )

    if "trial_type" not in events.columns:
        raise ValueError(f"'trial_type' not found in {events_path}")

    if "usable" not in events.columns:
        logger.info("  No 'usable' column; assuming all trials are usable.")
        events["usable"] = 1

    return events


# =============================================================================
# PHASE-SPECIFIC PROCESSING
# =============================================================================

def process_phase2(layout, glasser_img, tian_img,
                   parcel_indices, parcel_names, parcel_atlas, logger):
    """
    Build X_ext, y_ext, subjects_ext from phase2 LSS (extinction).
    """
    pattern = os.path.join(FIRSTLEVEL_DIR, "sub-*_task-phase2_contrast1.nii*")
    lss_files = sorted(glob.glob(pattern))
    subjects = []

    for f in lss_files:
        base = os.path.basename(f)
        # sub-N101_task-phase2_contrast1.nii.gz
        parts = base.split("_")
        sub_part = parts[0]  # 'sub-N101'
        sub_id = sub_part.replace("sub-", "")
        subjects.append(sub_id)

    subjects = sorted(list(set(subjects)))
    logger.info(f"[PHASE2] Found {len(subjects)} subjects with phase2 LSS: {subjects}")

    if not subjects:
        logger.error("[PHASE2] No phase2 LSS files found.")
        return None, None, None

    events_template = load_phase2_events(logger)

    X_list = []
    y_list = []
    subj_list = []

    for sub in subjects:
        logger.info("------------------------------------------------------------")
        logger.info(f"[PHASE2] Processing subject: {sub}")

        nii_path = os.path.join(
            FIRSTLEVEL_DIR,
            f"sub-{sub}_task-phase2_contrast1.nii.gz"
        )
        if not os.path.exists(nii_path):
            nii_path = os.path.join(
                FIRSTLEVEL_DIR,
                f"sub-{sub}_task-phase2_contrast1.nii"
            )
        if not os.path.exists(nii_path):
            logger.warning(f"  [PHASE2] No LSS file found for {sub}, skipping.")
            continue

        logger.info(f"  [PHASE2] LSS image path: {nii_path}")
        LSS_img = load_img(nii_path)
        n_trials = LSS_img.shape[3]
        logger.info(f"  [PHASE2] LSS image shape: {LSS_img.shape}, n_trials={n_trials}")

        mask_path = find_brain_mask(layout, sub, "phase2", logger)
        if mask_path is None:
            logger.warning(f"  [PHASE2] Skipping {sub} due to missing mask.")
            continue
        mask_img = load_img(mask_path)

        events = events_template.copy()

        if len(events) < n_trials:
            logger.warning(
                f"  [PHASE2] Events rows ({len(events)}) < LSS trials ({n_trials}); "
                f"truncating LSS to {len(events)}."
            )
            n_trials = len(events)
            LSS_img = LSS_img.slicer[..., :n_trials]

        if len(events) > n_trials:
            logger.info(
                f"  [PHASE2] Events rows ({len(events)}) > LSS trials ({n_trials}); "
                f"truncating events."
            )
            events = events.iloc[:n_trials].reset_index(drop=True)

        mask_cs = events["trial_type"].isin(CS_LABELS) & (events["usable"] == 1)
        n_keep = mask_cs.sum()
        logger.info(f"  [PHASE2] Usable CS trials (CS-, CSS, CSR): {n_keep}")

        if n_keep == 0:
            logger.warning(f"  [PHASE2] No usable CS trials for {sub}, skipping.")
            continue

        events_cs = events[mask_cs].reset_index(drop=True)

        X_sub_all = extract_parcel_features(
            LSS_img, mask_img, glasser_img, tian_img,
            parcel_indices, parcel_atlas, logger
        )
        X_sub = X_sub_all[mask_cs.values, :]

        y_sub = events_cs["trial_type"].astype(str).values
        subj_vec = np.array([sub] * len(y_sub), dtype=object)

        X_list.append(X_sub)
        y_list.append(y_sub)
        subj_list.append(subj_vec)

    if not X_list:
        logger.error("[PHASE2] No valid subject data collected; nothing to save.")
        return None, None, None

    X_ext = np.vstack(X_list)
    y_ext = np.concatenate(y_list)
    subjects_all = np.concatenate(subj_list)

    logger.info("======================================================================")
    logger.info(f"[PHASE2] Final X_ext shape: {X_ext.shape}")
    logger.info(f"[PHASE2] Number of labels: {len(y_ext)}")
    unique_labels, counts = np.unique(y_ext, return_counts=True)
    logger.info(f"[PHASE2] Label distribution: {dict(zip(unique_labels, counts))}")
    logger.info("======================================================================")

    return X_ext, y_ext, subjects_all


def process_phase3(layout, glasser_img, tian_img,
                   parcel_indices, parcel_names, parcel_atlas, logger):
    """
    Build X_reinst, y_reinst, subjects_reinst from phase3 LSS (reinstatement).
    """
    pattern = os.path.join(FIRSTLEVEL_DIR, "sub-*_task-phase3_contrast1.nii*")
    lss_files = sorted(glob.glob(pattern))
    subjects = []

    for f in lss_files:
        base = os.path.basename(f)
        # sub-N101_task-phase3_contrast1.nii.gz
        parts = base.split("_")
        sub_part = parts[0]  # 'sub-N101'
        sub_id = sub_part.replace("sub-", "")
        subjects.append(sub_id)

    subjects = sorted(list(set(subjects)))
    logger.info(f"[PHASE3] Found {len(subjects)} subjects with phase3 LSS: {subjects}")

    if not subjects:
        logger.error("[PHASE3] No phase3 LSS files found.")
        return None, None, None

    X_list = []
    y_list = []
    subj_list = []

    for sub in subjects:
        logger.info("------------------------------------------------------------")
        logger.info(f"[PHASE3] Processing subject: {sub}")

        nii_path = os.path.join(
            FIRSTLEVEL_DIR,
            f"sub-{sub}_task-phase3_contrast1.nii.gz"
        )
        if not os.path.exists(nii_path):
            nii_path = os.path.join(
                FIRSTLEVEL_DIR,
                f"sub-{sub}_task-phase3_contrast1.nii"
            )
        if not os.path.exists(nii_path):
            logger.warning(f"  [PHASE3] No LSS file found for {sub}, skipping.")
            continue

        logger.info(f"  [PHASE3] LSS image path: {nii_path}")
        LSS_img = load_img(nii_path)
        n_trials = LSS_img.shape[3]
        logger.info(f"  [PHASE3] LSS image shape: {LSS_img.shape}, n_trials={n_trials}")

        mask_path = find_brain_mask(layout, sub, "phase3", logger)
        if mask_path is None:
            logger.warning(f"  [PHASE3] Skipping {sub} due to missing mask.")
            continue
        mask_img = load_img(mask_path)

        events = load_phase3_events_for_subject(sub, logger)

        if len(events) < n_trials:
            logger.warning(
                f"  [PHASE3] Events rows ({len(events)}) < LSS trials ({n_trials}); "
                f"truncating LSS to {len(events)}."
            )
            n_trials = len(events)
            LSS_img = LSS_img.slicer[..., :n_trials]

        if len(events) > n_trials:
            logger.info(
                f"  [PHASE3] Events rows ({len(events)}) > LSS trials ({n_trials}); "
                f"truncating events."
            )
            events = events.iloc[:n_trials].reset_index(drop=True)

        mask_cs = events["trial_type"].isin(CS_LABELS) & (events["usable"] == 1)
        n_keep = mask_cs.sum()
        logger.info(f"  [PHASE3] Usable CS trials (CS-, CSS, CSR): {n_keep}")

        if n_keep == 0:
            logger.warning(f"  [PHASE3] No usable CS trials for {sub}, skipping.")
            continue

        events_cs = events[mask_cs].reset_index(drop=True)

        X_sub_all = extract_parcel_features(
            LSS_img, mask_img, glasser_img, tian_img,
            parcel_indices, parcel_atlas, logger
        )
        X_sub = X_sub_all[mask_cs.values, :]

        y_sub = events_cs["trial_type"].astype(str).values
        subj_vec = np.array([sub] * len(y_sub), dtype=object)

        X_list.append(X_sub)
        y_list.append(y_sub)
        subj_list.append(subj_vec)

    if not X_list:
        logger.error("[PHASE3] No valid subject data collected; nothing to save.")
        return None, None, None

    X_reinst = np.vstack(X_list)
    y_reinst = np.concatenate(y_list)
    subjects_all = np.concatenate(subj_list)

    logger.info("======================================================================")
    logger.info(f"[PHASE3] Final X_reinst shape: {X_reinst.shape}")
    logger.info(f"[PHASE3] Number of labels: {len(y_reinst)}")
    unique_labels, counts = np.unique(y_reinst, return_counts=True)
    logger.info(f"[PHASE3] Label distribution: {dict(zip(unique_labels, counts))}")
    logger.info("======================================================================")

    return X_reinst, y_reinst, subjects_all


# =============================================================================
# MAIN
# =============================================================================

def main():
    logger = setup_logging()

    logger.info("======================================================================")
    logger.info("Preparing X_ext (phase2) and X_reinst (phase3) with Glasser+Tian parcels")
    logger.info("======================================================================")

    # Load atlases once
    logger.info(f"Loading Glasser atlas from: {GLASSER_ATLAS_PATH}")
    glasser_img = load_img(GLASSER_ATLAS_PATH)

    logger.info(f"Loading Tian atlas from: {TIAN_ATLAS_PATH}")
    tian_img = load_img(TIAN_ATLAS_PATH)

    # Build parcel metadata once
    parcel_indices, parcel_names, parcel_atlas = build_parcel_metadata(
        glasser_img, tian_img, logger
    )

    # Initialize BIDSLayout once
    logger.info("Initializing BIDSLayout for fMRIPrep derivatives...")
    layout = init_layout()

    # -------------------------
    # Phase 2: Extinction
    # -------------------------
    X_ext, y_ext, subjects_ext = process_phase2(
        layout, glasser_img, tian_img,
        parcel_indices, parcel_names, parcel_atlas, logger
    )

    if X_ext is not None:
        logger.info(f"Saving phase2 NPZ to: {PHASE2_OUTPUT_FILE}")
        np.savez_compressed(
            PHASE2_OUTPUT_FILE,
            X_ext=X_ext,
            y_ext=y_ext,
            subjects=subjects_ext,
            parcel_indices=parcel_indices,
            parcel_names=parcel_names,
            parcel_atlas=parcel_atlas
        )

    # -------------------------
    # Phase 3: Reinstatement
    # -------------------------
    X_reinst, y_reinst, subjects_reinst = process_phase3(
        layout, glasser_img, tian_img,
        parcel_indices, parcel_names, parcel_atlas, logger
    )

    if X_reinst is not None:
        logger.info(f"Saving phase3 NPZ to: {PHASE3_OUTPUT_FILE}")
        np.savez_compressed(
            PHASE3_OUTPUT_FILE,
            X_reinst=X_reinst,
            y_reinst=y_reinst,
            subjects=subjects_reinst,
            parcel_indices=parcel_indices,
            parcel_names=parcel_names,
            parcel_atlas=parcel_atlas
        )

    logger.info("======================================================================")
    logger.info("Done building group-level matrices for phase2 (ext) and phase3 (reinst).")
    logger.info("======================================================================")


if __name__ == "__main__":
    main()
