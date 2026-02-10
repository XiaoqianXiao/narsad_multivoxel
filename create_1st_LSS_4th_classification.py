#!/usr/bin/env python3
"""
Script to generate SLURM scripts for LSS ROI classification (Step 4).

This script generates SLURM job submission scripts that call
`first_LSS_4th_classification.py` to perform ROI-based MVPA / classification
using single-trial LSS outputs (phase2 = vicarious extinction, phase3 = reinstatement).

Each SLURM script is for ONE subject, and the classification script internally
uses BOTH phase2 and phase3 (needed for PSI / cross-phase generalization).

Usage:
    # Generate scripts for all subjects
    python create_1st_LSS_4th_classification.py

    # Verbose and custom SLURM resources
    python create_1st_LSS_4th_classification.py --cpus-per-task 8 --memory 16G --time 06:00:00 -v

    # Dry run (see what would be created, but don't write files)
    python create_1st_LSS_4th_classification.py --dry-run

Workflow Context:
    Step 1: create_1st_LSS_singleTrialEstimate.py -> run_LSS.py (individual trial GLM)
    Step 2: create_1st_LSS_2nd_cateAlltrials.py -> first_LSS_2nd_cateAlltrials.py (merge trials)
    Step 3: create_1st_LSS_3rd_similarity.py -> first_LSS_3rd_similarity.py (similarity analysis)
    Step 4: create_1st_LSS_4th_classification.py -> first_LSS_4th_classification.py [CURRENT STEP]

Author: Xiaoqian Xiao (adapted by ChatGPT)
"""

# =============================================================================
# IMPORTS AND CONFIGURATION
# =============================================================================

import os
import argparse
import logging
from glob import glob
from bids.layout import BIDSLayout

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging():
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# =============================================================================
# CONFIGURATION AND PATHS
# =============================================================================

# Base paths
ROOT_DIR = os.getenv('DATA_DIR', '/data')
PROJECT_NAME = 'NARSAD'
DATA_DIR = os.path.join(ROOT_DIR, PROJECT_NAME, 'MRI')
DERIVATIVES_DIR = os.path.join(DATA_DIR, 'derivatives')
BEHAV_DIR = os.path.join(DATA_DIR, 'source_data', 'behav')

# LSS (first-level single-trial) paths
OUTPUT_DIR = os.path.join(DERIVATIVES_DIR, 'fMRI_analysis', 'LSS')
LSS_DIR = os.path.join(OUTPUT_DIR, 'firstLevel')
LSS_ALL_SUB_DIR = os.path.join(LSS_DIR, 'all_subjects')

# Scrubbed directory and container
SCRUBBED_DIR  = '/scrubbed_dir'
CONTAINER_PATH = "/gscratch/scrubbed/fanglab/xiaoqian/images/narsad.sif"

# Where to put generated SLURM scripts / logs
WORKFLOW_NAME = 'Lss_step4'
WORKFLOW_BASE = os.path.join(SCRUBBED_DIR, PROJECT_NAME, 'work_flows', WORKFLOW_NAME)

LOG_DIR = os.path.join(SCRUBBED_DIR, PROJECT_NAME, 'work_flows', 'Lss', 'logs', WORKFLOW_NAME)
#os.makedirs(LOG_DIR, exist_ok=True)

# BIDS configuration
SPACE = 'MNI152NLin2009cAsym'

# Path to the classification script inside the container repo
CLASSIFICATION_SCRIPT = "/app/first_LSS_4th_classification.py"

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate SLURM scripts for LSS ROI classification (Step 4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_1st_LSS_4th_classification.py
  python create_1st_LSS_4th_classification.py --cpus-per-task 8 --memory 16G --time 06:00:00 -v
  python create_1st_LSS_4th_classification.py --dry-run
        """
    )

    # SLURM resource parameters
    parser.add_argument('--account', default='fang',
                        help='SLURM account (default: fang)')
    parser.add_argument('--partition', default='ckpt-all',
                        help='SLURM partition (default: ckpt-all)')
    parser.add_argument('--cpus-per-task', type=int, default=8,
                        help='CPUs per task (default: 8)')
    parser.add_argument('--memory', default='16G',
                        help='Memory allocation (default: 16G)')
    parser.add_argument('--time', default='06:00:00',
                        help='Time limit (default: 06:00:00)')

    # Classification parameters
    parser.add_argument('--phase2-task', default='phase2',
                        help='Task name for extinction (default: phase2)')
    parser.add_argument('--phase3-task', default='phase3',
                        help='Task name for reinstatement (default: phase3)')
    parser.add_argument('--n-splits', type=int, default=4,
                        help='Max number of CV folds in ROI classifier (default: 4)')

    # Misc / debugging
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output with detailed progress information')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be created without creating files')

    args = parser.parse_args()
    return args

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def verify_paths():
    """Verify that all required core paths exist."""
    logger.info("Verifying input paths...")

    paths_to_check = [
        (DATA_DIR, "Data directory"),
        (DERIVATIVES_DIR, "Derivatives directory"),
        (BEHAV_DIR, "Behavioral directory"),
        (LSS_ALL_SUB_DIR, "LSS all_subjects directory"),
    ]

    missing = []
    for path, desc in paths_to_check:
        if not os.path.exists(path):
            logger.error(f"{desc} does not exist: {path}")
            missing.append(desc)
        else:
            logger.info(f"{desc}: {path} ✓")

    if missing:
        logger.error("Missing required paths: " + ", ".join(missing))
        return False
    return True


def discover_subjects_with_LSS(phase2_task, phase3_task):
    """
    Discover subjects that have LSS 4D images for BOTH phase2 and phase3.

    Uses naming:
      sub-{subject}_task-{task}_contrast1.nii.gz
    in LSS_ALL_SUB_DIR.
    """
    logger.info("Discovering subjects with LSS 4D images for both tasks...")

    pattern_p2 = os.path.join(LSS_ALL_SUB_DIR, f"sub-*_task-{phase2_task}_contrast1.nii*")
    pattern_p3 = os.path.join(LSS_ALL_SUB_DIR, f"sub-*_task-{phase3_task}_contrast1.nii*")

    files_p2 = glob(pattern_p2)
    files_p3 = glob(pattern_p3)

    subs_p2 = {os.path.basename(f).split("_")[0].replace("sub-", "") for f in files_p2}
    subs_p3 = {os.path.basename(f).split("_")[0].replace("sub-", "") for f in files_p3}

    common_subs = sorted(list(subs_p2.intersection(subs_p3)))

    logger.info(f"Found {len(subs_p2)} subjects with {phase2_task} LSS.")
    logger.info(f"Found {len(subs_p3)} subjects with {phase3_task} LSS.")
    logger.info(f"{len(common_subs)} subjects have BOTH {phase2_task} and {phase3_task} LSS:")
    logger.info(f"  {common_subs}")

    return common_subs


def get_mask_for_subject(layout, subject, task):
    """
    Get mask image path for a subject (and optionally task) from BIDS derivatives.

    Prefer space=MNI152NLin2009cAsym and suffix='mask'.
    """
    logger.info(f"Finding mask image for sub-{subject}, task-{task} (space={SPACE})...")
    try:
        mask_files = layout.get(
            subject=subject,
            task=task,
            space=SPACE,
            suffix='mask',
            extension=['.nii', '.nii.gz'],
            return_type='file'
        )
        if not mask_files:
            # Try without task constraint, in case mask is task-agnostic
            mask_files = layout.get(
                subject=subject,
                space=SPACE,
                suffix='mask',
                extension=['.nii', '.nii.gz'],
                return_type='file'
            )
        if not mask_files:
            logger.warning(f"No mask file found for sub-{subject}.")
            return None

        mask_path = mask_files[0]
        logger.info(f"Mask path for sub-{subject}: {mask_path}")
        return mask_path

    except Exception as e:
        logger.error(f"Error finding mask for sub-{subject}: {e}")
        return None


def create_slurm_script_for_subject(
    sub,
    phase2_task,
    phase3_task,
    mask_img_path,
    args,
):
    """
    Create a SLURM script for one subject, calling first_LSS_4th_classification.py.

    Args:
        sub (str): Subject ID
        phase2_task (str): extinction task name (e.g., 'phase2')
        phase3_task (str): reinstatement task name (e.g., 'phase3')
        mask_img_path (str): full path to the subject-specific mask image
        args: parsed CLI arguments (SLURM config, n_splits, etc.)

    Returns:
        str or None: path to created SLURM script, or None on error
    """
    logger.info(f"Creating SLURM script for sub-{sub} (tasks: {phase2_task}, {phase3_task})")

    slurm_config = {
        'account': args.account,
        'partition': args.partition,
        'cpus_per_task': args.cpus_per_task,
        'memory': args.memory,
        'time': args.time
    }

    # Where to write this subject's slurm script
    subject_work_dir = os.path.join(WORKFLOW_BASE)
    os.makedirs(subject_work_dir, exist_ok=True)

    script_path = os.path.join(subject_work_dir, f"sub_{sub}_classification_slurm.sh")
    log_dir = '/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/Lss_step4/log'

    # Log files
    out_log = os.path.join(log_dir, f"{phase2_task}_{phase3_task}_sub_{sub}_%j.out")
    err_log = os.path.join(log_dir, f"{phase2_task}_{phase3_task}_sub_{sub}_%j.err")

    # SLURM script content
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=LSS4_{sub}
#SBATCH --account={slurm_config['account']}
#SBATCH --partition={slurm_config['partition']}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={slurm_config['cpus_per_task']}
#SBATCH --mem={slurm_config['memory']}
#SBATCH --time={slurm_config['time']}
#SBATCH --output={out_log}
#SBATCH --error={err_log}

module load apptainer

apptainer exec \\
    -B /gscratch/fang:/data \\
    -B /gscratch/scrubbed/fanglab/xiaoqian:/scrubbed_dir \\
    -B /gscratch/scrubbed/fanglab/xiaoqian/repo/hyak_narsad_0917:/app \\
    {CONTAINER_PATH} \\
    python3 {CLASSIFICATION_SCRIPT} \\
    --subject {sub} \\
    --phase2_task {phase2_task} \\
    --phase3_task {phase3_task} \\
    --mask_img_path {mask_img_path} \\
    --n_splits {args.n_splits}
"""

    try:
        if args.dry_run:
            logger.info(f"[DRY RUN] Would write SLURM script to: {script_path}")
            logger.debug("SLURM script content:\n" + slurm_script)
            return script_path

        with open(script_path, "w") as f:
            f.write(slurm_script)
        os.chmod(script_path, 0o755)

        logger.info(f"SLURM script created: {script_path}")
        return script_path

    except Exception as e:
        logger.error(f"Error writing SLURM script {script_path}: {e}")
        return None

# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_arguments()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info("=" * 70)
    logger.info("LSS ROI Classification Script Generator - Step 4")
    logger.info("=" * 70)
    logger.info(f"Phase2 task: {args.phase2_task}")
    logger.info(f"Phase3 task: {args.phase3_task}")
    logger.info(f"SLURM resources: {args.cpus_per_task} CPUs, {args.memory} RAM, {args.time}")
    logger.info(f"n_splits for CV: {args.n_splits}")
    if args.dry_run:
        logger.info("DRY RUN MODE - No files will be created")
    logger.info("=" * 70)

    # Verify base paths
    if not verify_paths():
        logger.error("Path verification failed. Exiting.")
        return

    # Discover subjects with both phase2 and phase3 LSS
    subjects = discover_subjects_with_LSS(args.phase2_task, args.phase3_task)
    if not subjects:
        logger.error("No subjects with both phase2 and phase3 LSS were found. Exiting.")
        return

    # Initialize BIDS layout to find mask images
    try:
        layout = BIDSLayout(str(DATA_DIR), validate=False, derivatives=str(DERIVATIVES_DIR))
        logger.info(f"Initialized BIDSLayout with data directory: {DATA_DIR}")
    except Exception as e:
        logger.error(f"Error initializing BIDS layout: {e}")
        return

    total_scripts = 0
    errors = []

    for sub in subjects:
        try:
            # Use phase2 task to get mask path (should be same space for phase3)
            mask_img_path = get_mask_for_subject(layout, sub, args.phase2_task)
            if mask_img_path is None:
                logger.warning(f"Skipping sub-{sub}: no mask found.")
                continue

            script_path = create_slurm_script_for_subject(
                sub=sub,
                phase2_task=args.phase2_task,
                phase3_task=args.phase3_task,
                mask_img_path=mask_img_path,
                args=args,
            )

            if script_path:
                total_scripts += 1

        except Exception as e:
            msg = f"Error processing sub-{sub}: {e}"
            logger.error(msg)
            errors.append(msg)
            continue

    # Summary
    logger.info("=" * 70)
    logger.info("SCRIPT GENERATION SUMMARY")
    logger.info("=" * 70)
    if args.dry_run:
        logger.info(f"DRY RUN: Would create {total_scripts} SLURM scripts")
        logger.info("To actually create the scripts, run without --dry-run")
    else:
        logger.info(f"Created {total_scripts} SLURM scripts")
        logger.info(f"Scripts are located in: {WORKFLOW_BASE}")
        logger.info(f"Logs will go to:       {LOG_DIR}")

    if errors:
        logger.info("")
        logger.info("Errors encountered:")
        for e in errors:
            logger.info(f"  - {e}")

    logger.info("")
    logger.info("LSS Analysis Pipeline Status:")
    logger.info("  Step 1: First-level LSS analysis (individual trials)")
    logger.info("  Step 2: Trial merging (4D image creation)")
    logger.info("  Step 3: Similarity analysis (searchlight/ROI)")
    logger.info("  Step 4: ROI classification (current step) [SLURM scripts generated]")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
