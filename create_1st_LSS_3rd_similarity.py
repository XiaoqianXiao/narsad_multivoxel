#!/usr/bin/env python3
"""
Script to generate SLURM scripts for LSS similarity analysis (Step 3).

This script generates SLURM job submission scripts that call first_LSS_3rd_similarity.py
to compute similarity analysis for LSS trial outputs. It's designed to work with
the LSS workflow and create batch jobs for similarity analysis.

Usage:
    # Generate scripts for all subjects and tasks
    python create_1st_LSS_3rd_similarity.py
    
    # Generate scripts with custom parameters
    python create_1st_LSS_3rd_similarity.py --batch-size 500 --n-jobs 8
    
    # Generate scripts with profiling enabled
    python create_1st_LSS_3rd_similarity.py --profile
    
    # Generate scripts for specific analysis types only
    python create_1st_LSS_3rd_similarity.py --analysis-types searchlight roi
    
    # Show help
    python create_1st_LSS_3rd_similarity.py --help

Features:
    - Automatically discovers subjects and tasks using BIDS layout
    - Generates SLURM scripts for searchlight, ROI, or both analysis types
    - Configurable computing resources and analysis parameters
    - Creates organized output directory structure
    - Provides detailed progress feedback and error handling
    - Supports profiling for performance debugging
    - Flexible analysis type selection
    - Automatic mask file discovery

Output:
    - SLURM scripts: sub_{subject}_slurm.sh in organized directories
    - Organized by task and analysis type
    - Ready for job submission using launch_1st_LSS_3rd_similarity.sh

Workflow Context:
    This script is Step 3 of the LSS analysis pipeline:
    
    Step 1: create_1st_LSS_singleTrialEstimate.py -> run_LSS.py (individual trial GLM)
    Step 2: create_1st_LSS_2nd_cateAlltrials.py -> first_LSS_2nd_cateAlltrials.py (merge trials)
    Step 3: create_1st_LSS_3rd_similarity.py -> first_LSS_3rd_similarity.py (similarity analysis) [CURRENT STEP]
    Step 4: Group-level analysis and statistical testing

Author: Xiaoqian Xiao (xiao.xiaoqian.320@gmail.com)
"""

# =============================================================================
# IMPORTS AND CONFIGURATION
# =============================================================================

import os
import argparse
import json
import pandas as pd
from bids.layout import BIDSLayout
from nipype import config, logging as nipype_logging
import logging

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

# =============================================================================
# CONFIGURATION AND PATHS
# =============================================================================

# Set FSL environment
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['FSLDIR'] = '/usr/local/fsl'
os.environ['PATH'] += os.pathsep + os.path.join(os.environ['FSLDIR'], 'bin')

# Nipype configuration
config.set('execution', 'remove_unnecessary_outputs', 'false')
nipype_logging.update_logging(config)

# Paths
ROOT_DIR = os.getenv('DATA_DIR', '/data')
PROJECT_NAME = 'NARSAD'
DATA_DIR = os.path.join(ROOT_DIR, PROJECT_NAME, 'MRI')
DERIVATIVES_DIR = os.path.join(DATA_DIR, 'derivatives')
BEHAV_DIR = os.path.join(DATA_DIR, 'source_data', 'behav')
OUTPUT_DIR = os.path.join(DERIVATIVES_DIR, 'fMRI_analysis', 'LSS')
LSS_DIR = os.path.join(OUTPUT_DIR, 'firstLevel')
RESULTS_DIR = os.path.join(LSS_DIR, 'all_subjects')

# Scrubbed directory and container
SCRUBBED_DIR = '/scrubbed_dir'
CONTAINER_PATH = "/gscratch/scrubbed/fanglab/xiaoqian/images/narsad.sif"

# Atlas and ROI paths
COMBINED_ATLAS_PATH = ('/scrubbed_dir/parcellation/Tian/3T/'
                       'Cortex-Subcortex/MNIvolumetric/Schaefer2018_100Parcels_7Networks_order_'
                       'Tian_Subcortex_S1_3T_MNI152NLin2009cAsym_2mm.nii.gz')
ROI_NAMES_FILE = ('/scrubbed_dir/parcellation/Tian/3T/'
                  'Cortex-Subcortex/Schaefer2018_100Parcels_7Networks_order_'
                  'Tian_Subcortex_S1_label.txt')

# BIDS configuration
SPACE = 'MNI152NLin2009cAsym'

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate SLURM scripts for LSS similarity analysis (Step 3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate scripts for all subjects and tasks
    python create_1st_LSS_3rd_similarity.py
    
    # Generate scripts with custom parameters
    python create_1st_LSS_3rd_similarity.py --batch-size 500 --n-jobs 8
    
    # Generate scripts with profiling enabled
    python create_1st_LSS_3rd_similarity.py --profile
    
    # Generate scripts for specific analysis types only
    python create_1st_LSS_3rd_similarity.py --analysis-types searchlight roi
    
    # Generate scripts with custom SLURM resources
    python create_1st_LSS_3rd_similarity.py --cpus-per-task 32 --memory 40G --time 08:00:00
        """
    )
    
    # Analysis parameters
    parser.add_argument('--analysis-types', nargs='+', 
                       choices=['searchlight', 'roi', 'both'], default=['both'],
                       help='Analysis types to generate scripts for (default: both)')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size for searchlight analysis (default: 1000)')
    parser.add_argument('--n-jobs', type=int, default=12,
                       help='Number of parallel jobs (default: 12)')
    
    # SLURM resource parameters
    parser.add_argument('--account', default='fang',
                       help='SLURM account (default: fang)')
    parser.add_argument('--partition', default='ckpt-all',
                       help='SLURM partition (default: ckpt-all)')
    parser.add_argument('--cpus-per-task', type=int, default=16,
                       help='CPUs per task (default: 16)')
    parser.add_argument('--memory', default='20G',
                       help='Memory allocation (default: 20G)')
    parser.add_argument('--time', default='04:00:00',
                       help='Time limit (default: 04:00:00)')
    
    # Debugging options
    parser.add_argument('--profile', action='store_true',
                       help='Enable profiling for debugging')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output with detailed progress information')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be created without creating files')
    
    args = parser.parse_args()
    
    # Validate analysis types
    if not args.analysis_types:
        parser.error("Must specify at least one analysis type")
    
    return args

# =============================================================================
# SLURM SCRIPT GENERATION
# =============================================================================

def create_slurm_script(sub, task, work_dir, mask_img_path, combined_atlas_path, 
                        roi_names_file, analysis_type='both', batch_size=1000, n_jobs=12, 
                        profile=False, slurm_config=None):
    """
    Create a SLURM script for LSS similarity analysis.
    
    Args:
        sub (str): Subject ID
        task (str): Task name
        work_dir (str): Working directory for output
        mask_img_path (str): Path to brain mask image
        combined_atlas_path (str): Path to combined atlas image
        roi_names_file (str): Path to ROI names file
        analysis_type (str): Type of analysis (searchlight, roi, or both)
        batch_size (int): Batch size for searchlight analysis
        n_jobs (int): Number of parallel jobs
        profile (bool): Whether to enable profiling
        slurm_config (dict): SLURM configuration parameters
    
    Returns:
        str: Path to the created SLURM script
    """
    logger.info(f"Creating SLURM script for sub-{sub}, task-{task}, analysis_type={analysis_type}")
    
    # Use default SLURM config if none provided
    if slurm_config is None:
        slurm_config = {
            'account': 'fang',
            'partition': 'ckpt-all',
            'cpus_per_task': 16,
            'memory': '20G',
            'time': '04:00:00'
        }
    
    profile_flag = "--profile" if profile else ""
    
    # Create SLURM script content
    slurm_script = f"""#!/bin/bash 
#SBATCH --job-name=LSS_3_{sub}_{task}_{analysis_type}                                                                    
#SBATCH --account={slurm_config['account']}                                                                                            
#SBATCH --partition={slurm_config['partition']}                                                                                      
#SBATCH --nodes=1                                                                                                 
#SBATCH --ntasks=1                                                                                                
#SBATCH --cpus-per-task={slurm_config['cpus_per_task']}                                                                                        
#SBATCH --mem={slurm_config['memory']}                                                                                                 
#SBATCH --time={slurm_config['time']}                                                                                           
#SBATCH --output=/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/Lss/logs/Lss_step3/{task}_sub_{sub}_{analysis_type}_%j.out
#SBATCH --error=/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/Lss/logs/Lss_step3/{task}_sub_{sub}_{analysis_type}_%j.err

module load apptainer
apptainer exec \\
    -B /gscratch/fang:/data \\
    -B /gscratch/scrubbed/fanglab/xiaoqian:/scrubbed_dir \\
    -B /gscratch/scrubbed/fanglab/xiaoqian/repo/hyak_narsad_0917:/app \\
    {CONTAINER_PATH} \\
    python3 /app/first_LSS_3rd_similarity.py \\
    --subject {sub} \\
    --task {task} \\
    --mask_img_path {mask_img_path} \\
    --combined_atlas_path {combined_atlas_path} \\
    --roi_names_file {roi_names_file} \\
    --analysis_type {analysis_type} \\
    --batch_size {batch_size} \\
    --n_jobs {n_jobs} \\
    {profile_flag}
"""
    
    # Create output directory
    script_dir = os.path.join(work_dir, analysis_type)
    os.makedirs(script_dir, exist_ok=True)
    
    script_path = os.path.join(script_dir, f'sub_{sub}_slurm.sh')
    
    try:
        with open(script_path, 'w') as f:
            f.write(slurm_script)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        logger.info(f"SLURM script created: {script_path}")
        return script_path
        
    except Exception as e:
        logger.error(f"Error writing SLURM script {script_path}: {e}")
        return None

# =============================================================================
# MAIN PROCESSING FUNCTIONS
# =============================================================================

def verify_paths():
    """Verify that all required paths exist."""
    logger.info("Verifying input paths...")
    
    paths_to_check = [
        (DATA_DIR, "Data directory"),
        (DERIVATIVES_DIR, "Derivatives directory"),
        (BEHAV_DIR, "Behavioral directory"),
        (COMBINED_ATLAS_PATH, "Atlas path"),
        (ROI_NAMES_FILE, "ROI names file")
    ]
    
    missing_paths = []
    for path, description in paths_to_check:
        if not os.path.exists(path):
            missing_paths.append(f"{description}: {path}")
            logger.error(f"{description} does not exist: {path}")
        else:
            logger.info(f"{description}: {path} ✓")
    
    if missing_paths:
        logger.error(f"Missing {len(missing_paths)} required paths:")
        for path in missing_paths:
            logger.error(f"  - {path}")
        return False
    
    logger.info("All required paths verified successfully")
    return True

def discover_subjects_and_tasks(layout):
    """Discover subjects and tasks from BIDS layout."""
    subjects = layout.get_subjects()
    tasks = layout.get_tasks()
    
    logger.info(f"Discovered {len(subjects)} subjects: {subjects}")
    logger.info(f"Discovered {len(tasks)} tasks: {tasks}")
    
    if not subjects:
        logger.warning("No subjects found in BIDS layout")
        return [], []
    
    if not tasks:
        logger.warning("No tasks found in BIDS layout")
        return [], []
    
    return subjects, tasks

def process_subject_task(sub, task, layout, args, work_dir):
    """
    Process a specific subject-task combination.
    
    Args:
        sub (str): Subject ID
        task (str): Task name
        layout: BIDS layout object
        args: Parsed arguments
        work_dir (str): Working directory
    
    Returns:
        int: Number of scripts created
    """
    logger.info(f"Processing sub-{sub}, task-{task}")
    
    # Query for BOLD files
    query = {
        'desc': 'preproc', 
        'suffix': 'bold', 
        'extension': ['.nii', '.nii.gz'],
        'subject': sub, 
        'task': task, 
        'space': SPACE
    }
    
    logger.info(f"Querying BOLD files for sub-{sub}, task-{task}")
    bold_files = layout.get(**query)
    
    if not bold_files:
        logger.warning(f"No BOLD files found for sub-{sub}, task-{task}")
        return 0
    
    logger.info(f"Found {len(bold_files)} BOLD files for sub-{sub}, task-{task}")
    
    # Get mask file
    try:
        mask_img_path = layout.get(
            suffix='mask', 
            return_type='file',
            extension=['.nii', '.nii.gz'],
            space=SPACE, 
            subject=sub, 
            task=task
        )[0]
        logger.info(f"Mask image path: {mask_img_path}, exists: {os.path.exists(mask_img_path)}")
    except IndexError:
        logger.warning(f"No mask file found for sub-{sub}, task-{task}")
        return 0
    
    # Create SLURM scripts for each analysis type
    scripts_created = 0
    
    for analysis_type in args.analysis_types:
        # Skip ROI analysis if atlas/ROI files are not available
        if analysis_type in ['roi', 'both']:
            if not os.path.exists(COMBINED_ATLAS_PATH):
                logger.warning(f"Skipping {analysis_type} analysis: Atlas file not found")
                continue
            if not os.path.exists(ROI_NAMES_FILE):
                logger.warning(f"Skipping {analysis_type} analysis: ROI names file not found")
                continue
        
        # Create SLURM script
        slurm_config = {
            'account': args.account,
            'partition': args.partition,
            'cpus_per_task': args.cpus_per_task,
            'memory': args.memory,
            'time': args.time
        }
        
        script_path = create_slurm_script(
            sub, task, work_dir, mask_img_path, COMBINED_ATLAS_PATH,
            ROI_NAMES_FILE, analysis_type, args.batch_size, args.n_jobs, 
            args.profile, slurm_config
        )
        
        if script_path:
            scripts_created += 1
            if args.dry_run:
                logger.info(f"[DRY RUN] Would create script: {script_path}")
            else:
                logger.info(f"Created script: {script_path}")
    
    return scripts_created

def main():
    """Main function to generate SLURM scripts."""
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    global logger
    logger = setup_logging()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info("=" * 70)
    logger.info("LSS Similarity Analysis Script Generator - Step 3")
    logger.info("=" * 70)
    logger.info(f"Analysis types: {args.analysis_types}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Parallel jobs: {args.n_jobs}")
    logger.info(f"SLURM resources: {args.cpus_per_task} CPUs, {args.memory} RAM, {args.time}")
    if args.profile:
        logger.info("Profiling enabled")
    if args.dry_run:
        logger.info("DRY RUN MODE - No files will be created")
    logger.info("=" * 70)
    
    # Verify paths
    if not verify_paths():
        logger.error("Path verification failed. Exiting.")
        return
    
    # Initialize BIDS layout
    try:
        layout = BIDSLayout(str(DATA_DIR), validate=False, derivatives=str(DERIVATIVES_DIR))
        logger.info(f"Initialized BIDSLayout with data directory: {DATA_DIR}")
    except Exception as e:
        logger.error(f"Error initializing BIDS layout: {e}")
        return
    
    # Discover subjects and tasks
    subjects, tasks = discover_subjects_and_tasks(layout)
    if not subjects or not tasks:
        logger.error("No subjects or tasks found. Exiting.")
        return
    
    # Create results directory
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        logger.info(f"Results directory: {RESULTS_DIR}")
    except Exception as e:
        logger.error(f"Error creating results directory {RESULTS_DIR}: {e}")
        return
    
    # Process each subject-task combination
    total_scripts = 0
    total_subjects = 0
    processing_errors = []
    
    for sub in subjects:
        subject_scripts = 0
        
        for task in tasks:
            try:
                # Create work directory
                work_dir = os.path.join(SCRUBBED_DIR, PROJECT_NAME, f'work_flows/Lss_step3/{task}')
                try:
                    os.makedirs(work_dir, exist_ok=True)
                    logger.info(f"Work directory: {work_dir}")
                except Exception as e:
                    logger.error(f"Error creating work directory {work_dir}: {e}")
                    continue
                
                # Process subject-task
                scripts_created = process_subject_task(sub, task, layout, args, work_dir)
                subject_scripts += scripts_created
                
            except Exception as e:
                error_msg = f"Error processing sub-{sub}, task-{task}: {e}"
                logger.error(error_msg)
                processing_errors.append(error_msg)
                continue
        
        if subject_scripts > 0:
            total_subjects += 1
            total_scripts += subject_scripts
            logger.info(f"Total scripts for sub-{sub}: {subject_scripts}")
        logger.info("")
    
    # Summary
    logger.info("=" * 70)
    logger.info("SCRIPT GENERATION SUMMARY")
    logger.info("=" * 70)
    
    if args.dry_run:
        logger.info(f"DRY RUN: Would create {total_scripts} SLURM scripts")
        logger.info(f"DRY RUN: Would process {total_subjects} subjects")
        logger.info("")
        logger.info("To actually create the scripts, run without --dry-run")
    else:
        logger.info(f"Created {total_scripts} SLURM scripts")
        logger.info(f"Processed {total_subjects} subjects")
        logger.info("")
        logger.info("Scripts are organized by task and analysis type in:")
        logger.info(f"  {SCRUBBED_DIR}/{PROJECT_NAME}/work_flows/Lss_step3/")
    
    if processing_errors:
        logger.info("")
        logger.info("Processing errors encountered:")
        for error in processing_errors:
            logger.info(f"  - {error}")
    
    logger.info("")
    logger.info("LSS Analysis Pipeline Status:")
    logger.info("  Step 1: First-level LSS analysis (individual trials)")
    logger.info("    - Scripts: create_1st_LSS_singleTrialEstimate.py")
    logger.info("    - Launcher: launch_1st_LSS_1st_singleTrialEstimate.sh")
    logger.info("")
    logger.info("  Step 2: Trial merging (4D image creation)")
    logger.info("    - Scripts: create_1st_LSS_2nd_cateAlltrials.py")
    logger.info("    - Launcher: launch_1st_LSS_2nd_cateAlltrials.sh")
    logger.info("")
    logger.info("  Step 3: Similarity analysis script generation [COMPLETED]")
    logger.info("    - Scripts: create_1st_LSS_3rd_similarity.py")
    logger.info("    - Launcher: launch_1st_LSS_3rd_similarity.sh")
    logger.info("")
    logger.info("  Step 4: Execute similarity analysis and group-level processing")
    logger.info("    - Ready to run similarity analysis")
    logger.info("")
    logger.info("Next step: Use launch_1st_LSS_3rd_similarity.sh to submit the generated scripts!")
    logger.info("=" * 70)

if __name__ == '__main__':
    main()