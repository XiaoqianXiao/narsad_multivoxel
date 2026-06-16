#!/usr/bin/env python3
"""
Script to generate SLURM scripts for LSS trial merging (Step 2).

This script generates SLURM scripts that call first_LSS_2_cateAlltrials.py to merge
individual LSS trial outputs into 4D NIfTI images. It's designed to work with the
LSS workflow outputs and create batch processing scripts for trial merging.

Usage:
    # Generate SLURM scripts for all subjects and tasks
    python create_1st_LSS_2nd_cateAlltrials.py
    
    # Generate scripts for specific subjects only
    python create_1st_LSS_2nd_cateAlltrials.py --subjects N101 N102 N103
    
    # Generate scripts for specific tasks only
    python create_1st_LSS_2nd_cateAlltrials.py --tasks phase2 phase3
    
    # Generate scripts with custom SLURM settings
    python create_1st_LSS_2nd_cateAlltrials.py --account fang --partition ckpt-all --memory 64G
    
    # Generate scripts for specific contrasts
    python create_1st_LSS_2nd_cateAlltrials.py --contrasts 1 2 3
    
    # Generate scripts for specific contrast ranges
    python create_1st_LSS_2nd_cateAlltrials.py --contrast-range 1 5
    
    # Show what would be created without creating files
    python create_1st_LSS_2nd_cateAlltrials.py --dry-run
    
    # Verbose output
    python create_1st_LSS_2nd_cateAlltrials.py --verbose
    
    # Show help
    python create_1st_LSS_2nd_cateAlltrials.py --help

Features:
    - Auto-detects subjects and tasks from BIDS layout
    - Generates SLURM scripts for trial merging
    - Calls first_LSS_2_cateAlltrials.py with proper arguments
    - Configurable SLURM resource requirements
    - Supports multiple subjects, tasks, and contrasts
    - Creates organized directory structure
    - Handles special case files and configurations
    - Contrast range filtering for selective processing
    - Comprehensive error handling and validation
    - Progress tracking and detailed reporting
    - Trial count validation and reporting

Output:
    - SLURM scripts in /gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/LSS_step2/{task}/
    - Script naming: sub_{subject}_slurm.sh
    - Each script runs first_LSS_2_cateAlltrials.py with appropriate parameters
    - Scripts are automatically made executable

Workflow Context:
    This script is Step 2 of the LSS analysis pipeline:
    
    Step 1: create_1st_LSS_singleTrialEstimate.py -> run_LSS.py (individual trial GLM)
    Step 2: create_1st_LSS_2nd_cateAlltrials.py -> first_LSS_2nd_cateAlltrials.py (merge trials) [CURRENT STEP]
    Step 3: Similarity analysis and group-level processing

Author: Xiaoqian Xiao (xiao.xiaoqian.320@gmail.com)
"""

# =============================================================================
# IMPORTS AND CONFIGURATION
# =============================================================================

import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from bids.layout import BIDSLayout
from nipype import config, logging

# Set FSL environment
os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
os.environ['FSLDIR'] = '/usr/local/fsl'
os.environ['PATH'] += os.pathsep + os.path.join(os.environ['FSLDIR'], 'bin')

# Nipype configuration
config.set('execution', 'remove_unnecessary_outputs', 'false')
logging.update_logging(config)

# =============================================================================
# PATHS AND DIRECTORIES
# =============================================================================

# Base paths
ROOT_DIR = os.getenv('DATA_DIR', '/data')
PROJECT_NAME = 'NARSAD'
DATA_DIR = os.path.join(ROOT_DIR, PROJECT_NAME, 'MRI')
DERIVATIVES_DIR = os.path.join(DATA_DIR, 'derivatives')
BEHAV_DIR = os.path.join(DATA_DIR, 'source_data', 'behav')

# Container and workflow paths
SCRUBBED_DIR = '/scrubbed_dir'
CONTAINER_PATH = "/gscratch/scrubbed/fanglab/xiaoqian/images/narsad.sif"

# BIDS configuration
SPACE = ['MNI152NLin2009cAsym']

# =============================================================================
# SLURM SCRIPT GENERATION
# =============================================================================

def create_slurm_script(subject, task, work_dir, slurm_config, contrasts=None):
    """
    Create a SLURM script for LSS trial merging.
    
    Args:
        subject (str): Subject ID
        task (str): Task name
        work_dir (str): Working directory for the script
        slurm_config (dict): SLURM configuration parameters
        contrasts (list): List of contrast numbers to process
    
    Returns:
        str: Path to the created SLURM script
    """
    # Build contrast argument string
    contrast_args = ""
    if contrasts and len(contrasts) > 1:
        contrast_args = " --contrasts " + " ".join(map(str, contrasts))
    elif contrasts and len(contrasts) == 1:
        contrast_args = f" --contrast {contrasts[0]}"
    
    # SLURM script template
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name=LSS_merge_{subject}_{task}
#SBATCH --account={slurm_config['account']}
#SBATCH --partition={slurm_config['partition']}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={slurm_config['cpus_per_task']}
#SBATCH --mem={slurm_config['memory']}
#SBATCH --time={slurm_config['time']}
#SBATCH --output=/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/LSS_step2/logs/{task}/sub_{subject}_%j.out
#SBATCH --error=/gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/LSS_step2/logs/{task}/sub_{subject}_%j.err

# Load required modules
module load apptainer

# Run LSS trial merging using the container
apptainer exec -B /gscratch/fang:/data -B /gscratch/scrubbed/fanglab/xiaoqian:/scrubbed_dir -B /gscratch/scrubbed/fanglab/xiaoqian/repo/hyak_narsad_0917:/app {CONTAINER_PATH} \\
    python3 /app/first_LSS_2nd_cateAlltrials.py \\
    --subject {subject} \\
    --task {task}{contrast_args}

echo "LSS trial merging completed for subject {subject}, task {task}"
"""
    
    # Create script file
    script_filename = f'sub_{subject}_slurm.sh'
    script_path = os.path.join(work_dir, script_filename)
    
    with open(script_path, 'w') as f:
        f.write(slurm_script)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    
    return script_path

def get_events_file_path(subject, task):
    """
    Get the path to the events file for a specific subject and task.
    
    Args:
        subject (str): Subject ID
        task (str): Task name
    
    Returns:
        str: Path to the events file
    """
    # Handle special case for N202 phase3
    if subject == 'N202' and task == 'phase3':
        events_file = os.path.join(BEHAV_DIR, 'single_trial_task-NARSAD_phase-3_sub-202_half_events.csv')
    else:
        # Standard naming convention
        events_file = os.path.join(BEHAV_DIR, f'single_trial_task-Narsad_{task}_half_events.csv')
    
    return events_file

def validate_contrast_range(contrasts, contrast_range):
    """
    Validate and filter contrasts based on the specified range.
    
    Args:
        contrasts (list): List of all available contrast numbers
        contrast_range (tuple): Tuple of (start, end) contrast numbers
    
    Returns:
        list: Filtered list of contrasts within the specified range
    """
    if not contrast_range:
        return contrasts
    
    start_contrast, end_contrast = contrast_range
    filtered_contrasts = [c for c in contrasts if start_contrast <= c <= end_contrast]
    
    print(f"    Contrast range filter: {start_contrast} to {end_contrast}")
    print(f"    Available contrasts: {len(contrasts)}, Filtered contrasts: {len(filtered_contrasts)}")
    
    return filtered_contrasts

def process_subject_task(layout, subject, task, slurm_config, contrasts=None, contrast_range=None, dry_run=False, verbose=False):
    """
    Process a specific subject-task combination and generate SLURM script.
    
    Args:
        layout: BIDS layout object
        subject (str): Subject ID
        task (str): Task name
        slurm_config (dict): SLURM configuration
        contrasts (list): List of contrast numbers
        contrast_range (tuple): Optional tuple of (start, end) contrast numbers to filter
        dry_run (bool): If True, don't create files, just show what would be done
        verbose (bool): If True, show detailed progress information
    
    Returns:
        int: Number of scripts created
    """
    # Query for BOLD files
    query = {
        'desc': 'preproc', 
        'suffix': 'bold', 
        'extension': ['.nii', '.nii.gz'],
        'subject': subject, 
        'task': task, 
        'space': SPACE[0]
    }
    
    bold_files = layout.get(**query)
    if not bold_files:
        print(f"  No BOLD files found for sub-{subject}, task-{task}")
        return 0
    
    # Get events file path
    events_file = get_events_file_path(subject, task)
    
    # Check if events file exists
    if not os.path.exists(events_file):
        print(f"  Events file not found: {events_file}")
        return 0
    
    try:
        # Read events file to get trial information with automatic separator detection
        from utils import read_csv_with_detection
        events_df = read_csv_with_detection(events_file)
        
        # Check if trial_ID column exists
        if 'trial_ID' not in events_df.columns:
            print(f"  Error: 'trial_ID' column not found in events file: {events_file}")
            print(f"  Available columns: {list(events_df.columns)}")
            return 0
        
        trial_ids = sorted(events_df['trial_ID'].unique())
        print(f"  Found {len(trial_ids)} trials in events file")
        
        # Apply contrast range filter if specified
        if contrast_range and contrasts:
            contrasts = validate_contrast_range(contrasts, contrast_range)
            if not contrasts:
                print(f"  No contrasts found within specified range")
                return 0
        
        if dry_run:
            contrast_info = f" for contrasts {contrasts}" if contrasts else ""
            print(f"  [DRY RUN] Would create SLURM script{contrast_info}")
            return 1
        
        # Create work directory
        work_dir = os.path.join(SCRUBBED_DIR, PROJECT_NAME, f'work_flows/LSS_step2/{task}')
        os.makedirs(work_dir, exist_ok=True)
        
        # Generate SLURM script
        script_path = create_slurm_script(subject, task, work_dir, slurm_config, contrasts)
        print(f"    Created: {os.path.basename(script_path)}")
        
        if verbose and contrasts:
            print(f"    Processing contrasts: {contrasts}")
        
        return 1
        
    except Exception as e:
        print(f"  Error processing events file: {e}")
        return 0

def main():
    """Main function to generate LSS trial merging SLURM scripts."""
    
    # =============================================================================
    # ARGUMENT PARSING
    # =============================================================================
    
    parser = argparse.ArgumentParser(
        description="Generate SLURM scripts for LSS trial merging (Step 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate scripts for all subjects and tasks
    python create_1st_LSS_2nd_cateAlltrials.py
    
    # Generate scripts for specific subjects only
    python create_1st_LSS_2nd_cateAlltrials.py --subjects N101 N102 N103
    
    # Generate scripts for specific tasks only
    python create_1st_LSS_2nd_cateAlltrials.py --tasks phase2 phase3
    
    # Generate scripts with custom SLURM settings
    python create_1st_LSS_2nd_cateAlltrials.py --account fang --partition ckpt-all --memory 64G
    
    # Generate scripts for specific contrasts
    python create_1st_LSS_2nd_cateAlltrials.py --contrasts 1 2 3
    
    # Generate scripts for contrast range
    python create_1st_LSS_2nd_cateAlltrials.py --contrast-range 1 5
    
    # Generate scripts for specific subjects and contrast range
    python create_1st_LSS_2nd_cateAlltrials.py --subjects N101 N102 --contrast-range 1 3
    
    # Show what would be created without creating files
    python create_1st_LSS_2nd_cateAlltrials.py --dry-run
    
    # Verbose output
    python create_1st_LSS_2nd_cateAlltrials.py --verbose
        """
    )
    
    # Subject and task filtering
    parser.add_argument('--subjects', nargs='+', 
                       help='Specific subjects to process (default: all)')
    parser.add_argument('--tasks', nargs='+', 
                       help='Specific tasks to process (default: all)')
    
    # Contrast specification
    parser.add_argument('--contrasts', nargs='+', type=int,
                       help='Specific contrast numbers to process (default: all available)')
    parser.add_argument('--contrast-range', nargs=2, type=int, metavar=('START', 'END'),
                       help='Range of contrast numbers to process (e.g., 1 5 for contrasts 1-5)')
    
    # SLURM configuration
    parser.add_argument('--account', default='fang',
                       help='SLURM account (default: fang)')
    parser.add_argument('--partition', default='ckpt-all',
                       help='SLURM partition (default: ckpt-all)')
    parser.add_argument('--cpus-per-task', type=int, default=4,
                       help='CPUs per task (default: 4)')
    parser.add_argument('--memory', default='40G',
                       help='Memory requirement (default: 40G)')
    parser.add_argument('--time', default='2:00:00',
                       help='Time limit (default: 2:00:00)')
    
    # Other options
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be created without creating files')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output with detailed progress information')
    
    args = parser.parse_args()
    
    # Validate contrast range if specified
    if args.contrast_range:
        start_contrast, end_contrast = args.contrast_range
        if start_contrast > end_contrast:
            print("Error: Start contrast must be less than or equal to end contrast")
            sys.exit(1)
        if start_contrast < 1:
            print("Error: Start contrast must be 1 or greater")
            sys.exit(1)
    
    # =============================================================================
    # INITIALIZATION
    # =============================================================================
    
    print("=" * 60)
    print("LSS Trial Merging SLURM Script Generator (Step 2)")
    print("=" * 60)
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be created")
        print()
    
    if args.contrast_range:
        print(f"Contrast range filter: {args.contrast_range[0]} to {args.contrast_range[1]}")
        print()
    
    # Create BIDS layout
    try:
        layout = BIDSLayout(str(DATA_DIR), validate=False, derivatives=str(DERIVATIVES_DIR))
        print(f"BIDS layout created from: {DATA_DIR}")
    except Exception as e:
        print(f"Error creating BIDS layout: {e}")
        sys.exit(1)
    
    # Get subjects and tasks
    all_subjects = layout.get_subjects()
    all_tasks = layout.get_tasks()
    
    # Apply filters
    subjects_to_process = args.subjects if args.subjects else all_subjects
    tasks_to_process = args.tasks if args.tasks else all_tasks
    
    print(f"Subjects to process: {len(subjects_to_process)}")
    print(f"Tasks to process: {len(tasks_to_process)}")
    if args.contrasts:
        print(f"Contrasts to process: {args.contrasts}")
    print()
    
    # SLURM configuration
    slurm_config = {
        'account': args.account,
        'partition': args.partition,
        'cpus_per_task': args.cpus_per_task,
        'memory': args.memory,
        'time': args.time
    }
    
    print("SLURM Configuration:")
    for key, value in slurm_config.items():
        print(f"  {key}: {value}")
    print()
    
    # =============================================================================
    # SCRIPT GENERATION
    # =============================================================================
    
    total_scripts = 0
    total_subjects = 0
    processing_errors = []
    
    for subject in subjects_to_process:
        if subject not in all_subjects:
            print(f"Warning: Subject {subject} not found in BIDS layout, skipping")
            continue
        
        subject_scripts = 0
        print(f"Processing subject: {subject}")
        
        for task in tasks_to_process:
            if task not in all_tasks:
                print(f"  Warning: Task {task} not found in BIDS layout, skipping")
                continue
            
            print(f"  Task: {task}")
            try:
                scripts_created = process_subject_task(
                    layout, subject, task, slurm_config, args.contrasts, 
                    args.contrast_range, args.dry_run, args.verbose
                )
                subject_scripts += scripts_created
            except Exception as e:
                error_msg = f"Error processing {subject}, {task}: {e}"
                print(f"  {error_msg}")
                processing_errors.append(error_msg)
        
        if subject_scripts > 0:
            total_subjects += 1
            total_scripts += subject_scripts
            print(f"  Total scripts for {subject}: {subject_scripts}")
        print()
    
    # =============================================================================
    # SUMMARY
    # =============================================================================
    
    print("=" * 60)
    print("SCRIPT GENERATION SUMMARY")
    print("=" * 60)
    
    if args.dry_run:
        print(f"DRY RUN: Would create {total_scripts} SLURM scripts")
        print(f"DRY RUN: Would process {total_subjects} subjects")
        print()
        print("To actually create the scripts, run without --dry-run")
    else:
        print(f"Created {total_scripts} SLURM scripts")
        print(f"Processed {total_subjects} subjects")
        print()
        print("Scripts are located in:")
        for task in tasks_to_process:
            work_dir = os.path.join(SCRUBBED_DIR, PROJECT_NAME, f'work_flows/LSS_step2//{task}')
            print(f"  {task}: {work_dir}")
    
    if processing_errors:
        print()
        print("Processing Errors:")
        for error in processing_errors:
            print(f"  - {error}")
    
    print()
    print("LSS Analysis Pipeline Status:")
    print("  Step 1: First-level LSS analysis (individual trials)")
    print("    - Scripts: create_1st_LSS_singleTrialEstimate.py")
    print("    - Launcher: launch_1st_LSS_1st_singleTrialEstimate.sh")
    print("")
    print("  Step 2: Trial merging (4D image creation) [COMPLETED]")
    print("    - Scripts: create_1st_LSS_2nd_cateAlltrials.py")
    print("    - Launcher: launch_1st_LSS_2nd_cateAlltrials.sh")
    print("")
    print("  Step 3: Similarity analysis and group processing")
    print("    - Scripts: Various similarity analysis scripts")
    print("    - Launcher: launch_group_LSS.sh")
    print("")
    print("To submit all jobs, you can use:")
    print("  for script in /gscratch/scrubbed/fanglab/xiaoqian/NARSAD/work_flows/LSS_step2//*/*.sh; do sbatch \"$script\"; done")
    print("")
    print("Or use the launch script:")
    print("  ./launch_1st_LSS_2nd_cateAlltrials.sh")
    print("")
    print("Next steps:")
    print("  1. Submit the generated SLURM scripts")
    print("  2. Wait for all trial merging jobs to complete")
    print("  3. Verify 4D images were created successfully")
    print("  4. Proceed to Step 3 (similarity analysis)")
    print("=" * 60)

if __name__ == '__main__':
    main()
