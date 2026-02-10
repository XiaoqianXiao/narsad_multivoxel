#!/usr/bin/env python3
"""
Script to merge LSS trial outputs into 4D NIfTI images.

This script takes the individual trial outputs from run_1st_LSS.py and merges them
into 4D NIfTI files for each subject-task combination. It's designed to work with
the LSS workflow outputs and create consolidated images for group-level analysis.

Usage:
    # Merge outputs for a specific subject and task
    python first_LSS_2nd_cateAlltrials.py --subject N101 --task phase2
    
    # Merge outputs for multiple subjects
    python first_LSS_2nd_cateAlltrials.py --subjects N101 N102 N103 --task phase2
    
    # Merge outputs for multiple tasks
    python first_LSS_2nd_cateAlltrials.py --subject N101 --tasks phase2 phase3
    
    # Merge multiple contrasts
    python first_LSS_2nd_cateAlltrials.py --subject N101 --task phase2 --contrasts 1 2 3
    
    # Merge contrast range
    python first_LSS_2nd_cateAlltrials.py --subject N101 --task phase2 --contrast-range 1 5
    
    # Merge with specific trial range
    python first_LSS_2nd_cateAlltrials.py --subject N101 --task phase2 --trial-range 1 20
    
    # Show what would be created without creating files
    python first_LSS_2nd_cateAlltrials.py --subject N101 --task phase2 --dry-run
    
    # Verbose output
    python first_LSS_2nd_cateAlltrials.py --subject N101 --task phase2 --verbose
    
    # Show help
    python first_LSS_2nd_cateAlltrials.py --help

Features:
    - Automatically discovers LSS trial outputs using BIDS layout
    - Merges multiple trials into single 4D images
    - Supports multiple subjects, tasks, and contrasts
    - Handles different file naming conventions and formats
    - Creates organized output directory structure
    - Provides detailed progress feedback and error handling
    - Validates data consistency and handles shape mismatches
    - Supports dry-run mode for testing
    - Contrast range filtering for selective processing
    - Trial range filtering for partial data processing
    - Comprehensive error reporting and validation
    - Progress tracking and detailed reporting
    - Memory-efficient processing for large datasets

Output:
    - 4D NIfTI files: sub-{subject}_task-{task}_contrast{contrast}.nii.gz
    - Organized by subject and task in all_subjects directory
    - Ready for group-level analysis
    - Maintains proper metadata and affine transformations
    - Compressed with .nii.gz format for storage efficiency

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
import re
import argparse
import numpy as np
import nibabel as nib
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

# LSS output paths
LSS_OUTPUT_DIR = os.path.join(DERIVATIVES_DIR, 'fMRI_analysis', 'LSS')
LSS_FIRST_LEVEL_DIR = os.path.join(LSS_OUTPUT_DIR, 'firstLevel')
RESULTS_DIR = os.path.join(LSS_FIRST_LEVEL_DIR, 'all_subjects')

# BIDS configuration
SPACE = 'MNI152NLin2009cAsym'

# =============================================================================
# FILE DISCOVERY AND PROCESSING
# =============================================================================

def discover_lss_outputs(layout, subject, task, contrast=1, trial_range=None):
    """
    Discover LSS output files for a specific subject-task combination.
    """
    # Query for LSS trial outputs
    # NOTE: we do NOT use 'desc' here because some pybids configs don't define it
    query = {
        'extension': ['.nii', '.nii.gz'],
        'suffix': 'bold',
        'subject': subject,
        'task': task,
        'space': SPACE,
    }

    # Get all matching files
    all_files = layout.get(**query, regex_search=True)

    if not all_files:
        print(f"    No bold files found for subject {subject}, task {task}")
        return []

    # ---- contrast filter ----
    # Your filenames look like:
    #   sub-N101_ses-pilot3mm_task-phase3_space-MNI..._desc-trial9_varcope2_bold.nii
    #
    # If you want VARCOPE images, use '_varcope{contrast}'
    # If you want COPE images, use '_cope{contrast}' and make sure those files exist.
    #contrast_pattern = f'_varcope{contrast}'  # <-- change to '_cope{contrast}' if needed
    #contrast_pattern = f'_cope{contrast}'  # <-- change to '_cope{contrast}' if needed
    contrast_pattern = f'_tstat{contrast}'

    contrast_files = [f for f in all_files if contrast_pattern in f.filename]

    if not contrast_files:
        print(f"    Warning: No files found with contrast {contrast} (pattern '{contrast_pattern}')")
        return []

    # ---- keep only trial images based on filename ----
    def extract_trial_num(file_obj):
        filename = file_obj.filename if hasattr(file_obj, 'filename') else str(file_obj)

        patterns = [
            r'desc-trial(\d+)',  # BIDS-style
            r'trial(\d+)',
            r'_trial(\d+)',
            r'trial_(\d+)',
        ]

        for pattern in patterns:
            m = re.search(pattern, filename)
            if m:
                return int(m.group(1))

        # fall back to path
        path_str = str(file_obj.path)
        for pattern in patterns:
            m = re.search(pattern, path_str)
            if m:
                return int(m.group(1))

        print(f"      Warning: Could not extract trial number from {filename}")
        return float('inf')

    # sort by trial number
    sorted_files = sorted(contrast_files, key=extract_trial_num)
    valid_files = [f for f in sorted_files if extract_trial_num(f) != float('inf')]

    if len(valid_files) != len(sorted_files):
        print(f"    Warning: {len(sorted_files) - len(valid_files)} files had invalid trial numbers")

    # optional trial range
    if trial_range:
        start_trial, end_trial = trial_range
        range_filtered = []
        for f in valid_files:
            t = extract_trial_num(f)
            if start_trial <= t <= end_trial:
                range_filtered.append(f)

        print(f"    Trial range filter: {start_trial} to {end_trial}")
        print(f"    Available trials: {len(valid_files)}, Filtered trials: {len(range_filtered)}")
        valid_files = range_filtered

    print(f"    Found {len(valid_files)} valid trial files for contrast {contrast}")
    return valid_files


def validate_and_prepare_data(files):
    """
    Validate and prepare data arrays for merging.
    
    Args:
        files (list): List of file objects to process
    
    Returns:
        tuple: (data_arrays, imgs, target_shape)
    """
    print(f"    Loading and validating {len(files)} trial files...")
    
    # Load all images
    imgs = []
    data_arrays = []
    valid_indices = []
    
    for i, file_obj in enumerate(files):
        try:
            file_path = file_obj.path if hasattr(file_obj, 'path') else str(file_obj)
            img = nib.load(file_path)
            
            # Basic validation
            if img is None or img.get_fdata() is None:
                print(f"      Warning: Invalid image data in file {i+1}")
                continue
            
            data = img.get_fdata()
            if data.size == 0:
                print(f"      Warning: Empty data in file {i+1}")
                continue
            
            imgs.append(img)
            data_arrays.append(data)
            valid_indices.append(i)
            
        except Exception as e:
            print(f"      Warning: Error loading file {i+1}: {e}")
            continue
    
    if not data_arrays:
        print("    Error: No valid data arrays found")
        return [], [], None
    
    # Check for consistent shapes
    shapes = [data.shape for data in data_arrays]
    unique_shapes = set(shapes)
    
    if len(unique_shapes) > 1:
        print(f"    Warning: Inconsistent shapes detected: {unique_shapes}")
        
        # Find most common shape
        shape_counts = {}
        for shape in shapes:
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
        
        target_shape = max(shape_counts.items(), key=lambda x: x[1])[0]
        print(f"    Using most common shape: {target_shape}")
        
        # Filter to only include files with target shape
        filtered_data = []
        filtered_imgs = []
        for i, (data, img) in enumerate(zip(data_arrays, imgs)):
            if data.shape == target_shape:
                filtered_data.append(data)
                filtered_imgs.append(img)
            else:
                print(f"      Skipping file with shape {data.shape}")
        
        data_arrays = filtered_data
        imgs = filtered_imgs
    
    else:
        target_shape = data_arrays[0].shape
    
    print(f"    Successfully loaded {len(data_arrays)} files with shape {target_shape}")
    return data_arrays, imgs, target_shape

def merge_trials_to_4d(data_arrays, imgs, target_shape, subject, task, contrast):
    """
    Merge trial data arrays into a single 4D NIfTI image.
    
    Args:
        data_arrays (list): List of 3D data arrays
        imgs (list): List of NIfTI image objects
        target_shape (tuple): Target 3D shape for the data
        subject (str): Subject ID
        task (str): Task name
        contrast (int): Contrast number
    
    Returns:
        str: Path to the created 4D file
    """
    if not data_arrays:
        print("    Error: No data arrays to merge")
        return None
    
    print(f"    Merging {len(data_arrays)} trials into 4D image...")
    
    # Stack data arrays along time dimension
    try:
        # Ensure all arrays have the same shape
        validated_arrays = []
        for i, data in enumerate(data_arrays):
            if data.shape == target_shape:
                validated_arrays.append(data)
            else:
                print(f"      Warning: Skipping array {i} with shape {data.shape}")
        
        if not validated_arrays:
            print("    Error: No valid arrays to merge")
            return None
        
        # Stack arrays
        merged_data = np.stack(validated_arrays, axis=-1)
        print(f"    Created 4D array with shape: {merged_data.shape}")
        
        # Create output filename
        output_filename = f"sub-{subject}_task-{task}_contrast{contrast}.nii.gz"
        output_path = os.path.join(RESULTS_DIR, output_filename)
        
        # Ensure output directory exists
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        # Create new NIfTI image
        if imgs and validated_arrays:
            # Use the first image as template for header
            template_img = imgs[0]
            new_img = nib.Nifti1Image(merged_data, template_img.affine, template_img.header)
            
            # Update header for 4D data
            new_img.header.set_data_shape(merged_data.shape)
            new_img.header.set_xyzt_units('mm', 'sec')
            
            # Save the image
            nib.save(new_img, output_path)
            
            # Get file size
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            print(f"    Saved: {output_filename}")
            print(f"    Final shape: {merged_data.shape}")
            print(f"    Data type: {merged_data.dtype}")
            print(f"    File size: {file_size:.2f} MB")
            
            return output_path
        else:
            print("    Error: No valid images for header template")
            return None
            
    except Exception as e:
        print(f"    Error merging trials: {e}")
        return None

def process_subject_task(layout, subject, task, contrasts, dry_run=False, trial_range=None, verbose=False):
    """
    Process a specific subject-task combination and merge trial outputs.
    
    Args:
        layout: BIDS layout object
        subject (str): Subject ID
        task (str): Task name
        contrasts (list): List of contrast numbers to process
        dry_run (bool): If True, don't create files, just show what would be done
        trial_range (tuple): Optional tuple of (start, end) trial numbers to filter
        verbose (bool): If True, show detailed progress information
    
    Returns:
        int: Number of files created
    """
    if not contrasts:
        contrasts = [1]  # Default to contrast 1
    
    files_created = 0
    
    for contrast in contrasts:
        if verbose:
            print(f"    Processing contrast {contrast}...")
        
        # Discover LSS outputs
        trial_files = discover_lss_outputs(layout, subject, task, contrast, trial_range)
        
        if not trial_files:
            print(f"    No trial files found for contrast {contrast}")
            continue
        
        if dry_run:
            print(f"    [DRY RUN] Would merge {len(trial_files)} trials for contrast {contrast}")
            files_created += 1
            continue
        
        # Validate and prepare data
        data_arrays, imgs, target_shape = validate_and_prepare_data(trial_files)
        
        if not data_arrays:
            print(f"    Skipping contrast {contrast} due to validation errors")
            continue
        
        # Merge trials to 4D
        output_path = merge_trials_to_4d(data_arrays, imgs, target_shape, subject, task, contrast)
        
        if output_path:
            files_created += 1
        else:
            print(f"    Failed to create 4D file for contrast {contrast}")
    
    return files_created

def main():
    """Main function to merge LSS trial outputs."""
    
    # =============================================================================
    # ARGUMENT PARSING
    # =============================================================================
    
    parser = argparse.ArgumentParser(
        description="Merge LSS trial outputs into 4D NIfTI images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Merge outputs for a specific subject and task
    python first_LSS_2nd_cateAlltrials.py --subject N101 --task phase2
    
    # Merge outputs for multiple subjects
    python first_LSS_2nd_cateAlltrials.py --subjects N101 N102 N103 --task phase2
    
    # Merge outputs for multiple tasks
    python first_LSS_2nd_cateAlltrials.py --subject N101 --tasks phase2 phase3
    
    # Merge multiple contrasts
    python first_LSS_2nd_cateAlltrials.py --subject N101 --task phase2 --contrasts 1 2 3
    
    # Merge contrast range
    python first_LSS_2nd_cateAlltrials.py --subject N101 --task phase2 --contrast-range 1 5
    
    # Merge with specific trial range
    python first_LSS_2nd_cateAlltrials.py --subject N101 --task phase2 --trial-range 1 20
    
    # Show what would be created without creating files
    python first_LSS_2nd_cateAlltrials.py --subject N101 --task phase2 --dry-run
    
    # Verbose output
    python first_LSS_2nd_cateAlltrials.py --subject N101 --task phase2 --verbose
        """
    )
    
    # Subject and task specification
    parser.add_argument('--subject', help='Single subject to process')
    parser.add_argument('--subjects', nargs='+', help='Multiple subjects to process')
    parser.add_argument('--task', help='Single task to process')
    parser.add_argument('--tasks', nargs='+', help='Multiple tasks to process')
    
    # Contrast specification
    parser.add_argument('--contrast', type=int, help='Single contrast number to process')
    parser.add_argument('--contrasts', nargs='+', type=int, help='Multiple contrast numbers to process')
    parser.add_argument('--contrast-range', nargs=2, type=int, metavar=('START', 'END'),
                       help='Range of contrast numbers to process (e.g., 1 5 for contrasts 1-5)')
    
    # Trial filtering
    parser.add_argument('--trial-range', nargs=2, type=int, metavar=('START', 'END'),
                       help='Range of trial numbers to process (e.g., 1 20 for trials 1-20)')
    
    # Other options
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be created without creating files')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output with detailed progress information')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.subject and not args.subjects:
        print("Error: Must specify either --subject or --subjects")
        sys.exit(1)
    
    if not args.task and not args.tasks:
        print("Error: Must specify either --task or --tasks")
        sys.exit(1)
    
    # Validate contrast range if specified
    if args.contrast_range:
        start_contrast, end_contrast = args.contrast_range
        if start_contrast > end_contrast:
            print("Error: Start contrast must be less than or equal to end contrast")
            sys.exit(1)
        if start_contrast < 1:
            print("Error: Start contrast must be 1 or greater")
            sys.exit(1)
    
    # Validate trial range if specified
    if args.trial_range:
        start_trial, end_trial = args.trial_range
        if start_trial > end_trial:
            print("Error: Start trial must be less than or equal to end trial")
            sys.exit(1)
        if start_trial < 1:
            print("Error: Start trial must be 1 or greater")
            sys.exit(1)
    
    # Determine contrasts to process
    if args.contrast:
        contrasts = [args.contrast]
    elif args.contrasts:
        contrasts = args.contrasts
    elif args.contrast_range:
        start_contrast, end_contrast = args.contrast_range
        contrasts = list(range(start_contrast, end_contrast + 1))
    else:
        contrasts = [1]  # Default to contrast 1
    
    # =============================================================================
    # INITIALIZATION
    # =============================================================================
    
    print("=" * 70)
    print("LSS Trial Output Merger - Enhanced Version")
    print("Step 2 of LSS Analysis Pipeline")
    print("=" * 70)
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be created")
        print()
    
    if args.contrast_range:
        print(f"Contrast range filter: {args.contrast_range[0]} to {args.contrast_range[1]}")
        print()
    
    if args.trial_range:
        print(f"Trial range filter: {args.trial_range[0]} to {args.trial_range[1]}")
        print()
    
    # Create BIDS layout
    try:
        layout = BIDSLayout(str(LSS_FIRST_LEVEL_DIR), validate=False)
        print(f"BIDS layout created from: {LSS_FIRST_LEVEL_DIR}")
    except Exception as e:
        print(f"Error creating BIDS layout: {e}")
        print(f"Please ensure the directory exists: {LSS_FIRST_LEVEL_DIR}")
        sys.exit(1)
    
    # Get subjects and tasks
    all_subjects = layout.get_subjects()
    all_tasks = layout.get_tasks()
    
    if not all_subjects:
        print(f"Warning: No subjects found in {LSS_FIRST_LEVEL_DIR}")
        print("This might indicate that LSS analysis hasn't been run yet.")
    
    if not all_tasks:
        print(f"Warning: No tasks found in {LSS_FIRST_LEVEL_DIR}")
        print("This might indicate that LSS analysis hasn't been run yet.")
    
    # Apply filters
    subjects_to_process = args.subjects if args.subjects else [args.subject]
    tasks_to_process = args.tasks if args.tasks else [args.task]
    
    print(f"Subjects to process: {subjects_to_process}")
    print(f"Tasks to process: {tasks_to_process}")
    print(f"Contrasts to process: {contrasts}")
    print()
    
    # =============================================================================
    # PROCESSING
    # =============================================================================
    
    total_files = 0
    total_subjects = 0
    processing_errors = []
    
    for subject in subjects_to_process:
        if subject not in all_subjects:
            print(f"Warning: Subject {subject} not found in BIDS layout, skipping")
            continue
        
        subject_files = 0
        print(f"Processing subject: {subject}")
        
        for task in tasks_to_process:
            if task not in all_tasks:
                print(f"  Warning: Task {task} not found in BIDS layout, skipping")
                continue
            
            try:
                files_created = process_subject_task(
                    layout, subject, task, contrasts, args.dry_run, 
                    args.trial_range, args.verbose
                )
                subject_files += files_created
            except Exception as e:
                error_msg = f"Error processing subject {subject}, task {task}: {e}"
                print(f"  {error_msg}")
                processing_errors.append(error_msg)
                continue
        
        if subject_files > 0:
            total_subjects += 1
            total_files += subject_files
            print(f"  Total files for {subject}: {subject_files}")
        print()
    
    # =============================================================================
    # SUMMARY
    # =============================================================================
    
    print("=" * 70)
    print("MERGE SUMMARY")
    print("=" * 70)
    
    if args.dry_run:
        print(f"DRY RUN: Would create {total_files} 4D NIfTI files")
        print(f"DRY RUN: Would process {total_subjects} subjects")
        print()
        print("To actually create the files, run without --dry-run")
    else:
        print(f"Created {total_files} 4D NIfTI files")
        print(f"Processed {total_subjects} subjects")
        print()
        print("Output files are located in:")
        print(f"  {RESULTS_DIR}")
    
    if processing_errors:
        print()
        print("Processing errors encountered:")
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
    print("The merged 4D files are ready for group-level analysis!")
    print("=" * 70)

if __name__ == '__main__':
    main()