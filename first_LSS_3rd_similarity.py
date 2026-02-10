#!/usr/bin/env python3
"""
Script to compute similarity analysis for LSS trial outputs (Step 3).

This script takes the merged 4D NIfTI images from the LSS pipeline and computes
similarity analyses using both searchlight and ROI-based approaches. It's designed
to work with the LSS workflow outputs and create similarity matrices for group-level analysis.

Usage:
    # Run both searchlight and ROI analysis
    python first_LSS_3rd_similarity.py --subject N101 --task phase2 --mask_img_path /path/to/mask.nii.gz --combined_atlas_path /path/to/atlas.nii.gz --roi_names_file /path/to/roi_names.txt
    
    # Run only searchlight analysis
    python first_LSS_3rd_similarity.py --subject N101 --task phase2 --mask_img_path /path/to/mask.nii.gz --analysis_type searchlight
    
    # Run only ROI analysis
    python first_LSS_3rd_similarity.py --subject N101 --task phase2 --mask_img_path /path/to/mask.nii.gz --combined_atlas_path /path/to/atlas.nii.gz --roi_names_file /path/to/roi_names.txt --analysis_type roi
    
    # Run with custom parameters
    python first_LSS_3rd_similarity.py --subject N101 --task phase2 --mask_img_path /path/to/mask.nii.gz --combined_atlas_path /path/to/atlas.nii.gz --roi_names_file /path/to/roi_names.txt --n_jobs 8 --batch_size 500
    
    # Enable profiling for debugging
    python first_LSS_3rd_similarity.py --subject N101 --task phase2 --mask_img_path /path/to/mask.nii.gz --combined_atlas_path /path/to/atlas.nii.gz --roi_names_file /path/to/roi_names.txt --profile
    
    # Show help
    python first_LSS_3rd_similarity.py --help

Features:
    - Automatically discovers LSS trial outputs using BIDS layout
    - Computes searchlight similarity using cubic searchlight approach
    - Computes ROI-based similarity using labeled atlas regions
    - Supports multiple analysis types (searchlight, roi, or both)
    - Handles different file naming conventions and formats
    - Creates organized output directory structure
    - Provides detailed progress feedback and error handling
    - Validates data consistency and handles shape mismatches
    - Supports profiling for performance debugging
    - Parallel processing for improved performance

Output:
    - Searchlight similarity maps: sub-{subject}_task-{task}_within-{trial_type}.nii.gz
    - ROI similarity matrices: sub-{subject}_task-{task}_within-{trial_type}.csv
    - NEW: Trial-by-trial similarity matrices:
        * searchlight: sub-{subject}_task-{task}_trial-by-trial_searchlight.csv
        * ROI: sub-{subject}_task-{task}_trial-by-trial_roi.csv
    - Organized by analysis type in similarity subdirectories
    - Ready for group-level analysis and statistical testing

Workflow Context:
    This script is Step 3 of the LSS analysis pipeline:
    
    Step 1: create_1st_LSS_singleTrialEstimate.py -> run_LSS.py (individual trial GLM)
    Step 2: create_1st_LSS_2nd_cateAlltrials.py -> first_LSS_2nd_cateAlltrials.py (merge trials)
    Step 3: first_LSS_3rd_similarity.py -> similarity analysis [CURRENT STEP]
    Step 4: Group-level analysis and statistical testing

Author: Xiaoqian Xiao (xiao.xiaoqian.320@gmail.com)
"""

# =============================================================================
# IMPORTS AND CONFIGURATION
# =============================================================================

import os
import argparse
import numpy as np
import pandas as pd
from itertools import combinations, product
from nilearn.image import load_img, index_img, new_img_like, resample_to_img
import nibabel as nib
from similarity import searchlight_similarity, roi_similarity, load_roi_names, get_roi_labels
from joblib import Parallel, delayed
import cProfile
import pstats
import time
import logging

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging():
    """Configure logging for the script."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    return logger

# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute similarity analysis for LSS trial outputs (Step 3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run both searchlight and ROI analysis
    python first_LSS_3rd_similarity.py --subject N101 --task phase2 --mask_img_path /path/to/mask.nii.gz --combined_atlas_path /path/to/atlas.nii.gz --roi_names_file /path/to/roi_names.txt
    
    # Run only searchlight analysis
    python first_LSS_3rd_similarity.py --subject N101 --task phase2 --mask_img_path /path/to/mask.nii.gz --analysis_type searchlight
    
    # Run only ROI analysis
    python first_LSS_3rd_similarity.py --subject N101 --task phase2 --mask_img_path /path/to/mask.nii.gz --combined_atlas_path /path/to/atlas.nii.gz --roi_names_file /path/to/roi_names.txt --analysis_type roi
    
    # Run with custom parameters
    python first_LSS_3rd_similarity.py --subject N101 --task phase2 --mask_img_path /path/to/mask.nii.gz --combined_atlas_path /path/to/atlas.nii.gz --roi_names_file /path/to/roi_names.txt --n_jobs 8 --batch_size 500
    
    # Enable profiling for debugging
    python first_LSS_3rd_similarity.py --subject N101 --task phase2 --mask_img_path /path/to/mask.nii.gz --combined_atlas_path /path/to/atlas.nii.gz --roi_names_file /path/to/roi_names.txt --profile
        """
    )
    
    # Required arguments
    parser.add_argument('--subject', required=True, help='Subject ID to process')
    parser.add_argument('--task', required=True, help='Task name to process')
    parser.add_argument('--mask_img_path', required=True, help='Path to brain mask image')
    
    # Conditional arguments (required for ROI analysis)
    parser.add_argument('--combined_atlas_path', help='Path to combined atlas image (required for ROI analysis)')
    parser.add_argument('--roi_names_file', help='Path to ROI names file (required for ROI analysis)')
    
    # Analysis options
    parser.add_argument('--analysis_type', choices=['searchlight', 'roi', 'both'], default='both',
                        help='Type of analysis to run: searchlight, roi, or both (default: both)')
    parser.add_argument('--batch_size', type=int, default=1000, 
                        help='Number of voxels per batch for searchlight (default: 1000)')
    parser.add_argument('--n_jobs', type=int, default=12, 
                        help='Number of parallel jobs (default: 12)')
    
    # Debugging options
    parser.add_argument('--profile', action='store_true', 
                        help='Enable cProfile for debugging')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output with detailed progress information')
    
    args = parser.parse_args()
    
    # Validate conditional arguments
    if args.analysis_type in ['roi', 'both']:
        if not args.combined_atlas_path:
            parser.error("--combined_atlas_path is required for ROI analysis")
        if not args.roi_names_file:
            parser.error("--roi_names_file is required for ROI analysis")
    
    return args

# =============================================================================
# MAIN PROCESSING FUNCTIONS
# =============================================================================

def load_and_validate_data(args, logger):
    """
    Load and validate all required data files.
    
    Args:
        args: Parsed arguments
        logger: Logger instance
    
    Returns:
        dict: Dictionary containing loaded data and validation results
    """
    logger.info(f"Loading and validating data for sub-{args.subject}, task-{args.task}")
    
    # Paths
    root_dir = os.getenv('DATA_DIR', '/data')
    project_name = 'NARSAD'
    derivatives_dir = os.path.join(root_dir, project_name, 'MRI', 'derivatives')
    behav_dir = os.path.join(root_dir, project_name, 'MRI', 'source_data', 'behav')
    data_dir = os.path.join(derivatives_dir, 'fMRI_analysis', 'LSS', 'firstLevel', 'all_subjects')
    output_dir = os.path.join(data_dir, 'similarity')
    
    # Create output directories
    try:
        if args.analysis_type in ['searchlight', 'both']:
            os.makedirs(os.path.join(output_dir, 'searchlight'), exist_ok=True)
        if args.analysis_type in ['roi', 'both']:
            os.makedirs(os.path.join(output_dir, 'roi'), exist_ok=True)
        logger.info(f"Output directories created: {output_dir}/searchlight, {output_dir}/roi")
    except Exception as e:
        logger.error(f"Error creating output directories: {e}")
        return None
    
    # Load BOLD data
    logger.info(f"Loading BOLD data from {data_dir}")
    bold_4d_path = os.path.join(data_dir, f'sub-{args.subject}_task-{args.task}.nii.gz')
    if not os.path.exists(bold_4d_path):
        # Try without .gz extension
        bold_4d_path = os.path.join(data_dir, f'sub-{args.subject}_task-{args.task}.nii')
    
    logger.info(f"BOLD path: {bold_4d_path}, exists: {os.path.exists(bold_4d_path)}")
    if not os.path.exists(bold_4d_path):
        logger.error(f"BOLD data file not found: {bold_4d_path}")
        return None
    
    try:
        bold_4d = load_img(bold_4d_path)
        logger.info(f"BOLD shape: {bold_4d.shape}, affine: {bold_4d.affine}")
        bold_data = bold_4d.get_fdata()
        nan_count = np.sum(np.isnan(bold_data))
        zero_count = np.sum(np.all(bold_data == 0, axis=3))
        logger.info(f"BOLD data NaN count: {nan_count}, all-zero voxel count: {zero_count}")
        if nan_count > 0 or zero_count > 0:
            logger.warning("BOLD data contains NaNs or all-zero voxels, which may cause computation issues")
    except Exception as e:
        logger.error(f"Error loading BOLD data {bold_4d_path}: {e}")
        return None
    
    # Load mask
    try:
        mask_img = load_img(args.mask_img_path)
        logger.info(f"Mask shape: {mask_img.shape}, affine: {mask_img.affine}")
        mask_data = mask_img.get_fdata()
        valid_voxels = np.sum(mask_data > 0)
        nan_count = np.sum(np.isnan(mask_data))
        logger.info(f"Number of voxels in mask: {valid_voxels}, NaN count: {nan_count}")
        if valid_voxels == 0:
            logger.error("Mask contains no valid voxels (all values <= 0)")
            return None
        if nan_count > 0:
            logger.warning("Mask contains NaNs, which may cause resampling issues")
    except Exception as e:
        logger.error(f"Error loading mask {args.mask_img_path}: {e}")
        return None
    
    # Load atlas and ROI names for ROI analysis
    combined_atlas = None
    combined_roi_labels = None
    roi_names = None
    
    if args.analysis_type in ['roi', 'both']:
        try:
            combined_atlas = load_img(args.combined_atlas_path)
            logger.info(f"Atlas shape: {combined_atlas.shape}, affine: {combined_atlas.affine}")
            atlas_data = combined_atlas.get_fdata()
            nan_count = np.sum(np.isnan(atlas_data))
            unique_labels = np.unique(atlas_data[atlas_data > 0])
            logger.info(f"Atlas NaN count: {nan_count}, unique positive labels: {len(unique_labels)}")
            if nan_count > 0:
                logger.warning("Atlas data contains NaNs, which may cause resampling issues")
            if len(unique_labels) == 0:
                logger.error("Atlas contains no valid ROIs (all values <= 0)")
                return None
        except Exception as e:
            logger.error(f"Error loading atlas {args.combined_atlas_path}: {e}")
            return None
        
        try:
            combined_roi_labels = get_roi_labels(combined_atlas, 'Schaefer+Tian')
            logger.info(f"ROI labels: {len(combined_roi_labels)} found")
            roi_names = load_roi_names(args.roi_names_file, combined_roi_labels)
            logger.info(f"Loaded {len(roi_names)} ROI names")
        except Exception as e:
            logger.error(f"Error processing ROI labels or names: {e}")
            return None
    
    # Load events file
    if args.subject == 'N202' and args.task == 'phase3':
        events_file = os.path.join(behav_dir, 'task-NARSAD_phase-3_sub-202_events.csv')
    else:
        events_file = os.path.join(behav_dir, f'task-Narsad_{args.task}_events.csv')
    
    logger.info(f"Events file: {events_file}, exists: {os.path.exists(events_file)}")
    try:
        from utils import read_csv_with_detection
        events = read_csv_with_detection(events_file)
        logger.info(f"Events loaded, shape: {events.shape}, columns: {events.columns}")
    except Exception as e:
        logger.error(f"Error loading events file {events_file}: {e}")
        return None
    
    # Validate events data
    if 'trial_type' not in events.columns:
        logger.error("Events file does not contain 'trial_type' column")
        return None
    
    trial_types = events['trial_type'].unique()
    logger.info(f"Trial types: {trial_types}")
    
    # Validate trial indices against BOLD data
    n_trials = bold_4d.shape[3]
    logger.info(f"Number of trials in BOLD data: {n_trials}")
    
    trial_to_type = {i: tt for i, tt in enumerate(events['trial_type'].values) if i < n_trials}
    if len(trial_to_type) < len(events):
        logger.warning(
            f"Event file has {len(events)} trials, but BOLD data has only {n_trials}. Truncating to {n_trials} trials.")
    
    type_to_indices = {t: [i for i, tt in trial_to_type.items() if tt == t] for t in trial_types}
    logger.info(f"Type to indices: {type_to_indices}")
    
    for ttype, indices in type_to_indices.items():
        if not indices:
            logger.warning(f"No valid trial indices for trial type {ttype}. Skipping.")
            type_to_indices[ttype] = []
    
    return {
        'bold_4d': bold_4d,
        'mask_img': mask_img,
        'mask_data': mask_data,
        'combined_atlas': combined_atlas,
        'combined_roi_labels': combined_roi_labels,
        'roi_names': roi_names,
        'events': events,
        'trial_types': trial_types,
        'type_to_indices': type_to_indices,
        'n_trials': n_trials,
        'output_dir': output_dir,
        'data_dir': data_dir
    }

def run_searchlight_analysis(data_dict, args, logger):
    """
    Run searchlight similarity analysis.
    
    Args:
        data_dict: Dictionary containing loaded data
        args: Parsed arguments
        logger: Logger instance
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Starting searchlight similarity analysis")
    
    try:
        # Compute all pair similarities
        all_pairs = list(combinations(range(data_dict['n_trials']), 2))
        logger.info(f"Computing searchlight similarity for {len(all_pairs)} total pairs")
        
        start_time = time.time()
        pair_results = searchlight_similarity(
            data_dict['bold_4d'], 
            data_dict['mask_img'], 
            radius=6, 
            trial_pairs=all_pairs,
            similarity='pearson', 
            n_jobs=args.n_jobs, 
            batch_size=args.batch_size
        )
        elapsed = time.time() - start_time
        logger.info(f"Searchlight similarity for all pairs completed in {elapsed:.2f} seconds")
        
        # Organize results
        pair_sims = {(i, j): sim.get_fdata() for i, j, sim in pair_results if sim is not None}
        logger.info(f"Valid similarity maps: {len(pair_sims)} out of {len(all_pairs)}")
        
        # Initialize output maps
        output_maps = {f"within-{ttype}": np.zeros(data_dict['mask_data'].shape, dtype=np.float32) 
                       for ttype in data_dict['trial_types']}
        for t1, t2 in combinations(data_dict['trial_types'], 2):
            output_maps[f"between-{t1}-{t2}"] = np.zeros(data_dict['mask_data'].shape, dtype=np.float32)
        
        # Within-type similarity
        for ttype in data_dict['trial_types']:
            indices = data_dict['type_to_indices'][ttype]
            pairs = list(combinations(indices, 2))
            if not pairs:
                logger.warning(f"No pairs for within-type {ttype}")
                continue
            
            sim_maps = [pair_sims.get((i, j)) for i, j in pairs if (i, j) in pair_sims]
            if sim_maps:
                avg_map = np.nanmean(np.stack(sim_maps, axis=0), axis=0)
                output_maps[f"within-{ttype}"] = avg_map
                output_img = new_img_like(data_dict['mask_img'], avg_map)
                output_path = os.path.join(data_dict['output_dir'], 'searchlight', 
                                         f"sub-{args.subject}_task-{args.task}_within-{ttype}.nii.gz")
                logger.info(f"Saving searchlight to {output_path}")
                try:
                    nib.save(output_img, output_path)
                    logger.info(f"Saved searchlight for {ttype}")
                except Exception as e:
                    logger.error(f"Error saving searchlight {output_path}: {e}")
            else:
                logger.warning(f"No valid searchlight maps for {ttype}")
        
        # Between-type similarity
        for t1, t2 in combinations(data_dict['trial_types'], 2):
            pairs = list(product(data_dict['type_to_indices'][t1], data_dict['type_to_indices'][t2]))
            if not pairs:
                logger.warning(f"No pairs for between-type {t1}-{t2}")
                continue
            
            sim_maps = [pair_sims.get((min(i, j), max(i, j))) 
                       for i, j in pairs if (min(i, j), max(i, j)) in pair_sims]
            if sim_maps:
                avg_map = np.nanmean(np.stack(sim_maps, axis=0), axis=0)
                output_maps[f"between-{t1}-{t2}"] = avg_map
                output_img = new_img_like(data_dict['mask_img'], avg_map)
                output_path = os.path.join(data_dict['output_dir'], 'searchlight', 
                                         f"sub-{args.subject}_task-{args.task}_between-{t1}-{t2}.nii.gz")
                logger.info(f"Saving searchlight to {output_path}")
                try:
                    nib.save(output_img, output_path)
                    logger.info(f"Saved searchlight for {t1}-{t2}")
                except Exception as e:
                    logger.error(f"Error saving searchlight {output_path}: {e}")
            else:
                logger.warning(f"No valid searchlight maps for {t1}-{t2}")
        
        logger.info("Searchlight analysis completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in searchlight analysis: {e}")
        return False

def run_roi_analysis(data_dict, args, logger):
    """
    Run ROI-based similarity analysis.
    
    Args:
        data_dict: Dictionary containing loaded data
        args: Parsed arguments
        logger: Logger instance
    
    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("Starting ROI-based similarity analysis")
    
    try:
        # Ensure atlas is aligned with BOLD data
        combined_atlas_aligned = data_dict['combined_atlas']
        if not np.allclose(data_dict['bold_4d'].affine, data_dict['combined_atlas'].affine) or \
           data_dict['bold_4d'].shape[:3] != data_dict['combined_atlas'].shape:
            logger.info("Resampling atlas to match BOLD data space")
            combined_atlas_aligned = resample_to_img(data_dict['combined_atlas'], 
                                                   data_dict['mask_img'], 
                                                   interpolation='nearest')
            logger.info(f"Resampled atlas shape: {combined_atlas_aligned.shape}")
        
        # Compute all pair similarities
        all_pairs = list(combinations(range(data_dict['n_trials']), 2))
        logger.info(f"Computing ROI similarity for {len(all_pairs)} total pairs")
        
        start_time = time.time()
        pair_results = roi_similarity(
            data_dict['bold_4d'], 
            combined_atlas_aligned, 
            data_dict['combined_roi_labels'], 
            trial_pairs=all_pairs,
            similarity='pearson', 
            n_jobs=args.n_jobs
        )
        elapsed = time.time() - start_time
        logger.info(f"ROI similarity for all pairs completed in {elapsed:.2f} seconds")
        
        pair_sims = {(i, j): sim for i, j, sim in pair_results if sim is not None}

        # ------------------------------------------------------------
        # NEW: trial-by-trial ROI similarity matrix (scalar per pair)
        # ------------------------------------------------------------
        n_trials = data_dict['n_trials']
        trial_sim_matrix = np.full((n_trials, n_trials), np.nan, dtype=np.float32)

        for (i, j), sim_matrix in pair_sims.items():
            if sim_matrix is not None:
                # Collapse ROI×ROI matrix to a single scalar similarity
                s = np.nanmean(sim_matrix)
                trial_sim_matrix[i, j] = s
                trial_sim_matrix[j, i] = s  # symmetry

        trial_labels = [f"trial_{k}" for k in range(n_trials)]
        trial_sim_df = pd.DataFrame(
            trial_sim_matrix,
            index=trial_labels,
            columns=trial_labels
        )

        trial_sim_path = os.path.join(
            data_dict['output_dir'],
            'roi',
            f"sub-{args.subject}_task-{args.task}_trial-by-trial_roi.csv"
        )
        logger.info(f"Saving trial-by-trial ROI similarity matrix to {trial_sim_path}")
        try:
            trial_sim_df.to_csv(trial_sim_path, index=True, index_label='trial')
            logger.info("Saved trial-by-trial ROI similarity matrix")
        except Exception as e:
            logger.error(f"Error saving trial-by-trial ROI matrix {trial_sim_path}: {e}")
        # ------------------------------------------------------------
        
        # Initialize DataFrames for averaged ROI×ROI similarities
        columns = [data_dict['roi_names'][label] for label in data_dict['combined_roi_labels']]
        index = [data_dict['roi_names'][label] for label in data_dict['combined_roi_labels']]
        roi_dfs = {f"within-{ttype}": pd.DataFrame(index=index, columns=columns, dtype=np.float32) 
                   for ttype in data_dict['trial_types']}
        for t1, t2 in combinations(data_dict['trial_types'], 2):
            roi_dfs[f"between-{t1}-{t2}"] = pd.DataFrame(index=index, columns=columns, dtype=np.float32)
        
        # Within-type similarity
        for ttype in data_dict['trial_types']:
            indices = data_dict['type_to_indices'][ttype]
            pairs = list(combinations(indices, 2))
            if not pairs:
                logger.warning(f"No pairs for within-type {ttype}")
                continue
            
            sim_matrices = [pair_sims.get((i, j)) for i, j in pairs if (i, j) in pair_sims]
            if sim_matrices:
                avg_sim_matrix = np.nanmean(np.stack(sim_matrices, axis=0), axis=0)
                df = roi_dfs[f"within-{ttype}"]
                for i in range(len(data_dict['combined_roi_labels'])):
                    for j in range(len(data_dict['combined_roi_labels'])):
                        df.iloc[i, j] = avg_sim_matrix[i, j]
            else:
                logger.warning(f"No valid ROI matrices for {ttype}")
        
        # Between-type similarity
        for t1, t2 in combinations(data_dict['trial_types'], 2):
            pairs = list(product(data_dict['type_to_indices'][t1], data_dict['type_to_indices'][t2]))
            if not pairs:
                logger.warning(f"No pairs for between-type {t1}-{t2}")
                continue
            
            sim_matrices = [pair_sims.get((min(i, j), max(i, j))) 
                           for i, j in pairs if (min(i, j), max(i, j)) in pair_sims]
            if sim_matrices:
                avg_sim_matrix = np.nanmean(np.stack(sim_matrices, axis=0), axis=0)
                df = roi_dfs[f"between-{t1}-{t2}"]
                for i in range(len(data_dict['combined_roi_labels'])):
                    for j in range(len(data_dict['combined_roi_labels'])):
                        df.iloc[i, j] = avg_sim_matrix[i, j]
            else:
                logger.warning(f"No valid ROI matrices for {t1}-{t2}")
        
        # Save ROI DataFrames
        for df_name, df in roi_dfs.items():
            output_path = os.path.join(data_dict['output_dir'], 'roi', 
                                     f"sub-{args.subject}_task-{args.task}_{df_name}.csv")
            logger.info(f"Saving ROI to {output_path}")
            try:
                df.to_csv(output_path, index=True, index_label='ROI1')
                logger.info(f"Saved {df_name} ROI similarities")
            except Exception as e:
                logger.error(f"Error saving ROI {output_path}: {e}")
        
        logger.info("ROI analysis completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in ROI analysis: {e}")
        return False

def main(args):
    """Main function to run similarity analysis."""
    
    # Setup logging
    logger = setup_logging()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info("=" * 70)
    logger.info("LSS Similarity Analysis - Step 3 of LSS Pipeline")
    logger.info("=" * 70)
    logger.info(f"Subject: {args.subject}")
    logger.info(f"Task: {args.task}")
    logger.info(f"Analysis type: {args.analysis_type}")
    logger.info(f"Mask path: {args.mask_img_path}")
    if args.analysis_type in ['roi', 'both']:
        logger.info(f"Atlas path: {args.combined_atlas_path}")
        logger.info(f"ROI names file: {args.roi_names_file}")
    logger.info(f"Parallel jobs: {args.n_jobs}")
    if args.analysis_type in ['searchlight', 'both']:
        logger.info(f"Batch size: {args.batch_size}")
    logger.info("=" * 70)
    
    # Load and validate data
    data_dict = load_and_validate_data(args, logger)
    if data_dict is None:
        logger.error("Failed to load or validate data. Exiting.")
        return
    
    # Run analysis based on type
    success = True
    
    if args.analysis_type in ['searchlight', 'both']:
        logger.info("Running searchlight analysis...")
        if not run_searchlight_analysis(data_dict, args, logger):
            success = False
    
    if args.analysis_type in ['roi', 'both']:
        logger.info("Running ROI analysis...")
        if not run_roi_analysis(data_dict, args, logger):
            success = False
    
    # Summary
    logger.info("=" * 70)
    logger.info("ANALYSIS SUMMARY")
    logger.info("=" * 70)
    
    if success:
        logger.info("All analyses completed successfully!")
        logger.info(f"Output directory: {data_dict['output_dir']}")
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
        logger.info("  Step 3: Similarity analysis [COMPLETED]")
        logger.info("    - Scripts: first_LSS_3rd_similarity.py")
        logger.info("    - Launcher: launch_1st_LSS_3rd_similarity.sh")
        logger.info("")
        logger.info("  Step 4: Group-level analysis and statistical testing")
        logger.info("    - Ready for group-level analysis")
        logger.info("")
        logger.info("The similarity analysis is complete and ready for group-level analysis!")
    else:
        logger.error("Some analyses failed. Check the logs above for details.")
    
    logger.info("=" * 70)

if __name__ == '__main__':
    args = parse_arguments()
    if args.profile:
        logger = setup_logging()
        logger.info("Running with cProfile")
        profiler = cProfile.Profile()
        profiler.enable()
        main(args)
        profiler.disable()
        
        # Save profiling stats
        output_path = os.path.join(
            os.getenv('DATA_DIR', '/data'), 'NARSAD', 'MRI', 'derivatives',
            'fMRI_analysis', 'LSS', 'firstLevel', 'all_subjects', 'similarity', 'searchlight',
            f"sub-{args.subject}_task-{args.task}_profile.prof"
        )
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            profiler.dump_stats(output_path)
            stats = pstats.Stats(output_path)
            stats.sort_stats('cumulative').print_stats(20)
            logger.info(f"Profiling stats saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving profiling stats to {output_path}: {e}")
    else:
        main(args)
