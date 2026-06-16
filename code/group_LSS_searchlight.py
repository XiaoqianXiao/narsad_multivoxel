"""
Group-Level LSS Searchlight Analysis Script

This script performs group-level statistical analysis on LSS (Least Squares Separate) 
searchlight similarity maps using either FLAMEO (parametric) or Randomise (non-parametric) methods.

Purpose:
    - Takes individual subject similarity maps from LSS analysis
    - Performs group-level statistical testing (patients vs controls)
    - Generates group-level statistical maps and cluster results

Workflow Context:
    This script is part of the LSS analysis pipeline:
    
    Step 1: First-level LSS analysis (individual trial GLM)
    Step 2: Trial merging (4D image creation)  
    Step 3: Similarity analysis (searchlight/ROI)
    Step 4: Group-level analysis [THIS SCRIPT] - Statistical testing across subjects

Prerequisites:
    - LSS similarity analysis must be completed (Step 3)
    - Similarity maps must exist in: derivatives/fMRI_analysis/LSS/firstLevel/all_subjects/similarity/searchlight/
    - Subject IDs must start with '1' (patients) or '2' (controls)
    - FSL must be available for FLAMEO/Randomise

Input Data:
    - Similarity maps: sub-{subject}_task-{task}_{map_type}.nii.gz
    - Variance maps (for FLAMEO): sub-{subject}_task-{task}_{map_type}_var.nii.gz
    - Brain mask from fmriprep derivatives

Output:
    - Group-level statistical maps (z-stats, t-stats)
    - Cluster results (if clustering enabled)
    - Design matrices and contrast files
    - Results saved to: derivatives/fMRI_analysis/LSS/groupLevel/searchlight/{method}/{task}/

Usage Examples:
    # Run FLAMEO analysis for within-FIXATION maps
    python group_LSS_searchlight.py --map_type within-FIXATION --method flameo
    
    # Run Randomise analysis for between-FIXATION-CS maps  
    python group_LSS_searchlight.py --map_type between-FIXATION-CS --method randomise
    
    # Default is FLAMEO method
    python group_LSS_searchlight.py --map_type within-FIXATION

Author: Xiaoqian Xiao (xiao.xiaoqian.320@gmail.com)
Date: 2024
"""

import os
import argparse
import numpy as np
import pandas as pd
from nipype.pipeline.engine import Workflow, Node
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.io import DataSink
import logging
from glob import glob

# Import workflows from group_level_workflows.py
from group_level_workflows import wf_flameo, wf_randomise, create_flexible_design_matrix, save_vest_file

# Configure logging
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

logger = setup_logging()

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run group-level searchlight analysis for a single map type.')
    parser.add_argument('--map_type', required=True, help='Map type to process (e.g., within-FIXATION, between-FIXATION-CS-_first, between-FIXATION-CS-)')
    parser.add_argument('--method', choices=['flameo', 'randomise'], default='flameo', help='Analysis method: flameo or randomise')
    args = parser.parse_args()
    map_type = args.map_type
    method = args.method
    logger.info(f"Processing group-level analysis for map type: {map_type}, method: {method}")

    # Validate map_type format
    if not (map_type.startswith('within-') or map_type.startswith('between-')):
        logger.error(f"Invalid map_type format: {map_type}. Must start with 'within-' or 'between-'")
        return

    # Paths
    root_dir = os.getenv('DATA_DIR', '/data')
    project_name = 'NARSAD'
    derivatives_dir = os.path.join(root_dir, project_name, 'MRI', 'derivatives')
    data_dir = os.path.join(derivatives_dir, 'fMRI_analysis', 'LSS', 'firstLevel', 'all_subjects', 'similarity', 'searchlight')

    # Check if data directory exists
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        logger.error("Please ensure the LSS similarity analysis has been completed first.")
        return

    # Process both tasks
    tasks = ['phase2', 'phase3']
    tasks_processed = 0
    
    for task in tasks:
        logger.info(f"Processing task: {task}")
        output_dir = os.path.join(derivatives_dir, 'fMRI_analysis', 'LSS', 'groupLevel', 'searchlight', method, task)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory for {task}: {output_dir}")

        # Collect subjects
        subject_files = glob(os.path.join(data_dir, f'sub-*_task-{task}_within-FIXATION.nii.gz'))
        if not subject_files:
            logger.warning(f"No subject files found for {task} in {data_dir}")
            logger.warning(f"Expected pattern: sub-*_task-{task}_within-FIXATION.nii.gz")
            continue
            
        subjects = sorted([os.path.basename(f).split('_')[0].replace('sub-', '') for f in subject_files])
        logger.info(f"Found {len(subjects)} subjects for {task}: {subjects}")

        # Common mask - find the correct session directory
        subject_fmriprep_dir = os.path.join(root_dir, project_name, 'MRI', 'derivatives', 'fmriprep', f'sub-{subjects[0]}')
        if not os.path.exists(subject_fmriprep_dir):
            logger.error(f"Subject fmriprep directory not found: {subject_fmriprep_dir}")
            continue
            
        # Find session directories
        session_dirs = [d for d in os.listdir(subject_fmriprep_dir) if d.startswith('ses-')]
        if not session_dirs:
            logger.error(f"No session directories found in {subject_fmriprep_dir}")
            continue
            
        # Use the first session found (you can modify this logic if needed)
        session_name = session_dirs[0]
        mask_file = os.path.join(
            subject_fmriprep_dir, session_name, 'func',
            f'sub-{subjects[0]}_{session_name}_task-{task}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
        )
        logger.info(f"Using mask for {task}: {mask_file}, exists: {os.path.exists(mask_file)}")
        if not os.path.exists(mask_file):
            logger.error(f"Mask file not found for {task}: {mask_file}")
            continue

        # Collect cope files for the specified map type
        cope_files = []
        subjects_with_data = []
        for sub in subjects:
            cope_file = os.path.join(data_dir, f'sub-{sub}_task-{task}_{map_type}.nii.gz')
            if os.path.exists(cope_file):
                cope_files.append(cope_file)
                subjects_with_data.append(sub)
            else:
                logger.warning(f"Cope file missing for sub-{sub}, {map_type}, {task}: {cope_file}")
        if len(cope_files) < 2:
            logger.error(f"Insufficient cope files for {map_type}, {task}: {len(cope_files)} found")
            continue
        logger.info(f"Found {len(cope_files)} cope files for {map_type}, {task}")
        
        # Collect var_cope files only if using FLAMEO
        var_cope_files = []
        if method == 'flameo':
            for sub in subjects_with_data:
                var_cope_file = os.path.join(data_dir, f'sub-{sub}_task-{task}_{map_type}_var.nii.gz')
                if os.path.exists(var_cope_file):
                    var_cope_files.append(var_cope_file)
                else:
                    logger.warning(f"Var_cope file missing for sub-{sub}, {map_type}, {task}: {var_cope_file}")
            if len(var_cope_files) != len(cope_files):
                logger.error(f"Var_cope files missing for FLAMEO: {len(var_cope_files)} vs {len(cope_files)} cope files")
                continue
            logger.info(f"Found {len(var_cope_files)} var_cope files for FLAMEO")

        # Create design matrix and contrasts using only subjects with data
        try:
            design, contrasts = create_flexible_design_matrix(subjects_with_data, group_coding='1/0', contrast_type='standard')
        except Exception as e:
            logger.error(f"Error creating design matrix for {task}: {e}")
            continue
            
        design_file = os.path.join(output_dir, f'design_{map_type}.mat')
        con_file = os.path.join(output_dir, f'contrast_{map_type}.con')
        
        try:
            save_vest_file(design, design_file)
            # Save all contrasts in VEST format
            contrast_matrix = np.array(contrasts)
            save_vest_file(contrast_matrix, con_file)
        except Exception as e:
            logger.error(f"Error saving design/contrast files for {task}: {e}")
            continue
            
        logger.info(f"Created design for {task}: {design_file}, contrasts: {con_file}")
        logger.info(f"Contrasts: 1) patients>controls, 2) patients<controls, 3) mean_effect_patients, 4) mean_effect_controls")
        
        # Log group information
        patients = [s for s in subjects_with_data if s.startswith('1')]
        controls = [s for s in subjects_with_data if s.startswith('2')]
        logger.info(f"Group analysis: {len(patients)} patients vs {len(controls)} controls")
        
        # Check group balance
        if len(patients) < 1 or len(controls) < 1:
            logger.error(f"Insufficient subjects in one or both groups for {map_type}, {task}: {len(patients)} patients, {len(controls)} controls")
            continue

        # Create workflow from group_level_workflows.py
        wf_name = f'wf_{method}_{map_type}_{task}'
        try:
            if method == 'flameo':
                # Create FLAMEO workflow with clustering enabled
                wf = wf_flameo(output_dir=output_dir, name=wf_name, use_clustering=True, cluster_threshold=3.2)
            else:  # randomise
                # Create Randomise workflow with TFCE and voxelwise p-values
                wf = wf_randomise(output_dir=output_dir, name=wf_name, num_perm=10000, use_tfce=True, use_voxelwise=True)
        except Exception as e:
            logger.error(f"Error creating workflow for {method}, {task}: {e}")
            continue

        # Set workflow inputs
        try:
            wf.inputs.inputnode.cope_files = cope_files
            wf.inputs.inputnode.mask_file = mask_file
            wf.inputs.inputnode.design_file = design_file
            wf.inputs.inputnode.con_file = con_file
            
            # Set additional inputs for FLAMEO
            if method == 'flameo':
                wf.inputs.inputnode.var_cope_files = var_cope_files
        except Exception as e:
            logger.error(f"Error setting workflow inputs for {task}: {e}")
            continue

        # Run workflow
        try:
            logger.info(f"Running {method} workflow for {map_type}, {task}")
            logger.info(f"Workflow inputs:")
            logger.info(f"  - Cope files: {len(cope_files)} files")
            logger.info(f"  - Mask file: {mask_file}")
            logger.info(f"  - Design file: {design_file}")
            logger.info(f"  - Contrast file: {con_file}")
            if method == 'flameo':
                logger.info(f"  - Var_cope files: {len(var_cope_files)} files")
            
            # Run the workflow
            wf.run()
            
            logger.info(f"Completed group-level analysis for {map_type}, {task} with {method}")
            logger.info(f"Results saved to: {output_dir}")
            tasks_processed += 1
            
        except Exception as e:
            logger.error(f"Error running {method} workflow for {map_type}, {task}: {e}")
            logger.error(f"Workflow inputs were:")
            logger.error(f"  - Cope files: {len(cope_files)} files")
            logger.error(f"  - Mask file: {mask_file}")
            logger.error(f"  - Design file: {design_file}")
            logger.error(f"  - Contrast file: {con_file}")
            if method == 'flameo':
                logger.error(f"  - Var_cope files: {len(var_cope_files)} files")
            continue

    # Final summary
    if tasks_processed > 0:
        logger.info(f"Successfully processed {tasks_processed} out of {len(tasks)} tasks")
        logger.info(f"Group-level analysis completed for map type: {map_type}")
        logger.info(f"Results saved to: {derivatives_dir}/fMRI_analysis/LSS/groupLevel/searchlight/{method}/")
    else:
        logger.error("No tasks were successfully processed")
        logger.error("Please check the input data and error messages above")

if __name__ == '__main__':
    main()