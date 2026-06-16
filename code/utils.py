# Author: Xiaoqian Xiao (xiao.xiaoqian.320@gmail.com)

import pandas as pd

def _get_tr(in_dict):
    return in_dict.get('RepetitionTime')


def _len(inlist):
    return len(inlist)


def _dof(inlist):
    return len(inlist) - 1


def _neg(val):
    return -val

def _dict_ds(in_dict, sub, order=['bold', 'mask', 'events', 'regressors', 'tr']):
    return tuple([in_dict[sub][k] for k in order])

def _dict_ds_lss(in_dict, sub, order=['bold', 'mask', 'events', 'regressors', 'tr', 'trial_ID']):
    return tuple([in_dict[sub][k] for k in order])

def _bids2nipypeinfo(in_file, events_file, regressors_file,
                     regressors_names=None,
                     motion_columns=None,
                     decimals=3, amplitude=1.0):
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from nipype.interfaces.base.support import Bunch

    # Process the events file with automatic separator detection
    # Import the function locally to ensure it's available
    from utils import read_csv_with_detection
    
    events = read_csv_with_detection(events_file)
    print("=== DEBUG: loaded event columns ===")
    print(events.columns.tolist())
    print(events.head())

    # Detect the condition column (try different possible names)
    condition_column = None
    possible_columns = ['trial_type', 'condition', 'event_type', 'type', 'stimulus', 'trial']
    for col in possible_columns:
        if col in events.columns:
            condition_column = col
            break
    
    if condition_column is None:
        # If no standard column found, try to use the first non-numeric column
        for col in events.columns:
            if not pd.api.types.is_numeric_dtype(events[col]):
                condition_column = col
                break
    
    if condition_column is None:
        raise ValueError(f"Could not find condition column in events file. Available columns: {events.columns.tolist()}")
    
    print(f"Using column '{condition_column}' for conditions")

    bunch_fields = ['onsets', 'durations', 'amplitudes']

    if not motion_columns:
        from itertools import product
        motion_columns = ['_'.join(v) for v in product(('trans', 'rot'), 'xyz')]

    out_motion = Path('motion.par').resolve()

    regress_data = read_csv_with_detection(regressors_file)
    np.savetxt(out_motion, regress_data[motion_columns].values, '%g')
    if regressors_names is None:
        regressors_names = sorted(set(regress_data.columns) - set(motion_columns))

    if regressors_names:
        bunch_fields += ['regressor_names']
        bunch_fields += ['regressors']

    # Create conditions list with proper CS- splitting
    raw_conditions = list(events[condition_column].values)
    
    # Count CS- trials and create proper condition names
    cs_count = raw_conditions.count('CS-')
    if cs_count > 1:
        # Multiple CS- trials: split into CS-_first and CS-_others
        conditions = ['CS-_first', 'CS-_others']
        # Add other unique conditions (excluding CS-)
        other_conditions = [c for c in set(raw_conditions) if c != 'CS-']
        conditions.extend(other_conditions)
        print(f"Split {cs_count} CS- trials into CS-_first and CS-_others. Total conditions: {len(conditions)}")
    else:
        # Single or no CS- trials: use original logic
        conditions = list(set(raw_conditions))
        print(f"Using standard conditions: {len(conditions)} total")
    
    runinfo = Bunch(
        scans=in_file,
        conditions=conditions,
        **{k: [] for k in bunch_fields})

    for condition in runinfo.conditions:
        if condition == 'CS-_first':
            # First CS- trial: get the first occurrence
            cs_events = events[events[condition_column] == 'CS-']
            if len(cs_events) > 0:
                first_cs = cs_events.iloc[0:1]  # Get first CS- trial
                runinfo.onsets.append(np.round(first_cs.onset.values, 3).tolist())
                runinfo.durations.append(np.round(first_cs.duration.values, 3).tolist())
                if 'amplitudes' in events.columns:
                    runinfo.amplitudes.append(np.round(first_cs.amplitudes.values, 3).tolist())
                else:
                    runinfo.amplitudes.append([amplitude] * len(first_cs))
            else:
                # Fallback if no CS- trials found
                runinfo.onsets.append([])
                runinfo.durations.append([])
                runinfo.amplitudes.append([])
                
        elif condition == 'CS-_others':
            # Other CS- trials: get all except the first
            cs_events = events[events[condition_column] == 'CS-']
            if len(cs_events) > 1:
                other_cs = cs_events.iloc[1:]  # Get all CS- trials except first
                runinfo.onsets.append(np.round(other_cs.onset.values, 3).tolist())
                runinfo.durations.append(np.round(other_cs.duration.values, 3).tolist())
                if 'amplitudes' in events.columns:
                    runinfo.amplitudes.append(np.round(other_cs.amplitudes.values, 3).tolist())
                else:
                    runinfo.amplitudes.append([amplitude] * len(other_cs))
            else:
                # Fallback if only 1 CS- trial
                runinfo.onsets.append([])
                runinfo.durations.append([])
                runinfo.amplitudes.append([])
                
        else:
            # Regular condition: use original logic
            event = events[events[condition_column].str.match(str(condition))]
            runinfo.onsets.append(np.round(event.onset.values, 3).tolist())
            runinfo.durations.append(np.round(event.duration.values, 3).tolist())
            if 'amplitudes' in events.columns:
                runinfo.amplitudes.append(np.round(event.amplitudes.values, 3).tolist())
            else:
                runinfo.amplitudes.append([amplitude] * len(event))

    if 'regressor_names' in bunch_fields:
        runinfo.regressor_names = regressors_names
        try:
            runinfo.regressors = regress_data[regressors_names]
        except KeyError:
            regressors_names = list(set(regressors_names).intersection(
                set(regress_data.columns)))
            runinfo.regressors = regress_data[regressors_names]
        runinfo.regressors = regress_data[regressors_names].fillna(0.0).values.T.tolist()

    return [runinfo], str(out_motion)


def _bids2nipypeinfo_lss(in_file, events_file, regressors_file,
                          trial_ID,
                          regressors_names=None,
                          motion_columns=None,
                          decimals=3,
                          amplitude=1.0):
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from nipype.interfaces.base.support import Bunch

    # Load events and regressors with automatic separator detection
    from utils import read_csv_with_detection
    
    events = read_csv_with_detection(events_file)
    print("LOADED EVENTS COLUMNS:", events.columns.tolist())
    print(events.head())
    regress_data = read_csv_with_detection(regressors_file)

    # ------------------------------------------------------------------
    # 1) MOTION REGRESSORS: 6 motion + 6 derivatives for motion.par
    # ------------------------------------------------------------------
    if not motion_columns:
        # Base 6 motion params
        base_motion = [
            'trans_x', 'trans_y', 'trans_z',
            'rot_x', 'rot_y', 'rot_z'
        ]
        # First derivatives (fMRIPrep-style names)
        deriv_motion = [
            f'{c}_derivative1' for c in base_motion
        ]
        motion_columns = base_motion + deriv_motion

    # Keep only motion columns that exist
    available_motion = [c for c in motion_columns if c in regress_data.columns]
    missing_motion = sorted(set(motion_columns) - set(available_motion))
    if missing_motion:
        print(f"[LSS WARNING] Missing motion columns in {regressors_file}: {missing_motion}")

    if not available_motion:
        raise ValueError(
            f"LSS: no motion columns found in {regressors_file}. "
            f"Tried: {motion_columns}"
        )

    out_motion = Path('motion.par').resolve()
    np.savetxt(out_motion, regress_data[available_motion].values, '%g')

    # ------------------------------------------------------------------
    # 2) SPIKE REGRESSORS (FD / DVARS outliers)
    # ------------------------------------------------------------------
    spike_cols = []
    fd = regress_data['framewise_displacement'] if 'framewise_displacement' in regress_data.columns else None
    dvars = regress_data['dvars'] if 'dvars' in regress_data.columns else None

    n_tp = len(regress_data)
    spike_mask = np.zeros(n_tp, dtype=bool)

    # FD threshold (mm)
    if fd is not None:
        spike_mask |= (fd.values > 0.5)

    # DVARS threshold (relative) – simple robust cut: mean + 1.5*std
    if dvars is not None:
        thr = dvars.mean() + 1.5 * dvars.std()
        spike_mask |= (dvars.values > thr)

    # Create one column per spike
    spike_indices = np.where(spike_mask)[0]
    for idx in spike_indices:
        col_name = f'spike_{idx:04d}'
        spike_vec = np.zeros(n_tp, dtype=float)
        spike_vec[idx] = 1.0
        regress_data[col_name] = spike_vec
        spike_cols.append(col_name)

    if len(spike_cols) > 0:
        print(f"[LSS INFO] Added {len(spike_cols)} spike regressors: first few = {spike_cols[:5]}")

    # ------------------------------------------------------------------
    # 3) NUISANCE REGRESSORS: 2 aCompCor + cosines + spikes
    # ------------------------------------------------------------------
    if regressors_names is None:
        # aCompCor: take at most 2 components if available
        acompcor_cols = sorted([c for c in regress_data.columns if c.startswith('a_comp_cor_')])
        acompcor_cols = acompcor_cols[:2]

        # Cosine drifts (any cosineXX-like columns)
        cosine_cols = sorted([c for c in regress_data.columns if c.startswith('cosine')])

        # Combine: 2 aCompCor + cosines + spike regressors
        regressors_names = acompcor_cols + cosine_cols + spike_cols

    # Keep only nuisance regressors that actually exist
    if regressors_names:
        available_regs = [c for c in regressors_names if c in regress_data.columns]
        missing_regs = sorted(set(regressors_names) - set(available_regs))
        if missing_regs:
            print(f"[LSS WARNING] Missing nuisance regressors in {regressors_file}: {missing_regs}")
    else:
        available_regs = []

    # ------------------------------------------------------------------
    # 4) Locate the trial of interest and build 'trial' vs 'others'
    # ------------------------------------------------------------------
    if 'trial_ID' not in events.columns:
        raise ValueError(
            f"'trial_ID' column not found in events file {events_file}. "
            f"Available columns: {events.columns.tolist()}"
        )

    trial = events[events['trial_ID'] == trial_ID]
    if trial.empty:
        raise ValueError(f"Trial ID {trial_ID} not found in events file.")
    if len(trial) > 1:
        raise ValueError(
            f"Trial ID {trial_ID} is not unique in events file. "
            f"Found {len(trial)} rows."
        )

    other_trials = events[events['trial_ID'] != trial_ID]

    conditions = ['trial', 'others']
    onsets = [
        np.round(trial['onset'].values.tolist(), decimals),
        np.round(other_trials['onset'].values.tolist(), decimals)
    ]
    durations = [
        np.round(trial['duration'].values.tolist(), decimals),
        np.round(other_trials['duration'].values.tolist(), decimals)
    ]
    amplitudes = [
        [amplitude] * len(onsets[0]),
        [amplitude] * len(onsets[1])
    ]

    runinfo = Bunch(
        scans=in_file,
        conditions=conditions,
        onsets=onsets,
        durations=durations,
        amplitudes=amplitudes
    )

    # ------------------------------------------------------------------
    # 5) Attach nuisance regressors (2 aCompCor + cosines + spikes)
    # ------------------------------------------------------------------
    if available_regs:
        regress_subset = regress_data[available_regs].fillna(0.0)
        runinfo.regressor_names = available_regs
        runinfo.regressors = regress_subset.values.T.tolist()

    return [runinfo], str(out_motion)



def print_input_traits(interface_class):
    """
    Print all input traits of a Nipype interface class, with mandatory inputs listed first,
    and then extract any mutually‐exclusive input groups from the interface’s help().

    Parameters:
    - interface_class: A Nipype interface class (e.g., SpecifyModel)
    """
    import io, sys

    # 1) List all traits
    spec = interface_class().inputs
    traits = spec.traits().items()
    sorted_traits = sorted(traits, key=lambda item: not item[1].mandatory)

    print("Name                           | mandatory")
    print("-------------------------------|----------")
    for name, trait in sorted_traits:
        print(f"{name:30} | {trait.mandatory}")

    # 2) Capture help() output to find the "Mutually exclusive inputs" line
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        interface_class().help()
    finally:
        sys.stdout = old_stdout

    help_text = buf.getvalue().splitlines()
    mux_line = next((line for line in help_text if 'mutually_exclusive' in line), None)

    # 3) Parse and print mutually‐exclusive inputs if present
    if mux_line:
        # e.g. "Mutually exclusive inputs: subject_info, event_files, bids_event_file"
        _, fields = mux_line.split(':', 1)
        names = [n.strip() for n in fields.split(',')]
        print("\nMutually exclusive inputs:")
        for n in names:
            print(f"  - {n}")
    else:
        print("\nNo mutually exclusive inputs found in help().")


def print_output_traits(interface_class):
    """
    Print all input traits of a Nipype interface class, with mandatory inputs listed first,
    and then extract any mutually‐exclusive input groups from the interface’s help().

    Parameters:
    - interface_class: A Nipype interface class (e.g., SpecifyModel)
    """
    import io, sys

    # 1) List all traits
    spec = interface_class().output_spec()  # same as inst.output_spec(), but bound
    traits = spec.traits().items()
    sorted_traits = sorted(traits, key=lambda item: not item[1].mandatory)

    print("Name                           | mandatory")
    print("-------------------------------|----------")
    for name, trait in sorted_traits:
        print(f"{name:30} | {trait.mandatory}")


def detect_csv_separator(file_path, sample_size=1024):
    """
    Automatically detect the separator used in a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        sample_size (int): Number of characters to read for detection
    
    Returns:
        str: Detected separator ('\t' for tab, ',' for comma)
    """
    try:
        with open(file_path, 'r') as f:
            sample = f.read(sample_size)
        
        # Count occurrences of potential separators
        comma_count = sample.count(',')
        tab_count = sample.count('\t')
        
        # Determine the most likely separator
        if tab_count > comma_count:
            return '\t'
        else:
            return ','
    except Exception as e:
        # Default to comma if detection fails
        return ','

def read_csv_with_detection(file_path, **kwargs):
    """
    Read a CSV file with automatic separator detection.
    
    Args:
        file_path (str): Path to the CSV file
        **kwargs: Additional arguments to pass to pd.read_csv
    
    Returns:
        pandas.DataFrame: Loaded CSV data
    """
    separator = detect_csv_separator(file_path)
    return pd.read_csv(file_path, sep=separator, **kwargs)





