#!/usr/bin/env python3
"""
gaze_converter.py

gaze_converter.py

This script processes raw gaze data captured by an open‑source system (e.g. WebGazer)
and converts it into a fixation‑grouped format similar to commercial outputs.
It uses DBSCAN clustering on spatial features (and optionally on time) to group
individual gaze points into fixations. In addition, even points that might have
a fixation validity of 0 are grouped with their neighbors so that the sequential
structure is preserved. (Only values with a fixation validity of 1 are used for some analyses.)

Usage:
    python gaze_converter.py input_file.csv output_file.csv

Parameters:
    input_file  : Path to the raw gaze data CSV file.
    output_file : Path to save the processed CSV with fixation IDs.

Updated:
- FPOGX = x / 1920, FPOGY = y / 1080 (monitor-based, not min/max)
- Spatio-temporal DBSCAN + time-gap splitting; bridge zero-ID runs; renumber to 1..N
- FPOGS = seeded start (one median-Δt earlier per fixation, clamped ≥ 0)
- FPOGD = row-wise (TIME_NUM − FPOGS) in seconds; first sample forced = 0
- AOI assignment
"""

import os
import glob
import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# dependency from eren-ck in order to get rid of the warning line in the terminal
# delete everything above line 13 in the __init__ file for this library.
from st_dbscan import ST_DBSCAN

# Function to normalize a value given a min and max. (unchanged, not used bc its wrong)
def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

# Function to check if a point (normalized) falls within an AOI (unchanged).
def is_point_in_aoi(x, y, aoi):
    x_min = aoi['x']
    x_max = aoi['x'] + aoi['width']
    y_min = aoi['y']
    y_max = aoi['y'] + aoi['height']
    return x_min <= x <= x_max and y_min <= y <= y_max

# Function to calculate saccade magnitude. (unchanged)
def calculate_saccade_magnitude(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Function to calculate saccade direction. (unchanged)
def calculate_saccade_direction(x1, y1, x2, y2):
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))


# Function to calculate fixation validity. (checked elsewhere so not used)
# def calculate_fixation_validity(data):
    # data['FIXATION_VALIDITY'] = data.apply(lambda row: 1 if row['FPOGD'] >= 100 else 0, axis=1)
    # return data

# (Optional) Function to calculate AOI metrics (unchanged).
def calculate_aoi_metrics(data, aoi_config):
    results = []
    total_fixations = data[data['FPOGV'] == 1].shape[0]
    total_duration = data[data['FPOGV'] == 1]['FPOGD'].sum()
    for index, aoi in enumerate(aoi_config):
        aoi_name = f"AOI_{index + 1}"
        fixations_in_aoi = data[(data[aoi_name] == 1) & (data['FPOGV'] == 1)]
        fixation_proportion = len(fixations_in_aoi) / total_fixations if total_fixations > 0 else 0
        duration_proportion = fixations_in_aoi['FPOGD'].sum() / total_duration if total_duration > 0 else 0
        results.append({
            'AOI': aoi_name,
            'Proportion_of_Fixations': fixation_proportion,
            'Proportion_of_Duration': duration_proportion
        })
    return results

# (Optional) Function to calculate AOI transitions. (unchanged)
def calculate_aoi_transitions(data):
    transitions = []
    for i in range(len(data) - 1):
        current_aoi = data.iloc[i]['AOI']
        next_aoi = data.iloc[i + 1]['AOI']
        if current_aoi != next_aoi:
            transitions.append((current_aoi, next_aoi))
    return transitions

# Function to add saccade features. (unchanged)
def add_saccade_features(data):
    data['SACCADE_MAG'] = data.apply(
        lambda row: calculate_saccade_magnitude(
            data['FPOGX'].shift(1).loc[row.name],
            data['FPOGY'].shift(1).loc[row.name],
            row['FPOGX'],
            row['FPOGY']
        ) if row.name > 0 else 0.0,
        axis=1
    )
    data['SACCADE_DIR'] = data.apply(
        lambda row: calculate_saccade_direction(
            data['FPOGX'].shift(1).loc[row.name],
            data['FPOGY'].shift(1).loc[row.name],
            row['FPOGX'],
            row['FPOGY']
        ) if row.name > 0 else 0.0,
        axis=1
    )
    return data

# Function to add placeholder fields. (unchanged)
def add_placeholder_fields(data):
    placeholders = {
        'MEDIA_ID': '', 'MEDIA_NAME': '', 'CNT': np.arange(0, len(data)),
        'FPOGV': 1, 'BPOGX': 0.0, 'BPOGY': 0.0, 'BPOGV': 0,
        'CX': 0, 'CY': 0, 'CS': '', 'KB': 0, 'KBS': 0, 'USER': '',
        'LPCX': 0.0, 'LPCY': 0.0, 'LPD': 0, 'LPS': 0, 'LPV': 0,
        'RPCX': 0.0, 'RPCY': 0.0, 'RPD': 0, 'RPS': 0, 'RPV': 0,
        'BKID': 0, 'BKDUR': 0, 'BKPMIN': 0, 'LPMM': 4, 'LPMMV': 1,
        'RPMM': 4, 'RPMMV': 1, 'DIAL': 0, 'DIALV': 0,
        'GSR': 0, 'GSRV': 0, 'HR': 0, 'HRV': 0, 'HRP': 0, 'IBI': 0,
        'TTL0': 0, 'TTL1': 0, 'TTL2': 0, 'TTL3': 0, 'TTL4': 0,
        'TTL5': 0, 'TTL6': 0, 'TTLV': 0, 'PIXS': 0, 'PIXV': 0, 'VID_FRAME': 0
    }
    for column, default in placeholders.items():
        if column not in data.columns:
            data[column] = default
    return data

# Process gaze data with AOI detection and DBSCAN-based fixation analysis.
def process_gaze_data(gaze_file, aoi_config_file='aoi_config.json',
                      output_file='processed_gaze_data.csv', fixation_file='fixations.csv'):
    #added screen width and height
    SCREEN_W, SCREEN_H = 1920.0, 1080.0

    # --- STEP 1: Read header to extract base time ---
    raw_columns = pd.read_csv(gaze_file, nrows=0).columns
    time_header_candidates = [col for col in raw_columns if col.startswith("TIME(")]
    if not time_header_candidates:
        raise ValueError("Could not find a TIME column in the expected format.")
    raw_time_header = time_header_candidates[0]
    base_time_str = raw_time_header[raw_time_header.find("(")+1 : raw_time_header.find(")")]

    # --- STEP 2: Load the raw gaze data ---
    data = pd.read_csv(gaze_file)
    data = data.rename(columns={raw_time_header: "TIME_REL"})

    # Build numeric time (seconds) with TIMETICK priority; start at 0; stable sort
    if 'TIMETICK(f=10000000)' in data.columns:
        data['TIME_NUM'] = data['TIMETICK(f=10000000)'].astype(float) / 1e7
    else:
        data['TIME_NUM'] = data['TIME_REL'].astype(float)
    data['TIME_NUM'] -= data['TIME_NUM'].iloc[0]
    data = data.sort_values(by='TIME_NUM', kind='mergesort').reset_index(drop=True)

    # --- STEP 3: Normalize X and Y coordinates to monitor (FIXED) ---
    if 'x' not in data.columns or 'y' not in data.columns:
        raise ValueError("Input must contain 'x' and 'y' columns.")
    data['FPOGX'] = data['x'] / SCREEN_W
    data['FPOGY'] = data['y'] / SCREEN_H

    # --- STEP 4: Fixation detection via spatio-temporal DBSCAN (FIXED) ---
    # for now let the min_samples be ln(X) where X is the length of the dataframe
    # it a heuristic that is followed in the ST_DBSCAN paper.
    # another heuristic amount of features * 2 = 6 # int(np.log(len(data)))

    epsSpat, epsT, min_samples = 0.03, 0.8, 5

    Time = data['TIME_NUM'].astype(float).to_numpy()
    x = data['FPOGX'].astype(float).to_numpy()
    y = data['FPOGY'].astype(float).to_numpy()

    X_feat = np.column_stack([Time, x, y])

    # the order that must enter the fit function must be ['t', 'x', 'y']
    st_db = ST_DBSCAN(eps1=epsSpat, eps2=epsT, min_samples=min_samples).fit(X_feat)
    data['st_db_label'] = st_db.labels
    print("Unique ST-DBSCAN labels:", np.unique(data['st_db_label']))

    # --- STEP 5: INITIAL FIXATION ID ASSIGNMENT  ---
    labels = data['st_db_label'].to_numpy()  # ints; -1 = noise
    N = len(labels)
    idx = np.arange(N)
    is_labeled = labels != -1

    # Nearest labeled indices to the LEFT and RIGHT (by row/time order)
    left_idx = np.where(is_labeled, idx, np.nan)
    left_idx = pd.Series(left_idx).ffill().to_numpy()  # last labeled index at/before i (or NaN)
    right_idx = np.where(is_labeled, idx, np.nan)
    right_idx = pd.Series(right_idx).bfill().to_numpy()  # first labeled index at/after i (or NaN)

    # Time gaps to those neighbors (∞ if neighbor doesn't exist) <- what?
    left_gap = np.full(N, np.inf)
    right_gap = np.full(N, np.inf)

    mL = ~np.isnan(left_idx) # True where a left neighbor exists, False otherwise.
    mR = ~np.isnan(right_idx) # True where a right neighbor exists, False otherwise.

    if mL.any():
        left_gap[mL] = Time[mL] - Time[left_idx[mL].astype(int)]
    if mR.any():
        right_gap[mR] = Time[right_idx[mR].astype(int)] - Time[mR]

    use_left = left_gap <= right_gap

    # Start from original labels; replace only noise rows by nearest labeled neighbor in time
    bridged = labels.astype(float)
    noise = ~is_labeled

    src_idx = np.full(N, np.nan)  # where to copy label from
    # choose left if it's closer and exists
    src_idx[noise & use_left & mL] = left_idx[noise & use_left & mL]
    # otherwise choose right if it exists
    src_idx[noise & ~use_left & mR] = right_idx[noise & ~use_left & mR]

    assignable = ~np.isnan(src_idx)
    bridged[assignable] = labels[src_idx[assignable].astype(int)]

    # Renumber by first occurrence time → 1..K (chronological)
    b = pd.Series(bridged, name='label')  # float but integer-like values per cluster
    first_idx = b.reset_index().groupby('label')['index'].min().sort_values()
    remap = {old: new for new, old in enumerate(first_idx.index, start=1)}
    mapped = b.map(remap)

    # Safety: ensure no zeros; if any NaNs remain (edge cases), use nearest-time fallback then clamp to 1
    if mapped.isna().any():
        mapped = mapped.ffill().bfill().fillna(1)

    data['FPOGID'] = mapped.astype(int)

    # Optional: quick stats
    print("Final FPOGIDs:", data['FPOGID'].nunique(),
          " assigned rows %:", 100 * (data['FPOGID'] > 0).mean())

    # # --- STEP 5b: Renumber FPOGIDs sequentially starting at 1 (FIXED) ---
    # unique_ids = [int(x) for x in pd.unique(data['FPOGID']) if x > 0]
    # unique_ids.sort()
    # new_fix_ids = {old: new for new, old in enumerate(unique_ids, start=1)}
    # data['FPOGID'] = data['FPOGID'].map(new_fix_ids).fillna(1).astype(int)

    # --- STEP 5c: Seeded FPOGS and row-wise FPOGD (FIXED) ---
    fix_group = data.groupby('FPOGID')['TIME_NUM']
    true_start = fix_group.min()
    true_end   = fix_group.max()

    med_dt_per_fix = data.groupby('FPOGID')['TIME_NUM'].apply(
        lambda s: np.median(np.diff(s.values)) if len(s) > 1 and np.any(np.diff(s.values) > 0) else np.nan
    )
    global_dt = float(np.median(np.diff(data['TIME_NUM'].values))) if len(data) > 1 else 0.033
    med_dt_per_fix = med_dt_per_fix.fillna(max(1e-3, global_dt))

    seeded_start = (true_start - med_dt_per_fix).clip(lower=0)
    data['FPOGS'] = data['FPOGID'].map(seeded_start).astype(float)
    data['FPOGD'] = (data['TIME_NUM'] - data['FPOGS']).clip(lower=0.0)
    if len(data) > 0:
        data.loc[0, 'FPOGD'] = 0.0  # keep very first row exactly 0

    # Validity based on TRUE (unseeded) duration
    # total_dur = (true_end - true_start).astype(float)
    # valid_map = (total_dur >= 0.100).astype(int)
    # data['FPOGV'] = data['FPOGID'].map(valid_map).fillna(0).astype(int)

    # --- STEP 6: AOI assignment ---
    AOI_Q, AOI_V = None, None
    if os.path.exists(aoi_config_file):
        try:
            with open(aoi_config_file, 'r') as f:
                aoi_config = json.load(f)
            if isinstance(aoi_config, list):
                if len(aoi_config) >= 1:
                    AOI_Q = aoi_config[0]
                if len(aoi_config) >= 2:
                    AOI_V = aoi_config[1]
        except json.JSONDecodeError:
            print(f"Error: AOI configuration file '{aoi_config_file}' is not valid JSON.")
            aoi_config = []
    else:
        print(f"Warning: AOI configuration file '{aoi_config_file}' not found. Proceeding without AOIs.")
        aoi_config = []

    def assign_aoi(row):
        # Convert normalized to pixel before checking (FIXED)
        x_px = row['FPOGX'] * SCREEN_W
        y_px = row['FPOGY'] * SCREEN_H

        # Any (0,0) coordinate is None
        if x_px == 0 and y_px == 0:
            return "None"

        if AOI_Q and is_point_in_aoi(x_px, y_px, AOI_Q):
            return "AOI_Q"
        if AOI_V and is_point_in_aoi(x_px, y_px, AOI_V):
            return "AOI_V"
        return "None"

    data['AOI'] = data.apply(assign_aoi, axis=1)

    # --- STEP 7: (Code 1 had "Compute fixation metrics") —
    # We already computed FPOGS/FPOGD above

    # --- STEP 8: Add placeholder fields ---
    data = add_placeholder_fields(data)

    # --- STEP 9: Add saccade features (same helpers as Code 1) ---
    data = add_saccade_features(data)

    # --- STEP 10: Prepare final output (same ordering as Code 1) ---
    new_time_header = f"TIME({base_time_str})"
    data = data.rename(columns={"TIME_REL": new_time_header})
    # Drop intermediate columns.
    data = data.drop(columns=["db_label"], errors='ignore')
    columns_order = [
        'MEDIA_ID', 'MEDIA_NAME', 'CNT', new_time_header, 'TIMETICK(f=10000000)',
        'FPOGX', 'FPOGY', 'FPOGS', 'FPOGD', 'FPOGID', 'FPOGV',
        'BPOGX', 'BPOGY', 'BPOGV', 'CX', 'CY', 'CS',
        'KB', 'KBS', 'USER', 'LPCX', 'LPCY', 'LPD', 'LPS', 'LPV',
        'RPCX', 'RPCY', 'RPD', 'RPS', 'RPV', 'BKID', 'BKDUR', 'BKPMIN',
        'LPMM', 'LPMMV', 'RPMM', 'RPMMV', 'DIAL', 'DIALV',
        'GSR', 'GSRV', 'HR', 'HRV', 'HRP', 'IBI',
        'TTL0', 'TTL1', 'TTL2', 'TTL3', 'TTL4', 'TTL5', 'TTL6', 'TTLV',
        'PIXS', 'PIXV', 'AOI', 'SACCADE_MAG', 'SACCADE_DIR', 'VID_FRAME'
    ]
    for c in columns_order:
        if c not in data.columns:
            data[c] = 0 if c not in ('MEDIA_ID', 'MEDIA_NAME', 'CS', 'USER', 'AOI') else ''
    data = data[columns_order]
    data.to_csv(output_file, index=False)
    print(f"Processed gaze data with AOIs saved to {output_file}")

if __name__ == "__main__":
    # Use glob to search for all files matching "p*_webcam_gaze_data.csv"
    raw_files = glob.glob("p*_webcam_gaze_data.csv")
    if not raw_files:
        print("No raw gaze data files matching 'p*_webcam_gaze_data.csv' were found.")
    else:
        aoi_config_file = 'aoi_config.json'
        for gaze_file in raw_files:
            base_name = os.path.basename(gaze_file)
            parts = base_name.split("_")
            if len(parts) < 2:
                print(f"Filename {base_name} does not match expected pattern. Skipping.")
                continue
            participant_id = parts[0]  # e.g., "p7"
            output_file = f"{participant_id}_all_gaze.csv"
            fixation_file = f"{participant_id}_fixations.csv"  # if needed
            print(f"Processing {gaze_file} -> {output_file}")
            process_gaze_data(gaze_file, aoi_config_file, output_file, fixation_file)
