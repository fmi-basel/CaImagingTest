#%%
import os
import re
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from helpers_figures import plot_fly_trial_panels, plot_group_identity_boxplots
from postUtils import (
    normalize_odor_name,
    safe_filename,
    pvalue_to_stars,
    sem,
    nanmean_padded,
    mean_and_sem_padded,
)
from utilities import load_experiment_metadata

#%%
main_dir = "/Volumes/tungsten/scratch/gfelsenb/Ana/2p-imaging/burak/"
container_id = '2026_02_Alpha3_dendrites'

experiment_dir = os.path.join(main_dir, container_id)  # Update this path accordingly
processed_data_dir = os.path.join(main_dir, container_id, f'{container_id}_processed_data')
results_dir = os.path.join(experiment_dir, 'results')
pass_fail_dir = os.path.join(results_dir, 'pass_fail')
os.makedirs(results_dir, exist_ok=True)
os.makedirs(pass_fail_dir, exist_ok=True)

# Read the database
db_path = os.path.join(experiment_dir, f'{container_id}_database.csv')
if not os.path.exists(db_path):
    print(f"Error: Database file {db_path} does not exist.")
    exit('Terminating script.')
# Read all the files that end with .pkl in the processed data directory
pkl_files = [f for f in os.listdir(processed_data_dir) if f.endswith('.pkl')]
processed_data_files = [f for f in pkl_files if re.search(r'_processed_data\.pkl$', f)]
print(f"Found {len(processed_data_files)} processed data files in {processed_data_dir}.")

# %%
roi_rows = []
roi_trial_response_dict = {}
roi_trial_response_norm_dict = {}
roi_avg_response_by_odor_dict = {}
roi_avg_response_by_odor_norm_dict = {}
roi_trial_pass_dict = {}
roi_trial_plot_windows = {}
roi_trial_plot_windows_norm = {}
alpha_response = 0.05



for p_file in processed_data_files:
    p_path = os.path.join(processed_data_dir, p_file)
    with open(p_path, 'rb') as f:
        p_data = pickle.load(f)

    metadata = p_data.get('metadata', {})
    series_meta = load_experiment_metadata(db_path, metadata['seriesID'])
    series_meta['experimentID'] = f"{metadata['seriesID']}_{metadata.get('experimentID', '')}"
    # Discard 
    if series_meta.get('discard', False):
        print(f"Skipping {metadata['seriesID']} due to discard flag in database.")
        continue

    # Update the metadata with the most up-to-date information from the database
    metadata.update(series_meta)
    
    rois = p_data.get('rois', {})
    frame_rate_hz = 6 # Hard coded 
    curr_flyID = metadata.get('flyID')
    curr_CSp = metadata.get('CSp')
    curr_CSm = metadata.get('CSm')
    curr_session_type = metadata.get('sessionType')
    curr_genotype = (
        metadata.get('genotype')
        if metadata.get('genotype') is not None
        else metadata.get('genoType')
    )
    curr_group = metadata.get('group')

    csp_name = normalize_odor_name(curr_CSp) 
    csm_name = normalize_odor_name(curr_CSm)

    for roi_unique_name, roi_data in rois.items():
        single_trial_ctx = roi_data.get('single_trial_traces_with_context', {})
        trial_response_by_odor = {}
        trial_pass_by_odor = {}
        avg_response_by_odor = {}
        trial_plot_windows_by_odor = {}
        all_trial_pass_flags = []

        if isinstance(single_trial_ctx, dict):
            for stimulus_name, stimulus_data in single_trial_ctx.items():
                if not isinstance(stimulus_data, dict):
                    continue

                context_window_s = stimulus_data.get('context_window_s', 0.0)
                try:
                    context_window_s = float(context_window_s)
                except (TypeError, ValueError):
                    context_window_s = 0.0

                trial_traces = stimulus_data.get('trial_traces', [])
                stim_start_indices = stimulus_data.get('stim_start_indices_in_window', [])
                stim_end_indices = stimulus_data.get('stim_end_indices_in_window', [])
 
                context_window_frames = 0
                if frame_rate_hz is not None and context_window_s > 0:
                    context_window_frames = int(round(frame_rate_hz * context_window_s))
                if context_window_frames <= 0 and len(stim_start_indices) > 0:
                    try:
                        context_window_frames = int(round(float(np.nanmedian(stim_start_indices))))
                    except (TypeError, ValueError):
                        context_window_frames = 0

                stimulus_key = normalize_odor_name(stimulus_name)
                if stimulus_key is None:
                    continue

                trial_responses = []
                trial_is_pass = []
                trial_windows = []
                for trial_index, trial_trace in enumerate(trial_traces):
                    trace_array = np.asarray(trial_trace, dtype=float)
                    if trace_array.size == 0:
                        trial_responses.append(np.nan)
                        trial_is_pass.append(False)
                        trial_windows.append(
                            {
                                'trace': np.array([], dtype=float),
                                'stim_start': 0,
                                'stim_end': 0,
                                'context_start': 0,
                                'context_end': 0,
                                'is_pass': False,
                            }
                        )
                        continue

                    default_stim_start = context_window_frames
                    stim_start = stim_start_indices[trial_index] if trial_index < len(stim_start_indices) else default_stim_start
                    stim_end = stim_end_indices[trial_index] if trial_index < len(stim_end_indices) else trace_array.size

                    try:
                        stim_start = int(round(float(stim_start)))
                    except (TypeError, ValueError):
                        stim_start = int(default_stim_start)
                    try:
                        stim_end = int(round(float(stim_end)))
                    except (TypeError, ValueError):
                        stim_end = int(trace_array.size)

                    stim_start = int(np.clip(stim_start, 0, trace_array.size))
                    stim_end = int(np.clip(stim_end, stim_start, trace_array.size))

                    context_end = stim_start
                    context_start = max(0, context_end - max(0, context_window_frames))

                    stim_segment = trace_array[stim_start:stim_end]
                    context_segment = trace_array[context_start:context_end]

                    stim_segment = stim_segment[np.isfinite(stim_segment)]
                    context_segment = context_segment[np.isfinite(context_segment)]

                    if stim_segment.size == 0 or context_segment.size == 0:
                        trial_responses.append(np.nan)
                        trial_is_pass.append(False)
                        trial_windows.append(
                            {
                                'trace': trace_array,
                                'stim_start': int(stim_start),
                                'stim_end': int(stim_end),
                                'context_start': int(context_start),
                                'context_end': int(context_end),
                                'is_pass': False,
                            }
                        )
                        continue

                    stim_mean = np.nanmean(stim_segment)
                    context_mean = np.nanmean(context_segment)
                    response_value = float(stim_mean - context_mean)
                    trace_array_adjusted = trace_array - context_mean

                    p_value = np.nan
                    try:
                        test_result = mannwhitneyu(stim_segment, context_segment, alternative='greater')
                        p_value = float(test_result.pvalue)
                    except ValueError:
                        p_value = np.nan

                    is_pass_trial = bool(np.isfinite(p_value) and p_value < alpha_response and response_value > 0)

                    trial_responses.append(response_value)
                    trial_is_pass.append(is_pass_trial)
                    trial_windows.append(
                        {
                            'trace': trace_array_adjusted,
                            'stim_start': int(stim_start),
                            'stim_end': int(stim_end),
                            'context_start': int(context_start),
                            'context_end': int(context_end),
                            'is_pass': is_pass_trial,
                        }
                    )

                trial_response_by_odor[stimulus_key] = trial_responses
                trial_pass_by_odor[stimulus_key] = trial_is_pass
                trial_plot_windows_by_odor[stimulus_key] = trial_windows
                all_trial_pass_flags.extend(trial_is_pass)

                if len(trial_responses) == 0:
                    avg_response_by_odor[stimulus_key] = np.nan
                else:
                    avg_response_by_odor[stimulus_key] = float(np.nanmean(np.asarray(trial_responses, dtype=float)))

        if len(all_trial_pass_flags) == 0:
            responsiveness_index = np.nan
            is_responsive_roi = False
        else:
            responsiveness_index = float(np.mean(np.asarray(all_trial_pass_flags, dtype=float)))
            is_responsive_roi = bool(np.any(all_trial_pass_flags))

        roi_trial_response_dict[roi_unique_name] = trial_response_by_odor
        roi_trial_pass_dict[roi_unique_name] = trial_pass_by_odor
        roi_trial_plot_windows[roi_unique_name] = trial_plot_windows_by_odor

        avg_trial_traces_by_odor = []
        for trial_windows_list in trial_plot_windows_by_odor.values():
            odor_trial_traces = []
            for window_info in trial_windows_list:
                trace_values = np.asarray(window_info.get('trace', np.array([], dtype=float)), dtype=float)
                if trace_values.size == 0:
                    continue
                odor_trial_traces.append(trace_values)

            if len(odor_trial_traces) == 0:
                continue

            max_len = max(trace.size for trace in odor_trial_traces)
            padded = np.full((len(odor_trial_traces), max_len), np.nan, dtype=float)
            for trace_idx, trace_values in enumerate(odor_trial_traces):
                padded[trace_idx, :trace_values.size] = trace_values

            avg_trace_this_odor = np.nanmean(padded, axis=0)
            finite_avg = avg_trace_this_odor[np.isfinite(avg_trace_this_odor)]
            if finite_avg.size > 0:
                avg_trial_traces_by_odor.append(finite_avg)

        roi_norm_max = np.nan
        if avg_trial_traces_by_odor:
            all_values_concat = np.concatenate(avg_trial_traces_by_odor)
            if all_values_concat.size > 0:
                roi_norm_max = float(np.nanmax(all_values_concat))
                if (not np.isfinite(roi_norm_max)) or roi_norm_max <= 0:
                    roi_norm_max = float(np.nanmax(np.abs(all_values_concat)))

        trial_plot_windows_norm_by_odor = {}
        for odor_key, trial_windows_list in trial_plot_windows_by_odor.items():
            norm_trial_windows_list = []
            for window_info in trial_windows_list:
                trace_values = np.asarray(window_info.get('trace', np.array([], dtype=float)), dtype=float)
                if np.isfinite(roi_norm_max) and roi_norm_max > 0 and trace_values.size > 0:
                    trace_norm = trace_values / roi_norm_max
                else:
                    trace_norm = trace_values
                norm_window_info = dict(window_info)
                norm_window_info['trace'] = trace_norm
                norm_trial_windows_list.append(norm_window_info)
            trial_plot_windows_norm_by_odor[odor_key] = norm_trial_windows_list

        trial_response_by_odor_norm = {}
        avg_response_by_odor_norm = {}
        for odor_key, norm_trial_windows_list in trial_plot_windows_norm_by_odor.items():
            norm_trial_responses = []
            for window_info in norm_trial_windows_list:
                trace_values = np.asarray(window_info.get('trace', np.array([], dtype=float)), dtype=float)
                if trace_values.size == 0:
                    norm_trial_responses.append(np.nan)
                    continue

                stim_start = int(window_info.get('stim_start', 0))
                stim_end = int(window_info.get('stim_end', 0))
                context_start = int(window_info.get('context_start', 0))
                context_end = int(window_info.get('context_end', 0))

                stim_start = int(np.clip(stim_start, 0, trace_values.size))
                stim_end = int(np.clip(stim_end, stim_start, trace_values.size))
                context_start = int(np.clip(context_start, 0, trace_values.size))
                context_end = int(np.clip(context_end, context_start, trace_values.size))

                stim_segment = trace_values[stim_start:stim_end]
                context_segment = trace_values[context_start:context_end]
                stim_segment = stim_segment[np.isfinite(stim_segment)]
                context_segment = context_segment[np.isfinite(context_segment)]

                if stim_segment.size == 0 or context_segment.size == 0:
                    norm_trial_responses.append(np.nan)
                    continue

                norm_trial_responses.append(float(np.nanmean(stim_segment) - np.nanmean(context_segment)))

            trial_response_by_odor_norm[odor_key] = norm_trial_responses
            if len(norm_trial_responses) == 0:
                avg_response_by_odor_norm[odor_key] = np.nan
            else:
                avg_response_by_odor_norm[odor_key] = float(np.nanmean(np.asarray(norm_trial_responses, dtype=float)))

        roi_trial_plot_windows_norm[roi_unique_name] = trial_plot_windows_norm_by_odor
        roi_trial_response_norm_dict[roi_unique_name] = trial_response_by_odor_norm
        roi_avg_response_by_odor_dict[roi_unique_name] = avg_response_by_odor
        roi_avg_response_by_odor_norm_dict[roi_unique_name] = avg_response_by_odor_norm

        roi_rows.append(
            {
                'roi_unique_name': roi_unique_name,
                'flyID': curr_flyID,
                'frame_rate_hz': frame_rate_hz,
                'responsiveness_index': responsiveness_index,
                'is_responsive': is_responsive_roi,
                'sessionType': curr_session_type,
                'CSp': curr_CSp,
                'CSm': curr_CSm,
                'genotype': curr_genotype,
                'group': curr_group,
            }
        )

roi_database = pd.DataFrame(
    roi_rows,
    columns=[
        'roi_unique_name',
        'flyID',
        'frame_rate_hz',
        'responsiveness_index',
        'is_responsive',
        'sessionType',
        'CSp',
        'CSm',
        'genotype',
        'group',
    ],
)

print(f"Built ROI dataframe with {len(roi_database)} rows.")
print(f"Built ROI trial-response dictionary with {len(roi_trial_response_dict)} entries.")
print(f"Built ROI normalized trial-response dictionary with {len(roi_trial_response_norm_dict)} entries.")
print(f"Built ROI trial-pass dictionary with {len(roi_trial_pass_dict)} entries.")
print(f"Built ROI normalized trial-trace dictionary with {len(roi_trial_plot_windows_norm)} entries.")
print(f"Built ROI odor-averaged response dictionary with {len(roi_avg_response_by_odor_dict)} entries.")
print(f"Built ROI normalized odor-averaged response dictionary with {len(roi_avg_response_by_odor_norm_dict)} entries.")

fly_responsive_df = (
    roi_database
    .groupby('flyID', as_index=False)['is_responsive']
    .any()
    .rename(columns={'is_responsive': 'fly_is_responsive'})
)
responsive_fly_ids = set(fly_responsive_df.loc[fly_responsive_df['fly_is_responsive'], 'flyID'])
nonresponsive_fly_ids = set(fly_responsive_df.loc[~fly_responsive_df['fly_is_responsive'], 'flyID'])
roi_database_all = roi_database.copy()
roi_database = roi_database.loc[roi_database['flyID'].isin(responsive_fly_ids)].copy()
print(
    f"Filtered to responsive flies: {len(responsive_fly_ids)}/{len(fly_responsive_df)} flies kept, "
    f"{len(roi_database)} ROI rows remain."
)


print(f"Plotting responsive flies: {len(responsive_fly_ids)}")
plot_fly_trial_panels(
    responsive_fly_ids,
    status_label='PASS FILTER',
    roi_database_all=roi_database_all,
    roi_trial_plot_windows=roi_trial_plot_windows,
    pass_fail_dir=pass_fail_dir,
    normalize_odor_name=normalize_odor_name,
    safe_filename=safe_filename,
)

print(f"Plotting non-responsive flies: {len(nonresponsive_fly_ids)}")
plot_fly_trial_panels(
    nonresponsive_fly_ids,
    status_label='FAIL FILTER',
    roi_database_all=roi_database_all,
    roi_trial_plot_windows=roi_trial_plot_windows,
    pass_fail_dir=pass_fail_dir,
    normalize_odor_name=normalize_odor_name,
    safe_filename=safe_filename,
)

roi_database.head()

   

# %%
colors_hex = {'MCH': '#d95f02', 'OCTT': "#7570b3", "IAA": '#999999'}
roi_identity_rows = []

for _, roi_row in roi_database.iterrows():
    roi_name = roi_row['roi_unique_name']
    group_name = roi_row.get('group')

    if pd.isna(group_name):
        continue

    odor_response_dict = roi_avg_response_by_odor_dict.get(roi_name, {})
    if not isinstance(odor_response_dict, dict) or len(odor_response_dict) == 0:
        continue

    normalized_odor_values = {}
    for odor_name, response_value in odor_response_dict.items():
        odor_norm = normalize_odor_name(odor_name)
        if odor_norm is None:
            continue
        normalized_odor_values.setdefault(odor_norm, []).append(response_value)

    csp_norm = normalize_odor_name(roi_row.get('CSp'))
    csm_norm = normalize_odor_name(roi_row.get('CSm'))

    csp_values = normalized_odor_values.get(csp_norm, []) if csp_norm is not None else []
    csm_values = normalized_odor_values.get(csm_norm, []) if csm_norm is not None else []

    novel_values = []
    for odor_norm, values in normalized_odor_values.items():
        if odor_norm in {csp_norm, csm_norm}:
            continue
        novel_values.extend(values)

    csp_response = float(np.nanmean(np.asarray(csp_values, dtype=float))) if len(csp_values) else np.nan
    csm_response = float(np.nanmean(np.asarray(csm_values, dtype=float))) if len(csm_values) else np.nan
    novel_response = float(np.nanmean(np.asarray(novel_values, dtype=float))) if len(novel_values) else np.nan

    roi_identity_rows.extend([
        {'group': group_name, 'flyID': roi_row.get('flyID'), 'roi_unique_name': roi_name, 'odor_identity': 'CSp', 'response': csp_response},
        {'group': group_name, 'flyID': roi_row.get('flyID'), 'roi_unique_name': roi_name, 'odor_identity': 'CSm', 'response': csm_response},
        {'group': group_name, 'flyID': roi_row.get('flyID'), 'roi_unique_name': roi_name, 'odor_identity': 'Novel', 'response': novel_response},
    ])


roi_identity_df = pd.DataFrame(roi_identity_rows)


if roi_identity_df.empty:
    print("No ROI identity responses available for group-level plotting.")
else:
    plot_group_identity_boxplots(
        roi_identity_subset=roi_identity_df,
        roi_database_subset=roi_database,
        figure_title='ROI averaged responses by group and odor identity',
        output_path=os.path.join(results_dir, f"{safe_filename(container_id)}_roi_group_boxplots.png"),
        normalize_odor_name=normalize_odor_name,
        colors_hex=colors_hex,
        roi_avg_response_by_odor_dict=roi_avg_response_by_odor_dict,
        pvalue_to_stars=pvalue_to_stars,
    )

    for csp_target in ['MCH', 'OCTT']:
        roi_names_for_csp = set(
            roi_database.loc[
                roi_database['CSp'].apply(normalize_odor_name) == csp_target,
                'roi_unique_name',
            ]
        )
        roi_db_subset = roi_database.loc[roi_database['roi_unique_name'].isin(roi_names_for_csp)].copy()
        roi_identity_subset = roi_identity_df.loc[roi_identity_df['roi_unique_name'].isin(roi_names_for_csp)].copy()

        plot_group_identity_boxplots(
            roi_identity_subset=roi_identity_subset,
            roi_database_subset=roi_db_subset,
            figure_title=f"ROI responses by group (CSp = {csp_target})",
            output_path=os.path.join(results_dir, f"{safe_filename(container_id)}_roi_group_boxplots_csp_{safe_filename(csp_target)}.png"),
            normalize_odor_name=normalize_odor_name,
            colors_hex=colors_hex,
            roi_avg_response_by_odor_dict=roi_avg_response_by_odor_dict,
            pvalue_to_stars=pvalue_to_stars,
        )

roi_identity_rows_norm = []

for _, roi_row in roi_database.iterrows():
    roi_name = roi_row['roi_unique_name']
    group_name = roi_row.get('group')

    if pd.isna(group_name):
        continue

    odor_response_dict_norm = roi_avg_response_by_odor_norm_dict.get(roi_name, {})
    if not isinstance(odor_response_dict_norm, dict) or len(odor_response_dict_norm) == 0:
        continue

    normalized_odor_values_norm = {}
    for odor_name, response_value in odor_response_dict_norm.items():
        odor_norm = normalize_odor_name(odor_name)
        if odor_norm is None:
            continue
        normalized_odor_values_norm.setdefault(odor_norm, []).append(response_value)

    csp_norm = normalize_odor_name(roi_row.get('CSp'))
    csm_norm = normalize_odor_name(roi_row.get('CSm'))

    csp_values_norm = normalized_odor_values_norm.get(csp_norm, []) if csp_norm is not None else []
    csm_values_norm = normalized_odor_values_norm.get(csm_norm, []) if csm_norm is not None else []

    novel_values_norm = []
    for odor_norm, values in normalized_odor_values_norm.items():
        if odor_norm in {csp_norm, csm_norm}:
            continue
        novel_values_norm.extend(values)

    csp_response_norm = float(np.nanmean(np.asarray(csp_values_norm, dtype=float))) if len(csp_values_norm) else np.nan
    csm_response_norm = float(np.nanmean(np.asarray(csm_values_norm, dtype=float))) if len(csm_values_norm) else np.nan
    novel_response_norm = float(np.nanmean(np.asarray(novel_values_norm, dtype=float))) if len(novel_values_norm) else np.nan

    roi_identity_rows_norm.extend([
        {'group': group_name, 'flyID': roi_row.get('flyID'), 'roi_unique_name': roi_name, 'odor_identity': 'CSp', 'response': csp_response_norm},
        {'group': group_name, 'flyID': roi_row.get('flyID'), 'roi_unique_name': roi_name, 'odor_identity': 'CSm', 'response': csm_response_norm},
        {'group': group_name, 'flyID': roi_row.get('flyID'), 'roi_unique_name': roi_name, 'odor_identity': 'Novel', 'response': novel_response_norm},
    ])


roi_identity_df_norm = pd.DataFrame(roi_identity_rows_norm)
if roi_identity_df_norm.empty:
    print("No normalized ROI identity responses available for group-level plotting.")
else:
    plot_group_identity_boxplots(
        roi_identity_subset=roi_identity_df_norm,
        roi_database_subset=roi_database,
        figure_title='Normalized ROI averaged responses by group and odor identity',
        output_path=os.path.join(results_dir, f"{safe_filename(container_id)}_normalized_roi_group_boxplots.png"),
        normalize_odor_name=normalize_odor_name,
        colors_hex=colors_hex,
        roi_avg_response_by_odor_dict=roi_avg_response_by_odor_dict,
        pvalue_to_stars=pvalue_to_stars,
    )


# %%
odor_order = ['CSp', 'CSm', 'Novel']
roi_trace_identity_rows = []

for _, roi_row in roi_database.iterrows():
    roi_name = roi_row['roi_unique_name']
    group_name = roi_row.get('group')
    fly_id = roi_row.get('flyID')
    roi_frame_rate_hz = roi_row.get('frame_rate_hz')
    try:
        roi_frame_rate_hz = float(roi_frame_rate_hz)
    except (TypeError, ValueError):
        roi_frame_rate_hz = np.nan
    if (not np.isfinite(roi_frame_rate_hz)) or roi_frame_rate_hz <= 0:
        roi_frame_rate_hz = 1.0

    if pd.isna(group_name) or pd.isna(fly_id):
        continue

    roi_odor_windows = roi_trial_plot_windows.get(roi_name, {})
    if not isinstance(roi_odor_windows, dict) or len(roi_odor_windows) == 0:
        continue

    csp_norm = normalize_odor_name(roi_row.get('CSp'))
    csm_norm = normalize_odor_name(roi_row.get('CSm'))
    novel_norm = next((odor for odor in roi_odor_windows.keys() if odor not in {csp_norm, csm_norm}), None)

    identity_to_odor = {
        'CSp': csp_norm,
        'CSm': csm_norm,
        'Novel': novel_norm,
    }

    for odor_identity in odor_order:
        odor_key = identity_to_odor.get(odor_identity)
        trial_windows = roi_odor_windows.get(odor_key, []) if odor_key is not None else []
        trial_traces = [
            np.asarray(window_info.get('trace', np.array([], dtype=float)), dtype=float)
            for window_info in trial_windows
        ]

        stim_starts = [
            float(window_info.get('stim_start', np.nan))
            for window_info in trial_windows
            if np.isfinite(window_info.get('stim_start', np.nan))
        ]
        stim_ends = [
            float(window_info.get('stim_end', np.nan))
            for window_info in trial_windows
            if np.isfinite(window_info.get('stim_end', np.nan))
        ]

        stim_start_s = np.nan if len(stim_starts) == 0 else float(np.nanmedian(stim_starts) / roi_frame_rate_hz)
        stim_end_s = np.nan if len(stim_ends) == 0 else float(np.nanmedian(stim_ends) / roi_frame_rate_hz)

        roi_avg_trace = nanmean_padded(trial_traces)
        if roi_avg_trace.size == 0:
            continue

        roi_trace_identity_rows.append(
            {
                'group': group_name,
                'flyID': fly_id,
                'roi_unique_name': roi_name,
                'odor_identity': odor_identity,
                'trace': roi_avg_trace,
                'stim_start_s': stim_start_s,
                'stim_end_s': stim_end_s,
            }
        )


if len(roi_trace_identity_rows) == 0:
    print('No traces available for fly-averaged odor identity plotting.')
else:
    fly_trace_rows = []
    for (group_name, fly_id, odor_identity), fly_subset in pd.DataFrame(roi_trace_identity_rows).groupby(['group', 'flyID', 'odor_identity']):
        fly_trace = nanmean_padded(fly_subset['trace'].tolist())
        if fly_trace.size == 0:
            continue
        fly_stim_start_s = float(np.nanmedian(fly_subset['stim_start_s'].to_numpy(dtype=float))) if np.any(np.isfinite(fly_subset['stim_start_s'].to_numpy(dtype=float))) else np.nan
        fly_stim_end_s = float(np.nanmedian(fly_subset['stim_end_s'].to_numpy(dtype=float))) if np.any(np.isfinite(fly_subset['stim_end_s'].to_numpy(dtype=float))) else np.nan
        fly_trace_rows.append(
            {
                'group': group_name,
                'flyID': fly_id,
                'odor_identity': odor_identity,
                'trace': fly_trace,
                'stim_start_s': fly_stim_start_s,
                'stim_end_s': fly_stim_end_s,
            }
        )

    fly_trace_df = pd.DataFrame(fly_trace_rows)

    if fly_trace_df.empty:
        print('No fly-averaged traces available for plotting.')
    else:
        group_order = sorted(fly_trace_df['group'].dropna().unique())
        fig, axes = plt.subplots(len(group_order), 1, figsize=(4, 2.6 * len(group_order)), sharex=False, sharey=True)
        if len(group_order) == 1:
            axes = [axes]

        for axis, group_name in zip(axes, group_order):
            group_subset = fly_trace_df.loc[fly_trace_df['group'] == group_name]
            group_frame_rates = roi_database.loc[roi_database['group'] == group_name, 'frame_rate_hz'].to_numpy(dtype=float)
            group_frame_rates = group_frame_rates[np.isfinite(group_frame_rates) & (group_frame_rates > 0)]
            group_frame_rate_hz = float(np.nanmedian(group_frame_rates)) if group_frame_rates.size > 0 else 1.0
            group_meta = roi_database.loc[roi_database['group'] == group_name, ['CSp', 'CSm']]
            csp_group_odor = normalize_odor_name(group_meta['CSp'].dropna().iloc[0]) if not group_meta['CSp'].dropna().empty else None
            csm_group_odor = normalize_odor_name(group_meta['CSm'].dropna().iloc[0]) if not group_meta['CSm'].dropna().empty else None
            group_roi_names = set(roi_database.loc[roi_database['group'] == group_name, 'roi_unique_name'])
            all_group_odors = {
                normalize_odor_name(k)
                for roi_name_key, odor_dict in roi_avg_response_by_odor_dict.items()
                if roi_name_key in group_roi_names
                if isinstance(odor_dict, dict)
                for k in odor_dict.keys()
            }
            novel_group_odor = next(
                (odor for odor in all_group_odors if odor not in {csp_group_odor, csm_group_odor}),
                None,
            )
            identity_to_odor = {
                'CSp': csp_group_odor,
                'CSm': csm_group_odor,
                'Novel': novel_group_odor,
            }

            for odor_identity in odor_order:
                group_traces = group_subset.loc[group_subset['odor_identity'] == odor_identity, 'trace'].tolist()
                mean_trace, sem_trace = mean_and_sem_padded(group_traces)
                if mean_trace.size == 0:
                    continue

                time_axis = np.arange(mean_trace.size) / group_frame_rate_hz
                trace_color = colors_hex.get(identity_to_odor.get(odor_identity), '#999999')
                axis.plot(
                    time_axis,
                    mean_trace,
                    linewidth=2,
                    color=trace_color,
                    label=f"{odor_identity} (n={len(group_traces)})",
                )
                if sem_trace.size == mean_trace.size:
                    axis.fill_between(time_axis, mean_trace - sem_trace, mean_trace + sem_trace, alpha=0.2, color=trace_color)

                odor_subset = group_subset.loc[group_subset['odor_identity'] == odor_identity]
                odor_starts = odor_subset['stim_start_s'].to_numpy(dtype=float) if 'stim_start_s' in odor_subset else np.array([], dtype=float)
                odor_ends = odor_subset['stim_end_s'].to_numpy(dtype=float) if 'stim_end_s' in odor_subset else np.array([], dtype=float)
                odor_starts = odor_starts[np.isfinite(odor_starts)]
                odor_ends = odor_ends[np.isfinite(odor_ends)]
                if odor_starts.size > 0 and odor_ends.size > 0:
                    stim_start_s = float(np.nanmedian(odor_starts))
                    stim_end_s = float(np.nanmedian(odor_ends))
                    if stim_end_s > stim_start_s:
                        axis.axvspan(stim_start_s, stim_end_s, color=trace_color, alpha=0.08)

            axis.set_title(f"Group: {group_name}")
            axis.set_ylabel('dF/F')
            axis.grid(axis='y', alpha=0.25)
            axis.legend(loc='best', fontsize=8)

        axes[-1].set_xlabel('Time (s)')
        fig.suptitle('Fly-averaged traces by group', y=1.01)
        plt.tight_layout()
        fig.savefig(
            os.path.join(results_dir, f"{safe_filename(container_id)}_fly_averaged_traces_by_group.png"),
            dpi=200,
            bbox_inches='tight',
        )
        plt.show()

        fig_grid, axes_grid = plt.subplots(
            len(group_order),
            len(odor_order),
            figsize=(3.6 * len(odor_order), 2.3 * len(group_order)),
            sharex=False,
            sharey=True,
            squeeze=False,
        )

        for row_idx, group_name in enumerate(group_order):
            group_subset = fly_trace_df.loc[fly_trace_df['group'] == group_name]
            group_frame_rates = roi_database.loc[roi_database['group'] == group_name, 'frame_rate_hz'].to_numpy(dtype=float)
            group_frame_rates = group_frame_rates[np.isfinite(group_frame_rates) & (group_frame_rates > 0)]
            group_frame_rate_hz = float(np.nanmedian(group_frame_rates)) if group_frame_rates.size > 0 else 1.0
            group_meta = roi_database.loc[roi_database['group'] == group_name, ['CSp', 'CSm']]
            csp_group_odor = normalize_odor_name(group_meta['CSp'].dropna().iloc[0]) if not group_meta['CSp'].dropna().empty else None
            csm_group_odor = normalize_odor_name(group_meta['CSm'].dropna().iloc[0]) if not group_meta['CSm'].dropna().empty else None
            group_roi_names = set(roi_database.loc[roi_database['group'] == group_name, 'roi_unique_name'])
            all_group_odors = {
                normalize_odor_name(k)
                for roi_name_key, odor_dict in roi_avg_response_by_odor_dict.items()
                if roi_name_key in group_roi_names
                if isinstance(odor_dict, dict)
                for k in odor_dict.keys()
            }
            novel_group_odor = next(
                (odor for odor in all_group_odors if odor not in {csp_group_odor, csm_group_odor}),
                None,
            )
            identity_to_odor = {
                'CSp': csp_group_odor,
                'CSm': csm_group_odor,
                'Novel': novel_group_odor,
            }

            for col_idx, odor_identity in enumerate(odor_order):
                axis = axes_grid[row_idx, col_idx]
                group_traces = group_subset.loc[group_subset['odor_identity'] == odor_identity, 'trace'].tolist()

                for fly_trace in group_traces:
                    fly_trace = np.asarray(fly_trace, dtype=float)
                    if fly_trace.size == 0:
                        continue
                    axis.plot(np.arange(fly_trace.size) / group_frame_rate_hz, fly_trace, color='0.75', linewidth=0.8, alpha=0.7)

                mean_trace, sem_trace = mean_and_sem_padded(group_traces)
                if mean_trace.size > 0:
                    time_axis = np.arange(mean_trace.size) / group_frame_rate_hz
                    trace_color = colors_hex.get(identity_to_odor.get(odor_identity), '#999999')
                    axis.plot(time_axis, mean_trace, linewidth=2.0, color=trace_color)
                    if sem_trace.size == mean_trace.size:
                        axis.fill_between(time_axis, mean_trace - sem_trace, mean_trace + sem_trace, color=trace_color, alpha=0.25)

                    odor_subset = group_subset.loc[group_subset['odor_identity'] == odor_identity]
                    odor_starts = odor_subset['stim_start_s'].to_numpy(dtype=float) if 'stim_start_s' in odor_subset else np.array([], dtype=float)
                    odor_ends = odor_subset['stim_end_s'].to_numpy(dtype=float) if 'stim_end_s' in odor_subset else np.array([], dtype=float)
                    odor_starts = odor_starts[np.isfinite(odor_starts)]
                    odor_ends = odor_ends[np.isfinite(odor_ends)]
                    if odor_starts.size > 0 and odor_ends.size > 0:
                        stim_start_s = float(np.nanmedian(odor_starts))
                        stim_end_s = float(np.nanmedian(odor_ends))
                        if stim_end_s > stim_start_s:
                            axis.axvspan(stim_start_s, stim_end_s, color=trace_color, alpha=0.08)

                if row_idx == 0:
                    axis.set_title(odor_identity)
                if col_idx == 0:
                    axis.set_ylabel(f"{group_name}\ndF/F")
                if row_idx == len(group_order) - 1:
                    axis.set_xlabel('Time (s)')
                axis.grid(axis='y', alpha=0.25)

        fig_grid.suptitle('Fly traces by group (rows) and odor identity (columns)', y=1.01)
        plt.tight_layout()
        fig_grid.savefig(
            os.path.join(results_dir, f"{safe_filename(container_id)}_fly_traces_grid_by_group_and_odor.png"),
            dpi=200,
            bbox_inches='tight',
        )
        plt.show()

        roi_trace_identity_rows_norm = []
        for _, roi_row in roi_database.iterrows():
            roi_name = roi_row['roi_unique_name']
            group_name = roi_row.get('group')
            fly_id = roi_row.get('flyID')
            roi_frame_rate_hz = roi_row.get('frame_rate_hz')
            try:
                roi_frame_rate_hz = float(roi_frame_rate_hz)
            except (TypeError, ValueError):
                roi_frame_rate_hz = np.nan
            if (not np.isfinite(roi_frame_rate_hz)) or roi_frame_rate_hz <= 0:
                roi_frame_rate_hz = 1.0

            if pd.isna(group_name) or pd.isna(fly_id):
                continue

            roi_odor_windows_norm = roi_trial_plot_windows_norm.get(roi_name, {})
            if not isinstance(roi_odor_windows_norm, dict) or len(roi_odor_windows_norm) == 0:
                continue

            csp_norm = normalize_odor_name(roi_row.get('CSp'))
            csm_norm = normalize_odor_name(roi_row.get('CSm'))
            novel_norm = next((odor for odor in roi_odor_windows_norm.keys() if odor not in {csp_norm, csm_norm}), None)

            identity_to_odor = {
                'CSp': csp_norm,
                'CSm': csm_norm,
                'Novel': novel_norm,
            }

            for odor_identity in odor_order:
                odor_key = identity_to_odor.get(odor_identity)
                trial_windows = roi_odor_windows_norm.get(odor_key, []) if odor_key is not None else []
                trial_traces = [
                    np.asarray(window_info.get('trace', np.array([], dtype=float)), dtype=float)
                    for window_info in trial_windows
                ]

                stim_starts = [
                    float(window_info.get('stim_start', np.nan))
                    for window_info in trial_windows
                    if np.isfinite(window_info.get('stim_start', np.nan))
                ]
                stim_ends = [
                    float(window_info.get('stim_end', np.nan))
                    for window_info in trial_windows
                    if np.isfinite(window_info.get('stim_end', np.nan))
                ]

                stim_start_s = np.nan if len(stim_starts) == 0 else float(np.nanmedian(stim_starts) / roi_frame_rate_hz)
                stim_end_s = np.nan if len(stim_ends) == 0 else float(np.nanmedian(stim_ends) / roi_frame_rate_hz)

                roi_avg_trace = nanmean_padded(trial_traces)
                if roi_avg_trace.size == 0:
                    continue

                roi_trace_identity_rows_norm.append(
                    {
                        'group': group_name,
                        'flyID': fly_id,
                        'roi_unique_name': roi_name,
                        'odor_identity': odor_identity,
                        'trace': roi_avg_trace,
                        'stim_start_s': stim_start_s,
                        'stim_end_s': stim_end_s,
                    }
                )

        if len(roi_trace_identity_rows_norm) == 0:
            print('No normalized traces available for plotting.')
        else:
            fly_trace_rows_norm = []
            for (group_name, fly_id, odor_identity), fly_subset in pd.DataFrame(roi_trace_identity_rows_norm).groupby(['group', 'flyID', 'odor_identity']):
                fly_trace = nanmean_padded(fly_subset['trace'].tolist())
                if fly_trace.size == 0:
                    continue
                fly_stim_start_s = float(np.nanmedian(fly_subset['stim_start_s'].to_numpy(dtype=float))) if np.any(np.isfinite(fly_subset['stim_start_s'].to_numpy(dtype=float))) else np.nan
                fly_stim_end_s = float(np.nanmedian(fly_subset['stim_end_s'].to_numpy(dtype=float))) if np.any(np.isfinite(fly_subset['stim_end_s'].to_numpy(dtype=float))) else np.nan
                fly_trace_rows_norm.append(
                    {
                        'group': group_name,
                        'flyID': fly_id,
                        'odor_identity': odor_identity,
                        'trace': fly_trace,
                        'stim_start_s': fly_stim_start_s,
                        'stim_end_s': fly_stim_end_s,
                    }
                )

            fly_trace_df_norm = pd.DataFrame(fly_trace_rows_norm)

            if fly_trace_df_norm.empty:
                print('No normalized fly-averaged traces available for plotting.')
            else:
                group_order_norm = sorted(fly_trace_df_norm['group'].dropna().unique())

                fig_norm, axes_norm = plt.subplots(
                    len(group_order_norm),
                    1,
                    figsize=(4, 2.6 * len(group_order_norm)),
                    sharex=False,
                    sharey=True,
                )
                if len(group_order_norm) == 1:
                    axes_norm = [axes_norm]

                for axis, group_name in zip(axes_norm, group_order_norm):
                    group_subset = fly_trace_df_norm.loc[fly_trace_df_norm['group'] == group_name]
                    group_frame_rates = roi_database.loc[roi_database['group'] == group_name, 'frame_rate_hz'].to_numpy(dtype=float)
                    group_frame_rates = group_frame_rates[np.isfinite(group_frame_rates) & (group_frame_rates > 0)]
                    group_frame_rate_hz = float(np.nanmedian(group_frame_rates)) if group_frame_rates.size > 0 else 1.0
                    group_meta = roi_database.loc[roi_database['group'] == group_name, ['CSp', 'CSm']]
                    csp_group_odor = normalize_odor_name(group_meta['CSp'].dropna().iloc[0]) if not group_meta['CSp'].dropna().empty else None
                    csm_group_odor = normalize_odor_name(group_meta['CSm'].dropna().iloc[0]) if not group_meta['CSm'].dropna().empty else None
                    group_roi_names = set(roi_database.loc[roi_database['group'] == group_name, 'roi_unique_name'])
                    all_group_odors = {
                        normalize_odor_name(k)
                        for roi_name_key, odor_dict in roi_avg_response_by_odor_dict.items()
                        if roi_name_key in group_roi_names
                        if isinstance(odor_dict, dict)
                        for k in odor_dict.keys()
                    }
                    novel_group_odor = next(
                        (odor for odor in all_group_odors if odor not in {csp_group_odor, csm_group_odor}),
                        None,
                    )
                    identity_to_odor = {
                        'CSp': csp_group_odor,
                        'CSm': csm_group_odor,
                        'Novel': novel_group_odor,
                    }

                    for odor_identity in odor_order:
                        group_traces = group_subset.loc[group_subset['odor_identity'] == odor_identity, 'trace'].tolist()
                        mean_trace, sem_trace = mean_and_sem_padded(group_traces)
                        if mean_trace.size == 0:
                            continue

                        time_axis = np.arange(mean_trace.size) / group_frame_rate_hz
                        trace_color = colors_hex.get(identity_to_odor.get(odor_identity), '#999999')
                        axis.plot(
                            time_axis,
                            mean_trace,
                            linewidth=2,
                            color=trace_color,
                            label=f"{odor_identity} (n={len(group_traces)})",
                        )
                        if sem_trace.size == mean_trace.size:
                            axis.fill_between(time_axis, mean_trace - sem_trace, mean_trace + sem_trace, alpha=0.2, color=trace_color)

                        odor_subset = group_subset.loc[group_subset['odor_identity'] == odor_identity]
                        odor_starts = odor_subset['stim_start_s'].to_numpy(dtype=float) if 'stim_start_s' in odor_subset else np.array([], dtype=float)
                        odor_ends = odor_subset['stim_end_s'].to_numpy(dtype=float) if 'stim_end_s' in odor_subset else np.array([], dtype=float)
                        odor_starts = odor_starts[np.isfinite(odor_starts)]
                        odor_ends = odor_ends[np.isfinite(odor_ends)]
                        if odor_starts.size > 0 and odor_ends.size > 0:
                            stim_start_s = float(np.nanmedian(odor_starts))
                            stim_end_s = float(np.nanmedian(odor_ends))
                            if stim_end_s > stim_start_s:
                                axis.axvspan(stim_start_s, stim_end_s, color=trace_color, alpha=0.08)

                    axis.set_title(f"Group: {group_name}")
                    axis.set_ylabel('Normalized dF/F')
                    axis.grid(axis='y', alpha=0.25)
                    axis.legend(loc='best', fontsize=8)

                axes_norm[-1].set_xlabel('Time (s)')
                fig_norm.suptitle('Normalized fly-averaged traces by group', y=1.01)
                plt.tight_layout()
                fig_norm.savefig(
                    os.path.join(results_dir, f"{safe_filename(container_id)}_normalized_fly_averaged_traces_by_group.png"),
                    dpi=200,
                    bbox_inches='tight',
                )
                plt.show()

                fig_grid_norm, axes_grid_norm = plt.subplots(
                    len(group_order_norm),
                    len(odor_order),
                    figsize=(3.6 * len(odor_order), 2.3 * len(group_order_norm)),
                    sharex=False,
                    sharey=True,
                    squeeze=False,
                )

                for row_idx, group_name in enumerate(group_order_norm):
                    group_subset = fly_trace_df_norm.loc[fly_trace_df_norm['group'] == group_name]
                    group_frame_rates = roi_database.loc[roi_database['group'] == group_name, 'frame_rate_hz'].to_numpy(dtype=float)
                    group_frame_rates = group_frame_rates[np.isfinite(group_frame_rates) & (group_frame_rates > 0)]
                    group_frame_rate_hz = float(np.nanmedian(group_frame_rates)) if group_frame_rates.size > 0 else 1.0
                    group_meta = roi_database.loc[roi_database['group'] == group_name, ['CSp', 'CSm']]
                    csp_group_odor = normalize_odor_name(group_meta['CSp'].dropna().iloc[0]) if not group_meta['CSp'].dropna().empty else None
                    csm_group_odor = normalize_odor_name(group_meta['CSm'].dropna().iloc[0]) if not group_meta['CSm'].dropna().empty else None
                    group_roi_names = set(roi_database.loc[roi_database['group'] == group_name, 'roi_unique_name'])
                    all_group_odors = {
                        normalize_odor_name(k)
                        for roi_name_key, odor_dict in roi_avg_response_by_odor_dict.items()
                        if roi_name_key in group_roi_names
                        if isinstance(odor_dict, dict)
                        for k in odor_dict.keys()
                    }
                    novel_group_odor = next(
                        (odor for odor in all_group_odors if odor not in {csp_group_odor, csm_group_odor}),
                        None,
                    )
                    identity_to_odor = {
                        'CSp': csp_group_odor,
                        'CSm': csm_group_odor,
                        'Novel': novel_group_odor,
                    }

                    for col_idx, odor_identity in enumerate(odor_order):
                        axis = axes_grid_norm[row_idx, col_idx]
                        group_traces = group_subset.loc[group_subset['odor_identity'] == odor_identity, 'trace'].tolist()

                        for fly_trace in group_traces:
                            fly_trace = np.asarray(fly_trace, dtype=float)
                            if fly_trace.size == 0:
                                continue
                            axis.plot(np.arange(fly_trace.size) / group_frame_rate_hz, fly_trace, color='0.75', linewidth=0.8, alpha=0.7)

                        mean_trace, sem_trace = mean_and_sem_padded(group_traces)
                        if mean_trace.size > 0:
                            time_axis = np.arange(mean_trace.size) / group_frame_rate_hz
                            trace_color = colors_hex.get(identity_to_odor.get(odor_identity), '#999999')
                            axis.plot(time_axis, mean_trace, linewidth=2.0, color=trace_color)
                            if sem_trace.size == mean_trace.size:
                                axis.fill_between(time_axis, mean_trace - sem_trace, mean_trace + sem_trace, color=trace_color, alpha=0.25)

                            odor_subset = group_subset.loc[group_subset['odor_identity'] == odor_identity]
                            odor_starts = odor_subset['stim_start_s'].to_numpy(dtype=float) if 'stim_start_s' in odor_subset else np.array([], dtype=float)
                            odor_ends = odor_subset['stim_end_s'].to_numpy(dtype=float) if 'stim_end_s' in odor_subset else np.array([], dtype=float)
                            odor_starts = odor_starts[np.isfinite(odor_starts)]
                            odor_ends = odor_ends[np.isfinite(odor_ends)]
                            if odor_starts.size > 0 and odor_ends.size > 0:
                                stim_start_s = float(np.nanmedian(odor_starts))
                                stim_end_s = float(np.nanmedian(odor_ends))
                                if stim_end_s > stim_start_s:
                                    axis.axvspan(stim_start_s, stim_end_s, color=trace_color, alpha=0.08)

                        if row_idx == 0:
                            axis.set_title(odor_identity)
                        if col_idx == 0:
                            axis.set_ylabel(f"{group_name}\nNormalized dF/F")
                        if row_idx == len(group_order_norm) - 1:
                            axis.set_xlabel('Time (s)')
                        axis.grid(axis='y', alpha=0.25)

                fig_grid_norm.suptitle('Normalized fly traces by group (rows) and odor identity (columns)', y=1.01)
                plt.tight_layout()
                fig_grid_norm.savefig(
                    os.path.join(results_dir, f"{safe_filename(container_id)}_normalized_fly_traces_grid_by_group_and_odor.png"),
                    dpi=200,
                    bbox_inches='tight',
                )
                plt.show()


 # %%
