#%%
import os
import re
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import integrate
from scipy.stats import mannwhitneyu, kruskal, f_oneway, wilcoxon as wilcoxon_paired
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
container_id = '2025_10_Gamma1_CC_extinction'

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
roi_trial_post_stim_response_dict = {}
roi_avg_post_stim_response_by_odor_dict = {}
roi_avg_post_stim_response_by_odor_norm_dict = {}
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
        trial_post_stim_response_by_odor = {}
        avg_post_stim_response_by_odor = {}
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
                trial_post_stim_responses = []
                for trial_index, trial_trace in enumerate(trial_traces):
                    trace_array = np.asarray(trial_trace, dtype=float)
                    if trace_array.size == 0:
                        trial_responses.append(np.nan)
                        trial_post_stim_responses.append(np.nan)
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
                        trial_post_stim_responses.append(np.nan)
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
                    
                    # Mean of stimulus segment minus mean of context segment as response value
                    stim_mean = np.nanmean(stim_segment)
                    context_mean = np.nanmean(context_segment)

                    # AUC calculation
                    stim_auc = integrate.trapezoid(stim_segment)
                    context_auc = integrate.trapezoid(context_segment)

                    # Response calculation using mean
                    response_value = float(stim_mean - context_mean)

                    # Response calculation using AUC
                    # response_value = float(stim_auc - context_auc)

                    # Post-stimulus response: mean of trace after stim end, baseline-subtracted
                    post_stim_segment = trace_array[stim_end:]
                    post_stim_segment = post_stim_segment[np.isfinite(post_stim_segment)]
                    if post_stim_segment.size == 0:
                        post_stim_response_value = np.nan
                    else:
                        post_stim_response_value = float(np.nanmean(post_stim_segment) - context_mean)

                    trace_array_adjusted = trace_array - context_mean

                    p_value = np.nan
                    try:
                        test_result = mannwhitneyu(stim_segment, context_segment, alternative='two-sided')
                        p_value = float(test_result.pvalue)
                    except ValueError:
                        p_value = np.nan

                    is_pass_trial = bool(np.isfinite(p_value) and p_value < alpha_response)

                    trial_responses.append(response_value)
                    trial_post_stim_responses.append(post_stim_response_value)
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
                trial_post_stim_response_by_odor[stimulus_key] = trial_post_stim_responses
                all_trial_pass_flags.extend(trial_is_pass)

                if len(trial_responses) == 0:
                    avg_response_by_odor[stimulus_key] = np.nan
                else:
                    avg_response_by_odor[stimulus_key] = float(np.nanmean(np.asarray(trial_responses, dtype=float)))

                if len(trial_post_stim_responses) == 0:
                    avg_post_stim_response_by_odor[stimulus_key] = np.nan
                else:
                    avg_post_stim_response_by_odor[stimulus_key] = float(np.nanmean(np.asarray(trial_post_stim_responses, dtype=float)))

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
        trial_post_stim_response_by_odor_norm = {}
        avg_post_stim_response_by_odor_norm = {}
        for odor_key, norm_trial_windows_list in trial_plot_windows_norm_by_odor.items():
            norm_trial_responses = []
            norm_trial_post_stim_responses = []
            for window_info in norm_trial_windows_list:
                trace_values = np.asarray(window_info.get('trace', np.array([], dtype=float)), dtype=float)
                if trace_values.size == 0:
                    norm_trial_responses.append(np.nan)
                    norm_trial_post_stim_responses.append(np.nan)
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
                    norm_trial_post_stim_responses.append(np.nan)
                    continue

                norm_trial_responses.append(float(np.nanmean(stim_segment) - np.nanmean(context_segment)))

                post_stim_segment_norm = trace_values[stim_end:]
                post_stim_segment_norm = post_stim_segment_norm[np.isfinite(post_stim_segment_norm)]
                if post_stim_segment_norm.size == 0:
                    norm_trial_post_stim_responses.append(np.nan)
                else:
                    norm_trial_post_stim_responses.append(float(np.nanmean(post_stim_segment_norm) - np.nanmean(context_segment)))

            trial_response_by_odor_norm[odor_key] = norm_trial_responses
            if len(norm_trial_responses) == 0:
                avg_response_by_odor_norm[odor_key] = np.nan
            else:
                avg_response_by_odor_norm[odor_key] = float(np.nanmean(np.asarray(norm_trial_responses, dtype=float)))

            trial_post_stim_response_by_odor_norm[odor_key] = norm_trial_post_stim_responses
            if len(norm_trial_post_stim_responses) == 0:
                avg_post_stim_response_by_odor_norm[odor_key] = np.nan
            else:
                avg_post_stim_response_by_odor_norm[odor_key] = float(np.nanmean(np.asarray(norm_trial_post_stim_responses, dtype=float)))

        roi_trial_plot_windows_norm[roi_unique_name] = trial_plot_windows_norm_by_odor
        roi_trial_response_norm_dict[roi_unique_name] = trial_response_by_odor_norm
        roi_avg_response_by_odor_dict[roi_unique_name] = avg_response_by_odor
        roi_avg_response_by_odor_norm_dict[roi_unique_name] = avg_response_by_odor_norm
        roi_trial_post_stim_response_dict[roi_unique_name] = trial_post_stim_response_by_odor
        roi_avg_post_stim_response_by_odor_dict[roi_unique_name] = avg_post_stim_response_by_odor
        roi_avg_post_stim_response_by_odor_norm_dict[roi_unique_name] = avg_post_stim_response_by_odor_norm

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
print(f"Built ROI post-stimulus response dictionary with {len(roi_trial_post_stim_response_dict)} entries.")
print(f"Built ROI post-stimulus odor-averaged response dictionary with {len(roi_avg_post_stim_response_by_odor_dict)} entries.")
print(f"Built ROI normalized post-stimulus odor-averaged response dictionary with {len(roi_avg_post_stim_response_by_odor_norm_dict)} entries.")

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


# print(f"Plotting responsive flies: {len(responsive_fly_ids)}")
# plot_fly_trial_panels(
#     responsive_fly_ids,
#     status_label='PASS FILTER',
#     roi_database_all=roi_database_all,
#     roi_trial_plot_windows=roi_trial_plot_windows,
#     pass_fail_dir=pass_fail_dir,
#     normalize_odor_name=normalize_odor_name,
#     safe_filename=safe_filename,
# )

# print(f"Plotting non-responsive flies: {len(nonresponsive_fly_ids)}")
# plot_fly_trial_panels(
#     nonresponsive_fly_ids,
#     status_label='FAIL FILTER',
#     roi_database_all=roi_database_all,
#     roi_trial_plot_windows=roi_trial_plot_windows,
#     pass_fail_dir=pass_fail_dir,
#     normalize_odor_name=normalize_odor_name,
#     safe_filename=safe_filename,
# )

roi_database.head()

   

# %%
colors_hex = {'MCH': '#d7191c', 'OCTT': "#e6b800", "IAA": '#a6d96a', 'CSp': '#d95f02', 'CSm': "#7570b3", 'Novel': '#999999'}
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

# %%
# --- Statistical analysis for response boxplots ---
# Available test keys for plot annotation:
#   'wilcoxon_p'          — paired Wilcoxon (within-ROI), uncorrected
#   'wilcoxon_p_bonf'     — paired Wilcoxon, Bonferroni-corrected
#   'mannwhitney_p'       — Mann-Whitney U (independent samples), uncorrected
#   'mannwhitney_p_bonf'  — Mann-Whitney U, Bonferroni-corrected
annotation_test_key = 'wilcoxon_p_bonf' # HERE PICK WHICH ONE YOU WANT TO BE PLOTTED

_odor_pairs = [('CSp', 'CSm')]
_n_comparisons = len(_odor_pairs)  # Bonferroni correction factor

# Build subsets: all groups combined + per-CSp-odor subsets
_bp_subsets = [
    (
        'all',
        roi_identity_df,
        roi_database,
        None,
        'ROI averaged responses by group and odor identity',
        os.path.join(results_dir, f"{safe_filename(container_id)}_roi_group_boxplots.png"),
    ),
]
for _cspt in ['MCH', 'OCTT', 'IAA']:
    _names_csp = set(roi_database.loc[roi_database['CSp'].apply(normalize_odor_name) == _cspt, 'roi_unique_name'])
    _bp_subsets.append((
        _cspt,
        roi_identity_df.loc[roi_identity_df['roi_unique_name'].isin(_names_csp)].copy(),
        roi_database.loc[roi_database['roi_unique_name'].isin(_names_csp)].copy(),
        _cspt,
        f"ROI responses by group (CSp = {_cspt})",
        os.path.join(results_dir, f"{safe_filename(container_id)}_roi_group_boxplots_csp_{safe_filename(_cspt)}.png"),
    ))

bp_stats_results = {}
for _lbl, _id_df, _db_df, _csp_od, _ttl, _opath in _bp_subsets:
    if _id_df.empty:
        continue
    print(f"\n{'='*60}\nStats: {_ttl}")
    _stats_ann, _omni = {}, {}
    for _grp, _gdf in _id_df.groupby('group'):
        _arrs = {od: _gdf.loc[_gdf['odor_identity'] == od, 'response'].dropna().to_numpy(dtype=float) for od in ['CSp', 'CSm', 'Novel']}

        # Omnibus: Kruskal-Wallis (non-parametric)
        _kw_stat, _kw_p = np.nan, np.nan
        if all(a.size >= 2 for a in _arrs.values()):
            try: _kw_stat, _kw_p = kruskal(*_arrs.values())
            except ValueError: pass

        # Omnibus: one-way ANOVA (parametric)
        _an_stat, _an_p = np.nan, np.nan
        if all(a.size >= 2 for a in _arrs.values()):
            try: _an_stat, _an_p = f_oneway(*_arrs.values())
            except ValueError: pass

        print(f"\n  Group: {_grp}")
        print(f"    Kruskal-Wallis: H={_kw_stat:.3g}, p={_kw_p:.4g}  {pvalue_to_stars(_kw_p)}")
        print(f"    One-way ANOVA:  F={_an_stat:.3g}, p={_an_p:.4g}  {pvalue_to_stars(_an_p)}")
        _omni[_grp] = {
            'kruskal': {'statistic': float(_kw_stat), 'pvalue': float(_kw_p)},
            'anova':   {'statistic': float(_an_stat), 'pvalue': float(_an_p)},
        }

        # Post-hoc pairwise tests
        _pv = _gdf.pivot_table(index='roi_unique_name', columns='odor_identity', values='response', aggfunc='mean')
        _pair_res = {}
        for _oa, _ob in _odor_pairs:
            # Paired Wilcoxon signed-rank (within-ROI)
            _wil_p = np.nan
            if _oa in _pv.columns and _ob in _pv.columns:
                _pd2 = _pv[[_oa, _ob]].dropna()
                if len(_pd2) >= 2:
                    try: _wil_p = float(wilcoxon_paired(_pd2[_oa].to_numpy(dtype=float), _pd2[_ob].to_numpy(dtype=float), alternative='two-sided', zero_method='wilcox').pvalue)
                    except ValueError: pass
            # Mann-Whitney U (independent samples)
            _mwu_p = np.nan
            if _arrs[_oa].size >= 2 and _arrs[_ob].size >= 2:
                try: _mwu_p = float(mannwhitneyu(_arrs[_oa], _arrs[_ob], alternative='two-sided').pvalue)
                except ValueError: pass
            _wil_bonf = min(1.0, _wil_p * _n_comparisons) if np.isfinite(_wil_p) else np.nan
            _mwu_bonf = min(1.0, _mwu_p * _n_comparisons) if np.isfinite(_mwu_p) else np.nan
            _pair_res[(_oa, _ob)] = {
                'wilcoxon_p':        _wil_p,    'wilcoxon_p_bonf':   _wil_bonf,
                'mannwhitney_p':     _mwu_p,    'mannwhitney_p_bonf': _mwu_bonf,
            }
            print(f"    {_oa} vs {_ob}:")
            print(f"      Wilcoxon paired:  p={_wil_p:.4g}  [ Bonferroni: p={_wil_bonf:.4g}  {pvalue_to_stars(_wil_bonf)} ]")
            print(f"      Mann-Whitney U:   p={_mwu_p:.4g}  [ Bonferroni: p={_mwu_bonf:.4g}  {pvalue_to_stars(_mwu_bonf)} ]")
        _stats_ann[_grp] = _pair_res
    bp_stats_results[_lbl] = {'annotations': _stats_ann, 'omnibus': _omni}

# %%
# --- Boxplots (non-normalized) ---
for _lbl, _id_df, _db_df, _csp_od, _ttl, _opath in _bp_subsets:
    if _id_df.empty:
        print(f"No data for: {_ttl}")
        continue
    _grp_ann = {
        gname: {pair: ps[annotation_test_key] for pair, ps in gstats.items()}
        for gname, gstats in bp_stats_results.get(_lbl, {}).get('annotations', {}).items()
    }
    plot_group_identity_boxplots(
        roi_identity_subset=_id_df,
        roi_database_subset=_db_df,
        figure_title=_ttl,
        output_path=_opath,
        normalize_odor_name=normalize_odor_name,
        colors_hex=colors_hex,
        pvalue_to_stars=pvalue_to_stars,
        csp_odor=_csp_od,
        stats_annotations=_grp_ann,
        omnibus_stats=bp_stats_results.get(_lbl, {}).get('omnibus', {}),
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

        normalized_odor_values_norm.setdefault(odor_name, []).append(response_value)

    csp_name = roi_row.get('CSp')
    csm_name = roi_row.get('CSm')

    csp_values_norm = normalized_odor_values_norm.get(csp_name, []) if csp_name is not None else []
    csm_values_norm = normalized_odor_values_norm.get(csm_name, []) if csm_name is not None else []

    novel_values_norm = []
    for odor_name, values in normalized_odor_values_norm.items():
        if odor_name in {csp_name, csm_name}:
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

# %%
# --- Statistical analysis for normalized boxplots ---
bp_norm_stats = {'annotations': {}, 'omnibus': {}}
if not roi_identity_df_norm.empty:
    print(f"\n{'='*60}\nStats: Normalized ROI averaged responses by group and odor identity")
    for _grp, _gdf in roi_identity_df_norm.groupby('group'):
        _arrs = {od: _gdf.loc[_gdf['odor_identity'] == od, 'response'].dropna().to_numpy(dtype=float) for od in ['CSp', 'CSm', 'Novel']}

        # Omnibus: Kruskal-Wallis
        _kw_stat, _kw_p = np.nan, np.nan
        if all(a.size >= 2 for a in _arrs.values()):
            try: _kw_stat, _kw_p = kruskal(*_arrs.values())
            except ValueError: pass

        # Omnibus: one-way ANOVA
        _an_stat, _an_p = np.nan, np.nan
        if all(a.size >= 2 for a in _arrs.values()):
            try: _an_stat, _an_p = f_oneway(*_arrs.values())
            except ValueError: pass

        print(f"\n  Group: {_grp}")
        print(f"    Kruskal-Wallis: H={_kw_stat:.3g}, p={_kw_p:.4g}  {pvalue_to_stars(_kw_p)}")
        print(f"    One-way ANOVA:  F={_an_stat:.3g}, p={_an_p:.4g}  {pvalue_to_stars(_an_p)}")
        bp_norm_stats['omnibus'][_grp] = {
            'kruskal': {'statistic': float(_kw_stat), 'pvalue': float(_kw_p)},
            'anova':   {'statistic': float(_an_stat), 'pvalue': float(_an_p)},
        }

        _pv = _gdf.pivot_table(index='roi_unique_name', columns='odor_identity', values='response', aggfunc='mean')
        _pair_res = {}
        for _oa, _ob in _odor_pairs:
            _wil_p = np.nan
            if _oa in _pv.columns and _ob in _pv.columns:
                _pd2 = _pv[[_oa, _ob]].dropna()
                if len(_pd2) >= 2:
                    try: _wil_p = float(wilcoxon_paired(_pd2[_oa].to_numpy(dtype=float), _pd2[_ob].to_numpy(dtype=float), alternative='two-sided', zero_method='wilcox').pvalue)
                    except ValueError: pass
            _mwu_p = np.nan
            if _arrs[_oa].size >= 2 and _arrs[_ob].size >= 2:
                try: _mwu_p = float(mannwhitneyu(_arrs[_oa], _arrs[_ob], alternative='two-sided').pvalue)
                except ValueError: pass
            _wil_bonf = min(1.0, _wil_p * _n_comparisons) if np.isfinite(_wil_p) else np.nan
            _mwu_bonf = min(1.0, _mwu_p * _n_comparisons) if np.isfinite(_mwu_p) else np.nan
            _pair_res[(_oa, _ob)] = {
                'wilcoxon_p':        _wil_p,    'wilcoxon_p_bonf':   _wil_bonf,
                'mannwhitney_p':     _mwu_p,    'mannwhitney_p_bonf': _mwu_bonf,
            }
            print(f"    {_oa} vs {_ob}:")
            print(f"      Wilcoxon paired:  p={_wil_p:.4g}  [ Bonferroni: p={_wil_bonf:.4g}  {pvalue_to_stars(_wil_bonf)} ]")
            print(f"      Mann-Whitney U:   p={_mwu_p:.4g}  [ Bonferroni: p={_mwu_bonf:.4g}  {pvalue_to_stars(_mwu_bonf)} ]")
        bp_norm_stats['annotations'][_grp] = _pair_res

# %%
# --- Boxplots (normalized) ---
if not roi_identity_df_norm.empty:
    _grp_ann_norm = {
        gname: {pair: ps[annotation_test_key] for pair, ps in gstats.items()}
        for gname, gstats in bp_norm_stats['annotations'].items()
    }
    plot_group_identity_boxplots(
        roi_identity_subset=roi_identity_df_norm,
        roi_database_subset=roi_database,
        figure_title='Normalized ROI averaged responses by group and odor identity',
        output_path=os.path.join(results_dir, f"{safe_filename(container_id)}_normalized_roi_group_boxplots.png"),
        normalize_odor_name=normalize_odor_name,
        colors_hex=colors_hex,
        pvalue_to_stars=pvalue_to_stars,
        stats_annotations=_grp_ann_norm,
        omnibus_stats=bp_norm_stats['omnibus'],
    )


# %%
# --- Post-stimulus response analysis ---
roi_post_stim_identity_rows = []

for _, roi_row in roi_database.iterrows():
    roi_name = roi_row['roi_unique_name']
    group_name = roi_row.get('group')

    if pd.isna(group_name):
        continue

    odor_post_stim_dict = roi_avg_post_stim_response_by_odor_dict.get(roi_name, {})
    if not isinstance(odor_post_stim_dict, dict) or len(odor_post_stim_dict) == 0:
        continue

    normalized_odor_values = {}
    for odor_name, response_value in odor_post_stim_dict.items():
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

    roi_post_stim_identity_rows.extend([
        {'group': group_name, 'flyID': roi_row.get('flyID'), 'roi_unique_name': roi_name, 'odor_identity': 'CSp', 'response': csp_response},
        {'group': group_name, 'flyID': roi_row.get('flyID'), 'roi_unique_name': roi_name, 'odor_identity': 'CSm', 'response': csm_response},
        {'group': group_name, 'flyID': roi_row.get('flyID'), 'roi_unique_name': roi_name, 'odor_identity': 'Novel', 'response': novel_response},
    ])

roi_post_stim_identity_df = pd.DataFrame(roi_post_stim_identity_rows)

# %%
# --- Statistical analysis for post-stimulus response boxplots ---
_ps_bp_subsets = [
    (
        'all',
        roi_post_stim_identity_df,
        roi_database,
        None,
        'Post-stimulus ROI averaged responses by group and odor identity',
        os.path.join(results_dir, f"{safe_filename(container_id)}_roi_group_boxplots_post_stim.png"),
    ),
]
for _cspt in ['MCH', 'OCTT', 'IAA']:
    _names_csp = set(roi_database.loc[roi_database['CSp'].apply(normalize_odor_name) == _cspt, 'roi_unique_name'])
    _ps_bp_subsets.append((
        _cspt,
        roi_post_stim_identity_df.loc[roi_post_stim_identity_df['roi_unique_name'].isin(_names_csp)].copy(),
        roi_database.loc[roi_database['roi_unique_name'].isin(_names_csp)].copy(),
        _cspt,
        f"Post-stimulus ROI responses by group (CSp = {_cspt})",
        os.path.join(results_dir, f"{safe_filename(container_id)}_roi_group_boxplots_post_stim_csp_{safe_filename(_cspt)}.png"),
    ))

bp_post_stim_stats_results = {}
for _lbl, _id_df, _db_df, _csp_od, _ttl, _opath in _ps_bp_subsets:
    if _id_df.empty:
        continue
    print(f"\n{'='*60}\nStats: {_ttl}")
    _stats_ann, _omni = {}, {}
    for _grp, _gdf in _id_df.groupby('group'):
        _arrs = {od: _gdf.loc[_gdf['odor_identity'] == od, 'response'].dropna().to_numpy(dtype=float) for od in ['CSp', 'CSm', 'Novel']}

        _kw_stat, _kw_p = np.nan, np.nan
        if all(a.size >= 2 for a in _arrs.values()):
            try: _kw_stat, _kw_p = kruskal(*_arrs.values())
            except ValueError: pass

        _an_stat, _an_p = np.nan, np.nan
        if all(a.size >= 2 for a in _arrs.values()):
            try: _an_stat, _an_p = f_oneway(*_arrs.values())
            except ValueError: pass

        print(f"\n  Group: {_grp}")
        print(f"    Kruskal-Wallis: H={_kw_stat:.3g}, p={_kw_p:.4g}  {pvalue_to_stars(_kw_p)}")
        print(f"    One-way ANOVA:  F={_an_stat:.3g}, p={_an_p:.4g}  {pvalue_to_stars(_an_p)}")
        _omni[_grp] = {
            'kruskal': {'statistic': float(_kw_stat), 'pvalue': float(_kw_p)},
            'anova':   {'statistic': float(_an_stat), 'pvalue': float(_an_p)},
        }

        _pv = _gdf.pivot_table(index='roi_unique_name', columns='odor_identity', values='response', aggfunc='mean')
        _pair_res = {}
        for _oa, _ob in _odor_pairs:
            _wil_p = np.nan
            if _oa in _pv.columns and _ob in _pv.columns:
                _pd2 = _pv[[_oa, _ob]].dropna()
                if len(_pd2) >= 2:
                    try: _wil_p = float(wilcoxon_paired(_pd2[_oa].to_numpy(dtype=float), _pd2[_ob].to_numpy(dtype=float), alternative='two-sided', zero_method='wilcox').pvalue)
                    except ValueError: pass
            _mwu_p = np.nan
            if _arrs[_oa].size >= 2 and _arrs[_ob].size >= 2:
                try: _mwu_p = float(mannwhitneyu(_arrs[_oa], _arrs[_ob], alternative='two-sided').pvalue)
                except ValueError: pass
            _wil_bonf = min(1.0, _wil_p * _n_comparisons) if np.isfinite(_wil_p) else np.nan
            _mwu_bonf = min(1.0, _mwu_p * _n_comparisons) if np.isfinite(_mwu_p) else np.nan
            _pair_res[(_oa, _ob)] = {
                'wilcoxon_p':        _wil_p,    'wilcoxon_p_bonf':   _wil_bonf,
                'mannwhitney_p':     _mwu_p,    'mannwhitney_p_bonf': _mwu_bonf,
            }
            print(f"    {_oa} vs {_ob}:")
            print(f"      Wilcoxon paired:  p={_wil_p:.4g}  [ Bonferroni: p={_wil_bonf:.4g}  {pvalue_to_stars(_wil_bonf)} ]")
            print(f"      Mann-Whitney U:   p={_mwu_p:.4g}  [ Bonferroni: p={_mwu_bonf:.4g}  {pvalue_to_stars(_mwu_bonf)} ]")
        _stats_ann[_grp] = _pair_res
    bp_post_stim_stats_results[_lbl] = {'annotations': _stats_ann, 'omnibus': _omni}

# %%
# --- Boxplots (post-stimulus, non-normalized) ---
for _lbl, _id_df, _db_df, _csp_od, _ttl, _opath in _ps_bp_subsets:
    if _id_df.empty:
        print(f"No data for: {_ttl}")
        continue
    _grp_ann = {
        gname: {pair: ps[annotation_test_key] for pair, ps in gstats.items()}
        for gname, gstats in bp_post_stim_stats_results.get(_lbl, {}).get('annotations', {}).items()
    }
    plot_group_identity_boxplots(
        roi_identity_subset=_id_df,
        roi_database_subset=_db_df,
        figure_title=_ttl,
        output_path=_opath,
        normalize_odor_name=normalize_odor_name,
        colors_hex=colors_hex,
        pvalue_to_stars=pvalue_to_stars,
        csp_odor=_csp_od,
        stats_annotations=_grp_ann,
        omnibus_stats=bp_post_stim_stats_results.get(_lbl, {}).get('omnibus', {}),
    )

# %%
# --- Post-stimulus normalized response analysis ---
roi_post_stim_identity_rows_norm = []

for _, roi_row in roi_database.iterrows():
    roi_name = roi_row['roi_unique_name']
    group_name = roi_row.get('group')

    if pd.isna(group_name):
        continue

    odor_post_stim_dict_norm = roi_avg_post_stim_response_by_odor_norm_dict.get(roi_name, {})
    if not isinstance(odor_post_stim_dict_norm, dict) or len(odor_post_stim_dict_norm) == 0:
        continue

    normalized_odor_values_norm = {}
    for odor_name, response_value in odor_post_stim_dict_norm.items():
        normalized_odor_values_norm.setdefault(odor_name, []).append(response_value)

    csp_name = roi_row.get('CSp')
    csm_name = roi_row.get('CSm')

    csp_values_norm = normalized_odor_values_norm.get(csp_name, []) if csp_name is not None else []
    csm_values_norm = normalized_odor_values_norm.get(csm_name, []) if csm_name is not None else []

    novel_values_norm = []
    for odor_name, values in normalized_odor_values_norm.items():
        if odor_name in {csp_name, csm_name}:
            continue
        novel_values_norm.extend(values)

    csp_response_norm = float(np.nanmean(np.asarray(csp_values_norm, dtype=float))) if len(csp_values_norm) else np.nan
    csm_response_norm = float(np.nanmean(np.asarray(csm_values_norm, dtype=float))) if len(csm_values_norm) else np.nan
    novel_response_norm = float(np.nanmean(np.asarray(novel_values_norm, dtype=float))) if len(novel_values_norm) else np.nan

    roi_post_stim_identity_rows_norm.extend([
        {'group': group_name, 'flyID': roi_row.get('flyID'), 'roi_unique_name': roi_name, 'odor_identity': 'CSp', 'response': csp_response_norm},
        {'group': group_name, 'flyID': roi_row.get('flyID'), 'roi_unique_name': roi_name, 'odor_identity': 'CSm', 'response': csm_response_norm},
        {'group': group_name, 'flyID': roi_row.get('flyID'), 'roi_unique_name': roi_name, 'odor_identity': 'Novel', 'response': novel_response_norm},
    ])

roi_post_stim_identity_df_norm = pd.DataFrame(roi_post_stim_identity_rows_norm)

# %%
# --- Statistical analysis for normalized post-stimulus response boxplots ---
bp_post_stim_norm_stats = {'annotations': {}, 'omnibus': {}}
if not roi_post_stim_identity_df_norm.empty:
    print(f"\n{'='*60}\nStats: Normalized post-stimulus ROI averaged responses by group and odor identity")
    for _grp, _gdf in roi_post_stim_identity_df_norm.groupby('group'):
        _arrs = {od: _gdf.loc[_gdf['odor_identity'] == od, 'response'].dropna().to_numpy(dtype=float) for od in ['CSp', 'CSm', 'Novel']}

        _kw_stat, _kw_p = np.nan, np.nan
        if all(a.size >= 2 for a in _arrs.values()):
            try: _kw_stat, _kw_p = kruskal(*_arrs.values())
            except ValueError: pass

        _an_stat, _an_p = np.nan, np.nan
        if all(a.size >= 2 for a in _arrs.values()):
            try: _an_stat, _an_p = f_oneway(*_arrs.values())
            except ValueError: pass

        print(f"\n  Group: {_grp}")
        print(f"    Kruskal-Wallis: H={_kw_stat:.3g}, p={_kw_p:.4g}  {pvalue_to_stars(_kw_p)}")
        print(f"    One-way ANOVA:  F={_an_stat:.3g}, p={_an_p:.4g}  {pvalue_to_stars(_an_p)}")
        bp_post_stim_norm_stats['omnibus'][_grp] = {
            'kruskal': {'statistic': float(_kw_stat), 'pvalue': float(_kw_p)},
            'anova':   {'statistic': float(_an_stat), 'pvalue': float(_an_p)},
        }

        _pv = _gdf.pivot_table(index='roi_unique_name', columns='odor_identity', values='response', aggfunc='mean')
        _pair_res = {}
        for _oa, _ob in _odor_pairs:
            _wil_p = np.nan
            if _oa in _pv.columns and _ob in _pv.columns:
                _pd2 = _pv[[_oa, _ob]].dropna()
                if len(_pd2) >= 2:
                    try: _wil_p = float(wilcoxon_paired(_pd2[_oa].to_numpy(dtype=float), _pd2[_ob].to_numpy(dtype=float), alternative='two-sided', zero_method='wilcox').pvalue)
                    except ValueError: pass
            _mwu_p = np.nan
            if _arrs[_oa].size >= 2 and _arrs[_ob].size >= 2:
                try: _mwu_p = float(mannwhitneyu(_arrs[_oa], _arrs[_ob], alternative='two-sided').pvalue)
                except ValueError: pass
            _wil_bonf = min(1.0, _wil_p * _n_comparisons) if np.isfinite(_wil_p) else np.nan
            _mwu_bonf = min(1.0, _mwu_p * _n_comparisons) if np.isfinite(_mwu_p) else np.nan
            _pair_res[(_oa, _ob)] = {
                'wilcoxon_p':        _wil_p,    'wilcoxon_p_bonf':   _wil_bonf,
                'mannwhitney_p':     _mwu_p,    'mannwhitney_p_bonf': _mwu_bonf,
            }
            print(f"    {_oa} vs {_ob}:")
            print(f"      Wilcoxon paired:  p={_wil_p:.4g}  [ Bonferroni: p={_wil_bonf:.4g}  {pvalue_to_stars(_wil_bonf)} ]")
            print(f"      Mann-Whitney U:   p={_mwu_p:.4g}  [ Bonferroni: p={_mwu_bonf:.4g}  {pvalue_to_stars(_mwu_bonf)} ]")
        bp_post_stim_norm_stats['annotations'][_grp] = _pair_res

# %%
# --- Boxplots (post-stimulus, normalized) ---
if not roi_post_stim_identity_df_norm.empty:
    _grp_ann_ps_norm = {
        gname: {pair: ps[annotation_test_key] for pair, ps in gstats.items()}
        for gname, gstats in bp_post_stim_norm_stats['annotations'].items()
    }
    plot_group_identity_boxplots(
        roi_identity_subset=roi_post_stim_identity_df_norm,
        roi_database_subset=roi_database,
        figure_title='Normalized post-stimulus ROI averaged responses by group and odor identity',
        output_path=os.path.join(results_dir, f"{safe_filename(container_id)}_normalized_roi_group_boxplots_post_stim.png"),
        normalize_odor_name=normalize_odor_name,
        colors_hex=colors_hex,
        pvalue_to_stars=pvalue_to_stars,
        stats_annotations=_grp_ann_ps_norm,
        omnibus_stats=bp_post_stim_norm_stats['omnibus'],
    )


# %%
# --- Response distribution histograms ---
# Subsets: all groups combined, then per-CSp-odor
odor_order = ['CSp', 'CSm', 'Novel']

_hist_subsets = [
    (
        'all',
        roi_identity_df_norm,
        None,
        'ROI response distributions by group and odor identity',
        os.path.join(results_dir, f"{safe_filename(container_id)}_response_histograms.png"),
    ),
]
for _cspt in ['MCH', 'OCTT', 'IAA']:
    _names_csp_h = set(roi_database.loc[roi_database['CSp'].apply(normalize_odor_name) == _cspt, 'roi_unique_name'])
    _hist_subsets.append((
        _cspt,
        roi_identity_df_norm.loc[roi_identity_df_norm['roi_unique_name'].isin(_names_csp_h)].copy(),
        _cspt,
        f"ROI response distributions (CSp = {_cspt})",
        os.path.join(results_dir, f"{safe_filename(container_id)}_response_histograms_csp_{safe_filename(_cspt)}.png"),
    ))

for _lbl_h, _id_df_h, _csp_od_h, _ttl_h, _opath_h in _hist_subsets:
    if _id_df_h.empty:
        print(f"No data for histogram: {_ttl_h}")
        continue

    _groups_h = sorted(_id_df_h['group'].dropna().unique())
    _n_grp = len(_groups_h)
    if _n_grp == 0:
        continue

    # Determine x-axis range shared across all groups for comparability
    _all_vals = _id_df_h['response'].dropna().to_numpy(dtype=float)
    _x_min = float(np.nanpercentile(_all_vals, 1)) if _all_vals.size > 0 else -0.5
    _x_max = float(np.nanpercentile(_all_vals, 99)) if _all_vals.size > 0 else 1.5

    # Box color logic (same as boxplots: odor-specific when CSp is fixed, identity-based otherwise)
    if _csp_od_h is not None:
        _csm_col_h = roi_database.loc[roi_database['CSp'].apply(normalize_odor_name) == _csp_od_h, 'CSm'].dropna()
        _csm_od_h = normalize_odor_name(_csm_col_h.iloc[0]) if not _csm_col_h.empty else None
        _identity_keys_h = {'CSp', 'CSm', 'Novel'}
        _novel_od_h = next((k for k in colors_hex if k not in _identity_keys_h and k != _csp_od_h and k != _csm_od_h), None)
        _hist_colors = {
            'CSp':   colors_hex.get(_csp_od_h, '#999999'),
            'CSm':   colors_hex.get(_csm_od_h, '#999999'),
            'Novel': colors_hex.get(_novel_od_h, '#999999'),
        }
    else:
        _hist_colors = {od: colors_hex.get(od, '#999999') for od in odor_order}

    fig_h, axes_h = plt.subplots(1, _n_grp, figsize=(3.8 * _n_grp, 3.2), sharey=False, squeeze=False)

    for _col, _grp_h in enumerate(_groups_h):
        ax = axes_h[0, _col]
        _gdf_h = _id_df_h.loc[_id_df_h['group'] == _grp_h]

        for _oi in odor_order:
            _vals = _gdf_h.loc[_gdf_h['odor_identity'] == _oi, 'response'].dropna().to_numpy(dtype=float)
            if _vals.size == 0:
                continue
            _col_h = _hist_colors.get(_oi, '#999999')
            sns.kdeplot(_vals, ax=ax, color=_col_h, fill=True, alpha=0.35, linewidth=1.8, label=f"{_oi} (n={_vals.size})", clip=(_x_min, _x_max))

        ax.axvline(0, color='k', lw=0.8, ls='--')
        ax.set_title(f"Group: {_grp_h}", fontsize=9)
        ax.set_xlabel('ΔF/F response')
        if _col == 0:
            ax.set_ylabel('Density')
        ax.legend(fontsize=7)
        ax.grid(axis='y', alpha=0.25)

    fig_h.suptitle(_ttl_h, y=1.02)
    plt.tight_layout()
    fig_h.savefig(_opath_h, dpi=200, bbox_inches='tight')
    plt.show()

# %%
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

    csp_name = roi_row.get('CSp')
    csm_name = roi_row.get('CSm')
    novel_name = next((odor for odor in roi_odor_windows.keys() if odor not in {csp_name, csm_name}), None)

    identity_to_odor = {
        'CSp': csp_name,
        'CSm': csm_name,
        'Novel': novel_name,
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
        fig, axes = plt.subplots(len(group_order), 1, figsize=(3, 3 * len(group_order)), sharex=False, sharey=True)
        if len(group_order) == 1:
            axes = [axes]

        for axis, group_name in zip(axes, group_order):
            group_subset = fly_trace_df.loc[fly_trace_df['group'] == group_name]
            group_frame_rates = roi_database.loc[roi_database['group'] == group_name, 'frame_rate_hz'].to_numpy(dtype=float)
            group_frame_rates = group_frame_rates[np.isfinite(group_frame_rates) & (group_frame_rates > 0)]
            group_frame_rate_hz = float(np.nanmedian(group_frame_rates)) if group_frame_rates.size > 0 else 1.0
            for odor_identity in odor_order:
                group_traces = group_subset.loc[group_subset['odor_identity'] == odor_identity, 'trace'].tolist()
                mean_trace, sem_trace = mean_and_sem_padded(group_traces)
                if mean_trace.size == 0:
                    continue

                time_axis = np.arange(mean_trace.size) / group_frame_rate_hz
                trace_color = colors_hex.get(odor_identity, '#999999')
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
            figsize=(2 * len(odor_order), 3 * len(group_order)),
            sharex=False,
            sharey=True,
            squeeze=False,
        )

        for row_idx, group_name in enumerate(group_order):
            group_subset = fly_trace_df.loc[fly_trace_df['group'] == group_name]
            group_frame_rates = roi_database.loc[roi_database['group'] == group_name, 'frame_rate_hz'].to_numpy(dtype=float)
            group_frame_rates = group_frame_rates[np.isfinite(group_frame_rates) & (group_frame_rates > 0)]
            group_frame_rate_hz = float(np.nanmedian(group_frame_rates)) if group_frame_rates.size > 0 else 1.0
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
                    trace_color = colors_hex.get(odor_identity, '#999999')
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
                    figsize=(3, 3 * len(group_order_norm)),
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
                    for odor_identity in odor_order:
                        group_traces = group_subset.loc[group_subset['odor_identity'] == odor_identity, 'trace'].tolist()
                        mean_trace, sem_trace = mean_and_sem_padded(group_traces)
                        if mean_trace.size == 0:
                            continue

                        time_axis = np.arange(mean_trace.size) / group_frame_rate_hz
                        trace_color = colors_hex.get(odor_identity, '#999999')
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
                    figsize=(2 * len(odor_order), 3 * len(group_order_norm)),
                    sharex=False,
                    sharey=True,
                    squeeze=False,
                )

                for row_idx, group_name in enumerate(group_order_norm):
                    group_subset = fly_trace_df_norm.loc[fly_trace_df_norm['group'] == group_name]
                    group_frame_rates = roi_database.loc[roi_database['group'] == group_name, 'frame_rate_hz'].to_numpy(dtype=float)
                    group_frame_rates = group_frame_rates[np.isfinite(group_frame_rates) & (group_frame_rates > 0)]
                    group_frame_rate_hz = float(np.nanmedian(group_frame_rates)) if group_frame_rates.size > 0 else 1.0
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
                            trace_color = colors_hex.get(odor_identity, '#999999')
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
# --- Trace figures per CSp-specific odor ---
for _cspt_tr in ['MCH', 'OCTT', 'IAA']:
    _tr_roi_names = set(roi_database.loc[roi_database['CSp'].apply(normalize_odor_name) == _cspt_tr, 'roi_unique_name'])
    if not _tr_roi_names:
        continue

    # Resolve odor-specific colors for this CSp
    _tr_csm_col = roi_database.loc[roi_database['CSp'].apply(normalize_odor_name) == _cspt_tr, 'CSm'].dropna()
    _tr_csm_od = normalize_odor_name(_tr_csm_col.iloc[0]) if not _tr_csm_col.empty else None
    _tr_id_keys = {'CSp', 'CSm', 'Novel'}
    _tr_novel_od = next((k for k in colors_hex if k not in _tr_id_keys and k != _cspt_tr and k != _tr_csm_od), None)
    _tr_colors = {
        'CSp':   colors_hex.get(_cspt_tr, '#999999'),
        'CSm':   colors_hex.get(_tr_csm_od, '#999999'),
        'Novel': colors_hex.get(_tr_novel_od, '#999999'),
    }

    # --- Non-normalized traces ---
    if 'fly_trace_df' in dir() and not fly_trace_df.empty:
        _tr_fly_df = fly_trace_df.loc[fly_trace_df['flyID'].isin(
            roi_database.loc[roi_database['roi_unique_name'].isin(_tr_roi_names), 'flyID']
        )].copy()

        if not _tr_fly_df.empty:
            _tr_groups = sorted(_tr_fly_df['group'].dropna().unique())

            # Overlaid figure
            _tr_fig, _tr_axes = plt.subplots(len(_tr_groups), 1, figsize=(3, 3 * len(_tr_groups)), sharex=False, sharey=True)
            if len(_tr_groups) == 1:
                _tr_axes = [_tr_axes]
            for _tr_ax, _tr_grp in zip(_tr_axes, _tr_groups):
                _tr_gsub = _tr_fly_df.loc[_tr_fly_df['group'] == _tr_grp]
                _tr_frs = roi_database.loc[roi_database['group'] == _tr_grp, 'frame_rate_hz'].to_numpy(dtype=float)
                _tr_frs = _tr_frs[np.isfinite(_tr_frs) & (_tr_frs > 0)]
                _tr_fr = float(np.nanmedian(_tr_frs)) if _tr_frs.size > 0 else 1.0
                for _tr_oi in odor_order:
                    _tr_traces = _tr_gsub.loc[_tr_gsub['odor_identity'] == _tr_oi, 'trace'].tolist()
                    _tr_mean, _tr_sem = mean_and_sem_padded(_tr_traces)
                    if _tr_mean.size == 0:
                        continue
                    _tr_t = np.arange(_tr_mean.size) / _tr_fr
                    _tr_c = _tr_colors[_tr_oi]
                    _tr_ax.plot(_tr_t, _tr_mean, lw=2, color=_tr_c, label=f"{_tr_oi} (n={len(_tr_traces)})")
                    if _tr_sem.size == _tr_mean.size:
                        _tr_ax.fill_between(_tr_t, _tr_mean - _tr_sem, _tr_mean + _tr_sem, alpha=0.2, color=_tr_c)
                    _tr_osub = _tr_gsub.loc[_tr_gsub['odor_identity'] == _tr_oi]
                    _tr_starts = _tr_osub['stim_start_s'].to_numpy(dtype=float)
                    _tr_ends = _tr_osub['stim_end_s'].to_numpy(dtype=float)
                    _tr_starts = _tr_starts[np.isfinite(_tr_starts)]
                    _tr_ends = _tr_ends[np.isfinite(_tr_ends)]
                    if _tr_starts.size > 0 and _tr_ends.size > 0:
                        _tr_ss = float(np.nanmedian(_tr_starts))
                        _tr_se = float(np.nanmedian(_tr_ends))
                        if _tr_se > _tr_ss:
                            _tr_ax.axvspan(_tr_ss, _tr_se, color=_tr_c, alpha=0.08)
                _tr_ax.set_title(f"Group: {_tr_grp}")
                _tr_ax.set_ylabel('dF/F')
                _tr_ax.grid(axis='y', alpha=0.25)
                _tr_ax.legend(loc='best', fontsize=8)
            _tr_axes[-1].set_xlabel('Time (s)')
            _tr_fig.suptitle(f"Fly-averaged traces by group (CSp = {_cspt_tr})", y=1.01)
            plt.tight_layout()
            _tr_fig.savefig(os.path.join(results_dir, f"{safe_filename(container_id)}_fly_averaged_traces_by_group_csp_{safe_filename(_cspt_tr)}.png"), dpi=200, bbox_inches='tight')
            plt.show()

            # Grid figure
            _tr_fig_g, _tr_axes_g = plt.subplots(len(_tr_groups), len(odor_order), figsize=(2 * len(odor_order), 3 * len(_tr_groups)), sharex=False, sharey=True, squeeze=False)
            for _ri, _tr_grp in enumerate(_tr_groups):
                _tr_gsub = _tr_fly_df.loc[_tr_fly_df['group'] == _tr_grp]
                _tr_frs = roi_database.loc[roi_database['group'] == _tr_grp, 'frame_rate_hz'].to_numpy(dtype=float)
                _tr_frs = _tr_frs[np.isfinite(_tr_frs) & (_tr_frs > 0)]
                _tr_fr = float(np.nanmedian(_tr_frs)) if _tr_frs.size > 0 else 1.0
                for _ci, _tr_oi in enumerate(odor_order):
                    _tr_ax = _tr_axes_g[_ri, _ci]
                    _tr_traces = _tr_gsub.loc[_tr_gsub['odor_identity'] == _tr_oi, 'trace'].tolist()
                    for _ft in _tr_traces:
                        _ft = np.asarray(_ft, dtype=float)
                        if _ft.size > 0:
                            _tr_ax.plot(np.arange(_ft.size) / _tr_fr, _ft, color='0.75', lw=0.8, alpha=0.7)
                    _tr_mean, _tr_sem = mean_and_sem_padded(_tr_traces)
                    if _tr_mean.size > 0:
                        _tr_t = np.arange(_tr_mean.size) / _tr_fr
                        _tr_c = _tr_colors[_tr_oi]
                        _tr_ax.plot(_tr_t, _tr_mean, lw=2.0, color=_tr_c)
                        if _tr_sem.size == _tr_mean.size:
                            _tr_ax.fill_between(_tr_t, _tr_mean - _tr_sem, _tr_mean + _tr_sem, color=_tr_c, alpha=0.25)
                        _tr_osub = _tr_gsub.loc[_tr_gsub['odor_identity'] == _tr_oi]
                        _tr_starts = _tr_osub['stim_start_s'].to_numpy(dtype=float)
                        _tr_ends = _tr_osub['stim_end_s'].to_numpy(dtype=float)
                        _tr_starts = _tr_starts[np.isfinite(_tr_starts)]
                        _tr_ends = _tr_ends[np.isfinite(_tr_ends)]
                        if _tr_starts.size > 0 and _tr_ends.size > 0:
                            _tr_ss = float(np.nanmedian(_tr_starts))
                            _tr_se = float(np.nanmedian(_tr_ends))
                            if _tr_se > _tr_ss:
                                _tr_ax.axvspan(_tr_ss, _tr_se, color=_tr_c, alpha=0.08)
                    if _ri == 0:
                        _tr_ax.set_title(_tr_oi)
                    if _ci == 0:
                        _tr_ax.set_ylabel(f"{_tr_grp}\ndF/F")
                    if _ri == len(_tr_groups) - 1:
                        _tr_ax.set_xlabel('Time (s)')
                    _tr_ax.grid(axis='y', alpha=0.25)
            _tr_fig_g.suptitle(f"Fly traces by group and odor identity (CSp = {_cspt_tr})", y=1.01)
            plt.tight_layout()
            _tr_fig_g.savefig(os.path.join(results_dir, f"{safe_filename(container_id)}_fly_traces_grid_by_group_and_odor_csp_{safe_filename(_cspt_tr)}.png"), dpi=200, bbox_inches='tight')
            plt.show()

    # --- Normalized traces ---
    if 'fly_trace_df_norm' in dir() and not fly_trace_df_norm.empty:
        _tr_fly_df_n = fly_trace_df_norm.loc[fly_trace_df_norm['flyID'].isin(
            roi_database.loc[roi_database['roi_unique_name'].isin(_tr_roi_names), 'flyID']
        )].copy()

        if not _tr_fly_df_n.empty:
            _tr_groups_n = sorted(_tr_fly_df_n['group'].dropna().unique())

            # Overlaid figure
            _tr_fig_n, _tr_axes_n = plt.subplots(len(_tr_groups_n), 1, figsize=(3, 3 * len(_tr_groups_n)), sharex=False, sharey=True)
            if len(_tr_groups_n) == 1:
                _tr_axes_n = [_tr_axes_n]
            for _tr_ax, _tr_grp in zip(_tr_axes_n, _tr_groups_n):
                _tr_gsub = _tr_fly_df_n.loc[_tr_fly_df_n['group'] == _tr_grp]
                _tr_frs = roi_database.loc[roi_database['group'] == _tr_grp, 'frame_rate_hz'].to_numpy(dtype=float)
                _tr_frs = _tr_frs[np.isfinite(_tr_frs) & (_tr_frs > 0)]
                _tr_fr = float(np.nanmedian(_tr_frs)) if _tr_frs.size > 0 else 1.0
                for _tr_oi in odor_order:
                    _tr_traces = _tr_gsub.loc[_tr_gsub['odor_identity'] == _tr_oi, 'trace'].tolist()
                    _tr_mean, _tr_sem = mean_and_sem_padded(_tr_traces)
                    if _tr_mean.size == 0:
                        continue
                    _tr_t = np.arange(_tr_mean.size) / _tr_fr
                    _tr_c = _tr_colors[_tr_oi]
                    _tr_ax.plot(_tr_t, _tr_mean, lw=2, color=_tr_c, label=f"{_tr_oi} (n={len(_tr_traces)})")
                    if _tr_sem.size == _tr_mean.size:
                        _tr_ax.fill_between(_tr_t, _tr_mean - _tr_sem, _tr_mean + _tr_sem, alpha=0.2, color=_tr_c)
                    _tr_osub = _tr_gsub.loc[_tr_gsub['odor_identity'] == _tr_oi]
                    _tr_starts = _tr_osub['stim_start_s'].to_numpy(dtype=float)
                    _tr_ends = _tr_osub['stim_end_s'].to_numpy(dtype=float)
                    _tr_starts = _tr_starts[np.isfinite(_tr_starts)]
                    _tr_ends = _tr_ends[np.isfinite(_tr_ends)]
                    if _tr_starts.size > 0 and _tr_ends.size > 0:
                        _tr_ss = float(np.nanmedian(_tr_starts))
                        _tr_se = float(np.nanmedian(_tr_ends))
                        if _tr_se > _tr_ss:
                            _tr_ax.axvspan(_tr_ss, _tr_se, color=_tr_c, alpha=0.08)
                _tr_ax.set_title(f"Group: {_tr_grp}")
                _tr_ax.set_ylabel('Normalized dF/F')
                _tr_ax.grid(axis='y', alpha=0.25)
                _tr_ax.legend(loc='best', fontsize=8)
            _tr_axes_n[-1].set_xlabel('Time (s)')
            _tr_fig_n.suptitle(f"Normalized fly-averaged traces by group (CSp = {_cspt_tr})", y=1.01)
            plt.tight_layout()
            _tr_fig_n.savefig(os.path.join(results_dir, f"{safe_filename(container_id)}_normalized_fly_averaged_traces_by_group_csp_{safe_filename(_cspt_tr)}.png"), dpi=200, bbox_inches='tight')
            plt.show()

            # Grid figure
            _tr_fig_gn, _tr_axes_gn = plt.subplots(len(_tr_groups_n), len(odor_order), figsize=(2 * len(odor_order), 3 * len(_tr_groups_n)), sharex=False, sharey=True, squeeze=False)
            for _ri, _tr_grp in enumerate(_tr_groups_n):
                _tr_gsub = _tr_fly_df_n.loc[_tr_fly_df_n['group'] == _tr_grp]
                _tr_frs = roi_database.loc[roi_database['group'] == _tr_grp, 'frame_rate_hz'].to_numpy(dtype=float)
                _tr_frs = _tr_frs[np.isfinite(_tr_frs) & (_tr_frs > 0)]
                _tr_fr = float(np.nanmedian(_tr_frs)) if _tr_frs.size > 0 else 1.0
                for _ci, _tr_oi in enumerate(odor_order):
                    _tr_ax = _tr_axes_gn[_ri, _ci]
                    _tr_traces = _tr_gsub.loc[_tr_gsub['odor_identity'] == _tr_oi, 'trace'].tolist()
                    for _ft in _tr_traces:
                        _ft = np.asarray(_ft, dtype=float)
                        if _ft.size > 0:
                            _tr_ax.plot(np.arange(_ft.size) / _tr_fr, _ft, color='0.75', lw=0.8, alpha=0.7)
                    _tr_mean, _tr_sem = mean_and_sem_padded(_tr_traces)
                    if _tr_mean.size > 0:
                        _tr_t = np.arange(_tr_mean.size) / _tr_fr
                        _tr_c = _tr_colors[_tr_oi]
                        _tr_ax.plot(_tr_t, _tr_mean, lw=2.0, color=_tr_c)
                        if _tr_sem.size == _tr_mean.size:
                            _tr_ax.fill_between(_tr_t, _tr_mean - _tr_sem, _tr_mean + _tr_sem, color=_tr_c, alpha=0.25)
                        _tr_osub = _tr_gsub.loc[_tr_gsub['odor_identity'] == _tr_oi]
                        _tr_starts = _tr_osub['stim_start_s'].to_numpy(dtype=float)
                        _tr_ends = _tr_osub['stim_end_s'].to_numpy(dtype=float)
                        _tr_starts = _tr_starts[np.isfinite(_tr_starts)]
                        _tr_ends = _tr_ends[np.isfinite(_tr_ends)]
                        if _tr_starts.size > 0 and _tr_ends.size > 0:
                            _tr_ss = float(np.nanmedian(_tr_starts))
                            _tr_se = float(np.nanmedian(_tr_ends))
                            if _tr_se > _tr_ss:
                                _tr_ax.axvspan(_tr_ss, _tr_se, color=_tr_c, alpha=0.08)
                    if _ri == 0:
                        _tr_ax.set_title(_tr_oi)
                    if _ci == 0:
                        _tr_ax.set_ylabel(f"{_tr_grp}\nNormalized dF/F")
                    if _ri == len(_tr_groups_n) - 1:
                        _tr_ax.set_xlabel('Time (s)')
                    _tr_ax.grid(axis='y', alpha=0.25)
            _tr_fig_gn.suptitle(f"Normalized fly traces by group and odor identity (CSp = {_cspt_tr})", y=1.01)
            plt.tight_layout()
            _tr_fig_gn.savefig(os.path.join(results_dir, f"{safe_filename(container_id)}_normalized_fly_traces_grid_by_group_and_odor_csp_{safe_filename(_cspt_tr)}.png"), dpi=200, bbox_inches='tight')
            plt.show()

 # %%
