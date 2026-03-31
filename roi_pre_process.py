#%% Imports
import os
import glob
import numpy as np
import pickle
import matplotlib.pyplot as plt
from utilities import (
    map_stimulus_ids_from_osf,
    temporal_denoise,
)
from roi_processor import run_roi_selection
from helpers_figures import (
    plot_mean_std_projection,
    plot_roi_masks_and_traces,
    plot_trial_averaged_roi_responses,
    plot_trial_overlaid_roi_responses,
)
from roi_extractor_params import get_auto_roi_params
from batch_utilities import load_series_metadata

#%% Adjust the following depending on the experiment
base_dir = "/Volumes/tungsten/scratch/gfelsenb/Ana/2p-imaging/burak/"
container_id = '2025_10_Gamma1_CC_extinction'
day_id = '2025_11_13'

motion_correction_profile = 'dendrites'  # boutons, dendrites
roi_names_input = None  # e.g. ['dendrite_1', 'spine_1'] or None for auto-naming
colors_hex = {'MCH': '#e41a1c', 'OCTT': "#ffff99", "IAA": '#4daf4a'}

if motion_correction_profile == 'boutons':
    auto_roi_profile = 'boutons'
    roi_selection_mode = 'custom-automatic'
elif motion_correction_profile == 'dendrites':
    auto_roi_profile = None
    roi_selection_mode = 'manual'

#%% Find all the series for the given day and container
experiment_dir = os.path.join(base_dir, container_id)
day_dir = os.path.join(experiment_dir, day_id)
if not os.path.exists(day_dir):
    raise FileNotFoundError(f"Directory {day_dir} does not exist. Please check the path and try again.")

db_path = os.path.join(experiment_dir, f'{container_id}_database.csv')
series_paths = sorted(glob.glob(os.path.join(day_dir, 'S1-T*')))

#%% Process ROIs series by series
for series_path in series_paths:
    series_id = os.path.basename(series_path)
    print(f"\n{'='*60}")
    print(f"ROI processing: {series_id}")
    print(f"{'='*60}")

    series_dir = os.path.join(day_dir, series_id)
    results_dir = os.path.join(series_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    experiment_id = f"{day_id}_{series_id}"

    series_meta, vial_to_odor = load_series_metadata(db_path, series_id, experiment_id)

    # Load motion-corrected movie and registration metadata
    suite2p_dir = os.path.join(series_dir, 'suite2p_corrected')
    processed_movie = np.load(os.path.join(suite2p_dir, f'{series_id}_corrected.npy'), mmap_mode='r')
    ops = np.load(os.path.join(suite2p_dir, 'motion_input_ops.npy'), allow_pickle=True).item()
    output_ops = np.load(os.path.join(suite2p_dir, 'motion_output_ops.npy'), allow_pickle=True)
    downsampled_fr = ops['fs']

    # Crop to stable region — uncomment to apply output_ops bounds
    # yrange = output_ops[-2]
    # xrange = output_ops[-1]
    # processed_movie_cropped = processed_movie[:, yrange[0]:yrange[1], xrange[0]:xrange[1]]
    processed_movie_cropped = processed_movie

    # Map stimulus IDs from saved stim trace + OSF file
    stim_on_trace_downsampled_interp = np.load(os.path.join(series_dir, f'{series_id}_stim_trace.npy'))
    osf_path = os.path.join(series_dir, f'{series_id}.osf')
    stimulus_id_trace, stimulus_sequence, stim_starts, stim_ends = map_stimulus_ids_from_osf(
        stim_on_trace_downsampled_interp,
        osf_path,
    )
    print(f"Stimulus sequence from OSF: {stimulus_sequence}")
    print(f"Detected {len(stim_starts)} stimulus periods")
    mapped_ids = np.unique(stimulus_id_trace[stimulus_id_trace > 0]).astype(int)
    mapped_stimuli = [vial_to_odor.get(vial_id, f"V{vial_id}") for vial_id in mapped_ids]
    print(f"Mapped stimuli: {mapped_stimuli}")

    # Visualize mean and std projections of the corrected movie
    plot_mean_std_projection(
        processed_movie_cropped,
        save_path=os.path.join(results_dir, f'{series_id}_mean_std_images.png'),
        figsize=(16, 4),
        mean_cmap='magma',
        std_cmap='magma',
        mean_title='Mean Image (Motion Corrected)',
        std_title='Std Image (Motion Corrected)',
        show_axes=False,
        dpi=150,
    )

    # ROI selection (manual or automatic) — requires interactive backend
    %matplotlib qt
    extraction_image = processed_movie_cropped.mean(axis=0)
    auto_roi_params = get_auto_roi_params(auto_roi_profile)

    roi_selection_result = run_roi_selection(
        mode=roi_selection_mode,
        movie=processed_movie_cropped,
        extraction_image=extraction_image,
        fs=downsampled_fr,
        results_dir=results_dir,
        series_id=series_id,
        stimulus_id_trace=stimulus_id_trace,
        auto_roi_params=auto_roi_params,
        df_f_method='1-11s',
        roi_names=roi_names_input,
    )

    roi_masks = roi_selection_result['roi_masks']
    roi_names = roi_selection_result['roi_names']
    raw_traces = roi_selection_result['raw_traces']
    bg_subtracted_df_traces = roi_selection_result['bg_subtracted_df_traces']
    background_mask = roi_selection_result['background_mask']
    background_polygon = roi_selection_result['background_polygon']
    background_raw_trace = roi_selection_result['background_raw_trace']

    bg_subtracted_dict = {name: trace for name, trace in zip(roi_names, bg_subtracted_df_traces)}
    print(f"ROI names: {list(bg_subtracted_dict.keys())}")
    for roi_name, trace in bg_subtracted_dict.items():
        print(f"  {roi_name}: shape {trace.shape}")

    # Plot ROI masks and background-subtracted traces with stimulus periods
    %matplotlib inline
    traces_fig_path = os.path.join(results_dir, f'{series_id}_roi_analysis.pdf')
    plot_roi_masks_and_traces(
        plot_image=extraction_image,
        roi_masks=roi_masks,
        roi_names=roi_names,
        roi_traces=bg_subtracted_df_traces,
        downsampled_fr=downsampled_fr,
        series_id=series_id,
        background_mask=background_mask,
        background_polygon=background_polygon,
        stimulus_id_trace=stimulus_id_trace,
        vial_to_odor=vial_to_odor,
        colors_hex=colors_hex,
        save_path=traces_fig_path,
        figsize=(10, 12),
        dpi=300,
    )
    print(f"Saved ROI analysis figure to: {traces_fig_path}")

    # Build stimulus segments
    stim_ids = np.asarray(stimulus_id_trace).astype(int)
    if stim_ids.size == 0:
        stim_segments = []
    else:
        change_points = np.where(np.diff(stim_ids) != 0)[0] + 1
        segment_starts = np.concatenate(([0], change_points))
        segment_ends = np.concatenate((change_points, [stim_ids.size]))
        stim_segments = [
            (start_idx, end_idx, int(stim_ids[start_idx]))
            for start_idx, end_idx in zip(segment_starts, segment_ends)
            if int(stim_ids[start_idx]) > 0
        ]

    stimulus_ids_unique = sorted({seg_stim_id for _, _, seg_stim_id in stim_segments})
    stim_ids_full_trace = np.asarray(stimulus_id_trace).astype(int)
    context_window_s = 5.0
    context_window_frames = int(round(context_window_s * downsampled_fr))

    # Build nested ROI dictionary with single-trial and trial-averaged traces
    roi_data_nested = {}

    for roi_idx, roi_name in enumerate(roi_names):
        raw_trace = np.asarray(raw_traces[roi_idx])
        bg_sub_trace = np.asarray(bg_subtracted_df_traces[roi_idx])
        roi_mask = np.asarray(roi_masks[roi_idx]).astype(bool)

        repeats_by_stimulus = {}
        repeats_by_stimulus_with_context = {}

        for stim_id in stimulus_ids_unique:
            stim_name = vial_to_odor.get(stim_id, f"V{stim_id}")
            trial_segments = [
                (start_idx, end_idx)
                for start_idx, end_idx, seg_stim_id in stim_segments
                if seg_stim_id == stim_id
            ]
            trial_traces_only_stim = [
                bg_sub_trace[start_idx:end_idx]
                for start_idx, end_idx in trial_segments
                if end_idx > start_idx
            ]

            trial_traces_with_context = []
            trial_stim_id_traces_with_context = []
            stim_start_indices_in_window = []
            stim_end_indices_in_window = []
            for start_idx, end_idx in trial_segments:
                if end_idx <= start_idx:
                    continue
                window_start = max(0, start_idx - context_window_frames)
                window_end = min(bg_sub_trace.shape[0], end_idx + context_window_frames)
                trial_traces_with_context.append(bg_sub_trace[window_start:window_end])
                trial_stim_id_traces_with_context.append(stim_ids_full_trace[window_start:window_end])
                stim_start_indices_in_window.append(int(start_idx - window_start))
                stim_end_indices_in_window.append(int(end_idx - window_start))

            if len(trial_traces_only_stim) == 0:
                repeats_by_stimulus[stim_name] = {
                    'stimulus_id': int(stim_id),
                    'stimulus_name': stim_name,
                    'trial_traces': [],
                    'trial_average_trace': np.array([], dtype=float),
                }
                repeats_by_stimulus_with_context[stim_name] = {
                    'stimulus_id': int(stim_id),
                    'stimulus_name': stim_name,
                    'context_window_s': float(context_window_s),
                    'trial_traces': [],
                    'trial_stimulus_id_traces': [],
                    'stim_start_indices_in_window': [],
                    'stim_end_indices_in_window': [],
                    'trial_average_trace': np.array([], dtype=float),
                    'trial_average_stimulus_id_trace': np.array([], dtype=float),
                }
                continue

            trial_max_len = max(len(t) for t in trial_traces_only_stim)
            padded_trials = np.full((len(trial_traces_only_stim), trial_max_len), np.nan, dtype=float)
            for trial_i, trial_trace in enumerate(trial_traces_only_stim):
                padded_trials[trial_i, :len(trial_trace)] = trial_trace
            avg_trace_only_stim = np.nanmean(padded_trials, axis=0)

            repeats_by_stimulus[stim_name] = {
                'stimulus_id': int(stim_id),
                'stimulus_name': stim_name,
                'trial_traces': trial_traces_only_stim,
                'trial_average_trace': avg_trace_only_stim,
            }

            trial_max_len_context = max(len(t) for t in trial_traces_with_context)
            padded_context_trials = np.full(
                (len(trial_traces_with_context), trial_max_len_context), np.nan, dtype=float,
            )
            padded_context_stim = np.full(
                (len(trial_stim_id_traces_with_context), trial_max_len_context), np.nan, dtype=float,
            )
            for trial_i, trial_trace in enumerate(trial_traces_with_context):
                padded_context_trials[trial_i, :len(trial_trace)] = trial_trace
            for trial_i, trial_stim_trace in enumerate(trial_stim_id_traces_with_context):
                padded_context_stim[trial_i, :len(trial_stim_trace)] = trial_stim_trace

            avg_trace_with_context = np.nanmean(padded_context_trials, axis=0)
            avg_stim_trace_with_context = np.nanmax(padded_context_stim, axis=0)

            repeats_by_stimulus_with_context[stim_name] = {
                'stimulus_id': int(stim_id),
                'stimulus_name': stim_name,
                'context_window_s': float(context_window_s),
                'trial_traces': trial_traces_with_context,
                'trial_stimulus_id_traces': trial_stim_id_traces_with_context,
                'stim_start_indices_in_window': stim_start_indices_in_window,
                'stim_end_indices_in_window': stim_end_indices_in_window,
                'trial_average_trace': avg_trace_with_context,
                'trial_average_stimulus_id_trace': avg_stim_trace_with_context,
            }

        trial_averaged_traces_by_stimulus = {
            stim_name: stim_data['trial_average_trace']
            for stim_name, stim_data in repeats_by_stimulus.items()
        }

        roi_data_nested[roi_name] = {
            'unique_id': f"{container_id}_{day_id}_{series_id}_{roi_name}",
            'downsampled_fr': float(downsampled_fr),
            'raw_trace': raw_trace,
            'bg_subtracted_df_trace': bg_sub_trace,
            'mask': roi_mask,
            'analyzed_traces': repeats_by_stimulus,
            'single_trial_traces_only_stim': repeats_by_stimulus,
            'single_trial_traces_with_context': repeats_by_stimulus_with_context,
            'trial_averaged_traces_by_stimulus': trial_averaged_traces_by_stimulus,
            'mean_image': extraction_image,
        }

    # Save the dataset
    series_meta['stim_info'] = {
        'aurora_vial_info': vial_to_odor,
        'stim_id_trace': stimulus_id_trace,
    }
    series_meta['mean_image'] = extraction_image
    series_meta['background_info'] = {
        'mask': background_mask,
        'polygon': background_polygon,
        'raw_trace': background_raw_trace,
    }

    session_results = {'metadata': series_meta, 'rois': {}}
    for roi_name, roi_data in roi_data_nested.items():
        session_results['rois'][roi_data['unique_id']] = {
            'downsampled_fr': roi_data['downsampled_fr'],
            'raw_trace': roi_data['raw_trace'],
            'bg_subtracted_df_trace': roi_data['bg_subtracted_df_trace'],
            'mask': roi_data['mask'],
            'trial_averaged_traces_by_stimulus': roi_data['trial_averaged_traces_by_stimulus'],
            'single_trial_traces_only_stim': roi_data['single_trial_traces_only_stim'],
            'single_trial_traces_with_context': roi_data['single_trial_traces_with_context'],
        }

    p_data_save_path = os.path.join(experiment_dir, f'{container_id}_processed_data')
    os.makedirs(p_data_save_path, exist_ok=True)
    save_file = os.path.join(p_data_save_path, f'{series_id}_processed_data.pkl')
    with open(save_file, 'wb') as f:
        pickle.dump(session_results, f)
    print(f"Dataset saved to: {save_file}")

    # Plot trial-averaged and trial-overlaid responses
    roi_trial_plot_dir = os.path.join(results_dir, 'roi_trial_average_plots')
    os.makedirs(roi_trial_plot_dir, exist_ok=True)

    pre_window_s = 5.0
    post_window_s = 15.0
    pre_window_frames = int(round(pre_window_s * downsampled_fr))
    post_window_frames = int(round(post_window_s * downsampled_fr))

    stim_durations_by_id = {
        stim_id: [
            int(end_idx - start_idx)
            for start_idx, end_idx, seg_stim_id in stim_segments
            if seg_stim_id == stim_id
        ]
        for stim_id in stimulus_ids_unique
    }

    n_rois = len(roi_names)
    all_durations = [d for durations in stim_durations_by_id.values() for d in durations]
    bg_subtracted_df_traces_smoothed = np.array(
        [temporal_denoise(trace, window_size=5) for trace in bg_subtracted_df_traces]
    )

    if len(all_durations) > 0 and n_rois > 0:
        _ = plot_trial_averaged_roi_responses(
            roi_names=roi_names,
            roi_masks=roi_masks,
            bg_subtracted_df_traces=bg_subtracted_df_traces_smoothed,
            extraction_image=extraction_image,
            stim_segments=stim_segments,
            stimulus_ids_unique=stimulus_ids_unique,
            vial_to_odor=vial_to_odor,
            colors_hex=colors_hex,
            stim_durations_by_id=stim_durations_by_id,
            downsampled_fr=downsampled_fr,
            series_id=series_id,
            series_meta=series_meta,
            roi_trial_plot_dir=roi_trial_plot_dir,
            pre_window_frames=pre_window_frames,
            post_window_frames=post_window_frames,
            max_cols=5,
            cell_w=9,
            cell_h=3,
            font_size=6,
        )

        _ = plot_trial_overlaid_roi_responses(
            roi_names=roi_names,
            roi_masks=roi_masks,
            bg_subtracted_df_traces=bg_subtracted_df_traces_smoothed,
            extraction_image=extraction_image,
            stim_segments=stim_segments,
            stimulus_ids_unique=stimulus_ids_unique,
            vial_to_odor=vial_to_odor,
            downsampled_fr=downsampled_fr,
            series_id=series_id,
            series_meta=series_meta,
            roi_trial_plot_dir=roi_trial_plot_dir,
            pre_window_frames=pre_window_frames,
            post_window_frames=post_window_frames,
            colors_hex=colors_hex,
            font_size=6,
            cell_w=3.5,
            cell_h=2.6,
        )

    print(f"Saved ROI trial plots to: {roi_trial_plot_dir}")
    print(f"Series {series_id} complete.")

# %%
