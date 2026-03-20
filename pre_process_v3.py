#%% 
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from suite2p.default_ops import default_ops
from utilities import (
    load_experiment_metadata,
    export_visualization_video,
    temporal_denoise,
    map_stimulus_ids_from_osf,
    read_ini_file,
    read_lvd_data,
    plot_lvd_channels,
    get_recording_frame_bounds,
    build_frame_alignment_traces,
    estimate_frame_rate,
    load_video_memmap,
    downsample_and_align_traces,
    parse_aurora_vial_info,
    run_motion_correction_suite2p,
)
from roi_processor import run_roi_selection
from helpers_figures import (
    plot_mean_std_projection,
    plot_roi_masks_and_traces,
    plot_trial_averaged_roi_responses,
    plot_trial_overlaid_roi_responses,
)
from suite_2d_params import get_motion_correction_params
from roi_extractor_params import get_auto_roi_params
from scipy.interpolate import interp1d
#%% Set up paths and parameters
# Path to the data folder
# Clear all variables
# %reset -f
main_dir = "/Volumes/tungsten/scratch/gfelsenb/Ana/2p-imaging/burak/"

container_id = '2026_03_Beta1_counterconditioned_dendrites'
day_id = '2025_12_11'
series_id = 'S1-T23767'
motion_correction_profile = 'dendrites'  # boutons, dendrites

if motion_correction_profile == 'boutons':
    auto_roi_profile = 'boutons'
    roi_selection_mode = 'custom-automatic'  # Options: 'custom-automatic' or 'manual'
elif motion_correction_profile == 'dendrites':
    auto_roi_profile = None
    roi_selection_mode = 'manual'  # Options: 'custom-automatic' or 'manual'

experiment_dir = os.path.join(main_dir, container_id)  # Update this path accordingly
day_dir = os.path.join(experiment_dir, day_id)  # Update this path accordingly
series_dir = os.path.join(day_dir, series_id)  # Update this path accordingly

experiment_id = f"{day_id}_{series_id}"
results_dir = os.path.join(series_dir, 'results')
os.makedirs(results_dir, exist_ok=True)

colors_hex = {'MCH': '#e41a1c', 'OCTT': "#ffff99", "IAA": '#4daf4a'}

#%% Read the database and find the information for the current series
db_path = os.path.join(experiment_dir, f'{container_id}_database.csv')
series_meta = load_experiment_metadata(db_path, series_id)
series_meta['experimentID'] = experiment_id
vial_to_odor = parse_aurora_vial_info(series_meta.get('AuroraVialInfo'))
if len(vial_to_odor) > 0:
    print(f"Stimulus vial mapping: {vial_to_odor}")


# Frame times etc.
data_width=750 # Hard coded since .ini doesn't seem to provide it
lvd_channels = {'shutter': 0, 'galvo': 1, 'protocol_end': 2, 'stim_on': 3}
config_path = os.path.join(series_dir, f'{series_id}_ch525.ini')
_, data_height, data_frames = read_ini_file(config_path)

lvd_path = os.path.join(series_dir, f'{series_id}.lvd')
lvd_data, lvd_samplerate = read_lvd_data(lvd_path)

start_lvd_plot = 0
end_lvd_plot = -1
plot_lvd_channels(lvd_data, lvd_channels, lvd_samplerate,   start=start_lvd_plot, end=end_lvd_plot, save_path=os.path.join(results_dir, f'{series_id}_lvd_channels.png'))

frame_peaks_raw, start_frame, end_frame = get_recording_frame_bounds(lvd_data, lvd_channels)
print(f"Estimated recording start frame: {start_frame}, end frame: {end_frame}, total frames: {end_frame - start_frame}")

frame_num_trace, frame_time_trace_ms, stim_on_trace_frames = build_frame_alignment_traces(
    lvd_data,
    frame_peaks_raw,
    lvd_channels['stim_on'],
    lvd_samplerate,
    data_frames,
)
frame_rate = estimate_frame_rate(frame_peaks_raw, lvd_samplerate)
print(f"Estimated frame rate based on galvo peaks: {frame_rate:.2f} Hz")

series_meta['recording_settings'] = {
    'image_width': int(data_width),
    'image_height': int(data_height),
    'frames': int(data_frames),
    'frame_rate': int(round(frame_rate)),
    }

#%% Read the binary file using memory mapping
# Read the video
video_name = f'{series_id}_ch525.bin'
video_path = os.path.join(series_dir, video_name)
dtype = 'uint16' # Our microscope writes data in 16bits!

raw_data = load_video_memmap(
    video_path,
    series_meta['recording_settings']['frames'],
    series_meta['recording_settings']['image_height'],
    series_meta['recording_settings']['image_width'],
    dtype=dtype,
    mode='c',
)

print(f"Binary Mapped: {video_path}")
print(f"Data Shape: {raw_data.shape}") 
print(f"Duration: {raw_data.shape[0] / series_meta['recording_settings']['frame_rate']:.2f} seconds")

# %% Downsampling by temporal binning (movie mean, stim mode)
downsampled_fr = 6.0
print(
    f"Temporal bin downsampling from ~{frame_rate:.2f} Hz "
    f"to ~{downsampled_fr:.2f} Hz using frame_time_trace_ms"
)

# Downsampling to increase SNR (using mean)
data_downsampled, frame_time_trace_downsampled, stim_on_trace_downsampled = downsample_and_align_traces(
    raw_data,
    start_frame,
    end_frame,
    downsampled_fr,
    frame_time_trace_ms,
    stim_on_trace_frames
)

# Now interpolate to a proper time trace of 6Hz since the above can have jitter due to frame drops etc.
expected_time_points = np.arange(frame_time_trace_downsampled[0], frame_time_trace_downsampled[-1], 1000 / downsampled_fr)

# Interpolate data using linear interpolation
data_downsampled_interp = np.empty((len(expected_time_points), data_downsampled.shape[1], data_downsampled.shape[2]), dtype=data_downsampled.dtype)
for y in range(data_downsampled.shape[1]):
    for x in range(data_downsampled.shape[2]):
        interp_func = interp1d(frame_time_trace_downsampled, data_downsampled[:, y, x], kind='linear')
        data_downsampled_interp[:, y, x] = interp_func(expected_time_points)

# Interpolate stimulus trace using nearest neighbor
stim_interp_func = interp1d(frame_time_trace_downsampled, stim_on_trace_downsampled, kind='nearest')
stim_on_trace_downsampled_interp = stim_interp_func(expected_time_points)
#%%
# Map stimulus sequence to frames based on ON periods
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

#%% Visualize mean and std projections to check for signal
projection_path = os.path.join(results_dir, f'{series_id}_mean_std_projections.png')
plot_mean_std_projection(
    data_downsampled_interp,
    save_path=projection_path,
    figsize=(16, 4),
    mean_cmap='gray',
    std_cmap='hot',
    mean_title='Mean Projection',
    std_title='Std Projection',
    show_axes=True,
    dpi=150,
)
print(f"Saved projections to: {projection_path}")

# Denoising
# denoised_median = median_denoise_temporal(data_downsampled_interp, size=3)
# data_denoised = temporal_denoise(data_downsampled_interp, window_size=7)
#%% Motion correction with temporary binary (without saving the .bin)
save_path = os.path.join(series_dir, 'suite2p_corrected')
motion_params = get_motion_correction_params(motion_correction_profile)
print(f"Using Suite2p motion profile: {motion_correction_profile}")


motion_ops_profile = default_ops()

motion_ops_profile['fs'] = downsampled_fr
motion_ops_profile['nonrigid'] = False
motion_ops_profile['block_size'] = (128, 128)
motion_ops_profile['smooth_sigma'] = 2.5
motion_ops_profile['save_path0'] = save_path
motion_ops_profile['main_chan'] = 0
motion_ops_profile['maxregshift'] = 0.3
motion_ops_profile['maxregshiftNR'] = 10

motion_result = run_motion_correction_suite2p(
    movie_data=data_downsampled_interp[:,:,:],  
    save_path=save_path,
    series_id=series_id,
    motion_ops_profile=motion_ops_profile,
    dtype=np.int16,
)
reg_npy = motion_result['reg_npy']
print("Metadata (ops and output_ops) saved successfully.")
print("Saved .npy and cleaned up temp binary.")


processed_movie = np.load(reg_npy, mmap_mode='r')

ops = np.load(motion_result['ops_path'], allow_pickle=True).item()
output_ops = np.load(motion_result['output_ops_path'], allow_pickle=True).item()

yoff = output_ops['yoff']
xoff = output_ops['xoff']
corr = output_ops['corrXY'] # This is the registration correlation per frame

fig = plt.figure(figsize=(12, 4))
plt.plot(xoff, label='X offset', alpha=0.7)
plt.plot(yoff, label='Y offset', alpha=0.7)
plt.title("Registration Offsets (Pixels)")
plt.xlabel("Frame Number")
plt.ylabel("Shift")
plt.legend()
plt.ylim(-100, 100) 
plt.show()
fig.savefig(os.path.join(results_dir, f'{series_id}_registration_offsets.png'), dpi=150)


export_visualization_video(
    processed_movie,
    second_data=data_downsampled_interp,
    fps=ops['fs'],
    playback_speed=3,
    target_dir=series_dir,
    output_name=f"{series_id}_mot_corr_vs_input_x3_compressed.mp4",
    panel_titles=["Motion-corrected", "Input"],
)

# #%% USE HERE if you want to start from the already corrected dataset
# save_path = os.path.join(series_dir, 'suite2p_corrected')
# reg_file = os.path.join(save_path, f'{series_id}_corrected.bin')
# reg_npy = os.path.join(save_path, f'{series_id}_corrected.npy')
# bin_factor = 10
# # Loading the movie back as a memory map (fast)
# processed_movie = np.load(reg_npy, mmap_mode='r')
# downsampled_fr = series_meta['recording_settings']['frame_rate'] / bin_factor

# ops = np.load(os.path.join(save_path, 'motion_input_ops.npy'), allow_pickle=True).item()
# output_ops = np.load(os.path.join(save_path, 'motion_output_ops.npy'), allow_pickle=True)




#%% Crop the motion-corrected movie to the stable region

# Extract yrange and xrange from the end of the tuple
yrange = output_ops['yrange'] 
xrange = output_ops['xrange'] 

# Apply the crop to your corrected movie
# Format: movie[frames, y_start:y_end, x_start:x_end]
# processed_movie_cropped = processed_movie[:, yrange[0]:yrange[1], xrange[0]:xrange[1]]
processed_movie_cropped = processed_movie

# %% Mean and std image save
mean_image_path = os.path.join(results_dir, f'{series_id}_mean_std_images.png')
plot_mean_std_projection(
    processed_movie_cropped,
    save_path=mean_image_path,
    figsize=(16, 4),
    mean_cmap='magma',
    std_cmap='magma',
    mean_title='Mean Image (Motion Corrected & Cropped)',
    std_title='Std Image (Motion Corrected & Cropped)',
    show_axes=False,
    dpi=150,
)
print(f"Saved mean and std images to: {mean_image_path}")
#%% ROI selection (manual or automatic)
roi_names_input = ['dendrites', 'AT']  # Optional list of custom names e.g. ['dendrite_1', 'spine_1']; set to None for auto-naming (ROI1, ROI2 ...)


%matplotlib qt
extraction_image = processed_movie_cropped.mean(axis=0)
auto_roi_params = get_auto_roi_params(auto_roi_profile)
if auto_roi_params is not None:
    auto_roi_params= {'n_samples': 3,
    'footprint_size': 11,
    'gaussian_sigma': 1.0,
    'threshold_percentile': 98,
    'min_distance_factor': 0.5,
    'figsize': (20, 10),
    'watershed_threshold_percentile': 98,
    'compactness': 0.01,
    'min_area_factor': 0.4,
    'max_area_factor': 7.0,
    'relative_peak_fraction': 0.8}

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
    print(f"{roi_name}: shape {trace.shape}")


#%% Plot ROI masks and background-subtracted traces with stimulus periods
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



#%% Build stim segments
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

#%% Build nested ROI dictionary (raw, bg-subtracted, mask, trial-averaged traces)
stimulus_ids_unique = sorted({seg_stim_id for _, _, seg_stim_id in stim_segments})
stimulus_names_unique = [vial_to_odor.get(stim_id, f"V{stim_id}") for stim_id in stimulus_ids_unique]
stim_ids_full_trace = np.asarray(stimulus_id_trace).astype(int)

context_window_s = 5.0
context_window_frames = int(round(context_window_s * downsampled_fr))

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
        padded_context_trials = np.full((len(trial_traces_with_context), trial_max_len_context), np.nan, dtype=float)
        padded_context_stim = np.full((len(trial_stim_id_traces_with_context), trial_max_len_context), np.nan, dtype=float)
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
#%% Save the dataset
series_meta['stim_info'] = {
    'aurora_vial_info': vial_to_odor,
    'stim_id_trace': stimulus_id_trace,
    }
series_meta['mean_image'] = extraction_image
series_meta['background_info'] = {
    'mask': background_mask,
    'polygon': background_polygon,
    'raw_trace': background_raw_trace
}


session_results = {
    'metadata': series_meta,
    'rois': {}
}
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

# 2. Save to disk
p_data_save_path = os.path.join(experiment_dir, f'{container_id}_processed_data')
os.makedirs(p_data_save_path, exist_ok=True)
save_file = os.path.join(p_data_save_path, f'{series_id}_processed_data.pkl')
with open(save_file, 'wb') as f:
    pickle.dump(session_results, f)

print(f"Datasets saved to: {save_file}")

#%% Plot trial-averaged responses with ROI mask image + dF/F traces (+/- 5 s)
roi_trial_plot_dir = os.path.join(results_dir, 'roi_trial_average_plots')
os.makedirs(roi_trial_plot_dir, exist_ok=True)

pre_window_s = 5.0
post_window_s = 15
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
# Smooth traces for plotting
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
        roi_responsive_df=None,
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

print(f"Saved ROI mask + trial response plots to: {roi_trial_plot_dir}")

    # %%
