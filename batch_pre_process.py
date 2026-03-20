#%% Imports
import os
import glob
import numpy as np
import suite2p
from helpers_figures import plot_mean_std_projection
from batch_utilities import (
    load_series_metadata,
    process_signals,
    load_and_downsample_video,
    run_motion_correction,
)


#%% Adjust the following depending on the experiment
base_dir = "/Volumes/tungsten/scratch/gfelsenb/Ana/2p-imaging/burak/"
container_id = '2026_03_Beta1_counterconditioned_dendrites'
day_id = '2026_02_11'

downsampled_fr = 6.0
data_width = 750  # Hard-coded: .ini does not provide image width
lvd_channels = {'shutter': 0, 'galvo': 1, 'protocol_end': 2, 'stim_on': 3}

#%% Find all the series for the given day and container
experiment_dir = os.path.join(base_dir, container_id)
day_dir = os.path.join(experiment_dir, day_id)
if not os.path.exists(day_dir):
    raise FileNotFoundError(f"Directory {day_dir} does not exist. Please check the path and try again.")

db_path = os.path.join(experiment_dir, f'{container_id}_database.csv')
series_paths = sorted(glob.glob(os.path.join(day_dir, 'S1-T*')))
for series_path in series_paths:
    series_id = os.path.basename(series_path)
    print(f"Found series: {series_id}")

#%% Batch process each series up to and including motion correction
for series_path in series_paths:
    series_id = os.path.basename(series_path)
    print(f"\n{'='*60}")
    print(f"Processing series: {series_id}")
    print(f"{'='*60}")

    series_dir = os.path.join(day_dir, series_id)
    # If motion corrected output already exists, skip processing
    motion_corrected_dir = os.path.join(series_dir, 'suite2p_corrected')
    if os.path.exists(motion_corrected_dir) and len(os.listdir(motion_corrected_dir)) == 3:  # we output 3 files
        print(f"Motion corrected data already exists for {series_id}. Skipping processing.")
        continue    
    results_dir = os.path.join(series_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    experiment_id = f"{day_id}_{series_id}"

    series_meta, vial_to_odor = load_series_metadata(db_path, series_id, experiment_id)

    frame_rate, frame_time_trace_ms, stim_on_trace_frames, start_frame, end_frame = process_signals(
        series_dir, series_id, series_meta, lvd_channels, results_dir, data_width,
    )

    data_downsampled_interp, stim_on_trace_downsampled_interp = load_and_downsample_video(
        series_dir, series_id, series_meta, start_frame, end_frame,
        frame_time_trace_ms, stim_on_trace_frames, frame_rate, downsampled_fr,
    )
    np.save(os.path.join(series_dir, f'{series_id}_stim_trace.npy'), stim_on_trace_downsampled_interp)

    plot_mean_std_projection(
        data_downsampled_interp,
        save_path=os.path.join(results_dir, f'{series_id}_mean_std_projections.png'),
        figsize=(16, 4),
        mean_cmap='gray',
        std_cmap='hot',
        mean_title='Mean Projection',
        std_title='Std Projection',
        show_axes=True,
        dpi=150,
    )

    # Motion correction parameters — adjust per experiment type
    motion_ops_profile = suite2p.default_ops()
    motion_ops_profile['fs'] = downsampled_fr
    motion_ops_profile['nonrigid'] = False
    motion_ops_profile['block_size'] = (128, 128)
    motion_ops_profile['smooth_sigma'] = 2.5
    motion_ops_profile['save_path0'] = os.path.join(series_dir, 'suite2p_corrected')
    motion_ops_profile['main_chan'] = 0
    motion_ops_profile['maxregshift'] = 0.3
    motion_ops_profile['maxregshiftNR'] = 10

    processed_movie_cropped, ops, output_ops = run_motion_correction(
        data_downsampled_interp, series_id, series_dir, results_dir, motion_ops_profile,
    )

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

    print(f"Series {series_id} complete.")

