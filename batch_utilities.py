import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from utilities import (
    load_experiment_metadata,
    export_visualization_video,
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


def load_series_metadata(db_path, series_id, experiment_id):
    """Load experiment metadata and parse stimulus vial info."""
    series_meta = load_experiment_metadata(db_path, series_id)
    series_meta['experimentID'] = experiment_id
    vial_to_odor = parse_aurora_vial_info(series_meta.get('AuroraVialInfo'))
    if len(vial_to_odor) > 0:
        print(f"Stimulus vial mapping: {vial_to_odor}")
    return series_meta, vial_to_odor


def process_signals(series_dir, series_id, series_meta, lvd_channels, results_dir, data_width=750):
    """Read ini/lvd files, extract frame timing, and update recording_settings in series_meta."""
    config_path = os.path.join(series_dir, f'{series_id}_ch525.ini')
    _, data_height, data_frames = read_ini_file(config_path)

    lvd_path = os.path.join(series_dir, f'{series_id}.lvd')
    lvd_data, lvd_samplerate = read_lvd_data(lvd_path)

    plot_lvd_channels(
        lvd_data, lvd_channels, lvd_samplerate, start=0, end=-1,
        save_path=os.path.join(results_dir, f'{series_id}_lvd_channels.png'),
    )

    frame_peaks_raw, start_frame, end_frame = get_recording_frame_bounds(lvd_data, lvd_channels)
    print(f"Estimated recording start frame: {start_frame}, end frame: {end_frame}, "
          f"total frames: {end_frame - start_frame}")

    _, frame_time_trace_ms, stim_on_trace_frames = build_frame_alignment_traces(
        lvd_data, frame_peaks_raw, lvd_channels['stim_on'], lvd_samplerate, data_frames,
    )

    frame_rate = estimate_frame_rate(frame_peaks_raw, lvd_samplerate)
    print(f"Estimated frame rate: {frame_rate:.2f} Hz")

    series_meta['recording_settings'] = {
        'image_width': int(data_width),
        'image_height': int(data_height),
        'frames': int(data_frames),
        'frame_rate': int(round(frame_rate)),
    }
    return frame_rate, frame_time_trace_ms, stim_on_trace_frames, start_frame, end_frame


def load_and_downsample_video(series_dir, series_id, series_meta, start_frame, end_frame,
                               frame_time_trace_ms, stim_on_trace_frames, frame_rate, downsampled_fr):
    """Load raw binary video, temporally downsample and interpolate to a uniform time grid."""
    video_path = os.path.join(series_dir, f'{series_id}_ch525.bin')
    raw_data = load_video_memmap(
        video_path,
        series_meta['recording_settings']['frames'],
        series_meta['recording_settings']['image_height'],
        series_meta['recording_settings']['image_width'],
        dtype='uint16',
        mode='c',
    )
    print(f"Binary mapped: {video_path}")
    print(f"Data shape: {raw_data.shape}")
    print(f"Duration: {raw_data.shape[0] / series_meta['recording_settings']['frame_rate']:.2f} seconds")

    print(f"Temporal bin downsampling from ~{frame_rate:.2f} Hz to ~{downsampled_fr:.2f} Hz")
    data_downsampled, frame_time_trace_downsampled, stim_on_trace_downsampled = downsample_and_align_traces(
        raw_data, start_frame, end_frame, downsampled_fr, frame_time_trace_ms, stim_on_trace_frames,
    )

    # Interpolate to a uniform time grid (removes jitter from frame drops)
    expected_time_points = np.arange(
        frame_time_trace_downsampled[0], frame_time_trace_downsampled[-1], 1000 / downsampled_fr,
    )
    data_downsampled_interp = np.empty(
        (len(expected_time_points), data_downsampled.shape[1], data_downsampled.shape[2]),
        dtype=data_downsampled.dtype,
    )
    for y in range(data_downsampled.shape[1]):
        for x in range(data_downsampled.shape[2]):
            interp_func = interp1d(frame_time_trace_downsampled, data_downsampled[:, y, x], kind='linear')
            data_downsampled_interp[:, y, x] = interp_func(expected_time_points)

    stim_interp_func = interp1d(frame_time_trace_downsampled, stim_on_trace_downsampled, kind='nearest')
    stim_on_trace_downsampled_interp = stim_interp_func(expected_time_points)

    return data_downsampled_interp, stim_on_trace_downsampled_interp


def run_motion_correction(data_downsampled_interp, series_id, series_dir, results_dir, motion_ops_profile):
    """Run suite2p motion correction and save the registration offset plot and comparison video."""
    save_path = os.path.join(series_dir, 'suite2p_corrected')
    motion_result = run_motion_correction_suite2p(
        movie_data=data_downsampled_interp[:, :, :],
        save_path=save_path,
        series_id=series_id,
        motion_ops_profile=motion_ops_profile,
        dtype=np.int16,
    )

    processed_movie = np.load(motion_result['reg_npy'], mmap_mode='r')
    ops = np.load(motion_result['ops_path'], allow_pickle=True).item()
    output_ops = np.load(motion_result['output_ops_path'], allow_pickle=True)

    yoff = output_ops[4][0]
    xoff = output_ops[4][1]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(xoff, label='X offset', alpha=0.7)
    ax.plot(yoff, label='Y offset', alpha=0.7)
    ax.set_title("Registration Offsets (Pixels)")
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Shift")
    ax.legend()
    ax.set_ylim(-100, 100)
    fig.savefig(os.path.join(results_dir, f'{series_id}_registration_offsets.png'), dpi=150)
    plt.close(fig)

    export_visualization_video(
        processed_movie,
        second_data=data_downsampled_interp,
        fps=ops['fs'],
        playback_speed=3,
        target_dir=series_dir,
        output_name=f"{series_id}_mot_corr_vs_input_x3_compressed.mp4",
        panel_titles=["Motion-corrected", "Input"],
    )

    processed_movie_cropped = processed_movie  # Adjust using output_ops[-2]/[-1] if cropping needed
    return processed_movie_cropped, ops, output_ops
