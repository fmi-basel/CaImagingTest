import numpy as np
import os
import cv2
import configparser
import matplotlib.pyplot as plt
import tempfile
import suite2p
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy import stats
from tqdm import tqdm
import pandas as pd
import torch


def load_experiment_metadata(db_csv_path, series_id_value):
    df = pd.read_csv(db_csv_path)
    match = df.loc[df['seriesID'] == series_id_value]
    if match.empty:
        raise ValueError(f"No row found for series_id={series_id_value}")
    return match.iloc[0].to_dict()

def find_bin_files(source_dir):
	bin_files = []
	for root, _, files in os.walk(source_dir):
		for filename in files:
			if filename.lower().endswith('.bin'):
				bin_files.append(os.path.join(root, filename))
	return bin_files

def convert_bin_to_npy(file_path, width, height, dtype='uint16'):
    # Read the binary data
    raw_data = np.fromfile(file_path, dtype=dtype)
    
    # Reshape to (Frames, Height, Width)
    # The number of frames is inferred from the total size
    num_frames = len(raw_data) // (width * height)
    video_array = raw_data.reshape((num_frames, height, width))
    
    # Save as .npy
    save_path = file_path.replace('.bin', '.npy')
    np.save(save_path, video_array)
    return save_path


def downsample_video(data, bin_factor, show_progress=False, chunk_size_out_frames=128):
    num_frames, height, width = data.shape
    new_num_frames = num_frames // bin_factor

    if new_num_frames == 0:
        return np.empty((0, height, width), dtype=np.float32)
    
    # Trim data to be perfectly divisible by bin_factor
    trimmed_data = data[:new_num_frames * bin_factor]

    if not show_progress:
        # Reshape to (NewFrames, BinFactor, Height, Width)
        # Then mean across the BinFactor axis
        downsampled = trimmed_data.reshape(new_num_frames, bin_factor, height, width).mean(axis=1, dtype=np.float32)
        return downsampled.astype('float32') # Float32 is better for precision after averaging

    downsampled = np.empty((new_num_frames, height, width), dtype=np.float32)
    chunk_size_out_frames = max(1, int(chunk_size_out_frames))
    total_chunks = (new_num_frames + chunk_size_out_frames - 1) // chunk_size_out_frames

    for chunk_idx in tqdm(range(total_chunks), desc="Downsampling", unit="chunk"):
        out_start = chunk_idx * chunk_size_out_frames
        out_end = min(new_num_frames, (chunk_idx + 1) * chunk_size_out_frames)

        in_start = out_start * bin_factor
        in_end = out_end * bin_factor

        chunk = trimmed_data[in_start:in_end].reshape(out_end - out_start, bin_factor, height, width)
        downsampled[out_start:out_end] = chunk.mean(axis=1, dtype=np.float32)

    return downsampled


def export_visualization_video(
    data,
    target_dir=None,
    fps=None,
    playback_speed=3,
    output_name="analysis_check.mp4",
    second_data=None,
    panel_titles=None,
):
    num_frames, height, width = data.shape

    if fps is None:
        fps = 62.5  # Default frame rate; adjust if your data has a different frame rate
    # Calculate output FPS (e.g., 6.25 * 3 = 18.75 fps)
    out_fps = fps * playback_speed

    # Define the codec and create VideoWriter object
    # 'mp4v' is good for compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if target_dir is not None:
        os.makedirs(target_dir, exist_ok=True)
        output_path = os.path.join(target_dir, output_name)
    else:
        output_path = output_name

    is_two_panel = second_data is not None
    out_size = (width * 2, height) if is_two_panel else (width, height)
    out = cv2.VideoWriter(output_path, fourcc, out_fps, out_size)

    # Normalize data for visualization (Scale 0 to 255)
    # We use the 1st and 99th percentile for contrast stretching
    vmin, vmax = np.percentile(data, [1, 99])
    if is_two_panel:
        vmin2, vmax2 = np.percentile(second_data, [1, 99])

    def normalize_frame(frame, frame_min, frame_max):
        frame = np.clip(frame, frame_min, frame_max)
        return ((frame - frame_min) / (frame_max - frame_min) * 255).astype(np.uint8)

    if is_two_panel:
        num_frames = min(num_frames, second_data.shape[0])

    for i in range(num_frames):
        # 1. Normalize frame to uint8
        frame_left = normalize_frame(data[i], vmin, vmax)
        frame_left_bgr = cv2.cvtColor(frame_left, cv2.COLOR_GRAY2BGR)

        # 2. Add Timestamp text
        timestamp = i / fps
        text = f"{timestamp:.2f}s"
        cv2.putText(frame_left_bgr, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if is_two_panel:
            frame_right = normalize_frame(second_data[i], vmin2, vmax2)
            if frame_right.shape != (height, width):
                frame_right = cv2.resize(frame_right, (width, height))
            frame_right_bgr = cv2.cvtColor(frame_right, cv2.COLOR_GRAY2BGR)
            cv2.putText(frame_right_bgr, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if panel_titles and len(panel_titles) >= 2:
                cv2.putText(frame_left_bgr, panel_titles[0], (10, height - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame_right_bgr, panel_titles[1], (10, height - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            frame_bgr = np.hstack([frame_left_bgr, frame_right_bgr])
            # Divider line between panels
            cv2.line(frame_bgr, (width, 0), (width, height - 1), (255, 255, 255), 2)
        else:
            frame_bgr = frame_left_bgr

        # 3. Write frame
        out.write(frame_bgr)

    out.release()
    print(f"Video saved as {output_path}")

def spatial_denoise(data, sigma=1):
    """
    Applies a 2D Gaussian filter to each frame.
    sigma: standard deviation for Gaussian kernel. 
           Start with 1.0; increase for more smoothing.
    """
    smoothed_video = np.zeros_like(data)
    for i in range(data.shape[0]):
        smoothed_video[i] = gaussian_filter(data[i], sigma=sigma)
    return smoothed_video

def temporal_denoise(data, window_size=5, poly_order=2):
    """
    Applies Savitzky-Golay filter along the time axis.
    window_size: must be an odd integer.
    poly_order: order of the polynomial used for fitting.
    """
    # axis=0 is the time/frame axis
    return savgol_filter(data, window_size, poly_order, axis=0)

def median_denoise_temporal(data, size=3):
    # Apply a 3-frame median filter across time
    return median_filter(data, size=(size, 1, 1))


def parse_osf_stimulus_sequence(osf_path):
    with open(osf_path, 'r') as file:
        lines = file.readlines()

    sequence = []
    for line in lines[2:]:
        parts = line.strip().split()
        if parts and parts[0].isdigit():
            sequence.append(int(parts[0]))
    return sequence


def map_stimulus_ids_from_osf(stim_on_trace, osf_path):
    stim_on_trace = np.asarray(stim_on_trace).astype(int)
    stimulus_id_trace = np.zeros(stim_on_trace.size, dtype=int)

    padded = np.concatenate(([0], stim_on_trace, [0]))
    changes = np.diff(padded)
    stim_starts = np.where(changes == 1)[0]
    stim_ends = np.where(changes == -1)[0]

    stimulus_sequence = parse_osf_stimulus_sequence(osf_path)

    for stim_idx, (start_idx, end_idx) in enumerate(zip(stim_starts, stim_ends)):
        if stim_idx < len(stimulus_sequence):
            stimulus_id_trace[start_idx:end_idx] = stimulus_sequence[stim_idx]

    return stimulus_id_trace, stimulus_sequence, stim_starts, stim_ends


def parse_aurora_vial_info(vial_info):
    """Parse strings like '1:IAA,7:OCTT,8:MCH' into {1: 'IAA', 7: 'OCTT', 8: 'MCH'}."""
    vial_to_odor = {}

    if vial_info is None:
        return vial_to_odor

    if isinstance(vial_info, float) and np.isnan(vial_info):
        return vial_to_odor

    for entry in str(vial_info).split(','):
        entry = entry.strip()
        if not entry or ':' not in entry:
            continue

        vial_str, odor_name = entry.split(':', 1)
        vial_str = vial_str.strip()
        odor_name = odor_name.strip()

        if vial_str.isdigit() and odor_name:
            vial_to_odor[int(vial_str)] = odor_name

    return vial_to_odor


def read_ini_file(config_path, data_width=750):
    config = configparser.ConfigParser()
    config.read(config_path)
    if 'main' not in config:
        raise ValueError(f"INI file {config_path} is missing.")
    data_height = float(config.get('main', 'CRS.noScanLines').strip('"'))
    data_frames = float(config.get('main', 'scan.number.set').strip('"'))

    return data_width, data_height, data_frames


def read_lvd_data(lvd_path):
    with open(lvd_path, 'rb') as file:
        lvd_samplerate = np.fromfile(file, dtype='>f8', count=1)
        numchannels = np.fromfile(file, dtype='>f8', count=1)
        timestamp = np.fromfile(file, dtype='>f8', count=1)
        inputrange = np.fromfile(file, dtype='>f8', count=1)
        lvd_array = np.fromfile(file, dtype='>f8')

    lvd_data = np.reshape(lvd_array, (int(len(lvd_array) / numchannels[0]), int(numchannels[0])))
    return lvd_data, float(lvd_samplerate[0])


def plot_lvd_channels(lvd_data, lvd_channels, lvd_samplerate, start=0, end=-1, save_path=None):
    fig = plt.figure(figsize=(12, 6))
    time_trace = np.arange(len(lvd_data)) * (1000.0 / lvd_samplerate) / 1000
    for channel_name, channel_idx in lvd_channels.items():
        plt.plot(time_trace[start:end], lvd_data[start:end, channel_idx], label=channel_name)
    plt.legend()
    plt.title("LVD Channels Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Signal")
    if save_path:
        fig.savefig(save_path)
    plt.show()


def get_recording_frame_bounds(lvd_data, lvd_channels, galvo_prominence=1, digital_threshold=4):
    frame_peaks_raw, _ = find_peaks(lvd_data[:, lvd_channels['galvo']], prominence=galvo_prominence)
    shutter_opening = np.where(lvd_data[:, lvd_channels['shutter']] > digital_threshold)[0][0]
    start_frame = np.where(frame_peaks_raw < shutter_opening)[0][-1] + 1

    protocol_end = np.where(lvd_data[:, lvd_channels['protocol_end']] > digital_threshold)[0][0]
    end_frame = np.where(frame_peaks_raw > protocol_end)[0][0]
    return frame_peaks_raw, start_frame, end_frame


def build_frame_alignment_traces(lvd_data, frame_peaks_raw, stim_channel_idx, lvd_samplerate, data_frames):
    frame_num_trace = np.zeros(len(lvd_data), dtype=int)
    current_frame = 0
    peak_indices = set(frame_peaks_raw.tolist())

    for sample_idx in range(len(lvd_data)):
        if sample_idx in peak_indices:
            current_frame += 1
        frame_num_trace[sample_idx] = current_frame

    time_trace_ms = np.arange(len(lvd_data)) * (1000.0 / lvd_samplerate)
    stim_on_trace = (lvd_data[:, stim_channel_idx] > 1).astype(int)

    n_frames = int(data_frames)
    frame_time_trace_ms = np.zeros(n_frames)
    stim_on_trace_frames = np.zeros(n_frames, dtype=int)

    for frame_idx in range(1, n_frames + 1):
        frame_mask = frame_num_trace == frame_idx
        if np.any(frame_mask):
            frame_time_trace_ms[frame_idx - 1] = np.mean(time_trace_ms[frame_mask])
            stim_on_trace_frames[frame_idx - 1] = int(np.round(np.mean(stim_on_trace[frame_mask])))

    return frame_num_trace, frame_time_trace_ms, stim_on_trace_frames


def estimate_frame_rate(frame_peaks_raw, lvd_samplerate):
    periods = np.diff(frame_peaks_raw)
    return lvd_samplerate / np.mean(periods)


def load_video_memmap(video_path, frames, image_height, image_width, dtype='uint16', mode='c'):
    raw_data = np.memmap(
        video_path,
        dtype=dtype,
        mode=mode,
        shape=(int(frames), int(image_height), int(image_width)),
        order='C'
    )
    return raw_data


def downsample_and_align_traces(
    movie_data,
    start_frame,
    end_frame,
    target_fs,
    frame_time_trace_ms,
    stim_on_trace_frames
):
    frame_time_trace_clipped = np.asarray(frame_time_trace_ms[start_frame:end_frame], dtype=float)
    stim_on_trace_clipped = np.asarray(stim_on_trace_frames[start_frame:end_frame], dtype=int)
    movie_clipped = movie_data[start_frame:end_frame]

    if movie_clipped.shape[0] == 0:
        raise ValueError("No frames available after clipping. Check start_frame and end_frame values.")

    valid_time = frame_time_trace_clipped[np.isfinite(frame_time_trace_clipped)]
    if valid_time.size < 2:
        raise ValueError("Not enough valid frame times to estimate bin factor.")

    median_dt_ms = np.median(np.diff(valid_time))
    if median_dt_ms <= 0:
        raise ValueError("Invalid frame time trace. Non-positive frame interval detected.")

    estimated_fs = 1000.0 / median_dt_ms
    bin_factor = max(1, int(round(float(estimated_fs) / float(target_fs))))

    data_downsampled = downsample_video(movie_clipped, bin_factor, show_progress=False)

    num_downsampled_frames = data_downsampled.shape[0]
    num_frames_to_keep = num_downsampled_frames * bin_factor

    if num_downsampled_frames == 0:
        return (
            np.empty((0,) + movie_clipped.shape[1:], dtype=np.float32),
            np.array([], dtype=float),
            np.array([], dtype=int),
        )

    frame_time_trace_downsampled = frame_time_trace_clipped[:num_frames_to_keep].reshape(
        num_downsampled_frames,
        bin_factor,
    ).mean(axis=1)

    stim_on_trace_downsampled = stats.mode(
        stim_on_trace_clipped[:num_frames_to_keep].reshape(num_downsampled_frames, bin_factor),
        axis=1,
        keepdims=False,
    ).mode.astype(int)

    return data_downsampled, frame_time_trace_downsampled, stim_on_trace_downsampled




def run_motion_correction_suite2p(
    movie_data,
    save_path,
    series_id,
    motion_ops_profile,
    dtype=np.int16,
):
    os.makedirs(save_path, exist_ok=True)


    to_correct_data = movie_data.copy().astype(dtype)
    n_frames, ly, lx = to_correct_data.shape

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_reg_file = os.path.join(temp_dir, 'temp_reg.bin')
        # 1. Directly write the array to disk. 
        # This bypasses all suite2p 'write' naming bugs.
        to_correct_data.tofile(temp_reg_file)

        f_reg = suite2p.io.BinaryFile(
            Ly=ly,
            Lx=lx,
            filename=temp_reg_file,
            n_frames=n_frames,
            dtype=dtype,
            write=True,
        )
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        output_ops = suite2p.registration.registration_wrapper(
            f_reg=f_reg,
            f_raw=None,  
            save_path=temp_dir,
            settings=motion_ops_profile,
            device=device
        )

        corrected_movie = np.memmap(temp_reg_file, dtype=dtype, mode='r', shape=(n_frames, ly, lx))
        reg_npy = os.path.join(save_path, f'{series_id}_corrected.npy')
        np.save(reg_npy, corrected_movie)

        # windows fix
        del f_reg
        del corrected_movie

    motion_input_ops_path = os.path.join(save_path, 'motion_input_ops.npy') # input ops for motion correction
    motion_output_ops_path = os.path.join(save_path, 'motion_output_ops.npy') # output ops for motion correction
    np.save(motion_input_ops_path, motion_ops_profile)
    np.save(motion_output_ops_path, np.array(output_ops, dtype=object))

    return {
        'reg_npy': reg_npy,
        'ops_path': motion_input_ops_path,
        'output_ops_path': motion_output_ops_path,
        'save_path': save_path,
    }

