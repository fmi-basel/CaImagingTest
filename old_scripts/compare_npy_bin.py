#%%
import os
import numpy as np
from utilities import downsample_video, export_visualization_video, temporal_denoise
#%% Load the bin file using memory mapping
# Path to your converted file


main_dir = "/Volumes/tungsten/scratch/gfelsenb/Ana/2p-imaging/burak/"

container_id = '2026_02_Gamma1_boutons'
day_id = 'discarded/260205_anabg'
series_id = 'S1-T23893'
video_dir = os.path.join(container_id, day_id, series_id)
video_name = 'S1-T23893_ch525'  # Update this filename accordingly
bin_path = os.path.join(main_dir, video_dir, f'{video_name}.bin')
npy_path = os.path.join(main_dir, video_dir, f'{video_name}.npy')

# Microscope settings (Verify these!)
data_width = 750
data_height = 400
dtype = 'uint16' # Raw data is almost always uint16 or int16
frame_rate = 62.5

# 1. Calculate how many frames are in the file based on file size
file_size = os.path.getsize(bin_path)
bytes_per_pixel = np.dtype(dtype).itemsize
frames_in_file = file_size // (data_height * data_width * bytes_per_pixel)

# 2. Map the binary file
# 'r' for read-only. Use 'c' (copy-on-write) if you want to modify it in RAM 
# without changing the file on disk.
bin_data = np.memmap(
    bin_path, 
    dtype=dtype, 
    mode='c', 
    shape=(frames_in_file, data_height, data_width),
    order='C' # Most 2P software writes in C-order (Row-major)
)

print(f"Binary Mapped: {bin_path}")
print(f"Data Shape: {bin_data.shape}") 
print(f"Duration: {bin_data.shape[0] / frame_rate:.2f} seconds")

#npy data
npy_data = np.load(npy_path, mmap_mode='r')
print(f"Numpy Mapped: {npy_path}")
print(f"Data Shape: {npy_data.shape}")
print(f"Duration: {npy_data.shape[0] / frame_rate:.2f} seconds")

# %%
bin_factor = 5 

bin_downsampled = downsample_video(bin_data[:4000].copy(), bin_factor)
npy_downsampled = downsample_video(npy_data[:4000].copy(), bin_factor)


bin_denoised = temporal_denoise(bin_downsampled, window_size=7)
npy_denoised = temporal_denoise(npy_downsampled, window_size=7)
#%% Compare the two arrays
print("\nComparing binary and numpy data...")
export_visualization_video(
    bin_data[:4000],
    second_data=npy_data[:4000],
    fps=frame_rate,
    target_dir=os.path.join(main_dir, video_dir),
    panel_titles=["Binary Data", "Numpy Data"],
    output_name="binary_vs_numpy_raw.mp4"
)
 # %%
