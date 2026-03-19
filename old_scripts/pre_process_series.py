#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle 
import suite2p
import suite2p.io.utils
from suite2p import io
from utilities import downsample_video, export_visualization_video, median_denoise_temporal, temporal_denoise
#%% Load the bin file using memory mapping
# Path to your converted file
main_dir = "/Volumes/tungsten/scratch/gfelsenb/Ana/2p-imaging/burak/"
video_dir = '2026_02_Gamma1_boutons/20260205_g1_fly1/S1-T23893'  # Update this path accordingly
video_name = 'S1-T23893_ch525'  # Update this filename accordingly
file_path = os.path.join(main_dir, video_dir, f'{video_name}.bin')

# Microscope settings (Verify these!)
data_width = 750
data_height = 400
dtype = 'uint16' # Raw data is almost always uint16 or int16
frame_rate = 62.5

# 1. Calculate how many frames are in the file based on file size
file_size = os.path.getsize(file_path)
bytes_per_pixel = np.dtype(dtype).itemsize
frames_in_file = file_size // (data_height * data_width * bytes_per_pixel)

# 2. Map the binary file
# 'r' for read-only. Use 'c' (copy-on-write) if you want to modify it in RAM 
# without changing the file on disk.
raw_data = np.memmap(
    file_path, 
    dtype=dtype, 
    mode='r', 
    shape=(frames_in_file, data_height, data_width),
    order='C' # Most 2P software writes in C-order (Row-major)
)
print(f"Binary Mapped: {file_path}")
print(f"Data Shape: {raw_data.shape}") 
print(f"Duration: {raw_data.shape[0] / frame_rate:.2f} seconds")

#%%
# Known dimensions from your previous steps
Ly, Lx = 400, 750
n_frames = 1200 # or however many frames your log file indicates

# Get the actual file size in bytes
file_size_bytes = os.path.getsize(file_path)

# Calculate potential sizes
bytes_8bit = n_frames * Ly * Lx * 1  # 1 byte per pixel
bytes_16bit = n_frames * Ly * Lx * 2 # 2 bytes per pixel
bytes_32bit = n_frames * Ly * Lx * 4 # 4 bytes per pixel

print(f"Actual File Size: {file_size_bytes} bytes")
print(f"Expected if 8-bit:  {bytes_8bit}")
print(f"Expected if 16-bit: {bytes_16bit}")
print(f"Expected if 32-bit: {bytes_32bit}")
#%%
dtype = np.int16
try:
    with open(file_path, "rb") as f:
        numpy_data = np.fromfile(f, dtype)
    print(numpy_data)
except IOError:
    print('Error While Opening the file!')

#%%
data_PMT1 = numpy_data
x_dim = 400
    # compute the y dimension
y_dim = 750
n_frames = int((data_PMT1.shape[0] / x_dim) / y_dim)#int(float(config.get('main', 'scan.number.set')[1:-1]))
    
# extracting the total lenght of the file
# reshaping data: first dimension n of frames, than x and y
reshaped_PMT1 = np.reshape(data_PMT1[:-2], (n_frames, x_dim, y_dim))
print('This is my stack shape (t, x, y): ' + str(reshaped_PMT1.shape))

#%% Load .npy file (if you have already converted)

npy_file_path = os.path.join(main_dir, video_dir, f'{video_name}_PMT1.npy')
data_npy = np.load(npy_file_path, mmap_mode='r+').astype('uint16')  # Ensure dtype matches raw data
print(f"Data Shape: {data_npy.shape}")
print(f"Data Type: {data_npy.dtype}")

#%%

# Compare SNR characteristics between data and data_npy
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Mean and Std projections for both datasets
mean_data = np.mean(raw_data[:100], axis=0)
std_data = np.std(raw_data[:100], axis=0)
mean_npy = np.mean(reshaped_PMT1[:100], axis=0)
std_npy = np.std(reshaped_PMT1[:100], axis=0)


axes[0, 0].imshow(mean_data, cmap='gray')
axes[0, 0].set_title("Mean Projection (data)")
axes[0, 1].imshow(mean_npy, cmap='gray')
axes[0, 1].set_title("Mean Projection (data_npy)")
axes[1, 0].imshow(std_data, cmap='hot')
axes[1, 0].set_title("Std Projection (data) - SNR proxy")
axes[1, 1].imshow(std_npy, cmap='hot')
axes[1, 1].set_title("Std Projection (data_npy) - SNR proxy")

for ax in axes.flat:
    ax.axis('off')
    

plt.tight_layout()
plt.show()

# Quantitative SNR comparison
snr_data = np.mean(mean_data) / (np.mean(std_data) + 1e-6)
snr_npy = np.mean(mean_npy) / (np.mean(std_npy) + 1e-6)
print(f"SNR (data): {snr_data:.3f}")
print(f"SNR (data_npy): {snr_npy:.3f}")
fig.savefig(os.path.join(main_dir, video_dir, "data_vs_npy_snr_comparison.png"), dpi=300)


# %%
# Calculate projections using a subset of frames to save time
# (e.g., first 1000 frames)
sample_data = raw_data[:1000].astype('float32')
mean_img = np.mean(sample_data, axis=0)
std_img = np.std(sample_data, axis=0)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(mean_img, cmap='gray')
ax[0].set_title('Mean Projection (Anatomy)')
ax[1].imshow(std_img, cmap='hot')
ax[1].set_title('Std Projection (Activity)')
plt.show()
fig.savefig(os.path.join(main_dir, video_dir, "projections.png"), dpi=300)
# %%
# Example: Pixel at y=100, x=150
plt.figure(figsize=(15, 4))
plt.plot(raw_data[:, 100, 150])
plt.title("Raw Intensity Trace (62.5 Hz)")
plt.xlabel("Frames")
plt.ylabel("Intensity")
plt.show()
# %% Downsampling, denoising, and visualization
bin_factor = 5 

data_downsampled = downsample_video(raw_data, bin_factor)

print(f"Original shape: {raw_data.shape}")
print(f"Downsampled shape: {data_downsampled.shape} at {62.5/bin_factor} Hz")

# Denoising
# denoised_median = median_denoise_temporal(data_downsampled, size=3)
data_denoised = temporal_denoise(data_downsampled, window_size=7)

#%%
# Sample videos
start_second = 30
end_second = 60
export_visualization_video(
    data_downsampled[:1000],
    second_data=data_denoised[:1000],
    fps=frame_rate/bin_factor,
    target_dir=os.path.join(main_dir, video_dir),
    panel_titles=["Downsampled Data", "Downsampled smoothened Data"],
    output_name="movie.mp4"
)
# export_visualization_video(denoised_median[:int(end_second*62.5)//bin_factor] , fps=62.5/bin_factor, target_dir=os.path.join(main_dir, video_dir), output_name="downsampled_median_denoised_visualization.mp4")
#%%
pixel_y, pixel_x = 100, 150 # Replace with coordinates of a visible bouton
raw_trace = raw_data[int(start_second*62.5):int(end_second*62.5), pixel_y, pixel_x]
ds_trace = data_downsampled[int(start_second*62.5)//bin_factor:int(end_second*62.5)//bin_factor, pixel_y, pixel_x]
denoised_trace = data_denoised[int(start_second*62.5)//bin_factor:int(end_second*62.5)//bin_factor, pixel_y, pixel_x]

plt.figure(figsize=(12, 4))
plt.plot(np.linspace(0, (end_second-start_second), int((end_second-start_second)*62.5)), raw_trace, label='Raw (62.5 Hz)', alpha=0.5)
plt.plot(np.linspace(0, (end_second-start_second), int((end_second-start_second)*62.5/bin_factor)), ds_trace, label='Downsampled (12.5 Hz)', linewidth=2)
plt.plot(np.linspace(0, (end_second-start_second), int((end_second-start_second)*62.5/bin_factor)), denoised_trace, label='Downsampled + Denoised', linewidth=2)    
plt.legend()
plt.title("Effect of Temporal Binning on Signal SNR")
plt.xlabel("Seconds")
plt.ylabel("Intensity")
plt.show()

plt.savefig(os.path.join(main_dir, video_dir, "signal_comparison.png"), dpi=300)

#%% data to use
# For interactive ROI selection and ΔF/F extraction, we will use the downsampled and denoised data
processed_data = data_downsampled
roi_data = processed_data[int(start_second*62.5)//bin_factor:int(end_second*62.5)//bin_factor]  # Subset for interactive analysis
# %%
%matplotlib qt
# Assuming 'data_downsampled' is your (Frames, H, W) array
mean_img = np.mean(processed_data, axis=0)
std_img = np.std(processed_data, axis=0)

# Assuming 'data_downsampled' is (Frames, H, W)
mean_img = np.mean(processed_data, axis=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
plt.subplots_adjust(bottom=0.2)

# Anatomy Display
im = ax1.imshow(std_img, cmap='hot')
ax1.set_title("Click to select Boutons")
ax2.set_title("ΔF/F Traces")
ax2.set_xlabel("Time (frames)")

# Data storage
roi_coords = []
traces = []
spatial_kernel_size = 10
color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
background_trace = None
iROI=0
def onclick(event):
    global background_trace, iROI
    if event.inaxes != ax1:
        return
    
    # 1. Get coordinates
    ix, iy = int(round(event.xdata)), int(round(event.ydata))
    roi_coords.append((ix, iy))
    
    # Pick color based on ROI index
    color = color_cycle[(len(roi_coords) - 1) % len(color_cycle)]
    
    # 2. Draw the ROI on the image (a square)
    rect = Rectangle(
        (ix - spatial_kernel_size / 2, iy - spatial_kernel_size / 2),
        spatial_kernel_size, spatial_kernel_size,
        linewidth=1, edgecolor=color, facecolor='none'
    )
    ax1.add_patch(rect)
    
    # 3. Extract and Process Signal
    roi_data = processed_data[:, iy-int(spatial_kernel_size/2):iy+int(spatial_kernel_size/2),
                              ix-int(spatial_kernel_size/2):ix+int(spatial_kernel_size/2)]
    raw_trace = np.mean(roi_data, axis=(1, 2))
    if background_trace is None:
        background_trace = raw_trace
    else:
        raw_trace = raw_trace - background_trace
        iROI += 1
    
    if iROI==0:
        return
    # Quick ΔF/F calculation: (F - F_median) / F_median
    # f0 = np.percentile(raw_trace, 5)
    # f0 = np.median(raw_trace)
    # if f0 == 0: f0 = 1e-6
    # df_f = (raw_trace - f0) / f0
    traces.append((raw_trace, color))
    
    # 4. Update the Trace Plot
    ax2.clear()
    for i, (t, c) in enumerate(traces):
        ax2.plot(t + (i * 10), color=c, label=f'ROI {i}')
    
    ax2.set_title(f"Active ROIs: {len(traces)}")
    ax1.figure.canvas.draw()

# Connect the click event
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

# %% Motion correction
# Which data to proceed with
to_correct_data = data_downsampled[:1000] # data_downsampled, data_denoised

# Create a reference image from frames with actual signal 
# Signal is captured here as high std pixels, which correspond to active regions
ref_img = to_correct_data.std(axis=0).astype(np.int16)


# 1. Define a temporary binary file for Suite2p to write into
# Suite2p needs a file path to 'f_reg' to actually save the result
save_path = os.path.join(main_dir, video_dir, 'suite2p_run')
if not os.path.exists(save_path): os.mkdir(save_path)
reg_file = os.path.join(save_path, 'data.bin')

# 3. Create the binary file on disk first
# This creates a placeholder that Suite2p can write into
n_frames, Ly, Lx = to_correct_data.shape
f_reg = io.BinaryFile(Ly=Ly, Lx=Lx, filename=reg_file, n_frames=n_frames, dtype=np.int16)

# Configure motion correction parameters
ops = suite2p.default_ops()
ops['fs'] = frame_rate/bin_factor           # Your downsampled FR
ops['nonrigid'] = True     # Critical for small structures
ops['block_size'] = [128, 128] # Patch size for non-rigid
ops['smooth_sigma'] = 2 # Smoothing for non-rigid shifts
ops['smooth_sigma_time'] = 1 # Temporal smoothing of shifts
ops['threshold_scaling'] = 0.5 # Better for small signals
ops['save_path0'] = save_path
ops['main_chan'] = 0               # Ensure it's looking at the GCaMP channel

ops['snr_thresh'] = 1.0            # Only align if signal is above this noise floor
ops['maxregshift'] = 0.1           # Max rigid shift as fraction of image size
ops['maxregshiftNR'] = 20           # Max pixels a non-rigid block can move (tighten this!)

# 4. Run Registration
# We pass the BinaryFile object (f_reg) and the data array (f_raw)
output_ops = suite2p.registration.registration_wrapper(
    f_reg=f_reg, 
    f_raw=to_correct_data, 
    ops=ops,
    refImg=ref_img
)
corrected_movie = np.memmap(reg_file, dtype='int16', mode='r', shape=(n_frames, Ly, Lx))
print(f"Registration successful! Final shape: {corrected_movie.shape}")
#%%
# 5. Access the stabilized movie
# Save to local drive
local_path = '/Users/guerbura/Desktop/current/2p-movies'
export_visualization_video(
    corrected_movie,
    second_data=to_correct_data,
    fps=ops['fs'],
    target_dir=local_path,
    output_name=f"{video_name}_mot_corr_vs_input.mp4",
    panel_titles=["Motion-corrected", "Input"],
)

# %%
