#%%
import os
import numpy as np
from utilities import downsample_video, export_visualization_video, temporal_denoise
import matplotlib.pyplot as plt
from roi_processor import ROISelector


def plot_roi_dashboard(mean_image, roi_masks, roi_names, roi_traces, fs=6.25):
    """Simple ROI dashboard: mask overlay and trace for each ROI (no 'lost vals')."""
    if len(roi_masks) == 0:
        print("No ROI masks found. Please select at least one ROI first.")
        return

    n_rois = len(roi_masks)
    fig, axes = plt.subplots(n_rois, 2, figsize=(10, 4.5 * n_rois), squeeze=False,
                             gridspec_kw={'width_ratios': [1.35, 1.35]})

    for i, (mask, trace) in enumerate(zip(roi_masks, roi_traces)):
        roi_name = roi_names[i] if i < len(roi_names) else f"ROI{i + 1}"

        ax0 = axes[i, 0]
        ax0.imshow(mean_image, cmap='gray', origin='upper')
        overlay = np.ma.masked_where(~mask, mask)
        ax0.imshow(overlay, cmap='Pastel1', alpha=0.5, origin='upper')
        ax0.set_title(f'{roi_name} - Mask')
        ax0.axis('off')

        ax1 = axes[i, 1]
        trace = np.asarray(trace)
        trace_time = np.arange(trace.shape[0]) / fs
        ax1.plot(trace_time, trace, color='black', linewidth=1)
        ax1.set_title(f'{roi_name} - Trace')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Signal')

    plt.tight_layout()
    plt.show()
from roi_processor import ROISelector

#%% Load the bin file using memory mapping
# Path to your converted file


dir = '/Users/guerbura/Desktop/current/data/2p/hanna/'
series_id = 'S1-T24250'
video_dir = os.path.join(dir, series_id)

bin_path = os.path.join(video_dir, f'{series_id}_ch525.bin')

# Microscope settings (Verify these!)
data_width = 750
data_height = 400
dtype = 'uint16' # Raw data is almost always uint16 or int16
frames = 13200


# 2. Map the binary file
# 'r' for read-only. Use 'c' (copy-on-write) if you want to modify it in RAM 
# without changing the file on disk.
bin_data_16 = np.memmap(
    bin_path, 
    dtype=dtype, 
    mode='r', 
    shape=(frames, data_height, data_width),
    order='C' # Most 2P software writes in C-order (Row-major)
)

print(f"Binary Mapped: {bin_path}")
print(f"Data Shape: {bin_data_16.shape}") 

#%% Sample data for testing
bin_sample_16 = bin_data_16[:].copy()

#%% Downsample 
bin16_downsampled = downsample_video(bin_sample_16, 10)

#%% plot the mean and std images of the downsampled video
mean_16 = bin16_downsampled.mean(axis=0)
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(mean_16, cmap='gray')
plt.title('Mean Image - 16-bit')
plt.axis('off')

std_16 = bin16_downsampled.std(axis=0)
plt.subplot(1, 2, 2)
plt.imshow(std_16, cmap='hot')
plt.title('Std Image - 16-bit')
plt.axis('off')
fig.savefig(os.path.join(video_dir, f'{series_id}_mean_std_images.png'), dpi=150)
#%% Manual ROI selection on mean image of bin_sample_16

selected_data = bin16_downsampled
%matplotlib qt
selector = ROISelector(
    movie=selected_data,
    sd_map=std_16,
    fs=62.5,
)

plt.show(block=True)

roi_masks = selector.roi_masks
roi_names = selector.roi_names
roi_traces = selector.raw_traces

print(f"Selected {len(roi_masks)} ROIs: {roi_names}")

#%% Plot ROI dashboard: mask + trace (no lost vals)
%matplotlib qt
plot_roi_dashboard(
    mean_image=std_16,
    roi_masks=roi_masks,
    roi_names=roi_names,
    roi_traces=roi_traces,
    fs=6.25,
)

# %%

# %%
