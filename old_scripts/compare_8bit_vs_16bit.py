#%%
import os
import numpy as np
from utilities import downsample_video, export_visualization_video, temporal_denoise
import matplotlib.pyplot as plt
from roi_processor import ROISelector


def plot_roi_lost_vals_dashboard(mean_image, roi_masks, roi_names, roi_traces, lost_vals, fs=6.25):
    if len(roi_masks) == 0:
        print("No ROI masks found. Please select at least one ROI first.")
        return

    n_rois = len(roi_masks)
    fig, axes = plt.subplots(
        n_rois,
        4,
        figsize=(20, 4.5 * n_rois),
        squeeze=False,
        gridspec_kw={'width_ratios': [1.35, 1.35, 0.55, 0.45]},
    )

    for i, (mask, trace) in enumerate(zip(roi_masks, roi_traces)):
        roi_name = roi_names[i] if i < len(roi_names) else f"ROI{i + 1}"

        masked_vals = lost_vals[:, mask]
        zero_prop = (masked_vals == 0).mean(axis=1)
        nonzero_prop = 1.0 - zero_prop
        overall_zero = float((masked_vals == 0).mean())
        overall_nonzero = 1.0 - overall_zero

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

        ax2 = axes[i, 2]
        lost_time = np.arange(lost_vals.shape[0]) / fs
        ax2.plot(lost_time, zero_prop *100, label='0s', linewidth=1.2)
        ax2.plot(lost_time, nonzero_prop *100, label='non-zeros', linewidth=1.2)
        ax2.set_title(f'{roi_name} - Per-frame proportion')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Proportion (%)')
        ax2.set_ylim(-0.1, 100.1)
        ax2.set_yticks(np.arange(0, 101, 10))
        ax2.set_yticks(np.arange(0, 101, 5), minor=True)
        ax2.grid(True, which='major', alpha=0.5, linestyle='-')
        ax2.grid(True, which='minor', alpha=0.25, linestyle='--')
        ax2.legend(loc='best')

        ax3 = axes[i, 3]
        ax3.bar(1, [overall_zero * 100], width=0.5, color='green', label='0s')
        ax3.bar(1, [overall_nonzero * 100], width=0.5, bottom=[overall_zero * 100], color='magenta', label='non-zeros')
        ax3.set_title(f'{roi_name} - Overall proportion')
        ax3.set_ylabel('Proportion (%)')
        ax3.set_ylim(-0.1, 100.1)
        ax3.set_xlim(0.5, 1.5)
        ax3.set_yticks(np.arange(0, 101, 10))
        ax3.set_yticks(np.arange(0, 101, 5), minor=True)
        ax3.grid(True, which='major', alpha=0.5, linestyle='-')
        ax3.grid(True, which='minor', alpha=0.25, linestyle='--')
        ax3.legend(loc='best', fontsize='x-small')

    plt.tight_layout()
    plt.show()
from roi_processor import ROISelector

#%% Load the bin file using memory mapping
# Path to your converted file


dir = '/Volumes/tungsten/scratch/gfelsenb/Hanna/2p-imaging/2026_02_27/fly3/session_post'
series_id = 'S1-T24250'
video_dir = os.path.join(dir, series_id)
video_dir = dir

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

dtype = np.dtype('B')
try:
    with open(bin_path, "rb") as f:
        numpy_data = np.fromfile(f, dtype = dtype)
    print(numpy_data)
except IOError:
    print('Error While Opening the file!')

bin_data_8_raw = numpy_data[::2]  # take the downsampled data
lost_data_raw = numpy_data[1::2]  # take the lost data

n_frames = int((lost_data_raw.shape[0] / data_width) / data_height)#int(float(config.get('main', 'scan.number.set')[1:-1]))

# extracting the total lenght of the file
# reshaping data: first dimension n of frames, than x and y
lost_data_8 = np.reshape(lost_data_raw[:-2], (n_frames, data_height, data_width))
bin_data_8 = np.reshape(bin_data_8_raw[:-2], (n_frames, data_height, data_width))

#%% Sample data for testing
bin_sample_16 = bin_data_16[:].copy()
lost_vals = lost_data_8[:].copy() # 2x
bin_sample_8 = bin_data_8[:].copy() # 2x

#%% Downsample 
bin16_downsampled = downsample_video(bin_sample_16, 10)
bin8_downsampled = downsample_video(bin_sample_8, 10)
#%% Compare the mean images
mean_16 = bin16_downsampled.mean(axis=0)
mean_8 = bin8_downsampled.mean(axis=0)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(mean_16, cmap='gray')
plt.title('Mean Image - 16-bit')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(mean_8, cmap='gray')
plt.title('Mean Image - 8-bit')
plt.axis('off')
plt.tight_layout()
plt.show()
#%% Manual ROI selection on mean image of bin_sample_16
mean_img = bin8_downsampled.std(axis=0)
mean_img = bin16_downsampled.std(axis=0)

# selected_data = bin8_downsampled
selected_data = bin16_downsampled
%matplotlib qt
selector = ROISelector(
    movie=selected_data,
    sd_map=mean_img,
    fs=62.5,
)

plt.show(block=True)

roi_masks = selector.roi_masks
roi_names = selector.roi_names
roi_traces = selector.raw_traces

print(f"Selected {len(roi_masks)} ROIs: {roi_names}")

#%% Plot ROI dashboard: mask, trace, per-frame proportions, overall proportions
%matplotlib qt
plot_roi_lost_vals_dashboard(
    mean_image=mean_img_16,
    roi_masks=roi_masks,
    roi_names=roi_names,
    roi_traces=roi_traces,
    lost_vals=lost_vals,
    fs=6.25,
)

# %%

# %%
