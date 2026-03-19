#%% 
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
import suite2p
import suite2p.io.utils
from suite2p import io
from utilities import downsample_video, export_visualization_video, median_denoise_temporal, temporal_denoise
from scipy.signal import find_peaks
from meta_utils import load_experiment_metadata
import configparser

#%% Set up paths and parameters
# Path to the data folder
main_dir = "/Volumes/tungsten/scratch/gfelsenb/Ana/2p-imaging/burak/"

container_id = '2026_02_Gamma1_boutons'
day_id = '260209_anabg'
series_id = 'S1-T23901'


experiment_dir = os.path.join(main_dir, container_id)  # Update this path accordingly
day_dir = os.path.join(experiment_dir, day_id)  # Update this path accordingly
series_dir = os.path.join(day_dir, series_id)  # Update this path accordingly

experiment_id = f"{day_id}_{series_id}"
#%% Read the database and find the information for the current series
db_path = os.path.join(experiment_dir, f'{container_id}_database.csv')
series_meta = load_experiment_metadata(db_path, series_id)
series_meta['experimentID'] = experiment_id

# Microscope settings
config = configparser.ConfigParser()
config_path = os.path.join(series_dir, f'{series_id}_ch525.ini')
config.read(config_path)

data_width = 700 # Couldn't extract from the ini file so hardcoded
data_height = float(config.get('main', 'CRS.noScanLines').strip('"'))
data_frames = float(config.get('main', 'scan.number.set').strip('"'))

# Frame times etc.
lvd_channels = {'shutter': 0, 'galvo': 1, 'protocol_end': 2, 'stim_on': 3}

lvd_path = os.path.join(series_dir, f'{series_id}.lvd')
with open(lvd_path, 'rb') as fid:
    # the first 4 values are header = we don't need them
    scanrateA = np.fromfile(fid, dtype='>f8', count=1) # Scan rate of the acquisition (Hz)
    numchannels = np.fromfile(fid, dtype='>f8', count=1)
    timestamp = np.fromfile(fid, dtype='>f8', count=1)
    inputrange = np.fromfile(fid, dtype='>f8', count=1)
    # extracting the real data
    # N.B: the dtype MUST be >f8
    lvd_array = np.fromfile(fid, dtype='>f8') # Acquired at 100Hz

# reshaping the data
lvd_data = np.reshape(lvd_array, (int(len(lvd_array) / numchannels), int(numchannels)))


start_lvd_plot = 0
end_lvd_plot = -1
plt.plot(lvd_data[start_lvd_plot:end_lvd_plot, lvd_channels['shutter']], label='Shutter')
plt.plot(lvd_data[start_lvd_plot:end_lvd_plot, lvd_channels['galvo']], label='Galvo')
plt.plot(lvd_data[start_lvd_plot:end_lvd_plot, lvd_channels['protocol_end']], label='End of Protocol')
plt.plot(lvd_data[start_lvd_plot:end_lvd_plot, lvd_channels['stim_on']], label='Stimulus On')
plt.legend()
plt.show()

# Find frame boundaries using the galvo signal and find the start and end of recording based on shutter opening and protocol end times
frame_peaks_raw, _ = find_peaks(lvd_data[:, lvd_channels['galvo']], prominence=1)
shutter_opening = np.where(lvd_data[:, lvd_channels['shutter']] > 4)[0][0]
start_frame = np.where(frame_peaks_raw < shutter_opening)[0][-1]

protocol_end = np.where(lvd_data[:, lvd_channels['protocol_end']] > 4)[0][0]
end_frame = np.where(frame_peaks_raw > protocol_end)[0][0]

print(f"Estimated recording start frame: {start_frame}, end frame: {end_frame}, total frames: {end_frame - start_frame}")

# Frame number: assign same frame number between peaks, increment at each peak
frame_num_trace = np.zeros(len(lvd_data), dtype=int)
current_frame = 0
for i in range(len(lvd_data)):
    if i in frame_peaks_raw:
        current_frame += 1
    frame_num_trace[i] = current_frame
# Create alignment arrays
lvd_samplerate = float(scanrateA[0])  # 1000 Hz
time_trace_ms = np.arange(len(lvd_data)) * (1000.0 / lvd_samplerate)  # Time in milliseconds
stim_on_trace = (lvd_data[:, lvd_channels['stim_on']] > 1).astype(int)  # Binary stimulus (1 or 0)

periods = [val - frame_peaks_raw[i - 1] for i, val in enumerate(frame_peaks_raw) if i > 0]
frame_rate = lvd_samplerate/np.mean(periods)
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
dtype = 'uint16' # Verify this is correct for your data (could be uint16, uint8, etc.)


# 1. Calculate how many frames are in the file based on file size
file_size = os.path.getsize(video_path)
bytes_per_pixel = np.dtype(dtype).itemsize

# 2. Map the binary file
# 'r' for read-only. Use 'c' (copy-on-write) if you want to modify it in RAM 
# without changing the file on disk.
raw_data = np.memmap(
    video_path, 
    dtype=dtype, 
    mode='c', 
    shape=(series_meta['recording_settings']['frames'], series_meta['recording_settings']['image_height'], series_meta['recording_settings']['image_width']),
    order='C' 
)

print(f"Binary Mapped: {video_path}")
print(f"Data Shape: {raw_data.shape}") 
print(f"Duration: {raw_data.shape[0] / series_meta['recording_settings']['frame_rate']:.2f} seconds")

# %% Downsampling, denoising, and visualization
bin_factor = 10

data_downsampled = downsample_video(raw_data, bin_factor)

print(f"Original shape: {raw_data.shape}")
print(f"Downsampled shape: {data_downsampled.shape} at {62.5/bin_factor} Hz")

# Denoising
# denoised_median = median_denoise_temporal(data_downsampled, size=3)
# data_denoised = temporal_denoise(data_downsampled, window_size=7)
#%% Plot SNR characteristics of the raw data
# Calculate mean and std projections as a quick SNR proxy
mean_projection = np.mean(raw_data[:], axis=0)
std_projection = np.std(raw_data[:], axis=0)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(mean_projection, cmap='gray')
axes[0].set_title("Mean Projection (Anatomy)")
axes[1].imshow(std_projection, cmap='hot')
axes[1].set_title("Std Projection (Activity)")
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()





# %% Motion correction
# Which data to proceed with
to_correct_data = data_downsampled[:1000].copy().astype(np.int16) # data_downsampled, data_denoised

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
ops['smooth_sigma'] = 2.0
ops['save_path0'] = save_path
ops['main_chan'] = 0               # Ensure it's looking at the GCaMP channel
ops['maxregshift'] = 0.1           # Max rigid shift as fraction of image size
ops['maxregshiftNR'] = 10           # Max pixels a non-rigid block can move 

# 4. Run Registration
# We pass the BinaryFile object (f_reg) and the data array (f_raw)
output_ops = suite2p.registration.registration_wrapper(
    f_reg=f_reg, 
    f_raw=to_correct_data, 
    ops=ops,
)
corrected_movie = np.memmap(reg_file, dtype='int16', mode='r', shape=(n_frames, Ly, Lx))
print(f"Registration successful! Final shape: {corrected_movie.shape}")
#%%
# 5. Access the stabilized movie
# Save to local drive
local_path = '/Users/guerbura/Desktop/current/2p-movies'
export_visualization_video(
    corrected_movie,
    second_data=data_downsampled[:1000],
    fps=ops['fs'],
    target_dir=local_path,
    output_name=f"{video_name}_mot_corr_vs_input.mp4",
    panel_titles=["Motion-corrected", "Input"],
)
#%% Crop the motion-corrected movie to the stable region

# Extract yrange and xrange from the end of the tuple
yrange = output_ops[-2] 
xrange = output_ops[-1] 

# Apply the crop to your corrected movie
# Format: movie[frames, y_start:y_end, x_start:x_end]
corrected_movie_cropped = corrected_movie[:, yrange[0]:yrange[1], xrange[0]:xrange[1]]
#%%

# Calculate the Standard Deviation across time
sd_map = corrected_movie_cropped.std(axis=0)

fig = plt.figure(figsize=(10, 5))
plt.imshow(sd_map, cmap='magma')
plt.title("Cropped SD Projection (Active Dendrites/Boutons)")
plt.colorbar(label='Intensity SD')
plt.show()
fig.savefig(os.path.join(main_dir, video_dir, "motion_corrected_sd_projection.png"), dpi=300)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
from matplotlib.patches import Polygon
from matplotlib import cm

class StackedROIProcessor:
    def __init__(self, movie, sd_map, fs=6.25, odor_onsets=None, spacing_factor=1.5):
        self.movie = movie
        self.sd_map = sd_map
        self.fs = fs
        self.odor_onsets = odor_onsets
        self.spacing_factor = spacing_factor # Multiplier for the stack offset
        self.rois = []
        self.traces = []
        self.current_offset = 0
        
        self.cmap = cm.get_cmap('tab10') 
        
        self.fig, (self.ax_map, self.ax_trace) = plt.subplots(1, 2, figsize=(16, 8))
        self.ax_map.imshow(sd_map, cmap='magma', interpolation='nearest', origin='upper')
        self.ax_map.set_title("Select Dendrites\n'c': clear | 'q': quit")
        
        self.ax_trace.set_facecolor('#0f0f0f') 
        self.ax_trace.set_title("Stacked $\Delta F/F$ Traces (Waterfall Plot)")
        
        self._init_selector()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        plt.tight_layout()
        plt.show()

    def _init_selector(self):
        props = dict(color='white', linestyle='-', linewidth=2, alpha=1)
        self.poly = PolygonSelector(self.ax_map, self.onselect, props=props)

    def onselect(self, verts):
        if len(verts) > 2:
            self.add_roi(verts)
            self.poly.disconnect_events()
            self._init_selector()

    def calculate_df_f(self, raw_trace):
        # Using 5th percentile for baseline F0
        f0 = np.percentile(raw_trace, 5)
        return (raw_trace - f0) / (f0 + 1e-6)

    def add_roi(self, verts):
        ny, nx = self.sd_map.shape
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        points = np.vstack((x.flatten(), y.flatten())).T
        path = Path(verts)
        mask = path.contains_points(points).reshape((ny, nx))

        color = self.cmap(len(self.rois) % 10)
        poly_patch = Polygon(verts, closed=True, fill=True, edgecolor=color, 
                             linewidth=1.5, facecolor=color, alpha=0.3)
        self.ax_map.add_patch(poly_patch)

        # Extracting data from your stabilized M3 Pro pipeline
        raw_f = np.mean(self.movie[:, mask], axis=1)
        df_f = self.calculate_df_f(raw_f)

        # Calculate spacing relative to trace strength
        # We shift the trace up by the current offset
        shifted_trace = df_f + self.current_offset
        
        self.rois.append(verts)
        self.traces.append(df_f)

        time_axis = np.arange(len(df_f)) / self.fs
        self.ax_trace.plot(time_axis, shifted_trace, color=color, linewidth=1.2)
        
        # Add a label at the start of the trace for clarity
        self.ax_trace.text(-0.5, self.current_offset, f"ROI {len(self.rois)}", 
                           color=color, va='center', ha='right', fontsize=9)

        # Update offset for the NEXT trace based on current trace max
        # This keeps the gap proportional to the response size
        self.current_offset += (np.max(df_f) * self.spacing_factor)

        if self.odor_onsets:
            for onset in self.odor_onsets:
                self.ax_trace.axvline(onset / self.fs, color='red', linestyle=':', alpha=0.2)

        self.ax_trace.set_xlabel("Time (s)")
        self.ax_trace.set_ylabel("Stacked $\Delta F/F$ (Relative units)")
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'c':
            self.ax_trace.clear()
            self.ax_trace.set_facecolor('#0f0f0f')
            self.current_offset = 0
            self.fig.canvas.draw()
        elif event.key == 'q':
            plt.close(self.fig)

# --- Run ---
my_onsets = [100]  # Example odor onsets in frames (adjust as needed)

processor = StackedROIProcessor(corrected_movie_cropped, sd_map, fs=frame_rate/bin_factor, odor_onsets=my_onsets)


# %%
