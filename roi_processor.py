import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
from matplotlib.patches import Polygon
from matplotlib import cm
from automatic_roi import (
    estimate_bouton_diameter_px,
    centers_to_watershed_masks,
    plot_watershed_rois,
    optimize_roi_parameters,
    detect_centers_from_image,
    select_polygon_mask,
    filter_centers_by_mask,
)


class ROISelector:
    def __init__(
        self,
        movie,
        sd_map,
        fs=6,
        spacing_factor=1.5,
        stimulus_id_trace=None,
        roi_names=None,
    ):
        self.movie = movie
        self.sd_map = sd_map
        self.fs = fs
        self.spacing_factor = spacing_factor
        self.stimulus_id_trace = stimulus_id_trace
        self._custom_roi_names = list(roi_names) if roi_names is not None else None
        self._stim_drawn = False
        self.rois = []
        self.roi_names = []
        self.roi_masks = []
        self.traces = []
        self.raw_traces = []
        self.roi_patches = []
        self.roi_name_texts = []
        self.roi_colors = []
        self.current_offset = 0
        self.background_idx = None

        self.cmap = cm.get_cmap('tab10')
        self.stim_cmap = cm.get_cmap('Set3')  # Color map for stimulus vials

        self.fig, (self.ax_map, self.ax_trace) = plt.subplots(1, 2, figsize=(16, 8))
        self.ax_map.imshow(sd_map, cmap='magma', interpolation='nearest', origin='upper')
        self.ax_map.set_title("Select ROIs\n'b': bg | 'z': undo | 'r': rename last | 'c': clear all | 'q': finish")

        self.ax_trace.set_facecolor('white')
        self.ax_trace.set_title('Stacked Delta F/F Traces (Waterfall Plot)')

        self._next_is_background = False
        self._init_selector()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        plt.tight_layout()
        # Avoid entering a nested Qt event loop when already running inside a Qt GUI
        # (important on Windows where a second exec_() call deadlocks).
        try:
            from PyQt5.QtWidgets import QApplication as _QApp
            if _QApp.instance() is not None:
                plt.show(block=False)
                return
        except ImportError:
            pass
        plt.show()

    def _init_selector(self):
        props = dict(color='white', linestyle='-', linewidth=2, alpha=1)
        self.poly = PolygonSelector(self.ax_map, self.onselect, props=props)

    def onselect(self, verts):
        if len(verts) > 2:
            is_bg = self._next_is_background
            self.add_roi(verts, is_background=is_bg)
            self._next_is_background = False
            self.poly.disconnect_events()
            self._init_selector()
            if not is_bg:
                self._prompt_and_rename_last()

    def calculate_df_f(self, raw_trace):
        # Calculate baseline (F0) as mean signal over first 10 seconds
        baseline_samples = int(self.fs * 10)  # 10 seconds of samples
        baseline_samples = min(baseline_samples, len(raw_trace))  # Cap at trace length
        f0 = np.mean(raw_trace[:baseline_samples])
        return (raw_trace - f0) / (f0 + 1e-6)

    def add_roi(self, verts, is_background=False):
        ny, nx = self.sd_map.shape
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        points = np.vstack((x.flatten(), y.flatten())).T
        path = Path(verts)
        mask = path.contains_points(points).reshape((ny, nx))

        # Auto-generate ROI name - count non-background ROIs separately
        if is_background:
            bg_count = sum(1 for name in self.roi_names if name.startswith('BG'))
            roi_name = f"BG{bg_count + 1}"
        else:
            non_bg_count = sum(1 for name in self.roi_names if not name.startswith('BG'))
            if self._custom_roi_names is not None and non_bg_count < len(self._custom_roi_names):
                roi_name = self._custom_roi_names[non_bg_count]
            else:
                roi_count = sum(1 for name in self.roi_names if name.startswith('ROI'))
                roi_name = f"ROI{roi_count + 1}"

        color = self.cmap(len(self.rois) % 10)
        self.roi_colors.append(color)
        poly_patch = Polygon(
            verts,
            closed=True,
            fill=True,
            edgecolor='white',
            linewidth=2.5,
            facecolor=color,
            alpha=0.4,
        )
        self.ax_map.add_patch(poly_patch)
        self.roi_patches.append(poly_patch)

        raw_f = np.mean(self.movie[:, mask], axis=1)
        df_f = self.calculate_df_f(raw_f)

        shifted_trace = df_f + self.current_offset

        self.rois.append(verts)
        self.roi_names.append(roi_name)
        self.roi_masks.append(mask)
        self.traces.append(df_f)
        self.raw_traces.append(raw_f)
        
        if is_background:
            self.background_idx = len(self.rois) - 1

        time_axis = np.arange(len(df_f)) / self.fs
        self.ax_trace.plot(time_axis, shifted_trace, color=color, linewidth=1.2)
        text_obj = self.ax_trace.text(
            -0.5,
            self.current_offset,
            roi_name,
            color=color,
            va='center',
            ha='right',
            fontsize=9,
        )
        self.roi_name_texts.append(text_obj)

        self.current_offset += (np.max(df_f) * self.spacing_factor)

        if not self._stim_drawn and self.stimulus_id_trace is not None:
            self._draw_stimulus_periods()
            self._stim_drawn = True

        self.ax_trace.set_xlabel('Time (s)')
        self.ax_trace.set_ylabel('Stacked Delta F/F (Relative units)')
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'b':
            print("Next ROI will be marked as BACKGROUND")
            self._next_is_background = True
        elif event.key == 'z':
            self._undo_last_roi()
        elif event.key == 'r':
            self._prompt_and_rename_last()
        elif event.key == 'c':
            for patch in self.roi_patches:
                patch.remove()
            self.roi_patches = []
            self.roi_name_texts = []
            self.roi_colors = []
            self.ax_trace.clear()
            self.ax_trace.set_facecolor('white')
            self.current_offset = 0
            self.rois = []
            self.roi_names = []
            self.roi_masks = []
            self.traces = []
            self.raw_traces = []
            self.background_idx = None
            self._stim_drawn = False
            self.fig.canvas.draw()
        elif event.key == 'q':
            plt.close(self.fig)

    def _prompt_roi_name(self, default_name):
        """Show a Qt input dialog to let the user name the ROI."""
        try:
            from PyQt5.QtWidgets import QApplication, QInputDialog
            if QApplication.instance() is not None:
                name, ok = QInputDialog.getText(
                    None, 'Name ROI', 'Enter a name for this ROI:', text=default_name
                )
                if ok and name.strip():
                    return name.strip()
        except Exception:
            pass
        return default_name

    def _prompt_and_rename_last(self):
        """Prompt the user to rename the most recently drawn ROI."""
        if not self.roi_names:
            return
        current_name = self.roi_names[-1]
        new_name = self._prompt_roi_name(current_name)
        if new_name != current_name:
            self.roi_names[-1] = new_name
            if self.roi_name_texts:
                self.roi_name_texts[-1].set_text(new_name)
            self.fig.canvas.draw()

    def _undo_last_roi(self):
        """Remove the most recently drawn ROI."""
        if not self.rois:
            print('No ROIs to undo.')
            return
        last_idx = len(self.rois) - 1
        was_background = (self.background_idx == last_idx)
        removed_name = self.roi_names.pop()
        self.rois.pop()
        self.roi_masks.pop()
        self.traces.pop()
        self.raw_traces.pop()
        self.roi_colors.pop()
        if self.roi_patches:
            self.roi_patches.pop().remove()
        if was_background:
            self.background_idx = None
        elif self.background_idx is not None and self.background_idx >= len(self.rois):
            self.background_idx = None
        self._redraw_traces()
        print(f'Removed ROI: {removed_name}')
        self.fig.canvas.draw()

    def _redraw_traces(self):
        """Clear and redraw the trace panel from the current ROI list."""
        self.ax_trace.clear()
        self.ax_trace.set_facecolor('white')
        self.roi_name_texts = []
        self.current_offset = 0
        self._stim_drawn = False
        for name, trace, color in zip(self.roi_names, self.traces, self.roi_colors):
            shifted_trace = trace + self.current_offset
            time_axis = np.arange(len(trace)) / self.fs
            self.ax_trace.plot(time_axis, shifted_trace, color=color, linewidth=1.2)
            text_obj = self.ax_trace.text(
                -0.5, self.current_offset, name,
                color=color, va='center', ha='right', fontsize=9,
            )
            self.roi_name_texts.append(text_obj)
            self.current_offset += (np.max(trace) * self.spacing_factor)
        if self.stimulus_id_trace is not None and self.roi_names:
            self._draw_stimulus_periods()
            self._stim_drawn = True
        self.ax_trace.set_xlabel('Time (s)')
        self.ax_trace.set_ylabel('Stacked Delta F/F (Relative units)')
    
    def get_background_subtracted_traces(self):
        """Return traces with background subtraction applied"""
        if self.background_idx is None:
            print("Warning: No background ROI defined. Returning original traces.")
            return self.traces
        
        bg_trace = self.raw_traces[self.background_idx]
        bg_subtracted = []
        
        for i, raw_trace in enumerate(self.raw_traces):
            if i == self.background_idx:
                bg_subtracted.append(self.traces[i])  # Keep background as is
            else:
                subtracted = raw_trace - bg_trace
                bg_subtracted.append(subtracted)  # Return raw background-subtracted trace; can also return df/f if desired
        
        return bg_subtracted
    
    def get_traces_dict(self):
        """Return dictionary of ROI names to traces"""
        return {name: trace for name, trace in zip(self.roi_names, self.traces)}
    
    def get_background_subtracted_dict(self):
        """Return dictionary of ROI names to background-subtracted traces"""
        bg_traces = self.get_background_subtracted_traces()
        return {name: trace for name, trace in zip(self.roi_names, bg_traces)}
    
    def rename_roi(self, old_name, new_name):
        """Rename an ROI"""
        if old_name in self.roi_names:
            idx = self.roi_names.index(old_name)
            self.roi_names[idx] = new_name
            print(f"Renamed '{old_name}' to '{new_name}'")
        else:
            print(f"ROI '{old_name}' not found. Available ROIs: {self.roi_names}")
    
    def rename_roi_by_index(self, idx, new_name):
        """Rename an ROI by its index"""
        if 0 <= idx < len(self.roi_names):
            old_name = self.roi_names[idx]
            self.roi_names[idx] = new_name
            print(f"Renamed ROI {idx} from '{old_name}' to '{new_name}'")
        else:
            print(f"Index {idx} out of range. Valid indices: 0-{len(self.roi_names)-1}")

    def _draw_stimulus_periods(self):
        """Draw color-coded stimulus periods based on vial ID"""
        stim_ids = np.asarray(self.stimulus_id_trace).astype(int)
        if stim_ids.size == 0:
            return

        # Find periods where stimulus ID changes
        padded = np.concatenate(([0], stim_ids, [0]))
        changes = np.diff(padded) != 0
        change_indices = np.where(changes)[0]

        # Draw each stimulus period with vial-specific color
        for i in range(len(change_indices) - 1):
            start_idx = change_indices[i]
            end_idx = change_indices[i + 1]
            vial_id = stim_ids[start_idx]

            if vial_id > 0:  # Only draw if there's an actual stimulus
                start_s = start_idx / self.fs
                end_s = end_idx / self.fs
                
                # Use vial ID to select color
                color = self.stim_cmap(vial_id % 12)
                
                self.ax_trace.axvspan(
                    start_s,
                    end_s,
                    color=color,
                    alpha=0.5,
                    zorder=0,
                )
                
                # Add vial label
                mid_s = (start_s + end_s) / 2
                self.ax_trace.text(
                    mid_s,
                    self.ax_trace.get_ylim()[1] * 0.95,
                    f'V{vial_id}',
                    ha='center',
                    va='top',
                    fontsize=8,
                    color='black',
                    weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7, edgecolor='none')
                )


def calculate_df_f(trace, fs, method='1-11s'):
    if method == '1-11s': # 1-11s to discard initial fluctuations
        baseline_frame_start = int(fs * 1)
        baseline_frame_end = int(fs * 11)
    else:
        raise ValueError(f"Unknown baseline method for df/f calculation: {method}")
    f0 = np.mean(trace[baseline_frame_start:baseline_frame_end])
    return (trace - f0) / (f0 + 1e-6)


def run_roi_selection(
    mode, 
    movie,
    extraction_image,
    fs, #sampling rate in Hz
    results_dir=None,
    series_id='series',
    stimulus_id_trace=None,
    auto_roi_params=None,
    df_f_method='1-11s',
    roi_names=None,
    gui_mode=False,
):
    mode = str(mode).lower().strip()

    if mode == 'manual':
        selector = ROISelector(
            movie=movie,
            sd_map=extraction_image,
            fs=fs,
            stimulus_id_trace=stimulus_id_trace,
            roi_names=roi_names,
        )

        # Block until the ROI window is closed.
        if gui_mode:
            # When running inside a Qt GUI, avoid re-entering QApplication.exec_().
            import time
            from PyQt5.QtWidgets import QApplication
            plt.show(block=False)
            while plt.fignum_exists(selector.fig.number):
                QApplication.processEvents()
                time.sleep(0.02)
        else:
            # Ensure manual selection blocks until the ROI window is closed.
            # In some backends, ROISelector's internal show() can be non-blocking.
            try:
                plt.show(block=True)
            except TypeError:
                plt.show()

            # Wait for user interaction to finish (close window with 'q')
            while plt.fignum_exists(selector.fig.number):
                plt.pause(0.05)

        # Close the figure after selection is done
        plt.close(selector.fig)
        
        roi_masks = list(selector.roi_masks)
        raw_traces = list(selector.raw_traces)
        bg_subtracted = selector.get_background_subtracted_traces()
        bg_subtracted_df_traces = [calculate_df_f(trace, fs, method=df_f_method) for trace in bg_subtracted]
        roi_names = list(selector.roi_names) if len(selector.roi_names) > 0 else [f"ROI{i+1}" for i in range(len(roi_masks))]
        bg_idx = selector.background_idx

        if bg_idx is not None and 0 <= bg_idx < len(roi_masks):
            background_mask = roi_masks[bg_idx]
            background_polygon = np.asarray(selector.rois[bg_idx], dtype=float)
            # pop the background ROI from the lists to keep only actual ROIs
            roi_masks.pop(bg_idx)
            roi_names.pop(bg_idx)
            bg_raw_trace = raw_traces[bg_idx]
            raw_traces.pop(bg_idx)
            bg_subtracted_df_traces.pop(bg_idx)
        else:
            background_mask = np.zeros_like(extraction_image, dtype=bool)
            background_polygon = np.empty((0, 2), dtype=float)
            # throw error if no background is selected, since manual mode requires it
            if bg_idx is None:
                raise ValueError("No background ROI selected. Please select a background ROI by pressing 'b' before drawing the ROI.")

        return {
            'roi_masks': roi_masks,
            'roi_names': roi_names,
            'raw_traces': raw_traces,
            'bg_subtracted_df_traces': bg_subtracted_df_traces,
            'bg_idx': bg_idx,
            'background_mask': background_mask,
            'background_polygon': background_polygon,
            'background_raw_trace': bg_raw_trace if bg_idx is not None else None,
        }

    elif mode == 'custom-automatic':
   

        params = {
            'n_samples': 3,
            'footprint_size': 11,
            'gaussian_sigma': 1.0,
            'threshold_percentile': 98,
            'min_distance_factor': 0.5,
            'figsize': (20, 10),
            'watershed_threshold_percentile': 98,
            'compactness': 0.01,
            'min_area_factor': 0.5,
            'max_area_factor': 5.0,
            'relative_peak_fraction': 0.9,
        }
        if auto_roi_params is not None:
            params.update(auto_roi_params)

        common_figsize = tuple(params.get('figsize', (20, 10)))
        inclusion_figsize = tuple(params.get('inclusion_figsize', common_figsize))
        center_preview_figsize = tuple(params.get('center_preview_figsize', common_figsize))
        background_figsize = tuple(params.get('background_figsize', common_figsize))

        bouton_diameter_px = estimate_bouton_diameter_px(extraction_image, n_samples=params['n_samples'])
        bouton_radius_px = max(1, int(round(bouton_diameter_px / 2)))

        footprint_size = params['footprint_size']
        gaussian_sigma = params['gaussian_sigma']
        threshold_percentile = params['threshold_percentile']
        min_distance_factor = params['min_distance_factor']

        coordinates, clean_sd_smoothed, threshold_abs, min_distance_px = detect_centers_from_image(
            extraction_image,
            bouton_diameter_px=bouton_diameter_px,
            footprint_size=footprint_size,
            gaussian_sigma=gaussian_sigma,
            min_distance_factor=min_distance_factor,
            threshold_percentile=threshold_percentile,
        )

        all_coordinates = coordinates.copy()
        selection_mask, selection_polygon = select_polygon_mask(
            extraction_image,
            title="Define ROI inclusion region",
            figsize=inclusion_figsize,
        )
        if selection_mask.any():
            coordinates = filter_centers_by_mask(all_coordinates, selection_mask)
            print(f"Mask selected: kept {len(coordinates)} / {len(all_coordinates)} detected centers")
        else:
            coordinates = all_coordinates
            print("No valid mask drawn; using all detected centers")

        print(
            "Auto-ROI parameters | "
            f"diameter={bouton_diameter_px:.2f}px, radius={bouton_radius_px}px, "
            f"footprint={footprint_size}px, sigma={gaussian_sigma:.2f}, "
            f"min_distance={min_distance_px}px, threshold={threshold_abs:.2f}"
        )
        print(f"Detected {len(coordinates)} potential boutons.")

        fig = plt.figure(figsize=center_preview_figsize)
        plt.imshow(extraction_image)
        plt.autoscale(False)
        if len(coordinates) > 0:
            plt.plot(coordinates[:, 1], coordinates[:, 0], 'r.', markersize=3, alpha=0.6)
        if len(selection_polygon) >= 3:
            closed_polygon = np.vstack([selection_polygon, selection_polygon[0]])
            plt.plot(closed_polygon[:, 0], closed_polygon[:, 1], 'c-', linewidth=1.5, alpha=0.9)
        plt.title(f"Automated Selection: {len(coordinates)} Boutons")
        if results_dir is not None:
            fig.savefig(f"{results_dir}/{series_id}_auto_roi_selection_centers.png", dpi=150, bbox_inches='tight')
        plt.show()

        threshold_percentile = params['watershed_threshold_percentile']
        compactness = params['compactness']
        min_area_factor = params['min_area_factor']
        max_area_factor = params['max_area_factor']
        relative_peak_fraction = params['relative_peak_fraction']
        labels_ws, roi_masks, kept_centers = centers_to_watershed_masks(
            extraction_image,
            coordinates,
            expected_diameter_px=bouton_diameter_px,
            threshold_percentile=threshold_percentile,
            marker_radius_px=max(1, int(round(bouton_radius_px / 3))),
            smooth_sigma=gaussian_sigma,
            compactness=compactness,
            min_area_factor=min_area_factor,
            max_area_factor=max_area_factor,
            relative_peak_fraction=relative_peak_fraction,
        )

        ws_fig = plot_watershed_rois(clean_sd_smoothed, kept_centers, labels_ws)
        if results_dir is not None:
            ws_fig.savefig(f"{results_dir}/{series_id}_watershed_roi_masks.png", dpi=150, bbox_inches='tight')
        print(f"Watershed segmentation completed: {len(roi_masks)} ROIs kept after filtering based on area criteria.")

        roi_names = [f"ROI{i+1}" for i in range(len(roi_masks))]
        raw_traces = [np.mean(movie[:, mask], axis=1) for mask in roi_masks]

        background_mask, background_polygon = select_polygon_mask(
            extraction_image,
            title="Define BACKGROUND ROI for subtraction",
            figsize=background_figsize,
        )

        if background_mask.any():
            background_raw_trace = np.mean(movie[:, background_mask], axis=1)
            bg_subtracted_raw = [trace - background_raw_trace for trace in raw_traces]
            bg_subtracted_df_traces = [calculate_df_f(trace, fs, method=df_f_method) for trace in bg_subtracted_raw]
            print(f"Background ROI selected: area {int(background_mask.sum())} px")
        else:
            bg_subtracted_df_traces = [calculate_df_f(trace, fs, method=df_f_method) for trace in raw_traces]
            background_polygon = np.empty((0, 2), dtype=float)
            print("No valid background ROI selected; using non-subtracted traces")

        return {
            'roi_masks': roi_masks,
            'roi_names': roi_names,
            'raw_traces': raw_traces,
            'bg_subtracted_df_traces': bg_subtracted_df_traces,
            'background_mask': background_mask,
            'background_polygon': background_polygon,
            'background_raw_trace': background_raw_trace if background_mask.any() else None,
    }

    else :
        raise ValueError("`ROI selection mode` not found'")
        return None
    
  