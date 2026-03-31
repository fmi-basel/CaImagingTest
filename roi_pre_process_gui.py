"""Two-photon ROI pre-processing GUI.

Replaces the for-loop workflow in roi_pre_process.py with a PyQt5 application
that lets the user browse folders, select series from a list, run the interactive
ROI selection (with per-ROI naming and undo), analyse the data, and then move on
to the next series without restarting.

Usage
-----
    python roi_pre_process_gui.py
    # or from an IPython session:
    %run roi_pre_process_gui.py
"""

import sys
import os
import glob
import io
import contextlib
import traceback
import numpy as np
import pickle

import matplotlib
try:
    matplotlib.use('Qt5Agg')
except Exception:
    pass  # backend may already be set (e.g. running inside IPython)
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton,
    QListWidget, QListWidgetItem,
    QTextEdit, QFileDialog,
    QComboBox, QGroupBox,
    QMessageBox, QSplitter,
    QCheckBox, QDialog, QScrollArea,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor, QPixmap

from utilities import map_stimulus_ids_from_osf, temporal_denoise
from roi_processor import run_roi_selection
from helpers_figures import (
    plot_mean_std_projection,
    plot_roi_masks_and_traces,
    plot_trial_averaged_roi_responses,
    plot_trial_overlaid_roi_responses,
)
from roi_extractor_params import get_auto_roi_params, AUTO_ROI_PROFILES
from batch_utilities import load_series_metadata


# ── stdout/stderr redirector ────────────────────────────────────────────────

class _LogWriter(io.TextIOBase):
    """Redirect print() calls to the GUI log widget."""

    def __init__(self, log_fn):
        self._log_fn = log_fn

    def write(self, text):
        if text and text.strip():
            self._log_fn(text.rstrip('\n'))
        return len(text)

    def flush(self):
        pass


# ── Main window ─────────────────────────────────────────────────────────────

class ROIProcessingGUI(QMainWindow):
    """Main application window for the ROI pre-processing pipeline."""

    # Default odour colours – edit here or expose in a future settings panel.
    DEFAULT_COLORS = {'MCH': '#e41a1c', 'OCTT': '#ffff99', 'IAA': '#4daf4a'}

    def __init__(self):
        super().__init__()
        self.setWindowTitle('ROI Processing')
        self.resize(1150, 860)
        self._series_paths = []
        self._series_status = {}   # path -> 'pending' | 'processing' | 'done' | 'error'
        self._setup_ui()

    # ── UI construction ──────────────────────────────────────────────────────

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(8)
        root.setContentsMargins(10, 10, 10, 10)

        # ── Top row: Configuration (left) | Series (right) ───────────────────
        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        # ── Configuration ────────────────────────────────────────────────────
        cfg_group = QGroupBox('Configuration')
        cfg = QGridLayout()
        cfg.setColumnStretch(1, 1)

        cfg.addWidget(QLabel('Base Directory:'), 0, 0)
        self.base_dir_edit = QLineEdit()
        self.base_dir_edit.setPlaceholderText('/path/to/data  (or D:\\data on Windows)')
        cfg.addWidget(self.base_dir_edit, 0, 1)
        browse_btn = QPushButton('Browse…')
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse_base_dir)
        cfg.addWidget(browse_btn, 0, 2)

        cfg.addWidget(QLabel('Container ID:'), 1, 0)
        self.container_edit = QLineEdit()
        self.container_edit.setPlaceholderText('e.g. 2025_10_Gamma1_CC_extinction')
        self.container_edit.editingFinished.connect(self._scan_days)
        cfg.addWidget(self.container_edit, 1, 1)
        scan_btn = QPushButton('Scan')
        scan_btn.setFixedWidth(55)
        scan_btn.setToolTip('Scan for available day directories')
        scan_btn.clicked.connect(self._scan_days)
        cfg.addWidget(scan_btn, 1, 2)

        cfg.addWidget(QLabel('Day ID:'), 2, 0)
        self.day_combo = QComboBox()
        self.day_combo.setEditable(True)
        self.day_combo.lineEdit().setPlaceholderText('e.g. 2025_11_13')
        cfg.addWidget(self.day_combo, 2, 1, 1, 2)

        self.auto_roi_check = QCheckBox('Auto ROI Selection')
        self.auto_roi_check.toggled.connect(self._on_auto_roi_toggled)
        cfg.addWidget(self.auto_roi_check, 3, 0, 1, 3)

        cfg.addWidget(QLabel('Auto ROI Profile:'), 4, 0)
        self.auto_profile_combo = QComboBox()
        self.auto_profile_combo.addItems(sorted(AUTO_ROI_PROFILES.keys()))
        self.auto_profile_combo.setEnabled(False)
        cfg.addWidget(self.auto_profile_combo, 4, 1, 1, 2)

        cfg_group.setLayout(cfg)
        top_row.addWidget(cfg_group, stretch=1)

        # ── Series list ──────────────────────────────────────────────────────
        series_group = QGroupBox('Series')
        series_layout = QVBoxLayout()

        load_btn = QPushButton('Load Series')
        load_btn.clicked.connect(self._load_series)
        series_layout.addWidget(load_btn)

        self.series_list = QListWidget()
        self.series_list.setSelectionMode(QListWidget.SingleSelection)
        self.series_list.itemSelectionChanged.connect(self._on_selection_changed)
        series_layout.addWidget(self.series_list)

        btn_row = QHBoxLayout()
        self.process_btn = QPushButton('Process Selected Series')
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self._process_selected)
        btn_row.addWidget(self.process_btn)

        self.process_all_btn = QPushButton('Process All Pending')
        self.process_all_btn.setEnabled(False)
        self.process_all_btn.clicked.connect(self._process_all_pending)
        btn_row.addWidget(self.process_all_btn)

        series_layout.addLayout(btn_row)
        series_group.setLayout(series_layout)
        top_row.addWidget(series_group, stretch=1)

        root.addLayout(top_row)

        # ── Log ──────────────────────────────────────────────────────────────
        log_group = QGroupBox('Log')
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont('Courier New', 9))
        clear_log_btn = QPushButton('Clear Log')
        clear_log_btn.setFixedWidth(90)
        clear_log_btn.clicked.connect(self.log_text.clear)
        log_top = QHBoxLayout()
        log_top.addStretch()
        log_top.addWidget(clear_log_btn)
        log_layout.addLayout(log_top)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        root.addWidget(log_group, stretch=1)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _log(self, text):
        self.log_text.append(str(text))
        QApplication.processEvents()

    def _browse_base_dir(self):
        directory = QFileDialog.getExistingDirectory(self, 'Select Base Directory')
        if directory:
            self.base_dir_edit.setText(directory)
            self._scan_days()

    def _scan_days(self):
        """Populate the Day ID combo with subdirectories found inside container_dir."""
        base_dir = self.base_dir_edit.text().strip()
        container_id = self.container_edit.text().strip()
        if not base_dir or not container_id:
            return
        container_dir = os.path.join(base_dir, container_id)
        if not os.path.isdir(container_dir):
            return
        days = sorted(
            d for d in os.listdir(container_dir)
            if os.path.isdir(os.path.join(container_dir, d)) and not d.startswith('.')
        )
        current_text = self.day_combo.currentText()
        self.day_combo.blockSignals(True)
        self.day_combo.clear()
        self.day_combo.addItems(days)
        idx = self.day_combo.findText(current_text)
        if idx >= 0:
            self.day_combo.setCurrentIndex(idx)
        elif current_text:
            self.day_combo.setEditText(current_text)
        self.day_combo.blockSignals(False)
        self._log(f'Found {len(days)} day directories in {container_dir}')

    def _on_auto_roi_toggled(self, checked):
        self.auto_profile_combo.setEnabled(checked)

    def _show_image_dialog(self, png_path, title='Plot'):
        """Open a modal dialog displaying a saved PNG file."""
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(960, 640)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(4, 4, 4, 4)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        label = QLabel()
        pixmap = QPixmap(png_path)
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)
        scroll.setWidget(label)
        layout.addWidget(scroll)
        close_btn = QPushButton('Close')
        close_btn.setFixedWidth(80)
        close_btn.clicked.connect(dialog.accept)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)
        dialog.exec_()

    def _on_selection_changed(self):
        self.process_btn.setEnabled(bool(self.series_list.selectedItems()))

    def _set_series_status(self, path, status):
        """Update list item text and colour for a given series path."""
        self._series_status[path] = status
        icons = {'pending': '[ ]', 'processing': '[...]', 'done': '[done]', 'error': '[!]'}
        colors = {
            'pending': '#aaaaaa',
            'processing': '#f0a500',
            'done': '#27ae60',
            'error': '#e74c3c',
        }
        for i in range(self.series_list.count()):
            item = self.series_list.item(i)
            if item.data(Qt.UserRole) == path:
                label = os.path.basename(path)
                item.setText(f"{icons.get(status, '')}  {label}")
                item.setForeground(QColor(colors.get(status, '#aaaaaa')))
                break

    # ── series loading ────────────────────────────────────────────────────────

    def _load_series(self):
        base_dir = self.base_dir_edit.text().strip()
        container_id = self.container_edit.text().strip()
        day_id = self.day_combo.currentText().strip()

        if not all([base_dir, container_id, day_id]):
            QMessageBox.warning(self, 'Missing Fields',
                                'Please fill in Base Directory, Container ID and Day ID.')
            return

        day_dir = os.path.join(base_dir, container_id, day_id)  # day_id from combo
        if not os.path.isdir(day_dir):
            QMessageBox.warning(self, 'Directory Not Found',
                                f'The following directory does not exist:\n{day_dir}')
            return

        series_paths = sorted(glob.glob(os.path.join(day_dir, 'S1-T*')))
        if not series_paths:
            QMessageBox.warning(self, 'No Series Found',
                                f'No S1-T* folders found in:\n{day_dir}')
            return

        self._series_paths = series_paths
        self._series_status = {}
        self.series_list.clear()

        for path in series_paths:
            item = QListWidgetItem()
            item.setData(Qt.UserRole, path)
            self.series_list.addItem(item)
            self._set_series_status(path, 'pending')

        self.process_all_btn.setEnabled(True)
        self._log(f'Loaded {len(series_paths)} series from:\n  {day_dir}')

    # ── processing ───────────────────────────────────────────────────────────

    def _process_selected(self):
        selected = self.series_list.selectedItems()
        if selected:
            self._run_pipeline(selected[0].data(Qt.UserRole))

    def _process_all_pending(self):
        for path in self._series_paths:
            if self._series_status.get(path) == 'pending':
                self._run_pipeline(path)

    def _run_pipeline(self, series_path):
        series_id = os.path.basename(series_path)
        self._set_series_status(series_path, 'processing')
        self.process_btn.setEnabled(False)
        self.process_all_btn.setEnabled(False)
        QApplication.processEvents()

        try:
            log_writer = _LogWriter(self._log)
            with contextlib.redirect_stdout(log_writer):
                with contextlib.redirect_stderr(log_writer):
                    self._analyze_series(series_path)
            self._set_series_status(series_path, 'done')
            self._log(f'Series {series_id} complete.')
        except Exception:
            self._set_series_status(series_path, 'error')
            self._log(f'ERROR in {series_id}:\n{traceback.format_exc()}')
        finally:
            has_sel = bool(self.series_list.selectedItems())
            self.process_btn.setEnabled(has_sel)
            self.process_all_btn.setEnabled(True)

    # ── core analysis (mirrors roi_pre_process.py per-series body) ───────────

    def _analyze_series(self, series_path):
        base_dir = self.base_dir_edit.text().strip()
        container_id = self.container_edit.text().strip()
        day_id = self.day_combo.currentText().strip()

        experiment_dir = os.path.join(base_dir, container_id)
        day_dir = os.path.join(experiment_dir, day_id)
        series_id = os.path.basename(series_path)

        print(f"\n{'='*60}")
        print(f"ROI processing: {series_id}")
        print(f"{'='*60}")

        series_dir = os.path.join(day_dir, series_id)
        results_dir = os.path.join(series_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        experiment_id = f"{day_id}_{series_id}"

        db_path = os.path.join(experiment_dir, f'{container_id}_database.csv')
        series_meta, vial_to_odor = load_series_metadata(db_path, series_id, experiment_id)

        # ── Load motion-corrected movie ──────────────────────────────────────
        suite2p_dir = os.path.join(series_dir, 'suite2p_corrected')
        processed_movie = np.load(
            os.path.join(suite2p_dir, f'{series_id}_corrected.npy'), mmap_mode='r'
        )
        ops = np.load(
            os.path.join(suite2p_dir, 'motion_input_ops.npy'), allow_pickle=True
        ).item()
        downsampled_fr = ops['fs']
        processed_movie_cropped = processed_movie

        # ── Map stimulus IDs ─────────────────────────────────────────────────
        stim_trace = np.load(os.path.join(series_dir, f'{series_id}_stim_trace.npy'))
        osf_path = os.path.join(series_dir, f'{series_id}.osf')
        stimulus_id_trace, stimulus_sequence, stim_starts, stim_ends = map_stimulus_ids_from_osf(
            stim_trace, osf_path,
        )
        print(f"Stimulus sequence from OSF: {stimulus_sequence}")
        print(f"Detected {len(stim_starts)} stimulus periods")
        mapped_ids = np.unique(stimulus_id_trace[stimulus_id_trace > 0]).astype(int)
        mapped_stimuli = [vial_to_odor.get(i, f"V{i}") for i in mapped_ids]
        print(f"Mapped stimuli: {mapped_stimuli}")

        extraction_image = processed_movie_cropped.mean(axis=0)

        # ── Mean/std projection figure ───────────────────────────────────────
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
        plt.close('all')

        # ── ROI selection (interactive) ──────────────────────────────────────
        if self.auto_roi_check.isChecked():
            auto_roi_profile = self.auto_profile_combo.currentText()
            roi_selection_mode = 'custom-automatic'
        else:
            auto_roi_profile = None
            roi_selection_mode = 'manual'

        auto_roi_params = get_auto_roi_params(auto_roi_profile)
        print("Opening ROI selection window…")
        print("  Controls: 'b' bg | 'z' undo | 'r' rename last | 'c' clear all | 'q' finish")

        roi_result = run_roi_selection(
            mode=roi_selection_mode,
            movie=processed_movie_cropped,
            extraction_image=extraction_image,
            fs=downsampled_fr,
            results_dir=results_dir,
            series_id=series_id,
            stimulus_id_trace=stimulus_id_trace,
            auto_roi_params=auto_roi_params,
            df_f_method='1-11s',
            gui_mode=True,
        )

        roi_masks = roi_result['roi_masks']
        roi_names = roi_result['roi_names']
        raw_traces = roi_result['raw_traces']
        bg_subtracted_df_traces = roi_result['bg_subtracted_df_traces']
        background_mask = roi_result['background_mask']
        background_polygon = roi_result['background_polygon']
        background_raw_trace = roi_result['background_raw_trace']

        print(f"ROI names: {roi_names}")
        for roi_name, trace in zip(roi_names, bg_subtracted_df_traces):
            print(f"  {roi_name}: shape {trace.shape}")

        # ── ROI masks + traces figure ────────────────────────────────────────
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
            colors_hex=self.DEFAULT_COLORS,
            save_path=os.path.join(results_dir, f'{series_id}_roi_analysis.pdf'),
            figsize=(10, 12),
            dpi=300,
        )
        plt.close('all')

        # ── Build stimulus segments ──────────────────────────────────────────
        stim_ids = np.asarray(stimulus_id_trace).astype(int)
        if stim_ids.size == 0:
            stim_segments = []
        else:
            change_points = np.where(np.diff(stim_ids) != 0)[0] + 1
            seg_starts = np.concatenate(([0], change_points))
            seg_ends = np.concatenate((change_points, [stim_ids.size]))
            stim_segments = [
                (int(s), int(e), int(stim_ids[s]))
                for s, e in zip(seg_starts, seg_ends)
                if int(stim_ids[s]) > 0
            ]

        stimulus_ids_unique = sorted({sid for _, _, sid in stim_segments})
        stim_ids_full_trace = np.asarray(stimulus_id_trace).astype(int)
        context_window_s = 5.0
        context_window_frames = int(round(context_window_s * downsampled_fr))

        # ── Build nested ROI data dict ───────────────────────────────────────
        roi_data_nested = {}

        for roi_idx, roi_name in enumerate(roi_names):
            raw_trace = np.asarray(raw_traces[roi_idx])
            bg_sub_trace = np.asarray(bg_subtracted_df_traces[roi_idx])
            roi_mask = np.asarray(roi_masks[roi_idx]).astype(bool)

            repeats_by_stimulus = {}
            repeats_by_stimulus_with_context = {}

            for stim_id in stimulus_ids_unique:
                stim_name = vial_to_odor.get(stim_id, f"V{stim_id}")
                trial_segs = [(s, e) for s, e, sid in stim_segments if sid == stim_id]

                trial_traces_only_stim = [
                    bg_sub_trace[s:e] for s, e in trial_segs if e > s
                ]

                trial_traces_with_context = []
                trial_stim_id_traces_with_context = []
                stim_start_indices_in_window = []
                stim_end_indices_in_window = []

                for start_idx, end_idx in trial_segs:
                    if end_idx <= start_idx:
                        continue
                    w_start = max(0, start_idx - context_window_frames)
                    w_end = min(bg_sub_trace.shape[0], end_idx + context_window_frames)
                    trial_traces_with_context.append(bg_sub_trace[w_start:w_end])
                    trial_stim_id_traces_with_context.append(stim_ids_full_trace[w_start:w_end])
                    stim_start_indices_in_window.append(int(start_idx - w_start))
                    stim_end_indices_in_window.append(int(end_idx - w_start))

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
                padded = np.full((len(trial_traces_only_stim), trial_max_len), np.nan, dtype=float)
                for k, t in enumerate(trial_traces_only_stim):
                    padded[k, :len(t)] = t
                avg_stim = np.nanmean(padded, axis=0)

                repeats_by_stimulus[stim_name] = {
                    'stimulus_id': int(stim_id),
                    'stimulus_name': stim_name,
                    'trial_traces': trial_traces_only_stim,
                    'trial_average_trace': avg_stim,
                }

                ctx_max = max(len(t) for t in trial_traces_with_context)
                padded_ctx = np.full((len(trial_traces_with_context), ctx_max), np.nan, dtype=float)
                padded_sid = np.full((len(trial_stim_id_traces_with_context), ctx_max), np.nan, dtype=float)
                for k, t in enumerate(trial_traces_with_context):
                    padded_ctx[k, :len(t)] = t
                for k, t in enumerate(trial_stim_id_traces_with_context):
                    padded_sid[k, :len(t)] = t

                repeats_by_stimulus_with_context[stim_name] = {
                    'stimulus_id': int(stim_id),
                    'stimulus_name': stim_name,
                    'context_window_s': float(context_window_s),
                    'trial_traces': trial_traces_with_context,
                    'trial_stimulus_id_traces': trial_stim_id_traces_with_context,
                    'stim_start_indices_in_window': stim_start_indices_in_window,
                    'stim_end_indices_in_window': stim_end_indices_in_window,
                    'trial_average_trace': np.nanmean(padded_ctx, axis=0),
                    'trial_average_stimulus_id_trace': np.nanmax(padded_sid, axis=0),
                }

            trial_averaged_traces = {
                sn: sd['trial_average_trace']
                for sn, sd in repeats_by_stimulus.items()
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
                'trial_averaged_traces_by_stimulus': trial_averaged_traces,
                'mean_image': extraction_image,
            }

        # ── Save dataset ─────────────────────────────────────────────────────
        series_meta['stim_info'] = {
            'aurora_vial_info': vial_to_odor,
            'stim_id_trace': stimulus_id_trace,
        }
        series_meta['mean_image'] = extraction_image
        series_meta['background_info'] = {
            'mask': background_mask,
            'polygon': background_polygon,
            'raw_trace': background_raw_trace,
        }

        session_results = {'metadata': series_meta, 'rois': {}}
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

        p_data_dir = os.path.join(experiment_dir, f'{container_id}_processed_data')
        os.makedirs(p_data_dir, exist_ok=True)
        save_file = os.path.join(p_data_dir, f'{series_id}_processed_data.pkl')
        with open(save_file, 'wb') as f:
            pickle.dump(session_results, f)
        print(f"Dataset saved to: {save_file}")

        # ── Trial-averaged and trial-overlaid plots ──────────────────────────
        roi_trial_plot_dir = os.path.join(results_dir, 'roi_trial_average_plots')
        os.makedirs(roi_trial_plot_dir, exist_ok=True)

        pre_window_frames = int(round(5.0 * downsampled_fr))
        post_window_frames = int(round(15.0 * downsampled_fr))

        stim_durations_by_id = {
            stim_id: [int(e - s) for s, e, sid in stim_segments if sid == stim_id]
            for stim_id in stimulus_ids_unique
        }
        all_durations = [d for dlist in stim_durations_by_id.values() for d in dlist]
        bg_smooth = np.array(
            [temporal_denoise(trace, window_size=5) for trace in bg_subtracted_df_traces]
        )

        if all_durations and roi_names:
            overlaid_plot_path = plot_trial_overlaid_roi_responses(
                roi_names=roi_names,
                roi_masks=roi_masks,
                bg_subtracted_df_traces=bg_smooth,
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
                colors_hex=self.DEFAULT_COLORS,
                font_size=6, cell_w=3.5, cell_h=2.6,
            )
            if overlaid_plot_path:
                self._show_image_dialog(overlaid_plot_path, f'{series_id} — Single-Trial Responses')

            avg_plot_path = plot_trial_averaged_roi_responses(
                roi_names=roi_names,
                roi_masks=roi_masks,
                bg_subtracted_df_traces=bg_smooth,
                extraction_image=extraction_image,
                stim_segments=stim_segments,
                stimulus_ids_unique=stimulus_ids_unique,
                vial_to_odor=vial_to_odor,
                colors_hex=self.DEFAULT_COLORS,
                stim_durations_by_id=stim_durations_by_id,
                downsampled_fr=downsampled_fr,
                series_id=series_id,
                series_meta=series_meta,
                roi_trial_plot_dir=roi_trial_plot_dir,
                pre_window_frames=pre_window_frames,
                post_window_frames=post_window_frames,
                max_cols=5, cell_w=9, cell_h=3, font_size=6,
            )
            if avg_plot_path:
                self._show_image_dialog(avg_plot_path, f'{series_id} — Trial-Averaged Responses')

            

        print(f"Saved ROI trial plots to: {roi_trial_plot_dir}")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    # Enable high-DPI scaling on Windows before creating QApplication
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication.instance() or QApplication(sys.argv)
    window = ROIProcessingGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
