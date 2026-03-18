import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.stats import wilcoxon


def plot_mean_std_projection(
	data,
	save_path=None,
	figsize=(14, 6),
	mean_cmap='gray',
	std_cmap='hot',
	mean_title='Mean Projection',
	std_title='Std Projection',
	show_axes=True,
	dpi=150,
):
	mean_projection = data.mean(axis=0)
	std_projection = data.std(axis=0)

	fig, axes = plt.subplots(1, 2, figsize=figsize)

	im1 = axes[0].imshow(mean_projection, cmap=mean_cmap)
	axes[0].set_title(mean_title)
	plt.colorbar(im1, ax=axes[0])

	im2 = axes[1].imshow(std_projection, cmap=std_cmap)
	axes[1].set_title(std_title)
	plt.colorbar(im2, ax=axes[1])

	if show_axes:
		axes[0].set_xlabel('X (pixels)')
		axes[0].set_ylabel('Y (pixels)')
		axes[1].set_xlabel('X (pixels)')
		axes[1].set_ylabel('Y (pixels)')
	else:
		axes[0].axis('off')
		axes[1].axis('off')

	plt.tight_layout()

	if save_path is not None:
		fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

	plt.show()
	return fig, mean_projection, std_projection


def plot_roi_masks_and_traces(
	plot_image,
	roi_masks,
	roi_names,
	roi_traces,
	downsampled_fr,
	series_id,
	background_mask=None,
	background_polygon=None,
	stimulus_id_trace=None,
	vial_to_odor=None,
	colors_hex=None,
	save_path=None,
	figsize=(10, 12),
	dpi=300,
):
	if background_mask is None:
		background_mask = np.zeros_like(plot_image, dtype=bool)
	if background_polygon is None:
		background_polygon = np.empty((0, 2), dtype=float)
	if vial_to_odor is None:
		vial_to_odor = {}
	if colors_hex is None:
		colors_hex = {}

	fig = plt.figure(figsize=figsize)
	gs = fig.add_gridspec(2, 1, hspace=0.15)
	ax_map = fig.add_subplot(gs[0])
	ax_trace = fig.add_subplot(gs[1])

	ax_map.imshow(plot_image, cmap='gray')
	ax_map.set_title('ROI Masks', fontsize=14)
	ax_map.axis('off')

	roi_cmap = cm.get_cmap('tab10')
	stim_cmap = cm.get_cmap('Set3')

	if background_mask.any():
		bg_overlay = np.zeros((*background_mask.shape, 4))
		bg_overlay[background_mask, :3] = [0.0, 1.0, 1.0]
		bg_overlay[background_mask, 3] = 0.25
		ax_map.imshow(bg_overlay, origin='upper')

		if len(background_polygon) >= 3:
			closed_bg = np.vstack([background_polygon, background_polygon[0]])
			ax_map.plot(closed_bg[:, 0], closed_bg[:, 1], 'c-', linewidth=1, alpha=0.9)

	for i, (name, mask) in enumerate(zip(roi_names, roi_masks)):
		color = roi_cmap(i % 10)
		overlay = np.zeros((*mask.shape, 4))
		overlay[mask, :3] = color[:3]
		overlay[mask, 3] = 0.4
		ax_map.imshow(overlay, origin='upper')

		y_coords, x_coords = np.where(mask)
		if len(y_coords) > 0:
			min_y, min_x = y_coords.min(), x_coords.min()
			ax_map.text(min_x + 5, min_y + 5, i + 1, color='white', ha='left', va='top', fontsize=6, weight='bold')

	spacing_factor = 1.5
	current_offset = 0
	time_axis = np.arange(len(roi_traces[0])) / downsampled_fr if len(roi_traces) > 0 else np.array([])

	for i, (name, trace) in enumerate(zip(roi_names, roi_traces)):
		
		color = roi_cmap(i % 10)
		shifted_trace = trace + current_offset
		ax_trace.plot(time_axis, shifted_trace, color=color, linewidth=1, label=name)
		ax_trace.text(-0.5, current_offset, i + 1, color=color, va='center', ha='right', fontsize=9)
		current_offset += np.max(trace) * spacing_factor

	if stimulus_id_trace is not None:
		stim_ids = np.asarray(stimulus_id_trace).astype(int)
		padded = np.concatenate(([0], stim_ids, [0]))
		changes = np.diff(padded) != 0
		change_indices = np.where(changes)[0]

		for i in range(len(change_indices) - 1):
			start_idx = change_indices[i]
			end_idx = change_indices[i + 1]
			vial_id = stim_ids[start_idx]

			if vial_id > 0:
				start_s = start_idx / downsampled_fr
				end_s = end_idx / downsampled_fr
				odor_name = vial_to_odor.get(int(vial_id), f"V{int(vial_id)}")
				color = colors_hex.get(odor_name, stim_cmap(vial_id % 12))

				ax_trace.axvspan(start_s, end_s, color=color, alpha=0.5, zorder=0)

				mid_s = (start_s + end_s) / 2
				ax_trace.text(
					mid_s,
					ax_trace.get_ylim()[1] * 0.95,
					odor_name,
					ha='center',
					va='top',
					fontsize=8,
					color='black',
					weight='bold',
					bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7, edgecolor='none')
				)

	ax_trace.set_xlabel('Time (s)', fontsize=12)
	ax_trace.set_ylabel('ΔF/F', fontsize=12)
	ax_trace.set_title('ROI traces', fontsize=14)
	ax_trace.set_facecolor('white')

	fig.suptitle(f'{series_id} - ROI Analysis', fontsize=16, y=0.98)

	if save_path is not None:
		fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

	plt.show()
	return fig


def plot_trial_averaged_roi_responses(
	roi_names,
	roi_masks,
	bg_subtracted_df_traces,
	extraction_image,
	stim_segments,
	stimulus_ids_unique,
	vial_to_odor,
	colors_hex,
	stim_durations_by_id,
	downsampled_fr,
	series_id,
	series_meta,
	roi_responsive_df,
	roi_trial_plot_dir,
	pre_window_frames,
	post_window_frames,
	max_cols=5,
	cell_w=8,
	cell_h=3,
	font_size=6,
):
	n_rois = len(roi_names)
	all_durations = [d for durations in stim_durations_by_id.values() for d in durations]

	if len(all_durations) == 0 or n_rois == 0:
		return None

	def _normalize_odor_name(value):
		if value is None:
			return None
		return str(value).strip().upper()

	def _ordered_stimulus_ids(stimulus_ids, vial_map, csp_name, csm_name):
		stim_name_by_id = {
			int(stim_id): _normalize_odor_name(vial_map.get(stim_id, f"V{stim_id}"))
			for stim_id in stimulus_ids
		}

		csp_id = next((sid for sid in stimulus_ids if stim_name_by_id[int(sid)] == csp_name), None)
		csm_id = next((sid for sid in stimulus_ids if stim_name_by_id[int(sid)] == csm_name), None)
		novel_id = next((sid for sid in stimulus_ids if sid not in {csp_id, csm_id}), None)

		ordered = []
		for sid in (csp_id, csm_id, novel_id):
			if sid is not None and sid in stimulus_ids and sid not in ordered:
				ordered.append(sid)

		for sid in stimulus_ids:
			if sid not in ordered:
				ordered.append(sid)

		return ordered

	def _compute_trial_average(trace_1d, trial_segments_for_stim, pre_frames, post_frames, window_len):
		aligned_trials = []

		for start_idx, end_idx in trial_segments_for_stim:
			if end_idx <= start_idx:
				continue

			win_start = start_idx - pre_frames
			win_end = end_idx + post_frames
			trial_window = np.full(window_len, np.nan, dtype=float)

			src_start = max(0, win_start)
			src_end = min(trace_1d.size, win_end)
			if src_end <= src_start:
				continue

			dst_start = src_start - win_start
			dst_end = dst_start + (src_end - src_start)
			trial_window[dst_start:dst_end] = trace_1d[src_start:src_end]
			aligned_trials.append(trial_window)

		if len(aligned_trials) == 0:
			return None

		aligned_trials = np.asarray(aligned_trials)
		return np.nanmean(aligned_trials, axis=0)

	max_stim_frames = int(max(all_durations))
	common_window_len = pre_window_frames + max_stim_frames + post_window_frames
	time_axis_s = (np.arange(common_window_len) - pre_window_frames) / downsampled_fr

	csp_name = _normalize_odor_name(series_meta.get('CSp'))
	csm_name = _normalize_odor_name(series_meta.get('CSm'))
	stimulus_ids_plot_order = _ordered_stimulus_ids(stimulus_ids_unique, vial_to_odor, csp_name, csm_name)

	if roi_responsive_df is not None and not roi_responsive_df.empty:
		roi_responsive_map = dict(
			zip(
				roi_responsive_df['roi'].astype(str),
				roi_responsive_df['is_responsive'].astype(bool),
			)
		)
	else:
		roi_responsive_map = {str(name): False for name in roi_names}

	n_cols = min(max_cols, n_rois)
	n_rows = int(np.ceil(n_rois / n_cols))
	fig = plt.figure(figsize=(n_cols * cell_w, n_rows * cell_h), dpi=150)
	outer_gs = fig.add_gridspec(n_rows, n_cols, wspace=0.1, hspace=0.1)
	stim_colors_fallback = cm.get_cmap('tab20', max(1, len(stimulus_ids_plot_order)))
	trace_axes = []

	for roi_idx, roi_name in enumerate(roi_names):
		grid_row = roi_idx // n_cols
		grid_col = roi_idx % n_cols
		cell_gs = outer_gs[grid_row, grid_col].subgridspec(1, 2, width_ratios=[2.5, 1], wspace=0.2)

		ax_img = fig.add_subplot(cell_gs[0, 0])
		if len(trace_axes) == 0:
			ax_trace = fig.add_subplot(cell_gs[0, 1])
		else:
			ax_trace = fig.add_subplot(cell_gs[0, 1], sharey=trace_axes[0])
		trace_axes.append(ax_trace)

		roi_mask = np.asarray(roi_masks[roi_idx]).astype(bool)
		trace = np.asarray(bg_subtracted_df_traces[roi_idx], dtype=float)

		ax_img.imshow(extraction_image, cmap='gray')
		roi_overlay = np.ma.masked_where(~roi_mask, roi_mask.astype(float))
		ax_img.imshow(roi_overlay, cmap='spring', alpha=0.45)
		ax_img.set_title(f'{roi_name} mask', fontsize=font_size)
		ax_img.set_xticks([])
		ax_img.set_yticks([])

		for stim_idx, stim_id in enumerate(stimulus_ids_plot_order):
			trial_segments_this_stim = [
				(start_idx, end_idx)
				for start_idx, end_idx, seg_stim_id in stim_segments
				if seg_stim_id == stim_id
			]
			if len(trial_segments_this_stim) == 0:
				continue

			avg_trace = _compute_trial_average(
				trace_1d=trace,
				trial_segments_for_stim=trial_segments_this_stim,
				pre_frames=pre_window_frames,
				post_frames=post_window_frames,
				window_len=common_window_len,
			)
			if avg_trace is None:
				continue

			stim_name = vial_to_odor.get(stim_id, f"V{stim_id}")
			stim_color = colors_hex.get(stim_name, stim_colors_fallback(stim_idx))

			stim_duration_list = stim_durations_by_id.get(stim_id, [])
			stim_duration_ref_s = (
				float(np.median(stim_duration_list) / downsampled_fr)
				if len(stim_duration_list) > 0
				else 0.0
			)

			ax_trace.axvspan(0, stim_duration_ref_s, color=stim_color, alpha=0.10, zorder=0)
			ax_trace.plot(
				time_axis_s,
				avg_trace,
				lw=1.5,
				color=stim_color,
				label=f"{stim_name} ({stim_id})",
			)

		ax_trace.axvline(0, color='k', ls='--', lw=1)
		ax_trace.axhline(0, color='0.7', lw=0.8)
		is_resp = bool(roi_responsive_map.get(str(roi_name), False))
		frame_color = 'green' if is_resp else 'red'
		for spine in ax_img.spines.values():
			spine.set_visible(True)
			spine.set_edgecolor(frame_color)
			spine.set_linewidth(2.0)
		resp_label = 'responsive' if is_resp else 'non-responsive'
		ax_trace.set_title(f'{roi_name} ({resp_label})', fontsize=font_size)
		ax_trace.set_xlabel('Time (s)', fontsize=font_size)
		ax_trace.set_ylabel('ΔF/F', fontsize=font_size)
		ax_trace.tick_params(labelsize=font_size)

	for empty_idx in range(n_rois, n_rows * n_cols):
		grid_row = empty_idx // n_cols
		grid_col = empty_idx % n_cols
		empty_ax = fig.add_subplot(outer_gs[grid_row, grid_col])
		empty_ax.axis('off')

	handles, labels = trace_axes[0].get_legend_handles_labels() if len(trace_axes) > 0 else ([], [])
	if len(handles) > 0:
		fig.legend(
			handles,
			labels,
			loc='lower center',
			bbox_to_anchor=(0.5, 0.005),
			ncol=min(4, len(labels)),
			frameon=True,
			fontsize=font_size,
		)

	fig.suptitle(
		f"{series_id}, CSp: {series_meta['CSp']} | Trial-averaged ROI responses",
		fontsize=max(font_size, 8),
	)
	fig.tight_layout(rect=[0, 0.06, 1, 0.97])

	combined_plot_path = f"{roi_trial_plot_dir}/{series_id}_all_stimuli_roi_mask_and_trace_trialavg.png"
	fig.savefig(combined_plot_path, bbox_inches='tight')
	plt.close(fig)
	return combined_plot_path


def plot_fly_trial_panels(
	fly_ids,
	status_label,
	roi_database_all,
	roi_trial_plot_windows,
	pass_fail_dir,
	normalize_odor_name,
	safe_filename,
):
	odor_order = ['CSp', 'CSm', 'Novel']
	n_trials_to_plot = 2

	for fly_id in sorted(fly_ids):
		fly_roi_df = roi_database_all.loc[roi_database_all['flyID'] == fly_id]
		if fly_roi_df.empty:
			continue

		roi_names = fly_roi_df['roi_unique_name'].tolist()
		n_rows = len(roi_names)
		n_cols = len(odor_order) * n_trials_to_plot

		fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 2.2 * n_rows), squeeze=False)

		for row_idx, (_, roi_row) in enumerate(fly_roi_df.iterrows()):
			roi_name = roi_row['roi_unique_name']
			roi_frame_rate_hz = roi_row.get('frame_rate_hz')
			try:
				roi_frame_rate_hz = float(roi_frame_rate_hz)
			except (TypeError, ValueError):
				roi_frame_rate_hz = np.nan
			if (not np.isfinite(roi_frame_rate_hz)) or roi_frame_rate_hz <= 0:
				roi_frame_rate_hz = 1.0
			roi_odor_windows = roi_trial_plot_windows.get(roi_name, {})

			csp_norm = normalize_odor_name(roi_row.get('CSp'))
			csm_norm = normalize_odor_name(roi_row.get('CSm'))
			novel_norm = next((odor for odor in roi_odor_windows.keys() if odor not in {csp_norm, csm_norm}), None)

			identity_to_odor = {
				'CSp': csp_norm,
				'CSm': csm_norm,
				'Novel': novel_norm,
			}

			for odor_idx, odor_identity in enumerate(odor_order):
				odor_key = identity_to_odor.get(odor_identity)
				trial_windows = roi_odor_windows.get(odor_key, []) if odor_key is not None else []

				for trial_idx in range(n_trials_to_plot):
					col_idx = odor_idx * n_trials_to_plot + trial_idx
					axis = axes[row_idx, col_idx]

					if trial_idx < len(trial_windows):
						window_info = trial_windows[trial_idx]
						trace = np.asarray(window_info.get('trace', np.array([], dtype=float)), dtype=float)
						stim_start = int(window_info.get('stim_start', 0))
						stim_end = int(window_info.get('stim_end', 0))
						is_pass = bool(window_info.get('is_pass', False))

						if trace.size > 0:
							time_axis = np.arange(trace.size) / roi_frame_rate_hz
							axis.plot(time_axis, trace, color='black', linewidth=1.0)
							axis.axvspan(stim_start / roi_frame_rate_hz, stim_end / roi_frame_rate_hz, color='red', alpha=0.18)
						axis.set_title(f"{odor_identity} T{trial_idx + 1} {'PASS' if is_pass else 'FAIL'}", fontsize=8)
					else:
						axis.set_title(f"{odor_identity} T{trial_idx + 1} NA", fontsize=8)

					if col_idx == 0:
						axis.set_ylabel(roi_name, fontsize=8)
					if row_idx == n_rows - 1:
						axis.set_xlabel('Time (s)', fontsize=8)
					axis.tick_params(axis='both', which='major', labelsize=7)

		fig.suptitle(f"Fly {fly_id} | {status_label}", y=1.01)
		plt.tight_layout()
		fly_file = f"fly_{safe_filename(fly_id)}_{safe_filename(status_label.lower())}_trial_panels.png"
		fig.savefig(f"{pass_fail_dir}/{fly_file}", dpi=200, bbox_inches='tight')
		plt.show()


def plot_group_identity_boxplots(
	roi_identity_subset,
	roi_database_subset,
	figure_title,
	output_path,
	normalize_odor_name,
	colors_hex,
	roi_avg_response_by_odor_dict,
	pvalue_to_stars,
):
	if roi_identity_subset.empty or roi_database_subset.empty:
		print(f"No ROI identity responses available for: {figure_title}")
		return

	odor_order = ['CSp', 'CSm', 'Novel']
	x_positions = np.arange(len(odor_order))

	grouped = list(roi_identity_subset.groupby('group'))
	n_groups = len(grouped)
	if n_groups == 0:
		print(f"No groups available for: {figure_title}")
		return

	fig, axes = plt.subplots(n_groups, 1, figsize=(8, 3.2 * n_groups), sharex=True)
	if n_groups == 1:
		axes = [axes]

	rng = np.random.default_rng(seed=42)

	for axis, (group_name, group_df) in zip(axes, grouped):
		box_data = []
		n_by_odor = []
		group_meta = roi_database_subset.loc[roi_database_subset['group'] == group_name, ['CSp', 'CSm']]
		csp_group_odor = normalize_odor_name(group_meta['CSp'].dropna().iloc[0]) if not group_meta['CSp'].dropna().empty else None
		csm_group_odor = normalize_odor_name(group_meta['CSm'].dropna().iloc[0]) if not group_meta['CSm'].dropna().empty else None
		group_roi_names = set(roi_database_subset.loc[roi_database_subset['group'] == group_name, 'roi_unique_name'])
		all_group_odors = {
			normalize_odor_name(k)
			for roi_name_key, odor_dict in roi_avg_response_by_odor_dict.items()
			if roi_name_key in group_roi_names
			if isinstance(odor_dict, dict)
			for k in odor_dict.keys()
		}
		novel_group_odor = next(
			(odor for odor in all_group_odors if odor not in {csp_group_odor, csm_group_odor}),
			None,
		)
		identity_to_odor = {
			'CSp': csp_group_odor,
			'CSm': csm_group_odor,
			'Novel': novel_group_odor,
		}
		box_colors = [
			colors_hex.get(identity_to_odor.get(odor_identity), '#999999')
			for odor_identity in odor_order
		]

		for odor in odor_order:
			odor_values = group_df.loc[group_df['odor_identity'] == odor, 'response'].to_numpy(dtype=float)
			odor_values = odor_values[np.isfinite(odor_values)]
			box_data.append(odor_values)
			n_by_odor.append(odor_values.size)

		boxplot_artists = axis.boxplot(
			box_data,
			positions=x_positions,
			widths=0.35,
			showfliers=False,
			patch_artist=True,
			medianprops={'linewidth': 2, 'color': 'black'},
		)

		for patch, color in zip(boxplot_artists['boxes'], box_colors):
			patch.set_facecolor(color)
			patch.set_alpha(0.45)

		for odor_idx, odor_values in enumerate(box_data):
			if odor_values.size == 0:
				continue
			jitter = rng.uniform(-0.10, 0.10, size=odor_values.size)
			axis.scatter(
				np.full(odor_values.size, x_positions[odor_idx]) + jitter,
				odor_values,
				s=22,
				alpha=0.7,
				color=box_colors[odor_idx],
				edgecolors='black',
				linewidths=0.25,
			)

		csp_csm_pairs = (
			group_df
			.pivot_table(index='roi_unique_name', columns='odor_identity', values='response', aggfunc='mean')
			.reindex(columns=['CSp', 'CSm'])
			.dropna()
		)
		csp_csm_pvalue = np.nan
		if len(csp_csm_pairs) >= 2:
			try:
				csp_csm_pvalue = float(
					wilcoxon(
						csp_csm_pairs['CSp'].to_numpy(dtype=float),
						csp_csm_pairs['CSm'].to_numpy(dtype=float),
						alternative='two-sided',
						zero_method='wilcox',
					).pvalue
				)
			except ValueError:
				csp_csm_pvalue = np.nan

		y_annot = 0.92
		axis.plot([0, 0, 1, 1], [y_annot - 0.02, y_annot, y_annot, y_annot - 0.02], color='black', linewidth=1.0)
		axis.text(
			0.5,
			y_annot + 0.015,
			f"CSp vs CSm: p={csp_csm_pvalue:.3g} ({pvalue_to_stars(csp_csm_pvalue)})",
			ha='center',
			va='bottom',
			fontsize=9,
		)

		axis.set_ylabel('Response')
		axis.set_ylim(-0.2, 1.0)
		axis.set_title(
			f"Group: {group_name} | n(CSp, CSm, Novel)=({n_by_odor[0]}, {n_by_odor[1]}, {n_by_odor[2]})"
		)
		axis.grid(axis='y', alpha=0.25)

	axes[-1].set_xticks(x_positions)
	axes[-1].set_xticklabels(odor_order)
	axes[-1].set_xlabel('Odor identity')
	fig.suptitle(figure_title, y=1.02)
	plt.tight_layout()
	fig.savefig(output_path, dpi=200, bbox_inches='tight')
	plt.show()


def plot_trial_overlaid_roi_responses(
	roi_names,
	roi_masks,
	bg_subtracted_df_traces,
	extraction_image,
	stim_segments,
	stimulus_ids_unique,
	vial_to_odor,
	downsampled_fr,
	series_id,
	series_meta,
	roi_trial_plot_dir,
	pre_window_frames,
	post_window_frames,
	colors_hex=None,
	font_size=6,
	cell_w=3.2,
	cell_h=2.4,
):
	if colors_hex is None:
		colors_hex = {}

	n_rois = len(roi_names)
	if n_rois == 0 or len(stimulus_ids_unique) == 0:
		return None

	def _normalize_odor_name(value):
		if value is None:
			return None
		return str(value).strip().upper()

	def _ordered_stimulus_ids(stimulus_ids, vial_map, csp_name, csm_name):
		stim_name_by_id = {
			int(stim_id): _normalize_odor_name(vial_map.get(stim_id, f"V{stim_id}"))
			for stim_id in stimulus_ids
		}

		csp_id = next((sid for sid in stimulus_ids if stim_name_by_id[int(sid)] == csp_name), None)
		csm_id = next((sid for sid in stimulus_ids if stim_name_by_id[int(sid)] == csm_name), None)
		novel_id = next((sid for sid in stimulus_ids if sid not in {csp_id, csm_id}), None)

		ordered = []
		for sid in (csp_id, csm_id, novel_id):
			if sid is not None and sid in stimulus_ids and sid not in ordered:
				ordered.append(sid)

		for sid in stimulus_ids:
			if sid not in ordered:
				ordered.append(sid)

		return ordered

	def _extract_onset_aligned_trial(trace_1d, stim_start, pre_frames, post_frames):
		window_len = pre_frames + post_frames
		win_start = stim_start - pre_frames
		win_end = stim_start + post_frames

		trial_window = np.full(window_len, np.nan, dtype=float)
		src_start = max(0, win_start)
		src_end = min(trace_1d.size, win_end)

		if src_end <= src_start:
			return trial_window

		dst_start = src_start - win_start
		dst_end = dst_start + (src_end - src_start)
		trial_window[dst_start:dst_end] = trace_1d[src_start:src_end]
		return trial_window

	csp_name = _normalize_odor_name(series_meta.get('CSp'))
	csm_name = _normalize_odor_name(series_meta.get('CSm'))
	stimulus_ids_plot_order = _ordered_stimulus_ids(stimulus_ids_unique, vial_to_odor, csp_name, csm_name)[:3]

	if len(stimulus_ids_plot_order) < 3:
		missing = 3 - len(stimulus_ids_plot_order)
		stimulus_ids_plot_order = stimulus_ids_plot_order + ([None] * missing)

	n_rows = 4
	fig, axes = plt.subplots(
		n_rows,
		n_rois,
		figsize=(max(1, n_rois) * cell_w, n_rows * cell_h),
		dpi=150,
		squeeze=False,
	)

	time_axis_s = (np.arange(pre_window_frames + post_window_frames) - pre_window_frames) / downsampled_fr
	pre_s = pre_window_frames / downsampled_fr
	post_s = post_window_frames / downsampled_fr

	for roi_idx, roi_name in enumerate(roi_names):
		roi_mask = np.asarray(roi_masks[roi_idx]).astype(bool)
		trace = np.asarray(bg_subtracted_df_traces[roi_idx], dtype=float)

		ax_mask = axes[0, roi_idx]
		ax_mask.imshow(extraction_image, cmap='gray')
		roi_overlay = np.ma.masked_where(~roi_mask, roi_mask.astype(float))
		ax_mask.imshow(roi_overlay, cmap='spring', alpha=0.45)
		ax_mask.set_title(str(roi_name), fontsize=font_size)
		ax_mask.set_xticks([])
		ax_mask.set_yticks([])

		for odor_row, stim_id in enumerate(stimulus_ids_plot_order, start=1):
			ax = axes[odor_row, roi_idx]
			ax.axvline(0, color='k', ls='--', lw=0.8)
			ax.axhline(0, color='0.75', lw=0.7)

			if stim_id is None:
				ax.text(0.5, 0.5, 'No odor', ha='center', va='center', transform=ax.transAxes, fontsize=font_size)
				ax.set_xlim(-pre_s, post_s)
				continue

			trial_segments_this_stim = [
				(start_idx, end_idx)
				for start_idx, end_idx, seg_stim_id in stim_segments
				if seg_stim_id == stim_id
			]

			stim_name = vial_to_odor.get(stim_id, f"V{stim_id}")
			stim_color = colors_hex.get(stim_name, None)

			if len(trial_segments_this_stim) == 0:
				ax.text(0.5, 0.5, f"{stim_name}\n(no trials)", ha='center', va='center', transform=ax.transAxes, fontsize=font_size)
				ax.set_xlim(-pre_s, post_s)
				ax.set_title(str(stim_name), fontsize=font_size)
				continue

			stim_durations = [max(0, int(end_idx - start_idx)) for start_idx, end_idx in trial_segments_this_stim]
			stim_dur_s = float(np.median(stim_durations) / downsampled_fr) if len(stim_durations) > 0 else 0.0
			stim_dur_s = min(stim_dur_s, post_s)
			if stim_dur_s > 0:
				ax.axvspan(0, stim_dur_s, color=stim_color if stim_color is not None else '0.8', alpha=0.10, zorder=0)

			trial_cmap = cm.get_cmap('viridis', max(2, len(trial_segments_this_stim)))
			for trial_i, (start_idx, end_idx) in enumerate(trial_segments_this_stim, start=1):
				if end_idx <= start_idx:
					continue
				trial_trace = _extract_onset_aligned_trial(
					trace_1d=trace,
					stim_start=start_idx,
					pre_frames=pre_window_frames,
					post_frames=post_window_frames,
				)
				ax.plot(
					time_axis_s,
					trial_trace,
					lw=0.9,
					alpha=0.9,
					color=trial_cmap(trial_i - 1),
					label=f"T{trial_i}",
				)

			ax.set_xlim(-pre_s, post_s)
			ax.set_title(f"{stim_name} (n={len(trial_segments_this_stim)})", fontsize=font_size)
			if roi_idx == 0:
				ax.set_ylabel('ΔF/F', fontsize=font_size)
			if odor_row == 3:
				ax.set_xlabel('Time (s)', fontsize=font_size)
			ax.tick_params(labelsize=font_size)

	fig.suptitle(
		f"{series_id}, CSp: {series_meta.get('CSp', 'NA')} | Single-trial odor responses (5 s pre, 15 s post)",
		fontsize=max(font_size, 8),
	)
	fig.tight_layout(rect=[0, 0.02, 1, 0.96])

	combined_plot_path = f"{roi_trial_plot_dir}/{series_id}_all_stimuli_roi_mask_and_trace_singletrials.png"
	fig.savefig(combined_plot_path, bbox_inches='tight')
	plt.close(fig)
	return combined_plot_path
