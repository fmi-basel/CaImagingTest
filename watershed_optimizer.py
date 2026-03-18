#%%


from automatic_roi import centers_to_watershed_masks, detect_centers_from_image, detect_centers_from_image, estimate_bouton_diameter_px, estimate_bouton_diameter_px,plot_watershed_rois
from roi_processor import filter_centers_by_mask, select_polygon_mask
extraction_image = processed_movie_cropped.mean(axis=0)


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
    threshold_percentile=99.8,
)
#%%
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
#%%
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
#%%

threshold_percentile = 99
compactness = 0.01
min_area_factor = 0.4
max_area_factor = 5.0
relative_peak_fraction = 0.8
labels_ws, roi_masks, kept_centers = centers_to_watershed_masks(
    extraction_image,
    coordinates,
    expected_diameter_px=bouton_diameter_px,
    threshold_percentile=threshold_percentile,
    marker_radius_px=max(1, int(round(bouton_radius_px / 3))),
    smooth_sigma=1,
    compactness=compactness,
    min_area_factor=min_area_factor,
    max_area_factor=max_area_factor,
    relative_peak_fraction=relative_peak_fraction,
)

ws_fig = plot_watershed_rois(clean_sd_smoothed, kept_centers, labels_ws)
if results_dir is not None:
    ws_fig.savefig(f"{results_dir}/{series_id}_watershed_roi_masks.png", dpi=150, bbox_inches='tight')
# %%
