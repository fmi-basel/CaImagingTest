import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, label as ndi_label
from skimage.segmentation import watershed
from skimage.draw import disk
from scipy.ndimage import white_tophat
from skimage.feature import peak_local_max
from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector

def estimate_bouton_diameter_px(image, n_samples=3):
    """Estimate bouton diameter (pixels) by manually clicking diameter endpoints."""
    fig, ax = plt.subplots(figsize=(20,8))
    ax.imshow(
        image,
        cmap='magma'
    )
    ax.set_title(
        f"Click 2 points per bouton diameter ({n_samples} boutons).\\n"
        "Press Enter when done.",
        fontsize=11,
    )
    ax.axis('off')

    points = plt.ginput(2 * n_samples, timeout=0)
    plt.close(fig)

    if len(points) < 2:
        print("No manual size selected. Falling back to default diameter = 12 px")
        return 12

    if len(points) % 2 != 0:
        points = points[:-1]

    diameters = []
    for i in range(0, len(points), 2):
        p1 = np.array(points[i])
        p2 = np.array(points[i + 1])
        diameters.append(np.linalg.norm(p2 - p1))

    if len(diameters) == 0:
        print("Could not compute diameter from clicks. Falling back to 12 px")
        return 12

    diameter_px = float(np.median(diameters))
    print(f"Estimated bouton diameter: {diameter_px:.2f} px (from {len(diameters)} samples)")
    return diameter_px


def centers_to_watershed_masks(
    image,
    centers,
    expected_diameter_px,
    threshold_percentile=85,
    marker_radius_px=1,
    smooth_sigma=1.0,
    compactness=0.0,
    min_area_factor=0.25,
    max_area_factor=3.0,
    relative_peak_fraction=0.55,
):
    """
    Convert center coordinates into ROI masks using marker-controlled watershed.

    Parameters
    ----------
    image : np.ndarray
        2D extraction image (typically std map).
    centers : array-like, shape (N, 2)
        ROI center coordinates in (row, col) format.
    expected_diameter_px : float
        Estimated bouton diameter in pixels.
    threshold_percentile : float
        Percentile used as a global floor for ROI intensity.
    marker_radius_px : int
        Radius of each marker seed around center.
    smooth_sigma : float
        Gaussian smoothing sigma before watershed.
    compactness : float
        Watershed compactness parameter.
    min_area_factor, max_area_factor : float
        Area limits as multipliers of expected circular area.
    relative_peak_fraction : float
        Fraction of each ROI's own peak intensity used for adaptive thresholding.

    Returns
    -------
    labels_filtered : np.ndarray
        2D label image after area filtering (0 is background).
    roi_masks : list[np.ndarray]
        List of boolean masks, one per kept ROI.
    kept_centers : np.ndarray
        Centers corresponding to kept ROI labels.
    """
    image = np.asarray(image)
    centers = np.asarray(centers, dtype=float)

    if image.ndim != 2:
        raise ValueError("`image` must be a 2D array")
    if centers.size == 0:
        return np.zeros_like(image, dtype=np.int32), [], np.empty((0, 2), dtype=int)

    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("`centers` must have shape (N, 2) in (row, col) format")

    smoothed = gaussian_filter(image.astype(np.float32), sigma=smooth_sigma)

    intensity_floor = np.percentile(smoothed, threshold_percentile)
    foreground = np.ones_like(smoothed, dtype=bool)

    markers = np.zeros_like(image, dtype=np.int32)
    valid_centers = []

    for idx, (row_f, col_f) in enumerate(centers, start=1):
        row = int(round(row_f))
        col = int(round(col_f))

        if row < 0 or row >= image.shape[0] or col < 0 or col >= image.shape[1]:
            continue

        rr, cc = disk((row, col), radius=max(1, int(marker_radius_px)), shape=image.shape)
        markers[rr, cc] = idx
        valid_centers.append((row, col))

    if len(valid_centers) == 0:
        return np.zeros_like(image, dtype=np.int32), [], np.empty((0, 2), dtype=int)

    valid_centers = np.array(valid_centers, dtype=int)

    labels = watershed(-smoothed, markers=markers, mask=foreground, compactness=compactness)

    expected_radius = max(1.0, float(expected_diameter_px) / 2.0)
    expected_area = np.pi * (expected_radius ** 2)
    min_area = max(4, int(round(expected_area * min_area_factor)))
    max_area = int(round(expected_area * max_area_factor))

    labels_filtered = np.zeros_like(labels, dtype=np.int32)
    roi_masks = []
    kept_centers = []
    next_label = 1

    for label_id in range(1, labels.max() + 1):
        label_mask = labels == label_id
        if not np.any(label_mask):
            continue

        center_idx = label_id - 1
        if center_idx < len(valid_centers):
            center_rc = valid_centers[center_idx]
        else:
            centroid = np.argwhere(label_mask).mean(axis=0)
            center_rc = np.round(centroid).astype(int)

        peak_intensity = float(np.max(smoothed[label_mask]))
        adaptive_threshold = max(float(intensity_floor), peak_intensity * float(relative_peak_fraction))
        mask = label_mask & (smoothed >= adaptive_threshold)

        row_c, col_c = int(center_rc[0]), int(center_rc[1])
        row_c = np.clip(row_c, 0, image.shape[0] - 1)
        col_c = np.clip(col_c, 0, image.shape[1] - 1)

        if np.any(mask):
            connected_labels, _ = ndi_label(mask)
            center_component = connected_labels[row_c, col_c]
            if center_component != 0:
                mask = connected_labels == center_component
            else:
                mask[row_c, col_c] = True
        else:
            mask[row_c, col_c] = True

        area = int(mask.sum())

        if area < min_area or area > max_area:
            continue

        labels_filtered[mask] = next_label
        roi_masks.append(mask)

        kept_centers.append(np.array([row_c, col_c], dtype=int))

        next_label += 1

    if len(kept_centers) == 0:
        kept_centers = np.empty((0, 2), dtype=int)
    else:
        kept_centers = np.array(kept_centers, dtype=int)

    return labels_filtered, roi_masks, kept_centers


def detect_centers_from_image(
    extraction_image,
    bouton_diameter_px,
    footprint_size,
    gaussian_sigma,
    min_distance_factor=0.5,
    threshold_percentile=98,
):
    """Detect candidate bouton centers from std/projection image."""
    min_distance_px = max(2, int(round(float(bouton_diameter_px) * min_distance_factor)))
    footprint_size = int(max(3, footprint_size))
    if footprint_size % 2 == 0:
        footprint_size += 1

    footprint = np.ones((footprint_size, footprint_size))
    clean_sd = white_tophat(extraction_image, footprint=footprint)
    clean_sd_smoothed = gaussian_filter(clean_sd, sigma=float(gaussian_sigma))

    threshold_abs = np.percentile(clean_sd_smoothed, threshold_percentile)
    centers = peak_local_max(
        clean_sd_smoothed,
        min_distance=min_distance_px,
        threshold_abs=threshold_abs,
    )

    return centers, clean_sd_smoothed, threshold_abs, min_distance_px


def select_polygon_mask(image, title="Draw polygon mask and press Enter", figsize=(10, 8)):
    """Interactively draw a polygon and return a boolean mask for selected region."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image, cmap='magma')
    ax.set_title(f"{title}\nLeft-click to add points, Enter to confirm, Esc to clear")
    ax.axis('off')

    selected = {'polygon': None}

    def on_select(verts):
        if len(verts) >= 3:
            selected['polygon'] = np.array(verts, dtype=float)
            plt.close(fig)

    selector = PolygonSelector(
        ax,
        on_select,
        props=dict(color='white', linestyle='-', linewidth=2, alpha=1),
    )

    def on_key(event):
        if event.key in ('enter', 'return'):
            verts = np.array(selector.verts, dtype=float)
            if len(verts) >= 3:
                selected['polygon'] = verts
                plt.close(fig)
        elif event.key == 'escape':
            selector.clear()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show(block=False)

    while plt.fignum_exists(fig.number) and selected['polygon'] is None:
        plt.pause(0.05)

    mask = np.zeros_like(image, dtype=bool)
    if selected['polygon'] is None or len(selected['polygon']) < 3:
        return mask, np.empty((0, 2), dtype=float)

    polygon = selected['polygon']
    poly_path = Path(polygon)

    h, w = image.shape
    yy, xx = np.indices((h, w))
    pixel_points = np.column_stack((xx.ravel(), yy.ravel()))
    inside = poly_path.contains_points(pixel_points).reshape(h, w)
    mask[inside] = True

    return mask, polygon


def filter_centers_by_mask(centers, mask):
    """Keep only center coordinates (row, col) that fall within mask."""
    centers = np.asarray(centers)
    if centers.size == 0:
        return np.empty((0, 2), dtype=int)

    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("`centers` must have shape (N, 2) in (row, col) format")

    rows = np.clip(np.round(centers[:, 0]).astype(int), 0, mask.shape[0] - 1)
    cols = np.clip(np.round(centers[:, 1]).astype(int), 0, mask.shape[1] - 1)
    keep = mask[rows, cols]
    return centers[keep].astype(int)


def optimize_roi_parameters(
    extraction_image,
    bouton_diameter_px,
    results_dir=None,
    series_id='series',
    footprint_multipliers=None,
    sigma_values=None,
    threshold_percentile=98,
    watershed_threshold_percentile=85,
):
    """
    Sweep footprint/sigma, score candidates, save inspection grid, and return best parameters.
    """
    if footprint_multipliers is None:
        footprint_multipliers = [0.6, 0.8, 1.0, 1.2, 1.6, 2.0, 2.4]

    if sigma_values is None:
        sigma_values = [
            max(0.6, bouton_diameter_px / 18.0),
            max(0.6, bouton_diameter_px / 16.0),
            max(0.6, bouton_diameter_px / 14.0),
            max(0.6, bouton_diameter_px / 12.0),
            max(0.6, bouton_diameter_px / 10.0),
            max(0.8, bouton_diameter_px / 8.0),
        ]

    bouton_radius_px = max(1, int(round(float(bouton_diameter_px) / 2.0)))
    expected_area = np.pi * (bouton_radius_px ** 2)

    candidate_results = []
    fig, axes = plt.subplots(len(sigma_values), len(footprint_multipliers), figsize=(18, 16))
    axes = np.atleast_2d(axes)

    for i, sigma_test in enumerate(sigma_values):
        for j, mult in enumerate(footprint_multipliers):
            footprint_size_test = int(max(5, round(float(bouton_diameter_px) * mult)))
            if footprint_size_test % 2 == 0:
                footprint_size_test += 1

            centers_test, filtered_test, _, _ = detect_centers_from_image(
                extraction_image,
                bouton_diameter_px=bouton_diameter_px,
                footprint_size=footprint_size_test,
                gaussian_sigma=float(sigma_test),
                min_distance_factor=0.5,
                threshold_percentile=threshold_percentile,
            )

            labels_test, roi_masks_test, kept_centers_test = centers_to_watershed_masks(
                extraction_image,
                centers_test,
                expected_diameter_px=bouton_diameter_px,
                threshold_percentile=watershed_threshold_percentile,
                marker_radius_px=max(1, int(round(bouton_radius_px / 3))),
                smooth_sigma=float(sigma_test),
                compactness=0.001,
                min_area_factor=0.2,
                max_area_factor=3.5,
            )

            areas_test = np.array([mask.sum() for mask in roi_masks_test]) if len(roi_masks_test) > 0 else np.array([])
            large_rois = int(np.sum(areas_test > (1.8 * expected_area))) if len(areas_test) > 0 else 0
            tiny_rois = int(np.sum(areas_test < (0.2 * expected_area))) if len(areas_test) > 0 else 0
            kept_ratio = (len(roi_masks_test) / max(1, len(centers_test)))
            score = (3.0 * kept_ratio) + (0.02 * len(roi_masks_test)) - (0.6 * large_rois) - (0.2 * tiny_rois)

            candidate = {
                'footprint_size': footprint_size_test,
                'sigma': float(sigma_test),
                'n_centers': int(len(centers_test)),
                'n_roi': int(len(roi_masks_test)),
                'kept_ratio': float(kept_ratio),
                'large_rois': int(large_rois),
                'tiny_rois': int(tiny_rois),
                'score': float(score),
            }
            candidate_results.append(candidate)

            ax = axes[i, j]
            ax.imshow(filtered_test, cmap='magma')
            if len(kept_centers_test) > 0:
                ax.plot(kept_centers_test[:, 1], kept_centers_test[:, 0], 'c.', ms=2, alpha=0.8)
            if np.max(labels_test) > 0:
                ax.contour(labels_test > 0, levels=[0.5], colors='w', linewidths=0.4, alpha=0.7)
            ax.set_title(
                f"fp={footprint_size_test}, σ={sigma_test:.2f}\\n"
                f"roi={len(roi_masks_test)}/{len(centers_test)}",
                fontsize=8,
            )
            ax.axis('off')

    plt.tight_layout()

    sweep_figure_path = None
    if results_dir is not None:
        sweep_figure_path = f"{results_dir}/{series_id}_roi_param_sweep.png"
        plt.savefig(sweep_figure_path, dpi=200, bbox_inches='tight')
    plt.show()

    candidate_results_sorted = sorted(candidate_results, key=lambda x: x['score'], reverse=True)
    best_candidate = candidate_results_sorted[0]

    return {
        'best': best_candidate,
        'all_candidates': candidate_results_sorted,
        'sweep_figure_path': sweep_figure_path,
    }


def plot_watershed_rois(image, centers, labels, figsize=(10, 8)):
    """Quick diagnostic plot for centers and watershed labels."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image, cmap='gray')

    if centers is not None and len(centers) > 0:
        centers = np.asarray(centers)
        ax.plot(centers[:, 1], centers[:, 0], 'c.', ms=4, alpha=0.9, label='centers')

    if labels is not None and np.max(labels) > 0:
        mask = np.where(labels==0, np.nan, labels)
        ax.imshow(mask, cmap='tab20', alpha=0.3)

    ax.set_title(f"Watershed ROI masks (n={int(np.max(labels))})")
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    return fig