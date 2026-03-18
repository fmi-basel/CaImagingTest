AUTO_ROI_PROFILES = {
    'boutons': {
        'n_samples': 3,
        'footprint_size': 11,
        'gaussian_sigma': 1.0,
        'threshold_percentile': 98,
        'min_distance_factor': 0.5,
        'figsize': (20, 10),
        'watershed_threshold_percentile': 97.5,
        'compactness': 0.01,
        'min_area_factor': 0.4,
        'max_area_factor': 5.0,
        'relative_peak_fraction': 0.90,
    }
    
}


def get_auto_roi_params(profile_name):
    if profile_name is None:
        return None
    if profile_name not in AUTO_ROI_PROFILES:
        available_profiles = ', '.join(sorted(AUTO_ROI_PROFILES.keys()))
        raise ValueError(f"Unknown auto ROI profile '{profile_name}'. Available: {available_profiles}")

    return dict(AUTO_ROI_PROFILES[profile_name])
