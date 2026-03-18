MOTION_CORRECTION_PROFILES = {
	'dendrites': {
		'nonrigid': False,
		'block_size': (128, 128),
		'smooth_sigma': 2.0,
		'main_chan': 0,
		'maxregshift': 0.1,
		'maxregshiftNR': 10,
	},
	'boutons': {
		'nonrigid': False,
		'block_size': (128, 128),
		'smooth_sigma': 2.0,
		'main_chan': 0,
		'maxregshift': 0.1,
		'maxregshiftNR': 10,
	},
}
def get_motion_correction_params(profile_name):
	if profile_name not in MOTION_CORRECTION_PROFILES:
		available_profiles = ', '.join(sorted(MOTION_CORRECTION_PROFILES.keys()))
		raise ValueError(f"Unknown motion profile '{profile_name}'. Available: {available_profiles}")

	return dict(MOTION_CORRECTION_PROFILES[profile_name])
