# Expose Wildtrack IO and cross-camera matcher utilities
from .wildtrack_io import (
	WildtrackBBox, parse_annotation_file, load_all_annotations,
	annotations_to_per_camera_per_frame, list_camera_frames, list_annotation_files,
	WILDTRACK_NUM_CAMERAS,
)
from .matcher import (
	LocalTrack, CrossCameraMatcher, load_camera_tracks,
)
from .evaluator import (
	Prediction,
	load_predictions_from_mot_files,
	match_predictions_to_gt,
	compute_identity_metrics,
	evaluate_wildtrack,
)
