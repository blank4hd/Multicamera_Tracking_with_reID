# Expose Wildtrack IO and cross-camera matcher utilities
from .wildtrack_io import (
	WildtrackBBox, parse_annotation_file, load_all_annotations,
	annotations_to_per_camera_per_frame, list_camera_frames, list_annotation_files,
	WILDTRACK_NUM_CAMERAS, WILDTRACK_FRAME_STEP,
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
from .visualize import (
	color_for_global_id,
	draw_global_id_boxes,
	draw_global_id_boxes_with_palette,
	HIGHLIGHT_PALETTE_BGR,
	make_grid_2x4,
)
