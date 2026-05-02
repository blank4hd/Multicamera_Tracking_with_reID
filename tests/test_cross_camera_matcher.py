import numpy as np
import pytest
from src.cross_camera import LocalTrack, CrossCameraMatcher

def make_track(cam, tid, embedding, first=1, last=10, n_obs=10):
    e = np.asarray(embedding, dtype=np.float32)
    e = e / max(np.linalg.norm(e), 1e-12)
    return LocalTrack(
        camera_id=cam, local_track_id=tid, gallery_mean=e,
        first_frame=first, last_frame=last, n_observations=n_obs,
    )

def test_empty_input():
    matcher = CrossCameraMatcher()
    result = matcher.match([])
    assert result == {}

def test_two_similar_tracks_different_cameras_merge():
    matcher = CrossCameraMatcher(threshold=0.3)
    tracks = [
        make_track(0, 1, [1, 0, 0, 0]),
        make_track(1, 2, [1, 0.05, 0, 0]),
    ]
    result = matcher.match(tracks)
    assert result[(0, 1)] == result[(1, 2)]

def test_two_dissimilar_tracks_dont_merge():
    matcher = CrossCameraMatcher(threshold=0.3)
    tracks = [
        make_track(0, 1, [1, 0, 0, 0]),
        make_track(1, 2, [0, 0, 0, 1]),
    ]
    result = matcher.match(tracks)
    assert result[(0, 1)] != result[(1, 2)]

def test_same_camera_overlap_blocked():
    matcher = CrossCameraMatcher(threshold=0.5, same_camera_overlap_blocked=True)
    tracks = [
        make_track(0, 1, [1, 0, 0, 0], first=1, last=10),
        make_track(0, 2, [1, 0, 0, 0], first=5, last=15),
    ]
    result = matcher.match(tracks)
    assert result[(0, 1)] != result[(0, 2)]

def test_same_camera_non_overlap_can_merge():
    matcher = CrossCameraMatcher(threshold=0.5, same_camera_overlap_blocked=True)
    tracks = [
        make_track(0, 1, [1, 0, 0, 0], first=1, last=10),
        make_track(0, 2, [1, 0, 0, 0], first=20, last=30),
    ]
    result = matcher.match(tracks)
    assert result[(0, 1)] == result[(0, 2)]

def test_three_camera_chain():
    matcher = CrossCameraMatcher(threshold=0.4)
    tracks = [
        make_track(0, 1, [1.0, 0.0, 0.0]),
        make_track(1, 2, [0.95, 0.3, 0.0]),
        make_track(2, 3, [0.9, 0.4, 0.1]),
    ]
    result = matcher.match(tracks)
    assert result[(0, 1)] == result[(1, 2)] == result[(2, 3)]

def test_no_three_way_merge_via_chain_blocked_by_average_link():
    matcher = CrossCameraMatcher(threshold=0.25)
    tracks = [
        make_track(0, 1, [1.0, 0.0, 0.0, 0.0]),
        make_track(1, 2, [0.9, 0.4, 0.0, 0.0]),
        make_track(2, 3, [0.0, 0.9, 0.4, 0.0]),
    ]
    result = matcher.match(tracks)
    assert result[(0, 1)] == result[(1, 2)]
    assert result[(2, 3)] != result[(0, 1)]

def test_cluster_summary():
    matcher = CrossCameraMatcher(threshold=0.3)
    tracks = [
        make_track(0, 1, [1, 0, 0, 0]),
        make_track(1, 2, [1, 0.05, 0, 0]),
        make_track(2, 3, [0, 0, 0, 1]),
    ]
    assignments = matcher.match(tracks)
    summary = matcher.cluster_summary(tracks, assignments)
    assert summary["n_tracks"] == 3
    assert summary["n_clusters"] == 2
    assert summary["max_cluster_size"] == 2
    assert summary["cross_camera_clusters"] == 1
    assert summary["single_track_clusters"] == 1
