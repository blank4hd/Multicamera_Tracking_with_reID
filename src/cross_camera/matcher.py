from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class LocalTrack:
    camera_id: int
    local_track_id: int
    gallery_mean: np.ndarray
    first_frame: int
    last_frame: int
    n_observations: int

    @property
    def key(self) -> tuple[int, int]:
        return (self.camera_id, self.local_track_id)


def load_camera_tracks(npz_path: str | Path, camera_id: int) -> list[LocalTrack]:
    data = np.load(npz_path, allow_pickle=True)
    tracks: list[LocalTrack] = []
    for i, tid in enumerate(data["track_ids"]):
        tracks.append(
            LocalTrack(
                camera_id=int(camera_id),
                local_track_id=int(tid),
                gallery_mean=data["gallery_means"][i].astype(np.float32),
                first_frame=int(data["first_frames"][i]),
                last_frame=int(data["last_frames"][i]),
                n_observations=int(data["n_observations"][i]),
            )
        )
    return tracks


def _frames_overlap(a: LocalTrack, b: LocalTrack) -> bool:
    return not (a.last_frame < b.first_frame or b.last_frame < a.first_frame)


def _build_distance_matrix(tracks: list[LocalTrack], same_camera_overlap_blocked: bool = True) -> np.ndarray:
    n = len(tracks)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    means = np.stack([t.gallery_mean for t in tracks], axis=0)
    sim = means @ means.T
    dist = 1.0 - sim
    np.fill_diagonal(dist, np.inf)
    if same_camera_overlap_blocked:
        for i in range(n):
            for j in range(i + 1, n):
                if tracks[i].camera_id == tracks[j].camera_id and _frames_overlap(tracks[i], tracks[j]):
                    dist[i, j] = np.inf
                    dist[j, i] = np.inf
    return dist.astype(np.float32)


def _cluster_average_link(
    initial_dist: np.ndarray,
    threshold: float,
    track_camera_ids: list[int],
    track_first_frames: list[int],
    track_last_frames: list[int],
) -> list[int]:
    n = initial_dist.shape[0]
    if n == 0:
        return []
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    cluster_cam_frames: dict[int, dict[int, list[tuple[int, int]]]] = {
        i: {track_camera_ids[i]: [(track_first_frames[i], track_last_frames[i])]} for i in range(n)
    }
    dist = initial_dist.copy()
    active = np.ones(n, dtype=bool)
    sizes = np.ones(n, dtype=np.int64)
    next_cluster_id = n
    parent = list(range(n))

    def find_root(c: int) -> int:
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    while True:
        active_indices = np.where(active)[0]
        if len(active_indices) < 2:
            break
        sub = dist[np.ix_(active_indices, active_indices)]
        flat_min_idx = np.argmin(sub)
        i_local, j_local = np.unravel_index(flat_min_idx, sub.shape)
        min_dist = sub[i_local, j_local]
        if min_dist >= threshold:
            break
        i = active_indices[i_local]
        j = active_indices[j_local]
        if i == j:
            break

        new_cluster_id = next_cluster_id
        next_cluster_id += 1
        size_i, size_j = sizes[i], sizes[j]
        new_size = size_i + size_j

        merged_cam_frames: dict[int, list[tuple[int, int]]] = {}
        for cam, ranges in cluster_cam_frames[i].items():
            merged_cam_frames.setdefault(cam, []).extend(ranges)
        for cam, ranges in cluster_cam_frames[j].items():
            merged_cam_frames.setdefault(cam, []).extend(ranges)
        cluster_cam_frames[i] = merged_cam_frames

        for k in active_indices:
            if k == i or k == j:
                continue
            d_ik = dist[i, k]
            d_jk = dist[j, k]
            if np.isinf(d_ik) or np.isinf(d_jk):
                new_d = np.inf
            else:
                new_d = (size_i * d_ik + size_j * d_jk) / new_size
            blocked = False
            for cam, ranges_in_merged in merged_cam_frames.items():
                if cam in cluster_cam_frames[k]:
                    for fa, la in ranges_in_merged:
                        for fb, lb in cluster_cam_frames[k][cam]:
                            if not (la < fb or lb < fa):
                                blocked = True
                                break
                        if blocked:
                            break
                if blocked:
                    break
            if blocked:
                new_d = np.inf
            dist[i, k] = new_d
            dist[k, i] = new_d

        sizes[i] = new_size
        active[j] = False
        parent[j] = i
        members[i] = members[i] + members[j]
        del members[j]
        cluster_cam_frames.pop(j, None)
        dist[j, :] = np.inf
        dist[:, j] = np.inf
        dist[j, j] = np.inf

    root_to_label: dict[int, int] = {}
    labels = [0] * n
    for orig_idx in range(n):
        root = find_root(orig_idx)
        if root not in root_to_label:
            root_to_label[root] = len(root_to_label)
        labels[orig_idx] = root_to_label[root]
    return labels


class CrossCameraMatcher:
    def __init__(self, threshold: float = 0.35, same_camera_overlap_blocked: bool = True):
        self.threshold = threshold
        self.same_camera_overlap_blocked = same_camera_overlap_blocked

    def match(self, tracks: list[LocalTrack]) -> dict[tuple[int, int], int]:
        if not tracks:
            return {}
        dist = _build_distance_matrix(tracks, self.same_camera_overlap_blocked)
        labels = _cluster_average_link(
            dist,
            self.threshold,
            track_camera_ids=[t.camera_id for t in tracks],
            track_first_frames=[t.first_frame for t in tracks],
            track_last_frames=[t.last_frame for t in tracks],
        )
        return {tracks[i].key: int(labels[i]) for i in range(len(tracks))}

    def cluster_summary(self, tracks: list[LocalTrack], assignments: dict[tuple[int, int], int]) -> dict:
        n_tracks = len(tracks)
        n_clusters = len(set(assignments.values())) if assignments else 0
        cluster_sizes: dict[int, int] = {}
        cluster_cameras: dict[int, set[int]] = {}
        for t in tracks:
            cid = assignments[t.key]
            cluster_sizes[cid] = cluster_sizes.get(cid, 0) + 1
            cluster_cameras.setdefault(cid, set()).add(t.camera_id)
        size_distribution = sorted(cluster_sizes.values(), reverse=True)
        cross_cam_clusters = sum(1 for cams in cluster_cameras.values() if len(cams) > 1)
        return {
            "n_tracks": n_tracks,
            "n_clusters": n_clusters,
            "max_cluster_size": max(size_distribution) if size_distribution else 0,
            "mean_cluster_size": float(np.mean(size_distribution)) if size_distribution else 0.0,
            "size_distribution": size_distribution,
            "cross_camera_clusters": cross_cam_clusters,
            "single_track_clusters": sum(1 for s in size_distribution if s == 1),
        }
