import json

from src.cross_camera.wildtrack_io import (
    WildtrackBBox,
    annotations_to_per_camera_per_frame,
    load_all_annotations,
    parse_annotation_file,
)


def make_sample_annotation():
    return [
        {
            "personID": 10,
            "positionID": 435095,
            "views": [
                {"viewNum": 0, "xmin": 1202, "ymin": 128, "xmax": 1248, "ymax": 284},
                {"viewNum": 1, "xmin": 1357, "ymin": 3, "xmax": 1440, "ymax": 304},
                {"viewNum": 2, "xmin": -1, "ymin": -1, "xmax": -1, "ymax": -1},
                {"viewNum": 3, "xmin": -1, "ymin": -1, "xmax": -1, "ymax": -1},
                {"viewNum": 4, "xmin": -1, "ymin": -1, "xmax": -1, "ymax": -1},
                {"viewNum": 5, "xmin": 429, "ymin": 121, "xmax": 463, "ymax": 255},
                {"viewNum": 6, "xmin": 1781, "ymin": 87, "xmax": 1899, "ymax": 323},
            ],
        },
        {
            "personID": 37,
            "positionID": 486745,
            "views": [
                {"viewNum": 0, "xmin": 953, "ymin": 104, "xmax": 987, "ymax": 232},
                {"viewNum": 1, "xmin": -1, "ymin": -1, "xmax": -1, "ymax": -1},
                {"viewNum": 2, "xmin": -1, "ymin": -1, "xmax": -1, "ymax": -1},
                {"viewNum": 3, "xmin": -1, "ymin": -1, "xmax": -1, "ymax": -1},
                {"viewNum": 4, "xmin": -1, "ymin": -1, "xmax": -1, "ymax": -1},
                {"viewNum": 5, "xmin": 31, "ymin": 116, "xmax": 68, "ymax": 250},
                {"viewNum": 6, "xmin": 1608, "ymin": 14, "xmax": 1685, "ymax": 188},
            ],
        },
    ]


def test_parse_filters_invisible_views(tmp_path):
    sample_path = tmp_path / "00000000.json"
    sample_path.write_text(json.dumps(make_sample_annotation()))
    bboxes = parse_annotation_file(sample_path, frame_idx=1)
    assert len(bboxes) == 7
    person_10 = [b for b in bboxes if b.person_id == 10]
    person_37 = [b for b in bboxes if b.person_id == 37]
    assert len(person_10) == 4
    assert len(person_37) == 3
    for b in bboxes:
        assert b.x1 >= 0 and b.y1 >= 0 and b.x2 > b.x1 and b.y2 > b.y1
        assert b.frame_idx == 1


def test_load_all_annotations_assigns_sequential_frame_idx(tmp_path):
    for _, name in enumerate(["00000000.json", "00000005.json", "00000010.json"], start=1):
        (tmp_path / name).write_text(json.dumps(make_sample_annotation()))
    bboxes = load_all_annotations(tmp_path)
    frame_indices = sorted({b.frame_idx for b in bboxes})
    assert frame_indices == [1, 2, 3]


def test_per_camera_per_frame_organization():
    bboxes = [
        WildtrackBBox(camera_id=0, person_id=10, frame_idx=1, x1=0, y1=0, x2=10, y2=10),
        WildtrackBBox(camera_id=0, person_id=20, frame_idx=1, x1=20, y1=20, x2=30, y2=30),
        WildtrackBBox(camera_id=2, person_id=10, frame_idx=1, x1=0, y1=0, x2=10, y2=10),
        WildtrackBBox(camera_id=0, person_id=10, frame_idx=2, x1=5, y1=5, x2=15, y2=15),
    ]
    result = annotations_to_per_camera_per_frame(bboxes, num_cameras=7)
    assert len(result) == 7
    assert len(result[0][1]) == 2
    assert len(result[0][2]) == 1
    assert len(result[2][1]) == 1
    assert 1 not in result[1]
