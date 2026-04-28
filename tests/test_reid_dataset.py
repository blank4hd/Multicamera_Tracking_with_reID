import os

import pytest

from src.reid import parse_market1501_filename


def test_parse_standard():
    pid, camid = parse_market1501_filename("0002_c1s1_000451_03.jpg")
    assert pid == 2
    assert camid == 1


def test_parse_camera_6():
    pid, camid = parse_market1501_filename("1501_c6s4_001877_02.jpg")
    assert pid == 1501
    assert camid == 6


def test_parse_distractor_zero():
    pid, camid = parse_market1501_filename("0000_c1s1_000000_00.jpg")
    assert pid == 0
    assert camid == 1


def test_parse_junk_minus_one():
    pid, camid = parse_market1501_filename("-1_c1s1_000000_00.jpg")
    assert pid == -1
    assert camid == 1


def test_parse_invalid():
    assert parse_market1501_filename("not_a_market_file.txt") is None
    assert parse_market1501_filename("Thumbs.db") is None


DATA_ROOT = "data/Market-1501-v15.09.15"
HAS_DATA = os.path.isdir(os.path.join(DATA_ROOT, "bounding_box_train"))


@pytest.mark.skipif(not HAS_DATA, reason="Market-1501 not available")
def test_train_dataset_loads():
    from src.reid import Market1501Dataset, build_train_transform

    ds = Market1501Dataset(DATA_ROOT, split="train", transform=build_train_transform())
    assert len(ds) > 0
    assert 12000 < len(ds) < 13000
    assert ds.num_classes == 751
    img, label, camid = ds[0]
    assert img.shape == (3, 256, 128)
    assert 0 <= label < ds.num_classes
    assert 1 <= camid <= 6


@pytest.mark.skipif(not HAS_DATA, reason="Market-1501 not available")
def test_query_dataset_no_relabel():
    from src.reid import Market1501Dataset, build_eval_transform

    ds = Market1501Dataset(DATA_ROOT, split="query", transform=build_eval_transform())
    assert len(ds) > 3000
    img, pid, camid = ds[0]
    assert pid > 0


@pytest.mark.skipif(not HAS_DATA, reason="Market-1501 not available")
def test_random_identity_sampler():
    from collections import Counter

    from src.reid import Market1501Dataset, RandomIdentitySampler, build_train_transform

    ds = Market1501Dataset(DATA_ROOT, split="train", transform=build_train_transform())
    sampler = RandomIdentitySampler(ds, batch_size=64, num_instances=4)
    indices = list(sampler)
    assert len(indices) % 64 == 0
    for i in range(0, min(len(indices), 64 * 5), 64):
        chunk = indices[i:i + 64]
        labels_in_chunk = [ds.samples[idx][1] for idx in chunk]
        counts = Counter(labels_in_chunk)
        assert len(counts) == 16, f"Expected 16 unique IDs in batch, got {len(counts)}"
        for label, count in counts.items():
            assert count == 4, f"Expected 4 instances per ID, got {count} for label {label}"
