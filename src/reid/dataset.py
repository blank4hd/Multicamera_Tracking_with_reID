import os
import re
from collections import defaultdict
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as T


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
REID_INPUT_HEIGHT = 256
REID_INPUT_WIDTH = 128


def parse_market1501_filename(filename: str) -> tuple[int, int] | None:
    pattern = re.compile(r"^(?P<pid>-?\d+)_c(?P<camid>\d)s\d+_\d+_\d+\.jpg$")
    match = pattern.match(filename)
    if match is None:
        return None
    person_id = int(match.group("pid"))
    camera_id = int(match.group("camid"))
    return person_id, camera_id


class Market1501Dataset(Dataset):
    SPLIT_DIRS = {
        "train": "bounding_box_train",
        "query": "query",
        "gallery": "bounding_box_test",
    }

    def __init__(self, root, split="train", transform=None, relabel=None, filter_distractors=None):
        self.root = Path(root)
        if split not in self.SPLIT_DIRS:
            raise ValueError(f"Unknown split: {split}")

        self.split = split
        self.transform = transform
        self.relabel = relabel if relabel is not None else split == "train"
        self.filter_distractors = filter_distractors if filter_distractors is not None else split == "train"

        split_dir = self.root / self.SPLIT_DIRS[split]
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Market-1501 split directory not found: {split_dir}")

        samples = []
        for filename in sorted(os.listdir(split_dir)):
            parsed = parse_market1501_filename(filename)
            if parsed is None:
                continue
            person_id, camera_id = parsed
            if self.filter_distractors and person_id in {0, -1}:
                continue
            samples.append((str(split_dir / filename), person_id, camera_id))

        self.samples = samples
        if self.relabel:
            unique_ids = sorted({person_id for _, person_id, _ in self.samples})
            self.id_to_label = {person_id: label for label, person_id in enumerate(unique_ids)}
            self.num_classes = len(self.id_to_label)
        else:
            self.id_to_label = {}
            self.num_classes = 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, person_id, camera_id = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform is None:
            image = build_eval_transform()(image)
        else:
            image = self.transform(image)
        label = self.id_to_label[person_id] if self.relabel else person_id
        return image, label, camera_id


def build_train_transform():
    return T.Compose([
        T.Resize((REID_INPUT_HEIGHT, REID_INPUT_WIDTH), interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((REID_INPUT_HEIGHT, REID_INPUT_WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        T.RandomErasing(p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.3), value=0),
    ])


def build_eval_transform():
    return T.Compose([
        T.Resize((REID_INPUT_HEIGHT, REID_INPUT_WIDTH), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class RandomIdentitySampler(Sampler):
    def __init__(self, dataset, batch_size, num_instances):
        if batch_size % num_instances != 0:
            raise ValueError("batch_size must be divisible by num_instances")

        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances
        self.index_map = defaultdict(list)

        for index, (_, person_id, _) in enumerate(dataset.samples):
            if getattr(dataset, "relabel", False):
                label = dataset.id_to_label[person_id]
            else:
                label = person_id
            self.index_map[label].append(index)

        self.labels = list(self.index_map.keys())
        self.length = (len(self.labels) // self.num_pids_per_batch) * self.num_pids_per_batch * self.num_instances

    def __iter__(self):
        labels = self.labels[:]
        random.shuffle(labels)

        batch_indices = []
        usable = len(labels) - (len(labels) % self.num_pids_per_batch)
        for start in range(0, usable, self.num_pids_per_batch):
            selected_labels = labels[start:start + self.num_pids_per_batch]
            for label in selected_labels:
                indices = self.index_map[label]
                if len(indices) >= self.num_instances:
                    chosen = random.sample(indices, self.num_instances)
                else:
                    # Pad with random repeats only for the shortage — avoids
                    # sampling an identity as its own positive/negative pair.
                    shortage = self.num_instances - len(indices)
                    chosen = indices + random.choices(indices, k=shortage)
                    random.shuffle(chosen)
                batch_indices.extend(chosen)

        return iter(batch_indices)

    def __len__(self):
        return self.length
