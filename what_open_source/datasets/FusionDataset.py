import os
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


def _letter_to_label(name: str) -> int:
    """Map the first letter of a folder name (A-Z) to class id 0-25."""
    if not name:
        raise ValueError("Empty folder name for class.")
    c = name[0].upper()
    if c < "A" or c > "Z":
        raise ValueError(f"Unexpected class letter '{c}', expected A-Z.")
    return ord(c) - ord("A")


def _load_image(path: str) -> Image.Image:
    """Load an RGB PIL image for downstream transforms."""
    return Image.open(path).convert("RGB")


def _load_depth(path: str) -> torch.Tensor:
    arr = np.load(path)
    # Normalize to shape (1, H, W)
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 3:
        if arr.shape[-1] == 1:
            arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr.astype(np.float32))


def _load_points(path: str) -> torch.Tensor:
    """Load and normalize a point cloud (ModelNet-style).

    The txt file has header "X Y Z" followed by 1024 coordinate rows.
    """
    try:
        pts = np.loadtxt(path, dtype=np.float32, skiprows=1)
    except Exception as e:
        raise RuntimeError(f"Failed to load point cloud from {path}: {e}")

    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"Unexpected point shape {pts.shape} in {path}")

    pts = pts[:, :3]

    centroid = np.mean(pts, axis=0)
    pts = pts - centroid
    m = np.max(np.sqrt(np.sum(pts ** 2, axis=1)))
    if m > 0:
        pts = pts / m

    return torch.from_numpy(pts.astype(np.float32))


class FusionDataset(Dataset):
    """Multi-modal dataset loader for RGB, depth, and point cloud views.

    Expected layout per instance folder:
        camera{0,2,4}/{sid}.png, depth{0,2,4}/{sid}.npy, point{0,2,4}/{sid}.txt

    Train/test splits are defined by separate root directories (``data/train`` vs ``data/test``).
    """

    def __init__(
        self,
        root: str,
        subset: str = "train",
        img_transform: Optional[Callable] = None,
        depth_transform: Optional[Callable] = None,
        point_transform: Optional[Callable] = None,
        shuffle: bool = True,
        seed: int = 42,
        classes: Optional[List[str]] = None,
        config=None,
    ) -> None:
        super().__init__()

        # Legacy PointMamba-style config object (optional).
        if config is not None:
            root = getattr(config, "ROOT", root)
            subset = getattr(config, "subset", subset)

        if root is None:
            raise ValueError("FusionDataset: root must be provided.")

        self.root = root
        self.subset = subset
        self.img_transform = img_transform
        self.depth_transform = depth_transform
        self.point_transform = point_transform
        self.shuffle = shuffle
        self.seed = seed

        # Remap letter classes to contiguous labels, e.g. ["A","C","D"] -> {0,1,2}
        self.class_label_map = None
        if classes is not None:
            ordered_letters = []
            seen = set()
            for c in classes:
                if not c:
                    continue
                ch = c[0].upper()
                if ch < "A" or ch > "Z":
                    continue
                if ch in seen:
                    continue
                seen.add(ch)
                ordered_letters.append(ch)

            self.class_label_map = {}
            for new_idx, ch in enumerate(ordered_letters):
                old_label = ord(ch) - ord("A")
                self.class_label_map[old_label] = new_idx

        self.samples: List[Dict] = []
        self._build_index()

    def _build_index(self) -> None:
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Root not found: {self.root}")

        all_instances: List[Dict] = []

        for inst_name in sorted(os.listdir(self.root)):
            inst_dir = os.path.join(self.root, inst_name)
            if not os.path.isdir(inst_dir):
                continue
            try:
                label = _letter_to_label(inst_name)
            except ValueError:
                continue

            if self.class_label_map is not None:
                if label not in self.class_label_map:
                    continue
                mapped_label = self.class_label_map[label]
            else:
                mapped_label = label

            cam0_dir = os.path.join(inst_dir, "camera0")
            if not os.path.isdir(cam0_dir):
                continue

            sids = []
            for fname in sorted(os.listdir(cam0_dir)):
                lower = fname.lower()
                if not lower.endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    continue
                sid, ext = os.path.splitext(fname)
                sids.append((sid, ext))

            for sid, ext in sids:
                paths = {
                    "camera0": os.path.join(inst_dir, "camera0", f"{sid}{ext}"),
                    "camera2": os.path.join(inst_dir, "camera2", f"{sid}{ext}"),
                    "camera4": os.path.join(inst_dir, "camera4", f"{sid}{ext}"),
                    "depth0": os.path.join(inst_dir, "depth0", f"{sid}.npy"),
                    "depth2": os.path.join(inst_dir, "depth2", f"{sid}.npy"),
                    "depth4": os.path.join(inst_dir, "depth4", f"{sid}.npy"),
                    "point0": os.path.join(inst_dir, "point0", f"{sid}.txt"),
                    "point2": os.path.join(inst_dir, "point2", f"{sid}.txt"),
                    "point4": os.path.join(inst_dir, "point4", f"{sid}.txt"),
                }

                if not all(os.path.isfile(p) for p in paths.values()):
                    continue

                all_instances.append({
                    "id": f"{inst_name}_{sid}",
                    "paths": paths,
                    "label": mapped_label,
                })

        if not all_instances:
            raise RuntimeError(f"No valid samples found in {self.root}")

        all_instances.sort(key=lambda x: x["id"])
        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(all_instances)

        # Train/test are separate roots; subset is metadata only.
        if self.subset not in ("train", "test", "all"):
            raise ValueError(f"Unknown subset: {self.subset}. Use train, test, or all.")
        self.samples = all_instances

        print(f"Dataset loaded: {self.subset} split with {len(self.samples)} samples.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], int]:
        info = self.samples[index]
        paths = info["paths"]
        label = info["label"]
        sample: Dict[str, torch.Tensor] = {}

        try:
            for k in ["camera0", "camera2", "camera4"]:
                img = _load_image(paths[k])
                if self.img_transform is not None:
                    img = self.img_transform(img)
                else:
                    img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                sample[k] = img

            for k in ["depth0", "depth2", "depth4"]:
                depth = _load_depth(paths[k])
                if self.depth_transform is not None:
                    depth = self.depth_transform(depth)
                sample[k] = depth

            for k in ["point0", "point2", "point4"]:
                pts = _load_points(paths[k])
                if self.point_transform is not None:
                    pts = self.point_transform(pts)
                sample[k] = pts

            return sample, label

        except Exception as e:
            print(f"Error loading sample {info['id']}: {e}")
            return self.__getitem__((index + 1) % len(self))
