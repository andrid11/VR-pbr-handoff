"""PyTorch dataset for MatSynth PBR materials via HuggingFace streaming."""

import os
import time
import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from PIL import Image

from src.transforms import get_resize_transform, get_train_transform, MAP_NAMES

# Set MATSYNTH_DEBUG=1 to enable per-sample timing logs
_DEBUG = os.environ.get("MATSYNTH_DEBUG", "0") == "1"


class MatSynthDataset(IterableDataset):
    """Streams MatSynth materials and returns 4 PBR maps as tensors.

    Each sample yields a dict:
        - "basecolor":  (3, H, W) float32 tensor [0,1]
        - "normal":     (3, H, W) float32 tensor [0,1]
        - "roughness":  (3, H, W) float32 tensor [0,1]
        - "metallic":   (3, H, W) float32 tensor [0,1]
        - "name":       str
        - "category":   str
    """

    def __init__(
        self,
        split: str = "train",
        size: int = 256,
        max_samples: int | None = None,
        use_augmentation: bool = False,
        seed: int = 42,
    ):
        self.split = split
        self.size = size
        self.max_samples = max_samples
        self.seed = seed
        self.transform = get_train_transform(size) if use_augmentation else get_resize_transform(size)

    def _load_stream(self):
        """Create a fresh streaming dataset (needed for multi-epoch iteration)."""
        ds = load_dataset(
            "gvecchio/MatSynth",
            split=self.split,
            streaming=True,
        )
        # Drop heavy columns we don't need (avoids select_columns downloading everything)
        keep = {"name", "metadata", *MAP_NAMES}
        try:
            drop = [c for c in ds.column_names if c not in keep]
            if drop:
                ds = ds.remove_columns(drop)
        except Exception:
            pass  # column_names may not be available on all streaming configs
        # buffer_size=500 with 4096x4096 images causes OOM (~500 * 4 * 64MB = 128GB)
        # Use small buffer; images are already diverse across the dataset
        ds = ds.shuffle(seed=self.seed, buffer_size=20)
        return ds

    def _process_sample(self, sample: dict) -> dict | None:
        """Convert a HF sample to tensors. Returns None if maps are missing."""
        tensors = {}
        for map_name in MAP_NAMES:
            img = sample.get(map_name)
            if img is None:
                return None
            # HF returns PIL Image; ensure RGB
            if not isinstance(img, Image.Image):
                return None
            img = img.convert("RGB")
            tensors[map_name] = self.transform(img)

        # Extract metadata
        meta = sample.get("metadata", {})
        category = meta.get("category", "unknown")
        if isinstance(category, (list, dict)):
            category = str(category)

        tensors["name"] = sample.get("name", "unknown")
        tensors["category"] = category
        return tensors

    def __iter__(self):
        t0 = time.perf_counter()
        ds = self._load_stream()
        if _DEBUG:
            print(f"[DEBUG] _load_stream: {time.perf_counter() - t0:.2f}s")
        count = 0
        for sample in ds:
            t1 = time.perf_counter()
            if _DEBUG and count == 0:
                print(f"[DEBUG] first sample from stream: {t1 - t0:.2f}s")
            result = self._process_sample(sample)
            if _DEBUG:
                print(f"[DEBUG] _process_sample #{count}: {time.perf_counter() - t1:.3f}s"
                      f"  (result={'ok' if result else 'None'})")
            if result is None:
                continue
            yield result
            count += 1
            if self.max_samples is not None and count >= self.max_samples:
                break


class CachedMatSynthDataset(torch.utils.data.Dataset):
    """Reads pre-downloaded .pt samples from disk. Supports random access and shuffling."""

    def __init__(self, cache_dir: str, use_augmentation: bool = False, size: int = 256):
        self.cache_dir = cache_dir
        self.files = sorted(f for f in os.listdir(cache_dir) if f.endswith(".pt"))
        if not self.files:
            raise FileNotFoundError(f"No .pt files in {cache_dir}")
        self.augment = None
        if use_augmentation:
            from src.transforms import PBRAugmentation
            self.augment = PBRAugmentation()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.cache_dir, self.files[idx])
        sample = torch.load(path, weights_only=False)
        if self.augment is not None:
            sample = self.augment(sample)
        return sample


def create_dataloader(
    split: str = "train",
    size: int = 256,
    batch_size: int = 4,
    max_samples: int | None = None,
    use_augmentation: bool = False,
    num_workers: int = 0,
    cache_dir: str | None = None,
) -> DataLoader:
    """Create a DataLoader for MatSynth PBR materials.

    Args:
        cache_dir: Path to pre-downloaded .pt files (from predownload.py).
                   If provided, loads from disk instead of streaming.
                   Default path: data/processed/{split}_{size}
    """
    if cache_dir is not None:
        dataset = CachedMatSynthDataset(cache_dir, use_augmentation=use_augmentation, size=size)
    else:
        dataset = MatSynthDataset(
            split=split,
            size=size,
            max_samples=max_samples,
            use_augmentation=use_augmentation,
        )

    def collate_fn(batch):
        """Custom collate to handle mixed tensor/string fields."""
        result = {}
        for key in MAP_NAMES:
            stacked = torch.stack([b[key] for b in batch])
            if key in ("roughness", "metallic"):
                stacked = stacked[:, :1, :, :]
            result[key] = stacked
        result["name"] = [b["name"] for b in batch]
        result["category"] = [b["category"] for b in batch]
        return result

    is_map_style = isinstance(dataset, CachedMatSynthDataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        shuffle=is_map_style,
    )
