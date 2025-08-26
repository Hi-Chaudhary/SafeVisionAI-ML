# src/data_loader.py
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video

# ---- Utilities ----

CLASS_TO_LABEL = {
    "fight": 1,
    "non-fight": 0,
}

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def _list_videos(root: Path) -> List[Tuple[Path, int]]:
    """
    Expects a folder structure:
      root/
        fight/*.mp4
        non-fight/*.mp4
    Returns a list of (path, label).
    """
    items: List[Tuple[Path, int]] = []
    for cls_name, label in CLASS_TO_LABEL.items():
        cls_dir = root / cls_name
        if not cls_dir.exists():
            # quietly skip missing class folders to be flexible
            continue
        for p in cls_dir.glob("*"):
            if p.suffix.lower() in VIDEO_EXTS:
                items.append((p, label))
    return items


def _evenly_spaced_indices(total: int, num: int) -> torch.Tensor:
    """
    Pick 'num' indices evenly from [0, total-1].
    If total < num, we repeat the last frame.
    """
    if total <= 0:
        return torch.zeros(num, dtype=torch.long)
    if total >= num:
        # linspace inclusive of both ends
        idx = torch.linspace(0, total - 1, steps=num).round().long()
    else:
        # take all frames, then pad with last index
        base = torch.arange(0, total, dtype=torch.long)
        pad = base[-1].repeat(num - total)
        idx = torch.cat([base, pad], dim=0)
    return idx


def _resize_video_tensor(frames: torch.Tensor, size: int = 112) -> torch.Tensor:
    """
    frames: [T, C, H, W] uint8 or float
    Returns: [T, C, size, size] float in [0,1]
    """
    if frames.dtype != torch.float32:
        frames = frames.float()  # [0..255] -> float
    frames = frames / 255.0
    # interpolate expects [N, C, H, W]
    frames = F.interpolate(
        frames.permute(1, 0, 2, 3),  # [C, T, H, W]
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    ).permute(1, 0, 2, 3)  # back to [T, C, H, W]
    return frames


# ---- Dataset ----

class RWF2000VideoDataset(Dataset):
    """
    Loads short video clips and returns 16 frames as [C, T, H, W] and label int {0,1}.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        clip_len: int = 16,
        resize: int = 112,
        normalize: bool = True,
        augment: bool = False,
    ):
        """
        root_dir: path like 'data/video/rwf2000'
        split: 'train' or 'val' (or 'test' if you have it)
        """
        self.root = Path(root_dir) / split
        self.clip_len = clip_len
        self.resize_to = resize
        self.normalize = normalize
        self.augment = augment

        self.items = _list_videos(self.root)
        if len(self.items) == 0:
            raise FileNotFoundError(
                f"No videos found under {self.root}. "
                f"Expected folders 'fight/' and 'non-fight/' with video files."
            )

        # Normalization stats (Kinetics/ImageNet-ish for videos)
        self.mean = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]

        # torchvision.io.read_video -> video: [T, H, W, C], uint8
        # Some videos might be variable length; we only need frames.
        video, _, info = read_video(str(path), pts_unit="sec")

        if video.numel() == 0:
            # Edge case: unreadable/corrupted file -> return zeros
            frames = torch.zeros(self.clip_len, 3, self.resize_to, self.resize_to)
        else:
            # Convert to [T, C, H, W]
            video = video.permute(0, 3, 1, 2)  # [T, C, H, W]
            T = video.shape[0]
            indices = _evenly_spaced_indices(T, self.clip_len)
            frames = video[indices]  # [clip_len, C, H, W]
            frames = _resize_video_tensor(frames, self.resize_to)  # [T, C, 112, 112]

            # Simple augmentation: (only on train) random horizontal flip
            if self.augment and torch.rand(1).item() < 0.5:
                frames = torch.flip(frames, dims=[3])  # flip width

            # Normalize per-channel
            if self.normalize:
                frames = (frames - self.mean) / self.std

        # Model-friendly shape: [C, T, H, W]
        frames = frames.permute(1, 0, 2, 3).contiguous()

        return frames, torch.tensor(label, dtype=torch.long)


# ---- Dataloaders ----

def make_rwf2000_loaders(
    root_dir: str = "data/video/rwf2000",
    batch_size: int = 8,
    num_workers: int = 2,
    clip_len: int = 16,
    resize: int = 112,
) -> Dict[str, DataLoader]:
    """
    Convenience factory for train/val dataloaders.
    """
    train_ds = RWF2000VideoDataset(
        root_dir, split="train", clip_len=clip_len, resize=resize, augment=True
    )
    val_ds = RWF2000VideoDataset(
        root_dir, split="val", clip_len=clip_len, resize=resize, augment=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return {"train": train_loader, "val": val_loader}
