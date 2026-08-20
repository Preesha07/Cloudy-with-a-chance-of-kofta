"""
Lazy per-patch dataset built from three per-band scaled JPEG directories.

Slicing, padding, and band-merging happen in __init__ (one-time, in memory).
Each __getitem__ returns the patch tensor plus its (y, x) position for canvas accumulation.
"""
from __future__ import annotations

import re
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

_NORM = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def _read_gray(path: Path) -> np.ndarray:
    """Read a single-channel JPEG as a uint8 numpy array [H, W]."""
    with Image.open(path) as img:
        return np.array(img.convert("L"))


def merge_bands(b2_path: Path, b3_path: Path, b4_path: Path) -> np.ndarray:
    """Merge three per-band grayscale images into a single [H, W, 3] uint8 array.

    Channel order: R=BAND3 (red), G=BAND2 (green), B=BAND4 (NIR).
    This matches the model's training convention.
    """
    r = _read_gray(b3_path)
    g = _read_gray(b2_path)
    b = _read_gray(b4_path)
    return np.stack([r, g, b], axis=-1)


class LazyPatchDataset(Dataset):
    """Slice a merged 3-band scene image into 256×256 patches, served lazily.

    The full merged image is loaded into RAM once in __init__.
    Each __getitem__ returns (tensor [3,256,256], y_pos, x_pos).
    y_pos and x_pos are the patch's top-left corner in the *padded* image —
    these are also the coordinates used when accumulating into the stitch canvas.
    """

    def __init__(
        self,
        band2_dir: Path,
        band3_dir: Path,
        band4_dir: Path,
        patch_size: int = 256,
        stride: int = 128,
        stretch_input: bool = False,
        scene_transform=None,
    ):
        # Discover the scene file (only one per band dir after processing)
        b2_files = sorted(Path(band2_dir).glob("*.jpg"))
        if not b2_files:
            raise FileNotFoundError(f"No .jpg files in {band2_dir}")
        b2_path = b2_files[0]
        b3_path = Path(band3_dir) / b2_path.name
        b4_path = Path(band4_dir) / b2_path.name

        arr = merge_bands(b2_path, b3_path, b4_path)  # [H, W, 3] uint8
        h, w = arr.shape[:2]
        self.orig_h = h
        self.orig_w = w

        if stretch_input:
            # Per-channel 2-98 percentile stretch over the whole scene so the
            # model always receives input in the [0, 255] range it was trained on.
            # This is input normalisation, not output post-processing.
            arr = arr.astype(np.float32)
            for c in range(arr.shape[2]):
                ch = arr[:, :, c]
                lo, hi = float(np.percentile(ch, 2)), float(np.percentile(ch, 98))
                if hi > lo:
                    arr[:, :, c] = np.clip((ch - lo) / (hi - lo) * 255.0, 0, 255)
            arr = arr.astype(np.uint8)

        if scene_transform is not None:
            # Adapter-provided callable: [H, W, 3] uint8 → [H, W, 3] uint8.
            # Applied after stretch_input so the two can compose (though most
            # adapters set stretch_input=False and do everything here).
            arr = scene_transform(arr)

        pad_h = (stride - (h - patch_size) % stride) % stride
        pad_w = (stride - (w - patch_size) % stride) % stride
        self._arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

        self._patch_size = patch_size
        self._stride = stride
        self._positions = [
            (y, x)
            for y in range(0, self._arr.shape[0] - patch_size + 1, stride)
            for x in range(0, self._arr.shape[1] - patch_size + 1, stride)
        ]

    def __len__(self) -> int:
        return len(self._positions)

    def __getitem__(self, idx: int):
        y, x = self._positions[idx]
        p = self._patch_size
        patch = self._arr[y : y + p, x : x + p]  # [256, 256, 3] uint8
        tensor = T.functional.to_tensor(patch)     # [3, 256, 256] float [0,1]
        tensor = _NORM(tensor)                     # [3, 256, 256] float [-1,1]
        return tensor, y, x


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """Convert a [-1, 1] float tensor [B, C, H, W] → uint8 numpy [B, H, W, C]."""
    arr = t.clamp(-1.0, 1.0).add(1.0).mul(127.5).byte().cpu().numpy()
    return arr.transpose(0, 2, 3, 1)  # [B, H, W, C]


def attn_to_uint8(t: torch.Tensor) -> np.ndarray:
    """Convert an attention map [B, 1, H, W] in [0,1] → uint8 [B, H, W, 1]."""
    arr = t.clamp(0.0, 1.0).mul(255.0).byte().cpu().numpy()
    return arr.transpose(0, 2, 3, 1)  # [B, H, W, 1]


def tensor_to_hwc(t: torch.Tensor) -> torch.Tensor:
    """[-1,1] float [B,C,H,W] → 0-255 float [B,H,W,C], staying on its device.

    The device-resident counterpart of tensor_to_uint8, for feeding Stitcher
    without a host round-trip. Stays float rather than quantising to uint8 —
    the values are about to be Hann-weighted and summed anyway.
    """
    return t.clamp(-1.0, 1.0).add(1.0).mul(127.5).permute(0, 2, 3, 1)


def attn_to_hwc(t: torch.Tensor) -> torch.Tensor:
    """[0,1] float [B,1,H,W] → 0-255 float [B,H,W,1], staying on its device."""
    return t.clamp(0.0, 1.0).mul(255.0).permute(0, 2, 3, 1)
