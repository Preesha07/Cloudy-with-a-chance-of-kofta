"""SEN12MS-CR input pipeline for smoke-testing the ViT MAE backbone.

The backbone is destined for LISS-IV (3 bands: R/G/NIR). To keep the
pretraining domain aligned, we read only the 3 Sentinel-2 bands that match
LISS-IV, in R/G/NIR order, from the cloud-free tiles.

    LISS-IV band   Sentinel-2   index in 13-band stack (0-based)
    Red            B4 (665nm)   3
    Green          B3 (560nm)   2
    NIR            B8 (842nm)   7
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

S2_CLIP = 10_000.0
LISS4_BAND_INDICES = (3, 2, 7)  # R=B4, G=B3, NIR=B8


class SEN12MSCROptical(Dataset):
    """3-band (R/G/NIR) Sentinel-2 cloud-free tiles from SEN12MS-CR."""

    def __init__(
        self,
        root: str | Path,
        pattern: str = "**/*.tif",
        img_size: int = 256,
        band_indices: tuple[int, ...] = LISS4_BAND_INDICES,
    ) -> None:
        self.root = Path(root)
        self.img_size = img_size
        self.band_indices = band_indices
        self.files = sorted(self.root.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"no tiles matched {pattern!r} under {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        with rasterio.open(self.files[idx]) as src:
            bands = [i + 1 for i in self.band_indices]  # rasterio is 1-based
            arr = src.read(bands)  # (3, H, W), uint16, R/G/NIR order

        arr = np.clip(arr.astype(np.float32), 0.0, S2_CLIP) / S2_CLIP
        tile = torch.from_numpy(arr)
        if tile.shape[-1] != self.img_size or tile.shape[-2] != self.img_size:
            raise ValueError(
                f"{self.files[idx].name}: got {tuple(tile.shape[-2:])}, "
                f"expected {self.img_size}x{self.img_size}"
            )
        return tile  # (3, 256, 256)