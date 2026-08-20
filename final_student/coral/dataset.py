"""Dataset for CORAL domain adaptation fine-tuning.

Each batch item contains:
  - A matched SEN12MS-CR (cloudy / clear) pair — the source domain.
  - A randomly sampled pre-filtered LISS-4 patch — the target domain.

LISS-4 paths must already be filtered before constructing this dataset.
Use patch_filter.filter_liss4_patches() to produce the filtered list.
"""
from __future__ import annotations

import random
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

Image.MAX_IMAGE_PIXELS = None

_SENTINEL_TRANSFORM = T.Compose([
    T.Resize((256, 256), Image.BICUBIC),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

_LISS4_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

_VALID_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}


class CoralDataset(Dataset):
    """Source (SEN12MS-CR) + target (pre-filtered LISS-4 patches) dataset.

    Args:
        sentinel_root:  Root of the SEN12MS-CR prepared data (contains trainA/trainB).
        liss4_paths:    Pre-filtered LISS-4 patch paths (cloud fraction < threshold).
        fine_size:      Patch size expected by the model (default 256).
    """

    def __init__(
        self,
        sentinel_root: Path | str,
        liss4_paths: list[Path],
        fine_size: int = 256,
    ) -> None:
        sentinel_root = Path(sentinel_root)
        self.fine_size = fine_size

        dir_A = (sentinel_root / 'trainA'
                 if (sentinel_root / 'trainA').exists()
                 else sentinel_root / 'A')
        dir_B = (sentinel_root / 'trainB'
                 if (sentinel_root / 'trainB').exists()
                 else sentinel_root / 'B')

        self.paths_A = sorted(p for p in dir_A.glob('*') if p.suffix.lower() in _VALID_EXTS)
        self.paths_B = sorted(p for p in dir_B.glob('*') if p.suffix.lower() in _VALID_EXTS)
        self.liss4_paths = list(liss4_paths)

        if not self.paths_A:
            raise FileNotFoundError(f"No images found in {dir_A}")
        if not self.liss4_paths:
            raise ValueError("liss4_paths is empty — run filter_liss4_patches first")

    def _load_sentinel(self, path: Path) -> torch.Tensor:
        return _SENTINEL_TRANSFORM(Image.open(path).convert('RGB'))

    def _load_liss4(self, index: int) -> torch.Tensor:
        """Random-crop a LISS-4 patch to fine_size, with a retry loop for corrupt files."""
        for offset in range(10):
            path = self.liss4_paths[(index + offset) % len(self.liss4_paths)]
            try:
                img = Image.open(path).convert('RGB')
                w, h = img.size
                if w < self.fine_size or h < self.fine_size:
                    img = img.resize(
                        (max(w, self.fine_size), max(h, self.fine_size)),
                        Image.BICUBIC,
                    )
                    w, h = img.size
                x = random.randint(0, w - self.fine_size)
                y = random.randint(0, h - self.fine_size)
                img = img.crop((x, y, x + self.fine_size, y + self.fine_size))
                return _LISS4_TRANSFORM(img)
            except Exception:
                continue
        return torch.zeros(3, self.fine_size, self.fine_size)

    def __getitem__(self, index: int) -> dict:
        tensor_A = self._load_sentinel(self.paths_A[index])
        tensor_B = self._load_sentinel(self.paths_B[index])
        liss4    = self._load_liss4(index)

        return {
            'A':       tensor_A,
            'B':       tensor_B,
            'C':       tensor_A,            # SAR placeholder (unused during adaptation)
            'A_paths': str(self.paths_A[index]),
            'B_paths': str(self.paths_B[index]),
            'C_paths': str(self.paths_A[index]),
            'LISS4':   liss4,
        }

    def __len__(self) -> int:
        return len(self.paths_A)
