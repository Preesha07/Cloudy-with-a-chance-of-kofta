"""Filter LISS-4 patches by cloud fraction before CORAL training.

For each 256×256 patch, the frozen netA attention module produces a cloud
probability map in [0, 1] (1 = cloud). Patches whose mean cloud probability
exceeds `cloud_threshold` are discarded so CORAL never aligns cloudy LISS-4
features with clear Sentinel-2 features.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

Image.MAX_IMAGE_PIXELS = None

_TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


class _PatchDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        path = self.paths[idx]
        try:
            img = Image.open(path).convert('RGB')
            return _TRANSFORM(img), str(path)
        except Exception:
            return torch.zeros(3, 256, 256), str(path)


def find_liss4_patches(liss4_root: Path | str) -> list[Path]:
    """Recursively collect all JPEG image paths under liss4_root."""
    root = Path(liss4_root)
    paths = sorted(
        p for p in root.rglob('*.jpg')
        if p.is_file() and not p.name.startswith('.')
    )
    print(f"[patch_filter] Found {len(paths)} LISS-4 patches in {root}")
    return paths


def filter_liss4_patches(
    paths: list[Path],
    netA: nn.Module,
    device: torch.device,
    cloud_threshold: float = 0.30,
    batch_size: int = 64,
    num_workers: int = 4,
) -> list[Path]:
    """Return paths of LISS-4 patches with cloud fraction below threshold.

    Args:
        paths:           All candidate LISS-4 patch paths.
        netA:            Frozen attention module (produces cloud probability maps).
        device:          Device to run netA on.
        cloud_threshold: Discard patches with mean(att_A) >= this value.
        batch_size:      Inference batch size.
        num_workers:     DataLoader workers.

    Returns:
        Filtered list of paths where mean cloud probability < cloud_threshold.
    """
    if not paths:
        return []

    dataset = _PatchDataset(paths)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )

    clear_paths: list[Path] = []
    netA.eval()

    with torch.no_grad():
        for tensors, path_strs in loader:
            tensors = tensors.to(device)
            att = netA(tensors)                           # (B, 1, H, W) in [0, 1]
            cloud_fracs = att.mean(dim=(1, 2, 3))        # (B,)
            for cf, p in zip(cloud_fracs.tolist(), path_strs):
                if cf < cloud_threshold:
                    clear_paths.append(Path(p))

    kept = len(clear_paths)
    total = len(paths)
    print(f"[patch_filter] Kept {kept}/{total} patches "
          f"({100 * kept / max(total, 1):.1f}%) "
          f"with cloud fraction < {cloud_threshold:.0%}")
    return clear_paths
