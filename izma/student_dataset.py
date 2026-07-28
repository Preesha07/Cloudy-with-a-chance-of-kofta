"""
Paired dataset for Phase-2 student GAN training.

Provides matched (cloudy optical, SAR) tile pairs drawn from the raw SEN12MS-CR
TIF archives.  Both modalities are normalised to [-1, 1], matching the range the
teacher model (AttentionGAN-for-Cloud-removal) was trained on.

    cloudy_optical : (3, 256, 256) float32  — G/R/NIR bands (S2 B3,B4,B8)
                     clip [0, 10 000] → [0,1] → [-1,1]
    sar            : (3, 256, 256) float32  — VV, VH, VV-VH (S1 2-band dB TIF)
                     VV/VH: clip [-25, 0] dB → [0,1] → [-1,1]
                     VV-VH: clip [0,  25] dB → [0,1] → [-1,1]

Directory layout expected under `sen12mscr_root`:
    ROIs1158_spring_s2_cloudy/ROIs1158_spring_s2_cloudy/s2_cloudy_{roi}/
        ROIs1158_spring_s2_cloudy_{roi}_p{patch}.tif   (13-band S2, uint16)
    ROIs1158_spring_s1/ROIs1158_spring_s1/s1_{roi}/
        ROIs1158_spring_s1_{roi}_p{patch}.tif           (2-band S1, float32 dB)

Both trees must be present under the same root (the layout produced by
download_sen12mscrts_roi.py / the SEN12MSCR download).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset

# S2 bands to extract (0-based index into the 13-band S2 stack)
# R = B4 (665 nm) → idx 3,  G = B3 (560 nm) → idx 2,  NIR = B8 (842 nm) → idx 7
LISS4_BANDS = (3, 2, 7)   # rasterio reads as band_number = idx + 1
S2_CLIP = 10_000.0

# SAR dB clipping ranges matching prepare_teacher_data.py / SEN12MS-CR-TS process_SAR
SAR_VV_VH_MIN = -25.0
SAR_VV_VH_MAX =  0.0
SAR_DIFF_MAX  = 25.0       # VV-VH difference always ≥ 0 for natural surfaces


def _to_minus1_1(x: np.ndarray) -> np.ndarray:
    """[0, 1] → [-1, 1]."""
    return x * 2.0 - 1.0


def _read_optical(path: Path) -> torch.Tensor:
    """Read S2 cloudy TIF and return (3, H, W) float32 in [-1, 1]."""
    band_nums = [i + 1 for i in LISS4_BANDS]   # rasterio is 1-based
    with rasterio.open(path) as src:
        arr = src.read(band_nums).astype(np.float32)   # (3, H, W) uint16 values
    arr = np.clip(arr, 0.0, S2_CLIP) / S2_CLIP        # [0, 1]
    arr = _to_minus1_1(arr)                             # [-1, 1]
    return torch.from_numpy(arr)


def _read_sar(path: Path) -> torch.Tensor:
    """Read S1 2-band dB TIF and return (3, H, W) float32 in [-1, 1].

    Channel order: R=VV, G=VH, B=VV-VH  — matches prepare_teacher_data.py.
    """
    with rasterio.open(path) as src:
        vv = src.read(1).astype(np.float32)   # band 1 = VV
        vh = src.read(2).astype(np.float32)   # band 2 = VH

    def clip_norm(band, lo, hi):
        return (np.clip(band, lo, hi) - lo) / (hi - lo)

    vv_n  = clip_norm(vv, SAR_VV_VH_MIN, SAR_VV_VH_MAX)           # [0, 1]
    vh_n  = clip_norm(vh, SAR_VV_VH_MIN, SAR_VV_VH_MAX)           # [0, 1]
    diff_n = clip_norm(vv - vh, 0.0, SAR_DIFF_MAX)                 # [0, 1]

    arr = np.stack([vv_n, vh_n, diff_n], axis=0)                   # (3, H, W)
    arr = _to_minus1_1(arr)                                         # [-1, 1]
    return torch.from_numpy(arr)


def _patch_id(path: Path) -> tuple[str, str]:
    """Extract (roi, patch) from filenames like ROIs1158_spring_s?_{roi}_p{patch}.tif."""
    m = re.search(r'_(\d+)_(p\d+)\.tif$', path.name)
    if m is None:
        raise ValueError(f"Unexpected filename format: {path.name}")
    return m.group(1), m.group(2)


class StudentDataset(Dataset):
    """Paired (cloudy optical, SAR) tiles for student GAN training."""

    def __init__(
        self,
        sen12mscr_root: str | Path,
        augment: bool = False,
        split: str = "train",
        val_fraction: float = 0.05,
        split_seed: int = 0,
    ) -> None:
        root = Path(sen12mscr_root)
        # S2 cloudy: flat layout  → <root>/ROIs1158_spring_s2_cloudy/s2_cloudy_{roi}/
        # S1 SAR:   nested layout → <root>/ROIs1158_spring_s1/ROIs1158_spring_s1/s1_{roi}/
        s2c_root = root / "ROIs1158_spring_s2_cloudy"
        s1_root  = root / "ROIs1158_spring_s1" / "ROIs1158_spring_s1"

        if not s2c_root.is_dir():
            raise FileNotFoundError(f"S2 cloudy dir not found: {s2c_root}")
        if not s1_root.is_dir():
            raise FileNotFoundError(f"S1 SAR dir not found: {s1_root}")

        # Index all S1 files by (roi, patch) → path
        s1_index: dict[tuple[str, str], Path] = {}
        for f in s1_root.rglob("*.tif"):
            s1_index[_patch_id(f)] = f

        # Collect all S2 cloudy files that have a matching S1 patch
        pairs: list[tuple[Path, Path]] = []
        for f in sorted(s2c_root.rglob("*.tif")):
            key = _patch_id(f)
            if key in s1_index:
                pairs.append((f, s1_index[key]))

        if not pairs:
            raise RuntimeError("No matched (S2-cloudy, S1) pairs found under "
                               f"{sen12mscr_root}. Check directory layout.")

        # Deterministic train/val split
        if split not in ("all", "train", "val"):
            raise ValueError(f"split must be all/train/val, got {split!r}")
        if split != "all":
            rng = torch.Generator().manual_seed(split_seed)
            order = torch.randperm(len(pairs), generator=rng).tolist()
            n_val = max(1, int(round(len(pairs) * val_fraction)))
            keep = order[:n_val] if split == "val" else order[n_val:]
            pairs = [pairs[i] for i in sorted(keep)]

        self.pairs = pairs
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s2_path, s1_path = self.pairs[idx]
        optical = _read_optical(s2_path)   # (3, 256, 256) in [-1, 1]
        sar     = _read_sar(s1_path)       # (3, 256, 256) in [-1, 1]

        if self.augment:
            if torch.rand(1).item() < 0.5:
                optical = torch.flip(optical, dims=[-1])
                sar     = torch.flip(sar,     dims=[-1])
            if torch.rand(1).item() < 0.5:
                optical = torch.flip(optical, dims=[-2])
                sar     = torch.flip(sar,     dims=[-2])
            k = int(torch.randint(4, (1,)).item())
            if k:
                optical = torch.rot90(optical, k=k, dims=[-2, -1])
                sar     = torch.rot90(sar,     k=k, dims=[-2, -1])
            optical = optical.contiguous()
            sar     = sar.contiguous()

        return {"cloudy_optical": optical, "sar": sar}


def build_student_loader(
    sen12mscr_root: str | Path,
    split: str = "train",
    batch_size: int = 1,
    num_workers: int = 4,
    augment: bool = True,
    val_fraction: float = 0.05,
) -> DataLoader:
    dataset = StudentDataset(
        sen12mscr_root,
        augment=(augment and split == "train"),
        split=split,
        val_fraction=val_fraction,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )
