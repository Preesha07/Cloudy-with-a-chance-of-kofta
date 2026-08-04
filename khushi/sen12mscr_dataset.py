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

import json
import math
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

S2_CLIP = 10_000.0
LISS4_BAND_INDICES = (3, 2, 7)  # R=B4, G=B3, NIR=B8

# Per-channel statistics of reflectance/S2_CLIP over 800 random tiles of
# ROIs1158_spring_s2 (52.4M pixels/channel). Land cover occupies a narrow slice
# of [0, 1] — without standardizing, the patch embedding sees inputs with a mean
# of ~0.15 and a std of ~0.1, which is badly conditioned for the first Linear.
CHANNEL_MEAN = (0.1014, 0.1111, 0.2427)  # R, G, NIR
CHANNEL_STD = (0.1079, 0.0914, 0.1086)


def denormalize(tile: torch.Tensor) -> torch.Tensor:
    """Undo `normalize=True` standardization -> reflectance in [0, 1].

    Accepts (C, H, W) or (B, C, H, W). Needed by anything that reports metrics
    or renders images in physical units rather than standardized units.
    """
    mean = torch.tensor(CHANNEL_MEAN, dtype=tile.dtype, device=tile.device)
    std = torch.tensor(CHANNEL_STD, dtype=tile.dtype, device=tile.device)
    shape = (1, -1, 1, 1) if tile.ndim == 4 else (-1, 1, 1)
    return tile * std.reshape(shape) + mean.reshape(shape)


def tile_dynamic_range(arr: np.ndarray) -> float:
    """Largest per-channel 98th-2nd percentile span of a (C, H, W) tile.

    The measure of "is there any spatial structure here". Computed per channel
    and maxed, NOT pooled over channels: pooling would also count the constant
    brightness offset between R/G/NIR, so a flat tile whose three channels sit
    at different levels scores as though it had content.
    """
    lo = np.percentile(arr, 2, axis=(-2, -1))
    hi = np.percentile(arr, 98, axis=(-2, -1))
    return float(np.max(hi - lo))


class SEN12MSCROptical(Dataset):
    """3-band (R/G/NIR) Sentinel-2 cloud-free tiles from SEN12MS-CR."""

    def __init__(
        self,
        root: str | Path,
        pattern: str = "**/*.tif",
        img_size: int = 256,
        band_indices: tuple[int, ...] = LISS4_BAND_INDICES,
        augment: bool = False,
        photometric: float = 0.0,
        rrc_scale: tuple[float, float] | None = None,
        rrc_ratio: tuple[float, float] = (3 / 4, 4 / 3),
        normalize: bool = True,
        split: str = "all",
        val_fraction: float = 0.05,
        split_seed: int = 0,
        min_dynamic_range: float = 0.0,
        stats_cache: str | Path | None = None,
        verbose: bool = True,
    ) -> None:
        self.root = Path(root)
        self.img_size = img_size
        self.band_indices = band_indices
        # Flip/rot90 augmentation. Satellite tiles have no canonical orientation,
        # so these are label-preserving and cheaply multiply an otherwise small
        # dataset (~8x via the dihedral group) — valuable for MAE pretraining.
        self.augment = augment
        # Radiometric jitter strength (0 = off). Flips and rotations leave a
        # tile's radiometry untouched, so without this the encoder only ever
        # sees Sentinel-2's exact calibration — brittle when the student is
        # handed LISS-4, a different instrument with different gains and band
        # responses. SatDINO's ablation puts augmentation at the single largest
        # effect they measured (kNN 68.89 -> 63.35 with augmentation removed),
        # while intensity barely mattered, so a moderate default is enough.
        self.photometric = photometric
        # Random-resized-crop: (min, max) fraction of tile AREA to crop, then
        # resize back to img_size. This is scale/GSD augmentation — LISS-4 is
        # 5.8m and Sentinel-2 10m, so the encoder should not assume one scale.
        # Crops use a non-square aspect ratio and are squashed back to square;
        # SatDINO found that distortion itself worth ~3 kNN points over square
        # crops (68.98 vs 65.77).
        self.rrc_scale = rrc_scale
        self.rrc_ratio = rrc_ratio
        # Per-channel standardization to zero mean / unit variance. Off means
        # raw reflectance in [0, 1] (the pre-2026-07 behavior).
        self.normalize = normalize
        self.files = sorted(self.root.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"no tiles matched {pattern!r} under {self.root}")

        # Drop near-blank tiles (deep water, shadow). Their content is a fraction
        # of a percent of reflectance, so per-patch normalization amplifies pure
        # sensor artifact to unit variance: the model learns to reproduce noise
        # patterns, and the near-zero reconstruction error flatters val metrics.
        # Filtering happens BEFORE the split so train/val stay disjoint and
        # neither contains junk.
        self.min_dynamic_range = min_dynamic_range
        if min_dynamic_range > 0:
            ranges = self._load_or_build_stats(stats_cache, verbose)
            keep = [f for f in self.files if ranges.get(f.name, 1.0) >= min_dynamic_range]
            dropped = len(self.files) - len(keep)
            if not keep:
                raise ValueError(
                    f"min_dynamic_range={min_dynamic_range} removed every tile; "
                    "typical land cover sits near 0.26, blank tiles near 0.002"
                )
            if verbose and dropped:
                print(f"dropped {dropped}/{len(self.files)} tiles "
                      f"({dropped / len(self.files):.1%}) below dynamic range "
                      f"{min_dynamic_range}")
            self.files = keep

        # Deterministic train/val split: a fixed permutation of the sorted file
        # list, so the same tile always lands on the same side regardless of
        # machine, worker count, or how many times training is resumed.
        if split not in ("all", "train", "val"):
            raise ValueError(f"split must be all/train/val, got {split!r}")
        if split != "all":
            order = torch.randperm(
                len(self.files),
                generator=torch.Generator().manual_seed(split_seed),
            ).tolist()
            n_val = max(1, int(round(len(self.files) * val_fraction)))
            keep = order[:n_val] if split == "val" else order[n_val:]
            self.files = [self.files[i] for i in sorted(keep)]
        self.split = split

    def _load_or_build_stats(
        self, stats_cache: str | Path | None, verbose: bool
    ) -> dict[str, float]:
        """Per-tile dynamic range, read from a JSON cache or computed once.

        Scanning every tile costs a full pass over the dataset (minutes), so the
        result is cached next to the data and reused. Tiles missing from an
        existing cache are computed and appended, which keeps the cache valid
        when new tiles are downloaded into the same root.
        """
        cache_path = Path(stats_cache) if stats_cache else self.root / ".tile_stats.json"
        ranges: dict[str, float] = {}
        if cache_path.exists():
            try:
                ranges = json.loads(cache_path.read_text())
            except (json.JSONDecodeError, OSError):
                ranges = {}  # corrupt cache: rebuild rather than fail

        missing = [f for f in self.files if f.name not in ranges]
        if missing:
            if verbose:
                print(f"scanning {len(missing)} tiles for dynamic range "
                      f"(one-time, cached to {cache_path.name})...")
            for i, f in enumerate(missing, 1):
                ranges[f.name] = tile_dynamic_range(self._read_reflectance(f))
                if verbose and i % 5000 == 0:
                    print(f"  {i}/{len(missing)}")
            try:
                cache_path.write_text(json.dumps(ranges))
            except OSError as exc:  # read-only data dir: proceed uncached
                if verbose:
                    print(f"  could not write cache ({exc}); continuing")
        return ranges

    def _random_resized_crop(self, tile: torch.Tensor) -> torch.Tensor:
        """Crop a random area/aspect region and resize back to img_size.

        Ten attempts to find a crop that fits inside the tile, then a centre
        crop as fallback — the standard Inception-style RRC procedure.
        """
        lo, hi = self.rrc_scale
        _, h, w = tile.shape
        area = h * w
        for _ in range(10):
            target = area * (lo + (hi - lo) * torch.rand(1).item())
            log_lo, log_hi = math.log(self.rrc_ratio[0]), math.log(self.rrc_ratio[1])
            ratio = math.exp(log_lo + (log_hi - log_lo) * torch.rand(1).item())
            cw = int(round((target * ratio) ** 0.5))
            ch = int(round((target / ratio) ** 0.5))
            if 0 < cw <= w and 0 < ch <= h:
                top = int(torch.randint(0, h - ch + 1, (1,)).item())
                left = int(torch.randint(0, w - cw + 1, (1,)).item())
                tile = tile[:, top : top + ch, left : left + cw]
                break
        else:
            side = min(h, w)
            top, left = (h - side) // 2, (w - side) // 2
            tile = tile[:, top : top + side, left : left + side]

        # Squash back to square: the aspect distortion is deliberate, not a
        # side effect — it is what makes the crop augmentation informative.
        return F.interpolate(
            tile.unsqueeze(0), size=(self.img_size, self.img_size),
            mode="bilinear", align_corners=False,
        ).squeeze(0)

    def _photometric(self, tile: torch.Tensor) -> torch.Tensor:
        """Per-channel gain/offset plus gamma, in reflectance space.

        Applied before standardization so the jitter is in physical units and
        means the same thing across channels. Strength scales with
        ``self.photometric``: 1.0 gives +-25% gain, +-0.02 offset, gamma in
        [1/1.3, 1.3].
        """
        s = self.photometric
        c = tile.shape[0]
        gain = 1.0 + (torch.rand(c, 1, 1) * 2 - 1) * 0.25 * s
        offset = (torch.rand(c, 1, 1) * 2 - 1) * 0.02 * s
        gamma = math.exp((torch.rand(1).item() * 2 - 1) * math.log(1.3) * s)
        tile = tile.clamp_min(0.0).pow(gamma) * gain + offset
        return tile.clamp(0.0, 1.0)

    def _read_reflectance(self, path: Path) -> np.ndarray:
        """(3, H, W) float32 reflectance in [0, 1], R/G/NIR order."""
        with rasterio.open(path) as src:
            arr = src.read([i + 1 for i in self.band_indices])  # rasterio is 1-based
        return np.clip(arr.astype(np.float32), 0.0, S2_CLIP) / S2_CLIP

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        tile = torch.from_numpy(self._read_reflectance(self.files[idx]))
        if tile.shape[-1] != self.img_size or tile.shape[-2] != self.img_size:
            raise ValueError(
                f"{self.files[idx].name}: got {tuple(tile.shape[-2:])}, "
                f"expected {self.img_size}x{self.img_size}"
            )

        # Geometry first (crop/flip/rotate), then radiometry, then
        # standardization — so photometric jitter acts on physical reflectance.
        if self.rrc_scale is not None:
            tile = self._random_resized_crop(tile)

        if self.augment:
            if torch.rand(1).item() < 0.5:
                tile = torch.flip(tile, dims=[-1])  # horizontal
            if torch.rand(1).item() < 0.5:
                tile = torch.flip(tile, dims=[-2])  # vertical
            k = int(torch.randint(4, (1,)).item())
            if k:
                tile = torch.rot90(tile, k=k, dims=[-2, -1])
            tile = tile.contiguous()

        if self.photometric > 0:
            tile = self._photometric(tile)

        if self.normalize:
            mean = torch.tensor(CHANNEL_MEAN).reshape(-1, 1, 1)
            std = torch.tensor(CHANNEL_STD).reshape(-1, 1, 1)
            tile = (tile - mean) / std

        return tile  # (3, 256, 256)