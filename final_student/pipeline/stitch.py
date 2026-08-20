"""Utilities for stitching overlapping patches and re-applying pushbroom shear."""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def create_hann_window(size: int = 256) -> np.ndarray:
    w1d = np.hanning(size)
    return np.outer(w1d, w1d).astype(np.float32)


def make_canvas(orig_h: int, orig_w: int, channels: int, patch_size: int = 256, stride: int = 128):
    """Allocate a float32 canvas and matching weight map for Hann-blended stitching."""
    pad_h = (stride - (orig_h - patch_size) % stride) % stride
    pad_w = (stride - (orig_w - patch_size) % stride) % stride
    padded_h = orig_h + pad_h
    padded_w = orig_w + pad_w
    canvas = np.zeros((padded_h, padded_w, channels), dtype=np.float32)
    return canvas


def finalize_canvas(
    canvas: np.ndarray,
    weight: np.ndarray,
    orig_h: int,
    orig_w: int,
) -> np.ndarray:
    """Normalize by accumulated Hann weights and crop to original dimensions.

    Returns a uint8 array of shape [orig_h, orig_w, C].
    """
    w = np.clip(weight, 1e-8, None)[:, :, None]
    result = np.clip(canvas / w, 0, 255)[:orig_h, :orig_w]
    return result.astype(np.uint8)


def accumulate(
    canvas: np.ndarray,
    weight: np.ndarray,
    patch_np: np.ndarray,
    y: int,
    x: int,
    window: np.ndarray,
) -> None:
    """Hann-accumulate one patch into the canvas in place.

    patch_np: [H, W, C] uint8 — the model output for this tile.
    weight:   [padded_H, padded_W] float32 — shared across all canvases.
    """
    p = window.shape[0]
    canvas[y : y + p, x : x + p] += patch_np.astype(np.float32) * window[:, :, None]
    weight[y : y + p, x : x + p] += window


class Stitcher:
    """Hann overlap-add accumulator for several co-registered output canvases.

    Exists because the free-function pair above is only safe with ONE canvas.
    `accumulate()` bumps the shared weight map on every call, so a caller
    holding N canvases and calling it N times per patch over-counts the weight
    N-fold, and `finalize_canvas` then divides every output down by N. This
    class accumulates the weight exactly once per patch regardless of how many
    canvases it feeds.

    Patches arrive as torch tensors already in [B, H, W, C] float 0-255, and the
    canvases live on `device` — keeping them on the GPU avoids a host transfer
    and a float32 CPU multiply-add per patch per canvas.
    """

    def __init__(
        self,
        specs: dict[str, int],
        orig_h: int,
        orig_w: int,
        patch_size: int = 256,
        stride: int = 128,
        device: str = "cpu",
    ):
        pad_h = (stride - (orig_h - patch_size) % stride) % stride
        pad_w = (stride - (orig_w - patch_size) % stride) % stride
        self.orig_h, self.orig_w = orig_h, orig_w
        self.p = patch_size
        self.device = torch.device(device)

        padded = (orig_h + pad_h, orig_w + pad_w)
        self.canvases = {
            name: torch.zeros((*padded, c), dtype=torch.float32, device=self.device)
            for name, c in specs.items()
        }
        self.weight = torch.zeros((*padded, 1), dtype=torch.float32, device=self.device)
        self.window = torch.from_numpy(create_hann_window(patch_size)).to(
            self.device
        ).unsqueeze(-1)  # [p, p, 1]

    @staticmethod
    def bytes_needed(specs: dict[str, int], orig_h: int, orig_w: int,
                     patch_size: int = 256, stride: int = 128) -> int:
        """float32 VRAM the canvases plus the weight map will occupy."""
        pad_h = (stride - (orig_h - patch_size) % stride) % stride
        pad_w = (stride - (orig_w - patch_size) % stride) % stride
        px = (orig_h + pad_h) * (orig_w + pad_w)
        return px * (sum(specs.values()) + 1) * 4

    def add(self, patches: dict, ys, xs) -> None:
        """Accumulate one batch. `patches` maps canvas name -> [B, H, W, C] float."""
        p = self.p
        for i in range(len(ys)):
            y, x = int(ys[i]), int(xs[i])
            for name, canvas in self.canvases.items():
                canvas[y : y + p, x : x + p] += patches[name][i] * self.window
            # Once per patch, NOT once per canvas — see class docstring.
            self.weight[y : y + p, x : x + p] += self.window

    def finalize(self, name: str) -> np.ndarray:
        """Normalize by accumulated weight, crop to original size, return uint8."""
        w = self.weight.clamp_min(1e-8)
        out = (self.canvases[name] / w).clamp_(0, 255)
        out = out[: self.orig_h, : self.orig_w]
        return out.to(dtype=torch.uint8).cpu().numpy()


def read_manifest(manifest_path: Path) -> dict:
    """Parse the manifest.csv produced by bhoonidhi_rectangle.py.

    Returns a dict keyed by scene_id with the full manifest row as values.
    """
    result = {}
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            result[row["scene_id"]] = row
    return result


def get_shear_params(manifest_row: dict) -> tuple[float, float, int, int, int, int]:
    """Extract (m, s0, r0, rect_h, in_h, in_w) from a manifest row.

    m   = shear_col_per_row (float)
    s0  = starting column of valid data at row r0 (estimated from geometry)
    r0  = first row of the rectangular crop in the raw image (inferred)
    rect_h = height of rectangular output (= out_height from manifest)
    in_h, in_w = dimensions of the original raw image
    """
    m     = float(manifest_row["shear_col_per_row"])
    in_h  = int(manifest_row["in_height"])
    in_w  = int(manifest_row["in_width"])
    rect_h = int(manifest_row["out_height"])
    rect_w = int(manifest_row["out_width"])

    # When in_height == out_height the full-height block was used → r0 = 0.
    # s0 is the column offset at row 0: for negative shear (data shifts left going down)
    # it equals in_w - rect_w (the right-side margin).
    if in_h == rect_h:
        r0 = 0
        s0 = in_w - rect_w if m < 0 else 0
    else:
        # Partial-height crop: approximate s0 conservatively.
        r0 = 0
        s0 = in_w - rect_w if m < 0 else 0

    return m, s0, r0, rect_h, in_h, in_w


def apply_pushbroom_shear(
    img_rect: np.ndarray,
    m: float,
    s0: float,
    r0: int,
    in_h: int,
    in_w: int,
) -> np.ndarray:
    """Place a rectangular stitched image back into the original pushbroom frame.

    img_rect: [rect_H, rect_W, C] uint8
    Returns:  [in_h, in_w, C] uint8 (zeros outside valid region)
    """
    rect_h, rect_w = img_rect.shape[:2]
    C = img_rect.shape[2]
    out = np.zeros((in_h, in_w, C), dtype=np.uint8)

    for row in range(rect_h):
        x_start = int(round(s0 + m * row))
        x_start = max(0, min(x_start, in_w - rect_w))
        out[r0 + row, x_start : x_start + rect_w] = img_rect[row]

    return out


def _stretch_uint8(arr: np.ndarray) -> np.ndarray:
    """Per-channel percentile stretch of a [H, W, C] uint8 array.

    Useful when model outputs span only a narrow range of [0, 255] due to
    domain gap (e.g. TTBN on LISS-4 with a SEN12MS-CR-trained model).
    Masks zero pixels as nodata so border padding doesn't skew the stretch.
    """
    out = np.zeros_like(arr)
    for c in range(arr.shape[2]):
        ch = arr[:, :, c].astype(np.float32)
        valid = ch[ch > 0]
        if valid.size == 0:
            continue
        lo, hi = float(np.percentile(valid, 2)), float(np.percentile(valid, 98))
        if hi <= lo:
            hi = lo + 1.0
        stretched = np.clip((ch - lo) / (hi - lo) * 255.0, 0, 255)
        stretched[ch == 0] = 0
        out[:, :, c] = stretched.astype(np.uint8)
    return out


def save_image(arr: np.ndarray, path: Path, quality: int = 95,
               stretch: bool = False) -> None:
    """Save a [H, W, C] uint8 array as JPEG (or grayscale if C=1).

    stretch is accepted for backward compatibility but is permanently ignored —
    per-channel contrast stretching has been removed from the pipeline.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.shape[2] == 1:
        img = Image.fromarray(arr[:, :, 0], mode="L")
    else:
        img = Image.fromarray(arr, mode="RGB")
    img.save(path, quality=quality)
