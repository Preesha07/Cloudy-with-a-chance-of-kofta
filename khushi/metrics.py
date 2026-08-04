"""Reconstruction metrics for MAE evaluation.

Two PSNR variants are reported, and the distinction matters:

- **normalized PSNR** — computed on per-patch mean/std-normalized targets, i.e.
  exactly what the model predicts when ``norm_pix_loss=True``. This is the honest
  number: nothing the model didn't produce is fed back in.
- **pixel PSNR** — computed in reflectance space after restoring each patch's
  *ground-truth* mean and std. Comparable to numbers reported in the literature,
  but optimistic: with ``norm_pix_loss=True`` the model never learns absolute
  brightness, so this metric hands it back for free.

SSIM needs real spatial structure, so it is only defined on the pixel-space
composite (visible ground-truth patches + predicted masked patches).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """PSNR in dB over all supplied elements. Returns inf on an exact match."""
    mse = F.mse_loss(pred.float(), target.float()).item()
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(data_range**2 / mse)


def masked_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    data_range: float = 1.0,
) -> float:
    """PSNR over masked patches only.

    Args:
        pred/target: (B, N, patch_dim)
        mask: (B, N), 1 = masked (a reconstruction target)
    """
    sel = mask.bool()
    if not sel.any():
        return float("nan")
    return psnr(pred[sel], target[sel], data_range=data_range)


def _gaussian_window(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return g.outer(g)  # (w, w)


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """Mean SSIM over a (B, C, H, W) batch, channel-averaged.

    Standard Gaussian-window SSIM (Wang et al. 2004) with the usual K1=0.01,
    K2=0.03. Implemented here rather than pulled from a dependency to keep this
    directory free of extra requirements — the vendored teacher repo has its own
    `pytorch_ssim`, but importing across repo boundaries is worse than 20 lines.
    """
    pred = pred.float()
    target = target.float()
    c = pred.shape[1]
    win = _gaussian_window(window_size, sigma, pred.device, pred.dtype)
    win = win.expand(c, 1, window_size, window_size).contiguous()
    pad = window_size // 2

    mu1 = F.conv2d(pred, win, padding=pad, groups=c)
    mu2 = F.conv2d(target, win, padding=pad, groups=c)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, win, padding=pad, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(target * target, win, padding=pad, groups=c) - mu2_sq
    sigma12 = F.conv2d(pred * target, win, padding=pad, groups=c) - mu1_mu2

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    num = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    return (num / den).mean().item()
