"""Plain Deep CORAL loss (Sun & Saenko, ECCV 2016).

Aligns second-order statistics (channel covariance) between source and target
feature maps without any spatial masking. Cloud filtering is handled upstream
by patch_filter.filter_liss4_patches() before training begins.
"""
from __future__ import annotations

import torch


def channel_covariance(feat: torch.Tensor) -> torch.Tensor:
    """Per-image channel covariance over spatial positions.

    Args:
        feat: (B, C, H, W) feature map.

    Returns:
        (B, C, C) covariance matrices, one per image.
    """
    B, C, H, W = feat.shape
    f = feat.reshape(B, C, -1)                        # (B, C, N)
    f = f - f.mean(dim=2, keepdim=True)               # centre
    return torch.bmm(f, f.transpose(1, 2)) / (H * W - 1 + 1e-8)


def coral_loss(feat_s: torch.Tensor, feat_t: torch.Tensor) -> torch.Tensor:
    """Frobenius norm of covariance difference, normalised by 4C².

    Args:
        feat_s: (B, C, H, W) source domain features.
        feat_t: (B, C, H, W) target domain features.

    Returns:
        Scalar loss tensor.
    """
    cov_s = channel_covariance(feat_s)
    cov_t = channel_covariance(feat_t)
    C = feat_s.size(1)
    return ((cov_s - cov_t) ** 2).mean() / (4.0 * C * C)
