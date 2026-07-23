"""Visualize MAE masking + reconstruction on a few sample tiles.

Needs a checkpoint with a `model_state_dict` (encoder + decoder), saved by
the current pretrain_mae.py. Checkpoints containing only `backbone_state_dict`
(pre-decoder-checkpointing runs) can't reconstruct pixels — the decoder would
be randomly initialized.

For each sample, saves a single PNG with three tiles side by side:
original | masked input (masked patches zeroed) | reconstruction
(visible original patches + predicted masked patches).

Channels are R/G/NIR (LISS-IV order) mapped directly to RGB for display —
this is a false-color composite (NIR takes the blue slot), not true color.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sen12mscr_dataset import SEN12MSCROptical
from vit_pretrain import MAEConfig, MAEPretrainer


def to_uint8(img_chw: torch.Tensor, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Per-channel contrast stretch (bounds from the original tile) then to uint8.

    Reflectance scaled by S2_CLIP=10000 is very dark for typical land cover
    (raw values are often a few hundred to ~2000 out of 10000), so a plain
    [0,1]->uint8 cast renders near-black. Stretching by the original image's
    own percentile range makes the tile human-visible without touching what
    the model actually sees/trains on.
    """
    arr = img_chw.detach().cpu().numpy()
    arr = (arr - lo) / np.clip(hi - lo, 1e-6, None)
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).round().astype(np.uint8)
    return np.transpose(arr, (1, 2, 0))  # HWC


def stretch_bounds(img_chw: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """2nd/98th percentile per channel, as (C,1,1) arrays for broadcasting."""
    arr = img_chw.detach().cpu().numpy()
    lo = np.percentile(arr, 2, axis=(1, 2)).reshape(-1, 1, 1)
    hi = np.percentile(arr, 98, axis=(1, 2)).reshape(-1, 1, 1)
    return lo, hi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="../SEN12MSCR/ROIs1158_spring_s2")
    p.add_argument("--pattern", default="**/*.tif")
    p.add_argument("--checkpoint", default="mae_backbone.pt")
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--num-samples", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="viz_out")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = SEN12MSCROptical(args.data_root, pattern=args.pattern, img_size=args.img_size)
    idxs = torch.randperm(len(ds))[: args.num_samples].tolist()
    batch = torch.stack([ds[i] for i in idxs]).to(device)

    model = MAEPretrainer(
        MAEConfig(img_size=args.img_size, patch_size=args.patch_size, in_channels=3)
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"loaded full model (encoder+decoder) from {args.checkpoint}")
    else:
        model.encoder.load_state_dict(ckpt["backbone_state_dict"])
        print(
            "WARNING: checkpoint only has encoder weights (backbone-only export) "
            "— decoder is randomly initialized, reconstructions will be meaningless."
        )

    model.eval()
    with torch.no_grad():
        loss, pred, mask = model(batch)
    print(f"reconstruction loss on these {args.num_samples} samples: {loss.item():.4f}")

    target = model.patchify(batch)
    mean = target.mean(dim=-1, keepdim=True)
    std = (target.var(dim=-1, keepdim=True) + 1e-6) ** 0.5
    pred_pixels = pred * std + mean  # invert norm_pix_loss normalization for display

    mask_exp = mask.unsqueeze(-1)  # (B, N, 1)
    recon_patches = mask_exp * pred_pixels + (1 - mask_exp) * target
    recon_imgs = model.unpatchify(recon_patches)
    masked_imgs = model.unpatchify(target * (1 - mask_exp))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    for i, idx in enumerate(idxs):
        lo, hi = stretch_bounds(batch[i])
        tiles = [
            to_uint8(batch[i], lo, hi),
            to_uint8(masked_imgs[i], lo, hi),
            to_uint8(recon_imgs[i], lo, hi),
        ]
        grid = np.concatenate(tiles, axis=1)  # (H, 3W, C): original | masked | recon
        out_path = out_dir / f"sample_{idx}.png"
        Image.fromarray(grid).save(out_path)
        print(f"saved {out_path}  (original | masked | reconstruction)")


if __name__ == "__main__":
    main()
