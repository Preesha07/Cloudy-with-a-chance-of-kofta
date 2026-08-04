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
from PIL import Image, ImageDraw

import metrics
from sen12mscr_dataset import SEN12MSCROptical, denormalize, tile_dynamic_range
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


MIN_STRETCH_SPAN = 0.02
"""Floor on the contrast-stretch range, in reflectance units.

Without it, a near-blank tile (deep water/shadow, span ~0.002) gets its range
multiplied by ~500x, turning sensor noise into saturated colour and making a
numerically near-perfect reconstruction look catastrophically broken. Ordinary
land cover spans ~0.26, far above this floor, so normal tiles are unaffected.
"""


def stretch_bounds(img_chw: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """2nd/98th percentile per channel, as (C,1,1) arrays for broadcasting.

    The span is floored at MIN_STRETCH_SPAN so blank tiles render as flat grey
    rather than amplified noise.
    """
    arr = img_chw.detach().cpu().numpy()
    lo = np.percentile(arr, 2, axis=(1, 2)).reshape(-1, 1, 1)
    hi = np.percentile(arr, 98, axis=(1, 2)).reshape(-1, 1, 1)
    short = (hi - lo) < MIN_STRETCH_SPAN
    if short.any():
        mid = (lo + hi) / 2.0
        lo = np.where(short, mid - MIN_STRETCH_SPAN / 2, lo)
        hi = np.where(short, mid + MIN_STRETCH_SPAN / 2, hi)
    return lo, hi


def label_grid(grid: np.ndarray, text: str) -> np.ndarray:
    """Add a caption strip under the panel row so an image is self-describing."""
    bar_h = 16
    bar = Image.new("RGB", (grid.shape[1], bar_h), (16, 16, 16))
    ImageDraw.Draw(bar).text((4, 3), text, fill=(235, 235, 235))
    return np.concatenate([grid, np.array(bar)], axis=0)


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

    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" not in ckpt:
        raise ValueError(
            f"{args.checkpoint} only contains backbone weights — "
            "pass a checkpoint saved by pretrain_mae.py (which includes model_state_dict) "
            "or use mae_backbone.pt.ckpt from a run with --ckpt-every."
        )

    # Rebuild the architecture the checkpoint was trained with. Older checkpoints
    # predate `mae_config` and are all ViT-Base at the CLI's img/patch size.
    if "mae_config" in ckpt:
        cfg = MAEConfig(**ckpt["mae_config"])
    else:
        cfg = MAEConfig(
            img_size=args.img_size, patch_size=args.patch_size, in_channels=3
        )
        print("checkpoint has no mae_config — assuming ViT-Base defaults")

    # The input pipeline must match how the checkpoint was trained: a model
    # trained on standardized input produces garbage on raw reflectance.
    normalize = bool(ckpt.get("config", {}).get("normalize_input", True))
    if "config" in ckpt and "no_normalize" in ckpt["config"]:
        normalize = not ckpt["config"]["no_normalize"]
    elif "mae_config" not in ckpt:
        normalize = False  # pre-standardization checkpoint
    ds = SEN12MSCROptical(
        args.data_root, pattern=args.pattern, img_size=cfg.img_size,
        normalize=normalize,
    )
    idxs = torch.randperm(len(ds))[: args.num_samples].tolist()
    batch = torch.stack([ds[i] for i in idxs]).to(device)

    model = MAEPretrainer(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"loaded full model (encoder+decoder) from {args.checkpoint}  "
          f"(embed_dim={cfg.embed_dim}, mask_ratio={cfg.mask_ratio}, "
          f"normalize={normalize})")

    model.eval()
    with torch.no_grad():
        loss, pred, mask = model(batch)
    print(f"reconstruction loss on these {args.num_samples} samples: {loss.item():.4f}")

    target = model.patchify(batch)
    if model.config.norm_pix_loss:
        # NOTE: this hands the model each patch's *ground-truth* mean and std.
        # Under norm_pix_loss the model only ever predicts the within-patch
        # pattern, never absolute brightness — so these panels flatter it.
        # Judge convergence by the val metrics from pretrain_mae.py, not by eye.
        mean = target.mean(dim=-1, keepdim=True)
        std = (target.var(dim=-1, keepdim=True) + 1e-6) ** 0.5
        pred_pixels = pred * std + mean
    else:
        pred_pixels = pred

    mask_exp = mask.unsqueeze(-1)  # (B, N, 1)
    recon_patches = mask_exp * pred_pixels + (1 - mask_exp) * target
    recon_imgs = model.unpatchify(recon_patches)

    # Render in reflectance units regardless of how the model was fed.
    if normalize:
        batch = denormalize(batch)
        recon_imgs = denormalize(recon_imgs)
    mask_img = model.unpatchify(mask_exp.expand(-1, -1, target.shape[-1]))
    masked_imgs = batch * (1 - mask_img)

    m = metrics.masked_psnr(
        model.patchify(recon_imgs.clamp(0, 1)),
        model.patchify(batch.clamp(0, 1)),
        mask,
    )
    print(f"masked-patch PSNR on these samples: {m:.2f} dB "
          f"(optimistic — see note in the source)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    for i, idx in enumerate(idxs):
        # Per-tile numbers, so a panel can be judged without re-deriving context.
        # A near-blank tile scoring 40+ dB is the model working, not failing.
        dyn = tile_dynamic_range(batch[i].clamp(0, 1).cpu().numpy())
        per_tile_psnr = metrics.masked_psnr(
            model.patchify(recon_imgs[i : i + 1].clamp(0, 1)),
            model.patchify(batch[i : i + 1].clamp(0, 1)),
            mask[i : i + 1],
        )
        blank = dyn < MIN_STRETCH_SPAN

        lo, hi = stretch_bounds(batch[i])
        tiles = [
            to_uint8(batch[i], lo, hi),
            to_uint8(masked_imgs[i], lo, hi),
            to_uint8(recon_imgs[i], lo, hi),
        ]
        grid = np.concatenate(tiles, axis=1)  # (H, 3W, C): original | masked | recon
        caption = (f"tile {idx}  range {dyn:.4f}  psnr {per_tile_psnr:.1f}dB"
                   f"{'   NEAR-BLANK TILE - contrast floored' if blank else ''}")
        grid = label_grid(grid, caption)

        out_path = out_dir / f"sample_{idx}.png"
        Image.fromarray(grid).save(out_path)
        print(f"saved {out_path}  | {caption}")


if __name__ == "__main__":
    main()
