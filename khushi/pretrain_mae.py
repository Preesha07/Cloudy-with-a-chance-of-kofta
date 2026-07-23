"""Short MAE pretraining run to verify the ViT backbone learns on Sentinel data."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sen12mscr_dataset import SEN12MSCROptical
from vit_pretrain import MAEConfig, MAEPretrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="../SEN12MSCR/ROIs1158_spring_s2")
    p.add_argument("--pattern", default="**/*.tif")
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--lr", type=float, default=1.5e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--out", default="mae_backbone.pt")
    return p.parse_args()


def lr_at(step: int, total: int, base_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    ds = SEN12MSCROptical(args.data_root, pattern=args.pattern, img_size=args.img_size)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
        drop_last=True, persistent_workers=args.num_workers > 0,
    )
    print(f"tiles={len(ds)}  batches/epoch={len(loader)}")

    model = MAEPretrainer(
        MAEConfig(img_size=args.img_size, patch_size=args.patch_size, in_channels=3)
    ).to(device)

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    total_steps = args.max_steps or args.epochs * len(loader)
    step, t0 = 0, time.time()
    model.train()

    for epoch in range(args.epochs):
        for imgs in loader:
            imgs = imgs.to(device, non_blocking=True)
            for g in optim.param_groups:
                g["lr"] = lr_at(step, total_steps, args.lr, args.warmup_steps)

            optim.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=args.amp):
                loss, _, _ = model(imgs)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            if step % 20 == 0:
                rate = (step + 1) / (time.time() - t0)
                print(f"epoch {epoch:3d} | step {step:5d}/{total_steps} "
                      f"| loss {loss.item():.4f} | lr {optim.param_groups[0]['lr']:.2e} "
                      f"| {rate:.1f} it/s")

            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        if args.max_steps and step >= args.max_steps:
            break

    backbone = model.export_backbone()
    torch.save(
        {
            "backbone_state_dict": backbone.state_dict(),
            "model_state_dict": model.state_dict(),
            "config": vars(args),
        },
        args.out,
    )
    print(f"saved backbone + full model -> {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()