"""CORAL domain adaptation fine-tuning script.

Workflow
--------
1. Initialize the CoralStudentModel (loads all weights including the frozen netA).
2. Scan all LISS-4 patches and discard those where netA reports cloud fraction
   >= cloud_threshold.  This runs once at startup.
3. Train on SEN12MS-CR (source) + filtered LISS-4 (target) batches.

Run from domain_adaptation/methods/coral/:
    source ../../../AttentionGAN-for-Cloud-removal/.venv/bin/activate
    python train.py [args]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
# Makes the vendored student/ models available (needed by parent model classes).
_PKG_ROOT  = Path(__file__).resolve().parents[2]   # domain_adaptation/
_REPO_ROOT = Path(__file__).resolve().parents[3]   # repo root
for _p in (str(_PKG_ROOT / "student"), str(_PKG_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from torch.utils.data import DataLoader

from coral_model import CoralStudentModel
from dataset import CoralDataset
from patch_filter import filter_liss4_patches, find_liss4_patches


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CORAL domain adaptation fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    p.add_argument("--dataroot", default=str(_REPO_ROOT / "SEN12MSCR_prepared"),
                   help="SEN12MS-CR prepared root (contains trainA / trainB)")
    p.add_argument("--dataroot_liss4",
                   default=str(_REPO_ROOT / "Bhoonidhi-Data" / "data_images" /
                               "R2F13APR2026077748011600054SSANSTUC00GTDD"),
                   help="Root dir for LISS-4 scene images (scanned recursively for .jpg)")
    p.add_argument("--cloud_threshold", type=float, default=0.30,
                   help="Discard LISS-4 patches with mean cloud probability >= this value")

    # Model / teacher
    p.add_argument("--checkpoints_dir",
                   default=str(_REPO_ROOT / "domain_adaptation" / "methods" / "outputs" / "checkpoints"),
                   help="Directory where model checkpoints are saved")
    p.add_argument("--name", default="sen12mscr_student_coral_adapted",
                   help="Experiment name (checkpoint subdirectory)")
    p.add_argument("--teacher_checkpoints_dir",
                   default=str(_REPO_ROOT / "AttentionGAN-for-Cloud-removal" / "outputs" / "checkpoints"),
                   help="Directory containing teacher checkpoint")
    p.add_argument("--teacher_name", default="sen12mscr_nir_teacher_v4c")
    p.add_argument("--teacher_which_epoch", default="latest")

    # Optimisation
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--coral_weight", type=float, default=1.0)
    p.add_argument("--gamma_hall", type=float, default=1.0)
    p.add_argument("--cross_depth_weight", type=float, default=1.0)
    p.add_argument("--freqkd_weight", type=float, default=1.0)

    # Logging / saving
    p.add_argument("--save_epoch_freq", type=int, default=5)
    p.add_argument("--print_freq", type=int, default=10)

    # Architecture (must match the checkpoint)
    p.add_argument("--fineSize", type=int, default=256)
    p.add_argument("--which_model_netG", default="unet_256")
    p.add_argument("--which_model_netA", default="resnet_9blocks")
    p.add_argument("--which_model_netG2", default="unet_256")
    p.add_argument("--which_model_netD", default="basic")
    p.add_argument("--n_layers_D", type=int, default=3)
    p.add_argument("--input_nc", type=int, default=3)
    p.add_argument("--output_nc", type=int, default=3)
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--ndf", type=int, default=64)
    p.add_argument("--norm", default="instance")
    p.add_argument("--no_dropout", action="store_true", default=True)
    p.add_argument("--init_type", default="normal")
    p.add_argument("--init_gain", type=float, default=0.02)
    p.add_argument("--no_lsgan", action="store_true", default=False)
    p.add_argument("--which_direction", default="AtoB")
    p.add_argument("--gpu_ids", default="0")

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    opt = parse_args()

    # Device setup
    gpu_ids = [int(i) for i in opt.gpu_ids.split(",") if int(i) >= 0]
    if gpu_ids and torch.cuda.is_available():
        torch.cuda.set_device(gpu_ids[0])
        opt.device = torch.device(f"cuda:{gpu_ids[0]}")
    else:
        opt.device = torch.device("cpu")

    # Translate CLI names to what BaseModel / parent expects
    opt.isTrain            = True
    opt.batchSize          = opt.batch_size
    opt.gpu_ids            = gpu_ids
    opt.continue_train     = False
    opt.which_epoch        = 'latest'
    opt.pool_size          = 50
    opt.beta1              = 0.5
    opt.lambda3            = 1.0
    opt.lambda_A           = 100.0
    opt.CORAL_WEIGHT       = opt.coral_weight
    opt.lr_policy          = 'linear'
    opt.epoch_count        = 1
    opt.niter              = opt.epochs // 2
    opt.niter_decay        = opt.epochs - opt.niter

    # ── Model init ────────────────────────────────────────────────────────────
    print("[INFO] Initializing CoralStudentModel …")
    model = CoralStudentModel()
    model.initialize(opt)

    # ── Patch filtering ───────────────────────────────────────────────────────
    # netA is frozen and loaded; use it now to filter LISS-4 patches once.
    print("[INFO] Scanning and filtering LISS-4 patches …")
    all_liss4 = find_liss4_patches(opt.dataroot_liss4)
    clear_liss4 = filter_liss4_patches(
        all_liss4,
        netA=model.netA,
        device=opt.device,
        cloud_threshold=opt.cloud_threshold,
    )
    if not clear_liss4:
        raise RuntimeError(
            "No LISS-4 patches survived filtering. "
            "Lower --cloud_threshold or check --dataroot_liss4."
        )

    # ── Dataset / dataloader ─────────────────────────────────────────────────
    dataset = CoralDataset(
        sentinel_root=opt.dataroot,
        liss4_paths=clear_liss4,
        fine_size=opt.fineSize,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    print(f"[INFO] Dataset: {len(dataset)} source pairs | "
          f"{len(clear_liss4)} target patches | "
          f"batch_size={opt.batch_size}")

    # ── Training loop ─────────────────────────────────────────────────────────
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"[INFO] Training for {opt.epochs} epochs …")
    for epoch in range(1, opt.epochs + 1):
        t0 = time.time()
        for i, batch in enumerate(dataloader):
            model.set_input(batch)
            model.optimize_parameters()

            if i % opt.print_freq == 0:
                errors = model.get_current_errors()
                row = " | ".join(f"{k}: {v:.4f}" for k, v in errors.items())
                print(f"Epoch {epoch}/{opt.epochs}  iter {i}/{len(dataloader)}  {row}")

        elapsed = int(time.time() - t0)

        if epoch % opt.save_epoch_freq == 0 or epoch == opt.epochs:
            print(f"[INFO] Saving checkpoint at epoch {epoch} …")
            model.save(f"epoch_{epoch}")
            model.save("latest")

        print(f"Epoch {epoch} done in {elapsed}s")

    print("[SUCCESS] CORAL domain adaptation complete.")


if __name__ == "__main__":
    main()
