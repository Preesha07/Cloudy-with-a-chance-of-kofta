"""
Phase-2 student GAN training entry point.

Usage (from the repo root or from izma/):
    python izma/train_student.py \
        --sen12mscr  SEN12MSCR \
        --phase1-ckpt khushi/mae_backbone.pt \
        --teacher-ckpt-dir AttentionGAN-for-Cloud-removal/outputs/checkpoints/sen12mscr_teacher_v1 \
        --epochs 50 \
        --out student_gan.pt

All paths are resolved relative to the current working directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Phase-2 student GAN")
    p.add_argument("--sen12mscr", required=True,
                   help="Root of the SEN12MS-CR dataset (contains ROIs1158_spring_s1/ etc.)")
    p.add_argument("--phase1-ckpt", required=True,
                   help="Phase-1 backbone checkpoint (khushi/mae_backbone.pt)")
    p.add_argument("--teacher-ckpt-dir", required=True,
                   help="Teacher checkpoint dir (…/sen12mscr_teacher_v1)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1,
                   help="Batch size per GPU. Teacher was trained with 1; keep 1 for safety.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--out", default="student_gan.pt",
                   help="Where to save the final student checkpoint")
    p.add_argument("--save-every", type=int, default=5,
                   help="Save a checkpoint every N epochs")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--device", default="auto",
                   help="cuda / cpu / auto")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Teacher loader
# ---------------------------------------------------------------------------

def load_teacher(teacher_ckpt_dir: str | Path, device: str):
    """Instantiate the trained Pix2Pix_attn teacher and load its weights.

    Uses the teacher's own model class directly, bypassing the framework's
    TestOptions parser to avoid sys.argv side-effects.
    """
    teacher_ckpt_dir = Path(teacher_ckpt_dir).resolve()
    checkpoints_dir  = str(teacher_ckpt_dir.parent)
    name             = teacher_ckpt_dir.name

    # Make AttentionGAN importable
    attn_gan_path = str(REPO_ROOT / "AttentionGAN-for-Cloud-removal")
    if attn_gan_path not in sys.path:
        sys.path.insert(0, attn_gan_path)

    from models.pix2pix_attn_model import Pix2Pix_attn_Model   # noqa: E402

    gpu_ids = [0] if (device != "cpu" and torch.cuda.is_available()) else []

    opt = argparse.Namespace(
        # architecture — must match the checkpoint
        batchSize=1, fineSize=256,
        input_nc=3, output_nc=3,
        ngf=64, ndf=64,
        which_model_netG="unet_256",
        which_model_netG2="unet_256",
        which_model_netA="unet_256",
        which_model_netD="basic",
        n_layers_D="3",
        loss_weight=None,
        norm="instance",
        no_dropout=True,
        init_type="xavier",
        which_direction="AtoB",
        # loading
        isTrain=False,
        which_epoch="latest",
        continue_train=False,
        checkpoints_dir=checkpoints_dir,
        name=name,
        gpu_ids=gpu_ids,
    )

    model = Pix2Pix_attn_Model()
    model.initialize(opt)    # allocates netG/netG2/netA and loads latest weights
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ---- device ----
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[train_student] device={device}")

    # ---- import student components ----
    # student_gan.py already inserts khushi/ into sys.path on import
    sys.path.insert(0, str(HERE))
    from student_gan import (  # noqa: E402
        CFG, Phase1ViTEncoder, StudentGenerator, StudentDiscriminator,
        TeacherWrapper, build_projections, train_student_gan,
    )
    from student_dataset import build_student_loader   # noqa: E402

    # ---- Phase-1 backbone ----
    phase1_ckpt = str(Path(args.phase1_ckpt).resolve())
    print(f"[train_student] loading Phase-1 backbone from {phase1_ckpt}")
    encoder = Phase1ViTEncoder(
        phase1_ckpt,
        embed_dim=CFG.embed_dim,
        depth=CFG.depth,
        freeze=True,
        in_chans=CFG.in_chans,
    )

    # ---- student models ----
    student_G = StudentGenerator(encoder, CFG)
    student_D = StudentDiscriminator(phase1_ckpt, CFG)

    # ---- teacher ----
    print(f"[train_student] loading teacher from {args.teacher_ckpt_dir}")
    teacher_model = load_teacher(args.teacher_ckpt_dir, device)
    teacher = TeacherWrapper(teacher_model)

    # ---- projections ----
    # "sar" Conv2d projects student token spatial map (embed_dim channels) to match
    # teacher.fake_C channel count (3).  "feat" stays as Identity since L_feat = 0
    # until teacher hint hooks are wired up.
    projections = build_projections(CFG, teacher_sar_chans=3)

    # ---- data ----
    print(f"[train_student] building dataloaders from {args.sen12mscr}")
    train_loader = build_student_loader(
        args.sen12mscr,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=not args.no_augment,
    )
    val_loader = build_student_loader(
        args.sen12mscr,
        split="val",
        batch_size=1,
        num_workers=2,
        augment=False,
    )
    print(f"[train_student] train={len(train_loader.dataset)} val={len(val_loader.dataset)} tiles")

    # ---- train ----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train_student_gan(
        student_G, student_D, teacher,
        train_loader, projections,
        cfg=CFG,
        epochs=args.epochs,
        device=device,
        lr=args.lr,
        save_every=args.save_every,
        out_path=str(out_path),
        val_loader=val_loader,
    )

    print(f"[train_student] done — final weights saved to {out_path}")


if __name__ == "__main__":
    main()
