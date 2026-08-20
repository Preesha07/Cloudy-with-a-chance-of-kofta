#!/usr/bin/env bash
# NIR teacher v4c-FFL-SAR — v4c + Focal Frequency Loss (on G AND G2) + emphasised
# cloud-region sharpness + a new explicit SAR-reliance loss.
#
# WHY: v4c_scratch's netG2 (SAR->optical) is trained with only L1+SSIM, which is
# the classic "regress to the mean" blur recipe. A blurry fake_C gives netG
# nothing sharp to borrow from under clouds, so netG falls back on the
# (uninformative) cloudy real_A there instead of leaning on the SAR translation.
# This run targets that directly: sharpens fake_C itself (new FFL term on G2),
# sharpens fake_B harder in cloud regions (FFL + raised cloud-Sobel weight on G),
# and adds a new loss that explicitly pulls netG's raw output toward netG2's
# SAR-derived translation inside attention-flagged cloud regions
# (loss_sar_reliance = mean(att_A * |g_B - fake_C.detach()|)).
#
# Warm-starts from sen12mscr_nir_teacher_v4c_scratch (epoch 120, its latest
# saved checkpoint) and trains for 60 more epochs (121 -> 180).
#
# Architecture: identical to v4c (single SAR-conditioned netD, frozen netA
# oracle, BatchNorm on netG2). No netD2 — this experiment is about the loss
# function, not the discriminator setup.
#
# Checkpoints land in: outputs/checkpoints/sen12mscr_nir_teacher_v4c_ffl_sar/
# Visuals to:          outputs/visuals_nir/sen12mscr_nir_teacher_v4c_ffl_sar/
#
# Run from AttentionGAN-for-Cloud-removal/:
#   source .venv/bin/activate
#   nohup bash scripts/train_sen12mscr_nir_v4c_ffl_sar.sh > nohup_v4c_ffl_sar.out 2>&1 &

set -e

DATAROOT="/workspace/Cloudy-with-a-chance-of-kofta/SEN12MSCR_student"
NAME="sen12mscr_nir_teacher_v4c_ffl_sar"
CHECKPOINTS_DIR="./outputs/checkpoints"

python train_nir_vis.py \
  --dataroot            "$DATAROOT" \
  --name                "$NAME" \
  --model               pix2pix_attn_nir_v4c_ffl_sar \
  --dataset_mode        unaligned_sar \
  --checkpoints_dir     "$CHECKPOINTS_DIR" \
  --niter               150 \
  --niter_decay         30 \
  --epoch_count         121 \
  --lr                  0.0001 \
  --lr_policy           step \
  --lr_decay_iters      30 \
  --batchSize           16 \
  --nThreads            8 \
  --gpu_ids             0 \
  --no_flip \
  --display_id          0 \
  --save_epoch_freq     10 \
  --display_freq        500 \
  --no_html \
  --pool_size           25 \
  --no_dropout \
  --norm_G2             batch \
  --lambda_sharp        75.0 \
  --lambda_sharp_cloud  200.0 \
  --lambda_clear        0.1 \
  --lambda_ffl          300.0 \
  --lambda_ffl_g2       150.0 \
  --lambda_sar_reliance 20.0 \
  --frozen_attn_name    sen12mscr_nir_teacher_v3 \
  --frozen_attn_epoch   latest \
  --warmstart_name      sen12mscr_nir_teacher_v4c_scratch \
  --warmstart_epoch     latest
