#!/usr/bin/env bash
# Experiment 1: AttentionGAN + dedicated D2 discriminator on G2 (SAR→optical).
# Model: pix2pix_attn_d2  |  Name: sen12mscr_teacher_v2_disc
#
# Checkpoints → outputs/checkpoints/sen12mscr_teacher_v2_disc/
# Visuals     → outputs/visuals_v2/   (written by train_v2_disc.py)
#
# Run from AttentionGAN-for-Cloud-removal/:
#   source .venv/bin/activate
#   bash scripts/train_sen12mscr_v2_disc.sh

set -e

DATAROOT="/workspace/Cloudy-with-a-chance-of-kofta/SEN12MSCR_prepared"
NAME="sen12mscr_teacher_v2_disc"
CHECKPOINTS_DIR="./outputs/checkpoints"

python train_v2_disc.py \
  --dataroot        "$DATAROOT" \
  --name            "$NAME" \
  --model           pix2pix_attn_d2 \
  --dataset_mode    unaligned_sar \
  --checkpoints_dir "$CHECKPOINTS_DIR" \
  --niter           15 \
  --niter_decay     0 \
  --lr_policy       step \
  --lr_decay_iters  7 \
  --batchSize       1 \
  --gpu_ids         0 \
  --no_flip \
  --display_id      0 \
  --save_epoch_freq 5 \
  --display_freq    500 \
  --no_html \
  --pool_size       50 \
  --no_dropout
