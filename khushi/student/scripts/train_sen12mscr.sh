#!/usr/bin/env bash
# Train the AttentionGAN teacher on SEN12MS-CR data.
# Run from the repo root:  bash scripts/train_sen12mscr.sh
#
# Step 1 — prepare PNGs (only needed once):
#   python ../../prepare_teacher_data.py
#
# Step 2 — activate venv and train:
#   source .venv/bin/activate
#   bash scripts/train_sen12mscr.sh

set -e

DATAROOT="/workspace/Cloudy-with-a-chance-of-kofta/SEN12MSCR_prepared"
NAME="sen12mscr_teacher_v1"
CHECKPOINTS_DIR="./outputs/checkpoints"

python train.py \
  --dataroot        "$DATAROOT" \
  --name            "$NAME" \
  --model           pix2pix_attn \
  --dataset_mode    unaligned_sar \
  --checkpoints_dir "$CHECKPOINTS_DIR" \
  --niter           100 \
  --niter_decay     100 \
  --lr_policy       step \
  --batchSize       1 \
  --gpu_ids         0 \
  --no_flip \
  --display_id      0 \
  --save_epoch_freq 20 \
  --display_freq    500 \
  --no_html \
  --pool_size       50 \
  --no_dropout
