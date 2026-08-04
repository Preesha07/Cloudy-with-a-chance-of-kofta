#!/usr/bin/env bash
# Test the student model (SAR-free inference).
# Only real_A (cloudy RG+NIR) is used at test time — real_C (SAR) is ignored.
#
# Run from AttentionGAN-for-Cloud-removal/:
#   source .venv/bin/activate
#   bash scripts/test_sen12mscr_student.sh

set -e

DATAROOT="/workspace/Cloudy-with-a-chance-of-kofta/SEN12MSCR_student"
NAME="sen12mscr_student_v1"
CHECKPOINTS_DIR="./outputs/checkpoints"

python test_student.py \
  --dataroot        "$DATAROOT" \
  --name            "$NAME" \
  --model           pix2pix_attn_student \
  --dataset_mode    unaligned_sar \
  --checkpoints_dir "$CHECKPOINTS_DIR" \
  --which_epoch     latest \
  --phase           val \
  --gpu_ids         0 \
  --no_flip \
  --no_html \
  --display_id      0
