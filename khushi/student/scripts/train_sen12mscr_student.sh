#!/usr/bin/env bash
# Train the SAR-free student model on SEN12MS-CR data (B4/B3/B8 channels).
#
# Prerequisites:
#   1. Teacher trained:  outputs/checkpoints/sen12mscr_teacher_v1/latest_net_*.pth
#   2. Student data prepared:
#        python ../../prepare_student_data.py \
#            --prepared  ../SEN12MSCR_prepared \
#            --sen12mscr ../SEN12MSCR \
#            --out       ../SEN12MSCR_student \
#            --workers   8
#
# Run from AttentionGAN-for-Cloud-removal/:
#   source .venv/bin/activate
#   bash scripts/train_sen12mscr_student.sh

set -e

DATAROOT="/workspace/Cloudy-with-a-chance-of-kofta/SEN12MSCR_student"
NAME="sen12mscr_student_v1"
CHECKPOINTS_DIR="./outputs/checkpoints"

TEACHER_CKPTS_DIR="./outputs/checkpoints"
TEACHER_NAME="sen12mscr_teacher_v1"
TEACHER_EPOCH="latest"

# gamma_hall = 100 (initial default; tune after inspecting L_hall in first ~100 iters
# relative to L_G and L_H magnitudes — adjust so gamma * L_hall ~ 10 * L_G_GAN)
GAMMA_HALL=100
LAMBDA3=1.0
# Hook layers 3-7 (0=outermost, 7=innermost for unet_256 with 8 levels)
HOOK_LAYERS="3,4,5,6,7"

python train_student.py \
  --dataroot               "$DATAROOT" \
  --name                   "$NAME" \
  --model                  pix2pix_attn_student \
  --dataset_mode           unaligned_sar \
  --checkpoints_dir        "$CHECKPOINTS_DIR" \
  --teacher_checkpoints_dir "$TEACHER_CKPTS_DIR" \
  --teacher_name           "$TEACHER_NAME" \
  --teacher_which_epoch    "$TEACHER_EPOCH" \
  --niter                  50 \
  --niter_decay            50 \
  --lr_policy              lambda \
  --batchSize              1 \
  --gpu_ids                0 \
  --no_flip \
  --display_id             0 \
  --save_epoch_freq        10 \
  --display_freq           500 \
  --no_html \
  --pool_size              50 \
  --no_dropout \
  --gamma_hall             "$GAMMA_HALL" \
  --lambda3                "$LAMBDA3" \
  --hook_layers            "$HOOK_LAYERS"
