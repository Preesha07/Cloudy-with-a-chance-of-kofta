#!/usr/bin/env bash
# Train the SAR-free student with DIST relational distillation (B4/B3/B8 channels).
#
# Difference from train_sen12mscr_student.sh: the hallucination match is done with
# Pearson-correlation relational losses (Huang et al. NeurIPS 2022) instead of the
# exact sigmoid-L2 match. GAMMA_HALL=0 below means "pure DIST"; set it >0 to blend.
#
# Prerequisites:
#   1. Teacher trained:  outputs/checkpoints/sen12mscr_teacher_v1/latest_net_*.pth
#   2. Student data prepared (run once, from the repo root):
#        python prepare_student_data.py \
#            --prepared  SEN12MSCR_prepared \
#            --sen12mscr SEN12MSCR \
#            --out       SEN12MSCR_student \
#            --workers   8
#
# Run from khushi/student/:
#   source ../../AttentionGAN-for-Cloud-removal/.venv/bin/activate
#   bash scripts/train_sen12mscr_student_dist.sh

set -e

DATAROOT="/workspace/Cloudy-with-a-chance-of-kofta/SEN12MSCR_student"
NAME="sen12mscr_student_dist_v1"
CHECKPOINTS_DIR="./outputs/checkpoints"

TEACHER_CKPTS_DIR="./outputs/checkpoints"
TEACHER_NAME="sen12mscr_teacher_v1"
TEACHER_EPOCH="latest"

# ── Distillation weights ──────────────────────────────────────────────────────
# GAMMA_HALL=0  -> pure DIST (the intended ablation against the v1 baseline,
#                  which ran GAMMA_HALL=100 with no DIST terms).
# DIST_BETA / DIST_GAMMA start at the paper's classification values (2/2). These
# are almost certainly too low here: L_G carries a 100x L1 term, so a relational
# loss of order 1 contributes little. Watch L_inter / L_intra in the first ~100
# iterations (both are logged unweighted) and scale up so that
# beta*L_inter ~ 10 * L_G_GAN, the same rule used to pick GAMMA_HALL=100.
GAMMA_HALL=0
DIST_BETA=2.0
DIST_GAMMA=2.0
DIST_ACT="none"

LAMBDA3=1.0
# Hook layers 3-7 (0=outermost, 7=innermost for unet_256 with 8 levels)
HOOK_LAYERS="3,4,5,6,7"

python train_student_dist.py \
  --dataroot               "$DATAROOT" \
  --name                   "$NAME" \
  --model                  pix2pix_attn_student_dist \
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
  --hook_layers            "$HOOK_LAYERS" \
  --dist_beta              "$DIST_BETA" \
  --dist_gamma             "$DIST_GAMMA" \
  --dist_act               "$DIST_ACT"
