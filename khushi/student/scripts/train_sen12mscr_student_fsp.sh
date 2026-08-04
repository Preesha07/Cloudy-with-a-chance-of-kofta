#!/usr/bin/env bash
# Train the SAR-free student with cross-depth FSP covariance distillation
# (Yim et al., CVPR 2017) added on top of the Hoffman exact-match term.
#
# Difference from train_sen12mscr_student.sh: adds L_fsp, which matches the
# teacher's INTER-LAYER structure (how features evolve between hooked depths)
# rather than only comparing activations at the same depth. The within-layer
# DIST terms are off by default here (--no_dist), so this run isolates
# "exact match + cross-depth" against the v1 baseline's "exact match" alone.
#
# Prerequisites:
#   1. Teacher trained:  outputs/checkpoints/sen12mscr_teacher_v1/latest_net_*.pth
#      (verified 2026-08-03: 'latest' is epoch 23, 32.72 dB on the 635-tile val
#       split — the best teacher checkpoint in the repo)
#   2. Student data prepared — SEN12MSCR_student/ already exists, do not re-run.
#
# Run from khushi/student/:
#   source ../../AttentionGAN-for-Cloud-removal/.venv/bin/activate
#   bash scripts/train_sen12mscr_student_fsp.sh

set -e

DATAROOT="/workspace/Cloudy-with-a-chance-of-kofta/SEN12MSCR_student"
NAME="sen12mscr_student_fsp_v1"
CHECKPOINTS_DIR="./outputs/checkpoints"

TEACHER_CKPTS_DIR="./outputs/checkpoints"
TEACHER_NAME="sen12mscr_teacher_v1"
TEACHER_EPOCH="latest"

# ── Distillation terms ────────────────────────────────────────────────────────
# GAMMA_HALL=100 matches the v1 baseline exactly, so the only difference between
# this run and sen12mscr_student_v1 is the added cross-depth term.
GAMMA_HALL=100
LAMBDA3=1.0

# CROSS_DEPTH_WEIGHT: calibrated from a real run, not estimated.
#
# The a-priori guess was L_fsp ~ 4 at init (4 pairs x 1-rho, rho~0 for unrelated
# nets) which suggested weight 2.0. That was wrong: netH is WARM-STARTED from the
# teacher's netG2, so the FSP matrices already correlate strongly from iteration 1
# -- measured 1-rho ~ 0.15-0.39 per pair, L_fsp ~ 0.97 total, not ~4.
#
# Measured over the first 700 iters of the weight-2.0 run:
#     L_fsp    ~0.97  -> weighted  1.95   (4x under target)
#     L_hall   ~0.077 -> weighted  7.7    (gamma_hall 100)
#     L_G_GAN  ~0.76  -> 10x       7.6    <- the target scale
# 8.0 puts the cross-depth term at ~7.8, matching both.
#
# If you change --fsp_distance to frobenius this needs recalibrating from scratch;
# the two distances are not on the same scale.
CROSS_DEPTH_WEIGHT=8.0
FSP_DISTANCE="pearson"     # or 'frobenius' for the Yim et al. original
FSP_PAIRS="adjacent"       # 4 pairs (3-4,4-5,5-6,6-7); 'all' gives 10
FSP_MIN_POSITIONS=0        # 0 = keep all pairs; see the rank caveat below

# Hook layers 3-7 (0=outermost, 7=innermost for unet_256 with 8 levels).
# Measured geometry (out-half channels @ spatial): 256@32x32, 512@16x16,
# 512@8x8, 512@4x4, 512@2x2. rank(G) <= H_j*W_j, so pairs 5-6 and 6-7 build
# 512x512 matrices from only 16 and 4 spatial samples. The per-pair diagnostics
# in fsp_metrics.txt show whether those pairs carry signal; set
# FSP_MIN_POSITIONS=16 or 64 to drop them if they do not.
HOOK_LAYERS="3,4,5,6,7"

python train_student_fsp.py \
  --dataroot               "$DATAROOT" \
  --name                   "$NAME" \
  --model                  pix2pix_attn_student_fsp \
  --dataset_mode           unaligned_sar \
  --checkpoints_dir        "$CHECKPOINTS_DIR" \
  --teacher_checkpoints_dir "$TEACHER_CKPTS_DIR" \
  --teacher_name           "$TEACHER_NAME" \
  --teacher_which_epoch    "$TEACHER_EPOCH" \
  --niter                  15 \
  --niter_decay            15 \
  --lr_policy              lambda \
  --batchSize              1 \
  --gpu_ids                0 \
  --no_flip \
  --display_id             0 \
  --save_epoch_freq        5 \
  --display_freq           500 \
  --no_html \
  --pool_size              50 \
  --no_dropout \
  --gamma_hall             "$GAMMA_HALL" \
  --lambda3                "$LAMBDA3" \
  --hook_layers            "$HOOK_LAYERS" \
  --no_dist \
  --use_cross_depth \
  --cross_depth_weight     "$CROSS_DEPTH_WEIGHT" \
  --fsp_distance           "$FSP_DISTANCE" \
  --fsp_pairs              "$FSP_PAIRS" \
  --fsp_min_positions      "$FSP_MIN_POSITIONS"
