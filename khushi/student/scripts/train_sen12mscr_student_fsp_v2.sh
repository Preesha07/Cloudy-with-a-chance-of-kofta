#!/usr/bin/env bash
# FSP v2 — weight=2.0, drop the two deepest (near-rank-degenerate) layer pairs.
#
# What changed from fsp_v1 (cross_depth_weight=8.0, fsp_min_positions=0):
#
#   1. CROSS_DEPTH_WEIGHT: 8.0 → 2.0
#      The v1 run peaked at epoch 10 then degraded to 25.52 dB by epoch 30
#      (−0.62 dB from peak). The bottleneck pair (6-7) collapsed 96% in 30
#      epochs, deforming netH faster than the outer layers could adapt.
#      Lower weight reduces that pressure.
#
#   2. FSP_MIN_POSITIONS: 0 → 17
#      Drops pairs whose deeper layer has < 17 spatial positions.
#      Pair 5-6: 4×4 = 16 positions → DROPPED (< 17)
#      Pair 6-7: 2×2 =  4 positions → DROPPED (< 17)
#      Pair 3-4: 16×16 = 256 positions → kept
#      Pair 4-5:  8×8  =  64 positions → kept
#      Motivation: the SAR linear probe showed layers 4-6 carry the signal;
#      layer 7 (innermost) is at the random-init floor. The rank analysis
#      (rank(G) ≤ H_j×W_j) flags 5-6 and 6-7 as near-degenerate. Those
#      pairs were the ones converging too fast in v1 — dropping them isolates
#      the two pairs that are both well-conditioned and information-bearing.
#
# Everything else is held identical to fsp_v1 and the baseline (v1):
#   - gamma_hall 100, no DIST, hook_layers 3-7 (3,4,5,6,7 still hooked for
#     L_hall; FSP simply skips the 5-6 and 6-7 pairs at loss-computation time)
#   - same teacher, same dataset, same LR schedule, same epoch budget
#
# Run from khushi/student/:
#   source ../../AttentionGAN-for-Cloud-removal/.venv/bin/activate
#   bash scripts/train_sen12mscr_student_fsp_v2.sh 2>&1 | tee train_fsp_v2.log

set -e

DATAROOT="/workspace/Cloudy-with-a-chance-of-kofta/SEN12MSCR_student"
NAME="sen12mscr_student_fsp_v2"
CHECKPOINTS_DIR="./outputs/checkpoints"

TEACHER_CKPTS_DIR="./outputs/checkpoints"
TEACHER_NAME="sen12mscr_teacher_v1"
TEACHER_EPOCH="latest"

GAMMA_HALL=100
LAMBDA3=1.0

CROSS_DEPTH_WEIGHT=2.0
FSP_DISTANCE="pearson"
FSP_PAIRS="adjacent"
FSP_MIN_POSITIONS=17   # drops 5-6 (16 positions) and 6-7 (4 positions)
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
