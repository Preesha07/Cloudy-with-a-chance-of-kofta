#!/usr/bin/env bash
# CORAL domain adaptation fine-tuning.
#
# Filters LISS-4 patches by cloud coverage (netA, threshold 30%), then
# fine-tunes the student model with plain Deep CORAL on the clear patches.
#
# Run from repo root:
#   source AttentionGAN-for-Cloud-removal/.venv/bin/activate
#   bash domain_adaptation/scripts/train_coral.sh

set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CORAL_DIR="${REPO_ROOT}/domain_adaptation/methods/coral"
CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/domain_adaptation/methods/outputs/checkpoints}"

python "${CORAL_DIR}/train.py" \
  --dataroot              "${REPO_ROOT}/SEN12MSCR_prepared" \
  --dataroot_liss4        "${REPO_ROOT}/Bhoonidhi-Data/data_images/R2F13APR2026077748011600054SSANSTUC00GTDD" \
  --cloud_threshold       0.30 \
  --checkpoints_dir       "${CKPT_ROOT}" \
  --name                  sen12mscr_student_coral_adapted \
  --teacher_checkpoints_dir "${REPO_ROOT}/AttentionGAN-for-Cloud-removal/outputs/checkpoints" \
  --teacher_name          sen12mscr_nir_teacher_v4c \
  --teacher_which_epoch   latest \
  --lr                    1e-5 \
  --epochs                15 \
  --batch_size            16 \
  --coral_weight          1.0 \
  --gamma_hall            1.0 \
  --cross_depth_weight    1.0 \
  --freqkd_weight         1.0 \
  --save_epoch_freq       5 \
  --print_freq            10
