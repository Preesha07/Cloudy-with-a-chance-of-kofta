#!/usr/bin/env bash
# CORAL domain adaptation fine-tuning — how the final weights were produced.
#
# Filters LISS-4 patches by cloud coverage (netA, threshold 30%), then
# fine-tunes the student with Deep CORAL on the clear patches.
#
# Starts from the frozen teacher checkpoint (sen12mscr_nir_teacher_v4c):
# netG/netA/netD are loaded from it as initialization, netH is warm-started
# from the teacher's netG2 with its first conv re-initialized.
#
# Run from repo root:
#   source AttentionGAN-for-Cloud-removal/.venv/bin/activate
#   bash final_student/scripts/train.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PKG="${REPO_ROOT}/final_student"

python "${PKG}/coral/train.py" \
  --dataroot                "${REPO_ROOT}/SEN12MSCR_prepared" \
  --dataroot_liss4          "${REPO_ROOT}/Bhoonidhi-Data/data_images/R2F13APR2026077748011600054SSANSTUC00GTDD" \
  --cloud_threshold         0.30 \
  --checkpoints_dir         "${PKG}/outputs/checkpoints" \
  --name                    sen12mscr_student_coral_adapted \
  --teacher_checkpoints_dir "${REPO_ROOT}/AttentionGAN-for-Cloud-removal/outputs/checkpoints" \
  --teacher_name            sen12mscr_nir_teacher_v4c \
  --teacher_which_epoch     latest \
  --lr                      1e-5 \
  --epochs                  15 \
  --batch_size              16 \
  --coral_weight            1.0 \
  --gamma_hall              1.0 \
  --cross_depth_weight      1.0 \
  --freqkd_weight           1.0 \
  --save_epoch_freq         5 \
  --print_freq              10 \
  "$@"
