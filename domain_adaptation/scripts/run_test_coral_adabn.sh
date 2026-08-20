#!/usr/bin/env bash
# CORAL-adapted student + AdaBN on the Final/ test set.
#
# Identical model/adapter config to run_coral_adabn.sh — same CORAL weights,
# --method dataset_norm, --model_variant coral, --adabn 50, --save_patches.
# Only the roots differ, so the test set never mixes with the main run:
#     Bhoonidhi-Data/test_data/{data_scaled,data_rect} -> test_data/outputs_adabn
#
# Prereq: python domain_adaptation/scripts/prep_test_data.py --group Cloud
#
# Run from repo root:
#   source AttentionGAN-for-Cloud-removal/.venv/bin/activate
#   bash domain_adaptation/scripts/run_test_coral_adabn.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TEST="${REPO_ROOT}/Bhoonidhi-Data/test_data"
CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/domain_adaptation/methods/outputs/checkpoints}"
WEIGHTS="${CKPT_ROOT}/sen12mscr_student_coral_adapted/latest_net_G.pth"

python "${REPO_ROOT}/domain_adaptation/run_inference.py" \
  --method        dataset_norm \
  --model_variant coral \
  --scaled_root   "${TEST}/data_scaled" \
  --weights       "${WEIGHTS}" \
  --out_root      "${TEST}/outputs_adabn" \
  --rect_root     "${TEST}/data_rect" \
  --batch_size    64 \
  --num_workers   8 \
  --adabn \
  --adabn_batches 50 \
  "$@"
