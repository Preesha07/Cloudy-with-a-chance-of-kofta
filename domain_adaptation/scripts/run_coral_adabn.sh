#!/usr/bin/env bash
# CORAL-adapted student inference + AdaBN (warms up InstanceNorm running stats on
# LISS-4 batches before inference, on top of the always-on TTBN).
#
# Same as run_coral.sh except for --adabn and a separate OUT_ROOT, so results
# never overwrite the plain-TTBN run. To revert to plain TTBN, just run
# run_coral.sh instead (or drop --adabn from this file) — no other change needed.
#
# Run from repo root:
#   source AttentionGAN-for-Cloud-removal/.venv/bin/activate
#   bash domain_adaptation/scripts/run_coral_adabn.sh

set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/domain_adaptation/methods/outputs/checkpoints}"
WEIGHTS="${CKPT_ROOT}/sen12mscr_student_coral_adapted/latest_net_G.pth"
SCALED_ROOT="${REPO_ROOT}/Bhoonidhi-Data/data_scaled"
RECT_ROOT="${REPO_ROOT}/Bhoonidhi-Data/data_rect"
OUT_ROOT="${REPO_ROOT}/Bhoonidhi-Data/outputs_adabn"

python "${REPO_ROOT}/domain_adaptation/run_inference.py" \
  --method        dataset_norm \
  --model_variant coral \
  --scaled_root   "${SCALED_ROOT}" \
  --weights       "${WEIGHTS}" \
  --out_root      "${OUT_ROOT}" \
  --rect_root     "${RECT_ROOT}" \
  --batch_size    64 \
  --num_workers   8 \
  --save_patches \
  --adabn \
  --adabn_batches 50
