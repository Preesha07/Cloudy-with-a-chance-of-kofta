#!/usr/bin/env bash
# Final model inference: CORAL-adapted student + dataset_norm adapter + AdaBN.
#
# This reproduces the run that produced the newest LISS-4 outputs
# (Bhoonidhi-Data/test_data/outputs_adabn/dataset_norm/).
#
# Prereq: scenes rectified + scaled into ${TEST}/data_rect and ${TEST}/data_scaled
#         (see final_student/README.md, "Preparing input scenes").
#
# Run from repo root:
#   source AttentionGAN-for-Cloud-removal/.venv/bin/activate
#   bash final_student/scripts/run_inference.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PKG="${REPO_ROOT}/final_student"
TEST="${REPO_ROOT}/Bhoonidhi-Data/test_data"

# Trained weights are NOT shipped inside final_student/. Point this at the
# checkpoint dir holding latest_net_{G,A,H,D}.pth.
WEIGHTS="${WEIGHTS:-${REPO_ROOT}/domain_adapt/methods/outputs/checkpoints/sen12mscr_student_coral_adapted/latest_net_G.pth}"

if [[ ! -f "${WEIGHTS}" ]]; then
  echo "ERROR: weights not found at ${WEIGHTS}" >&2
  echo "       Set WEIGHTS=/path/to/latest_net_G.pth and re-run." >&2
  exit 1
fi

python "${PKG}/run_inference.py" \
  --method        dataset_norm \
  --scaled_root   "${TEST}/data_scaled" \
  --weights       "${WEIGHTS}" \
  --out_root      "${TEST}/outputs_adabn" \
  --rect_root     "${TEST}/data_rect" \
  --batch_size    64 \
  --num_workers   8 \
  --adabn \
  --adabn_batches 50 \
  "$@"
