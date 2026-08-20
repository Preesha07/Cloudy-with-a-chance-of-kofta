# domain_adaptation — CORAL-adapted student on LISS-4

Clean extract of the **final** domain-adaptation path from `domain_adapt/`: the CORAL-adapted
student run with TTBN + AdaBN. The earlier adapters (histogram, FDA, cloud_quantile, baseline)
and the `base`/`fsp` student variants are **not** carried over — they live in `domain_adapt/` if
you need them.

Self-contained: the student model code is vendored under `student/`, so nothing here imports
from the repo-root `student/` or `khushi/student/` trees. **No datasets, no checkpoints.**

## Layout

```
run_inference.py              tile → adapt → student → stitch, per scene
pipeline/
  io.py                       LazyPatchDataset (BAND2/3/4 merge + on-the-fly tiling)
  model.py                    loads netG/netA/netH; apply_ttbn / run_adabn
  stitch.py                   Stitcher (Hann-window overlap-add), manifest/shear helpers
  starters_for_data_prep/     bhoonidhi_rectangle.py, scale_liss4_data.py
methods/
  base.py  dataset_norm.py    adapter interface + the TTBN adapter (the only one kept)
  coral/                      CoralStudentModel, coral_loss, datasets, patch_filter, train.py
student/                      vendored student model code (models/options/util/pytorch_ssim)
scripts/
  prep_test_data.py           test_data/Final/ zips → rectify → scale
  train_coral.sh              CORAL fine-tuning
  run_coral_adabn.sh          inference on Bhoonidhi-Data/data_scaled
  run_test_coral_adabn.sh     inference on the Final/ test set
```

## What the method is

`--method dataset_norm --model_variant coral --adabn 50`:

1. **CORAL fine-tune** (`methods/coral/train.py`) — filters LISS-4 patches by `netA` cloud
   fraction (< 0.30), then fine-tunes the student to align second-order feature statistics
   (covariance) between SEN12MS-CR and LISS-4 activations.
2. **TTBN** (always on, `apply_ttbn`) — sets `track_running_stats=False` on every norm layer.
   The student's `norm='instance'` builds `InstanceNorm2d(affine=False, track_running_stats=True)`,
   which at eval uses SEN12MS-CR running stats on LISS-4 input — that is what produced
   near-black outputs. TTBN restores training-time per-instance behaviour.
3. **AdaBN** (`--adabn`) — re-accumulates `BatchNorm2d` running stats over LISS-4 batches.
   A genuine no-op for InstanceNorm-only nets; live for the CORAL variant's BN layers.

## Running

```bash
source AttentionGAN-for-Cloud-removal/.venv/bin/activate

# fine-tune (writes to $CKPT_ROOT)
bash domain_adaptation/scripts/train_coral.sh

# inference
bash domain_adaptation/scripts/run_coral_adabn.sh        # bulk scenes
bash domain_adaptation/scripts/run_test_coral_adabn.sh   # Final/ test set
```

**Checkpoints**: the scripts default `CKPT_ROOT` to
`domain_adaptation/methods/outputs/checkpoints`, which is empty here. Either train into it, or
reuse the existing weights:

```bash
CKPT_ROOT=domain_adapt/methods/outputs/checkpoints bash domain_adaptation/scripts/run_coral_adabn.sh
```

`train_coral.sh` also needs the teacher at
`AttentionGAN-for-Cloud-removal/outputs/checkpoints/sen12mscr_nir_teacher_v4c/`.

Outputs land in `<out_root>/dataset_norm/<scene_id>/`: `final.jpg`, `final_para.jpg` (re-sheared
to pushbroom shape, needs `--rect_root`), `fake_b.jpg`, `fake_c.jpg`, `g_b.jpg`, `attention.jpg`.

## Gotchas

- **Band order**: `R = BAND3`, `G = BAND2`, `B = BAND4 (NIR)` — LISS-4 band numbers are off by
  one from Sentinel-2's, and a swap yields a plausible but wrong-coloured image with nothing to
  flag it.
- **Use `Stitcher`, not the bare `accumulate()`** in `pipeline/stitch.py`. `accumulate()` bumps
  the shared weight map per call, so N canvases over-count N-fold and every output comes out at
  1/N. That bug produced all pre-2026-08-07 outputs at 1/5 contrast. The bias-correction block
  in `run_inference.py` is now a near-no-op left in to catch real DC offsets.
- **`prep_test_data.py` rewrites `scene_id` in each `manifest.csv`** to the folder name.
  `run_inference.py` looks shear params up by scaled-dir name; without the rewrite the lookup
  misses and `final_para.jpg` is silently skipped.
- `--scratch` defaults inside the workspace volume on purpose — `/tmp` is a 20 GiB overlay and
  scenes unzip to ~1.7 GiB each.
