# `final_student/` — the final cloud-removal student model

This folder is a **self-contained copy of the model that produced the final LISS-4
inference results**. Everything it imports lives inside this folder — nothing is pulled
from `student/`, `khushi/student/`, or `domain_adapt/`. You can read, run, hand off, or
archive this directory on its own.

It was assembled by tracing the actual import chain of the inference entry point, not by
copying whatever looked relevant. See [What "final" means](#what-final-means) for the
provenance and for two things people commonly get wrong about it.

---

## What "final" means

The final results are the ones under
`Bhoonidhi-Data/test_data/outputs_adabn/dataset_norm/` — the CORAL-adapted student run
with the `dataset_norm` adapter and AdaBN, over the hand-curated `Final/` NER test set.

The model class chain that inference actually instantiates:

| Level | Class | File in this folder |
|---|---|---|
| top (loaded) | `CoralStudentModel` | `coral/coral_model.py` |
| parent | `Pix2Pix_attn_Student_FSP_FFL_Model` | `coral/pix2pix_attn_student_fsp_ffl_model.py` |
| grandparent | `Pix2Pix_attn_Student_FSP_Model` | `models/pix2pix_attn_student_fsp_model_changed.py` |
| base | `BaseModel` | `models/base_model.py` |
| layers | `define_G` / `define_G2` / `define_D`, U-Net + ResNet blocks | `models/networks.py` |

**Two things that are easy to get wrong:**

1. **The final model is not "the FSP-changed model".** `Pix2Pix_attn_Student_FSP_Model`
   (from `pix2pix_attn_student_fsp_model_changed.py`) is the *grandparent base class*.
   The object actually built and loaded is `CoralStudentModel`, which adds FFL and Deep
   CORAL on top, and its weights come from CORAL fine-tuning — not from any FSP run.
2. **The FSP-changed file came from `khushi/student/models/`, not `student/models/`.**
   Both trees contain a file of that name and **they differ**. The original loader put
   `khushi/student` ahead of `student` on `sys.path`, so `khushi/`'s copy is the one that
   was imported. That is the copy vendored here. (`domain_adapt/methods/coral/` also had a
   byte-identical copy, but it was *not* the one imported — the import is
   `from models.pix2pix_attn_student_fsp_model_changed`, which resolved through
   `khushi/student/models/`.)

### Deliberately left out

`domain_adapt/methods/coral/` also contains `train_coral.py`,
`pix2pix_attn_student_fsp_ffl_coral_model.py`, and `aligned_coral_dataset.py`. These are a
**superseded parallel training branch** (also duplicated under `vandita/`) — no inference
path loads them, and the final weights did not come from them. They are not copied here.
The live training driver is `coral/train.py`, which builds `CoralStudentModel`.

Likewise only the `dataset_norm` adapter is included, since that is what the final runs
used. The exploratory adapters (`histogram`, `fda`, `cloud_quantile`, `baseline`) remain in
`domain_adapt/methods/`.

---

## Folder layout

```
final_student/
├── README.md                 ← you are here
├── run_inference.py          ← INFERENCE ENTRY POINT
│
├── coral/                    ← the CORAL layer + training code (flat modules)
│   ├── coral_model.py                        CoralStudentModel — the final model
│   ├── pix2pix_attn_student_fsp_ffl_model.py parent: adds Focal Frequency Loss
│   ├── coral_loss.py                         Deep CORAL covariance-alignment loss
│   ├── train.py                              TRAINING ENTRY POINT
│   ├── dataset.py                            CoralDataset (SEN12MS-CR + LISS-4 batches)
│   └── patch_filter.py                       netA cloud-fraction gate on LISS-4 patches
│
├── models/                   ← base model + network definitions
│   ├── pix2pix_attn_student_fsp_model_changed.py   base class (FSP + DIST + FreqKD + hall)
│   ├── base_model.py                               save/load plumbing
│   └── networks.py                                 define_G/G2/D, U-Net, ResNet, GAN losses
│
├── pipeline/                 ← inference plumbing
│   ├── model.py              model loading, TTBN, AdaBN, torch.compile
│   ├── io.py                 LazyPatchDataset (on-the-fly tiling), tensor↔image helpers
│   └── stitch.py             Stitcher (Hann-window overlap-add), pushbroom re-shear
│
├── methods/                  ← input/output domain adaptation
│   ├── base.py               BaseAdapter contract
│   └── dataset_norm.py       the adapter used for the final results (pass-through + TTBN)
│
├── util/, pytorch_ssim/      ← vendored helpers the model classes need
└── scripts/
    ├── run_inference.sh      reproduces the final inference run
    └── train.sh              reproduces the CORAL fine-tuning that made the weights
```

---

## Weights are not included

Per request, no `.pth` files are shipped here. The final checkpoint lives at:

```
domain_adapt/methods/outputs/checkpoints/sen12mscr_student_coral_adapted/
    latest_net_G.pth      latest_net_A.pth      latest_net_H.pth      latest_net_D.pth
    epoch_{5,10,15}_net_{G,A,H,D}.pth
```

Point `--weights` at the `latest_net_G.pth` file. **`--weights` names a file but is used as
a directory + epoch tag**: the parent directory becomes the checkpoint dir and experiment
name, and the epoch tag is the filename prefix, so `latest_net_G.pth` → epoch `latest` and
`epoch_15_net_G.pth` → epoch `epoch`. `netG`, `netA`, and `netH` are then all loaded by the
model wrapper — the single `_net_G` filename is just a handle for all four.

Verified: all three networks load with **0 missing, 0 unexpected, 0 mismatched** keys
through this copy.

---

## Running inference

```bash
# From repo root
source AttentionGAN-for-Cloud-removal/.venv/bin/activate

# Easiest — reproduces the final run:
bash final_student/scripts/run_inference.sh

# Or point it at your own weights:
WEIGHTS=/path/to/latest_net_G.pth bash final_student/scripts/run_inference.sh

# Or call it directly:
python final_student/run_inference.py \
    --method        dataset_norm \
    --scaled_root   Bhoonidhi-Data/test_data/data_scaled \
    --weights       domain_adapt/methods/outputs/checkpoints/sen12mscr_student_coral_adapted/latest_net_G.pth \
    --out_root      Bhoonidhi-Data/test_data/outputs_adabn \
    --rect_root     Bhoonidhi-Data/test_data/data_rect \
    --batch_size    64 --num_workers 8 \
    --adabn --adabn_batches 50
```

### Flags worth knowing

| Flag | Notes |
|---|---|
| `--scaled_root` | Root with per-scene subdirs, each holding `BAND2/BAND3/BAND4/` scaled JPEGs. Tiling happens on the fly — there is **no separate slicing step**. |
| `--rect_root` | Root with per-scene `manifest.csv`. Needed only for `final_para.jpg`; if the lookup misses, that output is **silently skipped**. |
| `--scene` | Process a single scene instead of all of them. |
| `--adabn` / `--adabn_batches` | BatchNorm warmup on LISS-4 batches. Near-no-op for this InstanceNorm-only model (TTBN already covers it) — kept because it was part of the final run. Omitting the flag is the one-click revert. |
| `--stitch_device` | `auto` (default) keeps stitch canvases on GPU when they fit in free VRAM with 3 GiB spare, else CPU. Worth ~9× end-to-end. |
| `--save_patches` | Dumps every 256×256 output tile. ~24k JPEG writes per scene — **leave it off** unless debugging. |
| `--batch_size` | 64 is what the final run used. TTBN estimates variance per batch, so very small batches degrade output. |

### Outputs

One folder per scene at `<out_root>/<method>/<scene_id>/`:

| File | What it is |
|---|---|
| `final.jpg` | full-scene cloud removal output, after domain restoration |
| `final_para.jpg` | `final.jpg` re-sheared back to the original pushbroom parallelogram |
| `fake_b.jpg` | raw model composite, before domain restoration |
| `fake_c.jpg` | `netH`'s hallucinated SAR-like image |
| `g_b.jpg` | raw `netG` decoder output, before attention blending |
| `attention.jpg` | `att_A` cloud-probability heatmap |

The final composite is `fake_B = g_B · att_A + real_A · (1 − att_A)` — `g_b.jpg` and
`attention.jpg` are the two halves of that blend, which makes them the right things to look
at when output looks wrong.

---

## Preparing input scenes (upstream of this folder)

`run_inference.py` starts from **already rectified and scaled** scenes, so raw-zip
preparation is not part of this folder. Use the existing repo scripts:

```bash
# Bhoonidhi .zip → data_rect/ (rectified per-band TIFs + manifest.csv) → data_scaled/
python domain_adapt/scripts/prep_test_data.py --group Cloud --skip_done
```

That walks `Bhoonidhi-Data/test_data/Final/<group>/<NN City>/<ProductID>.zip`, runs
rectify → scale (5.8 m → 10 m GSD, matching Sentinel-2), and names each output folder
`<NN>_<City>__<ProductID>`. It also rewrites `scene_id` inside each `manifest.csv` to that
folder name — without that rewrite the shear lookup misses and `final_para.jpg` is silently
skipped.

**Note:** `--scratch` defaults inside the workspace volume on purpose. `/tmp` here is a
20 GiB overlay and scenes unzip to ~1.7 GiB each, so parallel workers will fill it.

---

## Retraining

```bash
source AttentionGAN-for-Cloud-removal/.venv/bin/activate
bash final_student/scripts/train.sh
```

Checkpoints land in `final_student/outputs/checkpoints/sen12mscr_student_coral_adapted/`
(the script's default; the original run wrote to `domain_adapt/methods/outputs/checkpoints/`).

### What training does

Deep CORAL unsupervised domain adaptation, Sentinel-2 (source) → LISS-4 (target):

1. `patch_filter.filter_liss4_patches()` runs `netA` over LISS-4 patches and discards any
   with mean cloud probability ≥ `--cloud_threshold` (0.30). CORAL then never sees a cloudy
   target patch, so no spatial masking is needed inside the loss.
2. Each step draws a paired batch: SEN12MS-CR source patches plus a `LISS4` key of filtered
   target patches. **If `LISS4` is absent, CORAL is silently skipped** and the model behaves
   exactly like its FSP+FFL parent — which is what happens at inference.
3. CORAL aligns second-order statistics of `netG`'s **encoder** activations at blocks
   `(2, 3, 4)` (64²×128ch, 32²×256ch, 16²×512ch), computed independently per block and
   summed. An earlier
   version aligned `netH`'s output instead — that is the SAR-hallucination output space, not
   the shared representation, which is why it moved.

The total objective, all backpropagated together:

```
L = L_H + L_G
  + gamma_hall         · L_hall     (Hoffman modality-hallucination match)
  + L_dist                          (relaxed Pearson-correlation distillation)
  + cross_depth_weight · L_fsp      (FSP cross-depth covariance, Yim et al.)
  + freqkd_weight      · L_freqkd   (radial low/high-frequency FFT split)
  + FFL_WEIGHT         · L_ffl      (focal frequency loss)
  + CORAL_WEIGHT       · L_coral    (Deep CORAL, Sun & Saenko)
```

Training initializes from the frozen teacher `sen12mscr_nir_teacher_v4c`: `netG`/`netA`/
`netD` are loaded from it, and `netH` is warm-started from the teacher's `netG2` with only
its first conv re-initialized (deeper layers already produce optical-like statistics and
transfer; the first conv saw SAR and does not). `netA` stays frozen.

Four loss weights are CLI flags on `coral/train.py` (`--coral_weight`, `--gamma_hall`,
`--cross_depth_weight`, `--freqkd_weight`). Two are **class attributes, not flags** —
edit them in place: `FFL_WEIGHT = 300.0` in `coral/pix2pix_attn_student_fsp_ffl_model.py`
(1.0 makes FFL negligible against `100·L_G_L1`; 300 puts it in range), and
`CORAL_BLOCK_INDICES` / `CORAL_WEIGHT` in `coral/coral_model.py`.

---

## Things that will bite you

- **Architecture options are reconstructed by hand.** CORAL training never wrote an
  `opt.pkl`, so `pipeline/model.py:ModelOpt` hardcodes the architecture to match
  `coral/train.py`'s defaults. `_load_from_dir` uses `strict=False`, so a mismatch loads
  **silently with missing keys** and produces noise rather than raising. If output looks
  like garbage, check `ModelOpt` against the training flags first.
- **`which_model_netA` is `resnet_9blocks`, not `unet_256`.** It is the one module that
  differs, and it is the single easiest thing to break here.
- **Band order is R = BAND3, G = BAND2, B = BAND4.** LISS-4 band numbering is off by one
  from Sentinel-2's: `LISS4 BANDn ≡ S2 B(n+1)`. The tensor is Red-first (R, G, NIR) to match
  the B4/B3/B8 order the student was trained on. Get this wrong and the output is plausible
  but wrong-coloured, with nothing to flag it.
- **`norm='instance'` here builds `InstanceNorm2d(affine=False, track_running_stats=True)`**
  — *not* the PyTorch default. With running stats on, it behaves like BatchNorm at eval time
  and applies SEN12MS-CR-domain statistics to LISS-4 input, which produced near-black
  outputs. `pipeline/model.py:apply_ttbn` fixes this by permanently setting
  `track_running_stats=False`, which reproduces training-time behaviour exactly.
  `run_inference.py` calls it on all three nets — do not remove it.
- **Use `Stitcher`, never the bare `accumulate()` free function, with multiple canvases.**
  `accumulate()` bumps the shared weight map on every call, so N canvases over-count the
  weight N-fold and every output comes out at 1/N of its true value. This was a real bug
  (5 canvases → everything at 1/5 brightness). `run_inference.py` uses `Stitcher` correctly.
  **Any output under `Bhoonidhi-Data/outputs*/` produced before 2026-08-07 is
  contrast-compressed** and should be regenerated before comparison.
- **The bias-correction block in `run_inference.py` is now a near-no-op** (it fell from
  +61.5 to +0.9 per channel once the stitch bug was fixed). It is kept only to catch a
  genuine DC offset. Its in-file comment blaming InstanceNorm running stats is a leftover
  from the wrong diagnosis.
- **The pipeline is lossy end to end** (JPEG at quality 95/100 at several stages). Do not
  use it to produce numbers you intend to report as reconstruction metrics.
- **There is no ground truth for LISS-4.** The `Cloud Free` scenes in the test set are
  same-city visual references at different dates and orbits, **not** co-registered ground
  truth — they do not license PSNR/SSIM against the cloudy outputs. All quantitative claims
  in this project come from the SEN12MS-CR val split.

---

## Verifying the copy still works

```bash
source AttentionGAN-for-Cloud-removal/.venv/bin/activate
python -c "
import sys, torch; sys.path.insert(0, 'final_student')
from pipeline.model import load_student_model, apply_ttbn
W='domain_adapt/methods/outputs/checkpoints/sen12mscr_student_coral_adapted/latest_net_G.pth'
netG, netA, netH, dev = load_student_model(W)
for n in (netG, netA, netH): apply_ttbn(n)
x = torch.randn(2, 3, 256, 256, device=dev)
with torch.no_grad():
    fc = netH(x); att = netA(x); gb = netG(torch.cat([x, fc], 1))
print('OK', fc.shape, att.shape, gb.shape)
"
```

Expect `fake_c (2,3,256,256)`, `att (2,1,256,256)`, `g_b (2,3,256,256)`.
