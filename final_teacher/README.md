# final_teacher — NIR teacher `v4c_ffl_sar`

A self-contained extract of the AttentionGAN teacher code needed to run one experiment:

```
scripts/train_sen12mscr_nir_v4c_ffl_sar.sh
```

Nothing else from `AttentionGAN-for-Cloud-removal/` is required. **No dataset and no
checkpoints are included** — see [What you must supply](#what-you-must-supply).

---

## What this model is

A SAR + optical cloud-removal GAN (Zhang et al. 2023, "Cloud Removal Using SAR and Optical
Images via Attention Mechanism–Based GAN"), trained here on **R/G/NIR** bands (Sentinel-2
B4/B3/B8, the LISS-4-equivalent band set) rather than true colour.

Four modules, all `unet_256` / PatchGAN with InstanceNorm:

| Module | Role |
|---|---|
| `netG2` | SAR → optical translation. Produces `fake_C` from `real_C`. |
| `netG`  | Main generator. Takes `cat([real_A, fake_C])` (cloudy optical + SAR translation) → `g_B`. |
| `netA`  | Attention / cloud-probability net → `att_A`. **Frozen oracle** here, loaded from a prior run and never trained. |
| `netD`  | SAR-conditioned PatchGAN discriminator on `cat([real_C, fake_B])`. |

Final output is the attention blend, not `g_B`:

```
fake_B = g_B * att_A + real_A * (1 - att_A)
```

`fake_C` is an *intermediate*, never the answer.

### Tensor / band conventions

- Optical channels are **(Red, Green, NIR)** = S2 (B4, B3, B8) — Red first, not band-number order.
- SAR channels are **(VV, VH, VV−VH)**, dB-clipped. Reverse of the paper's display convention;
  harmless because it is applied uniformly to every patch.
- Visuals are written in **false-colour infrared** display order (NIR into red) via
  `util.tensor2im_fcc`, because judging cloud-fill sharpness by eye with NIR-in-blue is hard.
  `real_C` and the attention maps use plain `tensor2im`.

### What `v4c_ffl_sar` adds over `v4c`

`v4c` = attention-weighted L1 + SSIM + full-image Sobel + cloud-region Sobel, frozen `netA`,
no `netD2`. This variant targets one failure mode: `netG2` trained on L1+SSIM alone regresses
to the mean, so `fake_C` is blurry, so `netG` has nothing sharp to borrow under clouds and
falls back on the (uninformative) cloudy `real_A`.

| Term | Flag | Script value | What it does |
|---|---|---|---|
| Focal Frequency Loss on `fake_B` vs `real_B` | `--lambda_ffl` | 300.0 | Spectral sharpening of the final output (Jiang et al., ICCV 2021). |
| **FFL on `fake_C` vs `real_B`** | `--lambda_ffl_g2` | 150.0 | New. Sharpens the SAR-derived intermediate at the source of the blur. |
| **SAR-reliance** | `--lambda_sar_reliance` | 20.0 | New. `mean(att_A * abs(g_B - fake_C.detach()))` — pulls `netG` toward the SAR translation inside cloud regions. `fake_C` is detached, so gradient flows only into `netG`; `att_A` is detached, so it never reshapes the mask. |
| Full-image Sobel | `--lambda_sharp` | 75.0 | Raised from v4c's 50.0 so clear-sky edges keep up. |
| Cloud-region Sobel | `--lambda_sharp_cloud` | 200.0 | 4× the full-image weight. Gradients computed on the full image, `att_A` applied *after* — masking before convolution would inject spurious mask-edge gradients. |
| Clear-pixel downweight in att-L1 | `--lambda_clear` | 0.1 | Inherited from v4c. |

FFL is forced to fp32 inside its own `autocast(enabled=False)` block: `torch.fft.fft2` has no
safe fp16 path and silently yields NaN under AMP.

---

## Layout

```
scripts/train_sen12mscr_nir_v4c_ffl_sar.sh   the run
train_nir_vis.py                             driver = train.py + periodic visual dumps
models/
  models.py                                  dispatch table (trimmed, see below)
  base_model.py  networks.py
  pix2pix_attn_nir_v4c_model.py              parent: v4c
  pix2pix_attn_nir_v4c_ffl_sar_model.py      this experiment
data/
  data_loader.py  custom_dataset_data_loader.py  base_data_loader.py
  base_dataset.py  image_folder.py  unaligned_dataset_sar.py
options/    base_options.py  train_options.py
util/       util.py  visualizer.py  image_pool.py  html.py
pytorch_ssim/__init__.py
requirements.txt  environment.yaml
```

Two files were **trimmed** relative to the originals, since the full versions dispatch to model
and dataset files that are not part of this experiment:

- `models/models.py` — only `pix2pix_attn_nir_v4c` and `pix2pix_attn_nir_v4c_ffl_sar`.
- `data/custom_dataset_data_loader.py` — `CreateDataset` only handles `unaligned_sar`.

Everything else is byte-identical to `AttentionGAN-for-Cloud-removal/`.

---

## What you must supply

The script is otherwise ready to run, but three things live outside this folder:

**1. Dataset** — `--dataroot` points at `/workspace/Cloudy-with-a-chance-of-kofta/SEN12MSCR_student`,
built by the repo-root `prepare_student_data.py`. Expected layout (`unaligned_sar` dataset mode):

```
<dataroot>/trainA/   cloudy optical PNGs   (B4/B3/B8)
<dataroot>/trainB/   clear optical PNGs    (B4/B3/B8)
<dataroot>/trainC/   SAR PNGs              (VV/VH/VV-VH)
```

**2. Frozen attention checkpoint** — `--frozen_attn_name sen12mscr_nir_teacher_v3
--frozen_attn_epoch latest`, resolved as
`<checkpoints_dir>/sen12mscr_nir_teacher_v3/latest_net_A.pth`. This is **required**; the model
raises `FileNotFoundError` without it.

**3. Warm-start checkpoint** — `--warmstart_name sen12mscr_nir_teacher_v4c_scratch
--warmstart_epoch latest`, loading `latest_net_{G,G2,D}.pth` from that directory. The run is a
continuation (`--epoch_count 121`), not a from-scratch train.

All three resolve under `--checkpoints_dir ./outputs/checkpoints`. Edit the variables at the top
of the `.sh` to repoint them.

---

## Running

```bash
cd final_teacher
source /path/to/venv/bin/activate     # needs torch + torchvision + CUDA
nohup bash scripts/train_sen12mscr_nir_v4c_ffl_sar.sh > nohup_v4c_ffl_sar.out 2>&1 &
```

Outputs land in:

```
outputs/checkpoints/sen12mscr_nir_teacher_v4c_ffl_sar/   weights + opt_train.txt + loss_log.txt
outputs/visuals_nir/sen12mscr_nir_teacher_v4c_ffl_sar/   PNG grids every 4000 images
```

Schedule: epochs **121 → 180** at lr 1e-4, `step` policy decaying ×0.1 every 30 epochs,
`batchSize 16` on one GPU, checkpoint every 10 epochs. `train_nir_vis.py` sets
`cudnn.benchmark = True` and uses AMP via `optimize_parameters_amp` (both models implement it).

`outputs/checkpoints/<name>/opt_train.txt` is written automatically by `BaseOptions.parse()` and
is the authoritative record of what a run actually used — read it rather than assuming the
script reproduces a checkpoint.

### Calibrating the new loss weights

The three new terms are **not tuned**. Watch the first ~100 iterations of the loss log:

- `ffl_g2` should not swamp `G2_L1` / `G2_SSIM` — if it does, lower `--lambda_ffl_g2`.
- `sar_reliance` should sit below `100 * G_L1` — if it dominates, lower `--lambda_sar_reliance`.

All three are logged unweighted in `get_current_errors()` as `ffl`, `ffl_g2`, `sar_reliance`.

---

## Gotchas

- **`--norm_G2 batch` in the script is inert for this model.** Only
  `pix2pix_attn_nir_v4c_model_izma` reads that flag; `v4c` builds `netG2` with `opt.norm`
  (instance). Passing it is harmless but does nothing — do not read the script and conclude
  `netG2` is BatchNorm here.
- **Checkpoint loading uses `strict=False`.** A checkpoint whose architecture flags
  (`--ngf` / `--norm` / `--which_model_netG`) don't match what is constructed loads silently
  with missing keys instead of raising. If results look like noise, verify the warm-start's
  `opt_train.txt` against this run's.
- **`netD` is conditioned on `real_C` (SAR)**, so SAR is used during training. That is fine for
  a teacher — it is exactly the privileged information the student later distils from — but it
  means this model cannot be run without SAR.
- **`v4c` writes `map_nir_v4c.npy` into the CWD** during visualisation. Cosmetic, but it means
  two concurrent runs from the same directory will overwrite each other's copy.
- Keep `--no_html --display_id 0` (already in the script). Visdom is not running, and
  `--use_html` litters the checkpoint directory with a `web/` folder.
