# AttentionGAN Teacher — Experiment Log & Context

**Paper:** Zhang et al. 2023, "Cloud Removal Using SAR and Optical Images via Attention Mechanism–Based GAN"
**Codebase:** This repo — AttentionGAN-for-Cloud-removal.
**Upstream summary of model architecture:** see `../Teacher_Model_codebase_summary.md`

---

## Dataset & Baseline Training

- **Data:** SEN12MS-CR mono-temporal dataset (~10K triplet patches)
- **Prepared with:** `../prepare_teacher_data.py` → writes to `../SEN12MSCR_prepared/trainA|B|C`
- **Training script:** `scripts/train_sen12mscr.sh` → `--model pix2pix_attn --name sen12mscr_teacher_v1`
- **Checkpoints:** `outputs/checkpoints/sen12mscr_teacher_v1/`
- **Baseline reached:** epoch 23, ~202 500 iterations
- **Periodic visuals** are saved to `outputs/visuals/` (custom instrumentation already in the code; keep the `--no_html` flag to avoid also creating the `web/` folder under checkpoints)

### Image naming from `get_current_visuals()`

| Filename tag | What it is |
|---|---|
| `real_A` | Cloudy optical input |
| `real_B` | Clear optical ground truth |
| `real_C` | SAR input |
| `fake_C` | G2's SAR→optical translation (intermediate, NOT the final answer) |
| `g_B` | netG's raw output **before** attention masking |
| `fake_B` | **Final output** — `g_B * att_A + real_A * (1 − att_A)` |
| `attn_A` / `attn_A2` / `attn_A3` | Attention map (heatmap / overlay / raw) |

`fake_B` is the actual generator output: it goes to the discriminator, to the L1+SSIM losses, and is the result at test time.

---

## Known Issues & Observations (Baseline v1)

### fake_C has wrong colour (dark, blue-orange rather than optical green/brown)

**What we see:** `fake_C` images at epoch 23 are dark and have blue/orange tones — nothing like natural optical imagery.

**Root causes (in order of severity):**

1. **No adversarial loss on G2 (paper design choice).** `loss_G2 = L1(fake_C, real_B) + SSIM(fake_C, real_B)`. Pure L1+SSIM without a GAN term causes the classic "regress to the mean" artifact: the network outputs a blurred average of all training targets rather than any perceptually realistic image.

2. **Cross-modal translation difficulty.** G2 must map SAR backscatter (unitless radar energy) to visible reflectance (RGB). This is a hard distribution shift. L1+SSIM can push pixel values closer but cannot enforce optical-domain realism.

3. **No early stopping metric on G2.** Because fake_C is an intermediate, there is no direct discriminator forcing it to look real; training just continues to minimise L1+SSIM without a quality floor.

### SAR channel ordering: our convention vs the paper's display

The paper (Zhang et al. 2023) displays SAR with **VH in the Red channel and VV in the Green channel** (a common remote-sensing visualisation convention where VH highlights double-bounce/urban and VV highlights rough surfaces). Our `prepare_teacher_data.py` uses the **opposite**: `R=VV, G=VH, B=VV-VH`.

**Does this affect network training?**
No — not as long as the convention is *consistent* across all 10K training pairs. The ResnetGenerator has no concept of "red" or "green"; it just sees three floating-point channels. What matters is that channel 0 of every SAR patch always carries VV (or always carries VH), and that relationship between the SAR channel ordering and the optical target is the same for every sample. Since `prepare_teacher_data.py` applies the same conversion uniformly, the model can learn the correct mapping regardless of which visual colour it corresponds to.

**The orange/blue fake_C is therefore NOT caused by a channel swap in the SAR input.** It is caused by the L1-only training dynamics described above.

---

## Planned Experiments

### Experiment 1 — G2 with a dedicated discriminator (netD2)

**Motivation:** Force fake_C to be perceptually realistic in the optical domain by adding adversarial training to the SAR→optical translation module. Not in the original paper.

**Model name / outputs:** `--name sen12mscr_teacher_v2_disc` — checkpoints go to `outputs/checkpoints/sen12mscr_teacher_v2_disc/`. Do **not** run with `--use_html`; keep `--no_html` so no web folder is created there. Periodic visuals should be saved to `outputs/visuals_v2/` (add/change the custom visual-save path accordingly).

**Architecture changes needed (in `models/pix2pix_attn_model.py`):**

1. **Add `netD2`** in `initialize()`:
   ```python
   self.netD2 = networks.define_D(
       opt.input_nc + opt.output_nc,   # 3 (SAR) + 3 (optical) = 6 channels
       opt.ndf,
       opt.which_model_netD,
       opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids
   )
   ```

2. **Add `optimizer_D2`** and schedule it alongside the other optimisers.

3. **Add `backward_D2()`** — conditional PatchGAN on the SAR→optical pair:
   ```python
   def backward_D2(self):
       # Fake: (SAR, fake_C) → should output 0
       fake_CC = torch.cat((self.real_C, self.fake_C), 1)
       self.loss_D2_fake = self.criterionGAN(self.netD2(fake_CC.detach()), False)
       # Real: (SAR, real_B) → should output 1
       real_CC = torch.cat((self.real_C, self.real_B), 1)
       self.loss_D2_real = self.criterionGAN(self.netD2(real_CC), True)
       self.loss_D2 = 0.5 * (self.loss_D2_fake + self.loss_D2_real)
       self.loss_D2.backward()
   ```

4. **Update `backward_G2()`** — add the adversarial term:
   ```python
   fake_CC = torch.cat((self.real_C, self.fake_C), 1)
   self.loss_G2_GAN = self.criterionGAN(self.netD2(fake_CC), True)
   self.loss_G2 = self.loss_G2_GAN + self.loss_G2_L1 + 1 * self.ssimloss2
   ```

5. **Update `optimize_parameters()`** to call `backward_D2()` and step `optimizer_D2`.

6. **Update `save()` / `load_network()`** to include `netD2`.

**What the netD2 discriminator is and why it was never in the paper:**
The paper treats G2 as a purely supervised pre-processor — a black box that translates SAR to a "SAR-like-but-optical" representation so netG can more easily fuse it with real_A. The authors likely omitted a D2 to keep the model simpler and because the main discriminator netD already provides adversarial training end-to-end (it sees fake_B, which depends on fake_C). The risk with adding D2 is that it can make training less stable: G2 now has to simultaneously satisfy netD2 (look optical) and netG (be a useful hint). We may need to weight the adversarial term lower initially (e.g. `0.1 * loss_G2_GAN`) to let the L1 loss anchor the spatial structure first.

**Results:** *(fill in after run)*
- fake_C quality: ?
- fake_B SSIM / PSNR vs baseline: ?
- Training stability: ?

#### Fix — netD2 was overpowering G2 (2026-07-28)

**Symptom:** netD2 converged much faster than G2, so `loss_D2` collapsed near-zero early and
`loss_G2_GAN` stopped providing a useful gradient — the classic discriminator-wins-too-early GAN
failure mode.

**Changes made in `models/pix2pix_attn_d2_model.py`:**
1. **`optimizer_D2` LR dropped 10x** (`opt.lr * 0.1` instead of `opt.lr`) — D2 now learns an
   order of magnitude slower than every other optimizer (G, G2, D, A all still use `opt.lr`).
2. **One-sided label smoothing on D2's real target** — `backward_D2()` no longer uses the hard
   `1.0` target from `criterionGAN(..., True)` for the real branch. Instead it draws an
   independent random label per element from `Uniform(0.75, 1.0)` and calls the underlying
   `criterionGAN.loss` (MSELoss, since LSGAN) directly against that smoothed target. Fake target
   is unchanged (hard 0.0). netD (the main discriminator) and netD2's fake-side loss are
   untouched — smoothing only applies to netD2's real-side loss, scoped to the SAR→optical mini
   GAN that was misbehaving.
3. **`scripts/train_sen12mscr_v2_disc.sh`: training length cut to 15 epochs** (`--niter 15
   --niter_decay 0`, was `100/100`) and `--save_epoch_freq` dropped from 20 to 5 so a labeled
   checkpoint actually lands inside the shorter run.
4. **`scripts/train_sen12mscr_v2_disc.sh`: `--lr_decay_iters 7`** (was the 100 default, which
   never fired inside a 15-epoch run). `--lr_policy step` builds a `StepLR(step_size=lr_decay_iters,
   gamma=0.1)` per optimizer (`get_scheduler` in `models/networks.py`), applied identically to
   G, G2, D, D2, and A — so this decays **every** optimizer's LR by 10x at epoch 7 (roughly the
   run's midpoint) and again at epoch 14. This is on top of, not instead of, D2's separately
   lowered base LR from fix #1 above — D2 still trains at 1/10th the others' LR at every point
   in the schedule.

**Current netG2/netD2 mini-GAN configuration** (none of these flags are overridden by the train
script, so these are the `options/base_options.py` / `train_options.py` defaults actually in
effect):

| Component | Value |
|---|---|
| `netG2` architecture | `UnetGenerator` (`unet_256`, i.e. 8 downsampling levels) — **not** ResnetGenerator, despite the upstream summary doc's generic description |
| `netG2` filters (`ngf`) | 64 |
| `netD2` architecture | `NLayerDiscriminator` (`which_model_netD=basic` → hardcoded `n_layers=3`), PatchGAN |
| `netD2` filters (`ndf`) | 64 |
| Normalization | **InstanceNorm2d** (`affine=False, track_running_stats=True`) — `--norm` defaults to `instance`, not `batch`. **No BatchNorm anywhere in this pipeline.** |
| Weight init | Xavier (`--init_type` default) |
| Dropout | Off (`--no_dropout` passed in the script) |
| GAN loss type | LSGAN (MSELoss) — `--no_lsgan` not passed, so `use_sigmoid=False`, no sigmoid on D2's output |
| G2 optimizer | Adam, lr=`opt.lr` (0.0002 default), betas=(0.5, 0.999) |
| D2 optimizer | Adam, lr=`opt.lr * 0.1` (**new**, was `opt.lr`), betas=(0.5, 0.999) |
| D2 real-label smoothing | **new**: `Uniform(0.75, 1.0)` per element (was hard 1.0); fake label still hard 0.0 |
| G2 adversarial weight | `G2_GAN_WEIGHT = 0.1` (class constant, unchanged) — `loss_G2 = 0.1*loss_G2_GAN + loss_G2_L1 + ssimloss2` |
| Batch size | 1 |
| Total epochs | **15** (new: `niter=15, niter_decay=0`, was `niter=100, niter_decay=100` = 200) |
| LR schedule | `step`, `lr_decay_iters=7` (**new**, was the 100 default) — every optimizer's LR ×0.1 at epoch 7 and epoch 14 |

**Results:** *(fill in after this run)*

---

### Experiment 2 — Feed SAR directly to netG (discard G2)

**Motivation:** Skip the SAR→optical translation step entirely and let the main generator learn to use raw SAR features directly. Simpler architecture; fewer failure modes.

**Architecture changes needed:**

- In `initialize()`, remove `netG2`, `netA` stays. Change netG input channels:
  ```python
  # Before: input_nc + input_nc (optical 3ch + fake_C 3ch)
  # After:  input_nc + input_nc (optical 3ch + real_C 3ch) — same shape, different semantic
  ```
  The channel count is identical (both 6), so `define_G` call is unchanged.

- In `forward()`:
  ```python
  # Remove: self.fake_C = self.netG2.forward(self.real_C)
  # Change: netG input from [real_A, fake_C] to [real_A, real_C]
  fake_B = self.netG.forward(torch.cat([self.real_A, self.real_C], dim=1))
  ```

- Remove all of `backward_G2()`, `optimizer_G2`, and the G2_L1 / G2_SSIM losses.
- Remove `netG2` from `save()` / load.
- Update `get_current_errors()` and `get_current_visuals()` to drop G2-specific entries.

**Model name / outputs:** `--name sen12mscr_teacher_v3_direct_sar` → `outputs/checkpoints/sen12mscr_teacher_v3_direct_sar/`. Visuals to `outputs/visuals_v3/`.

**Expected outcome:** netG will have to learn cross-modal SAR-optical fusion internally (within its ResNet blocks) rather than relying on a pre-translated intermediate. This may be harder to train but removes the translation bottleneck.

**Results:** *(fill in after run)*

---

## Errors Encountered

*(append here as experiments run)*

| Date | Experiment | Error / Observation | Fix |
|------|-----------|---------------------|-----|
| — | v1 baseline | fake_C dark, blue/orange tones at epoch 23 | Identified as L1-only training artifact + cross-modal difficulty; motivates Exp 1 |

---

## File / Flag Reference

| Concern | Flag / File |
|---|---|
| Avoid web folder | `--no_html` (already in `scripts/train_sen12mscr.sh`) |
| Separate experiment outputs | Change `--name` per experiment; checkpoints go to `outputs/checkpoints/<name>/` |
| Display visuals during training | `--display_freq 500 --display_id 0` (Visdom off; custom visuals save path in visualizer) |
| SAR channel order | `prepare_teacher_data.py`: R=VV G=VH B=VV-VH. Consistent across all patches. |
| Model file for new experiments | Copy `models/pix2pix_attn_model.py` to e.g. `models/pix2pix_attn_d2_model.py`; register in `models/models.py` under a new string key |
