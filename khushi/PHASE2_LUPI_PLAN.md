# Phase 2 — LUPI / teacher-quality audit

**Status of this file:** it was empty (0 bytes) from 2026-08-01 until 2026-08-03. The
repo-root `CLAUDE.md` cited it for a set of measurements about the teacher's SAR
dependence, but the writeup itself had been lost. This is a reconstruction, based on
re-running the measurements directly rather than on the lost text.

**Headline correction:** the earlier conclusion attributed to this file — that "the teacher's
encoder currently holds very little SAR-derived information", which "caps what any
distillation approach can transfer" — **is wrong and should not be relied on.** The
underlying `fake_C` measurements were real and reproduce almost exactly, but the conclusion
drawn from them does not survive contact with a direct measurement of SAR's contribution to
the output. SAR contributes substantially. See below.

All numbers below are on the **full 635-tile `SEN12MSCR_prepared` val split**, teacher
`sen12mscr_teacher_v1`, measured 2026-08-03.

---

## 1. SAR does contribute — substantially

Measured with `AttentionGAN-for-Cloud-removal/test_ig.py`
(`evaluate_teacher_sar_dependence`, expected-gradients attribution):

| Path | Attribution |
|---|---|
| Optical (`real_A`) | 81.27 % |
| **SAR (`real_C` → `netG2` → `fake_C`)** | **18.73 %** |

**18.73 % is a floor, not a ceiling**, for two structural reasons:

1. **The optical path is double-counted.** `fake_B = g_B·att_A + real_A·(1 − att_A)`, so
   `real_A` receives gradient through *both* `netG`'s direct input *and* the
   `(1 − att_A)` passthrough. SAR reaches the output through one path only. The split is
   biased toward optical by construction.
2. **The passthrough dominates wherever attention is low.** In non-cloudy regions
   `att_A → 0`, so `fake_B ≈ real_A` and essentially all attribution is optical — that is
   the model working *correctly*, not SAR failing. Most of a tile is not cloudy, which drags
   the whole-tile average down hard. SAR's share **inside cloudy regions** — the only place
   it is supposed to matter — is necessarily higher than 18.73 %.

The end-to-end result corroborates this. On the same split the teacher scores **32.72 dB**
against a **21.86 dB** do-nothing baseline (feeding the cloudy input straight through) —
**+10.86 dB**, SSIM 0.8457 vs 0.7703. A model whose privileged branch carried no usable
information could not produce that gap.

**Therefore: SAR-derived knowledge is present and worth distilling.** A null result from a
distillation method is a fact about that method, not an artifact of a SAR-blind teacher.

---

## 2. What *is* wrong: `netG2`'s intermediate is partly degenerate

The original measurements reproduce. `fake_C` vs its own SAR input `real_C`, and vs
`real_B` (the target `netG2` is actually trained on), at `latest` (= **epoch 23**, the
checkpoint both student runs distil from):

| Channel | `corr(fake_C, real_C)` | `std(fake_C)` | `std(real_B)` target | Ratio |
|---|---|---|---|---|
| ch0 (R / VV) | −0.173 ± 0.204 | 0.0388 | 0.0707 | 0.55× |
| ch1 (G / VH) | −0.097 ± 0.071 | **0.0042** | 0.0471 | **0.09× — 11× too flat** |
| ch2 (B / VV−VH) | −0.090 ± 0.056 | 0.1877 | 0.0372 | **5.0× too much variance** |

At epoch 20 the same pattern holds (−0.245 / −0.083 / −0.119; std 0.0443 / **0.0057** / 0.1818).
Both fall inside the −0.08 … −0.29 range the lost writeup quoted, so that measurement was sound.

Against `real_B`, `netG2` is also **beaten by a flat patch**: L1 0.142 / 0.207 / 0.143 per
channel versus 0.053 / 0.035 / 0.027 for simply predicting each image's own per-channel mean,
i.e. 2.7–6× worse.

### Reading these correctly

- **`corr(fake_C, real_C)` near zero is weak evidence on its own.** `netG2` performs a
  *cross-domain* translation (SAR backscatter → optical reflectance); its output is not
  supposed to correlate with its input. Do not cite this number alone.
- **The `std` comparison against `real_B` is the decisive one**, because `real_B` is
  literally `netG2`'s regression target. An 11× flat green channel is a genuine degeneracy,
  and a 5× over-dispersed third channel is genuine miscalibration.
- Both are consistent with the diagnosis already in
  `AttentionGAN-for-Cloud-removal/CLAUDE.md`: `loss_G2 = L1 + SSIM` with **no adversarial
  term** on `netG2` (the `loss_G_GAN2` lines are commented out in
  `models/pix2pix_attn_model.py`).

### How both facts coexist

Gradient attribution measures **sensitivity**; the `std` figures measure **calibration**.
SAR signal does reach the output — concentrated in ch2, the over-dispersed channel — while
ch1 contributes almost nothing spatially. So: real SAR information, flowing through a
partly-degraded representation. `netG` extracts enough of it to deliver +10.86 dB despite
the degeneracy.

---

## 3. Which teacher checkpoint to use

| Checkpoint | PSNR | SSIM | mean `att_A` | vs. do-nothing |
|---|---|---|---|---|
| **`teacher_v1` `latest` (= epoch 23)** | **32.72 dB** | 0.8457 | 0.732 | **+10.86 dB** |
| `teacher_v1` epoch 20 | 32.69 dB | 0.8461 | 0.733 | +10.83 dB |
| `teacher_dcond_cloudy` epoch 20 | 29.89 dB | 0.8045 | 0.890 | +8.03 dB |
| `teacher_dcond_cloudy` epoch 40 | 23.31 dB | 0.7794 | **0.9999** | +1.45 dB |
| cloudy input `real_A` | 21.86 dB | 0.7703 | — | — |

**`sen12mscr_teacher_v1` `latest` is the best teacher in the repo.** Use it.

**`sen12mscr_teacher_dcond_cloudy` is the *latest-trained* run but not the best — it
collapsed between epoch 20 and 40**, losing 6.6 dB. Two independent signatures:

- **Attention saturated**: `att_A` 0.890 → 0.9999, so the `(1 − att_A)` passthrough
  vanishes and the model overwrites every pixel including clear ones.
- **`netG2` output became noise**: against `real_B`, corr ≈ 0 (−0.033 / 0.002 / 0.017) and
  L1 **9–31× worse than a flat grey patch** (0.48 / 0.61 / 0.82 vs 0.053 / 0.035 / 0.027).
  Its high channel stds (0.54–0.61, ~10× the optical target) are noise, not healthy variance.

Note this run has *no* dead channel — because all three channels became noise. Absence of a
dead channel is not evidence of health.

### Verified: the student runs use the right teacher

All three student scripts (`khushi/student/scripts/train_sen12mscr_student{,_dist}.sh`,
`student/scripts/train_sen12mscr_student.sh`) set `TEACHER_NAME=sen12mscr_teacher_v1`,
`TEACHER_EPOCH=latest`. `TEACHER_CKPTS_DIR` is `./outputs/checkpoints`, i.e. *relative to
each student tree*, so each loads its own local copy — but all three `latest_net_G.pth` are
byte-identical (`md5 ce22dbb976f5ac00e30d7181d233539c`). Nothing points at the collapsed run.

**`latest` is epoch 23, not epoch 20** — `loss_log.txt` ends at epoch 23, and the `latest_*`
files carry a later mtime than the `20_*` files. Do not assume `latest` == the last numbered
checkpoint.

---

## 4. Implications for Phase 2

1. **Distillation is worth pursuing.** ~19 %+ whole-tile SAR attribution (higher in cloud)
   means there is real privileged knowledge to transfer. The DIST negative result in
   `khushi/student/DIST_EXPERIMENT_LOG.md` is a fact about DIST, not about a starved teacher.
2. **The hooked pathway is the degraded one.** `--hook_layers 3,4,5,6,7` sits on `netG2`'s
   deep blocks — the branch that produces the 11×-flat green channel. Every current
   distillation term (`L_hall`, DIST inter/intra, and the new cross-depth FSP term) matches
   activations along it, so the degeneracy is a **shared ceiling on all of them**. This is a
   plausible reason baseline and DIST landed within ±0.1 dB.
3. **Highest-leverage teacher fix: Experiment 1** (adversarial `netD2` on `netG2`,
   `AttentionGAN-for-Cloud-removal/CLAUDE.md`). It targets exactly the L1-only training
   dynamic causing the degeneracy, and `sen12mscr_teacher_v2_disc` currently has **no saved
   epoch checkpoints** — it was set up but never completed.
4. **Cheap student-side alternatives that avoid the degraded branch:** distil `netA`'s
   attention map (currently undistilled, and `netA` exists in both nets), or hook shallower
   blocks.
5. **Evaluate with cloud-region-masked metrics.** SAR's contribution concentrates where
   whole-tile averaging dilutes it. Add SAM (spectral angle) and NDVI consistency too — both
   are far more sensitive to the per-channel miscalibration above than PSNR is.

---

## 5. Reproducing these numbers

Scripts live in `AttentionGAN-for-Cloud-removal/` (added 2026-08-03):

```bash
cd AttentionGAN-for-Cloud-removal && source .venv/bin/activate

# SAR vs optical attribution
python test_ig.py          --dataroot ../SEN12MSCR_prepared --checkpoints_dir ./outputs/checkpoints \
  --name sen12mscr_teacher_v1 --model pix2pix_attn --dataset_mode unaligned_sar \
  --phase val --which_epoch latest --batchSize 4 --no_flip

# fake_C channel stats + corr against the SAR input
python check_fakeC.py      --dataroot ../SEN12MSCR_prepared --checkpoints_dir ./outputs/checkpoints \
  --name sen12mscr_teacher_v1 --model pix2pix_attn --dataset_mode unaligned_sar \
  --phase val --which_epoch latest --batchSize 4 --no_flip

# fake_C against real_B, its actual training target, vs a flat-mean baseline
python check_fakeC_vs_B.py <same flags>

# final output PSNR/SSIM vs the do-nothing baseline
python check_fakeB.py      <same flags>
```
