# Student Distillation Match: Exact (baseline) vs DIST (relaxed)

**Context:** `pix2pix_attn_student_model.py` distills the teacher's SAR→optical module (`netG2`)
into an optical-only `netH` (modality hallucination, Hoffman et al. 2016) via an exact sigmoid-L2
match on hooked UNet activations (`L_hall`). `pix2pix_attn_student_dist_model.py` forks it to
additionally offer a Pearson-correlation relational match (DIST, Huang et al. NeurIPS 2022:
`L_inter` + `L_intra`), which is invariant to per-location/per-channel affine rescaling and
doesn't demand exact activation recovery. See the docstring at the top of
`models/pix2pix_attn_student_dist_model.py` for the full derivation.

**Why this comparison matters:** `netH` is asked to reproduce the teacher's SAR-derived signal
from optical input alone, under cloud where there may be no optical evidence of the surface at
all. An exact match (`L_hall`) is hard there, and DIST's premise is that relaxing to a relational
match avoids fighting an objective that can't be hit. This run is the empirical test of that
argument, not a given — DIST changes *how* the teacher is matched, it cannot invent information
the teacher doesn't have.

> **⚠️ Corrected 2026-08-03 — the teacher is NOT SAR-blind.** An earlier version of this section
> argued from `corr(fake_C, real_C) ≈ −0.08…−0.29` that the teacher held almost no SAR-derived
> information, and therefore that any distillation method was capped at the source. **That
> inference was wrong.** The correlation figure is real and reproduces (−0.173 / −0.097 / −0.090
> at `latest`), but it is weak evidence on its own: `netG2` performs a *cross-domain* translation,
> so its output is not supposed to correlate with its SAR input. Direct measurement instead shows:
>
> - **SAR contributes 18.73 %** of the output attribution over the full 635-tile val split
>   (`test_ig.py`) — and that is a **floor**, because the optical path is double-counted (it gets
>   gradient through both `netG` and the `(1 − att_A)` passthrough) and because the passthrough
>   dominates the mostly-clear pixels that make up most of a tile. Inside cloud, SAR's share is
>   necessarily higher.
> - **The teacher scores 32.72 dB vs a 21.86 dB do-nothing baseline (+10.86 dB)** — impossible if
>   its privileged branch carried nothing usable.
>
> **Consequence for this log: the DIST null result below stands as a fact about DIST**, not as an
> artifact of a starved teacher. What *is* degraded is `netG2`'s intermediate calibration — the
> green channel of `fake_C` is 11× flatter than its `real_B` target and the third channel 5× over-
> dispersed — which is a shared ceiling on every term that hooks that branch, DIST and `L_hall`
> alike. Full audit and reproduction commands: `khushi/PHASE2_LUPI_PLAN.md`.

Architecture, warm-start, hook layers, `L_G`, `L_H`, optimizers, and epoch budget are held
identical between the two runs; only the hallucination-match term differs. That is what makes
this a fair ablation rather than two independently-tuned models.

---

## Runs

| | Baseline (exact match) | DIST (relaxed match) |
|---|---|---|
| Model key | `pix2pix_attn_student` | `pix2pix_attn_student_dist` |
| `--name` | `sen12mscr_student_v1` | `sen12mscr_student_dist_v1` |
| Driver | `train_student.py` | `train_student_dist.py` |
| Raw log | `train_baseline.log` | `train_dist.log` |
| tmux session | `student_baseline` | `student_dist` |
| `--gamma_hall` | 100 | 0 (pure DIST; set >0 to blend) |
| `--dist_beta` / `--dist_gamma` | n/a | 2.0 / 2.0 |
| `--dist_act` | n/a | `none` |
| `--hook_layers` | `3,4,5,6,7` | `3,4,5,6,7` |
| `--lambda3` | 1.0 | 1.0 |
| `--niter` / `--niter_decay` | 15 / 15 | 15 / 15 |
| Data | `SEN12MSCR_student/` (9127 train / 635 val, B4/B3/B8) | same |
| Teacher checkpoint | `outputs/checkpoints/sen12mscr_teacher_v1` (latest) | same |
| Started | 2026-07-31 18:02 | 2026-07-31 ~18:05 |
| Finished | 2026-08-01 00:18 | 2026-08-01 00:29 |
| Wall-clock / epoch (final epoch) | 785 s | 566 s |
| Best checkpoint (full val PSNR) | **epoch 20**, not `latest` | **epoch 20**, not `latest` |

Full commands are in the assistant conversation that produced this file (tmux launch commands,
2026-07-31); reproduced in each run's `outputs/checkpoints/<name>/opt_train.txt`, which
`BaseOptions.parse()` writes automatically — that file is the authoritative record of exact flags
used, this table is the human-readable summary.

---

## Quantitative comparison

Run after both finish (or against any shared `--which_epoch` to compare mid-training):

```bash
python eval_compare.py \
  --dataroot ../../SEN12MSCR_student \
  --dataset_mode unaligned_sar \
  --checkpoints_dir ./outputs/checkpoints \
  --phase val --which_epoch latest --how_many 635 \
  --names  sen12mscr_student_v1 sen12mscr_student_dist_v1 \
  --models pix2pix_attn_student pix2pix_attn_student_dist
```

PSNR/SSIM/L1 of `fake_B` vs `real_B` (cloud-free ground truth), SAR-free inference (`model.test()`),
full 635-tile val split.

| Epoch | Model | n | L1 ↓ | PSNR (dB) ↑ | SSIM ↑ |
|---|---|---|---|---|---|
| 10 (interim) | baseline | 150 | 0.0571 | 27.68 ± 1.53 | 0.8095 ± 0.0305 |
| 10 (interim) | DIST | 150 | 0.0555 | 28.07 ± 1.35 | 0.8099 ± 0.0289 |
| 15 (interim, LR decay just started) | baseline | 150 | 0.0562 | 27.92 ± 1.33 | 0.8050 ± 0.0297 |
| 15 (interim) | DIST | 150 | 0.0567 | 27.80 ± 1.38 | 0.8063 ± 0.0294 |
| 20 (interim) | baseline | 150 | 0.0578 | 27.60 ± 1.39 | 0.8003 ± 0.0297 |
| 20 (interim) | DIST | 150 | 0.0558 | 27.89 ± 1.32 | 0.8020 ± 0.0291 |
| **20 (final ckpt, full val)** | baseline | 635 | 0.0705 | 25.86 ± 2.12 | 0.8048 ± 0.0642 |
| **20 (final ckpt, full val)** | DIST | 635 | 0.0709 | 25.76 ± 2.25 | 0.8035 ± 0.0614 |
| **30 (final, full val)** | baseline | 635 | 0.0725 | 25.57 ± 2.06 | 0.7954 ± 0.0661 |
| **30 (final, full val)** | DIST | 635 | 0.0715 | 25.65 ± 2.01 | 0.7969 ± 0.0659 |
| 30 (n=150, for comparability with the interim rows) | baseline | 150 | 0.0603 | 27.24 ± 1.30 | 0.7860 ± 0.0306 |
| 30 (n=150) | DIST | 150 | 0.0591 | 27.40 ± 1.24 | 0.7877 ± 0.0295 |

**Delta at epoch 10 (DIST − baseline):** L1 −0.0017, PSNR +0.39 dB, SSIM +0.0004.
**Delta at epoch 15:** L1 +0.0005, PSNR −0.12 dB, SSIM +0.0014.
**Delta at epoch 20:** L1 −0.0020, PSNR +0.29 dB, SSIM +0.0018.

Oscillating: DIST ahead / roughly tied / DIST ahead across the three interim checks, all deltas
inside one std (~1.3-1.5 dB PSNR) of either run at n=150 — **no settled winner yet.** Also notable:
both runs' absolute PSNR is drifting down slightly (baseline 27.68→27.92→27.60; DIST
28.07→27.80→27.89) rather than climbing — neither model is clearly still improving on this metric
at n=150; could be LR-decay-phase noise or a genuine plateau, can't tell apart at this sample size.
`L_inter`/`L_intra` in `train_dist.log` are still ~1.6-2.0 (unweighted) around epoch 21, i.e. the
correlation match itself is far from tight. *(Originally read as evidence the teacher had little
SAR signal to transfer — **superseded**, see the correction at the top of this file: SAR
contributes ~19 % of output attribution. The loose match is better explained by `netG2`'s
per-channel miscalibration than by absent SAR information.)* Don't trust any delta under ~0.5 dB
at this sample size; re-run on the full 635-tile val split at epoch 30 before drawing a conclusion.

**Final delta (epoch 30, n=635):** L1 −0.0011, PSNR +0.09 dB, SSIM +0.0017 — DIST ahead on all three.
**Delta at epoch 20 (n=635):** L1 +0.0005, PSNR −0.11 dB, SSIM −0.0016 — DIST behind on all three.

### Two things the n=150 interim rows got wrong

**1. The ~2 dB drop between the interim and final rows is the sample, not the training.**
At a *fixed* checkpoint (epoch 20), n=150 scores 25.86 → 27.60 dB baseline: the first 150 tiles
in serial order are an easy subset, worth ~1.7 dB and half the spread (std 1.3 vs 2.1). Any
absolute number in this file taken at n=150 is inflated by roughly that much. `eval_compare.py`
sets `serial_batches=True`, so `--how_many N` is always the *same* first N tiles, never a sample.

**2. Both runs peak before epoch 30 — `latest` is not the best checkpoint.**
On the full val split, baseline goes 25.86 (ep20) → 25.57 (ep30) and DIST 25.76 → 25.65; SSIM
drops for both too (0.8048 → 0.7954, 0.8035 → 0.7969). The last 10 epochs cost ~0.1–0.3 dB. The
interim rows' worry that "both runs are drifting down" was directionally right, just inflated by
the subset effect above. **Use epoch 20 for anything downstream, not `latest`.**

### The DIST effect is significant and still not real

Because both models see identical tiles, the per-image *difference* is the right unit of
analysis — `eval_compare.py`'s ±std is the between-tile spread (~2 dB), which swamps a ~0.1 dB
model effect and makes everything look like noise. Paired tests on the 635 per-tile differences:

| Checkpoint | Metric | Mean diff (DIST−base) | 95% CI | paired t | Wilcoxon | DIST wins |
|---|---|---|---|---|---|---|
| ep30 | PSNR | **+0.094 dB** | [+0.023, +0.165] | p=0.010 | p=0.0035 | 57.3% |
| ep30 | SSIM | **+0.0017** | [+0.0009, +0.0026] | p=7.4e−05 | p=0.00042 | 56.4% |
| ep30 | L1 | **−0.0011** | [−0.0017, −0.0005] | p=0.00032 | p=0.00020 | 57.6% |
| ep20 | PSNR | **−0.109 dB** | [−0.191, −0.026] | p=0.010 | p=0.027 | 50.1% |
| ep20 | SSIM | **−0.0016** | [−0.0026, −0.0006] | p=0.0016 | p=0.0044 | 46.8% |
| ep20 | L1 | +0.0005 | [−0.0001, +0.0012] | p=0.12 | p=0.43 | 51.3% |

**Read this as a negative result, not a win for DIST.** At n=635 paired, the test resolves
differences of ~0.1 dB — so it returns p<0.01 at *both* checkpoints, with **opposite signs and
nearly equal magnitude** (+0.094 dB at ep30, −0.109 dB at ep20). A method effect does not reverse
between two checkpoints of the same pair of runs; what the test is detecting is checkpoint-to-
checkpoint jitter within each run. The median per-tile |difference| is 0.42–0.54 dB, four to five
times the mean difference, i.e. the two models disagree substantially per tile and it very nearly
cancels. DIST wins on 50–57% of tiles — a coin flip.

**Conclusion: no measurable benefit from the relaxed match, in either direction.** The log's
earlier "don't trust any delta under ~0.5 dB" rule survives the move to the full val split; the
lesson is only that a *significant* p-value at n=635 does not rescue a delta that small. This is
consistent with the `PHASE2_LUPI_PLAN.md` measurement: DIST changes *how* `netH` matches the
teacher, and if the teacher's `netG2` carries little SAR-derived information, no matching scheme
recovers information that isn't there. `L_inter`/`L_intra` ending at ~1.6–2.0 unweighted (never
tightening) says the relational match itself was never satisfied either.

**What would actually move the needle** — in rough order of expected value, none of which is a
distillation-objective change: fix the teacher so `fake_C` carries real SAR signal (TODO items 1,
3, 4); or drop the hallucination pathway and condition the student on something that isn't
degraded. Tuning `--dist_beta`/`--dist_gamma` upward is the cheap next thing to try but is
unlikely to clear 0.5 dB given the above.

---

## Qualitative check

```bash
python test_student.py --dataroot ../../SEN12MSCR_student --name sen12mscr_student_v1      --model pix2pix_attn_student      --dataset_mode unaligned_sar --checkpoints_dir ./outputs/checkpoints --phase val --which_epoch latest --gpu_ids 0
python test_student.py --dataroot ../../SEN12MSCR_student --name sen12mscr_student_dist_v1 --model pix2pix_attn_student_dist --dataset_mode unaligned_sar --checkpoints_dir ./outputs/checkpoints --phase val --which_epoch latest --gpu_ids 0
```

Writes to `results/sen12mscr_student_v1/val_latest/` and `results/sen12mscr_student_dist_v1/val_latest/`.

*(fill in after visual inspection — cloud-region artifacts, colour cast, sharpness, anything the
numbers in the table above don't capture)*

---

## Observations / issues encountered

*(append here as the runs progress — OOM, loss spikes, GPU contention from running both
concurrently, anything that would explain an anomalous number in the table above)*

| Date | Run | Observation | Action taken |
|------|-----|-------------|--------------|
| 2026-08-01 | both | Both runs completed 30/30 epochs cleanly; no OOM or loss spikes. LR reached 0.0000000 at the end as scheduled. Ran concurrently on one RTX 4090 — DIST's epochs were *faster* (566 s vs 785 s) despite the extra relational terms, so the difference is GPU contention, not model cost; don't read anything into it. | none needed |
| 2026-08-01 | both | Full 635-tile val scored at ep20 and ep30 + paired per-tile tests. **No measurable DIST benefit** (see above); both runs peaked at ep20. | Log conclusion recorded; `latest` deprecated in favour of epoch 20 for downstream use |
| 2026-08-01 | both | `eval_compare.py --how_many N` is `serial_batches=True`, so it always scores the *same first N* tiles. The first 150 are ~1.7 dB easier than the full split, which inflated every interim row in the table above. | Always score the full 635 before comparing; interim rows kept but labelled |

---

# Experiment 2 — Cross-depth FSP covariance distillation (added 2026-08-03)

**Status:** implemented, unit- and smoke-tested, **not yet run against real data.**

**Model key:** `pix2pix_attn_student_fsp` → `models/pix2pix_attn_student_fsp_model.py`
**Driver:** `train_student_fsp.py`  **Script:** `scripts/train_sen12mscr_student_fsp.sh`
**Planned run name:** `sen12mscr_student_fsp_v1`

## What is new

`L_hall` and the DIST terms both compare a teacher and student activation **at the same depth**.
Neither constrains how information *evolves between* depths. This adds a third, independently
toggleable term based on the FSP ("flow of solution procedure") matrix of
**Yim et al., "A Gift from Knowledge Distillation", CVPR 2017**.

For each **adjacent** pair of hooked levels (3-4, 4-5, 5-6, 6-7 — 4 pairs, not all 10), per
instance:

```
F_i' = adaptive_avg_pool2d(F_i, (H_j, W_j))                        # pool DOWN only
G    = (1/(H_j*W_j)) * F_i'.reshape(C_i,-1) @ F_j.reshape(C_j,-1).T   # [C_i, C_j]
loss_pair = 1 - pearson(vec(G_teacher), vec(G_student))            # default
```

Three properties worth stating explicitly:

- **Works at batchSize 1.** The contraction is over *spatial positions within one instance*, not
  across batch members. This is the key difference from DIST's `L_intra`, which needs multiple
  instances and is why `dist_intra_min_positions` exists.
- **Pearson between the flattened FSP matrices, not covariance directly.** `C_i*C_j` and the
  activation magnitudes differ by orders of magnitude between the shallow and deep pairs, so an
  unnormalised distance would let one pair dominate the sum. `--fsp_distance frobenius` gives the
  Yim et al. original (`||G_t − G_s||²_F / (C_i·C_j)`) for ablation.
- **Skip-echoed input channels are excluded.** See below.

## Measured facts about this codebase that shaped the implementation

Verified on a real 256×256 forward through `define_G2(3, 3, 64, 'unet_256', 'instance', ...)`:

| Block | Total C | C_in (echoed) | Output-half C | H × W |
|---|---|---|---|---|
| 3 | 512 | 256 | 256 | 32 × 32 |
| 4 | 1024 | 512 | 512 | 16 × 16 |
| 5 | 1024 | 512 | 512 | 8 × 8 |
| 6 | 1024 | 512 | 512 | 4 × 4 |
| 7 | 1024 | 512 | 512 | 2 × 2 |

1. **`UnetSkipConnectionBlock.forward` returns `cat([x, model(x)])`, and `out[:, :C_in]` is
   *bitwise* equal to the block input `x`** (verified, max|diff| = 0.000e+00 at every hooked
   level). That echoed half is near-identical between teacher and student — both derive from the
   same input path — so including it would inflate the measured similarity independently of
   anything `netH` learns. The cross-depth term therefore uses the **output half only**.
   **The split is 256/512 at block 3, i.e. NOT an even half** — it is read from the hook's own
   `inp[0].shape[1]` per level, never hardcoded as `C//2`.
2. **The parent block's `uprelu = nn.ReLU(inplace=True)` mutates the child's hooked tensor after
   the hook fires.** Measured drift `max|live − snapshot|` is 3.2e-1 … 9.7e-3 across levels, and
   the tensors the losses actually see are **post-ReLU: `frac<0 = 0.0000`, min exactly +0.0000**
   at every hooked level. This applies to the *existing* `L_hall` and DIST terms too — so
   `--dist_act none` has never meant "raw activations", it means "post-ReLU, non-negative", and
   `sigmoid` on such input only ever maps to [0.5, 1). Deliberately left unchanged: all three
   terms read the same dicts, keeping the three-way ablation clean.

## Rank caveat — watch the diagnostics before trusting the deep pairs

`rank(G) ≤ H_j·W_j`, so on the real geometry:

| Pair | G shape | Pooled positions | Max rank |
|---|---|---|---|
| 3-4 | (256, 512) | 256 | 256 |
| 4-5 | (512, 512) | 64 | 64 |
| 5-6 | (512, 512) | 16 | **16** |
| 6-7 | (512, 512) | 4 | **4** |

Pair 6-7 builds a 262,144-entry matrix from **4 spatial positions**. The `eps` guard does not
catch this — such matrices have healthy variance, they are merely near-rank-deficient. All 4
pairs are enabled by default (`--fsp_min_positions 0`) *because* the per-pair diagnostics are the
instrument for deciding whether to drop them; set `--fsp_min_positions 16` or `64` if they turn
out to carry no signal.

**Smoke test on untrained nets already shows pair 6-7 flagged DEGENERATE** (std of vec(G) below
`fsp_eps`), exactly as the rank analysis predicts. The other three pairs return `1 − ρ ≈ 1.00`,
the correct value for two independently-initialised networks.

## Diagnostics

Per epoch, `<checkpoints_dir>/<name>/fsp_metrics.txt` records for every pair the mean distance
and `*_degen`, the fraction of instances whose FSP matrix was degenerate. Per-iteration per-pair
values also appear in `loss_log.txt` alongside `L_fsp` (logged **unweighted**, like every other
distillation term, so it stays comparable across runs with different weights).

## Flags

| Flag | Default | Purpose |
|---|---|---|
| `--use_hall` / `--no_hall` | on | Toggle the Hoffman exact-match term (skips its compute, unlike `--gamma_hall 0`) |
| `--use_dist` / `--no_dist` | on | Toggle the within-layer DIST terms |
| `--use_cross_depth` / `--no_cross_depth` | on | Toggle the new cross-depth term |
| `--cross_depth_weight` | 1.0 | Weight on the summed cross-depth loss |
| `--fsp_distance` | `pearson` | `pearson` (scale-invariant) or `frobenius` (Yim et al.) |
| `--fsp_pairs` | `adjacent` | 4 adjacent pairs, or `all` for 10 |
| `--fsp_min_positions` | 0 | Skip pairs whose deeper layer has fewer positions |
| `--fsp_eps` | 1e-6 | Degeneracy threshold on std(vec(G)) |

## Run that was executed

Two attempts are captured in `train_fsp_weight2_aborted.log` (both in the same file):

1. **First attempt** started fresh, killed mid-epoch-9 (before any checkpoint after epoch 5 was
   saved). The restart began from epoch 1 again — no resume.
2. **Second attempt** ran to completion, 30 epochs. All six checkpoints (5, 10, 15, 20, 25, 30)
   plus `latest` are on disk.

The config actually used was **`--cross_depth_weight 8.0`** (not the planned 2.0), hence the
"weight2" tag in the log filename. Everything else matches the plan: `--gamma_hall 100 --no_dist
--use_cross_depth`, `fsp_pairs=adjacent`, `fsp_distance=pearson`. The log shows `L_inter` and
`L_intra` were 0.000 every iteration — DIST was correctly off. `fsp_degen` was **0.000 at every
epoch for every pair** — no feature collapse occurred.

Eval command (requires `--dataset_mode unaligned_sar`, which `eval_compare.py` does not default
to, causing a `KeyError: 'C'` without it):

```bash
python eval_compare.py \
  --dataroot ../../SEN12MSCR_student --checkpoints_dir ./outputs/checkpoints \
  --dataset_mode unaligned_sar \
  --phase val --which_epoch <ep> --how_many 635 \
  --names  sen12mscr_student_fsp_v1 \
  --models pix2pix_attn_student_fsp
```

## Results — per-epoch val PSNR/SSIM (full 635 tiles, 2026-08-04)

| Epoch | n | L1 ↓ | PSNR (dB) ↑ | SSIM ↑ | fsp_6_7 (train) |
|---|---|---|---|---|---|
| 5 | 635 | 0.0707 | 25.81 ± 2.29 | 0.8101 ± 0.0616 | 0.161 |
| **10** | **635** | **0.0684** | **26.14 ± 2.11** | **0.8117 ± 0.0645** | 0.092 |
| 15 | 635 | 0.0717 | 25.77 ± 2.18 | 0.8059 ± 0.0646 | 0.052 |
| 20 | 635 | 0.0722 | 25.61 ± 2.21 | 0.8018 ± 0.0618 | 0.027 |
| 25 | 635 | 0.0716 | 25.60 ± 2.06 | 0.7984 ± 0.0638 | 0.014 |
| 30 | 635 | 0.0727 | 25.52 ± 2.04 | 0.7956 ± 0.0653 | 0.008 |

**Best checkpoint: epoch 10** (not epoch 20 as for the baseline runs). Use `--which_epoch 10`
for any downstream comparison or LISS-4 inference with this model.

### Direct comparison at epoch 10

```
python eval_compare.py --dataset_mode unaligned_sar --phase val --which_epoch 10 --how_many 635 \
  --names  sen12mscr_student_fsp_v1  sen12mscr_student_v1 \
  --models pix2pix_attn_student_fsp  pix2pix_attn_student
```

| Model | n | L1 | PSNR (dB) | SSIM |
|---|---|---|---|---|
| FSP (ep 10) | 635 | 0.0684 | 26.13 ± 2.10 | 0.8115 ± 0.0645 |
| Baseline (ep 10) | 635 | 0.0701 | 25.95 ± 2.17 | 0.8117 ± 0.0636 |
| **Delta (FSP − base)** | | **−0.0017** | **+0.18 dB** | **−0.0002** |

**Best vs best** (FSP ep10 vs baseline ep20 = 25.86 dB): **FSP +0.28 dB**.

## Analysis

**FSP peaks earlier and harder than the baseline.** The model reaches its best quality at epoch 10
(26.14 dB) rather than epoch 20, then degrades to 25.52 dB by epoch 30 — a 0.62 dB drop from
peak over the decay phase. The baseline degrades only 0.29 dB (25.86 → 25.57) over the same span.
The FSP term with weight=8 is creating a fast early alignment that subsequently fights the LR
decay schedule.

**The culprit is `fsp_6_7` converging to near-zero.** The bottleneck pair drops from 0.199 →
0.008 (96% reduction) while `fsp_3_4` only drops 41% (0.152 → 0.089). This aggressive bottleneck
alignment pushes `netH` toward the teacher's innermost representation very fast — faster than
the upstream reconstruction losses can stabilise the outer layers around it. Once
`fsp_6_7 ≈ 0` (after epoch 15), the FSP gradient mostly vanishes from the bottleneck and the
remaining outer-layer signals are too weak to compensate for the displacement the early
over-alignment created. The probe result above (layer 7 is at the random-init floor in SAR R²)
further suggests the bottleneck was not the productive part anyway.

**The +0.18 dB gap at epoch 10 is inside the noise floor established by DIST.** The DIST
experiment showed deltas of +0.39 / −0.12 / +0.29 dB at n=150 and −0.11 / +0.09 dB at
n=635 across two checkpoints, with sign flips. This FSP result (+0.18 dB at a single checkpoint,
n=635) is consistent with checkpoint jitter rather than a real method effect. A paired per-tile
test would be needed to determine whether it survives; it was not run, since the model degrades
past the baseline by epoch 20 anyway.

**SSIM does not corroborate the PSNR.** At epoch 10, SSIM is essentially tied (FSP 0.8115,
baseline 0.8117), despite the +0.18 dB PSNR gap. This is a further sign the PSNR difference is
within noise.

## What "weight2" means

This was the second FSP weight tried. A "weight1" run (presumably at the planned 2.0) presumably
existed or was planned; this run used 8.0. The 4× heavier weight accelerated the bottleneck
collapse and caused the early peak / later decay pattern. The right follow-up would be
`--cross_depth_weight 2.0` with `--fsp_min_positions 16` (dropping pairs 6-7 and 5-6 which the
rank analysis flags as near-degenerate), aligned with what the probe says: layers 4–6, not 7,
carry the SAR signal.

## Verification (pre-run, recorded here for completeness)

- `py_compile` on the model, dispatch table, and driver; `bash -n` on the script.
- Unit tests (`scratchpad/fsp_loss.py`, all passing): FSP shapes match the probed geometry;
  identical teacher/student → loss ~0 (`-1.99e-08`); a **genuinely different cross-depth
  relationship** (same shallow map, different mixing matrix into the deep map — invisible to any
  within-layer loss) scores 1.0040 vs `-1.19e-07` for the matched one; the degeneracy guard fires
  at frac 1.00 on a constant input and 0.50 on a mixed batch; pearson is scale-invariant under a
  3× student rescale where frobenius is not (1.1716); gradient reaches both layers of a pair.
- Smoke test on the real UNet (`scratchpad/smoke_fsp.py`): gradient reaches the student
  (13/17 param tensors, `|grad| = 1.14e+04`) and **zero gradient leaks into the frozen teacher**.

**Bug caught by the unit tests, worth remembering:** writing the Pearson denominator as
`(t_std*s_std + eps)` biases every correlation down by `eps/(σ_tσ_s + eps)` — identical inputs
scored `1 − 6.1e-4` instead of 0. Small, but it puts a permanent floor under the loss and is
worst exactly where `G` has low variance. Fixed with `.clamp_min(eps)`; degenerate instances are
excluded by the mask anyway. The same additive-eps pattern is still present in
`pix2pix_attn_student_dist_model.py`'s `pearson_corr` (`dist_eps=1e-8`, so the bias is ~100×
smaller there) — left alone so the completed DIST run stays reproducible.

---

# Experiment 3 — FSP v2: lower weight + drop degenerate deep pairs (2026-08-04)

**Status:** running (PID 17784, launched 2026-08-04 ~11:53)

**Model key / name:** `pix2pix_attn_student_fsp` / `sen12mscr_student_fsp_v2`
**Script:** `scripts/train_sen12mscr_student_fsp_v2.sh`
**Log:** `train_fsp_v2.log`

## What changed from fsp_v1

| | fsp_v1 | **fsp_v2** |
|---|---|---|
| `cross_depth_weight` | 8.0 | **2.0** |
| `fsp_min_positions` | 0 (all 4 pairs) | **17 (drops 5-6 and 6-7)** |
| Active FSP pairs | 3-4, 4-5, 5-6, 6-7 | **3-4, 4-5 only** |

Everything else — `gamma_hall=100`, `--no_dist`, hook layers, teacher, dataset, LR schedule,
epoch budget — is held identical.

**Why these two changes:**

1. **Drop 5-6 and 6-7**: The SAR linear probe showed layer 7 is at the random-init floor.
   The rank analysis shows pair 6-7 builds a 512×512 FSP matrix from only 4 spatial positions
   (2×2), and pair 5-6 from only 16. In fsp_v1, `fsp_6_7` collapsed 96% by epoch 30 while
   `fsp_3_4` only dropped 41% — the degenerate pairs were doing nearly all the work and
   over-aligning the bottleneck. Dropping them leaves only the two pairs that are both
   well-conditioned (≥64 positions) and carry SAR signal (layers 4–6 per the probe).

2. **Lower weight**: fsp_v1 peaked at epoch 10 then degraded 0.62 dB through epoch 30. With
   only 2 pairs (vs 4), `L_fsp` is roughly half the scale, so effective contribution is
   `2 × ~0.5 = ~1.0` vs the hall term's `100 × ~0.077 = ~7.7`. This makes FSP a lighter
   regulariser rather than a dominant force. If this is too weak to show an effect, the next
   step is weight=4 with the same pair selection.

## Early diagnostics (epoch 1, first ~1900 iters)

FSP pairs confirmed: only `fsp_3_4` and `fsp_4_5` appear in loss_log.txt — the drop worked.
`L_fsp` values at init (~0.55, dropping toward ~0.32 in first 1900 iters) vs fsp_v1 init
(~0.76 with 4 pairs, weighted ×8 = 6.1). With weight=2, contribution is ~0.64–1.1 here —
about 8× lighter than fsp_v1's bottleneck pressure.

## Results

*(fill in after run — score every saved epoch on full 635 tiles, compare to baseline ep10/20 and fsp_v1 ep10)*

```bash
python eval_compare.py \
  --dataroot ../../SEN12MSCR_student --checkpoints_dir ./outputs/checkpoints \
  --dataset_mode unaligned_sar \
  --phase val --which_epoch <ep> --how_many 635 \
  --names  sen12mscr_student_fsp_v2 \
  --models pix2pix_attn_student_fsp
```

| Epoch | n | L1 ↓ | PSNR (dB) ↑ | SSIM ↑ |
|---|---|---|---|---|
| 5 | | | | |
| 10 | | | | |
| 15 | | | | |
| 20 | | | | |
| 25 | | | | |
| 30 | | | | |

---

# Diagnostic — linear probe: is there SAR information in `netH`? (2026-08-03)

**Why.** Every result in this log so far is a downstream image-quality number. Those are a
low-bandwidth readout of whether distillation worked: if PSNR does not move, it is impossible to
tell whether (a) the teacher had no SAR knowledge to give, (b) it did but the loss failed to
transfer it, or (c) it transferred but did not help image quality. `khushi/PHASE2_LUPI_PLAN.md`
made (a) look likely — `corr(fake_C, real_C) ≈ −0.08…−0.29` for the teacher. This measures the
transfer directly instead of inferring it.

**Method.** `probe_sar.py` (new, this dir). Ridge-regression linear probe fitting
`real_C ~ W·activations + b` at each hooked UNet block, on the 635-tile val split, seeded random
tile split 60/15/25 (fit / λ-selection / report). SAR target is average-pooled to each block's
resolution. Sufficient statistics (`XᵀX`, `Xᵀy`, `Σy²`) accumulate in one pass, so SSE for any λ
is closed-form and no features are stored. Four feature sources under an identical split and λ
grid: trained `netH` on `real_A`; **randomly-initialised `netH` on `real_A` (the floor)**; the
teacher's `netG2` on `real_C` (sanity check); raw optical pooled to block resolution.

**Sanity check passes:** `teacher_G2` scores R² 0.97 / 0.97 / 0.93 / 0.87 / 0.71 at layers 3–7.
A net that is fed SAR does linearly encode SAR, so the probe is measuring what it claims to.

## Result — `sen12mscr_student_v1`, epoch 20, full 635 tiles

| layer | netH_trained | netH_random (floor) | raw_optical | **Δ (training effect)** |
|---|---|---|---|---|
| 3 | 0.2122 | 0.1259 | 0.0646 | **+0.0863** |
| 4 | 0.2860 | 0.1539 | 0.0815 | **+0.1320** |
| 5 | 0.2971 | 0.1590 | 0.0996 | **+0.1381** |
| 6 | 0.2609 | 0.1471 | 0.1132 | **+0.1138** |
| 7 | 0.1226 | 0.0932 | 0.1235 | **+0.0295** |

**Transfer is real but partial.** `netH` beats its random-init floor at every hooked layer, so
training did put SAR-related information into it that was not already trivially available from a
random nonlinear map of the optical input. This **contradicts the pessimistic reading of
`PHASE2_LUPI_PLAN.md`** — the hallucination pathway is not dead. But the absolute level is low:
R² ≈ 0.30 at best against the teacher's 0.93–0.97, i.e. `netH` recovers roughly a third of the
linearly-decodable SAR variance, and that is against *pooled* (smoothed) SAR, which is the easy
version of the target.

**Per-channel, the informative channel is the one that fails.** R²_VV is 0.17–0.24 while
R²_VH is 0.24–0.31 and R²_(VV−VH) is 0.25–0.43. Raw optical predicts VV at R² ≈ 0.00 at *every*
layer while reaching 0.15–0.41 on VV−VH. So most of what the probe recovers is concentrated in
the channel most correlated with ordinary optical texture, and true backscatter — the part SAR
contributes that optical cannot — is where transfer is weakest.

**Layer 7 (innermost) is at the floor** and is the one place raw optical matches `netH`. The
deep end of `--hook_layers 3,4,5,6,7` is contributing nothing; 4–6 carry the signal.

## Fidelity — did the student actually match the teacher?

Val-split `L_hall` (Hoffman sigmoid-L2, the exact training objective), 159 held-out tiles:

| | L_hall | vs random init |
|---|---|---|
| `netH` trained (v1) | 0.0572 ± 0.0111 | −44.1% |
| `netH` trained (dist_v1) | 0.0639 ± 0.0107 | −37.6% |
| `netH` random init | 0.1024 ± 0.0160 | — |

The student closes **less than half** the gap to the teacher on held-out data. This is the
Stanton et al. (NeurIPS 2021) fidelity/generalization split: the run is a partial *optimization*
failure, not only a method failure. `--gamma_hall` is untuned (CLAUDE.md already flags this), so
there is headroom that no change of loss function addresses.

## The first signal that separates DIST from the baseline

`sen12mscr_student_dist_v1` at epoch 20, same probe: **higher R² at every layer**
(+0.0157 / +0.0176 / +0.0442 / +0.0481 / +0.0208 over the baseline; Δ-vs-random of
+0.10 / +0.15 / +0.18 / +0.16 / +0.05) while scoring **worse** `L_hall` (0.0639 vs 0.0572).

That is coherent: DIST deliberately relaxes exact matching, so it matches the teacher less
closely and transfers more SAR information. It is also **invisible in PSNR** — the settled
result above remains that the two are indistinguishable downstream (±0.1 dB, sign-flipping
across checkpoints). Caveat: one seed, one checkpoint, one random-init floor; treat as a lead,
not a finding, until repeated across seeds and epochs.

## Caveats

- Val tiles from one ROI overlap spatially, so the seeded random split makes absolute R²
  optimistic. The split is identical across sources, so the *comparisons* hold.
- Layers 6 and 7 fit 6096 and 1524 rows for 1024 features — ridge keeps them solvable but the
  probe is near-interpolating there. λ selection is on held-out data, so this is not silent, but
  those two rows are weak evidence.
- A *linear* probe lower-bounds the information present; a nonlinear probe would read higher.

## What this changes

The no-KD and random-teacher control runs are still the missing controls for the *downstream*
claim, but the mechanism question is now answered: there is something to transfer, transfer is
happening, and it is inefficient. Priorities in order: tune `--gamma_hall` against the val
`L_hall` curve; drop layer 7 from `--hook_layers`; re-run the probe across seeds/epochs to see
whether the DIST advantage survives.

**Reproduce:**
```bash
python probe_sar.py --dataroot ../../SEN12MSCR_student --checkpoints_dir ./outputs/checkpoints \
  --name sen12mscr_student_v1 --teacher_name sen12mscr_teacher_v1 \
  --model pix2pix_attn_student --dataset_mode unaligned_sar \
  --phase val --which_epoch 20 --how_many 635 --out_csv probe_student_v1_e20.csv
```
