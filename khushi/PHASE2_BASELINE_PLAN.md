# Plan: Phase-2 baseline student (no ViT) + real feature distillation

## Context

The Phase-2 student in `izma/` runs and has produced 20 epochs of checkpoints, but a read of the
code shows it cannot be learning what it is supposed to be learning. Three things are structurally
broken (details in **Findings**), the most serious being that **the SAR/privileged pathway and the
feature-hint pathway both carry exactly zero gradient into the student**, and that **the student is
never shown a cloud-free ground-truth image** — every loss targets the frozen teacher's output, so
the student is hard-capped at a teacher that the teacher experiment log still lists as degraded.

We are dropping the Phase-1 ViT backbone for now to get a defensible baseline quickly. Decisions
confirmed with the user:

- **Backbone:** student generator becomes a UNet twin of the teacher's `netG` (`unet_256`), 3-channel
  optical in / 3-channel out, built with the vendored `networks.define_G`.
- **Supervision:** cloud-free ground truth becomes the primary target; teacher terms sit on top.
- **Distillation:** feature-level, tapping `netG`'s decoder (fused optical+SAR) **and** `netG2`'s
  SAR bottleneck (LUPI) — replacing today's output-only matching.
- **Band mismatch:** teacher keeps its native B4/B3/B2; student stays on B4/B3/B8. Pixel-level
  teacher matching is restricted to the shared R,G channels; the rest transfers via `att_A` and
  features.

Outcome: a baseline that can be scored against ground truth, plus a distillation path where the
SAR knowledge actually reaches student weights.

---

## Findings — is the current code correct?

No. Ordered by severity; all verified in source.

### Blocking

1. **The SAR and feature pathways train nothing.** `Phase1ViTEncoder.forward` wraps the backbone in
   `torch.no_grad()` when `freeze=True` ([student_gan.py:154-160](izma/student_gan.py#L154-L160)),
   so `student_out["tokens"]` and `student_out["skips"]` are detached. `sar_mimicry_loss` therefore
   updates only the `projections["sar"]` 1×1 conv — **no gradient reaches any generator weight**.
   The entire LUPI group is a no-op on the student. Same would apply to `L_feat` if it were wired.
2. **No ground truth exists in the pipeline.** `StudentDataset` yields only `cloudy_optical` and
   `sar` ([student_dataset.py:170](izma/student_dataset.py#L170)); clear S2 is never opened, though
   29 117 tiles sit in `SEN12MSCR/ROIs1158_spring_s2`. Every loss — including the discriminator's
   "real" branch — targets `teacher.fake_B`. The student can only converge *to* the teacher.
3. **The teacher is fed out-of-distribution input.** It was trained on B4/B3/B2 8-bit PNGs
   (`prepare_teacher_data.py`), but receives B4/B3/B8 raw reflectance: channel 2 is NIR where the
   teacher expects Blue, and the radiometric scaling differs. `att_A`, `fake_B` and `fake_C` are all
   computed off-distribution, so every distillation target is corrupted at the source.
4. **`L_feat` is identically zero.** `TeacherWrapper.forward` returns `{}` for hints, so
   `feature_hint_loss`'s loop never executes and it returns the Python float `0.0`.

### Significant

5. **L1 is counted three times on the same tensor pair.** `L_1` (λ=10) and `L_out` (α=5) are both
   `F.l1_loss(I_g, I_t)` — the same function on the same inputs — and `L_cw` (δ=1) is a
   channel-weighted L1 on that pair again. Effective L1 weight ≈16 vs `L_adv` at 1, so the
   adversarial and attention terms are swamped.
6. **`sar_mimicry_loss` matches an image, not features.** It consumes `teacher.fake_C`, a 3-channel
   output image, while its own docstring and the file header claim "netG2's bottleneck features".
7. **`val_loader` is accepted and never used** ([student_gan.py:515-518](izma/student_gan.py#L515-L518)) —
   no validation, no metrics, no best-checkpoint selection, across a 50-epoch run.

### Minor

8. `TeacherWrapper.to(device)` is a no-op — `Pix2Pix_attn_Model` is not an `nn.Module`, so its
   nets are never registered. Works only because `initialize()` already placed them on `gpu_ids[0]`.
9. `logs` is referenced after the batch loop; an empty loader raises `NameError` (`d_loss`/`g_loss`
   are guarded, `logs` is not).
10. `ssim()` uses `C1/C2` calibrated for `data_range=1` on `[-1,1]` inputs (range 2).
11. `student_dataset.py`'s docstring says "G/R/NIR" but `LISS4_BANDS = (3, 2, 7)` is R/G/NIR.
12. `khushi/student_gan.py` is a byte-identical stale copy of `izma/student_gan.py`.
13. No AMP, no LR schedule, `batch_size=1` — a 50-epoch run over ~9.7k tiles is needlessly slow.

---

## Architecture — what exists today

**Student generator** (`StudentGenerator`), input `I_c` (B,3,256,256) in [-1,1]:

| Stage | Shape | Notes |
|---|---|---|
| `PatchEmbed` 16×16 | (B,256,768) | 256 patches |
| ViT blocks ×12 | (B,256,768) | frozen, tapped at layers 2/5/8/11 |
| `AttentionHead` | (B,1,256,256) | conv stack → bilinear upsample → sigmoid → `I_M` |
| `UNetDecoder` | (B,3,256,256) | 1×1 reduce to 512, 3× `UpBlock` (16→32→64→128), final ConvT→256 + tanh → `G_S` |
| Blend | (B,3,256,256) | `I_g = I_M·G_S + (1−I_M)·I_c` |

All four ViT taps sit at the same 16×16 grid, so the "skips" are bilinearly resized up to each
decoder scale rather than being genuine multi-resolution features.

**Discriminator:** a second, unfrozen copy of the same ViT + a 3-layer conv `PatchGANHead` →
(B,1,16,16) logits. Unconditional. Its "real" input is the *teacher's output*, not real imagery.

**Teacher** (frozen `Pix2Pix_attn_Model`): `att_A = netA(real_A)`, `fake_C = netG2(real_C)`,
`g_B = netG([real_A | fake_C])`, `fake_B = g_B·att_A + real_A·(1−att_A)`.

---

## Which kind of distillation is running now? — Output-level, not feature-level

**It is response/output distillation, and nothing else.** Every teacher-derived target is a finished
image tensor:

| Term | Target | Kind |
|---|---|---|
| `L_1`, `L_ssim`, `L_cw`, `L_out`, `L_cw2`, `L_freq` | `teacher.fake_B` (output image) | output |
| `L_attn` | `teacher.att_A` (output mask) | output |
| `L_sar_mimic` | `teacher.fake_C` (G2's output image) | output |
| `L_feat` | `{}` | **dead** |

So the SAR knowledge reaches the student only as whatever survives into the teacher's final RGB
pixels — and per finding #1 even that LUPI term touches no student weight. This is exactly the case
for switching to feature distillation.

---

## Target architecture

### Student (new — `izma/student_baseline.py`)

Built from the vendored factory functions so shapes match the teacher exactly:

```python
netG_S = define_G(3, 3, ngf=64, 'unet_256', norm='instance',
                  use_dropout=False, init_type='xavier', gpu_ids=gpu_ids)   # G_S, tanh [-1,1]
netA_S = define_A(3, 1, ngf=64, 'unet_256', ...)   # I_M in [0,1] (UnetGenerator + Norm())
netD_S = define_D(3 + 3, ndf=64, 'basic', 3, norm='instance', use_sigmoid=False, ...)
I_g = I_M * G_S(I_c) + (1 - I_M) * I_c
```

`netD_S` is a **conditional** PatchGAN on `cat([I_c, ·])` (standard pix2pix), with LSGAN loss via
`networks.GANLoss` — matching the teacher's setup rather than the current hinge loss.

`unet_256` = `UnetGenerator(num_downs=8)`, a nest of `UnetSkipConnectionBlock`s where each
non-outermost block returns `cat([x, model(x)])`. Traced shapes for a 256×256 input:

| Block (outer→inner) | block output | spatial |
|---|---|---|
| innermost 512/512 | 1024 | 2×2 |
| mid 512/512 ×3 | 1024 | 4×4, 8×8, **16×16** |
| 256/512 | 512 | **32×32** |
| 128/256 | 256 | **64×64** |
| 64/128 | 128 | **128×128** |
| outermost 3/64 | 3 (tanh) | 256×256 |

The student twin differs from `netG` **only** at the outermost block's `input_nc` (3 vs 6), so every
bolded tap has identical shape in both — no reshaping needed anywhere.

### Teacher feature taps (new) — encoder *and* decoder

A `TeacherFeatureTaps` helper registers forward hooks on `UnetSkipConnectionBlock` instances found
via `named_modules()`, ordered by nesting depth (count of `.model` in the name) so indexing is
deterministic rather than relying on fragile `model[1]`/`model[3]` arithmetic.

**Where the encoder features live.** `UnetSkipConnectionBlock.forward` is
`return torch.cat([x, self.model(x)], 1)` — `x` is the **encoder-side** activation at that scale and
`self.model(x)` is the decoder-side upconv output, each `outer_nc` channels wide. So a single hook on
a block yields both halves, sliceable as:

```
enc_k = out[:, :outer_nc]     # encoder activation at this scale
dec_k = out[:, outer_nc:]     # decoder activation at this scale
```

Probed empirically on `unet_256`, teacher (6ch in) vs student (3ch in):

| depth | encoder activation | block output (enc\|dec) |
|---|---|---|
| 1 | 64 @ 128² | 128 @ 128² |
| 2 | 128 @ 64² | 256 @ 64² |
| 3 | 256 @ 32² | 512 @ 32² |
| 4 | 512 @ 16² | 1024 @ 16² |
| 5–7 | 512 @ 8² / 4² / 2² | 1024 @ 8² / 4² / 2² |
| innermost downconv | **512 @ 1×1** (true bottleneck) | — |

Every one of these is **identical between teacher and student**; the only difference in the whole
network is the outermost downconv's `input_nc` (6 vs 3), and its *output* is 64@128² either way. The
512@1×1 bottleneck is not present in any block output and needs its own hook on the innermost
block's `model[1]` (`Conv2d`), confirmed by probe.

**Why the encoder taps matter more than the decoder ones for this project.** The teacher's `netG`
consumes `cat([real_A, fake_C])` — SAR enters the network *at the input*, so the encoder is where
optical and SAR-derived structure first fuse. Matching those activations is precisely the LUPI
objective: force the optical-only student to reconstruct, from 3 bands, the internal representation
the teacher could only build *because* it had SAR. Decoder activations are downstream of that fusion
and already partly decoded toward pixels, which makes them both a weaker signal and largely
redundant with the output-level loss we already have. Encoder taps are the ones doing the work; the
plan taps both, weighted toward the encoder.

Tap set:

- **`netG` encoder → `L_feat_enc`** (primary): 64@128², 128@64², 256@32², 512@16², plus 512@1×1.
- **`netG` decoder → `L_feat_dec`** (secondary, lower weight): the matching `dec_k` halves.
- **`netG2` → `L_sar`** (pure-SAR LUPI): the 512@1×1 bottleneck plus the 16² tap. *Note:* the 1×1
  bottleneck alone is spatially degenerate, so the 16² tap is included to carry SAR spatial
  structure — a global descriptor cannot teach cloud localisation.

Adapters are `nn.Conv2d(C_s, C_t, 1)` per tap by default (settable to `Identity`). Shapes already
match, but student and teacher occupy different band spaces, so a cheap learned 1×1 is the safer
default. Each feature map is normalised to unit RMS before MSE so the 1024-channel taps do not
dominate the 64-channel ones.

### Loss set (new)

```
vs cloud-free GT:   L_adv + 100*L1(I_g, clear) + 10*L_ssim(I_g, clear)
vs teacher:         1*L_out(I_g[:, :2], I_t[:, :2])   # shared R,G only
                  + 2*L_attn(I_M, att_A)
                  + 1.0*L_feat_enc  (netG encoder taps + 1x1 bottleneck)  <- primary LUPI signal
                  + 0.3*L_feat_dec  (netG decoder taps)
                  + 1.0*L_sar       (netG2 taps)
                  + 0.5*L_freq(I_g, clear)
```

The GT weights mirror the teacher's own `loss_G = GAN + 10*ssim + 100*L1`. Dropped: `L_cw` and the
duplicate `L_out`-as-L1 (finding #5). `L_cw2` (channel covariance) is retained only if it is
retargeted at GT.

---

## Implementation

| File | Change |
|---|---|
| `izma/student_dataset.py` | Yield 4 tensors per tile: `cloudy_optical` (B4/B3/B8), `clear_optical` (B4/B3/B8, **new** GT), `teacher_optical` (B4/B3/B2, **new**, teacher-space), `sar`. Triple-match `(roi, patch)` across `ROIs1158_spring_s2`, `_s2_cloudy`, `_s1` (S1 has 9 763 tiles and is the binding constraint). Replicate `prepare_teacher_data.py`'s clip→uint8→[-1,1] path for `teacher_optical` so the teacher sees its training distribution. Apply flips/rot90 identically to all four. Fix the G/R/NIR docstring. |
| `izma/student_baseline.py` | **New.** `BaselineStudent` (netG_S/netA_S + blend), `TeacherFeatureTaps`, adapters, the loss set above, and a training loop with AMP, a `step` LR schedule, and per-epoch validation. |
| `izma/train_student.py` | Build the baseline nets instead of the ViT ones; drop `--phase1-ckpt`; add `--ngf`, `--amp`, `--val-every`. Actually use `val_loader`. Keep `load_teacher()` as-is — it is correct. |
| `izma/visualize_student.py` | Point at the baseline checkpoint; add the GT panel; take paths as CLI args instead of hardcoding. |
| `khushi/PHASE2_BASELINE_PLAN.md` | **New.** A copy of this plan committed into `khushi/` so the Phase-1 side has the Phase-2 contract in-repo (checkpoint handoff, band conventions, tap shapes) rather than only in a session plan file. |
| `khushi/student_gan.py` | Delete the stale duplicate. |
| `izma/student_gan.py` | Leave in place as the ViT variant for when Phase-1 is picked back up. |

**Reuse rather than rewrite:** `psnr` / `ssim` from [khushi/metrics.py](khushi/metrics.py) for
validation metrics; `define_G` / `define_A` / `define_D` / `GANLoss` / `get_scheduler` from
[AttentionGAN-for-Cloud-removal/models/networks.py](AttentionGAN-for-Cloud-removal/models/networks.py);
`load_teacher()` from [izma/train_student.py](izma/train_student.py).

Also worth adding to `.gitignore`: `*.pt.ckpt*`, since the current `*.pt.ckpt` pattern does not
match `student_gan.pt.ckpt10` and leaves ~5.9 GB untracked-but-not-ignored.

---

## Verification

1. **CPU smoke test** — instantiate the student on random `(2,3,256,256)` input; assert `I_g`,
   `I_M`, and every tap have the shapes tabulated above.
2. **Tap shape equality** — assert each student tap (encoder, decoder, and the 512@1×1 bottleneck)
   matches its teacher counterpart channel-for-channel, and that the `enc_k`/`dec_k` split lands on
   `outer_nc`. This is the assumption the whole design rests on and it should fail loudly if `ngf` or
   `num_downs` ever diverge. The shape table above was produced by exactly this probe.
3. **Gradient regression test (guards finding #1)** — run one backward pass with *only* `L_feat`
   and then only `L_sar` active, and assert `netG_S` parameters have non-zero grad norm. This is the
   exact failure the current code has, so it must be a standing check, not a one-off.
4. **Teacher sanity** — dump `att_A` and `fake_B` for a few tiles fed the corrected B4/B3/B2 input
   and confirm they differ visibly from the same tiles fed the old B4/B3/B8 input (proves finding #3
   was real and is fixed).
5. **Short run** — 2 epochs on a ~500-tile subset; confirm `L_feat`/`L_sar` are non-zero and
   decreasing, and that val PSNR/SSIM against GT are logged.
6. **Ablation** — same budget with `beta_feat = mu_sar = 0` (GT-only) vs full distillation. This is
   the number that answers whether distillation is helping at all, and it is only measurable now
   that GT is in the pipeline.

**Caveat to carry forward:** the teacher itself is still mid-debug — `AttentionGAN-for-Cloud-removal/CLAUDE.md`
records `fake_C` as dark and blue/orange at epoch 23, with Experiment 1 (netD2) unresolved. Feature
distillation from a degraded teacher will inherit that. The GT-primary design limits the damage, and
step 6 measures it.
