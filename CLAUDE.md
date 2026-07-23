# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A research project on SAR-free cloud removal for ISRO's LISS-4 optical satellite imagery over
India's northeast region (NER), via knowledge distillation from a SAR-optical teacher GAN. The
full research plan, architecture, and literature review live in
[LISS4_Cloud_Removal_KD_Architecture.md](LISS4_Cloud_Removal_KD_Architecture.md) — **read that
file before making architectural changes**, since it is the design doc the rest of the repo
implements. In short:

- **Problem:** LISS-4 (Resourcesat-2/2A) has only 3 optical bands (G, R, NIR) and no SAR
  instrument, so thick-cloud regions are unrecoverable from LISS-4 data alone.
- **Approach:** Train a teacher GAN on Sentinel-1 (SAR) + Sentinel-2 (optical) pairs where SAR
  can see through clouds, then distill its SAR-derived knowledge into a student GAN that only
  ever sees optical input at inference time (SAR as "privileged information", per Vapnik's LUPI
  framework).
- **Two-phase student training:** Phase 1 pretrains a shared ViT backbone via GAN-MAE (masked
  autoencoding unified with adversarial training, so generator and discriminator co-evolve from a
  balanced start) on cloud-free NER imagery. Phase 2 forks that backbone into a student
  generator + discriminator and trains it via multiple distillation losses (output-level,
  attention-map, feature-level/hint-based, plus SAR-privileged losses: translation mimicry,
  cloud-region-weighted reconstruction, frequency-decoupled alignment) against the frozen teacher.

## Repository layout

This is not a single application — it's a workspace combining two vendored third-party
repositories (the teacher GAN and a reference multi-temporal cloud-removal toolbox), the
project's own in-progress Phase-1/Phase-2 implementation, and dataset directories.

- **`AttentionGAN-for-Cloud-removal/`** — vendored code for the teacher model, "Cloud removal
  using SAR and optical images via attention mechanism-based GAN" (Zhang et al. 2023). A
  pix2pix/CycleGAN-derived codebase (`train.py` / `test.py`, `models/`, `options/`, `data/`)
  trained on `trainA`/`trainB`/`trainC` (cloudy / cloud-free / SAR) PNG triplets in the
  SEN12MS-CR format. Treat this as upstream reference code — mirror its conventions rather than
  refactoring it wholesale.
- **`SEN12MS-CR-TS/`** — vendored toolbox for the SEN12MS-CR-TS dataset (Ebel et al. 2022),
  supporting multi-modal, multi-temporal cloud removal. Provides `data/dataLoader.py`
  (`SEN12MSCRTS` and `SEN12MSCR` PyTorch `Dataset` classes — usable standalone, see
  `standalone_dataloader.py`), plus its own `train.py`/`test.py`/`models/`/`options/` for
  STGAN-derived temporal-branched models. Also vendored/reference code.
- **`khushi/`** — the project's own from-scratch implementation of Phase 1 (GAN-MAE / ViT
  pretraining), separate from the two vendored repos above:
  - `vit_pretrain.py` — the ViT MAE backbone (`PatchEmbed`, `ViTBackbone`, `MAEPretrainer`,
    `MAEConfig`), built for arbitrary channel counts (3-band LISS-4 R/G/NIR, or 4-band with
    synthetic SWIR). `export_backbone()` is the Phase 1 → Phase 2 handoff point.
  - `sen12mscr_dataset.py` — `SEN12MSCROptical`, a `Dataset` that reads only the 3 Sentinel-2
    bands matching LISS-4 (B4→R, B3→G, B8→NIR) from cloud-free SEN12MS-CR tiles, used to keep
    the ViT pretraining domain aligned with the LISS-4 student's eventual input.
  - `pretrain_mae.py` — the training driver/CLI for the above (cosine LR schedule, AMP, exports
    `mae_backbone.pt`).
  - `mae_backbone.pt` — a large (~340MB) trained backbone checkpoint; treat as build output, not
    something to hand-edit.
- **`Bhoonidhi-Data/`** — data acquisition from ISRO's Bhoonidhi portal (`trial.py`, a
  `BhoonidhiDataFetcher` using OAuth Bearer tokens against `bhoonidhi-api.nrsc.gov.in`).
  Credentials load from `Bhoonidhi-Data/.env` (already gitignored) — never hardcode or print
  credentials from this file.
- **`download_sen12mscrts_roi.py`** (repo root) — standalone downloader for SEN12MS-CR
  (mono-temporal) or SEN12MS-CR-TS (time-series) archives from the TUM MediaTUM servers. Supports
  `--dataset {mono,ts}`, `--region` (asiaEast is the India-side shard for the TS set),
  `--include-s1`, `--download-only`/`--extract-only` for staged runs. Downloads land in
  `SEN12MSCR/` by default.
- **`SEN12MSCR/`** — downloaded dataset content (Sentinel-1/Sentinel-2 ROI tiles), fetched via
  the downloader script above. Large binary data, not source.
- **Model/data science identity note:** distinct contributors' work is split into per-person
  top-level directories (e.g. `khushi/`); check there before assuming code lives only under the
  vendored repo names.

## Working in the vendored repos

`AttentionGAN-for-Cloud-removal` and `SEN12MS-CR-TS` each have their own environment files
(`environment.yaml`/`requirements.txt` for the former, a `Dockerfile` for the latter) and their
own `train.py`/`test.py` CLI conventions — see each repo's `README.md` for exact flags. Their
`options/` directories (`base_options.py`, `train_options.py`, `test_options.py`) are the
authoritative source for available CLI flags; don't guess flags, read the parser.

Example commands (paths are illustrative — adjust to your local data location):

```bash
# AttentionGAN teacher: train (see AttentionGAN-for-Cloud-removal/scripts/train.sh)
python train.py --dataroot <path> --name <run_name> --model pix2pix_attn \
  --niter 210 --lr_policy step --dataset_mode unaligned_sar --no_flip

# AttentionGAN teacher: test (see AttentionGAN-for-Cloud-removal/scripts/test.sh)
python test.py --dataroot <path> --name <run_name> --model pix2pix_attn \
  --dataset_mode unaligned_sar --no_flip

# SEN12MS-CR-TS reference model (see SEN12MS-CR-TS/README.md for the full flag list)
python train.py --dataroot <path> --dataset_mode sen12mscrts --model temporal_branched \
  --netG resnet3d_9blocks_withoutBottleneck --include_S1 --input_nc 15 --output_nc 13
```

## khushi/ (Phase 1 GAN-MAE) commands

```bash
cd khushi
python pretrain_mae.py --data-root ../SEN12MSCR/ROIs1158_spring_s2 --epochs 50 --amp
```

Key flags: `--img-size`/`--patch-size` (must divide evenly), `--batch-size`, `--max-steps` (caps
total steps for smoke tests, overriding `--epochs`), `--out` (checkpoint path, default
`mae_backbone.pt`). `vit_pretrain.py` also runs standalone as a smoke test
(`python vit_pretrain.py`) — constructs a `mae_vit_base` model, runs one forward pass on random
4-channel input, and prints backbone param count.

No test suite, linter, or build step exists anywhere in this repository — verification is done by
running the training/pretraining scripts directly and inspecting loss curves or sample outputs.

## Data conventions to keep straight

- **Band ordering:** LISS-4 provides only Green, Red, NIR (no blue, no SWIR natively). When
  deriving LISS-4-equivalent bands from Sentinel-2, the mapping is S2 B3→Green, B4→Red, B8→NIR
  (see the table in the architecture doc and `khushi/sen12mscr_dataset.py`). Don't assume
  standard RGB channel order when touching this data path.
- **SEN12MS-CR vs SEN12MS-CR-TS:** mono-temporal (`SEN12MSCR`, single cloudy/cloud-free/SAR
  triplet per ROI) vs. multi-temporal (`SEN12MS-CR-TS`, whole-year time series). The vendored
  repos and dataloaders are specific to one or the other — check which dataset a given piece of
  code expects before wiring them together.
- Never read or print the contents of `.env` files (e.g. `Bhoonidhi-Data/.env`) when working in
  this repo — they hold live NRSC/ISRO credentials.
