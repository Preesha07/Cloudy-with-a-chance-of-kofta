# Optical-Only Cloud Removal for India's Northeast via Knowledge Distillation from a SAR-Optical Teacher

## A Complete Research & Architecture Plan

---

## 1. Sensor Comparison: LISS-4 vs Sentinel-1 vs Sentinel-2

### 1.1 LISS-4 (Resourcesat-2/2A)

LISS-4 is a high-resolution pushbroom camera aboard India's Resourcesat-2 (launched 2011) and Resourcesat-2A (launched 2016). It operates in **3 spectral bands** in the Visible and Near-Infrared (VNIR) range:

| Band | Name  | Wavelength (µm) | Resolution |
|------|-------|------------------|------------|
| B2   | Green | 0.52 – 0.59      | 5.8 m      |
| B3   | Red   | 0.62 – 0.68      | 5.8 m      |
| B4   | NIR   | 0.77 – 0.86      | 5.8 m      |

**Key characteristics:**
- **Spatial resolution:** 5.8 m (all bands)
- **Swath width:** 70 km (mono mode) / 23.9 km (multispectral mode on RS-1; 70 km on RS-2)
- **Radiometric resolution:** 10-bit (upgraded from 7-bit on RS-1)
- **Orbit repeat cycle:** 24 days (sun-synchronous, 817 km altitude)
- **Revisit capability:**
  - **24 days** — nadir-pointing systematic coverage (single satellite)
  - **~5 days** — targeted revisit only, using ±26° cross-track steering to view adjacent paths off-nadir
  - **25–26 days** — combined RS-2 + RS-2A systematic LISS-4 Mx coverage (per eoPortal/ISRO 2018)
  - **Note:** The 5-day figure widely cited refers to the *steerable targeted* mode, not systematic mapping. For building a multitemporal NER dataset, plan for ~24-day gaps between nadir acquisitions of the same area.
- **No blue band, no SWIR band, no SAR capability**

**Sources:**
- eoPortal Resourcesat-2: https://www.eoportal.org/satellite-missions/resourcesat-2
- Resourcesat-2 Data Users' Handbook: https://earth.esa.int/eogateway/documents/20142/37627/ResourceSat-2-Data-User-Handbook.pdf
- ESA Earth Online: https://earth.esa.int/eogateway/missions/resourcesat-2
- Spatial Thoughts LISS-4 guide: https://spatialthoughts.com/2023/12/25/liss4-processing-xarray/
- USGS LISS-4 characterization: https://www.usgs.gov/publications/system-characterization-report-resourcesat-2a-linear-imaging-self-scanning-4-sensor

**Critical limitation:** LISS-4 has no SWIR band. The co-flying LISS-3 sensor does carry a SWIR band (1.55–1.70 µm) at 24 m resolution. Recent work (James et al., 2025) has demonstrated CNN-based super-resolution to synthesize a 5 m SWIR band for LISS-4 from the LISS-3 SWIR, which could be leveraged as a fourth input channel.

### 1.2 Sentinel-1 (C-band SAR)

Sentinel-1 is an active C-band SAR sensor at 5.405 GHz (~5.5 cm wavelength):

| Parameter        | Specification                              |
|------------------|--------------------------------------------|
| Sensor type      | C-band SAR (active radar)                  |
| Frequency        | 5.405 GHz                                  |
| Polarization     | Dual-pol: VV+VH (over land), HH+HV        |
| IW Mode resolution | 5 m × 20 m (single-look); 10 m GRD      |
| Swath (IW)       | 250 km                                     |
| Revisit          | 6 days (constellation) / 12 days (single)  |

**Key capability:** All-weather, day/night imaging. Microwave signals penetrate clouds, making SAR the go-to auxiliary data source for cloud removal.

### 1.3 Sentinel-2 (MSI)

Sentinel-2's Multi-Spectral Instrument has 13 bands spanning VNIR to SWIR:

| Band | Central λ (nm) | Resolution | Primary Use          |
|------|-----------------|------------|----------------------|
| B1   | 443             | 60 m       | Aerosols             |
| B2   | 490             | 10 m       | Blue                 |
| B3   | 560             | 10 m       | Green                |
| B4   | 665             | 10 m       | Red                  |
| B5   | 705             | 20 m       | Red-edge 1           |
| B6   | 740             | 20 m       | Red-edge 2           |
| B7   | 783             | 20 m       | Red-edge 3           |
| B8   | 842             | 10 m       | NIR                  |
| B8A  | 865             | 20 m       | NIR narrow           |
| B9   | 945             | 60 m       | Water vapour         |
| B10  | 1375            | 60 m       | Cirrus               |
| B11  | 1610            | 20 m       | SWIR 1               |
| B12  | 2190            | 20 m       | SWIR 2               |

### 1.4 Key Comparison for This Project

| Feature               | LISS-4           | Sentinel-1        | Sentinel-2         |
|-----------------------|------------------|-------------------|--------------------|
| Sensor type           | Optical (passive)| SAR (active)      | Optical (passive)  |
| Bands                 | 3 (G,R,NIR)      | 2 pol (VV,VH)     | 13 (VNIR+SWIR)     |
| Spatial resolution    | 5.8 m            | 10 m (GRD-IW)     | 10/20/60 m         |
| Revisit (systematic)  | 24 days (nadir)  | 6 days (2-sat)    | 5 days (2-sat)     |
| Cloud penetration     | **No**           | **Yes**            | **No**             |
| SWIR capability       | No (but LISS-3)  | N/A               | Yes (B11, B12)     |
| India NE coverage     | Available (NRSC) | Free (Copernicus) | Free (Copernicus)  |

**The core challenge:** LISS-4 is purely optical with only 3 VNIR bands and **no paired SAR sensor** in the Resourcesat constellation. The teacher model (cloud-attention GAN) was designed for Sentinel-1 SAR + Sentinel-2 optical pairs. Our student must work with LISS-4 optical data alone.

---

## 2. The Teacher: Cloud-Attention GAN (Zhang et al., 2023)

### 2.1 Architecture Summary

The cloud-attention GAN is an end-to-end cGAN with four modules:

1. **Translation module** (U-Net): Converts SAR image → fake optical image, bridging the domain gap between radar and optical modalities.
2. **Attention module** (U-Net): Takes the cloudy optical image and produces a soft attention map (values 0–1) indicating cloud thickness, without needing a separate cloud detection step.
3. **Generator** (U-Net, 8 down + 8 up with skip connections): Takes the cloudy image and fake optical image to generate recovered content for the cloudy regions.
4. **Discriminator** (70×70 PatchGAN): Judges real vs. generated cloud-free images (concatenated with SAR).

The final output is formed as:
$$I_g = I_M \odot G(I_c, I_f) + (1 - I_M) \odot I_c$$

where $I_M$ is the attention map, $G$ is the generator output, and $I_c$ is the cloudy input.

### 2.2 Loss Functions

- **Translation loss** $L_T$: L1 + SSIM between fake optical and target
- **Attention loss** $L_A$: L1 norm of attention map (encourages sparsity)
- **Generator loss** $L_G$: adversarial (LSGAN) + λ₁·L1 + λ₂·SSIM
- **Discriminator loss** $L_D$: LSGAN loss

Hyperparameters: λ₁=100, λ₂=10, λ₃=1, δ=1

### 2.3 Results (SEN12MS-CR Dataset)

| Method                     | mPSNR | mRMSE | mSSIM |
|----------------------------|-------|-------|-------|
| pix2pix                    | 23.24 | 20.29 | 0.606 |
| SAR-opt-GAN                | 28.17 | 11.72 | 0.807 |
| Simulation-Fusion GAN      | 28.52 | 11.27 | 0.829 |
| GLF-CR                     | 28.67 | 13.22 | 0.820 |
| **Cloud-attention GAN**    | **29.65** | **10.15** | **0.864** |

### 2.4 GitHub Repository

Official code: `https://github.com/Shuaizhang7/AttentionGAN-for-Cloud-removal`

- Based on PyTorch
- Builds on Attention-GAN and CycleGAN codebases
- Data format: trainA (cloudy), trainB (cloud-free), trainC (SAR), as PNG triplets
- Dataset: SEN12MS-CR (Sentinel-1 + Sentinel-2 pairs)

---

## 3. The Challenge: Why Knowledge Distillation?

### 3.1 Problem Statement

India's Resourcesat satellites do not carry SAR instruments. For India's northeast region (NER), which suffers from persistent cloud cover (especially during the monsoon season), we need a cloud removal model that works with **LISS-4 optical data alone**—without any SAR input at inference time.

### 3.2 Why Not Just Train an Optical-Only Model?

A purely optical model for thick cloud removal faces a fundamental information bottleneck: the ground truth under thick clouds is completely invisible to the optical sensor. SAR provides the missing structural and textural information. Without it, the model must hallucinate content, leading to blurred, spectrally inconsistent, or structurally incorrect reconstructions.

### 3.3 The Knowledge Distillation Approach

The key insight: during training, we *do* have access to SAR data (from Sentinel-1, which covers India). We can train the teacher (cloud-attention GAN) on co-registered Sentinel-1 SAR + Sentinel-2 optical pairs over India's NER. Then, we distill the teacher's "understanding" of how SAR information helps reconstruct optical images into a student that operates on optical data only.

The student learns an implicit mapping: *given these optical features in cloud-free regions, the teacher would have produced this particular reconstruction in the cloudy regions using SAR*. Essentially, the student learns to infer what SAR would have revealed from the optical context alone.

---

## 4. Proposed Architecture: Two-Phase Training Pipeline

**The student model is a GAN throughout both phases.** The key insight — originating from GAN-MAE (Fei et al., CVPR 2023) — is that the BERT-like masked reconstruction pretraining and the adversarial training can be unified into a single GAN framework. Instead of pretraining an autoencoder separately and then awkwardly transplanting its encoder into a GAN (which creates a generator-discriminator imbalance), we train the generator and discriminator together on the masking task from the start. Both sides co-evolve, stay balanced, and learn NER terrain representations simultaneously.

```
Phase 1:  GAN-MAE pretraining on NER imagery
          └── Generator = ViT encoder + decoder (reconstructs masked patches)
          └── Discriminator = ViT (shared backbone) + classification head
                              (detects real vs synthesized patches)
          └── Both sides share the ViT backbone → no imbalance
                              │
                              ▼  (fork the shared backbone)
Phase 2:  Student cloud-removal GAN trained via KD from teacher
          └── Generator = shared ViT backbone + U-Net decoder + attention head
          └── Discriminator = shared ViT backbone + PatchGAN head
          └── Both sides start from same pretrained weights → balanced
```

---

### 4.1 Phase 1: GAN-MAE Pretraining on NER Imagery

**What this phase produces:** A shared ViT backbone that both the student generator and discriminator inherit in Phase 2, ensuring balanced initialization.

**Why GAN-MAE instead of plain MAE?** Three reasons:
1. **Balance:** In a plain MAE, only the encoder gets pretrained. Plugging a strong pretrained encoder into a GAN generator while the discriminator starts from scratch creates an asymmetry — the generator is "too good" initially, the discriminator can't provide useful gradients, and training destabilizes. GAN-MAE avoids this because both sides train adversarially from the start with a shared backbone.
2. **Efficiency:** GAN-MAE achieves comparable representation quality to MAE-1600 epochs in just 200 epochs (Fei et al., CVPR 2023), because the discriminator's adversarial signal forces sharper, more perceptually meaningful patch reconstructions than the MSE loss alone.
3. **Relevance to downstream task:** The pretraining task (reconstruct missing image patches from surrounding context) is structurally analogous to the downstream task (reconstruct cloud-occluded regions from surrounding cloud-free context). The GAN learns to both generate plausible patch content and judge its realism — exactly the skills needed for cloud removal.

**GAN-MAE architecture (following Fei et al., CVPR 2023):**

- **Generator:** Standard MAE structure. A ViT encoder processes only the visible (unmasked) patches, then a lightweight transformer decoder reconstructs the masked patches. Loss: pixel-level MSE on reconstructed patches.
- **Discriminator:** Takes the "corrupt image" (visible real patches + generator's synthesized patches stitched back together) and performs per-patch binary classification: is each patch real or synthesized?
- **Shared backbone:** The ViT backbone parameters are shared between the generator's encoder and the discriminator's encoder. This is the critical design choice — both sides see the same representation space and co-evolve at the same rate.
- **Training:** Alternating generator/discriminator updates, standard GAN training dynamics. The generator tries to reconstruct patches that the discriminator can't distinguish from real ones; the discriminator tries to detect fakes.

**Data source:** Cloud-free LISS-4 tiles of India's NER from NRSC/Bhuvan/Bhoonidhi, supplemented with Sentinel-2 data resampled and spectrally aligned to LISS-4 bands (S2-B3→Green, S2-B4→Red, S2-B8→NIR).

**Training details:**
- Input channels: 3 (G, R, NIR); optionally 4 with synthetic SWIR from LISS-3
- Patch size: 16×16 pixels at 5.8 m → each patch covers ~93 m × 93 m
- Image size: 256×256 (~1.5 km × 1.5 km)
- Masking ratio: 75%
- Spectral encoding: learnable per-band embedding added to patch embeddings (following SatMAE)
- Augmentation: random rotation (0/90/180/270°), horizontal/vertical flip, random crop
- Target epochs: ~200–400 (GAN-MAE converges much faster than vanilla MAE)

**Output of Phase 1:** A shared pretrained ViT backbone. The MAE decoder and discriminator classification head are discarded. The backbone is forked into the student generator and discriminator for Phase 2.

**Key consideration — spectral alignment:** LISS-4 and Sentinel-2 share overlapping but not identical bands:

| LISS-4 | Wavelength     | Closest S2 Band | S2 Wavelength  |
|--------|----------------|-----------------|----------------|
| B2 (G) | 0.52–0.59 µm  | B3              | 0.543–0.578 µm|
| B3 (R) | 0.62–0.68 µm  | B4              | 0.650–0.680 µm|
| B4 (NIR)| 0.77–0.86 µm | B8              | 0.785–0.900 µm|

The spectral overlap is high enough for transfer. During pretraining, mix LISS-4 and spectrally-aligned Sentinel-2 NER data to increase the corpus size.

### 4.2 Phase 2: Training the Student Cloud-Removal GAN via Knowledge Distillation

**What this phase produces:** The final deployable model — a cloud-removal GAN (generator + discriminator) that operates on LISS-4 optical images alone, without SAR.

#### 4.2.1 Teacher Preparation

**Step A:** Acquire co-registered Sentinel-1 (SAR) + Sentinel-2 (optical) pairs over India's NER from the SEN12MS-CR dataset or by direct download from Copernicus Open Access Hub. Focus on monsoon-season acquisitions (June–September) for maximum cloud variety.

**Step B:** Train the cloud-attention GAN teacher on these NER-specific Sentinel data pairs (or fine-tune from the authors' pretrained weights on SEN12MS-CR). The teacher is now an expert at NER cloud removal using SAR+optical.

**Step C:** Freeze the teacher. Extract intermediate representations for distillation.

#### 4.2.2 Student GAN Architecture

The student is a **conditional GAN** mirroring the teacher's structure, but operating on optical input only. Both its generator and discriminator are initialized from the shared GAN-MAE backbone from Phase 1, ensuring balanced starting points.

**Student Generator (G_S):**
- **Encoder backbone:** The shared ViT from GAN-MAE Phase 1.
- **Cloud attention head:** A lightweight convolutional head branching off the ViT features to predict a soft attention map (values 0–1), mirroring the teacher's attention module.
- **Decoder:** U-Net style CNN decoder (trained from scratch) that takes ViT features + attention map and produces the reconstructed content for cloudy regions.
- **Output combination (same as teacher):** $I_g = I_M \odot G_S(I_c) + (1 - I_M) \odot I_c$, where $I_M$ is the student's attention map. Cloud-free regions pass through unchanged.
- **Input:** Cloudy LISS-4 image only (3 channels: G, R, NIR; optionally +SWIR)
- **Output:** Cloud-free image (3 channels) + attention map

**Student Discriminator (D_S):**
- **Encoder backbone:** The same shared ViT from GAN-MAE Phase 1 (forked copy — starts identical to the generator's backbone, then diverges during training).
- **Classification head:** PatchGAN-style head on top of the ViT features, conditioned on the optical input only (no SAR concatenation).

**Why this works:** Both the generator and discriminator start from the same pretrained representation. The generator has a slight structural advantage (it gets the decoder and attention head), but the discriminator has a matching representational advantage (same backbone strength). The adversarial game begins from a balanced position, avoiding the collapse modes that plagued the earlier pretrained-encoder-only approach.

#### 4.2.3 Knowledge Distillation Losses with SAR as Privileged Information

This setup is a textbook case of **Learning Using Privileged Information (LUPI)** (Vapnik & Vashist, 2009): SAR is the privileged modality — available during training (from Sentinel-1) but absent at inference (LISS-4 has no SAR). The goal is to encode as much SAR-derived knowledge as possible into the student's loss function so that the student internalizes what SAR would have revealed, even though it will never see SAR at test time.

The student's total loss combines standard GAN losses, teacher-matching distillation losses, and SAR-aware privileged information losses:

**A. Standard GAN losses (ground truth supervision):**
- Adversarial: $L_{adv}$ (LSGAN)
- Pixel-wise: $L_1(I_g^S, I_t)$
- Structural: $L_{SSIM}(I_g^S, I_t)$
- Attention sparsity penalty: $L_A^S = \| A_S(I_c) \|_1$

**B. Output-level distillation (response-based KD):**
$$L_{out} = \| G_S(I_c^{LISS4}) - G_T(I_c^{S2}, I_s^{S1}) \|_1 + \lambda_{ssim} \cdot L_{SSIM}(G_S, G_T)$$

The student's output should match the teacher's SAR-aided output. This is the most direct channel for SAR knowledge transfer — the teacher "saw through the clouds" using SAR and produced a specific reconstruction; the student must match it using optical alone.

**C. Attention map distillation:**
$$L_{attn} = \| A_S(I_c) - A_T(I_c) \|_1$$

Both attention modules take only the optical input, so this is directly transferable.

**D. Feature-level distillation (hint-based KD):**
$$L_{feat} = \sum_{l \in \mathcal{L}} \| f_S^l(I_c) - \phi_l(f_T^l(I_c, I_s)) \|_2^2$$

Where $f^l$ denotes feature maps at selected layers, and $\phi_l$ is a learned linear projection to handle dimensionality mismatches. Note: the teacher's features at these layers are conditioned on SAR input — so the student is being pushed to produce SAR-informed internal representations from optical input alone.

**E. SAR-aware privileged information losses (novel to this architecture):**

These losses go beyond generic distillation by explicitly encoding what SAR contributes:

**E1. SAR translation mimicry loss:** The teacher's translation module learns a mapping SAR → fake optical. We can use this pretrained translation module as a frozen feature extractor during student training. Given a Sentinel-1 SAR image paired with the training sample, we extract the teacher's internal SAR-to-optical translation features and force the student's generator to produce features that are consistent with what the SAR translation would have predicted:
$$L_{SAR\text{-}mimic} = \| f_S^{bottleneck}(I_c) - f_T^{translation}(I_s) \|_2^2$$

This directly injects "what does SAR say the ground looks like here?" into the student's feature space. The student learns an implicit SAR hallucination — it can't see the SAR, but its internal features should be consistent with what SAR would have produced.

**E2. Cloud-region-weighted reconstruction loss:** SAR's primary advantage is in thick cloud regions where optical is blind. We weight the pixel-level losses by the teacher's attention map (which indicates cloud thickness) to force the student to pay disproportionate attention to exactly the regions where SAR matters most:
$$L_{cloud\text{-}weighted} = \| A_T(I_c) \odot (G_S(I_c) - I_t) \|_1$$

This ensures that reconstruction errors in thick-cloud regions (where SAR information is most critical) are penalized far more heavily than errors in thin-cloud or clear regions.

**E3. Frequency-decoupled cross-modal alignment:** Following recent work on cross-modal KD (Liu et al., 2025), low-frequency features (large-scale structure, land cover layout) tend to be consistent across SAR and optical modalities, while high-frequency features (textures, speckle patterns) are modality-specific. We apply strong alignment loss on low-frequency teacher features and relaxed alignment on high-frequency features to avoid forcing the student to learn SAR-specific noise patterns:
$$L_{freq} = \lambda_{low} \| \text{LPF}(f_S) - \text{LPF}(f_T) \|_2^2 + \lambda_{high} \max(0, \| \text{HPF}(f_S) - \text{HPF}(f_T) \|_2^2 - \epsilon)$$

where LPF/HPF are low-pass/high-pass filters applied to feature maps, and $\epsilon$ is a margin allowing high-frequency divergence.

**F. Total student loss:**
$$L_{student} = \underbrace{L_{adv} + \lambda_1 L_1 + \lambda_2 L_{SSIM} + \delta L_A^S}_{\text{standard GAN}} + \underbrace{\alpha L_{out} + \beta L_{feat} + \gamma L_{attn}}_{\text{teacher distillation}} + \underbrace{\mu L_{SAR\text{-}mimic} + \nu L_{cloud\text{-}weighted} + \eta L_{freq}}_{\text{SAR privileged information}}$$

#### 4.2.4 Student GAN Training Protocol

Because both the generator and discriminator start from the same GAN-MAE pretrained backbone, the adversarial game begins balanced. The training proceeds in two stages:

1. **Distillation-heavy training (epochs 1–80):** High distillation weight (α,β,γ large), moderate adversarial weight. Train all components: ViT backbones (both generator and discriminator, with low LR), U-Net decoder, attention head, PatchGAN head. The student GAN learns the cloud removal mapping primarily by imitating the teacher's outputs, attention maps, and intermediate features. The discriminator co-evolves because it started from the same backbone and receives standard adversarial gradients throughout.
2. **Adversarial refinement (epochs 81–150):** Reduce distillation weight, increase adversarial weight. The discriminator drives the generator toward sharper, more photorealistic results. The distillation signal acts as a regularizer preventing mode collapse, while the adversarial signal pushes perceptual quality beyond what pixel-level distillation alone can achieve.

---

## 5. Data Pipeline for India's Northeast

### 5.1 Data Sources

| Source | Sensor | Use | Access |
|--------|--------|-----|--------|
| SEN12MS-CR | S1+S2 | Teacher training, paired triplets | Public (TUM) |
| Copernicus Hub | S1 GRD, S2 L2A | NER-specific teacher fine-tuning | Free (Copernicus) |
| NRSC Bhuvan | LISS-4 Mx | Student pretraining & inference | Visualization free; download requires NRSC registration |
| NRSC Bhoonidhi | LISS-4 orders | Targeted NER data procurement | Paid/research license |

### 5.2 NER-Specific Considerations

India's NER states (Assam, Meghalaya, Arunachal Pradesh, Nagaland, Manipur, Mizoram, Tripura, Sikkim) present unique challenges:
- **Extremely high cloud cover:** 80%+ during monsoon (June–September)
- **Diverse terrain:** From Brahmaputra floodplains to Himalayan foothills
- **Dense vegetation:** Tropical/subtropical forests dominate
- **Limited LISS-4 cloud-free reference data:** Precisely why we need this model

### 5.3 Data Preparation Strategy

1. **Build a Sentinel-1 + Sentinel-2 NER paired dataset** using Google Earth Engine or the Copernicus API, filtering for scenes with partial cloud cover (10–80%) to get both cloudy regions and cloud-free ground truth within the same tile.
2. **Create simulated cloudy data** from cloud-free LISS-4 tiles by applying extracted cloud masks (from real S2 cloudy scenes), following the same protocol as the teacher paper.
3. **Spectral band alignment:** Resample Sentinel-2 B3/B4/B8 to 5.8 m to match LISS-4 resolution. Apply spectral response function corrections using known filter curves.

---

## 6. Using All Available LISS-4 Channels

LISS-4 natively provides 3 channels (G, R, NIR). To maximize information:

### 6.1 Native 3-Band Input
Use all three LISS-4 bands directly: B2 (Green, 0.52–0.59 µm), B3 (Red, 0.62–0.68 µm), B4 (NIR, 0.77–0.86 µm).

### 6.2 Synthetic SWIR (4th Channel)
LISS-3 flies on the same satellite and acquires a SWIR band (1.55–1.70 µm) at 24 m simultaneously. Use the GSR-SWIR method (James et al., 2025) — a CNN-based guided super-resolution — to generate a 5 m SWIR band from the 24 m LISS-3 SWIR guided by the three LISS-4 VNIR bands. SWIR is particularly useful because it can partially see through thin clouds and haze.

### 6.3 Derived Indices as Additional Channels
Compute vegetation and moisture indices as pseudo-channels to give the model richer representations:
- **NDVI** = (NIR − R) / (NIR + R) — vegetation density
- **Green NDVI** = (NIR − G) / (NIR + G) — chlorophyll sensitivity
- **If SWIR available: NDWI** = (NIR − SWIR) / (NIR + SWIR) — moisture/water

This gives us up to 6 input channels: G, R, NIR, SWIR_synth, NDVI, GNDVI.

---

## 7. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│              PHASE 1: GAN-MAE PRETRAINING (NER imagery)             │
│                                                                     │
│  Cloud-free LISS-4 / Sentinel-2 NER tiles                          │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────┐     ┌──────────────────────────────────┐          │
│  │ Mask 75%    │────▶│  GENERATOR                       │          │
│  │ of patches  │     │  ViT Encoder ──▶ Decoder ──▶     │          │
│  └─────────────┘     │  (shared       reconstructed     │          │
│                      │   backbone)     patches           │          │
│                      └──────────┬───────────┬────────────┘          │
│                                 │           │                       │
│                          Shared │    Stitch into                    │
│                         weights │    "corrupt image"                │
│                                 │           │                       │
│                      ┌──────────┴───────────▼────────────┐          │
│                      │  DISCRIMINATOR                     │          │
│                      │  ViT Encoder ──▶ Per-patch         │          │
│                      │  (shared         real/fake?        │          │
│                      │   backbone)                        │          │
│                      └───────────────────────────────────┘          │
│                                 │                                   │
│                     Save shared ViT backbone                        │
│                     (discard decoder + classif. head)                │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          │  Fork backbone into:  │
          │  • Generator copy     │
          │  • Discriminator copy │
          └───────────┬───────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────────┐
│       PHASE 2: KNOWLEDGE DISTILLATION (cloud removal)               │
│                                                                     │
│  ┌─── TEACHER (frozen) ──────────────────────────────────────────┐ │
│  │  S1 SAR ──▶ Translation ──▶ Fake Optical                      │ │
│  │  S2 Cloudy ──▶ Attention ──▶ Attn Map                         │ │
│  │       └────────────────▶ Generator ──▶ Teacher Output          │ │
│  │                                  │                             │ │
│  │                          Teacher Features (hint layers)        │ │
│  └──────────────────────────────────┼─────────────────────────────┘ │
│                                     │                               │
│           Distillation signals:     │                               │
│           L_out, L_feat, L_attn ◄───┘                               │
│                     │                                               │
│  ┌─── STUDENT GAN ──▼────────────────────────────────────────────┐ │
│  │                                                                │ │
│  │  LISS-4 Cloudy ──▶ ViT Backbone ──┬──▶ Attn Head ──▶ Attn Map│ │
│  │  (G,R,NIR)         (from Phase 1)  │                          │ │
│  │                                    └──▶ U-Net Dec ──▶ Recon   │ │
│  │                                                       │       │ │
│  │                    I_g = M⊙G(I_c) + (1-M)⊙I_c ◄──────┘       │ │
│  │                              │                                 │ │
│  │                              ▼                                 │ │
│  │                    ViT Backbone ──▶ PatchGAN Head              │ │
│  │                    (from Phase 1)   (real/fake?)               │ │
│  │                    [Discriminator]                              │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              │                                      │
│                              ▼                                      │
│                    Cloud-free LISS-4 output                          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 8. Expected Outcomes and Limitations

### 8.1 What the Student Can Realistically Achieve

- **Thin cloud / haze removal:** High confidence. Optical features under thin clouds are partially visible; the pretrained encoder should learn to enhance them.
- **Cloud-free region preservation:** High confidence. The attention mechanism directly ensures clean regions are passed through unchanged.
- **Moderate cloud reconstruction:** Moderate confidence. The distilled knowledge from SAR-aided reconstruction, combined with contextual priors from GAN-MAE pretraining on NER data, should enable reasonable infilling.
- **Thick cloud reconstruction:** Lower confidence. Without SAR, thick cloud regions are a pure hallucination problem. The student will rely on contextual priors (what land cover typically surrounds this region?). Performance will degrade compared to the SAR-aided teacher.

### 8.2 Key Limitations

1. **Information ceiling:** No amount of distillation can fully recover SAR's cloud-penetrating information from optical data alone. The student will always be bounded by what context the cloud-free portions of the image can reveal.
2. **LISS-4 data availability:** LISS-4 is not as freely available as Sentinel-2. Building the pretraining corpus may require NRSC cooperation.
3. **Spectral mismatch:** Only 3 (or 4 with synthetic SWIR) channels vs. the teacher's richer spectral input. The student has strictly less spectral information.
4. **Resolution mismatch:** LISS-4 (5.8 m) is higher resolution than Sentinel-2 (10 m), which means the distillation from S2-trained teacher to LISS-4 student involves a resolution domain shift that needs careful handling.

### 8.3 Mitigation Strategies

- Use **multitemporal** LISS-4 data (multiple dates) as additional context. If a recent cloud-free acquisition exists, it can serve as a strong prior.
- Incorporate **topographic data** (e.g., CartoDEM at 10 m from NRSC) as an auxiliary channel, since terrain strongly constrains land cover.
- Consider training a secondary **SAR-to-optical hallucination network** on the Sentinel data that can be used to generate pseudo-SAR features at inference, though this adds complexity.

---

## 9. Implementation Roadmap

### Step 1: Data Collection & Preparation
- Download SEN12MS-CR dataset; filter for ROIs near India's NER latitudes
- Acquire NER Sentinel-1/2 pairs via Copernicus
- Acquire LISS-4 cloud-free tiles from NRSC Bhuvan/Bhoonidhi
- Build data preprocessing pipeline (co-registration, spectral alignment, patch extraction)
- Generate simulated cloudy LISS-4 data using extracted cloud masks

### Step 2: Teacher Training
- Train cloud-attention GAN on SEN12MS-CR (baseline reproduction)
- Fine-tune on NER-specific Sentinel data
- Validate teacher on held-out NER test set
- Freeze teacher; extract and cache intermediate features for distillation

### Step 3: GAN-MAE Pretraining
- Prepare cloud-free LISS-4 + Sentinel-2 NER corpus
- Implement GAN-MAE with shared ViT backbone (adapt from Fei et al. CVPR 2023)
- Train GAN-MAE for ~200–400 epochs on NER imagery
- Evaluate backbone quality via linear probing on land cover classification

### Step 4: Student GAN Training via Knowledge Distillation
- Fork GAN-MAE backbone into student generator and discriminator
- Add U-Net decoder + attention head (generator) and PatchGAN head (discriminator)
- Implement the full loss function including SAR privileged information losses (L_SAR-mimic, L_cloud-weighted, L_freq)
- Run distillation training with the frozen teacher
- Hyperparameter sweep on loss weights (α, β, γ, μ, ν, η)

### Step 5: Evaluation & Ablation Studies
- Quantitative evaluation (PSNR, SSIM, RMSE) on simulated cloudy LISS-4 data
- Qualitative evaluation on real monsoon-season LISS-4 scenes
- Comparison with optical-only baselines (pix2pix, inpainting methods)
- Key ablations:
  - GAN-MAE pretrained student vs. from-scratch student (Phase 1 contribution)
  - With vs. without SAR privileged information losses (E1/E2/E3 contribution)
  - Number of input channels (3 vs 4 vs 6)
  - Impact of cloud thickness on reconstruction quality (thin vs moderate vs thick)

---

## 10. References

**Core method papers:**
- Zhang, S., Li, X., Zhou, X., Wang, Y., & Hu, Y. (2023). Cloud removal using SAR and optical images via attention mechanism-based GAN. *Pattern Recognition Letters*, 175, 8–15. [Teacher model]
- GitHub (teacher code): `https://github.com/Shuaizhang7/AttentionGAN-for-Cloud-removal`
- Fei, Z., Fan, M., Zhu, L., Huang, J., Wei, X., & Wei, X. (2023). Masked Auto-Encoders Meet Generative Adversarial Networks and Beyond. *CVPR 2023*, 24449–24459. [GAN-MAE pretraining — Phase 1]

**Knowledge distillation and privileged information:**
- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. *NIPS Workshop*.
- Vapnik, V. & Vashist, A. (2009). A new learning paradigm: Learning using privileged information. *Neural Networks*, 22(5–6), 544–557. [LUPI framework — SAR as privileged modality]
- Liu, J., et al. (2025). Distilling Cross-Modal Knowledge via Feature Disentanglement. [Frequency-decoupled cross-modal KD] GitHub: `https://github.com/Johumliu/FD-CMKD`

**GAN balance and pretraining:**
- Ham, H., Jun, T. J., & Kim, D. (2020). Unbalanced GANs: Pre-training the Generator of GAN using VAE. *arXiv:2002.02112*.
- Sauer, A., Chitta, K., Müller, J., & Geiger, A. (2021). Projected GANs Converge Faster. *NeurIPS*.
- Wei, B., Wang, D., Wang, Z., & Zhang, L. (2022). PRAGAN: Progressive Recurrent Attention GAN with Pretrained ViT Discriminator. *Sensors*, 22(24), 9587.

**Remote sensing foundation models:**
- He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). Masked Autoencoders Are Scalable Vision Learners. *CVPR*.
- Cong, Y., et al. (2022). SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery. *NeurIPS*.

**Datasets and sensors:**
- Ebel, P., Meraner, A., Schmitt, M., & Zhu, X. X. (2020). Multisensor data fusion for cloud removal (SEN12MS-CR). *IEEE TGRS*.
- James, L. & Nidamanuri, R. R. (2025). GSR-SWIR: SWIR band for Resourcesat LISS-4 from LISS-3 using guided super-resolution. *Remote Sensing Letters*, 16(9).
