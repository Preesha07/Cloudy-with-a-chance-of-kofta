"""
=================================================================================================
PHASE 2 : STUDENT CLOUD-REMOVAL GAN  (knowledge distillation from the frozen Teacher)
=================================================================================================

    LISS-4 optical input (I_c: G, R, NIR — 3 bands only, no SWIR)
        -> ViT Encoder (weights carried over from Phase-1 GAN-MAE pretraining)
        -> Attention Head        -> soft cloud mask I_M
        -> U-Net Decoder         -> pixel reconstruction G_S(I_c)
        -> Blend:  I_g = I_M ⊙ G_S(I_c) + (1 - I_M) ⊙ I_c
        -> Discriminator: forked ViT (separate copy, not frozen) + PatchGAN head

    Frozen Teacher (SAR + optical, already trained) supplies three distillation signals:
        - I_t      : teacher's cloud-free output (teacher.fake_B)         -> L_out
        - hints    : intermediate ResNet feature maps from teacher.netG    -> L_feat
        - A_c      : teacher's SAR-aware attention map (teacher.att_A)    -> L_attn
        - SAR feats: teacher.netG2's bottleneck features (LUPI signal)    -> L_SAR-mimic

    Combined loss (mirrors your slide's equation):
        L_student = [L_adv + λ1·L1 + λ2·L_SSIM + δ·L_cw]                 (standard GAN)
                  + [α·L_out + β·L_feat + γ·L_attn]                       (teacher distillation)
                  + [μ·L_SAR-mimic + ν·L_cw' + η·L_freq]                  (SAR privileged info)

INTEGRATION NOTES:
    1. Phase-1 backbone  — TappedViTBackbone wraps khushi/vit_pretrain.ViTBackbone and
       captures intermediate-layer tokens at CFG.tap_layers for U-Net skip connections.
       Point PHASE1_CKPT at the checkpoint written by khushi/pretrain_mae.py
       (either mae_backbone.pt — backbone_state_dict key — or the .ckpt resumable file).
    2. Teacher          — TeacherWrapper calls the AttentionGAN teacher via its real API:
       set_input() then forward(), then reads .fake_B / .att_A / hook-captured features.
       Input I_c is 3-band optical (G/R/NIR, matching LISS-4); I_s is 3-band SAR (VV/VH/VV-VH).
    3. Dataset          — batches must supply "cloudy_optical" (B,3,H,W) and "sar" (B,3,H,W).
       See khushi/sen12mscr_dataset.py for the 3-band LISS-4-equivalent loader.
=================================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =================================================================================================
# 0. CONFIG  (the greek-letter weights from the slide's loss equation)
# =================================================================================================
class CFG:
    img_size   = 256
    patch_size = 16
    in_chans   = 3                      # G, R, NIR  (LISS-4 native bands — no SWIR)
    embed_dim  = 768
    depth      = 12
    grid       = img_size // patch_size  # 16 -> 256 patches

    # which Phase-1 transformer blocks to tap for U-Net skip connections (shallow -> deep)
    tap_layers = (2, 5, 8, 11)

    # --- standard GAN group ---
    lambda_l1   = 10.0
    lambda_ssim = 5.0
    delta_cw    = 1.0

    # --- teacher distillation group ---
    alpha_out  = 5.0
    beta_feat  = 1.0
    gamma_attn = 2.0

    # --- SAR privileged-information group ---
    mu_sar_mimic = 1.0
    nu_cw        = 1.0
    eta_freq     = 0.5


# =================================================================================================
# 1. PHASE-1 BACKBONE WRAPPER  (already trained — this just loads & optionally freezes it)
# =================================================================================================
import sys, os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", "khushi"))
from vit_pretrain import ViTBackbone  # noqa: E402  — Phase-1 encoder class


class TappedViTBackbone(nn.Module):
    """Runs ViTBackbone block-by-block, capturing intermediate token tensors at
    tap_layers for use as U-Net skip connections (UNETR-style).

    Returns:
        tokens : (B, N, D)                 final normalized patch tokens (CLS dropped)
        skips  : dict{layer_idx: (B,N,D)}  raw (pre-norm) tokens at each tap layer
    """

    def __init__(self, backbone: ViTBackbone, tap_layers=(2, 5, 8, 11)):
        super().__init__()
        self.backbone = backbone
        self.tap_layers = set(tap_layers)

    def forward(self, x):
        tokens = self.backbone.embed_patches(x)          # (B, N, D)

        B = tokens.shape[0]
        cls = self.backbone.cls_token + self.backbone.pos_embed[:, :1, :]
        cls = cls.expand(B, -1, -1)
        h = torch.cat([cls, tokens], dim=1)              # (B, 1+N, D)

        skips = {}
        for i, block in enumerate(self.backbone.blocks):
            h = block(h)
            if i in self.tap_layers:
                skips[i] = h[:, 1:, :]                  # drop CLS; raw, pre-norm

        final = self.backbone.norm(h)[:, 1:, :]          # (B, N, D)
        return final, skips


def load_phase1_backbone(checkpoint_path, embed_dim, depth, in_chans=3,
                         img_size=256, patch_size=16, device="cpu",
                         tap_layers=CFG.tap_layers):
    """Load the Phase-1 ViT encoder from a khushi/pretrain_mae.py checkpoint.

    Checkpoint layouts understood:
        - Full resumable .ckpt  : state["backbone_state_dict"]
        - Final export .pt      : state["backbone_state_dict"]
        - Raw state dict        : loaded directly

    Returns a TappedViTBackbone whose forward(x) yields (tokens, skips).
    """
    raw = ViTBackbone(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=in_chans,
        embed_dim=embed_dim,
        depth=depth,
    )
    state = torch.load(checkpoint_path, map_location=device)
    sd = state.get("backbone_state_dict", state)
    raw.load_state_dict(sd, strict=True)
    return TappedViTBackbone(raw, tap_layers=tap_layers)


class Phase1ViTEncoder(nn.Module):
    """Thin wrapper so the rest of this file doesn't care about checkpoint internals."""

    def __init__(self, checkpoint_path, embed_dim=768, depth=12, freeze=True,
                 tap_layers=CFG.tap_layers, in_chans=CFG.in_chans):
        super().__init__()
        self.backbone = load_phase1_backbone(
            checkpoint_path, embed_dim, depth,
            in_chans=in_chans, tap_layers=tap_layers,
        )
        self.freeze = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        self.embed_dim = embed_dim
        self.tap_layers = tap_layers

    def forward(self, x):
        if self.freeze:
            with torch.no_grad():
                tokens, skips = self.backbone(x)
        else:
            tokens, skips = self.backbone(x)
        return tokens, skips


def tokens_to_map(tokens, grid):
    """(B, N, D) patch tokens -> (B, D, grid, grid) spatial feature map."""
    B, N, D = tokens.shape
    return tokens.transpose(1, 2).reshape(B, D, grid, grid)


# =================================================================================================
# 2. STUDENT GENERATOR  =  ViT encoder + Attention Head + U-Net Decoder + Blend
# =================================================================================================
class AttentionHead(nn.Module):
    """Produces the soft cloud-probability map I_M in [0,1] (student's counterpart to the
    teacher's attention module on the architecture slide)."""

    def __init__(self, embed_dim, grid, out_size):
        super().__init__()
        self.grid, self.out_size = grid, out_size
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, tokens):
        fmap = tokens_to_map(tokens, self.grid)
        mask = self.conv(fmap)
        mask = F.interpolate(mask, size=self.out_size, mode="bilinear", align_corners=False)
        return torch.sigmoid(mask)  # I_M


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1), nn.InstanceNorm2d(out_ch), nn.LeakyReLU(0.2),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.InstanceNorm2d(out_ch), nn.LeakyReLU(0.2),
        )

    def forward(self, x, skip):
        x = self.up(x)                                   # spatial size doubles
        if x.shape[-2:] != skip.shape[-2:]:
            # All ViT skips are at the same grid (16×16). Upsample the skip TO x,
            # not x down to the skip, so the decoder can grow toward full resolution.
            skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNetDecoder(nn.Module):
    """Pixel reconstruction head G_S(I_c). Consumes the final ViT tokens plus three shallower
    tapped layers as skip connections (UNETR-style)."""

    def __init__(self, embed_dim, grid, out_chans, tap_layers):
        super().__init__()
        self.grid = grid
        self.tap_layers = sorted(tap_layers)          # e.g. [2,5,8,11], shallow -> deep
        self.reduce = nn.Conv2d(embed_dim, 512, 1)
        self.skip_reduce = nn.ModuleDict({
            str(l): nn.Conv2d(embed_dim, 512, 1) for l in self.tap_layers[:-1]
        })
        self.up1 = UpBlock(512, 512, 256)
        self.up2 = UpBlock(256, 512, 128)
        self.up3 = UpBlock(128, 512, 64)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.InstanceNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, out_chans, 3, padding=1), nn.Tanh(),
        )

    def forward(self, tokens_final, skips):
        x = tokens_to_map(tokens_final, self.grid)
        x = self.reduce(x)
        deepest_first = self.tap_layers[:-1][::-1]
        skip_maps = [self.skip_reduce[str(l)](tokens_to_map(skips[l], self.grid)) for l in deepest_first]
        x = self.up1(x, skip_maps[0])
        x = self.up2(x, skip_maps[1])
        x = self.up3(x, skip_maps[2])
        return self.final(x)  # G_S(I_c), range [-1, 1]


class StudentGenerator(nn.Module):
    """I_g = I_M ⊙ G_S(I_c) + (1 - I_M) ⊙ I_c  — the "Blend" box on your architecture slide.
    Clear pixels pass through untouched; only the region the model believes is cloudy gets
    replaced by the generated content."""

    def __init__(self, encoder: Phase1ViTEncoder, cfg=CFG):
        super().__init__()
        self.encoder = encoder
        self.attn_head = AttentionHead(cfg.embed_dim, cfg.grid, cfg.img_size)
        self.decoder = UNetDecoder(cfg.embed_dim, cfg.grid, cfg.in_chans, encoder.tap_layers)

    def forward(self, I_c):
        tokens, skips = self.encoder(I_c)
        I_M = self.attn_head(tokens)
        G_S = self.decoder(tokens, skips)
        I_g = I_M * G_S + (1 - I_M) * I_c
        return {"I_g": I_g, "I_M": I_M, "G_S": G_S, "tokens": tokens, "skips": skips}


# =================================================================================================
# 3. STUDENT DISCRIMINATOR  =  forked ViT (independent copy, unfrozen) + PatchGAN head
# =================================================================================================
class PatchGANHead(nn.Module):
    def __init__(self, embed_dim, grid):
        super().__init__()
        self.grid = grid
        self.net = nn.Sequential(
            nn.Conv2d(embed_dim, 256, 3, padding=1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 128, 3, padding=1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 3, padding=1),
        )

    def forward(self, tokens):
        return self.net(tokens_to_map(tokens, self.grid))  # (B,1,grid,grid) real/fake per patch


class StudentDiscriminator(nn.Module):
    """A *forked* copy of the Phase-1 backbone: same architecture, initialised from the same
    checkpoint, but NOT frozen and NOT shared with the generator's encoder — so it's free to
    specialise for the real/fake task."""

    def __init__(self, checkpoint_path, cfg=CFG):
        super().__init__()
        self.backbone_encoder = Phase1ViTEncoder(checkpoint_path, cfg.embed_dim, cfg.depth, freeze=False)
        self.head = PatchGANHead(cfg.embed_dim, cfg.grid)

    def forward(self, x):
        tokens, _ = self.backbone_encoder(x)
        return self.head(tokens)


# =================================================================================================
# 4. TEACHER WRAPPER  (frozen — supplies the three distillation signals)
# =================================================================================================
class TeacherWrapper(nn.Module):
    """Wraps the trained Pix2Pix_attn teacher (AttentionGAN-for-Cloud-removal).

    The teacher uses the CycleGAN-style framework API: set_input() then forward(),
    after which results live as attributes.

        I_t       : teacher.fake_B   — final cloud-free output        (B,3,H,W) in [-1,1]
        A_c       : teacher.att_A    — SAR-aware cloud attention map   (B,1,H,W) in [0,1]
        sar_feats : teacher.fake_C   — G2's SAR→optical output         (B,3,H,W) in [-1,1]
                    Used as the LUPI privileged signal: it encodes the SAR information in
                    an optical feature space that the student must learn to mimic from
                    optical-only input.
        hints     : empty dict — UNet's recursive structure makes intermediate tapping
                    complex; L_feat term is zero. Wire up hooks here if needed later.

    I_c : (B,3,H,W)  cloudy optical in [-1,1]  (student's training input)
    I_s : (B,3,H,W)  SAR in [-1,1]             (teacher-only privileged signal at train time)
    """

    def __init__(self, teacher_model):
        super().__init__()
        self.teacher = teacher_model
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, I_c, I_s):
        # Teacher API: set_input populates the teacher's pre-allocated internal tensors,
        # then forward() runs netA + netG2 + netG and writes results as attributes.
        # real_B (cloud-free GT) is not needed at distillation time — zero placeholder.
        batch = {
            "A": I_c,
            "B": torch.zeros_like(I_c),
            "C": I_s,
            "A_paths": [""] * I_c.shape[0],
            "C_paths": [""] * I_s.shape[0],
        }
        self.teacher.set_input(batch)
        self.teacher.forward()

        I_t       = self.teacher.fake_B.detach()   # (B, 3, H, W) cloud-free reconstruction
        A_c       = self.teacher.att_A.detach()    # (B, 1, H, W) attention / cloud mask
        sar_feats = self.teacher.fake_C.detach()   # (B, 3, H, W) SAR→optical LUPI signal

        return I_t, {}, A_c, sar_feats


# =================================================================================================
# 5. LOSSES
# =================================================================================================

# ---------- standard GAN group ----------
def hinge_d_loss(real_logits, fake_logits):
    return F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()


def hinge_g_loss(fake_logits):
    return -fake_logits.mean()


def _gaussian_window(window_size=11, sigma=1.5, channels=1, device="cpu"):
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    window_2d = g.t() @ g
    return window_2d.expand(channels, 1, window_size, window_size).contiguous().to(device)


def ssim(pred, target, window_size=11, C1=0.01 ** 2, C2=0.03 ** 2):
    B, C, H, W = pred.shape
    window = _gaussian_window(window_size, channels=C, device=pred.device)
    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=C)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=C) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def ssim_loss(pred, target):
    return 1 - ssim(pred, target)


def channel_weighted_l1(pred, target, weights):
    """weights: (C,) tensor — upweight NIR (index 2) since it carries haze-penetration info."""
    diff = (pred - target).abs() * weights.view(1, -1, 1, 1)
    return diff.mean()


# ---------- teacher distillation group ----------
def output_matching_loss(I_g_student, I_t_teacher):
    return F.l1_loss(I_g_student, I_t_teacher.detach())


def feature_hint_loss(student_feats: dict, teacher_feats: dict, projections: nn.ModuleDict):
    """student_feats / teacher_feats: dict of same keys -> (B,D,h,w) or (B,N,D) tensors.
    `projections` 1x1-conv/linear-projects student dims to teacher dims where they differ."""
    loss, n = 0.0, 0
    for k, t_feat in teacher_feats.items():
        if k not in student_feats:
            continue
        s_feat = student_feats[k]
        if k in projections:
            s_feat = projections[k](s_feat)
        loss = loss + F.mse_loss(s_feat, t_feat.detach())
        n += 1
    return loss / max(n, 1)


def attention_matching_loss(I_M_student, A_c_teacher):
    A_c_teacher = F.interpolate(A_c_teacher, size=I_M_student.shape[-2:], mode="bilinear", align_corners=False)
    return F.mse_loss(I_M_student, A_c_teacher.detach())


# ---------- SAR privileged-information group (LUPI) ----------
def sar_mimicry_loss(student_tokens, teacher_sar_feats, proj: nn.Module, grid: int):
    """Forces the optical-only student to reproduce G2's SAR→optical feature map.

    student_tokens  : (B, N, D)   — final encoder tokens
    teacher_sar_feats: (B, C, H, W) — teacher.fake_C, G2's SAR translation
    proj            : Conv2d(D, C, 1) projecting student spatial map to teacher channels
    grid            : int — sqrt(N), the spatial grid side for reshaping tokens
    """
    if teacher_sar_feats is None:
        return torch.tensor(0.0, device=student_tokens.device)
    s_map  = tokens_to_map(student_tokens, grid)                           # (B, D, grid, grid)
    s_proj = proj(s_map)                                                   # (B, C, grid, grid)
    t_down = F.adaptive_avg_pool2d(teacher_sar_feats, s_proj.shape[-2:])  # (B, C, grid, grid)
    return F.mse_loss(s_proj, t_down.detach())


def channel_covariance_loss(pred, target):
    """Matches inter-band correlation structure (Gram matrix over channels) — a second,
    complementary reading of the slide's 'L_cw' term inside the SAR-privileged group,
    distinct from the plain per-band L1 used in the standard-GAN group."""
    def gram(x):
        B, C, H, W = x.shape
        xf = x.view(B, C, -1)
        xf = xf - xf.mean(dim=2, keepdim=True)
        return torch.einsum("bcn,bdn->bcd", xf, xf) / (H * W)
    return F.mse_loss(gram(pred), gram(target.detach()))


def frequency_decoupled_loss(pred, target, cutoff_ratio=0.35):
    """FFT-domain loss restricted to low/mid frequencies, so the student is not pushed to
    reproduce high-frequency SAR speckle texture baked into the teacher's SAR-informed
    target — exactly the 'skip radar speckle' behaviour noted on your slide."""
    B, C, H, W = pred.shape
    pf = torch.fft.rfft2(pred, norm="ortho")
    tf = torch.fft.rfft2(target.detach(), norm="ortho")
    freq_h = torch.fft.fftfreq(H).to(pred.device)
    freq_w = torch.fft.rfftfreq(W).to(pred.device)
    radius = torch.sqrt(freq_h.view(-1, 1) ** 2 + freq_w.view(1, -1) ** 2)
    mask = (radius <= cutoff_ratio).float()
    return ((pf - tf).abs() * mask).mean()


# =================================================================================================
# 6. COMBINED LOSS  (assembles exactly the equation on your "Detailed Overview" slide)
# =================================================================================================
def student_generator_loss(student_out, teacher_out, disc_fake_logits, projections, cfg=CFG):
    I_g, I_M = student_out["I_g"], student_out["I_M"]
    I_t, teacher_hints, A_c, teacher_sar_tokens = teacher_out

    # --- standard GAN ---
    L_adv = hinge_g_loss(disc_fake_logits)
    L_1 = F.l1_loss(I_g, I_t)
    L_ssim = ssim_loss(I_g, I_t)
    band_weights = torch.tensor([1.0, 1.0, 1.2], device=I_g.device)        # G, R, NIR
    L_cw = channel_weighted_l1(I_g, I_t, band_weights)
    L_standard = L_adv + cfg.lambda_l1 * L_1 + cfg.lambda_ssim * L_ssim + cfg.delta_cw * L_cw

    # --- teacher distillation ---
    L_out = output_matching_loss(I_g, I_t)
    L_feat = feature_hint_loss(student_out["skips"], teacher_hints, projections["feat"])
    L_attn = attention_matching_loss(I_M, A_c)
    L_distill = cfg.alpha_out * L_out + cfg.beta_feat * L_feat + cfg.gamma_attn * L_attn

    # --- SAR privileged information (LUPI) ---
    L_sar_mimic = sar_mimicry_loss(student_out["tokens"], teacher_sar_tokens, projections["sar"], cfg.grid)
    L_cw2 = channel_covariance_loss(I_g, I_t)
    L_freq = frequency_decoupled_loss(I_g, I_t)
    L_privileged = cfg.mu_sar_mimic * L_sar_mimic + cfg.nu_cw * L_cw2 + cfg.eta_freq * L_freq

    L_total = L_standard + L_distill + L_privileged
    logs = {k: (v.item() if torch.is_tensor(v) else v) for k, v in dict(
        L_adv=L_adv, L_1=L_1, L_ssim=L_ssim, L_cw=L_cw,
        L_out=L_out, L_feat=L_feat, L_attn=L_attn,
        L_sar_mimic=L_sar_mimic, L_cw2=L_cw2, L_freq=L_freq,
        L_total=L_total,
    ).items()}
    return L_total, logs


def build_projections(cfg=CFG, teacher_sar_chans: int = 3):
    """Adapters bridging student (ViT) and teacher (UNet) feature spaces.

    "feat" : per-tap-layer projections for feature hint loss.  Currently nn.Identity()
             because the UNet's recursive structure makes layer-level tapping complex and
             the loss will be zero (no key overlap).  Replace each with a suitable
             Conv2d/Linear once you wire up teacher hint hooks.
    "sar"  : Conv2d(embed_dim, teacher_sar_chans, 1) projects the student's spatial token
             map to match teacher.fake_C's channel count (3 for a 3-band UNet output).
    """
    return nn.ModuleDict({
        "feat": nn.ModuleDict({str(l): nn.Identity() for l in cfg.tap_layers}),
        "sar":  nn.Conv2d(cfg.embed_dim, teacher_sar_chans, 1),
    })


# =================================================================================================
# 7. TRAINING LOOP
# =================================================================================================
def train_student_gan(student_G, student_D, teacher, dataloader, projections, cfg=CFG,
                       epochs=50, device="cuda", lr=2e-4,
                       save_every=5, out_path="student_gan.pt", val_loader=None):
    """Train the student GAN with knowledge distillation from the frozen teacher.

    Saves a checkpoint every `save_every` epochs to <out_path>.ckptN and writes the
    final weights to `out_path`.
    """
    student_G.to(device)
    student_D.to(device)
    teacher.to(device)
    projections.to(device)

    opt_G = torch.optim.AdamW(
        [p for p in student_G.parameters() if p.requires_grad] + list(projections.parameters()),
        lr=lr, betas=(0.5, 0.999),
    )
    opt_D = torch.optim.AdamW(student_D.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        student_G.train()
        student_D.train()
        d_loss = g_loss = None

        for batch in dataloader:
            # cloudy_optical: (B,3,H,W) G/R/NIR in [-1,1] — ONLY student input at inference
            # sar:            (B,3,H,W) VV/VH/VV-VH in [-1,1] — teacher-only training signal
            I_c = batch["cloudy_optical"].to(device)
            I_s = batch["sar"].to(device)

            student_out = student_G(I_c)
            teacher_out = teacher(I_c, I_s)  # (I_t, hints, A_c, sar_feats), already no_grad

            # ---- discriminator step ----
            opt_D.zero_grad()
            real_logits = student_D(teacher_out[0].detach())
            fake_logits = student_D(student_out["I_g"].detach())
            d_loss = hinge_d_loss(real_logits, fake_logits)
            d_loss.backward()
            opt_D.step()

            # ---- generator step ----
            opt_G.zero_grad()
            fake_logits = student_D(student_out["I_g"])
            g_loss, logs = student_generator_loss(student_out, teacher_out, fake_logits, projections, cfg)
            g_loss.backward()
            opt_G.step()

        d_val = d_loss.item() if d_loss is not None else float("nan")
        g_val = g_loss.item() if g_loss is not None else float("nan")
        print(f"[epoch {epoch:03d}] D={d_val:.4f} G={g_val:.4f} | {logs}")

        if save_every and (epoch + 1) % save_every == 0:
            ckpt = {
                "epoch": epoch,
                "student_G": student_G.state_dict(),
                "student_D": student_D.state_dict(),
                "projections": projections.state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_D": opt_D.state_dict(),
            }
            ckpt_path = f"{out_path}.ckpt{epoch + 1}"
            torch.save(ckpt, ckpt_path)
            print(f"[epoch {epoch:03d}] checkpoint → {ckpt_path}")

    # Final export: generator weights only (what Phase-3 / inference needs)
    torch.save({
        "student_G": student_G.state_dict(),
        "student_D": student_D.state_dict(),
        "projections": projections.state_dict(),
        "cfg": {k: v for k, v in vars(cfg).items() if not k.startswith("_")},
    }, out_path)
    print(f"[train_student_gan] saved final weights → {out_path}")


# =================================================================================================
# 8. WIRING EXAMPLE
# =================================================================================================
if __name__ == "__main__":
    # Point at the checkpoint written by khushi/pretrain_mae.py
    # (mae_backbone.pt carries backbone_state_dict; .ckpt checkpoints also work).
    PHASE1_CKPT = "../khushi/mae_backbone.pt"

    encoder = Phase1ViTEncoder(PHASE1_CKPT, embed_dim=CFG.embed_dim, depth=CFG.depth,
                                freeze=True, in_chans=CFG.in_chans)
    student_G = StudentGenerator(encoder, CFG)
    student_D = StudentDiscriminator(PHASE1_CKPT, CFG)

    # Load the trained teacher from AttentionGAN-for-Cloud-removal checkpoints, e.g.:
    #   import sys; sys.path.insert(0, "../AttentionGAN-for-Cloud-removal")
    #   from options.test_options import TestOptions
    #   from models.models import create_model
    #   opt = TestOptions().parse(["--dataroot", "...", "--name", "sen12mscr_teacher_v1",
    #                              "--model", "pix2pix_attn", "--which_epoch", "latest"])
    #   teacher_model = create_model(opt)
    #   teacher = TeacherWrapper(teacher_model)

    projections = build_projections(CFG)

    # Dataset: use SEN12MSCROptical (3-band G/R/NIR) for optical and the paired SAR channel.
    # Your DataLoader batches must provide:
    #   batch["cloudy_optical"]  (B, 3, H, W)   — cloudy G/R/NIR tiles
    #   batch["sar"]             (B, 3, H, W)   — paired SAR VV/VH/VV-VH tiles
    # dataloader = build_your_dataloader(...)
    # train_student_gan(student_G, student_D, teacher, dataloader, projections, epochs=50)