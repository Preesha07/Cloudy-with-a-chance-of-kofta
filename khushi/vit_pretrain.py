"""ViT pretraining backbone (Phase-1) for the LISS-IV cloud-removal project.

Self-supervised Masked Autoencoder (MAE) pretraining over multi-band
satellite tiles. The encoder learns an "inherent understanding of satellite
data" by reconstructing randomly masked patches; after pretraining, the
encoder (``MAEPretrainer.export_backbone``) is transplanted into the Phase-2
student generator for SAR-free cloud removal.

Input tiles are (B, C, H, W) where C is the number of spectral bands:
    - LISS-IV optical      -> 3 bands (R / G / NIR)
    - + LISS-III SWIR aux  -> 4 bands (contingency: SWIR as auxiliary channel)

Reference: He et al., "Masked Autoencoders Are Scalable Vision Learners"
(CVPR 2022), adapted for arbitrary channel counts and satellite reflectance.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Positional embedding
# --------------------------------------------------------------------------- #
def build_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """Fixed 2D sine-cosine positional embedding.

    Returns:
        (grid_size * grid_size, embed_dim) tensor. No CLS slot; the CLS
        token is given a separate learnable/zero embedding by the caller.
    """
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D sincos"

    coords = torch.arange(grid_size, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")

    # Half the channels encode the x axis, half encode the y axis.
    dim_quarter = embed_dim // 4
    omega = torch.arange(dim_quarter, dtype=torch.float32) / dim_quarter
    omega = 1.0 / (10000.0**omega)  # (embed_dim/4,)

    def _axis_embed(pos: torch.Tensor) -> torch.Tensor:
        out = pos.flatten()[:, None] * omega[None, :]  # (N, embed_dim/4)
        return torch.cat([out.sin(), out.cos()], dim=1)  # (N, embed_dim/2)

    emb = torch.cat([_axis_embed(grid_x), _axis_embed(grid_y)], dim=1)
    return emb  # (grid_size**2, embed_dim)


# --------------------------------------------------------------------------- #
# Patch embedding
# --------------------------------------------------------------------------- #
class PatchEmbed(nn.Module):
    """Split an image into non-overlapping patches and linearly project them.

    Input shape:
        x: (B, C, H, W)
    Output shape:
        tokens: (B, num_patches, embed_dim)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size**2
        # A strided conv is an efficient, equivalent form of "flatten patch
        # then Linear": kernel == stride == patch_size.
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        assert h == self.img_size and w == self.img_size, (
            f"input {h}x{w} does not match configured img_size {self.img_size}"
        )
        x = self.proj(x)  # (B, embed_dim, grid, grid)
        return x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)


# --------------------------------------------------------------------------- #
# Transformer block
# --------------------------------------------------------------------------- #
class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block (MHSA + MLP)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


# --------------------------------------------------------------------------- #
# Encoder backbone (the artefact Phase-2 actually reuses)
# --------------------------------------------------------------------------- #
class ViTBackbone(nn.Module):
    """Vision Transformer encoder for satellite tiles.

    Used two ways:
        - Pretraining: fed the *visible* subset of tokens by the MAE wrapper.
        - Downstream:  ``forward`` runs the full dense token grid and returns
          per-patch features for the Phase-2 cloud-removal decoder.

    Output shape (``forward``):
        features: (B, num_patches, embed_dim)   # CLS token dropped
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Fixed sincos pos-embed (+1 zero slot for CLS). Registered as a buffer
        # so it moves with .to(device) but is never optimized.
        pos_embed = build_2d_sincos_pos_embed(embed_dim, self.patch_embed.grid_size)
        pos_embed = torch.cat([torch.zeros(1, embed_dim), pos_embed], dim=0)
        self.register_buffer("pos_embed", pos_embed.unsqueeze(0), persistent=False)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.embed_dim = embed_dim
        self.num_patches = num_patches
        nn.init.normal_(self.cls_token, std=0.02)

    def embed_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Patchify + add positional embedding (excludes CLS)."""
        tokens = self.patch_embed(x)
        return tokens + self.pos_embed[:, 1:, :]

    def forward_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Run the transformer stack over a (possibly masked) token sequence.

        ``tokens`` must already carry positional information. A CLS token
        (with its own pos-embed) is prepended here.
        """
        b = tokens.shape[0]
        cls = self.cls_token + self.pos_embed[:, :1, :]
        cls = cls.expand(b, -1, -1)
        x = torch.cat([cls, tokens], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dense encode a full image; returns per-patch features (no CLS)."""
        tokens = self.embed_patches(x)
        x = self.forward_tokens(tokens)
        return x[:, 1:, :]


# --------------------------------------------------------------------------- #
# Lightweight MAE decoder (discarded after pretraining)
# --------------------------------------------------------------------------- #
class MAEDecoder(nn.Module):
    """Shallow, narrow decoder that reconstructs pixel patches."""

    def __init__(
        self,
        num_patches: int,
        patch_dim: int,
        encoder_embed_dim: int = 768,
        decoder_embed_dim: int = 512,
        depth: int = 8,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        grid = int(round(num_patches**0.5))
        pos_embed = build_2d_sincos_pos_embed(decoder_embed_dim, grid)
        pos_embed = torch.cat([torch.zeros(1, decoder_embed_dim), pos_embed], dim=0)
        self.register_buffer(
            "pos_embed", pos_embed.unsqueeze(0), persistent=False
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(decoder_embed_dim, num_heads, mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(decoder_embed_dim)
        # Predict raw pixel values for every channel in the patch.
        self.pred = nn.Linear(decoder_embed_dim, patch_dim)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(
        self, latent: torch.Tensor, ids_restore: torch.Tensor
    ) -> torch.Tensor:
        """Re-insert mask tokens, decode, and predict pixels.

        Args:
            latent: (B, 1 + num_visible, encoder_embed_dim) — includes CLS.
            ids_restore: (B, num_patches) — inverse of the shuffle used to mask.
        Returns:
            pred: (B, num_patches, patch_dim)
        """
        x = self.decoder_embed(latent)  # (B, 1 + num_visible, D)

        b, _, d = x.shape
        num_patches = ids_restore.shape[1]
        num_visible = x.shape[1] - 1  # exclude CLS

        # Append shared mask tokens, then unshuffle back to canonical order.
        mask_tokens = self.mask_token.expand(b, num_patches - num_visible, -1)
        no_cls = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # (B, num_patches, D)
        no_cls = torch.gather(
            no_cls, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, d)
        )
        x = torch.cat([x[:, :1, :], no_cls], dim=1)  # re-attach CLS

        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.pred(x[:, 1:, :])  # drop CLS -> (B, num_patches, patch_dim)


# --------------------------------------------------------------------------- #
# MAE pretraining wrapper
# --------------------------------------------------------------------------- #
@dataclass
class MAEConfig:
    """Hyperparameters for Phase-1 MAE pretraining."""

    img_size: int = 224
    patch_size: int = 16
    in_channels: int = 3  # 3 = R/G/NIR ; 4 adds the LISS-III SWIR aux band
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    decoder_embed_dim: int = 512
    decoder_depth: int = 8
    decoder_num_heads: int = 16
    mlp_ratio: float = 4.0
    mask_ratio: float = 0.75
    norm_pix_loss: bool = True  # per-patch normalized target — sharper features


class MAEPretrainer(nn.Module):
    """End-to-end MAE: encoder + decoder + masking + reconstruction loss.

    Typical loop:
        model = MAEPretrainer(MAEConfig(in_channels=4))
        loss, _, _ = model(batch)          # batch: (B, C, H, W)
        loss.backward()
        ...
        backbone = model.export_backbone() # hand off to Phase-2
    """

    def __init__(self, config: MAEConfig | None = None) -> None:
        super().__init__()
        self.config = config or MAEConfig()
        c = self.config

        self.encoder = ViTBackbone(
            img_size=c.img_size,
            patch_size=c.patch_size,
            in_channels=c.in_channels,
            embed_dim=c.embed_dim,
            depth=c.depth,
            num_heads=c.num_heads,
            mlp_ratio=c.mlp_ratio,
        )
        patch_dim = c.patch_size * c.patch_size * c.in_channels
        self.decoder = MAEDecoder(
            num_patches=self.encoder.num_patches,
            patch_dim=patch_dim,
            encoder_embed_dim=c.embed_dim,
            decoder_embed_dim=c.decoder_embed_dim,
            depth=c.decoder_depth,
            num_heads=c.decoder_num_heads,
            mlp_ratio=c.mlp_ratio,
        )

    # -- patch <-> image conversion ---------------------------------------- #
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, num_patches, patch_size**2 * C)."""
        p = self.config.patch_size
        c = self.config.in_channels
        b, _, h, w = imgs.shape
        gh, gw = h // p, w // p
        x = imgs.reshape(b, c, gh, p, gw, p)
        x = torch.einsum("bchpwq->bhwpqc", x)
        return x.reshape(b, gh * gw, p * p * c)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """(B, num_patches, patch_size**2 * C) -> (B, C, H, W)."""
        p = self.config.patch_size
        c = self.config.in_channels
        b, n, _ = patches.shape
        g = int(round(n**0.5))
        x = patches.reshape(b, g, g, p, p, c)
        x = torch.einsum("bhwpqc->bchpwq", x)
        return x.reshape(b, c, g * p, g * p)

    # -- random masking ---------------------------------------------------- #
    @staticmethod
    def random_masking(
        tokens: torch.Tensor, mask_ratio: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-sample random masking by shuffling.

        Returns:
            kept:        (B, num_keep, D) visible tokens
            mask:        (B, num_patches) 1 = masked (a reconstruction target)
            ids_restore: (B, num_patches) inverse permutation for the decoder
        """
        b, n, d = tokens.shape
        num_keep = max(1, int(round(n * (1.0 - mask_ratio))))

        noise = torch.rand(b, n, device=tokens.device)
        ids_shuffle = torch.argsort(noise, dim=1)  # ascending noise
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :num_keep]
        kept = torch.gather(
            tokens, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, d)
        )

        mask = torch.ones(b, n, device=tokens.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return kept, mask, ids_restore

    # -- loss -------------------------------------------------------------- #
    def reconstruction_loss(
        self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean squared error over masked patches only."""
        target = self.patchify(imgs)
        if self.config.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6) ** 0.5

        loss = (pred - target).pow(2).mean(dim=-1)  # (B, num_patches)
        # Average over masked patches only (mask sums to the masked count).
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)

    # -- forward ----------------------------------------------------------- #
    def forward(
        self, imgs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one MAE step.

        Returns:
            loss: scalar reconstruction loss
            pred: (B, num_patches, patch_dim) predicted pixels
            mask: (B, num_patches) which patches were reconstruction targets
        """
        tokens = self.encoder.embed_patches(imgs)  # pos-embedded, no CLS
        kept, mask, ids_restore = self.random_masking(
            tokens, self.config.mask_ratio
        )
        latent = self.encoder.forward_tokens(kept)  # includes CLS
        pred = self.decoder(latent, ids_restore)
        loss = self.reconstruction_loss(imgs, pred, mask)
        return loss, pred, mask

    # -- handoff to Phase-2 ------------------------------------------------ #
    def export_backbone(self) -> ViTBackbone:
        """Return the trained encoder; the decoder is discarded."""
        return self.encoder


# --------------------------------------------------------------------------- #
# Preset factory functions
# --------------------------------------------------------------------------- #
def mae_vit_base(in_channels: int = 3, img_size: int = 224) -> MAEPretrainer:
    """ViT-Base/16 MAE (86M-param encoder)."""
    return MAEPretrainer(
        MAEConfig(img_size=img_size, in_channels=in_channels)
    )


def mae_vit_large(in_channels: int = 3, img_size: int = 224) -> MAEPretrainer:
    """ViT-Large/16 MAE."""
    return MAEPretrainer(
        MAEConfig(
            img_size=img_size,
            in_channels=in_channels,
            embed_dim=1024,
            depth=24,
            num_heads=16,
        )
    )


if __name__ == "__main__":
    # Smoke test: 4-band (R/G/NIR + SWIR) LISS-IV-style tiles.
    torch.manual_seed(0)
    model = mae_vit_base(in_channels=4, img_size=224)
    x = torch.randn(2, 4, 224, 224)
    loss, pred, mask = model(x)
    print(f"loss={loss.item():.4f}  pred={tuple(pred.shape)}  masked/patch={mask.mean():.2f}")

    backbone = model.export_backbone()
    feats = backbone(x)
    n_params = sum(p.numel() for p in backbone.parameters()) / 1e6
    print(f"backbone features={tuple(feats.shape)}  encoder params={n_params:.1f}M")