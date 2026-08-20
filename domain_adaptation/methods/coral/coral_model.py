"""CORAL domain adaptation student model.

Extends Pix2Pix_attn_Student_FSP_FFL_Model with plain Deep CORAL
(Sun & Saenko, ECCV 2016) for Unsupervised Domain Adaptation from
Sentinel-2 (source) to LISS-4 (target).

Key design decisions vs. the previous masked implementation
-----------------------------------------------------------
1. No attention-map downsampling or spatial masking inside CORAL.
   Cloud-contaminated patches are discarded *before* training via
   patch_filter.filter_liss4_patches(), so every LISS-4 batch item
   that reaches this model has already passed the cloud-fraction gate.

2. Features aligned: output-half of netG encoder at CORAL_BLOCK_IDX
   (block 3 = 32×32 spatial, 256 channels for unet_256).  netH output
   was used before — that is the SAR-hallucination output space, not
   the encoder representation.  Aligning netG encoder activations pushes
   the shared encoder to produce domain-invariant representations, which
   is the standard CORAL domain adaptation goal.

3. Gradient flows through both source (SEN12MS-CR) and target (LISS-4)
   netG paths.  The source path also receives gradients from all existing
   distillation losses (FSP, hall, DIST, FFL), which together prevent
   catastrophic forgetting of the pre-trained reconstruction quality.
"""
from __future__ import annotations

from collections import OrderedDict

import torch

from pix2pix_attn_student_fsp_ffl_model import Pix2Pix_attn_Student_FSP_FFL_Model
from models.pix2pix_attn_student_fsp_model_changed import _get_unet_blocks
from coral_loss import coral_loss


class CoralStudentModel(Pix2Pix_attn_Student_FSP_FFL_Model):
    """FSP + FFL + plain Deep CORAL student model.

    Usage
    -----
    During training, batches must contain a 'LISS4' key with pre-filtered
    target-domain patches (cloud fraction < threshold).  When 'LISS4' is
    absent (e.g. at test/inference time), CORAL is silently skipped and
    the model behaves identically to Pix2Pix_attn_Student_FSP_FFL_Model.
    """

    #: Weight on the CORAL loss term.  Tune so CORAL_WEIGHT * loss_coral
    #: sits in the same order of magnitude as loss_G and loss_H (~0.2-1.0).
    CORAL_WEIGHT: float = 1.0

    #: netG encoder blocks at which features are captured and aligned.
    #: For unet_256 (256×256 input, ngf=64):
    #:   block 2 → 64×64,  128 output-half channels (4096 positions)
    #:   block 3 → 32×32,  256 output-half channels (1024 positions)
    #:   block 4 → 16×16,  512 output-half channels  (256 positions)
    #: CORAL is computed independently at each block and summed.
    CORAL_BLOCK_INDICES: tuple[int, ...] = (2, 3, 4)

    def name(self) -> str:
        return 'CoralStudentModel'

    # ── Initialization ────────────────────────────────────────────────────────

    def initialize(self, opt) -> None:
        super().initialize(opt)
        self.device = getattr(
            opt, 'device',
            torch.device(f"cuda:{opt.gpu_ids[0]}" if opt.gpu_ids else "cpu"),
        )
        self.CORAL_WEIGHT = getattr(opt, 'CORAL_WEIGHT', self.CORAL_WEIGHT)

        self.real_LISS4: torch.Tensor | None = None
        self._coral_feat_src: dict[int, torch.Tensor] = {}
        self._coral_feat_tgt: dict[int, torch.Tensor] = {}
        self._coral_capture: str = 'src'   # routes hooks to src or tgt dicts
        self.loss_coral = torch.zeros((), device=self.device)

        if self.isTrain:
            self._register_coral_encoder_hooks()

    def _register_coral_encoder_hooks(self) -> None:
        """Attach forward hooks on netG blocks CORAL_BLOCK_INDICES.

        Each hook captures the output-half of its block (dropping the echoed
        skip-connection input, which is near-identical between domains and would
        inflate the measured similarity).  A flag `_coral_capture` routes each
        firing to either `_coral_feat_src[idx]` or `_coral_feat_tgt[idx]`.
        """
        blocks = _get_unet_blocks(self.netG)
        n = len(blocks)
        for idx in self.CORAL_BLOCK_INDICES:
            if idx >= n:
                raise ValueError(
                    f"CORAL_BLOCK_INDICES contains {idx}, but netG only has "
                    f"{n} blocks (0-indexed)"
                )

            def make_hook(block_idx: int):
                def hook_fn(module, inp, out):
                    c_in = inp[0].shape[1]
                    feat = out[:, c_in:] if c_in < out.shape[1] else out
                    if self._coral_capture == 'src':
                        self._coral_feat_src[block_idx] = feat
                    else:
                        self._coral_feat_tgt[block_idx] = feat
                return hook_fn

            blocks[idx].register_forward_hook(make_hook(idx))

        print(
            f"[CORAL] Encoder hooks on netG blocks {self.CORAL_BLOCK_INDICES} "
            f"(out of {n} blocks, 0 = outermost)"
        )

    # ── Input handling ────────────────────────────────────────────────────────

    def set_input(self, batch: dict) -> None:
        super().set_input(batch)
        self.real_LISS4 = (
            batch['LISS4'].to(self.device) if 'LISS4' in batch else None
        )

    # ── Forward passes ────────────────────────────────────────────────────────

    def forward(self) -> None:
        # ── Source pass (SEN12MS-CR) ─────────────────────────────────────────
        # Populates: _teacher_acts, _student_acts, fake_C, fake_B, att_A, g_B.
        # The CORAL hook fires on netG and stores the encoder feature in
        # _coral_feat_src.
        self._coral_capture = 'src'
        super().forward()

        # ── Target pass (LISS-4) ─────────────────────────────────────────────
        if self.real_LISS4 is not None:
            # Save the distillation hook state populated by the source pass.
            # The netH call below fires hallucination hooks and would overwrite
            # _student_acts; restoring them afterwards keeps the FSP / DIST
            # losses consistent with the source-domain computation.
            saved_student_acts = dict(self._student_acts)

            self._coral_capture = 'tgt'

            # fake_c is only used as netG's 2nd-channel input; no distillation
            # loss targets it, so we can safely detach it.
            with torch.no_grad():
                fake_c_liss4 = self.netH(self.real_LISS4)

            # netG forward fires the CORAL hook → stores _coral_feat_tgt.
            # Gradient flows into netG parameters (they always require grad).
            _ = self.netG(torch.cat([self.real_LISS4, fake_c_liss4], dim=1))

            self._student_acts = saved_student_acts  # restore for distillation
            self._coral_capture = 'src'

    # ── Optimization ─────────────────────────────────────────────────────────

    def optimize_parameters(self) -> None:
        self.forward()
        self._compute_losses()

        zero = torch.zeros((), device=self.device)
        if self.real_LISS4 is not None and self._coral_feat_src:
            self.loss_coral = sum(
                coral_loss(self._coral_feat_src[idx], self._coral_feat_tgt[idx])
                for idx in self.CORAL_BLOCK_INDICES
                if idx in self._coral_feat_src and idx in self._coral_feat_tgt
            )
            if not torch.is_tensor(self.loss_coral):
                self.loss_coral = zero
        else:
            self.loss_coral = zero

        self.optimizer_G.zero_grad()
        self.optimizer_H.zero_grad()

        total_loss = (
            self.loss_H
            + self.loss_G
            + self.opt.gamma_hall     * self.loss_hall
            + self.loss_dist
            + self.cross_depth_weight * self.loss_fsp
            + self.freqkd_weight      * self.loss_freqkd
            + self.FFL_WEIGHT         * self.loss_ffl
            + self.CORAL_WEIGHT       * self.loss_coral
        )
        total_loss.backward()

        student_params = (
            list(self.netG.parameters()) + list(self.netH.parameters())
        )
        torch.nn.utils.clip_grad_norm_(student_params, max_norm=10.0)

        self.optimizer_G.step()
        self.optimizer_H.step()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    # ── Logging ───────────────────────────────────────────────────────────────

    def get_current_errors(self) -> OrderedDict:
        errors = super().get_current_errors()
        errors['coral'] = self.loss_coral.item() if torch.is_tensor(self.loss_coral) else 0.0
        # Per-layer breakdown (logged unweighted, same as other distillation terms)
        for idx in self.CORAL_BLOCK_INDICES:
            if idx in self._coral_feat_src and idx in self._coral_feat_tgt:
                errors[f'coral_b{idx}'] = coral_loss(
                    self._coral_feat_src[idx].detach(),
                    self._coral_feat_tgt[idx].detach(),
                ).item()
        return errors
