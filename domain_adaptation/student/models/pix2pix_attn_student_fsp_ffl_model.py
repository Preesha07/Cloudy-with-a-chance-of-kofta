"""
FSP (changed) + Focal Frequency Loss student model.

Combines:
  - pix2pix_attn_student_fsp_model_changed.py  (FSP cross-depth distillation
    with the upsample-before-Pearson fix in dist_intra_loss, all three fixes:
    _out_half in hall, range(3, n_blocks-1) default, frozen netA)
  - pix2pix_attn_student_ffl_model.py  (Focal Frequency Loss on fake_B vs real_B)

The only additions over the changed FSP model are:
  _focal_frequency_loss()  — identical to the FFL model's implementation
  loss_ffl added to total_student_loss in optimize_parameters()
  loss_ffl logged in get_current_errors()

FFL targets blurriness by adaptively up-weighting the spectral components the
model is currently getting most wrong — the same failure mode that L1 loss
(dominated by easy low-frequency pixels) systematically under-penalises.

*** WEIGHT CALIBRATION ***
FFL_WEIGHT = 300.0.  At a typical FFL_WEIGHT=1.0, loss_ffl ≈ 0.001–0.03,
which is negligible next to 100*loss_G_L1 (~0.2–1.0).  FFL_WEIGHT=300 puts
the weighted contribution ~0.3–9, in range with L_hall (gamma*loss_hall ~3)
and L_G_L1.  Verify from the first ~100 iterations: if the weighted term is an
order of magnitude above L_G or L_hall, halve FFL_WEIGHT before continuing.

*** KNOWN FAILURE MODE (VGG, NOT FFL) ***
The VGG perceptual loss variant caused InstanceNorm running_var explosion on
this architecture family.  FFL has no such reported failure here; however, if
a fake_B channel goes dead (std→0) or any InstanceNorm running_var exceeds ~10×
its neighbours, treat that as the same instability and reduce FFL_WEIGHT.

Run from khushi/student/:
    source ../../AttentionGAN-for-Cloud-removal/.venv/bin/activate
    bash scripts/train_sen12mscr_student_fsp_ffl_v1.sh 2>&1 | tee train_fsp_ffl_v1.log
"""

import torch
from collections import OrderedDict

from .pix2pix_attn_student_fsp_model_changed import Pix2Pix_attn_Student_FSP_Model


class Pix2Pix_attn_Student_FSP_FFL_Model(Pix2Pix_attn_Student_FSP_Model):

    # Weight on the focal frequency loss.  Calibrated so the weighted term
    # sits in the same order of magnitude as the L1 and hall losses.
    # See module docstring before adjusting.
    FFL_WEIGHT = 300.0

    # Spectral weight exponent (paper's alpha, Jiang et al. ICCV 2021 Eq. 3-4).
    # alpha=1 is the paper default; higher values concentrate more sharply on
    # worst-matched frequencies at the cost of a noisier early gradient.
    FFL_ALPHA = 1.0

    def name(self):
        return 'Pix2Pix_attn_Student_FSP_FFL_Model'

    # ── Focal Frequency Loss ──────────────────────────────────────────────────

    def _focal_frequency_loss(self, pred, target):
        """Focal Frequency Loss between two (B,C,H,W) images in [-1,1].

        1. 2D FFT per channel (orthonormal norm).
        2. freq_distance = squared magnitude of the complex spectral error.
        3. weight = (freq_distance normalised per-image)^alpha, detached.
        4. loss = mean(weight * freq_distance).
        """
        pred_freq = torch.fft.fft2(pred, norm='ortho')
        targ_freq = torch.fft.fft2(target, norm='ortho')
        diff = pred_freq - targ_freq
        freq_distance = diff.real ** 2 + diff.imag ** 2

        with torch.no_grad():
            w = freq_distance / (freq_distance.amax(dim=(2, 3), keepdim=True) + 1e-8)
            w = w.clamp(0.0, 1.0) ** self.FFL_ALPHA

        return (w * freq_distance).mean()

    # ── Losses ───────────────────────────────────────────────────────────────

    def _compute_losses(self):
        """All FSP losses plus FFL on the output image."""
        super()._compute_losses()
        self.loss_ffl = self._focal_frequency_loss(self.fake_B, self.real_B)

    def optimize_parameters(self):
        self.forward()
        self._compute_losses()

        self.optimizer_G.zero_grad()
        self.optimizer_H.zero_grad()

        total_student_loss = (self.loss_H
                              + self.loss_G
                              + self.opt.gamma_hall     * self.loss_hall
                              + self.loss_dist
                              + self.cross_depth_weight * self.loss_fsp
                              + self.freqkd_weight      * self.loss_freqkd
                              + self.FFL_WEIGHT         * self.loss_ffl)
        total_student_loss.backward()

        student_params = (list(self.netG.parameters())
                          + list(self.netH.parameters()))
        torch.nn.utils.clip_grad_norm_(student_params, max_norm=10.0)

        self.optimizer_G.step()
        self.optimizer_H.step()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def get_current_errors(self):
        errors = super().get_current_errors()
        errors['ffl'] = self.loss_ffl.item()
        return errors
