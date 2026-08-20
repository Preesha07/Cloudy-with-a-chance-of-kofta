"""
NIR teacher v4c-FFL-SAR — v4c + focal frequency sharpening on BOTH G and G2,
emphasised cloud-region sharpness, and a new explicit SAR-reliance loss.

Motivation (2026-08-07): v4c's netG2 (SAR->optical translation) is trained with
only L1 + SSIM (see pix2pix_attn_nir_v4c_model.py::_compute_G2_losses). That
combination is exactly the "regress to the mean" recipe documented at the top
of this repo's teacher CLAUDE.md — it produces a blurry fake_C. Since netG's
only channel of SAR information is `cat([real_A, fake_C])`, a blurry fake_C
gives netG nothing sharp to borrow from in cloud-covered regions, so netG
falls back on the (uninformative, cloudy) real_A there and fake_B stays blurry
under clouds specifically — the failure mode this experiment targets.

Three additions on top of v4c:

  1. Focal Frequency Loss (FFL, Jiang et al. ICCV 2021) on fake_B vs real_B —
     same implementation already used by v4adv_ffl and the student's fsp_ffl
     model. Weight: --lambda_ffl (default 300.0, same calibration as those).

  2. Focal Frequency Loss on fake_C vs real_B — NEW, not present in any prior
     teacher variant. Sharpens the SAR-derived intermediate directly, at the
     source of the blur, rather than only patching it up after the fact in
     fake_B. Weight: --lambda_ffl_g2 (default 150.0 — half of lambda_ffl,
     since G2's L1+SSIM anchor is already weaker than G's att-weighted L1;
     inspect loss_ffl_g2 in the first ~100 iters and rescale if it swamps
     loss_G2_L1/ssimloss2).

  3. SAR-reliance loss — NEW. Directly penalises netG's raw decoder output
     (g_B, pre-attention-blend) for disagreeing with netG2's SAR-derived
     translation (fake_C, detached) inside attention-flagged cloud regions:

         loss_sar_reliance = mean( att_A * |g_B - fake_C.detach()| ) * lambda_sar_reliance

     fake_C is detached so gradient flows only into netG (pulling it toward
     what SAR already suggests), not into netG2 (which keeps its own
     independent real_B supervision and does not get pulled toward whatever
     netG currently outputs — that direction would risk both nets collapsing
     onto a shared blurry compromise instead of netG2 staying SAR-accurate).
     att_A is detached (frozen oracle, as elsewhere in v4c) so this loss only
     ever pushes netG, never reshapes the cloud mask itself.
     Weight: --lambda_sar_reliance (default 20.0 — calibrate the same way:
     watch loss_sar_reliance vs 100*loss_G_L1 in the first ~100 iters).

  4. Cloud-region Sobel weight is raised relative to v4c_scratch's own run
     (50.0) to the repo's already-registered default of 200.0
     (--lambda_sharp_cloud), i.e. 4x the full-image Sobel — "even more
     emphasis on sharpness in cloud-flagged areas" as requested. Full-image
     Sobel (--lambda_sharp) is also nudged up from v4c_scratch's 50.0 default
     to 75.0 so clear-sky edges do not fall further behind.

Everything else (frozen netA oracle, att-weighted L1, warm-start loading,
netD/netD2-free architecture) is inherited unchanged from v4c.

Training flags (--model pix2pix_attn_nir_v4c_ffl_sar):
  --lambda_ffl            FFL weight on fake_B vs real_B          (default 300.0)
  --lambda_ffl_g2          FFL weight on fake_C vs real_B          (default 150.0)
  --lambda_sar_reliance    att_A-weighted |g_B - fake_C.detach()|  (default 20.0)
  --lambda_sharp           full-image Sobel                        (default 50.0, pass 75.0 to emphasise)
  --lambda_sharp_cloud     att_A-weighted Sobel                    (default 200.0)
  --lambda_clear           clear-pixel downweight in att-L1        (default 0.1)
  --warmstart_name         experiment to load G/G2/D from          (default: sen12mscr_nir_teacher_v4c_scratch)
  --warmstart_epoch        epoch tag                                (default: latest)
  --frozen_attn_name       frozen netA source                       (default: sen12mscr_nir_teacher_v3)
  --frozen_attn_epoch      epoch tag for netA                       (default: latest)
"""

from collections import OrderedDict

import torch

import util.util as util
from .pix2pix_attn_nir_v4c_model import Pix2Pix_attn_NIR_v4c_Model


class Pix2Pix_attn_NIR_v4c_FFL_SAR_Model(Pix2Pix_attn_NIR_v4c_Model):
    """v4c + FFL on G and G2 + emphasised cloud-Sobel + explicit SAR-reliance loss."""

    FFL_WEIGHT_DEFAULT       = 300.0   # G: fake_B vs real_B
    FFL_G2_WEIGHT_DEFAULT    = 150.0   # G2: fake_C vs real_B (NEW)
    SAR_RELIANCE_DEFAULT     = 20.0    # NEW: att_A-weighted |g_B - fake_C.detach()|
    FFL_ALPHA                = 1.0     # spectral weight exponent (paper Eq. 3-4)

    def name(self):
        return 'Pix2Pix_attn_NIR_v4c_FFL_SAR_Model'

    def initialize(self, opt):
        # Default warm-start is v4c_scratch; override so users don't have to pass the flag.
        if not hasattr(opt, 'warmstart_name') or not opt.warmstart_name:
            opt.warmstart_name = 'sen12mscr_nir_teacher_v4c_scratch'

        super().initialize(opt)

        if self.isTrain:
            self._lambda_ffl         = getattr(opt, 'lambda_ffl',          self.FFL_WEIGHT_DEFAULT)
            self._lambda_ffl_g2      = getattr(opt, 'lambda_ffl_g2',       self.FFL_G2_WEIGHT_DEFAULT)
            self._lambda_sar_reliance = getattr(opt, 'lambda_sar_reliance', self.SAR_RELIANCE_DEFAULT)

            print('--- NIR v4c-FFL-SAR extra losses ---')
            print(f'  lambda_ffl={self._lambda_ffl}  '
                  f'lambda_ffl_g2={self._lambda_ffl_g2}  '
                  f'lambda_sar_reliance={self._lambda_sar_reliance}  '
                  f'lambda_sharp={self.lambda_sharp}  lambda_sharp_cloud={self.lambda_sharp_cloud}')

    # ── Focal Frequency Loss (identical formula to v4adv_ffl / student fsp_ffl) ──

    def _focal_frequency_loss(self, pred, target):
        # torch.fft.fft2 has no safe fp16 path (ComplexHalf is "experimental" and
        # silently produces NaN under autocast once spectral magnitudes over/underflow
        # fp16's range — observed directly in this model's own smoke test). Force this
        # computation to fp32 regardless of the surrounding AMP context; it's cheap
        # relative to the generator forward/backward so there's no real perf cost.
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred_f32 = pred.float()
            targ_f32 = target.float()
            pred_freq = torch.fft.fft2(pred_f32, norm='ortho')
            targ_freq = torch.fft.fft2(targ_f32, norm='ortho')
            diff      = pred_freq - targ_freq
            freq_dist = diff.real ** 2 + diff.imag ** 2

            with torch.no_grad():
                w = freq_dist / (freq_dist.amax(dim=(2, 3), keepdim=True) + 1e-8)
                w = w.clamp(0.0, 1.0) ** self.FFL_ALPHA

            return (w * freq_dist).mean()

    # ── Loss overrides ────────────────────────────────────────────────────────

    def _compute_G2_losses(self):
        """v4c's G2 L1+SSIM, plus FFL to directly sharpen the SAR-derived fake_C."""
        super()._compute_G2_losses()   # sets loss_G2_L1, ssimloss2, loss_G2

        self.loss_ffl_g2 = self._focal_frequency_loss(self.fake_C, self.real_B)
        self.loss_G2 = self.loss_G2 + self._lambda_ffl_g2 * self.loss_ffl_g2

    def _compute_G_losses(self):
        """v4c losses + FFL on fake_B + explicit SAR-reliance on g_B."""
        super()._compute_G_losses()   # sets loss_G_GAN, loss_G_L1, ssimloss,
                                      # loss_sharp, loss_sharp_cloud, and self.loss_G

        att = self.att_A.detach()

        # 1. Focal Frequency Loss on full fake_B vs real_B
        self.loss_ffl = self._focal_frequency_loss(self.fake_B, self.real_B)

        # 2. SAR-reliance: pull netG's raw output toward netG2's SAR-derived
        #    translation specifically where the cloud mask says to (fake_C
        #    detached — this only pulls netG, never netG2).
        pixel_diff = torch.abs(self.g_B - self.fake_C.detach())
        self.loss_sar_reliance = (att * pixel_diff).mean()

        self.loss_G = (
            self.loss_G
            + self._lambda_ffl          * self.loss_ffl
            + self._lambda_sar_reliance * self.loss_sar_reliance
        )

    def get_current_errors(self):
        errors = super().get_current_errors()
        errors['ffl']          = self.loss_ffl.item()
        errors['ffl_g2']       = self.loss_ffl_g2.item()
        errors['sar_reliance'] = self.loss_sar_reliance.item()
        return errors

    def get_current_visuals(self):
        """Same set as v4c, but the optical composites (real_A/real_B/fake_B/
        fake_C/g_B) are saved in false-colour-infrared display order (NIR in
        the red channel) instead of the raw (Red,Green,NIR) storage order — the
        whole point of this experiment is judging sharpness/SAR-reliance in
        cloud-fill regions by eye, and NIR-in-blue is hard to read at a glance.
        real_C (SAR, VV/VH/VV-VH — a different channel semantic entirely) and
        the attention maps are left on plain tensor2im, unswapped."""
        real_A  = util.tensor2im_fcc(self.real_A.data)
        fake_B  = util.tensor2im_fcc(self.fake_B.data)
        real_B  = util.tensor2im_fcc(self.real_B.data)
        real_C  = util.tensor2im(self.real_C.data)
        fake_C  = util.tensor2im_fcc(self.fake_C.data)
        g_B     = util.tensor2im_fcc(self.g_B.data)
        attn_A  = util.mask2heatmap(self.att_A.data)
        attn_A2 = util.overlay(real_A, attn_A)
        attn_A3 = util.tensor2im(self.att_A.data)
        return OrderedDict([
            ('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B),
            ('real_C', real_C), ('fake_C', fake_C), ('g_B', g_B),
            ('attn_A', attn_A), ('attn_A2', attn_A2), ('attn_A3', attn_A3),
        ])
