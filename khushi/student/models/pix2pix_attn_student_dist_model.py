"""
Student model for SAR-free cloud removal via modality hallucination, with the
hallucination match relaxed using DIST (Huang et al., NeurIPS 2022).

Difference from `pix2pix_attn_student_model.py` (which this file forks):

  Baseline  L_hall = Sum_l  mean( (sigma(T_l) - sigma(S_l))^2 )        <- exact match
  This file L_dist = beta * L_inter + gamma * L_intra                  <- relaxed match

`L_hall` is minimised only when the student's activations are *literally identical*
to the teacher's.  That is the "exact match" DIST argues against.  It is a
particularly bad fit here: netH must reproduce, from cloudy optical input,
activations the teacher computed from SAR.  Under thick cloud there is no optical
evidence of the surface at all, so an exact match is not just hard, it is
unsatisfiable, and the residual shows up as gradient noise on every weight.

Pearson's distance d_p(u,v) = 1 - rho_p(u,v) is invariant to positive affine
rescaling (paper Eq. 5), so it asks netH to preserve the *relative structure* of
the teacher's response rather than its absolute values.

Adapting DIST from classification logits to conv feature maps
-------------------------------------------------------------
DIST operates on a (B, C) matrix of logits: rows = instances, cols = classes.
A hooked UNet activation is (B, C, H, W).  We flatten it to (N, C) with
N = B*H*W, treating **each spatial location as an instance** and **each channel
as a class**:

  L_inter  Pearson along the channel axis, per location.   (paper Eq. 8)
           "at this pixel, do teacher and student rank the channels the same way"
  L_intra  Pearson along the location axis, per channel.   (paper Eq. 9)
           "for this channel, does the student's spatial response pattern track
            the teacher's"

Using spatial position (not batch) as the instance axis is forced by this repo:
batchSize is 1, so a batch-wise intra-relation would correlate length-1 vectors
and be undefined.

Reference: Huang et al. 2022, "Knowledge Distillation from A Stronger Teacher";
Hoffman et al. 2016, "Learning with Side Information through Modality
Hallucination"; Zhang et al. 2023 for the underlying cloud-attention GAN.

Band mapping (S2 -> LISS-4 equivalent):
  channel 0 = B4  (Red,  665 nm)
  channel 1 = B3  (Green, 560 nm)
  channel 2 = B8  (NIR,   842 nm)
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init as nn_init
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .networks import UnetSkipConnectionBlock
import pytorch_ssim


def _get_unet_blocks(gen):
    """Return list of UnetSkipConnectionBlock instances, outermost-first.

    Works for any UnetGenerator with arbitrarily nested blocks.
    """
    blocks = []
    block = gen.model          # UnetGenerator.model = outermost block
    while block is not None:
        blocks.append(block)
        nxt = None
        for child in block.model.children():
            if isinstance(child, UnetSkipConnectionBlock):
                nxt = child
                break
        block = nxt
    return blocks


# ── DIST primitives ───────────────────────────────────────────────────────────

def pearson_corr(a, b, dim, eps=1e-8):
    """Pearson correlation coefficient between `a` and `b` along `dim`.

    Implemented as a dot product of mean-centred, L2-normalised vectors, which is
    algebraically identical to paper Eq. 7 but avoids forming the covariance and
    the two standard deviations separately (fewer chances to divide by ~0).

    Returns a tensor with `dim` reduced.
    """
    a = a - a.mean(dim=dim, keepdim=True)
    b = b - b.mean(dim=dim, keepdim=True)
    num = (a * b).sum(dim=dim)
    den = a.norm(dim=dim) * b.norm(dim=dim)
    return num / (den + eps)


def _flatten_to_instances(x):
    """(B, C, H, W) -> (N, C) with N = B*H*W.  Rows are spatial locations."""
    b, c, h, w = x.shape
    return x.permute(0, 2, 3, 1).reshape(b * h * w, c)


def dist_inter_loss(t_act, s_act, eps=1e-8):
    """Inter-channel relation (paper Eq. 8, channels playing the role of classes).

    For each spatial location, correlate the teacher's and student's vector of
    channel responses.  Invariant to a per-location scale/shift of the whole
    response vector.
    """
    t = _flatten_to_instances(t_act)
    s = _flatten_to_instances(s_act)
    return (1.0 - pearson_corr(t, s, dim=1, eps=eps)).mean()


def dist_intra_loss(t_act, s_act, eps=1e-8):
    """Intra-channel relation (paper Eq. 9, channels playing the role of classes).

    For each channel, correlate the teacher's and student's spatial response
    pattern.  Invariant to a per-channel scale/shift, which is exactly the
    freedom netH needs: it may fire on a different absolute scale than the
    SAR-driven teacher so long as it lights up in the same places.
    """
    t = _flatten_to_instances(t_act)
    s = _flatten_to_instances(s_act)
    return (1.0 - pearson_corr(t, s, dim=0, eps=eps)).mean()


class Pix2Pix_attn_Student_DIST_Model(BaseModel):

    def name(self):
        return 'Pix2Pix_attn_Student_DIST_Model'

    # ── Initialization ────────────────────────────────────────────────────────

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        nb   = opt.batchSize
        size = opt.fineSize
        self.zeros = self.Tensor(nb, 1, size, size)

        # DIST hyperparameters.  getattr so that test_student.py — which does not
        # define the dist_* flags — can still instantiate this model for inference.
        self.dist_beta  = getattr(opt, 'dist_beta',  2.0)
        self.dist_gamma = getattr(opt, 'dist_gamma', 2.0)
        self.dist_act   = getattr(opt, 'dist_act',   'none')
        self.dist_intra_min_positions = getattr(opt, 'dist_intra_min_positions', 16)
        self.dist_eps   = getattr(opt, 'dist_eps',   1e-8)

        # Placeholder tensors (same as teacher)
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,  opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
        self.input_C = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)

        # ── Student modules (same architectures as teacher) ───────────────────
        # netG: generator (takes 6ch = cloudy + fake_C)
        self.netG = networks.define_G(
            opt.input_nc + opt.input_nc, opt.output_nc, opt.ngf,
            opt.which_model_netG, opt.norm, not opt.no_dropout,
            opt.init_type, self.gpu_ids)
        # netA: attention module
        self.netA = networks.define_A(
            opt.input_nc, 1, opt.ngf,
            opt.which_model_netA, opt.norm, not opt.no_dropout,
            opt.init_type, self.gpu_ids)
        # netH: hallucination module — same architecture as teacher's netG2
        # Input: RG+NIR optical (3ch, same channel count as SAR, different domain)
        self.netH = networks.define_G2(
            opt.input_nc, opt.output_nc, opt.ngf,
            opt.which_model_netG2, opt.norm, not opt.no_dropout,
            opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(
                opt.input_nc + opt.output_nc, opt.ndf,
                opt.which_model_netD, opt.n_layers_D,
                opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if self.isTrain:
            # ── Frozen teacher (only needed during training) ──────────────────
            self.netG2_teacher = networks.define_G2(
                opt.input_nc, opt.output_nc, opt.ngf,
                opt.which_model_netG2, opt.norm, not opt.no_dropout,
                opt.init_type, self.gpu_ids)

            teacher_dir   = os.path.join(opt.teacher_checkpoints_dir, opt.teacher_name)
            teacher_epoch = opt.teacher_which_epoch

            self._load_from_dir(self.netG,          'G',  teacher_epoch, teacher_dir)
            self._load_from_dir(self.netA,          'A',  teacher_epoch, teacher_dir)
            self._load_from_dir(self.netG2_teacher, 'G2', teacher_epoch, teacher_dir)
            self._load_from_dir(self.netD,          'D',  teacher_epoch, teacher_dir)

            for param in self.netG2_teacher.parameters():
                param.requires_grad = False

            # Warm-start netH: copy teacher G2 weights, re-init first conv.
            # First conv sees optical domain (not SAR) so its kernels don't transfer.
            # Deeper layers produce optical-like feature statistics → do transfer.
            self.netH.load_state_dict(self.netG2_teacher.state_dict())
            self._reinit_first_conv(self.netH, opt.init_type)

            # ── Hallucination hooks ───────────────────────────────────────────
            if hasattr(opt, 'hook_layers') and opt.hook_layers:
                self.hook_layer_indices = opt.hook_layers
            else:
                n_blocks = len(_get_unet_blocks(self.netG2_teacher))
                self.hook_layer_indices = list(range(3, n_blocks))

            self._teacher_acts = {}
            self._student_acts = {}
            self._hooks = []
            self._register_hooks()
            print('Hook layers (0=outermost): %s' % self.hook_layer_indices)
            print('DIST: beta(inter)=%g  gamma(intra)=%g  act=%s  intra_min_positions=%d'
                  % (self.dist_beta, self.dist_gamma, self.dist_act,
                     self.dist_intra_min_positions))
            print('DIST: gamma_hall=%g (Hoffman exact-match term; 0 = pure DIST)'
                  % opt.gamma_hall)

            # Continue-train: replace teacher-init weights with student checkpoint
            if opt.continue_train:
                self.load_network(self.netG, 'G', opt.which_epoch)
                self.load_network(self.netA, 'A', opt.which_epoch)
                self.load_network(self.netH, 'H', opt.which_epoch)
                self.load_network(self.netD, 'D', opt.which_epoch)
        else:
            # Test time: load student checkpoint directly
            self.load_network(self.netG, 'G', opt.which_epoch)
            self.load_network(self.netA, 'A', opt.which_epoch)
            self.load_network(self.netH, 'H', opt.which_epoch)

        # ── Optimizers (student params only) ─────────────────────────────────
        if self.isTrain:
            self.fake_AB_pool   = ImagePool(opt.pool_size)
            self.old_lr         = opt.lr
            self.criterionGAN   = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1    = torch.nn.L1Loss()

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_H = torch.optim.Adam(self.netH.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_A = torch.optim.Adam(self.netA.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers  = [self.optimizer_G, self.optimizer_H, self.optimizer_A, self.optimizer_D]
            self.schedulers  = [networks.get_scheduler(o, opt) for o in self.optimizers]

        print('---------- Student (DIST) networks initialized -------------')
        networks.print_network(self.netG)
        networks.print_network(self.netH)
        networks.print_network(self.netA)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_from_dir(self, network, label, epoch, ckpt_dir):
        path = os.path.join(ckpt_dir, '%s_net_%s.pth' % (epoch, label))
        print('Loading %-3s from %s' % (label, path))
        network.load_state_dict(torch.load(path), strict=False)

    def _reinit_first_conv(self, net, init_type='xavier'):
        """Re-initialize the first downsampling conv of a UnetGenerator.

        The outermost UnetSkipConnectionBlock.model[0] is always downconv.
        This must be re-initialized because the teacher's netG2 first conv learned
        SAR-specific patterns; the hallucination module's first conv must learn
        optical patterns instead.
        """
        first_conv = net.model.model[0]
        if init_type == 'xavier':
            nn_init.xavier_normal_(first_conv.weight.data, gain=1)
        elif init_type == 'kaiming':
            nn_init.kaiming_normal_(first_conv.weight.data, a=0, mode='fan_in')
        else:
            nn_init.normal_(first_conv.weight.data, 0.0, 0.02)
        if first_conv.bias is not None:
            nn_init.constant_(first_conv.bias.data, 0.0)
        print('Re-initialized netH first conv: %s' % first_conv)

    def _register_hooks(self):
        """Register forward hooks on teacher and student UNet blocks.

        Hooks capture the output of each UnetSkipConnectionBlock at the configured
        layer indices. Teacher hooks store detached tensors; student hooks store
        live tensors (with grad) for backprop through the DIST losses.

        Note: for non-outermost blocks `forward` returns `cat([x, model(x)], 1)`,
        so a hooked activation carries the block's *input* in its leading channels
        as well as its output.  This is inherited from the baseline hallucination
        model and kept identical here so the two are directly comparable.
        """
        teacher_blocks = _get_unet_blocks(self.netG2_teacher)
        student_blocks = _get_unet_blocks(self.netH)

        for idx in self.hook_layer_indices:
            def make_t_hook(i):
                def fn(module, inp, out):
                    self._teacher_acts[i] = out.detach()
                return fn
            def make_s_hook(i):
                def fn(module, inp, out):
                    self._student_acts[i] = out
                return fn
            self._hooks.append(teacher_blocks[idx].register_forward_hook(make_t_hook(idx)))
            self._hooks.append(student_blocks[idx].register_forward_hook(make_s_hook(idx)))

    def _transform_acts(self, x):
        """Optional squashing applied before correlating.

        'none'    — correlate raw activations.  Default, and the principled choice:
                    Pearson already absorbs scale and shift, which is the only
                    reason the baseline needed a sigmoid in the first place.
        'sigmoid' — Hoffman's squash.  Use to isolate the effect of swapping only
                    the *metric* (L2 -> Pearson) while holding the input identical.
        'softmax' — closest literal analogue of DIST, which correlates softmax
                    probabilities.  Applied over the channel axis, so each spatial
                    location becomes a distribution over channels.  Discards
                    magnitude entirely.
        """
        if self.dist_act == 'none':
            return x
        if self.dist_act == 'sigmoid':
            return torch.sigmoid(x)
        if self.dist_act == 'softmax':
            return F.softmax(x, dim=1)
        raise ValueError("dist_act must be one of 'none', 'sigmoid', 'softmax'; got %r"
                         % self.dist_act)

    # ── Input / forward ───────────────────────────────────────────────────────

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        input_C = input['C']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.input_C.resize_(input_C.size()).copy_(input_C)
        self.image_paths  = input['A_paths' if AtoB else 'B_paths']
        self.image_paths2 = input['C_paths']

    def mask_layer(self, foreground, background, mask):
        return foreground * mask + background * (1 - mask)

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_C = Variable(self.input_C)
        self.zeros_attn = Variable(self.zeros, requires_grad=False)

        # Run frozen teacher's translation module to capture supervision targets
        with torch.no_grad():
            self.netG2_teacher(self.real_C)
        # _teacher_acts now populated (detached tensors, no grad)

        # Student hallucination: optical → SAR-like intermediate features
        self.fake_C = self.netH(self.real_A)
        # _student_acts now populated (live tensors, grad flows through netH)

        # Attention + main generator (same as teacher)
        self.att_A  = self.netA(self.real_A)
        fake_B      = self.netG(torch.cat([self.real_A, self.fake_C], dim=1))
        self.g_B    = fake_B
        self.fake_B = self.mask_layer(fake_B, self.real_A, self.att_A)
        self.real_B = Variable(self.input_B)

    def test(self):
        """SAR-free inference: only RG+NIR optical input needed."""
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_C = Variable(self.input_C, volatile=True)  # kept for visualisation only
        self.fake_C = self.netH(self.real_A)
        self.att_A  = self.netA(self.real_A)
        fake_B      = self.netG(torch.cat([self.real_A, self.fake_C], dim=1))
        self.g_B    = fake_B
        self.fake_B = self.mask_layer(fake_B, self.real_A, self.att_A)
        self.real_B = Variable(self.input_B, volatile=True)

    # ── Losses ────────────────────────────────────────────────────────────────

    def _hallucination_loss(self):
        """Hoffman exact-match term, retained for ablation.

        L_hallucinate = Sum_l mean( (sigma(T_l) - sigma(S_l))^2 )

        Weighted by opt.gamma_hall.  Set --gamma_hall 0 to train with DIST alone;
        leave it nonzero to blend the exact and relaxed matches.
        """
        total = 0.0
        for idx in self.hook_layer_indices:
            t_act = self._teacher_acts[idx]   # detached → stable supervision
            s_act = self._student_acts[idx]   # differentiable → grad into netH
            total = total + torch.mean((torch.sigmoid(t_act) - torch.sigmoid(s_act)) ** 2)
        return total

    def _dist_losses(self):
        """DIST relational losses, summed over hooked layers.

        Returns (L_inter, L_intra) as 0-D tensors.

        The intra term is skipped at layers with fewer than
        `dist_intra_min_positions` spatial locations.  In unet_256 the hooked
        blocks sit at 32x32 down to 2x2, and a Pearson coefficient over 4 samples
        (block 7 at batchSize 1) is noise, not signal — including it would inject
        a large, meaningless gradient into the deepest layers.
        """
        l_inter = torch.zeros((), device=self.real_A.device, dtype=torch.float32)
        l_intra = torch.zeros((), device=self.real_A.device, dtype=torch.float32)
        self._intra_skipped = []

        for idx in self.hook_layer_indices:
            t_act = self._transform_acts(self._teacher_acts[idx].float())
            s_act = self._transform_acts(self._student_acts[idx].float())

            l_inter = l_inter + dist_inter_loss(t_act, s_act, eps=self.dist_eps)

            n_positions = t_act.shape[0] * t_act.shape[2] * t_act.shape[3]
            if n_positions >= self.dist_intra_min_positions:
                l_intra = l_intra + dist_intra_loss(t_act, s_act, eps=self.dist_eps)
            else:
                self._intra_skipped.append(idx)

        return l_inter, l_intra

    def _compute_losses(self):
        """Compute all student losses.  Called inside optimize_parameters()."""
        ssim_fn = pytorch_ssim.SSIM()

        # ── L_H: hallucination module translation loss (same as teacher's L_G2) ──
        self.loss_H_L1   = self.criterionL1(self.fake_C, self.real_B)
        self.loss_H_SSIM = 1.0 - ssim_fn(self.fake_C, self.real_B)
        self.loss_H      = self.loss_H_L1 + self.opt.lambda3 * self.loss_H_SSIM

        # ── L_G: adversarial + L1 + SSIM + attention sparsity ────────────────
        fake_AB = torch.cat((self.real_C, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN        = self.criterionGAN(pred_fake, True)
        self.loss_G_L1         = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
        self.loss_attnsparse_A = self.criterionL1(self.att_A, self.zeros_attn) * self.opt.loss_attn_A
        self.ssimloss          = 1.0 - ssim_fn(self.fake_B, self.real_B)
        self.loss_G = (self.loss_G_GAN
                       + 10.0 * self.ssimloss
                       + self.loss_attnsparse_A
                       + 100.0 * self.loss_G_L1)

        # ── Distillation: exact match (optional) + DIST relaxed match ─────────
        self.loss_hall = self._hallucination_loss()
        self.loss_inter, self.loss_intra = self._dist_losses()
        self.loss_dist = self.dist_beta * self.loss_inter + self.dist_gamma * self.loss_intra

    def backward_D(self):
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_C, self.fake_B), 1))
        self.pred_fake  = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

        real_AB = torch.cat((self.real_C, self.real_B), 1)
        self.pred_real  = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)

        self.loss_D = 0.5 * (self.loss_D_fake + self.loss_D_real)
        self.loss_D.backward()

    def optimize_parameters(self):
        self.forward()
        self._compute_losses()

        # ── Combined student backward (G + H + A) ─────────────────────────────
        self.optimizer_G.zero_grad()
        self.optimizer_H.zero_grad()
        self.optimizer_A.zero_grad()

        total_student_loss = (self.loss_H
                              + self.loss_G
                              + self.opt.gamma_hall * self.loss_hall
                              + self.loss_dist)
        total_student_loss.backward()

        # Gradient clipping guards against distillation loss spikes (Hoffman et al.)
        student_params = (list(self.netG.parameters())
                          + list(self.netH.parameters())
                          + list(self.netA.parameters()))
        torch.nn.utils.clip_grad_norm_(student_params, max_norm=10.0)

        self.optimizer_G.step()
        self.optimizer_H.step()
        self.optimizer_A.step()

        # ── Discriminator (detached from generator graph) ─────────────────────
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    # ── Logging / saving / loading ────────────────────────────────────────────

    def get_image_paths(self):
        return self.image_paths, self.image_paths2

    def get_current_errors(self):
        return OrderedDict([
            ('G_GAN',    self.loss_G_GAN.item()),
            ('G_L1',     self.loss_G_L1.item()),
            ('G_SSIM',   self.ssimloss.item()),
            ('H_L1',     self.loss_H_L1.item()),
            ('H_SSIM',   self.loss_H_SSIM.item()),
            ('D_real',   self.loss_D_real.item()),
            ('D_fake',   self.loss_D_fake.item()),
            ('att_A',    self.loss_attnsparse_A.item()),
            # Distillation terms, all logged UNWEIGHTED so the printed value is
            # comparable across runs with different beta/gamma/gamma_hall.
            ('L_hall',   self.loss_hall.item() if torch.is_tensor(self.loss_hall)
                         else float(self.loss_hall)),
            ('L_inter',  self.loss_inter.item()),
            ('L_intra',  self.loss_intra.item()),
        ])

    def get_current_visuals(self):
        image_numpy = self.att_A.data[0, 0].cpu().float().numpy()
        np.save('map.npy', image_numpy)

        real_A     = util.tensor2im(self.real_A.data)
        fake_B     = util.tensor2im(self.fake_B.data)
        real_B     = util.tensor2im(self.real_B.data)
        real_C     = util.tensor2im(self.real_C.data)
        fake_C     = util.tensor2im(self.fake_C.data)
        g_B        = util.tensor2im(self.g_B.data)
        attn_A     = util.mask2heatmap(self.att_A.data)
        attn_A3    = util.tensor2im(self.att_A.data)
        attn_A2    = util.overlay(real_A, attn_A)
        return OrderedDict([
            ('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B),
            ('real_C', real_C), ('fake_C', fake_C), ('g_B', g_B),
            ('attn_A', attn_A), ('attn_A2', attn_A2), ('attn_A3', attn_A3),
        ])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netH, 'H', label, self.gpu_ids)  # hallucination module
        self.save_network(self.netD, 'D', label, self.gpu_ids)
        self.save_network(self.netA, 'A', label, self.gpu_ids)
