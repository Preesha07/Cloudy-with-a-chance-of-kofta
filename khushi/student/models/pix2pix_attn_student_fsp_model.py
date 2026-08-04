"""
Student model for SAR-free cloud removal via modality hallucination, adding a
cross-depth FSP-style covariance term (Yim et al., CVPR 2017) on top of the
exact-match and DIST relational terms.

Forked from `pix2pix_attn_student_dist_model.py`.  Three distillation terms are
available and each is independently toggleable, so any subset can be ablated:

  --use_hall         L_hall = Sum_l mean( (sigma(T_l) - sigma(S_l))^2 )   exact match
  --use_dist         L_dist = beta*L_inter + gamma*L_intra                within-layer
  --use_cross_depth  L_fsp  = Sum_pairs d( G_t, G_s )                     CROSS-DEPTH (new)

The new term
------------
`L_hall` and `L_dist` both compare a teacher and student activation *at the same
depth*.  Neither constrains how information EVOLVES between depths.  The FSP
("flow of solution procedure") matrix of Yim et al. captures exactly that: for an
adjacent pair of hooked levels it contracts the two feature maps over space,

    F_i' = adaptive_avg_pool2d(F_i, (H_j, W_j))          # pool DOWN to deeper grid
    G    = (1/(H_j*W_j)) * F_i'.reshape(C_i,-1) @ F_j.reshape(C_j,-1).T    # [C_i, C_j]

leaving a channel-by-channel matrix that says "how much does shallow channel a
co-activate with deep channel b".  Matching G_teacher to G_student asks netH to
reproduce the teacher's *inter-layer* structure.

Because the contraction is over spatial positions within a single instance, this
works at batchSize 1 -- unlike DIST's intra-relation term, which needs multiple
instances and is why `dist_intra_min_positions` exists.

Distance (--fsp_distance):
  'pearson'   (default) 1 - pearson(vec(G_t), vec(G_s)).  Scale-invariant, which
              matters here because C_i*C_j and the activation magnitudes differ by
              orders of magnitude between the shallow and deep pairs.
  'frobenius' ||G_t - G_s||_F^2 / (C_i*C_j).  The Yim et al. original.

Two measured facts about this codebase that shape the implementation
--------------------------------------------------------------------
1. Hooked tensors are `cat([x, model(x)])`, and `out[:, :C_in]` is bitwise equal to
   the block input x.  That echoed half is near-identical between teacher and
   student (both derive from the same real_A/real_C path), so including it would
   inflate the measured similarity.  The cross-depth term therefore uses the
   OUTPUT HALF ONLY, sliced by the per-hook C_in captured at hook time.
   Verified geometry (unet_256, ngf 64, 256x256 input):
       block 3: C 512  C_in 256  -> out-half 256 @ 32x32
       block 4: C 1024 C_in 512  -> out-half 512 @ 16x16
       block 5: C 1024 C_in 512  -> out-half 512 @  8x8
       block 6: C 1024 C_in 512  -> out-half 512 @  4x4
       block 7: C 1024 C_in 512  -> out-half 512 @  2x2
   Note block 3's split is 256/512, NOT an even half -- never hardcode C//2.
2. The parent block's `uprelu = nn.ReLU(inplace=True)` mutates the child's hooked
   tensor AFTER the hook fires.  Every distillation term in this file therefore
   sees post-ReLU (non-negative) activations.  This is inherited from the baseline
   and deliberately NOT changed: all three terms read the same dicts, so the
   three-way ablation stays clean.

Rank caveat, worth watching in the diagnostics
----------------------------------------------
rank(G) <= H_j*W_j.  On the real geometry that is 256 / 64 / 16 / 4 positions for
pairs 3-4 / 4-5 / 5-6 / 6-7, so the two deepest pairs build a 512x512 matrix from
16 and 4 spatial samples respectively.  The eps guard will NOT flag these -- they
have healthy variance, they are merely near-rank-deficient.  Per-pair diagnostics
are logged so this can be judged from data rather than assumed; --fsp_min_positions
can then skip them.

Original DIST notes retained below.

  Baseline  L_hall = Sum_l  mean( (sigma(T_l) - sigma(S_l))^2 )        <- exact match
  DIST      L_dist = beta * L_inter + gamma * L_intra                  <- relaxed match

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


# ── Cross-depth FSP primitives (Yim et al., CVPR 2017) ───────────────────────

def fsp_matrix(f_shallow, f_deep):
    """FSP/Gram matrix between two feature maps at different depths.

    Args:
        f_shallow: [B, C_i, H_i, W_i]  (larger spatial extent)
        f_deep:    [B, C_j, H_j, W_j]  (smaller spatial extent)
    Returns:
        [B, C_i, C_j]

    The shallower map is pooled DOWN to the deeper map's grid.  Never interpolate
    the deeper one upward -- that invents detail the network never computed.
    """
    b, ci, hi, wi = f_shallow.shape
    _, cj, hj, wj = f_deep.shape
    if (hi, wi) != (hj, wj):
        f_shallow = F.adaptive_avg_pool2d(f_shallow, (hj, wj))
    a = f_shallow.reshape(b, ci, hj * wj)
    d = f_deep.reshape(b, cj, hj * wj)
    return torch.bmm(a, d.transpose(1, 2)) / float(hj * wj)


def fsp_pearson(g_t, g_s, eps=1e-6):
    """Pearson correlation between flattened FSP matrices, per instance.

    Returns (corr [B], degenerate [B] bool).

    The denominator uses clamp_min rather than "+ eps": an additive epsilon biases
    every correlation downward by eps/(std_t*std_s + eps), which is not negligible
    for low-variance G -- identical inputs scored 1-6e-4 instead of 1 in testing.
    Degenerate instances are masked out anyway, so the clamp only ever guards a
    division whose result is discarded.
    """
    b = g_t.shape[0]
    t = g_t.reshape(b, -1)
    s = g_s.reshape(b, -1)
    t = t - t.mean(dim=1, keepdim=True)
    s = s - s.mean(dim=1, keepdim=True)
    t_std = t.pow(2).mean(dim=1).sqrt()
    s_std = s.pow(2).mean(dim=1).sqrt()
    degenerate = (t_std < eps) | (s_std < eps)
    corr = (t * s).mean(dim=1) / (t_std * s_std).clamp_min(eps)
    return corr, degenerate


def build_fsp_pairs(hook_indices, mode='adjacent'):
    """Layer-index pairs (shallower, deeper) to build FSP matrices between."""
    idx = sorted(hook_indices)
    if mode == 'adjacent':
        return [(idx[k], idx[k + 1]) for k in range(len(idx) - 1)]
    if mode == 'all':
        return [(idx[a], idx[b])
                for a in range(len(idx)) for b in range(a + 1, len(idx))]
    raise ValueError("fsp_pairs must be 'adjacent' or 'all', got %r" % mode)


class Pix2Pix_attn_Student_FSP_Model(BaseModel):

    def name(self):
        return 'Pix2Pix_attn_Student_FSP_Model'

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

        # ── Term toggles.  Each distillation term is independently switchable so
        # any subset can be ablated.  getattr-with-default so test_student.py,
        # which does not register these flags, can still instantiate the model.
        self.use_hall        = bool(getattr(opt, 'use_hall', True))
        self.use_dist        = bool(getattr(opt, 'use_dist', True))
        self.use_cross_depth = bool(getattr(opt, 'use_cross_depth', True))

        # ── Cross-depth FSP hyperparameters ──────────────────────────────────
        self.cross_depth_weight = getattr(opt, 'cross_depth_weight', 1.0)
        self.fsp_distance       = getattr(opt, 'fsp_distance', 'pearson')
        self.fsp_pairs_mode     = getattr(opt, 'fsp_pairs', 'adjacent')
        self.fsp_eps            = getattr(opt, 'fsp_eps', 1e-6)
        # 0 disables the skip; see the rank caveat in the module docstring.  The
        # per-pair diagnostics are the instrument for deciding whether to raise it.
        self.fsp_min_positions  = getattr(opt, 'fsp_min_positions', 0)
        if self.fsp_distance not in ('pearson', 'frobenius'):
            raise ValueError("fsp_distance must be 'pearson' or 'frobenius', got %r"
                             % self.fsp_distance)

        # Per-hook input-channel counts, filled by the forward hooks.  Used to
        # slice the echoed skip input off each captured tensor.  Captured rather
        # than hardcoded because the split is NOT an even half at every level
        # (block 3 is 256/512).
        self._hook_c_in = {}
        # Per-pair epoch accumulators for the diagnostics.
        self._fsp_epoch = {}

        # ── FreqKD hyperparameters (Thaker et al., arXiv 2606.11572, 2026) ───
        # Frequency-decoupled cross-modal distillation: applies strict MSE on
        # the low-frequency band (shared structural content across modalities)
        # and relaxed log-MSE on the high-frequency band (SAR speckle/texture
        # that netH cannot reproduce from optical input, so penalising it
        # heavily injects harmful gradient).
        self.use_freqkd         = bool(getattr(opt, 'use_freqkd', False))
        self.freqkd_weight      = getattr(opt, 'freqkd_weight', 10.0)
        self.freqkd_rc          = getattr(opt, 'freqkd_rc', 0.5)
        self.freqkd_eta         = getattr(opt, 'freqkd_eta', 0.1)
        self.freqkd_min_spatial = getattr(opt, 'freqkd_min_spatial', 8)
        self._freqkd_masks      = {}   # (H, W) -> radial mask; cached per spatial size
        self._freqkd_epoch      = {}   # per-layer epoch accumulators

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

            self._fsp_pairs = build_fsp_pairs(self.hook_layer_indices,
                                              self.fsp_pairs_mode)
            print('Terms enabled: hall=%s  dist=%s  cross_depth=%s'
                  % (self.use_hall, self.use_dist, self.use_cross_depth))
            print('FSP: pairs=%s (%s)  distance=%s  weight=%g  min_positions=%d'
                  % (self._fsp_pairs, self.fsp_pairs_mode, self.fsp_distance,
                     self.cross_depth_weight, self.fsp_min_positions))
            print('FreqKD: enabled=%s  weight=%g  rc=%g  eta=%g  min_spatial=%d'
                  % (self.use_freqkd, self.freqkd_weight, self.freqkd_rc,
                     self.freqkd_eta, self.freqkd_min_spatial))

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
                    # Channel count of the echoed skip input, needed to slice the
                    # tensor down to the block's own computed output.  Read from
                    # the live input rather than assumed, since the split is not
                    # an even half at every level.
                    self._hook_c_in[i] = inp[0].shape[1]
                return fn
            def make_s_hook(i):
                def fn(module, inp, out):
                    self._student_acts[i] = out
                return fn
            self._hooks.append(teacher_blocks[idx].register_forward_hook(make_t_hook(idx)))
            self._hooks.append(student_blocks[idx].register_forward_hook(make_s_hook(idx)))

    def _out_half(self, act, idx):
        """Drop the echoed skip-connection input, keeping the block's own output.

        `UnetSkipConnectionBlock.forward` returns cat([x, model(x)]) for every
        non-outermost block, and out[:, :C_in] is bitwise equal to x.  Both nets
        carry near-identical signal there, so leaving it in would inflate the
        measured similarity independently of anything netH learns.
        """
        c_in = self._hook_c_in.get(idx)
        if c_in is None or c_in >= act.shape[1]:
            return act          # outermost block, or hook never fired
        return act[:, c_in:]

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

    def _cross_depth_loss(self):
        """FSP-style cross-depth covariance matching, summed over layer pairs.

        Returns a 0-D tensor.  Per-pair statistics land in self._fsp_stats for
        logging, and are accumulated into self._fsp_epoch for the epoch summary.

        Operates on OUTPUT-HALF channels only (see _out_half) and per instance,
        so it is well-defined at batchSize 1.
        """
        device = self.real_A.device
        total = torch.zeros((), device=device, dtype=torch.float32)
        self._fsp_stats = {}

        for (i, j) in self._fsp_pairs:
            t_i = self._out_half(self._teacher_acts[i].float(), i)
            t_j = self._out_half(self._teacher_acts[j].float(), j)
            s_i = self._out_half(self._student_acts[i].float(), i)
            s_j = self._out_half(self._student_acts[j].float(), j)

            n_positions = t_j.shape[2] * t_j.shape[3]
            if self.fsp_min_positions and n_positions < self.fsp_min_positions:
                continue

            g_t = fsp_matrix(t_i, t_j)
            g_s = fsp_matrix(s_i, s_j)

            if self.fsp_distance == 'pearson':
                corr, degen = fsp_pearson(g_t, g_s, eps=self.fsp_eps)
                keep = ~degen
                if keep.any():
                    pair_loss = (1.0 - corr[keep]).mean()
                else:
                    # Every instance degenerate: contribute nothing rather than a
                    # fabricated gradient.  The diagnostic below records it.
                    pair_loss = torch.zeros((), device=device, dtype=torch.float32)
                degen_frac = degen.float().mean().item()
            else:  # frobenius
                ci, cj = g_t.shape[1], g_t.shape[2]
                pair_loss = ((g_t - g_s).pow(2).flatten(1).sum(1)
                             / float(ci * cj)).mean()
                degen_frac = 0.0

            self._fsp_stats[(i, j)] = (pair_loss.item(), degen_frac)
            acc = self._fsp_epoch.setdefault((i, j), [0.0, 0.0, 0])
            acc[0] += pair_loss.item()
            acc[1] += degen_frac
            acc[2] += 1

            total = total + pair_loss

        return total

    # ── FreqKD: frequency-decoupled cross-modal distillation ─────────────────

    def _channel_normalize(self, x):
        """Per-channel L2 normalization over spatial dims (FreqKD Eq. 1).

        Removes the mean and scale difference between SAR-driven teacher
        features and optical-driven student features before spectral comparison.
        Without this, MSE in the FFT domain is dominated by the DC offset
        rather than the spatial pattern.
        x: (B, C, H, W) -> (B, C, H, W), each channel zero-mean unit-norm.
        """
        mu = x.mean(dim=(2, 3), keepdim=True)
        centered = x - mu
        norm = centered.norm(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return centered / norm

    def _get_radial_mask(self, H, W, device):
        """Radial low-pass binary mask centred in the FFT spectrum (FreqKD Eq. 3).

        Cached per (H, W) to avoid recomputation each forward pass.
        rho = 0 at DC centre, rho = 1 at the corner.  Pixels with rho <= rc
        form the low-pass band; the rest form the high-pass band.
        Returns a (H, W) float32 tensor: 1 inside the disc, 0 outside.
        """
        if (H, W) not in self._freqkd_masks:
            cy, cx = H // 2, W // 2
            ys = torch.arange(H).float() - cy
            xs = torch.arange(W).float() - cx
            rho = torch.sqrt(
                (ys[:, None] / max(cy, 1)) ** 2
                + (xs[None, :] / max(cx, 1)) ** 2
            ) / (2 ** 0.5)
            self._freqkd_masks[(H, W)] = (rho <= self.freqkd_rc).float()
        return self._freqkd_masks[(H, W)].to(device)

    def _freqkd_layer_loss(self, t_act, s_act):
        """Frequency-decoupled distillation loss for one hooked layer.

        Returns (l_low, l_high) as 0-D tensors (before weighting by eta).

        Low-band  (structural): strict MSE — shape/layout transfers across
                  the SAR-optical modality gap.
        High-band (textural):   log(1+MSE) — saturates on large residuals
                  from SAR speckle/double-bounce that netH cannot reproduce
                  from cloudy optical input.  Enforcing this strictly would
                  inject harmful gradient from an unsatisfiable target.
        """
        t_n = self._channel_normalize(t_act)
        s_n = self._channel_normalize(s_act)

        H, W = t_act.shape[-2:]

        # 2D FFT with DC at centre (Eq. 2)
        t_fft = torch.fft.fftshift(torch.fft.fft2(t_n), dim=(-2, -1))
        s_fft = torch.fft.fftshift(torch.fft.fft2(s_n), dim=(-2, -1))

        mask = self._get_radial_mask(H, W, t_act.device)          # (H, W)
        mask = mask.unsqueeze(0).unsqueeze(0)                      # (1, 1, H, W)

        def to_spatial(spec, m):
            return torch.fft.ifft2(
                torch.fft.ifftshift(spec * m, dim=(-2, -1))
            ).real

        # Teacher bands detached — teacher is frozen; sg() matches paper notation
        t_low  = to_spatial(t_fft, mask).detach()
        s_low  = to_spatial(s_fft, mask)
        t_high = to_spatial(t_fft, 1.0 - mask).detach()
        s_high = to_spatial(s_fft, 1.0 - mask)

        l_low  = (s_low  - t_low ).pow(2).mean()                  # Eq. 4
        l_high = torch.log(1.0 + (s_high - t_high).pow(2).mean()) # Eq. 5
        return l_low, l_high

    def _freqkd_loss(self):
        """FreqKD loss summed over hooked layers with sufficient spatial size.

        Skips layers where H or W < freqkd_min_spatial: 2D FFT on a 4x4 or
        2x2 map yields 16 / 4 frequency bins — not enough for a meaningful
        low/high split.  On unet_256 with hook_layers 3,4,5,6,7 this skips
        blocks 6 and 7 (4x4, 2x2) and fires on 3,4,5 (32x32, 16x16, 8x8).

        Per-layer (l_low, l_high) land in self._freqkd_stats for per-iter
        logging and are accumulated into self._freqkd_epoch for epoch summary.
        """
        device = self.real_A.device
        total  = torch.zeros((), device=device, dtype=torch.float32)
        self._freqkd_stats = {}

        for idx in self.hook_layer_indices:
            t_act = self._out_half(self._teacher_acts[idx].float(), idx)
            s_act = self._out_half(self._student_acts[idx].float(), idx)
            H, W = t_act.shape[-2:]

            if H < self.freqkd_min_spatial or W < self.freqkd_min_spatial:
                continue

            l_low, l_high = self._freqkd_layer_loss(t_act, s_act)

            self._freqkd_stats[idx] = (l_low.item(), l_high.item())
            acc = self._freqkd_epoch.setdefault(idx, [0.0, 0.0, 0])
            acc[0] += l_low.item()
            acc[1] += l_high.item()
            acc[2] += 1

            total = total + l_low + self.freqkd_eta * l_high      # Eq. 6

        return total

    def get_freqkd_layer_errors(self):
        """Per-layer FreqKD band losses for the current iteration (unweighted)."""
        out = OrderedDict()
        for idx, (l_low, l_high) in sorted(
                getattr(self, '_freqkd_stats', {}).items()):
            out['fkd_l%d_lo' % idx] = l_low
            out['fkd_l%d_hi' % idx] = l_high
        return out

    # ── Epoch-level FSP diagnostics ──────────────────────────────────────────

    def reset_epoch_metrics(self):
        self._fsp_epoch    = {}
        self._freqkd_epoch = {}

    def get_epoch_metrics(self):
        """Per-pair FSP diagnostics and per-layer FreqKD band losses over the epoch."""
        out = OrderedDict()
        for (i, j), (vsum, dsum, n) in sorted(self._fsp_epoch.items()):
            if n == 0:
                continue
            out['fsp_%d_%d' % (i, j)]       = vsum / n
            out['fsp_%d_%d_degen' % (i, j)] = dsum / n
        for idx, (lsum, hsum, n) in sorted(self._freqkd_epoch.items()):
            if n == 0:
                continue
            out['fkd_l%d_low'  % idx] = lsum / n
            out['fkd_l%d_high' % idx] = hsum / n
        return out

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

        # ── Distillation terms, each independently toggleable ─────────────────
        # Disabled terms are set to a zero tensor rather than skipped, so
        # get_current_errors() keeps a stable set of columns across ablations.
        zero = torch.zeros((), device=self.real_A.device, dtype=torch.float32)

        self.loss_hall = self._hallucination_loss() if self.use_hall else zero

        if self.use_dist:
            self.loss_inter, self.loss_intra = self._dist_losses()
        else:
            self.loss_inter, self.loss_intra = zero, zero
        self.loss_dist = self.dist_beta * self.loss_inter + self.dist_gamma * self.loss_intra

        self.loss_fsp    = self._cross_depth_loss() if self.use_cross_depth else zero
        self.loss_freqkd = self._freqkd_loss()      if self.use_freqkd     else zero

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
                              + self.loss_dist
                              + self.cross_depth_weight  * self.loss_fsp
                              + self.freqkd_weight       * self.loss_freqkd)
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
            ('L_fsp',    self.loss_fsp.item()),
            ('L_freqkd', self.loss_freqkd.item() if torch.is_tensor(self.loss_freqkd)
                         else float(self.loss_freqkd)),
        ])

    def get_fsp_pair_errors(self):
        """Per-pair cross-depth stats for the current iteration."""
        out = OrderedDict()
        for (i, j), (val, degen) in sorted(getattr(self, '_fsp_stats', {}).items()):
            out['fsp_%d_%d' % (i, j)] = val
            if degen > 0:
                out['fsp_%d_%d_degen' % (i, j)] = degen
        return out

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
