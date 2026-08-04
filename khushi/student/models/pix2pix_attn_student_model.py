"""
Student model for SAR-free cloud removal via modality hallucination.

Architecture: same four-module structure as pix2pix_attn (netG, netA, netD, netG2),
but netG2 (the SAR→optical translation module) is replaced by netH (the hallucination
module), which takes RG+NIR optical input instead of SAR.

At inference: only cloudy RG+NIR input is needed — no SAR.

Reference: Hoffman et al. (2016) "Learning with Side Information through Modality
Hallucination", applied to the cloud-attention GAN of Zhang et al. (2023).

Band mapping (S2 → LISS-4 equivalent):
  channel 0 = B4  (Red,  665 nm)
  channel 1 = B3  (Green, 560 nm)
  channel 2 = B8  (NIR,   842 nm)
"""

import os
import numpy as np
import torch
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


class Pix2Pix_attn_Student_Model(BaseModel):

    def name(self):
        return 'Pix2Pix_attn_Student_Model'

    # ── Initialization ────────────────────────────────────────────────────────

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        nb   = opt.batchSize
        size = opt.fineSize
        self.zeros = self.Tensor(nb, 1, size, size)

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

        print('---------- Student networks initialized -------------')
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
        live tensors (with grad) for backprop through the hallucination loss.
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
        """L_hallucinate = Σ_ℓ mean( (σ(teacher_act_ℓ) - σ(student_act_ℓ))² )

        Applies sigmoid before L2 distance (Hoffman et al. Eq. 1).
        Summation is over all hooked layers; the result is weighted by γ outside.
        """
        total = 0.0
        for idx in self.hook_layer_indices:
            t_act = self._teacher_acts[idx]   # detached → stable supervision
            s_act = self._student_acts[idx]   # differentiable → grad into netH
            total = total + torch.mean((torch.sigmoid(t_act) - torch.sigmoid(s_act)) ** 2)
        return total  # 0-D tensor (scalar) after first loop iteration

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

        # ── L_hallucination: sigmoid L2 at hooked UNet levels ─────────────────
        self.loss_hall = self._hallucination_loss()

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

        total_student_loss = self.loss_H + self.loss_G + self.opt.gamma_hall * self.loss_hall
        total_student_loss.backward()

        # Gradient clipping guards against hallucination loss spikes (Hoffman et al.)
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
            ('L_hall',   self.loss_hall.item()),           # hallucination loss (unweighted)
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
