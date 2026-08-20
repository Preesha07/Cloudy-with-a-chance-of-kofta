"""
NIR teacher v4c — v4a + cloud-region sharpness loss.

Diff vs v4a:
  ADDED : loss_sharp_cloud
            Per-pixel Sobel gradient difference (fake_B vs real_B), weighted
            by att_A AFTER computing gradients (so mask edges don't inject
            artificial gradient signals).  Gradient computation runs on the full
            image; att_A acts as a soft selector that routes the sharpness
            penalty toward cloud-covered pixels.

            loss_sharp_cloud = mean( att_A * (|∇x_err| + |∇y_err|) ) * λ_sharp_cloud

  KEPT  : loss_sharp (full-image Sobel, λ_sharp=50) — clear-sky edges still
            supervised so the generator does not lose sharpness outside clouds.

Warm-start loading:
  On first launch (not --continue_train), if --warmstart_name is set, loads
  netG / netG2 / netD from that checkpoint.  Intended use: start from
  sen12mscr_nir_teacher_v4a epoch 100.

Training flags (--model pix2pix_attn_nir_v4c):
  --warmstart_name      experiment to warm-start G/G2/D from  (default: sen12mscr_nir_teacher_v4a)
  --warmstart_epoch     epoch tag for that checkpoint         (default: latest)
  --frozen_attn_name    checkpoint to load frozen netA from   (default: sen12mscr_nir_teacher_v3)
  --frozen_attn_epoch   epoch tag for that checkpoint         (default: latest)
  --lambda_clear        clear-pixel weight in att-weighted L1 (default: 0.1)
  --lambda_sharp        full-image Sobel weight               (default: 50.0)
  --lambda_sharp_cloud  cloud-region Sobel weight             (default: 50.0)
"""

import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import pytorch_ssim


class _GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        self.register_buffer('kx', kx.view(1, 1, 3, 3))
        self.register_buffer('ky', ky.view(1, 1, 3, 3))

    def _per_pixel(self, pred, target):
        """Return per-pixel |∇pred − ∇target| as [B, C, H, W]."""
        B, C, H, W = pred.shape
        p = pred.view(B * C, 1, H, W)
        t = target.view(B * C, 1, H, W)
        gx_p = F.conv2d(p, self.kx, padding=1)
        gy_p = F.conv2d(p, self.ky, padding=1)
        gx_t = F.conv2d(t, self.kx, padding=1)
        gy_t = F.conv2d(t, self.ky, padding=1)
        return (torch.abs(gx_p - gx_t) + torch.abs(gy_p - gy_t)).view(B, C, H, W)

    def forward(self, pred, target):
        """Uniform mean over all pixels."""
        return self._per_pixel(pred, target).mean()

    def forward_masked(self, pred, target, mask):
        """Gradient-difference weighted by mask [B,1,H,W] ∈ [0,1].

        Gradients are computed on the full image first; att_A acts as a soft
        selector afterwards.  This avoids artifact gradients at mask edges that
        would appear if we masked the images before convolution.
        """
        diff = self._per_pixel(pred, target)          # [B, C, H, W]
        return (diff * mask.expand_as(diff)).mean()


class Pix2Pix_attn_NIR_v4c_Model(BaseModel):
    def name(self):
        return 'Pix2Pix_attn_NIR_v4c_Model'

    def _load_net_from(self, network, net_label, ckpt_dir, experiment, epoch):
        path = os.path.join(ckpt_dir, experiment, f'{epoch}_net_{net_label}.pth')
        if not os.path.exists(path):
            raise FileNotFoundError(f'v4c: checkpoint not found: {path}')
        network.load_state_dict(torch.load(path, map_location='cpu'), strict=False)

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,  opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
        self.input_C = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)

        self.netG = networks.define_G(
            opt.input_nc + opt.input_nc, opt.output_nc, opt.ngf,
            opt.which_model_netG, opt.norm, not opt.no_dropout,
            opt.init_type, self.gpu_ids)
        self.netG2 = networks.define_G(
            opt.input_nc, opt.output_nc, opt.ngf,
            opt.which_model_netG, opt.norm, not opt.no_dropout,
            opt.init_type, self.gpu_ids)
        self.netA = networks.define_A(
            opt.input_nc, 1, opt.ngf, opt.which_model_netA, opt.norm,
            not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(
                opt.input_nc + opt.output_nc, opt.ndf,
                opt.which_model_netD, opt.n_layers_D, opt.norm,
                use_sigmoid, opt.init_type, self.gpu_ids)

        # ── Frozen netA oracle (always from v3 or configured source) ────────
        frozen_attn_name  = getattr(opt, 'frozen_attn_name',  'sen12mscr_nir_teacher_v3')
        frozen_attn_epoch = getattr(opt, 'frozen_attn_epoch', 'latest')
        self._load_net_from(self.netA, 'A', opt.checkpoints_dir, frozen_attn_name, frozen_attn_epoch)
        self.netA.eval()
        for p in self.netA.parameters():
            p.requires_grad_(False)

        # ── Checkpoint loading: resume this experiment OR warm-start from v4a ─
        if not self.isTrain or opt.continue_train:
            # Normal resume: load from this experiment's own checkpoint dir
            self.load_network(self.netG,  'G',  opt.which_epoch)
            self.load_network(self.netG2, 'G2', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)
        else:
            warmstart_name  = getattr(opt, 'warmstart_name',  'sen12mscr_nir_teacher_v4a')
            warmstart_epoch = getattr(opt, 'warmstart_epoch', 'latest')
            if warmstart_name:
                print(f'v4c: warm-starting G/G2/D from {warmstart_name}/{warmstart_epoch}')
                self._load_net_from(self.netG,  'G',  opt.checkpoints_dir, warmstart_name, warmstart_epoch)
                self._load_net_from(self.netG2, 'G2', opt.checkpoints_dir, warmstart_name, warmstart_epoch)
                if self.isTrain:
                    self._load_net_from(self.netD, 'D', opt.checkpoints_dir, warmstart_name, warmstart_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            self.criterionGAN  = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1   = torch.nn.L1Loss(reduction='none')
            self.criterionSSIM = pytorch_ssim.SSIM()
            self.criterionGrad = _GradientLoss()

            if len(self.gpu_ids) > 0:
                dev = self.gpu_ids[0]
                self.criterionSSIM = self.criterionSSIM.cuda(dev)
                self.criterionGrad = self.criterionGrad.cuda(dev)

            self.lambda_sharp       = getattr(opt, 'lambda_sharp',       50.0)
            self.lambda_sharp_cloud = getattr(opt, 'lambda_sharp_cloud', 50.0)
            self.lambda_clear       = getattr(opt, 'lambda_clear',        0.1)

            self.optimizers = []
            self.schedulers = []
            self.optimizer_G  = torch.optim.Adam(self.netG.parameters(),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G2 = torch.optim.Adam(self.netG2.parameters(),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D  = torch.optim.Adam(self.netD.parameters(),
                                                  lr=opt.lr * 0.01, betas=(opt.beta1, 0.999))
            for opt_ in [self.optimizer_G, self.optimizer_G2, self.optimizer_D]:
                self.optimizers.append(opt_)
                self.schedulers.append(networks.get_scheduler(opt_, opt))

        print('--- NIR v4c: frozen netA oracle + att-weighted L1 + cloud-region Sobel ---')
        print(f'  netA loaded from {frozen_attn_name}/{frozen_attn_epoch} (FROZEN)')
        if self.isTrain:
            print(f'  lambda_sharp={self.lambda_sharp}  lambda_sharp_cloud={self.lambda_sharp_cloud}  lambda_clear={self.lambda_clear}')

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
        self.real_B = Variable(self.input_B)
        with torch.no_grad():
            self.att_A = self.netA(self.real_A)
        self.fake_C = self.netG2(self.real_C)
        g_B = self.netG(torch.cat([self.real_A, self.fake_C], dim=1))
        self.g_B    = g_B
        self.fake_B = self.mask_layer(g_B, self.real_A, self.att_A)

    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_C = Variable(self.input_C, volatile=True)
        self.real_B = Variable(self.input_B, volatile=True)
        with torch.no_grad():
            self.att_A = self.netA(self.real_A)
        self.fake_C = self.netG2(self.real_C)
        g_B = self.netG(torch.cat([self.real_A, self.fake_C], dim=1))
        self.g_B    = g_B
        self.fake_B = self.mask_layer(g_B, self.real_A, self.att_A)

    def get_image_paths(self):
        return self.image_paths, self.image_paths2

    # ── Loss helpers ──────────────────────────────────────────────────────────

    def _compute_G2_losses(self):
        self.loss_G2_L1 = self.criterionL1(self.fake_C, self.real_B).mean()
        self.ssimloss2  = 1 - self.criterionSSIM(self.fake_C, self.real_B)
        self.loss_G2    = self.loss_G2_L1 + self.ssimloss2

    def _compute_G_losses(self):
        fake_AB = torch.cat((self.real_C, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        att = self.att_A.detach()
        pixel_l1 = self.criterionL1(self.fake_B, self.real_B)
        weighted_l1 = (att * pixel_l1 + (1.0 - att) * pixel_l1 * self.lambda_clear).mean()
        self.loss_G_L1 = weighted_l1 * self.opt.lambda_A

        self.ssimloss = 1 - self.criterionSSIM(self.fake_B, self.real_B)

        # Full-image Sobel (retained from v4a)
        self.loss_sharp = self.criterionGrad(self.fake_B, self.real_B) * self.lambda_sharp

        # Cloud-region Sobel: gradient computed on full image, then weighted by att_A
        self.loss_sharp_cloud = self.criterionGrad.forward_masked(
            self.fake_B, self.real_B, att
        ) * self.lambda_sharp_cloud

        self.loss_G = (
            self.loss_G_GAN
            + 10  * self.ssimloss
            + 100 * self.loss_G_L1
            + self.loss_sharp
            + self.loss_sharp_cloud
        )

    def _compute_D_losses(self):
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_C, self.fake_B), 1))
        self.pred_fake   = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

        real_AB = torch.cat((self.real_C, self.real_B), 1)
        self.pred_real = self.netD(real_AB)
        smooth_label = torch.empty_like(self.pred_real).uniform_(0.75, 1.0)
        self.loss_D_real = self.criterionGAN.loss(self.pred_real, smooth_label)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

    # ── Backward / optimize ───────────────────────────────────────────────────

    def backward_G2(self):
        self._compute_G2_losses()
        self.loss_G2.backward(retain_graph=True)

    def backward_G(self):
        self._compute_G_losses()
        self.loss_G.backward()

    def backward_D(self):
        self._compute_D_losses()
        self.loss_D.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_G2.zero_grad()
        self.backward_G2()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.optimizer_G2.step()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def optimize_parameters_amp(self, scaler):
        ctx = torch.amp.autocast('cuda')
        with ctx:
            self.forward()

        self.optimizer_G2.zero_grad()
        self.optimizer_G.zero_grad()
        with ctx:
            self._compute_G2_losses()
            self._compute_G_losses()
            loss_gen = self.loss_G2 + self.loss_G
        scaler.scale(loss_gen).backward()
        scaler.unscale_(self.optimizer_G2)
        scaler.unscale_(self.optimizer_G)
        scaler.step(self.optimizer_G2)
        scaler.step(self.optimizer_G)

        self.optimizer_D.zero_grad()
        with ctx:
            self._compute_D_losses()
        scaler.scale(self.loss_D).backward()
        scaler.unscale_(self.optimizer_D)
        scaler.step(self.optimizer_D)
        scaler.update()

    # ── Logging / saving ─────────────────────────────────────────────────────

    def get_current_errors(self):
        return OrderedDict([
            ('G_GAN',        self.loss_G_GAN.item()),
            ('G_L1',         self.loss_G_L1.item()),
            ('G2_L1',        self.loss_G2_L1.item()),
            ('G2_SSIM',      self.ssimloss2.item()),
            ('D_real',       self.loss_D_real.item()),
            ('D_fake',       self.loss_D_fake.item()),
            ('G_SSIM',       self.ssimloss.item()),
            ('G_sharp',      self.loss_sharp.item()),
            ('G_sharp_cld',  self.loss_sharp_cloud.item()),
        ])

    def get_current_visuals(self):
        image_numpy = self.att_A.data[0, 0].cpu().float().numpy()
        np.save('map_nir_v4c.npy', image_numpy)
        real_A  = util.tensor2im(self.real_A.data)
        fake_B  = util.tensor2im(self.fake_B.data)
        real_B  = util.tensor2im(self.real_B.data)
        real_C  = util.tensor2im(self.real_C.data)
        fake_C  = util.tensor2im(self.fake_C.data)
        g_B     = util.tensor2im(self.g_B.data)
        attn_A  = util.mask2heatmap(self.att_A.data)
        attn_A2 = util.overlay(real_A, attn_A)
        attn_A3 = util.tensor2im(self.att_A.data)
        return OrderedDict([
            ('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B),
            ('real_C', real_C), ('fake_C', fake_C), ('g_B', g_B),
            ('attn_A', attn_A), ('attn_A2', attn_A2), ('attn_A3', attn_A3),
        ])

    def save(self, label):
        self.save_network(self.netG,  'G',  label, self.gpu_ids)
        self.save_network(self.netG2, 'G2', label, self.gpu_ids)
        self.save_network(self.netD,  'D',  label, self.gpu_ids)
        self.save_network(self.netA,  'A',  label, self.gpu_ids)
