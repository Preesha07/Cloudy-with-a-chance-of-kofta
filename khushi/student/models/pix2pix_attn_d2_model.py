import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import pytorch_ssim


class Pix2Pix_attn_D2_Model(BaseModel):
    """Experiment 1: AttentionGAN + dedicated discriminator (netD2) on G2 (SAR→optical).

    Identical to Pix2Pix_attn_Model except that G2's backward pass now includes an
    adversarial loss from netD2, which forces fake_C to look like real optical imagery.
    The adversarial weight (G2_GAN_WEIGHT) is kept low so the L1 anchor dominates early.
    """

    G2_GAN_WEIGHT = 0.1  # weight for adversarial term in backward_G2

    def name(self):
        return 'Pix2Pix_attn_D2_Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        nb = opt.batchSize
        size = opt.fineSize
        self.zeros = self.Tensor(nb, 1, size, size)
        self.ones = self.Tensor(nb, 1, size, size)

        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_C = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)

        # --- networks ---
        self.netG = networks.define_G(
            opt.input_nc + opt.input_nc, opt.output_nc, opt.ngf,
            opt.which_model_netG, opt.norm, not opt.no_dropout,
            opt.init_type, self.gpu_ids,
        )
        self.netG2 = networks.define_G(
            opt.input_nc, opt.output_nc, opt.ngf,
            opt.which_model_netG, opt.norm, not opt.no_dropout,
            opt.init_type, self.gpu_ids,
        )
        self.netA = networks.define_A(
            opt.input_nc, 1, opt.ngf, opt.which_model_netA, opt.norm,
            not opt.no_dropout, opt.init_type, self.gpu_ids,
        )

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            # netD: main discriminator on (SAR, fake_B/real_B) — unchanged from baseline
            self.netD = networks.define_D(
                opt.input_nc + opt.output_nc, opt.ndf,
                opt.which_model_netD,
                opt.n_layers_D, opt.norm, use_sigmoid,
                opt.init_type, self.gpu_ids,
            )
            # netD2: auxiliary discriminator on (SAR, fake_C/real_B)
            self.netD2 = networks.define_D(
                opt.input_nc + opt.output_nc, opt.ndf,
                opt.which_model_netD,
                opt.n_layers_D, opt.norm, use_sigmoid,
                opt.init_type, self.gpu_ids,
            )

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            self.load_network(self.netG2, 'G2', opt.which_epoch)
            self.load_network(self.netA, 'A', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)
                self.load_network(self.netD2, 'D2', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.fake_CC_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G2 = torch.optim.Adam(self.netG2.parameters(),
                                                 lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            # netD2 was overpowering G2 (converging much faster than the generator),
            # so its LR is dropped an order of magnitude below every other optimizer.
            self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(),
                                                 lr=opt.lr * 0.1, betas=(opt.beta1, 0.999))
            self.optimizer_A = torch.optim.Adam(self.netA.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = [
                self.optimizer_G, self.optimizer_G2,
                self.optimizer_D, self.optimizer_D2,
                self.optimizer_A,
            ]
            self.schedulers = []
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        networks.print_network(self.netG2)
        networks.print_network(self.netA)
        if self.isTrain:
            networks.print_network(self.netD)
            networks.print_network(self.netD2)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        input_C = input['C']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.input_C.resize_(input_C.size()).copy_(input_C)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.image_paths2 = input['C_paths']

    def mask_layer(self, foreground, background, mask):
        return foreground * mask + background * (1 - mask)

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_C = Variable(self.input_C)
        self.zeros_attn = Variable(self.zeros, requires_grad=False)
        self.att_A = self.netA.forward(self.real_A)
        self.fake_C = self.netG2.forward(self.real_C)
        fake_B = self.netG.forward(torch.cat([self.real_A, self.fake_C], dim=1))
        self.g_B = fake_B
        self.fake_B = self.mask_layer(fake_B, self.real_A, self.att_A)
        self.real_B = Variable(self.input_B)

    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_C = Variable(self.input_C, volatile=True)
        self.att_A = self.netA.forward(self.real_A)
        self.fake_C = self.netG2.forward(self.real_C)
        fake_B = self.netG.forward(torch.cat([self.real_A, self.fake_C], dim=1))
        self.g_B = fake_B
        self.fake_B = self.mask_layer(fake_B, self.real_A, self.att_A)
        self.real_B = Variable(self.input_B, volatile=True)

    def get_image_paths(self):
        return self.image_paths, self.image_paths2

    def backward_D(self):
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_C, self.fake_B), 1))
        self.pred_fake = self.netD.forward(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

        real_AB = torch.cat((self.real_C, self.real_B), 1)
        self.pred_real = self.netD.forward(real_AB)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)

        self.loss_D = 0.5 * (self.loss_D_fake + self.loss_D_real)
        self.loss_D.backward()

    def backward_D2(self):
        # Fake: (SAR, fake_C) — netD2 should output 0
        fake_CC = self.fake_CC_pool.query(torch.cat((self.real_C, self.fake_C), 1))
        self.loss_D2_fake = self.criterionGAN(self.netD2(fake_CC.detach()), False)

        # Real: (SAR, real_B) — one-sided label smoothing. Instead of a hard 1.0
        # target, each element gets an independent random label in [0.75, 1.0), so
        # netD2 can never drive its real-side loss to ~0 and dominate G2 early on.
        real_CC = torch.cat((self.real_C, self.real_B), 1)
        pred_real = self.netD2(real_CC)
        smooth_labels = Variable(
            self.Tensor(pred_real.size()).uniform_(0.75, 1.0), requires_grad=False)
        self.loss_D2_real = self.criterionGAN.loss(pred_real, smooth_labels)

        self.loss_D2 = 0.5 * (self.loss_D2_fake + self.loss_D2_real)
        self.loss_D2.backward()

    def backward_G2(self):
        self.loss_G2_L1 = self.criterionL1(self.fake_C, self.real_B)
        ssim_loss2 = pytorch_ssim.SSIM()
        self.ssimloss2 = 1 - ssim_loss2(self.fake_C, self.real_B)

        # Adversarial term: fool netD2 into thinking fake_C is real optical
        fake_CC = torch.cat((self.real_C, self.fake_C), 1)
        self.loss_G2_GAN = self.criterionGAN(self.netD2(fake_CC), True)

        self.loss_G2 = (self.G2_GAN_WEIGHT * self.loss_G2_GAN
                        + self.loss_G2_L1
                        + self.ssimloss2)
        self.loss_G2.backward(retain_graph=True)

    def backward_G(self):
        fake_AB = torch.cat((self.real_C, self.fake_B), 1)
        pred_fake = self.netD.forward(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
        self.loss_attnsparse_A = self.criterionL1(self.att_A, self.zeros_attn) * self.opt.loss_attn_A

        ssim_loss = pytorch_ssim.SSIM()
        self.ssimloss = 1 - ssim_loss(self.fake_B, self.real_B)

        self.loss_G = (self.loss_G_GAN
                       + 10 * self.ssimloss
                       + self.loss_attnsparse_A
                       + 100 * self.loss_G_L1)
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        # Accumulate G2 gradients first (retain graph — fake_C feeds into netG)
        self.optimizer_G2.zero_grad()
        self.backward_G2()

        # Accumulate G + A gradients (backward_G backprops through fake_B → fake_C → netG2,
        # so both G and G2 grads accumulate before any weight update)
        self.optimizer_G.zero_grad()
        self.optimizer_A.zero_grad()
        self.backward_G()

        # Step all generators together after both backward passes
        self.optimizer_G.step()
        self.optimizer_G2.step()
        self.optimizer_A.step()

        # D2 uses fake_C via the pool (.detach() inside backward_D2), so safe after G2 step
        self.optimizer_D2.zero_grad()
        self.backward_D2()
        self.optimizer_D2.step()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

    def get_current_errors(self):
        return OrderedDict([
            ('G_GAN',    self.loss_G_GAN.item()),
            ('G_L1',     self.loss_G_L1.item()),
            ('G2_GAN',   self.loss_G2_GAN.item()),
            ('G2_L1',    self.loss_G2_L1.item()),
            ('G2_SSIM',  self.ssimloss2.item()),
            ('D_real',   self.loss_D_real.item()),
            ('D_fake',   self.loss_D_fake.item()),
            ('D2_real',  self.loss_D2_real.item()),
            ('D2_fake',  self.loss_D2_fake.item()),
            ('G_SSIM',   self.ssimloss.item()),
            ('att_A',    self.loss_attnsparse_A.item()),
        ])

    def get_current_visuals(self):
        image_numpy = self.att_A.data[0, 0].cpu().float().numpy()
        np.save('map.npy', image_numpy)

        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        real_C = util.tensor2im(self.real_C.data)
        fake_C = util.tensor2im(self.fake_C.data)
        g_B = util.tensor2im(self.g_B.data)
        attn_real_A = util.mask2heatmap(self.att_A.data)
        attn_real_A3 = util.tensor2im(self.att_A.data)
        attn_real_A2 = util.overlay(real_A, attn_real_A)
        return OrderedDict([
            ('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B),
            ('real_C', real_C), ('fake_C', fake_C), ('g_B', g_B),
            ('attn_A', attn_real_A), ('attn_A2', attn_real_A2), ('attn_A3', attn_real_A3),
        ])

    def save(self, label):
        self.save_network(self.netG,  'G',  label, self.gpu_ids)
        self.save_network(self.netG2, 'G2', label, self.gpu_ids)
        self.save_network(self.netD,  'D',  label, self.gpu_ids)
        self.save_network(self.netD2, 'D2', label, self.gpu_ids)
        self.save_network(self.netA,  'A',  label, self.gpu_ids)
