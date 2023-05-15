import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from models.base_model import BaseModel
from . import networks_complete
import sys
import scipy.stats as st
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# ~~~~~~
from .model import Hed


def no_sigmoid_cross_entropy(sig_logits, label):
    # print(sig_logits)
    count_neg = torch.sum(1.-label)
    count_pos = torch.sum(label)

    beta = count_neg / (count_pos+count_neg)
    pos_weight = beta / (1-beta)

    cost = pos_weight * label * (-1) * torch.log(sig_logits) + (1-label)* (-1) * torch.log(1-sig_logits)
    cost = torch.mean(cost * (1-beta))

    return cost
# ~~~~~~


class CycleGANModel(BaseModel):
    def name2(self):
        return 'CycleGANModel'

    def gauss_kernel(self, kernlen=21, nsig=3, channels=1):
        interval = (2 * nsig + 1.) / (kernlen)
        x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw / kernel_raw.sum()
        out_filter = np.array(kernel, dtype=np.float32)
        out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
        out_filter = np.repeat(out_filter, channels, axis=2)
        return out_filter

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        self.netG_A = networks_complete.define_G_A(opt.input_nc, opt.output_nc,
                                                   opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type,
                                                   self.gpu_ids)
        self.netG_B = networks_complete.define_G_B(opt.output_nc, opt.input_nc,
                                                   opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks_complete.define_D(opt.output_nc, opt.ndf,
                                                     opt.which_model_netD,
                                                     opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_B = networks_complete.define_D(opt.input_nc, opt.ndf,
                                                     opt.which_model_netD,
                                                     opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

            self.netD_ink = networks_complete.define_D(opt.output_nc, opt.ndf,
                                                       opt.which_model_netD,
                                                       opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

            g_kernel = self.gauss_kernel(21, 3, 1).transpose((3, 2, 1, 0))
            self.gauss_conv = nn.Conv2d(1, 1, kernel_size=21, stride=1, padding=1, bias=False)
            self.gauss_conv.weight.data.copy_(torch.from_numpy(g_kernel))  # 创建一个与g_kernel相同的tensor
            self.gauss_conv.weight.requires_grad = False
            self.gauss_conv.cuda()

            # ~~~~~~
            self.hed_model = Hed()
            self.hed_model.cuda()
            save_path = './35.pth'
            self.hed_model.load_state_dict(torch.load(save_path))
            for param in self.hed_model.parameters():
                param.requires_grad = False
            # ~~~~~~

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)
                self.load_network(self.netD_ink, 'D_ink', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.ink_fake_B_pool = ImagePool(opt.pool_size)

            # define loss functions
            self.criterionGAN = networks_complete.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.style = torch.nn.MSELoss()

            # # initialize optimizers
            # self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
            #                                     lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_A = torch.optim.Adam(self.netG_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_B = torch.optim.Adam(self.netG_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr_D, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr_D, betas=(opt.beta1, 0.999))
            self.optimizer_D_ink = torch.optim.Adam(self.netD_ink.parameters(), lr=opt.lr_D, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            # self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_G_A)
            self.optimizers.append(self.optimizer_G_B)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_B)
            self.optimizers.append(self.optimizer_D_ink)
            for optimizer in self.optimizers:
                self.schedulers.append(networks_complete.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks_complete.print_network(self.netG_A)
        networks_complete.print_network(self.netG_B)
        if self.isTrain:
            networks_complete.print_network(self.netD_A)
            networks_complete.print_network(self.netD_B)
            networks_complete.print_network(self.netD_ink)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.input_A = input['A' if AtoB else 'B'].to(self.device)
        self.input_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

        kernel_size = 5
        pad_size = kernel_size//2
        p1d = (pad_size, pad_size, pad_size, pad_size)

        p_real_B = F.pad(self.real_B, p1d, mode="constant", value=1)
        erode_real_B = -1*(F.max_pool2d(-1*p_real_B, kernel_size, 1))

        res1 = self.gauss_conv(erode_real_B[:, 0, :, :].unsqueeze(1))
        res2 = self.gauss_conv(erode_real_B[:, 1, :, :].unsqueeze(1))
        res3 = self.gauss_conv(erode_real_B[:, 2, :, :].unsqueeze(1))

        self.ink_real_B = torch.cat((res1, res2, res3), dim=1)

    def test(self):
        real_A = Variable(self.input_A, volatile=True)
        real_B = Variable(self.input_B, volatile=True)
        fake_B, _ = self.netG_A(real_A, real_B)
        # fake_B = self.netG_A(real_A)
        # self.rec_A, _ = self.netG_B(fake_B).data
        self.rec_A = self.netG_B(fake_B).data
        self.fake_B = fake_B.data
        self.edge_fake_B = self.fake_B
        self.edge_real_A = self.fake_B
        fake_A = self.netG_B(real_B)
        # self.rec_B, _ = self.netG_A(fake_A).data
        rec_B, _ = self.netG_A(fake_A, fake_B)
        # rec_B = self.netG_A(fake_A)
        self.rec_B = rec_B.data
        self.fake_A = fake_A.data

        self.ink_real_B = fake_A
        self.ink_fake_B = self.fake_A

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        self.loss_D_A = loss_D_A.item()

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        self.loss_D_B = loss_D_B.item()

    def backward_D_ink(self):
        ink_fake_B = self.ink_fake_B_pool.query(self.ink_fake_B)
        loss_D_ink = self.backward_D_basic(self.netD_ink, self.ink_real_B, ink_fake_B)
        self.loss_D_ink = loss_D_ink.item()

    def backward_G(self, lambda_sup):
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_ink = self.opt.lambda_ink
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            idt_A, _ = self.netG_A(self.real_B, self.real_B)
            # idt_A = self.netG_A(self.real_B)
            loss_idt_A = self.criterionIdt(idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            idt_B = self.netG_B(self.real_A)
            loss_idt_B = self.criterionIdt(idt_B, self.real_A) * lambda_A * lambda_idt

            self.idt_A = idt_A.data
            self.idt_B = idt_B.data
            self.loss_idt_A = loss_idt_A.item()
            self.loss_idt_B = loss_idt_B.item()
        else:
            loss_idt_A = 0
            loss_idt_B = 0
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        fake_B, _ = self.netG_A(self.real_A, self.real_B)
        # fake_B = self.netG_A(self.real_A)
        edge_real_A = F.sigmoid(self.hed_model(self.real_A).detach())
        edge_fake_B = F.sigmoid(self.hed_model(fake_B))
        loss_edge_1 = no_sigmoid_cross_entropy(edge_fake_B, edge_real_A) * lambda_sup
        # print('edge')
        # print(loss_edge_1)

        pred_fake = self.netD_A(fake_B)
        loss_G_A = self.criterionGAN(pred_fake, True)

        kernel_size = 5
        pad_size = kernel_size//2
        p1d = (pad_size, pad_size, pad_size, pad_size)
        p_fake_B = F.pad(fake_B, p1d, "constant", 1)
        erode_fake_B = -1*(F.max_pool2d(-1*p_fake_B, kernel_size, 1))

        res1 = self.gauss_conv(erode_fake_B[:, 0, :, :].unsqueeze(1))
        res2 = self.gauss_conv(erode_fake_B[:, 1, :, :].unsqueeze(1))
        res3 = self.gauss_conv(erode_fake_B[:, 2, :, :].unsqueeze(1))

        ink_fake_B = torch.cat((res1, res2, res3), dim=1)
        pred_fake_ink = self.netD_ink(ink_fake_B)
        loss_G_ink = self.criterionGAN(pred_fake_ink, True) * lambda_ink

        # GAN loss D_B(G_B(B))
        fake_A = self.netG_B(self.real_B)
        pred_fake = self.netD_B(fake_A)
        loss_G_B = self.criterionGAN(pred_fake, True)

        # Forward cycle loss
        rec_A = self.netG_B(fake_B)
        loss_cycle_A = self.criterionCycle(rec_A, self.real_A) * lambda_A

        # Backward cycle loss
        rec_B, _ = self.netG_A(fake_A, self.real_B)
        # rec_B = self.netG_A(fake_A)
        loss_cycle_B = self.criterionCycle(rec_B, self.real_B) * lambda_B

        # style loss
        _, style_fake_B = self.netG_A(self.real_A, fake_B)
        _, style_real_B = self.netG_A(self.real_A, self.real_B)
        loss_style = self.style(style_real_B, style_fake_B)

        # combined loss
        # loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B + loss_style + loss_edge_1 + loss_G_ink
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B + loss_G_ink + loss_style + loss_edge_1
        loss_G.backward()

        self.fake_B = fake_B.data
        self.fake_A = fake_A.data
        self.rec_A = rec_A.data
        self.rec_B = rec_B.data
        self.edge_real_A = edge_real_A.data
        self.edge_fake_B = edge_fake_B.data
        self.ink_fake_B = ink_fake_B.data


        self.loss_G_A = loss_G_A.item()
        self.loss_G_B = loss_G_B.item()
        self.loss_G_ink = loss_G_ink.item()
        self.loss_cycle_A = loss_cycle_A.item()
        self.loss_cycle_B = loss_cycle_B.item()
        self.loss_edge_1 = loss_edge_1.item()
        self.loss_style = loss_style.item()

    def optimize_parameters(self, lambda_sup):
        # forward
        self.forward()
        # G_A and G_B
        # self.optimizer_G.zero_grad()
        # self.backward_G(lambda_sup)
        # self.optimizer_G.step()
        # G_A
        self.optimizer_G_A.zero_grad()
        self.backward_G(lambda_sup)
        self.optimizer_G_A.step()
        # G_B
        self.optimizer_G_B.zero_grad()
        self.backward_G(lambda_sup)
        self.optimizer_G_B.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()
        # D_ink
        self.optimizer_D_ink.zero_grad()
        self.backward_D_ink()
        self.optimizer_D_ink.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([('D_A', self.loss_D_A), ('G_A', self.loss_G_A), ('Cyc_A', self.loss_cycle_A),
                                 ('D_B', self.loss_D_B), ('G_B', self.loss_G_B), ('Cyc_B',  self.loss_cycle_B),
                                  ('edge1', self.loss_edge_1), ('D_ink', self.loss_D_ink), ('G_ink', self.loss_G_ink),
                                  ('styleloss', self.loss_style)])
        if self.opt.identity > 0.0:
            ret_errors['idt_A'] = self.loss_idt_A
            ret_errors['idt_B'] = self.loss_idt_B
        return ret_errors

    def get_current_visuals(self):
        real_A = util.tensor2im(self.input_A)
        fake_B = util.tensor2im(self.fake_B)
        rec_A = util.tensor2im(self.rec_A)
        real_B = util.tensor2im(self.input_B)
        fake_A = util.tensor2im(self.fake_A)
        rec_B = util.tensor2im(self.rec_B)
        edge_fake_B = util.tensor2im(self.edge_fake_B)
        edge_real_A = util.tensor2im(self.edge_real_A)
        ink_real_B = util.tensor2im(self.ink_real_B.data)
        ink_fake_B = util.tensor2im(self.ink_fake_B)
        ret_visuals = OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_B', rec_B), ('rec_A', rec_A),
                                   ('real_B', real_B),  ('fake_A', fake_A),
                                   ('edge_fake_B', edge_fake_B), ('edge_real_A', edge_real_A),
                                   ('ink_real_B', ink_real_B), ('ink_fake_B', ink_fake_B)])
        # if self.opt.isTrain and self.opt.identity > 0.0:
        #     ret_visuals['idt_A'] = util.tensor2im(self.idt_A)
        #     ret_visuals['idt_B'] = util.tensor2im(self.idt_B)
        return ret_visuals

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)
        self.save_network(self.netD_ink, 'D_ink', label, self.gpu_ids)
