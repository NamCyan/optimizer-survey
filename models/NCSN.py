#Souce https://github.com/ermongroup/ncsn/blob/master/models/scorenet.py

import torch.nn as nn
import functools
import torch
from torchvision.models import ResNet
import torch.nn.functional as F
# from .pix2pix import init_net, UnetSkipConnectionBlock, get_norm_layer, init_weights, ResnetBlock, \
#     UnetSkipConnectionBlockWithResNet


class ConvResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, resize=False, act='relu'):
        super().__init__()
        self.resize = resize

        def get_act():
            if act == 'relu':
                return nn.ReLU(inplace=True)
            elif act == 'softplus':
                return nn.Softplus()
            elif act == 'elu':
                return nn.ELU()
            elif act == 'leakyrelu':
                return nn.LeakyReLU(0.2, inplace=True)

        if not resize:
            self.main = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
                nn.GroupNorm(8, out_channel),
                get_act(),
                nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1),
                nn.GroupNorm(8, out_channel)
            )
        else:
            self.main = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, stride=2, padding=1),
                nn.GroupNorm(8, out_channel),
                get_act(),
                nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1),
                nn.GroupNorm(8, out_channel)
            )
            self.residual = nn.Conv2d(in_channel, out_channel, 3, stride=2, padding=1)

        self.final_act = get_act()

    def forward(self, inputs):
        if not self.resize:
            h = self.main(inputs)
            h += inputs
        else:
            h = self.main(inputs)
            res = self.residual(inputs)
            h += res
        return self.final_act(h)


class DeconvResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, resize=False, act='relu'):
        super().__init__()
        self.resize = resize

        def get_act():
            if act == 'relu':
                return nn.ReLU(inplace=True)
            elif act == 'softplus':
                return nn.Softplus()
            elif act == 'elu':
                return nn.ELU()
            elif act == 'leakyrelu':
                return nn.LeakyReLU(0.2, True)

        if not resize:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, 3, stride=1, padding=1),
                nn.GroupNorm(8, out_channel),
                get_act(),
                nn.ConvTranspose2d(out_channel, out_channel, 3, stride=1, padding=1),
                nn.GroupNorm(8, out_channel)
            )
        else:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, 3, stride=1, padding=1),
                nn.GroupNorm(8, out_channel),
                get_act(),
                nn.ConvTranspose2d(out_channel, out_channel, 3, stride=2, padding=1, output_padding=1),
                nn.GroupNorm(8, out_channel)
            )
            self.residual = nn.ConvTranspose2d(in_channel, out_channel, 3, stride=2, padding=1, output_padding=1)

        self.final_act = get_act()

    def forward(self, inputs):
        if not self.resize:
            h = self.main(inputs)
            h += inputs
        else:
            h = self.main(inputs)
            res = self.residual(inputs)
            h += res
        return self.final_act(h)

# training:
#   batch_size: 128
#   n_epochs: 500000
#   n_iters: 50001
#   ngpu: 1
#   noise_std: 0.01
#   algo: "ssm"
#   snapshot_freq: 5000

# data:
#   dataset: "CIFAR10"
#   image_size: 32
#   channels: 3
#   logit_transform: false

# model:
#   n_particles: 1
#   lam: 10
#   z_dim: 100
#   nef: 32
#   ndf: 32

# optim:
#   weight_decay: 0.000
#   optimizer: "Adam"
#   lr: 0.001
#   beta1: 0.9

class Net(nn.Module): #ResScore
    def __init__(self):
        super().__init__()
        self.nef = 32 #config.model.nef
        self.ndf = 32 #config.model.ndf
        act = 'elu'
        self.convs = nn.Sequential(
            nn.Conv2d(3, self.nef, 3, 1, 1),
            ConvResBlock(self.nef, self.nef, act=act),
            ConvResBlock(self.nef, 2 * self.nef, resize=True, act=act),
            ConvResBlock(2 * self.nef, 2 * self.nef, act=act),
            # ConvResBlock(2 * self.nef, 2 * self.nef, resize=True, act=act),
            # ConvResBlock(2 * self.nef, 2 * self.nef, act=act),
            ConvResBlock(2 * self.nef, 4 * self.nef, resize=True, act=act),
            ConvResBlock(4 * self.nef, 4 * self.nef, act=act),
        )

        self.deconvs = nn.Sequential(
            # DeconvResBlock(2 * self.ndf, 2 * self.ndf, act=act),
            # DeconvResBlock(2 * self.ndf, 2 * self.ndf, resize=True, act=act),
            DeconvResBlock(4 * self.ndf, 4 * self.ndf, act=act),
            DeconvResBlock(4 * self.ndf, 2 * self.ndf, resize=True, act=act),
            DeconvResBlock(2 * self.ndf, 2 * self.ndf, act=act),
            DeconvResBlock(2 * self.ndf, self.ndf, resize=True, act=act),
            DeconvResBlock(self.ndf, self.ndf, act=act),
            nn.Conv2d(self.ndf, 3, 3, 1, 1)
        )

    def forward(self, x):
        x = 2 * x - 1.
        res = self.deconvs(self.convs(x))
        return res + 1e-10
        
    def reconstruction_loss(self, x_reconstructed, x):
        return nn.BCELoss(size_average=False)(x_reconstructed, x) / x.size(0)

    def kl_divergence_loss(self, mean, logvar):
        return ((mean**2 + logvar.exp() - 1 - logvar) / 2).mean()
    
    # sample
    def sample(self, num_of_sample, epoch):
        with torch.no_grad():
            
            z = torch.randn(num_of_sample, 2).cuda()
            sample = self.decoder(z).cuda()
            save_image(sample.view(num_of_sample, 3, 32, 32), './results/sample/NCSN_sample_' + str(epoch) + '.png')