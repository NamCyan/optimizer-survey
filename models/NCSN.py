#Souce https://github.com/ermongroup/ncsn/blob/master/models/scorenet.py

import torch.nn as nn
import functools
import torch
from torchvision.models import ResNet
import torch.nn.functional as F
from .pix2pix import init_net, UnetSkipConnectionBlock, get_norm_layer, init_weights, ResnetBlock, \
    UnetSkipConnectionBlockWithResNet


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
        return res
        
    def reconstruction_loss(self, x_reconstructed, x):
        return nn.BCELoss(size_average=False)(x_reconstructed, x) / x.size(0)

    def kl_divergence_loss(self, mean, logvar):
        return ((mean**2 + logvar.exp() - 1 - logvar) / 2).mean()
    
    # calculate frechet inception distance
    def calculate_fid(self, images1, images2):
        with torch.no_grad():
            act1 = self.encoder(images1).view(images1.shape[0], -1).cpu().numpy()
            # print(act1.shape)
            mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
            
            act2 = self.encoder(images2).view(images1.shape[0], -1).cpu().numpy()
            mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    
            # calculate sum squared difference between means
            ssdiff = numpy.sum((mu1 - mu2)**2.0)
            # calculate sqrt of product between cov
            covmean = sqrtm(sigma1.dot(sigma2))
            # check and correct imaginary numbers from sqrt
            if iscomplexobj(covmean):
                covmean = covmean.real
                # calculate score
            fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid
        
    def sample(self, size):
        z = Variable(
            torch.randn(size, self.z_size).cuda() if self._is_on_cuda() else
            torch.randn(size, self.z_size)
        )
        z_projected = self.project(z).view(
            -1, self.kernel_num,
            self.feature_size,
            self.feature_size,
        )
        return self.decoder(z_projected).data