### From https://github.com/SashaMalysheva/Pytorch-VAE/blob/master/model.py

import torch
from torch.autograd import Variable
from torch import nn
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from torchvision.utils import save_image
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, label='mnist', image_size=28, channel_num=1, z_size=2, args = None):
    #     pass
    # def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(Net, self).__init__()
        x_dim = image_size*image_size*channel_num
        h_dim1 = 512
        h_dim2 = 256
        z_dim=z_size
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
        self.args = args
        
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        h = F.sigmoid(self.fc6(h))
        return h + 1e-12
    
    def forward(self, x):
        # print(type(x))
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var
        
        
    def loss_function(self, recon_x, x, mu, log_var):
        # print(recon_x.max(), x.max())
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='mean')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        # print(BCE)
        # print(KLD) 
        return BCE + KLD
        
        
    def sample(self, num_of_sample, epoch):
        with torch.no_grad():
            
            z = torch.randn(num_of_sample, 2).cuda()
            sample = self.decoder(z).cuda()
            save_image(sample.view(num_of_sample, 1, 28, 28), self.args.img_file + '_' + str(epoch) + '.png')