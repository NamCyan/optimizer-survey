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

class Net(nn.Module):
    def __init__(self, label='mnist', image_size=28, channel_num=1, z_size=2):
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
        return F.sigmoid(self.fc6(h)) 
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var
        
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
        
    def loss_function(self, recon_x, x, mu, log_var):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD
        