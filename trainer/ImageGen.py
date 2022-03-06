### From https://github.com/SashaMalysheva/Pytorch-VAE/blob/master/train.py

import math
import torch.optim as optim
import torch
import copy
from time import time
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from copy import deepcopy
from trainer.optimizer import get_optimizer
from utils.fid_score import calculate_fid
from utils.inception_score import calculate_is
from utils.sliced_sm import sliced_score_estimation_vr
from arguments import get_args
import math
import random
import numpy as np
import torch
import json
from time import time
import os, csv



class Trainer():
    def __init__(self, args, model, train_data, valid_data, test_data, log_file = None):
        super(Trainer, self).__init__()
        self.model = model.to(args.device)
        self.device = args.device
        self.log_file = args.log_file
        # print(self.log_file)
        self.args = args
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.T = len(train_data)*args.epochs
        self.optimizer = get_optimizer(args, self.model, self.T)
        self.total_training_time = 0
        
        self.fid = []
        self.inception = []
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones= milestones, gamma= 0.3)
        print(args)

    def train(self):
        self.model.train()
        self.total_valid_time = 0
        
        best_val_loss = 0
        self.best_model = None
        
        print("Training ...")

        for epoch in range(self.args.epochs):
            # self.model.train()
            time1 = time()
            Total_loss = 0
            for step, (x, _) in enumerate(self.train_data):
                x_real = x.cuda() #torch.sigmoid(x.cuda()) #
                
                if self.args.model_name == 'VAE':
                    x_reconstructed, mean, logvar = self.model(x_real)
                    loss = self.model.loss_function(x_reconstructed, x_real, mean, logvar)
                else:
                    sigma = 0.01
                    scaled_score = lambda x: self.model(x)
                    x_real = x_real + torch.randn_like(x_real) * sigma
                    loss, *_ = sliced_score_estimation_vr(scaled_score, x_real.detach(), n_particles=1)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                Total_loss += loss.item()
                # print(loss.item())
                
            time2 = time()
            self.total_training_time += (time2 - time1)
            Total_loss = Total_loss/len(self.train_data.dataset)
            
            # Total_loss = Total_loss/100

            # print('hehe')
            if epoch % 1 == 0 or epoch == self.args.epochs - 1:
                time3 = time()
                _, _, _, val_loss = self.eval(self.model, self.valid_data)
                fid_score, inception_score, is_std, test_loss = self.eval(self.model, self.test_data, test_mode =  True)
                
                if val_loss > best_val_loss:
                    best_val_loss = val_loss
                    self.best_model = self.get_model(self.model)
                    
                self.model.sample(64, epoch)
                time4 = time()
                self.total_valid_time += (time4 - time3)
                    
                print("Epoch {:3d} |Train_loss: {:.3f}, Val_loss: {:.3f}, Test_loss: {:.3f}, FID: {:.3f}, IS: {:.3f} ({:.3f})|| ".format(epoch, Total_loss, val_loss, test_loss, fid_score, inception_score, is_std), end= "\n")
                
                 # save result
                header = ['epoch', 'train_loss', 'valid_loss', 'test_loss', 'FID',  'IS', 'IS_std', 'time']
                # print(self.log_file)
                with open(self.log_file, "a") as f:
                    writer = csv.writer(f)
                    if epoch == 0:
                        writer.writerow(header)
                    writer.writerow([epoch, Total_loss, val_loss, test_loss, fid_score, inception_score, is_std, time2 - time1])


        print("Training time: {}s".format(self.total_training_time))
        print('Eval time: {}s'.format(self.total_valid_time))
        print("Epoch time: {}s".format((self.total_training_time + self.total_valid_time)/self.args.epochs))
        self.set_model_(self.model, self.best_model) # set back to the best model found

    def eval(self, model, dataloader, test_mode = False):
        model.eval()
        test_loss= 0
        fid_score, inception_score, is_std = 0, 0, 0
        x_fake = None
        X_real = None
        with torch.no_grad():
            for x_real, _ in dataloader:
                batch_size = x_real.shape[0]
                x_real =  x_real.cuda() #torch.sigmoid(x_real.cuda()) #
                # recon, mu, log_var = model(x_real)
                
                # sum up batch loss
                if self.args.model_name == 'VAE':
                    recon, mean, logvar = self.model(x_real)
                    loss = self.model.loss_function(recon, x_real, mean, logvar)
                else:
                    sigma = 0.01
                    scaled_score = lambda x: self.model(x)
                    x_real = x_real + torch.randn_like(x_real) * sigma
                    loss, *_ = sliced_score_estimation_vr(scaled_score, x_real.detach(), n_particles=1)
                    recon = self.model(x_real)
                test_loss += loss.item()
                    
                if test_mode: 
                    
                    if self.args.dataset == 'mnist':
                        inp_shape = (1, 28, 28)
                    else:
                        inp_shape = (3, 32, 32)
                        
                    if x_fake is None:
                        x_fake = recon.view(-1, inp_shape[0], inp_shape[1], inp_shape[2])
                    else:
                        x_fake = torch.cat((x_fake, recon.view(-1, inp_shape[0], inp_shape[1], inp_shape[2])))
                
                    if X_real is None:
                        X_real = x_real.view(-1, inp_shape[0], inp_shape[1], inp_shape[2])
                    else:
                        X_real = torch.cat((X_real, x_real.view(-1, inp_shape[0], inp_shape[1], inp_shape[2])))
                        # print(X_real.shape)
            

            if test_mode:
                fake_set = CustomDataset(x_fake)
                fake_loader = torch.utils.data.DataLoader(fake_set, batch_size = batch_size)
                real_set = CustomDataset(X_real)
                real_loader = torch.utils.data.DataLoader(real_set, batch_size = batch_size)
                
                fid_score, inception_score, is_std = self.FID_IS_score(image_loaders = [real_loader, fake_loader], device = self.device)
            
            test_loss /= len(dataloader.dataset)
        
        return fid_score, inception_score, is_std, test_loss

    
    def FID_IS_score(self, image_loaders, device, dims = 2048):
        FID = calculate_fid(image_loaders, device, dims)
        
        IS_mu, IS_std = calculate_is(image_loaders[1], device, resize=False, splits=20)
        return FID, IS_mu, IS_std

    @staticmethod
    def set_model_(model, state_dict):
        model.load_state_dict(deepcopy(state_dict))
        return

    @staticmethod
    def get_model(model):
        return deepcopy(model.state_dict())
        
    def logit_transform(self, image, lam=1e-6):
        image = lam + (1 - 2 * lam) * image
        return torch.log(image) - torch.log1p(-image)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, list_images, transform = None):
        self.list_image = list_images
        self.transform = transform
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        sample = self.list_image[idx]
        if self.transform:
            sample = self.transform(sample)
            
        return sample
        
    def __len__(self):
        return len(self.list_image)