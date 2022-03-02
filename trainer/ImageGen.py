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


class Trainer():
    def __init__(self, args, model, train_data, valid_data, test_data, log_file = None):
        super(Trainer, self).__init__()
        self.model = model.to(args.device)
        self.device = args.device
        self.log_file = log_file
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

    def train(self):
        self.model.train()
        
        best_FID = 0
        self.best_model = None
        
        print("Training ...")

        for epoch in range(self.args.epochs):
            # self.model.train()
            time1 = time()
            Total_loss = 0
            for step, batch in enumerate(self.train_data):
                batch = [t.to(self.device) for t in batch]
                x_real, _ = batch
                x_reconstructed, mean, logvar = self.model(x_real)
                loss = self.model.loss_function(x_reconstructed, x_real, mean, logvar)
                # print(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                Total_loss += loss.item()
                # print(loss.item())
                
            time2 = time()
            self.total_training_time += (time2 - time1)
            # Total_loss = Total_loss/len(self.train_data.dataset)
            
            Total_loss = Total_loss/100

            # print('hehe')
            if epoch % 1 == 0 or epoch == self.args.epochs - 1:
                fid_score, inception_score, test_loss = self.eval(self.model, self.valid_data)
                if fid_score > best_FID:
                    best_FID = fid_score
                    self.best_model = self.get_model(self.model)
                    
                self.model.sample(64, epoch)
                    
                print("Epoch {:3d} |Train_loss = {:.3f}, Test_loss = {:.3f}, FID = {:.3f}, Inception = {:.3f} || ".format(epoch + 1,Total_loss, test_loss, fid_score, inception_score), end= "\n")


        print("Training time: {}s".format(self.total_training_time))
        print("Epoch time: {}s".format(self.total_training_time/self.args.epochs))
        self.set_model_(self.model, self.best_model) # set back to the best model found

    def eval(self, model, dataloader):
        model.eval()
        test_loss= 0
        fid_score, inception_score = 0, 0
        with torch.no_grad():
            for x_real, _ in dataloader:
                x_real = x_real.cuda()
                recon, mu, log_var = model(x_real)
                
                # sum up batch loss
                test_loss += model.loss_function(recon, x_real, mu, log_var).item()
            
        # test_loss /= len(dataloader.dataset)
        test_loss = test_loss/100
        

        return fid_score, inception_score, test_loss

    

    @staticmethod
    def set_model_(model, state_dict):
        model.load_state_dict(deepcopy(state_dict))
        return

    @staticmethod
    def get_model(model):
        return deepcopy(model.state_dict())

