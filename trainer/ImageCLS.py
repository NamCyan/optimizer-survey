import math
import torch.optim as optim
import torch
import copy
from time import time
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
from copy import deepcopy

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
        self.optimizer = self.get_optimizer()
        self.total_training_time = 0
        self.train_losses, self.train_accs = [], []
        self.valid_losses, self.valid_accs = [], []
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones= milestones, gamma= 0.3)

    def get_optimizer(self):
        if self.args.optim == "Adam":
            return torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        elif self.args.optim == "SGD":
            return torch.optim.SGD(self.model.parameters(), lr=self.args.lr)
        elif self.args.optim == "AggMo":
          betas = []
          for i in range(self.args.num_betas):
              beta = 1 - math.pow(0.1, i)
              betas.append(beta)
          optimizer = optim.AggMo(self.model.parameters(), lr=self.args.lr, betas= betas)
          return optimizer
        elif self.args.optim == "QHM":
            betas = self.args.betas
            optimizer = optim.QHM(self.model.parameters(), lr=self.args.lr, momentum=betas, nu=self.args.nu)
            return optimizer
        elif self.args.optim == "QHAdam":
            betas = [0.9, 0.999]
            optimizer = optim.QHAdam(self.model.parameters(), lr=self.args.lr, nu=self.args.nu, betas= betas)
            return optimizer
        else:
            raise Exception('Have not implement {} optimizer yet'.format(self.args.optim))

    def train(self):
        self.model.train()
        
        best_valid_loss = 1e8
        self.best_model = None
        
        print("Training ...")

        for epoch in range(self.args.epochs):
            self.model.train()
            time1 = time()
            for step, batch in enumerate(self.train_data):
                batch = [t.to(self.device) for t in batch]
                input, label = batch
                logits = self.model(input)
                loss = nn.CrossEntropyLoss()(logits, label)
                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()
            time2 = time()

            self.total_training_time += (time2 - time1)
            # scheduler.step()
            valid_loss, valid_acc, valid_macro_f1, valid_cls_report = self.eval(self.model, self.valid_data)
            train_loss, train_acc, train_macro_f1, train_cls_report = self.eval(self.model, self.train_data)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.valid_losses.append(valid_loss)
            self.valid_accs.append(valid_acc)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self.best_model = self.get_model(self.model)

            # print("Epoch {:3d} | Train: loss= {:.3f}, acc= {:.3f}% || Valid: loss= {:.3f}, acc= {:.3f}%, macro_f1= {:.3f}% || Best macro_f1: {:.3f}% || Time= {:.3f}s".format(epoch + 1, train_loss, train_acc, valid_loss, valid_acc*100, valid_macro_f1*100, best_valid_macro_f1*100, time2- time1))
            
            if epoch % 10 == 0 or epoch == self.args.epochs - 1:
                print("Epoch {:3d} | Train: loss= {:.3f}, acc= {:.3f}% || ".format(epoch + 1, train_loss, train_acc*100), end= "")
                print("Valid: loss= {:.3f}, acc= {:.3f}%, macro_f1= {:.3f}% || Best loss: {:.3f} || Time= {:.4f}s ||".format(valid_loss, valid_acc*100, valid_macro_f1*100, best_valid_loss, time2- time1), end= "\n")

        print("Training time: {}s".format(self.total_training_time))
        print("Epoch time: {}s".format(self.total_training_time/self.args.epochs))
        self.set_model_(self.model, self.best_model) # set back to the best model found

    def eval(self, model, data):
        self.model.eval()
        losses = []
        preds = []
        golds = []

        for step, batch in enumerate(data):
            batch = [t.to(self.device) for t in batch]
            input, label = batch
            logits = model(input)
            loss = nn.CrossEntropyLoss()(logits, label)

            output = F.log_softmax(logits, dim=1)
            _, pred = output.max(1)

            assert len(label) == len(pred)
            
            preds.extend(pred.cpu().numpy())
            golds.extend(label.cpu().numpy())
            losses.append(loss.item())

        loss = np.mean(losses)
        acc = accuracy_score(golds, preds)
        macro = f1_score(golds, preds, average='macro')
        report = classification_report(golds, preds)
        return loss, acc, macro, report

    @staticmethod
    def set_model_(model, state_dict):
        model.load_state_dict(deepcopy(state_dict))
        return

    @staticmethod
    def get_model(model):
        return deepcopy(model.state_dict())

