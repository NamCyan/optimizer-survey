import os
import csv
import math
import torch_optimizer as optim
import torch
import copy
from time import time
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
from copy import deepcopy
from trainer.optimizer import get_optimizer


class Trainer():
    def __init__(self, args, model, train_data, valid_data, test_data):
        super(Trainer, self).__init__()
        self.model = model.to(args.device)
        self.device = args.device
        self.log_file = args.log_file
        self.args = args
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.optimizer = get_optimizer(args, self.model)
        self.total_training_time = 0
        self.train_losses, self.train_accs = [], []
        self.valid_losses, self.valid_accs = [], []
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones= milestones, gamma= 0.3)

    def train(self):
        self.model.train()
        
        best_valid_loss = 1e8
        self.best_model = None
        
        print("Training ...")

        for epoch in range(self.args.epochs):
            train_loss = []
            train_preds = []
            train_golds = []

            self.model.train()
            time1 = time()
            for step, batch in enumerate(self.train_data):
                batch = [t.to(self.device) for t in batch]
                if "bert" in self.args.model_name.lower():
                    input_ids, attention_mask, token_type_ids, labels = batch
                    outputs = self.model(input_ids= input_ids, 
                                              attention_mask= attention_mask, 
                                              token_type_ids= token_type_ids, 
                                              labels= labels)
                    loss, logits = outputs.loss, outputs.logits
                else:
                    input, labels = batch
                    logits = self.model(input)
                    loss = nn.CrossEntropyLoss()(logits, labels)
                
                output = F.log_softmax(logits, dim=1)
                _, pred = output.max(1)

                train_loss.append(loss.item())
                train_preds.extend(pred.cpu().numpy())
                train_golds.extend(labels.cpu().numpy())

                loss.backward()

                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            time2 = time()

            self.total_training_time += (time2 - time1)
            # scheduler.step()
            valid_loss, valid_acc, valid_macro_f1, valid_cls_report = self.eval(self.model, self.valid_data)
            train_loss, train_acc, train_macro_f1 = np.mean(train_loss), accuracy_score(train_golds, train_preds), f1_score(train_golds, train_preds, average='macro')
            
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.valid_losses.append(valid_loss)
            self.valid_accs.append(valid_acc)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                self.best_model = self.get_model(self.model)
            
            if epoch % 10 == 0 or epoch == self.args.epochs - 1:
                print("Epoch {:3d} | Train: loss= {:.3f}, acc= {:.3f}% || ".format(epoch + 1, train_loss, train_acc*100), end= "")
                print("Valid: loss= {:.3f}, acc= {:.3f}%, macro_f1= {:.3f}% || Best loss: {:.3f} || Time= {:.4f}s ||".format(valid_loss, valid_acc*100, valid_macro_f1*100, best_valid_loss, time2- time1), end= "\n")

            # save result
            header = ['epoch', 'train_loss', 'train_acc', 'train_f1', 'valid_loss', 'valid_acc', 'valid_f1', 'test_loss', 'test_acc',  'test_f1', 'time']
            with open(self.log_file, "a") as f:
                writer = csv.writer(f)
                if epoch == 0:
                    writer.writerow(header)
                writer.writerow([epoch, 
                              train_loss, train_acc, train_macro_f1, 
                              valid_loss, valid_acc, valid_macro_f1,
                              None, None, None,
                              time2 - time1])

        print("Training time: {:.3f}s".format(self.total_training_time))
        print("Epoch time: {:.3f}s".format(self.total_training_time/self.args.epochs))
        self.set_model_(self.model, self.best_model) # set back to the best model found

    def eval(self, model, data):
        self.model.eval()
        losses = []
        preds = []
        golds = []

        for step, batch in enumerate(data):
            batch = [t.to(self.device) for t in batch]

            if "bert" in self.args.model_name.lower():
                input_ids, attention_mask, token_type_ids, labels = batch
                outputs = self.model(input_ids= input_ids, 
                                          attention_mask= attention_mask, 
                                          token_type_ids= token_type_ids, 
                                          labels= labels)
                loss, logits = outputs.loss, outputs.logits
            else:
                input, labels = batch
                logits = self.model(input)
                loss = nn.CrossEntropyLoss()(logits, labels)

            output = F.log_softmax(logits, dim=1)
            _, pred = output.max(1)

            assert len(label) == len(pred)
            
            preds.extend(pred.cpu().numpy())
            golds.extend(labels.cpu().numpy())
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

