import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import math
import torch.nn.init as init


class Net(nn.Module):  
    def __init__(self, num_classes, input_dim = 200, hidden_linear_dim= 256, rnn_hidden_dim= 256, droprate= 0.5, num_layers= 1):
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.hidden_linear_dim = hidden_linear_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_layers = num_layers
        # First linear layer
        self.fc1 = nn.Linear(self.input_dim, self.hidden_linear_dim)

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.hidden_linear_dim, self.rnn_hidden_dim, self.num_layers, batch_first=True, bidirectional=False)

        self.drop = nn.Dropout(droprate)
        # Define the output layer
        self.fc2 = nn.Linear(self.rnn_hidden_dim, num_classes)
        self.init_weight()

        self.relu = nn.ReLU()

    def init_weight(self):
        self.fc1.weight.data.normal_(0, 1/ math.sqrt(self.input_dim))
        self.fc1.bias.data.uniform_(0, 0)
        self.fc2.weight.data.normal_(0, 1/ math.sqrt(2*self.rnn_hidden_dim))
        self.fc2.bias.data.uniform_(0, 0)

    def forward(self, input):
        x = self.fc1(input)
        x = self.relu(x)
        x, hidden = self.lstm(x)
        logits = self.fc2(hidden[0].squeeze(0))
        return logits
