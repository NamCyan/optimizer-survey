import torch.nn as nn
import torch
import numpy as np
import torch.nn.init as init
import math

class Net(nn.Module):
  def __init__(self, num_classes= 10):
        super(Net, self).__init__()
 
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.relu = nn.ReLU()
  
  def init_weight(self):
        self.fc1.weight.data.normal_(0, 1/ math.sqrt(16 * 5 * 5))
        self.fc1.bias.data.uniform_(0, 0)
        self.fc2.weight.data.normal_(0, 1/ math.sqrt(120))
        self.fc2.bias.data.uniform_(0, 0)
        self.fc3.weight.data.normal_(0, 1/ math.sqrt(84))
        self.fc3.bias.data.uniform_(0, 0)

  def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x)) 
        logits = self.fc3(x)
        return logits
