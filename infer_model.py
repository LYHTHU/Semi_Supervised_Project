from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision.models as models

import numpy as np

from SemiSupervised import SemiSupervised
from modelVae_linear import Linear_Model

class Infer_model(nn.Module):
    def __init__(self, ftm_path, model_path):
        self.ft = Linear_Model()
        self.ft.eval()
        self.layer1 = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=5, stride=1), # 16* 92 * 92
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # 16 * 46 * 46
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1), # 32 * 42 * 42
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3, padding = 0)) # 32 * 14 * 14
        
        self.drop_out = nn.Dropout()
        
        self.fc1 = nn.Linear(14 * 14 * 32, 1000)
        
        self.classification = nn.ModuleList([self.layer1, self.layer2, self.drop_out, self.fc1])
        
    def forward(self, x):
        add_feature = self.ft(x)
        input_size = [*x.size()]
        dim = np.prod(input_size)
        
        # resize the input data
        # input data size is batch_size * 6 * 96 * 96
        input_size[1] = 2*input_size[1]
        new_data = torch.cat([x.view(-1, dim), add_feature.view(-1, dim)]).view(*input_size)
        x = new_data
        
        for layer in self.classification:
            x = layer(x)
        
        return x