from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

import numpy as np

from SemiSupervised import SemiSupervised

class Base_Model(nn.Module):
    def __init__(self, n_class = 1000):
        super(Base_Model, self).__init__()
        

        # Architecture
        # TODO
        # encoder_dim is a list of dimension
    
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1), # 16* 92 * 92
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # 16 * 46 * 46
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1), # 32 * 42 * 42
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3, padding = 0)) # 32 * 14 * 14
        
        self.drop_out = nn.Dropout()
        
        self.fc1 = nn.Linear(14 * 14 * 32, 3*n_class)
        self.fc2 = nn.Linear(3*n_class, n_class)
        
        self.encoder_hidden = nn.ModuleList([self.layer1, self.layer2, self.drop_out])

        # Load pre-trained model
        self.load_weights('./baseline_weights.pt', cuda=torch.cuda.is_available())

    def load_weights(self, pretrained_model_path, cuda=True):
        # Load pretrained model
        pretrained_model = torch.load(f=pretrained_model_path, map_location="cuda" if cuda else "cpu")

        # Load pre-trained weights in current model
        with torch.no_grad():
            self.load_state_dict(pretrained_model, strict=True)

        # Debug loading
        print('Parameters found in pretrained model:')
        pretrained_layers = pretrained_model.keys()
        for l in pretrained_layers:
            print('\t' + l)
        print('')

        for name, module in self.state_dict().items():
            if name in pretrained_layers:
                assert torch.equal(pretrained_model[name].cpu(), module.cpu())
                print('{} have been loaded correctly in current model.'.format(name))
            else:
                raise ValueError("state_dict() keys do not match")

    def forward(self, x):
        # TODO
        for layer in self.encoder_hidden:
            x = layer(x)
            
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        # reparametrization
        
        return x