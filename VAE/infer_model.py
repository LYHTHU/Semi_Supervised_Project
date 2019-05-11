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
from modelVae_convnet import Conv_Model

class Infer_model(nn.Module):
    def __init__(self, latent=256, pretrained=False, pretrained_model_path='/scratch/yc3329/infer_conv11.pt'):
        super(Infer_model, self).__init__()
        
        self.ft = Conv_Model(pretrained=True)
        self.ft.eval()
        self.latent = latent
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1), # 16* 92 * 92
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # 16 * 46 * 46
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1), # 32 * 42 * 42
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3, padding = 0)) # 64 * 14 * 14
        
        self.drop_out = nn.Dropout()
        
        self.fc1 = nn.Linear(14 * 14 * 64 + self.latent, 1000)
        
        self.classification = nn.ModuleList([self.layer1, self.layer2, self.drop_out])

        if pretrained:
            self.load_weights(pretrained_model_path, cuda=torch.cuda.is_available())
        
    def forward(self, x):
        add_feature = self.ft.encode(x)
        
        # resize the input data
        # input data size is batch_size * 6 * 96 * 96
        
        for layer in self.classification:
            x = layer(x)
            
        input_size = [*x.size()][1:]
        dim = np.prod(input_size)
           
        new_data = torch.cat((x.view(-1, dim), add_feature.view(-1, self.latent)), 1)
        #print(new_data.size())
        
        
        return self.fc1(new_data)
    
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
