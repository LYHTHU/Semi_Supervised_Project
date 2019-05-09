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
    def __init__(self, latent=256, pretrained=True, pretrained_model_path='../weights/infer_weights.pt'):
        super(Infer_model, self).__init__()
        
        self.ft = Conv_Model_plus(pretrained=True)
        self.ft.eval()
        self.latent = latent


        self.resnet = models.resnet18(pretrained = False)

        num_ftrs = self.resnet.fc.in_features

        self.resnet.fc = nn.Linear(num_ftrs, 1000 + self.latent)
        
        self.classification = nn.Linear(1000 + self.latent, 1000)

        if pretrained:
            self.load_weights(pretrained_model_path, cuda=torch.cuda.is_available())
        
    def forward(self, x):
        add_feature = self.ft.encode(x)
        
        # resize the input data
        #x = 
        x = self.resnet(x)
        
            
        input_size = [*x.size()][1:]
        dim = np.prod(input_size)
           
        new_data = torch.cat((x.view(-1, dim), add_feature.view(-1, self.latent)), 1)
        #print(new_data.size())
        
        
        return self.classification(new_data)
    
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