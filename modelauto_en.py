from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import numpy as np

from SemiSupervised import SemiSupervised

class AutoEn_Model(nn.Module):
    def __init__(self, y_dim=3*96*96, decoder_dim=None, latent_dim=20):
        super(AutoEn_Model, self).__init__()
        self.decoder_dim = []

        # Architecture
        # TODO
        # encoder_dim is a list of dimension
        
        if decoder_dim is None:
            self.decoder_dim = [latent_dim] + [200, 4000]
        else:
            self.decoder_dim = [latent_dim] + decoder_dim
            
        #encoder
        #nets_en = [*self.encoder_dim]
        # input parameters: [[c_in, c_out, kernel_size, stride]]
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1), # 16* 92 * 92
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # 16 * 46 * 46
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1), # 32 * 42 * 42
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3, padding = 0)) # 32 * 14 * 14
        
        #self.drop_out = nn.Dropout()
        
        self.fc1 = nn.Linear(14 * 14 * 32, 2*latent_dim)
        
        self.encoder_hidden = nn.ModuleList([self.layer1, self.layer2])
        
        # reparametrization for latent var
        in_features = 2*latent_dim
        out_features = latent_dim
        
        self.mu = nn.Linear(in_features, out_features)
        
        #decode
        #nets_de = [self.encoder_dim[-1], *self.decoder_dim]
        linear_layers_de = [nn.Linear(self.decoder_dim[i-1], self.decoder_dim[i]) for i in range(1, len(self.decoder_dim))]
        
        self.decoder_hidden = nn.ModuleList(linear_layers_de)
        self.reconstruction = nn.Linear(self.decoder_dim[-1], y_dim)

        # Load pre-trained model
        # self.load_weights('weights.pth')

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
        for i, layer in enumerate(self.encoder_hidden):
            #print(i)
            x = layer(x)
        
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        
        # reparametrization
        z = self.mu(x)
        
        for layer in self.decoder_hidden:
            z = F.relu(layer(z))
            
        z = self.reconstruction(z)
        return torch.sigmoid(z)