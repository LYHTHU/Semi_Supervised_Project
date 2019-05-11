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

class Conv_Model(nn.Module):
    def __init__(self, y_dim=3*96*96, decoder_dim=None, latent_dim=256, pretrained=False):
        super(Conv_Model, self).__init__()
        self.dim_factor = 16
        self.decoder_dim = []

        #encoder
        #nets_en = [*self.encoder_dim]
        # input parameters: [[c_in, c_out, kernel_size, stride]]
        self.layer1 = nn.Sequential(
            nn.ZeroPad2d(2),
            nn.Conv2d(3, self.dim_factor, kernel_size=5, stride=2),
            nn.ReLU()) # 16* 48 * 48

        self.layer2 = nn.Sequential(
            nn.ZeroPad2d(2),
            nn.Conv2d(self.dim_factor, self.dim_factor, kernel_size=5, stride=2), # 32 * 42 * 42
            nn.ReLU()) # 16 * 24 * 24

        self.layer3 = nn.Sequential(
            nn.ZeroPad2d(2),
            nn.Conv2d(self.dim_factor, self.dim_factor*2, kernel_size=5, stride=2), # 32 * 42 * 42
            nn.ReLU()) # 32 * 12 * 12

        self.encoder_hidden = nn.ModuleList([self.layer1, self.layer2, self.layer3])
        #self.fc1 = nn.Linear(32 * 6 * 6, 2*latent_dim)
        # reparametrization for latent var
        in_features = self.dim_factor * 2 * 12 * 12
        out_features = latent_dim

        self.mu = nn.Linear(in_features, latent_dim)
        self.log_var = nn.Linear(in_features, latent_dim)

        #decode
        #nets_de = [self.encoder_dim[-1], *self.decoder_dim]
        self.delinear = nn.Linear(latent_dim, in_features)
        self.deconv1 = nn.ConvTranspose2d(self.dim_factor*2, self.dim_factor,
                                          4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(self.dim_factor, self.dim_factor,
                                          4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(self.dim_factor, 3,
                                          4, stride=2, padding=1)

        self.decoder_hidden = nn.ModuleList([self.deconv1, self.deconv2])
        #self.reconstruction = nn.Linear(self.decoder_dim[-1], y_dim)

        # Load pre-trained model
        if pretrained:
            self.load_weights('/scratch/hl3420/conv_encoder11_weights.pt', cuda=torch.cuda.is_available())

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

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

    def encode(self, x):
        for i, layer in enumerate(self.encoder_hidden):
            #print(i)
            x = layer(x)

        x = x.view(x.size(0), -1)
        #x = self.fc1(x)

        # reparametrization
        mu = self.mu(x)
        return mu

    def forward(self, x):
        # TODO
        for i, layer in enumerate(self.encoder_hidden):
            #print(i)
            x = layer(x)

        x = x.view(x.size(0), -1)
        #x = self.fc1(x)

        # reparametrization
        mu = self.mu(x)
        log_var = F.softplus(self.log_var(x))

        z = self.reparametrize(mu, log_var)

        z = F.relu(self.delinear(z))
        z = z.view(z.size(0), -1, 12, 12)
        for layer in self.decoder_hidden:
            z = F.relu(layer(z))

        z = self.deconv3(z)
        return torch.sigmoid(z), mu, log_var
