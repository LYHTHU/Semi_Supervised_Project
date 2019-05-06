from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from SemiSupervised import SemiSupervised

class Linear_Model(nn.Module):
    def __init__(self, x_dim=3*96*96, y_dim=3*96*96, encoder_dim=None, decoder_dim=None, latent_dim = 20):
        super(Linear_Model, self).__init__()
        self.encoder_dim = []
        self.decoder_dim = []

        # Architecture
        # TODO
        # encoder_dim is a list of dimension
        if encoder_dim is None:
            self.encoder_dim = [x_dim] + [4000, 200]
        else:
            self.encoder_dim = [x_dim] + encoder_dim
        
        if decoder_dim is None:
            self.decoder_dim = [400, 6000]
        else:
            self.decoder_dim = decoder_dim
            
        #encoder
        nets_en = [*self.encoder_dim]
        linear_layers_en = [nn.Linear(nets_en[i-1], nets_en[i]) for i in range(1, len(nets_en))]
        self.encoder_hidden = nn.ModuleList(linear_layers_en)
        
        # reparametrization for latent var
        in_features = nets_en[-1]
        out_features = latent_dim
        self.mu = nn.Linear(in_features, out_features)
        self.log_var = nn.Linear(in_features, out_features)
        
        #decode
        nets_de = [latent_dim, *self.decoder_dim]
        linear_layers_de = [nn.Linear(nets_de[i-1], nets_de[i]) for i in range(1, len(nets_de))]
        
        self.decoder_hidden = nn.ModuleList(linear_layers_de)
        self.reconstruction = nn.Linear(nets_de[-1], y_dim)

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
    
    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        #eps = torch.Tensor(torch.randn(mu.size()))
        #print(eps.size())
        #print(std.size())
        return mu + eps*std

    def forward(self, x):
        # TODO
        x = x.view(-1, 3*96*96)
        #data = x
        for layer in self.encoder_hidden:
            x = F.relu(layer(x))
        
        print(x)
        # reparametrization
        mu = self.mu(x)
        #print("mean is {}".format(mu))
        log_var = self.log_var(x)
        #print("variance is {}".format(log_var))
        
        z = self.reparametrize(mu, log_var)
        #print(z.size())
        
        for i, layer in enumerate(self.decoder_hidden):
            data = z
            z = layer(z)
            if (z != z).any() == 1:
                print(data)
                print(z)
                print(self.decoder_hidden[i])
                print("Not Relu")
                break
            z = F.relu(z)
            if (z != z).any() == 1:
                print(z)
                print(self.decoder_hidden[i])
                print("Relu")
                break
     
        
        z = self.reconstruction(z)
        
        return torch.sigmoid(z), mu, log_var