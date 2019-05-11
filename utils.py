import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
from torch.utils.data import Dataset
from torch.autograd import grad, Variable
import os
from torch.utils.data.sampler import SubsetRandomSampler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Data:
    def __init__(self, path="../ssl_data_96/", batch_size=32, num_workers=2):
        self.sup_train_root_path = path+"supervised/train"
        self.unsup_train_root_path = path+"unsupervised"
        self.sup_val_root_path = path+"supervised/val"
        self.model = None
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.w = 96
        self.h = 96
        self.ch = 3

    # Load and union the supervisd and unsupervised training data, ignoring labels
    # Input: transform
    # Return a DataLoader.
    def load_train_data_mix(self, transform=None):
        print("Start loading mix")
        data_sup = torchvision.datasets.ImageFolder(root=self.sup_train_root_path, transform=transform)
        data_unsup = torchvision.datasets.ImageFolder(root=self.unsup_train_root_path, transform=transform)

        concat_dataset = torch.utils.data.ConcatDataset((data_sup, data_unsup))

        loader_unsup = torch.utils.data.DataLoader(concat_dataset, batch_size=self.batch_size, shuffle=True,
                                                   num_workers=self.num_workers)
        print("End loading mix")
        return loader_unsup

    # Load and union the supervised training data
    # Input: transform
    # Return a DataLoader.
    def load_train_data_sup(self, transform=None):
        print("Start load supurvised training data")
        data = torchvision.datasets.ImageFolder(root=self.sup_train_root_path, transform=transform)
        loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        print("End load supurvised training data")
        return loader

    # Load and union the supervised validation data
    # Input: transform
    # Return a DataLoader.
    def load_val_data(self, transform=None):
        print("Start load val")
        data = torchvision.datasets.ImageFolder(root=self.sup_val_root_path, transform=transform)
        loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        print("End loading val")
        return loader

    def main(self):
        train_data_sup = self.load_train_data_sup()
        print("Supervised training dataset: size =", len(train_data_sup))

    def laod_train_val_data(self, valid_size=0.2, transform=None):
        print("Start load supurvised training and val data")
        data_train = torchvision.datasets.ImageFolder(root=self.sup_train_root_path, transform=transform)
        train_loader1 = torch.utils.data.DataLoader(data_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        data_val = torchvision.datasets.ImageFolder(root=self.sup_val_root_path, transform=transform)
        num_val = len(data_val)
        indices = list(range(num_val))
        split = int(np.floor(valid_size * num_val))
        valid_idx1, valid_idx2 = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(valid_idx1)
        valid_sampler = SubsetRandomSampler(valid_idx2)

        train_loader2 = torch.utils.data.DataLoader(data_val, batch_size=self.batch_size, sampler=train_sampler, shuffle=False, num_workers=self.num_workers)
        val_loader = torch.utils.data.DataLoader(data_val, batch_size=self.batch_size, sampler=valid_sampler, shuffle=False, num_workers=self.num_workers)

        print("End load supurvised training and val data")
        print("train loader size: {}, val loader size: {}".format(
            len(train_loader1.dataset)+int(num_val*(1-valid_size)),
            int(num_val*valid_size)))
        return train_loader1, train_loader2, val_loader

class Discriminator(nn.Module):
    """
    General Discriminator for small dataset image GAN models
    """
    def __init__(self, image_shape=(3, 32, 32), dim_factor=64):
        """
        Inputs:
            image_shape (tuple of int): Shape of input images (C, H, W)
            dim_factor (int): Base factor to use for number of hidden
                              dimensions at each layer
        """
        super(Discriminator, self).__init__()
        C, H, W = image_shape
        assert H % 2**3 == 0, "Image height %i not compatible with architecture" % H
        H_out = int(H / 2**3)  # divide by 2^3 bc 3 convs with stride 2
        assert W % 2**3 == 0, "Image width %i not compatible with architecture" % W
        W_out = int(W / 2**3)  # divide by 2^3 bc 3 convs with stride 2

        self.pad = nn.ZeroPad2d(2)
        self.conv1 = nn.Conv2d(C, dim_factor, 5, stride=2)
        self.conv2 = nn.Conv2d(dim_factor, 2 * dim_factor, 5,
                               stride=2)
        self.conv3 = nn.Conv2d(2 * dim_factor, 4 * dim_factor, 5,
                               stride=2)
        self.linear = nn.Linear(4 * dim_factor * H_out * W_out, 1)

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Tensor): Batch of images to predict real or fake on
        Outputs:
            disc_out (PyTorch Vector): Vector of classification values for each
                                       input image (higher for more real, lower
                                       for more fake)
        """
        H1 = F.leaky_relu(self.conv1(self.pad(X)), negative_slope=0.2)
        H2 = F.leaky_relu(self.conv2(self.pad(H1)), negative_slope=0.2)
        H3 = F.leaky_relu(self.conv3(self.pad(H2)), negative_slope=0.2)
        H3_resh = H3.view(H3.size(0), -1)  # reshape for linear layer
        disc_out = self.linear(H3_resh)
        return disc_out


class Generator(nn.Module):
    """
    General Generator for small dataset image GAN models
    """
    def __init__(self, image_shape=(3, 32, 32), noise_dim=128, dim_factor=64):
        """
        Inputs:
            image_shape (tuple of int): Shape of output images (H, W, C)
            noise_dim (int): Number of dimensions for input noise
            dim_factor (int): Base factor to use for number of hidden
                              dimensions at each layer
        """
        super(Generator, self).__init__()
        C, H, W = image_shape
        assert H % 2**3 == 0, "Image height %i not compatible with architecture" % H
        self.H_init = int(H / 2**3)  # divide by 2^3 bc 3 deconvs with stride 0.5
        assert W % 2**3 == 0, "Image width %i not compatible with architecture" % W
        self.W_init = int(W / 2**3)  # divide by 2^3 bc 3 deconvs with stride 0.5

        self.linear = nn.Linear(noise_dim,
                                4 * dim_factor * self.H_init * self.W_init)
        self.deconv1 = nn.ConvTranspose2d(4 * dim_factor, 2 * dim_factor,
                                          4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(2 * dim_factor, dim_factor,
                                          4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(dim_factor, C,
                                          4, stride=2, padding=1)

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Tensor): Random noise input
        Outputs:
            img_out (PyTorch Tensor): Generated batch of images
        """
        H1 = F.relu(self.linear(X))
        H1_resh = H1.view(H1.size(0), -1, self.W_init, self.H_init)
        H2 = F.relu(self.deconv1(H1_resh))
        H3 = F.relu(self.deconv2(H2))
        img_out = torch.tanh(self.deconv3(H3))
        return img_out


def wgan_generator_loss(gen_noise, gen_net, disc_net):
    """
    Generator loss for Wasserstein GAN (same for WGAN-GP)

    Inputs:
        gen_noise (PyTorch Tensor): Noise to feed through generator
        gen_net (PyTorch Module): Network to generate images from noise
        disc_net (PyTorch Module): Network to determine whether images are real
                                   or fake
    Outputs:
        loss (PyTorch scalar): Generator Loss
    """
    # draw noise
    gen_noise.data.normal_()
    # get generated data
    gen_data = gen_net(gen_noise)
    # feed data through discriminator
    disc_out = disc_net(gen_data)
    # get loss
    loss = -disc_out.mean()
    return loss


def wgan_gp_discriminator_loss(gen_noise, real_data, gen_net, disc_net, lmbda,
                               gp_alpha):
    """
    Discriminator loss with gradient penalty for Wasserstein GAN (WGAN-GP)

    Inputs:
        gen_noise (PyTorch Tensor): Noise to feed through generator
        real_data (PyTorch Tensor): Noise to feed through generator
        gen_net (PyTorch Module): Network to generate images from noise
        disc_net (PyTorch Module): Network to determine whether images are real
                                   or fake
        lmbda (float): Hyperparameter for weighting gradient penalty
        gp_alpha (PyTorch Tensor): Values to use to randomly interpolate
                                   between real and fake data for GP penalty
    Outputs:
        loss (PyTorch scalar): Discriminator Loss
    """
    # draw noise
    gen_noise.data.normal_()
    # get generated data
    gen_data = gen_net(gen_noise)
    # feed data through discriminator
    disc_out_gen = disc_net(gen_data)
    disc_out_real = disc_net(real_data)
    # get loss (w/o GP)
    loss = disc_out_gen.mean() - disc_out_real.mean()
    # draw interpolation values
    gp_alpha.uniform_()
    # interpolate between real and generated data
    interpolates = gp_alpha * real_data.data + (1 - gp_alpha) * gen_data.data
    interpolates = Variable(interpolates, requires_grad=True)
    # feed interpolates through discriminator
    disc_out_interp = disc_net(interpolates)
    # get gradients of discriminator output with respect to input
    gradients = grad(outputs=disc_out_interp.sum(), inputs=interpolates,
                     create_graph=True)[0]
    # calculate gradient penalty
    grad_pen = ((gradients.contiguous().view(gradients.size(0), -1).norm(
        2, dim=1) - 1)**2).mean()
    # add gradient penalty to loss
    loss += lmbda * grad_pen
    return loss


def enable_gradients(net):
    for p in net.parameters():
        p.requires_grad = True


def disable_gradients(net):
    for p in net.parameters():
        p.requires_grad = False
