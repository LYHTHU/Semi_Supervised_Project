from __future__ import print_function
from __future__ import division
import argparse
import os
import copy
import time

import numpy as np

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision.models as models

import torch.optim as optim
from torch.optim import lr_scheduler

from SemiSupervised import SemiSupervised
#from modelVae_linear import Linear_Model
from modelVae_convnet import Conv_Model
#from modelVae_convnet_plus import Conv_Model_plus

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

semi = SemiSupervised(batch_size=args.batch_size)
train_loader = semi.load_train_data_mix(transform=transforms.ToTensor())
test_loader = semi.load_val_data(transform=transforms.ToTensor())

def loss_function(recon_x, x, mu, logvar):
    #print(recon_x.size())
    #print(x.view(-1, 3*96*96).size())
    print(x.size(), recon_x.size())
    BCE = F.binary_cross_entropy(recon_x.view(-1, 3*96*96), x.view(-1, 3*96*96), reduction = 'sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def train_model(model, criterion, optimizer, save_path, num_epoch = 10):
    since = time.time()

    best_mode_wts = copy.deepcopy(model.state_dict())
    best_loss = -1.0

    for epoch in range(num_epoch):
        print("Epoch {}/{}".format(epoch, num_epoch-1))
        print('-'*10)
        for phase in ['train']:
            if phase == 'train':
                #scheduler.step()
                model.train()
            else:
                continue
                model.eval()

            run_loss = 0.0
            #run_correct = 0
            proc = 0

            for inputs, _ in train_loader:
                proc += args.batch_size

                inputs = inputs.to(device)
                #labels = labels.to(device)

                #zeros para
                optimizer.zero_grad()

                #forward
                #track history if and only if in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    recon_x, mu, logvar = model(inputs)
                    #print(recon_x.size())
                    #print(x.size())
                    #_, preds = torch.max(outputs, 1)
                    #print(outputs.size())
                    #print(preds)
                    #
                    loss = criterion(recon_x, inputs, mu, logvar)
                    # backwarda and optimize only in training
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                run_loss += loss.item()*inputs.size(0)
                #run_correct += torch.sum(labels.data == preds)


            epoch_loss = run_loss/len(train_loader)
            #epoch_acc = run_correct/dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))


            # deep copy the model
            if best_loss < 0:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), './'+save_path+str(epoch)+'.pt')

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), './'+save_path+str(epoch)+'.pt')

                #torch.save(model.state_dict(), save_path)

            print('{} epoch time: {:.4f}'.format(epoch, time.time() - since))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
    #print('Best Acc: {:.4f}'.format(best_acc))

    #model.load(best_model_wts)

    return best_mode_wts

if __name__ == "__main__":

    save_path = './conv_encoder_12.pt'
    check_path = 'conv_encoder_12_check'
    model = Conv_Model()
    model = model.to(device)

    criterion = loss_function

    #optimizer_ft = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
    optimizer_ft = optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.9))

    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size = 7, gamma = 0.1)

    model_best_weights = train_model(model, criterion, optimizer_ft, check_path, num_epoch = 10)
    torch.save(model_best_weights, save_path)
