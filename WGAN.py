from __future__ import print_function
import torch
import argparse
import numpy as np
from os import makedirs
from os.path import join, exists
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import *
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import math
import time


class WGAN(nn.Module):
    def __init__(self, data, noise_dim, dim_factor, model_path=None):
        super(WGAN, self).__init__()
        self.data = data
        self.generator = Generator(image_shape=(data.ch, data.h, data.w),
                                   noise_dim=noise_dim, dim_factor=dim_factor)
        self.discriminator = Discriminator(image_shape=(data.ch, data.h, data.w),
                                           dim_factor=dim_factor)
        if model_path:
            self.load_weights(model_path)

    def load_weights(self, model_path, cuda=True):
        pretrained_model = torch.load(f=model_path, map_location="cuda" if cuda else "cpu")
        # Load pre-trained weights in current model
        with torch.no_grad():
            self.discriminator.load_state_dict(pretrained_model, strict=True)

        # Debug loading
        print('Parameters found in pretrained model:')
        pretrained_layers = pretrained_model.keys()
        for l in pretrained_layers:
            print('\t' + l)
        print('')

        for name, module in self.discriminator.state_dict().items():
            if name in pretrained_layers:
                assert torch.equal(pretrained_model[name].cpu(), module.cpu())
                print('{} have been loaded correctly in current model.'.format(name))
            else:
                raise ValueError("state_dict() keys do not match")

    def save(self, model_path):
        torch.save(self.discriminator.state_dict(), model_path)

    def forward(self, x):
        return self.discriminator.forward(x)

def unsup_train(model, device, learning_rate, batch_size, noise_dim, epochs, K, lmbda, params_path):
    data_loader = model.data.load_train_data_mix(transform=transforms.ToTensor())
    gen_optimizer = optim.Adam(model.generator.parameters(), lr=learning_rate,
                               betas=(0.5, 0.9))
    disc_optimizer = optim.Adam(model.discriminator.parameters(), lr=learning_rate,
                                betas=(0.5, 0.9))
    # create tensors for input to algorithm
    gen_noise_tensor = torch.FloatTensor(batch_size, noise_dim)
    gp_alpha_tensor = torch.FloatTensor(batch_size, 1, 1, 1)
    gen_noise_tensor = gen_noise_tensor.to(device)
    gp_alpha_tensor = gp_alpha_tensor.to(device)
    # wrap noise as variable so we can backprop through the graph
    gen_noise_var = Variable(gen_noise_tensor, requires_grad=False).to(device)
    # calculate batches per epoch
    bpe = len(data_loader.dataset) // batch_size
    print("total batch number per epoch: %i" % (bpe))
    # create lists to store training loss
    gen_loss = []
    disc_loss = []
    last_loss = math.inf
    for i in range(1, epochs + 1):
        time_epoch = time.monotonic()
        print("-> Entering epoch %i out of %i" % (i, epochs))
        # iterate over data
        for batch_idx, (data, _) in enumerate(data_loader):
            if (batch_idx+1) % 500 == 0:
                print("process: {:.2f}%, batch {} out of {}"\
                    .format((batch_idx+1)*100/bpe, batch_idx+1, bpe))
            # wrap data in torch Tensor
            X_tensor = torch.Tensor(data).to(device)
            X_var = Variable(X_tensor, requires_grad=False).to(device)
            if (batch_idx % K) == (K - 1):
                #print("train generator.")
                # train generator
                enable_gradients(model.generator)  # enable gradients for gen net
                disable_gradients(model.discriminator)  # saves computation on backprop
                model.generator.zero_grad()
                loss = wgan_generator_loss(gen_noise_var, model.generator, model.discriminator)
                loss.backward()
                gen_optimizer.step()
                # append loss to list
                gen_loss.append(loss.data.cpu().numpy())
            # train discriminator
            enable_gradients(model.discriminator)  # enable gradients for disc net
            disable_gradients(model.generator)  # saves computation on backprop
            model.discriminator.zero_grad()
            loss = wgan_gp_discriminator_loss(gen_noise_var, X_var, model.generator,
                                              model.discriminator, lmbda, gp_alpha_tensor)
            loss.backward()
            disc_optimizer.step()
            # append loss to list
            disc_loss.append(loss.data.cpu().numpy())

        # calculate and print mean discriminator loss for past epoch
        print("Epoch time: {:.2f}s".format(time.monotonic()-time_epoch))
        mean_loss = np.array(disc_loss[-bpe:]).mean()
        print("Mean discriminator loss over epoch: %.2f" % mean_loss)
        if (mean_loss < last_loss):
            last_loss = mean_loss
        model.save(params_path)
    np.save('gen_loss', gen_loss)
    np.save('disc_loss', disc_loss)

class classifier(nn.Module):
    def __init__(self, image_shape=(3, 96, 96), dim_factor=64, model_path=None):
        super(classifier, self).__init__()
        C, H, W = image_shape
        assert H % 2**5 == 0, "Image height %i not compatible with architecture" % H
        self.H_out = int(H / 2**3)  # divide by 2^3 bc 3 convs with stride 2
        assert W % 2**5 == 0, "Image width %i not compatible with architecture" % W
        self.W_out = int(W / 2**3)  # divide by 2^3 bc 3 convs with stride 2
        self.dim_factor = dim_factor

        self.pad = nn.ZeroPad2d(2)
        self.conv1 = nn.Conv2d(C, dim_factor, 5, stride=2)
        self.conv2 = nn.Conv2d(dim_factor, 2 * dim_factor, 5,
                               stride=2)
        self.conv3 = nn.Conv2d(2 * dim_factor, 4 * dim_factor, 5,
                               stride=2)
        self.linear = nn.Linear(4 * dim_factor * self.H_out * self.W_out, 1)

        if model_path:
            self.load_WGAN(model_path)

    def load_WGAN(self, model_path, cuda=True):
        pretrained_model = torch.load(f=model_path, map_location="cuda" if cuda else "cpu")
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

    def reconstruct(self, hidden_dim=2000, class_num=1000):
        """
        structure 1
        """
        self.H_out = int(self.H_out / 2)
        self.W_out = int(self.W_out / 2)
        self.conv4 = nn.Conv2d(4 * self.dim_factor, 2 * self.dim_factor, 5,
                               stride=2)
        #self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear = nn.Linear(2 * self.dim_factor * self.H_out * self.W_out, class_num)
        #self.output = nn.Linear(hidden_dim, hidden_dim, class_num)
        print("reconstruction finished")

    def forward(self, X):
        H1 = F.leaky_relu(self.conv1(self.pad(X)), negative_slope=0.2)
        H2 = F.leaky_relu(self.conv2(self.pad(H1)), negative_slope=0.2)
        H3 = F.leaky_relu(self.conv3(self.pad(H2)), negative_slope=0.2)
        """
        structure 1
        """
        H4 = F.leaky_relu(self.conv4(self.pad(H3)), negative_slope=0.2)
        #H5 = self.maxpool(H4)
        H4_resh = H4.view(H4.size(0), -1)  # reshape for linear layer
        #hidden = F.relu(self.linear(H4_resh))
        out = self.linear(H4_resh)
        return out

    def partial_nograd(self, layer_num):
        print("set the first {} layer to no grad".format(layer_num))
        cntr = 0
        for child in self.children():
            cntr += 1
            if cntr <= layer_num:
                print(child)
                for param in child.parameters():
                    param.requires_grad = False

def sup_train(model, train_loader, val_loader, device, learning_rate, batch_size, epochs, params_path):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.9))
    bpe = int(len(train_loader.dataset) // batch_size)
    criterion = nn.CrossEntropyLoss().to(device=device)
    print("total batch number per epoch: %i" % (bpe))
    # create lists to store training loss
    loss_list = []
    epochs2 = 0
    last_loss = math.inf

    model.partial_nograd(4)
    print("training with partial frozen.")
    for i in range(1, epochs + 1):
        model.train()
        time_epoch = time.monotonic()
        print("-> Entering epoch %i out of %i" % (i, epochs+epochs2))
        # iterate over data
        for batch_idx, (X, target) in enumerate(train_loader):
            if (batch_idx+1) % 100 == 0:
                print("training process: {:.2f}%, batch {} out of {}"\
                       .format((batch_idx+1)*100/bpe, batch_idx+1, bpe))
            X = X.to(device)
            target = target.to(device)
            model.zero_grad()
            pred = model.forward(X)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            # append loss to list
            loss_list.append(loss.data.cpu().numpy())

        # calculate and print mean discriminator loss for past epoch
        print("Epoch time: {:.2f}s".format(time.monotonic()-time_epoch))
        mean_loss = np.array(loss_list[-bpe:]).mean()
        print("cross entropy loss over epoch: %.2f" % mean_loss)
        if (mean_loss < last_loss):
            last_loss = mean_loss
            torch.save(model.state_dict(), params_path)
        #print("Test on training")
        #test(model, device, train_loader)
        print("Test on val")
        test(model, device, val_loader)

    enable_gradients(model)
    print("training with no frozen.")
    for i in range(1, epochs2+1):
        model.train()
        time_epoch = time.monotonic()
        print("-> Entering epoch %i out of %i" % (i+epochs, epochs+epochs2))
        # iterate over data
        for batch_idx, (X, target) in enumerate(train_loader):
            if (batch_idx+1) % 100 == 0:
                print("training process: {:.2f}%, batch {} out of {}"\
                       .format((batch_idx+1)*100/bpe1, batch_idx+1, bpe))
            X = X.to(device)
            target = target.to(device)
            model.zero_grad()
            pred = model.forward(X)
            if (batch_idx+1) % 100 == 0:
                print(pred.data.cpu())
                print(target.data.cpu())
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            # append loss to list
            loss_list.append(loss.data.cpu().numpy())

        # calculate and print mean discriminator loss for past epoch
        print("Epoch time: {:.2f}s".format(time.monotonic()-time_epoch))
        mean_loss = np.array(loss_list[-(bpe1+bpe2):]).mean()
        print("cross entropy loss over epoch: %.2f" % mean_loss)
        if (mean_loss < last_loss):
            last_loss = mean_loss
            torch.save(model.state_dict(), params_path)
        #print("Test on training")
        #test(model, device, train_loader)
        print("Test on val")
        test(model, device, val_loader)

def test(model, device, test_loader):
    time_test = time.monotonic()
    criterion = nn.CrossEntropyLoss().to(device=device)
    # Set model to evaluation mode
    model.eval()
    # Variable for the total loss
    test_loss = 0
    # Counter for the correct predictions
    num_correct = 0
    # define negative log likelihood loss

    with torch.no_grad():
        # Loop through data points
        i = 0
        for data, target in test_loader:
            # Send data to device
            data = data.to(device)
            target = target.to(device)
            # Pass data through model
            pred = model.forward(data)
            if i == 0:
                i = 1
                #print(pred.data.cpu())
            loss = criterion(pred, target)
            test_loss += loss.data.cpu().numpy()

            # Get predictions from the model for each data point
            pred_label = pred.data.max(1, keepdim=True)[1]

            # Add number of correct predictions to total num_correct
            num_correct += pred_label.eq(target.data.view_as(pred_label)).cpu().sum().item()

    # Compute the average test_loss
    num = len(test_loader.dataset)
    avg_test_loss = test_loss / num
    print("Test time: {:.2f}s".format(time.monotonic()-time_test))
    # Print loss (uncomment lines below once implemented)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_test_loss, num_correct, num,
        100. * num_correct / num))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='On WGAN')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument("--noise_dim",
                        default=128, type=int,
                        help="Noise dim for generator")
    parser.add_argument("--dim_factor",
                        default=128, type=int,
                        help="Dimension factor to use for hidden layers")
    parser.add_argument("--K",
                        default=3, type=int,
                        help="Iterations of discriminator per generator")
    parser.add_argument("--lmbda",
                        default=10., type=float,
                        help="Gradient penalty hyperparameter")
    parser.add_argument("--learning_rate",
                        default=1e-4, type=float,
                        help="Learning rate of the model")

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")
    if args.cuda:
        print("Using GPU")

    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}

    data = Data(path="/scratch/hl3420/ssl_data_96/", batch_size=args.batch_size)
    train_loader = data.load_train_data_sup(transform=transforms.ToTensor())
    val_loader = data.load_val_data(transform=transforms.ToTensor())

    model = classifier(model_path="wgan_best6.pth", dim_factor=args.dim_factor)
    model.reconstruct()
    model.to(device)
    sup_train(model=model, train_loader=train_loader, val_loader=val_loader,
              device=device, learning_rate=args.learning_rate, batch_size=args.batch_size,
              epochs=args.epochs, params_path="WGCL-v1.pth")
    #test(model, device, test_loader)
    #wgan = WGAN(data, noise_dim=args.noise_dim, dim_factor=args.dim_factor).to(device)
    #wgan.load_weights("wgan_best.pth")
    #unsup_train(model=wgan, device=device, learning_rate=args.learning_rate,
    #            batch_size=args.batch_size, noise_dim=args.noise_dim,
    #            epochs=args.epochs, K=args.K, lmbda=args.lmbda,
    #            params_path='wgan_best-4.pth')
