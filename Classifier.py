from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from Data import Data
import math
import time
from VaeModel import VAE

parser = argparse.ArgumentParser(description='Classifier')
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
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")
if args.cuda:
    print("Using GPU")

# kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}


data = Data(batch_size=args.batch_size)
train_loader = data.load_train_data_sup(transform=transforms.ToTensor())
test_loader = data.load_val_data(transform=transforms.ToTensor())


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.vae = torch.load("vae_best.pth")
        self.n_latent = self.vae.n_latent
        self.n_in = data.w * data.h * data.ch
        # self.n_latent = 200
        self.n_class = 1000
        self.fc1 = nn.Linear(self.n_in, self.n_latent)

        self.cls = nn.Sequential(
            nn.Linear(self.n_latent, 4000),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=4000),
            nn.Linear(4000, self.n_class),
            nn.Sigmoid()
        )

    def forward(self, x):
        mu, logvar = self.vae.encode(x)
        x = self.vae.reparameterize(mu, logvar)
        # x = F.relu(self.fc1(x.view(-1, self.n_in)))
        x = x.view(-1, self.n_latent)
        x = self.cls(x)
        return x


def train(model, device, train_loader, optimizer, epoch, log_interval=100):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        # for p in model.parameters():
        #     p.data.add_(-20, p.grad.data)

        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    return train_loss


def model_test(model, device, test_loader, log_interval=100):
    model.eval()
    test_loss = 0
    num_correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target, reduction="sum")
            test_loss += loss.item()
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            num_correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

            if batch_idx % log_interval == 0:
                print('Test : [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

    # Compute the average test_loss
    avg_test_loss = test_loss / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(avg_test_loss, num_correct, len(test_loader.dataset), 100. * num_correct / len(test_loader.dataset)))


if __name__ == '__main__':
    model = Classifier().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
    last_loss = math.inf
    params_path = "classifier.pth"
    for epoch in range(1, 10 + 1):
        start_time = time.time()
        # Train model
        train_loss = train(model, device, train_loader, optimizer, epoch)
        if train_loss < last_loss:
            last_loss = train_loss
            torch.save(model, params_path)
        end_time = time.time()
        print("Epoch:{}, running {} seconds", epoch, end_time-start_time)
        # Test model
    model = torch.load(params_path)
    model_test(model, device, test_loader)
