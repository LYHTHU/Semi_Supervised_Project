import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SemiSupervised:
    def __init__(self, batch_size=32, num_workers=2):
        self.sup_train_root_path = "../ssl_data_96/supervised/train"
        self.unsup_train_root_path = "../ssl_data_96/unsupervised"
        self.sup_val_root_path = "../ssl_data_96/supervised/val"
        self.model = None
        self.batch_size = batch_size
        self.num_workers = num_workers

    # Load and union the supervisd and unsupervised training data, ignoring labels
    # Input: transform
    # Return a DataLoader.
    def load_train_data_mix(self, transform=None):
        print("Start loading mix")
        data_sup = torchvision.datasets.ImageFolder(root=self.sup_train_root_path, transform=transform)
        data_unsup = torchvision.datasets.ImageFolder(root=self.unsup_train_root_path, target_transform=transform)

        concat_dataset = torch.utils.data.ConcatDataset((data_sup, data_unsup))

        loader_unsup = torch.utils.data.DataLoader(concat_dataset, batch_size=self.batch_size, shuffle=True,
                                                   num_workers=self.num_workers)
        return loader_unsup
        print("End loading mix")

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
        train_data_mix = self.load_train_data_mix()
        val_data = self.load_val_data()

        print("Supervised training dataset: size =", len(train_data_sup))
        print("Mixed training dataset: size =", len(train_data_mix))
        print("Validation dataset: size =", len(val_data))


if __name__ == '__main__':
    semi = SemiSupervised(batch_size=1)
    semi.main()
