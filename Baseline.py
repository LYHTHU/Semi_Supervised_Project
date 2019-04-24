from ImageLoader import ImageLoader
import torch
import torchvision
import torchvision.transforms as transforms


if __name__ == '__main__':
    # Using torch loader
    # data = torchvision.datasets.ImageFolder(root="./ssl_data_96")
    #
    # loader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=True, num_workers=8)

    cv_loader = ImageLoader()
    images, labels = ImageLoader.parse_supervised(cv_loader.load_supervised("train"))
