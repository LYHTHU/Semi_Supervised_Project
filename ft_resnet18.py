from __future__ import print_function
from __future__ import division
import time
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from skimage import io, transform

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = "/scratch/yc3329/ssl_data_96" + "/supervised"

b_size = 8
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=b_size,
                                             shuffle=True, num_workers=4)
              for x in ['train']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, save_path, num_epoch = 10):
    since = time.time()
    
    best_mode_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epoch):
        print("Epoch {}/{}".format(epoch, num_epoch-1))
        print('-'*10)
        for phase in ['train', 'val']:
            if phase == 'train':
                #scheduler.step()
                model.train()
            else:
                model.eval()
            
            run_loss = 0.0
            run_correct = 0
            proc = 0
            
            for inputs, labels in dataloaders[phase]:
                proc += b_size
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                #zeros para
                optimizer.zero_grad()
                
                #forward
                #track history if and only if in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    #print(outputs.size())
                    #print(preds)
                    loss = criterion(outputs, labels)
                    
                    # backwarda and optimize only in training
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                run_loss += loss.item()*inputs.size(0)
                run_correct += torch.sum(labels.data == preds)
                
                
            epoch_loss = run_loss/dataset_sizes[phase]
            epoch_acc = run_correct/dataset_sizes[phase]

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': epoch_loss
            }, save_path)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepycopy(model.state_dict())
                
        print()
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
    print('Best Acc: {:.4f}'.format(best_acc))
        
    model.load(best_model_wts)
        
    return model
    
# pass in parameter
if __name__ == '__main__':
    
    save_path = './ft_resnet152.pt'
    check_path = './ft_resnet152_check.pt'

    model_ft = models.resnet152(pretrained = False)
    num_ftrs = model_ft.fc.in_features

    n_class = 1000
    model_ft.fc = nn.Linear(num_ftrs, n_class)

    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr = 0.001, momentum = 0.9)

    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size = 7, gamma = 0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, check_path, num_epoch = 10)
    torch.save(model_ft.state_dict(), save_path)
