import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from dataloaders import load_cifar10
import torchvision
import glob
from PIL import Image
import PIL
from torchvision import transforms

mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

task = 3


img = Image.open("horse.jpeg")
img = transforms.ToTensor()(img)
img = transforms.Normalize(mean, std)(img)
img = img.view(1, *img.shape)
img = nn.functional.interpolate(img, size=(256, 256))

layer1 = torchvision.models.resnet18(pretrained=True).conv1
model = torchvision.models.resnet18(pretrained=True)
modules = list(model.children())[:-2]

if task==2:
    init = True
    for i in modules:
        if init:
            pred = i(img)
            init = False
        else:
            pred = i(pred)

if task==3:
    pred = layer1.weight.data

if task==1:
    pred = layer1(img)
print(pred.shape)


fig=plt.figure(figsize=(10, 10))
columns = 8
rows = 8

if task ==1:
    for i in range(1, columns*rows +1):
        img =  pred[0][i-1].view(128,128)
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.imshow(img.detach().numpy())
    plt.show()

if task ==2:
    for i in range(1, columns*rows +1):
        img =  pred[0][i-1].view(8, 8)
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.imshow(img.detach().numpy())
    plt.show()


if task==3:
    for i in range(1, columns*rows +1):
        #img =  pred[0][i].view(8, 8)
        img =  pred[i-1].numpy().transpose(1,2,0)
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.imshow(img)
    plt.show()
