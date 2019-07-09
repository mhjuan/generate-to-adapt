import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

def init_weights(m):
    if type(m) == nn.Conv2d:
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        nn.init.constant_(m.bias, 0.0)
    elif type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight, mean=1.0, std=0.02)
        nn.init.constant_(m.bias, 0.0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0.0, std=0.1)
        nn.init.constant_(m.bias, 0.0)

def transform_n_normalize(img):
    return transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])(img)

def unnormalize(img):
    return transforms.Compose([
        transforms.Normalize(mean=[0, 0, 0], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1])
    ])(img)
