import torch
import torch.nn as nn
from torchvision import models

class DogCatResNet(nn.Module):
    def __init__(self, freeze_backbone=True):
        super(DogCatResNet, self).__init__()
        # Load pretrained ResNet18
        self.model = models.resnet18(pretrained=True)

        # Optionally freeze backbone layers
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Replace final fully connected layer for 2 classes (cat, dog)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 2)

    def forward(self, x):
        return self.model(x)