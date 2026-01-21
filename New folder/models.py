import torch
import torch.nn as nn
from torchvision import models

def build_resnet18(num_classes=2):
    """
    Builds a ResNet-18 model adapted for binary classification (or num_classes).
    Pretrained on ImageNet.
    """
    model = models.resnet18(weights="IMAGENET1K_V1")
    # ResNet fc layer: (fc): Linear(in_features=512, out_features=1000, bias=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def build_effnet_b0(num_classes=2):
    """
    Builds an EfficientNet-B0 model adapted for binary classification.
    Pretrained on ImageNet.
    """
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    # EfficientNet classifier: Sequential(..., (1): Linear(in_features=1280, out_features=1000, bias=True))
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, num_classes
    )
    return model

def build_convnext_tiny(num_classes=2):
    """
    Builds a ConvNeXt Tiny model adapted for binary classification.
    Pretrained on ImageNet.
    """
    model = models.convnext_tiny(weights="IMAGENET1K_V1")
    # ConvNeXt classifier: Sequential(..., (2): Linear(in_features=768, out_features=1000, bias=True))
    # We replace the last layer (index 2)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model
