import torch.nn as nn
from torchvision import models
from config import *

def build_model():
    model = models.efficientnet_b2(weights="DEFAULT")
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        NUM_CLASSES
    )
    return model.to(DEVICE)
