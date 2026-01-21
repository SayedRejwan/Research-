import torch.nn as nn
import torch.nn.functional as F

class ScratchCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        def block(in_c, out_c):
            layers = [nn.Conv2d(in_c, out_c, 3, padding=1)]
            if cfg["use_batchnorm"]:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            block(3, 32),
            block(32, 64),
            block(64, 128),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(cfg["dropout"])
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)
