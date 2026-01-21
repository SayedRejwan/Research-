import torch
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, m, i, o):
        self.activations = o.detach()

    def backward_hook(self, m, g_in, g_out):
        self.gradients = g_out[0].detach()

    def generate(self, x, class_idx):
        self.model.zero_grad()
        out = self.model(x)
        out[:, class_idx].backward()

        weights = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = cam.cpu().numpy()
        cam = np.maximum(cam, 0)
        cam /= cam.max()
        return cam
