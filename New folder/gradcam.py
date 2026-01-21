import torch
import torch.nn.functional as F
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient) # Note: usage might vary by torch version

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        self.model.eval()
        
        # Forward pass
        logit = self.model(x)
        
        if class_idx is None:
            class_idx = logit.argmax(dim=1).item()
            
        # Backward pass
        self.model.zero_grad()
        score = logit[0, class_idx]
        score.backward()
        
        # Generate CAM
        gradients = self.gradients
        activations = self.activations
        b, k, u, v = gradients.size()
        
        # Global Average Pooling of gradients
        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)
        
        # Linear combination of activations
        cam = (weights * activations).sum(1, keepdim=True)
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-7)
        
        # Resize to input size
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return cam.detach().cpu().numpy()[0, 0]

# Helper to attach specific layers
def get_target_layer(model, model_name):
    if "resnet" in model_name.lower():
        return model.layer4[-1]
    elif "efficientnet" in model_name.lower():
        # Last block of features
        return model.features[-1]
    elif "convnext" in model_name.lower():
        return model.features[-1]
    return None
