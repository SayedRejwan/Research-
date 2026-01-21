# gradcam.py
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ------------------------------
# GradCAM Core
# ------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x, class_idx):
        self.model.zero_grad()

        logits = self.model(x)
        score = logits[:, class_idx]
        score.backward()

        # Global Average Pooling on gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)

        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam[0].cpu().numpy(), logits.detach()


# ------------------------------
# Visualization
# ------------------------------
def overlay_heatmap(image, cam, alpha=0.5):
    cam = Image.fromarray(np.uint8(cam * 255)).resize(image.size, Image.BILINEAR)
    cam = np.array(cam)

    heatmap = plt.get_cmap("jet")(cam / 255.0)[:, :, :3]
    heatmap = Image.fromarray(np.uint8(heatmap * 255))

    blended = Image.blend(image.convert("RGB"), heatmap, alpha)
    return blended


# ------------------------------
# Runner
# ------------------------------
def run_gradcam(model, dataloader, device, class_names):
    os.makedirs("outputs", exist_ok=True)
    model.eval()

    cam_engine = GradCAM(model, model.features[-1])

    for images, paths in dataloader:
        images = images.to(device)

        for i in range(images.size(0)):
            x = images[i].unsqueeze(0)

            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=1)

            class_idx = probs.argmax(dim=1).item()
            confidence = probs[0, class_idx].item()

            cam, _ = cam_engine.generate(x, class_idx)

            img = Image.open(paths[i]).convert("RGB")
            cam_img = overlay_heatmap(img, cam)

            fname = os.path.basename(paths[i])
            out_path = f"outputs/gradcam_{fname}"
            cam_img.save(out_path)

            # --------- INSIGHTS ----------
            print("\n[Grad-CAM Insight]")
            print(f"Image      : {paths[i]}")
            print(f"Prediction : {class_names[class_idx]}")
            print(f"Confidence : {confidence:.4f}")
            print(f"Saved CAM  : {out_path}")

            if confidence > 0.85:
                print("Insight    : Model is highly confident — focus regions are decisive.")
            elif confidence > 0.6:
                print("Insight    : Moderate confidence — multiple visual cues used.")
            else:
                print("Insight    : Low confidence — ambiguous or weak discriminative features.")
