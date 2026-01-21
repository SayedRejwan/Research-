import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np
import os
import torch
import cv2

def plot_training_curves(train_losses, train_accs, output_dir):
    plt.figure(figsize=(10, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy', color='orange')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def plot_roc_curve(y_true, y_scores, output_dir):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()

def apply_gradcam(model, image_tensor, target_class=None):
    """
    Generates GradCAM heatmap for a specific image tensor.
    image_tensor: (1, C, H, W)
    """
    model.eval()
    
    # Hook to capture gradients and activations
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
        
    def forward_hook(module, input, output):
        activations.append(output)
        
    # Hook the last convolutional layer
    # In ScratchCNN, features is a Sequential. The last item is features[2].
    # features[2] is a Sequential logic block. Let's hook the Conv2d inside it for safety?
    # Or just hook the whole features block output.
    # The user model is: self.features -> gap -> flatten -> dropout -> fc
    # So self.features output is the spatial map (before GAP).
    
    handle_f = model.features.register_forward_hook(forward_hook)
    handle_b = model.features.register_full_backward_hook(backward_hook)
    
    # Forward
    output = model(image_tensor)
    if target_class is None:
        target_class = output.argmax(dim=1).item()
        
    # Backward
    model.zero_grad()
    score = output[0, target_class]
    score.backward()
    
    # Get captures
    grads = gradients[0].cpu().data.numpy()[0] # (C, H, W)
    fmaps = activations[0].cpu().data.numpy()[0] # (C, H, W)
    
    handle_f.remove()
    handle_b.remove()
    
    # Weights = Global Average Pooling of Gradients
    weights = np.mean(grads, axis=(1, 2)) # (C,)
    
    # Weighted combination of feature maps
    cam = np.zeros(fmaps.shape[1:], dtype=np.float32) # (H, W)
    for i, w in enumerate(weights):
        cam += w * fmaps[i]
        
    # ReLU
    cam = np.maximum(cam, 0)
    
    # Normalize
    if np.max(cam) > 0:
        cam = cam / np.max(cam)
    else:
        cam = cam # avoid div by zero
        
    return cam

def save_gradcam_image(image_tensor, cam, output_path, label_text):
    # Denormalize image for visualization
    # Assuming standard normalization was applied or just ToTensor
    # If just ToTensor, it's 0-1.
    img = image_tensor.cpu().numpy().transpose(1, 2, 0) # (H, W, C)
    img = (img - img.min()) / (img.max() - img.min()) # Scale to 0-1
    img = np.uint8(255 * img)
    
    # Resize cam to image size
    heatmap = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose
    superimposed_img = heatmap * 0.4 + img * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    # Add label
    cv2.putText(superimposed_img, label_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imwrite(output_path, superimposed_img)

