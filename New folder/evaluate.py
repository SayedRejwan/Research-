import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from config import *

def evaluate(model, loader):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)

            y_true.extend(labels.numpy())
            y_pred.extend(outputs.argmax(1).cpu().numpy())
            y_prob.extend(probs[:,1].cpu().numpy())

    print("\nCONFUSION MATRIX")
    print(confusion_matrix(y_true, y_pred))

    print("\nCLASSIFICATION REPORT")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    print("ROC-AUC:", roc_auc_score(y_true, y_prob))

    conf = np.max(np.stack([1-np.array(y_prob), y_prob]), axis=0)
    print(f"Confidence Mean={conf.mean():.4f} Std={conf.std():.4f}")
