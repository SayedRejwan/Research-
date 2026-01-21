import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from config import *

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    for x, y in loader:
        x = x.to(DEVICE)
        out = model(x)
        prob = torch.softmax(out, 1)

        y_true.extend(y.numpy())
        y_pred.extend(out.argmax(1).cpu().numpy())
        y_prob.extend(prob[:,1].cpu().numpy())

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    print("ROC-AUC:", roc_auc_score(y_true, y_prob))
