from sklearn.metrics import roc_auc_score, classification_report
import torch
import numpy as np

def evaluate(model, loader, device):
    model.eval()
    y_true, y_score = [], []
    y_pred = []

    print("\nRunning Evaluation...")
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            # Forward pass
            logits = model(x)
            # Get probabilities
            probs = logits.softmax(1)[:, 1].cpu()
            # Get hard predictions
            preds = logits.argmax(1).cpu()

            y_true.extend(y.numpy())
            y_score.extend(probs.numpy())
            y_pred.extend(preds.numpy())

    # Check if we have gathered any data
    if not y_true:
        print("No evaluation data found.")
        return

    try:
        auc = roc_auc_score(y_true, y_score)
        print(f"ROC-AUC: {auc:.4f}")
    except ValueError:
        print("ROC-AUC: N/A (Only one class present in targets)")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))
