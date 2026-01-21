# main_effnet.py
# --------------------------------------------------
# EfficientNet training + evaluation (MLMI-2024)
# --------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from config import DEVICE, EPOCHS
from dataset import get_dataloaders


# --------------------------------------------------
# MODEL
# --------------------------------------------------
def build_model(num_classes: int):
    model = models.efficientnet_b0(weights="DEFAULT")
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


# --------------------------------------------------
# TRAIN LOOP
# --------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    correct, total, running_loss = 0, 0, 0.0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    return running_loss / len(loader), acc


# --------------------------------------------------
# EVAL LOOP
# --------------------------------------------------
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []

    for images, labels in loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu()

        y_true.extend(labels.numpy())
        y_pred.extend(preds.numpy())

    return y_true, y_pred


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    print("=" * 60)
    print("[SYSTEM] Starting EfficientNet Training")
    print(f"[SYSTEM] Device: {DEVICE}")
    print("=" * 60)

    # Load data
    train_dl, val_dl, test_dl = get_dataloaders()
    num_classes = len(train_dl.dataset.classes)

    print(f"[DATA] Classes ({num_classes}): {train_dl.dataset.classes}")
    print("=" * 60)

    # Build model
    model = build_model(num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Training
    for epoch in range(EPOCHS):
        loss, acc = train_one_epoch(
            model, train_dl, criterion, optimizer
        )
        print(
            f"[EPOCH {epoch+1:02d}/{EPOCHS}] "
            f"Loss: {loss:.4f} | Train Acc: {acc:.4f}"
        )

    print("=" * 60)
    print("[EVAL] Running final evaluation on TEST set")

    # Evaluation
    y_true, y_pred = evaluate(model, test_dl)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true,
        y_pred,
        target_names=test_dl.dataset.classes,
        digits=4
    )

    print(f"[RESULT] Test Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(report)

    print("=" * 60)
    print("[DONE] Training & evaluation complete")


# --------------------------------------------------
if __name__ == "__main__":
    main()
