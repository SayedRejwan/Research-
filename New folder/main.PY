# main.py
import torch
import torch.nn as nn
from torchvision import models
from dataset import get_dataloaders, get_ood_loader
from utils import evaluate, predict_ood
from gradcam import run_gradcam

# ---------------- CONFIG ---------------- #
DATA_ROOT = "F:/MLMI_SPLIT"
OOD_ROOT  = "G:/Downloads/archive/Mosquito"
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 3
CLASS_NAMES = ["Aedes aegypti", "Culex quinquefasciatus"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[SYSTEM] Device: {DEVICE}")

# ---------------- DATA ---------------- #
train_dl, val_dl, test_dl = get_dataloaders(
    DATA_ROOT, IMG_SIZE, BATCH_SIZE
)

ood_dl = get_ood_loader(
    OOD_ROOT, IMG_SIZE, BATCH_SIZE
)

# ---------------- MODEL ---------------- #
model = models.efficientnet_b0(weights="DEFAULT")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ---------------- TRAIN ---------------- #
for epoch in range(EPOCHS):
    model.train()
    correct, total, loss_sum = 0, 0, 0

    for x, y in train_dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    print(f"[EPOCH {epoch+1}/{EPOCHS}] "
          f"Loss={loss_sum/len(train_dl):.4f} "
          f"Acc={correct/total:.4f}")

# ---------------- TEST ---------------- #
evaluate(model, test_dl, DEVICE, CLASS_NAMES)

# ---------------- OOD PREDICTION ---------------- #
predict_ood(model, ood_dl, DEVICE)

# ---------------- GRADCAM (on OOD) ---------------- #
CLASS_NAMES = ["Aedes aegypti", "Culex quinquefasciatus"]
run_gradcam(model, ood_dl, DEVICE, CLASS_NAMES)
