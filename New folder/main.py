import os
import random
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report

# ======================================================
# CONFIG
# ======================================================
ROOT_DATA = r"F:\MLMI-2024 (Mosquito Larvae Microscopic Images)\MLMI-2024"
WORK_DIR  = r"F:\MLMI_SPLIT"

TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1

IMG_SIZE   = 224
BATCH_SIZE = 16
EPOCHS     = 10
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

CLASSES = ["Aedes aegypti", "Culex quinquefasciatus"]
VIEWS   = ["abdomen", "full body", "head", "siphon"]

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# ======================================================
# SAFETY CHECKS
# ======================================================
print("Running on OS:", os.name)
assert os.name == "nt", "❌ Must run locally on Windows"

assert os.path.exists(ROOT_DATA), f"❌ Dataset not found: {ROOT_DATA}"

print("Dataset path OK")

# ======================================================
# COLLECT ALL IMAGES (IMAGE-LEVEL)
# ======================================================
all_images = []

for cls in CLASSES:
    for view in VIEWS:
        view_dir = os.path.join(ROOT_DATA, cls, view)
        for img in os.listdir(view_dir):
            if img.lower().endswith(VALID_EXT):
                all_images.append((cls, os.path.join(view_dir, img)))

assert len(all_images) > 0, "❌ No images found"

print(f"Total images found: {len(all_images)}")

# ======================================================
# SPLIT DATA
# ======================================================
random.shuffle(all_images)

n_total = len(all_images)
n_train = int(n_total * TRAIN_RATIO)
n_val   = int(n_total * VAL_RATIO)

splits = {
    "train": all_images[:n_train],
    "val":   all_images[n_train:n_train + n_val],
    "test":  all_images[n_train + n_val:]
}

# ======================================================
# CREATE SPLIT FOLDERS
# ======================================================
if os.path.exists(WORK_DIR):
    shutil.rmtree(WORK_DIR)

for split in splits:
    for cls in CLASSES:
        os.makedirs(os.path.join(WORK_DIR, split, cls), exist_ok=True)

# ======================================================
# COPY FILES SAFELY
# ======================================================
for split, items in splits.items():
    for cls, src_path in items:
        dst_path = os.path.join(
            WORK_DIR,
            split,
            cls,
            os.path.basename(src_path)
        )
        shutil.copy2(src_path, dst_path)

print("Data split completed")

# ======================================================
# VERIFY SPLIT CONTENT
# ======================================================
for split in ["train", "val", "test"]:
    for cls in CLASSES:
        p = os.path.join(WORK_DIR, split, cls)
        print(f"{split}/{cls}: {len(os.listdir(p))} images")

# ======================================================
# DATA LOADERS
# ======================================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

datasets_dict = {
    split: datasets.ImageFolder(os.path.join(WORK_DIR, split), transform)
    for split in ["train", "val", "test"]
}

loaders = {
    split: DataLoader(datasets_dict[split], batch_size=BATCH_SIZE, shuffle=True)
    for split in datasets_dict
}

# ======================================================
# MODEL
# ======================================================
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ======================================================
# TRAINING
# ======================================================
for epoch in range(EPOCHS):
    model.train()
    correct, total = 0, 0

    for x, y in loaders["train"]:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Accuracy: {correct/total:.4f}")

# ======================================================
# EVALUATION
# ======================================================
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for x, y in loaders["test"]:
        x = x.to(DEVICE)
        out = model(x)
        pred = out.argmax(1).cpu().tolist()
        y_true.extend(y.tolist())
        y_pred.extend(pred)

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASSES))

# ======================================================
# SAVE MODEL
# ======================================================
torch.save(model.state_dict(), "mosquito_resnet18.pth")
print("Model saved: mosquito_resnet18.pth")
