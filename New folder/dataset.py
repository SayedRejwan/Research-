# dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(DATA_ROOT, IMG_SIZE, BATCH_SIZE):
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    test_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    train_ds = datasets.ImageFolder(f"{DATA_ROOT}/train", transform=train_tf)
    val_ds   = datasets.ImageFolder(f"{DATA_ROOT}/val", transform=test_tf)
    test_ds  = datasets.ImageFolder(f"{DATA_ROOT}/test", transform=test_tf)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_dl  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"[DATA] Train={len(train_ds)} Val={len(val_ds)} Test={len(test_ds)}")
    print(f"[DATA] Classes: {train_ds.classes}")

    return train_dl, val_dl, test_dl


# ---------------- OOD DATASET ---------------- #

class OODDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.images = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.images[idx]


def get_ood_loader(OOD_ROOT, IMG_SIZE, BATCH_SIZE):
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])

    ds = OODDataset(OOD_ROOT, tf)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    print(f"[OOD] Loaded {len(ds)} OOD images from {OOD_ROOT}")
    return dl
