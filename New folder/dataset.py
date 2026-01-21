# dataset.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import DATA_ROOT, IMG_SIZE, BATCH_SIZE

def get_dataloaders():
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
    val_ds   = datasets.ImageFolder(f"{DATA_ROOT}/val",   transform=test_tf)
    test_ds  = datasets.ImageFolder(f"{DATA_ROOT}/test",  transform=test_tf)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    print(f"[DATA] Train={len(train_ds)} Val={len(val_ds)} Test={len(test_ds)}")

    return train_dl, val_dl, test_dl
