from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_dataloaders(data_root, batch_size=32):
    """
    Creates DataLoaders for train, val, and test splits.
    Expects data_root to contain 'train', 'val', and 'test' subdirectories.
    """
    # Standard ImageNet normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    loaders = {}
    splits = ["train", "val", "test"]
    
    # robust check for folders
    for split in splits:
        path = os.path.join(data_root, split)
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Skipping loader for {split}.")
            continue
            
        ds = datasets.ImageFolder(path, transform)
        # Shuffle only for training
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split=="train"))

    return loaders
