from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import os

def get_transforms(cfg, train=True):
    t = [transforms.Resize((cfg["img_size"], cfg["img_size"]))]

    if train and cfg["augment"]:
        t += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2),
        ]

    t += [transforms.ToTensor()]
    return transforms.Compose(t)

def get_dataloaders(cfg):
    """
    Creates DataLoaders for train and validation sets using the structure:
    dataset/
      train/
      val/
    """
    train_transform = get_transforms(cfg, train=True)
    val_transform = get_transforms(cfg, train=False)

    # Use data path from config
    base_path = cfg.get("data_path", "dataset")
    train_dir = os.path.join(base_path, "train")
    val_dir = os.path.join(base_path, "val")

    # Check if directories exist
    if not os.path.exists(train_dir):
        print(f"WARNING: Train directory not found at {train_dir}")
        print("Please ensure 'dataset/train' and 'dataset/val' exist.")
        # Return dummy loaders or raise error depending on robustness needed
        # For this script, we'll let ImageFolder raise the error if called, 
        # or maybe we should just simpler:
    
    try:
        train_ds = datasets.ImageFolder(train_dir, train_transform)
        val_ds = datasets.ImageFolder(val_dir, val_transform)

        train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=2)
        val_dl = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=2)
        
        return train_dl, val_dl
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return None, None
