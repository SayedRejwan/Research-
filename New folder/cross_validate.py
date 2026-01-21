import os
import glob
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import models, transforms
from sklearn.model_selection import KFold
from PIL import Image
from config import *  # Imports paths and config

# ---------------- CONFIG ---------------- #
K_FOLDS = 5
EPOCHS_PER_FOLD = 3  # Fast check
BATCH_SIZE = 16

# ---------------- 1. DATASET AGGREGATOR ---------------- #
class UnifiedMosquitoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []
        self.classes = CLASS_NAMES
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Crawl ALL folders (train, val, test)
        for split in ['train', 'val', 'test']:
            path = os.path.join(root_dir, split)
            for cls_name in self.classes:
                cls_folder = os.path.join(path, cls_name)
                if not os.path.exists(cls_folder):
                    continue
                    
                # Find images
                for img_name in os.listdir(cls_folder):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append((
                            os.path.join(cls_folder, img_name),
                            self.class_to_idx[cls_name]
                        ))
        
        print(f"[DATA] Consolidated Dataset Size: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# ---------------- 2. HELPER FUNCTIONS ---------------- #
def reset_weights(m):
    '''
    Try resetting model weights to avoid weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def train_one_fold(fold, model, train_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    print(f"\n   >>> FOLD {fold+1}/{K_FOLDS} STARTING")
    
    for epoch in range(EPOCHS_PER_FOLD):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        print(f"       [Epoch {epoch+1}] Loss: {running_loss/len(train_loader):.4f} | Acc: {100.*correct/total:.2f}%")

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    acc = 100. * correct / total
    print(f"   >>> FOLD {fold+1} RESULT: {acc:.2f}% Accuracy")
    return acc

# ---------------- 3. MAIN CROSS-VALIDATION LOOP ---------------- #
if __name__ == "__main__":
    device = torch.device(DEVICE)
    print(f"[SYSTEM] Running {K_FOLDS}-Fold Cross Validation on {device}")
    
    # Define Transforms
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    
    # Load Unified Dataset
    dataset = UnifiedMosquitoDataset(DATA_ROOT, transform=tf)
    
    # K-Fold Split
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    results = {}
    
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # Samplers
        train_subsampler = SubsetRandomSampler(train_ids)
        test_subsampler = SubsetRandomSampler(test_ids)
        
        # Loaders
        trainloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_subsampler)
        testloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=test_subsampler)
        
        # Init New Model
        model = models.efficientnet_b0(weights="DEFAULT")
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
        model = model.to(device)
        
        # Run Fold
        fold_acc = train_one_fold(fold, model, trainloader, testloader, device)
        results[fold] = fold_acc

    # ---------------- 4. FINAL REPORT ---------------- #
    print("\n" + "="*40)
    print("      CROSS VALIDATION SUMMARY")
    print("="*40)
    accuracies = list(results.values())
    for f, acc in results.items():
        print(f"Fold {f+1}: {acc:.2f}%")
        
    print("-" * 40)
    print(f"AVERAGE ACCURACY: {np.mean(accuracies):.2f}%")
    print(f"STD DEVIATION:    {np.std(accuracies):.2f}%")
    print("="*40)
    
    if np.mean(accuracies) > 98.0:
        print("\n[VERDICT] The 100% was REAL. Your model is robust.")
    else:
        print("\n[VERDICT] The 100% might have been lucky. Real performance is lower.")
