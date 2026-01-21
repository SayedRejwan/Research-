import torch
import os
import sys

# Add current directory to path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import build_resnet18, build_effnet_b0, build_convnext_tiny
from dataset import get_dataloaders
from train_single import train_model
from ensemble_eval import ensemble_predict, ensemble_predict_weighted

# --- CONFIGURATION ---
# UPDATE THIS PATH to your actual dataset location
# Expected structure:
# dataset/
#   train/
#     class1/
#     class2/
#   test/
#     ...
DATA_ROOT = "../dataset"  # Assuming dataset is in the parent directory
if not os.path.exists(DATA_ROOT):
    # Fallback to check if we are in root
    if os.path.exists("dataset"):
        DATA_ROOT = "dataset"
    else:
        print(f"WARNING: Dataset not found at {DATA_ROOT}. Please edit DATA_ROOT in main.py")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def main():
    # 1. Load Data
    print("Loading data...")
    loaders = get_dataloaders(DATA_ROOT)
    
    if "train" not in loaders or "test" not in loaders:
        print("Detailed instruction: Ensure your dataset has 'train' and 'test' folders.")
        print("Exiting...")
        return

    # 2. Train Individual Models
    print("\n=== Training ResNet-18 ===")
    resnet = build_resnet18()
    resnet = train_model(resnet, loaders, DEVICE, epochs=5) # Reduced epochs for demo

    print("\n=== Training EfficientNet-B0 ===")
    effnet = build_effnet_b0()
    effnet = train_model(effnet, loaders, DEVICE, epochs=5)

    # 3. Standard Ensemble Evaluation (2 Models)
    print("\n=== Evaluating 2-Model Ensemble (ResNet + EffNet) ===")
    acc, auc, _, _ = ensemble_predict(
        [resnet, effnet],
        loaders["test"],
        DEVICE
    )
    print(f"Ensemble (2) Results -> Accuracy: {acc:.4f}, ROC-AUC: {auc:.4f}")

    # 4. Extended Ensemble (3 Models) - As requested
    # We train the 3rd model (ConvNeXt) and add it
    print("\n=== Training ConvNeXt (3rd Model) ===")
    try:
        convnext = build_convnext_tiny()
        convnext = train_model(convnext, loaders, DEVICE, epochs=5)
        
        print("\n=== Evaluating 3-Model Ensemble ===")
        acc_3, auc_3, _, _ = ensemble_predict(
            [resnet, effnet, convnext],
            loaders["test"],
            DEVICE
        )
        print(f"Ensemble (3) Results -> Accuracy: {acc_3:.4f}, ROC-AUC: {auc_3:.4f}")
        
        # 5. Weighted Ensemble Example
        # Giving more weight to the likely stronger models (e.g. EffNet & ConvNext)
        print("\n=== Evaluating Weighted Ensemble (3 Models) ===")
        # Weights: ResNet (0.2), EffNet (0.4), ConvNext (0.4)
        acc_w, auc_w = ensemble_predict_weighted(
            [resnet, effnet, convnext],
            loaders["test"],
            DEVICE,
            weights=[0.2, 0.4, 0.4]
        )
        print(f"Weighted Ensemble Results -> Accuracy: {acc_w:.4f}, ROC-AUC: {auc_w:.4f}")

    except Exception as e:
        print(f"\nSkipping 3rd model extension due to error (possibly hardware/memory): {e}")

if __name__ == "__main__":
    main()
