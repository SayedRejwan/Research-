import torch
import torch.nn.functional as F
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, 
    confusion_matrix, precision_score, recall_score, roc_curve
)
from tqdm import tqdm

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import build_resnet18, build_effnet_b0
from dataset import get_dataloaders
from train_single import train_model

# Config
DATA_ROOT = "dataset"
RESULTS_DIR = "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    plt.title('Ensemble Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_probs, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_val = roc_auc_score(y_true, y_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_val:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Ensemble)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def stress_test(models, device, input_shape=(1, 3, 224, 224), iterations=100):
    print(f"\n⚡ STARTING STRESS TEST ({iterations} iterations)...")
    
    # Generate random input
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warmup
    for model in models:
        model.eval()
        model.to(device)
        with torch.no_grad():
            _ = model(dummy_input)
            
    # Timing
    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.time()
            # Simulate ensemble forward pass
            probs = []
            for model in models:
                out = model(dummy_input)
                probs.append(F.softmax(out, dim=1))
            _ = torch.mean(torch.stack(probs), dim=0)
            end = time.time()
            latencies.append((end - start) * 1000) # ms
            
    avg_latency = np.mean(latencies)
    fps = 1000 / avg_latency
    
    print(f"Average Ensemble Latency: {avg_latency:.2f} ms")
    print(f"Throughput: {fps:.2f} FPS")
    return avg_latency, fps

def main():
    print(f"Running on {DEVICE}")
    
    # 1. Load Data
    if not os.path.exists(DATA_ROOT):
        print("Dataset not found! Please run create_dummy_data.py first.")
        return
        
    loaders = get_dataloaders(DATA_ROOT)
    
    # 2. Train Models
    print("\n--- Training Base Models ---")
    resnet = build_resnet18()
    resnet = train_model(resnet, loaders, DEVICE, epochs=3)
    
    effnet = build_effnet_b0()
    effnet = train_model(effnet, loaders, DEVICE, epochs=3)
    
    models_list = [resnet, effnet]
    
    # 3. Evaluation
    print("\n--- Evaluating Ensemble ---")
    y_true = []
    y_probs_all = []
    
    resnet.eval()
    effnet.eval()
    
    with torch.no_grad():
        for x, y in tqdm(loaders["test"], desc="Inference"):
            x = x.to(DEVICE)
            
            # Forward pass
            out1 = F.softmax(resnet(x), dim=1)
            out2 = F.softmax(effnet(x), dim=1)
            
            # Soft Voting
            avg_prob = (out1 + out2) / 2.0
            
            y_probs_all.append(avg_prob.cpu())
            y_true.append(y)
            
    y_probs_all = torch.cat(y_probs_all)
    y_true = torch.cat(y_true)
    y_pred = y_probs_all.argmax(1)
    
    # 4. Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_probs_all[:, 1])
    except:
        auc = 0.5
        
    print("\n=== FINAL ENSEMBLE METRICS ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC AUC  : {auc:.4f}")
    
    # 5. Visualizations
    print("\ngenerating plots in ./results/...")
    plot_confusion_matrix(y_true, y_pred, os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    
    # Only plot ROC if binary classification and we have both classes
    if len(np.unique(y_true)) > 1:
        plot_roc_curve(y_true, y_probs_all[:, 1], os.path.join(RESULTS_DIR, "roc_curve.png"))
    
    # 6. Stress Test
    stress_test(models_list, DEVICE)

if __name__ == "__main__":
    main()
