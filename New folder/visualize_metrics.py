import torch
import torch.nn.functional as F
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, 
    confusion_matrix, precision_score, recall_score, 
    roc_curve, precision_recall_curve, classification_report
)
from sklearn.calibration import calibration_curve
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

def plot_scalar_metrics(metrics_dict, save_path):
    """Plots a bar chart for scalar metrics like Accuracy, F1, etc."""
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    plt.title('Ensemble Model Performance Metrics', fontsize=16)
    plt.ylim(0, 1.1)
    plt.ylabel('Score')
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.4f}',
                 ha='center', va='bottom', fontsize=12)
                 
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_precision_recall_curve(y_true, y_probs, save_path):
    """Plots Precision-Recall Curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='purple', lw=2, label='PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_calibration_curve(y_true, y_probs, save_path):
    """Plots Calibration Curve (Reliability Diagram)."""
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)
    
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Ensemble')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve (Reliability Diagram)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_class_report_heatmap(y_true, y_pred, class_names, save_path):
    """Visualizes the Classification Report as a Heatmap."""
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    df = pd.DataFrame(report).transpose()
    
    # Drop support column for heatmap visualization (it's scale is different)
    df_plot = df.drop(columns=['support'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_plot, annot=True, cmap='viridis', fmt='.3f', cbar=True)
    plt.title('Classification Report Heatmap')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    print(f"Generating rich visualizations on {DEVICE}...")
    
    # 1. Load Data
    loaders = get_dataloaders(DATA_ROOT)
    
    # 2. Train Models (Quickly retrain or assume weights if saved - for now we retrain quickly for standalone)
    print("Training models to generate valid metrics...")
    resnet = build_resnet18()
    resnet = train_model(resnet, loaders, DEVICE, epochs=1) # Fast epoch
    effnet = build_effnet_b0()
    effnet = train_model(effnet, loaders, DEVICE, epochs=1)
    
    # 3. Inference
    y_true = []
    y_probs_all = []
    
    resnet.eval()
    effnet.eval()
    
    with torch.no_grad():
        for x, y in tqdm(loaders["test"], desc="Inference"):
            x = x.to(DEVICE)
            out1 = F.softmax(resnet(x), dim=1)
            out2 = F.softmax(effnet(x), dim=1)
            # Soft Voting
            avg_prob = (out1 + out2) / 2.0
            y_probs_all.append(avg_prob.cpu())
            y_true.append(y)
            
    y_probs_all = torch.cat(y_probs_all)
    y_true = torch.cat(y_true)
    y_pred = y_probs_all.argmax(1)
    
    # 4. Generate All Plots
    print("Generating final images...")
    
    # A. Scalar Metrics Bar Chart
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred, average='weighted'),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'AUC': roc_auc_score(y_true, y_probs_all[:, 1]) if len(np.unique(y_true)) > 1 else 0.5
    }
    plot_scalar_metrics(metrics, os.path.join(RESULTS_DIR, "metrics_summary_bar.png"))
    
    # B. Precision-Recall Curve
    if len(np.unique(y_true)) > 1:
        plot_precision_recall_curve(y_true, y_probs_all[:, 1], os.path.join(RESULTS_DIR, "precision_recall_curve.png"))
        
        # C. Calibration Curve
        plot_calibration_curve(y_true, y_probs_all[:, 1], os.path.join(RESULTS_DIR, "calibration_curve.png"))
    
    # D. Class-wise Heatmap
    plot_class_report_heatmap(y_true, y_pred, ["larva_negative", "larva_positive"], os.path.join(RESULTS_DIR, "classification_report_heatmap.png"))
    
    print(f"Done! All images saved to {os.path.abspath(RESULTS_DIR)}")

if __name__ == "__main__":
    main()
