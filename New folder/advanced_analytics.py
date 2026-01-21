"""
Advanced Analytics Generation Script
Generates ROC curves, Precision-Recall curves, F1 charts, and detailed metric tables.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    auc, 
    precision_recall_curve, 
    f1_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    matthews_corrcoef, 
    cohen_kappa_score
)
from config import DATA_ROOT, IMG_SIZE, BATCH_SIZE, DEVICE

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12

def load_data():
    """Load test dataset"""
    test_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()
    ])
    
    test_ds = datasets.ImageFolder(f"{DATA_ROOT}/test", transform=test_tf)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    return test_dl, test_ds.classes

def load_model(num_classes):
    """Load trained ResNet-18 model"""
    print(f"Loading model from: mosquito_resnet18.pth")
    model = models.resnet18(weights=None) # Structure only
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Load weights
    state_dict = torch.load("mosquito_resnet18.pth", map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

def get_predictions(model, loader):
    """Run inference to get probabilities and true labels"""
    y_true = []
    y_prob = []  # Probability of positive class (Culex)
    y_pred = []  # Hard predictions
    
    print("Running inference on test set...")
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            out = model(x)
            prob = torch.softmax(out, dim=1)
            
            y_true.extend(y.numpy())
            y_prob.extend(prob[:, 1].cpu().numpy()) # Probability of class 1
            y_pred.extend(out.argmax(1).cpu().numpy())
            
    return np.array(y_true), np.array(y_pred), np.array(y_prob)

def plot_roc_curve(y_true, y_prob, classes):
    """Generate and save ROC Curve"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='#d62728', lw=3, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='#7f7f7f', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Fill area
    plt.fill_between(fpr, tpr, alpha=0.1, color='#d62728')
    
    plt.tight_layout()
    plt.savefig('advanced_roc_curve.png', dpi=300)
    print("✅ Saved: advanced_roc_curve.png")
    plt.close()

def plot_confusion_matrix_detailed(y_true, y_pred, classes):
    """Generate detailed Confusion Matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    
    # Custom annotations with count and percentage
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_percent[i, j]
            if i == j:
                s = cm_percent.sum()
                annot[i, j] = f'{c}\n({p:.1%})'
            else:
                annot[i, j] = f'{c}\n({p:.1%})'
                
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', linewidths=1, linecolor='black',
                xticklabels=classes, yticklabels=classes, annot_kws={"size": 14})
    
    plt.ylabel('True Label', fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.title('Detailed Confusion Matrix', fontweight='bold', fontsize=16)
    
    plt.tight_layout()
    plt.savefig('advanced_confusion_matrix.png', dpi=300)
    print("✅ Saved: advanced_confusion_matrix.png")
    plt.close()

def plot_f1_score_chart(y_true, y_pred, classes):
    """Generate F1 Score and other metrics chart"""
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    
    # Extract metrics for plotting
    metrics = ['precision', 'recall', 'f1-score']
    data = []
    
    for cls in classes:
        for m in metrics:
            data.append({
                'Class': cls,
                'Metric': m.capitalize(),
                'Score': report[cls][m]
            })
            
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Class', y='Score', hue='Metric', data=df, palette='viridis')
    
    plt.ylim(0.9, 1.01)  # Zoom in since accuracy is high
    plt.ylabel('Score', fontweight='bold')
    plt.xlabel('Class', fontweight='bold')
    plt.title('Performance Metrics per Class (Precision, Recall, F1)', fontweight='bold', fontsize=16)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add values on top of bars
    for container in plt.gca().containers:
        plt.gca().bar_label(container, fmt='%.3f', padding=3)
        
    plt.tight_layout()
    plt.savefig('advanced_f1_metrics_chart.png', dpi=300)
    print("✅ Saved: advanced_f1_metrics_chart.png")
    plt.close()

def generate_metrics_table(y_true, y_pred, y_prob):
    """Generate a comprehensive table image of all metrics"""
    
    # Calculate scalar metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # Specificity calculation
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    
    # Create DataFrame
    metrics_data = [
        ["Overall Accuracy", f"{acc:.4f}"],
        ["Weighted Precision", f"{prec:.4f}"],
        ["Weighted Recall", f"{rec:.4f}"],
        ["Weighted F1-Score", f"{f1:.4f}"],
        ["Matthews Correlation Coeff (MCC)", f"{mcc:.4f}"],
        ["Cohen's Kappa", f"{kappa:.4f}"],
        ["Sensitivity (Recall)", f"{sensitivity:.4f}"],
        ["Specificity", f"{specificity:.4f}"],
        ["False Positive Rate", f"{fp/(tn+fp):.4f}"],
        ["False Negative Rate", f"{fn/(tp+fn):.4f}"]
    ]
    
    # Plot table
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=metrics_data, colLabels=["Metric", "Value"], loc='center', cellLoc='left')
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Style logic
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#40466e')
            cell.set_text_props(color='white', weight='bold')
        elif row % 2 == 1:
            cell.set_facecolor('#f1f1f2')
            
    plt.title('Comprehensive Performance Metrics', fontweight='bold', fontsize=16, y=0.95)
    plt.savefig('advanced_metrics_table.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: advanced_metrics_table.png")
    plt.close()

def main():
    print("="*60)
    print("🚀 STARTING ADVANCED ANALYTICS GENERATION")
    print("="*60)
    
    # Load
    test_dl, classes = load_data()
    model = load_model(len(classes))
    
    # Inference
    y_true, y_pred, y_prob = get_predictions(model, test_dl)
    
    print("\ngenerating plots...")
    
    # 1. ROC Curve
    plot_roc_curve(y_true, y_prob, classes)
    
    # 2. Detailed Confusion Matrix
    plot_confusion_matrix_detailed(y_true, y_pred, classes)
    
    # 3. F1 Score Chart
    plot_f1_score_chart(y_true, y_pred, classes)
    
    # 4. Metrics Table
    generate_metrics_table(y_true, y_pred, y_prob)
    
    print("\n" + "="*60)
    print("🎉 ALL ADVANCED CHARTS & TABLES GENERATED!")
    print("="*60)

if __name__ == "__main__":
    main()
