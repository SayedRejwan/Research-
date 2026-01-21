# utils.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    auc, 
    accuracy_score, 
    precision_recall_fscore_support
)

def evaluate(model, dl, device, class_names, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    y_true = []
    y_pred = []
    y_probs = []

    print("[EVAL] Running evaluation...")
    
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            logits = model(x)
            
            # Get probabilities for ROC
            probs = torch.softmax(logits, dim=1)
            
            # Get predictions
            preds = logits.argmax(1)
            
            y_probs.extend(probs.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(y.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # ------------------------------------------
    # 1. Metrics Calculation
    # ------------------------------------------
    try:
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        
        print(f"\n[METRICS] Accuracy: {acc:.4f}")
        print(f"[METRICS] F1 Score (Weighted): {f1:.4f}")

        # Save Metrics to CSV
        report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
        df_metrics = pd.DataFrame(report_dict).transpose()
        df_metrics.to_csv(f"{output_dir}/metrics_report.csv")
        print(f"[OUTPUT] Saved detailed metrics to {output_dir}/metrics_report.csv")
    except Exception as e:
        print(f"[ERROR] Failed to calculate metrics: {e}")

    # ------------------------------------------
    # 2. Confusion Matrix Heatmap
    # ------------------------------------------
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        
        # Calculate percentages for annotations
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        # Avoid division by zero
        cm_sum[cm_sum == 0] = 1
        
        cm_perc = cm / cm_sum.astype(float) * 100
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                c = cm[i, j]
                p = cm_perc[i, j]
                if i == j:
                    s = cm_sum[i]
                    annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                elif c == 0:
                    annot[i, j] = ''
                else:
                    annot[i, j] = '%.1f%%\n%d' % (p, c)
                    
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300)
        plt.close()
        print(f"[OUTPUT] Saved Confusion Matrix chart to {output_dir}/confusion_matrix.png")
    except Exception as e:
         print(f"[ERROR] Failed to plot Confusion Matrix: {e}")

    # ------------------------------------------
    # 3. ROC Curve (Multiclass handled as simple or average)
    # ------------------------------------------
    try:
        # Assuming 2 classes for this specific project as per config
        if len(class_names) == 2:
            # Check if we have at least one positive and one negative sample to plot ROC
            if len(np.unique(y_true)) > 1:
                fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1], pos_label=1)
                roc_auc = auc(fpr, tpr)

                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC)')
                plt.legend(loc="lower right")
                plt.grid(alpha=0.3)
                plt.savefig(f"{output_dir}/roc_curve.png", dpi=300)
                plt.close()
                print(f"[OUTPUT] Saved ROC Curve to {output_dir}/roc_curve.png")
            else:
                print("[WARN] Skipping ROC Curve: Test set contains only one class.")
    except Exception as e:
        print(f"[ERROR] Failed to plot ROC Curve: {e}")

    # ------------------------------------------
    # 4. F1 Score & Accuracy Comparison Chart
    # ------------------------------------------
    try:
        # Extract per-class F1 scores
        metrics_data = {
            'Class': [],
            'Metric': [],
            'Value': []
        }
        
        for cls in class_names:
            metrics_data['Class'].append(cls)
            metrics_data['Metric'].append('Precision')
            metrics_data['Value'].append(report_dict[cls]['precision'])
            
            metrics_data['Class'].append(cls)
            metrics_data['Metric'].append('Recall')
            metrics_data['Value'].append(report_dict[cls]['recall'])
            
            metrics_data['Class'].append(cls)
            metrics_data['Metric'].append('F1 Score')
            metrics_data['Value'].append(report_dict[cls]['f1-score'])

        df_plot = pd.DataFrame(metrics_data)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Class', y='Value', hue='Metric', data=df_plot, palette='viridis')
        plt.ylim(0, 1.1)
        plt.title('Performance Metrics per Class')
        plt.ylabel('Score')
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(f"{output_dir}/metrics_chart.png", dpi=300)
        plt.close()
        print(f"[OUTPUT] Saved Metrics Chart to {output_dir}/metrics_chart.png")
    except Exception as e:
        print(f"[ERROR] Failed to plot Metrics Chart: {e}")


def predict_ood(model, dl, device):
    model.eval()
    print("\n[OOD] Predictions:")
    
    # We will also save a CSV for this
    results = []

    with torch.no_grad():
        for x, path in dl:
            x = x.to(device)
            out = model(x)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            
            img_name = os.path.basename(path[0])
            results.append({
                "Image": img_name,
                "Aedes_Prob": probs[0],
                "Culex_Prob": probs[1],
                "Predicted": "Aedes" if probs[0] > probs[1] else "Culex"
            })

            print(f"{path[0]}")
            print(f"  Aedes: {probs[0]:.4f} | Culex: {probs[1]:.4f}")
    
    df = pd.DataFrame(results)
    df.to_csv("outputs/ood_predictions.csv", index=False)
    print("\n[OUTPUT] Saved OOD predictions to outputs/ood_predictions.csv")
