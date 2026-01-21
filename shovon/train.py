import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from config import CONFIG
from model import ScratchCNN
from data import get_dataloaders
from evaluate import evaluate
from utils import plot_training_curves, plot_confusion_matrix, plot_roc_curve, apply_gradcam, save_gradcam_image
import time
import os
import random

def get_optimizer(cfg, model):
    if cfg["optimizer"] == "adam":
        print(f"Using Adam optimizer with lr={cfg['lr']}, weight_decay={cfg['weight_decay']}")
        return Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    print(f"Using SGD optimizer with lr={cfg['lr']}, momentum=0.9, weight_decay={cfg['weight_decay']}")
    return SGD(model.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=cfg["weight_decay"])

def train_model():
    # 0. Setup Output Dir
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    print(f"Outputs will be saved in: {CONFIG['output_dir']}")

    # 1. Setup
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {DEVICE}")
    
    # 2. Data
    print("Loading data...")
    train_dl, val_dl = get_dataloaders(CONFIG)
    
    if train_dl is None or val_dl is None:
        print("Data loading failed.")
        return

    # 3. Model
    print("Initializing from-scratch model...")
    model = ScratchCNN(CONFIG).to(DEVICE)
    
    # 4. Optimization
    optimizer = get_optimizer(CONFIG, model)
    
    # Calculate weights to handle potential difficulty imbalance
    # Previous run showed bias towards Aedes (Class 0).
    # Model mistakenly predicts Aedes when it sees Culex.
    # We need to penalize errors on Class 1 (Culex) MORE.
    # Current counts: Aedes=~310, Culex=~375.
    # Standard balancing gave Culex LESS weight (0.82). We need to reverse this.
    
    # We will manually boost Culex weight to force the model to learn it.
    # [Aedes Weight, Culex Weight]
    weight = torch.tensor([1.0, 5.0], device=DEVICE) 
    print(f"Using manual bias-correction weights: {weight}")

    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=CONFIG["label_smoothing"])
    
    # Trackers
    train_losses = []
    train_accs = []
    
    # 5. Training Loop
    print(f"Starting training for {CONFIG['epochs']} epochs...")
    start_time = time.time()
    
    for epoch in range(CONFIG["epochs"]):
        model.train()
        correct, total, loss_sum = 0, 0, 0
        
        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            loss_sum += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
            
        # Stats
        epoch_acc = correct / total if total > 0 else 0
        avg_loss = loss_sum / len(train_dl) if len(train_dl) > 0 else 0
        
        train_losses.append(avg_loss)
        train_accs.append(epoch_acc)
        
        print(f"[Epoch {epoch+1:02d}/{CONFIG['epochs']}] Loss={avg_loss:.4f} Acc={epoch_acc:.4f}")
        
    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.2f}s")
    
    # Save training curves
    print("Generating training curves...")
    plot_training_curves(train_losses, train_accs, CONFIG["output_dir"])
    
    # 6. Evaluation
    print("\nEvaluating on Validation Set...")
    model.eval()
    y_true, y_score, y_pred = [], [], []
    
    with torch.no_grad():
        for x, y in val_dl:
            x = x.to(DEVICE)
            logits = model(x)
            probs = logits.softmax(1)[:, 1].cpu()
            preds = logits.argmax(1).cpu()

            y_true.extend(y.numpy())
            y_score.extend(probs.numpy())
            y_pred.extend(preds.numpy())
            
    # Save Model
    save_path = os.path.join(CONFIG["output_dir"], "scratch_cnn_mosq.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")
    
    # Generate Plots
    print("Generating ROC Curve...")
    plot_roc_curve(y_true, y_score, CONFIG["output_dir"])
    
    print("Generating Confusion Matrix...")
    classes = ["Aedes", "Culex"] # verify order if possible, usually alphabetical from ImageFolder
    cm = confusion_matrix(y_true, y_pred)
    print("\nCONFUSION MATRIX (Text):")
    print(cm)
    print(f"Aedes Correct: {cm[0][0]}, Aedes Wrong: {cm[0][1]}")
    print(f"Culex Wrong: {cm[1][0]}, Culex Correct: {cm[1][1]}")
    
    plot_confusion_matrix(y_true, y_pred, classes, CONFIG["output_dir"])
    
    # 7. GradCAM Generation
    if CONFIG["use_gradcam"]:
        print("Generating GradCAM samples...")
        model.eval()
        gradcam_dir = os.path.join(CONFIG["output_dir"], "gradcam")
        os.makedirs(gradcam_dir, exist_ok=True)
        
        # Pick 5 random images from validation set
        # We need to iterate again or cache them. Let's just grab a batch.
        inputs, labels = next(iter(val_dl))
        indices = random.sample(range(len(inputs)), min(5, len(inputs)))
        
        for idx in indices:
            img_tensor = inputs[idx].unsqueeze(0).to(DEVICE) # (1, 3, H, W)
            label = labels[idx].item()
            class_name = classes[label]
            
            # Predict
            out = model(img_tensor)
            pred = out.argmax(1).item()
            pred_name = classes[pred]
            
            # Generate CAM
            cam = apply_gradcam(model, img_tensor, target_class=pred)
            
            # Save
            fname = f"cam_{idx}_true_{class_name}_pred_{pred_name}.jpg"
            save_path = os.path.join(gradcam_dir, fname)
            label_text = f"True: {class_name}, Pred: {pred_name}"
            save_gradcam_image(img_tensor[0], cam, save_path, label_text)
            print(f"Saved {save_path}")

    # Text Evaluation
    evaluate(model, val_dl, DEVICE)

if __name__ == "__main__":
    train_model()
