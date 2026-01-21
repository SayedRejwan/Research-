import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

@torch.no_grad()
def ensemble_predict(models, loader, device):
    """
    Performs soft-voting ensemble prediction.
    Averages the probabilities from all models.
    """
    y_true = []
    y_probs = []

    # Ensure models are in eval mode
    for model in models:
        model.eval()
        model.to(device)

    print(f"Running ensemble inference with {len(models)} models...")

    for x, y in loader:
        x = x.to(device)

        probs_list = []
        for model in models:
            out = model(x)
            # Softmax to get probabilities
            probs_list.append(F.softmax(out, dim=1))

        # Average probabilities across models (Soft Voting)
        # stack: [num_models, batch_size, num_classes] -> mean(dim=0): [batch_size, num_classes]
        avg_prob = torch.mean(torch.stack(probs_list), dim=0)
        
        y_probs.append(avg_prob.cpu())
        y_true.append(y)

    y_probs = torch.cat(y_probs)
    y_true = torch.cat(y_true)

    preds = y_probs.argmax(1)

    acc = accuracy_score(y_true, preds)
    try:
        auc = roc_auc_score(y_true, y_probs[:, 1])
    except ValueError:
        auc = 0.0 # Handle case with only one class in test set

    return acc, auc, y_probs, y_true

@torch.no_grad()
def ensemble_predict_weighted(models, loader, device, weights=None):
    """
    Performs weighted soft-voting.
    weights: list of floats, same length as models.
    """
    y_true = []
    y_probs = []

    if weights is None:
        weights = [1.0] * len(models)
    
    # Normalize weights
    weights = torch.tensor(weights, device=device)
    weights = weights / weights.sum()

    for model in models:
        model.eval()
        model.to(device)

    for x, y in loader:
        x = x.to(device)

        probs_list = []
        for model in models:
            out = model(x)
            probs_list.append(F.softmax(out, dim=1))

        # Stack: [num_models, batch_size, num_classes]
        stacked_probs = torch.stack(probs_list)
        
        # Weighted Average
        # Multiply each model's probs by its weight
        # We need to broadcast weights: [num_models] -> [num_models, 1, 1]
        w_expanded = weights.view(-1, 1, 1)
        weighted_probs = stacked_probs * w_expanded
        
        # Sum across models
        avg_prob = weighted_probs.sum(dim=0)
        
        y_probs.append(avg_prob.cpu())
        y_true.append(y)

    y_probs = torch.cat(y_probs)
    y_true = torch.cat(y_true)

    preds = y_probs.argmax(1)
    acc = accuracy_score(y_true, preds)
    try:
        auc = roc_auc_score(y_true, y_probs[:, 1])
    except:
        auc = 0.0

    return acc, auc
