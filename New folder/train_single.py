import torch
import torch.nn as nn
from torch.optim import Adam
import time

def train_model(model, loaders, device, epochs=10):
    """
    Trains a single model.
    """
    print(f"Training {model.__class__.__name__} on {device}...")
    model.to(device)
    opt = Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        correct = total = 0
        loss_sum = 0

        if "train" not in loaders:
            print("No training data found in loaders.")
            return model

        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()

            loss_sum += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        
        epoch_time = time.time() - start_time
        print(f"[Epoch {epoch+1}/{epochs}] Loss={loss_sum:.4f} Acc={correct/total:.4f} Time={epoch_time:.1f}s")
        
        # Optional: Validation loop could be added here

    return model
