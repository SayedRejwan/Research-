import torch
from tqdm import tqdm
from config import *

def train_model(model, loader, optimizer, criterion):
    model.train()
    correct = total = 0

    for x, y in tqdm(loader):
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    acc = correct / total
    print(f"[TRAIN] Accuracy={acc:.4f}")
