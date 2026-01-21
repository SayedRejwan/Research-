import torch
import numpy as np

@torch.no_grad()
def evaluate_ood(model, loader):
    model.eval()
    conf = []

    for x,_ in loader:
        x = x.to(model.device)
        prob = torch.softmax(model(x),1)
        conf.extend(prob.max(1).values.cpu().numpy())

    conf = np.array(conf)
    print("\n[OOD]")
    print("Mean confidence:", conf.mean())
    print("Min confidence:", conf.min())
    print("Max confidence:", conf.max())
