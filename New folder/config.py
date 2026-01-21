# config.py
import torch

DATA_ROOT = r"F:\MLMI_SPLIT"  # <-- this must contain train/val/test
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
