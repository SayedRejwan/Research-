import torch

# =========================
# PATHS
# =========================

DATA_ROOT = r"F:\MLMI_SPLIT"      # train/val/test
OOD_ROOT  = r"G:\Downloads\archive\Mosquito"       # public insect images

# =========================
# TRAINING
# =========================

IMG_SIZE = 260
BATCH_SIZE = 16
EPOCHS = 3
LR = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 2

CLASS_NAMES = [
    "Aedes aegypti",
    "Culex quinquefasciatus"
]
