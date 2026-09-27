import random
from pathlib import Path

import numpy as np
import torch

SEED = 42

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"
MODELS_DIR = ROOT_DIR / "models"
METRICS_PATH = RESULTS_DIR / "metrics.json"

HF_DATASET_ID = "davanstrien/WELFake"
PLM_MODEL_NAME = "distilbert-base-uncased"
MAX_SEQ_LENGTH = 256

TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1

BONUS_TEST_SUBSET_SIZE = 200


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


for _dir in (DATA_DIR, RESULTS_DIR, CONFUSION_DIR, MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
