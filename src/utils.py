import json
import time
from contextlib import contextmanager

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from config import CONFUSION_DIR, METRICS_PATH


@contextmanager
def timer():
    start = time.perf_counter()
    result = {}
    yield result
    result["seconds"] = time.perf_counter() - start


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="binary"),
    }


def plot_confusion_matrix(y_true, y_pred, approach_name: str, labels=("fake", "real")) -> str:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — {approach_name}")
    fig.tight_layout()

    out_path = CONFUSION_DIR / f"{approach_name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return str(out_path)


def load_all_metrics() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {}


def save_approach_metrics(approach_key: str, metrics: dict) -> None:
    all_metrics = load_all_metrics()
    all_metrics[approach_key] = metrics
    METRICS_PATH.write_text(json.dumps(all_metrics, indent=2))
    print(f"[{approach_key}] " + json.dumps(metrics, indent=2))
