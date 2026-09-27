"""Part D: aggregate metrics from all approaches into a comparison table
and print sampled misclassified examples from the best PLM approach for
error analysis."""

import numpy as np
import pandas as pd

from config import RESULTS_DIR
from data import load_splits
from utils import load_all_metrics

COMPARISON_TABLE_PATH = RESULTS_DIR / "comparison_table.csv"


def build_comparison_table() -> pd.DataFrame:
    all_metrics = load_all_metrics()
    rows = []
    for key, m in all_metrics.items():
        rows.append(
            {
                "approach": m.get("approach", key),
                "accuracy": round(m.get("accuracy", float("nan")), 4),
                "f1": round(m.get("f1", float("nan")), 4),
                "time_seconds": round(
                    m.get("train_time_seconds")
                    or m.get("train_embedding_extraction_time_seconds")
                    or m.get("api_call_time_seconds")
                    or float("nan"),
                    2,
                ),
                "peak_gpu_memory_mb": round(m.get("peak_gpu_memory_mb", float("nan")), 1)
                if "peak_gpu_memory_mb" in m
                else None,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(COMPARISON_TABLE_PATH, index=False)
    return df


def main():
    df = build_comparison_table()
    print("\n=== Comparison table (also saved to results/comparison_table.csv) ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
