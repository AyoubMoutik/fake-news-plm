"""Part A (data half): load WELFake, clean, and produce a shared stratified
train/val/test split reused identically by every approach (A/B/C) so
comparisons are fair and there is no test leakage."""

import json

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from config import DATA_DIR, SEED, TEST_FRAC, TRAIN_FRAC, VAL_FRAC, HF_DATASET_ID, set_seed

SPLIT_PATHS = {
    "train": DATA_DIR / "train.parquet",
    "val": DATA_DIR / "val.parquet",
    "test": DATA_DIR / "test.parquet",
}
STATS_PATH = DATA_DIR / "stats.json"


def load_and_clean() -> pd.DataFrame:
    ds = load_dataset(HF_DATASET_ID, split="train")
    df = ds.to_pandas()

    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df["full_text"] = (df["title"] + " " + df["text"]).str.strip()

    # The HF dataset declares ClassLabel(names=["fake", "real"]) (0=fake, 1=real),
    # but this is inverted relative to actual content: ~61% of label-0 articles
    # carry genuine Reuters wire attribution ("(Reuters)") vs ~0.05% of label-1,
    # and manual inspection confirms label-0 reads as real wire copy while
    # label-1 reads as clickbait/tabloid style. Flip so 0=fake, 1=real matches
    # the names actually used everywhere else in this project (labels=("fake",
    # "real") in utils.plot_confusion_matrix, the few-shot bonus prompt, etc.).
    df["label"] = 1 - df["label"]

    before = len(df)
    df = df[df["full_text"].str.len() > 20]
    df = df.drop_duplicates(subset="full_text")
    after = len(df)
    print(f"Cleaned {before} -> {after} rows (removed empty/near-empty + duplicates)")

    return df[["full_text", "label"]].reset_index(drop=True)


def make_splits(df: pd.DataFrame):
    assert abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) < 1e-9

    train_df, temp_df = train_test_split(
        df, train_size=TRAIN_FRAC, stratify=df["label"], random_state=SEED
    )
    val_relative = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
    val_df, test_df = train_test_split(
        temp_df, train_size=val_relative, stratify=temp_df["label"], random_state=SEED
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_splits(train_df, val_df, test_df) -> None:
    train_df.to_parquet(SPLIT_PATHS["train"])
    val_df.to_parquet(SPLIT_PATHS["val"])
    test_df.to_parquet(SPLIT_PATHS["test"])

    stats = {}
    for name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        counts = split_df["label"].value_counts().to_dict()
        stats[name] = {"n": len(split_df), "label_counts": {str(k): int(v) for k, v in counts.items()}}
    STATS_PATH.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def load_splits():
    return (
        pd.read_parquet(SPLIT_PATHS["train"]),
        pd.read_parquet(SPLIT_PATHS["val"]),
        pd.read_parquet(SPLIT_PATHS["test"]),
    )


def main():
    set_seed()
    df = load_and_clean()
    train_df, val_df, test_df = make_splits(df)

    train_ids = set(train_df["full_text"])
    val_ids = set(val_df["full_text"])
    test_ids = set(test_df["full_text"])
    assert not (train_ids & val_ids), "train/val overlap detected"
    assert not (train_ids & test_ids), "train/test overlap detected"
    assert not (val_ids & test_ids), "val/test overlap detected"

    save_splits(train_df, val_df, test_df)
    print("No train/val/test overlap. Splits saved to", DATA_DIR)


if __name__ == "__main__":
    main()
