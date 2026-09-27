"""One-off helper: print a few misclassified test examples from the
fine-tuned model, for the error-analysis question in the report."""

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import MAX_SEQ_LENGTH, MODELS_DIR, get_device
from data import load_splits

FINETUNED_DIR = MODELS_DIR / "distilbert_finetuned"


def main():
    device = get_device()
    _, _, test_df = load_splits()

    tokenizer = AutoTokenizer.from_pretrained(str(FINETUNED_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(FINETUNED_DIR)).to(device)
    model.eval()

    texts = test_df["full_text"].tolist()
    labels = test_df["label"].tolist()

    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            batch = texts[i : i + 64]
            enc = tokenizer(
                batch, padding=True, truncation=True, max_length=MAX_SEQ_LENGTH, return_tensors="pt"
            ).to(device)
            logits = model(**enc).logits
            preds.extend(np.argmax(logits.cpu().numpy(), axis=-1).tolist())

    errors = [
        (t, y, p) for t, y, p in zip(texts, labels, preds) if y != p
    ]
    print(f"{len(errors)} misclassified examples out of {len(texts)}")
    for text, true_label, pred_label in errors[:5]:
        label_name = {0: "fake", 1: "real"}
        print(f"\ntrue={label_name[true_label]} pred={label_name[pred_label]}")
        print(text[:400])


if __name__ == "__main__":
    main()
