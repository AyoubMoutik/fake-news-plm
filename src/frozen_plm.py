"""Part B: frozen PLM (DistilBERT) used as a feature extractor.

Mean-pooled last-hidden-state embeddings are extracted on GPU in batches,
then a Logistic Regression classifier is trained on top of them.
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer

from config import MAX_SEQ_LENGTH, PLM_MODEL_NAME, SEED, get_device, set_seed
from data import load_splits
from utils import compute_metrics, plot_confusion_matrix, save_approach_metrics, timer

BATCH_SIZE = 32


@torch.no_grad()
def extract_embeddings(texts, tokenizer, model, device) -> np.ndarray:
    model.eval()
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt",
        ).to(device)

        outputs = model(**encoded)
        last_hidden = outputs.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).expand(last_hidden.size()).float()
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        mean_pooled = summed / counts

        all_embeddings.append(mean_pooled.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def main():
    set_seed()
    device = get_device()
    train_df, val_df, test_df = load_splits()

    tokenizer = AutoTokenizer.from_pretrained(PLM_MODEL_NAME)
    model = AutoModel.from_pretrained(PLM_MODEL_NAME).to(device)
    for param in model.parameters():
        param.requires_grad = False

    with timer() as t:
        X_train = extract_embeddings(train_df["full_text"].tolist(), tokenizer, model, device)
    extraction_time = t["seconds"]
    X_test = extract_embeddings(test_df["full_text"].tolist(), tokenizer, model, device)

    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(X_train, train_df["label"])

    y_pred = clf.predict(X_test)
    y_true = test_df["label"]

    metrics = compute_metrics(y_true, y_pred)
    metrics["train_embedding_extraction_time_seconds"] = extraction_time
    metrics["approach"] = "B - Frozen DistilBERT (feature extractor) + Logistic Regression"
    if device.type == "cuda":
        metrics["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2

    plot_confusion_matrix(y_true, y_pred, "B_frozen_plm")
    save_approach_metrics("B_frozen_plm", metrics)


if __name__ == "__main__":
    main()
