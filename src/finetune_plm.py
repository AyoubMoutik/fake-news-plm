"""Part C: fine-tune DistilBERT on the task with the Hugging Face Trainer.

Uses fp16 + a held-out validation set each epoch to monitor overfitting
(early stopping on validation F1).
"""

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from config import MAX_SEQ_LENGTH, MODELS_DIR, PLM_MODEL_NAME, SEED, get_device, set_seed
from data import load_splits
from utils import compute_metrics as sklearn_metrics
from utils import plot_confusion_matrix, save_approach_metrics, timer

FINETUNED_DIR = MODELS_DIR / "distilbert_finetuned"


def to_hf_dataset(df, tokenizer):
    ds = Dataset.from_pandas(df.rename(columns={"full_text": "text"}))

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_SEQ_LENGTH)

    ds = ds.map(tokenize, batched=True, remove_columns=["text"])
    ds = ds.rename_column("label", "labels")
    return ds


def hf_compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return sklearn_metrics(labels, preds)


def main():
    set_seed()
    device = get_device()
    train_df, val_df, test_df = load_splits()

    tokenizer = AutoTokenizer.from_pretrained(PLM_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(PLM_MODEL_NAME, num_labels=2).to(device)

    train_ds = to_hf_dataset(train_df, tokenizer)
    val_ds = to_hf_dataset(val_df, tokenizer)
    test_ds = to_hf_dataset(test_df, tokenizer)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir=str(FINETUNED_DIR / "checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=500,
        fp16=device.type == "cuda",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=100,
        seed=SEED,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=hf_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    with timer() as t:
        trainer.train()
    train_time = t["seconds"]

    trainer.save_model(str(FINETUNED_DIR))
    tokenizer.save_pretrained(str(FINETUNED_DIR))

    predictions = trainer.predict(test_ds)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = test_df["label"]

    metrics = sklearn_metrics(y_true, y_pred)
    metrics["train_time_seconds"] = train_time
    metrics["approach"] = "C - Fine-tuned DistilBERT (HF Trainer)"
    if device.type == "cuda":
        metrics["peak_gpu_memory_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2

    plot_confusion_matrix(y_true, y_pred, "C_finetuned_plm")
    save_approach_metrics("C_finetuned_plm", metrics)


if __name__ == "__main__":
    main()
