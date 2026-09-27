# TP1 — Fake News Detection with a Pre-trained Language Model

Compares three approaches on the [WELFake](https://huggingface.co/datasets/davanstrien/WELFake) dataset:

- **A** — TF-IDF + Logistic Regression baseline
- **B** — Frozen DistilBERT used as a feature extractor + Logistic Regression
- **C** — Fine-tuned DistilBERT (Hugging Face `Trainer`)
- **Bonus** — Gemini (few-shot prompted, free tier) on a test subset

## Setup

Requires [`uv`](https://docs.astral.sh/uv/).

```powershell
uv sync
```

This creates `.venv` with Python 3.11 and installs all pinned dependencies, including a CUDA-enabled build of PyTorch (`torch==2.5.1+cu121`).

For the bonus part, copy `.env.example` to `.env` and fill in your own `GEMINI_API_KEY` (free key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey)).

## Reproducing the results

One command runs the whole pipeline in order (data → A → B → C → bonus → D):

```powershell
uv run python src/run_all.py
```

If `GEMINI_API_KEY` isn't set, the bonus step is skipped with a warning instead of failing the run. Part C (fine-tuning) is the slow part, ~15 minutes on a GPU.

To run a single stage instead (e.g. to only redo the baseline), call its script directly, in this order:

```powershell
uv run python src/data.py            # download WELFake, clean, and save a fixed stratified split
uv run python src/baseline.py        # Part A
uv run python src/frozen_plm.py      # Part B
uv run python src/finetune_plm.py    # Part C
uv run python src/bonus_llm.py       # Bonus (needs GEMINI_API_KEY)
uv run python src/compare.py         # Part D - aggregates everything into results/comparison_table.csv
```

Every script uses a fixed seed (`SEED = 42` in `src/config.py`) and reuses the exact same train/val/test split (cached in `data/`) so results are reproducible and comparable across approaches.

## Notebook

`notebook.ipynb` is a read-through summary of the whole project: it doesn't retrain anything, it loads the results the scripts above already produced (`results/metrics.json`, the confusion matrix images, `results/comparison_table.csv`) and walks through the dataset, the label-inversion bug found in the source data, all four approaches, and the final comparison/decision, in one place. Re-run it with:

```powershell
uv run jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
```

## Outputs

- `results/metrics.json` — raw metrics per approach
- `results/comparison_table.csv` — the Part D comparison table
- `results/confusion_matrices/*.png` — one confusion matrix per approach
- `models/distilbert_finetuned/` — the fine-tuned model from Part C
- `report/TP1_Basics_PLM.docx` — the filled compte-rendu
