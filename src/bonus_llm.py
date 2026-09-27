"""Bonus: few-shot prompt Gemini on a bounded test subset and compare
accuracy/F1/cost against the local PLM approaches.

Requires GEMINI_API_KEY in a .env file (see .env.example). Uses Gemini's
free tier (gemini-3.8-flash), so this costs nothing to run.

Progress is checkpointed to disk so a transient API error (e.g. a 503
"model overloaded") doesn't waste the whole run — re-running the script
resumes from where it left off.
"""

import json
import os
import re
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from config import BONUS_TEST_SUBSET_SIZE, RESULTS_DIR, SEED, set_seed
from data import load_splits
from utils import compute_metrics, plot_confusion_matrix, save_approach_metrics, timer

MODEL_NAME = "gemini-flash-lite-latest"
SECONDS_BETWEEN_CALLS = 4.5  # stay under the free tier's ~15 requests/minute limit
MAX_RETRIES = 5
CHECKPOINT_PATH = RESULTS_DIR / "bonus_llm_checkpoint.json"

FEW_SHOT_EXAMPLES = [
    (
        "Local Man Wins Lottery Twice In Same Week, Experts Baffled",
        "fake",
    ),
    (
        "Senate Passes Infrastructure Bill After Months Of Negotiation",
        "real",
    ),
    (
        "Scientists Discover Chocolate Cures All Diseases, Doctors Hate This",
        "fake",
    ),
]

SYSTEM_PROMPT = (
    "You are a fact-checking classifier. Given a news headline/article, "
    "decide if it is 'fake' or 'real'. Reply with exactly one word: fake or real."
)


def build_prompt(text: str) -> str:
    examples = "\n\n".join(
        f"Text: {ex_text}\nLabel: {ex_label}" for ex_text, ex_label in FEW_SHOT_EXAMPLES
    )
    return f"{examples}\n\nText: {text}\nLabel:"


def parse_label(response_text: str) -> int:
    match = re.search(r"\b(fake|real)\b", response_text.strip().lower())
    if not match:
        return 0
    return 0 if match.group(1) == "fake" else 1


def call_with_retry(client, prompt: str):
    delay = 5
    for attempt in range(MAX_RETRIES):
        try:
            return client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=20,
                    temperature=0,
                ),
            )
        except errors.ServerError:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"  transient server error, retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2


def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        return json.loads(CHECKPOINT_PATH.read_text())
    return {}


def save_checkpoint(predictions: dict) -> None:
    CHECKPOINT_PATH.write_text(json.dumps(predictions))


def main():
    load_dotenv()
    set_seed()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set. Copy .env.example to .env and fill it in.")

    client = genai.Client(api_key=api_key)

    _, _, test_df = load_splits()
    subset = test_df.sample(n=min(BONUS_TEST_SUBSET_SIZE, len(test_df)), random_state=SEED)

    predictions = load_checkpoint()
    n_already_done = len(predictions)
    if n_already_done:
        print(f"Resuming from checkpoint: {n_already_done}/{len(subset)} already done.")

    with timer() as t:
        for i, (_, row) in enumerate(subset.iterrows()):
            if str(i) in predictions:
                continue

            truncated_text = row["full_text"][:2000]
            prompt = build_prompt(truncated_text)

            response = call_with_retry(client, prompt)
            predicted_label = parse_label(response.text or "")

            predictions[str(i)] = {"true": int(row["label"]), "pred": predicted_label}
            save_checkpoint(predictions)

            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(subset)} done")

            time.sleep(SECONDS_BETWEEN_CALLS)
    api_time = t["seconds"]

    y_true = [predictions[str(i)]["true"] for i in range(len(subset))]
    y_pred = [predictions[str(i)]["pred"] for i in range(len(subset))]

    metrics = compute_metrics(y_true, y_pred)
    metrics["api_call_time_seconds"] = api_time
    metrics["n_examples"] = len(subset)
    metrics["approach"] = f"Bonus - {MODEL_NAME} few-shot prompting"

    plot_confusion_matrix(y_true, y_pred, "Bonus_llm_prompted")
    save_approach_metrics("Bonus_llm_prompted", metrics)

    CHECKPOINT_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
