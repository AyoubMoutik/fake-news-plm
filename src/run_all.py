"""Run the full pipeline in one command: data -> A -> B -> C -> bonus -> compare.

The bonus step needs GEMINI_API_KEY (see .env.example); if it's missing,
this skips it with a warning instead of failing the whole run.
"""

import os

from dotenv import load_dotenv

import baseline
import bonus_llm
import compare
import data
import finetune_plm
import frozen_plm

load_dotenv()


def main():
    print("=== Data: downloading WELFake, cleaning, splitting ===")
    data.main()

    print("\n=== Part A: baseline (TF-IDF + logistic regression) ===")
    baseline.main()

    print("\n=== Part B: frozen PLM (feature extraction) ===")
    frozen_plm.main()

    print("\n=== Part C: fine-tuning (~15 min on a GPU) ===")
    finetune_plm.main()

    print("\n=== Bonus: LLM prompted ===")
    if os.environ.get("GEMINI_API_KEY"):
        bonus_llm.main()
    else:
        print("GEMINI_API_KEY not set, skipping the bonus step (see .env.example).")

    print("\n=== Part D: comparison ===")
    compare.main()


if __name__ == "__main__":
    main()
