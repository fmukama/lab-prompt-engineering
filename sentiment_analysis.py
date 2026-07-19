"""
Multi-Stage Sentiment Analysis Pipeline -- Stage 2 (IDE AI output)

Consumes the raw scraped snippets produced upstream by a chat AI
(saved to scraped_data.json), filters them to a single target entity,
applies a lightweight rule-based sentiment classifier, and prints a
metrics summary.

Run:  python sentiment_analysis.py
"""

import json
import sys
from pathlib import Path

# --- Configuration -----------------------------------------------------------
# Everything the pipeline targets lives here, so retargeting it to a new
# entity or vocabulary is a one-line edit -- no logic changes required.
DATA_FILE = "scraped_data.json"
TARGET_ENTITY = "Lionel Messi"

POSITIVE_WORDS = ["magic", "gifted", "praise", "generosity", "admiration", "humility"]
NEGATIVE_WORDS = ["frustrating", "cheaply", "flattest", "fatigue"]


def load_records(path):
    """Load the scraped JSON array from disk, failing clearly if it is missing."""
    file_path = Path(__file__).with_name(path)
    if not file_path.exists():
        sys.exit(f"Error: could not find '{path}' next to this script.")
    with file_path.open(encoding="utf-8") as f:
        return json.load(f)


def score_snippet(text):
    """Return (score, positive_hits, negative_hits) for one snippet.

    Matching is case-insensitive and counts every keyword occurrence, so a
    snippet that leans heavily positive or negative scores accordingly.
    """
    lowered = text.lower()
    positive_hits = sum(lowered.count(word.lower()) for word in POSITIVE_WORDS)
    negative_hits = sum(lowered.count(word.lower()) for word in NEGATIVE_WORDS)
    return positive_hits - negative_hits, positive_hits, negative_hits


def classify(score):
    """Map a numeric score to a sentiment label."""
    if score > 0:
        return "Positive"
    if score < 0:
        return "Negative"
    return "Neutral"


def main():
    records = load_records(DATA_FILE)

    # Stage 2a -- filter: keep only articles that mention the target entity.
    mentions = [
        r for r in records
        if TARGET_ENTITY in r.get("content_snippet", "")
    ]

    positive = negative = neutral = 0

    print(f"Target entity : {TARGET_ENTITY}")
    print(f"Records loaded: {len(records)}\n")
    print("Per-article breakdown")
    print("-" * 64)

    # Stage 2b -- score each matching snippet and tally the labels.
    for r in mentions:
        score, pos, neg = score_snippet(r["content_snippet"])
        label = classify(score)
        if label == "Positive":
            positive += 1
        elif label == "Negative":
            negative += 1
        else:
            neutral += 1
        print(
            f"[{r['source_site']:<7}] {r['article_id']}: "
            f"score {score:+d}  (positive {pos} / negative {neg})  -> {label}"
        )

    # Stage 2c -- report the aggregate metrics.
    print("-" * 64)
    print("\nSUMMARY")
    print("=" * 64)
    print(f"Total Mentions found : {len(mentions)}")
    print(f"Positive articles    : {positive}")
    print(f"Negative articles    : {negative}")
    if neutral:
        print(f"Neutral articles     : {neutral}")


if __name__ == "__main__":
    main()
