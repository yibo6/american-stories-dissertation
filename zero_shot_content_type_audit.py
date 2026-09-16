import argparse
import re
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm


LABELS = [
    "standard news article",
    "advertisement or commercial notice",
    "legal notice",
    "market or shipping list",
    "event or entertainment listing",
    "personal advice or opinion column",
]


NON_ARTICLE_LABELS = set(LABELS) - {"standard news article"}


def compact_text(value, max_chars):
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def classify_batch(classifier, texts, batch_size):
    rows = []
    for start in tqdm(range(0, len(texts), batch_size), desc="zero-shot batches"):
        batch = texts[start : start + batch_size]
        preds = classifier(
            batch,
            candidate_labels=LABELS,
            hypothesis_template="This newspaper text is a {}.",
            multi_label=False,
            truncation=True,
        )
        if isinstance(preds, dict):
            preds = [preds]
        for pred in preds:
            scores = dict(zip(pred["labels"], pred["scores"]))
            top_label = pred["labels"][0]
            top_score = float(pred["scores"][0])
            non_article_score = max(scores[label] for label in NON_ARTICLE_LABELS)
            rows.append(
                {
                    "zero_shot_top_label": top_label,
                    "zero_shot_top_score": top_score,
                    "zero_shot_standard_news_score": scores["standard news article"],
                    "zero_shot_non_article_score": non_article_score,
                    "zero_shot_is_non_article_like": int(
                        top_label in NON_ARTICLE_LABELS and top_score >= 0.50
                    ),
                    **{
                        f"score_{label.replace(' ', '_').replace('/', '_')}": score
                        for label, score in scores.items()
                    },
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Run a zero-shot content-type audit on American Stories features."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/processed/american_stories_features_1940s_sample/"
            "american_stories_article_features.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/american_stories_features_1940s_sample/"
            "zero_shot_content_type_audit.csv"
        ),
    )
    parser.add_argument(
        "--model",
        default="facebook/bart-large-mnli",
        help="Hugging Face zero-shot classification model.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional random sample size from the input file.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-chars", type=int, default=1400)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.sample_size:
        df = df.sample(min(len(df), args.sample_size), random_state=20260816)
    df = df.reset_index(drop=True)

    try:
        import torch
        from transformers import pipeline
    except ImportError as exc:
        raise SystemExit(
            "Missing package. Install dependencies with: python3 -m pip install transformers torch"
        ) from exc

    device = 0 if torch.cuda.is_available() else -1
    print("Zero-shot device:", "GPU" if device == 0 else "CPU")
    classifier = pipeline("zero-shot-classification", model=args.model, device=device)

    texts = (
        df.get("headline", "").fillna("").astype(str)
        + ". "
        + df["text"].fillna("").astype(str)
    ).map(lambda text: compact_text(text, args.max_chars)).tolist()

    audit = classify_batch(classifier, texts, batch_size=args.batch_size)
    out = pd.concat([df.reset_index(drop=True), audit], axis=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print("Saved:", args.output)
    print("Top label counts:")
    print(out["zero_shot_top_label"].value_counts().to_string())
    print("Non-article-like flagged:")
    print(int(out["zero_shot_is_non_article_like"].sum()), "of", len(out))


if __name__ == "__main__":
    main()
