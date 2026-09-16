import argparse
import csv
import re
from pathlib import Path

from filter_selected_sources_with_optional_ocr_model import (
    correct_text_with_model,
    correction_improves_text,
    load_ocr_model,
    rule_based_label,
    text_metrics,
)


def clean_excerpt(text, limit=900):
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:limit]


def iter_candidate_rows(input_path, max_articles):
    """
    Yield only records where OCR correction might matter.

    We skip clean pass records because the model has little to improve there.
    We also skip clearly severe removals unless they are close to borderline,
    because a transformer should not be trusted to reconstruct unreadable text.
    """
    yielded = 0
    with input_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            metrics = text_metrics(row.get("text", ""))
            label, reason = rule_based_label(metrics)
            if label == "pass":
                continue
            if label == "remove" and metrics["word_like_token_ratio"] < 0.55:
                continue
            yield row, metrics, label, reason
            yielded += 1
            if max_articles and yielded >= max_articles:
                return


def process_year(input_path, output_dir, tokenizer, model, max_articles):
    year_match = re.search(r"(\d{4})", input_path.name)
    year = year_match.group(1) if year_match else input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    examples_path = output_dir / f"ocr_model_examples_{year}.csv"
    summary_path = output_dir / f"ocr_model_summary_{year}.csv"

    total_tested = 0
    would_drop_before = 0
    keep_after_model = 0
    rescued_by_model = 0
    unchanged_remove = 0

    fields = [
        "date",
        "publisher",
        "headline",
        "before_rule_label",
        "before_reason",
        "after_rule_label",
        "after_reason",
        "model_result",
        "original_text",
        "corrected_text",
    ]

    with examples_path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()

        for row, before_metrics, before_label, before_reason in iter_candidate_rows(
            input_path, max_articles
        ):
            total_tested += 1
            if before_label != "pass":
                would_drop_before += 1

            corrected = correct_text_with_model(row.get("text", ""), tokenizer, model)
            after_metrics = text_metrics(corrected)
            after_label, after_reason = rule_based_label(after_metrics)

            improved = correction_improves_text(before_metrics, after_metrics)
            if improved:
                model_result = "keep_corrected_text"
                keep_after_model += 1
                if before_label != "pass":
                    rescued_by_model += 1
            else:
                model_result = "still_drop"
                unchanged_remove += 1

            writer.writerow(
                {
                    "date": row.get("date", ""),
                    "publisher": row.get("publisher", ""),
                    "headline": row.get("headline", ""),
                    "before_rule_label": before_label,
                    "before_reason": before_reason,
                    "after_rule_label": after_label,
                    "after_reason": after_reason,
                    "model_result": model_result,
                    "original_text": clean_excerpt(row.get("text", "")),
                    "corrected_text": clean_excerpt(corrected),
                }
            )

    drop_rate_before = would_drop_before / total_tested if total_tested else 0
    drop_rate_after = unchanged_remove / total_tested if total_tested else 0

    with summary_path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(
            out,
            fieldnames=[
                "year",
                "model_tested_articles",
                "would_drop_before_model",
                "keep_after_model",
                "rescued_by_model",
                "still_drop_after_model",
                "drop_rate_before_model",
                "drop_rate_after_model",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "year": year,
                "model_tested_articles": total_tested,
                "would_drop_before_model": would_drop_before,
                "keep_after_model": keep_after_model,
                "rescued_by_model": rescued_by_model,
                "still_drop_after_model": unchanged_remove,
                "drop_rate_before_model": round(drop_rate_before, 4),
                "drop_rate_after_model": round(drop_rate_after, 4),
            }
        )

    print(f"Examples: {examples_path}")
    print(f"Summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare historical OCR correction model outputs on selected sources."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--ocr-model", default="pykale/bart-base-ocr")
    parser.add_argument("--max-articles", type=int, default=100)
    args = parser.parse_args()

    tokenizer, model = load_ocr_model(args.ocr_model)

    for year in range(args.start_year, args.end_year + 1):
        input_path = args.input_dir / f"selected_sources_{year}.csv"
        if not input_path.exists():
            print(f"Skipping missing input: {input_path}")
            continue
        process_year(input_path, args.output_dir, tokenizer, model, args.max_articles)


if __name__ == "__main__":
    main()
