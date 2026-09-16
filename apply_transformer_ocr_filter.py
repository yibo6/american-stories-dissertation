import argparse
import csv
import re
from pathlib import Path

from filter_selected_sources_with_optional_ocr_model import (
    correct_texts_with_model_batch,
    correction_improves_text,
    load_ocr_model,
    rule_based_label,
    text_metrics,
)


OUTPUT_FIELDS = ["date", "publisher", "headline", "text"]

CONTENT_LABELS = [
    "article-like newspaper report",
    "advertisement or commercial notice",
    "legal notice or public notice",
    "market shipping or timetable listing",
    "short fragment table or corrupted layout",
    "literary poem or entertainment listing",
]

ARTICLE_LIKE_LABEL = "article-like newspaper report"


def worth_transformer_correction(metrics):
    """
    Send only recoverable borderline records to the transformer.

    The model is useful for mild OCR problems, such as occasional broken words,
    odd capitalisation, or mixed characters. It is usually not worth running on
    texts that are too fragmented, too short, or already far from readable
    English, because those records are slow to process and rarely improve.
    """
    if metrics["clean_word_count"] < 120:
        return False
    if metrics["word_like_token_ratio"] < 0.82:
        return False
    if metrics["damaged_token_ratio"] > 0.05:
        return False
    if metrics["gibberish_like_token_ratio"] > 0.025:
        return False
    if metrics["short_fragment_line_ratio"] > 0.40:
        return False

    has_correctable_noise = (
        metrics["damaged_token_ratio"] > 0.005
        or metrics["gibberish_like_token_ratio"] > 0.002
        or metrics["odd_case_token_ratio"] > 0.005
        or metrics["word_like_token_ratio"] < 0.92
    )
    return has_correctable_noise


def initial_decision(text, use_ocr_model):
    """
    Decide whether to keep, drop, or send a text to transformer correction.

    This separates fast rule-based screening from the slow model step, so the
    model can process borderline records in batches. Borderline records are not
    kept as-is: they must be corrected by the transformer and then pass the same
    quality rules. This prevents visibly damaged OCR from entering the final
    usable dataset just because it was considered potentially recoverable.
    """
    metrics = text_metrics(text)
    label, _ = rule_based_label(metrics)

    if label == "pass":
        return "keep", metrics
    if label == "remove":
        return "drop", metrics
    if use_ocr_model and worth_transformer_correction(metrics):
        return "model", metrics
    return "drop", metrics


def write_kept_row(writer, row, text):
    writer.writerow(
        {
            "date": row.get("date", ""),
            "publisher": row.get("publisher", ""),
            "headline": row.get("headline", ""),
            "text": text,
        }
    )


def load_content_classifier(model_name):
    """
    Load an optional zero-shot content classifier.

    This is not an OCR correction model. It is only used to judge whether a
    retained text looks like an article-like newspaper record or more like a
    notice/listing/advertisement. It is intentionally optional because it is
    slower and more subjective than OCR/readability filtering.
    """
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device,
    )
    device_name = "cuda" if device == 0 else "cpu"
    print(f"Content classifier loaded on device: {device_name}", flush=True)
    return classifier


def is_article_like(
    headline,
    text,
    classifier,
    min_article_score,
):
    """
    Return True when the classifier thinks the text is article-like.

    To keep this conservative, the classifier only removes a record when a
    non-article label beats the article-like label by the confidence threshold.
    This avoids dropping borderline historical news prose too aggressively.
    """
    if classifier is None:
        return True

    combined = f"{headline or ''}\n{text or ''}"
    combined = re.sub(r"\s+", " ", combined).strip()
    combined = " ".join(combined.split()[:350])
    if not combined:
        return False

    result = classifier(combined, candidate_labels=CONTENT_LABELS, multi_label=False)
    top_label = result["labels"][0]
    top_score = float(result["scores"][0])

    if top_label == ARTICLE_LIKE_LABEL:
        return True
    return top_score < min_article_score


def process_year(
    input_path,
    output_dir,
    tokenizer,
    model,
    max_model_articles,
    use_ocr_model,
    content_classifier,
    max_classifier_articles,
    min_article_score,
    batch_size,
    progress_every,
):
    year = input_path.stem.split("_")[-1]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"usable_articles_transformer_{year}.csv"

    processed = 0
    kept = 0
    dropped = 0
    model_counter = 0
    classifier_counter = 0

    model_status = "with OCR transformer" if use_ocr_model else "rule-based only"
    print(f"{year}: starting {input_path} ({model_status})", flush=True)

    with input_path.open(newline="", encoding="utf-8") as inf, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as outf:
        reader = csv.DictReader(inf)
        writer = csv.DictWriter(outf, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()

        model_batch = []

        def flush_model_batch():
            nonlocal kept, dropped, model_counter, classifier_counter
            if not model_batch:
                return

            original_texts = [item["text"] for item in model_batch]
            corrected_texts = correct_texts_with_model_batch(
                original_texts,
                tokenizer,
                model,
                batch_size=batch_size,
            )
            for item, corrected_text in zip(model_batch, corrected_texts):
                corrected_metrics = text_metrics(corrected_text)
                corrected_label, _ = rule_based_label(corrected_metrics)
                if (
                    corrected_label != "pass"
                    or not correction_improves_text(item["metrics"], corrected_metrics)
                ):
                    dropped += 1
                    continue

                final_text = corrected_text
                if content_classifier is not None:
                    if (
                        max_classifier_articles
                        and classifier_counter >= max_classifier_articles
                    ):
                        pass
                    else:
                        classifier_counter += 1
                        if not is_article_like(
                            item["row"].get("headline", ""),
                            final_text,
                            content_classifier,
                            min_article_score,
                        ):
                            dropped += 1
                            continue
                write_kept_row(writer, item["row"], final_text)
                kept += 1

            model_counter += len(model_batch)
            model_batch.clear()

        for row in reader:
            processed += 1
            text = row.get("text", "")
            decision, metrics = initial_decision(text, use_ocr_model)

            if decision == "drop":
                dropped += 1
                continue

            if decision == "model":
                if max_model_articles and model_counter + len(model_batch) >= max_model_articles:
                    dropped += 1
                    continue
                else:
                    model_batch.append({"row": row, "text": text, "metrics": metrics})
                    if len(model_batch) >= batch_size:
                        flush_model_batch()
                    if progress_every and processed % progress_every == 0:
                        print(
                            f"{year}: processed={processed:,}, kept={kept:,}, "
                            f"dropped={dropped:,}, sent_to_model={model_counter:,}, "
                            f"pending_model_batch={len(model_batch):,}, "
                            f"sent_to_classifier={classifier_counter:,}",
                            flush=True,
                        )
                    continue

            final_text = text
            if content_classifier is not None:
                if (
                    max_classifier_articles
                    and classifier_counter >= max_classifier_articles
                ):
                    pass
                else:
                    classifier_counter += 1
                    if not is_article_like(
                        row.get("headline", ""),
                        final_text,
                        content_classifier,
                        min_article_score,
                    ):
                        dropped += 1
                        continue

            write_kept_row(writer, row, final_text)
            kept += 1

            if progress_every and processed % progress_every == 0:
                print(
                    f"{year}: processed={processed:,}, kept={kept:,}, "
                    f"dropped={dropped:,}, sent_to_model={model_counter:,}, "
                    f"sent_to_classifier={classifier_counter:,}",
                    flush=True,
                )

        flush_model_batch()

    print(
        f"{year}: finished. processed={processed:,}, kept={kept:,}, "
        f"dropped={dropped:,}, sent_to_model={model_counter:,}, "
        f"sent_to_classifier={classifier_counter:,}",
        flush=True,
    )
    print(f"Output: {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create clean four-column usable American Stories CSVs. "
            "Pass records are kept unchanged; severe bad OCR is dropped; "
            "borderline OCR records are corrected with a transformer model "
            "and kept only if correction improves text quality."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--ocr-model", default="pykale/bart-base-ocr")
    parser.add_argument(
        "--ocr-start-year",
        type=int,
        default=None,
        help=(
            "First year that should use transformer OCR correction. "
            "Leave empty to use the transformer for every processed year."
        ),
    )
    parser.add_argument(
        "--ocr-end-year",
        type=int,
        default=None,
        help=(
            "Last year that should use transformer OCR correction. "
            "Leave empty to use the transformer for every processed year."
        ),
    )
    parser.add_argument(
        "--content-classifier-model",
        default="",
        help=(
            "Optional zero-shot model for removing non-article-like records. "
            "Leave empty to skip content classification."
        ),
    )
    parser.add_argument(
        "--max-classifier-articles",
        type=int,
        default=0,
        help=(
            "Maximum kept articles to send to the content classifier per year. "
            "Use 0 for no limit. Only used when --content-classifier-model is set."
        ),
    )
    parser.add_argument(
        "--min-article-score",
        type=float,
        default=0.70,
        help=(
            "Only drop a record as non-article-like when the classifier's top "
            "non-article label reaches this confidence. Higher is more cautious."
        ),
    )
    parser.add_argument(
        "--max-model-articles",
        type=int,
        default=100,
        help=(
            "Maximum borderline articles to send to the model per year. "
            "Use 0 for no limit."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress after this many input articles. Use 0 to turn off.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help=(
            "Number of borderline articles corrected together in one transformer "
            "batch. Try 16 on a GPU; use 4 if memory is limited."
        ),
    )
    args = parser.parse_args()

    ocr_start_year = args.ocr_start_year or args.start_year
    ocr_end_year = args.ocr_end_year or args.end_year
    should_load_ocr_model = not (
        ocr_end_year < args.start_year or ocr_start_year > args.end_year
    )

    tokenizer = None
    model = None
    if should_load_ocr_model:
        tokenizer, model = load_ocr_model(args.ocr_model)

    content_classifier = None
    if args.content_classifier_model:
        content_classifier = load_content_classifier(args.content_classifier_model)

    for year in range(args.start_year, args.end_year + 1):
        input_path = args.input_dir / f"selected_sources_{year}.csv"
        if not input_path.exists():
            print(f"Skipping missing input: {input_path}")
            continue
        process_year(
            input_path=input_path,
            output_dir=args.output_dir,
            tokenizer=tokenizer,
            model=model,
            max_model_articles=args.max_model_articles,
            use_ocr_model=ocr_start_year <= year <= ocr_end_year,
            content_classifier=content_classifier,
            max_classifier_articles=args.max_classifier_articles,
            min_article_score=args.min_article_score,
            batch_size=args.batch_size,
            progress_every=args.progress_every,
        )


if __name__ == "__main__":
    main()
