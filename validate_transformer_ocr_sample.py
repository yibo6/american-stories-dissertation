import argparse
import csv
import random
from pathlib import Path

from filter_selected_sources_with_optional_ocr_model import (
    correct_texts_with_model_batch,
    correction_improves_text,
    load_ocr_model,
    rule_based_label,
    text_metrics,
)


METRIC_FIELDS = [
    "clean_word_count",
    "word_like_token_ratio",
    "damaged_token_ratio",
    "gibberish_like_token_ratio",
    "odd_case_token_ratio",
    "short_fragment_line_ratio",
    "common_function_word_ratio",
    "sentence_like_unit_count",
]


def year_from_path(path):
    return path.stem.split("_")[-1]


def reservoir_add(sample, item, sample_size, seen, rng):
    if len(sample) < sample_size:
        sample.append(item)
        return
    replace_at = rng.randrange(seen)
    if replace_at < sample_size:
        sample[replace_at] = item


def sample_matches(label, sample_mode):
    if sample_mode == "all":
        return True
    if sample_mode == "borderline":
        return label == "borderline"
    if sample_mode == "borderline_or_remove":
        return label in {"borderline", "remove"}
    raise ValueError(f"Unknown sample mode: {sample_mode}")


def load_year_sample(path, sample_size, sample_mode, rng):
    sample = []
    matching_seen = 0
    total_seen = 0

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_seen += 1
            text = row.get("text", "")
            original_metrics = text_metrics(text)
            original_label, original_reason = rule_based_label(original_metrics)
            if not sample_matches(original_label, sample_mode):
                continue

            matching_seen += 1
            item = {
                "row": row,
                "original_metrics": original_metrics,
                "original_label": original_label,
                "original_reason": original_reason,
            }
            reservoir_add(sample, item, sample_size, matching_seen, rng)

    return sample, total_seen, matching_seen


def metric_delta(corrected_metrics, original_metrics, field):
    return corrected_metrics[field] - original_metrics[field]


def classify_outcome(original_metrics, corrected_metrics, corrected_label):
    if corrected_label == "pass" and correction_improves_text(
        original_metrics, corrected_metrics
    ):
        return "improved_and_passed"

    word_like_delta = metric_delta(
        corrected_metrics, original_metrics, "word_like_token_ratio"
    )
    damaged_delta = metric_delta(
        corrected_metrics, original_metrics, "damaged_token_ratio"
    )
    gibberish_delta = metric_delta(
        corrected_metrics, original_metrics, "gibberish_like_token_ratio"
    )

    if word_like_delta > 0 or damaged_delta < 0 or gibberish_delta < 0:
        return "partly_improved_but_not_passed"
    if word_like_delta < 0 or damaged_delta > 0 or gibberish_delta > 0:
        return "worsened"
    return "no_clear_change"


def write_record(writer, year, index, item, corrected_text):
    row = item["row"]
    original_text = row.get("text", "")
    original_metrics = item["original_metrics"]
    corrected_metrics = text_metrics(corrected_text)
    corrected_label, corrected_reason = rule_based_label(corrected_metrics)
    outcome = classify_outcome(original_metrics, corrected_metrics, corrected_label)

    output = {
        "year": year,
        "sample_index": index,
        "date": row.get("date", ""),
        "publisher": row.get("publisher", ""),
        "headline": row.get("headline", ""),
        "original_label": item["original_label"],
        "original_reason": item["original_reason"],
        "corrected_label": corrected_label,
        "corrected_reason": corrected_reason,
        "validation_outcome": outcome,
        "original_text": original_text,
        "corrected_text": corrected_text,
    }

    for field in METRIC_FIELDS:
        output[f"original_{field}"] = original_metrics[field]
        output[f"corrected_{field}"] = corrected_metrics[field]
        output[f"delta_{field}"] = metric_delta(
            corrected_metrics, original_metrics, field
        )

    writer.writerow(output)
    return output


def summarise_year(year, total_seen, matching_seen, record_outputs):
    total_sampled = len(record_outputs)
    counts = {
        "improved_and_passed": 0,
        "partly_improved_but_not_passed": 0,
        "worsened": 0,
        "no_clear_change": 0,
    }
    for row in record_outputs:
        counts[row["validation_outcome"]] += 1

    def avg(field):
        if not record_outputs:
            return ""
        return sum(float(row[field]) for row in record_outputs) / len(record_outputs)

    return {
        "year": year,
        "total_records_in_file": total_seen,
        "records_matching_sample_mode": matching_seen,
        "sampled_records": total_sampled,
        **counts,
        "improvement_pass_rate": (
            counts["improved_and_passed"] / total_sampled if total_sampled else ""
        ),
        "avg_delta_word_like_token_ratio": avg("delta_word_like_token_ratio"),
        "avg_delta_damaged_token_ratio": avg("delta_damaged_token_ratio"),
        "avg_delta_gibberish_like_token_ratio": avg(
            "delta_gibberish_like_token_ratio"
        ),
        "avg_delta_short_fragment_line_ratio": avg(
            "delta_short_fragment_line_ratio"
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate transformer OCR correction on a random sample from each "
            "selected-source year file. This is for dissertation validation, "
            "not for creating the main usable corpus."
        )
    )
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start-year", type=int, default=1837)
    parser.add_argument("--end-year", type=int, default=1964)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument(
        "--sample-mode",
        choices=["all", "borderline", "borderline_or_remove"],
        default="borderline",
        help=(
            "Which records to sample before correction. 'borderline' is the "
            "most relevant mode for testing whether transformer correction can "
            "rescue moderately noisy OCR records."
        ),
    )
    parser.add_argument("--ocr-model", default="pykale/bart-base-ocr")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--progress-every-year", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_ocr_model(args.ocr_model)
    rng = random.Random(args.seed)

    record_fields = [
        "year",
        "sample_index",
        "date",
        "publisher",
        "headline",
        "original_label",
        "original_reason",
        "corrected_label",
        "corrected_reason",
        "validation_outcome",
        "original_text",
        "corrected_text",
    ]
    for field in METRIC_FIELDS:
        record_fields.extend(
            [f"original_{field}", f"corrected_{field}", f"delta_{field}"]
        )

    records_path = args.output_dir / "transformer_ocr_validation_records.csv"
    summary_path = args.output_dir / "transformer_ocr_validation_summary_by_year.csv"

    summary_rows = []
    with records_path.open("w", newline="", encoding="utf-8") as records_f:
        records_writer = csv.DictWriter(records_f, fieldnames=record_fields)
        records_writer.writeheader()

        for year in range(args.start_year, args.end_year + 1):
            input_path = args.input_dir / f"selected_sources_{year}.csv"
            if not input_path.exists():
                continue

            sample, total_seen, matching_seen = load_year_sample(
                input_path,
                args.sample_size,
                args.sample_mode,
                rng,
            )
            if not sample:
                summary_rows.append(
                    summarise_year(str(year), total_seen, matching_seen, [])
                )
                continue

            corrected_texts = correct_texts_with_model_batch(
                [item["row"].get("text", "") for item in sample],
                tokenizer,
                model,
                batch_size=args.batch_size,
            )
            record_outputs = []
            for index, (item, corrected_text) in enumerate(
                zip(sample, corrected_texts), start=1
            ):
                record_outputs.append(
                    write_record(
                        records_writer,
                        str(year),
                        index,
                        item,
                        corrected_text,
                    )
                )

            summary = summarise_year(
                str(year), total_seen, matching_seen, record_outputs
            )
            summary_rows.append(summary)
            if args.progress_every_year:
                print(
                    f"{year}: sampled={summary['sampled_records']}, "
                    f"improved_and_passed={summary['improved_and_passed']}",
                    flush=True,
                )

    if summary_rows:
        with summary_path.open("w", newline="", encoding="utf-8") as summary_f:
            writer = csv.DictWriter(summary_f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"Validation records: {records_path}")
    print(f"Summary by year: {summary_path}")


if __name__ == "__main__":
    main()
