import argparse
import csv
import hashlib
import re
import unicodedata
from pathlib import Path


STANDARD_PUNCTUATION = set(".,;:!?'-\"()[]{}$/%&@#+*=<>_")
VOWELS = set("aeiouy")
COMMON_FUNCTION_WORDS = {
    "the",
    "of",
    "and",
    "to",
    "in",
    "a",
    "is",
    "that",
    "for",
    "it",
    "as",
    "with",
    "was",
    "on",
    "be",
    "by",
    "at",
    "from",
    "or",
    "an",
    "this",
    "which",
    "are",
    "were",
    "his",
    "her",
    "their",
    "not",
    "have",
    "had",
}


def normalise_for_dedup(text):
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def first_alpha_is_lowercase(text):
    match = re.search(r"[A-Za-z]", text or "")
    return bool(match and match.group(0).islower())


def text_metrics(text):
    """
    Compute lightweight OCR/readability indicators.

    These metrics do not try to understand article meaning. They only ask
    whether the text still looks like analysable English newspaper text.
    """
    text = text or ""
    chars = [char for char in text if not char.isspace()]
    tokens = re.findall(r"\S+", text)
    letter_tokens = re.findall(r"[A-Za-z]+", text)
    cleaned_tokens = [re.sub(r"[^A-Za-z]", "", token) for token in tokens]
    cleaned_tokens = [token for token in cleaned_tokens if token]

    word_like_tokens = [
        token
        for token in tokens
        if re.fullmatch(
            r"[A-Za-z][A-Za-z'’-]*[A-Za-z]|[A-Za-z]",
            token.strip(".,;:!?()[]{}\"'"),
        )
    ]

    alpha_count = sum(char.isalpha() for char in chars)
    unusual_count = sum(
        not (char.isalnum() or char.isspace() or char in STANDARD_PUNCTUATION)
        for char in text
    )

    single_char_tokens = sum(len(token) == 1 for token in cleaned_tokens)
    short_alpha_tokens = sum(1 <= len(token) <= 2 for token in cleaned_tokens)
    long_tokens = sum(len(re.sub(r"[^A-Za-z]", "", token)) >= 25 for token in tokens)
    consonant_only_tokens = sum(
        len(token) >= 4 and not any(char in VOWELS for char in token.lower())
        for token in letter_tokens
    )
    repeated_runs = len(re.findall(r"(.)\1{5,}", text))

    damaged_tokens = 0
    gibberish_like_tokens = 0
    odd_case_tokens = 0
    for token in tokens:
        stripped = token.strip(".,;:!?()[]{}\"'")
        letters = sum(char.isalpha() for char in stripped)
        digits = sum(char.isdigit() for char in stripped)
        unusual = sum(not (char.isalnum() or char in "'’-") for char in stripped)
        if letters and (digits or unusual) and len(stripped) >= 3:
            damaged_tokens += 1

        word = re.sub(r"[^A-Za-z]", "", stripped)
        lower_word = word.lower()
        if len(lower_word) >= 5:
            vowel_count = sum(char in VOWELS for char in lower_word)
            rare_letter_count = sum(char in "jqxz" for char in lower_word)
            if vowel_count == 0 or rare_letter_count >= 2:
                gibberish_like_tokens += 1
            if re.search(r"[a-z][A-Z]|[A-Z][a-z][A-Z]", word):
                odd_case_tokens += 1

    common_function_words = sum(
        token.lower() in COMMON_FUNCTION_WORDS for token in cleaned_tokens
    )
    sentence_like_units = [
        unit
        for unit in re.split(r"[.!?]+", text)
        if len(re.findall(r"[A-Za-z]+", unit)) >= 8
    ]
    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    short_fragment_lines = [
        line for line in nonempty_lines if len(re.findall(r"[A-Za-z]+", line)) <= 3
    ]

    return {
        "clean_word_count": len(cleaned_tokens),
        "first_alpha_is_lowercase": int(first_alpha_is_lowercase(text)),
        "alpha_ratio": ratio(alpha_count, len(chars)),
        "unusual_char_ratio": ratio(unusual_count, max(len(text), 1)),
        "word_like_token_ratio": ratio(len(word_like_tokens), len(tokens)),
        "damaged_token_ratio": ratio(damaged_tokens, len(tokens)),
        "gibberish_like_token_ratio": ratio(gibberish_like_tokens, len(tokens)),
        "odd_case_token_ratio": ratio(odd_case_tokens, len(tokens)),
        "single_char_token_ratio": ratio(single_char_tokens, len(cleaned_tokens)),
        "short_alpha_token_ratio": ratio(short_alpha_tokens, len(cleaned_tokens)),
        "long_token_ratio": ratio(long_tokens, len(tokens)),
        "consonant_only_token_ratio": ratio(consonant_only_tokens, len(letter_tokens)),
        "common_function_word_ratio": ratio(
            common_function_words, len(cleaned_tokens)
        ),
        "sentence_like_unit_count": len(sentence_like_units),
        "short_fragment_line_ratio": ratio(
            len(short_fragment_lines), len(nonempty_lines)
        ),
        "repeated_char_runs": repeated_runs,
    }


def rule_based_label(metrics):
    """
    Split records into three groups:

    remove:
        Severe OCR/segmentation damage. These records are likely not useful even
        with transformer correction.
    borderline:
        Human-readable but noisy text. These are candidates for transformer OCR
        correction.
    pass:
        Readable enough to keep without correction.
    """
    hard_remove_rules = [
        (metrics["clean_word_count"] < 100, "too_short_under_100_clean_words"),
        (metrics["first_alpha_is_lowercase"] == 1, "likely_starts_mid_sentence"),
        (metrics["alpha_ratio"] < 0.70, "low_alphabetic_content"),
        (metrics["word_like_token_ratio"] < 0.70, "too_few_word_like_tokens"),
        (metrics["damaged_token_ratio"] > 0.08, "too_many_damaged_tokens"),
        (metrics["gibberish_like_token_ratio"] > 0.04, "too_many_gibberish_tokens"),
        (metrics["odd_case_token_ratio"] > 0.05, "too_much_mixed_capitalisation"),
        (metrics["short_fragment_line_ratio"] > 0.50, "too_many_fragmented_lines"),
        (
            metrics["sentence_like_unit_count"] == 0
            and metrics["clean_word_count"] >= 100,
            "no_sentence_like_structure",
        ),
    ]
    for failed, reason in hard_remove_rules:
        if failed:
            return "remove", reason

    borderline_rules = [
        (metrics["word_like_token_ratio"] < 0.92, "some_non_word_tokens"),
        (metrics["damaged_token_ratio"] > 0.015, "some_damaged_tokens"),
        (metrics["gibberish_like_token_ratio"] > 0.008, "some_gibberish_tokens"),
        (metrics["odd_case_token_ratio"] > 0.01, "some_mixed_capitalisation"),
        (metrics["short_alpha_token_ratio"] > 0.16, "many_short_word_fragments"),
        (metrics["common_function_word_ratio"] < 0.16, "few_common_function_words"),
        (metrics["short_fragment_line_ratio"] > 0.30, "some_fragmented_lines"),
        (metrics["repeated_char_runs"] >= 2, "repeated_character_runs"),
    ]
    for failed, reason in borderline_rules:
        if failed:
            return "borderline", reason

    return "pass", ""


def load_ocr_model(model_name):
    """
    Load a Hugging Face seq2seq model only when requested.

    Suggested models to test:
    - pykale/bart-base-ocr
    - PleIAs/OCRonos-Vintage
    """
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()
    print(f"OCR model loaded on device: {device}", flush=True)
    return tokenizer, model


def correct_text_with_model(text, tokenizer, model, max_words=450):
    """
    Correct only a short/pilot chunk. Full-article correction should be tested
    carefully before being used at scale, because OCR models may alter wording.
    """
    import torch

    words = re.findall(r"\S+", text or "")
    chunk = " ".join(words[:max_words])
    inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_length=512, num_beams=2)
    corrected = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if len(words) > max_words:
        corrected = corrected + " " + " ".join(words[max_words:])
    return corrected


def correct_texts_with_model_batch(texts, tokenizer, model, max_words=450, batch_size=8):
    """
    Correct several borderline OCR texts in one model call.

    Batch processing is much faster than calling the transformer once per
    article, especially on a GPU. The model still only sees the first
    max_words words of each article; the remaining text is appended unchanged.
    """
    import torch

    corrected_texts = []
    device = next(model.parameters()).device

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        chunks = []
        tails = []
        for text in batch_texts:
            words = re.findall(r"\S+", text or "")
            chunks.append(" ".join(words[:max_words]))
            tails.append(" ".join(words[max_words:]))

        inputs = tokenizer(
            chunks,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_length=512, num_beams=1)
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for corrected, tail in zip(decoded, tails):
            corrected = corrected.strip()
            if tail:
                corrected = corrected + " " + tail
            corrected_texts.append(corrected)

    return corrected_texts


def correction_improves_text(original_metrics, corrected_metrics):
    """
    A conservative acceptance test for transformer correction.

    Keep the corrected version only if it reduces OCR noise and still passes the
    same hard quality rules. This avoids using correction when the model makes a
    damaged text look fluent by changing too much.
    """
    corrected_label, _ = rule_based_label(corrected_metrics)
    if corrected_label == "remove":
        return False
    if corrected_metrics["clean_word_count"] < 100:
        return False

    noise_improved = (
        corrected_metrics["damaged_token_ratio"]
        < original_metrics["damaged_token_ratio"]
        or corrected_metrics["gibberish_like_token_ratio"]
        < original_metrics["gibberish_like_token_ratio"]
        or corrected_metrics["word_like_token_ratio"]
        > original_metrics["word_like_token_ratio"]
    )
    return noise_improved


def process_file(input_path, output_dir, tokenizer=None, model=None, max_model_articles=0):
    year_match = re.search(r"(\d{4})", input_path.name)
    year = year_match.group(1) if year_match else input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    usable_path = output_dir / f"usable_articles_{year}.csv"
    audit_path = output_dir / f"filtering_audit_{year}.csv"

    seen_hashes = set()
    corrected_count = 0

    usable_fields = ["date", "publisher", "headline", "text"]
    audit_fields = [
        "date",
        "publisher",
        "headline",
        "rule_label",
        "rule_reason",
        "final_decision",
        "used_transformer_correction",
        "text_excerpt",
        "corrected_excerpt",
        "clean_word_count",
        "word_like_token_ratio",
        "damaged_token_ratio",
        "gibberish_like_token_ratio",
        "short_fragment_line_ratio",
    ]

    with input_path.open(newline="", encoding="utf-8") as inf, usable_path.open(
        "w", newline="", encoding="utf-8"
    ) as usable_f, audit_path.open("w", newline="", encoding="utf-8") as audit_f:
        reader = csv.DictReader(inf)
        usable_writer = csv.DictWriter(usable_f, fieldnames=usable_fields)
        audit_writer = csv.DictWriter(audit_f, fieldnames=audit_fields)
        usable_writer.writeheader()
        audit_writer.writeheader()

        for row in reader:
            text = row.get("text", "")
            metrics = text_metrics(text)
            label, reason = rule_based_label(metrics)
            final_text = text
            final_decision = label
            corrected_excerpt = ""
            used_correction = 0

            if label == "borderline" and tokenizer is not None and model is not None:
                if max_model_articles <= 0 or corrected_count < max_model_articles:
                    corrected = correct_text_with_model(text, tokenizer, model)
                    corrected_metrics = text_metrics(corrected)
                    corrected_excerpt = re.sub(r"\s+", " ", corrected).strip()[:350]
                    corrected_count += 1
                    if correction_improves_text(metrics, corrected_metrics):
                        final_text = corrected
                        final_decision = "pass_after_transformer_correction"
                        used_correction = 1

            normalised = normalise_for_dedup(final_text)
            text_hash = hashlib.blake2b(
                normalised.encode("utf-8"), digest_size=16
            ).hexdigest()
            if text_hash in seen_hashes:
                final_decision = "remove_duplicate"
            else:
                seen_hashes.add(text_hash)

            if final_decision in {"pass", "pass_after_transformer_correction"}:
                usable_writer.writerow(
                    {
                        "date": row.get("date", ""),
                        "publisher": row.get("publisher", ""),
                        "headline": row.get("headline", ""),
                        "text": final_text,
                    }
                )

            audit_writer.writerow(
                {
                    "date": row.get("date", ""),
                    "publisher": row.get("publisher", ""),
                    "headline": row.get("headline", ""),
                    "rule_label": label,
                    "rule_reason": reason,
                    "final_decision": final_decision,
                    "used_transformer_correction": used_correction,
                    "text_excerpt": re.sub(r"\s+", " ", text).strip()[:350],
                    "corrected_excerpt": corrected_excerpt,
                    **{
                        key: metrics[key]
                        for key in [
                            "clean_word_count",
                            "word_like_token_ratio",
                            "damaged_token_ratio",
                            "gibberish_like_token_ratio",
                            "short_fragment_line_ratio",
                        ]
                    },
                }
            )

    print(f"Wrote {usable_path}")
    print(f"Wrote {audit_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Filter selected American Stories source CSVs with rule-based "
            "OCR screening and optional transformer OCR correction for "
            "borderline records."
        )
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-year", type=int, default=1837)
    parser.add_argument("--end-year", type=int, default=1964)
    parser.add_argument(
        "--ocr-model",
        default="",
        help=(
            "Optional Hugging Face OCR correction model. Example: "
            "pykale/bart-base-ocr"
        ),
    )
    parser.add_argument(
        "--max-model-articles",
        type=int,
        default=200,
        help="Maximum borderline records to correct per year during pilot tests.",
    )
    args = parser.parse_args()

    tokenizer = None
    model = None
    if args.ocr_model:
        tokenizer, model = load_ocr_model(args.ocr_model)

    for year in range(args.start_year, args.end_year + 1):
        input_path = args.input_dir / f"selected_sources_{year}.csv"
        if not input_path.exists():
            print(f"Skipping missing input: {input_path}")
            continue
        process_file(
            input_path=input_path,
            output_dir=args.output_dir,
            tokenizer=tokenizer,
            model=model,
            max_model_articles=args.max_model_articles,
        )


if __name__ == "__main__":
    main()
