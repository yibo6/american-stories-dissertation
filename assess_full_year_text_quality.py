import argparse
import csv
import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


STANDARD_PUNCTUATION = set(".,;:!?'-\"()[]{}$/%&@#+*=<>_")
VOWELS = set("aeiouy")
SYSTEM_WORD_LIST = Path("/usr/share/dict/words")
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


def load_reference_words():
    """
    Load a broad local English word list if it is available.

    This is not used as a strict modern-spelling filter. It is only a safety
    check for records where many alphabetic tokens look word-like but the text
    is still incoherent, e.g. OCR fragments such as "DTRD", "rense", "tsgc".
    """
    words = set(COMMON_FUNCTION_WORDS)
    if SYSTEM_WORD_LIST.exists():
        with SYSTEM_WORD_LIST.open(encoding="utf-8", errors="ignore") as f:
            for line in f:
                word = line.strip().lower()
                if re.fullmatch(r"[a-z][a-z'-]*", word):
                    words.add(word)
    return words


REFERENCE_WORDS = load_reference_words()


CONTENT_TERMS = {
    "advertisement": [
        "advertisement",
        "classified",
        "for sale",
        "for rent",
        "wanted",
        "apply to",
        "auction",
        "auctioneer",
        "auctioneers",
        "will sell",
        "sale of horses",
        "sale of mules",
        "horses and mules",
        "furniture",
        "piano",
        "piano forte",
        "wardrobes",
        "sofas",
        "chairs",
        "tables",
        "reward",
        "terms cash",
        "subscriber",
        "cash",
        "for cash",
        "cash paid",
        "best prices",
        "immediate sale",
        "bargain",
        "reasonable rates",
        "cash for your car",
        "will pay",
        "new or used",
        "patent",
        "manufacturing company",
        "notice hereby given",
        "directors",
        "elected president",
        "manufactured",
        "wholesale",
        "retail",
        "for sale wholesale",
        "sold at",
        "sold by",
        "store",
        "importers",
        "go and see",
        "opened",
        "opening",
        "establishment",
        "patronage",
        "museum",
        "concert",
        "admitted",
        "vocal and instrumental",
        "natural history",
        "wholesale and retail",
        "certificates",
        "testimonials",
        "bring your title",
        "drive in",
        "dealer",
        "motors",
        "pontiac",
        "plymouth",
        "chevrolet",
        "appraisal",
        "phone",
        "call",
        "telephone",
        "write to",
        "inquire",
        "inquire at",
        "orders",
        "orders left",
        "delivered",
        "subscriber",
        "subscribers",
        "received from",
        "assortment",
        "agent",
        "agents",
        "warranted",
        "room",
        "office",
        "address",
        "st.",
        "ave.",
        "nw",
        "ne",
        "sw",
        "se",
        "for hire",
        "available immediately",
        "hauling",
        "contract work",
        "insured",
        "rates",
        "vans",
        "pickups",
        "trucks",
        "automobile",
        "auto",
        "truck",
        "car",
        "cars",
    ],
    "legal_notice": [
        "notice is hereby given",
        "estate of",
        "probate",
        "sheriff",
        "court of",
        "decree",
        "summons",
        "administrator",
        "executor",
    ],
    "market_finance": [
        "market",
        "prices",
        "cotton",
        "wheat",
        "flour",
        "stock exchange",
        "bonds",
        "shares",
        "cattle",
    ],
    "politics_government": [
        "president",
        "congress",
        "senate",
        "governor",
        "election",
        "legislature",
        "minister",
        "cabinet",
        "tariff",
    ],
    "health_wellbeing": [
        "health",
        "hospital",
        "disease",
        "fever",
        "doctor",
        "physician",
        "mortality",
        "death",
        "insane",
        "mental",
    ],
    "employment_wages": [
        "employment",
        "unemployed",
        "wages",
        "salary",
        "strike",
        "labor",
        "labour",
        "workers",
        "factory",
    ],
    "crime_safety": [
        "crime",
        "murder",
        "robbery",
        "burglary",
        "forgery",
        "charged",
        "arrest",
        "arrested",
        "police",
        "prison",
        "prisoners",
        "trial",
        "violence",
        "accident",
    ],
    "housing_poverty": [
        "housing",
        "rent",
        "homeless",
        "poor",
        "poverty",
        "charity",
        "relief",
        "tenement",
    ],
    "education_family": [
        "school",
        "education",
        "teacher",
        "children",
        "family",
        "marriage",
        "divorce",
    ],
}


MEDICAL_AD_TERMS = [
    "gravel",
    "venereal",
    "delicate disease",
    "delicate diseases",
    "delicate complaints",
    "private diseases",
    "delicate and private diseases",
    "secret habit",
    "destructive habit",
    "royal college",
    "surgeons",
    "consulted",
    "consultation",
    "remedy",
    "specific",
    "medicine",
    "solvent medicine",
    "personal attendance",
    "advice",
    "certificates",
    "satisfactory references",
    "superior practice",
    "obtain relief",
    "bottle",
    "bottles",
    "directions",
    "price",
    "genuine",
    "sold only",
    "drug store",
    "druggist",
    "dispensary",
    "peck slip",
    "red drop",
    "syrup",
    "diarrhea",
    "dysentery",
    "cholera morbus",
    "summer complaints",
    "safe and reliable",
    "immediate relief",
    "no family should be without",
    "cough lozenges",
    "cough",
    "consumption",
    "invigorator",
    "uterine",
    "abdominal supporter",
    "utero abdominal supporter",
    "falling of the womb",
    "prolapsus",
    "visceral displacement",
    "preserving and restoring",
    "restoring human hair",
    "restoring hum n hair",
    "restoring him n hair",
    "medicated compound",
    "trichopher",
    "trieopher",
    "baldness",
    "clarified essence",
    "hoarhound",
    "hoarhound candy",
    "tonic mixture",
    "febrifuge",
    "lozenges",
    "plaster",
    "vapor baths",
    "female physician",
    "celebrated female physician",
    "mrs mott",
    "madame restell",
    "medical card",
    "rheumatism",
    "scrofula",
    "dyspepsia",
    "no mercury",
    "roots and essential oils",
    "cosmetic",
    "complexion",
    "lotion",
    "pills",
    "deobstruent pills",
    "dose",
    "doses",
    "cure",
    "cures",
    "female weakness",
    "french specific",
    "without change of diet",
    "does not affect the breath",
    "female complaints",
]


NEWS_CONTEXT_TERMS = [
    "correspondence",
    "by express",
    "foreign news",
    "old world items",
    "election",
    "votes",
    "voters",
    "mayor",
    "aldermen",
    "congress",
    "senate",
    "legislature",
    "governor",
    "president",
    "cabinet",
    "minister",
    "committee",
    "meeting",
    "bill",
    "law",
    "court",
    "judge",
    "trial",
    "charged",
    "burglary",
    "forgery",
    "habeas corpus",
    "prisoners",
    "police",
    "coroner",
    "inquest",
    "verdict",
    "prison",
    "arrested",
    "murder",
    "killed",
    "died",
    "death",
    "death of",
    "accident",
    "disaster",
    "explosion",
    "blew up",
    "sunk",
    "missing",
    "frozen",
    "fire",
    "burned",
    "war",
    "army",
    "troops",
    "railroad",
    "bank",
    "tax",
    "tariff",
    "public",
    "government",
]


def normalise_publisher(value):
    """Normalise spacing only; preserve historical publication wording."""
    return re.sub(r"\s+", " ", value or "").strip()


def normalise_for_exact_dedup(text):
    """Normalise harmless formatting differences before exact hashing."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_reference_word(token):
    """Return whether a cleaned token is recognisable as a broad English word."""
    token = token.lower()
    if token in REFERENCE_WORDS:
        return True
    # Simple suffix stripping catches normal inflected words without turning
    # this into a strict modern spell-checker.
    for suffix in ("ing", "edly", "ed", "ly", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            if token[: -len(suffix)] in REFERENCE_WORDS:
                return True
    return False


# ---------------------------------------------------------------------------
# OCR / READABILITY SCREENING PRINCIPLE FOR SUPERVISOR REVIEW
#
# 1. Allowed:
#    Minor OCR errors and historical spellings are not removed automatically.
#    A record can pass only if the main text is clearly readable and most
#    tokens still look like analysable English words.
#
# 2. Removed:
#    A record is marked as poor quality when it shows strong signs of unusable
#    OCR, including:
#    - short fragments under 100 words
#    - text whose first alphabetic character is lowercase
#    - low alphabetic content
#    - many unusual symbols
#    - too few word-like tokens
#    - many damaged tokens, such as "vi5ited"
#    - many gibberish-like tokens, such as "pfTyUTmeu"
#    - too many tokens not recognised by a broad English word list
#    - abnormal mixed capitalisation
#    - no sentence-like structure
#    - many very short fragment lines, suggesting layout/segmentation problems
# ---------------------------------------------------------------------------


def text_metrics(text):
    """
    Compute conservative OCR/readability indicators.

    These metrics do not penalise historical spelling merely for being old.
    They focus on damaged OCR patterns: very short texts, low alphabetic
    content, excessive symbols, long garbage tokens, and consonant-only tokens.
    """
    text = text or ""
    chars = [char for char in text if not char.isspace()]
    tokens = re.findall(r"\S+", text)
    letter_tokens = re.findall(r"[A-Za-z]+", text)
    first_alpha_match = re.search(r"[A-Za-z]", text)
    first_alpha_is_lowercase = int(
        bool(first_alpha_match and first_alpha_match.group(0).islower())
    )
    # Approximate how many tokens still look like analysable words.
    # This intentionally does not check whether a word is modern/correctly
    # spelled, because historical spellings should not be removed merely for
    # looking unfamiliar.
    word_like_tokens = [
        token
        for token in tokens
        if re.fullmatch(r"[A-Za-z][A-Za-z'’-]*[A-Za-z]|[A-Za-z]", token.strip(".,;:!?()[]{}\""))
    ]
    cleaned_word_tokens = [
        re.sub(r"[^A-Za-z]", "", token)
        for token in tokens
    ]
    cleaned_word_tokens = [token for token in cleaned_word_tokens if token]

    char_count = len(text)
    nonspace_count = len(chars)
    word_count = len(tokens)
    alpha_count = sum(char.isalpha() for char in chars)
    digit_count = sum(char.isdigit() for char in chars)
    # Newlines and tabs are normal layout whitespace in newspaper OCR.
    printable_count = sum(char.isprintable() or char.isspace() for char in text)

    unusual_count = sum(
        not (
            char.isalnum()
            or char.isspace()
            or char in STANDARD_PUNCTUATION
        )
        for char in text
    )

    single_char_tokens = sum(
        len(re.sub(r"[^A-Za-z]", "", token)) == 1 for token in tokens
    )
    long_tokens = sum(
        len(re.sub(r"[^A-Za-z]", "", token)) >= 25 for token in tokens
    )
    consonant_only_tokens = sum(
        len(token) >= 4 and not any(char in VOWELS for char in token.lower())
        for token in letter_tokens
    )
    short_alpha_tokens = sum(
        1 <= len(token) <= 2
        for token in cleaned_word_tokens
    )

    repeated_runs = len(re.findall(r"(.)\1{5,}", text))
    sentence_like_units = [
        unit.strip()
        for unit in re.split(r"[.!?]+", text)
        if len(re.findall(r"[A-Za-z]+", unit)) >= 5
    ]
    nonempty_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]
    line_word_counts = [
        len(re.findall(r"[A-Za-z]+", line))
        for line in nonempty_lines
    ]
    short_fragment_lines = sum(1 for count in line_word_counts if 1 <= count <= 3)
    very_short_fragment_lines = sum(1 for count in line_word_counts if count == 1)

    # Count OCR-damaged tokens such as "vi5ited" or "pr--dent". A few damaged
    # tokens are acceptable, but a high share suggests the article is no longer
    # reliable for downstream text analysis.
    damaged_tokens = 0
    gibberish_like_tokens = 0
    odd_case_tokens = 0
    for token in tokens:
        stripped = token.strip(".,;:!?()[]{}\"'")
        letters = sum(char.isalpha() for char in stripped)
        digits = sum(char.isdigit() for char in stripped)
        unusual = sum(
            not (char.isalnum() or char in "'’-")
            for char in stripped
        )
        # Treat OCR-like mixed alphanumeric/unusual tokens as damaged, but do
        # not penalise normal historical spelling merely for being unfamiliar.
        if letters and (digits or unusual) and len(stripped) >= 3:
            damaged_tokens += 1
        word = re.sub(r"[^A-Za-z]", "", stripped)
        lower_word = word.lower()
        if len(lower_word) >= 5:
            vowel_count = sum(char in VOWELS for char in lower_word)
            rare_letter_count = sum(char in "jqxz" for char in lower_word)
            # Flag tokens that look alphabetic but implausible, e.g. OCR fragments
            # such as "pfTyUTmeu" or "chgTeh". This is a heuristic for
            # unreadable OCR, not a dictionary check.
            if vowel_count == 0 or rare_letter_count >= 2:
                gibberish_like_tokens += 1
            if re.search(r"[a-z][A-Z]|[A-Z][a-z][A-Z]", word):
                odd_case_tokens += 1

    common_function_words = sum(
        token.lower() in COMMON_FUNCTION_WORDS
        for token in cleaned_word_tokens
    )
    reference_checked_tokens = [
        token.lower()
        for token in cleaned_word_tokens
        if len(token) >= 3
    ]
    recognised_tokens = sum(
        is_reference_word(token)
        for token in reference_checked_tokens
    )
    long_unknown_tokens = sum(
        len(token) >= 5 and not is_reference_word(token)
        for token in reference_checked_tokens
    )

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else 0.0

    return {
        "char_count": char_count,
        "word_count": word_count,
        "clean_word_count": len(cleaned_word_tokens),
        "first_alpha_is_lowercase": first_alpha_is_lowercase,
        "alpha_ratio": ratio(alpha_count, nonspace_count),
        "digit_ratio": ratio(digit_count, nonspace_count),
        "printable_ratio": ratio(printable_count, max(char_count, 1)),
        "unusual_char_ratio": ratio(unusual_count, max(char_count, 1)),
        "single_char_token_ratio": ratio(single_char_tokens, word_count),
        "short_alpha_token_ratio": ratio(
            short_alpha_tokens, len(cleaned_word_tokens)
        ),
        "long_token_ratio": ratio(long_tokens, word_count),
        "consonant_only_token_ratio": ratio(
            consonant_only_tokens, len(letter_tokens)
        ),
        "repeated_char_runs": repeated_runs,
        "word_like_token_ratio": ratio(len(word_like_tokens), word_count),
        "damaged_token_ratio": ratio(damaged_tokens, word_count),
        "gibberish_like_token_ratio": ratio(gibberish_like_tokens, word_count),
        "odd_case_token_ratio": ratio(odd_case_tokens, word_count),
        "common_function_word_ratio": ratio(
            common_function_words, len(cleaned_word_tokens)
        ),
        "recognised_word_ratio": ratio(
            recognised_tokens, len(reference_checked_tokens)
        ),
        "long_unknown_token_ratio": ratio(
            long_unknown_tokens, len(reference_checked_tokens)
        ),
        "sentence_like_unit_count": len(sentence_like_units),
        "nonempty_line_count": len(nonempty_lines),
        "short_fragment_line_ratio": ratio(
            short_fragment_lines, len(nonempty_lines)
        ),
        "very_short_fragment_line_ratio": ratio(
            very_short_fragment_lines, len(nonempty_lines)
        ),
    }


def quality_label(metrics):
    """
    Assign an aggressive OCR/readability screening label.

    Following the supervisor discussion, the filter is deliberately stricter:
    borderline or dodgy-looking records should be kept out of the final dataset
    when there is enough remaining data.
    """
    # Unusable means the article is likely to distort downstream text analysis:
    # too short, very low alphabetic content, excessive OCR noise, or many
    # tokens that no longer look like analysable words.
    if metrics["clean_word_count"] < 100:
        return "fail", ""
    if metrics["first_alpha_is_lowercase"]:
        return "fail", ""
    if metrics["alpha_ratio"] < 0.70:
        return "fail", ""
    if metrics["unusual_char_ratio"] > 0.03:
        return "fail", ""
    if metrics["printable_ratio"] < 0.95:
        return "fail", ""
    # Mild OCR errors are acceptable, but the final dataset should exclude
    # passages that look word-like only in fragments or are not coherent enough
    # for downstream language analysis.
    if metrics["word_like_token_ratio"] < 0.88:
        return "fail", ""
    if metrics["damaged_token_ratio"] > 0.04:
        return "fail", ""
    if (
        metrics["word_count"] >= 50
        and metrics["damaged_token_ratio"] > 0.025
        and metrics["word_like_token_ratio"] < 0.92
    ):
        return "fail", ""
    if metrics["gibberish_like_token_ratio"] > 0.02:
        return "fail", ""
    if (
        metrics["word_count"] >= 50
        and metrics["common_function_word_ratio"] < 0.17
        and metrics["gibberish_like_token_ratio"] > 0.01
    ):
        return "fail", ""
    if metrics["word_count"] >= 50 and metrics["odd_case_token_ratio"] > 0.035:
        return "fail", ""
    if (
        metrics["word_count"] >= 50
        and metrics["short_alpha_token_ratio"] > 0.20
        and metrics["damaged_token_ratio"] > 0.02
    ):
        return "fail", ""
    if metrics["word_count"] >= 40 and metrics["sentence_like_unit_count"] == 0:
        return "fail", ""
    if metrics["word_count"] >= 100 and metrics["recognised_word_ratio"] < 0.58:
        return "fail", ""
    if metrics["word_count"] >= 100 and metrics["long_unknown_token_ratio"] > 0.35:
        return "fail", ""
    if (
        metrics["nonempty_line_count"] >= 12
        and metrics["short_fragment_line_ratio"] > 0.55
        and metrics["sentence_like_unit_count"] < 4
    ):
        return "fail", ""
    if (
        metrics["nonempty_line_count"] >= 12
        and metrics["very_short_fragment_line_ratio"] > 0.25
    ):
        return "fail", ""

    # Borderline records are retained for auditing but are not included in the
    # final usable dataset, because analysis_decision() keeps only "pass".
    if metrics["single_char_token_ratio"] > 0.12:
        return "manual_review", ""
    if metrics["long_token_ratio"] > 0.02:
        return "manual_review", ""
    if metrics["consonant_only_token_ratio"] > 0.10:
        return "manual_review", ""
    if metrics["repeated_char_runs"] >= 2:
        return "manual_review", ""
    if metrics["word_like_token_ratio"] < 0.94:
        return "manual_review", ""
    if metrics["short_alpha_token_ratio"] > 0.14:
        return "manual_review", ""
    if metrics["damaged_token_ratio"] > 0.02:
        return "manual_review", ""
    if metrics["gibberish_like_token_ratio"] > 0.01:
        return "manual_review", ""
    if metrics["odd_case_token_ratio"] > 0.015:
        return "manual_review", ""
    if metrics["word_count"] >= 50 and metrics["common_function_word_ratio"] < 0.19:
        return "manual_review", ""
    if metrics["word_count"] >= 80 and metrics["sentence_like_unit_count"] < 2:
        return "manual_review", ""
    if metrics["recognised_word_ratio"] < 0.68:
        return "manual_review", ""
    if metrics["long_unknown_token_ratio"] > 0.25:
        return "manual_review", ""
    if (
        metrics["nonempty_line_count"] >= 8
        and metrics["short_fragment_line_ratio"] > 0.40
    ):
        return "manual_review", ""
    return "pass", ""


def count_terms(text, terms):
    count = 0
    for term in terms:
        escaped = re.escape(term)
        if re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text):
            count += 1
    return count


def content_profile(headline, text):
    """
    Keep content screening broad, but remove obvious non-article notices.

    The final dataset should not be limited to politics, but very formulaic
    advertisements, route schedules, service notices, and listings are not
    article-like enough for wellbeing language analysis.
    """
    combined = f"{headline or ''} {text or ''}".lower()
    combined = re.sub(r"\s+", " ", combined)
    non_article_notice_terms = [
        "daily mail line",
        "mail line",
        "steamboats",
        "steam boats",
        "steam packet",
        "passage and fare",
        "passage fare",
        "leaving the",
        "will leave",
        "will arrive",
        "running daily",
        "run in connection",
        "travellers",
        "travelers",
        "route for",
        "wharf",
        "agent",
        "for sale",
        "offers for sale",
        "wholesale",
        "retail",
        "apply to",
        "inquire at",
        "terms cash",
        "cash paid",
        "subscriber",
        "subscribers",
        "auction",
        "auctioneer",
        "notice is hereby given",
        "estate of",
        "probate",
        "executor",
        "administrator",
        "list of letters",
        "letters remaining",
        "prices current",
    ]
    notice_signal_count = count_terms(combined, non_article_notice_terms)
    route_notice_like = (
        count_terms(
            combined,
            [
                "daily mail line",
                "mail line",
                "steamboats",
                "steam packet",
                "passage and fare",
                "wharf",
            ],
        )
        >= 2
        and count_terms(
            combined,
            [
                "leaving",
                "will leave",
                "will arrive",
                "running daily",
                "travellers",
                "travelers",
                "route",
                "agent",
            ],
        )
        >= 2
    )
    commercial_notice_like = notice_signal_count >= 4
    wellbeing_signals = (
        count_terms(combined, CONTENT_TERMS["health_wellbeing"])
        + count_terms(combined, CONTENT_TERMS["employment_wages"])
        + count_terms(combined, CONTENT_TERMS["crime_safety"])
        + count_terms(combined, CONTENT_TERMS["housing_poverty"])
        + count_terms(combined, CONTENT_TERMS["education_family"])
    )
    primary_type = (
        "non_article_notice"
        if route_notice_like or commercial_notice_like
        else "broad_newspaper_content"
    )

    return {
        "wellbeing_related_term_count": wellbeing_signals,
        "notice_signal_count": notice_signal_count,
        "primary_content_type": primary_type,
    }


def analysis_decision(quality, is_duplicate, content_type, apply_content_filter=False):
    """Return the final inclusion decision for the analysis dataset."""
    if quality != "pass":
        return "exclude_quality"
    if is_duplicate:
        return "exclude_duplicate"
    if apply_content_filter and content_type == "non_article_notice":
        return "exclude_non_article_notice"
    return "keep"


def load_selected_publishers(selection_path):
    variants = set()
    with selection_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for variant in row["american_stories_title_variants"].split(" | "):
                if variant.strip():
                    variants.add(normalise_publisher(variant))
    return variants


def main():
    parser = argparse.ArgumentParser(
        description="Assess readability and exact duplicates in one full-year CSV."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path(
            "/Users/nora/Documents/Codex/2026-05-24/"
            "45-population-wellbeing-the-news-language/outputs/"
            "final_source_list_for_text_screening.csv"
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--usable-output",
        type=Path,
        help=(
            "Optional direct path for usable_articles output. If omitted, "
            "usable_articles.csv is written inside --output-dir."
        ),
    )
    parser.add_argument(
        "--year",
        help="Optional year value to add as a year column in usable_articles.csv.",
    )
    parser.add_argument(
        "--all-publishers",
        action="store_true",
        help="Assess every publisher rather than only selected sources.",
    )
    parser.add_argument(
        "--skip-article-flags",
        action="store_true",
        help=(
            "Do not write article_quality_flags.csv. Use this when only the "
            "keep-only usable_articles.csv and summary tables are needed."
        ),
    )
    parser.add_argument(
        "--include-usable-audit-columns",
        action="store_true",
        help=(
            "Add screening helper columns to usable_articles.csv. By default "
            "the usable file contains only date, text, headline, and publisher."
        ),
    )
    parser.add_argument(
        "--minimal-output",
        action="store_true",
        help="Write only usable_articles.csv and skip all audit/summary files.",
    )
    parser.add_argument(
        "--apply-content-filter",
        action="store_true",
        help=(
            "Optionally exclude records flagged as obvious non-article notices. "
            "By default, the main filter only applies source selection, "
            "OCR/readability screening, and exact duplicate removal."
        ),
    )
    args = parser.parse_args()
    if args.minimal_output:
        args.skip_article_flags = True

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_publishers = load_selected_publishers(args.selection)

    article_output = args.output_dir / "article_quality_flags.csv"
    usable_output = args.usable_output or args.output_dir / "usable_articles.csv"
    usable_output.parent.mkdir(parents=True, exist_ok=True)
    publisher_output = args.output_dir / "publisher_quality_summary.csv"
    duplicate_output = args.output_dir / "exact_duplicate_groups.csv"

    publisher_summary = defaultdict(
        lambda: Counter(
            total_articles=0,
            total_words=0,
            pass_articles=0,
            manual_review_articles=0,
            fail_articles=0,
            duplicate_articles=0,
            usable_articles=0,
            broad_newspaper_content_articles=0,
            non_article_notice_articles=0,
            news_or_public_affairs_articles=0,
            advertisement_or_classified_articles=0,
            legal_notice_articles=0,
            market_or_finance_listing_articles=0,
            shipping_listing_articles=0,
            literary_or_entertainment_articles=0,
            mixed_or_uncertain_articles=0,
            uncertain_articles=0,
            wellbeing_related_articles=0,
        )
    )
    first_hash_record = {}
    duplicate_groups = defaultdict(list)

    with args.input.open(newline="", encoding="utf-8") as inf, usable_output.open(
        "w", newline="", encoding="utf-8"
    ) as usable_f:
        reader = csv.DictReader(inf)
        output_fields = [
            "row_number",
            "date",
            "publisher",
            "headline",
            "text_excerpt",
            "quality_label",
            "quality_reasons",
            "analysis_decision",
            "is_normalised_exact_duplicate",
            "duplicate_of_row",
            "text_hash",
            "char_count",
            "word_count",
            "clean_word_count",
            "first_alpha_is_lowercase",
            "alpha_ratio",
            "digit_ratio",
            "printable_ratio",
            "unusual_char_ratio",
            "single_char_token_ratio",
            "short_alpha_token_ratio",
            "long_token_ratio",
            "consonant_only_token_ratio",
            "repeated_char_runs",
            "word_like_token_ratio",
            "damaged_token_ratio",
            "gibberish_like_token_ratio",
            "odd_case_token_ratio",
            "common_function_word_ratio",
            "recognised_word_ratio",
            "long_unknown_token_ratio",
            "sentence_like_unit_count",
            "nonempty_line_count",
            "short_fragment_line_ratio",
            "very_short_fragment_line_ratio",
            "primary_content_type",
            "wellbeing_related_term_count",
            "notice_signal_count",
            "classified_structure_count",
            "phone_like_count",
            "address_like_count",
            "money_like_count",
            "medical_ad_term_count",
            "medical_ad_like",
            "commercial_ad_like",
            "literary_like",
            "corporate_notice_like",
            "shipping_listing_like",
            "listing_or_notice_like",
            "news_context_term_count",
            "strong_news_signal_count",
            "advertisement_term_count",
            "legal_notice_term_count",
            "market_finance_term_count",
            "politics_government_term_count",
            "health_wellbeing_term_count",
            "employment_wages_term_count",
            "crime_safety_term_count",
            "housing_poverty_term_count",
            "education_family_term_count",
        ]
        writer = None
        article_flags_file = None
        if not args.skip_article_flags:
            article_flags_file = article_output.open(
                "w", newline="", encoding="utf-8"
            )
            writer = csv.DictWriter(article_flags_file, fieldnames=output_fields)
            writer.writeheader()
        usable_fields = [
            "date",
            "text",
            "headline",
            "publisher",
        ]
        if args.year:
            usable_fields = ["year", *usable_fields]
        if args.include_usable_audit_columns:
            usable_fields.extend(
                [
                    "row_number",
                    "primary_content_type",
                    "wellbeing_related_term_count",
                    "word_count",
                    "clean_word_count",
                    "text_hash",
                ]
            )
        usable_writer = csv.DictWriter(usable_f, fieldnames=usable_fields)
        usable_writer.writeheader()

        for row_number, row in enumerate(reader, start=1):
            publisher = normalise_publisher(row.get("publisher"))
            if not args.all_publishers and publisher not in selected_publishers:
                continue

            text = row.get("text") or ""
            headline = row.get("headline", "")
            metrics = text_metrics(text)
            label, reasons = quality_label(metrics)
            content = content_profile(headline, text)
            normalised_text = normalise_for_exact_dedup(text)
            dedup_eligible = metrics["word_count"] >= 20
            text_hash = (
                hashlib.blake2b(
                    normalised_text.encode("utf-8"), digest_size=16
                ).hexdigest()
                if dedup_eligible
                else ""
            )

            duplicate_of_row = first_hash_record.get(text_hash) if text_hash else None
            is_duplicate = int(
                duplicate_of_row is not None and dedup_eligible
            )
            if duplicate_of_row is None and dedup_eligible:
                first_hash_record[text_hash] = row_number
            elif dedup_eligible:
                duplicate_groups[text_hash].append(row_number)

            summary = publisher_summary[publisher]
            summary["total_articles"] += 1
            summary["total_words"] += metrics["word_count"]
            summary[f"{label}_articles"] += 1
            summary["duplicate_articles"] += is_duplicate
            summary[f"{content['primary_content_type']}_articles"] += 1
            summary["wellbeing_related_articles"] += int(
                content["wellbeing_related_term_count"] > 0
            )
            final_decision = analysis_decision(
                label,
                is_duplicate,
                content["primary_content_type"],
                args.apply_content_filter,
            )

            output_row = {
                "row_number": row_number,
                "date": row.get("date", ""),
                "publisher": publisher,
                "headline": headline,
                "text_excerpt": re.sub(r"\s+", " ", text).strip()[:500],
                "quality_label": label,
                "quality_reasons": reasons,
                "analysis_decision": final_decision,
                "is_normalised_exact_duplicate": is_duplicate,
                "duplicate_of_row": duplicate_of_row or "",
                "text_hash": text_hash,
                **metrics,
                **content,
            }
            if writer is not None:
                writer.writerow(output_row)

            if final_decision == "keep":
                summary["usable_articles"] += 1
                usable_row = {
                    "date": row.get("date", ""),
                    "text": text,
                    "headline": headline,
                    "publisher": publisher,
                }
                if args.year:
                    usable_row["year"] = args.year
                if args.include_usable_audit_columns:
                    usable_row.update(
                        {
                            "row_number": row_number,
                            "primary_content_type": content["primary_content_type"],
                            "wellbeing_related_term_count": content[
                                "wellbeing_related_term_count"
                            ],
                            "word_count": metrics["word_count"],
                            "clean_word_count": metrics["clean_word_count"],
                            "text_hash": text_hash,
                        }
                    )
                usable_writer.writerow(usable_row)

        if article_flags_file is not None:
            article_flags_file.close()

    if not args.minimal_output:
        with publisher_output.open("w", newline="", encoding="utf-8") as f:
            fields = [
                "publisher",
                "total_articles",
                "total_words",
                "pass_articles",
                "manual_review_articles",
                "fail_articles",
                "duplicate_articles",
                "usable_articles",
                "broad_newspaper_content_articles",
                "non_article_notice_articles",
                "news_or_public_affairs_articles",
                "advertisement_or_classified_articles",
                "legal_notice_articles",
                "market_or_finance_listing_articles",
                "shipping_listing_articles",
                "literary_or_entertainment_articles",
                "mixed_or_uncertain_articles",
                "uncertain_articles",
                "wellbeing_related_articles",
                "pass_share",
                "manual_review_share",
                "fail_share",
                "duplicate_share",
                "usable_share",
                "broad_newspaper_content_share",
                "non_article_notice_share",
                "news_or_public_affairs_share",
                "advertisement_or_classified_share",
                "legal_notice_share",
                "market_or_finance_listing_share",
                "shipping_listing_share",
                "literary_or_entertainment_share",
                "wellbeing_related_share",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for publisher, counts in sorted(publisher_summary.items()):
                total = counts["total_articles"]
                writer.writerow(
                    {
                        "publisher": publisher,
                        **counts,
                        "pass_share": counts["pass_articles"] / total,
                        "manual_review_share": counts["manual_review_articles"] / total,
                        "fail_share": counts["fail_articles"] / total,
                        "duplicate_share": counts["duplicate_articles"] / total,
                        "usable_share": counts["usable_articles"] / total,
                        "broad_newspaper_content_share": (
                            counts["broad_newspaper_content_articles"] / total
                        ),
                        "non_article_notice_share": (
                            counts["non_article_notice_articles"] / total
                        ),
                        "news_or_public_affairs_share": (
                            counts["news_or_public_affairs_articles"] / total
                        ),
                        "advertisement_or_classified_share": (
                            counts["advertisement_or_classified_articles"] / total
                        ),
                        "legal_notice_share": counts["legal_notice_articles"] / total,
                        "market_or_finance_listing_share": (
                            counts["market_or_finance_listing_articles"] / total
                        ),
                        "shipping_listing_share": (
                            counts["shipping_listing_articles"] / total
                        ),
                        "literary_or_entertainment_share": (
                            counts["literary_or_entertainment_articles"] / total
                        ),
                        "wellbeing_related_share": (
                            counts["wellbeing_related_articles"] / total
                        ),
                    }
                )

        with duplicate_output.open("w", newline="", encoding="utf-8") as f:
            fields = ["text_hash", "first_row", "duplicate_rows", "n_duplicates"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for text_hash, rows in duplicate_groups.items():
                writer.writerow(
                    {
                        "text_hash": text_hash,
                        "first_row": first_hash_record[text_hash],
                        "duplicate_rows": " | ".join(map(str, rows)),
                        "n_duplicates": len(rows),
                    }
                )

    print(f"Publishers assessed: {len(publisher_summary)}")
    if args.skip_article_flags:
        print("Article flags: skipped")
    else:
        print(f"Article flags: {article_output}")
    print(f"Usable articles: {usable_output}")
    if args.minimal_output:
        print("Publisher summary: skipped")
        print("Duplicate groups: skipped")
    else:
        print(f"Publisher summary: {publisher_output}")
        print(f"Duplicate groups: {duplicate_output}")


if __name__ == "__main__":
    main()
