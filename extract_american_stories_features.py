import argparse
import glob
import html
import math
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def clean_text(value):
    """Light text cleaning before NLP feature extraction."""
    if pd.isna(value):
        return None
    text = html.unescape(html.unescape(str(value)))
    text = unicodedata.normalize("NFKC", text)
    text = (
        text.replace("\xa0", " ")
        .replace("\u200b", "")
        .replace("\ufeff", "")
        .replace("\r", " ")
        .replace("\n", " ")
        .replace("\t", " ")
    )
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"([!?.,]){3,}", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def parse_year_from_filename(path):
    match = re.search(r"usable_articles_(\d{4})\.csv$", Path(path).name)
    return int(match.group(1)) if match else None


CONTENT_PATTERNS = {
    "advertisement_or_commercial_notice": [
        r"\badvertisement\b",
        r"\bclassified\b",
        r"\bfor sale\b",
        r"\boffers? for sale\b",
        r"\bfor rent\b",
        r"\bwanted\b",
        r"\bapply to\b",
        r"\binquire at\b",
        r"\bterms cash\b",
        r"\bcash paid\b",
        r"\bwholesale\b",
        r"\bretail\b",
        r"\bat your druggist\b",
        r"\bfrom your druggist\b",
        r"\bbottle\b",
        r"\bprice \$?\d",
        r"\bcall\b",
        r"\btelephone\b",
        r"\bphone\b",
    ],
    "legal_notice": [
        r"\bnotice is hereby given\b",
        r"\bestate of\b",
        r"\bprobate\b",
        r"\bexecutor\b",
        r"\badministrator\b",
        r"\bsheriff'?s sale\b",
        r"\bsummons\b",
        r"\bcourt of\b",
    ],
    "market_or_shipping_listing": [
        r"\bprices current\b",
        r"\bstock exchange\b",
        r"\bcotton market\b",
        r"\bgrain market\b",
        r"\bshipping intelligence\b",
        r"\bmarine intelligence\b",
        r"\barrived\b",
        r"\bcleared\b",
        r"\bschooner\b",
        r"\bsteamer\b",
        r"\bsteamship\b",
        r"\bbrig\b",
    ],
    "event_or_entertainment_listing": [
        r"\btheatre\b",
        r"\btheater\b",
        r"\bmatinee\b",
        r"\bperformance\b",
        r"\bconcert\b",
        r"\bprogram of music\b",
    ],
}


def count_pattern_matches(text, patterns):
    return sum(1 for pattern in patterns if re.search(pattern, text))


def classify_content_type(headline, text):
    """
    Flag likely non-article-like records for sensitivity analysis.

    This is intentionally conservative and transparent. It does not decide
    whether the article sentiment is valid; it only identifies records with
    strong notice/listing/advertisement signals so they can optionally be
    excluded or reported separately.
    """
    combined = f"{headline or ''} {text or ''}".lower()
    combined = re.sub(r"\s+", " ", combined)

    scores = {
        label: count_pattern_matches(combined, patterns)
        for label, patterns in CONTENT_PATTERNS.items()
    }
    reasons = [f"{label}:{count}" for label, count in scores.items() if count]

    headline_text = (headline or "").lower()
    headline_says_ad = bool(re.search(r"\badvertisement\b|\bclassified\b", headline_text))

    if headline_says_ad or scores["advertisement_or_commercial_notice"] >= 3:
        return "non_article_like", "advertisement_or_commercial_notice", ";".join(reasons)
    if scores["legal_notice"] >= 2:
        return "non_article_like", "legal_notice", ";".join(reasons)
    if scores["market_or_shipping_listing"] >= 3:
        return "non_article_like", "market_or_shipping_listing", ";".join(reasons)
    if scores["event_or_entertainment_listing"] >= 3:
        return "non_article_like", "event_or_entertainment_listing", ";".join(reasons)
    return "article_like_or_uncertain", "", ";".join(reasons)


def add_content_type_flags(df_all):
    flags = df_all.apply(
        lambda row: classify_content_type(row.get("headline"), row.get("text")),
        axis=1,
        result_type="expand",
    )
    flags.columns = ["content_type", "content_subtype", "content_filter_reasons"]
    return pd.concat([df_all, flags], axis=1)


def load_american_stories(
    input_dir,
    start_year,
    end_year,
    sample_size=None,
    exclude_non_article_like=False,
):
    """Read usable_articles_YYYY.csv files produced by the final text filter."""
    files = sorted(glob.glob(str(Path(input_dir) / "usable_articles_*.csv")))
    frames = []

    for file_path in files:
        source_year = parse_year_from_filename(file_path)
        if source_year is None:
            continue
        if start_year is not None and source_year < start_year:
            continue
        if end_year is not None and source_year > end_year:
            continue

        df = pd.read_csv(file_path)
        missing = {"date", "publisher", "headline", "text"} - set(df.columns)
        if missing:
            raise ValueError(f"{file_path} is missing columns: {sorted(missing)}")

        df = df[["date", "publisher", "headline", "text"]].copy()
        df["source_year"] = source_year
        frames.append(df)
        print(f"Read {file_path}: {len(df):,} rows")

    if not frames:
        raise SystemExit(f"No usable_articles_YYYY.csv files found in {input_dir}")

    df_all = pd.concat(frames, ignore_index=True)
    df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")
    df_all["year"] = df_all["date"].dt.year.fillna(df_all["source_year"]).astype(int)
    df_all["month"] = df_all["date"].dt.month
    df_all["decade"] = (df_all["year"] // 10) * 10

    df_all["text"] = df_all["text"].map(clean_text)
    df_all["headline"] = df_all["headline"].map(clean_text)
    df_all["publisher"] = df_all["publisher"].map(clean_text)
    df_all = df_all[df_all["text"].notna() & (df_all["text"].str.len() > 0)].copy()
    df_all["word_count"] = df_all["text"].str.count(r"\S+").astype(int)
    df_all = add_content_type_flags(df_all)

    print("Content-type flags before optional exclusion:")
    print(df_all["content_type"].value_counts(dropna=False).to_string())
    if exclude_non_article_like:
        before = len(df_all)
        df_all = df_all[df_all["content_type"] != "non_article_like"].copy()
        print(
            "Excluded likely non-article-like records: "
            f"{before - len(df_all):,} of {before:,}"
        )

    if sample_size:
        sampled_frames = []
        for _, group in df_all.groupby("year"):
            sampled_frames.append(
                group.sample(min(len(group), sample_size), random_state=20260815)
            )
        df_all = pd.concat(sampled_frames, ignore_index=True)
        print(f"Using sample: up to {sample_size:,} articles per year")

    df_all.insert(0, "article_id", [f"AS_{i:09d}" for i in range(len(df_all))])
    print(f"Total rows for feature extraction: {len(df_all):,}")
    return df_all


def load_transformer_pipeline(model_name, device):
    from transformers import pipeline

    return pipeline(
        "text-classification",
        model=model_name,
        tokenizer=model_name,
        return_all_scores=True,
        truncation=True,
        device=device,
    )


def batched_probs(texts, pipe, batch_size=8, max_length=512):
    """Run a Hugging Face classification pipeline and return label probabilities."""
    out_rows = []
    for start in tqdm(range(0, len(texts), batch_size), desc="model batches"):
        batch = texts[start : start + batch_size]
        preds = pipe(
            batch,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_all_scores=True,
        )
        for item in preds:
            out_rows.append({p["label"].lower(): float(p["score"]) for p in item})
    return pd.DataFrame(out_rows)


def add_sentiment_and_emotion(df_all, batch_size):
    """Replicate Emmy's transformer sentiment and emotion feature extraction."""
    import torch

    device = 0 if torch.cuda.is_available() else -1
    print("Sentiment/emotion device:", "GPU" if device == 0 else "CPU")

    sent_model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    emo_model = "j-hartmann/emotion-english-distilroberta-base"

    sent_pipe = load_transformer_pipeline(sent_model, device)
    emo_pipe = load_transformer_pipeline(emo_model, device)

    texts = df_all["text"].tolist()
    sent_df = batched_probs(texts, sent_pipe, batch_size=batch_size)
    emo_df = batched_probs(texts, emo_pipe, batch_size=batch_size)

    # Normalise possible model label variants.
    rename = {
        "label_0": "negative",
        "label_1": "sent_neutral",
        "label_2": "positive",
        "neutral": "sent_neutral",
    }
    sent_df = sent_df.rename(columns=rename)
    for col in ["negative", "sent_neutral", "positive"]:
        if col not in sent_df.columns:
            sent_df[col] = 0.0
    sent_df["sentiment_neg"] = sent_df["negative"]
    sent_df["sentiment_pos_minus_neg"] = sent_df["positive"] - sent_df["negative"]

    emo_df = emo_df.rename(columns={"neutral": "emo_neutral"})
    for col in ["anger", "disgust", "fear", "joy", "emo_neutral", "sadness", "surprise"]:
        if col not in emo_df.columns:
            emo_df[col] = 0.0
    emo_df["emotion_distress"] = (
        emo_df["fear"].fillna(0)
        + emo_df["sadness"].fillna(0)
        + emo_df["anger"].fillna(0)
    )

    return pd.concat(
        [df_all.reset_index(drop=True), sent_df.reset_index(drop=True), emo_df.reset_index(drop=True)],
        axis=1,
    )


def count_syllables(word):
    word = re.sub(r"[^a-z]", "", str(word).lower())
    if not word:
        return 0
    vowels = "aeiouy"
    syllables = 0
    prev_is_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_is_vowel:
            syllables += 1
        prev_is_vowel = is_vowel
    if word.endswith("e") and syllables > 1:
        syllables -= 1
    return max(1, syllables)


def basic_complexity(text):
    """Simple readability and length features."""
    text = "" if text is None else str(text)
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    words = [w for w in re.findall(r"\b[\w']+\b", text) if re.search(r"[A-Za-z]", w)]
    n_sent = max(1, len(sentences))
    n_words = max(1, len(words))
    syllables = sum(count_syllables(w) for w in words)
    avg_sent_len = n_words / n_sent
    avg_word_len = sum(len(re.sub(r"[^A-Za-z]", "", w)) for w in words) / n_words
    flesch = 206.835 - 1.015 * avg_sent_len - 84.6 * (syllables / n_words)
    fk_grade = 0.39 * avg_sent_len + 11.8 * (syllables / n_words) - 15.59
    return {
        "n_sentences": n_sent,
        "n_words_text": n_words,
        "avg_sentence_length": avg_sent_len,
        "avg_word_length": avg_word_len,
        "flesch_reading_ease": flesch,
        "flesch_kincaid_grade": fk_grade,
    }


def pseudo_perplexity_mlm_factory():
    """Load the RoBERTa masked-language model used for BERT-style complexity."""
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()

    @torch.no_grad()
    def pseudo_perplexity_mlm(text, max_length=512):
        text = "" if text is None else str(text).strip()
        if not text:
            return np.nan
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)
        special_mask = tokenizer.get_special_tokens_mask(
            input_ids[0].tolist(), already_has_special_tokens=True
        )
        special_mask = torch.tensor(special_mask, device=device).bool()
        score_positions = (attention_mask[0] == 1) & (~special_mask)
        total_nll = 0.0
        n_scored = 0
        for pos in torch.where(score_positions)[0].tolist():
            masked_ids = input_ids.clone()
            masked_ids[0, pos] = tokenizer.mask_token_id
            logits = model(masked_ids, attention_mask=attention_mask).logits
            log_probs = torch.log_softmax(logits[0, pos], dim=-1)
            total_nll += (-log_probs[input_ids[0, pos]]).item()
            n_scored += 1
        return float(np.exp(total_nll / n_scored)) if n_scored else np.nan

    print("BERT-style complexity device:", device)
    return pseudo_perplexity_mlm


def add_complexity(df_all, run_bert_complexity):
    comp = df_all["text"].map(basic_complexity).apply(pd.Series)
    df_all = pd.concat([df_all, comp], axis=1)
    if run_bert_complexity:
        scorer = pseudo_perplexity_mlm_factory()
        tqdm.pandas(desc="bert complexity")
        df_all["bert_complexity_ppl"] = df_all["text"].progress_apply(scorer)
    return df_all


def clean_for_empath(text):
    text = "" if text is None else str(text).lower()
    text = (
        text.replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
    )
    text = re.sub(r"[^a-z\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def add_empath_features(df_all):
    """Add Empath lexical features, similar to the LIWC-like step in Emmy's notebook."""
    from empath import Empath

    lex = Empath()

    def empath_features(text):
        cleaned = clean_for_empath(text)
        if not cleaned:
            return {}
        return lex.analyze(cleaned, normalize=True)

    tqdm.pandas(desc="empath")
    tmp = df_all["text"].progress_apply(empath_features)
    empath_df = tmp.apply(pd.Series).reindex(df_all.index).fillna(0.0)
    return pd.concat([df_all, empath_df.add_prefix("liwc_")], axis=1)


def add_embeddings(df_all, n_components):
    """Add sentence-transformer PCA components for semantic variation."""
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA

    model_name = "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        df_all["text"].astype(str).tolist(),
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    n_components = min(n_components, embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_components, random_state=20260815)
    pcs = pca.fit_transform(embeddings)
    pc_df = pd.DataFrame(
        pcs,
        columns=[f"emb_pc{i + 1}" for i in range(n_components)],
        index=df_all.index,
    )
    return pd.concat([df_all, pc_df], axis=1)


def weighted_average(values, weights):
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return np.nan
    return float((values[mask] * weights[mask]).sum() / weights[mask].sum())


def aggregate_scores(df_all, group_col):
    """Create yearly or decade-level descriptive feature summaries."""
    base_cols = [
        "sentiment_neg",
        "sentiment_pos_minus_neg",
        "negative",
        "sent_neutral",
        "positive",
        "anger",
        "disgust",
        "fear",
        "sadness",
        "joy",
        "surprise",
        "emo_neutral",
        "emotion_distress",
        "avg_sentence_length",
        "avg_word_length",
        "flesch_reading_ease",
        "flesch_kincaid_grade",
        "bert_complexity_ppl",
    ]
    liwc_cols = [c for c in df_all.columns if c.startswith("liwc_")]
    emb_cols = [c for c in df_all.columns if c.startswith("emb_pc")]
    score_cols = [c for c in base_cols + liwc_cols + emb_cols if c in df_all.columns]

    grouped = df_all.groupby(group_col)
    counts = grouped.agg(
        n_articles=("text", "size"),
        total_words=("word_count", "sum"),
        mean_words=("word_count", "mean"),
        n_publishers=("publisher", "nunique"),
    )
    if "content_type" in df_all.columns:
        content_counts = grouped["content_type"].value_counts().unstack(fill_value=0)
        for col in ["article_like_or_uncertain", "non_article_like"]:
            if col not in content_counts.columns:
                content_counts[col] = 0
        content_counts = content_counts[
            ["article_like_or_uncertain", "non_article_like"]
        ].rename(
            columns={
                "article_like_or_uncertain": "n_article_like_or_uncertain",
                "non_article_like": "n_non_article_like",
            }
        )
        counts = counts.join(content_counts)
        counts["share_non_article_like"] = (
            counts["n_non_article_like"] / counts["n_articles"]
        )
    means = grouped[score_cols].mean(numeric_only=True).add_prefix("mean_")

    weights = df_all["word_count"].where(df_all["word_count"] > 0, 1.0)
    wavg_rows = []
    for group_value, group in grouped:
        row = {group_col: group_value}
        for col in score_cols:
            row[f"wavg_{col}"] = weighted_average(group[col], weights.loc[group.index])
        wavg_rows.append(row)
    wavgs = pd.DataFrame(wavg_rows).set_index(group_col)
    return pd.concat([counts, means, wavgs], axis=1).reset_index()


def main():
    parser = argparse.ArgumentParser(
        description="Extract NOW-style NLP wellbeing features for filtered American Stories."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed/quality_by_year_final_sources"),
        help="Folder containing usable_articles_YYYY.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/american_stories_features"),
        help="Folder for article-level and aggregated feature outputs.",
    )
    parser.add_argument("--start-year", type=int, default=1837)
    parser.add_argument("--end-year", type=int, default=1964)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional max articles per year for test runs.",
    )
    parser.add_argument(
        "--exclude-non-article-like",
        action="store_true",
        help=(
            "Exclude records conservatively flagged as advertisements, legal notices, "
            "market/shipping listings, or event listings before feature extraction."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--skip-transformer-features",
        action="store_true",
        help="Skip transformer sentiment/emotion models.",
    )
    parser.add_argument(
        "--skip-empath",
        action="store_true",
        help="Skip Empath/LIWC-like lexical features.",
    )
    parser.add_argument(
        "--run-bert-complexity",
        action="store_true",
        help="Run slow RoBERTa pseudo-perplexity complexity feature.",
    )
    parser.add_argument(
        "--run-embeddings",
        action="store_true",
        help="Run sentence-transformer embeddings and PCA features.",
    )
    parser.add_argument("--embedding-components", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df_all = load_american_stories(
        args.input_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        sample_size=args.sample_size,
        exclude_non_article_like=args.exclude_non_article_like,
    )

    if not args.skip_transformer_features:
        df_all = add_sentiment_and_emotion(df_all, batch_size=args.batch_size)

    df_all = add_complexity(df_all, run_bert_complexity=args.run_bert_complexity)

    if not args.skip_empath:
        df_all = add_empath_features(df_all)

    if args.run_embeddings:
        df_all = add_embeddings(df_all, n_components=args.embedding_components)

    article_file = args.output_dir / "american_stories_article_features.csv"
    year_file = args.output_dir / "american_stories_yearly_features.csv"
    decade_file = args.output_dir / "american_stories_decade_features.csv"

    df_all.to_csv(article_file, index=False)
    aggregate_scores(df_all, "year").to_csv(year_file, index=False)
    aggregate_scores(df_all, "decade").to_csv(decade_file, index=False)

    print(f"Saved article features: {article_file}")
    print(f"Saved yearly features: {year_file}")
    print(f"Saved decade features: {decade_file}")


if __name__ == "__main__":
    main()
