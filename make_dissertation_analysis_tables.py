import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "article_id",
    "date",
    "publisher",
    "headline",
    "text",
    "year",
    "decade",
    "word_count",
    "content_type",
    "content_subtype",
    "content_filter_reasons",
    "negative",
    "sent_neutral",
    "positive",
    "sentiment_neg",
    "sentiment_pos_minus_neg",
    "anger",
    "disgust",
    "fear",
    "joy",
    "emo_neutral",
    "sadness",
    "surprise",
    "emotion_distress",
    "n_sentences",
    "n_words_text",
    "avg_sentence_length",
    "avg_word_length",
    "flesch_reading_ease",
    "flesch_kincaid_grade",
]

OUTCOME_COLUMNS = [
    "sentiment_pos_minus_neg",
    "negative",
    "sent_neutral",
    "positive",
    "emotion_distress",
    "anger",
    "disgust",
    "fear",
    "sadness",
    "joy",
    "surprise",
    "emo_neutral",
]

TEXT_COLUMNS = [
    "word_count",
    "n_sentences",
    "avg_sentence_length",
    "avg_word_length",
    "flesch_reading_ease",
    "flesch_kincaid_grade",
]


def flatten_columns(columns):
    names = []
    for col in columns:
        if isinstance(col, tuple):
            names.append("_".join(str(part) for part in col if part))
        else:
            names.append(str(col))
    return names


def summarise_by_group(df, group_col):
    value_cols = OUTCOME_COLUMNS + TEXT_COLUMNS
    summary = (
        df.groupby(group_col)
        .agg(
            n_articles=("article_id", "size"),
            total_words=("word_count", "sum"),
            n_publishers=("publisher", "nunique"),
            n_article_like_or_uncertain=(
                "content_type",
                lambda s: int((s != "non_article_like").sum()),
            ),
            n_non_article_like=(
                "content_type",
                lambda s: int((s == "non_article_like").sum()),
            ),
        )
        .reset_index()
    )

    stats = df.groupby(group_col)[value_cols].agg(["mean", "std", "median", "min", "max"])
    quantiles = df.groupby(group_col)[value_cols].quantile([0.25, 0.75]).unstack()
    stats.columns = flatten_columns(stats.columns)
    quantiles.columns = [
        f"{col}_q{int(q * 100)}"
        for col, q in quantiles.columns
    ]
    stats = stats.reset_index()
    quantiles = quantiles.reset_index()

    out = summary.merge(stats, on=group_col, how="left").merge(
        quantiles, on=group_col, how="left"
    )
    out["share_non_article_like_percent"] = (
        out["n_non_article_like"] / out["n_articles"] * 100
    )
    return out


def cluster_bootstrap_ci(df, group_col, variables, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    rows = []

    grouped = (
        df.groupby([group_col, "publisher"])[variables]
        .agg(["sum", "count"])
        .reset_index()
    )
    grouped.columns = flatten_columns(grouped.columns)

    for group_value, group_df in grouped.groupby(group_col):
        publishers = group_df["publisher"].to_numpy()
        if len(publishers) == 0:
            continue

        for variable in variables:
            sums = group_df[f"{variable}_sum"].to_numpy(dtype=float)
            counts = group_df[f"{variable}_count"].to_numpy(dtype=float)
            observed = sums.sum() / counts.sum()
            boot = []
            for _ in range(n_boot):
                sample_idx = rng.integers(0, len(publishers), size=len(publishers))
                boot_sum = sums[sample_idx].sum()
                boot_count = counts[sample_idx].sum()
                boot.append(boot_sum / boot_count if boot_count else np.nan)
            ci_low, ci_high = np.nanpercentile(boot, [2.5, 97.5])
            rows.append(
                {
                    group_col: group_value,
                    "variable": variable,
                    "mean": observed,
                    "cluster_bootstrap_ci_low": ci_low,
                    "cluster_bootstrap_ci_high": ci_high,
                    "n_publisher_clusters": len(publishers),
                    "n_bootstrap_samples": n_boot,
                }
            )

    return pd.DataFrame(rows)


def publisher_composition(df):
    rows = []
    for decade, group in df.groupby("decade"):
        total = len(group)
        counts = group["publisher"].value_counts().head(10)
        for rank, (publisher, count) in enumerate(counts.items(), start=1):
            rows.append(
                {
                    "decade": decade,
                    "rank": rank,
                    "publisher": publisher,
                    "articles": int(count),
                    "share_of_decade_percent": count / total * 100,
                }
            )
    return pd.DataFrame(rows)


def long_text_audit(df):
    rows = []
    for decade, group in df.groupby("decade"):
        n = len(group)
        row = {
            "decade": decade,
            "n_articles": n,
            "median_word_count": group["word_count"].median(),
            "mean_word_count": group["word_count"].mean(),
        }
        for threshold in [256, 512, 1000, 1500]:
            row[f"n_word_count_gt_{threshold}"] = int((group["word_count"] > threshold).sum())
            row[f"share_word_count_gt_{threshold}_percent"] = (
                (group["word_count"] > threshold).mean() * 100
            )
        rows.append(row)
    return pd.DataFrame(rows)


def content_type_summary(df):
    counts = (
        df.groupby(["decade", "content_type", "content_subtype"], dropna=False)
        .size()
        .reset_index(name="articles")
    )
    totals = df.groupby("decade").size().rename("decade_total").reset_index()
    counts = counts.merge(totals, on="decade", how="left")
    counts["share_of_decade_percent"] = counts["articles"] / counts["decade_total"] * 100
    return counts.sort_values(["decade", "articles"], ascending=[True, False])


def adjusted_decade_trends(df):
    """
    Descriptive adjustment, not a formal inferential model.

    For each outcome, regress out article length, readability, and publisher
    composition, then report decade-level means of the adjusted outcome. This
    helps check whether the raw decade pattern is mostly driven by changing
    text structure or newspaper mix.
    """
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except Exception as exc:
        return pd.DataFrame(
            [
                {
                    "note": (
                        "Adjusted trend not computed because sklearn could not be "
                        f"imported: {type(exc).__name__}: {exc}"
                    )
                }
            ]
        )

    model_df = df[
        [
            "decade",
            "publisher",
            "word_count",
            "flesch_reading_ease",
            "flesch_kincaid_grade",
            "sentiment_pos_minus_neg",
            "emotion_distress",
            "fear",
            "sadness",
            "joy",
        ]
    ].dropna()
    model_df["log_word_count"] = np.log1p(model_df["word_count"])

    x = model_df[
        ["publisher", "log_word_count", "flesch_reading_ease", "flesch_kincaid_grade"]
    ]
    preprocessor = ColumnTransformer(
        [
            ("publisher", OneHotEncoder(handle_unknown="ignore"), ["publisher"]),
            (
                "numeric",
                StandardScaler(),
                ["log_word_count", "flesch_reading_ease", "flesch_kincaid_grade"],
            ),
        ]
    )

    rows = []
    for outcome in [
        "sentiment_pos_minus_neg",
        "emotion_distress",
        "fear",
        "sadness",
        "joy",
    ]:
        y = model_df[outcome].to_numpy()
        pipe = make_pipeline(preprocessor, Ridge(alpha=1.0))
        pipe.fit(x, y)
        residual = y - pipe.predict(x)
        adjusted = residual + y.mean()
        tmp = pd.DataFrame(
            {
                "decade": model_df["decade"].to_numpy(),
                "outcome": outcome,
                "adjusted_mean": adjusted,
                "raw_mean": y,
            }
        )
        grouped = (
            tmp.groupby(["decade", "outcome"])
            .agg(
                adjusted_mean=("adjusted_mean", "mean"),
                raw_mean=("raw_mean", "mean"),
                n_articles=("raw_mean", "size"),
            )
            .reset_index()
        )
        rows.append(grouped)

    return pd.concat(rows, ignore_index=True)


def manual_validation_sample(df, n_per_decade, seed):
    sample_cols = [
        "article_id",
        "date",
        "year",
        "decade",
        "publisher",
        "headline",
        "content_type",
        "content_subtype",
        "word_count",
        "sentiment_pos_minus_neg",
        "emotion_distress",
        "text",
    ]
    return (
        df.groupby("decade", group_keys=False)
        .apply(lambda x: x.sample(min(n_per_decade, len(x)), random_state=seed))
        [sample_cols]
        .sort_values(["decade", "year", "publisher"])
    )


def stage_comparison_percent(input_path, output_path):
    if not input_path.exists():
        return
    df = pd.read_csv(input_path)
    out = df.copy()
    for col in out.columns:
        if col.endswith("_share_of_original_articles") or col.endswith(
            "_share_of_selected_articles"
        ) or "share" in col:
            out[col.replace("share", "percent")] = out[col] * 100
    out.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Create dissertation-ready descriptive and robustness tables."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path(
            "data/processed/american_stories_features_full/"
            "american_stories_article_features.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dissertation_analysis_tables"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--sample-per-decade", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {args.features}")
    df = pd.read_csv(args.features, usecols=FEATURE_COLUMNS)
    df = df[df["decade"].between(1840, 1960)].copy()
    df["five_year_start"] = (df["year"] // 5) * 5
    df["five_year_period"] = (
        df["five_year_start"].astype(str)
        + "-"
        + (df["five_year_start"] + 4).astype(str)
    )

    print("Writing decade descriptive statistics")
    decade = summarise_by_group(df, "decade")
    decade.to_csv(args.output_dir / "decade_descriptive_statistics.csv", index=False)

    print("Writing five-year robustness summary")
    five_year = summarise_by_group(df, "five_year_period")
    five_year.to_csv(args.output_dir / "five_year_robustness_summary.csv", index=False)

    print("Writing publisher composition")
    publisher_composition(df).to_csv(
        args.output_dir / "publisher_composition_by_decade_top10.csv",
        index=False,
    )

    print("Writing long-text truncation audit")
    long_text_audit(df).to_csv(args.output_dir / "long_text_truncation_audit.csv", index=False)

    print("Writing content-type summary")
    content_type_summary(df).to_csv(args.output_dir / "content_type_summary_by_decade.csv", index=False)

    print("Writing adjusted decade trends")
    adjusted_decade_trends(df).to_csv(args.output_dir / "adjusted_decade_trends.csv", index=False)

    print("Writing cluster bootstrap confidence intervals")
    cluster_bootstrap_ci(
        df,
        "decade",
        [
            "sentiment_pos_minus_neg",
            "negative",
            "positive",
            "emotion_distress",
            "fear",
            "sadness",
            "joy",
        ],
        n_boot=args.bootstrap_samples,
        seed=args.seed,
    ).to_csv(args.output_dir / "decade_cluster_bootstrap_confidence_intervals.csv", index=False)

    print("Writing manual validation sample")
    manual_validation_sample(df, args.sample_per_decade, args.seed).to_csv(
        args.output_dir / "manual_validation_sample_by_decade.csv",
        index=False,
    )

    print("Writing corpus-stage percentage tables")
    stage_comparison_percent(
        Path("outputs/corpus_stage_comparison/corpus_stage_comparison_by_decade.csv"),
        args.output_dir / "corpus_stage_comparison_by_decade_percent.csv",
    )
    stage_comparison_percent(
        Path("outputs/corpus_stage_comparison/corpus_stage_comparison_totals.csv"),
        args.output_dir / "corpus_stage_comparison_totals.csv",
    )

    print(f"Done: {args.output_dir}")


if __name__ == "__main__":
    main()
