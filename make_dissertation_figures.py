from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


BASE = Path("data/processed/american_stories_features_full")
COMPARISON = Path("outputs/corpus_stage_comparison")
OUT = Path("outputs/dissertation_figures")


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.frameon": False,
        "figure.dpi": 150,
    }
)


PALETTE = {
    "ink": "#1f2933",
    "muted": "#6b7280",
    "grid": "#d7dde5",
    "original": "#9aa6b2",
    "selected": "#4c78a8",
    "filtered": "#2f8f7f",
    "positive": "#2a9d8f",
    "negative": "#b44e4e",
    "neutral": "#7b8da4",
    "distress": "#7c4d8a",
    "anger": "#c44536",
    "fear": "#5b6fb7",
    "sadness": "#4f87a0",
    "joy": "#d99b2b",
}


def save_figure(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / stem
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(path.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def decade_label(decade):
    decade = int(decade)
    return f"{decade}s*" if decade == 1960 else f"{decade}s"


def load_main_data():
    decade = pd.read_csv(BASE / "american_stories_decade_features.csv")
    yearly = pd.read_csv(BASE / "american_stories_yearly_features.csv")
    stage_decade = pd.read_csv(COMPARISON / "corpus_stage_comparison_by_decade.csv")
    totals = pd.read_csv(COMPARISON / "corpus_stage_comparison_totals.csv")

    # Main dissertation figures start from the 1840s. The 1830s only includes
    # 1837-1839 and is better treated as a sensitivity or appendix period.
    decade = decade[decade["decade"] >= 1840].copy()
    yearly = yearly[yearly["year"] >= 1840].copy()
    stage_decade = stage_decade[stage_decade["decade"] != "1830s"].copy()

    decade["decade_label"] = decade["decade"].map(decade_label)
    stage_decade["decade_num"] = stage_decade["decade"].str[:4].astype(int)
    stage_decade["decade_label"] = stage_decade["decade_num"].map(decade_label)
    return decade, yearly, stage_decade, totals


def format_thousands(values):
    return [f"{int(v):,}" for v in values]


def format_count_label(value):
    """Compact count labels without superscripts on log axes."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{int(value)}"


def style_axis(ax, ygrid=True):
    ax.tick_params(axis="both", labelsize=7.5, colors=PALETTE["ink"])
    ax.xaxis.label.set_color(PALETTE["ink"])
    ax.yaxis.label.set_color(PALETTE["ink"])
    if ygrid:
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)


def rotate_decade_labels(ax):
    """Keep decade labels readable and stable after PDF/SVG export."""
    for label in ax.get_xticklabels():
        label.set_rotation(35)
        label.set_ha("right")
        label.set_rotation_mode("anchor")


def figure_1_corpus_construction(stage_decade, totals):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [0.9, 1.8]})

    order = ["original_american_stories", "selected_sources", "rule_based_filtered"]
    labels = ["Full American Stories", "Selected sources", "Filtered usable"]
    colors = [PALETTE["original"], PALETTE["selected"], PALETTE["filtered"]]
    total_map = totals.set_index("stage")["articles"].to_dict()
    values = [total_map[s] for s in order]

    ax = axes[0]
    bars = ax.bar(labels, values, color=colors, width=0.65)
    ax.set_yscale("log")
    ax.set_ylim(100_000, 1_000_000_000)
    ax.set_yticks([100_000, 1_000_000, 10_000_000, 100_000_000, 1_000_000_000])
    ax.set_yticklabels(["100k", "1M", "10M", "100M", "1B"])
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_ylabel("Articles, log scale")
    ax.set_title("a  Corpus construction stages", loc="left", fontweight="bold", pad=8)
    ax.tick_params(axis="x", rotation=35)
    rotate_decade_labels(ax)
    style_axis(ax)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value * 1.1,
            format_count_label(value),
            ha="center",
            va="bottom",
            fontsize=7,
            color=PALETTE["ink"],
        )

    ax = axes[1]
    x = np.arange(len(stage_decade))
    ax.plot(
        x,
        stage_decade["selected_sources_articles"],
        marker="o",
        linewidth=1.5,
        color=PALETTE["selected"],
        label="Selected sources",
    )
    ax.plot(
        x,
        stage_decade["rule_based_filtered_articles"],
        marker="o",
        linewidth=1.5,
        color=PALETTE["filtered"],
        label="Filtered usable",
    )
    ax.set_yscale("log")
    ax.set_ylim(1_000, 20_000_000)
    ax.set_yticks([1_000, 10_000, 100_000, 1_000_000, 10_000_000])
    ax.set_yticklabels(["1k", "10k", "100k", "1M", "10M"])
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xticks(x)
    ax.set_xticklabels(stage_decade["decade_label"], rotation=35, ha="right")
    rotate_decade_labels(ax)
    ax.set_ylabel("Articles by decade, log scale")
    ax.set_title("b  Retained records by decade", loc="left", fontweight="bold", pad=8)
    ax.legend(loc="upper left", fontsize=7)
    style_axis(ax)

    fig.text(
        0.99,
        -0.02,
        "Note: 1960s includes 1960-1963.",
        ha="right",
        va="top",
        fontsize=7,
        color=PALETTE["muted"],
    )
    fig.tight_layout()
    save_figure(fig, "fig1_corpus_construction")


def figure_2_coverage(decade, yearly):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))

    ax = axes[0]
    x = np.arange(len(decade))
    ax.bar(x, decade["n_articles"], color=PALETTE["filtered"], width=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(decade["decade_label"], rotation=35, ha="right")
    rotate_decade_labels(ax)
    ax.set_ylabel("Filtered articles")
    ax.set_title("a  Usable articles by decade", loc="left", fontweight="bold", pad=8)
    style_axis(ax)

    ax = axes[1]
    ax.plot(yearly["year"], yearly["n_articles"], color=PALETTE["filtered"], linewidth=1.2)
    ax.fill_between(yearly["year"], yearly["n_articles"], color=PALETTE["filtered"], alpha=0.18)
    ax.set_xlabel("Year")
    ax.set_ylabel("Filtered articles")
    ax.set_title("b  Yearly coverage after filtering", loc="left", fontweight="bold", pad=8)
    style_axis(ax)

    fig.tight_layout()
    save_figure(fig, "fig2_article_coverage")


def figure_3_sentiment(decade):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    x = np.arange(len(decade))

    ax = axes[0]
    ax.plot(x, decade["mean_positive"], marker="o", color=PALETTE["positive"], linewidth=1.6, label="Positive")
    ax.plot(x, decade["mean_negative"], marker="o", color=PALETTE["negative"], linewidth=1.6, label="Negative")
    ax.plot(x, decade["mean_sent_neutral"], marker="o", color=PALETTE["neutral"], linewidth=1.3, label="Neutral")
    ax.set_xticks(x)
    ax.set_xticklabels(decade["decade_label"], rotation=35, ha="right")
    rotate_decade_labels(ax)
    ax.set_ylabel("Mean model score")
    ax.set_ylim(0, 0.75)
    ax.set_title("a  Sentiment components", loc="left", fontweight="bold", pad=8)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=7)
    style_axis(ax)

    ax = axes[1]
    ax.axhline(0, color=PALETTE["grid"], linewidth=1.0)
    ax.plot(
        x,
        decade["mean_sentiment_pos_minus_neg"],
        marker="o",
        color=PALETTE["ink"],
        linewidth=1.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(decade["decade_label"], rotation=35, ha="right")
    rotate_decade_labels(ax)
    ax.set_ylabel("Positive minus negative")
    ax.set_title("b  Net sentiment", loc="left", fontweight="bold", pad=8)
    style_axis(ax)

    fig.tight_layout()
    save_figure(fig, "fig3_sentiment_by_decade")


def figure_4_emotion(decade):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    x = np.arange(len(decade))

    ax = axes[0]
    ax.plot(x, decade["mean_emotion_distress"], marker="o", color=PALETTE["distress"], linewidth=1.8, label="Distress")
    ax.plot(x, decade["mean_joy"], marker="o", color=PALETTE["joy"], linewidth=1.5, label="Joy")
    ax.set_xticks(x)
    ax.set_xticklabels(decade["decade_label"], rotation=35, ha="right")
    rotate_decade_labels(ax)
    ax.set_ylabel("Mean model score")
    ax.set_ylim(0, 0.6)
    ax.set_title("a  Distress and joy", loc="left", fontweight="bold", pad=8)
    ax.legend(loc="upper right", fontsize=7)
    style_axis(ax)

    ax = axes[1]
    ax.plot(x, decade["mean_anger"], marker="o", color=PALETTE["anger"], linewidth=1.5, label="Anger")
    ax.plot(x, decade["mean_fear"], marker="o", color=PALETTE["fear"], linewidth=1.5, label="Fear")
    ax.plot(x, decade["mean_sadness"], marker="o", color=PALETTE["sadness"], linewidth=1.5, label="Sadness")
    ax.set_xticks(x)
    ax.set_xticklabels(decade["decade_label"], rotation=35, ha="right")
    rotate_decade_labels(ax)
    ax.set_ylabel("Mean model score")
    ax.set_ylim(0, 0.28)
    ax.set_title("b  Distress components", loc="left", fontweight="bold", pad=8)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=7)
    style_axis(ax)

    fig.tight_layout()
    save_figure(fig, "fig4_emotion_by_decade")


def figure_5_text_characteristics(decade):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    x = np.arange(len(decade))

    ax = axes[0]
    ax.plot(x, decade["mean_words"], marker="o", color=PALETTE["selected"], linewidth=1.6)
    ax.set_xticks(x)
    ax.set_xticklabels(decade["decade_label"], rotation=35, ha="right")
    rotate_decade_labels(ax)
    ax.set_ylabel("Mean words per article")
    ax.set_title("a  Article length", loc="left", fontweight="bold", pad=8)
    style_axis(ax)

    ax = axes[1]
    ax.plot(
        x,
        decade["mean_flesch_reading_ease"],
        marker="o",
        color=PALETTE["ink"],
        linewidth=1.6,
        label="Flesch reading ease",
    )
    ax2 = ax.twinx()
    ax2.plot(
        x,
        decade["share_non_article_like"] * 100,
        marker="s",
        color=PALETTE["muted"],
        linewidth=1.2,
        label="Potential non-article-like records",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(decade["decade_label"], rotation=35, ha="right")
    rotate_decade_labels(ax)
    ax.set_ylabel("Reading ease")
    ax2.set_ylabel("Flagged records (%)")
    ax.set_title("b  Readability and residual content flags", loc="left", fontweight="bold", pad=8)
    style_axis(ax)
    ax2.tick_params(axis="y", labelsize=7.5, colors=PALETTE["muted"])
    ax2.spines["top"].set_visible(False)
    lines = ax.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax.legend(
        lines,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        fontsize=7,
    )

    fig.tight_layout()
    save_figure(fig, "fig5_text_characteristics")


def write_source_tables(decade, stage_decade):
    OUT.mkdir(parents=True, exist_ok=True)
    decade[
        [
            "decade",
            "n_articles",
            "total_words",
            "mean_words",
            "n_publishers",
            "mean_negative",
            "mean_positive",
            "mean_sentiment_pos_minus_neg",
            "mean_emotion_distress",
            "mean_anger",
            "mean_fear",
            "mean_sadness",
            "mean_joy",
        ]
    ].to_csv(OUT / "figure_source_decade_features_1840_1960s.csv", index=False)
    stage_decade[
        [
            "decade",
            "original_american_stories_articles",
            "selected_sources_articles",
            "rule_based_filtered_articles",
            "selected_share_of_original_articles",
            "rule_based_share_of_selected_articles",
        ]
    ].to_csv(OUT / "figure_source_corpus_stages_1840_1960s.csv", index=False)


def write_qa_notes(decade, stage_decade):
    min_decade = int(decade["decade"].min())
    max_decade = int(decade["decade"].max())
    total_articles = int(decade["n_articles"].sum())
    min_articles = int(decade["n_articles"].min())
    max_articles = int(decade["n_articles"].max())
    notes = f"""# Dissertation figure QA notes

Backend: Python matplotlib.

Input files:
- {BASE / 'american_stories_decade_features.csv'}
- {BASE / 'american_stories_yearly_features.csv'}
- {COMPARISON / 'corpus_stage_comparison_by_decade.csv'}
- {COMPARISON / 'corpus_stage_comparison_totals.csv'}

Main plotting window:
- Decades included in main figures: {min_decade}s to {max_decade}s.
- The 1830s are excluded from main figures because that decade only contains 1837-1839.
- The 1960s are marked with an asterisk because the available feature output covers 1960-1963.

Data-integrity notes:
- Main decade-level plotted articles: {total_articles:,}.
- Minimum articles in a plotted decade: {min_articles:,}.
- Maximum articles in a plotted decade: {max_articles:,}.
- No rows were randomly sampled for plotting.
- Source data tables for plotted values are exported in this folder.

Panel map:
- Figure 1: Corpus construction stages and retained records by decade.
- Figure 2: Coverage after filtering by decade and year.
- Figure 3: Sentiment components and net sentiment by decade.
- Figure 4: Distress, joy, and distress components by decade.
- Figure 5: Article length, readability, and residual content-type flags.
"""
    (OUT / "figure_qa_notes.md").write_text(notes, encoding="utf-8")


def main():
    decade, yearly, stage_decade, totals = load_main_data()
    write_source_tables(decade, stage_decade)
    figure_1_corpus_construction(stage_decade, totals)
    figure_2_coverage(decade, yearly)
    figure_3_sentiment(decade)
    figure_4_emotion(decade)
    figure_5_text_characteristics(decade)
    write_qa_notes(decade, stage_decade)
    print(f"Saved figures to {OUT}")


if __name__ == "__main__":
    main()
