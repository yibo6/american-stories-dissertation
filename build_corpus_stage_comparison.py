import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


def year_from_name(path):
    match = re.search(r"(\d{4})", path.name)
    return int(match.group(1)) if match else None


def decade_from_year(year):
    return f"{year // 10 * 10}s"


def word_count(text):
    return len(re.findall(r"\S+", text or ""))


def empty_counter():
    return {"articles": 0, "words": 0, "publishers": set()}


def add_row(summary, stage, year, publisher, words):
    key = (stage, year)
    summary[key]["articles"] += 1
    summary[key]["words"] += words
    if publisher:
        summary[key]["publishers"].add(publisher)


def load_original_summary(path, start_year, end_year):
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            decade = row["Decade"]
            decade_start = int(decade[:4])
            if decade_start > end_year or decade_start + 9 < start_year:
                continue
            rows.append(
                {
                    "stage": "original_american_stories",
                    "decade": decade,
                    "articles": int(row["Articles"]),
                    "words": int(row["Words"]),
                    "publisher": row["Publication"],
                }
            )
    return rows


def summarise_yearly_csvs(folder, pattern, stage, start_year, end_year):
    summary = defaultdict(empty_counter)
    for path in sorted(folder.glob(pattern)):
        year = year_from_name(path)
        if year is None or year < start_year or year > end_year:
            continue
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get("text", "")
                publisher = row.get("publisher", "")
                add_row(summary, stage, year, publisher, word_count(text))
    return summary


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compare Original, Selected Sources, and Rule-Based Filter stages."
    )
    parser.add_argument(
        "--original-summary",
        type=Path,
        default=Path("/Users/nora/Documents/AmericanStories_Summary_Table_1770_1964.csv"),
    )
    parser.add_argument(
        "--selected-dir",
        type=Path,
        default=Path("data/processed/selected_sources_by_year"),
    )
    parser.add_argument(
        "--rule-dir",
        type=Path,
        default=Path("data/processed/quality_by_year_final_sources"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/corpus_stage_comparison"),
    )
    parser.add_argument("--start-year", type=int, default=1837)
    parser.add_argument("--end-year", type=int, default=1964)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    original_rows = load_original_summary(
        args.original_summary, args.start_year, args.end_year
    )
    selected_summary = summarise_yearly_csvs(
        args.selected_dir,
        "selected_sources_*.csv",
        "selected_sources",
        args.start_year,
        args.end_year,
    )
    rule_summary = summarise_yearly_csvs(
        args.rule_dir,
        "usable_articles_*.csv",
        "rule_based_filtered",
        args.start_year,
        args.end_year,
    )

    original_by_decade = defaultdict(empty_counter)
    for row in original_rows:
        key = ("original_american_stories", row["decade"])
        original_by_decade[key]["articles"] += row["articles"]
        original_by_decade[key]["words"] += row["words"]
        original_by_decade[key]["publishers"].add(row["publisher"])

    yearly_rows = []
    for (stage, year), values in sorted(
        {**selected_summary, **rule_summary}.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        yearly_rows.append(
            {
                "year": year,
                "decade": decade_from_year(year),
                "stage": stage,
                "articles": values["articles"],
                "words": values["words"],
                "publishers": len(values["publishers"]),
            }
        )

    decade_stage = defaultdict(empty_counter)
    for (stage, decade), values in original_by_decade.items():
        decade_stage[(decade, stage)]["articles"] += values["articles"]
        decade_stage[(decade, stage)]["words"] += values["words"]
        decade_stage[(decade, stage)]["publishers"].update(values["publishers"])

    for summary in [selected_summary, rule_summary]:
        for (stage, year), values in summary.items():
            decade = decade_from_year(year)
            decade_stage[(decade, stage)]["articles"] += values["articles"]
            decade_stage[(decade, stage)]["words"] += values["words"]
            decade_stage[(decade, stage)]["publishers"].update(values["publishers"])

    decade_rows = []
    stages = [
        "original_american_stories",
        "selected_sources",
        "rule_based_filtered",
    ]
    for decade in sorted({key[0] for key in decade_stage}, key=lambda x: int(x[:4])):
        row = {"decade": decade}
        for stage in stages:
            values = decade_stage.get((decade, stage), empty_counter())
            row[f"{stage}_articles"] = values["articles"]
            row[f"{stage}_words"] = values["words"]
            row[f"{stage}_publishers"] = len(values["publishers"])
        original_articles = row["original_american_stories_articles"]
        selected_articles = row["selected_sources_articles"]
        row["selected_share_of_original_articles"] = (
            selected_articles / original_articles if original_articles else ""
        )
        row["rule_based_share_of_selected_articles"] = (
            row["rule_based_filtered_articles"] / selected_articles
            if selected_articles
            else ""
        )
        row["rule_based_share_of_original_articles"] = (
            row["rule_based_filtered_articles"] / original_articles
            if original_articles
            else ""
        )
        decade_rows.append(row)

    original_year_estimates = {}
    for row in original_rows:
        decade_start = int(row["decade"][:4])
        decade_years = [
            year
            for year in range(decade_start, decade_start + 10)
            if args.start_year <= year <= args.end_year
        ]
        if not decade_years:
            continue
        articles_per_year = row["articles"] / len(decade_years)
        words_per_year = row["words"] / len(decade_years)
        for year in decade_years:
            values = original_year_estimates.setdefault(
                year, {"articles": 0.0, "words": 0.0}
            )
            values["articles"] += articles_per_year
            values["words"] += words_per_year

    selected_by_year = {
        year: values for (stage, year), values in selected_summary.items()
    }
    rule_by_year = {
        year: values for (stage, year), values in rule_summary.items()
    }
    wide_year_rows = []
    for year in range(args.start_year, args.end_year + 1):
        original = original_year_estimates.get(year, {"articles": "", "words": ""})
        selected = selected_by_year.get(year, empty_counter())
        rule = rule_by_year.get(year, empty_counter())
        selected_articles = selected["articles"]
        rule_articles = rule["articles"]
        original_articles = original["articles"]
        wide_year_rows.append(
            {
                "year": year,
                "decade": decade_from_year(year),
                "original_articles_estimated_from_decade": (
                    round(original_articles) if original_articles != "" else ""
                ),
                "selected_articles": selected_articles,
                "rule_based_filtered_articles": rule_articles,
                "selected_share_of_original_estimate": (
                    selected_articles / original_articles
                    if original_articles
                    else ""
                ),
                "rule_based_share_of_selected": (
                    rule_articles / selected_articles if selected_articles else ""
                ),
                "original_words_estimated_from_decade": (
                    round(original["words"]) if original["words"] != "" else ""
                ),
                "selected_words": selected["words"],
                "rule_based_filtered_words": rule["words"],
                "selected_publishers": len(selected["publishers"]),
                "rule_based_filtered_publishers": len(rule["publishers"]),
            }
        )

    stage_totals = []
    for stage in stages:
        articles = 0
        words = 0
        publishers = set()
        if stage == "original_american_stories":
            source = original_by_decade.values()
        else:
            source = (
                selected_summary.values()
                if stage == "selected_sources"
                else rule_summary.values()
            )
        for values in source:
            articles += values["articles"]
            words += values["words"]
            publishers.update(values["publishers"])
        stage_totals.append(
            {
                "stage": stage,
                "articles": articles,
                "words": words,
                "publishers": len(publishers),
            }
        )

    write_csv(
        args.output_dir / "corpus_stage_comparison_by_year.csv",
        yearly_rows,
        ["year", "decade", "stage", "articles", "words", "publishers"],
    )
    write_csv(
        args.output_dir / "corpus_stage_comparison_by_year_wide_article_counts.csv",
        wide_year_rows,
        list(wide_year_rows[0].keys()) if wide_year_rows else [],
    )
    write_csv(
        args.output_dir / "corpus_stage_comparison_by_decade.csv",
        decade_rows,
        list(decade_rows[0].keys()) if decade_rows else [],
    )
    write_csv(
        args.output_dir / "corpus_stage_comparison_totals.csv",
        stage_totals,
        ["stage", "articles", "words", "publishers"],
    )

    try:
        import pandas as pd

        xlsx_path = args.output_dir / "corpus_stage_comparison.xlsx"
        with pd.ExcelWriter(xlsx_path) as writer:
            pd.DataFrame(stage_totals).to_excel(writer, "totals", index=False)
            pd.DataFrame(decade_rows).to_excel(writer, "by_decade", index=False)
            pd.DataFrame(wide_year_rows).to_excel(
                writer, "by_year_wide", index=False
            )
            pd.DataFrame(yearly_rows).to_excel(writer, "by_year", index=False)
        print(f"Excel: {xlsx_path}")
    except Exception as exc:
        print(f"Excel not written: {exc}")

    print(f"CSV outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
