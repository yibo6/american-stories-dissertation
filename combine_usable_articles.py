import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Combine yearly keep-only usable article files into one final CSV."
    )
    parser.add_argument("--quality-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    usable_files = sorted(args.quality_dir.glob("*/usable_articles.csv"))
    if not usable_files:
        raise SystemExit(f"No usable_articles.csv files found in {args.quality_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    fieldnames = None

    with args.output.open("w", newline="", encoding="utf-8") as out_f:
        writer = None
        for usable_file in usable_files:
            year = usable_file.parent.name
            with usable_file.open(newline="", encoding="utf-8") as in_f:
                reader = csv.DictReader(in_f)
                if fieldnames is None:
                    fieldnames = ["year", *reader.fieldnames]
                    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
                    writer.writeheader()

                for row in reader:
                    writer.writerow({"year": year, **row})
                    total_rows += 1

    print(f"Usable files combined: {len(usable_files)}")
    print(f"Final keep-only articles: {total_rows:,}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
