import argparse
import csv
import json
import subprocess
import tarfile
import time
from pathlib import Path


BASE_URL = (
    "https://huggingface.co/datasets/dell-research-harvard/"
    "AmericanStories/resolve/main"
)


def normalise_publisher(value):
    return " ".join((value or "").split())


def load_selected_publishers(selection_path):
    selected = set()
    with selection_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            variants = row["american_stories_title_variants"].split(" | ")
            for variant in variants:
                variant = normalise_publisher(variant)
                if variant:
                    selected.add(variant)
    return selected


def iter_articles_from_archive(archive_path, year):
    year_dir = f"faro_{year}"
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.startswith(year_dir):
                continue

            extracted = archive.extractfile(member)
            if extracted is None:
                continue

            try:
                data = json.loads(extracted.read().decode("utf-8"))
            except Exception:
                continue

            if "lccn" not in data:
                continue

            filename = Path(member.name).name
            scan_id = filename.split(".")[0]
            scan_date = filename.split("_")[0]
            publisher = normalise_publisher(data["lccn"]["title"])

            for article in data.get("full articles", []):
                yield {
                    "date": scan_date,
                    "publisher": publisher,
                    "headline": article.get("headline", ""),
                    "text": article.get("article", ""),
                }


def archive_is_complete(archive_path):
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            for _ in archive:
                pass
        return True
    except Exception:
        return False


def download_archive(year, archive_path, retries=10):
    url = f"{BASE_URL}/faro_{year}.tar.gz"
    part_path = archive_path.with_suffix(archive_path.suffix + ".part")

    if archive_path.exists() and archive_is_complete(archive_path):
        print(f"{year}: using existing archive", flush=True)
        return

    archive_path.unlink(missing_ok=True)

    for attempt in range(1, retries + 1):
        try:
            print(f"{year}: downloading attempt {attempt}/{retries}", flush=True)
            subprocess.run(
                [
                    "curl",
                    "-L",
                    "--fail",
                    "--retry",
                    "5",
                    "--retry-delay",
                    "5",
                    "--connect-timeout",
                    "30",
                    "-C",
                    "-",
                    "-o",
                    str(part_path),
                    url,
                ],
                check=True,
            )
            part_path.replace(archive_path)
            if not archive_is_complete(archive_path):
                archive_path.unlink(missing_ok=True)
                part_path.unlink(missing_ok=True)
                raise tarfile.TarError("downloaded file is not a complete tar.gz")
            return
        except Exception as exc:
            if attempt == retries:
                raise
            print(f"{year}: download failed, retrying ({exc})", flush=True)
            time.sleep(5)


def process_year(year, raw_dir, selected_dir, archive_dir, selected_publishers, overwrite):
    archive_path = archive_dir / f"faro_{year}.tar.gz"
    selected_path = selected_dir / f"selected_sources_{year}.csv"
    selected_part_path = selected_path.with_suffix(selected_path.suffix + ".part")

    if selected_path.exists() and not overwrite:
        print(f"{year}: already done", flush=True)
        return

    archive_dir.mkdir(parents=True, exist_ok=True)
    selected_dir.mkdir(parents=True, exist_ok=True)

    download_archive(year, archive_path)

    fields = ["date", "publisher", "headline", "text"]
    raw_count = 0
    selected_count = 0

    selected_part_path.unlink(missing_ok=True)

    with selected_part_path.open("w", newline="", encoding="utf-8") as selected_f:
        selected_writer = csv.DictWriter(selected_f, fieldnames=fields)
        selected_writer.writeheader()

        for row in iter_articles_from_archive(archive_path, year):
            raw_count += 1

            if row["publisher"] in selected_publishers:
                selected_writer.writerow(row)
                selected_count += 1

    selected_part_path.replace(selected_path)
    archive_path.unlink(missing_ok=True)
    print(
        f"{year}: raw={raw_count:,}, selected_sources={selected_count:,}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download American Stories yearly archives, create source-selected "
            "yearly CSVs, and delete each archive after processing."
        )
    )
    parser.add_argument("--start-year", type=int, default=1837)
    parser.add_argument("--end-year", type=int, default=1964)
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("outputs/final_source_list_for_text_screening.csv"),
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/AmericanStories"),
        help="Deprecated. Raw full-year CSVs are no longer written.",
    )
    parser.add_argument(
        "--selected-dir",
        type=Path,
        default=Path("data/processed/selected_sources_by_year"),
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=Path("data/raw/AmericanStories_archives"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    selected_publishers = load_selected_publishers(args.selection)
    print(f"Selected publication title variants: {len(selected_publishers):,}", flush=True)

    for year in range(args.start_year, args.end_year + 1):
        process_year(
            year=year,
            raw_dir=args.raw_dir,
            selected_dir=args.selected_dir,
            archive_dir=args.archive_dir,
            selected_publishers=selected_publishers,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
