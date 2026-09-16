import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
QUALITY_SCRIPT = SCRIPT_DIR / "assess_full_year_text_quality.py"


def extract_year(path):
    """Extract the four-digit year from a file named stories_YYYY.csv."""
    match = re.fullmatch(r"stories_(\d{4})\.csv", path.name)
    return int(match.group(1)) if match else None


def file_fingerprint(path):
    """Return a stable SHA-256 fingerprint for a small configuration file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run OCR readability checks and source filtering for the final "
            "American Stories newspaper sources in every stories_YYYY.csv file."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Folder containing downloaded stories_YYYY.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "data" / "processed" / "quality_by_year_final_sources",
        help="Folder in which yearly usable_articles_YYYY.csv files will be created.",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=(
            SCRIPT_DIR
            / "outputs"
            / "final_source_list_for_text_screening.csv"
        ),
        help=(
            "CSV containing the selected American Stories newspaper names. "
            "Defaults to the final historical sources selected for screening."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Process a year again even when its output already exists.",
    )
    parser.add_argument(
        "--skip-article-flags",
        action="store_true",
        help=(
            "Do not write per-article audit files. This keeps only "
            "usable_articles.csv, publisher summaries, duplicate groups, "
            "and run metadata for each year."
        ),
    )
    parser.add_argument(
        "--minimal-output",
        action="store_true",
        help="Write only usable_articles.csv for each year.",
    )
    parser.add_argument(
        "--apply-content-filter",
        action="store_true",
        help=(
            "Optionally exclude records flagged as obvious non-article notices. "
            "This is off by default because the main dataset focuses on "
            "OCR/readability and duplicate removal."
        ),
    )
    args = parser.parse_args()
    if args.minimal_output:
        args.skip_article_flags = True

    if not args.input_dir.is_dir():
        raise SystemExit(f"Input folder does not exist: {args.input_dir}")
    if not QUALITY_SCRIPT.exists():
        raise SystemExit(f"Quality-check script not found: {QUALITY_SCRIPT}")
    if not args.selection.exists():
        raise SystemExit(f"Source-selection file not found: {args.selection}")

    input_files = []
    for path in args.input_dir.glob("stories_*.csv"):
        year = extract_year(path)
        if year is not None:
            input_files.append((year, path))
    input_files.sort()

    if not input_files:
        raise SystemExit(
            f"No files named stories_YYYY.csv were found in {args.input_dir}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(input_files)} yearly CSV files.")
    selection_hash = file_fingerprint(args.selection)

    completed = 0
    skipped = 0

    for position, (year, input_path) in enumerate(input_files, start=1):
        completion_file = args.output_dir / f"usable_articles_{year}.csv"
        metadata_file = args.output_dir / f"usable_articles_{year}.metadata.json"
        input_stat = input_path.stat()
        expected_metadata = {
            "input_path": str(input_path.resolve()),
            "input_size": input_stat.st_size,
            "input_modified_ns": input_stat.st_mtime_ns,
            "selection_path": str(args.selection.resolve()),
            "selection_sha256": selection_hash,
            "skip_article_flags": args.skip_article_flags,
            "apply_content_filter": args.apply_content_filter,
        }

        if (
            completion_file.exists()
            and not args.overwrite
        ):
            print(f"[{position}/{len(input_files)}] {year}: already processed")
            skipped += 1
            continue

        print(f"[{position}/{len(input_files)}] Processing {year}: {input_path}")
        with tempfile.TemporaryDirectory(
            prefix=f"quality_{year}_", dir=args.output_dir
        ) as tmp_dir:
            year_output = Path(tmp_dir)
            command = [
                sys.executable,
                str(QUALITY_SCRIPT),
                "--input",
                str(input_path),
                "--selection",
                str(args.selection),
                "--output-dir",
                str(year_output),
            ]
            if args.skip_article_flags:
                command.append("--skip-article-flags")
            if args.minimal_output:
                command.append("--minimal-output")
            if args.apply_content_filter:
                command.append("--apply-content-filter")
            subprocess.run(command, check=True)
            generated_file = year_output / "usable_articles.csv"
            if not generated_file.exists():
                raise SystemExit(f"Expected output was not created: {generated_file}")
            generated_file.replace(completion_file)
            metadata_file.write_text(
                json.dumps(expected_metadata, indent=2),
                encoding="utf-8",
            )
        completed += 1

    print(
        f"Finished. Processed {completed} year(s), skipped {skipped} "
        f"completed year(s). Results: {args.output_dir}"
    )


if __name__ == "__main__":
    main()
