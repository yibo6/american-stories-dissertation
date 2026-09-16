# American Stories historical newspaper filtering and wellbeing-language analysis

This repository contains the code and selected reproducible outputs that correspond to the dissertation workflow for constructing and analysing a filtered American Stories historical newspaper corpus. The final analysis builds a source-selected and OCR/readability-screened corpus, extracts article-level sentiment and emotion-language features, and summarises long-run patterns by decade.

## Project structure

The repository is intentionally limited to the scripts and outputs that match the submitted dissertation. Earlier exploratory source-list scripts, draft-writing scripts and large article-level CSV files are kept locally but are not tracked here.

### Main corpus workflow

- `outputs/final_source_list_for_text_screening.csv` is the final source-selection file used for American Stories filtering.
- `download_and_filter_selected_sources.py` downloads yearly American Stories archives and writes selected-source CSV files.
- `assess_full_year_text_quality.py` applies the rule-based OCR/readability, fragmentation and duplicate screening used in the dissertation.
- `run_all_year_quality_checks.py` runs the quality screen across yearly files.
- `combine_usable_articles.py` combines yearly retained article files when needed.

### Feature extraction and dissertation analysis

- `extract_american_stories_features.py` extracts sentiment, emotion, readability and text-length features from retained articles.
- `build_corpus_stage_comparison.py` compares corpus-construction stages.
- `make_dissertation_analysis_tables.py` creates the decade-level, robustness and diagnostic summary tables reported in the dissertation.
- `make_dissertation_figures.py` and `make_dissertation_extra_figures.py` generate the final dissertation figures.
- `data/processed/american_stories_features_full/american_stories_decade_features.csv` and `american_stories_yearly_features.csv` are compact summary outputs from feature extraction.
- `outputs/dissertation_analysis_tables/` contains the final tables used for the dissertation results and robustness checks.
- `outputs/dissertation_figures/` contains the final figure files and figure-source CSVs.

### Exploratory checks mentioned in the dissertation

- `validate_transformer_ocr_sample.py`, `compare_ocr_correction_model.py`, `apply_transformer_ocr_filter.py` and `filter_selected_sources_with_optional_ocr_model.py` relate to exploratory transformer OCR-correction checks. These were not used as the main filtering method.
- `zero_shot_content_type_audit.py` relates to the exploratory zero-shot content-type audit. It was not used as the main filtering method.

## Data policy

The full local `data/` directory is not committed because it contains large yearly American Stories CSV files and intermediate article-level outputs. The committed data are limited to compact summary tables needed to inspect the reported results:

- `data/processed/american_stories_features_full/american_stories_decade_features.csv`
- `data/processed/american_stories_features_full/american_stories_yearly_features.csv`
- selected CSV and XLSX summaries under `outputs/`

The raw newspaper text is derived from the public American Stories dataset. To reproduce the full workflow, download or generate yearly American Stories files locally, then run the scripts in the order below.

## Reproduction outline

1. Prepare or download yearly American Stories CSV files.
2. Select comparable historical source units:

   ```bash
   python3 download_and_filter_selected_sources.py \
     --start-year 1837 \
     --end-year 1964 \
     --selection outputs/final_source_list_for_text_screening.csv \
     --selected-dir data/processed/selected_sources_by_year
   ```

3. Apply OCR/readability and duplicate screening:

   ```bash
   python3 run_all_year_quality_checks.py \
     --input-dir data/processed/selected_sources_by_year \
     --output-dir data/processed/quality_by_year_final_sources \
     --overwrite \
     --minimal-output
   ```

4. Extract NLP features:

   ```bash
   python3 extract_american_stories_features.py \
     --input-dir data/processed/quality_by_year_final_sources \
     --output-dir data/processed/american_stories_features_full
   ```

5. Generate analysis tables and figures:

   ```bash
   python3 build_corpus_stage_comparison.py
   python3 make_dissertation_analysis_tables.py
   python3 make_dissertation_figures.py
   python3 make_dissertation_extra_figures.py
   ```

Some scripts accept additional command-line options; run each script with `--help` where available.

## Notes

The repository intentionally preserves the rule-based screening code and selected outputs rather than uploading the entire 59GB local data directory. This keeps the GitHub repository lightweight while retaining the main computational workflow and the tables and figures needed to audit the dissertation results.
