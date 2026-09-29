# Data inventory

This file explains which data files are included in the GitHub repository and which large intermediate files are not included.

## Included in this repository

### Final NOW/American Stories source-selection list

- `outputs/final_source_list_for_text_screening.csv`

This is the final list of historical American Stories source units used for the dissertation workflow. It corresponds to the selected historical newspaper sources matched or made comparable to the modern NOW Corpus source-selection discussion.

### Rule-based filtered article files

- `data/processed/quality_by_year_final_sources/usable_articles_YYYY.csv`

These are the final OCR/readability-filtered article files used as the main retained American Stories corpus before feature extraction. They contain the article-level retained records after source selection, OCR/readability screening, fragmentation screening and duplicate removal.

### Feature summaries

- `data/processed/american_stories_features_full/american_stories_yearly_features.csv`
- `data/processed/american_stories_features_full/american_stories_decade_features.csv`

These are compact yearly and decade-level summaries derived from the feature extraction workflow.

### Analysis tables and figures

- `outputs/dissertation_analysis_tables/`
- `outputs/dissertation_figures/`

These contain the summary tables and figures used in the dissertation.

## Not included because of size

### Selected-source intermediate files

- Local path: `data/processed/selected_sources_by_year/`
- Approximate local size: 57GB

These files are the yearly selected-source intermediate CSVs before OCR/readability filtering. They are not uploaded to GitHub because of their size. They can be regenerated using:

```bash
python3 download_and_filter_selected_sources.py \
  --start-year 1837 \
  --end-year 1964 \
  --selection outputs/final_source_list_for_text_screening.csv \
  --selected-dir data/processed/selected_sources_by_year
```

### Full raw American Stories yearly files

The full raw American Stories files are public but very large. They are not included in this repository. The repository contains the scripts needed to download/filter the relevant selected-source files and the final filtered article files used in the dissertation analysis.
