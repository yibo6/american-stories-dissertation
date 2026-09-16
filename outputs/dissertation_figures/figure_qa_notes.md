# Dissertation figure QA notes

Backend: Python matplotlib.

Input files:
- data/processed/american_stories_features_full/american_stories_decade_features.csv
- data/processed/american_stories_features_full/american_stories_yearly_features.csv
- outputs/corpus_stage_comparison/corpus_stage_comparison_by_decade.csv
- outputs/corpus_stage_comparison/corpus_stage_comparison_totals.csv

Main plotting window:
- Decades included in main figures: 1840s to 1960s.
- The 1830s are excluded from main figures because that decade only contains 1837-1839.
- The 1960s are marked with an asterisk because the available feature output covers 1960-1963.

Data-integrity notes:
- Main decade-level plotted articles: 573,076.
- Minimum articles in a plotted decade: 4,566.
- Maximum articles in a plotted decade: 132,313.
- No rows were randomly sampled for plotting.
- Source data tables for plotted values are exported in this folder.

Panel map:
- Figure 1: Corpus construction stages and retained records by decade.
- Figure 2: Coverage after filtering by decade and year.
- Figure 3: Sentiment components and net sentiment by decade.
- Figure 4: Distress, joy, and distress components by decade.
- Figure 5: Article length, readability, and residual content-type flags.
