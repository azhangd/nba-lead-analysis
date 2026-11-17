# How likely is an NBA lead to hold?

This dashboard compares how likely a lead is to hold with what coaches actually did. Change the lead, time, and matchup, then look at actual games to see how the score changed as starters got pulled and subbed back in.

- Dashboard: https://azhangd.github.io/nba-lead-analysis/
- Video: TODO: video link

The dashboard uses Bokeh and runs entirely in the browser. [docs/index.html](docs/index.html) can also be downloaded and opened locally. No installation is needed.

## Notebooks

- [NBA lead analysis](nba_lead_analysis.ipynb) covers the data preparation, dashboard, and exploration. [HTML version](nba_lead_analysis.html).
- [Model appendix](model_appendix.ipynb) explains how the model was built and tested. [HTML version](model_appendix.html).

## Run the code

```sh
uv sync
uv run python dashboard.py
uv run pytest
```

This rebuilds `docs/index.html` and runs the model tests. Without uv, `pip install -r requirements.txt` installs the packages for the dashboard and notebooks, and `pip install pytest` adds the test runner. The included `data/` files are enough to build the dashboard and run the analysis. Rebuilding the data with `prep_data.py` requires the source DuckDB warehouses. The analysis notebook lists the warehouse settings.

## Data

The statistics come from [NBA Stats](https://www.nba.com/stats) through the [NBA-to-DuckDB pipeline](https://github.com/azhangd/nba-analytics-duckdb). The pipeline's code is MIT-licensed. The NBA data itself falls under NBA.com's terms of use, so the exported files only include derived values, with no player names. The model was fitted on 3,690 games from 2022–25 and tested on 1,427 games from 2025–27, through November 15, 2026.

The predictions reflect games where coaches could bring starters back in. The linked charts make it possible to compare those predictions with the substitutions.
