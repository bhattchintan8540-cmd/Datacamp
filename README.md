# Fit.ly churn analysis

Python package that loads a 400-customer Fit.ly workbook, summarizes churn by plan and engagement, tests those differences, fits a shallow random forest, and writes charts plus recommendations.

The customer file in this repo is a reproducible stand-in. `generate_data.py` (seed 42) builds `data/fitly_churn_analysis.xlsx` so the study margins match exactly: 400 customers (`C10000`–`C10399`), 114 churned, the plan and engagement mixes below, and 41 active customers in Low or No Engagement. Prices, tickets, activities, and resolution times vary around the study means. Every path in this package is relative to the repository.

Metrics are computed from that workbook when you run the scripts. The random forest prints its held-out accuracy. On this environment that score is **85.00%** (majority-class baseline **71.00%**, ROC AUC **0.91**) with `max_depth=6` and `min_samples_leaf=5`. A figure of 92.5% or 98.12% from an older writeup is not reproduced here and is not hardcoded.

## Install

From the repository root, with Python 3.12:

```bash
pip install -r requirements.txt
```

## Run

```bash
python generate_data.py
python fitly_churn_analysis.py
python fitly_advanced_analysis.py
python fitly_utils.py
python EXAMPLES.py
```

`FitlyDataLoader` calls `generate_data.py` if `data/fitly_churn_analysis.xlsx` is missing, so a fresh clone still runs. With the repository root as the working directory (or on `PYTHONPATH`):

```python
from fitly_churn_analysis import FitlyDataLoader
```

Charts and text exports land in `outputs/`. Matplotlib uses the non-interactive `Agg` backend. Nothing calls `plt.show()` or `input()`.

## What you should see

| Check | Value |
| --- | --- |
| Customers | 400 |
| Churned | 114 (28.5%) |
| Plans | Free 105, Basic 118, Pro 85, Enterprise 92 |
| Churned by plan | Free 43, Basic 28, Pro 19, Enterprise 24 |
| Engagement | High 105, Medium 231, Low 56, No Engagement 8 |
| Churned by engagement | High 17, Medium 74, Low 20, No Engagement 3 |
| At-risk (active, Low or No Engagement) | 41 |
| Active price sum | $13,177.00 |
| Churned price sum | $4,408.48 (within $1 of $4,409; plan means fix the total) |

Plan means: Free $0, Basic $19.87, Pro $53.74, Enterprise $116.01. Support tickets average 2.30, activities 1.11 (lower among churned customers), resolution time about 10.4 hours. Resolution time is only mildly related to churn (correlation about 0.12) and is not a label.

Cells with one or two customers (for example Pro or Enterprise with No Engagement) can show a 0% or 50% rate. Those rates are real counts, not stable segment estimates.

Business assumptions used for LTV and payback: 24-month lifetime, CAC $50, gross margin 0.80. Free payback is skipped because the price is 0.

## Open in DataCamp DataLab

Publishing requires your DataCamp account. This environment has no DataCamp credentials and does not publish the project for you.

1. Create a DataLab project (blank Python, or import this folder).
2. Upload the repo files, including `data/fitly_churn_analysis.xlsx`, `fitly_churn_analysis.py`, `fitly_advanced_analysis.py`, `fitly_utils.py`, `EXAMPLES.py`, `generate_data.py`, `requirements.txt`, and `notebooks/fitly_churn_analysis.ipynb`.
3. In the DataLab terminal: `pip install -r requirements.txt`
4. Open `notebooks/fitly_churn_analysis.ipynb` and Run All, or run `python fitly_churn_analysis.py`.
5. After the notebook runs, publish it with DataLab's Publish / Share control.

The notebook sets the `Agg` backend, puts the repo root on `sys.path`, generates the workbook if it is missing, and runs from top to bottom without widgets or `getpass`.

## Layout

```
fitly_churn_analysis.py      loader, EDA, segmentation, charts, recommendations
fitly_advanced_analysis.py   tests, random forest, cohorts, MRR / LTV / CAC
fitly_utils.py               export, reports, config, validators
EXAMPLES.py                  ten runnable examples
generate_data.py             seed-42 workbook
data/fitly_churn_analysis.xlsx
notebooks/fitly_churn_analysis.ipynb
outputs/                     charts and exports
```

See `QUICKSTART.md`, `PACKAGE_SUMMARY.md`, and `DATA_DICTIONARY.md` for the short path, the class list, and column definitions.
