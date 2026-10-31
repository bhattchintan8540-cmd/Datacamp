# Package summary

Modules live at the repository root so `python fitly_churn_analysis.py` and `from fitly_churn_analysis import FitlyDataLoader` both work when the root is the working directory.

## `generate_data.py`

`generate_dataset(path=None, seed=42)` writes `data/fitly_churn_analysis.xlsx` with sheets Customer Details, Churn by Plan, Churn by Engagement, Data Validation, Summary Statistics, and Support Analysis. `generate_data` is an alias.

## `fitly_churn_analysis.py`

| Class | Role |
| --- | --- |
| `FitlyDataLoader` | `load_data()` returns a dict of dataframes. `validate_data()` checks the study margins. Default workbook path is next to the module. Missing files are generated. Other sheets may be absent; Customer Details is enough. |
| `FitlyEDA` | `get_overall_metrics`, `analyze_churn_by_plan`, `analyze_churn_by_engagement`, `analyze_engagement_distribution`, `analyze_support_metrics`. Copies the frame so the caller's Engagement column is not recast. |
| `ChurnPredictors` | Plan and engagement impact versus the overall rate, feature correlations, and plan × engagement segments. |
| `CustomerSegmentation` | `segment_by_ltv`, `identify_at_risk(churn_threshold=0.35)`, `calculate_retention_potential`. Retention uses the historical churn rate of paid Low/No Engagement customers, including those who already churned, then applies the gap to the still-active count only. |
| `ChurnVisualizations` | `Agg` backend, `set_style`, four chart methods. Saves PNGs under `outputs/` and closes figures. No `plt.show()`. |
| `RecommendationEngine` | `generate_recommendations()` returns four dicts: `priority`, `title`, `current_issue`, `actions`, `owner`, `timeline`, `expected_impact`. |

`main()` runs the core pipeline and returns a results dict.

## `fitly_advanced_analysis.py`

| Class | Role |
| --- | --- |
| `StatisticalAnalysis` | Chi-square for plan and engagement, Welch t-tests for tickets and activities. Empty groups return an error dict. |
| `ChurnPredictionModel` | Ordinal maps (Free=0 … Enterprise=3, No Engagement=0 … High=3). Skips Avg Resolution Time when that column is missing. `max_depth=6`, `min_samples_leaf=5`. Prints real class counts and test accuracy. `predict_churn_risk` keeps `X_test.index`. |
| `CohortAnalysis` | `analyze_plan_cohorts`, `analyze_engagement_cohorts`. |
| `BusinessMetrics` | MRR, gross-margin LTV, CAC payback. Lifetime 24 months, CAC 50, gross margin 0.80. Payback is `None` when price is 0. |

`run_advanced_analysis(df)` returns stats, model results, predictions, cohorts, and business metrics. `__main__` loads the workbook through `FitlyDataLoader`.

## `fitly_utils.py`

| Class | Role |
| --- | --- |
| `DataExporter` | Excel, CSV, JSON, and text under the repo `outputs/` directory. JSON accepts numpy scalars, booleans, timestamps, and (optionally) dataframes. |
| `ReportGenerator` | Executive summary tolerates an empty recommendation list and full recommendation dicts. Technical report summarizes tests, the model, and predictions. |
| `AnalysisHelper` | Risk matrix, critical customers (Low/No Engagement and still active). A zero activity or ticket maximum does not raise. An empty filter returns an empty frame with `risk_score`. `calculate_intervention_roi` leaves ROI undefined when cost is 0. |
| `ConfigManager` | `DEFAULT_CONFIG`, `get`, `update`, `save`, `load`. |
| `MetricsValidator` | Churn rate, plan mix, engagement mix, and a quality report. |

`example_usage()` reads the repo workbook and writes `outputs/example_usage_metrics.json`.

## `EXAMPLES.py`

Ten examples covering load, EDA, non-overlapping risk bins, segmentation, LTV loss, tests, the forest, business metrics, and exports. Failures print a traceback and exit non-zero.
