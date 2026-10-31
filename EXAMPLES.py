"""Ten worked examples for the Fit.ly churn package.

Every path points at the repository workbook and the repository ``outputs``
directory. A failure prints a traceback and exits non-zero.

    python EXAMPLES.py
"""

from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fitly_advanced_analysis import (
    BusinessMetrics,
    ChurnPredictionModel,
    CohortAnalysis,
    StatisticalAnalysis,
    run_advanced_analysis,
)
from fitly_churn_analysis import (
    ChurnPredictors,
    ChurnVisualizations,
    CustomerSegmentation,
    FitlyDataLoader,
    FitlyEDA,
    RecommendationEngine,
)
from fitly_utils import (
    AnalysisHelper,
    ConfigManager,
    DataExporter,
    MetricsValidator,
    ReportGenerator,
    example_usage,
)

REPO_ROOT = Path(__file__).resolve().parent
DATA_PATH = REPO_ROOT / "data" / "fitly_churn_analysis.xlsx"
OUTPUT_DIR = REPO_ROOT / "outputs"


def _header(number: int, title: str) -> None:
    print(f"\n=== Example {number}: {title} ===")


def example_1() -> pd.DataFrame:
    _header(1, "Load and validate the workbook")
    loader = FitlyDataLoader(DATA_PATH)
    data = loader.load_data()
    valid = loader.validate_data()
    customers = data["Customer Details"]
    print(f"Sheets: {list(data)}")
    print(f"Rows: {len(customers)} | validated={valid}")
    print(customers.head(3).to_string(index=False))
    if not valid:
        raise RuntimeError("validate_data() returned False.")
    if len(customers) != 400 or int(customers["Churned"].sum()) != 114:
        raise RuntimeError("Customer margins do not match the study.")
    return customers


def example_2(customers: pd.DataFrame) -> dict:
    _header(2, "Overall EDA metrics")
    original_dtype = customers["Engagement"].dtype
    eda = FitlyEDA(customers)
    metrics = eda.get_overall_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    if customers["Engagement"].dtype != original_dtype:
        raise RuntimeError("EDA mutated the caller's Engagement dtype.")
    return metrics


def example_3(customers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    _header(3, "Plan, engagement, and support")
    eda = FitlyEDA(customers)
    plan = eda.analyze_churn_by_plan()
    engagement = eda.analyze_churn_by_engagement()
    distribution = eda.analyze_engagement_distribution()
    support = eda.analyze_support_metrics()
    print(plan.to_string(index=False))
    print(engagement.to_string(index=False))
    print(distribution.to_string(index=False))
    print(support)
    return plan, engagement


def example_4(customers: pd.DataFrame) -> pd.DataFrame:
    _header(4, "Non-overlapping segment risk bins")
    predictors = ChurnPredictors(customers)
    segments = predictors.get_high_risk_segments(threshold=0.0)
    # Bins are exclusive. A cumulative ">= threshold" loop would count a 50%
    # segment in every lower bin and overstate the customer total.
    bins = [
        ("Low", 0.00, 0.20),
        ("Moderate", 0.20, 0.35),
        ("High", 0.35, 0.50),
        ("Critical", 0.50, 1.01),
    ]
    accounted = 0
    for name, lower, upper in bins:
        subset = segments[(segments["churn_rate"] >= lower) & (segments["churn_rate"] < upper)]
        customers_in_bin = int(subset["customers"].sum()) if len(subset) else 0
        accounted += customers_in_bin
        print(
            f"  {name} [{lower:.0%}, {upper:.0%}): "
            f"{len(subset)} segments, {customers_in_bin} customers"
        )
    print(f"  Customers across bins: {accounted}")
    if accounted != len(customers):
        raise RuntimeError("Risk bins double-counted or dropped customers.")
    print("Top segments by churn rate:")
    print(segments.head(5).to_string(index=False))
    return segments


def example_5(customers: pd.DataFrame) -> pd.DataFrame:
    _header(5, "LTV bands and at-risk customers")
    segmentation = CustomerSegmentation(customers)
    ltv = segmentation.segment_by_ltv()
    summary = (
        ltv.groupby("LTV Segment", as_index=False)
        .agg(customers=("Customer ID", "count"), avg_ltv=("LTV", "mean"), churn_rate=("Churned", "mean"))
    )
    print(summary.to_string(index=False))
    at_risk = segmentation.identify_at_risk(churn_threshold=0.35)
    print(f"At-risk customers (threshold 0.35): {len(at_risk)}")
    if len(at_risk) != 41:
        raise RuntimeError(f"Expected 41 at-risk customers, found {len(at_risk)}.")
    print(at_risk.head(5).to_string(index=False))
    return at_risk


def example_6(customers: pd.DataFrame) -> None:
    _header(6, "Retention potential and total LTV loss")
    potential = CustomerSegmentation(customers).calculate_retention_potential()
    for key, value in potential.items():
        print(f"  {key}: {value}")
    if potential["segment_size"] == 0:
        raise RuntimeError("Paid low/no-engagement segment was empty.")
    if potential["current_churn_rate"] <= 0:
        raise RuntimeError("Historical churn rate collapsed to 0; active-only mean was used.")

    churned = customers[customers["Churned"] == 1]
    # Listed value of churned contracts over a 24-month horizon. Free prices are 0
    # and contribute nothing; the sum itself is safe on an empty frame too.
    total_ltv_loss = float((churned["Price"] * 24).sum()) if len(churned) else 0.0
    print(f"Total LTV loss (price x 24 months): ${total_ltv_loss:,.2f}")


def example_7(customers: pd.DataFrame) -> dict:
    _header(7, "Chi-square and t-tests")
    stats_runner = StatisticalAnalysis(customers)
    results = {
        "plan_churn": stats_runner.chi_square_plan_churn(),
        "engagement_churn": stats_runner.chi_square_engagement_churn(),
        "support_tickets": stats_runner.ttest_support_tickets_churn(),
        "activities": stats_runner.ttest_activities_churn(),
    }
    for name, result in results.items():
        print(
            f"  {name}: p={result.get('p_value')} "
            f"significant={result.get('significant_at_0_05')} error={result.get('error')}"
        )
    empty = StatisticalAnalysis(customers.iloc[0:0])
    empty_result = empty.ttest_activities_churn()
    if "error" not in empty_result:
        raise RuntimeError("Empty-frame t-test should return an error key.")
    print(f"  empty-frame guard: {empty_result['error']}")
    return results


def example_8(customers: pd.DataFrame) -> pd.DataFrame:
    _header(8, "Random forest on a held-out split")
    model = ChurnPredictionModel(customers)
    model.prepare_data()
    model.train_test_split_data()
    model.train_random_forest()
    importance = model.feature_importance_analysis()
    predictions = model.predict_churn_risk()
    if not predictions.index.equals(model.X_test.index):
        raise RuntimeError("predict_churn_risk index does not match X_test.")
    aligned_ids = customers.loc[model.X_test.index, "Customer ID"].to_numpy()
    if not np.array_equal(predictions["Customer ID"].to_numpy(), aligned_ids):
        raise RuntimeError("Customer IDs were not aligned to X_test.index.")
    print(f"Observed test accuracy: {model.accuracy:.4f}")
    print(predictions.head(5).to_string(index=False))
    return predictions


def example_9(customers: pd.DataFrame) -> None:
    _header(9, "Cohorts, MRR, LTV, and CAC payback")
    cohorts = CohortAnalysis(customers)
    print(cohorts.analyze_plan_cohorts().to_string(index=False))
    print(cohorts.analyze_engagement_cohorts().to_string(index=False))
    business = BusinessMetrics(customers)
    mrr = business.calculate_mrr_impact()
    ltv = business.calculate_ltv_impact()
    payback = business.calculate_cac_payback()
    print(f"Active MRR ${mrr['active_mrr']:.2f} | lost MRR ${mrr['mrr_lost']:.2f}")
    print(f"Gross-margin LTV lost ${ltv['ltv_lost']:.2f}")
    print(f"Payback months: {payback['payback_months_by_plan']}")
    if payback["payback_months_by_plan"].get("Free") is not None:
        raise RuntimeError("Free plan payback should be skipped when price is 0.")


def example_10(customers: pd.DataFrame, metrics: dict, plan: pd.DataFrame, engagement: pd.DataFrame) -> None:
    _header(10, "Recommendations, reports, exports, and ROI guards")
    recommendations = RecommendationEngine(customers).generate_recommendations()
    if len(recommendations) != 4:
        raise RuntimeError("Expected 4 recommendations.")
    required = {"priority", "title", "current_issue", "actions", "owner", "timeline", "expected_impact"}
    for item in recommendations:
        missing = required - set(item)
        if missing:
            raise RuntimeError(f"Recommendation missing keys: {missing}")
        print(f"  {item['priority']}: {item['title']}")

    reporter = ReportGenerator()
    empty_summary = reporter.generate_executive_summary(metrics, plan, engagement, [])
    if "No recommendations" not in empty_summary:
        raise RuntimeError("Empty recommendation list was not handled.")
    full_summary = reporter.generate_executive_summary(metrics, plan, engagement, recommendations)
    print(full_summary.splitlines()[0])

    charts = ChurnVisualizations(customers, OUTPUT_DIR)
    charts.set_style()
    written = [
        charts.plot_churn_by_plan(),
        charts.plot_churn_by_engagement(),
        charts.plot_heatmap_plan_engagement(),
        charts.plot_engagement_distribution(),
    ]
    for path in written:
        if not path.exists():
            raise RuntimeError(f"Missing chart {path}")

    exporter = DataExporter(OUTPUT_DIR)
    exporter.export_report_text(full_summary, "examples_executive_summary.txt")
    exporter.export_dataframe_csv(plan, "examples_churn_by_plan.csv")
    payload = {
        "rows": np.int64(len(customers)),
        "churn_rate": np.float64(metrics["churn_rate"]),
        "validated": np.bool_(True),
        "exported_at": pd.Timestamp(datetime.now(timezone.utc)),
    }
    json_path = exporter.export_dict_json(payload, "examples_metrics.json")
    reloaded = json.loads(json_path.read_text(encoding="utf-8"))
    if reloaded["rows"] != 400 or reloaded["validated"] is not True:
        raise RuntimeError("JSON export did not round-trip numpy types.")

    matrix = AnalysisHelper.create_risk_matrix(customers)
    print("Risk matrix (churn rate):")
    print(matrix.round(3).to_string())

    critical = AnalysisHelper.identify_critical_customers(customers, threshold=0.3)
    print(f"Critical customers: {len(critical)}")
    empty = AnalysisHelper.identify_critical_customers(customers.iloc[0:0], threshold=0.3)
    if not empty.empty or "risk_score" not in empty.columns:
        raise RuntimeError("Empty critical-customer frame is missing columns.")

    zero_activity = customers.head(4).copy()
    zero_activity["Activities"] = 0
    zero_activity["Support Tickets"] = 0
    zero_activity["Engagement"] = "Low"
    zero_activity["Churned"] = 0
    scored = AnalysisHelper.identify_critical_customers(zero_activity)
    if not np.isfinite(scored["risk_score"]).all():
        raise RuntimeError("Risk score divided by zero.")

    roi = AnalysisHelper.calculate_intervention_roi(
        n_customers=int((critical["Plan"] != "Free").sum()) or 1,
        cost_per_customer=25,
        success_rate=0.30,
        avg_monthly_revenue=float(critical.loc[critical["Plan"] != "Free", "Price"].mean() or 0),
    )
    print(f"Intervention ROI: {roi['roi']}")
    zero_cost = AnalysisHelper.calculate_intervention_roi(10, 0, 0.5, 20)
    if zero_cost["roi"] is not None:
        raise RuntimeError("Zero-cost ROI should be undefined, not a divide-by-zero crash.")
    print(f"Zero-cost ROI guard: {zero_cost['note']}")

    config = ConfigManager()
    config.update("churn_threshold", 0.4)
    config_path = config.save(OUTPUT_DIR / "examples_config.json")
    reloaded_config = ConfigManager().load(config_path)
    if reloaded_config["churn_threshold"] != 0.4:
        raise RuntimeError("Config save/load failed.")

    quality = MetricsValidator().data_quality_report(customers)
    print(
        "Quality: "
        f"churn_ok={quality['churn_rate_ok']} "
        f"plan_ok={quality['plan_distribution_ok']} "
        f"engagement_ok={quality['engagement_distribution_ok']}"
    )
    if not all(
        [
            quality["churn_rate_ok"],
            quality["plan_distribution_ok"],
            quality["engagement_distribution_ok"],
        ]
    ):
        raise RuntimeError("Metrics validator rejected the workbook.")

    advanced = run_advanced_analysis(customers)
    technical = reporter.generate_technical_report(
        advanced["stats"], advanced["model_results"], advanced["predictions"]
    )
    exporter.export_report_text(technical, "examples_technical_report.txt")
    example_usage()


def main() -> None:
    try:
        customers = example_1()
        metrics = example_2(customers)
        plan, engagement = example_3(customers)
        example_4(customers)
        example_5(customers)
        example_6(customers)
        example_7(customers)
        example_8(customers)
        example_9(customers)
        example_10(customers, metrics, plan, engagement)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    print("\nAll 10 examples completed successfully.")


if __name__ == "__main__":
    main()
