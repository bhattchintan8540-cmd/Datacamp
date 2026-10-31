"""Export, reporting, configuration, and validation helpers for the Fit.ly analysis.

Run from the repository root::

    python fitly_utils.py
"""

from __future__ import annotations

import json
import sys
import traceback
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "fitly_churn_analysis.xlsx"

PLAN_ORDER = ["Free", "Basic", "Pro", "Enterprise"]
ENGAGEMENT_ORDER = ["High", "Medium", "Low", "No Engagement"]
CRITICAL_COLUMNS = [
    "Customer ID",
    "Plan",
    "Price",
    "Churned",
    "Support Tickets",
    "Activities",
    "Engagement",
    "Avg Resolution Time",
    "risk_score",
]

__all__ = [
    "DataExporter",
    "ReportGenerator",
    "AnalysisHelper",
    "ConfigManager",
    "MetricsValidator",
    "NumpyJSONEncoder",
    "example_usage",
]


class NumpyJSONEncoder(json.JSONEncoder):
    """Serialize numpy and pandas scalars that the stdlib encoder rejects."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            if np.isnan(obj):
                return None
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


class DataExporter:
    """Write tables and reports. The default directory is the repo ``outputs`` folder."""

    def __init__(self, output_dir: str | Path = "./outputs"):
        candidate = Path(output_dir)
        if candidate == Path("./outputs") or candidate == Path("outputs"):
            resolved = DEFAULT_OUTPUT_DIR
        elif not candidate.is_absolute():
            resolved = REPO_ROOT / candidate
        else:
            resolved = candidate
        self.output_dir = resolved
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, filename: str | Path) -> Path:
        filename = Path(filename)
        if filename.is_absolute():
            filename.parent.mkdir(parents=True, exist_ok=True)
            return filename
        return self.output_dir / filename

    def export_dataframe_excel(self, df: pd.DataFrame, filename: str, sheet_name: str = "Sheet1") -> Path:
        path = self._path(filename)
        df.to_excel(path, index=False, sheet_name=sheet_name[:31], engine="openpyxl")
        print(f"Wrote {path}")
        return path

    def export_dataframe_csv(self, df: pd.DataFrame, filename: str) -> Path:
        path = self._path(filename)
        df.to_csv(path, index=False)
        print(f"Wrote {path}")
        return path

    def export_dict_json(self, payload: dict, filename: str) -> Path:
        path = self._path(filename)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, cls=NumpyJSONEncoder)
            handle.write("\n")
        print(f"Wrote {path}")
        return path

    def export_report_text(self, text: str, filename: str) -> Path:
        path = self._path(filename)
        path.write_text(text, encoding="utf-8")
        print(f"Wrote {path}")
        return path


def _as_text(obj) -> str:
    if obj is None:
        return "(none)"
    if isinstance(obj, pd.DataFrame):
        if obj.empty:
            return "(empty table)"
        return obj.to_string(index=False)
    if isinstance(obj, pd.Series):
        return obj.to_string()
    if isinstance(obj, dict):
        lines = []
        for key, value in obj.items():
            if isinstance(value, (dict, list, pd.DataFrame, pd.Series)):
                lines.append(f"{key}:")
                lines.append(_as_text(value))
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)
    if isinstance(obj, (list, tuple)):
        if not obj:
            return "(empty)"
        return "\n".join(_as_text(item) for item in obj)
    return str(obj)


class ReportGenerator:
    """Plain-text summaries. An empty recommendation list is valid."""

    def generate_executive_summary(
        self,
        overall_metrics,
        plan_analysis,
        engagement_analysis,
        recommendations,
    ) -> str:
        recommendations = list(recommendations or [])
        lines = [
            "Fit.ly churn analysis — executive summary",
            "=" * 44,
            "",
            "Overall",
            _as_text(overall_metrics),
            "",
            "Churn by plan",
            _as_text(plan_analysis),
            "",
            "Churn by engagement",
            _as_text(engagement_analysis),
            "",
            "Recommendations",
        ]
        if not recommendations:
            lines.append("No recommendations supplied.")
        else:
            for item in recommendations:
                if isinstance(item, dict):
                    priority = item.get("priority", "")
                    title = item.get("title", "Untitled")
                    lines.append(f"{priority}: {title}".strip())
                    if item.get("current_issue"):
                        lines.append(f"  Issue: {item['current_issue']}")
                    actions = item.get("actions", [])
                    if isinstance(actions, (list, tuple)):
                        for action in actions:
                            lines.append(f"  - {action}")
                    elif actions:
                        lines.append(f"  - {actions}")
                    if item.get("owner") or item.get("timeline"):
                        lines.append(
                            f"  Owner: {item.get('owner', 'unassigned')} | "
                            f"Timeline: {item.get('timeline', 'unspecified')}"
                        )
                    if item.get("expected_impact"):
                        lines.append(f"  Impact: {item['expected_impact']}")
                else:
                    lines.append(str(item))
                lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    def generate_technical_report(self, stats_results, model_results, predictions) -> str:
        lines = [
            "Fit.ly churn analysis — technical report",
            "=" * 42,
            "",
            "Statistical tests",
            _as_text(stats_results),
            "",
            "Model",
            _as_text(model_results),
            "",
            "Held-out predictions",
        ]
        if isinstance(predictions, pd.DataFrame):
            lines.append(f"Rows: {len(predictions)}")
            if "churn_probability" in predictions.columns and len(predictions):
                lines.append(
                    f"Mean predicted churn probability: {float(predictions['churn_probability'].mean()):.4f}"
                )
            lines.append(predictions.head(10).to_string(index=False))
        else:
            lines.append(_as_text(predictions))
        return "\n".join(lines).rstrip() + "\n"


class AnalysisHelper:
    """Risk matrix, critical-customer scoring, and intervention ROI."""

    @staticmethod
    def create_risk_matrix(df: pd.DataFrame) -> pd.DataFrame:
        matrix = pd.crosstab(
            df["Plan"],
            df["Engagement"],
            values=df["Churned"],
            aggfunc="mean",
        )
        return matrix.reindex(index=PLAN_ORDER, columns=ENGAGEMENT_ORDER)

    @staticmethod
    def identify_critical_customers(df: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
        """Active Low and No Engagement customers, scored only on that filtered frame.

        ``threshold`` is part of the public signature and is not used to change
        the engagement rule. An empty filter returns an empty frame with the
        expected columns. Activity or ticket maxima of 0 do not divide by zero.
        """
        del threshold  # retained for signature compatibility
        available = [column for column in CRITICAL_COLUMNS if column != "risk_score" and column in df.columns]
        filtered = df[
            df["Engagement"].isin(["Low", "No Engagement"]) & (df["Churned"] == 0)
        ].copy()
        if filtered.empty:
            return pd.DataFrame(columns=available + ["risk_score"])

        def _safe_max(series: pd.Series) -> float:
            observed = series.max() if len(series) else 0
            if pd.isna(observed) or float(observed) == 0.0:
                return 1.0
            return float(observed)

        activity_max = _safe_max(filtered["Activities"])
        ticket_max = _safe_max(filtered["Support Tickets"])
        filtered["risk_score"] = (
            0.6 * (1.0 - filtered["Activities"] / activity_max)
            + 0.4 * (filtered["Support Tickets"] / ticket_max)
        )
        ordered = [column for column in CRITICAL_COLUMNS if column in filtered.columns]
        extra = [column for column in filtered.columns if column not in ordered]
        return filtered[ordered + extra].sort_values("risk_score", ascending=False).reset_index(drop=True)

    @staticmethod
    def calculate_intervention_roi(
        n_customers: float,
        cost_per_customer: float,
        success_rate: float,
        avg_monthly_revenue: float,
        months: int = 24,
        gross_margin: float = 0.80,
    ) -> dict:
        """ROI of a save program. A zero program cost leaves ROI undefined."""
        customers = float(n_customers)
        cost = customers * float(cost_per_customer)
        benefit = customers * float(success_rate) * float(avg_monthly_revenue) * float(months) * float(gross_margin)
        if cost == 0:
            return {
                "cost": 0.0,
                "benefit": float(benefit),
                "net": float(benefit),
                "roi": None,
                "note": "Intervention cost is zero; ROI is undefined.",
            }
        return {
            "cost": float(cost),
            "benefit": float(benefit),
            "net": float(benefit - cost),
            "roi": float((benefit - cost) / cost),
            "note": "",
        }


class ConfigManager:
    """Small JSON config with the analysis assumptions."""

    DEFAULT_CONFIG = {
        "data_path": "data/fitly_churn_analysis.xlsx",
        "output_dir": "outputs",
        "churn_threshold": 0.35,
        "target_churn_rate": 0.20,
        "lifetime_months": 24,
        "cac": 50,
        "gross_margin": 0.80,
        "random_state": 42,
        "n_estimators": 100,
        "max_depth": 6,
        "min_samples_leaf": 5,
        "test_size": 0.25,
    }

    def __init__(self, config: dict | None = None):
        self.config = dict(self.DEFAULT_CONFIG)
        if config:
            self.config.update(config)

    def get(self, key: str, default=None):
        return self.config.get(key, default)

    def update(self, key, value=None):
        if isinstance(key, dict) and value is None:
            self.config.update(key)
        else:
            self.config[key] = value
        return self.config

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        if not destination.is_absolute():
            destination = REPO_ROOT / destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(self.config, handle, indent=2, cls=NumpyJSONEncoder)
            handle.write("\n")
        return destination

    def load(self, path: str | Path) -> dict:
        source = Path(path)
        if not source.is_absolute():
            source = REPO_ROOT / source
        with source.open(encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise ValueError(f"Config file {source} did not contain an object.")
        self.config.update(loaded)
        return self.config


class MetricsValidator:
    """Checks against the study margins stored in the generated workbook."""

    EXPECTED_ROWS = 400
    EXPECTED_CHURNED = 114
    EXPECTED_PLAN = {"Free": 105, "Basic": 118, "Pro": 85, "Enterprise": 92}
    EXPECTED_ENGAGEMENT = {"High": 105, "Medium": 231, "Low": 56, "No Engagement": 8}

    def validate_churn_rate(self, df: pd.DataFrame, expected_rate: float | None = None, tolerance: float = 1e-9) -> bool:
        if len(df) == 0:
            return False
        expected = self.EXPECTED_CHURNED / self.EXPECTED_ROWS if expected_rate is None else expected_rate
        return abs(float(df["Churned"].mean()) - expected) <= tolerance

    def validate_plan_distribution(self, df: pd.DataFrame) -> bool:
        counts = df["Plan"].value_counts().to_dict()
        return all(int(counts.get(name, 0)) == expected for name, expected in self.EXPECTED_PLAN.items())

    def validate_engagement_distribution(self, df: pd.DataFrame) -> bool:
        counts = df["Engagement"].value_counts().to_dict()
        return all(int(counts.get(name, 0)) == expected for name, expected in self.EXPECTED_ENGAGEMENT.items())

    def data_quality_report(self, df: pd.DataFrame) -> dict:
        missing = {column: int(value) for column, value in df.isna().sum().items()}
        return {
            "rows": int(len(df)),
            "duplicate_ids": int(df["Customer ID"].duplicated().sum()) if "Customer ID" in df.columns else None,
            "missing_by_column": missing,
            "churn_rate_ok": self.validate_churn_rate(df),
            "plan_distribution_ok": self.validate_plan_distribution(df),
            "engagement_distribution_ok": self.validate_engagement_distribution(df),
            "churned_values": sorted(int(value) for value in df["Churned"].dropna().unique()),
        }


def example_usage() -> None:
    """Load the repo workbook, summarize it, and write a JSON metrics file."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from fitly_churn_analysis import FitlyDataLoader, FitlyEDA

    loader = FitlyDataLoader(DEFAULT_DATA_PATH)
    data = loader.load_data()
    valid = loader.validate_data()
    customers = data["Customer Details"]
    metrics = FitlyEDA(customers).get_overall_metrics()
    exporter = DataExporter(DEFAULT_OUTPUT_DIR)
    payload = {
        "source": "data/fitly_churn_analysis.xlsx",
        "validated": bool(valid),
        "total_customers": np.int64(metrics["total_customers"]),
        "churned_customers": np.int64(metrics["churned_customers"]),
        "churn_rate": np.float64(metrics["churn_rate"]),
        "at_risk_customers": np.int64(metrics["at_risk_customers"]),
        "generated_at": datetime.now().astimezone().isoformat(),
    }
    exporter.export_dict_json(payload, "example_usage_metrics.json")
    print(
        f"example_usage: churn rate {metrics['churn_rate']:.2%}, "
        f"at-risk {metrics['at_risk_customers']}, validated={valid}"
    )


if __name__ == "__main__":
    try:
        example_usage()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
