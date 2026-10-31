"""Statistical tests, a regularized random forest, cohorts, and business metrics.

Run from the repository root::

    python fitly_advanced_analysis.py

Held-out accuracy is whatever the model scores on this workbook. It is printed
from ``accuracy_score`` and is not a fixed writeup number.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent
PLAN_ORDER = ["Free", "Basic", "Pro", "Enterprise"]
ENGAGEMENT_ORDER = ["High", "Medium", "Low", "No Engagement"]

# Explicit business order. Do not replace these with sklearn LabelEncoder,
# which assigns integers by sorted label and hides the intended scale.
PLAN_ORDINAL = {"Free": 0, "Basic": 1, "Pro": 2, "Enterprise": 3}
ENGAGEMENT_ORDINAL = {"No Engagement": 0, "Low": 1, "Medium": 2, "High": 3}

LIFETIME_MONTHS = 24
CAC = 50.0
GROSS_MARGIN = 0.80

__all__ = [
    "StatisticalAnalysis",
    "ChurnPredictionModel",
    "CohortAnalysis",
    "BusinessMetrics",
    "run_advanced_analysis",
]


def _empty_test(feature: str, reason: str) -> dict:
    return {
        "feature": feature,
        "test": None,
        "error": reason,
        "statistic": None,
        "p_value": None,
        "significant_at_0_05": False,
    }


class StatisticalAnalysis:
    """Chi-square and Welch t-tests. Empty groups return a structured error."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def _chi_square(self, column: str) -> dict:
        if column not in self.df.columns or "Churned" not in self.df.columns:
            return _empty_test(column, f"Column '{column}' or 'Churned' is missing.")
        if self.df.empty or self.df[column].nunique(dropna=True) < 2:
            return _empty_test(column, "Need at least two groups for a chi-square test.")
        if self.df["Churned"].nunique(dropna=True) < 2:
            return _empty_test(column, "Churned has fewer than two outcomes.")

        table = pd.crosstab(self.df[column], self.df["Churned"])
        if table.empty or (table.to_numpy() < 0).any():
            return _empty_test(column, "Contingency table is empty.")
        try:
            chi2, p_value, dof, _expected = stats.chi2_contingency(table)
        except ValueError as exc:
            return _empty_test(column, str(exc))

        sample = float(table.to_numpy().sum())
        dimensions = min(table.shape) - 1
        cramers_v = float(np.sqrt(chi2 / (sample * dimensions))) if sample and dimensions else None
        return {
            "feature": column,
            "test": "chi-square",
            "chi2": float(chi2),
            "statistic": float(chi2),
            "p_value": float(p_value),
            "dof": int(dof),
            "cramers_v": cramers_v,
            "significant_at_0_05": bool(p_value < 0.05),
            "contingency_table": table.to_dict(),
        }

    def chi_square_plan_churn(self) -> dict:
        return self._chi_square("Plan")

    def chi_square_engagement_churn(self) -> dict:
        return self._chi_square("Engagement")

    def _ttest(self, column: str) -> dict:
        if column not in self.df.columns:
            return _empty_test(column, f"Column '{column}' is missing.")
        churned = self.df.loc[self.df["Churned"] == 1, column].dropna()
        active = self.df.loc[self.df["Churned"] == 0, column].dropna()
        if len(churned) < 2 or len(active) < 2:
            return _empty_test(
                column,
                "Need at least two non-missing observations in both the churned and active groups.",
            )
        t_stat, p_value = stats.ttest_ind(churned, active, equal_var=False)
        return {
            "feature": column,
            "test": "Welch t-test",
            "t_statistic": float(t_stat),
            "statistic": float(t_stat),
            "p_value": float(p_value),
            "mean_churned": float(churned.mean()),
            "mean_active": float(active.mean()),
            "n_churned": int(len(churned)),
            "n_active": int(len(active)),
            "significant_at_0_05": bool(p_value < 0.05),
        }

    def ttest_support_tickets_churn(self) -> dict:
        return self._ttest("Support Tickets")

    def ttest_activities_churn(self) -> dict:
        return self._ttest("Activities")


class ChurnPredictionModel:
    """Random forest with a shallow tree so n=400 does not memorize one column."""

    def __init__(self, df: pd.DataFrame, random_state: int = 42):
        self.df = df.copy()
        self.random_state = random_state
        self.feature_columns: list[str] = []
        self.X: pd.DataFrame | None = None
        self.y: pd.Series | None = None
        self.X_train: pd.DataFrame | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.y_test: pd.Series | None = None
        self.model: RandomForestClassifier | None = None
        self.accuracy: float | None = None
        self.baseline_accuracy: float | None = None
        self.roc_auc: float | None = None
        self.report_text: str | None = None
        self.confusion: np.ndarray | None = None

    def prepare_data(self) -> tuple[pd.DataFrame, pd.Series]:
        frame = self.df
        features = pd.DataFrame(index=frame.index)
        features["Plan"] = frame["Plan"].map(PLAN_ORDINAL)
        features["Price"] = frame["Price"]
        features["Support Tickets"] = frame["Support Tickets"]
        features["Activities"] = frame["Activities"]
        features["Engagement"] = frame["Engagement"].map(ENGAGEMENT_ORDINAL)

        unknown_plan = frame.loc[features["Plan"].isna(), "Plan"].unique().tolist()
        unknown_engagement = frame.loc[features["Engagement"].isna(), "Engagement"].unique().tolist()
        if unknown_plan or unknown_engagement:
            raise ValueError(
                f"Unmapped categories. Plan={unknown_plan or 'ok'}, "
                f"Engagement={unknown_engagement or 'ok'}."
            )

        # Skip resolution time entirely when the column is absent or incomplete
        # rather than failing the fit or imputing a fake signal.
        if "Avg Resolution Time" in frame.columns and frame["Avg Resolution Time"].notna().all():
            features["Avg Resolution Time"] = frame["Avg Resolution Time"]

        target = frame["Churned"].astype(int)
        active_n = int((target == 0).sum())
        churned_n = int((target == 1).sum())
        print(f"Class 0 (active): {active_n}")
        print(f"Class 1 (churned): {churned_n}")
        print(f"Features: {list(features.columns)}")

        self.feature_columns = list(features.columns)
        self.X = features
        self.y = target
        return features, target

    def train_test_split_data(self, test_size: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        if self.X is None or self.y is None:
            self.prepare_data()
        assert self.X is not None and self.y is not None
        stratify = self.y if self.y.nunique() > 1 else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify,
        )
        print(f"Train rows: {len(self.X_train)} | Test rows: {len(self.X_test)}")
        print(f"Test class 0 (active): {(self.y_test == 0).sum()}")
        print(f"Test class 1 (churned): {(self.y_test == 1).sum()}")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_random_forest(self, n_estimators: int = 100) -> RandomForestClassifier:
        if self.X_train is None or self.y_train is None:
            self.train_test_split_data()
        assert self.X_train is not None and self.y_train is not None
        assert self.X_test is not None and self.y_test is not None

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=6,
            min_samples_leaf=5,
            random_state=self.random_state,
            n_jobs=1,
        )
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        accuracy = float(accuracy_score(self.y_test, predictions))
        majority = max(float((self.y_test == 0).mean()), float((self.y_test == 1).mean()))
        self.model = model
        self.accuracy = accuracy
        self.baseline_accuracy = majority
        self.report_text = classification_report(
            self.y_test, predictions, digits=3, zero_division=0
        )
        self.confusion = confusion_matrix(self.y_test, predictions, labels=[0, 1])
        try:
            probabilities = self._positive_probabilities(self.X_test)
            self.roc_auc = float(roc_auc_score(self.y_test, probabilities))
        except ValueError:
            self.roc_auc = None

        print(f"Majority-class baseline accuracy: {majority:.4f} ({majority * 100:.2f}%)")
        print(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        if self.roc_auc is not None:
            print(f"Test ROC AUC: {self.roc_auc:.4f}")
        print("Classification report:")
        print(self.report_text)
        return model

    def _positive_probabilities(self, features: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Train the random forest before predicting.")
        probabilities = self.model.predict_proba(features)
        classes = list(self.model.classes_)
        if 1 not in classes:
            return np.zeros(len(features), dtype=float)
        return probabilities[:, classes.index(1)]

    def feature_importance_analysis(self) -> pd.DataFrame:
        if self.model is None:
            self.train_random_forest()
        assert self.model is not None
        table = pd.DataFrame(
            {
                "feature": self.feature_columns,
                "importance": self.model.feature_importances_.astype(float),
            }
        ).sort_values("importance", ascending=False)
        table = table.reset_index(drop=True)
        print("Feature importance:")
        print(table.to_string(index=False))
        return table

    def predict_churn_risk(self) -> pd.DataFrame:
        """Score the held-out rows. The result index matches ``X_test.index``."""
        if self.model is None:
            self.train_random_forest()
        assert self.model is not None and self.X_test is not None and self.y_test is not None

        probabilities = self._positive_probabilities(self.X_test)
        predicted = self.model.predict(self.X_test).astype(int)
        scored = pd.DataFrame(
            {
                "Customer ID": self.df.loc[self.X_test.index, "Customer ID"].to_numpy(),
                "churn_probability": probabilities,
                "predicted_churn": predicted,
                "actual_churn": self.y_test.loc[self.X_test.index].to_numpy(),
            },
            index=self.X_test.index,
        )
        if not scored.index.equals(self.X_test.index):
            raise RuntimeError("Prediction index is not aligned to X_test.index.")
        return scored


class CohortAnalysis:
    """Plan and engagement cohorts with rates and revenue."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def _cohort(self, column: str, order: list[str]) -> pd.DataFrame:
        rows = []
        present = list(order) + [value for value in self.df[column].dropna().unique() if value not in order]
        for key in present:
            group = self.df[self.df[column] == key]
            if group.empty:
                continue
            resolution = (
                float(group["Avg Resolution Time"].mean())
                if "Avg Resolution Time" in group.columns
                else None
            )
            rows.append(
                {
                    column: key,
                    "customers": int(len(group)),
                    "churned": int(group["Churned"].sum()),
                    "active": int((group["Churned"] == 0).sum()),
                    "churn_rate": float(group["Churned"].mean()),
                    "avg_price": float(group["Price"].mean()),
                    "avg_support_tickets": float(group["Support Tickets"].mean()),
                    "avg_activities": float(group["Activities"].mean()),
                    "avg_resolution_time": resolution,
                    "active_mrr": float(group.loc[group["Churned"] == 0, "Price"].sum()),
                    "churned_mrr": float(group.loc[group["Churned"] == 1, "Price"].sum()),
                }
            )
        return pd.DataFrame(rows)

    def analyze_plan_cohorts(self) -> pd.DataFrame:
        return self._cohort("Plan", PLAN_ORDER)

    def analyze_engagement_cohorts(self) -> pd.DataFrame:
        return self._cohort("Engagement", ENGAGEMENT_ORDER)


class BusinessMetrics:
    """MRR, gross-margin LTV, and CAC payback. Free (price 0) has no payback."""

    LIFETIME_MONTHS = LIFETIME_MONTHS
    CAC = CAC
    GROSS_MARGIN = GROSS_MARGIN

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def calculate_mrr_impact(self) -> dict:
        frame = self.df
        active = frame[frame["Churned"] == 0]
        churned = frame[frame["Churned"] == 1]
        def _money(series: pd.Series) -> dict[str, float]:
            return {str(plan): round(float(amount), 2) for plan, amount in series.items()}

        return {
            "active_mrr": round(float(active["Price"].sum()), 2),
            "churned_mrr": round(float(churned["Price"].sum()), 2),
            "mrr_lost": round(float(churned["Price"].sum()), 2),
            "potential_mrr": round(float(frame["Price"].sum()), 2),
            "active_mrr_by_plan": _money(active.groupby("Plan")["Price"].sum()),
            "churned_mrr_by_plan": _money(churned.groupby("Plan")["Price"].sum()),
        }

    def calculate_ltv_impact(self) -> dict:
        months = self.LIFETIME_MONTHS
        margin = self.GROSS_MARGIN
        ltv = self.df["Price"] * months * margin
        churned_mask = self.df["Churned"] == 1
        active_mask = ~churned_mask
        return {
            "lifetime_months": months,
            "gross_margin": margin,
            "ltv_lost": round(float(ltv.loc[churned_mask].sum()), 2) if churned_mask.any() else 0.0,
            "avg_ltv_churned": round(float(ltv.loc[churned_mask].mean()), 2) if churned_mask.any() else 0.0,
            "avg_ltv_active": round(float(ltv.loc[active_mask].mean()), 2) if active_mask.any() else 0.0,
            "total_active_ltv": round(float(ltv.loc[active_mask].sum()), 2) if active_mask.any() else 0.0,
        }

    def calculate_cac_payback(self) -> dict:
        """Months to repay CAC from gross-margin revenue. Price 0 is skipped."""
        payback: dict[str, float | None] = {}
        for plan in PLAN_ORDER:
            group = self.df[self.df["Plan"] == plan]
            if group.empty:
                continue
            price = float(group["Price"].mean())
            monthly_gross = price * self.GROSS_MARGIN
            if price <= 0 or monthly_gross <= 0:
                payback[plan] = None
                continue
            payback[plan] = float(self.CAC / monthly_gross)
        return {
            "cac": self.CAC,
            "gross_margin": self.GROSS_MARGIN,
            "lifetime_months": self.LIFETIME_MONTHS,
            "payback_months_by_plan": payback,
        }


def run_advanced_analysis(df: pd.DataFrame) -> dict:
    """Run stats, the forest, cohorts, and business metrics for one customer table."""
    stats_runner = StatisticalAnalysis(df)
    stats_results = {
        "plan_churn": stats_runner.chi_square_plan_churn(),
        "engagement_churn": stats_runner.chi_square_engagement_churn(),
        "support_tickets": stats_runner.ttest_support_tickets_churn(),
        "activities": stats_runner.ttest_activities_churn(),
    }
    for name, result in stats_results.items():
        statistic = result.get("statistic")
        p_value = result.get("p_value")
        if result.get("error"):
            print(f"{name}: {result['error']}")
        else:
            print(f"{name}: statistic={statistic:.4f} p={p_value:.4g} significant={result['significant_at_0_05']}")

    model = ChurnPredictionModel(df)
    model.prepare_data()
    model.train_test_split_data()
    model.train_random_forest()
    importance = model.feature_importance_analysis()
    predictions = model.predict_churn_risk()

    cohorts = CohortAnalysis(df)
    plan_cohorts = cohorts.analyze_plan_cohorts()
    engagement_cohorts = cohorts.analyze_engagement_cohorts()
    print("\nPlan cohorts:")
    print(plan_cohorts.to_string(index=False))
    print("\nEngagement cohorts:")
    print(engagement_cohorts.to_string(index=False))

    business = BusinessMetrics(df)
    mrr = business.calculate_mrr_impact()
    ltv = business.calculate_ltv_impact()
    payback = business.calculate_cac_payback()
    print(f"\nMRR lost to churn: ${mrr['mrr_lost']:.2f}")
    print(f"Gross-margin LTV lost: ${ltv['ltv_lost']:.2f}")
    print(f"CAC payback months by plan: {payback['payback_months_by_plan']}")

    model_results = {
        "accuracy": model.accuracy,
        "baseline_accuracy": model.baseline_accuracy,
        "roc_auc": model.roc_auc,
        "classification_report": model.report_text,
        "confusion_matrix": None if model.confusion is None else model.confusion.tolist(),
        "feature_importance": importance,
        "max_depth": 6,
        "min_samples_leaf": 5,
        "n_estimators": 100,
    }
    return {
        "stats": stats_results,
        "model_results": model_results,
        "predictions": predictions,
        "plan_cohorts": plan_cohorts,
        "engagement_cohorts": engagement_cohorts,
        "mrr": mrr,
        "ltv": ltv,
        "cac_payback": payback,
    }


def _load_customers() -> pd.DataFrame:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from fitly_churn_analysis import FitlyDataLoader

    loaded = FitlyDataLoader().load_data()
    return loaded["Customer Details"]


if __name__ == "__main__":
    try:
        run_advanced_analysis(_load_customers())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
