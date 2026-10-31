"""Fit.ly churn analysis: loading, EDA, segmentation, charts, and recommendations.

Run from the repository root::

    python fitly_churn_analysis.py

Importing this module forces the non-interactive Matplotlib backend so batch
runs and DataLab publishes do not block on ``plt.show()``.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "fitly_churn_analysis.xlsx"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"

PLAN_ORDER = ["Free", "Basic", "Pro", "Enterprise"]
ENGAGEMENT_ORDER = ["High", "Medium", "Low", "No Engagement"]
CUSTOMER_SHEET = "Customer Details"
REQUIRED_COLUMNS = [
    "Customer ID",
    "Plan",
    "Price",
    "Churned",
    "Support Tickets",
    "Activities",
    "Engagement",
    "Avg Resolution Time",
]

__all__ = [
    "FitlyDataLoader",
    "FitlyEDA",
    "ChurnPredictors",
    "CustomerSegmentation",
    "ChurnVisualizations",
    "RecommendationEngine",
    "main",
]


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _ordered_categories(series: pd.Series, order: list[str]) -> pd.Series:
    """Return a categorical copy. Never write this back onto a caller's frame."""
    return pd.Series(pd.Categorical(series.to_numpy(), categories=order, ordered=True), index=series.index)


class FitlyDataLoader:
    """Load the Fit.ly workbook. Missing files are generated from ``generate_data``."""

    def __init__(self, excel_path: str | Path | None = None):
        if excel_path is None:
            resolved = DEFAULT_DATA_PATH
        else:
            candidate = Path(excel_path)
            resolved = candidate if candidate.is_absolute() else (REPO_ROOT / candidate)
        self.excel_path = resolved
        self.data: dict[str, pd.DataFrame] | None = None

    def _ensure_workbook(self) -> None:
        if self.excel_path.exists():
            return
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from generate_data import generate_dataset

        self.excel_path.parent.mkdir(parents=True, exist_ok=True)
        generate_dataset(self.excel_path)

    def load_data(self) -> dict[str, pd.DataFrame]:
        """Return a dict of sheet name to dataframe.

        ``Customer Details`` is required. Any other sheet that is absent is
        skipped; callers can rebuild those views from the customer table.
        """
        self._ensure_workbook()
        workbook = pd.ExcelFile(self.excel_path, engine="openpyxl")
        if CUSTOMER_SHEET not in workbook.sheet_names:
            raise ValueError(
                f"{self.excel_path} is missing the '{CUSTOMER_SHEET}' sheet. "
                f"Found: {workbook.sheet_names}"
            )

        loaded: dict[str, pd.DataFrame] = {}
        for sheet in workbook.sheet_names:
            loaded[sheet] = pd.read_excel(workbook, sheet_name=sheet, engine="openpyxl")

        customers = loaded[CUSTOMER_SHEET].copy()
        missing = [column for column in REQUIRED_COLUMNS if column not in customers.columns]
        if missing:
            raise ValueError(f"Customer Details is missing columns: {missing}")

        customers["Customer ID"] = customers["Customer ID"].astype(str)
        customers["Plan"] = customers["Plan"].astype(str)
        customers["Engagement"] = customers["Engagement"].astype(str)
        customers["Churned"] = customers["Churned"].astype(int)
        customers["Support Tickets"] = customers["Support Tickets"].astype(int)
        customers["Activities"] = customers["Activities"].astype(int)
        customers["Price"] = customers["Price"].astype(float)
        if "Avg Resolution Time" in customers.columns:
            customers["Avg Resolution Time"] = customers["Avg Resolution Time"].astype(float)
        loaded[CUSTOMER_SHEET] = customers
        self.data = loaded
        return loaded

    def validate_data(self) -> bool:
        """Return True when the customer table matches the study structure."""
        if self.data is None:
            self.load_data()
        assert self.data is not None
        frame = self.data[CUSTOMER_SHEET]
        problems: list[str] = []

        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            problems.append(f"missing columns {missing}")
        if frame[REQUIRED_COLUMNS].isna().any().any():
            problems.append("missing values in required columns")
        if len(frame) != 400:
            problems.append(f"row count {len(frame)} != 400")
        if not frame["Customer ID"].is_unique:
            problems.append("duplicate customer IDs")
        if int(frame["Churned"].sum()) != 114:
            problems.append(f"churned count {int(frame['Churned'].sum())} != 114")
        if set(frame["Churned"].unique()) - {0, 1}:
            problems.append("Churned is not binary 0/1")

        plan_counts = frame["Plan"].value_counts().to_dict()
        expected_plans = {"Free": 105, "Basic": 118, "Pro": 85, "Enterprise": 92}
        if plan_counts != expected_plans:
            problems.append(f"plan counts {plan_counts}")

        engagement_counts = frame["Engagement"].value_counts().to_dict()
        expected_engagement = {"High": 105, "Medium": 231, "Low": 56, "No Engagement": 8}
        if engagement_counts != expected_engagement:
            problems.append(f"engagement counts {engagement_counts}")

        at_risk = frame[
            frame["Engagement"].isin(["Low", "No Engagement"]) & (frame["Churned"] == 0)
        ]
        if len(at_risk) != 41:
            problems.append(f"at-risk count {len(at_risk)} != 41")

        if problems:
            print("Data validation failed:")
            for problem in problems:
                print(f"  - {problem}")
            return False

        print(f"Data validation passed ({len(frame)} customers, {self.excel_path.name}).")
        return True


class FitlyEDA:
    """Exploratory summaries. Works on a copy so the caller's frame is unchanged."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def get_overall_metrics(self) -> dict:
        frame = self.df
        total = int(len(frame))
        churned = int(frame["Churned"].sum())
        active = total - churned
        active_mrr = round(float(frame.loc[frame["Churned"] == 0, "Price"].sum()), 2)
        churned_mrr = round(float(frame.loc[frame["Churned"] == 1, "Price"].sum()), 2)
        at_risk = int(
            (
                frame["Engagement"].isin(["Low", "No Engagement"]) & (frame["Churned"] == 0)
            ).sum()
        )
        churn_rate = (churned / total) if total else 0.0
        return {
            "total_customers": total,
            "churned_customers": churned,
            "active_customers": active,
            "churn_rate": churn_rate,
            "retention_rate": 1.0 - churn_rate,
            "active_mrr": active_mrr,
            "churned_mrr": churned_mrr,
            "total_listed_mrr": active_mrr + churned_mrr,
            "avg_price_active": float(frame.loc[frame["Churned"] == 0, "Price"].mean()) if active else 0.0,
            "avg_price_all": float(frame["Price"].mean()) if total else 0.0,
            "avg_support_tickets": float(frame["Support Tickets"].mean()) if total else 0.0,
            "avg_activities": float(frame["Activities"].mean()) if total else 0.0,
            "avg_resolution_time": float(frame["Avg Resolution Time"].mean())
            if total and "Avg Resolution Time" in frame.columns
            else 0.0,
            "at_risk_customers": at_risk,
        }

    def analyze_churn_by_plan(self) -> pd.DataFrame:
        rows = []
        for plan in PLAN_ORDER:
            group = self.df[self.df["Plan"] == plan]
            customers = int(len(group))
            churned = int(group["Churned"].sum()) if customers else 0
            rows.append(
                {
                    "Plan": plan,
                    "Customers": customers,
                    "Churned": churned,
                    "Active": customers - churned,
                    "Churn Rate": (churned / customers) if customers else 0.0,
                    "Avg Price": float(group["Price"].mean()) if customers else 0.0,
                    "Active MRR": float(group.loc[group["Churned"] == 0, "Price"].sum()) if customers else 0.0,
                    "Churned MRR": float(group.loc[group["Churned"] == 1, "Price"].sum()) if customers else 0.0,
                }
            )
        return pd.DataFrame(rows)

    def analyze_churn_by_engagement(self) -> pd.DataFrame:
        rows = []
        for engagement in ENGAGEMENT_ORDER:
            group = self.df[self.df["Engagement"] == engagement]
            customers = int(len(group))
            churned = int(group["Churned"].sum()) if customers else 0
            rows.append(
                {
                    "Engagement": engagement,
                    "Customers": customers,
                    "Churned": churned,
                    "Active": customers - churned,
                    "Churn Rate": (churned / customers) if customers else 0.0,
                }
            )
        return pd.DataFrame(rows)

    def analyze_engagement_distribution(self) -> pd.DataFrame:
        total = len(self.df)
        rows = []
        for engagement in ENGAGEMENT_ORDER:
            customers = int((self.df["Engagement"] == engagement).sum())
            rows.append(
                {
                    "Engagement": engagement,
                    "Customers": customers,
                    "Share": (customers / total) if total else 0.0,
                }
            )
        return pd.DataFrame(rows)

    def analyze_support_metrics(self) -> dict:
        frame = self.df
        active = frame[frame["Churned"] == 0]
        churned = frame[frame["Churned"] == 1]

        def _mean(group: pd.DataFrame, column: str) -> float:
            if group.empty or column not in group.columns:
                return 0.0
            return float(group[column].mean())

        return {
            "avg_tickets_active": _mean(active, "Support Tickets"),
            "avg_tickets_churned": _mean(churned, "Support Tickets"),
            "avg_activities_active": _mean(active, "Activities"),
            "avg_activities_churned": _mean(churned, "Activities"),
            "avg_resolution_active": _mean(active, "Avg Resolution Time"),
            "avg_resolution_churned": _mean(churned, "Avg Resolution Time"),
            "ticket_gap_churned_minus_active": _mean(churned, "Support Tickets")
            - _mean(active, "Support Tickets"),
            "activity_gap_churned_minus_active": _mean(churned, "Activities")
            - _mean(active, "Activities"),
        }


class ChurnPredictors:
    """Compare segments and rank simple associations with churn."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def analyze_plan_churn_impact(self) -> pd.DataFrame:
        overall = float(self.df["Churned"].mean()) if len(self.df) else 0.0
        table = FitlyEDA(self.df).analyze_churn_by_plan()
        table["Overall Churn Rate"] = overall
        table["Rate Difference"] = table["Churn Rate"] - overall
        table["Lift"] = np.where(overall > 0, table["Churn Rate"] / overall, np.nan)
        return table

    def analyze_engagement_churn_impact(self) -> pd.DataFrame:
        overall = float(self.df["Churned"].mean()) if len(self.df) else 0.0
        table = FitlyEDA(self.df).analyze_churn_by_engagement()
        table["Overall Churn Rate"] = overall
        table["Rate Difference"] = table["Churn Rate"] - overall
        table["Lift"] = np.where(overall > 0, table["Churn Rate"] / overall, np.nan)
        return table

    def correlate_features(self) -> pd.Series:
        """Pearson correlation of each modeled feature with ``Churned``."""
        plan_map = {name: index for index, name in enumerate(PLAN_ORDER)}
        engagement_map = {
            "No Engagement": 0,
            "Low": 1,
            "Medium": 2,
            "High": 3,
        }
        working = pd.DataFrame(
            {
                "Price": self.df["Price"],
                "Support Tickets": self.df["Support Tickets"],
                "Activities": self.df["Activities"],
                "Plan (ordinal)": self.df["Plan"].map(plan_map),
                "Engagement (ordinal)": self.df["Engagement"].map(engagement_map),
                "Churned": self.df["Churned"],
            }
        )
        if "Avg Resolution Time" in self.df.columns and self.df["Avg Resolution Time"].notna().all():
            working["Avg Resolution Time"] = self.df["Avg Resolution Time"]
        correlations = working.corr(numeric_only=True)["Churned"].drop(labels=["Churned"])
        return correlations.sort_values(key=lambda series: series.abs(), ascending=False)

    def get_high_risk_segments(self, threshold: float = 0.35) -> pd.DataFrame:
        grouped = (
            self.df.groupby(["Plan", "Engagement"], as_index=False)
            .agg(
                customers=("Customer ID", "count"),
                churned=("Churned", "sum"),
                churn_rate=("Churned", "mean"),
                avg_price=("Price", "mean"),
            )
        )
        grouped["churned"] = grouped["churned"].astype(int)
        grouped["customers"] = grouped["customers"].astype(int)
        risky = grouped[grouped["churn_rate"] >= threshold].copy()
        return risky.sort_values(["churn_rate", "customers"], ascending=[False, False]).reset_index(drop=True)


class CustomerSegmentation:
    """Lifetime-value bands, at-risk customers, and retention headroom."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def segment_by_ltv(self, lifetime_months: int = 24, gross_margin: float = 0.80) -> pd.DataFrame:
        """Attach gross-margin LTV and a price band. Churned customers keep a listed LTV.

        Tenure is not in the workbook, so LTV is price x lifetime x margin for
        every customer. That is a bookable-value view, not observed revenue.
        """
        segmented = self.df.copy()
        segmented["LTV"] = segmented["Price"] * lifetime_months * gross_margin

        def _band(row: pd.Series) -> str:
            if row["Plan"] == "Free" or row["Price"] <= 0:
                return "Free"
            if row["Price"] < 30:
                return "Low LTV"
            if row["Price"] < 80:
                return "Medium LTV"
            return "High LTV"

        segmented["LTV Segment"] = segmented.apply(_band, axis=1)
        return segmented

    def identify_at_risk(self, churn_threshold: float = 0.35) -> pd.DataFrame:
        """Active customers in engagement levels whose historical churn rate meets the threshold.

        With the study margins and the default threshold of 0.35 this is the
        Low and No Engagement active population (41 customers).
        """
        rates = self.df.groupby("Engagement")["Churned"].mean()
        risky_levels = [level for level, rate in rates.items() if rate >= churn_threshold]
        columns = list(self.df.columns) + ["segment_churn_rate"]
        if not risky_levels:
            return pd.DataFrame(columns=columns)

        at_risk = self.df[
            self.df["Engagement"].isin(risky_levels) & (self.df["Churned"] == 0)
        ].copy()
        at_risk["segment_churn_rate"] = at_risk["Engagement"].map(rates).astype(float)
        return at_risk.reset_index(drop=True)

    def calculate_retention_potential(self, target_churn_rate: float = 0.20) -> dict:
        """Expected saves among still-active paid customers with low or no engagement.

        ``current_churn_rate`` is the historical churn rate of that paid segment,
        including customers who already churned. The rate gap is applied only to
        customers who are still active. Empty segments return zeros.
        """
        paid = self.df[self.df["Plan"] != "Free"]
        segment = paid[paid["Engagement"].isin(["Low", "No Engagement"])]
        result = {
            "segment": "Paid Low/No Engagement",
            "segment_size": int(len(segment)),
            "active_customers": 0,
            "churned_customers": 0,
            "current_churn_rate": 0.0,
            "target_churn_rate": float(target_churn_rate),
            "potential_customers_saved": 0.0,
            "avg_monthly_price": 0.0,
            "potential_mrr_saved": 0.0,
            "potential_ltv_saved": 0.0,
        }
        if segment.empty:
            return result

        current = float(segment["Churned"].mean())
        active = segment[segment["Churned"] == 0]
        n_active = int(len(active))
        avg_price = float(active["Price"].mean()) if n_active else 0.0
        rate_gap = max(current - float(target_churn_rate), 0.0)
        saved = rate_gap * n_active
        result.update(
            {
                "active_customers": n_active,
                "churned_customers": int(segment["Churned"].sum()),
                "current_churn_rate": current,
                "potential_customers_saved": float(saved),
                "avg_monthly_price": round(avg_price, 2),
                "potential_mrr_saved": round(float(saved * avg_price), 2),
                "potential_ltv_saved": round(float(saved * avg_price * 24 * 0.80), 2),
            }
        )
        return result


class ChurnVisualizations:
    """Save churn charts under the repository ``outputs`` directory."""

    def __init__(self, df: pd.DataFrame, output_dir: str | Path | None = None):
        self.df = df.copy()
        self.output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def set_style(self) -> None:
        sns.set_theme(style="whitegrid", context="notebook")
        plt.rcParams["axes.titlesize"] = 13
        plt.rcParams["axes.labelsize"] = 11
        plt.rcParams["figure.dpi"] = 120

    def _save(self, fig: plt.Figure, filename: str) -> Path:
        path = self.output_dir / filename
        try:
            fig.savefig(path, dpi=120)
        finally:
            plt.close(fig)
        print(f"Saved {path}")
        return path

    def plot_churn_by_plan(self, filename: str = "churn_by_plan.png") -> Path:
        table = FitlyEDA(self.df).analyze_churn_by_plan()
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        sns.barplot(
            data=table,
            x="Plan",
            y="Churn Rate",
            hue="Plan",
            order=PLAN_ORDER,
            hue_order=PLAN_ORDER,
            palette="Blues_d",
            dodge=False,
            legend=False,
            ax=ax,
        )
        ax.set_ylim(0, max(0.5, float(table["Churn Rate"].max()) * 1.25))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _pos: f"{value:.0%}"))
        ax.set_title("Churn rate by plan")
        ax.set_xlabel("Plan")
        ax.set_ylabel("Churn rate")
        for patch, rate, customers in zip(ax.patches, table["Churn Rate"], table["Customers"]):
            ax.annotate(
                f"{rate:.0%}\n(n={customers})",
                (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                ha="center",
                va="bottom",
                fontsize=8,
            )
        return self._save(fig, filename)

    def plot_churn_by_engagement(self, filename: str = "churn_by_engagement.png") -> Path:
        table = FitlyEDA(self.df).analyze_churn_by_engagement()
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        sns.barplot(
            data=table,
            x="Engagement",
            y="Churn Rate",
            hue="Engagement",
            order=ENGAGEMENT_ORDER,
            hue_order=ENGAGEMENT_ORDER,
            palette="Oranges",
            dodge=False,
            legend=False,
            ax=ax,
        )
        ax.set_ylim(0, max(0.5, float(table["Churn Rate"].max()) * 1.25))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _pos: f"{value:.0%}"))
        ax.set_title("Churn rate by engagement")
        ax.set_xlabel("Engagement")
        ax.set_ylabel("Churn rate")
        for patch, rate in zip(ax.patches, table["Churn Rate"]):
            ax.annotate(
                f"{rate:.0%}",
                (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                ha="center",
                va="bottom",
                fontsize=9,
            )
        return self._save(fig, filename)

    def plot_heatmap_plan_engagement(self, filename: str = "heatmap_plan_engagement.png") -> Path:
        rates = pd.crosstab(
            self.df["Plan"],
            self.df["Engagement"],
            values=self.df["Churned"],
            aggfunc="mean",
        ).reindex(index=PLAN_ORDER, columns=ENGAGEMENT_ORDER)
        annot = rates.map(lambda value: "" if pd.isna(value) else f"{value:.0%}")
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        sns.heatmap(
            rates.fillna(0),
            annot=annot.fillna(""),
            fmt="",
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            linewidths=0.5,
            ax=ax,
            cbar_kws={"format": plt.FuncFormatter(lambda value, _pos: f"{value:.0%}")},
        )
        ax.set_title("Churn rate by plan and engagement")
        ax.set_xlabel("Engagement")
        ax.set_ylabel("Plan")
        return self._save(fig, filename)

    def plot_engagement_distribution(self, filename: str = "engagement_distribution.png") -> Path:
        counts = (
            self.df.groupby(["Engagement", "Churned"], as_index=False)
            .size()
            .rename(columns={"size": "Customers"})
        )
        counts["Status"] = counts["Churned"].map({0: "Active", 1: "Churned"})
        counts["Engagement"] = _ordered_categories(counts["Engagement"], ENGAGEMENT_ORDER)
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        sns.barplot(
            data=counts,
            x="Engagement",
            y="Customers",
            hue="Status",
            order=ENGAGEMENT_ORDER,
            hue_order=["Active", "Churned"],
            palette={"Active": "#4C78A8", "Churned": "#E45756"},
            ax=ax,
        )
        ax.set_title("Engagement distribution by churn status")
        ax.set_xlabel("Engagement")
        ax.set_ylabel("Customers")
        return self._save(fig, filename)


class RecommendationEngine:
    """Four actions computed from the customer table, not from hardcoded rates."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def generate_recommendations(self) -> list[dict]:
        frame = self.df
        eda = FitlyEDA(frame)
        plan = eda.analyze_churn_by_plan().set_index("Plan")
        engagement = eda.analyze_churn_by_engagement().set_index("Engagement")
        support = eda.analyze_support_metrics()
        retention = CustomerSegmentation(frame).calculate_retention_potential()
        at_risk = CustomerSegmentation(frame).identify_at_risk()

        free = plan.loc["Free"] if "Free" in plan.index else None
        basic_rate = float(plan.loc["Basic", "Churn Rate"]) if "Basic" in plan.index else 0.0
        if free is not None:
            free_rate = float(free["Churn Rate"])
            free_active = int(free["Active"])
            free_saved = max(free_rate - basic_rate, 0.0) * free_active
            free_issue = (
                f"Free churn is {_pct(free_rate)} "
                f"({int(free['Churned'])} of {int(free['Customers'])} customers), "
                f"above the Basic rate of {_pct(basic_rate)}."
            )
            free_impact = (
                f"Closing the gap with Basic on the {free_active} active Free customers "
                f"is about {free_saved:.1f} additional retained accounts. "
                "Revenue follows only if those accounts convert to a paid plan."
            )
        else:
            free_issue = "Free plan is not present in this extract."
            free_impact = "No Free-plan impact estimate."

        low_rate = float(engagement.loc["Low", "Churn Rate"]) if "Low" in engagement.index else 0.0
        none_rate = (
            float(engagement.loc["No Engagement", "Churn Rate"])
            if "No Engagement" in engagement.index
            else 0.0
        )
        at_risk_n = int(len(at_risk))
        paid_active = int(retention["active_customers"])
        engagement_issue = (
            f"Low engagement churns at {_pct(low_rate)} and No Engagement at {_pct(none_rate)}. "
            f"{at_risk_n} active customers are still in those two levels, "
            f"including {paid_active} on a paid plan."
        )
        engagement_impact = (
            f"Moving the paid low/no-engagement churn rate from "
            f"{_pct(retention['current_churn_rate'])} toward "
            f"{_pct(retention['target_churn_rate'])} is about "
            f"{retention['potential_customers_saved']:.1f} saved customers, "
            f"${retention['potential_mrr_saved']:.0f} MRR, and "
            f"${retention['potential_ltv_saved']:.0f} gross-margin LTV "
            "(24 months, 80% margin)."
        )

        medium = engagement.loc["Medium"] if "Medium" in engagement.index else None
        total_churned = int(frame["Churned"].sum()) if len(frame) else 0
        if medium is not None and total_churned:
            medium_share = float(medium["Churned"]) / total_churned
            medium_issue = (
                f"Medium engagement holds {int(medium['Churned'])} of {total_churned} "
                f"churned customers ({_pct(medium_share)}) at a {_pct(float(medium['Churn Rate']))} rate. "
                f"{int(medium['Active'])} Medium customers are still active."
            )
            medium_impact = (
                "A nurture program on the active Medium group addresses the largest "
                "slice of historical churn volume, even though its rate is below Low and No Engagement."
            )
        else:
            medium_issue = "Medium engagement volume could not be measured."
            medium_impact = "No Medium-engagement impact estimate."

        activity_gap = support["activity_gap_churned_minus_active"]
        activity_issue = (
            f"Churned customers average {support['avg_activities_churned']:.2f} activities "
            f"versus {support['avg_activities_active']:.2f} for active customers "
            f"(gap {activity_gap:.2f}). Support tickets average "
            f"{support['avg_tickets_churned']:.2f} among churned customers and "
            f"{support['avg_tickets_active']:.2f} among active customers."
        )
        activity_impact = (
            "Raising product activity among quiet accounts attacks the strongest "
            "simple association in this workbook. Resolution time is only a mild "
            "companion signal and should not be treated as a churn label."
        )

        return [
            {
                "priority": "P1",
                "title": "Fix Free-plan onboarding and the path to paid",
                "current_issue": free_issue,
                "actions": [
                    "Review the first-week Free experience for drop-off before the first activity.",
                    "Add a time-boxed prompt that converts active Free users to Basic.",
                    "Track Free churn weekly against the Basic churn rate.",
                ],
                "owner": "Growth",
                "timeline": "30 days",
                "expected_impact": free_impact,
            },
            {
                "priority": "P2",
                "title": "Re-engage active low and no-engagement customers",
                "current_issue": engagement_issue,
                "actions": [
                    f"Open a save playbook for the {at_risk_n} active Low and No Engagement customers.",
                    "Prioritize the paid subset, where a retained account still has monthly revenue.",
                    "Pair outreach with a single activation task rather than a discount by default.",
                ],
                "owner": "Customer Success",
                "timeline": "14 days",
                "expected_impact": engagement_impact,
            },
            {
                "priority": "P3",
                "title": "Nurture the Medium engagement cohort",
                "current_issue": medium_issue,
                "actions": [
                    "Build a lifecycle track for Medium customers who have not added a second activity.",
                    "Flag Medium accounts whose support tickets are rising.",
                    "Set a target to move Medium churn toward the High-engagement rate.",
                ],
                "owner": "Lifecycle Marketing",
                "timeline": "45 days",
                "expected_impact": medium_impact,
            },
            {
                "priority": "P4",
                "title": "Increase product activity before accounts go quiet",
                "current_issue": activity_issue,
                "actions": [
                    "Instrument an activation checklist and alert when a paid account stays at zero activities.",
                    "Route high-ticket, low-activity accounts to support before renewal.",
                    "Do not use average resolution time alone to score risk; the separation is modest.",
                ],
                "owner": "Product",
                "timeline": "60 days",
                "expected_impact": activity_impact,
            },
        ]


def main() -> dict:
    """Run the core analysis and return the collected results."""
    loader = FitlyDataLoader()
    data = loader.load_data()
    if not loader.validate_data():
        raise RuntimeError("Customer data failed validation.")

    customers = data[CUSTOMER_SHEET]
    eda = FitlyEDA(customers)
    overall = eda.get_overall_metrics()
    plan = eda.analyze_churn_by_plan()
    engagement = eda.analyze_churn_by_engagement()
    distribution = eda.analyze_engagement_distribution()
    support = eda.analyze_support_metrics()

    predictors = ChurnPredictors(customers)
    plan_impact = predictors.analyze_plan_churn_impact()
    engagement_impact = predictors.analyze_engagement_churn_impact()
    correlations = predictors.correlate_features()
    high_risk = predictors.get_high_risk_segments()

    segments = CustomerSegmentation(customers)
    ltv = segments.segment_by_ltv()
    at_risk = segments.identify_at_risk()
    retention = segments.calculate_retention_potential()

    charts = ChurnVisualizations(customers)
    charts.set_style()
    plot_paths = [
        charts.plot_churn_by_plan(),
        charts.plot_churn_by_engagement(),
        charts.plot_heatmap_plan_engagement(),
        charts.plot_engagement_distribution(),
    ]

    recommendations = RecommendationEngine(customers).generate_recommendations()

    from fitly_utils import ReportGenerator

    summary = ReportGenerator().generate_executive_summary(
        overall, plan, engagement, recommendations
    )
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = DEFAULT_OUTPUT_DIR / "executive_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")

    print("\nOverall churn rate: {:.2%} ({}/{})".format(
        overall["churn_rate"], overall["churned_customers"], overall["total_customers"]
    ))
    print(f"At-risk active low/no-engagement customers: {overall['at_risk_customers']}")
    print(f"Active MRR: ${overall['active_mrr']:.2f} | Churned MRR: ${overall['churned_mrr']:.2f}")
    print("\nChurn by plan:")
    print(plan.to_string(index=False))
    print("\nChurn by engagement:")
    print(engagement.to_string(index=False))
    print(f"\nWrote {summary_path}")
    print(f"Recommendations: {len(recommendations)}")

    return {
        "data": data,
        "overall_metrics": overall,
        "plan_analysis": plan,
        "engagement_analysis": engagement,
        "engagement_distribution": distribution,
        "support_metrics": support,
        "plan_impact": plan_impact,
        "engagement_impact": engagement_impact,
        "correlations": correlations,
        "high_risk_segments": high_risk,
        "ltv_segments": ltv,
        "at_risk": at_risk,
        "retention_potential": retention,
        "recommendations": recommendations,
        "plots": plot_paths,
        "executive_summary_path": summary_path,
    }


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
