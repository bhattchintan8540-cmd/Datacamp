"""Generate the Fit.ly churn workbook used by the analysis package.

The original spreadsheet is not part of this repository. This module builds
``data/fitly_churn_analysis.xlsx`` with seed 42 so a fresh clone is runnable
and the study margins (plan mix, engagement mix, and churn counts) match
exactly.

Run from the repository root::

    python generate_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_WORKBOOK = REPO_ROOT / "data" / "fitly_churn_analysis.xlsx"

PLAN_ORDER = ["Free", "Basic", "Pro", "Enterprise"]
ENGAGEMENT_ORDER = ["High", "Medium", "Low", "No Engagement"]

# Feasible plan x engagement totals. Row and column sums match the study.
JOINT_TOTALS = {
    ("Free", "High"): 15,
    ("Free", "Medium"): 61,
    ("Free", "Low"): 25,
    ("Free", "No Engagement"): 4,
    ("Basic", "High"): 35,
    ("Basic", "Medium"): 66,
    ("Basic", "Low"): 15,
    ("Basic", "No Engagement"): 2,
    ("Pro", "High"): 28,
    ("Pro", "Medium"): 48,
    ("Pro", "Low"): 8,
    ("Pro", "No Engagement"): 1,
    ("Enterprise", "High"): 27,
    ("Enterprise", "Medium"): 56,
    ("Enterprise", "Low"): 8,
    ("Enterprise", "No Engagement"): 1,
}

# Churned counts inside each plan x engagement cell. Consistent with both
# the plan churn margins and the engagement churn margins.
JOINT_CHURN = {
    ("Free", "High"): 8,
    ("Free", "Medium"): 21,
    ("Free", "Low"): 12,
    ("Free", "No Engagement"): 2,
    ("Basic", "High"): 4,
    ("Basic", "Medium"): 19,
    ("Basic", "Low"): 4,
    ("Basic", "No Engagement"): 1,
    ("Pro", "High"): 2,
    ("Pro", "Medium"): 15,
    ("Pro", "Low"): 2,
    ("Pro", "No Engagement"): 0,
    ("Enterprise", "High"): 3,
    ("Enterprise", "Medium"): 19,
    ("Enterprise", "Low"): 2,
    ("Enterprise", "No Engagement"): 0,
}

# min price, max price, target mean. Free is identically zero.
PLAN_PRICE = {
    "Free": (0.0, 0.0, 0.0),
    "Basic": (15.0, 25.0, 19.87),
    "Pro": (45.0, 65.0, 53.74),
    "Enterprise": (100.0, 119.0, 116.01),
}

TARGET_ACTIVE_PRICE_SUM = 13177.00
TARGET_CHURNED_TICKETS = 319  # mean about 2.80; active carries the rest
TARGET_TICKET_SUM = 920  # 400 * 2.30
TARGET_CHURNED_ACTIVITIES = 57  # mean 0.50
TARGET_ACTIVITY_SUM = 444  # 400 * 1.11
TARGET_RESOLUTION_MEAN = 10.4

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


def _repair_integer_sum(values: np.ndarray, target: int, low: int, high: int) -> np.ndarray:
    """Move integer values by 1 until they sum to ``target``, staying in range."""
    adjusted = np.array(values, dtype=int, copy=True)
    diff = int(target - adjusted.sum())
    guard = 0
    while diff != 0 and guard < 100000:
        guard += 1
        if diff > 0:
            candidates = np.flatnonzero(adjusted < high)
            if len(candidates) == 0:
                break
            chosen = candidates[np.argmin(adjusted[candidates])]
            adjusted[chosen] += 1
            diff -= 1
        else:
            candidates = np.flatnonzero(adjusted > low)
            if len(candidates) == 0:
                break
            chosen = candidates[np.argmax(adjusted[candidates])]
            adjusted[chosen] -= 1
            diff += 1
    if int(adjusted.sum()) != int(target):
        raise RuntimeError(
            f"Could not reach integer sum {target} within [{low}, {high}] "
            f"(landed on {int(adjusted.sum())})."
        )
    return adjusted


def _spread_cents(cents: np.ndarray, indexes: np.ndarray, amount: int) -> None:
    """Add ``amount`` cents across ``indexes`` as evenly as possible."""
    count = len(indexes)
    if count == 0 or amount == 0:
        return
    sign = 1 if amount > 0 else -1
    base, remainder = divmod(abs(int(amount)), count)
    if base:
        cents[indexes] += sign * base
    if remainder:
        cents[indexes[:remainder]] += sign


def _assign_prices(plan: np.ndarray, churned: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Build 2-decimal prices with the target plan means and active-revenue split."""
    cents = np.zeros(len(plan), dtype=int)

    for name in PLAN_ORDER:
        low, high, mean = PLAN_PRICE[name]
        indexes = np.flatnonzero(plan == name)
        count = len(indexes)
        if name == "Free":
            cents[indexes] = 0
            continue

        low_c = int(round(low * 100))
        high_c = int(round(high * 100))
        base_c = int(round(mean * 100))
        target_total = int(round(mean * count * 100))

        noise = rng.integers(-150, 151, size=count)
        noise = np.rint(noise - noise.mean()).astype(int)
        draft = base_c + noise
        cents[indexes] = _repair_integer_sum(draft, target_total, low_c, high_c)

    paid_active = np.flatnonzero((plan != "Free") & (churned == 0))
    active_cents = int(cents[paid_active].sum())
    delta = active_cents - int(round(TARGET_ACTIVE_PRICE_SUM * 100))

    # Move ``delta`` cents from active paid customers onto churned paid customers
    # inside the same plan so each plan's mean price stays put.
    churn_counts = {
        name: int(((plan == name) & (churned == 1)).sum())
        for name in PLAN_ORDER
        if name != "Free"
    }
    total_churned_paid = sum(churn_counts.values())
    if total_churned_paid == 0:
        raise RuntimeError("No paid churned customers available to rebalance price.")

    allocated = 0
    names = [name for name in PLAN_ORDER if name != "Free"]
    for position, name in enumerate(names):
        if position == len(names) - 1:
            share = delta - allocated
        else:
            share = int(round(delta * churn_counts[name] / total_churned_paid))
            allocated += share
        active_idx = np.flatnonzero((plan == name) & (churned == 0))
        churn_idx = np.flatnonzero((plan == name) & (churned == 1))
        _spread_cents(cents, active_idx, -share)
        _spread_cents(cents, churn_idx, share)

        low_c = int(round(PLAN_PRICE[name][0] * 100))
        high_c = int(round(PLAN_PRICE[name][1] * 100))
        if (cents[active_idx] < low_c).any() or (cents[churn_idx] > high_c).any():
            raise RuntimeError(f"Price rebalance pushed {name} outside its allowed range.")

    return cents / 100.0


def _assign_resolution_time(churned: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Hours to resolve a ticket. Mildly higher for churned customers, with overlap."""
    baseline = rng.normal(loc=9.6, scale=5.8, size=len(churned))
    # A small location shift, not a separator. Overlap stays large.
    shifted = baseline + np.where(churned == 1, 1.6, 0.0)
    values = np.clip(shifted, 0.5, 32.0)
    for _ in range(8):
        values = np.clip(values + (TARGET_RESOLUTION_MEAN - values.mean()), 0.5, 32.0)
    return np.round(values, 2)


def _customer_frame(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    cursor = 0
    for plan in PLAN_ORDER:
        for engagement in ENGAGEMENT_ORDER:
            key = (plan, engagement)
            total = JOINT_TOTALS[key]
            churned_n = JOINT_CHURN[key]
            if churned_n > total:
                raise RuntimeError(f"Churn count exceeds customers for {key}.")
            for offset in range(total):
                rows.append(
                    {
                        "Customer ID": f"C{10000 + cursor}",
                        "Plan": plan,
                        "Churned": 1 if offset < churned_n else 0,
                        "Engagement": engagement,
                    }
                )
                cursor += 1

    frame = pd.DataFrame(rows)
    plan = frame["Plan"].to_numpy()
    churned = frame["Churned"].to_numpy(dtype=int)

    frame["Price"] = _assign_prices(plan, churned, rng)

    tickets = np.zeros(len(frame), dtype=int)
    activities = np.zeros(len(frame), dtype=int)
    churn_idx = np.flatnonzero(churned == 1)
    active_idx = np.flatnonzero(churned == 0)

    tickets[churn_idx] = rng.binomial(6, 2.80 / 6.0, size=len(churn_idx))
    tickets[active_idx] = rng.binomial(6, 2.10 / 6.0, size=len(active_idx))
    tickets[churn_idx] = _repair_integer_sum(tickets[churn_idx], TARGET_CHURNED_TICKETS, 0, 6)
    tickets[active_idx] = _repair_integer_sum(
        tickets[active_idx], TARGET_TICKET_SUM - TARGET_CHURNED_TICKETS, 0, 6
    )

    activities[churn_idx] = rng.binomial(5, 0.10, size=len(churn_idx))
    activities[active_idx] = rng.binomial(5, 1.35 / 5.0, size=len(active_idx))
    activities[churn_idx] = _repair_integer_sum(
        activities[churn_idx], TARGET_CHURNED_ACTIVITIES, 0, 5
    )
    activities[active_idx] = _repair_integer_sum(
        activities[active_idx], TARGET_ACTIVITY_SUM - TARGET_CHURNED_ACTIVITIES, 0, 5
    )

    frame["Support Tickets"] = tickets
    frame["Activities"] = activities
    frame["Avg Resolution Time"] = _assign_resolution_time(churned, rng)
    frame["Churned"] = churned
    return frame[REQUIRED_COLUMNS]


def _assert_margins(frame: pd.DataFrame) -> None:
    if len(frame) != 400:
        raise AssertionError(f"Expected 400 customers, found {len(frame)}.")
    expected_ids = [f"C{i}" for i in range(10000, 10400)]
    if frame["Customer ID"].tolist() != expected_ids:
        raise AssertionError("Customer IDs are not C10000–C10399 in order.")
    if int(frame["Churned"].sum()) != 114:
        raise AssertionError("Churned total is not 114.")

    plan_counts = frame["Plan"].value_counts().to_dict()
    if plan_counts != {"Free": 105, "Basic": 118, "Pro": 85, "Enterprise": 92}:
        raise AssertionError(f"Unexpected plan counts: {plan_counts}")

    plan_churn = frame.groupby("Plan")["Churned"].sum().astype(int).to_dict()
    if plan_churn != {"Free": 43, "Basic": 28, "Pro": 19, "Enterprise": 24}:
        raise AssertionError(f"Unexpected plan churn counts: {plan_churn}")

    engagement_counts = frame["Engagement"].value_counts().to_dict()
    if engagement_counts != {"High": 105, "Medium": 231, "Low": 56, "No Engagement": 8}:
        raise AssertionError(f"Unexpected engagement counts: {engagement_counts}")

    engagement_churn = frame.groupby("Engagement")["Churned"].sum().astype(int).to_dict()
    expected_engagement_churn = {"High": 17, "Medium": 74, "Low": 20, "No Engagement": 3}
    if engagement_churn != expected_engagement_churn:
        raise AssertionError(f"Unexpected engagement churn counts: {engagement_churn}")

    at_risk = frame[
        frame["Engagement"].isin(["Low", "No Engagement"]) & (frame["Churned"] == 0)
    ]
    if len(at_risk) != 41:
        raise AssertionError(f"At-risk active low/no-engagement count is {len(at_risk)}, not 41.")

    if not (frame.loc[frame["Plan"] == "Free", "Price"] == 0).all():
        raise AssertionError("Free plan prices must all be 0.")

    for name, (_low, _high, mean) in PLAN_PRICE.items():
        if name == "Free":
            continue
        actual = float(frame.loc[frame["Plan"] == name, "Price"].mean())
        if abs(actual - mean) > 0.02:
            raise AssertionError(f"{name} mean price {actual:.4f} is far from {mean:.2f}.")
        prices = frame.loc[frame["Plan"] == name, "Price"]
        if prices.min() < PLAN_PRICE[name][0] - 1e-9 or prices.max() > PLAN_PRICE[name][1] + 1e-9:
            raise AssertionError(f"{name} prices left the allowed range.")

    active_sum = float(frame.loc[frame["Churned"] == 0, "Price"].sum())
    churned_sum = float(frame.loc[frame["Churned"] == 1, "Price"].sum())
    if abs(active_sum - TARGET_ACTIVE_PRICE_SUM) > 1.0:
        raise AssertionError(f"Active price sum {active_sum:.2f} is not near 13177.")
    if abs(churned_sum - 4409) > 2.0:
        raise AssertionError(f"Churned price sum {churned_sum:.2f} is not near 4409.")

    if abs(float(frame["Support Tickets"].mean()) - 2.30) > 1e-9:
        raise AssertionError("Support ticket mean is not 2.30.")
    if abs(float(frame["Activities"].mean()) - 1.11) > 1e-9:
        raise AssertionError("Activities mean is not 1.11.")

    active_activities = float(frame.loc[frame["Churned"] == 0, "Activities"].mean())
    churn_activities = float(frame.loc[frame["Churned"] == 1, "Activities"].mean())
    if churn_activities > active_activities - 0.4:
        raise AssertionError("Activities are not clearly lower among churned customers.")

    resolution = frame["Avg Resolution Time"]
    if resolution.min() < 0.5 - 1e-9 or resolution.max() > 32 + 1e-9:
        raise AssertionError("Resolution time left the 0.5–32 hour range.")
    if abs(float(resolution.mean()) - TARGET_RESOLUTION_MEAN) > 0.05:
        raise AssertionError(f"Resolution mean {resolution.mean():.3f} is not near 10.4.")

    # Guard against a near-perfect label leak through resolution time.
    corr = float(np.corrcoef(frame["Churned"].to_numpy(), resolution.to_numpy())[0, 1])
    if abs(corr) > 0.45:
        raise AssertionError(f"Resolution time correlation with churn is too strong ({corr:.3f}).")


def _churn_by_plan(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for plan in PLAN_ORDER:
        group = frame[frame["Plan"] == plan]
        churned = int(group["Churned"].sum())
        customers = int(len(group))
        active = customers - churned
        rows.append(
            {
                "Plan": plan,
                "Customers": customers,
                "Churned": churned,
                "Active": active,
                "Churn Rate": churned / customers,
                "Avg Price": float(group["Price"].mean()),
                "Active MRR": float(group.loc[group["Churned"] == 0, "Price"].sum()),
                "Churned MRR": float(group.loc[group["Churned"] == 1, "Price"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _churn_by_engagement(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for engagement in ENGAGEMENT_ORDER:
        group = frame[frame["Engagement"] == engagement]
        churned = int(group["Churned"].sum())
        customers = int(len(group))
        rows.append(
            {
                "Engagement": engagement,
                "Customers": customers,
                "Churned": churned,
                "Active": customers - churned,
                "Churn Rate": churned / customers,
            }
        )
    return pd.DataFrame(rows)


def _data_validation(frame: pd.DataFrame) -> pd.DataFrame:
    at_risk = int(
        (
            frame["Engagement"].isin(["Low", "No Engagement"]) & (frame["Churned"] == 0)
        ).sum()
    )
    checks = [
        ("Row count", "400", str(len(frame))),
        ("Unique customer IDs", "400", str(frame["Customer ID"].nunique())),
        ("Churned customers", "114", str(int(frame["Churned"].sum()))),
        ("Free customers", "105", str(int((frame["Plan"] == "Free").sum()))),
        ("Basic customers", "118", str(int((frame["Plan"] == "Basic").sum()))),
        ("Pro customers", "85", str(int((frame["Plan"] == "Pro").sum()))),
        ("Enterprise customers", "92", str(int((frame["Plan"] == "Enterprise").sum()))),
        ("High engagement", "105", str(int((frame["Engagement"] == "High").sum()))),
        ("Medium engagement", "231", str(int((frame["Engagement"] == "Medium").sum()))),
        ("Low engagement", "56", str(int((frame["Engagement"] == "Low").sum()))),
        ("No engagement", "8", str(int((frame["Engagement"] == "No Engagement").sum()))),
        ("Active low or no engagement", "41", str(at_risk)),
        ("Missing values", "0", str(int(frame.isna().sum().sum()))),
        ("Free prices equal to 0", "105", str(int((frame.loc[frame["Plan"] == "Free", "Price"] == 0).sum()))),
    ]
    table = pd.DataFrame(checks, columns=["Check", "Expected", "Actual"])
    table["Status"] = np.where(table["Expected"] == table["Actual"], "PASS", "FAIL")
    return table


def _summary_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame[["Price", "Churned", "Support Tickets", "Activities", "Avg Resolution Time"]]
    summary = numeric.describe().T.reset_index().rename(columns={"index": "Metric"})
    summary.insert(0, "Column", summary.pop("Metric"))
    return summary


def _support_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, mask in (("Active", frame["Churned"] == 0), ("Churned", frame["Churned"] == 1)):
        group = frame.loc[mask]
        rows.append(
            {
                "Status": label,
                "Customers": int(len(group)),
                "Avg Support Tickets": float(group["Support Tickets"].mean()),
                "Avg Resolution Time": float(group["Avg Resolution Time"].mean()),
                "Avg Activities": float(group["Activities"].mean()),
            }
        )
    return pd.DataFrame(rows)


def generate_dataset(path: str | Path | None = None, seed: int = 42) -> Path:
    """Write the Fit.ly workbook and return its path.

    Parameters
    ----------
    path:
        Destination ``.xlsx``. Defaults to ``data/fitly_churn_analysis.xlsx``
        next to this module.
    seed:
        NumPy Generator seed. The published workbook uses 42.
    """
    destination = Path(path) if path is not None else DEFAULT_WORKBOOK
    destination.parent.mkdir(parents=True, exist_ok=True)

    customers = _customer_frame(seed)
    _assert_margins(customers)

    with pd.ExcelWriter(destination, engine="openpyxl") as writer:
        customers.to_excel(writer, sheet_name="Customer Details", index=False)
        _churn_by_plan(customers).to_excel(writer, sheet_name="Churn by Plan", index=False)
        _churn_by_engagement(customers).to_excel(writer, sheet_name="Churn by Engagement", index=False)
        _data_validation(customers).to_excel(writer, sheet_name="Data Validation", index=False)
        _summary_statistics(customers).to_excel(writer, sheet_name="Summary Statistics", index=False)
        _support_analysis(customers).to_excel(writer, sheet_name="Support Analysis", index=False)

    active_sum = float(customers.loc[customers["Churned"] == 0, "Price"].sum())
    churned_sum = float(customers.loc[customers["Churned"] == 1, "Price"].sum())
    resolution_corr = float(
        np.corrcoef(customers["Churned"].to_numpy(), customers["Avg Resolution Time"].to_numpy())[0, 1]
    )
    print(f"Wrote {destination}")
    print(f"Customers: {len(customers)} | Churned: {int(customers['Churned'].sum())}")
    print(f"Active price sum: {active_sum:.2f} | Churned price sum: {churned_sum:.2f}")
    print(
        "Means | tickets "
        f"{customers['Support Tickets'].mean():.2f} | activities "
        f"{customers['Activities'].mean():.2f} | resolution "
        f"{customers['Avg Resolution Time'].mean():.2f}"
    )
    print(f"Resolution/churn correlation: {resolution_corr:.3f}")
    return destination


def generate_data(path: str | Path | None = None, seed: int = 42) -> Path:
    """Alias used by loaders that look up ``generate_data`` by name."""
    return generate_dataset(path=path, seed=seed)


if __name__ == "__main__":
    generate_dataset()
