# Data dictionary

Source file: `data/fitly_churn_analysis.xlsx`, produced by `generate_data.py` with seed 42. Engagement is stored on each row. The analysis does not recompute it from activities, so the counts below stay fixed.

## Customer Details (400 rows)

| Column | Type | Definition |
| --- | --- | --- |
| Customer ID | string | `C10000` through `C10399`. |
| Plan | string | `Free`, `Basic`, `Pro`, or `Enterprise`. |
| Price | float, USD per month | Free is 0. Basic mean 19.87 (about 15–25). Pro mean 53.74 (about 45–65). Enterprise mean 116.01 (about 100–119). |
| Churned | int | 1 churned, 0 active. 114 ones. |
| Support Tickets | int | 0–6. Overall mean 2.30. |
| Activities | int | 0–5. Overall mean 1.11. Mean is 0.50 among churned customers and about 1.35 among active customers. |
| Engagement | string | `High`, `Medium`, `Low`, or `No Engagement`. |
| Avg Resolution Time | float, hours | About 0.5–32, mean about 10.4. Correlation with churn is about 0.12. |

### Margins

| Plan | Customers | Churned |
| --- | --- | --- |
| Free | 105 | 43 |
| Basic | 118 | 28 |
| Pro | 85 | 19 |
| Enterprise | 92 | 24 |

| Engagement | Customers | Churned | Active |
| --- | --- | --- | --- |
| High | 105 | 17 | 88 |
| Medium | 231 | 74 | 157 |
| Low | 56 | 20 | 36 |
| No Engagement | 8 | 3 | 5 |

Active Low + active No Engagement = 41. That is the at-risk count used by `identify_at_risk(0.35)`, because those two levels are the ones whose churn rate is at least 35%.

### Plan × engagement (one feasible table)

Totals (churned in parentheses):

| Plan | High | Medium | Low | No Engagement |
| --- | --- | --- | --- | --- |
| Free | 15 (8) | 61 (21) | 25 (12) | 4 (2) |
| Basic | 35 (4) | 66 (19) | 15 (4) | 2 (1) |
| Pro | 28 (2) | 48 (15) | 8 (2) | 1 (0) |
| Enterprise | 27 (3) | 56 (19) | 8 (2) | 1 (0) |

Rates on the single-customer cells are 0% or 100% by arithmetic. Do not read them as a precise segment rate.

### Revenue totals

Active price sum: $13,177.00. Churned price sum: $4,408.48. The churned total is 52 cents under $4,409 because the three paid plan means, applied to 118, 85, and 92 customers, sum to $17,585.48 rather than $17,586.

## Other sheets

Derived from Customer Details so they cannot drift:

| Sheet | Contents |
| --- | --- |
| Churn by Plan | Customers, churned, active, churn rate, average price, active MRR, churned MRR. |
| Churn by Engagement | Customers, churned, active, churn rate. |
| Data Validation | Expected versus actual checks. Every row is PASS on the generated file. |
| Summary Statistics | `describe` for the numeric columns. |
| Support Analysis | Ticket, resolution, and activity means for active versus churned customers. |

## Analysis assumptions (not columns)

| Assumption | Value |
| --- | --- |
| LTV lifetime | 24 months |
| Gross margin | 0.80 |
| CAC | $50 |
| Listed LTV | Price × 24 × 0.80 |
| CAC payback | CAC / (mean price × 0.80), skipped when price is 0 |
| Retention target | 20% churn on the paid Low/No Engagement segment |
| Model | Random forest, 100 trees, `max_depth=6`, `min_samples_leaf=5`, stratified 25% holdout, `random_state=42` |
| Plan codes | Free 0, Basic 1, Pro 2, Enterprise 3 |
| Engagement codes | No Engagement 0, Low 1, Medium 2, High 3 |
