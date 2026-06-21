# Zoho SMB — Customer Churn Analysis & Revenue Health Dashboard

> **A data analyst portfolio project** — built to demonstrate real-world SaaS analytics skills  
> Tools used: Python · Pandas · Scikit-learn · Matplotlib · SQL

---

## Project Summary

Zoho serves **1M+ paying SMB customers** across 55+ products. Their biggest revenue challenge:  
**single-app customers churn at 3× the rate of Zoho One (multi-suite) customers.**

This project builds an end-to-end analytics pipeline that:
1. Identifies **why** customers churn (EDA + cohort analysis)
2. Quantifies **how much MRR** is at risk each month (MRR waterfall)
3. Predicts **who will churn next** (Random Forest, AUC = 0.843)
4. Scores **who is ready to upsell** to Zoho One (Gradient Boosting, AUC = 1.00)
5. Delivers an **executive dashboard** with all findings in one view

---

## Key Findings

| Insight | Finding |
|---|---|
| Free plan churn rate | ~56% |
| Zoho One churn rate | ~6% |
| Biggest churn window | Months 0–6 (danger zone) |
| 1-app users vs 9+ app users | 3× higher churn rate |
| Churn model AUC | **0.843** |
| At-risk MRR (score > 60) | $1,740/month |
| Upsell candidates identified | **1,230 customers** |
| Upsell revenue potential | $1.72M/month |

> **Bottom line:** If Zoho converts just 5% of upsell-ready customers to Zoho One,  
> that's **~$86,000 in new MRR** — without acquiring a single new customer.

---

## Project Structure

```
zoho_churn_project/
│
├── generate_data.py          # Synthetic SMB dataset (5,000 customers)
├── phase2_eda.py             # Exploratory data analysis + visualizations
├── phase3_cohort_mrr.py      # Cohort retention heatmap + MRR waterfall
├── phase4_ml_model.py        # Churn & upsell prediction models
├── phase5_dashboard.py       # Executive summary dashboard
├── zoho_churn_queries.sql    # 10 production-ready SQL queries
│
├── zoho_smb_data.csv         # Raw dataset
├── zoho_scored_data.csv      # Dataset with churn + upsell scores
│
└── outputs/
    ├── 01_eda_overview.png         # Churn by plan, apps, tenure, distribution
    ├── 02_eda_segmentation.png     # Churn by country and industry
    ├── 03_cohort_retention.png     # Quarterly cohort retention heatmap
    ├── 04_mrr_waterfall.png        # MRR movement + total MRR trend
    ├── 05_ltv_nrr.png              # LTV and NRR by plan
    ├── 06_ml_model.png             # ROC, feature importance, risk matrix
    └── 07_executive_dashboard.png  # Full executive summary (1-pager)
```

---

## How to Run

```bash
# 1. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# 2. Generate dataset
python generate_data.py

# 3. Run all phases in order
python phase2_eda.py
python phase3_cohort_mrr.py
python phase4_ml_model.py
python phase5_dashboard.py
```

All charts are saved to the `outputs/` folder.

---

## Dataset

The dataset is synthetically generated to mimic Zoho's SMB customer base:
- **5,000 customers** across Free, Standard, Professional, Enterprise, Zoho One plans
- Features: signup date, plan tier, apps used, monthly logins, support tickets, seats, company size, industry, country, tenure, churn flag, upsell flag
- Churn logic is grounded in real SaaS behavior patterns (single-app users, low engagement, early tenure = higher churn)

To use real data: replace `generate_data.py` with a SQL query to your Zoho Analytics / CRM database, keeping the same column names.

---

## SQL Queries

`zoho_churn_queries.sql` contains 10 production-ready queries:

1. Base customer snapshot table
2. Churn rate by plan tier
3. MRR movement (New · Expansion · Churn)
4. Cohort retention matrix
5. Customer LTV by plan
6. Net Revenue Retention (NRR)
7. High-risk customer watchlist
8. Zoho One upsell pipeline
9. Single-app vs multi-app vs Zoho One churn comparison
10. Business impact summary

---

## Business Recommendations

Based on the analysis, three actions would have the highest revenue impact:

**1. Intervene at months 3–5 (the danger zone)**  
Trigger an automated health check email + in-app nudge at month 3 for customers with < 2 apps used and < 8 logins/month. This is the highest-churn window.

**2. Push multi-app adoption early**  
Customers using 5+ apps churn at half the rate of single-app users. A free 30-day trial of a complementary app (e.g. Zoho Books for CRM users) at month 1 could significantly improve stickiness.

**3. Create an upsell fast lane for the 1,230 identified candidates**  
Customers scoring > 60 on the upsell model are already power users — they just need a Zoho One offer. A targeted in-app or email campaign with a discounted annual Zoho One plan could convert 5–10% of this segment.

---

## Tools & Technologies

| Tool | Used for |
|---|---|
| Python 3.12 | Core analysis, ML models |
| Pandas | Data wrangling, cohort logic |
| Scikit-learn | Random Forest (churn), Gradient Boosting (upsell) |
| Matplotlib / Seaborn | All visualizations |
| SQL (PostgreSQL syntax) | Production queries |
| Power BI / Tableau | Dashboard (see `/dashboard` folder) |

---

## Author

**Arun**  
Aspiring Data Analyst | Python · SQL · Power BI · Tableau  
[LinkedIn](#) · [GitHub](#) · [Tableau Public](#)

---

*This project was built as a portfolio piece specifically targeting SaaS companies like Zoho.  
The dataset is synthetic — no real customer data was used.*
