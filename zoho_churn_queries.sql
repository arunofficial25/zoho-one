-- ============================================================
--  ZOHO SMB CHURN ANALYSIS — SQL QUERIES
--  Author: Your Name | Built for Zoho Data Analyst role
-- ============================================================


-- ── 1. BASE TABLE: Customer snapshot ────────────────────────
CREATE TABLE zoho_customers AS
SELECT
    customer_id,
    signup_date,
    plan,
    monthly_price,
    apps_used,
    monthly_logins,
    support_tickets,
    seats,
    company_size,
    industry,
    country,
    tenure_months,
    churned,
    churn_month
FROM raw_customers;


-- ── 2. CHURN RATE BY PLAN TIER ───────────────────────────────
SELECT
    plan,
    COUNT(*)                                    AS total_customers,
    SUM(churned)                                AS churned_count,
    ROUND(100.0 * SUM(churned) / COUNT(*), 2)  AS churn_rate_pct,
    ROUND(AVG(monthly_price * seats), 2)        AS avg_monthly_revenue
FROM zoho_customers
GROUP BY plan
ORDER BY churn_rate_pct DESC;


-- ── 3. MRR MOVEMENT (New · Expansion · Churn) ────────────────
WITH monthly_mrr AS (
    SELECT
        DATE_TRUNC('month', signup_date)        AS month,
        plan,
        SUM(monthly_price * seats)              AS new_mrr
    FROM zoho_customers
    WHERE churned = 0
    GROUP BY 1, 2
),
churned_mrr AS (
    SELECT
        DATE_TRUNC('month', signup_date) +
            INTERVAL '1 month' * churn_month    AS month,
        SUM(monthly_price * seats)              AS churn_mrr
    FROM zoho_customers
    WHERE churned = 1
    GROUP BY 1
)
SELECT
    m.month,
    COALESCE(m.new_mrr, 0)                     AS new_mrr,
    COALESCE(c.churn_mrr, 0)                   AS churn_mrr,
    COALESCE(m.new_mrr, 0) -
        COALESCE(c.churn_mrr, 0)               AS net_new_mrr
FROM monthly_mrr m
LEFT JOIN churned_mrr c USING (month)
ORDER BY m.month;


-- ── 4. COHORT RETENTION MATRIX ────────────────────────────────
WITH cohorts AS (
    SELECT
        customer_id,
        DATE_TRUNC('quarter', signup_date)      AS cohort_quarter,
        tenure_months,
        churned
    FROM zoho_customers
)
SELECT
    cohort_quarter,
    COUNT(*)                                    AS cohort_size,
    ROUND(100.0 * SUM(CASE WHEN tenure_months >= 3  THEN 1 ELSE 0 END) / COUNT(*), 1) AS m3_retention,
    ROUND(100.0 * SUM(CASE WHEN tenure_months >= 6  THEN 1 ELSE 0 END) / COUNT(*), 1) AS m6_retention,
    ROUND(100.0 * SUM(CASE WHEN tenure_months >= 12 THEN 1 ELSE 0 END) / COUNT(*), 1) AS m12_retention,
    ROUND(100.0 * SUM(CASE WHEN tenure_months >= 18 THEN 1 ELSE 0 END) / COUNT(*), 1) AS m18_retention,
    ROUND(100.0 * SUM(CASE WHEN tenure_months >= 24 THEN 1 ELSE 0 END) / COUNT(*), 1) AS m24_retention
FROM cohorts
GROUP BY cohort_quarter
ORDER BY cohort_quarter;


-- ── 5. CUSTOMER LIFETIME VALUE (LTV) BY PLAN ─────────────────
SELECT
    plan,
    ROUND(AVG(monthly_price * seats), 2)                        AS avg_monthly_revenue,
    ROUND(AVG(tenure_months), 1)                                AS avg_tenure_months,
    ROUND(AVG(monthly_price * seats * tenure_months), 2)        AS avg_ltv,
    ROUND(100.0 * SUM(churned) / COUNT(*), 2)                   AS churn_rate_pct
FROM zoho_customers
GROUP BY plan
ORDER BY avg_ltv DESC;


-- ── 6. NET REVENUE RETENTION (NRR) ────────────────────────────
-- NRR = (Starting MRR + Expansion MRR - Churn MRR) / Starting MRR
WITH period_mrr AS (
    SELECT
        plan,
        SUM(CASE WHEN signup_date < '2024-01-01'
                 THEN monthly_price * seats ELSE 0 END)         AS start_mrr,
        SUM(CASE WHEN signup_date >= '2024-01-01' AND churned=0
                 THEN (monthly_price - 20) * seats ELSE 0 END)  AS expansion_mrr,
        SUM(CASE WHEN churned = 1
                 THEN monthly_price * seats ELSE 0 END)         AS churned_mrr
    FROM zoho_customers
    GROUP BY plan
)
SELECT
    plan,
    ROUND(start_mrr, 0)                                         AS start_mrr,
    ROUND(expansion_mrr, 0)                                     AS expansion_mrr,
    ROUND(churned_mrr, 0)                                       AS churned_mrr,
    ROUND(100.0 * (start_mrr + expansion_mrr - churned_mrr)
                / NULLIF(start_mrr, 0), 1)                      AS nrr_pct
FROM period_mrr
WHERE start_mrr > 0
ORDER BY nrr_pct DESC;


-- ── 7. HIGH-RISK CUSTOMER WATCHLIST ──────────────────────────
-- Customers who are single-app, low engagement, early tenure = danger
SELECT
    customer_id,
    plan,
    apps_used,
    monthly_logins,
    tenure_months,
    support_tickets,
    monthly_price * seats                                       AS monthly_revenue,
    CASE
        WHEN apps_used = 1 AND monthly_logins < 5
             AND tenure_months <= 5 THEN 'Critical'
        WHEN apps_used <= 2 AND monthly_logins < 10
             AND tenure_months <= 6 THEN 'High'
        WHEN monthly_logins < 8
             AND tenure_months <= 12                            THEN 'Medium'
        ELSE 'Low'
    END AS churn_risk_flag
FROM zoho_customers
WHERE churned = 0
ORDER BY churn_risk_flag, monthly_revenue DESC;


-- ── 8. ZOHO ONE UPSELL PIPELINE ───────────────────────────────
-- Active customers on lower plans with high engagement = upsell ready
SELECT
    customer_id,
    plan,
    apps_used,
    monthly_logins,
    seats,
    tenure_months,
    monthly_price * seats                                       AS current_mrr,
    (105 - monthly_price) * seats                              AS upsell_mrr_potential
FROM zoho_customers
WHERE
    churned = 0
    AND plan != 'Zoho One'
    AND apps_used >= 4
    AND monthly_logins >= 15
    AND tenure_months >= 6
ORDER BY upsell_mrr_potential DESC
LIMIT 100;


-- ── 9. CHURN PATTERN: SINGLE-APP vs MULTI-APP vs ZOHO ONE ────
SELECT
    CASE
        WHEN apps_used = 1          THEN 'Single-app (1)'
        WHEN apps_used BETWEEN 2 AND 4 THEN 'Multi-app (2–4)'
        WHEN apps_used >= 5
             AND plan != 'Zoho One' THEN 'Power user (5+)'
        ELSE 'Zoho One'
    END AS customer_segment,
    COUNT(*)                                                    AS customers,
    ROUND(100.0 * SUM(churned) / COUNT(*), 2)                  AS churn_rate_pct,
    ROUND(AVG(tenure_months), 1)                               AS avg_tenure,
    ROUND(AVG(monthly_price * seats), 2)                       AS avg_monthly_revenue
FROM zoho_customers
GROUP BY 1
ORDER BY churn_rate_pct DESC;


-- ── 10. BUSINESS IMPACT SUMMARY ──────────────────────────────
SELECT
    'Total customers'                                           AS metric,
    COUNT(*)::TEXT                                             AS value
FROM zoho_customers
UNION ALL
SELECT 'Churned customers',        SUM(churned)::TEXT
FROM zoho_customers
UNION ALL
SELECT 'Overall churn rate (%)',   ROUND(100.0*SUM(churned)/COUNT(*),2)::TEXT
FROM zoho_customers
UNION ALL
SELECT 'Total active MRR ($)',     ROUND(SUM(CASE WHEN churned=0
                                    THEN monthly_price*seats ELSE 0 END),0)::TEXT
FROM zoho_customers
UNION ALL
SELECT 'Avg LTV per customer ($)', ROUND(AVG(monthly_price*seats*tenure_months),0)::TEXT
FROM zoho_customers
UNION ALL
SELECT 'Zoho One customers (%)',   ROUND(100.0*SUM(CASE WHEN plan='Zoho One'
                                    THEN 1 ELSE 0 END)/COUNT(*),1)::TEXT
FROM zoho_customers;
