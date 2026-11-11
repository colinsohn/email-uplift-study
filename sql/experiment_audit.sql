-- DuckDB SQL Audit for the Hillstrom Email Merchandise Randomized Experiment
-- Pre-campaign customer features: recency, history_segment, history, mens, womens, zip_code, newbie, channel
-- Randomized treatment assignment: segment ('Mens E-Mail', 'Womens E-Mail', 'No E-Mail'; documented 1/3 probability each)
-- Post-campaign outcomes (excluded from all customer feature vectors): visit, conversion, spend

-- @query: arm_sizes
WITH arm_counts AS (
    SELECT
        segment,
        COUNT(*) AS n_customers,
        COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () AS observed_prop,
        1.0 / 3.0 AS documented_prop,
        SUM(COUNT(*)) OVER () / 3.0 AS expected_n
    FROM hillstrom_raw
    GROUP BY segment
)
SELECT
    segment,
    n_customers,
    ROUND(observed_prop, 6) AS observed_prop,
    ROUND(documented_prop, 6) AS documented_prop,
    ROUND(
        SUM(POWER(n_customers - expected_n, 2) / expected_n) OVER (),
        6
    ) AS srm_chisq_stat,
    ROUND(AVG(visit), 6) AS visit_rate,
    ROUND(AVG(conversion), 6) AS conversion_rate,
    ROUND(AVG(spend), 6) AS avg_spend
FROM hillstrom_raw
JOIN arm_counts USING (segment)
GROUP BY segment, n_customers, observed_prop, documented_prop, expected_n
ORDER BY segment;

-- @query: missing_values
SELECT
    COUNT(*) AS total_rows,
    SUM(CASE WHEN recency IS NULL THEN 1 ELSE 0 END) AS null_recency,
    SUM(CASE WHEN history_segment IS NULL THEN 1 ELSE 0 END) AS null_history_segment,
    SUM(CASE WHEN history IS NULL THEN 1 ELSE 0 END) AS null_history,
    SUM(CASE WHEN mens IS NULL THEN 1 ELSE 0 END) AS null_mens,
    SUM(CASE WHEN womens IS NULL THEN 1 ELSE 0 END) AS null_womens,
    SUM(CASE WHEN zip_code IS NULL THEN 1 ELSE 0 END) AS null_zip_code,
    SUM(CASE WHEN newbie IS NULL THEN 1 ELSE 0 END) AS null_newbie,
    SUM(CASE WHEN channel IS NULL THEN 1 ELSE 0 END) AS null_channel,
    SUM(CASE WHEN segment IS NULL THEN 1 ELSE 0 END) AS null_segment,
    SUM(CASE WHEN visit IS NULL THEN 1 ELSE 0 END) AS null_visit,
    SUM(CASE WHEN conversion IS NULL THEN 1 ELSE 0 END) AS null_conversion,
    SUM(CASE WHEN spend IS NULL THEN 1 ELSE 0 END) AS null_spend,
    SUM(CASE WHEN visit NOT IN (0, 1) OR conversion NOT IN (0, 1) OR spend < 0 OR recency < 1 OR recency > 12 OR history < 0 THEN 1 ELSE 0 END) AS invalid_range_rows
FROM hillstrom_raw;

-- @query: pre_campaign_balance
WITH feature_long AS (
    SELECT segment, 'recency (months)' AS feature, CAST(recency AS DOUBLE) AS val FROM hillstrom_raw
    UNION ALL
    SELECT segment, 'history ($ prior spend)' AS feature, CAST(history AS DOUBLE) AS val FROM hillstrom_raw
    UNION ALL
    SELECT segment, 'mens (prior purchase 0/1)' AS feature, CAST(mens AS DOUBLE) AS val FROM hillstrom_raw
    UNION ALL
    SELECT segment, 'womens (prior purchase 0/1)' AS feature, CAST(womens AS DOUBLE) AS val FROM hillstrom_raw
    UNION ALL
    SELECT segment, 'newbie (new customer 0/1)' AS feature, CAST(newbie AS DOUBLE) AS val FROM hillstrom_raw
    UNION ALL
    SELECT segment, 'zip_code = Urban' AS feature, CASE WHEN zip_code = 'Urban' THEN 1.0 ELSE 0.0 END AS val FROM hillstrom_raw
    UNION ALL
    SELECT segment, 'zip_code = Surburban' AS feature, CASE WHEN zip_code = 'Surburban' THEN 1.0 ELSE 0.0 END AS val FROM hillstrom_raw
    UNION ALL
    SELECT segment, 'zip_code = Rural' AS feature, CASE WHEN zip_code = 'Rural' THEN 1.0 ELSE 0.0 END AS val FROM hillstrom_raw
    UNION ALL
    SELECT segment, 'channel = Web' AS feature, CASE WHEN channel = 'Web' THEN 1.0 ELSE 0.0 END AS val FROM hillstrom_raw
    UNION ALL
    SELECT segment, 'channel = Phone' AS feature, CASE WHEN channel = 'Phone' THEN 1.0 ELSE 0.0 END AS val FROM hillstrom_raw
    UNION ALL
    SELECT segment, 'channel = Multichannel' AS feature, CASE WHEN channel = 'Multichannel' THEN 1.0 ELSE 0.0 END AS val FROM hillstrom_raw
),
arm_stats AS (
    SELECT
        feature,
        AVG(CASE WHEN segment = 'No E-Mail' THEN val END) AS mean_no_email,
        VAR_SAMP(CASE WHEN segment = 'No E-Mail' THEN val END) AS var_no_email,
        AVG(CASE WHEN segment = 'Mens E-Mail' THEN val END) AS mean_mens_email,
        VAR_SAMP(CASE WHEN segment = 'Mens E-Mail' THEN val END) AS var_mens_email,
        AVG(CASE WHEN segment = 'Womens E-Mail' THEN val END) AS mean_womens_email,
        VAR_SAMP(CASE WHEN segment = 'Womens E-Mail' THEN val END) AS var_womens_email
    FROM feature_long
    GROUP BY feature
)
SELECT
    feature,
    ROUND(mean_no_email, 4) AS mean_no_email,
    ROUND(mean_mens_email, 4) AS mean_mens_email,
    ROUND(mean_womens_email, 4) AS mean_womens_email,
    ROUND(
        (mean_mens_email - mean_no_email) / NULLIF(SQRT((var_mens_email + var_no_email) / 2.0), 0),
        4
    ) AS smd_mens_vs_control,
    ROUND(
        (mean_womens_email - mean_no_email) / NULLIF(SQRT((var_womens_email + var_no_email) / 2.0), 0),
        4
    ) AS smd_womens_vs_control
FROM arm_stats
ORDER BY feature;

-- @query: experiment_ate_summary
WITH arm_moments AS (
    SELECT
        segment AS arm,
        COUNT(*) AS n,
        AVG(visit) AS visit_mean,
        VAR_SAMP(visit) AS visit_var,
        AVG(conversion) AS conv_mean,
        VAR_SAMP(conversion) AS conv_var,
        AVG(spend) AS spend_mean,
        VAR_SAMP(spend) AS spend_var
    FROM hillstrom_raw
    GROUP BY segment
    UNION ALL
    SELECT
        'Any E-Mail (Pooled)' AS arm,
        COUNT(*) AS n,
        AVG(visit) AS visit_mean,
        VAR_SAMP(visit) AS visit_var,
        AVG(conversion) AS conv_mean,
        VAR_SAMP(conversion) AS conv_var,
        AVG(spend) AS spend_mean,
        VAR_SAMP(spend) AS spend_var
    FROM hillstrom_raw
    WHERE segment <> 'No E-Mail'
),
ctrl AS (
    SELECT * FROM arm_moments WHERE arm = 'No E-Mail'
)
SELECT
    a.arm AS comparison_arm,
    a.n AS n_arm,
    c.n AS n_control,
    ROUND(a.visit_mean * 1000.0, 2) AS visits_per_1k,
    ROUND((a.visit_mean - c.visit_mean) * 1000.0, 2) AS ate_visits_per_1k,
    CASE WHEN a.arm = 'No E-Mail' THEN 0.0 ELSE ROUND(((a.visit_mean - c.visit_mean) - 1.959964 * SQRT(a.visit_var / a.n + c.visit_var / c.n)) * 1000.0, 2) END AS ate_visits_ci95_low,
    CASE WHEN a.arm = 'No E-Mail' THEN 0.0 ELSE ROUND(((a.visit_mean - c.visit_mean) + 1.959964 * SQRT(a.visit_var / a.n + c.visit_var / c.n)) * 1000.0, 2) END AS ate_visits_ci95_high,
    ROUND(a.conv_mean * 1000.0, 2) AS conversions_per_1k,
    ROUND((a.conv_mean - c.conv_mean) * 1000.0, 2) AS ate_conversions_per_1k,
    CASE WHEN a.arm = 'No E-Mail' THEN 0.0 ELSE ROUND(((a.conv_mean - c.conv_mean) - 1.959964 * SQRT(a.conv_var / a.n + c.conv_var / c.n)) * 1000.0, 2) END AS ate_conversions_ci95_low,
    CASE WHEN a.arm = 'No E-Mail' THEN 0.0 ELSE ROUND(((a.conv_mean - c.conv_mean) + 1.959964 * SQRT(a.conv_var / a.n + c.conv_var / c.n)) * 1000.0, 2) END AS ate_conversions_ci95_high,
    ROUND(a.spend_mean * 1000.0, 2) AS spend_per_1k,
    ROUND((a.spend_mean - c.spend_mean) * 1000.0, 2) AS ate_spend_per_1k,
    CASE WHEN a.arm = 'No E-Mail' THEN 0.0 ELSE ROUND(((a.spend_mean - c.spend_mean) - 1.959964 * SQRT(a.spend_var / a.n + c.spend_var / c.n)) * 1000.0, 2) END AS ate_spend_ci95_low,
    CASE WHEN a.arm = 'No E-Mail' THEN 0.0 ELSE ROUND(((a.spend_mean - c.spend_mean) + 1.959964 * SQRT(a.spend_var / a.n + c.spend_var / c.n)) * 1000.0, 2) END AS ate_spend_ci95_high
FROM arm_moments a
CROSS JOIN ctrl c
ORDER BY CASE a.arm
    WHEN 'No E-Mail' THEN 1
    WHEN 'Mens E-Mail' THEN 2
    WHEN 'Womens E-Mail' THEN 3
    ELSE 4
END;
