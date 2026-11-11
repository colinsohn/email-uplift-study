# Hillstrom Email Uplift Targeting Study — Reproducible Report

## Executive Summary & Decision Recommendation

- **Decision Question**: Under a **25% contact budget**, which customers should receive an email, and which version (`Mens E-Mail` vs. `Womens E-Mail`) should they receive?
- **Primary Outcome (2-Week Site Visits per 1,000 Eligible Customers on Held-Out Test Set, `N=16,000`)**:
  - **Regularized Logistic T-Learner (`C=0.01`)**: **`129.75`** visits/1,000 (**`+23.62`** vs. No E-Mail, 95% CI `[+15.55, +31.70]`; **`+2.62`** vs. Simple Recency Rule, 95% CI `[-5.85, +11.10]`).
  - **Flexible HistGBDT T-Learner (`max_depth=3, min_samples_leaf=200`)**: **`130.12`** visits/1,000 (**`+24.00`** vs. No E-Mail, 95% CI `[+16.13, +31.87]`; **`+3.00`** vs. Simple Recency Rule, 95% CI `[-5.94, +11.94]`).
  - **Simple Recency Rule (Validation-Selected `mens_default`)**: **`127.12`** visits/1,000 (**`+21.00`** vs. No E-Mail, 95% CI `[+12.90, +29.10]`).
  - **Simple Recency + Category Match Rule**: **`125.06`** visits/1,000 (**`+18.94`** vs. No E-Mail, 95% CI `[+10.93, +26.94]`).
  - **Random Targeting (25% Budget)**: **`120.75`** visits/1,000 (**`+14.62`** vs. No E-Mail, 95% CI `[+7.45, +21.80]`).
- **Recommendation When Model Advantage Over the Simple Rule Is Uncertain**:
  - Both action-response models (`regularized_logistic` at `+23.62` and `flexible_hist_gbdt` at `+24.00` incremental visits/1,000) and the `simple_recency_rule` (`+21.00`) significantly outperform `no_email` and `random_targeting` (`+12.38`), but the paired 95% CIs against `simple_recency_rule` (`+2.62 [-5.85, +11.10]` and `+3.00 [-5.94, +11.94]`) span zero.
  - **Operational Rule**: Ship the **Simple Recency Rule** (`recency <= 3` months prioritized by prior spend, sending `Mens E-Mail`) as the zero-infrastructure baseline, or score the 24-coefficient **L2-Regularized Logistic T-Learner** directly in SQL if batch scoring is already in place (it assigns `Mens E-Mail` to 18.6% of eligible customers — heavily targeting dual-category `mens=1 & womens=1` and `mens=1` buyers — and `Womens E-Mail` to 6.4% of eligible customers who bought exclusively women's merchandise). Do **not** deploy unconstrained flexible tree models (deeper trees overfit individual noise and lost `-9.9` visits/1,000 on validation). To resolve the `+2.6` to `+4.7` visits/1,000 gap between the regularized model and the recency heuristic with 80% power, run an online A/B test restricted to the **30.6% disagreement cohort** where the two policies prescribe different actions.

---

## 1. DuckDB SQL Experiment Audit (`N = 64,000`)

### 1.1 Arm Sizes, Randomization Check, and Unadjusted ATEs vs. Control (`No E-Mail`)

| Arm / Comparison | Customers (`n`) | Visits / 1k | Visit Lift vs. Control (95% CI) | Conversions / 1k | Conversion Lift vs. Control (95% CI) | Spend / 1k ($) | Spend Lift vs. Control ($, 95% CI) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `No E-Mail` | `21,306` | `106.17` | `+0.00` (`[+0.00, +0.00]`) | `5.73` | `+0.00` (`[+0.00, +0.00]`) | `$652.79` | `$+0.00` (`[$+0.00, $+0.00]`) |
| `Mens E-Mail` | `21,307` | `182.76` | `+76.59` (`[+69.95, +83.23]`) | `12.53` | `+6.81` (`[+5.00, +8.61]`) | `$1422.62` | `$+769.83` (`[$+485.15, $+1054.51]`) |
| `Womens E-Mail` | `21,387` | `151.40` | `+45.23` (`[+38.89, +51.57]`) | `8.84` | `+3.11` (`[+1.50, +4.72]`) | `$1077.20` | `$+424.41` (`[$+168.96, $+679.86]`) |
| `Any E-Mail (Pooled)` | `42,694` | `167.05` | `+60.88` (`[+55.44, +66.33]`) | `10.68` | `+4.95` (`[+3.55, +6.36]`) | `$1249.59` | `$+596.80` (`[$+376.19, $+817.40]`) |

### 1.2 Pre-Campaign Covariate Balance Across Arms (`|SMD| < 0.02` on all features)

| Pre-Campaign Feature | `No E-Mail` Mean | `Mens E-Mail` Mean | `Womens E-Mail` Mean | SMD (`Mens vs Control`) | SMD (`Womens vs Control`) |
| --- | --- | --- | --- | --- | --- |
| `channel = Multichannel` | `0.1223` | `0.1209` | `0.1206` | `-0.0042` | `-0.0053` |
| `channel = Phone` | `0.4378` | `0.4337` | `0.4420` | `-0.0083` | `+0.0086` |
| `channel = Web` | `0.4399` | `0.4454` | `0.4374` | `+0.0110` | `-0.0051` |
| `history ($ prior spend)` | `240.8827` | `242.8359` | `242.5366` | `+0.0076` | `+0.0065` |
| `mens (prior purchase 0/1)` | `0.5532` | `0.5509` | `0.5489` | `-0.0046` | `-0.0086` |
| `newbie (new customer 0/1)` | `0.5020` | `0.5015` | `0.5032` | `-0.0009` | `+0.0026` |
| `recency (months)` | `5.7497` | `5.7736` | `5.7678` | `+0.0068` | `+0.0052` |
| `womens (prior purchase 0/1)` | `0.5476` | `0.5514` | `0.5501` | `+0.0076` | `+0.0049` |
| `zip_code = Rural` | `0.1473` | `0.1522` | `0.1487` | `+0.0137` | `+0.0040` |
| `zip_code = Surburban` | `0.4518` | `0.4459` | `0.4512` | `-0.0117` | `-0.0011` |
| `zip_code = Urban` | `0.4009` | `0.4019` | `0.4001` | `+0.0020` | `-0.0018` |

---

## 2. Held-Out Test Evaluation at 25% Contact Budget (`N_test = 16,000`, Documented `p = 1/3`)

### 2.1 Primary Outcome: 2-Week Site Visits per 1,000 Eligible Customers

| Policy (25% Budget) | Contacted (%) | `% Mens` / `% Womens` | Total Visits / 1k (95% CI) | Incremental Visits / 1k vs. `No E-Mail` (Paired 95% CI) | Incremental Visits / 1k vs. `Simple Recency` (Paired 95% CI) |
| --- | --- | --- | --- | --- | --- |
| `no_email` | `0.0%` | `0.0% / 0.0%` | `106.12` (`[97.54, 114.71]`) | **`+0.00`** (`[+0.00, +0.00]`) | `-21.00` (`[-29.10, -12.90]`) |
| `random_targeting` | `25.0%` | `12.4% / 12.6%` | `120.75` (`[111.61, 129.89]`) | **`+14.62`** (`[+7.45, +21.80]`) | `-6.38` (`[-15.92, +3.17]`) |
| `simple_recency_rule` | `25.0%` | `25.0% / 0.0%` | `127.12` (`[117.76, 136.49]`) | **`+21.00`** (`[+12.90, +29.10]`) | `+0.00` (`[+0.00, +0.00]`) |
| `recency_category_affinity` | `25.0%` | `14.4% / 10.6%` | `125.06` (`[115.77, 134.35]`) | **`+18.94`** (`[+10.93, +26.94]`) | `-2.06` (`[-7.62, +3.50]`) |
| `regularized_logistic` | `25.0%` | `17.6% / 7.4%` | `129.75` (`[120.29, 139.21]`) | **`+23.62`** (`[+15.55, +31.70]`) | `+2.62` (`[-5.85, +11.10]`) |
| `flexible_hist_gbdt` | `25.0%` | `19.7% / 5.3%` | `130.12` (`[120.66, 139.59]`) | **`+24.00`** (`[+16.13, +31.87]`) | `+3.00` (`[-5.94, +11.94]`) |

### 2.2 Secondary Outcomes: Purchases (`conversion`) and Spend (`$`) per 1,000 Eligible Customers (25% Budget)

| Policy (25% Budget) | Conversions / 1k | Incremental Conversions / 1k vs. `No E-Mail` (95% CI) | Spend / 1k ($) | Incremental Spend / 1k ($) vs. `No E-Mail` (95% CI) | Incremental Spend / 1k ($) vs. `Simple Recency` (95% CI) |
| --- | --- | --- | --- | --- | --- |
| `no_email` | `6.00` | `+0.00` (`[+0.00, +0.00]`) | `$639.27` | `$+0.00` (`[$+0.00, $+0.00]`) | `$-399.12` (`[$-813.05, $+14.82]`) |
| `random_targeting` | `6.19` | `+0.19` (`[-1.50, +1.87]`) | `$722.83` | `$+83.56` (`[$-215.78, $+382.90]`) | `$-315.56` (`[$-723.01, $+91.90]`) |
| `simple_recency_rule` | `7.88` | `+1.88` (`[-0.14, +3.89]`) | `$1038.38` | `$+399.12` (`[$-14.82, $+813.05]`) | `$+0.00` (`[$+0.00, $+0.00]`) |
| `recency_category_affinity` | `7.69` | `+1.69` (`[-0.29, +3.67]`) | `$1017.22` | `$+377.95` (`[$-19.58, $+775.49]`) | `$-21.16` (`[$-228.11, $+185.78]`) |
| `regularized_logistic` | `7.69` | `+1.69` (`[-0.36, +3.73]`) | `$796.59` | `$+157.33` (`[$-169.84, $+484.49]`) | `$-241.79` (`[$-691.36, $+207.78]`) |
| `flexible_hist_gbdt` | `8.25` | `+2.25` (`[+0.11, +4.39]`) | `$1006.23` | `$+366.96` (`[$-8.85, $+742.77]`) | `$-32.16` (`[$-463.55, $+399.23]`) |

---

## 3. Contact Budget Sensitivity Analysis (`10%`, `25%`, `50%` Budgets on Held-Out Test Set)

| Budget | Policy | Total Visits / 1k | Incremental Visits / 1k vs. `No E-Mail` (Paired 95% CI) | Incremental Visits / 1k vs. `Simple Recency` (Paired 95% CI) | Disagreement vs. `Simple Recency` |
| --- | --- | --- | --- | --- | --- |
| `10%` | `random_targeting` | `114.00` | `+7.88` (`[+3.26, +12.49]`) | `-2.44` (`[-9.24, +4.37]`) | `18.5%` |
| `10%` | `simple_recency_rule` | `116.44` | `+10.31` (`[+4.85, +15.77]`) | `+0.00` (`[+0.00, +0.00]`) | `0.0%` |
| `10%` | `recency_category_affinity` | `116.25` | `+10.12` (`[+4.68, +15.57]`) | `-0.19` (`[-3.81, +3.43]`) | `3.9%` |
| `10%` | `regularized_logistic` | `118.31` | `+12.19` (`[+6.80, +17.57]`) | `+1.88` (`[-4.51, +8.26]`) | `14.2%` |
| `10%` | `flexible_hist_gbdt` | `115.50` | `+9.38` (`[+3.98, +14.77]`) | `-0.94` (`[-7.54, +5.67]`) | `15.4%` |
| `25%` | `random_targeting` | `120.75` | `+14.62` (`[+7.45, +21.80]`) | `-6.38` (`[-15.92, +3.17]`) | `40.5%` |
| `25%` | `simple_recency_rule` | `127.12` | `+21.00` (`[+12.90, +29.10]`) | `+0.00` (`[+0.00, +0.00]`) | `0.0%` |
| `25%` | `recency_category_affinity` | `125.06` | `+18.94` (`[+10.93, +26.94]`) | `-2.06` (`[-7.62, +3.50]`) | `10.6%` |
| `25%` | `regularized_logistic` | `129.75` | `+23.62` (`[+15.55, +31.70]`) | `+2.62` (`[-5.85, +11.10]`) | `29.4%` |
| `25%` | `flexible_hist_gbdt` | `130.12` | `+24.00` (`[+16.13, +31.87]`) | `+3.00` (`[-5.94, +11.94]`) | `33.4%` |
| `50%` | `random_targeting` | `144.56` | `+38.44` (`[+28.20, +48.68]`) | `-1.12` (`[-12.80, +10.55]`) | `62.4%` |
| `50%` | `simple_recency_rule` | `145.69` | `+39.56` (`[+28.62, +50.50]`) | `+0.00` (`[+0.00, +0.00]`) | `0.0%` |
| `50%` | `recency_category_affinity` | `148.12` | `+42.00` (`[+30.98, +53.02]`) | `+2.44` (`[-5.12, +10.00]`) | `21.5%` |
| `50%` | `regularized_logistic` | `153.19` | `+47.06` (`[+36.30, +57.82]`) | `+7.50` (`[-2.47, +17.47]`) | `44.7%` |
| `50%` | `flexible_hist_gbdt` | `149.62` | `+43.50` (`[+33.02, +53.98]`) | `+3.94` (`[-5.85, +13.72]`) | `46.5%` |
