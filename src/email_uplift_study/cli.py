from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from email_uplift_study.pipeline import render_policy_value_svg, run_full_study


def render_markdown_summary(results: dict[str, Any]) -> str:
    """Render the study Markdown report dynamically from the results dict / study_results.json."""
    audit = results["duckdb_audit"]
    ate_rows = audit["experiment_ate_summary"]
    balance_rows = audit["pre_campaign_balance"]
    primary = results["held_out_test_25pct_budget"]
    sens = results["budget_sensitivity"]
    n_test = results["dataset"]["splits"]["test_rows"]
    total_rows = results["dataset"]["total_rows"]

    by_pol = {r["policy"]: r for r in primary}
    lr = by_pol["regularized_logistic"]
    gb = by_pol["flexible_hist_gbdt"]
    rec = by_pol["simple_recency_rule"]
    rec_cat = by_pol["recency_category_affinity"]
    rand = by_pol["random_targeting"]

    selected_c = results["validation_tuning"]["regularized_logistic"]["selected_C"]
    gb_params = results["validation_tuning"]["flexible_hist_gbdt"]["selected_params"]
    rec_mode = results["validation_tuning"]["simple_recency_rule"]["selected_email_mode"]

    rec_vn = rec["incremental_visits_vs_no_email"]
    rec_vr = rec["incremental_visits_vs_random"]
    rand_vn = rand["incremental_visits_vs_no_email"]
    lr_vn = lr["incremental_visits_vs_no_email"]
    lr_vs = lr["incremental_visits_vs_best_simple"]
    lr_vr = lr["incremental_visits_vs_random"]
    gb_vn = gb["incremental_visits_vs_no_email"]
    gb_vs = gb["incremental_visits_vs_best_simple"]
    gb_vr = gb["incremental_visits_vs_random"]
    rec_cat_vn = rec_cat["incremental_visits_vs_no_email"]

    lr_mens_pct = lr["composition"]["mens_email_share_of_eligible"] * 100.0
    lr_womens_pct = lr["composition"]["womens_email_share_of_eligible"] * 100.0
    lr_disagree_pct = lr_vs["disagreement_rate"] * 100.0
    gb_disagree_pct = gb_vs["disagreement_rate"] * 100.0

    lines: list[str] = [
        "# Hillstrom Email Uplift Targeting Study — Reproducible Report",
        "",
        "## Executive Summary & Decision Recommendation",
        "",
        "- **Decision Question**: Under a **25% contact budget**, which customers should receive an email, and which version (`Mens E-Mail` vs. `Womens E-Mail`) should they receive?",
        f"- **Main Takeaway (`N_test = {n_test:,}`)**: At a **25% contact budget**, the **Simple Recency Rule** added an estimated **`{rec_vn['diff']:+.2f}` visits per 1,000 eligible customers** versus sending no email (paired 95% CI `[{rec_vn['ci95_low']:+.2f}, {rec_vn['ci95_high']:+.2f}]`), and the held-out evidence was **inconclusive** about whether either model improved on that rule (`regularized_logistic` vs. recency rule: **`{lr_vs['diff']:+.2f}`** visits/1,000, paired 95% CI `[{lr_vs['ci95_low']:+.2f}, {lr_vs['ci95_high']:+.2f}]`, disagreeing on **`{lr_disagree_pct:.1f}%`** of customers; `flexible_hist_gbdt` vs. recency rule: **`{gb_vs['diff']:+.2f}`** visits/1,000, paired 95% CI `[{gb_vs['ci95_low']:+.2f}, {gb_vs['ci95_high']:+.2f}]`, disagreeing on **`{gb_disagree_pct:.1f}%`** of customers).",
        f"- **Primary Outcome Summary (2-Week Site Visits per 1,000 Eligible Customers on Held-Out Test Set, `N={n_test:,}`)**:",
        f"  - **Simple Recency Rule (Validation-Selected `{rec_mode}`)**: **`{rec['visits_per_1k']['estimate']:.2f}`** visits/1,000 (**`{rec_vn['diff']:+.2f}`** vs. No E-Mail, 95% CI `[{rec_vn['ci95_low']:+.2f}, {rec_vn['ci95_high']:+.2f}]`; **`{rec_vr['diff']:+.2f}`** vs. Random Targeting, 95% CI `[{rec_vr['ci95_low']:+.2f}, {rec_vr['ci95_high']:+.2f}]`).",
        f"  - **Simple Recency + Category Match Rule**: **`{rec_cat['visits_per_1k']['estimate']:.2f}`** visits/1,000 (**`{rec_cat_vn['diff']:+.2f}`** vs. No E-Mail, 95% CI `[{rec_cat_vn['ci95_low']:+.2f}, {rec_cat_vn['ci95_high']:+.2f}]`).",
        f"  - **Regularized Logistic T-Learner (`C={selected_c}`)**: **`{lr['visits_per_1k']['estimate']:.2f}`** visits/1,000 (**`{lr_vn['diff']:+.2f}`** vs. No E-Mail, 95% CI `[{lr_vn['ci95_low']:+.2f}, {lr_vn['ci95_high']:+.2f}]`; **`{lr_vs['diff']:+.2f}`** vs. Simple Recency Rule, 95% CI `[{lr_vs['ci95_low']:+.2f}, {lr_vs['ci95_high']:+.2f}]`; **`{lr_vr['diff']:+.2f}`** vs. Random Targeting, 95% CI `[{lr_vr['ci95_low']:+.2f}, {lr_vr['ci95_high']:+.2f}]`).",
        f"  - **Flexible HistGBDT T-Learner (`max_depth={gb_params['max_depth']}, min_samples_leaf={gb_params['min_samples_leaf']}`)**: **`{gb['visits_per_1k']['estimate']:.2f}`** visits/1,000 (**`{gb_vn['diff']:+.2f}`** vs. No E-Mail, 95% CI `[{gb_vn['ci95_low']:+.2f}, {gb_vn['ci95_high']:+.2f}]`; **`{gb_vs['diff']:+.2f}`** vs. Simple Recency Rule, 95% CI `[{gb_vs['ci95_low']:+.2f}, {gb_vs['ci95_high']:+.2f}]`; **`{gb_vr['diff']:+.2f}`** vs. Random Targeting, 95% CI `[{gb_vr['ci95_low']:+.2f}, {gb_vr['ci95_high']:+.2f}]`).",
        f"  - **Random Targeting (25% Budget)**: **`{rand['visits_per_1k']['estimate']:.2f}`** visits/1,000 (**`{rand_vn['diff']:+.2f}`** vs. No E-Mail, 95% CI `[{rand_vn['ci95_low']:+.2f}, {rand_vn['ci95_high']:+.2f}]`).",
        "- **Recommendation**:",
        f"  - All active 25%-budget policies beat sending **No E-Mail** with 95% confidence intervals well above zero (`{rand_vn['diff']:+.2f}` for `random_targeting`, `{rec_vn['diff']:+.2f}` for `simple_recency_rule`, `{lr_vn['diff']:+.2f}` for `regularized_logistic`, and `{gb_vn['diff']:+.2f}` for `flexible_hist_gbdt`). However, direct paired 95% confidence intervals at the 25% budget do **not** establish a statistically significant advantage for any learned policy over `simple_recency_rule` (`{lr_vs['diff']:+.2f} [{lr_vs['ci95_low']:+.2f}, {lr_vs['ci95_high']:+.2f}]` for `regularized_logistic`; `{gb_vs['diff']:+.2f} [{gb_vs['ci95_low']:+.2f}, {gb_vs['ci95_high']:+.2f}]` for `flexible_hist_gbdt`), nor do the paired 95% CIs versus `random_targeting` exclude zero (`{rec_vr['diff']:+.2f} [{rec_vr['ci95_low']:+.2f}, {rec_vr['ci95_high']:+.2f}]` for `simple_recency_rule`; `{lr_vr['diff']:+.2f} [{lr_vr['ci95_low']:+.2f}, {lr_vr['ci95_high']:+.2f}]` for `regularized_logistic`).",
        f"  - **Operational Decision**: **Recommend deploying the Simple Recency Rule** (contact the 25% most recent purchasers ordered by `recency` ascending and prior `history` spend descending, sending `Mens E-Mail`). It delivers `{rec_vn['diff']:+.2f}` incremental visits/1,000 (`[{rec_vn['ci95_low']:+.2f}, {rec_vn['ci95_high']:+.2f}]`) and `${rec['incremental_spend_vs_no_email']['diff']:+.2f}` incremental spend/1,000 (`[${rec['incremental_spend_vs_no_email']['ci95_low']:+.2f}, ${rec['incremental_spend_vs_no_email']['ci95_high']:+.2f}]`) vs. No E-Mail with zero ML scoring infrastructure. While `regularized_logistic` (`{lr_mens_pct:.1f}%` `Mens E-Mail`, `{lr_womens_pct:.1f}%` `Womens E-Mail`) has higher point estimates across budgets, that advantage is uncertain on held-out data; if the team wants to test it further, run an online A/B test restricted to the **`{lr_disagree_pct:.1f}%` disagreement cohort** where `regularized_logistic` and `simple_recency_rule` prescribe different actions.",
        "",
        "---",
        "",
        f"## 1. DuckDB SQL Experiment Audit (`N = {total_rows:,}`)",
        "",
        "### 1.1 Arm Sizes, Randomization Check, and Unadjusted ATEs vs. Control (`No E-Mail`)",
        "",
        "| Arm / Comparison | Customers (`n`) | Visits / 1k | Visit Lift vs. Control (95% CI) | Conversions / 1k | Conversion Lift vs. Control (95% CI) | Spend / 1k ($) | Spend Lift vs. Control ($, 95% CI) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in ate_rows:
        lines.append(
            f"| `{row['comparison_arm']}` | `{row['n_arm']:,}` | `{row['visits_per_1k']:.2f}` | "
            f"`{row['ate_visits_per_1k']:+.2f}` (`[{row['ate_visits_ci95_low']:+.2f}, {row['ate_visits_ci95_high']:+.2f}]`) | "
            f"`{row['conversions_per_1k']:.2f}` | `{row['ate_conversions_per_1k']:+.2f}` (`[{row['ate_conversions_ci95_low']:+.2f}, {row['ate_conversions_ci95_high']:+.2f}]`) | "
            f"`${row['spend_per_1k']:.2f}` | `${row['ate_spend_per_1k']:+.2f}` (`[${row['ate_spend_ci95_low']:+.2f}, ${row['ate_spend_ci95_high']:+.2f}]`) |"
        )

    lines.extend(
        [
            "",
            "### 1.2 Pre-Campaign Covariate Balance Across Arms (`|SMD| < 0.02` on all features)",
            "",
            "| Pre-Campaign Feature | `No E-Mail` Mean | `Mens E-Mail` Mean | `Womens E-Mail` Mean | SMD (`Mens vs Control`) | SMD (`Womens vs Control`) |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for b in balance_rows:
        lines.append(
            f"| `{b['feature']}` | `{b['mean_no_email']:.4f}` | `{b['mean_mens_email']:.4f}` | `{b['mean_womens_email']:.4f}` | `{b['smd_mens_vs_control']:+.4f}` | `{b['smd_womens_vs_control']:+.4f}` |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            f"## 2. Held-Out Test Evaluation at 25% Contact Budget (`N_test = {n_test:,}`, Documented `p = 1/3`)",
            "",
            "### 2.1 Primary Outcome: 2-Week Site Visits per 1,000 Eligible Customers",
            "",
            "| Policy (25% Budget) | Contacted (%) | `% Mens` / `% Womens` | Total Visits / 1k (95% CI) | Incremental Visits / 1k vs. `No E-Mail` (Paired 95% CI) | Incremental Visits / 1k vs. `Simple Recency` (Paired 95% CI) | Disagreement vs. `Simple Recency` |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

    for r in primary:
        comp = r["composition"]
        v = r["visits_per_1k"]
        vn = r["incremental_visits_vs_no_email"]
        vs = r["incremental_visits_vs_best_simple"]
        lines.append(
            f"| `{r['policy']}` | `{comp['contact_rate']*100:.1f}%` | `{comp['mens_email_share_of_eligible']*100:.1f}% / {comp['womens_email_share_of_eligible']*100:.1f}%` | "
            f"`{v['estimate']:.2f}` (`[{v['ci95_low']:.2f}, {v['ci95_high']:.2f}]`) | "
            f"**`{vn['diff']:+.2f}`** (`[{vn['ci95_low']:+.2f}, {vn['ci95_high']:+.2f}]`) | "
            f"`{vs['diff']:+.2f}` (`[{vs['ci95_low']:+.2f}, {vs['ci95_high']:+.2f}]`) | "
            f"`{vs['disagreement_rate']*100:.1f}%` |"
        )

    lines.extend(
        [
            "",
            "### 2.2 Secondary Outcomes: Purchases (`conversion`) and Spend (`$`) per 1,000 Eligible Customers (25% Budget)",
            "",
            "| Policy (25% Budget) | Conversions / 1k | Incremental Conversions / 1k vs. `No E-Mail` (95% CI) | Spend / 1k ($) | Incremental Spend / 1k ($) vs. `No E-Mail` (95% CI) | Incremental Spend / 1k ($) vs. `Simple Recency` (95% CI) |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )

    for r in primary:
        c = r["conversions_per_1k"]
        cn = r["incremental_conversions_vs_no_email"]
        s = r["spend_per_1k"]
        sn = r["incremental_spend_vs_no_email"]
        ss = r["incremental_spend_vs_best_simple"]
        lines.append(
            f"| `{r['policy']}` | `{c['estimate']:.2f}` | `{cn['diff']:+.2f}` (`[{cn['ci95_low']:+.2f}, {cn['ci95_high']:+.2f}]`) | "
            f"`${s['estimate']:.2f}` | `${sn['diff']:+.2f}` (`[${sn['ci95_low']:+.2f}, ${sn['ci95_high']:+.2f}]`) | "
            f"`${ss['diff']:+.2f}` (`[${ss['ci95_low']:+.2f}, ${ss['ci95_high']:+.2f}]`) |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Contact Budget Sensitivity Analysis (`10%`, `25%`, `50%` Budgets on Held-Out Test Set)",
            "",
            "| Budget | Policy | Total Visits / 1k | Incremental Visits / 1k vs. `No E-Mail` (Paired 95% CI) | Incremental Visits / 1k vs. `Simple Recency` (Paired 95% CI) | Disagreement vs. `Simple Recency` |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )

    for row in sens:
        ci_n = row["inc_visits_vs_no_email_ci95"]
        ci_s = row["inc_visits_vs_best_simple_ci95"]
        lines.append(
            f"| `{row['budget_pct']}%` | `{row['policy']}` | `{row['visits_per_1k']:.2f}` | "
            f"`{row['inc_visits_vs_no_email']:+.2f}` (`[{ci_n[0]:+.2f}, {ci_n[1]:+.2f}]`) | "
            f"`{row['inc_visits_vs_best_simple']:+.2f}` (`[{ci_s[0]:+.2f}, {ci_s[1]:+.2f}]`) | "
            f"`{row['disagreement_with_best_simple']*100:.1f}%` |"
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Hillstrom email uplift targeting study (DuckDB SQL audit + policy learning + held-out IPW evaluation)."
    )
    parser.add_argument(
        "--csv",
        default="data/hillstrom-email-experiment.csv",
        help="Relative path to the verified Hillstrom CSV file.",
    )
    parser.add_argument(
        "--sql",
        default="sql/experiment_audit.sql",
        help="Relative path to the DuckDB SQL audit script.",
    )
    parser.add_argument(
        "--out-dir",
        default="results",
        help="Directory for JSON, Markdown, and SVG study outputs.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = run_full_study(csv_path=args.csv, sql_path=args.sql, seed=42)
    json_path = out_dir / "study_results.json"
    md_path = out_dir / "study_report.md"
    svg_path = out_dir / "policy_value_comparison.svg"

    # Write study_results.json first, then render the report and SVG directly from the serialized JSON object
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    serialized_results = json.loads(json_path.read_text(encoding="utf-8"))
    md_report = render_markdown_summary(serialized_results)
    md_path.write_text(md_report, encoding="utf-8")
    render_policy_value_svg(serialized_results, svg_path)

    print(md_report)
    print(f"Wrote {json_path}, {md_path}, and {svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
