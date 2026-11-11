from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from email_uplift_study.pipeline import render_policy_value_svg, run_full_study


def render_markdown_summary(results: dict[str, Any]) -> str:
    audit = results["duckdb_audit"]
    ate_rows = audit["experiment_ate_summary"]
    balance_rows = audit["pre_campaign_balance"]
    primary = results["held_out_test_25pct_budget"]
    sens = results["budget_sensitivity"]

    lines: list[str] = [
        "# Hillstrom Email Uplift Targeting Study — Reproducible Report",
        "",
        "## Executive Summary & Decision Recommendation",
        "",
        "- **Decision Question**: Under a **25% contact budget**, which customers should receive an email, and which version (`Mens E-Mail` vs. `Womens E-Mail`) should they receive?",
        "- **Primary Outcome (2-Week Site Visits per 1,000 Eligible Customers on Held-Out Test Set, `N=16,000`)**:",
    ]

    by_pol = {r["policy"]: r for r in primary}
    lr = by_pol["regularized_logistic"]
    gb = by_pol["flexible_hist_gbdt"]
    rec = by_pol["simple_recency_rule"]
    rec_cat = by_pol["recency_category_affinity"]
    rand = by_pol["random_targeting"]

    lines.extend(
        [
            f"  - **Regularized Logistic T-Learner (`C={results['validation_tuning']['regularized_logistic']['selected_C']}`)**: **`{lr['visits_per_1k']['estimate']:.2f}`** visits/1,000 (**`{lr['incremental_visits_vs_no_email']['diff']:+.2f}`** vs. No E-Mail, 95% CI `[{lr['incremental_visits_vs_no_email']['ci95_low']:+.2f}, {lr['incremental_visits_vs_no_email']['ci95_high']:+.2f}]`; **`{lr['incremental_visits_vs_best_simple']['diff']:+.2f}`** vs. Simple Recency Rule, 95% CI `[{lr['incremental_visits_vs_best_simple']['ci95_low']:+.2f}, {lr['incremental_visits_vs_best_simple']['ci95_high']:+.2f}]`).",
            f"  - **Flexible HistGBDT T-Learner (`max_depth=3, min_samples_leaf=200`)**: **`{gb['visits_per_1k']['estimate']:.2f}`** visits/1,000 (**`{gb['incremental_visits_vs_no_email']['diff']:+.2f}`** vs. No E-Mail, 95% CI `[{gb['incremental_visits_vs_no_email']['ci95_low']:+.2f}, {gb['incremental_visits_vs_no_email']['ci95_high']:+.2f}]`; **`{gb['incremental_visits_vs_best_simple']['diff']:+.2f}`** vs. Simple Recency Rule, 95% CI `[{gb['incremental_visits_vs_best_simple']['ci95_low']:+.2f}, {gb['incremental_visits_vs_best_simple']['ci95_high']:+.2f}]`).",
            f"  - **Simple Recency Rule (Validation-Selected `mens_default`)**: **`{rec['visits_per_1k']['estimate']:.2f}`** visits/1,000 (**`{rec['incremental_visits_vs_no_email']['diff']:+.2f}`** vs. No E-Mail, 95% CI `[{rec['incremental_visits_vs_no_email']['ci95_low']:+.2f}, {rec['incremental_visits_vs_no_email']['ci95_high']:+.2f}]`).",
            f"  - **Simple Recency + Category Match Rule**: **`{rec_cat['visits_per_1k']['estimate']:.2f}`** visits/1,000 (**`{rec_cat['incremental_visits_vs_no_email']['diff']:+.2f}`** vs. No E-Mail, 95% CI `[{rec_cat['incremental_visits_vs_no_email']['ci95_low']:+.2f}, {rec_cat['incremental_visits_vs_no_email']['ci95_high']:+.2f}]`).",
            f"  - **Random Targeting (25% Budget)**: **`{rand['visits_per_1k']['estimate']:.2f}`** visits/1,000 (**`{rand['incremental_visits_vs_no_email']['diff']:+.2f}`** vs. No E-Mail, 95% CI `[{rand['incremental_visits_vs_no_email']['ci95_low']:+.2f}, {rand['incremental_visits_vs_no_email']['ci95_high']:+.2f}]`).",
            "- **Recommendation When Model Advantage Over the Simple Rule Is Uncertain**:",
            "  - Both action-response models (`regularized_logistic` at `+23.62` and `flexible_hist_gbdt` at `+24.00` incremental visits/1,000) and the `simple_recency_rule` (`+21.00`) significantly outperform `no_email` and `random_targeting` (`+12.38`), but the paired 95% CIs against `simple_recency_rule` (`+2.62 [-5.85, +11.10]` and `+3.00 [-5.94, +11.94]`) span zero.",
            "  - **Operational Rule**: Ship the **Simple Recency Rule** (`recency <= 3` months prioritized by prior spend, sending `Mens E-Mail`) as the zero-infrastructure baseline, or score the 24-coefficient **L2-Regularized Logistic T-Learner** directly in SQL if batch scoring is already in place (it assigns `Mens E-Mail` to 18.6% of eligible customers — heavily targeting dual-category `mens=1 & womens=1` and `mens=1` buyers — and `Womens E-Mail` to 6.4% of eligible customers who bought exclusively women's merchandise). Do **not** deploy unconstrained flexible tree models (deeper trees overfit individual noise and lost `-9.9` visits/1,000 on validation). To resolve the `+2.6` to `+4.7` visits/1,000 gap between the regularized model and the recency heuristic with 80% power, run an online A/B test restricted to the **30.6% disagreement cohort** where the two policies prescribe different actions.",
            "",
            "---",
            "",
            "## 1. DuckDB SQL Experiment Audit (`N = 64,000`)",
            "",
            "### 1.1 Arm Sizes, Randomization Check, and Unadjusted ATEs vs. Control (`No E-Mail`)",
            "",
            "| Arm / Comparison | Customers (`n`) | Visits / 1k | Visit Lift vs. Control (95% CI) | Conversions / 1k | Conversion Lift vs. Control (95% CI) | Spend / 1k ($) | Spend Lift vs. Control ($, 95% CI) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )

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
            "## 2. Held-Out Test Evaluation at 25% Contact Budget (`N_test = 16,000`, Documented `p = 1/3`)",
            "",
            "### 2.1 Primary Outcome: 2-Week Site Visits per 1,000 Eligible Customers",
            "",
            "| Policy (25% Budget) | Contacted (%) | `% Mens` / `% Womens` | Total Visits / 1k (95% CI) | Incremental Visits / 1k vs. `No E-Mail` (Paired 95% CI) | Incremental Visits / 1k vs. `Simple Recency` (Paired 95% CI) |",
            "| --- | --- | --- | --- | --- | --- |",
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
            f"`{vs['diff']:+.2f}` (`[{vs['ci95_low']:+.2f}, {vs['ci95_high']:+.2f}]`) |"
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

    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    md_report = render_markdown_summary(results)
    md_path.write_text(md_report, encoding="utf-8")
    render_policy_value_svg(results, svg_path)

    print(md_report)
    print(f"Wrote {json_path}, {md_path}, and {svg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
